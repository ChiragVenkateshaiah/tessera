"""LangGraph orchestration (Phase 5 plan §3.7, P5-7): the ``orchestration``
switch's ``graph`` value.

The same flow as ``NativePipeline``, as a ``StateGraph``::

    START -> route -+-> terminal -----------------------+
                    +-> experts -> generate_b ----------+-> assemble -> END
                    +-> documents -> generate ----------+

``documents`` is the corrective subgraph (``tessera.lc.corrective``):
retrieve, grade, and at most one rewrite-and-retry. With the
``corrective`` switch ``off`` it retrieves once and makes no call, so the
graph equals native.

**Rules it keeps** (plan §3.2.2, CLAUDE.md constraint #6):
- **State is plain data**: the query, the principal, the routing decision,
  the retrieval attempts, the generator's output, and usage summaries.
  No client, store or recorder ever enters state.
- **Dependencies arrive per invocation** through LangGraph's runtime
  context (``GraphContext`` as ``context_schema``; nodes read
  ``runtime.context``). The graph is compiled once, at import, with
  nothing bound — so one compiled graph serves every principal, model and
  store, and two concurrent invocations share nothing but the graph.
- **Usage is state, not a shared recorder.** Each node that calls a model
  wraps the context's client in its *own* ``UsageRecorder`` and returns
  that recorder's summary into an append-reducer list
  (``Annotated[list, operator.add]``). Concurrent invocations can't mix
  their usage, because no recorder outlives its node.
- **Retrieval attempts are an append-reducer list too**, so every attempt
  (the corrective loop's retry, later) stays visible to the leak checks.
- **Compiled without a checkpointer**: each question is answered
  independently (no conversation memory — CLAUDE.md's do-not-build list).
- **No agents and no tool calling**: conditional edges route on the
  router's plain decision.

**Equivalence.** With the same steps, the graph returns a ``PipelineRun``
equal (``==``) to ``NativePipeline.run``'s — checked on fakes for every
path and on replayed real responses (``tests/test_lc_graph.py``).
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from tessera.embedding.base import Embedder
from tessera.generation.answer import Citation, GeneratedAnswer, filter_relevant, generate_answer
from tessera.generation.base import LLMClient
from tessera.generation.expertise import generate_expertise_answer
from tessera.generation.usage import UsageRecorder, UsageSummary
from tessera.lc.chat_models import status_of
from tessera.lc.corrective import build_corrective_subgraph
from tessera.pipeline import (
    EXPERTISE_UNAVAILABLE_MESSAGE,
    AnswerResult,
    PipelineRun,
    document_trace,
    expertise_trace,
)
from tessera.principal import Principal
from tessera.retrieval.expertise import TOP_K as EXPERTISE_TOP_K
from tessera.retrieval.expertise import ExpertiseResult, find_experts
from tessera.retrieval.retriever import RetrievalResult
from tessera.retrieval.router import Archetype, RoutingDecision, route, terminal_response_for
from tessera.store.base import ExpertiseStore, PersonMatch, SearchResult, VectorStore
from tessera.trace import Trace

GRAPH_RUN_NAME = "tessera_graph"


# --- what a node may read: the runtime context ---


@dataclass(frozen=True)
class GraphContext:
    """Everything a node calls, passed per invocation (never in state,
    never bound at compile time). The step functions default to the core
    functions, exactly as ``NativePipeline``'s do; a composition root
    passes LangChain replacements or ``@traceable``-wrapped ones."""

    llm: LLMClient
    embedder: Embedder
    store: VectorStore
    expertise_store: ExpertiseStore | None = None
    router_llm: LLMClient | None = None
    expertise_k: int = EXPERTISE_TOP_K
    route_fn: Callable[..., RoutingDecision] | None = None
    retrieve_fn: Callable[..., RetrievalResult] | None = None
    generate_fn: Callable[..., GeneratedAnswer] | None = None
    find_experts_fn: Callable[..., ExpertiseResult] | None = None
    generate_expertise_fn: Callable[..., GeneratedAnswer] | None = None
    # The corrective loop: off · score · llm (tessera.lc.corrective).
    corrective: str = "off"
    # Multi-query as a Send map-reduce (tessera.lc.multiquery) in place of
    # retrieve_fn: the retriever switch's multiquery_send value. Needs
    # the LangChain store (it searches through a ScopedStore).
    multiquery_send: bool = False


# --- the state: plain data, two append reducers ---


class GraphState(TypedDict, total=False):
    """One question's state. ``retrievals``, ``routing_usage`` and
    ``generation_usage`` are append-only lists (every attempt and every
    call kept); every other key is last-write."""

    query: str
    principal: Principal | None
    # Whether a people index exists (B needs one). Plain data, so the
    # conditional edge after routing can read it without the context.
    people_index: bool
    decision: RoutingDecision
    retrievals: Annotated[list[RetrievalResult], operator.add]
    shown: tuple[SearchResult, ...]
    generated: GeneratedAnswer | None
    expertise: ExpertiseResult | None
    answer: str
    citations: list[Citation]
    experts: list[PersonMatch]
    trace: Trace
    routing_usage: Annotated[list[UsageSummary], operator.add]
    generation_usage: Annotated[list[UsageSummary], operator.add]
    # The corrective subgraph's keys: its grader and rewriter calls, the
    # verdict, and the rewritten query (None unless it retried).
    corrective_usage: Annotated[list[UsageSummary], operator.add]
    # Calls made inside retrieval (the Send multi-query's rephrasing).
    retrieval_usage: Annotated[list[UsageSummary], operator.add]
    sufficient: bool
    rewritten_query: str | None
    run: PipelineRun


# --- nodes: pure over state, dependencies from runtime.context ---


def route_node(state: GraphState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    recorder = UsageRecorder(ctx.router_llm if ctx.router_llm is not None else ctx.llm)
    decision = (ctx.route_fn or route)(state["query"], recorder)
    get_stream_writer()({"stage": "routed", "archetype": decision.archetype.value})
    return {"decision": decision, "routing_usage": [recorder.summary()]}


def after_route(state: GraphState) -> Literal["terminal", "experts", "documents"]:
    """The conditional edge: D (and B without a people index) end in a
    fixed message; B goes to people; A and C go to documents."""
    archetype = state["decision"].archetype
    if terminal_response_for(archetype) is not None:
        return "terminal"
    if archetype is Archetype.EXPERTISE:
        return "experts" if state.get("people_index") else "terminal"
    return "documents"


def terminal_node(state: GraphState) -> dict[str, Any]:
    decision = state["decision"]
    answer = terminal_response_for(decision.archetype) or EXPERTISE_UNAVAILABLE_MESSAGE
    return {"answer": answer, "trace": Trace(decision.reasoning, fixed_response=True)}


def experts_node(state: GraphState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    found = (ctx.find_experts_fn or find_experts)(
        state["query"], ctx.embedder, ctx.expertise_store, k=ctx.expertise_k
    )
    return {"expertise": found}


def generate_b_node(state: GraphState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    found = state["expertise"]
    assert found is not None
    recorder = UsageRecorder(ctx.llm)
    generated = (ctx.generate_expertise_fn or generate_expertise_answer)(found, recorder)
    return {
        "generated": generated,
        "answer": generated.answer,
        "experts": generated.experts,
        "trace": expertise_trace(state["decision"], found),
        "generation_usage": [recorder.summary()],
    }


def generate_node(state: GraphState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    """Generate from the last retrieval attempt (the only one, unless the
    corrective loop retried), answering the original question."""
    ctx = runtime.context
    retrieval = replace(state["retrievals"][-1], query=state["query"])
    recorder = UsageRecorder(ctx.llm)
    generated = (ctx.generate_fn or generate_answer)(retrieval, recorder)
    shown = tuple(filter_relevant(retrieval.results))
    get_stream_writer()({"stage": "answered", "citations": len(generated.citations)})
    return {
        "generated": generated,
        "answer": generated.answer,
        "citations": generated.citations,
        "shown": shown,
        "trace": document_trace(state["decision"], retrieval, generated, shown),
        "generation_usage": [recorder.summary()],
    }


def _merged(summaries: list[UsageSummary]) -> UsageSummary:
    return UsageSummary(
        calls=tuple(call for s in summaries for call in s.calls),
        unmetered_calls=sum(s.unmetered_calls for s in summaries),
    )


def assemble_node(state: GraphState) -> dict[str, Any]:
    return {"run": assemble_run(state)}


def assemble_run(state: GraphState) -> PipelineRun:
    """The ``PipelineRun``, shaped exactly as ``NativePipeline`` shapes it
    (shared by the graph and the Functional API variant)."""
    principal = state.get("principal")
    decision = state["decision"]
    routing = _merged(state.get("routing_usage", []))
    corrective = _merged([*state.get("retrieval_usage", []), *state.get("corrective_usage", [])])
    generation = _merged(state.get("generation_usage", []))
    run = PipelineRun(
        answer=AnswerResult(
            query=state["query"],
            archetype=decision.archetype,
            answer=state["answer"],
            citations=state.get("citations") or [],
            experts=state.get("experts") or [],
            usage=_merged([routing, corrective, generation]),
            trace=replace(state["trace"], principal=principal.person_id if principal else None),
        ),
        decision=decision,
        retrievals=tuple(state.get("retrievals", [])),
        shown=state.get("shown", ()),
        generated=state.get("generated"),
        expertise=state.get("expertise"),
        routing_usage=routing,
        generation_usage=generation,
    )
    return run


# --- retries as a graph concern: the ``retry`` switch's ``graph`` value ---


def is_transient(exc: BaseException) -> bool:
    """A 429 or 5xx, wherever the status hides (the native retry layer's
    rule, read with the same ``status_of``)."""
    code = status_of(exc)
    return code is not None and (code == 429 or 500 <= code < 600)


# The native layer's 5xx schedule (RetryingLLMClient: 15 s base, doubling,
# 120 s cap, 6 attempts). What a node-level RetryPolicy can't do is the
# difference: it never reads Retry-After, its schedule is the same for a
# 429 as for a 503, and its waits aren't shared with other callers.
GRAPH_RETRY_POLICY = RetryPolicy(
    initial_interval=15.0,
    backoff_factor=2.0,
    max_interval=120.0,
    max_attempts=6,
    jitter=True,
    retry_on=is_transient,
)


def build_graph(retry_policy: RetryPolicy | None = None) -> Any:
    """The orchestrator, compiled once and without a checkpointer.

    ``retry_policy`` goes on every node that calls a model (route,
    generate, generate_b, and the corrective subgraph's nodes). A failed
    node is re-run whole: its writes from the failed attempt are dropped,
    so a retry can't double-count usage or retrievals."""
    graph = StateGraph(GraphState, context_schema=GraphContext)
    graph.add_node("route", route_node, retry_policy=retry_policy)
    graph.add_node("terminal", terminal_node)
    graph.add_node("experts", experts_node)
    graph.add_node("generate_b", generate_b_node, retry_policy=retry_policy)
    graph.add_node("documents", build_corrective_subgraph(retry_policy))
    graph.add_node("generate", generate_node, retry_policy=retry_policy)
    graph.add_node("assemble", assemble_node)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, ["terminal", "experts", "documents"])
    graph.add_edge("experts", "generate_b")
    graph.add_edge("documents", "generate")
    for last in ("terminal", "generate_b", "generate"):
        graph.add_edge(last, "assemble")
    graph.add_edge("assemble", END)
    return graph.compile(name=GRAPH_RUN_NAME)


GRAPH = build_graph()
# The same graph with LangGraph's RetryPolicy on its model-calling nodes:
# TESSERA_LC_RETRY=graph. A second compiled graph, not a flag, so the
# default graph has no policy at all (under retry=lc, a policy would retry
# again what .with_retry() already gave up on).
GRAPH_WITH_RETRIES = build_graph(GRAPH_RETRY_POLICY)


class LangGraphPipeline:
    """A ``Pipeline`` whose orchestration is the LangGraph ``GRAPH``. Takes
    the same constructor parameters as ``NativePipeline`` (``*_fn``
    steps included), so a composition root swaps one for the other."""

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        store: VectorStore,
        expertise_store: ExpertiseStore | None = None,
        *,
        router_llm: LLMClient | None = None,
        expertise_k: int = EXPERTISE_TOP_K,
        route_fn: Callable[..., RoutingDecision] | None = None,
        retrieve_fn: Callable[..., RetrievalResult] | None = None,
        generate_fn: Callable[..., GeneratedAnswer] | None = None,
        find_experts_fn: Callable[..., ExpertiseResult] | None = None,
        generate_expertise_fn: Callable[..., GeneratedAnswer] | None = None,
        corrective: str = "off",
        multiquery_send: bool = False,
        graph: Any = None,
    ) -> None:
        # The compiled graph to run: GRAPH, or GRAPH_WITH_RETRIES under
        # retry=graph (tests pass one built with a fast policy).
        self.graph = graph if graph is not None else GRAPH
        self.context = GraphContext(
            llm=llm,
            embedder=embedder,
            store=store,
            expertise_store=expertise_store,
            router_llm=router_llm,
            expertise_k=expertise_k,
            route_fn=route_fn,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            find_experts_fn=find_experts_fn,
            generate_expertise_fn=generate_expertise_fn,
            corrective=corrective,
            multiquery_send=multiquery_send,
        )

    def initial_state(self, query: str, principal: Principal | None) -> GraphState:
        return {
            "query": query,
            "principal": principal,
            "people_index": self.context.expertise_store is not None,
        }

    def run(self, query: str, principal: Principal | None = None) -> PipelineRun:
        state = self.graph.invoke(
            self.initial_state(query, principal),
            context=self.context,
            config={"run_name": GRAPH_RUN_NAME},
        )
        return state["run"]


# --- streaming: the four stream modes, each for one purpose ---


@dataclass(frozen=True)
class StreamedRun:
    """One question answered through ``GRAPH.stream`` with every mode on:
    - ``values`` (the full state after each step) gives the final
      ``PipelineRun`` — ``snapshots`` counts them;
    - ``updates`` (each node's own output, as it finishes) gives
      ``node_seconds``: wall time per node, in order;
    - ``messages`` (LLM tokens as they're generated, from any LangChain
      chat model a node calls) gives ``first_token_s`` and ``tokens`` —
      the time-to-first-token measurement. Only a LangChain chat model
      streams tokens (``model_client=lc`` with ``prompt_chain=lc``);
      otherwise ``first_token_s`` is None;
    - ``custom`` (what nodes write with ``get_stream_writer``) gives
      ``progress``: stages and counts only, never retrieved text.
    """

    run: PipelineRun
    node_seconds: tuple[tuple[str, float], ...]
    first_token_s: float | None
    tokens: int
    progress: tuple[dict[str, Any], ...]
    snapshots: int
    total_s: float
    token_text: str = field(default="", repr=False)


STREAM_MODES = ("values", "updates", "messages", "custom")


def stream_run(
    pipeline: LangGraphPipeline,
    query: str,
    principal: Principal | None = None,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> StreamedRun:
    """Answer ``query`` through the pipeline's graph's ``stream`` with all four modes."""
    start = last = clock()
    node_seconds: list[tuple[str, float]] = []
    progress: list[dict[str, Any]] = []
    first_token: float | None = None
    tokens: list[str] = []
    final: dict[str, Any] = {}
    snapshots = 0
    # subgraphs=True: the corrective subgraph's progress writes and tokens
    # reach this stream too. Values and timings are the top level's
    # (namespace ()), so a subgraph's inner steps aren't double-counted.
    for namespace, mode, chunk in pipeline.graph.stream(
        pipeline.initial_state(query, principal),
        context=pipeline.context,
        config={"run_name": GRAPH_RUN_NAME},
        stream_mode=list(STREAM_MODES),
        subgraphs=True,
    ):
        now = clock()
        top = namespace == ()
        if mode == "values" and top:
            final, snapshots = chunk, snapshots + 1
        elif mode == "updates" and top:
            node_seconds.extend((node, now - last) for node in chunk)
            last = now
        elif mode == "messages":
            message, _meta = chunk
            text = message.content if isinstance(message.content, str) else ""
            if text:
                tokens.append(text)
                if first_token is None:
                    first_token = now - start
        elif mode == "custom":
            progress.append(chunk)
    return StreamedRun(
        run=final["run"],
        node_seconds=tuple(node_seconds),
        first_token_s=first_token,
        tokens=len(tokens),
        progress=tuple(progress),
        snapshots=snapshots,
        total_s=clock() - start,
        token_text="".join(tokens),
    )


__all__ = [
    "GRAPH",
    "GRAPH_RETRY_POLICY",
    "GRAPH_WITH_RETRIES",
    "STREAM_MODES",
    "GraphContext",
    "GraphState",
    "LangGraphPipeline",
    "StreamedRun",
    "assemble_run",
    "build_graph",
    "is_transient",
    "stream_run",
]
