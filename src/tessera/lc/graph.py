"""LangGraph orchestration (Phase 5 plan §3.7, P5-7): the ``orchestration``
switch's ``graph`` value.

The same flow as ``NativePipeline``, as a ``StateGraph``::

    START -> route -+-> terminal ----------------------+
                    +-> experts -> generate_b ---------+-> assemble -> END
                    +-> retrieve -> generate ----------+

(the corrective subgraph replaces ``retrieve`` when the ``corrective``
switch is on — added in the next P5-7 step).

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
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from tessera.embedding.base import Embedder
from tessera.generation.answer import Citation, GeneratedAnswer, filter_relevant, generate_answer
from tessera.generation.base import LLMClient
from tessera.generation.expertise import generate_expertise_answer
from tessera.generation.usage import UsageRecorder, UsageSummary
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
from tessera.retrieval.retriever import RetrievalResult, retrieve
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
    run: PipelineRun


# --- nodes: pure over state, dependencies from runtime.context ---


def route_node(state: GraphState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    recorder = UsageRecorder(ctx.router_llm if ctx.router_llm is not None else ctx.llm)
    decision = (ctx.route_fn or route)(state["query"], recorder)
    return {"decision": decision, "routing_usage": [recorder.summary()]}


def after_route(state: GraphState) -> Literal["terminal", "experts", "retrieve"]:
    """The conditional edge: D (and B without a people index) end in a
    fixed message; B goes to people; A and C go to documents."""
    archetype = state["decision"].archetype
    if terminal_response_for(archetype) is not None:
        return "terminal"
    if archetype is Archetype.EXPERTISE:
        return "experts" if state.get("people_index") else "terminal"
    return "retrieve"


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


def retrieve_node(state: GraphState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    retrieval = (ctx.retrieve_fn or retrieve)(
        state["query"],
        state["decision"].archetype,
        ctx.embedder,
        ctx.store,
        principal=state.get("principal"),
    )
    return {"retrievals": [retrieval]}


def generate_node(state: GraphState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    """Generate from the last retrieval attempt (the only one, unless the
    corrective loop retried)."""
    ctx = runtime.context
    retrieval = state["retrievals"][-1]
    recorder = UsageRecorder(ctx.llm)
    generated = (ctx.generate_fn or generate_answer)(retrieval, recorder)
    shown = tuple(filter_relevant(retrieval.results))
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
    """The ``PipelineRun``, shaped exactly as ``NativePipeline`` shapes it."""
    principal = state.get("principal")
    decision = state["decision"]
    routing = _merged(state.get("routing_usage", []))
    generation = _merged(state.get("generation_usage", []))
    run = PipelineRun(
        answer=AnswerResult(
            query=state["query"],
            archetype=decision.archetype,
            answer=state["answer"],
            citations=state.get("citations") or [],
            experts=state.get("experts") or [],
            usage=_merged([routing, generation]),
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
    return {"run": run}


def build_graph() -> Any:
    """The orchestrator, compiled once and without a checkpointer."""
    graph = StateGraph(GraphState, context_schema=GraphContext)
    graph.add_node("route", route_node)
    graph.add_node("terminal", terminal_node)
    graph.add_node("experts", experts_node)
    graph.add_node("generate_b", generate_b_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("assemble", assemble_node)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, ["terminal", "experts", "retrieve"])
    graph.add_edge("experts", "generate_b")
    graph.add_edge("retrieve", "generate")
    for last in ("terminal", "generate_b", "generate"):
        graph.add_edge(last, "assemble")
    graph.add_edge("assemble", END)
    return graph.compile(name=GRAPH_RUN_NAME)


GRAPH = build_graph()


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
    ) -> None:
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
        )

    def initial_state(self, query: str, principal: Principal | None) -> GraphState:
        return {
            "query": query,
            "principal": principal,
            "people_index": self.context.expertise_store is not None,
        }

    def run(self, query: str, principal: Principal | None = None) -> PipelineRun:
        state = GRAPH.invoke(
            self.initial_state(query, principal),
            context=self.context,
            config={"run_name": GRAPH_RUN_NAME},
        )
        return state["run"]


__all__ = ["GRAPH", "GraphContext", "GraphState", "LangGraphPipeline", "build_graph"]
