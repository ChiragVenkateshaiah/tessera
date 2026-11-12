"""The corrective retrieval loop as a LangGraph subgraph (Phase 5 plan
§3.7, P5-7): the ``corrective`` switch (``off`` · ``score`` · ``llm``).

::

    START -> retrieve -> grade -+-> END                 (sufficient, or off)
                                +-> rewrite -> retry -> END   (at most once)

- **retrieve** is the pipeline's own retrieval step (the context's
  ``retrieve_fn``, principal-bound), exactly as without the loop.
- **grade** judges whether the context can answer the question:
  - ``off``: always sufficient — no call, so the graph equals native;
  - ``score``: deterministic. Insufficient when too few chunks clear the
    generation floor (A: 1, C: 2) or the best chunk is weak
    (< ``WEAK_TOP_SCORE``). A retrieval-only probe on the eval set (the
    P5-7 report) found the best score does **not** separate the
    partial-recall cases from the full-recall ones, so this grader is
    expected to fire rarely and to help little here; the sweep measures it;
  - ``llm``: the routing model reads the question and the sources the
    answer model would see (already principal-scoped) and returns
    ``{"sufficient": bool, "reasoning": str}``. An unparseable verdict
    counts as sufficient (no retry): a retry is safe but costs calls.
- **rewrite** turns the question into one search query (the routing model;
  it sees only the question).
- **retry** runs the same principal-bound retrieval step on the rewritten
  query. Its attempt is appended to ``retrievals`` — every attempt stays
  visible to the leak and context-marker checks — and generation uses it,
  answering the **original** question. There is no second grade: the loop
  runs at most once.

Grader and rewriter calls are metered into ``corrective_usage`` (an
append-reducer list), which the answer's total usage includes.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from tessera.generation.answer import filter_relevant
from tessera.generation.usage import UsageRecorder, UsageSummary
from tessera.lc.generation import format_sources
from tessera.principal import Principal
from tessera.retrieval.retriever import RetrievalResult, retrieve
from tessera.retrieval.router import Archetype, RoutingDecision, _strip_markdown_fence

CORRECTIVE_VALUES = ("off", "score", "llm")

# The score grader (see the module docstring and the P5-7 report).
WEAK_TOP_SCORE = 0.45
MIN_SHOWN = {Archetype.LOOKUP: 1, Archetype.SYNTHESIS: 2}

GRADER_SYSTEM_PROMPT = (
    "You check whether sources retrieved from a consulting firm's internal "
    "knowledge base contain the information needed to answer a question. "
    "Judge only the sources given; do not answer the question.\n\n"
    'Reply with JSON only, no prose and no code fences: {"sufficient": true or '
    'false, "reasoning": "<one sentence>"}. Say false only when the sources '
    "are about something else or miss what the question asks for."
)
REWRITE_SYSTEM_PROMPT = (
    "You rewrite a consultant's question into one search query for the "
    "firm's internal knowledge base (a methodology wiki and published "
    "thought leadership). Keep every named entity, industry and constraint "
    "in the question, and use the terms such documents would use. Reply "
    "with the query only, on one line."
)


def grader_user_prompt(question: str, shown: list[Any]) -> str:
    return f"Question: {question}\n\nSources:\n\n{format_sources(shown)}"


def score_sufficient(retrieval: RetrievalResult) -> bool:
    """The deterministic grader."""
    shown = filter_relevant(retrieval.results)
    if len(shown) < MIN_SHOWN.get(retrieval.archetype, 1):
        return False
    return max(r.score for r in shown) >= WEAK_TOP_SCORE


def parse_verdict(text: str) -> bool:
    """The LLM grader's verdict; anything unreadable is "sufficient"."""
    try:
        value = json.loads(_strip_markdown_fence(text))
    except ValueError:
        return True
    return not (isinstance(value, dict) and value.get("sufficient") is False)


class CorrectiveState(TypedDict, total=False):
    """The keys the subgraph shares with the orchestrator's state."""

    query: str
    principal: Principal | None
    decision: RoutingDecision
    retrievals: Annotated[list[RetrievalResult], operator.add]
    retrieval_usage: Annotated[list[UsageSummary], operator.add]
    corrective_usage: Annotated[list[UsageSummary], operator.add]
    sufficient: bool
    rewritten_query: str | None


def _grading_llm(ctx: Any) -> Any:
    return ctx.router_llm if ctx.router_llm is not None else ctx.llm


# --- the loop's steps as plain functions (the graph nodes below and the
# Functional API variant both call these, so the two can't drift) ---


def grade(ctx: Any, query: str, retrieval: RetrievalResult) -> tuple[bool, UsageSummary | None]:
    """Whether ``retrieval`` can answer ``query``, per ``ctx.corrective``,
    and the grading call's usage (None when no call was made)."""
    if ctx.corrective == "off":
        return True, None
    if ctx.corrective == "score":
        return score_sufficient(retrieval), None
    shown = filter_relevant(retrieval.results)
    if not shown:
        # Nothing over the floor: the fixed "nothing on that" answer
        # would follow, so a retry can only help. No call needed.
        return False, None
    recorder = UsageRecorder(_grading_llm(ctx))
    verdict = recorder.complete(GRADER_SYSTEM_PROMPT, grader_user_prompt(query, shown))
    return parse_verdict(verdict), recorder.summary()


def rewrite(ctx: Any, query: str) -> tuple[str | None, UsageSummary]:
    """One search query for ``query`` (None if the model gave nothing)."""
    recorder = UsageRecorder(_grading_llm(ctx))
    text = recorder.complete(REWRITE_SYSTEM_PROMPT, f"Question: {query}")
    lines = [line.strip().strip('"') for line in text.strip().splitlines() if line.strip()]
    return (lines[0] if lines else None), recorder.summary()


def retrieve_with(
    ctx: Any, query: str, archetype: Archetype, principal: Principal | None
) -> tuple[RetrievalResult, UsageSummary | None]:
    """The pipeline's retrieval step, principal-bound, as without the loop
    — or the Send multi-query subgraph when the context selects it, whose
    rephrasing call is metered (its usage is returned)."""
    if getattr(ctx, "multiquery_send", False):
        from tessera.lc.multiquery import send_multiquery

        return send_multiquery(ctx, query, archetype, principal)
    return (ctx.retrieve_fn or retrieve)(query, archetype, ctx.embedder, ctx.store, principal=principal), None


def _with_usage(retrieval: RetrievalResult, usage: UsageSummary | None) -> dict[str, Any]:
    get_stream_writer()({"stage": "retrieved", "chunks": len(retrieval.results)})
    return {"retrievals": [retrieval], **({"retrieval_usage": [usage]} if usage is not None else {})}


def retrieve_node(state: CorrectiveState, runtime: Runtime[Any]) -> dict[str, Any]:
    return _with_usage(
        *retrieve_with(runtime.context, state["query"], state["decision"].archetype, state.get("principal"))
    )


def grade_node(state: CorrectiveState, runtime: Runtime[Any]) -> dict[str, Any]:
    sufficient, usage = grade(runtime.context, state["query"], state["retrievals"][-1])
    if runtime.context.corrective != "off":
        get_stream_writer()({"stage": "graded", "sufficient": sufficient})
    return {"sufficient": sufficient, **({"corrective_usage": [usage]} if usage is not None else {})}


def after_grade(state: CorrectiveState) -> Literal["rewrite", "__end__"]:
    if state.get("sufficient", True) or len(state["retrievals"]) > 1:
        return END
    return "rewrite"


def rewrite_node(state: CorrectiveState, runtime: Runtime[Any]) -> dict[str, Any]:
    rewritten, usage = rewrite(runtime.context, state["query"])
    return {"rewritten_query": rewritten, "corrective_usage": [usage]}


def after_rewrite(state: CorrectiveState) -> Literal["retry", "__end__"]:
    return "retry" if state.get("rewritten_query") else END


def retry_node(state: CorrectiveState, runtime: Runtime[Any]) -> dict[str, Any]:
    rewritten = state["rewritten_query"]
    assert rewritten
    return _with_usage(*retrieve_with(runtime.context, rewritten, state["decision"].archetype, state.get("principal")))


def build_corrective_subgraph(retry_policy: Any = None) -> Any:
    """The loop, compiled. ``retry_policy`` (a LangGraph ``RetryPolicy``, the
    ``retry`` switch's ``graph`` value) goes on every node that can call a
    model: retrieve and retry (the Send multi-query's rephrasing), grade
    and rewrite."""
    graph = StateGraph(CorrectiveState)
    graph.add_node("retrieve", retrieve_node, retry_policy=retry_policy)
    graph.add_node("grade", grade_node, retry_policy=retry_policy)
    graph.add_node("rewrite", rewrite_node, retry_policy=retry_policy)
    graph.add_node("retry", retry_node, retry_policy=retry_policy)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["rewrite", END])
    graph.add_conditional_edges("rewrite", after_rewrite, ["retry", END])
    graph.add_edge("retry", END)
    return graph.compile(name="corrective")


__all__ = [
    "CORRECTIVE_VALUES",
    "grade",
    "retrieve_with",
    "rewrite",
    "GRADER_SYSTEM_PROMPT",
    "REWRITE_SYSTEM_PROMPT",
    "build_corrective_subgraph",
    "parse_verdict",
    "score_sufficient",
]
