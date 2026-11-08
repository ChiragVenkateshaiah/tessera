"""Multi-query retrieval as a LangGraph map-reduce with ``Send`` (Phase 5
plan §3.7, P5-7): the ``multiquery_send`` value of the ``retriever``
switch, compared with P5-5's classic ``MultiQueryRetriever``.

::

    START -> rephrase --Send x N--> search (one per query, in parallel) -> reduce -> END

- **rephrase**: the routing model writes the rephrasings with
  ``MultiQueryRetriever``'s own prompt and line parser
  (``multi_query_chain``); the original question is added last, as
  ``include_original=True`` does. Unlike the classic retriever, the call
  is **metered**: it goes through a ``UsageRecorder`` built from the
  runtime context, and its usage is returned with the result.
- **search**: one ``Send`` per query. Each runs the vector candidate pool
  through a ``ScopedStore`` bound to the asking principal — the only way
  this subgraph reaches the index, so every search carries the
  permission filter.
- **reduce**: the hits in query order, then the shared retrieval contract
  (``ArchetypeRetriever``'s, with ``order_by_cosine``): cosine re-score in
  scope (repeats dropped), sort by cosine to the original question, then
  ``retrieve_from_candidates``.

With the same rephrasings it returns exactly what the classic
``MultiQueryRetriever`` returns (the parity test); what differs is that
the searches run in parallel and the rephrasing call is counted.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document as LCDocument
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from tessera.generation.usage import UsageRecorder, UsageSummary
from tessera.lc.retrievers import (
    ScopedStore,
    ScopedVectorRetriever,
    candidate_k,
    multi_query_chain,
    rescore,
)
from tessera.principal import Principal
from tessera.retrieval.retriever import RetrievalResult, retrieve_from_candidates
from tessera.retrieval.router import Archetype


class MultiQueryState(TypedDict, total=False):
    query: str
    archetype: Archetype
    principal: Principal | None
    queries: list[str]
    # (query index, its hits): appended by the parallel searches, ordered
    # by index in reduce.
    hits: Annotated[list[tuple[int, list[LCDocument]]], operator.add]
    usage: Annotated[list[UsageSummary], operator.add]
    result: RetrievalResult


class SearchInput(TypedDict):
    index: int
    query: str
    archetype: Archetype
    principal: Principal | None


def rephrase_node(state: MultiQueryState, runtime: Runtime[Any]) -> dict[str, Any]:
    ctx = runtime.context
    recorder = UsageRecorder(ctx.router_llm if ctx.router_llm is not None else ctx.llm)
    lines = multi_query_chain(recorder).invoke({"question": state["query"]})
    return {"queries": [*lines, state["query"]], "usage": [recorder.summary()]}


def fan_out(state: MultiQueryState) -> list[Send]:
    return [
        Send(
            "search",
            {"index": i, "query": q, "archetype": state["archetype"], "principal": state.get("principal")},
        )
        for i, q in enumerate(state["queries"])
    ]


def search_node(state: SearchInput, runtime: Runtime[Any]) -> dict[str, Any]:
    ctx = runtime.context
    scoped = ScopedStore(ctx.store, state["principal"])
    retriever = ScopedVectorRetriever(scoped=scoped, embedder=ctx.embedder, k=candidate_k(state["archetype"]))
    return {"hits": [(state["index"], retriever.invoke(state["query"]))]}


def reduce_node(state: MultiQueryState, runtime: Runtime[Any]) -> dict[str, Any]:
    ctx = runtime.context
    principal = state.get("principal")
    scoped = ScopedStore(ctx.store, principal)
    docs = [doc for _, hits in sorted(state["hits"], key=lambda h: h[0]) for doc in hits]
    query_embedding = ctx.embedder.embed_query(state["query"])
    candidates = rescore(docs, query_embedding, scoped)
    candidates.sort(key=lambda r: r.score, reverse=True)
    result = retrieve_from_candidates(
        state["query"], state["archetype"], query_embedding, candidates, scoped, principal=principal, expand=True
    )
    return {"result": result}


def build_multiquery_graph() -> Any:
    graph = StateGraph(MultiQueryState)
    graph.add_node("rephrase", rephrase_node)
    graph.add_node("search", search_node)
    graph.add_node("reduce", reduce_node)
    graph.add_edge(START, "rephrase")
    graph.add_conditional_edges("rephrase", fan_out, ["search"])
    graph.add_edge("search", "reduce")
    graph.add_edge("reduce", END)
    return graph.compile(name="multiquery_send")


MULTIQUERY_GRAPH = build_multiquery_graph()


def send_multiquery(
    ctx: Any, query: str, archetype: Archetype, principal: Principal | None
) -> tuple[RetrievalResult, UsageSummary]:
    """One multi-query retrieval through the Send subgraph, as the asking
    principal: its result and the rephrasing call's usage."""
    state = MULTIQUERY_GRAPH.invoke(
        {"query": query, "archetype": archetype, "principal": principal}, context=ctx
    )
    usage = state.get("usage", [])
    return state["result"], UsageSummary(
        calls=tuple(c for u in usage for c in u.calls), unmetered_calls=sum(u.unmetered_calls for u in usage)
    )


__all__ = ["MULTIQUERY_GRAPH", "build_multiquery_graph", "send_multiquery"]
