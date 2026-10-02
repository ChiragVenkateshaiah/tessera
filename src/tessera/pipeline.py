"""query -> route -> retrieve -> generate.

The single entry point tying Tasks 4-6 together. Pure with respect to
infrastructure per CLAUDE.md constraint #6 — every dependency
(LLMClient, Embedder, VectorStore) is injected, nothing is constructed
here — so a future edge/routing layer can call this via a plain function
call, subprocess, or HTTP wrapper without this module changing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tessera.embedding.base import Embedder
from tessera.generation.answer import (
    RELEVANCE_THRESHOLD,
    Citation,
    filter_relevant,
    generate_answer,
)
from tessera.generation.base import LLMClient
from tessera.generation.expertise import (
    EXPERTISE_PERSON_FLOOR,
    EXPERTISE_QUERY_FLOOR,
    filter_qualified,
    generate_expertise_answer,
)
from tessera.generation.usage import UsageRecorder, UsageSummary, combine
from tessera.retrieval.expertise import find_experts
from tessera.retrieval.retriever import retrieve
from tessera.retrieval.router import Archetype, route, terminal_response_for
from tessera.store.base import ExpertiseStore, PersonMatch, VectorStore
from tessera.trace import RetrievedItem, Trace

EXPERTISE_UNAVAILABLE_MESSAGE = (
    "The expertise index hasn't been built, so I can't look up people yet."
)


@dataclass(frozen=True)
class AnswerResult:
    """The pipeline's single return shape, regardless of archetype.

    D carries a terminal message and no citations (never reaches
    retrieval/generation); A/C carry a generated answer with document
    citations; B carries a generated answer with the ranked people (and
    their evidence) it named in ``experts``. One shape means callers
    (CLI, eval harness) don't need to branch on archetype to know what
    they got back.
    """

    query: str
    archetype: Archetype
    answer: str
    citations: list[Citation]
    experts: list[PersonMatch] = field(default_factory=list)
    # Tokens of every LLM call behind this answer (routing included);
    # callers turn it into dollars with a price table (Phase 4).
    usage: UsageSummary = field(default_factory=UsageSummary)
    # How the answer was produced — the composition roots log it with a
    # trace_id (Phase 4, plan §3.2.1).
    trace: Trace = field(default_factory=lambda: Trace(route_reasoning=""))


def answer_query(
    query: str,
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    expertise_store: ExpertiseStore | None = None,
    *,
    router_llm: LLMClient | None = None,
) -> AnswerResult:
    """Run a query through the full pipeline: route, then return a
    terminal response (D), find and describe people (B), or retrieve and
    generate a grounded, cited answer from documents (A/C).

    expertise_store is only needed for archetype B; without one a B query
    gets EXPERTISE_UNAVAILABLE_MESSAGE rather than an error.

    router_llm, when given, makes the routing call (a cheap classification
    — Haiku on Bedrock) while ``llm`` writes the answer; by default one
    client does both.
    """
    answer_llm = UsageRecorder(llm)
    routing_llm = UsageRecorder(router_llm) if router_llm is not None else answer_llm

    decision = route(query, routing_llm)

    def result(
        answer: str,
        trace: Trace,
        citations: list[Citation] | None = None,
        experts: list[PersonMatch] | None = None,
    ) -> AnswerResult:
        return AnswerResult(
            query=query,
            archetype=decision.archetype,
            answer=answer,
            citations=citations or [],
            experts=experts or [],
            usage=combine([routing_llm, answer_llm]),
            trace=trace,
        )

    terminal = terminal_response_for(decision.archetype)
    if terminal is not None:
        return result(terminal, Trace(decision.reasoning, fixed_response=True))

    if decision.archetype is Archetype.EXPERTISE:
        if expertise_store is None:
            return result(
                EXPERTISE_UNAVAILABLE_MESSAGE,
                Trace(decision.reasoning, fixed_response=True),
            )
        found = find_experts(query, embedder, expertise_store)
        generated = generate_expertise_answer(found, answer_llm)
        # The same filter the generator applied, so "used" is exactly who
        # the model was shown.
        shown = {m.person.person_id for m in filter_qualified(found.matches)}
        trace = Trace(
            decision.reasoning,
            retrieved_kind="person",
            retrieved=tuple(
                RetrievedItem(m.person.person_id, m.evidence_score, m.person.person_id in shown)
                for m in found.matches
            ),
            floors={
                "expertise_query_floor": EXPERTISE_QUERY_FLOOR,
                "expertise_person_floor": EXPERTISE_PERSON_FLOOR,
            },
            fixed_response=not shown,
        )
        return result(generated.answer, trace, experts=generated.experts)

    retrieval = retrieve(query, decision.archetype, embedder, store)
    generated = generate_answer(retrieval, answer_llm)
    shown_chunks = {r.chunk_id for r in filter_relevant(retrieval.results)}
    trace = Trace(
        decision.reasoning,
        retrieved_kind="chunk",
        retrieved=tuple(
            RetrievedItem(r.chunk_id, r.score, r.chunk_id in shown_chunks, r.document_path)
            for r in retrieval.results
        ),
        floors={"relevance_threshold": RELEVANCE_THRESHOLD},
        fixed_response=not shown_chunks,
    )
    return result(generated.answer, trace, citations=generated.citations)
