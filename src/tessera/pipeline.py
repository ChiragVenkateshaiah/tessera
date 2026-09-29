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
from tessera.generation.answer import Citation, generate_answer
from tessera.generation.base import LLMClient
from tessera.generation.expertise import generate_expertise_answer
from tessera.retrieval.expertise import find_experts
from tessera.retrieval.retriever import retrieve
from tessera.retrieval.router import Archetype, route, terminal_response_for
from tessera.store.base import ExpertiseStore, PersonMatch, VectorStore

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


def answer_query(
    query: str,
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    expertise_store: ExpertiseStore | None = None,
) -> AnswerResult:
    """Run a query through the full pipeline: route, then return a
    terminal response (D), find and describe people (B), or retrieve and
    generate a grounded, cited answer from documents (A/C).

    expertise_store is only needed for archetype B; without one a B query
    gets EXPERTISE_UNAVAILABLE_MESSAGE rather than an error.
    """
    decision = route(query, llm)

    terminal = terminal_response_for(decision.archetype)
    if terminal is not None:
        return AnswerResult(
            query=query,
            archetype=decision.archetype,
            answer=terminal,
            citations=[],
        )

    if decision.archetype is Archetype.EXPERTISE:
        if expertise_store is None:
            return AnswerResult(
                query=query,
                archetype=decision.archetype,
                answer=EXPERTISE_UNAVAILABLE_MESSAGE,
                citations=[],
            )
        experts = find_experts(query, embedder, expertise_store)
        generated = generate_expertise_answer(experts, llm)
        return AnswerResult(
            query=query,
            archetype=decision.archetype,
            answer=generated.answer,
            citations=[],
            experts=generated.experts,
        )

    retrieval = retrieve(query, decision.archetype, embedder, store)
    generated = generate_answer(retrieval, llm)
    return AnswerResult(
        query=query,
        archetype=decision.archetype,
        answer=generated.answer,
        citations=generated.citations,
    )
