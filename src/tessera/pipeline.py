"""query -> route -> retrieve -> generate.

The single entry point tying Tasks 4-6 together. Pure with respect to
infrastructure per CLAUDE.md constraint #6 — every dependency
(LLMClient, Embedder, VectorStore) is injected, nothing is constructed
here — so a future edge/routing layer can call this via a plain function
call, subprocess, or HTTP wrapper without this module changing.
"""

from __future__ import annotations

from dataclasses import dataclass

from tessera.embedding.base import Embedder
from tessera.generation.answer import Citation, generate_answer
from tessera.generation.base import LLMClient
from tessera.retrieval.retriever import retrieve
from tessera.retrieval.router import Archetype, route, terminal_response_for
from tessera.store.base import VectorStore


@dataclass(frozen=True)
class AnswerResult:
    """The pipeline's single return shape, regardless of archetype.

    B/D carry a terminal message and no citations (never reach
    retrieval/generation); A/C carry a generated, cited answer. One
    shape means callers (CLI, eval harness) don't need to branch on
    archetype to know what they got back.
    """

    query: str
    archetype: Archetype
    answer: str
    citations: list[Citation]


def answer_query(
    query: str,
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
) -> AnswerResult:
    """Run a query through the full pipeline: route, then either return a
    terminal response (B/D) or retrieve and generate a grounded, cited
    answer (A/C).
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

    retrieval = retrieve(query, decision.archetype, embedder, store)
    generated = generate_answer(retrieval, llm)
    return AnswerResult(
        query=query,
        archetype=decision.archetype,
        answer=generated.answer,
        citations=generated.citations,
    )
