"""Turns retrieved chunks into a grounded, cited answer (Task 6)."""

from __future__ import annotations

from dataclasses import dataclass

from tessera.generation.base import LLMClient
from tessera.generation.prompts import (
    LOOKUP_ANSWER_SYSTEM_PROMPT,
    SYNTHESIS_ANSWER_SYSTEM_PROMPT,
    build_grounded_answer_user_prompt,
)
from tessera.retrieval.retriever import RetrievalResult
from tessera.retrieval.router import Archetype
from tessera.store.base import SearchResult

# Below this cosine similarity, a chunk is treated as noise rather than a
# real match. Measured against the real corpus/embedder (LocalEmbedder):
# genuinely on-corpus queries score >=0.39 on their weakest top-3 result;
# a topic that's semantically adjacent but genuinely absent ("parental
# leave" near HR/org-design content) tops out around 0.31; unrelated
# queries score <0.15. 0.35 sits in that gap.
RELEVANCE_THRESHOLD = 0.35

NO_RESULTS_MESSAGE = (
    "We don't have anything on that in Meridian's corpus — nothing "
    "retrieved was relevant enough to ground an answer."
)

_SYSTEM_PROMPT_BY_ARCHETYPE = {
    Archetype.LOOKUP: LOOKUP_ANSWER_SYSTEM_PROMPT,
    Archetype.SYNTHESIS: SYNTHESIS_ANSWER_SYSTEM_PROMPT,
}


@dataclass(frozen=True)
class Citation:
    """One numbered source offered to the model for a generated answer —
    the marker matches the [n] reference the prompt asks the model to
    cite inline with.
    """

    marker: int
    document_path: str
    document_title: str
    heading_path: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedAnswer:
    """A grounded answer plus the sources it was allowed to draw from."""

    query: str
    archetype: Archetype
    answer: str
    citations: list[Citation]


def filter_relevant(
    results: list[SearchResult], threshold: float = RELEVANCE_THRESHOLD
) -> list[SearchResult]:
    """Chunks that clear RELEVANCE_THRESHOLD, in their original order.

    Factored out of generate_answer() so callers that need to know
    exactly which chunks the model was shown — e.g. the eval harness's
    LLM-judge, which grades groundedness against those same chunks —
    can reconstruct it without duplicating the filter.
    """
    return [r for r in results if r.score >= threshold]


def generate_answer(retrieval: RetrievalResult, llm: LLMClient) -> GeneratedAnswer:
    """Generate a grounded, cited answer from retrieved chunks.

    Pure with respect to infrastructure per CLAUDE.md constraint #6: the
    LLMClient is injected, not constructed. Only called for archetypes A
    and C — retrieve() already rejects B/D, and pipeline.py short-circuits
    them earlier via router.terminal_response_for() before generation
    would run.

    Chunks below RELEVANCE_THRESHOLD are dropped before the LLM ever sees
    them. If nothing clears the bar, this returns the fixed "nothing on
    that" message without spending an LLM call: a query with no on-corpus
    signal shouldn't cost anything against Gemini's daily quota, and the
    refusal is guaranteed rather than left to the model choosing to say
    so (CLAUDE.md constraint #2 — grounded generation only).
    """
    if retrieval.archetype not in _SYSTEM_PROMPT_BY_ARCHETYPE:
        raise ValueError(
            "generate_answer() only supports archetypes A and C, got "
            f"{retrieval.archetype.value!r}"
        )

    relevant = filter_relevant(retrieval.results)
    if not relevant:
        return GeneratedAnswer(
            query=retrieval.query,
            archetype=retrieval.archetype,
            answer=NO_RESULTS_MESSAGE,
            citations=[],
        )

    system = _SYSTEM_PROMPT_BY_ARCHETYPE[retrieval.archetype]
    answer = llm.complete(
        system=system,
        user=build_grounded_answer_user_prompt(retrieval.query, relevant),
    )
    citations = [
        Citation(
            marker=i,
            document_path=r.document_path,
            document_title=r.document_title,
            heading_path=r.heading_path,
        )
        for i, r in enumerate(relevant, start=1)
    ]
    return GeneratedAnswer(
        query=retrieval.query,
        archetype=retrieval.archetype,
        answer=answer,
        citations=citations,
    )
