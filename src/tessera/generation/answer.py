"""Turns retrieved chunks into a grounded, cited answer (Task 6)."""

from __future__ import annotations

from dataclasses import dataclass, field

from tessera.generation.base import LLMClient
from tessera.generation.prompts import (
    LOOKUP_ANSWER_SYSTEM_PROMPT,
    SYNTHESIS_ANSWER_SYSTEM_PROMPT,
    build_grounded_answer_user_prompt,
    group_by_document,
)
from tessera.retrieval.retriever import RetrievalResult, SupersededMatch
from tessera.retrieval.router import Archetype
from tessera.store.base import PersonMatch, SearchResult

# Below this cosine similarity, a chunk is treated as noise rather than a
# real match. Measured against the real corpus/embedder (LocalEmbedder).
# Recalibrated 2026-09-04 (P2-3, docs/Tessera_Phase2_Plan.md §4) against
# the current title-aware chunk embeddings (chunker.chunk_embedding_text,
# added 2026-08-27 — see that entry in checkpoint.md for the prior,
# now-stale numbers this replaces): on-corpus queries score >=0.58 on
# their weakest top-3 result; a topic that's semantically adjacent but
# genuinely absent ("parental leave"/"vacation policy" near HR/org-design
# content) tops out around 0.31; unrelated queries score <0.12. 0.35
# still sits cleanly in that gap (0.04 above the highest adjacent-but-
# absent probe, 0.22 below the weakest on-corpus one) — no change from
# the value Task 6 originally calibrated.
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
    cite inline with. One per document; heading_path is empty when more
    than one section of the document was shown.
    """

    marker: int
    document_path: str
    document_title: str
    heading_path: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedAnswer:
    """A grounded answer plus the sources it was allowed to draw from.

    Archetype B answers cite people rather than documents: ``citations``
    stays empty and ``experts`` carries the ranked people (with their
    evidence) the model was shown. A/C leave ``experts`` empty.
    """

    query: str
    archetype: Archetype
    answer: str
    citations: list[Citation]
    experts: list[PersonMatch] = field(default_factory=list)
    # Superseded documents the answer's closing note points away from, and
    # that note as appended to ``answer`` ("" when there is none). The eval
    # judge grades the model's text without it.
    superseded: tuple[SupersededMatch, ...] = ()
    notice: str = ""


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


def noted_superseded(
    retrieval: RetrievalResult, threshold: float = RELEVANCE_THRESHOLD
) -> tuple[SupersededMatch, ...]:
    """The excluded superseded documents worth telling the user about:
    those that would have cleared the same floor the shown chunks did.
    """
    return tuple(m for m in retrieval.superseded if m.score >= threshold)


def supersession_notice(
    superseded: tuple[SupersededMatch, ...], citations: list[Citation]
) -> str:
    """The fixed closing note for superseded matches (plan §3.3.2) —
    written here, not left to the model, so it is always there and always
    points at the current version's citation number when it was shown.
    """
    cited = {c.document_path: c for c in citations}
    lines = []
    for m in superseded:
        current = cited.get(m.superseded_by)
        newer = (
            f'"{current.document_title}" [{current.marker}]'
            if current
            else f"a newer version ({m.superseded_by})"
        )
        lines.append(
            f'Note: an older version, "{m.document_title}", also matches this '
            f"question but has been superseded by {newer}; this answer does not "
            "draw on the older version."
        )
    return "\n".join(lines)


def generate_answer(retrieval: RetrievalResult, llm: LLMClient) -> GeneratedAnswer:
    """Generate a grounded, cited answer from retrieved chunks.

    Pure with respect to infrastructure per CLAUDE.md constraint #6: the
    LLMClient is injected, not constructed. Only called for archetypes A
    and C — retrieve() already rejects B/D. B has its own generator
    (generation/expertise.py); D is short-circuited earlier via
    router.terminal_response_for().

    Chunks below RELEVANCE_THRESHOLD are dropped before the LLM ever sees
    them. If nothing clears the bar, this returns the fixed "nothing on
    that" message without spending an LLM call: a query with no on-corpus
    signal shouldn't cost anything against the LLM's daily quota, and the
    refusal is guaranteed rather than left to the model choosing to say
    so (CLAUDE.md constraint #2 — grounded generation only).

    When retrieval excluded a superseded document that would have cleared
    the floor, a fixed note naming it and its current version is appended
    (plan §3.3.2).
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
    # One citation per document, numbered as the prompt numbered them.
    # heading_path is the section only when a single section was shown.
    citations = [
        Citation(
            marker=i,
            document_path=group[0].document_path,
            document_title=group[0].document_title,
            heading_path=group[0].heading_path if len(group) == 1 else (),
        )
        for i, group in enumerate(group_by_document(relevant), start=1)
    ]
    superseded = noted_superseded(retrieval)
    notice = supersession_notice(superseded, citations) if superseded else ""
    if notice:
        answer = f"{answer.rstrip()}\n\n{notice}"
    return GeneratedAnswer(
        query=retrieval.query,
        archetype=retrieval.archetype,
        answer=answer,
        citations=citations,
        superseded=superseded,
        notice=notice,
    )
