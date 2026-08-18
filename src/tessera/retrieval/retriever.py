"""Archetype-aware retrieval (A: narrow+filtered, C: broad multi-source)."""

from __future__ import annotations

from dataclasses import dataclass

from tessera.embedding.base import Embedder
from tessera.retrieval.router import Archetype
from tessera.store.base import SearchResult, VectorStore

# Lookup (A): the user wants "the document," not a survey — a small,
# precise set of matches is the point.
LOOKUP_TOP_K = 5

# Synthesis (C): pull a wider candidate pool so multiple sources get a
# chance to surface, then cap how many chunks any single document can
# contribute so one high-scoring document can't crowd out the rest.
SYNTHESIS_CANDIDATE_K = 20
SYNTHESIS_MAX_RESULTS = 10
SYNTHESIS_MAX_PER_DOCUMENT = 2


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved chunks for one query, ready for grounded generation
    (Task 6) — carries the archetype alongside the results so the
    generation step (found-documents summary vs multi-source synthesis)
    knows which prompt shape to use without re-deriving it.
    """

    query: str
    archetype: Archetype
    results: list[SearchResult]


def retrieve(
    query: str,
    archetype: Archetype,
    embedder: Embedder,
    store: VectorStore,
    where: dict[str, object] | None = None,
) -> RetrievalResult:
    """Retrieve chunks for query using the strategy appropriate to archetype.

    Pure with respect to infrastructure per CLAUDE.md constraint #6:
    Embedder and VectorStore are injected, not constructed here. Only
    called for A and C — B and D are short-circuited by
    router.terminal_response_for() before retrieval would run.
    """
    if archetype not in (Archetype.LOOKUP, Archetype.SYNTHESIS):
        raise ValueError(
            f"retrieve() only supports archetypes A and C, got {archetype.value!r}"
        )

    query_embedding = embedder.embed_query(query)

    if archetype is Archetype.LOOKUP:
        results = store.query(query_embedding, k=LOOKUP_TOP_K, where=where)
    else:
        candidates = store.query(
            query_embedding, k=SYNTHESIS_CANDIDATE_K, where=where
        )
        results = _diversify_by_source(candidates)

    return RetrievalResult(query=query, archetype=archetype, results=results)


def _diversify_by_source(candidates: list[SearchResult]) -> list[SearchResult]:
    """Trim a best-match-first candidate list to SYNTHESIS_MAX_RESULTS,
    capping how many chunks come from any one document so synthesis pulls
    from multiple sources instead of one dominant document.
    """
    per_document_count: dict[str, int] = {}
    diversified: list[SearchResult] = []
    for result in candidates:
        count = per_document_count.get(result.document_path, 0)
        if count >= SYNTHESIS_MAX_PER_DOCUMENT:
            continue
        per_document_count[result.document_path] = count + 1
        diversified.append(result)
        if len(diversified) >= SYNTHESIS_MAX_RESULTS:
            break
    return diversified
