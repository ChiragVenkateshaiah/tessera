"""Archetype-aware retrieval (A: narrow+filtered, C: broad multi-source)."""

from __future__ import annotations

from dataclasses import dataclass

from tessera.embedding.base import Embedder
from tessera.retrieval.router import Archetype
from tessera.store.base import SearchResult, VectorStore

# Lookup (A): the user wants "the document(s)," not a survey — a small,
# precise set of matches is the point. But "the document" is often a
# family (market-entry-overview + market-sizing + competitive-landscape +
# ...), and a raw top-k over chunks lets one strong document's chunks fill
# every slot, hiding its siblings (P2-4: q001/ql003/ql004 recall floor).
# So A pulls a wider candidate pool and returns one chunk per document —
# the top LOOKUP_TOP_K *distinct* documents, best chunk first.
LOOKUP_CANDIDATE_K = 30
LOOKUP_TOP_K = 5
LOOKUP_MAX_PER_DOCUMENT = 1
# Parent-document expansion (P4, fb001): the top LOOKUP_EXPAND_DOCUMENTS
# documents are shown whole, not as their single best chunk. A chunk's
# similarity to the question doesn't say whether it holds the answer —
# "Where's our worked example for price elasticity?" matched the
# reference page's Overview (0.62) better than the Framework section
# holding the example (0.59/0.55, raw rank 6-7), so one chunk per document
# never showed it. Recall is unchanged (same documents); the context the
# model reads grows (~2.3k -> ~9k chars on the A cases, 2026-10-02).
LOOKUP_EXPAND_DOCUMENTS = 2
# Upper bound on chunks fetched for one expanded document; corpus
# documents have 5-8.
EXPAND_MAX_CHUNKS_PER_DOCUMENT = 30

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
        candidates = store.query(
            query_embedding, k=LOOKUP_CANDIDATE_K, where=where
        )
        selected = _diversify_by_source(
            candidates,
            max_results=LOOKUP_TOP_K,
            max_per_document=LOOKUP_MAX_PER_DOCUMENT,
        )
        results = _expand_top_documents(
            selected, query_embedding, store, LOOKUP_EXPAND_DOCUMENTS, where
        )
    else:
        candidates = store.query(
            query_embedding, k=SYNTHESIS_CANDIDATE_K, where=where
        )
        results = _diversify_by_source(candidates)

    return RetrievalResult(query=query, archetype=archetype, results=results)


def _diversify_by_source(
    candidates: list[SearchResult],
    max_results: int = SYNTHESIS_MAX_RESULTS,
    max_per_document: int = SYNTHESIS_MAX_PER_DOCUMENT,
) -> list[SearchResult]:
    """Trim a best-match-first candidate list to max_results, capping how
    many chunks come from any one document so retrieval pulls from
    multiple sources instead of one dominant document.

    Used by both archetypes: C (synthesis) with the SYNTHESIS_* caps for a
    broad multi-source briefing, and A (lookup) with LOOKUP_MAX_PER_DOCUMENT
    = 1 to return the top LOOKUP_TOP_K distinct documents rather than
    LOOKUP_TOP_K chunks. The defaults are C's values, kept for
    backwards-compatible callers; A and evals/tune_retrieval.py's grid
    search pass their own. This function is the single source of truth for
    the diversification logic either way.
    """
    per_document_count: dict[str, int] = {}
    diversified: list[SearchResult] = []
    for result in candidates:
        count = per_document_count.get(result.document_path, 0)
        if count >= max_per_document:
            continue
        per_document_count[result.document_path] = count + 1
        diversified.append(result)
        if len(diversified) >= max_results:
            break
    return diversified


def _chunk_index(result: SearchResult) -> int:
    """Position of a chunk within its document, from its "<stem>::<n>" id."""
    try:
        return int(result.chunk_id.rsplit("::", 1)[1])
    except (IndexError, ValueError):
        return 0


def _document_filter(
    document_path: str, where: dict[str, object] | None
) -> dict[str, object]:
    """``where`` narrowed to one document — the caller's filter (e.g.
    P4-6's permissions) still applies to every expanded chunk.
    """
    only = {"document_path": document_path}
    return only if not where else {"$and": [where, only]}


def _expand_top_documents(
    selected: list[SearchResult],
    query_embedding: list[float],
    store: VectorStore,
    n_documents: int,
    where: dict[str, object] | None = None,
) -> list[SearchResult]:
    """Replace each of the first ``n_documents`` selected chunks with every
    chunk of its document, in document order; the rest stay as they are.
    Document order (best first) is unchanged, so document-level recall and
    MRR are too. Each chunk keeps its own similarity score, so the
    generation floor still drops a section that isn't relevant at all.
    """
    expanded: list[SearchResult] = []
    seen: set[str] = set()
    for i, result in enumerate(selected):
        chunks = [result]
        if i < n_documents:
            whole = store.query(
                query_embedding,
                k=EXPAND_MAX_CHUNKS_PER_DOCUMENT,
                where=_document_filter(result.document_path, where),
            )
            # Re-checked here, not trusted to the store's filter.
            chunks = [c for c in whole if c.document_path == result.document_path] or [result]
        for c in sorted(chunks, key=_chunk_index):
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                expanded.append(c)
    return expanded
