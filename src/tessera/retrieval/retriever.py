"""Archetype-aware retrieval (A: narrow+filtered, C: broad multi-source)."""

from __future__ import annotations

from dataclasses import dataclass, field

from tessera.embedding.base import Embedder
from tessera.ingestion.loader import (
    SENSITIVITY_INTERNAL,
    SENSITIVITY_RESTRICTED,
    STATUS_SUPERSEDED,
)
from tessera.principal import Principal
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

# Freshness (Phase 4, plan §3.3.2): superseded chunks never enter an A/C
# candidate set. "$ne" rather than equality on "current", so an index
# built before chunks carried a status still returns everything.
CURRENT_ONLY: dict[str, object] = {"status": {"$ne": STATUS_SUPERSEDED}}
SUPERSEDED_ONLY: dict[str, object] = {"status": STATUS_SUPERSEDED}


def permission_filter(principal: Principal | None) -> dict[str, object]:
    """The store-side filter for what ``principal`` may retrieve (plan
    §3.5.4): internal chunks, plus restricted chunks of engagements they
    are cleared for. Deny by default — no principal, or one cleared for
    nothing, sees internal chunks only, and a restricted chunk with no
    engagement matches nobody's list.
    """
    internal: dict[str, object] = {"sensitivity": SENSITIVITY_INTERNAL}
    cleared = sorted(principal.engagements) if principal else []
    if not cleared:
        return internal
    return {
        "$or": [
            internal,
            {"$and": [{"sensitivity": SENSITIVITY_RESTRICTED}, {"engagement": {"$in": cleared}}]},
        ]
    }


def _withheld_filter(principal: Principal | None) -> dict[str, object]:
    """The complement of permission_filter(): what it withholds. Used
    only to count removals for the trace — never to retrieve content.
    """
    not_internal: dict[str, object] = {"sensitivity": {"$ne": SENSITIVITY_INTERNAL}}
    cleared = sorted(principal.engagements) if principal else []
    if not cleared:
        return not_internal
    return {"$and": [not_internal, {"engagement": {"$nin": cleared}}]}


def is_permitted(chunk: SearchResult, principal: Principal | None) -> bool:
    """The same rule as permission_filter(), applied to a returned chunk
    — retrieval re-checks every result rather than trusting the store.
    """
    if chunk.sensitivity == SENSITIVITY_INTERNAL:
        return True
    return bool(principal and chunk.engagement and chunk.engagement in principal.engagements)


@dataclass(frozen=True)
class SupersededMatch:
    """A superseded document that would have ranked among the results
    had it not been excluded. ``score`` is its best chunk's similarity;
    ``superseded_by`` is the replacing document's path (document_path form).
    """

    document_path: str
    document_title: str
    superseded_by: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved chunks for one query, ready for grounded generation
    (Task 6) — carries the archetype alongside the results so the
    generation step (found-documents summary vs multi-source synthesis)
    knows which prompt shape to use without re-deriving it.

    ``superseded`` lists excluded superseded documents that would have
    ranked (generation says a newer version exists); ``removed`` counts
    chunks each exclusion filter kept out of the candidate pool, by filter
    name — what the trace reports (plan §3.2.1).
    """

    query: str
    archetype: Archetype
    results: list[SearchResult]
    superseded: tuple[SupersededMatch, ...] = ()
    removed: dict[str, int] = field(default_factory=dict)


def retrieve(
    query: str,
    archetype: Archetype,
    embedder: Embedder,
    store: VectorStore,
    where: dict[str, object] | None = None,
    principal: Principal | None = None,
) -> RetrievalResult:
    """Retrieve chunks for query using the strategy appropriate to archetype.

    Pure with respect to infrastructure per CLAUDE.md constraint #6:
    Embedder and VectorStore are injected, not constructed here. Only
    called for A and C — B and D are short-circuited by
    router.terminal_response_for() before retrieval would run.

    Permissions (P4-6): every store query carries permission_filter(
    principal), so a chunk the principal may not see never enters a
    candidate set — and every returned chunk is re-checked. ``removed``
    counts what the filter withheld from the candidate pool.

    Superseded documents (P4-4) are excluded in the store query. A second
    query finds the ones that would have ranked; they are reported on the
    result, and each one's current version is placed in its slot if it
    didn't rank on its own.
    """
    if archetype not in (Archetype.LOOKUP, Archetype.SYNTHESIS):
        raise ValueError(
            f"retrieve() only supports archetypes A and C, got {archetype.value!r}"
        )

    query_embedding = embedder.embed_query(query)
    lookup = archetype is Archetype.LOOKUP
    candidate_k = LOOKUP_CANDIDATE_K if lookup else SYNTHESIS_CANDIDATE_K
    max_results = LOOKUP_TOP_K if lookup else SYNTHESIS_MAX_RESULTS
    max_per_document = LOOKUP_MAX_PER_DOCUMENT if lookup else SYNTHESIS_MAX_PER_DOCUMENT

    caller_where = where
    # From here on every query carries the permission filter.
    where = _and(caller_where, permission_filter(principal))

    # Re-checked here, not trusted to the store's filter.
    candidates = [
        c
        for c in store.query(query_embedding, k=candidate_k, where=_and(where, CURRENT_ONLY))
        if c.status != STATUS_SUPERSEDED and is_permitted(c, principal)
    ]
    selected = _diversify_by_source(
        candidates, max_results=max_results, max_per_document=max_per_document
    )

    excluded = [
        c
        for c in store.query(query_embedding, k=candidate_k, where=_and(where, SUPERSEDED_ONLY))
        if c.status == STATUS_SUPERSEDED and c.superseded_by and is_permitted(c, principal)
    ]
    superseded = _superseded_matches(excluded, candidates, candidate_k, selected, max_results)
    selected = _with_replacements(selected, superseded, query_embedding, store, where, max_results)

    results = (
        _expand_top_documents(selected, query_embedding, store, LOOKUP_EXPAND_DOCUMENTS, where)
        if lookup
        else selected
    )
    # Last line of defence: nothing the principal may not see leaves here.
    results = [r for r in results if is_permitted(r, principal)]

    # Counted from a probe over only what the filter withholds; the chunks
    # themselves go no further than this count.
    withheld = store.query(
        query_embedding, k=candidate_k, where=_and(caller_where, _withheld_filter(principal))
    )
    return RetrievalResult(
        query=query,
        archetype=archetype,
        results=results,
        superseded=superseded,
        removed={
            "superseded": _removed_from_pool(excluded, candidates, candidate_k),
            "restricted": _removed_from_pool(
                [c for c in withheld if not is_permitted(c, principal)], candidates, candidate_k
            ),
        },
    )


def _and(
    where: dict[str, object] | None, extra: dict[str, object]
) -> dict[str, object]:
    """The caller's filter (e.g. P4-6's permissions) plus one more clause."""
    return extra if not where else {"$and": [where, extra]}


def _removed_from_pool(
    excluded: list[SearchResult], candidates: list[SearchResult], candidate_k: int
) -> int:
    """How many excluded chunks would have been in the candidate pool:
    all of them while the pool isn't full, else those scoring at least as
    well as its weakest member.
    """
    if len(candidates) < candidate_k:
        return len(excluded)
    weakest = candidates[-1].score
    return sum(1 for c in excluded if c.score >= weakest)


def _superseded_matches(
    excluded: list[SearchResult],
    candidates: list[SearchResult],
    candidate_k: int,
    selected: list[SearchResult],
    max_results: int,
) -> tuple[SupersededMatch, ...]:
    """Superseded documents whose best chunk would have made the selected
    results: it beats the weakest selected chunk — or, when the selection
    had room left, the weakest candidate, since it had to reach the pool
    first. Best first, one per document.
    """
    if len(selected) >= max_results:
        cutoff = selected[-1].score
    elif len(candidates) >= candidate_k:
        cutoff = candidates[-1].score
    else:
        cutoff = float("-inf")
    matches: dict[str, SupersededMatch] = {}
    for c in excluded:  # best first, so a document's first chunk is its best
        if c.score >= cutoff and c.document_path not in matches:
            matches[c.document_path] = SupersededMatch(
                document_path=c.document_path,
                document_title=c.document_title,
                superseded_by=c.superseded_by or "",
                score=c.score,
            )
    return tuple(matches.values())


def _with_replacements(
    selected: list[SearchResult],
    superseded: tuple[SupersededMatch, ...],
    query_embedding: list[float],
    store: VectorStore,
    where: dict[str, object] | None,
    max_results: int,
) -> list[SearchResult]:
    """Make sure the current version of each superseded match is in the
    results, so the answer can cite it: a replacement that didn't rank on
    its own takes the slot the superseded document would have held, and
    the list is trimmed back to max_results.
    """
    present = {r.document_path for r in selected}
    out = list(selected)
    for match in superseded:
        if match.superseded_by in present:
            continue
        best = [
            c
            for c in store.query(
                query_embedding, k=1, where=_document_filter(match.superseded_by, where)
            )
            if c.document_path == match.superseded_by and c.status != STATUS_SUPERSEDED
        ]
        if not best:
            continue  # the replacement isn't visible under the caller's filter
        slot = sum(1 for r in out if r.score > match.score)
        out.insert(slot, best[0])
        present.add(match.superseded_by)
    return out[:max_results]


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
