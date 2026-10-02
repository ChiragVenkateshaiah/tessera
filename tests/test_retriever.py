"""Unit tests for archetype-aware retrieval (fake Embedder/VectorStore, no
real model or Chroma instance) — proves the A-vs-C strategy difference
called for by build plan Task 5's acceptance check.
"""

import pytest

from tessera.embedding.base import Embedder
from tessera.retrieval.retriever import (
    LOOKUP_CANDIDATE_K,
    LOOKUP_EXPAND_DOCUMENTS,
    LOOKUP_TOP_K,
    SYNTHESIS_CANDIDATE_K,
    SYNTHESIS_MAX_PER_DOCUMENT,
    SYNTHESIS_MAX_RESULTS,
    retrieve,
)
from tessera.retrieval.router import Archetype
from tessera.store.base import SearchResult, VectorStore


class FakeEmbedder(Embedder):
    """Returns a canned vector regardless of input — retriever.py only
    needs *an* embedding to hand to the store, not a meaningful one.
    """

    @property
    def dimension(self) -> int:
        return 1

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def _result(chunk_id: str, document_path: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        score=score,
        document_path=document_path,
        document_title=document_path,
        doc_type="methodology",
        industry="cross-industry",
        topics=["t1"],
        date="2024-01-01",
        heading_path=("Overview",),
    )


def _matches(result: SearchResult, where: dict[str, object] | None) -> bool:
    """The subset of Chroma's where syntax retriever.py uses: a single
    {"document_path": ...} or {"$and": [...]} over such clauses; other keys
    (e.g. doc_type) aren't modelled and always match.
    """
    if not where:
        return True
    if "$and" in where:
        return all(_matches(result, w) for w in where["$and"])  # type: ignore[union-attr]
    path = where.get("document_path")
    return path is None or result.document_path == path


class FakeVectorStore(VectorStore):
    """Records the k/where it was called with and returns a canned,
    best-match-first candidate list spread across several documents —
    enough for diversification behavior to be observable.
    """

    def __init__(self, candidates: list[SearchResult]) -> None:
        self._candidates = candidates
        self.calls: list[tuple[int, dict[str, object] | None]] = []

    def add(self, chunks: object, embeddings: object) -> None:
        raise NotImplementedError

    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        self.calls.append((k, where))
        matching = [c for c in self._candidates if _matches(c, where)]
        return matching[:k]

    @property
    def first_k(self) -> int:
        return self.calls[0][0]

    def count(self) -> int:
        return len(self._candidates)


def _spread_candidates(n: int, docs: int) -> list[SearchResult]:
    """n candidates, best-match-first, round-robined across `docs` distinct
    documents so a document's chunks aren't all consecutive.
    """
    return [
        _result(f"doc{i % docs}::{i}", f"doc{i % docs}.md", score=1.0 - i * 0.01)
        for i in range(n)
    ]


def test_lookup_fetches_a_candidate_pool_then_narrows_to_top_k() -> None:
    store = FakeVectorStore(_spread_candidates(40, docs=20))

    result = retrieve(
        "do we have a market entry template?", Archetype.LOOKUP, FakeEmbedder(), store
    )

    # Fetches the wider pool from the store...
    assert store.first_k == LOOKUP_CANDIDATE_K
    # ...but the caller only ever sees the top LOOKUP_TOP_K distinct docs.
    assert len({r.document_path for r in result.results}) == LOOKUP_TOP_K


def _documents_in_order(results: list[SearchResult]) -> list[str]:
    seen: list[str] = []
    for r in results:
        if r.document_path not in seen:
            seen.append(r.document_path)
    return seen


def test_lookup_picks_distinct_documents_before_expanding() -> None:
    # 40 candidates round-robined across 20 docs (2 chunks each). The
    # documents are chosen one chunk each (P2-4: a strong document can't
    # hide its siblings), best first...
    store = FakeVectorStore(_spread_candidates(40, docs=20))

    result = retrieve("market entry framework", Archetype.LOOKUP, FakeEmbedder(), store)

    assert _documents_in_order(result.results) == [f"doc{i}.md" for i in range(LOOKUP_TOP_K)]
    # ...then the top LOOKUP_EXPAND_DOCUMENTS are shown whole, the rest as
    # their single best chunk.
    per_doc = {d: sum(r.document_path == d for r in result.results) for d in _documents_in_order(result.results)}
    assert list(per_doc.values()) == [2] * LOOKUP_EXPAND_DOCUMENTS + [1] * (
        LOOKUP_TOP_K - LOOKUP_EXPAND_DOCUMENTS
    )


def test_an_expanded_document_comes_whole_and_in_document_order() -> None:
    # 30 candidates across 3 docs, 10 chunks each; doc0 is the best match.
    store = FakeVectorStore(_spread_candidates(30, docs=3))

    result = retrieve("market entry framework", Archetype.LOOKUP, FakeEmbedder(), store)

    doc0 = [r.chunk_id for r in result.results if r.document_path == "doc0.md"]
    assert doc0 == [f"doc0::{i}" for i in range(0, 30, 3)]
    # The third document is outside the expansion: one chunk.
    assert sum(r.document_path == "doc2.md" for r in result.results) == 1


def test_expansion_keeps_each_chunks_own_score() -> None:
    store = FakeVectorStore(_spread_candidates(30, docs=3))

    result = retrieve("market entry framework", Archetype.LOOKUP, FakeEmbedder(), store)

    by_id = {c.chunk_id: c.score for c in _spread_candidates(30, docs=3)}
    assert all(r.score == by_id[r.chunk_id] for r in result.results)


def test_synthesis_uses_broader_candidate_k() -> None:
    store = FakeVectorStore(_spread_candidates(30, docs=10))

    retrieve("get me up to speed on pricing strategy", Archetype.SYNTHESIS, FakeEmbedder(), store)

    assert store.first_k == SYNTHESIS_CANDIDATE_K
    assert len(store.calls) == 1  # synthesis doesn't expand


def test_same_query_returns_visibly_different_breadth_under_a_vs_c() -> None:
    """The Task 5 acceptance check, directly: same query, same index,
    archetype is the only thing that changes — A and C must differ in how
    much and how broadly they retrieve.
    """
    candidates = _spread_candidates(30, docs=10)
    store_a = FakeVectorStore(candidates)
    store_c = FakeVectorStore(candidates)
    query = "what have we done on supply chain resilience?"

    lookup = retrieve(query, Archetype.LOOKUP, FakeEmbedder(), store_a)
    synthesis = retrieve(query, Archetype.SYNTHESIS, FakeEmbedder(), store_c)

    assert len(synthesis.results) > len(lookup.results)
    lookup_sources = {r.document_path for r in lookup.results}
    synthesis_sources = {r.document_path for r in synthesis.results}
    assert len(synthesis_sources) > len(lookup_sources)


def test_synthesis_caps_chunks_per_document() -> None:
    # All 30 candidates from the same document — diversification must
    # still cap it at SYNTHESIS_MAX_PER_DOCUMENT rather than returning
    # SYNTHESIS_MAX_RESULTS chunks from one source.
    candidates = _spread_candidates(30, docs=1)
    store = FakeVectorStore(candidates)

    result = retrieve("topic synthesis query", Archetype.SYNTHESIS, FakeEmbedder(), store)

    assert len(result.results) == SYNTHESIS_MAX_PER_DOCUMENT


def test_synthesis_trims_to_max_results_when_sources_are_plentiful() -> None:
    candidates = _spread_candidates(30, docs=10)
    store = FakeVectorStore(candidates)

    result = retrieve("topic synthesis query", Archetype.SYNTHESIS, FakeEmbedder(), store)

    assert len(result.results) == SYNTHESIS_MAX_RESULTS


def test_results_stay_best_match_first_after_diversification() -> None:
    candidates = _spread_candidates(30, docs=10)
    store = FakeVectorStore(candidates)

    result = retrieve("topic synthesis query", Archetype.SYNTHESIS, FakeEmbedder(), store)

    scores = [r.score for r in result.results]
    assert scores == sorted(scores, reverse=True)


def test_where_filter_passed_through_to_store() -> None:
    store = FakeVectorStore(_spread_candidates(10, docs=5))

    retrieve(
        "do we have a template for this?",
        Archetype.LOOKUP,
        FakeEmbedder(),
        store,
        where={"doc_type": "methodology"},
    )

    first_where = store.calls[0][1]
    assert first_where == {"doc_type": "methodology"}
    # Expansion narrows to one document but keeps the caller's filter
    # (P4-6's permission filter will ride on this).
    for _, where in store.calls[1:]:
        assert where["$and"][0] == {"doc_type": "methodology"}
        assert set(where["$and"][1]) == {"document_path"}


@pytest.mark.parametrize("archetype", [Archetype.EXPERTISE, Archetype.COMPARATIVE])
def test_retrieve_rejects_archetypes_that_never_reach_it(archetype: Archetype) -> None:
    store = FakeVectorStore(_spread_candidates(5, docs=5))

    with pytest.raises(ValueError):
        retrieve("some query", archetype, FakeEmbedder(), store)


def test_retrieval_result_carries_query_and_archetype() -> None:
    store = FakeVectorStore(_spread_candidates(5, docs=5))

    result = retrieve("do we have anything on churn?", Archetype.LOOKUP, FakeEmbedder(), store)

    assert result.query == "do we have anything on churn?"
    assert result.archetype is Archetype.LOOKUP


def test_expansion_ignores_chunks_of_other_documents_from_a_loose_store() -> None:
    class LooseStore(FakeVectorStore):
        def query(self, embedding, k, where=None):  # ignores the filter entirely
            self.calls.append((k, where))
            return self._candidates[:k]

    store = LooseStore(_spread_candidates(30, docs=10))

    result = retrieve("market entry framework", Archetype.LOOKUP, FakeEmbedder(), store)

    ids = [r.chunk_id for r in result.results]
    assert len(ids) == len(set(ids))  # nothing twice
    assert _documents_in_order(result.results) == [f"doc{i}.md" for i in range(LOOKUP_TOP_K)]
