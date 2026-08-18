"""Unit tests for archetype-aware retrieval (fake Embedder/VectorStore, no
real model or Chroma instance) — proves the A-vs-C strategy difference
called for by build plan Task 5's acceptance check.
"""

import pytest

from tessera.embedding.base import Embedder
from tessera.retrieval.retriever import (
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


class FakeVectorStore(VectorStore):
    """Records the k/where it was called with and returns a canned,
    best-match-first candidate list spread across several documents —
    enough for diversification behavior to be observable.
    """

    def __init__(self, candidates: list[SearchResult]) -> None:
        self._candidates = candidates
        self.last_k: int | None = None
        self.last_where: dict[str, object] | None = None

    def add(self, chunks: object, embeddings: object) -> None:
        raise NotImplementedError

    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        self.last_k = k
        self.last_where = where
        return self._candidates[:k]

    def count(self) -> int:
        return len(self._candidates)


def _spread_candidates(n: int, docs: int) -> list[SearchResult]:
    """n candidates, best-match-first, round-robined across `docs` distinct
    documents so a document's chunks aren't all consecutive.
    """
    return [
        _result(f"c{i}", f"doc{i % docs}.md", score=1.0 - i * 0.01) for i in range(n)
    ]


def test_lookup_uses_narrow_k() -> None:
    store = FakeVectorStore(_spread_candidates(30, docs=10))

    retrieve("do we have a market entry template?", Archetype.LOOKUP, FakeEmbedder(), store)

    assert store.last_k == LOOKUP_TOP_K


def test_synthesis_uses_broader_candidate_k() -> None:
    store = FakeVectorStore(_spread_candidates(30, docs=10))

    retrieve("get me up to speed on pricing strategy", Archetype.SYNTHESIS, FakeEmbedder(), store)

    assert store.last_k == SYNTHESIS_CANDIDATE_K


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

    assert store.last_where == {"doc_type": "methodology"}


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
