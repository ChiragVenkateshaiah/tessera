"""Unit tests for evals/tune_retrieval.py (fake Embedder/VectorStore, no
network, no live index) — grid generation, scoring, selection, and
report formatting.
"""

from pathlib import Path

from evals.harness import EvalCase
from evals.tune_retrieval import (
    CURRENT_CONFIG,
    FetchedCase,
    RetrievalConfig,
    fetch_candidates,
    format_report,
    grid,
    score_config,
    select_best,
    widest_k,
)
from tessera.embedding.base import Embedder
from tessera.retrieval.router import Archetype
from tessera.store.base import SearchResult, VectorStore

CORPUS_DIR = Path("data/corpus")


def _result(document_path: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=f"{document_path}#1",
        text=f"text for {document_path}",
        score=score,
        document_path=document_path,
        document_title="Doc",
        doc_type="methodology",
        industry="cross-industry",
        topics=["t1"],
        date="2024-01-01",
        heading_path=("Overview",),
    )


class FakeEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 1

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class FakeVectorStore(VectorStore):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_k: int | None = None

    def add(self, chunks: object, embeddings: object) -> None:
        raise NotImplementedError

    def query(
        self, embedding: list[float], k: int, where: dict[str, object] | None = None
    ) -> list[SearchResult]:
        self.last_k = k
        return self._results[:k]

    def count(self) -> int:
        return len(self._results)


# --- grid() ---


def test_grid_has_72_points_including_current_config() -> None:
    configs = grid()

    assert len(configs) == 4 * 3 * 3 * 2
    assert CURRENT_CONFIG in configs


# --- fetch_candidates() ---


def test_fetch_candidates_only_fetches_ac_cases_at_widest_k() -> None:
    store = FakeVectorStore([_result("data/corpus/a.md")] * 30)
    cases = [
        EvalCase("a1", "q", Archetype.LOOKUP, ["a.md"], "ideal"),
        EvalCase("b1", "q", Archetype.EXPERTISE, [], ""),
        EvalCase("c1", "q", Archetype.SYNTHESIS, ["a.md"], "ideal"),
        EvalCase("d1", "q", Archetype.COMPARATIVE, [], ""),
    ]

    fetched = fetch_candidates(cases, FakeEmbedder(), store)

    assert [f.case.id for f in fetched] == ["a1", "c1"]
    assert store.last_k == widest_k()
    assert all(len(f.candidates) == widest_k() for f in fetched)


# --- score_config() ---


def test_score_config_lookup_slices_to_lookup_top_k() -> None:
    candidates = [
        _result("data/corpus/methodology/target.md"),
        _result("data/corpus/methodology/other1.md"),
        _result("data/corpus/methodology/other2.md"),
    ]
    case = EvalCase("q1", "q", Archetype.LOOKUP, ["methodology/target.md"], "ideal")
    fetched = [FetchedCase(case=case, candidates=candidates)]
    config = RetrievalConfig(
        lookup_top_k=1, synthesis_candidate_k=20, synthesis_max_results=10,
        synthesis_max_per_document=2,
    )

    score = score_config(fetched, config, CORPUS_DIR)

    assert score.mean_recall == 1.0
    assert score.mean_precision == 1.0
    assert score.mean_reciprocal_rank == 1.0
    assert score.min_recall == 1.0
    assert score.objective == 1.0
    assert score.feasible is True


def test_score_config_synthesis_diversifies_with_configured_caps() -> None:
    # 3 chunks from the same doc + 1 from another — with max_per_document=1
    # only the first chunk of the dominant doc plus the other doc survive.
    candidates = [
        _result("data/corpus/methodology/dominant.md"),
        _result("data/corpus/methodology/dominant.md"),
        _result("data/corpus/methodology/dominant.md"),
        _result("data/corpus/methodology/other.md"),
    ]
    case = EvalCase(
        "c1", "q", Archetype.SYNTHESIS,
        ["methodology/dominant.md", "methodology/other.md"], "ideal",
    )
    fetched = [FetchedCase(case=case, candidates=candidates)]
    config = RetrievalConfig(
        lookup_top_k=5, synthesis_candidate_k=20, synthesis_max_results=10,
        synthesis_max_per_document=1,
    )

    score = score_config(fetched, config, CORPUS_DIR)

    assert score.mean_recall == 1.0  # both docs still surface, one chunk each


def test_score_config_marks_infeasible_on_total_miss_or_low_mrr() -> None:
    case = EvalCase("q1", "q", Archetype.LOOKUP, ["methodology/missing.md"], "ideal")
    fetched = [FetchedCase(case=case, candidates=[_result("data/corpus/methodology/other.md")])]
    config = RetrievalConfig(5, 20, 10, 2)

    score = score_config(fetched, config, CORPUS_DIR)

    assert score.min_recall == 0.0
    assert score.feasible is False


# --- select_best() ---


def _score(config: RetrievalConfig, objective: float, feasible: bool = True) -> "type":
    from evals.tune_retrieval import ConfigScore

    return ConfigScore(
        config=config, mean_recall=0.9, mean_precision=0.8,
        mean_reciprocal_rank=0.95, min_recall=0.5, objective=objective,
        feasible=feasible,
    )


def test_select_best_keeps_current_when_improvement_is_marginal() -> None:
    better = RetrievalConfig(3, 20, 10, 2)
    scores = [
        _score(CURRENT_CONFIG, objective=0.70),
        _score(better, objective=0.71),  # +0.01, below the 0.02 tiebreak
    ]

    assert select_best(scores) == CURRENT_CONFIG


def test_select_best_switches_when_improvement_clears_tiebreak() -> None:
    better = RetrievalConfig(3, 20, 10, 2)
    scores = [
        _score(CURRENT_CONFIG, objective=0.70),
        _score(better, objective=0.73),  # +0.03, above the 0.02 tiebreak
    ]

    assert select_best(scores) == better


def test_select_best_ignores_infeasible_configs() -> None:
    better_but_infeasible = RetrievalConfig(3, 20, 10, 2)
    scores = [
        _score(CURRENT_CONFIG, objective=0.70),
        _score(better_but_infeasible, objective=0.99, feasible=False),
    ]

    assert select_best(scores) == CURRENT_CONFIG


def test_select_best_falls_back_to_current_when_nothing_feasible() -> None:
    scores = [_score(CURRENT_CONFIG, objective=0.70, feasible=False)]

    assert select_best(scores) == CURRENT_CONFIG


# --- format_report() ---


def test_format_report_includes_current_and_winner() -> None:
    scores = [
        _score(CURRENT_CONFIG, objective=0.70),
        _score(RetrievalConfig(3, 20, 10, 2), objective=0.71),
    ]

    text = format_report(scores, winner=CURRENT_CONFIG)

    assert "current" in text
    assert "winner   = current" in text
    assert "top 5 feasible" in text
