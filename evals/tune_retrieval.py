"""Retrieval-only grid search over the archetype-aware retrieval
constants in retriever.py — Phase 2 task P2-3
(docs/Tessera_Phase2_Plan.md §4).

Archetype comes from the case file, not the router, and there is no
LLM-judge step — this is why the full grid runs in seconds rather than
requiring hours of live NVIDIA calls: the entire objective is computed
from a single pre-fetched candidate pool per case, re-sliced and
re-diversified in memory for every grid point.

Tunes against evals/cases/query_log.yaml only; evals/cases/placeholder.yaml
is the held-out overfitting check-set (see its and query_log.yaml's
header comments, and checkpoint.md's 2026-09-04 P2-2 entry) — the chosen
config is reported against both, but only query_log.yaml drives the
selection.

`RetrievalConfig`/`ConfigScore`/`score_config`/`grid`/`select_best` are
pure with respect to infrastructure per CLAUDE.md constraint #6 — no I/O,
no env reads, return data. `fetch_candidates()` does the one round of
real I/O (embedding + vector-store queries) the grid search needs; `main()`
is the impure composition root, mirroring harness.py's main().
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

from tessera.embedding.base import Embedder
from tessera.retrieval.retriever import (
    LOOKUP_TOP_K,
    SYNTHESIS_CANDIDATE_K,
    SYNTHESIS_MAX_PER_DOCUMENT,
    SYNTHESIS_MAX_RESULTS,
    _diversify_by_source,
)
from tessera.retrieval.router import Archetype
from tessera.store.base import SearchResult, VectorStore

from evals.harness import EvalCase, unique_documents_by_rank
from evals.metrics import mean, precision_at_k, reciprocal_rank, recall_at_k

# Matches evals.harness.DEFAULT_K — the same cutoff the eval report's
# recall@k/precision@k are computed at, so this script's numbers are
# directly comparable to a `tessera eval` report.
METRIC_K = 5

# The grid (plan §4 P2-3). Each *_GRID includes retriever.py's current
# value, so "current" is always one of the scored points, not a separate
# case — select_best() below relies on that.
LOOKUP_TOP_K_GRID = (3, 5, 7, 10)
SYNTHESIS_CANDIDATE_K_GRID = (15, 20, 30)
SYNTHESIS_MAX_RESULTS_GRID = (8, 10, 12)
SYNTHESIS_MAX_PER_DOCUMENT_GRID = (2, 3)

# Feasibility floor (evals/QUALITY_BAR.md: mean MRR >= 0.90, gated).
MIN_MEAN_RECIPROCAL_RANK = 0.90

# Plan §4 P2-3: "prefer current values when improvement is marginal
# (< 0.02)" — avoids chasing noise on a ~50-case eval set.
IMPROVEMENT_TIEBREAK = 0.02


@dataclass(frozen=True)
class RetrievalConfig:
    lookup_top_k: int
    synthesis_candidate_k: int
    synthesis_max_results: int
    synthesis_max_per_document: int


CURRENT_CONFIG = RetrievalConfig(
    lookup_top_k=LOOKUP_TOP_K,
    synthesis_candidate_k=SYNTHESIS_CANDIDATE_K,
    synthesis_max_results=SYNTHESIS_MAX_RESULTS,
    synthesis_max_per_document=SYNTHESIS_MAX_PER_DOCUMENT,
)


@dataclass(frozen=True)
class FetchedCase:
    """One A/C case's widest-possible candidate pool, fetched once. Every
    grid point re-slices/re-diversifies this same list — no store query
    per grid point.
    """

    case: EvalCase
    candidates: list[SearchResult]


@dataclass(frozen=True)
class ConfigScore:
    config: RetrievalConfig
    mean_recall: float
    mean_precision: float
    mean_reciprocal_rank: float
    min_recall: float
    objective: float  # mean(recall_i * precision_i), plan §4's stated objective
    feasible: bool  # min_recall > 0.0 and mean_reciprocal_rank >= floor


def grid() -> list[RetrievalConfig]:
    """Every combination in the plan §4 grid (72 points)."""
    return [
        RetrievalConfig(l, ck, mr, mpd)
        for l, ck, mr, mpd in itertools.product(
            LOOKUP_TOP_K_GRID,
            SYNTHESIS_CANDIDATE_K_GRID,
            SYNTHESIS_MAX_RESULTS_GRID,
            SYNTHESIS_MAX_PER_DOCUMENT_GRID,
        )
    ]


def widest_k() -> int:
    """The candidate pool size that satisfies every grid point at once —
    the max of every LOOKUP_TOP_K/SYNTHESIS_CANDIDATE_K value in the grid.
    """
    return max(max(LOOKUP_TOP_K_GRID), max(SYNTHESIS_CANDIDATE_K_GRID))


def fetch_candidates(
    cases: list[EvalCase], embedder: Embedder, store: VectorStore
) -> list[FetchedCase]:
    """Fetch every A/C case's widest candidate pool once. The one real
    I/O step in this module (embedding + a vector-store query per case).
    """
    fetched = []
    for case in cases:
        if case.archetype not in (Archetype.LOOKUP, Archetype.SYNTHESIS):
            continue
        embedding = embedder.embed_query(case.query)
        candidates = store.query(embedding, k=widest_k())
        fetched.append(FetchedCase(case=case, candidates=candidates))
    return fetched


def score_config(
    fetched: list[FetchedCase],
    config: RetrievalConfig,
    corpus_dir: Path,
    k: int = METRIC_K,
) -> ConfigScore:
    """Score one retrieval config against pre-fetched candidates. Pure —
    no I/O, no re-querying the store; this is what makes a 72-point grid
    run in seconds.
    """
    recalls: list[float] = []
    precisions: list[float] = []
    rrs: list[float] = []

    for item in fetched:
        case = item.case
        if case.archetype is Archetype.LOOKUP:
            results = item.candidates[: config.lookup_top_k]
        else:
            narrowed = item.candidates[: config.synthesis_candidate_k]
            results = _diversify_by_source(
                narrowed,
                max_results=config.synthesis_max_results,
                max_per_document=config.synthesis_max_per_document,
            )
        docs = unique_documents_by_rank(results, corpus_dir)
        relevant = set(case.relevant_sources)
        recalls.append(recall_at_k(docs, relevant, k))
        precisions.append(precision_at_k(docs, relevant, k))
        rrs.append(reciprocal_rank(docs, relevant))

    mean_recall = mean(recalls)
    mean_rr = mean(rrs)
    min_recall = min(recalls) if recalls else 0.0
    objective = mean([r * p for r, p in zip(recalls, precisions)])
    return ConfigScore(
        config=config,
        mean_recall=mean_recall,
        mean_precision=mean(precisions),
        mean_reciprocal_rank=mean_rr,
        min_recall=min_recall,
        objective=objective,
        feasible=min_recall > 0.0 and mean_rr >= MIN_MEAN_RECIPROCAL_RANK,
    )


def select_best(
    scores: list[ConfigScore],
    current: RetrievalConfig = CURRENT_CONFIG,
    tiebreak: float = IMPROVEMENT_TIEBREAK,
) -> RetrievalConfig:
    """Highest-objective feasible config, but keep `current` unless the
    best feasible config beats it by more than `tiebreak` (plan §4 P2-3).
    Returns `current` if nothing is feasible or `current` isn't in
    `scores` at all (defensive — CURRENT_CONFIG is always one of the
    grid points in practice, since every *_GRID includes it).
    """
    feasible = [s for s in scores if s.feasible]
    if not feasible:
        return current

    best = max(feasible, key=lambda s: s.objective)
    current_score = next((s for s in scores if s.config == current), None)
    if current_score is not None and best.objective - current_score.objective < tiebreak:
        return current
    return best.config


def format_report(scores: list[ConfigScore], winner: RetrievalConfig) -> str:
    """Human-readable summary: current config's score, the winner (which
    may be the same config), and the top few feasible alternatives.
    """
    by_config = {s.config: s for s in scores}
    current_score = by_config.get(CURRENT_CONFIG)
    winner_score = by_config.get(winner)
    feasible = sorted(
        (s for s in scores if s.feasible), key=lambda s: -s.objective
    )

    lines = ["=== Retrieval Constant Grid Search ==="]
    lines.append(f"{len(scores)} configs scored, {len(feasible)} feasible "
                 f"(min_recall>0, mean_MRR>={MIN_MEAN_RECIPROCAL_RANK})")
    lines.append("")
    if current_score is not None:
        lines.append(f"current  {CURRENT_CONFIG}: obj={current_score.objective:.4f} "
                      f"recall={current_score.mean_recall:.3f} "
                      f"precision={current_score.mean_precision:.3f} "
                      f"mrr={current_score.mean_reciprocal_rank:.3f} "
                      f"min_recall={current_score.min_recall:.2f}")
    if winner == CURRENT_CONFIG:
        lines.append("winner   = current (no feasible config beat it by "
                      f">= {IMPROVEMENT_TIEBREAK})")
    elif winner_score is not None:
        lines.append(f"winner   {winner}: obj={winner_score.objective:.4f} "
                      f"recall={winner_score.mean_recall:.3f} "
                      f"precision={winner_score.mean_precision:.3f} "
                      f"mrr={winner_score.mean_reciprocal_rank:.3f} "
                      f"min_recall={winner_score.min_recall:.2f}")
    lines.append("")
    lines.append("top 5 feasible by objective:")
    for s in feasible[:5]:
        tag = "  <- current" if s.config == CURRENT_CONFIG else ""
        lines.append(f"  obj={s.objective:.4f} recall={s.mean_recall:.3f} "
                      f"precision={s.mean_precision:.3f} mrr={s.mean_reciprocal_rank:.3f} "
                      f"min_recall={s.min_recall:.2f} {s.config}{tag}")
    return "\n".join(lines)


def main() -> None:
    """Grid-search against query_log.yaml (tuning set), then report the
    winning config's score on placeholder.yaml (the held-out check-set)
    too. Impure composition root — real corpus, embedder, persisted
    index — mirroring evals/harness.py's main().
    """
    from tessera.config import Settings
    from tessera.embedding.local import LocalEmbedder
    from tessera.store.chroma import ChromaVectorStore

    from evals.harness import load_cases

    settings = Settings()
    embedder = LocalEmbedder()
    store = ChromaVectorStore(persist_dir=settings.vectorstore_dir)

    cases_dir = Path(__file__).parent / "cases"
    all_cases = load_cases(cases_dir)
    query_log_cases = [c for c in all_cases if c.id.startswith("ql")]
    placeholder_cases = [c for c in all_cases if not c.id.startswith("ql")]

    tuning_pool = fetch_candidates(query_log_cases, embedder, store)
    check_pool = fetch_candidates(placeholder_cases, embedder, store)

    scores = [
        score_config(tuning_pool, config, settings.corpus_dir)
        for config in grid()
    ]
    winner = select_best(scores)

    print(format_report(scores, winner))
    print()
    winner_on_checkset = score_config(check_pool, winner, settings.corpus_dir)
    current_on_checkset = score_config(check_pool, CURRENT_CONFIG, settings.corpus_dir)
    print("placeholder.yaml (held-out check-set):")
    print(f"  current config: obj={current_on_checkset.objective:.4f} "
          f"recall={current_on_checkset.mean_recall:.3f} "
          f"min_recall={current_on_checkset.min_recall:.2f}")
    print(f"  winner  config: obj={winner_on_checkset.objective:.4f} "
          f"recall={winner_on_checkset.mean_recall:.3f} "
          f"min_recall={winner_on_checkset.min_recall:.2f}")


if __name__ == "__main__":
    main()
