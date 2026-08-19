"""Loads evals/cases/*.yaml, runs them through the pipeline, reports metrics.

`load_cases`, `run_case`, `run_harness`, and `format_report` are pure with
respect to infrastructure per CLAUDE.md constraint #6 — LLMClient/Embedder/
VectorStore are injected, nothing is constructed inside them, and they
return data rather than printing. Only `main()` (this file's actual
`python evals/harness.py` entry point) touches the environment, builds the
real corpus index, and prints — the same exemption `loader.py` gets for
ingestion I/O.

Uses `route()`, `retrieve()`, and `generate_answer()` directly rather than
`pipeline.answer_query()` — the harness needs the full ranked
`RetrievalResult` for recall@k/precision@k/MRR and the exact chunks shown
to the model for the groundedness judge, neither of which survives
`answer_query()`'s collapsed `AnswerResult` shape. These are the same
functions `answer_query()` itself composes, so this still exercises the
real query-answering path end to end.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from evals.metrics import (
    JudgeScore,
    judge_answer,
    mean,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from tessera.embedding.base import Embedder
from tessera.generation.answer import NO_RESULTS_MESSAGE, filter_relevant, generate_answer
from tessera.generation.base import LLMClient
from tessera.retrieval.retriever import retrieve
from tessera.retrieval.router import Archetype, route, terminal_response_for
from tessera.store.base import SearchResult, VectorStore

DEFAULT_K = 5


@dataclass(frozen=True)
class EvalCase:
    """One eval case, loaded from evals/cases/*.yaml (build plan §5 Task 7
    schema). relevant_sources are corpus-relative paths, e.g.
    "methodology/market-entry-overview.md". Empty for B/D cases, which
    never reach retrieval.
    """

    id: str
    query: str
    archetype: Archetype
    relevant_sources: list[str]
    ideal_answer: str


def load_cases(cases_dir: Path) -> list[EvalCase]:
    """Load every case from every *.yaml file in cases_dir.

    A file containing only comments (yaml.safe_load returns None) —
    like the placeholder case file before it's populated — contributes
    zero cases rather than erroring, so an empty scaffold is a valid,
    runnable state.
    """
    cases: list[EvalCase] = []
    for path in sorted(cases_dir.glob("*.yaml")):
        entries = yaml.safe_load(path.read_text()) or []
        for entry in entries:
            cases.append(
                EvalCase(
                    id=entry["id"],
                    query=entry["query"],
                    archetype=Archetype(entry["archetype"]),
                    relevant_sources=entry.get("relevant_sources") or [],
                    ideal_answer=entry.get("ideal_answer") or "",
                )
            )
    return cases


def unique_documents_by_rank(
    results: list[SearchResult], corpus_dir: Path
) -> list[str]:
    """Collapse a ranked chunk list to unique, corpus-relative document
    paths, keeping each document's best (first-seen) rank — multiple
    chunks from one document shouldn't inflate precision/recall the way
    distinct documents should.
    """
    seen: list[str] = []
    for result in results:
        relative = Path(result.document_path).relative_to(corpus_dir).as_posix()
        if relative not in seen:
            seen.append(relative)
    return seen


@dataclass(frozen=True)
class CaseResult:
    """actual_archetype/routing_correct are None only when `error` is set
    — the case raised before routing could even complete (e.g. a
    malformed LLM response), so there's nothing to report beyond the
    failure itself. run_harness excludes such cases from every
    aggregate rather than guessing.
    """

    case_id: str
    query: str
    expected_archetype: Archetype
    actual_archetype: Archetype | None
    routing_correct: bool | None
    retrieved_documents: list[str]
    recall: float | None
    precision: float | None
    reciprocal_rank_score: float | None
    answer: str
    judge: JudgeScore | None
    latency_seconds: float
    error: str | None = None


def run_case(
    case: EvalCase,
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    corpus_dir: Path,
    k: int = DEFAULT_K,
) -> CaseResult:
    """Run one eval case through routing, then retrieval + generation
    unless the routed archetype is terminal (B/D).
    """
    start = time.perf_counter()
    decision = route(case.query, llm)
    routing_correct = decision.archetype is case.archetype

    terminal = terminal_response_for(decision.archetype)
    if terminal is not None:
        return CaseResult(
            case_id=case.id,
            query=case.query,
            expected_archetype=case.archetype,
            actual_archetype=decision.archetype,
            routing_correct=routing_correct,
            retrieved_documents=[],
            recall=None,
            precision=None,
            reciprocal_rank_score=None,
            answer=terminal,
            judge=None,
            latency_seconds=time.perf_counter() - start,
        )

    retrieval = retrieve(case.query, decision.archetype, embedder, store)
    generated = generate_answer(retrieval, llm)
    latency = time.perf_counter() - start

    retrieved_documents = unique_documents_by_rank(retrieval.results, corpus_dir)
    recall = precision = reciprocal_rank_score = None
    if case.relevant_sources:
        relevant = set(case.relevant_sources)
        recall = recall_at_k(retrieved_documents, relevant, k)
        precision = precision_at_k(retrieved_documents, relevant, k)
        reciprocal_rank_score = reciprocal_rank(retrieved_documents, relevant)

    judge = None
    if case.ideal_answer and generated.answer != NO_RESULTS_MESSAGE:
        source_descriptions = [
            f"{r.document_title} — {' > '.join(r.heading_path)}\n{r.text}"
            for r in filter_relevant(retrieval.results)
        ]
        judge = judge_answer(
            case.query, case.ideal_answer, source_descriptions, generated.answer, llm
        )

    return CaseResult(
        case_id=case.id,
        query=case.query,
        expected_archetype=case.archetype,
        actual_archetype=decision.archetype,
        routing_correct=routing_correct,
        retrieved_documents=retrieved_documents,
        recall=recall,
        precision=precision,
        reciprocal_rank_score=reciprocal_rank_score,
        answer=generated.answer,
        judge=judge,
        latency_seconds=latency,
    )


@dataclass(frozen=True)
class EvalReport:
    case_results: list[CaseResult]
    routing_accuracy: float
    mean_recall: float | None
    mean_precision: float | None
    mean_reciprocal_rank: float | None
    mean_groundedness: float | None
    mean_relevance: float | None
    mean_latency_by_archetype: dict[Archetype, float]


def run_harness(
    cases: list[EvalCase],
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    corpus_dir: Path,
    k: int = DEFAULT_K,
) -> EvalReport:
    """Run every case and aggregate metrics across all of them.

    One case failing (a malformed LLM response mid-sweep, most likely
    from the judge — see JudgeError/RoutingError) does not abort the
    rest: on a 20-request/day Gemini budget, losing every already-
    completed case's worth of quota to one bad response later in the
    list would be far worse than recording that one case as failed and
    moving on. The failure is captured on the case's own CaseResult
    (`error` set, routing/metric fields None) and excluded from every
    aggregate below, rather than silently corrupting them.
    """
    case_results: list[CaseResult] = []
    for case in cases:
        try:
            case_results.append(run_case(case, llm, embedder, store, corpus_dir, k))
        except Exception as exc:
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    query=case.query,
                    expected_archetype=case.archetype,
                    actual_archetype=None,
                    routing_correct=None,
                    retrieved_documents=[],
                    recall=None,
                    precision=None,
                    reciprocal_rank_score=None,
                    answer="",
                    judge=None,
                    latency_seconds=0.0,
                    error=str(exc),
                )
            )

    routing_flags = [
        r.routing_correct for r in case_results if r.routing_correct is not None
    ]
    routing_accuracy = mean([1.0 if flag else 0.0 for flag in routing_flags])
    recalls = [r.recall for r in case_results if r.recall is not None]
    precisions = [r.precision for r in case_results if r.precision is not None]
    reciprocal_ranks = [
        r.reciprocal_rank_score
        for r in case_results
        if r.reciprocal_rank_score is not None
    ]
    groundedness_scores = [
        float(r.judge.groundedness) for r in case_results if r.judge is not None
    ]
    relevance_scores = [
        float(r.judge.relevance) for r in case_results if r.judge is not None
    ]

    latency_by_archetype: dict[Archetype, list[float]] = defaultdict(list)
    for r in case_results:
        if r.error is None:
            latency_by_archetype[r.expected_archetype].append(r.latency_seconds)

    return EvalReport(
        case_results=case_results,
        routing_accuracy=routing_accuracy,
        mean_recall=mean(recalls) if recalls else None,
        mean_precision=mean(precisions) if precisions else None,
        mean_reciprocal_rank=mean(reciprocal_ranks) if reciprocal_ranks else None,
        mean_groundedness=mean(groundedness_scores) if groundedness_scores else None,
        mean_relevance=mean(relevance_scores) if relevance_scores else None,
        mean_latency_by_archetype={
            archetype: mean(values)
            for archetype, values in latency_by_archetype.items()
        },
    )


def format_report(report: EvalReport) -> str:
    """Human-readable text summary — aggregate metrics, then per-case
    detail so a specific miss can be traced back to its case.
    """
    lines = [
        "=== Tessera Eval Report ===",
        f"Cases: {len(report.case_results)}",
        "",
        f"Routing accuracy: {report.routing_accuracy:.1%}",
        "",
    ]

    if report.mean_recall is not None:
        lines += [
            "Retrieval (archetypes A/C with relevant_sources):",
            f"  Mean recall:    {report.mean_recall:.2f}",
            f"  Mean precision: {report.mean_precision:.2f}",
            f"  Mean MRR:       {report.mean_reciprocal_rank:.2f}",
            "",
        ]

    if report.mean_groundedness is not None:
        lines += [
            "Generation quality (LLM-judge, 1-5):",
            f"  Mean groundedness: {report.mean_groundedness:.2f}",
            f"  Mean relevance:    {report.mean_relevance:.2f}",
            "",
        ]

    lines.append("Latency by archetype (mean seconds):")
    for archetype in Archetype:
        if archetype in report.mean_latency_by_archetype:
            lines.append(
                f"  {archetype.value}: {report.mean_latency_by_archetype[archetype]:.2f}"
            )
    lines.append("")

    lines.append("Per-case detail:")
    for r in report.case_results:
        if r.error is not None:
            lines.append(f"  [{r.case_id}] ERROR: {r.error}")
            continue
        routing_mark = "OK" if r.routing_correct else "MISROUTED"
        parts = [f"[{r.case_id}] {r.actual_archetype.value} routing={routing_mark}"]
        if r.recall is not None:
            parts.append(
                f"recall={r.recall:.2f} precision={r.precision:.2f} "
                f"rr={r.reciprocal_rank_score:.2f}"
            )
        if r.judge is not None:
            parts.append(
                f"groundedness={r.judge.groundedness} relevance={r.judge.relevance}"
            )
        parts.append(f"latency={r.latency_seconds:.2f}s")
        lines.append("  " + " ".join(parts))

    return "\n".join(lines)


def main() -> None:
    """Build a real (temporary, non-persisted) index over the pilot
    corpus and run every case in evals/cases/ against live Gemini.

    Run as `python -m evals.harness` from the repo root (not
    `python evals/harness.py` directly — that would put evals/ itself on
    sys.path instead of the repo root, breaking this module's absolute
    `from evals.metrics import ...` import). Reads GEMINI_API_KEY from
    the environment first (`set -a; source .env; set +a`, same as every
    other live check in this repo — see checkpoint.md).

    Parameterized via environment variables rather than hardcoded, per
    CLAUDE.md — but this function itself, unlike the rest of this
    module, is deliberately impure: it's the one place allowed to read
    the environment, hit the network, and print, mirroring loader.py's
    exemption for ingestion I/O.
    """
    import os
    import tempfile

    from tessera.embedding.local import LocalEmbedder
    from tessera.generation.gemini import GeminiClient
    from tessera.ingestion.chunker import chunk_corpus
    from tessera.ingestion.loader import load_corpus
    from tessera.store.chroma import ChromaVectorStore

    corpus_dir = Path(os.environ.get("TESSERA_CORPUS_DIR", "data/corpus"))
    cases_dir = Path(__file__).parent / "cases"
    api_key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

    docs = load_corpus(corpus_dir)
    chunks = chunk_corpus(docs)
    embedder = LocalEmbedder()
    embeddings = embedder.embed_documents([c.text for c in chunks])

    with tempfile.TemporaryDirectory() as persist_dir:
        store = ChromaVectorStore(persist_dir=Path(persist_dir))
        store.add(chunks, embeddings)
        llm = GeminiClient(api_key=api_key, model=model)

        cases = load_cases(cases_dir)
        report = run_harness(cases, llm, embedder, store, corpus_dir)
        print(format_report(report))


if __name__ == "__main__":
    main()
