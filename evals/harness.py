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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from evals.metrics import (
    EXPERTISE_JUDGE_SYSTEM_PROMPT,
    JudgeScore,
    judge_answer,
    mean,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
    shortlist_recall_at_k,
)
from tessera.embedding.base import Embedder
from tessera.generation.answer import NO_RESULTS_MESSAGE, filter_relevant, generate_answer
from tessera.generation.base import LLMClient
from tessera.generation.expertise import NO_EXPERT_MESSAGE, generate_expertise_answer
from tessera.generation.prompts import format_person_record
from tessera.retrieval.expertise import find_experts
from tessera.retrieval.retriever import retrieve
from tessera.retrieval.router import Archetype, route, terminal_response_for
from tessera.store.base import ExpertiseStore, SearchResult, VectorStore

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
    # Archetype B: person_ids (from data/expertise/) who belong in the
    # shortlist. Empty for A/C/D and for no-match cases.
    relevant_people: list[str] = field(default_factory=list)
    # Archetype B no-match set: the query asks for expertise the firm
    # doesn't have; the correct outcome is the fixed no-match message with
    # zero generation LLM calls (evals/cases/expertise_nomatch.yaml).
    expect_no_match: bool = False


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
                    relevant_people=entry.get("relevant_people") or [],
                    expect_no_match=bool(entry.get("expect_no_match", False)),
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
    # Archetype B (kept apart from the document fields above so A/C means
    # are never blended with B).
    retrieved_people: list[str] = field(default_factory=list)
    person_recall: float | None = None
    person_precision: float | None = None
    person_reciprocal_rank: float | None = None
    expertise_judge: JudgeScore | None = None
    # No-match set only: True iff the fixed no-match message came back with
    # zero generation LLM calls. False also when the case misrouted.
    no_match_correct: bool | None = None


EXPERTISE_NOT_SCORED_NOTE = (
    "(archetype B: routing checked only — no expertise store was supplied "
    "to the harness, so the answer was not generated or scored)"
)


class _CountingLLM(LLMClient):
    """Delegates to the real client and counts calls — how the no-match
    set proves 'zero LLM calls' rather than assuming it.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.calls = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls += 1
        return self._inner.complete(system=system, user=user, temperature=temperature)


def _run_expertise_case(
    case: EvalCase,
    decision_archetype: Archetype,
    routing_correct: bool,
    llm: LLMClient,
    embedder: Embedder,
    expertise_store: ExpertiseStore,
    k: int,
    start: float,
) -> CaseResult:
    """Archetype B: find_experts -> generate_expertise_answer (the same
    two functions pipeline.answer_query() composes), scored on the people
    the answer actually presents.
    """
    result = find_experts(case.query, embedder, expertise_store, k=k)
    counting = _CountingLLM(llm)
    generated = generate_expertise_answer(result, counting)
    latency = time.perf_counter() - start

    presented = generated.experts
    retrieved_people = [m.person.person_id for m in presented]

    person_recall = person_precision = person_rr = None
    if case.relevant_people:
        relevant = set(case.relevant_people)
        person_recall = shortlist_recall_at_k(retrieved_people, relevant, k)
        person_precision = precision_at_k(retrieved_people, relevant, k)
        person_rr = reciprocal_rank(retrieved_people, relevant)

    no_match_correct = None
    if case.expect_no_match:
        no_match_correct = generated.answer == NO_EXPERT_MESSAGE and counting.calls == 0

    judge = None
    if case.ideal_answer and not case.expect_no_match and presented:
        records = [format_person_record(i, m) for i, m in enumerate(presented, start=1)]
        judge = judge_answer(
            case.query,
            case.ideal_answer,
            records,
            generated.answer,
            llm,
            system=EXPERTISE_JUDGE_SYSTEM_PROMPT,
        )

    return CaseResult(
        case_id=case.id,
        query=case.query,
        expected_archetype=case.archetype,
        actual_archetype=decision_archetype,
        routing_correct=routing_correct,
        retrieved_documents=[],
        recall=None,
        precision=None,
        reciprocal_rank_score=None,
        answer=generated.answer,
        judge=None,
        latency_seconds=latency,
        retrieved_people=retrieved_people,
        person_recall=person_recall,
        person_precision=person_precision,
        person_reciprocal_rank=person_rr,
        expertise_judge=judge,
        no_match_correct=no_match_correct,
    )


def run_case(
    case: EvalCase,
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    corpus_dir: Path,
    k: int = DEFAULT_K,
    expertise_store: ExpertiseStore | None = None,
) -> CaseResult:
    """Run one eval case through routing, then the archetype's path: D is
    terminal, B runs expertise retrieval + generation (needs
    expertise_store; without one a B case is scored on routing only), and
    A/C run document retrieval + generation.
    """
    start = time.perf_counter()
    decision = route(case.query, llm)
    routing_correct = decision.archetype is case.archetype

    terminal = terminal_response_for(decision.archetype)
    if decision.archetype is Archetype.EXPERTISE:
        if expertise_store is not None:
            return _run_expertise_case(
                case,
                decision.archetype,
                routing_correct,
                llm,
                embedder,
                expertise_store,
                k,
                start,
            )
        terminal = EXPERTISE_NOT_SCORED_NOTE
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
            no_match_correct=False if case.expect_no_match else None,
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
        no_match_correct=False if case.expect_no_match else None,
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
    # Archetype B (Phase 3). `expertise_scored` is True iff the run had an
    # expertise store, i.e. B was actually attempted; the B rows of the
    # quality bar only apply then (a run with no people index has not
    # tried B, as opposed to a run that tried and got no value).
    expertise_scored: bool = False
    mean_person_recall: float | None = None
    mean_person_precision: float | None = None
    mean_person_mrr: float | None = None
    mean_expertise_groundedness: float | None = None
    mean_expertise_relevance: float | None = None
    no_match_rate: float | None = None


def run_harness(
    cases: list[EvalCase],
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    corpus_dir: Path,
    k: int = DEFAULT_K,
    expertise_store: ExpertiseStore | None = None,
    on_case_complete: Callable[[int, int, CaseResult], None] | None = None,
) -> EvalReport:
    """Run every case and aggregate metrics across all of them.

    on_case_complete(done, total, result) is called after each case
    (errored ones included) so a long sweep can report progress; this
    function stays pure and never prints itself.

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
    for done, case in enumerate(cases, start=1):
        try:
            case_results.append(
                run_case(case, llm, embedder, store, corpus_dir, k, expertise_store)
            )
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
        if on_case_complete is not None:
            on_case_complete(done, len(cases), case_results[-1])

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

    person_recalls = [r.person_recall for r in case_results if r.person_recall is not None]
    person_precisions = [
        r.person_precision for r in case_results if r.person_precision is not None
    ]
    person_rrs = [
        r.person_reciprocal_rank
        for r in case_results
        if r.person_reciprocal_rank is not None
    ]
    expertise_groundedness = [
        float(r.expertise_judge.groundedness)
        for r in case_results
        if r.expertise_judge is not None
    ]
    expertise_relevance = [
        float(r.expertise_judge.relevance)
        for r in case_results
        if r.expertise_judge is not None
    ]
    no_match_flags = [
        r.no_match_correct for r in case_results if r.no_match_correct is not None
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
        expertise_scored=expertise_store is not None,
        mean_person_recall=mean(person_recalls) if person_recalls else None,
        mean_person_precision=mean(person_precisions) if person_precisions else None,
        mean_person_mrr=mean(person_rrs) if person_rrs else None,
        mean_expertise_groundedness=(
            mean(expertise_groundedness) if expertise_groundedness else None
        ),
        mean_expertise_relevance=(
            mean(expertise_relevance) if expertise_relevance else None
        ),
        no_match_rate=(
            mean([1.0 if f else 0.0 for f in no_match_flags]) if no_match_flags else None
        ),
    )


# --- Quality bar (Phase 2) ---
#
# The agreed internal bar from docs/Tessera_Phase2_Plan.md §2 / evals/
# QUALITY_BAR.md. Gated thresholds block a release and fail
# `tessera eval --check`; precision@k is reported but not gated (a
# labeling-completeness confound makes a low score ambiguous — see
# evals/QUALITY_BAR.md). Kept here, next to the metrics they check, and
# pure like the rest of this module.


@dataclass(frozen=True)
class QualityBar:
    min_routing_accuracy: float = 0.95
    min_mean_recall: float = 0.80
    min_mean_reciprocal_rank: float = 0.90
    min_mean_groundedness: float = 4.5
    min_mean_relevance: float = 4.5
    # Archetype B (Phase 3, docs/Tessera_Phase3_Plan.md §4.2). Person
    # recall is stricter than A/C's 0.80: a wrong shortlist sends the user
    # to the wrong person entirely.
    min_person_recall: float = 0.90
    min_person_mrr: float = 0.90
    min_expertise_groundedness: float = 4.5
    min_expertise_relevance: float = 4.5
    min_no_match_rate: float = 1.0
    # The B thresholds above are GATED. They entered provisional (reported,
    # not gated) for the first Phase 3 sweep (plan §4.2): person recall was
    # 0.89 vs 0.90, carried by ql019 ("who LED ... recently"). The
    # lead/recency-intent pass in retrieval/expertise.py lifted it, and this
    # switch was flipped in the same change. Set False to fall back to
    # report-only.
    gate_expertise: bool = True


DEFAULT_QUALITY_BAR = QualityBar()


@dataclass(frozen=True)
class ThresholdResult:
    """One row of the bar check. `passed` is always meaningful; the
    formatter only renders PASS/FAIL for gated rows.
    """

    name: str
    gated: bool
    requirement: str
    actual: str
    passed: bool


@dataclass(frozen=True)
class BarResult:
    thresholds: list[ThresholdResult]
    passed: bool  # every *gated* threshold passed

    @property
    def gated_failures(self) -> list[ThresholdResult]:
        return [t for t in self.thresholds if t.gated and not t.passed]


def evaluate_bar(
    report: EvalReport, bar: QualityBar = DEFAULT_QUALITY_BAR
) -> BarResult:
    """Check an EvalReport against the quality bar. Pure — no I/O.

    A gated metric with no value (e.g. mean_recall is None because the
    eval set had no A/C cases with relevant_sources) fails rather than
    being skipped: a sweep that can't measure a gated dimension has not
    cleared the bar.
    """
    thresholds: list[ThresholdResult] = []

    ra = report.routing_accuracy
    thresholds.append(
        ThresholdResult(
            "Routing accuracy",
            True,
            f">= {bar.min_routing_accuracy:.0%}",
            f"{ra:.1%}",
            ra >= bar.min_routing_accuracy,
        )
    )

    for name, value, floor in (
        ("Mean recall@k (A/C)", report.mean_recall, bar.min_mean_recall),
        ("Mean MRR (A/C)", report.mean_reciprocal_rank, bar.min_mean_reciprocal_rank),
        ("Mean groundedness", report.mean_groundedness, bar.min_mean_groundedness),
        ("Mean relevance", report.mean_relevance, bar.min_mean_relevance),
    ):
        thresholds.append(
            ThresholdResult(
                name,
                True,
                f">= {floor:.2f}",
                "n/a" if value is None else f"{value:.2f}",
                value is not None and value >= floor,
            )
        )

    zero_recall = sorted(
        r.case_id for r in report.case_results if r.recall == 0.0
    )
    thresholds.append(
        ThresholdResult(
            "Per-case recall > 0.00 (A/C)",
            True,
            "no A/C case at recall 0.00",
            "no total misses" if not zero_recall else "missed: " + ", ".join(zero_recall),
            not zero_recall,
        )
    )

    prec = report.mean_precision
    thresholds.append(
        ThresholdResult(
            "Mean precision@k (A/C)",
            False,
            "reported, not gated (labeling confound)",
            "n/a" if prec is None else f"{prec:.2f}",
            True,
        )
    )

    if report.expertise_scored:
        gated = bar.gate_expertise
        note = "" if gated else " (provisional — not yet gated)"
        for name, value, floor in (
            ("Person recall@k (B)", report.mean_person_recall, bar.min_person_recall),
            ("Person MRR (B)", report.mean_person_mrr, bar.min_person_mrr),
            (
                "B groundedness",
                report.mean_expertise_groundedness,
                bar.min_expertise_groundedness,
            ),
            ("B relevance", report.mean_expertise_relevance, bar.min_expertise_relevance),
        ):
            thresholds.append(
                ThresholdResult(
                    name,
                    gated,
                    f">= {floor:.2f}{note}",
                    "n/a" if value is None else f"{value:.2f}",
                    value is not None and value >= floor,
                )
            )

        zero_person_recall = sorted(
            r.case_id for r in report.case_results if r.person_recall == 0.0
        )
        thresholds.append(
            ThresholdResult(
                "Per-case person recall > 0.00 (B)",
                gated,
                f"no B case at recall 0.00{note}",
                "no total misses"
                if not zero_person_recall
                else "missed: " + ", ".join(zero_person_recall),
                not zero_person_recall,
            )
        )

        nm = report.no_match_rate
        thresholds.append(
            ThresholdResult(
                "No-match correct-refusal rate (B)",
                gated,
                f">= {bar.min_no_match_rate:.0%}{note}",
                "n/a" if nm is None else f"{nm:.0%}",
                nm is not None and nm >= bar.min_no_match_rate,
            )
        )

        pprec = report.mean_person_precision
        thresholds.append(
            ThresholdResult(
                "Person precision@k (B)",
                False,
                "reported, not gated (labeling confound)",
                "n/a" if pprec is None else f"{pprec:.2f}",
                True,
            )
        )

    return BarResult(
        thresholds=thresholds,
        passed=all(t.passed for t in thresholds if t.gated),
    )


def format_report(report: EvalReport) -> str:
    """Human-readable text summary — aggregate metrics, the quality-bar
    check, then per-case detail so a specific miss can be traced back to
    its case.
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

    if report.expertise_scored and report.mean_person_recall is not None:
        lines += [
            "Expertise (archetype B, cases with relevant_people):",
            f"  Mean person recall@k: {report.mean_person_recall:.2f}",
            f"  Mean person precision: {report.mean_person_precision:.2f}",
            f"  Mean person MRR:      {report.mean_person_mrr:.2f}",
        ]
        if report.mean_expertise_groundedness is not None:
            lines += [
                f"  Mean groundedness:    {report.mean_expertise_groundedness:.2f}",
                f"  Mean relevance:       {report.mean_expertise_relevance:.2f}",
            ]
        lines.append("")
    if report.no_match_rate is not None:
        lines += [f"No-match refusal rate (B): {report.no_match_rate:.0%}", ""]

    bar = evaluate_bar(report)
    lines.append("Quality bar (evals/QUALITY_BAR.md):")
    for t in bar.thresholds:
        if t.gated:
            lines.append(
                f"  [{'PASS' if t.passed else 'FAIL'}] {t.name}: {t.actual} "
                f"(needs {t.requirement})"
            )
        else:
            lines.append(f"  [----] {t.name}: {t.actual} ({t.requirement})")
    lines.append(f"  => {'PASS' if bar.passed else 'FAIL'} (gated thresholds)")
    lines.append("")

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
        if r.person_recall is not None:
            parts.append(
                f"person_recall={r.person_recall:.2f} "
                f"person_precision={r.person_precision:.2f} "
                f"person_rr={r.person_reciprocal_rank:.2f}"
            )
        if r.no_match_correct is not None:
            parts.append(f"no_match={'OK' if r.no_match_correct else 'WRONG'}")
        if r.judge is not None:
            parts.append(
                f"groundedness={r.judge.groundedness} relevance={r.judge.relevance}"
            )
        if r.expertise_judge is not None:
            parts.append(
                f"groundedness={r.expertise_judge.groundedness} "
                f"relevance={r.expertise_judge.relevance}"
            )
        parts.append(f"latency={r.latency_seconds:.2f}s")
        lines.append("  " + " ".join(parts))

    return "\n".join(lines)


def main() -> None:
    """Build a real (temporary, non-persisted) index over the pilot
    corpus and run every case in evals/cases/ against live NVIDIA NIM.

    Run as `python -m evals.harness` from the repo root (not
    `python evals/harness.py` directly — that would put evals/ itself on
    sys.path instead of the repo root, breaking this module's absolute
    `from evals.metrics import ...` import). Reads NVIDIA_API_KEY from
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
    from tessera.generation.nvidia import NvidiaClient
    from tessera.generation.resilient import RetryingLLMClient
    from tessera.ingestion.chunker import chunk_corpus, chunk_embedding_text
    from tessera.ingestion.expertise_loader import load_expertise, profile_summary_text
    from tessera.ingestion.loader import load_corpus
    from tessera.store.chroma import ChromaVectorStore
    from tessera.store.chroma_expertise import ChromaExpertiseStore

    corpus_dir = Path(os.environ.get("TESSERA_CORPUS_DIR", "data/corpus"))
    expertise_dir = Path(os.environ.get("TESSERA_EXPERTISE_DIR", "data/expertise/people"))
    cases_dir = Path(__file__).parent / "cases"
    api_key = os.environ["NVIDIA_API_KEY"]
    model = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

    docs = load_corpus(corpus_dir)
    chunks = chunk_corpus(docs)
    embedder = LocalEmbedder()
    embeddings = embedder.embed_documents([chunk_embedding_text(c) for c in chunks])

    with tempfile.TemporaryDirectory() as persist_dir:
        store = ChromaVectorStore(persist_dir=Path(persist_dir))
        store.add(chunks, embeddings)
        people = load_expertise(expertise_dir, corpus_dir=corpus_dir)
        expertise_store = ChromaExpertiseStore(persist_dir=Path(persist_dir))
        expertise_store.add(
            people, embedder.embed_documents([profile_summary_text(p) for p in people])
        )
        llm = RetryingLLMClient(
            NvidiaClient(api_key=api_key, model=model, sdk_max_retries=0),
            min_interval=3.0,
        )

        cases = load_cases(cases_dir)
        report = run_harness(
            cases, llm, embedder, store, corpus_dir, expertise_store=expertise_store
        )
        print(format_report(report))


if __name__ == "__main__":
    main()
