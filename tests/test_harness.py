"""Unit tests for evals/harness.py (fake LLM/Embedder/VectorStore, no
network) — case loading, per-case metric computation, aggregation, and
report formatting.
"""

from pathlib import Path

import pytest

from evals.harness import (
    DEFAULT_K,
    EXPERTISE_NOT_SCORED_NOTE,
    CaseResult,
    EvalCase,
    EvalReport,
    evaluate_bar,
    format_report,
    load_cases,
    run_case,
    run_harness,
    unique_documents_by_rank,
)
from evals.metrics import JUDGE_SYSTEM_PROMPT, JudgeScore
from tessera.embedding.base import Embedder
from tessera.generation.answer import RELEVANCE_THRESHOLD
from tessera.generation.base import LLMClient
from tessera.generation.prompts import (
    LOOKUP_ANSWER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SYNTHESIS_ANSWER_SYSTEM_PROMPT,
)
from tessera.retrieval.router import (
    COMPARATIVE_REFUSAL_MESSAGE,
    Archetype,
)
from tessera.store.base import SearchResult, VectorStore

CORPUS_DIR = Path("data/corpus")
CASES_DIR = Path(__file__).resolve().parents[1] / "evals" / "cases"


class ScriptedLLMClient(LLMClient):
    """Returns a canned response keyed by exact system-prompt match — a
    single case can call the LLM for routing, generation, AND judging,
    each with a different system prompt.
    """

    def __init__(self, responses_by_system: dict[str, str]) -> None:
        self._responses = responses_by_system
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        if system not in self._responses:
            raise AssertionError(f"no scripted response for system prompt: {system!r}")
        return self._responses[system]


class FakeEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 1

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def _result(document_path: str, score: float, title: str = "Doc") -> SearchResult:
    return SearchResult(
        chunk_id=f"{document_path}#1",
        text=f"text for {document_path}",
        score=score,
        document_path=document_path,
        document_title=title,
        doc_type="methodology",
        industry="cross-industry",
        topics=["t1"],
        date="2024-01-01",
        heading_path=("Overview",),
    )


class FakeVectorStore(VectorStore):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def add(self, chunks: object, embeddings: object) -> None:
        raise NotImplementedError

    def query(
        self, embedding: list[float], k: int, where: dict[str, object] | None = None
    ) -> list[SearchResult]:
        return self._results[:k]

    def count(self) -> int:
        return len(self._results)


def _router_response(archetype: str) -> str:
    return f'{{"archetype": "{archetype}", "reasoning": "test"}}'


# --- load_cases ---


def test_load_cases_parses_real_case_files() -> None:
    """Loads every *.yaml under the real evals/cases/ — placeholder.yaml
    (8 Discovery Findings workshop queries, held out as a P2-3
    overfitting check-set), query_log.yaml (41 synthesized query-log
    stand-in cases, see its header comment), expertise_nomatch.yaml
    (6 archetype-B no-match cases, ql022 moved there from query_log.yaml
    in P3-5), feedback.yaml (1 case promoted from a thumbs-down in
    P4-3), and freshness.yaml (7 superseded-document cases, P4-4). Counts
    below must be updated if any file's case count changes.
    """
    cases = load_cases(CASES_DIR)

    assert len(cases) == 63
    by_archetype = {a: 0 for a in Archetype}
    for case in cases:
        by_archetype[case.archetype] += 1
    assert by_archetype == {
        Archetype.LOOKUP: 26,
        Archetype.EXPERTISE: 15,
        Archetype.SYNTHESIS: 17,
        Archetype.COMPARATIVE: 5,
    }
    q001 = next(c for c in cases if c.id == "q001")
    assert q001.relevant_sources  # A-archetype case has real sources
    freshness = [c for c in cases if c.superseded_sources]
    assert [c.id for c in freshness] == [f"fr00{i}" for i in range(1, 8)]
    assert all(c.relevant_sources for c in freshness)
    # Every labelled path, current and superseded, exists in the corpus.
    for c in cases:
        for rel in [*c.relevant_sources, *c.superseded_sources]:
            assert (CASES_DIR.parents[1] / "data" / "corpus" / rel).is_file(), (c.id, rel)


def test_load_cases_returns_empty_list_for_comment_only_file(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "empty.yaml").write_text("# nothing here yet\n")

    assert load_cases(cases_dir) == []


def test_load_cases_parses_minimal_case(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "c.yaml").write_text(
        "- id: t1\n"
        "  query: 'do we have a template?'\n"
        "  archetype: A\n"
        "  relevant_sources: ['methodology/x.md']\n"
        "  ideal_answer: 'points to x'\n"
    )

    cases = load_cases(cases_dir)

    assert cases == [
        EvalCase(
            id="t1",
            query="do we have a template?",
            archetype=Archetype.LOOKUP,
            relevant_sources=["methodology/x.md"],
            ideal_answer="points to x",
        )
    ]


# --- unique_documents_by_rank ---


def test_unique_documents_by_rank_collapses_duplicates_keeping_best_rank() -> None:
    results = [
        _result("data/corpus/methodology/a.md", 0.9),
        _result("data/corpus/methodology/b.md", 0.8),
        _result("data/corpus/methodology/a.md", 0.7),
    ]

    assert unique_documents_by_rank(results, CORPUS_DIR) == [
        "methodology/a.md",
        "methodology/b.md",
    ]


# --- run_case ---


def test_run_case_lookup_computes_retrieval_metrics_and_judge() -> None:
    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("A"),
            LOOKUP_ANSWER_SYSTEM_PROMPT: "We have it, see [1].",
            JUDGE_SYSTEM_PROMPT: '{"groundedness": 5, "relevance": 5, "reasoning": "good"}',
        }
    )
    store = FakeVectorStore(
        [
            _result("data/corpus/methodology/target.md", 0.9),
            _result("data/corpus/methodology/other.md", 0.5),
        ]
    )
    case = EvalCase(
        id="q1",
        query="do we have X?",
        archetype=Archetype.LOOKUP,
        relevant_sources=["methodology/target.md"],
        ideal_answer="points to target",
    )

    result = run_case(case, llm, FakeEmbedder(), store, CORPUS_DIR)

    assert result.routing_correct is True
    assert result.actual_archetype is Archetype.LOOKUP
    assert result.retrieved_documents == ["methodology/target.md", "methodology/other.md"]
    assert result.recall == 1.0
    assert result.precision == pytest.approx(0.5)
    assert result.reciprocal_rank_score == 1.0
    assert result.answer == "We have it, see [1]."
    assert result.judge == JudgeScore(groundedness=5, relevance=5, reasoning="good")
    assert result.latency_seconds >= 0.0


def test_run_case_expertise_is_routing_only_without_retrieval_or_judge() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("B")})
    store = FakeVectorStore([_result("data/corpus/a.md", 0.9)])
    case = EvalCase(
        id="q2",
        query="who knows X?",
        archetype=Archetype.EXPERTISE,
        relevant_sources=[],
        ideal_answer="",
    )

    result = run_case(case, llm, FakeEmbedder(), store, CORPUS_DIR)

    assert result.routing_correct is True
    assert result.answer == EXPERTISE_NOT_SCORED_NOTE
    assert result.recall is None
    assert result.precision is None
    assert result.reciprocal_rank_score is None
    assert result.judge is None
    assert len(llm.calls) == 1  # only the router call


def test_run_case_comparative_short_circuits_with_refusal_message() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("D")})
    store = FakeVectorStore([_result("data/corpus/a.md", 0.9)])
    case = EvalCase(
        id="q3",
        query="compare X vs Y",
        archetype=Archetype.COMPARATIVE,
        relevant_sources=[],
        ideal_answer="",
    )

    result = run_case(case, llm, FakeEmbedder(), store, CORPUS_DIR)

    assert result.answer == COMPARATIVE_REFUSAL_MESSAGE
    assert result.judge is None


def test_run_case_marks_routing_incorrect_when_actual_differs_from_expected() -> None:
    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("C"),
            SYNTHESIS_ANSWER_SYSTEM_PROMPT: "a briefing [1].",
            JUDGE_SYSTEM_PROMPT: '{"groundedness": 3, "relevance": 3, "reasoning": "ok"}',
        }
    )
    store = FakeVectorStore([_result("data/corpus/a.md", 0.9)])
    case = EvalCase(
        id="q4",
        query="do we have X?",
        archetype=Archetype.LOOKUP,  # expected A, but router below returns C
        relevant_sources=["methodology/a.md"],
        ideal_answer="ideal",
    )

    result = run_case(case, llm, FakeEmbedder(), store, CORPUS_DIR)

    assert result.routing_correct is False
    assert result.actual_archetype is Archetype.SYNTHESIS


def test_run_case_skips_judge_when_no_ideal_answer() -> None:
    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("A"),
            LOOKUP_ANSWER_SYSTEM_PROMPT: "an answer [1].",
        }
    )
    store = FakeVectorStore([_result("data/corpus/a.md", 0.9)])
    case = EvalCase(
        id="q5",
        query="q",
        archetype=Archetype.LOOKUP,
        relevant_sources=["a.md"],
        ideal_answer="",
    )

    result = run_case(case, llm, FakeEmbedder(), store, CORPUS_DIR)

    assert result.judge is None
    assert JUDGE_SYSTEM_PROMPT not in [system for system, _ in llm.calls]


def test_run_case_skips_judge_when_answer_is_off_corpus_refusal() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("A")})
    store = FakeVectorStore([_result("data/corpus/a.md", RELEVANCE_THRESHOLD - 0.1)])
    case = EvalCase(
        id="q6",
        query="off corpus question",
        archetype=Archetype.LOOKUP,
        relevant_sources=["a.md"],
        ideal_answer="some ideal content",
    )

    result = run_case(case, llm, FakeEmbedder(), store, CORPUS_DIR)

    assert result.judge is None
    # only the router call happened — generate_answer's refusal path
    # never calls the LLM, and the harness doesn't judge a refusal
    assert len(llm.calls) == 1


# --- run_harness / format_report ---


def test_run_harness_aggregates_across_cases() -> None:
    # ScriptedLLMClient's system-prompt keying isn't enough here since
    # both cases share the router prompt but need different routing
    # answers — route per query text instead.
    class PerQueryLLMClient(LLMClient):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
            self.calls.append((system, user))
            if system == ROUTER_SYSTEM_PROMPT:
                if "lookup" in user:
                    return _router_response("A")
                return _router_response("B")
            if system == LOOKUP_ANSWER_SYSTEM_PROMPT:
                return "answer [1]."
            if system == JUDGE_SYSTEM_PROMPT:
                return '{"groundedness": 4, "relevance": 4, "reasoning": "fine"}'
            raise AssertionError(f"unexpected system prompt: {system!r}")

    per_query_llm = PerQueryLLMClient()
    store = FakeVectorStore([_result("data/corpus/target.md", 0.9)])
    cases = [
        EvalCase(
            id="a1",
            query="lookup query",
            archetype=Archetype.LOOKUP,
            relevant_sources=["target.md"],
            ideal_answer="ideal",
        ),
        EvalCase(
            id="b1",
            query="expertise query",
            archetype=Archetype.EXPERTISE,
            relevant_sources=[],
            ideal_answer="",
        ),
    ]

    report = run_harness(cases, per_query_llm, FakeEmbedder(), store, CORPUS_DIR)

    assert report.routing_accuracy == 1.0
    assert report.mean_recall == 1.0
    assert report.mean_groundedness == 4.0
    assert report.mean_relevance == 4.0
    assert report.mean_latency_by_archetype.keys() == {
        Archetype.LOOKUP,
        Archetype.EXPERTISE,
    }
    assert len(report.case_results) == 2


def test_run_harness_continues_past_a_case_that_raises() -> None:
    """A malformed judge/router response for one case must not abort the
    rest of the sweep or burn the quota already spent on earlier cases —
    genai-architect's Task 7 round-1 review flagged this as the
    highest-value follow-up.
    """

    class FlakyLLMClient(LLMClient):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
            self.calls.append((system, user))
            if system == ROUTER_SYSTEM_PROMPT:
                if "broken" in user:
                    return "not valid json"
                return _router_response("B")
            raise AssertionError(f"unexpected system prompt: {system!r}")

    llm = FlakyLLMClient()
    store = FakeVectorStore([_result("data/corpus/a.md", 0.9)])
    cases = [
        EvalCase(
            id="bad",
            query="broken query",
            archetype=Archetype.LOOKUP,
            relevant_sources=[],
            ideal_answer="",
        ),
        EvalCase(
            id="good",
            query="expertise query",
            archetype=Archetype.EXPERTISE,
            relevant_sources=[],
            ideal_answer="",
        ),
    ]

    report = run_harness(cases, llm, FakeEmbedder(), store, CORPUS_DIR)

    assert len(report.case_results) == 2
    bad, good = report.case_results
    assert bad.error is not None
    assert bad.actual_archetype is None
    assert bad.routing_correct is None
    assert good.error is None
    assert good.routing_correct is True
    # the errored case is excluded from routing_accuracy entirely, not
    # counted as a miss
    assert report.routing_accuracy == 1.0


def test_format_report_shows_error_cases_without_crashing() -> None:
    from evals.harness import EvalReport

    error_result = CaseResult(
        case_id="bad",
        query="broken query",
        expected_archetype=Archetype.LOOKUP,
        actual_archetype=None,
        routing_correct=None,
        retrieved_documents=[],
        recall=None,
        precision=None,
        reciprocal_rank_score=None,
        answer="",
        judge=None,
        latency_seconds=0.0,
        error="could not parse routing response: 'not valid json'",
    )
    report = EvalReport(
        case_results=[error_result],
        routing_accuracy=0.0,
        mean_recall=None,
        mean_precision=None,
        mean_reciprocal_rank=None,
        mean_groundedness=None,
        mean_relevance=None,
        mean_latency_by_archetype={},
    )

    text = format_report(report)

    assert "[bad] ERROR: could not parse routing response" in text


def test_format_report_includes_key_sections() -> None:
    case_result = CaseResult(
        case_id="q1",
        query="q",
        expected_archetype=Archetype.LOOKUP,
        actual_archetype=Archetype.LOOKUP,
        routing_correct=True,
        retrieved_documents=["a.md"],
        recall=1.0,
        precision=0.5,
        reciprocal_rank_score=1.0,
        answer="an answer",
        judge=JudgeScore(groundedness=4, relevance=5, reasoning="ok"),
        latency_seconds=1.234,
    )
    from evals.harness import EvalReport

    report = EvalReport(
        case_results=[case_result],
        routing_accuracy=1.0,
        mean_recall=1.0,
        mean_precision=0.5,
        mean_reciprocal_rank=1.0,
        mean_groundedness=4.0,
        mean_relevance=5.0,
        mean_latency_by_archetype={Archetype.LOOKUP: 1.234},
    )

    text = format_report(report)

    assert "Routing accuracy: 100.0%" in text
    assert "Mean recall:    1.00" in text
    assert "Mean groundedness: 4.00" in text
    assert "[q1]" in text
    assert "routing=OK" in text


# --- evaluate_bar ---


def _passing_report(**overrides: object) -> EvalReport:
    """An EvalReport that clears every gated threshold, minus any field
    overridden by a test.
    """
    defaults: dict[str, object] = dict(
        case_results=[],
        routing_accuracy=1.0,
        mean_recall=0.88,
        mean_precision=0.72,
        mean_reciprocal_rank=0.97,
        mean_groundedness=5.0,
        mean_relevance=4.9,
        mean_latency_by_archetype={},
        superseded_cited_cases=[],
    )
    defaults.update(overrides)
    return EvalReport(**defaults)  # type: ignore[arg-type]


def _ac_case(case_id: str, recall: float) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        query="q",
        expected_archetype=Archetype.LOOKUP,
        actual_archetype=Archetype.LOOKUP,
        routing_correct=True,
        retrieved_documents=["a.md"],
        recall=recall,
        precision=0.5,
        reciprocal_rank_score=1.0,
        answer="an answer [1].",
        judge=JudgeScore(groundedness=5, relevance=5, reasoning="ok"),
        latency_seconds=1.0,
    )


def test_evaluate_bar_passes_when_every_gated_threshold_is_met() -> None:
    result = evaluate_bar(_passing_report())

    assert result.passed is True
    assert result.gated_failures == []


def test_evaluate_bar_fails_on_low_mean_recall() -> None:
    result = evaluate_bar(_passing_report(mean_recall=0.62))

    assert result.passed is False
    assert [t.name for t in result.gated_failures] == ["Mean recall@k (A/C)"]


def test_evaluate_bar_does_not_fail_on_low_precision() -> None:
    result = evaluate_bar(_passing_report(mean_precision=0.20))

    assert result.passed is True
    precision_row = next(
        t for t in result.thresholds if t.name == "Mean precision@k (A/C)"
    )
    assert precision_row.gated is False


def test_evaluate_bar_fails_when_a_gated_metric_has_no_value() -> None:
    result = evaluate_bar(_passing_report(mean_reciprocal_rank=None))

    assert result.passed is False
    assert [t.name for t in result.gated_failures] == ["Mean MRR (A/C)"]


def test_evaluate_bar_fails_on_a_per_case_total_miss() -> None:
    report = _passing_report(
        case_results=[_ac_case("q001", 1.0), _ac_case("ql009", 0.0)]
    )

    result = evaluate_bar(report)

    assert result.passed is False
    miss_row = next(
        t for t in result.thresholds if t.name.startswith("Per-case recall")
    )
    assert miss_row.passed is False
    assert "ql009" in miss_row.actual


def test_format_report_renders_the_quality_bar_block() -> None:
    text = format_report(_passing_report())

    assert "Quality bar (evals/QUALITY_BAR.md):" in text
    assert "[PASS] Routing accuracy" in text
    assert "[----] Mean precision@k (A/C)" in text
    assert "=> PASS (gated thresholds)" in text


def test_format_report_quality_bar_block_shows_failure() -> None:
    text = format_report(_passing_report(mean_relevance=3.1))

    assert "[FAIL] Mean relevance" in text
    assert "=> FAIL (gated thresholds)" in text


def test_run_harness_reports_progress_for_every_case_including_errors() -> None:
    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("D"),
        }
    )

    class Boom(ScriptedLLMClient):
        def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
            if "explode" in user:
                raise RuntimeError("boom")
            return super().complete(system, user, temperature)

    boom = Boom({ROUTER_SYSTEM_PROMPT: _router_response("D")})
    cases = [
        EvalCase("c1", "compare X vs Y", Archetype.COMPARATIVE, [], ""),
        EvalCase("c2", "explode please", Archetype.COMPARATIVE, [], ""),
        EvalCase("c3", "compare A vs B", Archetype.COMPARATIVE, [], ""),
    ]
    seen: list[tuple[int, int, str, bool]] = []

    run_harness(
        cases,
        boom,
        FakeEmbedder(),
        FakeVectorStore([]),
        CORPUS_DIR,
        on_case_complete=lambda done, total, r: seen.append(
            (done, total, r.case_id, r.error is not None)
        ),
    )

    assert seen == [(1, 3, "c1", False), (2, 3, "c2", True), (3, 3, "c3", False)]


# --- Phase 4: router/judge clients and cost ---


class MeteredScripted(ScriptedLLMClient):
    def __init__(self, responses_by_system: dict[str, str], model: str) -> None:
        super().__init__(responses_by_system)
        self.model = model

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0):
        from tessera.generation.base import Completion, Usage

        return Completion(
            self.complete(system, user, temperature), Usage(self.model, 1_000, 100, 0.1)
        )


def _lookup_case() -> EvalCase:
    return EvalCase(
        id="q1",
        query="do we have X?",
        archetype=Archetype.LOOKUP,
        relevant_sources=["methodology/target.md"],
        ideal_answer="points to target",
    )


def _cost_setup():
    from tessera.generation.usage import ModelPrice

    router = MeteredScripted({ROUTER_SYSTEM_PROMPT: _router_response("A")}, "haiku")
    answerer = MeteredScripted({LOOKUP_ANSWER_SYSTEM_PROMPT: "See [1]."}, "opus")
    judge = ScriptedLLMClient(
        {JUDGE_SYSTEM_PROMPT: '{"groundedness": 5, "relevance": 4, "reasoning": "ok"}'}
    )
    store = FakeVectorStore([_result("data/corpus/methodology/target.md", 0.9)])
    prices = {"haiku": ModelPrice(1.0, 5.0), "opus": ModelPrice(4.0, 20.0)}
    return router, answerer, judge, store, prices


def test_run_case_uses_router_and_judge_clients_and_costs_only_the_answer_path() -> None:
    router, answerer, judge, store, prices = _cost_setup()

    result = run_case(
        _lookup_case(),
        answerer,
        FakeEmbedder(),
        store,
        CORPUS_DIR,
        router_llm=router,
        judge_llm=judge,
        prices=prices,
    )

    assert [s for s, _ in router.calls] == [ROUTER_SYSTEM_PROMPT]
    assert [s for s, _ in answerer.calls] == [LOOKUP_ANSWER_SYSTEM_PROMPT]
    assert [s for s, _ in judge.calls] == [JUDGE_SYSTEM_PROMPT]
    assert result.judge == JudgeScore(groundedness=5, relevance=4, reasoning="ok")
    assert (result.input_tokens, result.output_tokens) == (2_000, 200)
    # haiku 1000*1 + 100*5, opus 1000*4 + 100*20 — judge not included
    assert result.cost_usd == pytest.approx((1_500 + 6_000) / 1e6)


def test_run_harness_reports_cost_by_archetype_and_a_provisional_bar_row() -> None:
    router, answerer, judge, store, prices = _cost_setup()

    report = run_harness(
        [_lookup_case()],
        answerer,
        FakeEmbedder(),
        store,
        CORPUS_DIR,
        router_llm=router,
        judge_llm=judge,
        prices=prices,
    )

    assert report.mean_cost_per_answer == pytest.approx(7_500 / 1e6)
    assert report.total_cost_usd == pytest.approx(7_500 / 1e6)
    assert report.mean_cost_by_archetype == {Archetype.LOOKUP: pytest.approx(7_500 / 1e6)}
    assert (report.total_input_tokens, report.total_output_tokens) == (2_000, 200)

    row = next(t for t in evaluate_bar(report).thresholds if t.name == "Mean cost per answer")
    assert row.gated is False and row.passed is True
    assert "provisional" in row.requirement

    text = format_report(report)
    assert "A: $0.0075 per answer" in text
    assert "[----] Mean cost per answer: $0.0075 (provisional" in text
    assert "cost=$0.0075" in text


def test_unpriced_sweep_reports_tokens_but_no_cost_row() -> None:
    router, answerer, judge, store, _ = _cost_setup()

    report = run_harness(
        [_lookup_case()], answerer, FakeEmbedder(), store, CORPUS_DIR,
        router_llm=router, judge_llm=judge,
    )

    assert report.total_input_tokens == 2_000
    assert report.mean_cost_per_answer is None
    assert all(t.name != "Mean cost per answer" for t in evaluate_bar(report).thresholds)
    assert "Cost: n/a" in format_report(report)


def test_gated_cost_ceiling_fails_the_bar_when_exceeded() -> None:
    from evals.harness import QualityBar

    report = _passing_report(mean_cost_per_answer=0.05)

    ok = evaluate_bar(report, QualityBar(max_mean_cost_per_answer_usd=0.10, gate_cost=True))
    over = evaluate_bar(report, QualityBar(max_mean_cost_per_answer_usd=0.01, gate_cost=True))

    assert ok.passed is True
    assert over.passed is False
    assert [t.name for t in over.gated_failures] == ["Mean cost per answer"]


def test_judge_sees_sources_numbered_exactly_as_the_answer_prompt() -> None:
    """Answers cite one [n] per document; if the judge numbered chunks
    instead, every citation after a multi-chunk document would point at
    the wrong text (the 2026-10-02 groundedness drop).
    """
    from evals.metrics import build_judge_user_prompt
    from tessera.generation.prompts import build_grounded_answer_user_prompt

    chunks = [
        _result("data/corpus/methodology/ref.md", 0.9, title="Reference"),
        _result("data/corpus/methodology/ref.md", 0.8, title="Reference"),
        _result("data/corpus/methodology/other.md", 0.7, title="Other"),
    ]
    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("C"),
            SYNTHESIS_ANSWER_SYSTEM_PROMPT: "Briefing [1][2].",
            JUDGE_SYSTEM_PROMPT: '{"groundedness": 5, "relevance": 5, "reasoning": "ok"}',
        }
    )
    case = EvalCase(
        id="q1", query="brief me", archetype=Archetype.SYNTHESIS,
        relevant_sources=["methodology/ref.md"], ideal_answer="covers ref",
    )

    run_case(case, llm, FakeEmbedder(), FakeVectorStore(chunks), CORPUS_DIR)

    answer_prompt = next(u for s, u in llm.calls if s == SYNTHESIS_ANSWER_SYSTEM_PROMPT)
    judge_prompt = next(u for s, u in llm.calls if s == JUDGE_SYSTEM_PROMPT)
    answer_sources = answer_prompt.split("Sources:\n\n", 1)[1]
    judge_sources = judge_prompt.split("Sources the assistant could use:\n\n", 1)[1].split(
        "\n\nAssistant's actual answer:", 1
    )[0]
    assert judge_sources == answer_sources
    assert "[2] Other" in judge_sources and "[3]" not in judge_sources


# --- Freshness gate (P4-4) ---


def test_bar_fails_when_a_case_cites_a_superseded_document() -> None:
    result = evaluate_bar(_passing_report(superseded_cited_cases=["fr001"]))

    assert result.passed is False
    assert [t.name for t in result.gated_failures] == ["Superseded cited as current (A/C)"]
    assert "fr001" in result.gated_failures[0].actual


def test_superseded_gate_has_no_value_without_document_cases() -> None:
    result = evaluate_bar(_passing_report(superseded_cited_cases=None))

    assert "Superseded cited as current (A/C)" in [t.name for t in result.gated_failures]


def test_run_case_scores_a_freshness_case_and_judges_without_the_note() -> None:
    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("A"),
            LOOKUP_ANSWER_SYSTEM_PROMPT: "Use the current guide [1].",
            JUDGE_SYSTEM_PROMPT: '{"groundedness": 5, "relevance": 5, "reasoning": "good"}',
        }
    )
    old = SearchResult(
        **{
            **_result("data/corpus/methodology/old.md", 0.9, "Old Guide").__dict__,
            "status": "superseded",
            "superseded_by": "data/corpus/methodology/new.md",
        }
    )
    store = FakeVectorStore([old, _result("data/corpus/methodology/new.md", 0.7, "New Guide")])
    case = EvalCase(
        id="fr1",
        query="where's the old method?",
        archetype=Archetype.LOOKUP,
        relevant_sources=["methodology/new.md"],
        ideal_answer="points to the new guide",
        superseded_sources=["methodology/old.md"],
    )

    result = run_case(case, llm, FakeEmbedder(), store, CORPUS_DIR)

    assert result.retrieved_documents == ["methodology/new.md"]
    assert result.superseded_cited == []
    assert result.superseded_noted is True
    assert '"Old Guide"' in result.answer  # the user sees the note...
    judged = next(user for system, user in llm.calls if system == JUDGE_SYSTEM_PROMPT)
    assert "Old Guide" not in judged  # ...the judge grades the model's text


def test_run_harness_reports_freshness_cases() -> None:
    results = [
        CaseResult(
            case_id=cid, query="q", expected_archetype=Archetype.LOOKUP,
            actual_archetype=Archetype.LOOKUP, routing_correct=True,
            retrieved_documents=[], recall=1.0, precision=1.0,
            reciprocal_rank_score=1.0, answer="", judge=None, latency_seconds=0.0,
            superseded_cited=cited, superseded_noted=noted,
        )
        for cid, cited, noted in [("a", [], None), ("b", ["methodology/old.md"], True)]
    ]

    text = format_report(_passing_report(case_results=results, superseded_cited_cases=["b"]))

    assert "[FAIL] Superseded cited as current (A/C): 1: b" in text
    assert "superseded_cited=methodology/old.md" in text
