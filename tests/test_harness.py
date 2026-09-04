"""Unit tests for evals/harness.py (fake LLM/Embedder/VectorStore, no
network) — case loading, per-case metric computation, aggregation, and
report formatting.
"""

from pathlib import Path

import pytest

from evals.harness import (
    DEFAULT_K,
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
    NOT_YET_SUPPORTED_MESSAGE,
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
    overfitting check-set) plus query_log.yaml (42 synthesized
    query-log stand-in cases, see its header comment). Counts below must
    be updated if either file's case count changes.
    """
    cases = load_cases(CASES_DIR)

    assert len(cases) == 50
    by_archetype = {a: 0 for a in Archetype}
    for case in cases:
        by_archetype[case.archetype] += 1
    assert by_archetype == {
        Archetype.LOOKUP: 20,
        Archetype.EXPERTISE: 10,
        Archetype.SYNTHESIS: 15,
        Archetype.COMPARATIVE: 5,
    }
    assert cases[0].id == "q001"  # placeholder.yaml sorts before query_log.yaml
    assert cases[0].relevant_sources  # A-archetype case has real sources


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


def test_run_case_expertise_short_circuits_without_retrieval_or_judge() -> None:
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
    assert result.answer == NOT_YET_SUPPORTED_MESSAGE
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
