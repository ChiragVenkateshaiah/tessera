"""P3-5: archetype B in the eval harness — B metrics, the no-match set, and
the B rows of the quality bar. Deterministic (fake LLM / embedder / store).
"""

from pathlib import Path

import pytest

from evals.harness import (
    EXPERTISE_NOT_SCORED_NOTE,
    CaseResult,
    EvalCase,
    EvalReport,
    QualityBar,
    evaluate_bar,
    format_report,
    load_cases,
    run_case,
    run_harness,
)
from evals.metrics import (
    EXPERTISE_JUDGE_SYSTEM_PROMPT,
    JudgeScore,
    shortlist_recall_at_k,
)
from tessera.generation.expertise import NO_EXPERT_MESSAGE
from tessera.generation.prompts import EXPERTISE_ANSWER_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT
from tessera.retrieval.router import Archetype
from tests.test_expertise_generation import ScoredExpertiseStore, make_person, make_project
from tests.test_harness import FakeVectorStore, ScriptedLLMClient, _router_response
from tests.test_pipeline import PricingEmbedder

CORPUS_DIR = Path("data/corpus")
JUDGE_OK = '{"groundedness": 5, "relevance": 4, "reasoning": "fine"}'


def expert(pid: str):
    return make_person(pid, projects=[make_project("pricing"), make_project("pricing", 2026)])


def b_case(**kw) -> EvalCase:
    base = dict(
        id="b1",
        query="who knows about pricing?",
        archetype=Archetype.EXPERTISE,
        relevant_sources=[],
        ideal_answer="names pricing experts",
        relevant_people=["c1", "c2"],
    )
    base.update(kw)
    return EvalCase(**base)


def b_llm() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("B"),
            EXPERTISE_ANSWER_SYSTEM_PROMPT: "Ask Person c1 [1].",
            EXPERTISE_JUDGE_SYSTEM_PROMPT: JUDGE_OK,
        }
    )


# --- metric ---


def test_shortlist_recall_caps_the_denominator_at_k() -> None:
    relevant = {f"p{i}" for i in range(20)}
    assert shortlist_recall_at_k(["p0", "p1", "p2", "p3", "p4"], relevant, 5) == 1.0
    assert shortlist_recall_at_k(["p0", "x", "p2", "y", "z"], relevant, 5) == pytest.approx(0.4)


def test_shortlist_recall_equals_plain_recall_when_relevant_fits_in_k() -> None:
    assert shortlist_recall_at_k(["a", "x", "y"], {"a", "b"}, 5) == pytest.approx(0.5)


def test_shortlist_recall_rejects_empty_relevant() -> None:
    with pytest.raises(ValueError):
        shortlist_recall_at_k(["a"], set(), 5)


# --- load_cases ---


def test_load_cases_parses_relevant_people_and_expect_no_match(tmp_path: Path) -> None:
    (tmp_path / "c.yaml").write_text(
        "- id: b1\n  query: q\n  archetype: B\n  relevant_people: [c0001, c0002]\n"
        "  ideal_answer: names them\n"
        "- id: n1\n  query: q2\n  archetype: B\n  expect_no_match: true\n"
    )
    b, n = load_cases(tmp_path)
    assert b.relevant_people == ["c0001", "c0002"] and not b.expect_no_match
    assert n.expect_no_match and n.relevant_people == []


# --- run_case: scored B ---


def test_run_case_scores_person_metrics_and_b_judge_separately_from_documents() -> None:
    store = ScoredExpertiseStore([(expert("c1"), 0.6), (expert("c9"), 0.5)])

    result = run_case(
        b_case(), b_llm(), PricingEmbedder(), FakeVectorStore([]), CORPUS_DIR,
        expertise_store=store,
    )

    assert result.routing_correct is True
    assert result.retrieved_people == ["c1", "c9"]
    assert result.person_recall == pytest.approx(0.5)  # 1 of 2 relevant, min(2,5)=2
    assert result.person_precision == pytest.approx(0.5)  # 1 of the 2 shown
    assert result.person_reciprocal_rank == 1.0
    assert result.expertise_judge == JudgeScore(5, 4, "fine")
    # document-side fields untouched, so A/C aggregates can't be blended
    assert result.recall is None and result.judge is None
    assert result.no_match_correct is None


def test_run_case_b_without_expertise_store_is_routing_only() -> None:
    result = run_case(
        b_case(), b_llm(), PricingEmbedder(), FakeVectorStore([]), CORPUS_DIR
    )
    assert result.answer == EXPERTISE_NOT_SCORED_NOTE
    assert result.person_recall is None and result.expertise_judge is None


def test_b_case_that_gets_the_no_match_message_scores_recall_zero() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("B")})
    nobody = ScoredExpertiseStore([(make_person("c1"), 0.3)])

    result = run_case(
        b_case(), llm, PricingEmbedder(), FakeVectorStore([]), CORPUS_DIR,
        expertise_store=nobody,
    )

    assert result.answer == NO_EXPERT_MESSAGE
    assert result.retrieved_people == []
    assert result.person_recall == 0.0  # a missed expert must show as a total miss
    assert result.expertise_judge is None


# --- run_case: no-match set ---


def test_no_match_case_is_correct_only_with_message_and_zero_generation_calls() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("B")})
    nobody = ScoredExpertiseStore([(make_person("c1"), 0.3)])
    case = b_case(id="n1", relevant_people=[], ideal_answer="", expect_no_match=True)

    result = run_case(
        case, llm, PricingEmbedder(), FakeVectorStore([]), CORPUS_DIR,
        expertise_store=nobody,
    )

    assert result.no_match_correct is True
    assert len(llm.calls) == 1  # router only


def test_no_match_case_is_wrong_when_the_system_names_someone() -> None:
    store = ScoredExpertiseStore([(expert("c1"), 0.6)])
    case = b_case(id="n1", relevant_people=[], ideal_answer="", expect_no_match=True)

    result = run_case(
        case, b_llm(), PricingEmbedder(), FakeVectorStore([]), CORPUS_DIR,
        expertise_store=store,
    )

    assert result.no_match_correct is False
    assert result.expertise_judge is None  # no-match cases are never judged


def test_no_match_case_that_misroutes_counts_as_wrong() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("D")})
    case = b_case(id="n1", relevant_people=[], ideal_answer="", expect_no_match=True)

    result = run_case(
        case, llm, PricingEmbedder(), FakeVectorStore([]), CORPUS_DIR,
        expertise_store=ScoredExpertiseStore([]),
    )

    assert result.routing_correct is False
    assert result.no_match_correct is False


# --- run_harness aggregation ---


def test_run_harness_aggregates_b_metrics_apart_from_ac() -> None:
    store = ScoredExpertiseStore([(expert("c1"), 0.6), (expert("c2"), 0.5)])
    report = run_harness(
        [b_case(id="b1", relevant_people=["c1", "c2"])], b_llm(), PricingEmbedder(), FakeVectorStore([]), CORPUS_DIR,
        expertise_store=store,
    )

    assert report.expertise_scored
    assert report.mean_person_recall == 1.0
    assert report.mean_person_mrr == 1.0
    assert report.mean_expertise_groundedness == 5.0
    assert report.mean_expertise_relevance == 4.0
    assert report.no_match_rate is None
    # nothing leaked into the A/C aggregates
    assert report.mean_recall is None and report.mean_groundedness is None


# --- quality bar: B rows ---


def _b_case_result(cid: str, recall: float | None, **kw) -> CaseResult:
    return CaseResult(
        case_id=cid, query="q", expected_archetype=Archetype.EXPERTISE,
        actual_archetype=Archetype.EXPERTISE, routing_correct=True,
        retrieved_documents=[], recall=None, precision=None,
        reciprocal_rank_score=None, answer="a", judge=None, latency_seconds=1.0,
        person_recall=recall,
        person_precision=None if recall is None else recall,
        person_reciprocal_rank=None if recall is None else recall,
        **kw,
    )


def _b_report(**overrides) -> EvalReport:
    d = dict(
        case_results=[_b_case_result("b1", 1.0)],
        routing_accuracy=1.0, mean_recall=0.9, mean_precision=0.5,
        mean_reciprocal_rank=0.95, mean_groundedness=4.8, mean_relevance=4.8,
        mean_latency_by_archetype={}, expertise_scored=True,
        mean_person_recall=0.95, mean_person_precision=0.8, mean_person_mrr=1.0,
        mean_expertise_groundedness=4.8, mean_expertise_relevance=4.7,
        no_match_rate=1.0,
    )
    d.update(overrides)
    return EvalReport(**d)


GATED_B = QualityBar(gate_expertise=True)


def test_bar_includes_b_rows_and_passes_when_all_clear() -> None:
    result = evaluate_bar(_b_report(), GATED_B)
    names = [t.name for t in result.thresholds]
    for expected in (
        "Person recall@k (B)", "Person MRR (B)", "B groundedness", "B relevance",
        "Per-case person recall > 0.00 (B)", "No-match correct-refusal rate (B)",
        "Person precision@k (B)",
    ):
        assert expected in names
    assert result.passed
    assert all(
        t.gated for t in result.thresholds if "(B)" in t.name and "precision" not in t.name
    )
    assert not next(t for t in result.thresholds if t.name == "Person precision@k (B)").gated


@pytest.mark.parametrize(
    "override,failing",
    [
        ({"mean_person_recall": 0.85}, "Person recall@k (B)"),
        ({"mean_person_mrr": 0.85}, "Person MRR (B)"),
        ({"mean_expertise_groundedness": 4.4}, "B groundedness"),
        ({"mean_expertise_relevance": 4.4}, "B relevance"),
        ({"no_match_rate": 0.8}, "No-match correct-refusal rate (B)"),
        ({"no_match_rate": None}, "No-match correct-refusal rate (B)"),
        ({"mean_person_recall": None}, "Person recall@k (B)"),
        (
            {"case_results": [_b_case_result("b1", 1.0), _b_case_result("b2", 0.0)]},
            "Per-case person recall > 0.00 (B)",
        ),
    ],
)
def test_bar_fails_on_each_b_threshold(override, failing) -> None:
    result = evaluate_bar(_b_report(**override), GATED_B)
    assert not result.passed
    assert failing in [t.name for t in result.gated_failures]


def test_b_thresholds_are_gated_by_default() -> None:
    assert QualityBar().gate_expertise is True
    result = evaluate_bar(_b_report(mean_person_recall=0.85))
    assert not result.passed
    assert "Person recall@k (B)" in [t.name for t in result.gated_failures]


def test_b_thresholds_can_be_provisional() -> None:
    """With gate_expertise=False a failing B metric is shown but does not
    fail the bar (plan §4.2's first-sweep staging)."""
    report = _b_report(mean_person_recall=0.85, no_match_rate=0.8)

    result = evaluate_bar(report, QualityBar(gate_expertise=False))

    assert result.passed
    recall = next(t for t in result.thresholds if t.name == "Person recall@k (B)")
    assert not recall.gated and not recall.passed  # still reported honestly
    assert "provisional" in recall.requirement


def test_ac_failures_still_fail_the_bar_while_b_is_provisional() -> None:
    result = evaluate_bar(_b_report(mean_recall=0.5), QualityBar(gate_expertise=False))
    assert not result.passed
    assert "Mean recall@k (A/C)" in [t.name for t in result.gated_failures]


def test_bar_omits_b_rows_when_no_expertise_store_was_used() -> None:
    result = evaluate_bar(_b_report(expertise_scored=False))
    assert not any("(B)" in t.name or t.name.startswith("B ") for t in result.thresholds)


def test_quality_bar_b_defaults_match_the_plan() -> None:
    bar = QualityBar()
    assert (bar.min_person_recall, bar.min_person_mrr) == (0.90, 0.90)
    assert (bar.min_expertise_groundedness, bar.min_expertise_relevance) == (4.5, 4.5)
    assert bar.min_no_match_rate == 1.0


def test_format_report_shows_b_block_and_no_match_rate() -> None:
    text = format_report(_b_report())
    assert "[PASS] Person recall@k (B): 0.95" in text
    assert "Mean person recall@k: 0.95" in text
    assert "No-match refusal rate (B): 100%" in text


# --- the real case files ---

ROOT = Path(__file__).resolve().parents[1]


def test_every_relevant_people_id_resolves_to_a_real_record() -> None:
    from tessera.ingestion.expertise_loader import load_expertise

    real = {p.person_id for p in load_expertise(ROOT / "data" / "expertise" / "people")}
    cases = load_cases(ROOT / "evals" / "cases")
    labelled = [c for c in cases if c.relevant_people]

    assert len(labelled) >= 8
    for case in labelled:
        assert set(case.relevant_people) <= real, case.id
        assert len(set(case.relevant_people)) == len(case.relevant_people), case.id
        assert case.archetype is Archetype.EXPERTISE, case.id


def test_no_match_cases_are_b_and_carry_no_labels() -> None:
    cases = [c for c in load_cases(ROOT / "evals" / "cases") if c.expect_no_match]

    assert len(cases) >= 5
    for case in cases:
        assert case.archetype is Archetype.EXPERTISE
        assert case.relevant_people == [] and case.relevant_sources == []
