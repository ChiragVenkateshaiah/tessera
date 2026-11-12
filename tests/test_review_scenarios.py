"""The six review scenarios (Phase 5 plan §3.8.6, P5-8 acceptance) on a
temporary copy of the corpus, with the real embedders, measured on
native, the LangChain vector retriever and the LangChain hybrid retriever.
Zero network calls; slow (one index build per stack, shared, then ~20 s a
scenario). ``python -m evals.review_scenario`` is the recorded run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph", reason="the lc extra isn't installed")

from evals import review_scenario as rs  # noqa: E402

STACKS = set(rs.STACKS)


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("review") / "template"
    rs.build_template(root)
    return root


@pytest.fixture(scope="module")
def cases() -> list[Any]:
    return rs.scenario_cases()


def run(n: int, template: Path, tmp_path: Path, cases: list[Any]) -> dict[str, Any]:
    world = rs.World.copy_of(template, tmp_path / f"s{n}")
    with world.inside():
        return rs.SCENARIOS[n](world, cases)


def test_the_committed_corpus_has_no_marker_hits_and_no_case_study(template: Path, tmp_path: Path, cases: list[Any]) -> None:
    world = rs.World.copy_of(template, tmp_path / "base")
    with world.inside():
        measured = world.measure(cases)
    assert set(measured) == STACKS
    for m in measured.values():
        assert m["hit_cases"] == [] and m["retrieved"] == {rs.GROCER: [], rs.BANK: []}


def test_1_grocer_approved_as_internal_leaks_on_exactly_ac_l01_and_ac_l13(template: Path, tmp_path: Path, cases: list[Any]) -> None:
    result = run(1, template, tmp_path, cases)
    counts, indexed = result["index_counts"], result["indexed"]
    for stack in ("native", "lc"):  # incremental: nothing but the case study added, nothing deleted
        assert counts["after"][stack] - counts["before"][stack] == indexed[stack]["chunks"] > 0
    assert result["front_matter"] == {"sensitivity": "internal", "engagement": None, "review_status": "reviewed"}
    for stack, m in result["measured"].items():
        assert m["hit_cases"] == ["ac-l01", "ac-l13"], stack
    assert result["pass"]


def test_2_reclassified_to_halcyon_no_leaks_and_c0048_retrieves_it(template: Path, tmp_path: Path, cases: list[Any]) -> None:
    result = run(2, template, tmp_path, cases)
    assert result["indexed"]["native"]["labels"] == result["indexed"]["lc"]["labels"] == [["restricted", "halcyon"]]
    for stack, m in result["measured"].items():
        assert m["hit_cases"] == [], stack
        assert "ac-a01" in m["retrieved"][rs.GROCER], stack
    assert result["pass"]


def test_3_rejected_not_indexed_not_offered_again_absent_from_bm25(template: Path, tmp_path: Path, cases: list[Any]) -> None:
    result = run(3, template, tmp_path, cases)
    assert result["indexed"] == {"native": {"chunks": 0, "labels": []}, "lc": {"chunks": 0, "labels": []}}
    assert rs.GROCER not in result["offered_next_time"] and rs.BANK in result["offered_next_time"]
    assert result["in_bm25_after"] is False
    assert result["front_matter"]["review_status"] == "pending"
    assert result["pass"]


def test_4_bank_approved_as_internal_shows_no_detected_leak(template: Path, tmp_path: Path, cases: list[Any]) -> None:
    """The honest-limits case: a paraphrased Project Lantern, retrieved by
    the walled analyst's Lantern question, with none of Lantern's markers."""
    result = run(4, template, tmp_path, cases)
    for stack, m in result["measured"].items():
        assert m["hit_cases"] == [], stack
        assert "ac-l03" in m["retrieved"][rs.BANK], stack
    assert result["pass"]


def test_5_a_partially_cleared_reviewer_gets_the_uniform_notice(template: Path, tmp_path: Path, cases: list[Any]) -> None:
    result = run(5, template, tmp_path, cases)
    assert result["allowed"] == ["leave", "stop"]
    assert len(result["refused"]) == 2
    assert result["review_status"] == ["pending", "pending"]
    assert set(result["evidence_engagements"]) <= set(result["clearance"])
    assert result["pass"]


def test_6_resumed_in_a_new_process_applied_once_and_corrected_before_apply(
    template: Path, tmp_path: Path, cases: list[Any]
) -> None:
    result = run(6, template, tmp_path, cases)
    assert result["process_2"]["resumed_at"] == result["process_1"]["paused_at"] == rs.GROCER
    assert result["process_2"]["replayed_apply"] == {"outcome": "already applied"}
    assert result["audit_lines_for_grocer"] == [{"action": "reclassify", "engagement": "halcyon", "outcome": "applied"}]
    assert result["pass"]
