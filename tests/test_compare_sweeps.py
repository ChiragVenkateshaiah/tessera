"""evals/compare_sweeps.py: deterministic over two exports, zero calls."""

from __future__ import annotations

from typing import Any

from evals.compare_sweeps import compare, empirical_floor, format_markdown


def _row(case_id: str, **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id, "actual_archetype": "A", "retrieved_documents": ["d1"], "retrieved_people": [],
        "recall": 1.0, "reciprocal_rank_score": 1.0, "person_recall": None, "person_reciprocal_rank": None,
        "judge": {"groundedness": 5, "relevance": 5}, "expertise_judge": None, "error": None,
        "leaked": None, "access_set": None, "contract_held": None,
        "latency_seconds": 2.0, "input_tokens": 100, "output_tokens": 10,
    }
    row.update(kw)
    return row


def _export(*rows: dict[str, Any], **meta: Any) -> dict[str, Any]:
    return {"meta": meta, "case_results": list(rows)}


def test_cases_with_an_error_on_either_side_are_left_out() -> None:
    base = _export(_row("a"), _row("b", error="503"), _row("c"))
    cand = _export(_row("a"), _row("b"), _row("c", error="503", judge=None))

    c = compare(base, cand, "x")

    assert c.scored_in_both == 1 and c.errors == ["c"]


def test_deterministic_changes_are_listed_case_by_case() -> None:
    base = _export(_row("a"), _row("b"), _row("i", access_set="injection", contract_held=True, leaked=False))
    cand = _export(
        _row("a", actual_archetype="C", retrieved_documents=["d2"], recall=0.5),
        _row("b", leaked=True),
        _row("i", access_set="injection", contract_held=False, leaked=False),
    )

    c = compare(base, cand, "x")

    assert c.routing_changed == ["a A→C"]
    assert c.documents_changed == ["a"]
    assert c.recall_changed == ["a recall 1.0→0.5"]
    assert c.leaks == ["b"] and c.injection_failed == ["i"]


def test_judge_means_and_the_noise_floor() -> None:
    base = _export(_row("a"), _row("b"), _row("p", judge=None, expertise_judge={"groundedness": 5, "relevance": 4}))
    cand = _export(
        _row("a", judge={"groundedness": 5, "relevance": 4}),
        _row("b"),
        _row("p", judge=None, expertise_judge={"groundedness": 5, "relevance": 4}),
    )

    c = compare(base, cand, "x")

    assert c.judge["ac_relevance"] == (5.0, 4.5)
    assert c.judge["b_relevance"] == (4.0, 4.0)
    assert c.judge_changed == {"ac_relevance": ["a 5→4"]}
    assert c.outside_noise_floor() == ["ac_relevance"]
    assert c.outside_noise_floor({"ac_relevance": 0.5, "b_relevance": 0, "ac_groundedness": 0, "b_groundedness": 0}) == []


def test_the_empirical_floor_is_the_largest_pairwise_spread() -> None:
    one = _export(_row("a"), _row("b"))
    two = _export(_row("a", judge={"groundedness": 5, "relevance": 4}), _row("b"))
    three = _export(_row("a", judge={"groundedness": 4, "relevance": 4}), _row("b", judge={"groundedness": 5, "relevance": 3}))

    floor = empirical_floor([one, two, three])

    # relevance means 5.0 / 4.5 / 3.5, groundedness 5.0 / 5.0 / 4.5
    assert floor["ac_relevance"] == 1.5 and floor["ac_groundedness"] == 0.5
    assert floor["b_relevance"] == 0.0  # no B rows: nothing to spread


def test_the_markdown_names_the_floor_and_the_changed_cases() -> None:
    c = compare(_export(_row("a")), _export(_row("a", judge={"groundedness": 4, "relevance": 5})), "p5-6-lc-router")

    text = format_markdown([c], "base.json")

    assert "Noise floor (P5-0)" in text and "`p5-6-lc-router`" in text
    assert "judge ac_groundedness: a 5→4" in text
