"""The feedback loop: JSONL stores, thumbs-down → candidate cases, and the
guard that keeps unlabelled candidates out of the eval set.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from evals.harness import load_cases
from tessera.feedback.base import Feedback
from tessera.feedback.candidates import (
    CANDIDATE_STATUS,
    candidate_cases,
    render_candidates,
)
from tessera.feedback.local import JsonlFeedbackStore, JsonlTraceLog

T0 = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)
CORPUS = Path("data/corpus")


def _trace(trace_id: str, **extra: object) -> dict:
    record = {
        "trace_id": trace_id,
        "timestamp": T0.isoformat(),
        "query": "Do we have a value-based pricing framework?",
        "archetype": "A",
        "retrieved_kind": "chunk",
        "retrieved": [
            {"id": "a#1", "score": 0.7, "used": True, "document_path": "data/corpus/methodology/a.md"},
            {"id": "a#2", "score": 0.6, "used": True, "document_path": "data/corpus/methodology/a.md"},
            {"id": "b#1", "score": 0.5, "used": True, "document_path": "data/corpus/methodology/b.md"},
            {"id": "c#1", "score": 0.2, "used": False, "document_path": "data/corpus/methodology/c.md"},
        ],
        "fixed_response": False,
    }
    record.update(extra)
    return record


def _down(trace_id: str, minutes: int = 0, **kw: object) -> Feedback:
    return Feedback(trace_id, "down", T0 + timedelta(minutes=minutes), **kw)  # type: ignore[arg-type]


def test_trace_log_appends_and_finds_by_id(tmp_path: Path) -> None:
    log = JsonlTraceLog(tmp_path / "traces" / "t.jsonl")
    log.append(_trace("one"))
    log.append(_trace("two", query="other"))

    assert log.get("two")["query"] == "other"
    assert log.get("missing") is None


def test_trace_log_requires_a_trace_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        JsonlTraceLog(tmp_path / "t.jsonl").append({"query": "q"})


def test_trace_log_skips_a_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    log = JsonlTraceLog(path)
    log.append(_trace("one"))
    with path.open("a") as f:
        f.write('{"trace_id": "two", "que')

    assert log.get("one") is not None
    assert log.get("two") is None


def test_feedback_store_round_trips(tmp_path: Path) -> None:
    store = JsonlFeedbackStore(tmp_path / "fb.jsonl")
    item = _down("one", reason="wrong-source", comment="wanted the pricing playbook")
    store.add(item)

    assert store.list() == [item]
    assert JsonlFeedbackStore(tmp_path / "none.jsonl").list() == []


def test_thumbs_down_becomes_an_unlabelled_candidate() -> None:
    traces = {"abcdef1234": _trace("abcdef1234")}

    cases, missing = candidate_cases(
        [_down("abcdef1234", reason="wrong-source", comment="too broad")], traces.get, CORPUS
    )

    assert missing == []
    (case,) = cases
    assert case["id"] == "fb-abcdef12"
    assert case["status"] == CANDIDATE_STATUS
    assert case["archetype"] == "A"
    assert case["relevant_sources"] == [] and case["ideal_answer"] == ""
    seen = case["observed"]
    # used chunks only, one entry per document, corpus-relative
    assert seen["sources_shown"] == ["methodology/a.md", "methodology/b.md"]
    assert seen["reason"] == "wrong-source" and seen["comment"] == "too broad"


def test_latest_rating_per_trace_wins_and_thumbs_up_is_ignored() -> None:
    traces = {t: _trace(t) for t in ("t1", "t2", "t3")}
    feedback = [
        _down("t1", 0),
        Feedback("t1", "up", T0 + timedelta(minutes=5)),  # changed their mind
        Feedback("t2", "up", T0),
        _down("t3", 0),
        _down("t3", 1),  # rated twice: one candidate
    ]

    cases, _ = candidate_cases(feedback, traces.get, CORPUS)

    assert [c["observed"]["trace_id"] for c in cases] == ["t3"]


def test_feedback_without_a_trace_is_reported_not_guessed() -> None:
    cases, missing = candidate_cases([_down("gone")], {}.get, CORPUS)

    assert cases == [] and missing == ["gone"]


def test_person_candidates_list_the_people_shown() -> None:
    trace = _trace(
        "p1",
        archetype="B",
        retrieved_kind="person",
        retrieved=[
            {"id": "c0001", "score": 1.4, "used": True},
            {"id": "c0002", "score": 0.0, "used": False},
        ],
    )

    (case,) = candidate_cases([_down("p1")], {"p1": trace}.get, CORPUS)[0]

    assert case["observed"]["people_shown"] == ["c0001"]
    assert case["observed"]["sources_shown"] == []


def test_rendered_candidates_parse_back_and_carry_labelling_instructions() -> None:
    cases, _ = candidate_cases([_down("abcdef1234")], {"abcdef1234": _trace("abcdef1234")}.get, CORPUS)

    text = render_candidates(cases)

    assert text.startswith("# CANDIDATE eval cases")
    assert "delete the `status: candidate` line" in text
    assert yaml.safe_load(text) == cases
    assert yaml.safe_load(render_candidates([])) == []


def test_load_cases_refuses_an_unlabelled_candidate(tmp_path: Path) -> None:
    cases, _ = candidate_cases([_down("abcdef1234")], {"abcdef1234": _trace("abcdef1234")}.get, CORPUS)
    (tmp_path / "promoted.yaml").write_text(render_candidates(cases))

    with pytest.raises(ValueError, match="unlabelled feedback candidate"):
        load_cases(tmp_path)


def test_a_labelled_candidate_loads_once_status_is_removed(tmp_path: Path) -> None:
    cases, _ = candidate_cases([_down("abcdef1234")], {"abcdef1234": _trace("abcdef1234")}.get, CORPUS)
    (case,) = cases
    del case["status"], case["observed"]
    case.update(
        id="ql043",
        relevant_sources=["methodology/a.md"],
        ideal_answer="Points to the value-based pricing framework.",
    )
    (tmp_path / "labelled.yaml").write_text(yaml.safe_dump([case]))

    (loaded,) = load_cases(tmp_path)

    assert loaded.id == "ql043" and loaded.relevant_sources == ["methodology/a.md"]
