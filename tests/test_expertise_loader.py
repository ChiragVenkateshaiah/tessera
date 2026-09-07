"""Unit tests for the expertise dataset loader + a real-dataset check."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from tessera.ingestion.expertise_loader import (
    ExpertiseError,
    Person,
    load_expertise,
    load_person,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PEOPLE_DIR = REPO_ROOT / "data" / "expertise" / "people"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


def _valid_record() -> dict:
    return {
        "person_id": "c0001",
        "name": "Test Person",
        "title": "Consultant",
        "office": "London",
        "practice": "pricing",
        "skills": [
            {"topic": "pricing-strategy", "level": 4, "basis": "evidenced"},
            {"topic": "benchmarking", "level": 2, "basis": "self_reported"},
        ],
        "project_history": [
            {
                "industry": "retail",
                "topic": "pricing-strategy",
                "role": "workstream lead",
                "year": 2023,
            }
        ],
        "authored": [],
        "languages": ["English"],
        "last_updated": date(2026, 1, 15),
    }


# --- load_person validation -------------------------------------------


def test_valid_record_loads() -> None:
    person = load_person(_valid_record())
    assert isinstance(person, Person)
    assert person.person_id == "c0001"
    assert person.skills[0].topic == "pricing-strategy"
    assert person.project_history[0].year == 2023
    assert person.evidenced_topics == frozenset({"pricing-strategy"})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(title="Senior Partner"),
        lambda r: r.update(last_updated="2026-01-15"),  # string, not a date
        lambda r: r["skills"].append({"topic": "x", "level": 4, "basis": "guessed"}),
        lambda r: r["skills"].append({"topic": "x", "level": 9, "basis": "evidenced"}),
        lambda r: r["skills"].clear(),
        lambda r: r["project_history"].append(
            {"industry": "retail", "topic": "x", "role": "workstream lead", "year": 2023,
             "client": "Acme Corp"}  # extra free-text key — a client name could hide here
        ),
        lambda r: r["project_history"].append(
            {"industry": "moon", "topic": "x", "role": "workstream lead", "year": 2023}
        ),
        lambda r: r["project_history"].append(
            {"industry": "retail", "topic": "x", "role": "chief", "year": 2023}
        ),
        lambda r: r["project_history"].append(
            {"industry": "retail", "topic": "x", "role": "analyst", "year": 1990}
        ),
        lambda r: r.pop("languages"),
        lambda r: r["languages"].clear(),
        lambda r: r.update(extra_key="nope"),
    ],
)
def test_invalid_records_are_rejected(mutate) -> None:
    record = _valid_record()
    mutate(record)
    with pytest.raises(ExpertiseError):
        load_person(record)


def test_authored_path_validated_against_corpus_when_given() -> None:
    record = _valid_record()
    record["authored"] = ["methodology/does-not-exist.md"]
    with pytest.raises(ExpertiseError):
        load_person(record, corpus_paths=frozenset({"methodology/market-entry-overview.md"}))


def test_skill_topic_validated_against_corpus_vocab_when_given() -> None:
    record = _valid_record()
    record["skills"] = [{"topic": "underwater-basket-weaving", "level": 3, "basis": "self_reported"}]
    with pytest.raises(ExpertiseError):
        load_person(record, corpus_topics=frozenset({"pricing-strategy"}))


# --- load_expertise on fixtures --------------------------------------


def test_load_expertise_dedupes_and_sorts(tmp_path: Path) -> None:
    a, b = _valid_record(), _valid_record()
    b["person_id"] = "c0000"
    (tmp_path / "one.yaml").write_text(yaml.safe_dump([a]), encoding="utf-8")
    (tmp_path / "two.yaml").write_text(yaml.safe_dump([b]), encoding="utf-8")
    people = load_expertise(tmp_path)
    assert [p.person_id for p in people] == ["c0000", "c0001"]


def test_load_expertise_rejects_duplicate_ids(tmp_path: Path) -> None:
    (tmp_path / "one.yaml").write_text(
        yaml.safe_dump([_valid_record(), _valid_record()]), encoding="utf-8"
    )
    with pytest.raises(ExpertiseError, match="duplicate"):
        load_expertise(tmp_path)


def test_load_expertise_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ExpertiseError):
        load_expertise(tmp_path)


# --- the real committed dataset -------------------------------------


def test_real_dataset_loads_and_validates_against_corpus() -> None:
    people = load_expertise(PEOPLE_DIR, corpus_dir=CORPUS_DIR)
    assert 560 <= len(people) <= 640
    assert len({p.person_id for p in people}) == len(people)
    assert all(p.person_id.startswith("c") for p in people)


def test_real_dataset_has_a_weak_signal_tail() -> None:
    people = load_expertise(PEOPLE_DIR)
    thin = [
        p
        for p in people
        if all(s.basis == "self_reported" for s in p.skills)
        and len(p.project_history) <= 1
    ]
    # a genuine tail, but not most of the firm
    assert 0.10 <= len(thin) / len(people) <= 0.40


def test_real_dataset_has_evidenced_and_claimed_skills() -> None:
    people = load_expertise(PEOPLE_DIR)
    bases = [s.basis for p in people for s in p.skills]
    evidenced = bases.count("evidenced")
    # both kinds well represented — neither the split is degenerate
    assert 0.25 <= evidenced / len(bases) <= 0.65


def test_real_dataset_project_history_carries_no_free_text() -> None:
    # The structural "no client names" guarantee: every project entry is
    # exactly industry+topic+role+year, all from closed vocabularies —
    # load_expertise would have raised otherwise, so this just asserts the
    # invariant is actually exercised (there are projects to check).
    people = load_expertise(PEOPLE_DIR, corpus_dir=CORPUS_DIR)
    assert sum(len(p.project_history) for p in people) > 500
