from datetime import date
from pathlib import Path

import pytest

from tessera.ingestion.loader import CorpusError, load_corpus, load_document

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


VALID_FILE = """\
---
title: "Test Doc"
doc_type: methodology
industry: cross-industry
topics: [test, fixture]
date: 2024-01-15
---

## Overview

Some body text.
"""


def test_load_document_parses_front_matter_and_body(tmp_path: Path) -> None:
    path = _write(tmp_path, "test-doc.md", VALID_FILE)

    doc = load_document(path)

    assert doc.title == "Test Doc"
    assert doc.doc_type == "methodology"
    assert doc.industry == "cross-industry"
    assert doc.topics == ["test", "fixture"]
    assert doc.date == date(2024, 1, 15)
    assert "## Overview" in doc.body
    assert doc.path == path


@pytest.mark.parametrize(
    "missing_key",
    ["title", "doc_type", "industry", "topics", "date"],
)
def test_load_document_raises_on_missing_key(
    tmp_path: Path, missing_key: str
) -> None:
    lines = [
        line
        for line in VALID_FILE.splitlines()
        if not line.startswith(f"{missing_key}:")
    ]
    path = _write(tmp_path, "broken.md", "\n".join(lines))

    with pytest.raises(CorpusError, match=missing_key):
        load_document(path)


def test_load_document_rejects_invalid_doc_type(tmp_path: Path) -> None:
    content = VALID_FILE.replace("doc_type: methodology", "doc_type: powerpoint")
    path = _write(tmp_path, "bad-doctype.md", content)

    with pytest.raises(CorpusError, match="doc_type"):
        load_document(path)


def test_load_document_rejects_non_date_date(tmp_path: Path) -> None:
    content = VALID_FILE.replace("date: 2024-01-15", 'date: "not a date"')
    path = _write(tmp_path, "bad-date.md", content)

    with pytest.raises(CorpusError, match="date"):
        load_document(path)


def test_load_document_rejects_empty_topics(tmp_path: Path) -> None:
    content = VALID_FILE.replace("topics: [test, fixture]", "topics: []")
    path = _write(tmp_path, "empty-topics.md", content)

    with pytest.raises(CorpusError, match="topics"):
        load_document(path)


def test_load_corpus_raises_on_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="no markdown files"):
        load_corpus(tmp_path)


def test_load_corpus_is_sorted_and_deterministic(tmp_path: Path) -> None:
    _write(tmp_path, "z-doc.md", VALID_FILE.replace("Test Doc", "Z Doc"))
    _write(tmp_path, "a-doc.md", VALID_FILE.replace("Test Doc", "A Doc"))

    docs = load_corpus(tmp_path)

    assert [d.title for d in docs] == ["A Doc", "Z Doc"]


# --- Integration: the real Task 1 corpus ---


def test_real_corpus_loads_cleanly() -> None:
    docs = load_corpus(CORPUS_DIR)

    assert len(docs) == 71  # 57 internal + 12 restricted engagements + 2 case studies
    superseded = {d.path.name: d.superseded_by for d in docs if d.is_superseded}
    assert len(superseded) == 5  # P4-4's freshness set
    assert sum(d.is_quarantined for d in docs) == 2  # P4-5's pending-review case studies
    doc_types = {d.doc_type for d in docs}
    assert doc_types == {"methodology", "thought_leadership", "engagement", "case_study"}
    for d in docs:
        assert d.body.strip(), f"{d.path} has empty body"
        assert d.topics, f"{d.path} has no topics"


# --- Freshness and review metadata (Phase 4, P4-4) ---


def _superseded_file(superseded_by: str = "new.md") -> str:
    return VALID_FILE.replace(
        "date: 2024-01-15", f"date: 2019-03-01\nstatus: superseded\nsuperseded_by: {superseded_by}"
    )


def test_status_defaults_to_current(tmp_path: Path) -> None:
    doc = load_document(_write(tmp_path, "doc.md", VALID_FILE))

    assert doc.status == "current" and not doc.is_superseded
    assert doc.superseded_by is None and not doc.is_quarantined


def test_a_superseded_document_resolves_its_replacement(tmp_path: Path) -> None:
    _write(tmp_path, "new.md", VALID_FILE)
    _write(tmp_path, "old.md", _superseded_file())

    old = next(d for d in load_corpus(tmp_path) if d.path.name == "old.md")

    assert old.is_superseded
    assert old.superseded_by == "new.md"
    assert old.superseded_by_path == tmp_path / "new.md"


@pytest.mark.parametrize(
    ("replace_from", "replace_to", "match"),
    [
        ("date: 2024-01-15", "date: 2024-01-15\nstatus: retired", "status"),
        ("date: 2024-01-15", "date: 2024-01-15\nstatus: superseded", "superseded_by is not set"),
        ("date: 2024-01-15", "date: 2024-01-15\nsuperseded_by: x.md", "status is not superseded"),
        ("date: 2024-01-15", "date: 2024-01-15\nreview_status: maybe", "review_status"),
    ],
)
def test_invalid_freshness_metadata_is_rejected(
    tmp_path: Path, replace_from: str, replace_to: str, match: str
) -> None:
    path = _write(tmp_path, "doc.md", VALID_FILE.replace(replace_from, replace_to))

    with pytest.raises(CorpusError, match=match):
        load_document(path)


@pytest.mark.parametrize(
    ("files", "match"),
    [
        ({"old.md": _superseded_file("missing.md")}, "not in the corpus"),
        ({"old.md": _superseded_file("old.md")}, "points at itself"),
        (
            {"a.md": _superseded_file("b.md"), "b.md": _superseded_file("c.md"), "c.md": VALID_FILE},
            "itself superseded",
        ),
    ],
)
def test_superseded_by_must_name_a_current_corpus_document(
    tmp_path: Path, files: dict[str, str], match: str
) -> None:
    for name, content in files.items():
        _write(tmp_path, name, content)

    with pytest.raises(CorpusError, match=match):
        load_corpus(tmp_path)


def test_pending_review_marks_a_document_quarantined(tmp_path: Path) -> None:
    content = VALID_FILE.replace("date: 2024-01-15", "date: 2024-01-15\nreview_status: pending")

    assert load_document(_write(tmp_path, "doc.md", content)).is_quarantined
