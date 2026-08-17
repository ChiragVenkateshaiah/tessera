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

    assert len(docs) == 52
    doc_types = {d.doc_type for d in docs}
    assert doc_types == {"methodology", "thought_leadership"}
    for d in docs:
        assert d.body.strip(), f"{d.path} has empty body"
        assert d.topics, f"{d.path} has no topics"
