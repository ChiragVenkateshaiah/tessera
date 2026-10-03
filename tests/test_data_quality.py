"""Unit tests for the data-quality report (P4-4) and the lenient corpus
scan behind it — synthetic corpora and hand-built embeddings, no model.
"""

from datetime import date
from pathlib import Path

from tessera.ingestion.chunker import chunk_corpus
from tessera.ingestion.data_quality import build_report, format_data_report, near_duplicates
from tessera.ingestion.loader import indexable, scan_corpus

DOC = """\
---
title: "{title}"
doc_type: methodology
industry: cross-industry
topics: [test]
date: {date}
{extra}
---

## Overview

{body}

## Related Frameworks

Pricing strategy, market sizing.
"""


def _write(root: Path, name: str, title: str = "Doc", date_: str = "2025-01-01",
           extra: str = "", body: str = "Body text.") -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DOC.format(title=title, date=date_, extra=extra, body=body), encoding="utf-8")


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    _write(root, "methodology/new.md", "New", body="Current method.")
    _write(root, "methodology/old.md", "Old", "2019-01-01",
           "status: superseded\nsuperseded_by: methodology/new.md", body="Retired method.")
    _write(root, "methodology/aging.md", "Aging", "2020-06-01", body="Still current, long unreviewed.")
    _write(root, "methodology/pending.md", "Pending", extra="review_status: pending")
    (root / "methodology/broken.md").write_text(
        "---\ntitle: Broken\ndoc_type: deck\n---\n\nbody\n", encoding="utf-8"
    )
    return root


def test_scan_corpus_reports_bad_files_instead_of_raising(tmp_path: Path) -> None:
    documents, problems = scan_corpus(_corpus(tmp_path))

    assert sorted(d.path.name for d in documents) == ["aging.md", "new.md", "old.md", "pending.md"]
    assert list(problems) == ["methodology/broken.md"]
    assert any("missing front-matter keys" in p for p in problems["methodology/broken.md"])
    assert any("doc_type 'deck'" in p for p in problems["methodology/broken.md"])


def test_scan_corpus_attributes_a_dangling_superseded_by(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write(root, "a.md", extra="status: superseded\nsuperseded_by: gone.md")

    _, problems = scan_corpus(root)

    assert problems == {"a.md": ["superseded_by 'gone.md' is not in the corpus"]}


def test_report_lists_every_category(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    documents, problems = scan_corpus(root)
    chunks = chunk_corpus(indexable(documents))
    # Every chunk orthogonal except the Related Frameworks sections, which
    # are identical across documents (the deliberate hard negatives).
    def one_hot(i: int) -> list[float]:
        return [1.0 if j == i else 0.0 for j in range(len(chunks) + 1)]

    embeddings = [
        one_hot(0) if c.heading_path[-1] == "Related Frameworks" else one_hot(i + 1)
        for i, c in enumerate(chunks)
    ]

    report = build_report(documents, problems, chunks, embeddings, root, date(2026, 10, 3))

    assert report.documents == 5
    assert "pending.md" not in {c.document_path.name for c in chunks}
    assert list(report.metadata_problems) == ["methodology/broken.md"]
    assert [e.path for e in report.stale] == ["methodology/aging.md"]  # old.md is superseded, not stale
    assert [(e.path, e.detail) for e in report.superseded] == [
        ("methodology/old.md", "superseded by methodology/new.md")
    ]
    assert [e.path for e in report.quarantined] == ["methodology/pending.md"]
    assert report.restricted == []  # everything here sits in an open directory
    assert report.near_duplicates == []
    assert len(report.known_near_duplicates) == 3  # aging~new, aging~old, new~old
    assert sum(p.involves_superseded for p in report.known_near_duplicates) == 2

    text = format_data_report(report)
    for heading in ("Missing or invalid metadata: 1", "Near-duplicate chunks", "Known near-duplicates",
                    "Stale", "Superseded (excluded from answers): 1", "Quarantined"):
        assert heading in text
    assert "--show-known" in text


def test_near_duplicates_skip_same_document_pairs_and_flag_other_sections(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write(root, "a.md", body="Same words.")
    _write(root, "b.md", body="Same words.")
    documents, _ = scan_corpus(root)
    chunks = chunk_corpus(documents)  # a::0, a::1, b::0, b::1
    embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.6, 0.8]]

    found, known = near_duplicates(chunks, embeddings, root, threshold=0.9)

    assert [(p.chunk_a, p.chunk_b, p.similarity) for p in found] == [("a::0", "b::0", 1.0)]
    assert known == []  # a::1~b::1 is 0.8, under the threshold


def test_report_lists_restricted_documents_and_unlabelled_defaults(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write(root, "engagements/a.md", extra="sensitivity: restricted\nengagement: halcyon")
    _write(root, "inbox/b.md")  # unlabelled, outside the open directories
    documents, problems = scan_corpus(root)

    report = build_report(documents, problems, [], [], root, date(2026, 10, 3))

    assert [(e.path, e.detail) for e in report.restricted] == [
        ("engagements/a.md", "engagement: halcyon"),
        ("inbox/b.md", "no sensitivity label outside the open directories — restricted by default"),
    ]
    assert "Restricted (ethical walls; cleared principals only): 2" in format_data_report(report)
