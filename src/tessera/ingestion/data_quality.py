"""The corpus data-quality report behind `tessera data-report` (Phase 4,
plan §3.3.3).

Deterministic and LLM-free: front-matter problems, near-duplicate chunks
by embedding similarity, stale documents by date, superseded documents and
their replacements, and quarantined documents. Pure — the caller (cli.py)
reads the corpus and computes the embeddings; this module only analyses
what it is given and returns data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from tessera.ingestion.chunker import Chunk
from tessera.ingestion.loader import Document

# Cosine similarity of two chunks' body text (not the title-prefixed
# embedding text) at or above which they are reported as near-duplicates.
# Calibrated on the pilot corpus (2026-10-03, all-MiniLM-L6-v2): 19
# Related Frameworks pairs clear it (0.86-1.00, the five superseded
# documents' copies of their replacements' lists at 1.00); the most similar
# pair outside that section is 0.81 — two sibling documents' sections that
# refer to each other, related rather than duplicated.
NEAR_DUPLICATE_THRESHOLD = 0.85

# A current document dated more than this many years before the report
# date is flagged as stale — a prompt for its owner to re-review, not a
# claim that it is wrong.
STALE_AFTER_YEARS = 3

# The section the corpus repeats on purpose as retrieval hard negatives
# (checkpoint.md Notes): listed as known near-duplicates, not as problems.
KNOWN_DUPLICATE_SECTION = "Related Frameworks"


@dataclass(frozen=True)
class NearDuplicate:
    """Two chunks from different documents with near-identical text."""

    chunk_a: str
    chunk_b: str
    document_a: str
    document_b: str
    similarity: float
    # True when either side belongs to a superseded document — an old
    # version overlapping its replacement is expected, not a defect.
    involves_superseded: bool = False


@dataclass(frozen=True)
class DocumentEntry:
    """One document as a report row. ``detail`` is the row's reason: the
    age for stale, the replacement for superseded, empty otherwise.
    """

    path: str
    title: str
    date: date
    detail: str = ""


@dataclass(frozen=True)
class DataQualityReport:
    as_of: date
    documents: int
    chunks: int
    metadata_problems: dict[str, list[str]]
    near_duplicates: list[NearDuplicate]
    known_near_duplicates: list[NearDuplicate]
    stale: list[DocumentEntry]
    superseded: list[DocumentEntry]
    quarantined: list[DocumentEntry]
    # Restricted documents (Phase 4, §3.5.1), detail = engagement or a note
    # that the document is restricted only because it has no label.
    restricted: list[DocumentEntry] = field(default_factory=list)
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD
    stale_after_years: int = STALE_AFTER_YEARS


def _relative(path: Path, corpus_dir: Path) -> str:
    try:
        return path.relative_to(corpus_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _is_known_duplicate(chunk: Chunk) -> bool:
    return bool(chunk.heading_path) and chunk.heading_path[-1] == KNOWN_DUPLICATE_SECTION


def _years_between(earlier: date, later: date) -> float:
    return (later - earlier).days / 365.25


def near_duplicates(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    corpus_dir: Path,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> tuple[list[NearDuplicate], list[NearDuplicate]]:
    """Cross-document chunk pairs at or above ``threshold`` cosine
    similarity, most similar first: (unexpected, known). Known pairs are
    both from the deliberate Related Frameworks sections.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")
    vectors = [_normalized(e) for e in embeddings]
    found: list[NearDuplicate] = []
    known: list[NearDuplicate] = []
    for i, a in enumerate(chunks):
        for j in range(i + 1, len(chunks)):
            b = chunks[j]
            if a.document_path == b.document_path:
                continue
            similarity = sum(x * y for x, y in zip(vectors[i], vectors[j]))
            if similarity < threshold:
                continue
            pair = NearDuplicate(
                chunk_a=a.chunk_id,
                chunk_b=b.chunk_id,
                document_a=_relative(a.document_path, corpus_dir),
                document_b=_relative(b.document_path, corpus_dir),
                similarity=round(similarity, 4),
                involves_superseded=a.status == "superseded" or b.status == "superseded",
            )
            (known if _is_known_duplicate(a) and _is_known_duplicate(b) else found).append(pair)
    order = lambda p: (-p.similarity, p.chunk_a, p.chunk_b)  # noqa: E731
    return sorted(found, key=order), sorted(known, key=order)


def build_report(
    documents: list[Document],
    metadata_problems: dict[str, list[str]],
    chunks: list[Chunk],
    embeddings: list[list[float]],
    corpus_dir: Path,
    as_of: date,
    *,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    stale_after_years: int = STALE_AFTER_YEARS,
) -> DataQualityReport:
    """Assemble the report. ``documents`` are the ones that loaded,
    ``metadata_problems`` the per-path problems of the rest (both from
    loader.scan_corpus); ``chunks``/``embeddings`` are index-aligned and
    cover the documents that would be embedded.
    """
    found, known = near_duplicates(chunks, embeddings, corpus_dir, near_duplicate_threshold)

    def entry(d: Document, detail: str = "") -> DocumentEntry:
        return DocumentEntry(_relative(d.path, corpus_dir), d.title, d.date, detail)

    loaded = {entry(d).path for d in documents}
    stale = [
        entry(d, f"{_years_between(d.date, as_of):.1f} years old")
        for d in documents
        if not d.is_superseded
        and not d.is_quarantined
        and _years_between(d.date, as_of) > stale_after_years
    ]
    return DataQualityReport(
        as_of=as_of,
        # Loaded documents plus those that failed to load.
        documents=len(documents) + len(set(metadata_problems) - loaded),
        chunks=len(chunks),
        metadata_problems=dict(sorted(metadata_problems.items())),
        near_duplicates=found,
        known_near_duplicates=known,
        stale=sorted(stale, key=lambda e: (e.date, e.path)),
        superseded=[
            entry(d, f"superseded by {d.superseded_by}") for d in documents if d.is_superseded
        ],
        quarantined=[entry(d, "review_status: pending") for d in documents if d.is_quarantined],
        restricted=[
            entry(
                d,
                f"engagement: {d.engagement}"
                if d.engagement
                else "no sensitivity label outside the open directories — restricted by default",
            )
            for d in documents
            if d.is_restricted
        ],
        near_duplicate_threshold=near_duplicate_threshold,
        stale_after_years=stale_after_years,
    )


def format_data_report(report: DataQualityReport, *, show_known: bool = False) -> str:
    """The report as printed by `tessera data-report`."""
    lines = [
        "=== Tessera Data-Quality Report ===",
        f"As of {report.as_of.isoformat()} · {report.documents} documents · "
        f"{report.chunks} chunks embedded",
        "",
    ]

    def section(title: str, rows: list[str]) -> None:
        lines.append(f"{title}: {len(rows) if rows else 'none'}")
        lines.extend(f"  {r}" for r in rows)
        lines.append("")

    section(
        "Missing or invalid metadata",
        [f"{path}: {'; '.join(problems)}" for path, problems in report.metadata_problems.items()],
    )

    def pair_row(p: NearDuplicate) -> str:
        tag = "  (superseded version)" if p.involves_superseded else ""
        return f"{p.similarity:.2f}  {p.chunk_a}  ~  {p.chunk_b}{tag}"

    section(
        f"Near-duplicate chunks (cosine >= {report.near_duplicate_threshold:.2f}, "
        "across documents)",
        [pair_row(p) for p in report.near_duplicates],
    )
    known = report.known_near_duplicates
    lines.append(
        f"Known near-duplicates ({KNOWN_DUPLICATE_SECTION} hard negatives, kept on "
        f"purpose): {len(known)} pairs"
        + ("" if show_known or not known else " — `--show-known` lists them")
    )
    if show_known:
        lines.extend(f"  {pair_row(p)}" for p in known)
    lines.append("")

    section(
        f"Stale (current, dated more than {report.stale_after_years} years ago)",
        [f"{e.date.isoformat()}  {e.path} — {e.detail}" for e in report.stale],
    )
    section(
        "Superseded (excluded from answers)",
        [f"{e.date.isoformat()}  {e.path} — {e.detail}" for e in report.superseded],
    )
    section(
        "Quarantined (awaiting human review; not embedded)",
        [f"{e.date.isoformat()}  {e.path} — {e.detail}" for e in report.quarantined],
    )
    section(
        "Restricted (ethical walls; cleared principals only)",
        [f"{e.path} — {e.detail}" for e in report.restricted],
    )
    return "\n".join(lines).rstrip()
