"""Reads the corpus with front-matter metadata intact."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date as date_type
from pathlib import Path
from typing import Any

import frontmatter

REQUIRED_KEYS = {"title", "doc_type", "industry", "topics", "date"}
VALID_DOC_TYPES = {"methodology", "thought_leadership", "engagement", "case_study"}

# Access labels (Phase 4, plan §3.5.1). Ethical walls are per engagement:
# a restricted document names its engagement, and only principals cleared
# for that engagement may retrieve it. Internal documents are open to all.
SENSITIVITY_INTERNAL = "internal"
SENSITIVITY_RESTRICTED = "restricted"
VALID_SENSITIVITIES = {SENSITIVITY_INTERNAL, SENSITIVITY_RESTRICTED}
# The pilot corpus's two directories, low-sensitivity by construction: an
# unlabelled document there is internal. Anywhere else an unlabelled
# document is restricted — fail closed.
OPEN_DIRECTORIES = frozenset({"methodology", "thought_leadership"})

# Freshness (Phase 4, plan §3.3.1). A superseded document stays in the
# corpus — it is still the record of what the firm used to do — but names
# the corpus-relative path of the document that replaced it.
STATUS_CURRENT = "current"
STATUS_SUPERSEDED = "superseded"
VALID_STATUSES = {STATUS_CURRENT, STATUS_SUPERSEDED}

# Anonymized material awaiting a human review pass (plan §3.5.3) is
# quarantined: loaded and reported, never embedded.
REVIEW_PENDING = "pending"
REVIEW_REVIEWED = "reviewed"
VALID_REVIEW_STATUSES = {REVIEW_PENDING, REVIEW_REVIEWED}


class CorpusError(ValueError):
    """A corpus file is missing required front matter or fails to parse."""


@dataclass(frozen=True)
class Document:
    """A single corpus document with parsed front matter and markdown body."""

    path: Path
    title: str
    doc_type: str
    industry: str
    topics: list[str]
    date: date_type
    body: str
    status: str = STATUS_CURRENT
    # Corpus-relative path of the replacing document; set iff superseded.
    superseded_by: str | None = None
    # superseded_by resolved against the corpus dir, in the same form as
    # ``path`` — what the chunks and the store carry. Set by load_corpus().
    superseded_by_path: Path | None = None
    review_status: str | None = None
    # As labelled in front matter; load_corpus()/scan_corpus() resolve an
    # absent label by directory (see OPEN_DIRECTORIES), so a loaded
    # corpus never carries None here.
    sensitivity: str | None = None
    engagement: str | None = None

    @property
    def is_superseded(self) -> bool:
        return self.status == STATUS_SUPERSEDED

    @property
    def is_quarantined(self) -> bool:
        return self.review_status == REVIEW_PENDING

    @property
    def is_restricted(self) -> bool:
        # Fail closed: anything not positively internal is restricted.
        return self.sensitivity != SENSITIVITY_INTERNAL


def front_matter_problems(metadata: dict[str, Any]) -> list[str]:
    """Everything wrong with one document's front matter, in a stable
    order; empty when it is valid. load_document() raises on these and
    `tessera data-report` lists them.
    """
    problems: list[str] = []
    missing = REQUIRED_KEYS - metadata.keys()
    if missing:
        problems.append(f"missing front-matter keys {sorted(missing)}")

    if "doc_type" in metadata and metadata["doc_type"] not in VALID_DOC_TYPES:
        problems.append(
            f"doc_type {metadata['doc_type']!r} not in {sorted(VALID_DOC_TYPES)}"
        )
    if "date" in metadata and not isinstance(metadata["date"], date_type):
        problems.append(f"date {metadata['date']!r} is not a valid ISO date")
    if "topics" in metadata:
        topics = metadata["topics"]
        if not isinstance(topics, list) or not topics:
            problems.append("topics must be a non-empty list")
    if "title" in metadata:
        title = metadata["title"]
        if not isinstance(title, str) or not title.strip():
            problems.append("title is missing or empty")

    status = metadata.get("status", STATUS_CURRENT)
    superseded_by = metadata.get("superseded_by")
    if status not in VALID_STATUSES:
        problems.append(f"status {status!r} not in {sorted(VALID_STATUSES)}")
    elif status == STATUS_SUPERSEDED and not superseded_by:
        problems.append("status is superseded but superseded_by is not set")
    elif status == STATUS_CURRENT and superseded_by:
        problems.append("superseded_by is set but status is not superseded")
    if superseded_by is not None and not isinstance(superseded_by, str):
        problems.append(f"superseded_by {superseded_by!r} is not a path")

    sensitivity = metadata.get("sensitivity")
    engagement = metadata.get("engagement")
    if sensitivity is not None and sensitivity not in VALID_SENSITIVITIES:
        problems.append(
            f"sensitivity {sensitivity!r} not in {sorted(VALID_SENSITIVITIES)}"
        )
    if sensitivity == SENSITIVITY_RESTRICTED and not engagement:
        problems.append("sensitivity is restricted but engagement is not set")
    if engagement is not None and sensitivity != SENSITIVITY_RESTRICTED:
        problems.append("engagement is set but sensitivity is not restricted")
    if engagement is not None and not (
        isinstance(engagement, str) and engagement.isidentifier() and engagement.islower()
    ):
        problems.append(f"engagement {engagement!r} is not a lowercase codename")

    review_status = metadata.get("review_status")
    if review_status is not None and review_status not in VALID_REVIEW_STATUSES:
        problems.append(
            f"review_status {review_status!r} not in {sorted(VALID_REVIEW_STATUSES)}"
        )
    return problems


def load_document(path: Path) -> Document:
    """Parse one markdown file's front matter and body into a Document.

    Raises CorpusError if required front-matter keys are missing or hold
    an invalid value — ingestion fails loudly on a malformed corpus file
    rather than silently indexing something without citation metadata.
    """
    post = frontmatter.load(path)

    problems = front_matter_problems(post.metadata)
    if problems:
        raise CorpusError(f"{path}: {'; '.join(problems)}")

    return Document(
        path=path,
        title=post["title"],
        doc_type=post["doc_type"],
        industry=post["industry"],
        topics=list(post["topics"]),
        date=post["date"],
        body=post.content,
        status=post.get("status", STATUS_CURRENT),
        superseded_by=post.get("superseded_by"),
        review_status=post.get("review_status"),
        sensitivity=post.get("sensitivity"),
        engagement=post.get("engagement"),
    )


def supersession_problems(documents: list[Document], corpus_dir: Path) -> list[str]:
    """Cross-document checks on superseded_by: the target exists in the
    corpus, is not the document itself, and is current (no chains — the
    answer must be able to point straight at the version to use).
    """
    by_path = {d.path.relative_to(corpus_dir).as_posix(): d for d in documents}
    problems: list[str] = []
    for rel, doc in by_path.items():
        if doc.superseded_by is None:
            continue
        target = by_path.get(doc.superseded_by)
        if target is None:
            problems.append(f"{rel}: superseded_by {doc.superseded_by!r} is not in the corpus")
        elif doc.superseded_by == rel:
            problems.append(f"{rel}: superseded_by points at itself")
        elif target.is_superseded:
            problems.append(
                f"{rel}: superseded_by {doc.superseded_by!r} is itself superseded"
            )
    return problems


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every markdown document under corpus_dir, sorted for determinism."""
    paths = sorted(corpus_dir.rglob("*.md"))
    if not paths:
        raise CorpusError(f"no markdown files found under {corpus_dir}")
    documents = [load_document(p) for p in paths]
    problems = supersession_problems(documents, corpus_dir)
    if problems:
        raise CorpusError("; ".join(problems))
    return [_resolved(d, corpus_dir) for d in documents]


def _resolved(doc: Document, corpus_dir: Path) -> Document:
    """Fill in what needs the corpus root: the replacement's path, and a
    missing sensitivity label from the document's directory.
    """
    sensitivity = doc.sensitivity
    if sensitivity is None:
        top = doc.path.relative_to(corpus_dir).parts[0]
        sensitivity = (
            SENSITIVITY_INTERNAL if top in OPEN_DIRECTORIES else SENSITIVITY_RESTRICTED
        )
    return replace(
        doc,
        sensitivity=sensitivity,
        superseded_by_path=corpus_dir / doc.superseded_by if doc.superseded_by else None,
    )


def indexable(documents: list[Document]) -> list[Document]:
    """The documents ingestion embeds: everything except quarantined
    material awaiting human review (plan §3.5.3).
    """
    return [d for d in documents if not d.is_quarantined]


def scan_corpus(corpus_dir: Path) -> tuple[list[Document], dict[str, list[str]]]:
    """Lenient load for reporting (`tessera data-report`): every document
    that parses, plus the problems of every one that doesn't, keyed by
    corpus-relative path. load_corpus() is the strict version ingestion
    uses — it raises on the first bad file.
    """
    documents: list[Document] = []
    problems: dict[str, list[str]] = {}
    for path in sorted(corpus_dir.rglob("*.md")):
        rel = path.relative_to(corpus_dir).as_posix()
        try:
            metadata = frontmatter.load(path).metadata
        except Exception as exc:  # unparseable YAML front matter
            problems[rel] = [f"front matter does not parse: {exc}"]
            continue
        found = front_matter_problems(metadata)
        if found:
            problems[rel] = found
        else:
            documents.append(load_document(path))
    for problem in supersession_problems(documents, corpus_dir):
        rel, _, detail = problem.partition(": ")
        problems.setdefault(rel, []).append(detail)
    return [_resolved(d, corpus_dir) for d in documents], problems
