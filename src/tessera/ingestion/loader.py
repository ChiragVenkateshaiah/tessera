"""Reads the corpus with front-matter metadata intact."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import frontmatter

REQUIRED_KEYS = {"title", "doc_type", "industry", "topics", "date"}
VALID_DOC_TYPES = {"methodology", "thought_leadership"}


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


def load_document(path: Path) -> Document:
    """Parse one markdown file's front matter and body into a Document.

    Raises CorpusError if required front-matter keys are missing or hold
    an invalid value — ingestion fails loudly on a malformed corpus file
    rather than silently indexing something without citation metadata.
    """
    post = frontmatter.load(path)

    missing = REQUIRED_KEYS - post.metadata.keys()
    if missing:
        raise CorpusError(f"{path}: missing front-matter keys {sorted(missing)}")

    doc_type = post["doc_type"]
    if doc_type not in VALID_DOC_TYPES:
        raise CorpusError(
            f"{path}: doc_type {doc_type!r} not in {sorted(VALID_DOC_TYPES)}"
        )

    doc_date = post["date"]
    if not isinstance(doc_date, date_type):
        raise CorpusError(f"{path}: date {doc_date!r} is not a valid ISO date")

    topics = post["topics"]
    if not isinstance(topics, list) or not topics:
        raise CorpusError(f"{path}: topics must be a non-empty list")

    title = post["title"]
    if not isinstance(title, str) or not title.strip():
        raise CorpusError(f"{path}: title is missing or empty")

    return Document(
        path=path,
        title=title,
        doc_type=doc_type,
        industry=post["industry"],
        topics=list(topics),
        date=doc_date,
        body=post.content,
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every markdown document under corpus_dir, sorted for determinism."""
    paths = sorted(corpus_dir.rglob("*.md"))
    if not paths:
        raise CorpusError(f"no markdown files found under {corpus_dir}")
    return [load_document(p) for p in paths]
