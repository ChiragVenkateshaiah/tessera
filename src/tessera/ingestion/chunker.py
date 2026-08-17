"""Section-aware chunking that respects markdown heading boundaries.

Chunks split on headings, never on the arbitrary fixed-size windows a naive
chunker would use — this keeps a framework's steps, a table, a checklist, or
a fenced code block coherent rather than sliced mid-structure. A section
that still exceeds the word budget after that is subdivided further, but
only at blank-line paragraph boundaries, and never inside a fenced code
block — so a table or list too big to subdivide any further is kept intact
rather than torn in two, trading strict size-budget adherence for structural
correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path

from tessera.ingestion.loader import Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^```")

# Target max words per chunk before subdividing on paragraph boundaries.
# Chosen so most single-topic sections in this corpus (median well under
# this) stay as one chunk, while the corpus's longer sections (the
# multi-paragraph "Framework" sections, the network-optimization spec)
# actually exercise the subdivision path.
DEFAULT_MAX_CHUNK_WORDS = 180


@dataclass
class Chunk:
    """A retrievable unit of text with full source and citation metadata."""

    chunk_id: str
    document_path: Path
    document_title: str
    doc_type: str
    industry: str
    topics: list[str]
    date: date_type
    heading_path: tuple[str, ...]
    text: str
    chunk_index: int


@dataclass
class _Node:
    level: int
    title: str
    own_lines: list[str] = field(default_factory=list)
    children: list["_Node"] = field(default_factory=list)


def _parse_heading_tree(body: str) -> _Node:
    """Parse markdown into a heading tree.

    The root is a synthetic level-0 node holding any preamble text before
    the first heading. Lines inside a fenced code block are never treated
    as heading markers, so a `#` comment inside a code fence can't split
    the tree.
    """
    root = _Node(level=0, title="")
    stack: list[_Node] = [root]
    in_fence = False

    for line in body.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            stack[-1].own_lines.append(line)
            continue

        match = None if in_fence else HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack[-1].level >= level:
                stack.pop()
            node = _Node(level=level, title=title)
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].own_lines.append(line)

    return root


def _flatten(
    node: _Node, path: tuple[str, ...], out: list[tuple[tuple[str, ...], str]]
) -> None:
    """Depth-first walk collecting (heading_path, own_text) for every node
    that has non-empty own text — i.e. content directly under that heading,
    not counting its children's content.
    """
    own_text = "\n".join(node.own_lines).strip()
    node_path = path if node.level == 0 else (*path, node.title)
    if own_text:
        out.append((node_path, own_text))
    for child in node.children:
        _flatten(child, node_path, out)


def _split_paragraphs(text: str) -> list[str]:
    """Split into blank-line-delimited paragraph blocks.

    A blank line inside a fenced code block never counts as a paragraph
    break, so a fence is always kept as one atomic block. List items and
    table rows have no blank lines between them in this corpus, so they
    fall out of this naturally as single atomic blocks too — nothing
    table- or list-specific needed here.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    in_fence = False

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            current.append(line)
            continue
        if not in_fence and line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    return ["\n".join(b).strip() for b in blocks if any(l.strip() for l in b)]


def _split_oversized(text: str, max_words: int) -> list[str]:
    """Subdivide text exceeding max_words at paragraph boundaries.

    If the text can't be subdivided further (it's a single atomic
    paragraph block — e.g. one big table or one big fence), it's returned
    whole rather than split, since there's no boundary that wouldn't cut
    through a structural element.
    """
    if len(text.split()) <= max_words:
        return [text]

    paragraphs = _split_paragraphs(text)
    if len(paragraphs) <= 1:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    current_words = 0
    for para in paragraphs:
        para_words = len(para.split())
        if current and current_words + para_words > max_words:
            pieces.append("\n\n".join(current))
            current = []
            current_words = 0
        current.append(para)
        current_words += para_words
    if current:
        pieces.append("\n\n".join(current))

    return pieces


def chunk_document(
    doc: Document, max_words: int = DEFAULT_MAX_CHUNK_WORDS
) -> list[Chunk]:
    """Chunk a single document into heading-bounded, cite-able pieces."""
    tree = _parse_heading_tree(doc.body)
    sections: list[tuple[tuple[str, ...], str]] = []
    _flatten(tree, (), sections)

    chunks: list[Chunk] = []
    index = 0
    for heading_path, text in sections:
        for piece in _split_oversized(text, max_words):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.path.stem}::{index}",
                    document_path=doc.path,
                    document_title=doc.title,
                    doc_type=doc.doc_type,
                    industry=doc.industry,
                    topics=doc.topics,
                    date=doc.date,
                    heading_path=heading_path,
                    text=piece,
                    chunk_index=index,
                )
            )
            index += 1
    return chunks


def chunk_corpus(
    documents: list[Document], max_words: int = DEFAULT_MAX_CHUNK_WORDS
) -> list[Chunk]:
    """Chunk every document in the corpus."""
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, max_words=max_words))
    return chunks
