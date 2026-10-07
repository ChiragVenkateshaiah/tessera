"""The LangChain splitter (plan §3.3, the ``splitter`` switch).

``MarkdownHeaderTextSplitter`` on every heading level (``#`` to ``######``,
as native splits), then
``RecursiveCharacterTextSplitter`` for sections over ``chunk_size``
characters — the framework counterpart of ``chunker.py``'s section-aware
split. Chunks are built through the native ``make_chunk``, so ids stay
``<stem>::<n>`` and the labels match native exactly.

Recorded choices (plan §3.3):
- **``strip_headers=True`` by default**: native chunk text excludes the
  heading line (the heading lives in ``heading_path`` and the embedding
  prefix), and the stored text is what the answer prompt renders. With
  ``False`` the text starts with ``## Heading``; the comparison measures
  both.
- **All six heading levels**, as native: the corpus goes down to ``####``
  (the M&A day-one runbook's hour blocks).
- **An empty parent heading** (``## Approach`` directly followed by
  ``### Step one``) is folded into the child section's text when
  ``strip_headers=False``; native drops headings with no own text.
- **What it does that native won't** (P5-1 spike): the character splitter
  cuts through tables and fenced blocks when they exceed ``chunk_size``;
  native keeps them whole. The comparison counts both.
"""

from __future__ import annotations

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from tessera.ingestion.chunker import Chunk, make_chunk
from tessera.ingestion.loader import Document

DEFAULT_CHUNK_SIZE = 1600  # characters; chosen on query_log.yaml (evals/reports/p5-4-ingestion.md)
HEADERS = [("#" * level, f"h{level}") for level in range(1, 7)]


def split_document(
    doc: Document,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = 0,
    strip_headers: bool = True,
) -> list[Chunk]:
    """Split one document into chunks, the LangChain way."""
    sections = MarkdownHeaderTextSplitter(HEADERS, strip_headers=strip_headers).split_text(doc.body)
    pieces = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(sections)
    chunks: list[Chunk] = []
    for piece in pieces:
        text = piece.page_content.strip()
        if not text:
            continue
        heading_path = tuple(piece.metadata[key] for _, key in HEADERS if key in piece.metadata)
        chunks.append(make_chunk(doc, len(chunks), heading_path, text))
    return chunks
