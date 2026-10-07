"""Native indexing: delete-then-add (Phase 5, P5-4; plan §3.3).

Every document's chunks are deleted from the store before its new chunks
are added. Upsert alone (Phases 1–4) replaced chunks by id, so a document
whose chunk count shrank kept its old tail chunks — still carrying the
labels they were indexed with. Deleting first means nothing stale
survives:
- a document that shrinks loses its old tail chunks;
- a document that is relabelled (say restricted → internal, or a new
  engagement) keeps no chunk under its old labels;
- a document newly quarantined for review is removed from the index,
  because quarantined documents are deleted too and then not re-added.

A file removed from the corpus entirely is not seen here (nothing loads
it), so its chunks stay until the index is rebuilt. That is the same gap
LangChain's ``index(cleanup="incremental")`` has; ``cleanup="full"`` closes
it — see ``tessera.lc.indexing``.

Pure with respect to infrastructure (CLAUDE.md constraint #6): the
embedder and the store are ports passed in; the splitter and the
embedding text are parameters, so the LangChain splitter and the
``embed_prefix`` switch reuse this path unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from tessera.embedding.base import Embedder
from tessera.ingestion.chunker import Chunk, chunk_document, chunk_embedding_text
from tessera.ingestion.loader import Document, indexable
from tessera.store.base import VectorStore

Splitter = Callable[[Document], list[Chunk]]


@dataclass(frozen=True)
class IndexingResult:
    """What one indexing run did."""

    documents: int  # documents loaded (quarantined included)
    quarantined: int  # of those, held out of the index
    chunks_added: int
    chunks_deleted: int  # chunks removed before adding
    # Chunks left as they were. Always 0 for delete-then-add, which
    # re-embeds everything; LangChain's index() skips unchanged chunks.
    chunks_unchanged: int = 0


def index_corpus(
    documents: Iterable[Document],
    embedder: Embedder,
    store: VectorStore,
    *,
    split: Splitter = chunk_document,
    embedding_text: Callable[[Chunk], str] = chunk_embedding_text,
) -> IndexingResult:
    """Delete every loaded document's chunks, then add the chunks of the
    indexable ones (everything but quarantined documents)."""
    documents = list(documents)
    deleted = sum(store.delete_document(str(doc.path)) for doc in documents)
    kept = indexable(documents)
    chunks = [chunk for doc in kept for chunk in split(doc)]
    if chunks:
        store.add(chunks, embedder.embed_documents([embedding_text(c) for c in chunks]))
    return IndexingResult(
        documents=len(documents),
        quarantined=len(documents) - len(kept),
        chunks_added=len(chunks),
        chunks_deleted=deleted,
    )
