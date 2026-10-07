"""The LangChain store (plan §3.3, the ``store`` switch).

``TesseraChroma`` is ``langchain_chroma.Chroma`` (collection
``tessera_lc_chunks``, cosine space) with one change: ``add_documents``
embeds the **embedding text** — title and heading path ahead of the body,
as native's ``chunk_embedding_text`` does — but stores the body only. Plain
``Chroma.add_documents`` embeds ``page_content``, and putting the prefix
into ``page_content`` would change what the answer prompt renders (the
plan's prompt-identity concern). This is the P5-1 spike's mechanism (a),
precomputed embeddings upserted through the store's collection, placed in
``add_documents`` so that LangChain's ``index()`` uses it too.
``embed_prefix=False`` embeds the body alone (the ``embed_prefix`` switch).

``LangChainChromaStore`` puts the native ``VectorStore`` port over it, so
the native retriever can read an index LangChain wrote. Rows have the
same metadata as native's (``chunk_metadata``) plus ``chunk_id``, because
``index()`` gives rows content-hash ids: the chunk id the retriever's
``_chunk_index`` reads comes from metadata, never from the row id.

Scores: ``similarity_search_by_vector_with_relevance_scores`` returns
cosine **distance** here, despite its name (P5-1 spike item 3), so the
score is ``1 - value`` — the same conversion native makes.
"""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument

from tessera.ingestion.chunker import Chunk
from tessera.store.base import SearchResult, VectorStore
from tessera.store.chroma import chunk_metadata, search_result_from_row

LC_COLLECTION_NAME = "tessera_lc_chunks"


def lc_metadata(chunk: Chunk) -> dict[str, str]:
    """A chunk's row metadata in the LangChain collection: native's, plus
    the chunk id and ``source`` (``index()``'s cleanup key)."""
    return {**chunk_metadata(chunk), "chunk_id": chunk.chunk_id, "source": str(chunk.document_path)}


def to_lc_document(chunk: Chunk) -> LCDocument:
    return LCDocument(page_content=chunk.text, metadata=lc_metadata(chunk))


def embedding_text(text: str, metadata: dict[str, Any]) -> str:
    """``chunk_embedding_text`` from a row's text and metadata (the parity
    test checks they agree on every corpus chunk)."""
    heading = metadata.get("heading_path") or ""
    title = metadata["document_title"]
    prefix = f"{title}\n{heading}" if heading else title
    return f"{prefix}\n\n{text}"


class TesseraChroma(Chroma):
    """``Chroma`` whose ``add_documents`` embeds the prefixed text."""

    def __init__(self, *args: Any, embed_prefix: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.embed_prefix = embed_prefix

    def add_documents(self, documents: list[LCDocument], **kwargs: Any) -> list[str]:
        if not documents:
            return []
        ids = kwargs.get("ids") or [d.metadata["chunk_id"] for d in documents]
        texts = [
            embedding_text(d.page_content, d.metadata) if self.embed_prefix else d.page_content
            for d in documents
        ]
        assert self.embeddings is not None, "TesseraChroma needs an embedding function"
        self.collection.upsert(
            ids=list(ids),
            embeddings=self.embeddings.embed_documents(texts),  # type: ignore[arg-type]
            documents=[d.page_content for d in documents],
            metadatas=[d.metadata for d in documents],
        )
        return list(ids)

    @property
    def collection(self) -> Any:
        # langchain_chroma keeps the chromadb collection on ``_collection``;
        # this is the one place Tessera reaches for it.
        return self._collection


class LangChainChromaStore(VectorStore):
    """The native ``VectorStore`` port over a ``TesseraChroma``."""

    def __init__(self, chroma: TesseraChroma) -> None:
        self.chroma = chroma

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return
        self.chroma.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=[c.text for c in chunks],
            metadatas=[lc_metadata(c) for c in chunks],
        )

    def query(
        self, embedding: list[float], k: int, where: dict[str, object] | None = None
    ) -> list[SearchResult]:
        count = self.count()
        if count == 0:
            return []
        pairs = self.chroma.similarity_search_by_vector_with_relevance_scores(
            embedding, k=min(k, count), filter=where
        )
        return [
            search_result_from_row(doc.metadata["chunk_id"], doc.page_content, doc.metadata, distance)
            for doc, distance in pairs
        ]

    def count(self) -> int:
        return self.chroma.collection.count()

    def rows(self, where: dict[str, object] | None = None) -> list[SearchResult]:
        """Every chunk matching ``where`` (score 0.0) — what BM25 indexes."""
        got = self.chroma.get(where=where, include=["documents", "metadatas"])
        return [
            search_result_from_row(m["chunk_id"], text, m, 1.0)
            for text, m in zip(got["documents"], got["metadatas"])
        ]

    def vectors(
        self, chunk_ids: list[str], where: dict[str, object] | None = None
    ) -> dict[str, list[float]]:
        """The stored embeddings of ``chunk_ids`` that also match ``where``,
        by chunk id — for re-scoring a fused or reranked list by cosine."""
        if not chunk_ids:
            return {}
        only: dict[str, object] = {"chunk_id": {"$in": list(chunk_ids)}}
        got = self.chroma.get(
            where=only if not where else {"$and": [where, only]},
            include=["embeddings", "metadatas"],
        )
        return {m["chunk_id"]: list(v) for m, v in zip(got["metadatas"], got["embeddings"])}

    def delete_document(self, document_path: str) -> int:
        ids = self.chroma.get(where={"document_path": document_path}, include=[])["ids"]
        if ids:
            self.chroma.delete(ids=ids)
        return len(ids)
