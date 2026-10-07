"""Chroma implementation of the VectorStore interface."""

from __future__ import annotations

from pathlib import Path

import chromadb

from tessera.ingestion.chunker import Chunk
from tessera.ingestion.loader import SENSITIVITY_RESTRICTED, STATUS_CURRENT
from tessera.store.base import SearchResult, VectorStore

DEFAULT_COLLECTION_NAME = "tessera_chunks"


def chunk_metadata(chunk: Chunk) -> dict[str, str]:
    """Chroma metadata values must be flat scalars (str/int/float/bool) —
    Chunk's list/tuple/date fields get serialized to strings here and
    reconstructed in search_result_from_row.
    """
    return {
        "document_path": str(chunk.document_path),
        "document_title": chunk.document_title,
        "doc_type": chunk.doc_type,
        "industry": chunk.industry,
        "topics": ",".join(chunk.topics),
        "date": chunk.date.isoformat(),
        "heading_path": " > ".join(chunk.heading_path),
        "status": chunk.status,
        "superseded_by": chunk.superseded_by or "",
        "sensitivity": chunk.sensitivity,
        "engagement": chunk.engagement or "",
    }


def search_result_from_row(
    chunk_id: str, text: str, meta: dict[str, str], distance: float
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text,
        # Collection is created with cosine space, so Chroma's "distance"
        # is 1 - cosine_similarity; invert it back to a similarity score
        # so callers see a bigger number for a better match.
        score=1.0 - distance,
        document_path=meta["document_path"],
        document_title=meta["document_title"],
        doc_type=meta["doc_type"],
        industry=meta["industry"],
        topics=meta["topics"].split(",") if meta["topics"] else [],
        date=meta["date"],
        heading_path=tuple(meta["heading_path"].split(" > "))
        if meta["heading_path"]
        else (),
        # An index built before Phase 4 has no status: every chunk current.
        status=meta.get("status", STATUS_CURRENT),
        superseded_by=meta.get("superseded_by") or None,
        # Fail closed: a chunk with no sensitivity label (an index built
        # before Phase 4's access labels) reads as restricted — re-ingest.
        sensitivity=meta.get("sensitivity", SENSITIVITY_RESTRICTED),
        engagement=meta.get("engagement") or None,
    )


class ChromaVectorStore(VectorStore):
    """Local, persistent Chroma collection at persist_dir."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        # sentence-transformers embeddings are trained for cosine
        # similarity; Chroma's default space is L2, so this is set
        # explicitly rather than relying on the default.
        self._collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[chunk_metadata(c) for c in chunks],
        )

    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
            where=where,
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            search_result_from_row(chunk_id, text, meta, distance)
            for chunk_id, text, meta, distance in zip(
                ids, documents, metadatas, distances
            )
        ]

    def delete_document(self, document_path: str) -> int:
        ids = self._collection.get(where={"document_path": document_path}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self._collection.count()
