"""Proves the Task 3 acceptance criterion directly: swapping an Embedder
or VectorStore implementation requires no changes outside embedding/ and
store/. Demonstrated by running the exact same indexing/query function
against fake implementations defined only in this test file, and again
against the real LocalEmbedder/ChromaVectorStore — with zero changes to
the calling code between the two runs.
"""

import math
from datetime import date
from pathlib import Path

from tessera.embedding.base import Embedder
from tessera.embedding.local import LocalEmbedder
from tessera.ingestion.chunker import Chunk
from tessera.store.base import SearchResult, VectorStore
from tessera.store.chroma import ChromaVectorStore


class FakeEmbedder(Embedder):
    """Deterministic, hash-based fake — no model, no network."""

    @property
    def dimension(self) -> int:
        return 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        h = hash(text)
        raw = [((h >> (8 * i)) & 0xFF) / 255.0 + 0.01 for i in range(4)]
        # Normalized so FakeVectorStore's dot-product scoring behaves like
        # cosine similarity (matching real ChromaVectorStore's cosine
        # space) — otherwise a differently-hashed but larger-magnitude
        # vector could out-score the true match, which is exactly what
        # happened before this fix (Python's hash() is randomized per
        # process, so the bug wasn't visible on every run).
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


class FakeVectorStore(VectorStore):
    """In-memory fake — no Chroma, no disk."""

    def __init__(self) -> None:
        self._rows: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._rows.extend(zip(chunks, embeddings))

    def query(
        self, embedding: list[float], k: int, where: dict[str, object] | None = None
    ) -> list[SearchResult]:
        def score(vec: list[float]) -> float:
            return sum(a * b for a, b in zip(embedding, vec))

        ranked = sorted(self._rows, key=lambda row: score(row[1]), reverse=True)
        return [
            SearchResult(
                chunk_id=c.chunk_id,
                text=c.text,
                score=score(v),
                document_path=str(c.document_path),
                document_title=c.document_title,
                doc_type=c.doc_type,
                industry=c.industry,
                topics=c.topics,
                date=c.date.isoformat(),
                heading_path=c.heading_path,
            )
            for c, v in ranked[:k]
        ]

    def count(self) -> int:
        return len(self._rows)


def _index_and_query(
    embedder: Embedder, store: VectorStore, texts: list[str], query: str
) -> list[SearchResult]:
    """The calling code under test — identical regardless of which
    Embedder/VectorStore implementation is injected.
    """
    chunks = [
        Chunk(
            chunk_id=str(i),
            document_path=Path("doc.md"),
            document_title="Doc",
            doc_type="methodology",
            industry="cross-industry",
            topics=["t"],
            date=date(2024, 1, 1),
            heading_path=("Overview",),
            text=text,
            chunk_index=i,
        )
        for i, text in enumerate(texts)
    ]
    embeddings = embedder.embed_documents(texts)
    store.add(chunks, embeddings)
    return store.query(embedder.embed_query(query), k=2)


def test_fake_implementations_satisfy_the_interfaces() -> None:
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeVectorStore(), VectorStore)


def test_same_calling_code_works_with_fake_implementations() -> None:
    results = _index_and_query(
        FakeEmbedder(),
        FakeVectorStore(),
        ["alpha text", "beta text", "gamma text"],
        "alpha text",
    )

    assert len(results) == 2
    assert results[0].chunk_id == "0"  # exact text match scores highest


def test_same_calling_code_works_with_real_implementations(
    tmp_path: Path,
) -> None:
    results = _index_and_query(
        LocalEmbedder(),
        ChromaVectorStore(persist_dir=tmp_path),
        [
            "market entry analysis framework for new geographies",
            "parental leave policy for full-time employees",
        ],
        "how do we evaluate entering a new market",
    )

    assert len(results) == 2
    assert "market entry" in results[0].text.lower()
