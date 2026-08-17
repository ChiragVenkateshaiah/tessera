from datetime import date
from pathlib import Path

import pytest

from tessera.ingestion.chunker import Chunk
from tessera.store.base import VectorStore
from tessera.store.chroma import ChromaVectorStore


def _chunk(chunk_id: str, text: str, **overrides: object) -> Chunk:
    defaults: dict[str, object] = dict(
        document_path=Path("doc.md"),
        document_title="Doc",
        doc_type="methodology",
        industry="cross-industry",
        topics=["t1"],
        date=date(2024, 1, 1),
        heading_path=("Overview",),
        chunk_index=0,
    )
    defaults.update(overrides)
    return Chunk(chunk_id=chunk_id, text=text, **defaults)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(persist_dir=tmp_path)


def test_chroma_store_satisfies_vectorstore_interface(
    store: ChromaVectorStore,
) -> None:
    assert isinstance(store, VectorStore)


def test_add_and_count(store: ChromaVectorStore) -> None:
    chunks = [_chunk("a", "text a"), _chunk("b", "text b")]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    assert store.count() == 2


def test_query_returns_best_match_first(store: ChromaVectorStore) -> None:
    chunks = [_chunk("a", "text a"), _chunk("b", "text b"), _chunk("c", "text c")]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])

    results = store.query([1.0, 0.0], k=3)

    assert [r.chunk_id for r in results] == ["a", "c", "b"]
    assert results[0].score == pytest.approx(1.0)


def test_query_respects_k(store: ChromaVectorStore) -> None:
    chunks = [_chunk(str(i), f"text {i}") for i in range(5)]
    store.add(chunks, [[float(i), 0.0] for i in range(5)])

    results = store.query([0.0, 0.0], k=2)

    assert len(results) == 2


def test_query_with_metadata_filter(store: ChromaVectorStore) -> None:
    chunks = [
        _chunk("a", "text a", doc_type="methodology"),
        _chunk("b", "text b", doc_type="thought_leadership"),
    ]
    store.add(chunks, [[1.0, 0.0], [1.0, 0.0]])

    results = store.query([1.0, 0.0], k=5, where={"doc_type": "methodology"})

    assert [r.chunk_id for r in results] == ["a"]


def test_query_on_empty_store_returns_empty(store: ChromaVectorStore) -> None:
    assert store.query([1.0, 0.0], k=5) == []


def test_metadata_round_trips_through_serialization(
    store: ChromaVectorStore,
) -> None:
    chunk = _chunk(
        "a",
        "text a",
        topics=["alpha", "beta", "gamma"],
        heading_path=("Framework", "Model Formulation", "Objective Function"),
        date=date(2023, 6, 15),
    )
    store.add([chunk], [[1.0, 0.0]])

    result = store.query([1.0, 0.0], k=1)[0]

    assert result.topics == ["alpha", "beta", "gamma"]
    assert result.heading_path == (
        "Framework",
        "Model Formulation",
        "Objective Function",
    )
    assert result.date == "2023-06-15"


def test_add_rejects_mismatched_lengths(store: ChromaVectorStore) -> None:
    with pytest.raises(ValueError):
        store.add([_chunk("a", "text")], [[1.0], [2.0]])


def test_persists_across_store_instances(tmp_path: Path) -> None:
    first = ChromaVectorStore(persist_dir=tmp_path)
    first.add([_chunk("a", "text a")], [[1.0, 0.0]])

    second = ChromaVectorStore(persist_dir=tmp_path)

    assert second.count() == 1
