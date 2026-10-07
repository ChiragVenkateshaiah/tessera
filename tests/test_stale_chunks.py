"""The stale-chunk test (Phase 5 plan §3.3, P5-4 acceptance), on both
indexing paths.

A re-index must leave nothing behind under old labels when a document:
- **shrinks** (5 chunks → 3: no ``::3``/``::4`` left over);
- is **relabelled** (engagement halcyon → kestrel: no chunk still says
  halcyon, which the permission filter would match on);
- is **quarantined** for review (``review_status: pending``: all its
  chunks gone).

Upsert alone fails all three, which is the bug this task fixes. Each
indexer is run over a real Chroma store in a temp dir with a tiny
deterministic embedder, then the store is read back through its own
``query`` with a ``document_path`` filter.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from tessera.embedding.base import Embedder
from tessera.ingestion.chunker import chunk_corpus, chunk_embedding_text
from tessera.ingestion.indexing import index_corpus
from tessera.ingestion.loader import Document, indexable, load_corpus
from tessera.store.base import SearchResult, VectorStore
from tessera.store.chroma import ChromaVectorStore


class HashEmbedder(Embedder):
    """Deterministic 8-dim vectors from the text's hash."""

    dimension = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 + 0.01 for b in digest[:8]]


def _sections(n: int, word: str) -> str:
    return "\n\n".join(f"## Part {i}\n\n{word} section {i} body text." for i in range(n))


FRONT = {
    "methodology/pricing.md": (
        'title: "Pricing"\ndoc_type: methodology\nindustry: retail\n'
        "topics: [pricing-strategy]\ndate: 2025-01-01\n"
    ),
    "engagements/halcyon-grocer.md": (
        'title: "Project Halcyon"\ndoc_type: engagement\nindustry: retail\n'
        "topics: [pricing-strategy]\ndate: 2025-01-01\nsensitivity: restricted\n"
        "engagement: {engagement}\n"
    ),
    "case_studies/grocer-reset.md": (
        'title: "Grocer reset"\ndoc_type: case_study\nindustry: retail\n'
        "topics: [pricing-strategy]\ndate: 2025-01-01\nsensitivity: internal\n"
        "review_status: {review}\n"
    ),
}


def write_corpus(root: Path, *, pricing_parts: int, engagement: str, review: str) -> Path:
    bodies = {
        "methodology/pricing.md": _sections(pricing_parts, "pricing"),
        "engagements/halcyon-grocer.md": _sections(2, "engagement"),
        "case_studies/grocer-reset.md": _sections(2, "case"),
    }
    for rel, front in FRONT.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = front.format(engagement=engagement, review=review)
        path.write_text(f"---\n{text}---\n\n{bodies[rel]}\n", encoding="utf-8")
    return root


# An indexer takes (loaded documents, embedder, persist dir) and returns a
# VectorStore to read the result back through. Every implementation runs
# the same scenarios.
Indexer = Callable[[list[Document], Embedder, Path], VectorStore]


def native_indexer(documents: list[Document], embedder: Embedder, persist: Path) -> VectorStore:
    store = ChromaVectorStore(persist_dir=persist)
    index_corpus(documents, embedder, store)
    return store


def upsert_only(documents: list[Document], embedder: Embedder, persist: Path) -> VectorStore:
    """The pre-P5-4 behaviour, kept as the control: it must fail."""
    store = ChromaVectorStore(persist_dir=persist)
    chunks = chunk_corpus(indexable(documents))
    store.add(chunks, embedder.embed_documents([chunk_embedding_text(c) for c in chunks]))
    return store


def _lc_indexer(cleanup: str) -> Indexer:
    def run(documents: list[Document], embedder: Embedder, persist: Path) -> VectorStore:
        pytest.importorskip("langchain_core", reason="the lc extra isn't installed")
        from tessera.integrations.langchain import lc_chroma, sql_record_manager
        from tessera.lc.embeddings import as_langchain
        from tessera.lc.indexing import index_with_record_manager
        from tessera.lc.store import LangChainChromaStore

        chroma = lc_chroma(persist, as_langchain(embedder))
        records = sql_record_manager(persist / "records.sqlite")
        index_with_record_manager(documents, chroma, records, cleanup=cleanup)
        return LangChainChromaStore(chroma)

    return run


INDEXERS: dict[str, Indexer] = {
    "native delete-then-add": native_indexer,
    "lc index() cleanup=full": _lc_indexer("full"),
    "lc index() cleanup=incremental": _lc_indexer("incremental"),
}
# incremental cleans only the sources in this run's batch; a quarantined
# document is absent from the batch, so its chunks stay (tessera.lc.indexing).
QUARANTINE_GAP = {"lc index() cleanup=incremental"}


def chunks_of(store: VectorStore, embedder: Embedder, path: Path) -> list[SearchResult]:
    return store.query(embedder.embed_query("x"), k=100, where={"document_path": str(path)})


def run_twice(tmp_path: Path, indexer: Indexer, **after: object) -> tuple[VectorStore, Path]:
    """Index the 'before' corpus, change it, index again into the same store."""
    embedder = HashEmbedder()
    corpus = tmp_path / "corpus"
    persist = tmp_path / "index"
    before = {"pricing_parts": 5, "engagement": "halcyon", "review": "reviewed"}
    write_corpus(corpus, **before)  # type: ignore[arg-type]
    indexer(load_corpus(corpus), embedder, persist)
    write_corpus(corpus, **{**before, **after})  # type: ignore[arg-type]
    return indexer(load_corpus(corpus), embedder, persist), corpus


@pytest.mark.parametrize("name", list(INDEXERS))
def test_a_shrunk_document_leaves_no_tail_chunks(tmp_path: Path, name: str) -> None:
    store, corpus = run_twice(tmp_path, INDEXERS[name], pricing_parts=3)
    ids = sorted(r.chunk_id for r in chunks_of(store, HashEmbedder(), corpus / "methodology/pricing.md"))
    assert ids == ["pricing::0", "pricing::1", "pricing::2"]


@pytest.mark.parametrize("name", list(INDEXERS))
def test_a_relabelled_document_keeps_no_chunk_under_its_old_engagement(
    tmp_path: Path, name: str
) -> None:
    store, corpus = run_twice(tmp_path, INDEXERS[name], engagement="kestrel")
    found = chunks_of(store, HashEmbedder(), corpus / "engagements/halcyon-grocer.md")
    assert found and {r.engagement for r in found} == {"kestrel"}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n,
            marks=pytest.mark.xfail(strict=True, reason="incremental cleanup skips absent sources"),
        )
        if n in QUARANTINE_GAP
        else n
        for n in INDEXERS
    ],
)
def test_a_newly_quarantined_document_is_removed(tmp_path: Path, name: str) -> None:
    store, corpus = run_twice(tmp_path, INDEXERS[name], review="pending")
    assert chunks_of(store, HashEmbedder(), corpus / "case_studies/grocer-reset.md") == []
    # Everything else is still there.
    assert len(chunks_of(store, HashEmbedder(), corpus / "methodology/pricing.md")) == 5


def test_upsert_alone_leaves_stale_chunks_the_control(tmp_path: Path) -> None:
    store, corpus = run_twice(tmp_path, upsert_only, pricing_parts=3, review="pending")
    embedder = HashEmbedder()
    assert len(chunks_of(store, embedder, corpus / "methodology/pricing.md")) == 5
    assert chunks_of(store, embedder, corpus / "case_studies/grocer-reset.md") != []
