"""End-to-end integration: real corpus -> chunks -> embeddings -> Chroma
index -> similarity query. Unit tests in test_embedding.py/test_store.py
cover the interface pieces in isolation; this proves they work together —
which is what Task 3's acceptance check ("corpus indexed; a manual
similarity query returns plausible chunks") is actually about.
"""

from pathlib import Path

import pytest

from tessera.embedding.local import LocalEmbedder
from tessera.ingestion.chunker import chunk_corpus
from tessera.ingestion.loader import load_corpus
from tessera.store.chroma import ChromaVectorStore

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder()


@pytest.fixture(scope="module")
def indexed_store(tmp_path_factory: pytest.TempPathFactory, embedder: LocalEmbedder):
    docs = load_corpus(CORPUS_DIR)
    chunks = chunk_corpus(docs)
    embeddings = embedder.embed_documents([c.text for c in chunks])
    persist_dir = tmp_path_factory.mktemp("vectorstore")
    store = ChromaVectorStore(persist_dir=persist_dir)
    store.add(chunks, embeddings)
    return store


def test_full_corpus_gets_indexed(indexed_store: ChromaVectorStore) -> None:
    docs = load_corpus(CORPUS_DIR)
    chunks = chunk_corpus(docs)
    assert indexed_store.count() == len(chunks)


@pytest.mark.parametrize(
    "query,expected_title_substring",
    [
        ("Do we have a framework for market entry analysis?", "Market Entry"),
        (
            "How should we think about pricing when demand is inelastic?",
            "Pricing",
        ),
        (
            "What should I read to get up to speed on cost transformation?",
            "Cost Transformation",
        ),
    ],
)
def test_on_corpus_query_returns_plausible_top_result(
    indexed_store: ChromaVectorStore,
    embedder: LocalEmbedder,
    query: str,
    expected_title_substring: str,
) -> None:
    results = indexed_store.query(embedder.embed_query(query), k=3)

    assert results
    assert expected_title_substring in results[0].document_title


def test_off_corpus_query_scores_much_lower_than_on_corpus_query(
    indexed_store: ChromaVectorStore, embedder: LocalEmbedder
) -> None:
    on_corpus = indexed_store.query(
        embedder.embed_query("Do we have a framework for market entry analysis?"),
        k=1,
    )[0]
    off_corpus = indexed_store.query(
        embedder.embed_query("What is our parental leave policy?"), k=1
    )[0]

    # Not a hard cutoff here (that's Task 6's job) — just confirming the
    # score gap is large enough to be a usable signal for the "we don't
    # have anything on that" behavior later.
    assert off_corpus.score < on_corpus.score - 0.2
