import math

import pytest

from tessera.embedding.base import Embedder
from tessera.embedding.local import LocalEmbedder


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def test_local_embedder_satisfies_embedder_interface(embedder: LocalEmbedder) -> None:
    assert isinstance(embedder, Embedder)


def test_query_vector_length_matches_declared_dimension(
    embedder: LocalEmbedder,
) -> None:
    vec = embedder.embed_query("test")
    assert len(vec) == embedder.dimension


def test_embed_documents_returns_one_vector_per_text(
    embedder: LocalEmbedder,
) -> None:
    texts = ["first chunk of text", "second chunk of text", "third"]
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == len(texts)
    assert all(len(v) == embedder.dimension for v in vectors)


def test_embedder_is_deterministic(embedder: LocalEmbedder) -> None:
    a = embedder.embed_query("market entry analysis")
    b = embedder.embed_query("market entry analysis")
    assert a == b


def test_similar_text_embeds_closer_than_dissimilar_text(
    embedder: LocalEmbedder,
) -> None:
    market_a = embedder.embed_query("market entry analysis for a new geography")
    market_b = embedder.embed_query("evaluating entry into a new country market")
    unrelated = embedder.embed_query("parental leave policy for employees")

    assert _cosine(market_a, market_b) > _cosine(market_a, unrelated)
