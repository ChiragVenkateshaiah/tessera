"""Embeddings across the two stacks (plan §3.3, the ``embeddings`` switch).

- ``as_langchain(embedder)``: a native ``Embedder`` as LangChain
  ``Embeddings``, so a LangChain store can use the native model.
- ``LangChainEmbedder``: LangChain ``Embeddings`` (``HuggingFaceEmbeddings``,
  built by ``integrations.langchain.hf_embeddings``) as the native
  ``Embedder`` port, so the native store and retriever can use it.

The parity test checks that ``HuggingFaceEmbeddings`` over the same
``all-MiniLM-L6-v2`` gives native's vectors within tolerance.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from tessera.embedding.base import Embedder


class _NativeEmbeddings(Embeddings):
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)


def as_langchain(embedder: Embedder) -> Embeddings:
    return _NativeEmbeddings(embedder)


class LangChainEmbedder(Embedder):
    """The native ``Embedder`` port over LangChain ``Embeddings``."""

    def __init__(self, embeddings: Embeddings, dimension: int | None = None) -> None:
        self._embeddings = embeddings
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        # LangChain Embeddings don't expose it; one probe, then cached.
        if self._dimension is None:
            self._dimension = len(self._embeddings.embed_query("dimension"))
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)
