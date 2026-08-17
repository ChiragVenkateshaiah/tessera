"""Embedder interface — the swappable port between local sentence-transformers
(Phase 1) and Bedrock Titan/Cohere (Phase 4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Turns text into vectors.

    Documents and queries are embedded through separate methods rather than
    one, even though Phase 1's model treats them identically — some models
    (e5-style instruction embeddings, several Bedrock/Cohere options) need a
    different prefix or encoding path for a query versus a passage being
    indexed. Keeping the split in the interface means a future
    implementation can exploit that without every caller changing.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the embedding vectors this embedder produces."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for retrieval."""
