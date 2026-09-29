"""VectorStore interface — the swappable port between local Chroma (Phase 1)
and OpenSearch Serverless (Phase 4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tessera.ingestion.chunker import Chunk
from tessera.ingestion.expertise_loader import Person


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk plus its similarity score.

    Carries everything a caller needs for citation and for archetype-aware
    filtering (doc_type, industry, topics) without a second lookup back to
    the source document.

    score: higher is better (similarity, not distance) — Phase 1's Chroma
    implementation returns cosine similarity in [-1, 1]. Task 6's
    RELEVANCE_THRESHOLD filtering in generation/answer.py depends on this
    direction; a future VectorStore implementation must preserve it rather
    than returning a raw distance.
    """

    chunk_id: str
    text: str
    score: float
    document_path: str
    document_title: str
    doc_type: str
    industry: str
    topics: list[str]
    date: str
    heading_path: tuple[str, ...]


class VectorStore(ABC):
    """Persists chunk embeddings and answers similarity queries."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Index a batch of chunks with their pre-computed embeddings.

        chunks and embeddings must be the same length and index-aligned.
        """

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        """Return the k nearest chunks to embedding, best match first.

        where filters on chunk metadata (e.g. {"doc_type": "methodology"})
        — this is the hook archetype-aware retrieval (Task 5) uses to keep
        lookup queries narrow.
        """

    @abstractmethod
    def count(self) -> int:
        """Number of chunks currently indexed."""


@dataclass(frozen=True)
class PersonMatch:
    """One person returned by an ExpertiseStore search.

    score: higher is better (cosine similarity), same direction contract
    as SearchResult.score. ``evidence`` is empty as returned by a store —
    the retrieval layer (P3-3) fills in which skills/projects/authored
    documents actually matched the query; the store only knows the
    embedded profile, not why it matched.
    """

    person: Person
    score: float
    evidence: tuple[str, ...] = ()


class ExpertiseStore(ABC):
    """Persists embedded people profiles and answers similarity queries.

    Sibling of VectorStore for archetype B (Phase 3). Phase 4 swaps the
    local Chroma implementation for a real people-index behind this port.
    """

    @abstractmethod
    def add(self, people: list[Person], embeddings: list[list[float]]) -> None:
        """Index a batch of people with their pre-computed profile
        embeddings. people and embeddings must be the same length and
        index-aligned; re-adding a person_id replaces that person.
        """

    @abstractmethod
    def search(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, object] | None = None,
    ) -> list[PersonMatch]:
        """Return the k nearest people to embedding, best match first.

        where filters on structured fields: ``practice``, ``office`` and
        ``title`` (exact match), plus per-topic evidenced flags via the
        implementation's own convention — see ChromaExpertiseStore.
        """

    @abstractmethod
    def count(self) -> int:
        """Number of people currently indexed."""
