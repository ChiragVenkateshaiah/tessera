"""VectorStore interface — the swappable port between local Chroma (Phase 1)
and OpenSearch Serverless (Phase 4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tessera.ingestion.chunker import Chunk
from tessera.ingestion.expertise_loader import Person
from tessera.ingestion.loader import SENSITIVITY_INTERNAL, STATUS_CURRENT


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
    # Freshness (Phase 4): "current" or "superseded"; superseded_by is the
    # replacing document's path, in the same form as document_path.
    status: str = STATUS_CURRENT
    superseded_by: str | None = None
    # Access labels (Phase 4): "internal" or "restricted", and the
    # engagement codename a restricted chunk belongs to.
    sensitivity: str = SENSITIVITY_INTERNAL
    engagement: str | None = None


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
class Evidence:
    """One concrete reason a person matched a query.

    kind: "project" | "authored" | "skill". ``strength`` is the
    contribution to the person's evidence score. ``self_reported`` is True
    only for a skill claim with nothing behind it — generation (P3-4) must
    flag those rather than present them as equivalent to project or
    authorship evidence.
    """

    kind: str
    description: str
    strength: float
    self_reported: bool = False


@dataclass(frozen=True)
class PersonMatch:
    """One person returned by an ExpertiseStore search.

    score: higher is better (cosine similarity between query and profile),
    same direction contract as SearchResult.score. ``evidence``,
    ``evidence_score`` and ``rank_score`` are empty/zero as returned by a
    store — the retrieval layer (retrieval/expertise.py) fills them in; the
    store only knows the embedded profile, not why it matched.

    evidence_score is how much topical evidence backs the person,
    independent of how the query is phrased — the scale the generation
    floors were calibrated on. rank_score is what orders the shortlist: the
    same evidence re-weighted for query intent ("led", "recently"), plus
    semantic similarity. They are equal for a query with no such intent.
    """

    person: Person
    score: float
    evidence: tuple[Evidence, ...] = ()
    evidence_score: float = 0.0
    rank_score: float = 0.0

    @property
    def is_evidenced(self) -> bool:
        """True if any evidence is a project, authored doc, or evidenced
        skill — False when the match rests on self-reported skills alone.
        """
        return any(not e.self_reported for e in self.evidence)


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
