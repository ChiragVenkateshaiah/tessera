"""Chroma implementation of the ExpertiseStore interface."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import chromadb

from tessera.ingestion.expertise_loader import (
    Person,
    ProjectEntry,
    Skill,
    profile_summary_text,
)
from tessera.store.base import ExpertiseStore, PersonMatch

DEFAULT_EXPERTISE_COLLECTION_NAME = "tessera_people"

# where-filter convention: Chroma metadata can't hold lists, so each
# evidenced topic becomes a boolean flag keyed "topic:<name>". Callers
# filter with topic_filter_key("pricing") -> {"topic:pricing": True}.
_TOPIC_PREFIX = "topic:"


def topic_filter_key(topic: str) -> str:
    """Metadata key to filter on for 'has an evidenced project in topic'."""
    return f"{_TOPIC_PREFIX}{topic}"


def _person_to_json(person: Person) -> str:
    return json.dumps(
        {
            "person_id": person.person_id,
            "name": person.name,
            "title": person.title,
            "office": person.office,
            "practice": person.practice,
            "skills": [
                {"topic": s.topic, "level": s.level, "basis": s.basis}
                for s in person.skills
            ],
            "project_history": [
                {"industry": p.industry, "topic": p.topic, "role": p.role, "year": p.year}
                for p in person.project_history
            ],
            "authored": list(person.authored),
            "languages": list(person.languages),
            "last_updated": person.last_updated.isoformat(),
        }
    )


def _person_from_json(raw: str) -> Person:
    d = json.loads(raw)
    return Person(
        person_id=d["person_id"],
        name=d["name"],
        title=d["title"],
        office=d["office"],
        practice=d["practice"],
        skills=tuple(Skill(**s) for s in d["skills"]),
        project_history=tuple(ProjectEntry(**p) for p in d["project_history"]),
        authored=tuple(d["authored"]),
        languages=tuple(d["languages"]),
        last_updated=date.fromisoformat(d["last_updated"]),
    )


def _serialize_metadata(person: Person) -> dict[str, str | bool]:
    meta: dict[str, str | bool] = {
        "practice": person.practice,
        "office": person.office,
        "title": person.title,
        # Full record, so search() can rebuild the Person without a
        # second lookup back to the YAML dataset.
        "record": _person_to_json(person),
    }
    for topic in person.evidenced_topics:
        meta[topic_filter_key(topic)] = True
    return meta


class ChromaExpertiseStore(ExpertiseStore):
    """Local, persistent Chroma collection of people, separate from the
    document collection (its own collection name in the same client).
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = DEFAULT_EXPERTISE_COLLECTION_NAME,
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, people: list[Person], embeddings: list[list[float]]) -> None:
        if len(people) != len(embeddings):
            raise ValueError("people and embeddings must be the same length")
        if not people:
            return
        self._collection.upsert(
            ids=[p.person_id for p in people],
            embeddings=embeddings,
            documents=[profile_summary_text(p) for p in people],
            metadatas=[_serialize_metadata(p) for p in people],
        )

    def search(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, object] | None = None,
    ) -> list[PersonMatch]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
            where=where,
        )
        return [
            PersonMatch(person=_person_from_json(meta["record"]), score=1.0 - distance)
            for meta, distance in zip(result["metadatas"][0], result["distances"][0])
        ]

    def count(self) -> int:
        return self._collection.count()
