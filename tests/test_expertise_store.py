"""P3-2: ExpertiseStore port + local Chroma implementation.

The interface-swap acceptance is proven the same way test_interface_swap.py
does it for documents: one index/search function runs unchanged against a
fake store defined only in this file and against the real Chroma store.
The full-dataset test then indexes all ~600 real people with the real
embedder and checks a manual-style query returns plausible people.
"""

from pathlib import Path

import pytest

from tessera.embedding.base import Embedder
from tessera.embedding.local import LocalEmbedder
from tessera.ingestion.expertise_loader import (
    Person,
    ProjectEntry,
    Skill,
    load_expertise,
    profile_summary_text,
)
from tessera.store.base import ExpertiseStore, PersonMatch
from tessera.store.chroma_expertise import ChromaExpertiseStore, topic_filter_key
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
PEOPLE_DIR = ROOT / "data" / "expertise" / "people"
CORPUS_DIR = ROOT / "data" / "corpus"


def _person(pid: str, topic: str, practice: str = "pricing", office: str = "London") -> Person:
    return Person(
        person_id=pid,
        name=f"Person {pid}",
        title="Principal",
        office=office,
        practice=practice,
        skills=(Skill(topic=topic, level=4, basis="evidenced"),),
        project_history=(
            ProjectEntry(industry="retail", topic=topic, role="engagement lead", year=2023),
        ),
        authored=(),
        languages=("English",),
        last_updated=date(2026, 6, 30),
    )


class FakeEmbedder(Embedder):
    """Bag-of-topics fake: one dimension per known topic word."""

    VOCAB = ["pricing", "procurement", "integration"]

    @property
    def dimension(self) -> int:
        return len(self.VOCAB)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        raw = [float(text.lower().count(w)) + 0.01 for w in self.VOCAB]
        norm = sum(x * x for x in raw) ** 0.5
        return [x / norm for x in raw]


class FakeExpertiseStore(ExpertiseStore):
    def __init__(self) -> None:
        self._rows: dict[str, tuple[Person, list[float]]] = {}

    def add(self, people: list[Person], embeddings: list[list[float]]) -> None:
        for p, e in zip(people, embeddings):
            self._rows[p.person_id] = (p, e)

    def search(self, embedding, k, where=None) -> list[PersonMatch]:
        scored = [
            PersonMatch(person=p, score=sum(a * b for a, b in zip(embedding, e)))
            for p, e in self._rows.values()
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._rows)


def index_and_search(
    store: ExpertiseStore, embedder: Embedder, people: list[Person], query: str, k: int
) -> list[PersonMatch]:
    """The calling code that must not change between implementations."""
    store.add(people, embedder.embed_documents([profile_summary_text(p) for p in people]))
    return store.search(embedder.embed_query(query), k)


PEOPLE = [_person("a", "pricing"), _person("b", "procurement"), _person("c", "integration")]


def test_same_calling_code_runs_against_fake_and_chroma(tmp_path: Path) -> None:
    for store in (FakeExpertiseStore(), ChromaExpertiseStore(tmp_path / "vs")):
        matches = index_and_search(store, FakeEmbedder(), PEOPLE, "pricing expert", k=2)
        assert store.count() == 3
        assert matches[0].person.person_id == "a"
        assert matches[0].score >= matches[1].score
        assert matches[0].evidence == ()


def test_chroma_round_trips_the_full_person_record(tmp_path: Path) -> None:
    store = ChromaExpertiseStore(tmp_path / "vs")
    match = index_and_search(store, FakeEmbedder(), PEOPLE, "procurement", k=1)[0]
    assert match.person == PEOPLE[1]


def test_chroma_where_filters_on_structured_fields_and_topic(tmp_path: Path) -> None:
    people = [_person("a", "pricing", office="London"), _person("b", "pricing", office="Paris")]
    store = ChromaExpertiseStore(tmp_path / "vs")
    emb = FakeEmbedder()
    store.add(people, emb.embed_documents([profile_summary_text(p) for p in people]))
    q = emb.embed_query("pricing")

    assert [m.person.person_id for m in store.search(q, 5, where={"office": "Paris"})] == ["b"]
    assert len(store.search(q, 5, where={topic_filter_key("pricing"): True})) == 2
    assert store.search(q, 5, where={topic_filter_key("procurement"): True}) == []


def test_add_is_idempotent_and_length_checked(tmp_path: Path) -> None:
    store = ChromaExpertiseStore(tmp_path / "vs")
    emb = FakeEmbedder().embed_documents(["pricing"])
    store.add([PEOPLE[0]], emb)
    store.add([PEOPLE[0]], emb)
    assert store.count() == 1
    with pytest.raises(ValueError):
        store.add(PEOPLE, emb)


def test_empty_store_returns_no_matches(tmp_path: Path) -> None:
    assert ChromaExpertiseStore(tmp_path / "vs").search([1.0, 0.0, 0.0], 3) == []


def test_profile_summary_has_signal_but_no_name() -> None:
    text = profile_summary_text(PEOPLE[0])
    assert "pricing" in text and "retail" in text and "London" in text
    assert "Person a" not in text


def test_full_dataset_indexes_and_query_returns_plausible_people(tmp_path: Path) -> None:
    people = load_expertise(PEOPLE_DIR, corpus_dir=CORPUS_DIR)
    embedder = LocalEmbedder()
    store = ChromaExpertiseStore(tmp_path / "vs")
    store.add(people, embedder.embed_documents([profile_summary_text(p) for p in people]))
    assert store.count() == len(people) >= 600

    topic = next(s.topic for s in people[0].skills)
    matches = store.search(embedder.embed_query(f"expert in {topic.replace('-', ' ')}"), 10)
    assert len(matches) == 10
    # Plausible: most of the top 10 actually list the topic as a skill.
    hits = sum(any(s.topic == topic for s in m.person.skills) for m in matches)
    assert hits >= 7, f"only {hits}/10 top matches have skill {topic!r}"
