"""P3-4: generation/expertise.py — deterministic parts only (fake LLM).

Covers the prompt contents the model is shown, the two relevance floors,
and the zero-LLM-call no-match guarantee. Answer quality itself is the eval
harness's job (P3-5), not a unit test's.
"""

from datetime import date

from tessera.generation.base import LLMClient
from tessera.generation.expertise import (
    EXPERTISE_PERSON_FLOOR,
    EXPERTISE_QUERY_FLOOR,
    NO_EXPERT_MESSAGE,
    filter_qualified,
    generate_expertise_answer,
)
from tessera.generation.prompts import (
    EXPERTISE_ANSWER_SYSTEM_PROMPT,
    build_expertise_user_prompt,
)
from tessera.ingestion.expertise_loader import Person, ProjectEntry, Skill
from tessera.retrieval.expertise import ExpertiseResult
from tessera.retrieval.router import Archetype
from tessera.store.base import Evidence, ExpertiseStore, PersonMatch


def make_project(topic: str, year: int = 2025, industry: str = "energy") -> ProjectEntry:
    return ProjectEntry(industry=industry, topic=topic, role="workstream lead", year=year)


def make_person(pid: str, skills=(), projects=(), authored=()) -> Person:
    return Person(
        person_id=pid,
        name=f"Person {pid}",
        title="Consultant",
        office="London",
        practice="pricing",
        skills=tuple(skills),
        project_history=tuple(projects),
        authored=tuple(authored),
        languages=("English",),
        last_updated=date(2026, 6, 30),
    )


class ScoredExpertiseStore(ExpertiseStore):
    """Returns its people in order with fixed semantic scores."""

    def __init__(self, scored: list[tuple[Person, float]]) -> None:
        self._scored = scored

    def add(self, people, embeddings) -> None:  # pragma: no cover
        raise NotImplementedError

    def search(self, embedding, k, where=None) -> list[PersonMatch]:
        return [PersonMatch(person=p, score=s) for p, s in self._scored[:k]]

    def count(self) -> int:
        return len(self._scored)


class RecordingLLM(LLMClient):
    def __init__(self, response: str = "answer") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        return self.response


class ExplodingLLM(LLMClient):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise AssertionError("LLM called when nobody qualified")


def match(pid: str, evidence_score: float, *, self_reported: bool = False) -> PersonMatch:
    ev = (
        Evidence(
            "skill" if self_reported else "project",
            "pricing skill, level 5 (self-reported)"
            if self_reported
            else "pricing project in energy, 2025 (workstream lead)",
            evidence_score,
            self_reported=self_reported,
        ),
    )
    return PersonMatch(
        person=make_person(pid), score=0.5, evidence=ev, evidence_score=evidence_score
    )


def test_no_match_returns_fixed_message_with_zero_llm_calls() -> None:
    result = ExpertiseResult(query="q", matches=[match("a", 0.66), match("b", 0.0)])

    answer = generate_expertise_answer(result, ExplodingLLM())

    assert answer.answer == NO_EXPERT_MESSAGE
    assert answer.archetype is Archetype.EXPERTISE
    assert answer.experts == [] and answer.citations == []


def test_empty_result_is_no_match_too() -> None:
    answer = generate_expertise_answer(ExpertiseResult(query="q", matches=[]), ExplodingLLM())
    assert answer.answer == NO_EXPERT_MESSAGE


def test_query_floor_gates_on_the_best_match_not_the_first() -> None:
    # evidence_score can rank below semantic order; the gate uses the max.
    weak_first = [match("a", 0.2), match("b", EXPERTISE_QUERY_FLOOR + 0.5)]
    assert [m.person.person_id for m in filter_qualified(weak_first)] == ["a", "b"]
    assert filter_qualified([match("a", EXPERTISE_QUERY_FLOOR - 0.01)]) == []


def test_person_floor_drops_people_with_no_topical_backing_but_keeps_weak_ones() -> None:
    matches = [
        match("strong", 2.0),
        match("selfonly", EXPERTISE_PERSON_FLOOR + 0.02, self_reported=True),
        match("nothing", 0.0),
    ]
    assert [m.person.person_id for m in filter_qualified(matches)] == ["strong", "selfonly"]


def test_qualified_people_reach_the_model_and_come_back_as_experts() -> None:
    llm = RecordingLLM("Ask Person a [1].")
    result = ExpertiseResult(query="who knows pricing", matches=[match("a", 2.0), match("z", 0.0)])

    answer = generate_expertise_answer(result, llm)

    assert answer.answer == "Ask Person a [1]."
    assert [m.person.person_id for m in answer.experts] == ["a"]
    system, user = llm.calls[0]
    assert system == EXPERTISE_ANSWER_SYSTEM_PROMPT
    assert "Person a" in user and "Person z" not in user
    assert "who knows pricing" in user


def test_prompt_lists_evidence_and_flags_self_reported_only_people() -> None:
    prompt = build_expertise_user_prompt(
        "q", [match("solid", 2.0), match("claimer", 0.1, self_reported=True)]
    )

    solid, claimer = prompt.split("[2]")
    assert "pricing project in energy, 2025" in solid
    assert "Basis: EVIDENCED" in solid
    assert "Basis: SELF-REPORTED ONLY" in claimer
    assert "2026-06-30" in prompt  # snapshot date is surfaced


def test_system_prompt_requires_grounding_flagging_and_snapshot_date() -> None:
    p = EXPERTISE_ANSWER_SYSTEM_PROMPT
    assert "ONLY" in p
    assert "SELF-REPORTED ONLY" in p and "EVIDENCED first" in p
    assert "snapshot" in p.lower()
    assert "obvious expert" in p
