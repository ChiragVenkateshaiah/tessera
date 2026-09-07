"""Reads the synthesized firm expertise dataset (Phase 3, archetype B).

The dataset under ``data/expertise/people/`` is generated, not
hand-written — see ``data/expertise/README.md`` and
``data/expertise/generate.py``. This module is the read + validate side,
the archetype-B analogue of ``loader.py`` for documents. It is exempt
from the no-I/O part of CLAUDE.md constraint #6 the same way ``loader.py``
is (reading the dataset off disk is its job), but stays parameterized:
no hardcoded paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import yaml

# Seniority ladder, junior → senior. Ordered so callers can compare rank.
TITLES: tuple[str, ...] = (
    "Analyst",
    "Consultant",
    "Engagement Manager",
    "Principal",
    "Partner",
)

# How a claimed skill is backed. "evidenced" = corroborated by a
# project_history entry and/or an authored corpus document on that topic;
# "self_reported" = the person tagged it with nothing behind it. The
# retrieval path (Phase 3 P3-3) weights evidenced above self_reported and
# the B answer flags self_reported-only matches (plan §2.2).
SKILL_BASES: frozenset[str] = frozenset({"evidenced", "self_reported"})

# Roles a consultant holds on an engagement. A closed set on purpose:
# project_history carries industry + topic + role + year and nothing
# else, so there is no free-text field a client name could hide in
# (plan §2.3 — the "no client names" gate is structural).
PROJECT_ROLES: frozenset[str] = frozenset(
    {
        "analyst",
        "modeling support",
        "workstream member",
        "workstream lead",
        "engagement manager",
        "engagement lead",
        "engagement partner",
        "advisor",
    }
)

# Industries a project can sit in. Wider than the pilot corpus's five so
# expertise retrieval has to discriminate (a "retail pricing" query
# should not match every pricing project regardless of sector).
PROJECT_INDUSTRIES: frozenset[str] = frozenset(
    {
        "industrials",
        "retail",
        "financial-services",
        "pharma",
        "healthcare-providers",
        "technology",
        "energy",
        "consumer-goods",
        "public-sector",
        "telecom",
    }
)

_MIN_PROJECT_YEAR = 2015
_MAX_PROJECT_YEAR = 2026

_PERSON_KEYS = {
    "person_id",
    "name",
    "title",
    "office",
    "practice",
    "skills",
    "project_history",
    "authored",
    "languages",
    "last_updated",
}
_SKILL_KEYS = {"topic", "level", "basis"}
_PROJECT_KEYS = {"industry", "topic", "role", "year"}


class ExpertiseError(ValueError):
    """A record in the expertise dataset is missing a field or holds an
    invalid value — loading fails loudly rather than indexing a person
    whose evidence can't be trusted.
    """


@dataclass(frozen=True)
class Skill:
    """One claimed area of expertise. ``level`` is a 1–5 self-assessment;
    ``basis`` is ``evidenced`` or ``self_reported`` (see SKILL_BASES).
    """

    topic: str
    level: int
    basis: str


@dataclass(frozen=True)
class ProjectEntry:
    """One engagement in a consultant's history. Industry + topic + role +
    year only — never a client name (plan §2.3).
    """

    industry: str
    topic: str
    role: str
    year: int


@dataclass(frozen=True)
class Person:
    """A single consultant's expertise profile."""

    person_id: str
    name: str
    title: str
    office: str
    practice: str
    skills: tuple[Skill, ...]
    project_history: tuple[ProjectEntry, ...]
    authored: tuple[str, ...]
    languages: tuple[str, ...]
    last_updated: date_type

    @property
    def evidenced_topics(self) -> frozenset[str]:
        """Topics this person has a project or authored document behind —
        the set the B answer can cite as evidenced rather than claimed.
        """
        topics = {p.topic for p in self.project_history}
        return frozenset(topics)


def _require_str(record_id: str, field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExpertiseError(f"{record_id}: {field} is missing or empty")
    return value


def _load_skill(record_id: str, raw: object, corpus_topics: frozenset[str] | None) -> Skill:
    if not isinstance(raw, dict):
        raise ExpertiseError(f"{record_id}: each skill must be a mapping, got {raw!r}")
    extra = raw.keys() - _SKILL_KEYS
    missing = _SKILL_KEYS - raw.keys()
    if extra or missing:
        raise ExpertiseError(
            f"{record_id}: skill keys off — extra {sorted(extra)}, missing {sorted(missing)}"
        )
    topic = _require_str(record_id, "skill.topic", raw["topic"])
    if corpus_topics is not None and topic not in corpus_topics:
        raise ExpertiseError(
            f"{record_id}: skill topic {topic!r} not in the corpus topic vocabulary"
        )
    level = raw["level"]
    if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 5:
        raise ExpertiseError(f"{record_id}: skill.level {level!r} must be an int 1–5")
    basis = raw["basis"]
    if basis not in SKILL_BASES:
        raise ExpertiseError(
            f"{record_id}: skill.basis {basis!r} not in {sorted(SKILL_BASES)}"
        )
    return Skill(topic=topic, level=level, basis=basis)


def _load_project(
    record_id: str, raw: object, corpus_topics: frozenset[str] | None
) -> ProjectEntry:
    if not isinstance(raw, dict):
        raise ExpertiseError(
            f"{record_id}: each project_history entry must be a mapping, got {raw!r}"
        )
    extra = raw.keys() - _PROJECT_KEYS
    missing = _PROJECT_KEYS - raw.keys()
    if extra or missing:
        raise ExpertiseError(
            f"{record_id}: project_history keys off — extra {sorted(extra)}, "
            f"missing {sorted(missing)} (industry+topic+role+year only — "
            "no free-text field, so no client name can hide)"
        )
    industry = raw["industry"]
    if industry not in PROJECT_INDUSTRIES:
        raise ExpertiseError(
            f"{record_id}: project industry {industry!r} not in {sorted(PROJECT_INDUSTRIES)}"
        )
    topic = _require_str(record_id, "project.topic", raw["topic"])
    if corpus_topics is not None and topic not in corpus_topics:
        raise ExpertiseError(
            f"{record_id}: project topic {topic!r} not in the corpus topic vocabulary"
        )
    role = raw["role"]
    if role not in PROJECT_ROLES:
        raise ExpertiseError(
            f"{record_id}: project role {role!r} not in {sorted(PROJECT_ROLES)}"
        )
    year = raw["year"]
    if (
        not isinstance(year, int)
        or isinstance(year, bool)
        or not _MIN_PROJECT_YEAR <= year <= _MAX_PROJECT_YEAR
    ):
        raise ExpertiseError(
            f"{record_id}: project year {year!r} outside "
            f"{_MIN_PROJECT_YEAR}–{_MAX_PROJECT_YEAR}"
        )
    return ProjectEntry(industry=industry, topic=topic, role=role, year=year)


def load_person(
    raw: dict,
    *,
    corpus_paths: frozenset[str] | None = None,
    corpus_topics: frozenset[str] | None = None,
) -> Person:
    """Validate one raw record into a Person.

    ``corpus_paths`` / ``corpus_topics``, when given, tie the record back
    to the real pilot corpus: every ``authored`` path must resolve to a
    real file and every skill/project topic must be a real corpus topic.
    Both are optional so a unit test can build a Person without the corpus
    on disk.
    """
    if not isinstance(raw, dict):
        raise ExpertiseError(f"record must be a mapping, got {raw!r}")
    person_id = _require_str("<record>", "person_id", raw.get("person_id"))
    extra = raw.keys() - _PERSON_KEYS
    missing = _PERSON_KEYS - raw.keys()
    if extra or missing:
        raise ExpertiseError(
            f"{person_id}: keys off — extra {sorted(extra)}, missing {sorted(missing)}"
        )

    name = _require_str(person_id, "name", raw["name"])
    title = raw["title"]
    if title not in TITLES:
        raise ExpertiseError(f"{person_id}: title {title!r} not in {list(TITLES)}")
    office = _require_str(person_id, "office", raw["office"])
    practice = _require_str(person_id, "practice", raw["practice"])

    for field in ("skills", "project_history", "authored", "languages"):
        if not isinstance(raw[field], list):
            raise ExpertiseError(f"{person_id}: {field} must be a list")
    if not raw["skills"]:
        raise ExpertiseError(f"{person_id}: skills must be non-empty")
    if not raw["languages"]:
        raise ExpertiseError(f"{person_id}: languages must be non-empty")

    skills = tuple(_load_skill(person_id, s, corpus_topics) for s in raw["skills"])
    projects = tuple(
        _load_project(person_id, p, corpus_topics) for p in raw["project_history"]
    )

    authored: list[str] = []
    for path in raw["authored"]:
        path = _require_str(person_id, "authored entry", path)
        if corpus_paths is not None and path not in corpus_paths:
            raise ExpertiseError(
                f"{person_id}: authored path {path!r} does not resolve to a corpus file"
            )
        authored.append(path)

    languages = tuple(_require_str(person_id, "language", lang) for lang in raw["languages"])

    last_updated = raw["last_updated"]
    if not isinstance(last_updated, date_type):
        raise ExpertiseError(
            f"{person_id}: last_updated {last_updated!r} is not a valid ISO date"
        )

    return Person(
        person_id=person_id,
        name=name,
        title=title,
        office=office,
        practice=practice,
        skills=skills,
        project_history=projects,
        authored=tuple(authored),
        languages=languages,
        last_updated=last_updated,
    )


def _corpus_index(corpus_dir: Path) -> tuple[frozenset[str], frozenset[str]]:
    """(corpus-relative paths, topic vocabulary) for the corpus on disk —
    used to validate authored paths and skill/project topics.
    """
    import frontmatter

    paths: set[str] = set()
    topics: set[str] = set()
    for md in sorted(corpus_dir.rglob("*.md")):
        paths.add(md.relative_to(corpus_dir).as_posix())
        post = frontmatter.load(md)
        for topic in post.get("topics") or []:
            topics.add(topic)
    if not paths:
        raise ExpertiseError(f"no corpus files found under {corpus_dir}")
    return frozenset(paths), frozenset(topics)


def load_expertise(
    people_dir: Path, corpus_dir: Path | None = None
) -> list[Person]:
    """Load every ``*.yaml`` under people_dir, sorted for determinism.

    When corpus_dir is given, each record is validated against the real
    corpus (authored paths resolve, topics are real). Raises
    ExpertiseError on a duplicate person_id or any invalid record.
    """
    corpus_paths: frozenset[str] | None = None
    corpus_topics: frozenset[str] | None = None
    if corpus_dir is not None:
        corpus_paths, corpus_topics = _corpus_index(corpus_dir)

    files = sorted(people_dir.glob("*.yaml"))
    if not files:
        raise ExpertiseError(f"no expertise files found under {people_dir}")

    people: list[Person] = []
    seen: set[str] = set()
    for path in files:
        records = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ExpertiseError(f"{path}: expected a non-empty list of records")
        for record in records:
            person = load_person(
                record, corpus_paths=corpus_paths, corpus_topics=corpus_topics
            )
            if person.person_id in seen:
                raise ExpertiseError(f"duplicate person_id {person.person_id!r}")
            seen.add(person.person_id)
            people.append(person)

    people.sort(key=lambda p: p.person_id)
    return people
