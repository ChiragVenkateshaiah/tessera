"""Reads the ethical-wall data (Phase 4, plan §3.5.2): which people are
cleared for which restricted engagement.

The local stand-in for a real entitlement / ethical-wall system, the way
``expertise_loader.py`` stands in for an HR feed. Deny by default: a
person is cleared for an engagement only if its list names them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class AccessError(ValueError):
    """The wall data is malformed or inconsistent with the corpus/people."""


@dataclass(frozen=True)
class Walls:
    """Engagement codename -> person_ids cleared for it, plus the named
    demo personas (persona name -> person_id).
    """

    cleared: dict[str, frozenset[str]]
    personas: dict[str, str] = field(default_factory=dict)

    def is_cleared(self, person_id: str | None, engagement: str | None) -> bool:
        """True only if person_id is on engagement's list. No person, no
        engagement, or an engagement with no list: not cleared.
        """
        if person_id is None or engagement is None:
            return False
        return person_id in self.cleared.get(engagement, frozenset())

    def engagements_for(self, person_id: str | None) -> frozenset[str]:
        """Every engagement person_id is cleared for."""
        return frozenset(e for e, people in self.cleared.items() if person_id in people)


def load_walls(
    path: Path,
    *,
    person_ids: Iterable[str] | None = None,
    engagements: Iterable[str] | None = None,
) -> Walls:
    """Load and validate walls.yaml.

    When person_ids is given, every cleared person and persona must be a
    real person. When engagements (the codenames on restricted corpus
    documents) is given, the two sets must match exactly — a restricted
    document with no wall would be invisible to everyone, and a wall with
    no document is stale.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("walls"), dict):
        raise AccessError(f"{path}: expected a mapping with a 'walls' mapping")

    cleared: dict[str, frozenset[str]] = {}
    for engagement, people in raw["walls"].items():
        if not isinstance(people, list) or not all(isinstance(p, str) for p in people):
            raise AccessError(f"{path}: wall {engagement!r} must be a list of person_ids")
        if len(set(people)) != len(people):
            raise AccessError(f"{path}: wall {engagement!r} lists a person twice")
        cleared[engagement] = frozenset(people)

    personas: dict[str, str] = {}
    for name, entry in (raw.get("personas") or {}).items():
        if not isinstance(entry, dict) or not isinstance(entry.get("person_id"), str):
            raise AccessError(f"{path}: persona {name!r} needs a person_id")
        personas[name] = entry["person_id"]

    if person_ids is not None:
        named = {p for people in cleared.values() for p in people} | set(personas.values())
        unknown = sorted(named - set(person_ids))
        if unknown:
            raise AccessError(f"{path}: unknown person_ids {unknown}")

    if engagements is not None:
        expected = set(engagements)
        missing, stale = sorted(expected - set(cleared)), sorted(set(cleared) - expected)
        if missing:
            raise AccessError(f"{path}: no wall for restricted engagements {missing}")
        if stale:
            raise AccessError(f"{path}: walls for engagements with no document {stale}")

    return Walls(cleared=cleared, personas=personas)
