"""The person a question is asked as (Phase 4, plan §3.5.4).

A typed parameter, passed into ``answer_query()`` and down to retrieval —
never read from a session, a header or a global (CLAUDE.md constraint
#6). The composition roots resolve it: they look the person up in the
ethical-wall data and hand the core the engagements they are cleared for.

This is a DEMO identity, not authentication: Tessera takes whoever the
caller says they are at their word. Real identity (SSO) is Phase 6+.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    """A person and the restricted engagements they are cleared for.

    No principal at all (``None`` where one is accepted) means internal
    documents only — the same as a principal cleared for nothing.
    """

    person_id: str
    engagements: frozenset[str] = frozenset()
