"""Human-readable archetype names, shared by the CLI and the HTTP API."""

from __future__ import annotations

from tessera.retrieval.router import Archetype

ARCHETYPE_LABELS: dict[Archetype, str] = {
    Archetype.LOOKUP: "lookup",
    Archetype.EXPERTISE: "expertise",
    Archetype.SYNTHESIS: "synthesis",
    Archetype.COMPARATIVE: "comparative — declined",
}
