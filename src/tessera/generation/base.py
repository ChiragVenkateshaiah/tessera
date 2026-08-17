"""LLMClient interface — the swappable port between Gemini (Phase 1) and
Claude via Bedrock (Phase 4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Sends a system + user message pair to an LLM, returns its text
    response.

    Deliberately minimal — a single-turn system/user completion, no chat
    history — because nothing in Phase 1 needs more than that: archetype
    routing (Task 4) and grounded generation (Task 6) are both one-shot
    calls. Constructor takes credentials/model as parameters, never reads
    environment variables itself (CLAUDE.md constraint #6 — config-sourcing
    is the caller's job, not this module's).
    """

    @abstractmethod
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Send a system instruction + user message, return the model's
        text response.
        """
