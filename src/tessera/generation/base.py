"""LLMClient interface — the swappable port between NVIDIA NIM (Phase 1)
and Claude via Bedrock (Phase 4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    """Token usage of one LLM call, as reported by the provider."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


@dataclass(frozen=True)
class Completion:
    """A completion's text plus its usage — None when the client can't
    tell (a provider that doesn't report tokens, or a test fake).
    """

    text: str
    usage: Usage | None = None


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

    def complete_with_usage(
        self, system: str, user: str, temperature: float = 0.0
    ) -> Completion:
        """Like complete(), but also returns the call's token usage.

        Phase 4 (cost accounting). Clients that know their usage override
        this and implement complete() on top of it; the default here keeps
        every other client (and test fake) working, with usage=None.
        """
        return Completion(self.complete(system, user, temperature))
