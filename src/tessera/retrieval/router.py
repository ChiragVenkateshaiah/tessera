"""Archetype classification (A/B/C/D)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from tessera.generation.base import LLMClient
from tessera.generation.prompts import ROUTER_SYSTEM_PROMPT, build_router_user_prompt

logger = logging.getLogger(__name__)


class Archetype(str, Enum):
    """The four query archetypes (Discovery Findings §7)."""

    LOOKUP = "A"
    EXPERTISE = "B"
    SYNTHESIS = "C"
    COMPARATIVE = "D"


NOT_YET_SUPPORTED_MESSAGE = (
    "Expertise-finding isn't supported yet — that needs staffing/HR data "
    "this system doesn't have access to. I can help you find existing "
    "documents or get up to speed on a topic instead."
)

COMPARATIVE_REFUSAL_MESSAGE = (
    "I can't compare approaches across specific client engagements — "
    "that risks pulling from restricted or confidentiality-sensitive "
    "material. I can help with finding existing documents or synthesizing "
    "a topic from what's in the corpus instead."
)


def terminal_response_for(archetype: Archetype) -> str | None:
    """A fixed response for archetypes that never reach retrieval.

    Returns None for A/C, meaning "proceed to retrieval" — the caller
    (pipeline.py, Task 5/6) checks this before doing any retrieval work.
    """
    if archetype is Archetype.EXPERTISE:
        return NOT_YET_SUPPORTED_MESSAGE
    if archetype is Archetype.COMPARATIVE:
        return COMPARATIVE_REFUSAL_MESSAGE
    return None


@dataclass(frozen=True)
class RoutingDecision:
    """The classifier's output for one query — logged and returned as-is,
    so the routing decision is always inspectable (Task 4 acceptance
    check), not just implicit in downstream behavior.
    """

    query: str
    archetype: Archetype
    reasoning: str


class RoutingError(ValueError):
    """The LLM's routing response couldn't be parsed into a valid decision."""


def route(query: str, llm: LLMClient) -> RoutingDecision:
    """Classify a query into archetype A/B/C/D via the given LLM client.

    Pure with respect to infrastructure per CLAUDE.md constraint #6: the
    LLMClient is injected, not constructed here — this function doesn't
    know or care whether it's talking to Gemini, Bedrock, or a test fake.
    """
    raw = llm.complete(
        system=ROUTER_SYSTEM_PROMPT, user=build_router_user_prompt(query)
    )

    try:
        parsed = json.loads(_strip_markdown_fence(raw))
        archetype = Archetype(parsed["archetype"])
        reasoning = parsed["reasoning"]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise RoutingError(f"could not parse routing response: {raw!r}") from exc

    decision = RoutingDecision(query=query, archetype=archetype, reasoning=reasoning)
    logger.info(
        "routed query=%r -> archetype=%s reasoning=%r",
        query,
        archetype.value,
        reasoning,
    )
    return decision


def _strip_markdown_fence(text: str) -> str:
    """Some models wrap JSON in ```json ... ``` even when told not to —
    strip it rather than fail the parse over formatting.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped
