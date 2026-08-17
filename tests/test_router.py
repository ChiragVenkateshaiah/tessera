"""Unit tests for the router logic (fake LLM client, no network), plus a
live integration test against the 8 Discovery Findings §7 placeholder
queries.

The live test is opt-in via RUN_LIVE_LLM_TESTS=1, not just "GEMINI_API_KEY
is set" — Gemini's free tier caps gemini-3.6-flash at 20 requests/*day*
(not just per-minute), so a routine `pytest tests/` run must never spend
that budget by default. Run explicitly with:
    RUN_LIVE_LLM_TESTS=1 pytest tests/test_router.py -k live
"""

import os

import pytest

from tessera.generation.base import LLMClient
from tessera.retrieval.router import (
    Archetype,
    RoutingDecision,
    RoutingError,
    route,
    terminal_response_for,
)


class FakeLLMClient(LLMClient):
    """Returns a canned response regardless of input — for testing the
    parsing/error-handling logic in router.py in isolation.
    """

    def __init__(self, canned_response: str) -> None:
        self._response = canned_response

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self._response


def test_route_parses_valid_json_response() -> None:
    llm = FakeLLMClient('{"archetype": "A", "reasoning": "asking for a template"}')

    decision = route("Do we have a framework for X?", llm)

    assert decision.archetype is Archetype.LOOKUP
    assert decision.reasoning == "asking for a template"
    assert decision.query == "Do we have a framework for X?"


def test_route_strips_markdown_fence() -> None:
    llm = FakeLLMClient(
        '```json\n{"archetype": "C", "reasoning": "wants a briefing"}\n```'
    )

    decision = route("get me up to speed", llm)

    assert decision.archetype is Archetype.SYNTHESIS


@pytest.mark.parametrize(
    "bad_response",
    [
        "not json at all",
        '{"archetype": "Z", "reasoning": "invalid letter"}',
        '{"reasoning": "missing archetype key"}',
        '{"archetype": "A"}',  # missing reasoning
        "",
    ],
)
def test_route_raises_routing_error_on_malformed_response(
    bad_response: str,
) -> None:
    llm = FakeLLMClient(bad_response)

    with pytest.raises(RoutingError):
        route("some query", llm)


def test_terminal_response_none_for_lookup_and_synthesis() -> None:
    assert terminal_response_for(Archetype.LOOKUP) is None
    assert terminal_response_for(Archetype.SYNTHESIS) is None


def test_terminal_response_present_for_expertise_and_comparative() -> None:
    expertise_msg = terminal_response_for(Archetype.EXPERTISE)
    comparative_msg = terminal_response_for(Archetype.COMPARATIVE)

    assert expertise_msg is not None
    assert comparative_msg is not None
    assert expertise_msg != comparative_msg
    assert "not" in expertise_msg.lower() or "yet" in expertise_msg.lower()
    assert (
        "confidential" in comparative_msg.lower()
        or "restrict" in comparative_msg.lower()
    )


def test_routing_decision_is_a_plain_inspectable_object() -> None:
    llm = FakeLLMClient('{"archetype": "B", "reasoning": "asking for a person"}')

    decision = route("who knows about X?", llm)

    assert isinstance(decision, RoutingDecision)
    # every field is directly readable — no hidden state, nothing that
    # requires re-parsing raw LLM output to inspect the decision
    assert decision.archetype.value == "B"


# --- Live integration: the 8 Discovery Findings §7 placeholder queries ---

PLACEHOLDER_QUERIES: list[tuple[str, Archetype]] = [
    ("Do we have a framework for market entry analysis?", Archetype.LOOKUP),
    (
        "Have we done work in financial services on GenAI before?",
        Archetype.LOOKUP,
    ),
    ("Who at the firm knows about pharma pricing?", Archetype.EXPERTISE),
    (
        "Who's our expert on supply chain network optimization?",
        Archetype.EXPERTISE,
    ),
    (
        "I'm staffed on a retail-bank cost transformation Monday — "
        "what should I read first?",
        Archetype.SYNTHESIS,
    ),
    (
        "Client meeting in an hour, they asked about pricing elasticity — "
        "do we have anything?",
        Archetype.SYNTHESIS,
    ),
    (
        "How did we approach margin improvement for Client X vs. the "
        "standard playbook?",
        Archetype.COMPARATIVE,
    ),
    (
        "Compare our pricing engagement for Acme Corp against Beta Inc.",
        Archetype.COMPARATIVE,
    ),
]


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TESTS") != "1",
    reason=(
        "live LLM test opt-in only (RUN_LIVE_LLM_TESTS=1) — Gemini free "
        "tier caps gemini-3.6-flash at 20 requests/day, not just per-minute"
    ),
)
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — live routing check skipped",
)
@pytest.mark.parametrize("query,expected_archetype", PLACEHOLDER_QUERIES)
def test_placeholder_queries_route_correctly_live(
    query: str, expected_archetype: Archetype
) -> None:
    from tessera.generation.gemini import GeminiClient

    llm = GeminiClient(
        api_key=os.environ["GEMINI_API_KEY"],
        model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
    )

    decision = route(query, llm)

    assert decision.archetype is expected_archetype, (
        f"query={query!r} routed to {decision.archetype.value}, "
        f"expected {expected_archetype.value}. reasoning={decision.reasoning!r}"
    )
