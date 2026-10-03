"""Traces returned by the pipeline and the record built from them
(`tessera.trace`) — fakes only, no model or index.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tessera import pipeline
from tessera.generation.answer import NO_RESULTS_MESSAGE, RELEVANCE_THRESHOLD
from tessera.generation.base import Completion, Usage
from tessera.generation.expertise import EXPERTISE_PERSON_FLOOR, EXPERTISE_QUERY_FLOOR
from tessera.generation.prompts import (
    EXPERTISE_ANSWER_SYSTEM_PROMPT,
    LOOKUP_ANSWER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from tessera.generation.usage import ModelPrice
from tessera.pipeline import answer_query
from tessera.retrieval.expertise import ExpertiseResult
from tessera.retrieval.router import Archetype
from tessera.trace import trace_record
from tests.test_expertise_generation import match
from tests.test_pipeline import (
    FakeEmbedder,
    FakeVectorStore,
    ScriptedLLMClient,
    _result,
    _router_response,
)

ROUTE_A = '{"archetype": "A", "reasoning": "asks for a specific framework"}'


class Metered(ScriptedLLMClient):
    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0):
        return Completion(self.complete(system, user, temperature), Usage("m", 1000, 50, 0.2))


def test_lookup_trace_lists_every_chunk_and_which_cleared_the_floor() -> None:
    llm = ScriptedLLMClient(
        {ROUTER_SYSTEM_PROMPT: ROUTE_A, LOOKUP_ANSWER_SYSTEM_PROMPT: "See [1]."}
    )
    store = FakeVectorStore([_result("data/corpus/a.md", 0.61), _result("data/corpus/b.md", 0.20)])

    trace = answer_query("pricing framework?", llm, FakeEmbedder(), store).trace

    assert trace.route_reasoning == "asks for a specific framework"
    assert trace.retrieved_kind == "chunk"
    assert [(i.id, i.used) for i in trace.retrieved] == [
        ("data/corpus/a.md#1", True),
        ("data/corpus/b.md#1", False),
    ]
    assert trace.retrieved[0].document_path == "data/corpus/a.md"
    assert trace.floors == {"relevance_threshold": RELEVANCE_THRESHOLD}
    assert trace.fixed_response is False


def test_nothing_over_the_floor_is_traced_as_a_fixed_response() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: ROUTE_A})
    store = FakeVectorStore([_result("data/corpus/a.md", 0.10)])

    result = answer_query("parental leave policy?", llm, FakeEmbedder(), store)

    assert result.answer == NO_RESULTS_MESSAGE
    assert result.trace.fixed_response is True
    assert [i.used for i in result.trace.retrieved] == [False]


def test_comparative_trace_has_the_route_and_nothing_retrieved() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("D")})

    trace = answer_query("Acme vs Globex?", llm, FakeEmbedder(), FakeVectorStore([])).trace

    assert trace.retrieved_kind is None and trace.retrieved == ()
    assert trace.fixed_response is True


def test_expertise_trace_lists_people_by_evidence_score(monkeypatch: pytest.MonkeyPatch) -> None:
    found = ExpertiseResult(query="q", matches=[match("c0001", 1.4), match("c0002", 0.0)])
    monkeypatch.setattr(pipeline, "find_experts", lambda q, e, s: found)
    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("B"),
            EXPERTISE_ANSWER_SYSTEM_PROMPT: "Ask [1].",
        }
    )

    trace = answer_query(
        "who knows pricing?", llm, FakeEmbedder(), FakeVectorStore([]), expertise_store=object()
    ).trace

    assert trace.retrieved_kind == "person"
    assert [(i.id, i.score, i.used) for i in trace.retrieved] == [
        ("c0001", 1.4, True),
        ("c0002", 0.0, False),  # under EXPERTISE_PERSON_FLOOR, not shown
    ]
    assert trace.floors == {
        "expertise_query_floor": EXPERTISE_QUERY_FLOOR,
        "expertise_person_floor": EXPERTISE_PERSON_FLOOR,
    }


def test_trace_record_carries_route_retrieval_tokens_cost_and_latency() -> None:
    llm = Metered({ROUTER_SYSTEM_PROMPT: ROUTE_A, LOOKUP_ANSWER_SYSTEM_PROMPT: "See [1]."})
    store = FakeVectorStore([_result("data/corpus/a.md", 0.612345)])
    result = answer_query("pricing framework?", llm, FakeEmbedder(), store)

    record = trace_record(
        "abc123",
        result,
        timestamp=datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc),
        latency_s=3.14159,
        prices={"m": ModelPrice(1.0, 10.0)},
        llm="fake:model",
    )

    assert record["trace_id"] == "abc123"
    assert record["timestamp"] == "2026-10-02T12:00:00+00:00"
    assert record["query"] == "pricing framework?"
    assert record["archetype"] == "A"
    assert record["route_reasoning"] == "asks for a specific framework"
    assert record["retrieved"] == [
        {"id": "data/corpus/a.md#1", "score": 0.6123, "used": True, "document_path": "data/corpus/a.md"}
    ]
    assert record["citations"] == ["data/corpus/a.md"]
    assert record["usage"]["input_tokens"] == 2000
    assert record["cost_usd"] == pytest.approx(2 * (1000 * 1 + 50 * 10) / 1e6)
    assert record["latency_s"] == 3.14
    assert record["answer_chars"] == len("See [1].")
    assert record["llm"] == "fake:model"


def test_superseded_exclusion_is_traced_as_a_removal_count() -> None:
    from dataclasses import replace

    llm = ScriptedLLMClient(
        {ROUTER_SYSTEM_PROMPT: ROUTE_A, LOOKUP_ANSWER_SYSTEM_PROMPT: "See [1]."}
    )
    old = replace(
        _result("data/corpus/old.md", 0.9), status="superseded", superseded_by="data/corpus/a.md"
    )
    store = FakeVectorStore([old, _result("data/corpus/a.md", 0.61)])

    result = answer_query("pricing framework?", llm, FakeEmbedder(), store)

    assert result.trace.removed == {"superseded": 1, "restricted": 0}
    assert result.trace.superseded == ("data/corpus/old.md",)
    assert "data/corpus/old.md" not in [i.document_path for i in result.trace.retrieved]
    record = trace_record(
        "t", result, timestamp=datetime(2026, 10, 3, tzinfo=timezone.utc),
        latency_s=1.0, prices={}, llm="fake",
    )
    assert record["removed"] == {"superseded": 1, "restricted": 0}
    assert record["superseded"] == ["data/corpus/old.md"]
