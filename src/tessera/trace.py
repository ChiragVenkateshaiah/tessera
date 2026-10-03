"""Request traces (Phase 4, plan §3.2.1).

The pipeline returns a `Trace` as data on every answer — what the router
decided and why, what retrieval found, which of it cleared the floors and
reached the model. `trace_record()` turns that plus the request's id,
time, latency and usage into the one JSON-ready record a composition root
writes to the trace log. Nothing here does I/O (CLAUDE.md constraint #6):
the CLI and API are the only writers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from tessera.generation.usage import ModelPrice

if TYPE_CHECKING:
    from tessera.pipeline import AnswerResult


@dataclass(frozen=True)
class RetrievedItem:
    """One retrieval candidate: a chunk (A/C) or a person (B).

    score is the value the floor is applied to — chunk similarity for
    documents, evidence_score for people. ``used`` is True when it cleared
    the floor and was shown to the model.
    """

    id: str
    score: float
    used: bool
    document_path: str | None = None


@dataclass(frozen=True)
class Trace:
    """How one answer was produced, independent of transport."""

    route_reasoning: str
    retrieved_kind: Literal["chunk", "person"] | None = None
    retrieved: tuple[RetrievedItem, ...] = ()
    # The thresholds applied on this path, by name.
    floors: dict[str, float] = field(default_factory=dict)
    # True when a fixed message answered without a generation call: a
    # terminal archetype (D), nothing over the floor, or no people index.
    fixed_response: bool = False
    # Chunks each exclusion filter kept out of the candidate pool, by
    # filter name (e.g. {"superseded": 3}), and the superseded documents
    # the answer's note pointed away from.
    removed: dict[str, int] = field(default_factory=dict)
    superseded: tuple[str, ...] = ()
    # Who document retrieval was scoped to (a demo identity), or None for
    # internal-only.
    principal: str | None = None


def trace_record(
    trace_id: str,
    result: AnswerResult,
    *,
    timestamp: datetime,
    latency_s: float,
    prices: Mapping[str, ModelPrice],
    llm: str,
) -> dict[str, Any]:
    """The JSON-ready trace record for one answered request."""
    trace = result.trace
    usage = result.usage
    return {
        "trace_id": trace_id,
        "timestamp": timestamp.isoformat(),
        "query": result.query,
        "archetype": result.archetype.value,
        "route_reasoning": trace.route_reasoning,
        "retrieved_kind": trace.retrieved_kind,
        "retrieved": [
            {
                "id": item.id,
                "score": round(item.score, 4),
                "used": item.used,
                **({"document_path": item.document_path} if item.document_path else {}),
            }
            for item in trace.retrieved
        ],
        "floors": dict(trace.floors),
        "fixed_response": trace.fixed_response,
        "principal": trace.principal,
        "removed": dict(trace.removed),
        "superseded": list(trace.superseded),
        "citations": [c.document_path for c in result.citations],
        "experts": [m.person.person_id for m in result.experts],
        "llm": llm,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "calls": [
                {
                    "model": u.model,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "latency_s": round(u.latency_s, 2),
                }
                for u in usage.calls
            ],
        },
        "cost_usd": usage.cost_usd(prices),
        "latency_s": round(latency_s, 2),
        "answer_chars": len(result.answer),
    }
