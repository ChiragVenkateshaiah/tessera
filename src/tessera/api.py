"""HTTP transport over the pipeline — `POST /api/ask`, `POST
/api/feedback`, `GET /api/health`, and a placeholder page at `GET /`.

A composition-root-side adapter, like `cli.py` (CLAUDE.md constraint #6):
`create_app()` receives already-built dependencies (LLM client, embedder,
stores) and only translates between HTTP and `answer_query()`. It never
reads config itself — `tessera serve` builds the dependencies and hands
them over — so tests drive it with fakes and the query path stays pure.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from tessera.embedding.base import Embedder
from tessera.feedback.base import Feedback, FeedbackStore, Rating, TraceLog
from tessera.generation.base import LLMClient
from tessera.generation.usage import ModelPrice
from tessera.labels import ARCHETYPE_LABELS
from tessera.pipeline import AnswerResult, answer_query
from tessera.store.base import ExpertiseStore, VectorStore
from tessera.trace import trace_record

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000

ANSWER_FAILED_MESSAGE = (
    "Couldn't answer that question — the language model call failed. "
    "Try again in a moment."
)

# Whether a chat page replaces this is decided after P4-2 (Phase 4 plan §7).
PLACEHOLDER_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Tessera</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem;">
<h1>Tessera</h1>
<p>The API is running.</p>
<p>Ask a question with <code>POST /api/ask</code> and a JSON body
<code>{"question": "..."}</code>, or try it from <a href="/docs">/docs</a>.</p>
</body></html>
"""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    # Return the full trace record with the answer, not just its trace_id.
    include_trace: bool = False

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=64)
    rating: Rating
    reason: str | None = Field(default=None, max_length=200)
    comment: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)


def answer_to_dict(
    result: AnswerResult,
    latency_s: float,
    prices: Mapping[str, ModelPrice] | None = None,
) -> dict[str, Any]:
    """The JSON shape of one answer: everything `tessera query` prints,
    as data, plus the evidence behind each named person and the tokens
    (and, when every model is priced, the cost) it took.
    """
    usage = result.usage
    return {
        "question": result.query,
        "archetype": result.archetype.value,
        "archetype_label": ARCHETYPE_LABELS[result.archetype],
        "answer": result.answer,
        "citations": [
            {
                "marker": c.marker,
                "title": c.document_title,
                "heading_path": list(c.heading_path),
                "document_path": c.document_path,
            }
            for c in result.citations
        ],
        "experts": [
            {
                "rank": i,
                "person_id": m.person.person_id,
                "name": m.person.name,
                "title": m.person.title,
                "practice": m.person.practice,
                "office": m.person.office,
                "last_updated": m.person.last_updated.isoformat(),
                "evidenced": m.is_evidenced,
                "evidence": [
                    {
                        "kind": e.kind,
                        "description": e.description,
                        "self_reported": e.self_reported,
                    }
                    for e in m.evidence
                ],
            }
            for i, m in enumerate(result.experts, start=1)
        ],
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
        "cost_usd": usage.cost_usd(prices or {}),
        "latency_s": round(latency_s, 2),
    }


def create_app(
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    expertise_store: ExpertiseStore | None,
    *,
    llm_name: str,
    router_llm: LLMClient | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
    trace_log: TraceLog | None = None,
    feedback_store: FeedbackStore | None = None,
    new_trace_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> FastAPI:
    """Build the HTTP app around already-constructed dependencies.

    llm_name is reported by /api/health so a demo can show which model is
    answering; it must not contain secrets. router_llm (optional) makes the
    routing call; prices turn each answer's tokens into ``cost_usd``
    (None when a model is unpriced). Every answer gets a ``trace_id`` and
    its trace is written to trace_log; feedback on it goes to
    feedback_store (``/api/feedback`` answers 503 without both).
    """
    app = FastAPI(
        title="Tessera",
        summary="Internal knowledge assistant — grounded answers with citations.",
    )
    # One question at a time. The embedder and the LLM rate limits are the
    # bottleneck either way, and a single-user demo gains nothing from
    # concurrent pipeline runs; on Lambda each instance serves one request.
    pipeline_lock = threading.Lock()

    @app.post("/api/ask")
    def ask(request: AskRequest) -> JSONResponse:
        start = time.perf_counter()
        try:
            with pipeline_lock:
                result = answer_query(
                    request.question,
                    llm,
                    embedder,
                    store,
                    expertise_store,
                    router_llm=router_llm,
                )
        except Exception:  # the client gets a message, the log gets the trace
            logger.exception("answer_query failed for an /api/ask request")
            return JSONResponse(status_code=502, content={"error": ANSWER_FAILED_MESSAGE})
        latency = time.perf_counter() - start
        trace_id = new_trace_id()
        record = trace_record(
            trace_id,
            result,
            timestamp=now(),
            latency_s=latency,
            prices=prices or {},
            llm=llm_name,
        )
        if trace_log is not None:
            try:
                trace_log.append(record)
            except Exception:  # losing a trace must not lose the answer
                logger.exception("couldn't write trace %s", trace_id)
        body = answer_to_dict(result, latency, prices)
        body["trace_id"] = trace_id
        if request.include_trace:
            body["trace"] = record
        return JSONResponse(body)

    @app.post("/api/feedback", status_code=201)
    def feedback(request: FeedbackRequest) -> JSONResponse:
        if trace_log is None or feedback_store is None:
            return JSONResponse(
                status_code=503, content={"error": "Feedback isn't enabled on this server."}
            )
        if trace_log.get(request.trace_id) is None:
            return JSONResponse(
                status_code=404, content={"error": f"No answer with trace_id {request.trace_id!r}."}
            )
        feedback_store.add(
            Feedback(
                trace_id=request.trace_id,
                rating=request.rating,
                created_at=now(),
                reason=request.reason,
                comment=request.comment,
            )
        )
        return JSONResponse(status_code=201, content={"status": "recorded"})

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "document_chunks": store.count(),
            "people": expertise_store.count() if expertise_store is not None else 0,
            "people_search": expertise_store is not None,
            "llm": llm_name,
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PLACEHOLDER_PAGE

    return app
