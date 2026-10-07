"""LangSmith tracing through a redacting client (Phase 5 plan §3.9.1–3.9.4).

Never env-driven: a composition root builds a ``LangSmithTracer`` from
Tessera's own config (``build_tracer``) and every traced run happens
inside ``tracing_context(enabled=True, client=<the redacting client>)``.

**The native stack** is traced by wrapping its injected step functions and
its LLM ports here, at the composition root (``native_steps()``,
``llm()``), never inside core modules. ``VectorStore`` is not wrapped: its
withheld probe returns unpermitted restricted chunks.

**Redaction** (``RunRedactor``) uses the client's two hooks, because the
P5-1 spike found that only the pair reaches every field:
- ``anonymizer`` redacts inputs, outputs, metadata and errors as each run
  is created or updated;
- ``process_buffered_run_ops`` sees every buffered run dict whole, before
  serialization. It redacts every string field (names, tags, events,
  serialized, extra), drops the restricted-chunk removal count, and — for
  a trace whose retrieval touched restricted or quarantined content —
  hides every run's inputs, outputs and events in full and replaces its
  error text. A run of a trace it has no record of is hidden in full too
  (fail closed).

**Timing.** The client never flushes on its own (the buffer size and
timeout are set out of reach); ``LangSmithTracer.run`` records the
request's taint for its trace id when the run ends, *then* flushes. So
whole-trace hiding is decided with the whole trace known, including the
routing run that happened before retrieval.

Tessera's ``trace_id`` is the root run's ``run_id``, so a JSONL trace and
its LangSmith trace line up. The asker appears as an opaque per-process
reference, never their person_id.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from typing import Any

import langsmith as ls
import requests
from langsmith import Client, traceable

from tessera.generation.answer import GeneratedAnswer, generate_answer
from tessera.generation.base import Completion, LLMClient
from tessera.generation.expertise import generate_expertise_answer
from tessera.observability.taint import (
    CURRENT_REQUEST,
    Placeholders,
    Redactor,
    RequestTaint,
    TaintTerms,
)
from tessera.pipeline import Pipeline, PipelineRun
from tessera.principal import Principal
from tessera.retrieval.expertise import ExpertiseResult
from tessera.retrieval.expertise import find_experts as _core_find_experts
from tessera.retrieval.retriever import RetrievalResult, retrieve
from tessera.retrieval.router import RoutingDecision
from tessera.retrieval.router import route as _core_route

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.smith.langchain.com"

HIDDEN = "[hidden: this trace touched restricted or quarantined content]"
HIDDEN_ERROR = "[error text hidden: this trace touched restricted or quarantined content]"

# Run fields that are identifiers, timestamps or enums — left untouched so
# the SDK's id/order checks hold. Everything else is redacted.
_STRUCTURAL_FIELDS = frozenset(
    {
        "id",
        "trace_id",
        "parent_run_id",
        "dotted_order",
        "session_id",
        "session_name",
        "start_time",
        "end_time",
        "run_type",
        "reference_example_id",
    }
)
# The SDK only flushes the run-ops buffer at this size or after this long;
# both are out of reach, so only LangSmithTracer.run's flush sends.
_NEVER_FLUSH_SIZE = 1 << 30
_NEVER_FLUSH_MS = 1e15

_ERROR_TYPE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)")


def _without_removal_count(value: Any) -> Any:
    """``value`` with the restricted-chunk removal count dropped from every
    ``removed`` mapping: shown to a walled asker, it would say restricted
    material on their question exists (Phase 4 plan §3.5.4)."""
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if k == "removed" and isinstance(v, Mapping):
                v = {name: n for name, n in v.items() if name != "restricted"}
            out[k] = _without_removal_count(v)
        return out
    if isinstance(value, list):
        return [_without_removal_count(v) for v in value]
    return value


def _trace_key(value: Any) -> uuid.UUID | None:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


class RunRedactor:
    """The redacting client's hooks, plus the record of which traces have
    ended and how tainted they were."""

    def __init__(self, terms: TaintTerms) -> None:
        self._static = Redactor(terms)
        self._closed: dict[uuid.UUID, RequestTaint] = {}
        self._processed: set[uuid.UUID] = set()
        self._cond = threading.Condition()

    # --- the client's anonymizer: inputs, outputs, metadata, error ---

    def anonymize(self, data: Any) -> Any:
        request = CURRENT_REQUEST.get()
        if request is None:
            return self._static.value(data, Placeholders())
        return request.redactor().value(data, request.placeholders)

    # --- end of a request ---

    def close(self, trace_id: str, request: RequestTaint) -> None:
        """Record a finished request's taint; its runs can now be sent."""
        key = _trace_key(trace_id)
        if key is None:
            raise ValueError(f"trace_id {trace_id!r} is not a UUID")
        with self._cond:
            self._closed[key] = request

    def wait_processed(self, trace_id: str, timeout: float) -> bool:
        """Block until a batch holding ``trace_id``'s runs went through
        ``process``; False on timeout."""
        key = _trace_key(trace_id)
        with self._cond:
            done = self._cond.wait_for(lambda: key in self._processed, timeout=timeout)
            self._processed.discard(key)
            return done

    # --- the client's process_buffered_run_ops: every field, whole runs ---

    def process(self, runs: Iterable[dict]) -> list[dict]:
        runs = list(runs)
        keys = {_trace_key(r.get("trace_id")) for r in runs}
        try:
            with self._cond:
                requests_by_trace = {k: self._closed.get(k) for k in keys if k is not None}
            return [self._process_run(r, requests_by_trace) for r in runs]
        finally:
            with self._cond:
                for key in keys:
                    if key is not None:
                        self._processed.add(key)
                        # A trace's runs are all buffered before it closes,
                        # so it never comes back; a straggler would be
                        # unknown, hence hidden in full.
                        self._closed.pop(key, None)
                self._cond.notify_all()

    def _process_run(
        self, run: dict, requests_by_trace: Mapping[uuid.UUID, RequestTaint | None]
    ) -> dict:
        request = requests_by_trace.get(_trace_key(run.get("trace_id")))  # type: ignore[arg-type]
        if request is None:
            redactor, placeholders, hide = self._static, Placeholders(), True
        else:
            redactor, placeholders, hide = request.redactor(), request.placeholders, request.touched
        out: dict[str, Any] = {}
        for key, value in run.items():
            if key in _STRUCTURAL_FIELDS:
                out[key] = value
            elif key == "attachments":
                out[key] = {}
            elif hide and key in ("inputs", "outputs"):
                out[key] = {"withheld": HIDDEN} if value else value
            elif hide and key == "events":
                out[key] = []
            elif hide and key == "error" and value:
                match = _ERROR_TYPE.match(str(value))
                kind = redactor.text(match.group(1), placeholders) if match else "Error"
                out[key] = f"{kind}: {HIDDEN_ERROR}"
            else:
                out[key] = redactor.value(_without_removal_count(value), placeholders)
        return out


# --- what each traced step shows ---


def _or_empty(shape: Callable[[Any], dict[str, Any]]) -> Callable[[Any], dict[str, Any]]:
    """A run's output shape; a step that raised has no output (None)."""
    return lambda value: {} if value is None else shape(value)


def _search_result_doc(r: Any) -> dict[str, Any]:
    return {
        "type": "Document",
        "page_content": r.text,
        "metadata": {
            "chunk_id": r.chunk_id,
            "document_path": r.document_path,
            "title": r.document_title,
            "heading_path": list(r.heading_path),
            "score": round(r.score, 4),
            "sensitivity": r.sensitivity,
        },
    }


def _retrieval_outputs(result: RetrievalResult) -> dict[str, Any]:
    return {
        "documents": [_search_result_doc(r) for r in result.results],
        "superseded": [m.document_path for m in result.superseded],
        "removed": {k: v for k, v in result.removed.items() if k != "restricted"},
    }


def _generated_outputs(generated: GeneratedAnswer) -> dict[str, Any]:
    return {
        "answer": generated.answer,
        "citations": [c.document_path for c in generated.citations],
        "experts": [m.person.person_id for m in generated.experts],
    }


def _run_outputs(run: PipelineRun) -> dict[str, Any]:
    answer = run.answer
    return {
        "archetype": answer.archetype.value,
        "answer": answer.answer,
        "citations": [c.document_path for c in answer.citations],
        "experts": [m.person.person_id for m in answer.experts],
        "fixed_response": answer.trace.fixed_response,
        "retrieval_attempts": len(run.retrievals),
    }


class _TracedLLM(LLMClient):
    """An LLM port whose every call is an ``llm`` run: the rendered system
    and user messages in, the text and token usage out."""

    def __init__(self, inner: LLMClient, role: str) -> None:
        self._inner = inner
        self._call = traceable(
            run_type="llm",
            name=f"{role}-llm",
            process_inputs=lambda i: {
                "messages": [
                    {"role": "system", "content": i["system"]},
                    {"role": "user", "content": i["user"]},
                ],
                "temperature": i.get("temperature", 0.0),
            },
            process_outputs=_or_empty(_completion_outputs),
        )(self._complete)

    def _complete(self, system: str, user: str, temperature: float = 0.0) -> Completion:
        return self._inner.complete_with_usage(system, user, temperature)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Completion:
        return self._call(system, user, temperature)


def _completion_outputs(completion: Completion) -> dict[str, Any]:
    out: dict[str, Any] = {
        "choices": [{"message": {"role": "assistant", "content": completion.text}}]
    }
    if completion.usage is not None:
        u = completion.usage
        out["usage_metadata"] = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "total_tokens": u.input_tokens + u.output_tokens,
        }
        out["model"] = u.model
    return out


def _route_outputs(decision: RoutingDecision) -> dict[str, Any]:
    return {"archetype": decision.archetype.value, "reasoning": decision.reasoning}


def _expertise_outputs(found: ExpertiseResult) -> dict[str, Any]:
    return {
        "people": [
            {"person_id": m.person.person_id, "evidence_score": round(m.evidence_score, 4)}
            for m in found.matches
        ]
    }


class LangSmithTracer:
    """Traces pipeline runs to LangSmith through the redacting client.

    One per process (built by ``build_tracer``). ``native_steps()`` and
    ``llm()`` give a composition root the wrapped steps and ports to build
    a ``NativePipeline`` from; ``run()`` answers one question as one trace.
    """

    def __init__(
        self,
        client: Client,
        redactor: RunRedactor,
        terms: TaintTerms,
        *,
        project: str,
        stack: str = "native",
        switches: Mapping[str, str] | None = None,
        flush_timeout: float = 30.0,
    ) -> None:
        self.client = client
        self._redactor = redactor
        self._terms = terms
        self._project = project
        self._metadata = {"stack": stack, "switches": dict(switches or {})}
        self._flush_timeout = flush_timeout
        # Keyed per process: the reference is stable within a session and
        # can't be reversed by hashing the 600 person_ids.
        self._principal_key = secrets.token_bytes(32)
        self._root = traceable(
            run_type="chain",
            name="tessera",
            process_inputs=lambda i: {"query": i["query"], "principal": i["principal_ref"]},
            process_outputs=_or_empty(_run_outputs),
        )(_answer)

    # --- identities ---

    def principal_ref(self, principal: Principal | None) -> str:
        """An opaque reference to the asker: never the person_id."""
        if principal is None:
            return "internal-only"
        digest = hmac.new(self._principal_key, principal.person_id.encode(), hashlib.sha256)
        return f"principal-{digest.hexdigest()[:12]}"

    @staticmethod
    def new_trace_id() -> str:
        """A Tessera trace id that is also a LangSmith run id (UUID v7)."""
        return ls.uuid7().hex

    # --- what a composition root builds the native pipeline from ---

    def llm(self, inner: LLMClient, role: str) -> LLMClient:
        return _TracedLLM(inner, role)

    def native_steps(
        self,
        retrieve: Callable[..., RetrievalResult] | None = None,
        *,
        route: Callable[..., RoutingDecision] | None = None,
        generate: Callable[..., GeneratedAnswer] | None = None,
        find_experts: Callable[..., ExpertiseResult] | None = None,
        generate_expertise: Callable[..., GeneratedAnswer] | None = None,
    ) -> dict[str, Callable[..., Any]]:
        """``NativePipeline`` step keyword arguments, each wrapped as a
        run. By default each step is the core function; a LangChain
        stack's switch passes its replacement (``retrieve``, ``route``,
        ``generate``, ...), whose own LangChain runs nest under the step's."""
        ref = self.principal_ref
        retrieve_step = retrieve or _native_retrieve
        route_step = route or (lambda query, llm: _core_route(query, llm))
        generate_step = generate or (lambda retrieval, llm: generate_answer(retrieval, llm))
        experts_step = find_experts or (
            lambda query, embedder, store, *, k: _core_find_experts(query, embedder, store, k=k)
        )
        expertise_step = generate_expertise or (lambda found, llm: generate_expertise_answer(found, llm))

        @traceable(
            run_type="chain",
            name="route",
            process_inputs=lambda i: {"query": i["query"]},
            process_outputs=_or_empty(_route_outputs),
        )
        def route_fn(query: str, llm: LLMClient) -> RoutingDecision:
            return route_step(query, llm)

        @traceable(
            run_type="retriever",
            name="retrieve",
            process_inputs=lambda i: {
                "query": i["query"],
                "archetype": i["archetype"].value,
                "principal": ref(i.get("principal")),
            },
            process_outputs=_or_empty(_retrieval_outputs),
        )
        def retrieve_fn(
            query: str, archetype: Any, embedder: Any, store: Any, *, principal: Principal | None = None
        ) -> RetrievalResult:
            result = retrieve_step(query, archetype, embedder, store, principal=principal)
            request = CURRENT_REQUEST.get()
            if request is not None:
                request.record_results(result.results)
            return result

        @traceable(
            run_type="chain",
            name="generate_answer",
            process_inputs=lambda i: {
                "chunks": [r.chunk_id for r in i["retrieval"].results],
            },
            process_outputs=_or_empty(_generated_outputs),
        )
        def generate_fn(retrieval: RetrievalResult, llm: LLMClient) -> GeneratedAnswer:
            return generate_step(retrieval, llm)

        @traceable(
            run_type="retriever",
            name="find_experts",
            process_inputs=lambda i: {"query": i["query"], "k": i.get("k")},
            process_outputs=_or_empty(_expertise_outputs),
        )
        def find_experts_fn(query: str, embedder: Any, store: Any, *, k: int) -> ExpertiseResult:
            return experts_step(query, embedder, store, k=k)

        @traceable(
            run_type="chain",
            name="generate_expertise_answer",
            process_inputs=lambda i: {
                "people": [m.person.person_id for m in i["found"].matches],
            },
            process_outputs=_or_empty(_generated_outputs),
        )
        def generate_expertise_fn(found: ExpertiseResult, llm: LLMClient) -> GeneratedAnswer:
            return expertise_step(found, llm)

        return {
            "route_fn": route_fn,
            "retrieve_fn": retrieve_fn,
            "generate_fn": generate_fn,
            "find_experts_fn": find_experts_fn,
            "generate_expertise_fn": generate_expertise_fn,
        }

    # --- one traced question ---

    def run(
        self, pipeline: Pipeline, query: str, principal: Principal | None, trace_id: str
    ) -> PipelineRun:
        """Answer ``query`` as one trace whose root run id is ``trace_id``,
        then send it, redacted. Exceptions propagate after the send."""
        request = RequestTaint(self._terms)
        if principal is not None:
            request.add_words(principal.person_id)
        ref = self.principal_ref(principal)
        token = CURRENT_REQUEST.set(request)
        try:
            with ls.tracing_context(
                enabled=True,
                client=self.client,
                project_name=self._project,
                metadata={**self._metadata, "principal": ref},
            ):
                return self._root(
                    pipeline,
                    query,
                    principal,
                    ref,
                    langsmith_extra={"run_id": trace_id, "metadata": {"tessera_trace_id": trace_id}},
                )
        finally:
            CURRENT_REQUEST.reset(token)
            self._redactor.close(trace_id, request)
            self._send(trace_id)

    def _send(self, trace_id: str) -> None:
        try:
            self.client.flush(timeout=self._flush_timeout)
            if not self._redactor.wait_processed(trace_id, self._flush_timeout):
                logger.warning("LangSmith trace %s wasn't processed in time", trace_id)
            # The first flush hands the buffer to a worker thread; this one
            # sends what it queued.
            self.client.flush(timeout=self._flush_timeout)
        except Exception:  # losing a trace must not lose the answer
            logger.exception("couldn't send LangSmith trace %s", trace_id)

    def pipeline(self, inner: Pipeline) -> Pipeline:
        """``inner`` as a ``Pipeline`` whose every run is a trace (for the
        eval harness, which calls ``run(query, principal)``)."""
        return _TracedPipeline(self, inner)


def _native_retrieve(
    query: str, archetype: Any, embedder: Any, store: Any, *, principal: Principal | None = None
) -> RetrievalResult:
    return retrieve(query, archetype, embedder, store, principal=principal)


def _answer(
    pipeline: Pipeline, query: str, principal: Principal | None, principal_ref: str
) -> PipelineRun:
    run = pipeline.run(query, principal)
    run_tree = ls.get_current_run_tree()
    if run_tree is not None:
        run_tree.metadata["archetype"] = run.answer.archetype.value
    return run


class _TracedPipeline:
    def __init__(self, tracer: LangSmithTracer, inner: Pipeline) -> None:
        self._tracer = tracer
        self._inner = inner

    def run(self, query: str, principal: Principal | None = None) -> PipelineRun:
        return self._tracer.run(self._inner, query, principal, self._tracer.new_trace_id())


def tracing_off() -> AbstractContextManager[Any]:
    """A context where nothing traces, whatever the environment says (for
    invocations with Tessera's tracing off; the LangChain stack runs inside
    it from P5-4)."""
    return ls.tracing_context(enabled=False)


def build_tracer(
    *,
    api_key: str,
    terms: TaintTerms,
    project: str,
    api_url: str = DEFAULT_ENDPOINT,
    stack: str = "native",
    switches: Mapping[str, str] | None = None,
    session: requests.Session | None = None,
    info: Mapping[str, Any] | None = None,
    flush_timeout: float = 30.0,
) -> LangSmithTracer:
    """The redacting client and a tracer over it. Every setting is passed
    in — nothing is read from the environment. ``session`` and ``info``
    exist for tests (a mocked transport; no ``/info`` call)."""
    redactor = RunRedactor(terms)
    client = Client(
        api_key=api_key,
        api_url=api_url,
        session=session,
        info=info,  # type: ignore[arg-type]
        anonymizer=redactor.anonymize,
        process_buffered_run_ops=redactor.process,
        run_ops_buffer_size=_NEVER_FLUSH_SIZE,
        run_ops_buffer_timeout_ms=_NEVER_FLUSH_MS,
        omit_traced_runtime_info=True,
    )
    return LangSmithTracer(
        client,
        redactor,
        terms,
        project=project,
        stack=stack,
        switches=switches,
        flush_timeout=flush_timeout,
    )
