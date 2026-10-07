"""The gated LangSmith redaction test (Phase 5 plan §3.9.4, P5-3).

Drives a **real** ``langsmith.Client`` — the redacting one ``build_tracer``
makes — with HTTP mocked by patching ``requests.Session.send`` (the client
mounts its own adapter on any session it is given, so mounting a mock
adapter would capture nothing: the P5-1 spike's false "0 leaks"). Every
access case runs through the traced native pipeline over a real index of
the real corpus, as the principal the case names. The answer model is a
deliberately leaky fake: it echoes the rendered prompt — restricted chunk
text included, for a cleared partner — into its answer, its run's
metadata, a streamed-token event, a custom event, a child run's name and
inputs, and (in the error scenarios) an exception.

The captured multipart payloads are decoded part by part and must hold
**zero** restricted chunk text, engagement markers, codenames, restricted
paths, quarantined text, person_ids or restricted-removal counts. A
control run through a plain client proves the capture sees leaks when
there is no redaction.

The test grows with the stack: P5-5, P5-6, P5-7 and P5-9 extend it to
their run types, the review scenarios and dataset uploads.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import pytest

ls = pytest.importorskip("langsmith", reason="the lc extra isn't installed")
requests = pytest.importorskip("requests")

from evals.harness import EvalCase, case_principal, load_cases  # noqa: E402
from tessera.embedding.local import LocalEmbedder  # noqa: E402
from tessera.generation.base import LLMClient  # noqa: E402
from tessera.generation.prompts import ROUTER_SYSTEM_PROMPT  # noqa: E402
from tessera.ingestion.access_loader import Walls, load_walls  # noqa: E402
from tessera.ingestion.chunker import chunk_corpus, chunk_document, chunk_embedding_text  # noqa: E402
from tessera.ingestion.expertise_loader import load_expertise  # noqa: E402
from tessera.ingestion.loader import Document, indexable, load_corpus  # noqa: E402
from tessera.observability.langsmith_tracing import (  # noqa: E402
    HIDDEN,
    LangSmithTracer,
    build_tracer,
)
from tessera.observability.taint import corpus_taint  # noqa: E402
from tessera.pipeline import NativePipeline  # noqa: E402
from tessera.principal import Principal  # noqa: E402
from tessera.store.base import SearchResult  # noqa: E402
from tessera.store.chroma import ChromaVectorStore  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "corpus"
NO_INFO = {"version": "0.0"}  # no /info call; uncompressed multipart

# --- the corpus, its real index, the access cases ---


@pytest.fixture(scope="module")
def documents() -> list[Document]:
    return load_corpus(CORPUS)


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder()


@pytest.fixture(scope="module")
def store(
    documents: list[Document], embedder: LocalEmbedder, tmp_path_factory: pytest.TempPathFactory
) -> ChromaVectorStore:
    chunks = chunk_corpus(indexable(documents))
    built = ChromaVectorStore(persist_dir=tmp_path_factory.mktemp("index"))
    built.add(chunks, embedder.embed_documents([chunk_embedding_text(c) for c in chunks]))
    return built


@pytest.fixture(scope="module")
def walls(documents: list[Document]) -> Walls:
    people = [p.person_id for p in load_expertise(REPO / "data" / "expertise" / "people")]
    engagements = {d.engagement for d in documents if d.engagement}
    return load_walls(REPO / "data" / "access" / "walls.yaml", person_ids=people, engagements=engagements)


@pytest.fixture(scope="module")
def access_cases() -> list[EvalCase]:
    cases = [c for c in load_cases(REPO / "evals" / "cases") if c.access is not None]
    assert len(cases) == 29, "the access set changed — re-check what this test covers"
    return cases


# --- what must never leave ---


@pytest.fixture(scope="module")
def forbidden(documents: list[Document], access_cases: list[EvalCase], walls: Walls) -> dict[str, Any]:
    """Written independently of taint.py: whole chunk texts, 30+ character
    fragments of restricted and quarantined bodies, markers, codenames,
    paths, and the case principals' person_ids."""
    secret_docs = [d for d in documents if d.is_restricted or d.is_quarantined]
    texts: set[str] = set()
    for doc in secret_docs:
        texts.update(c.text for c in chunk_document(doc))
        texts.update(
            f.strip(" -*#|>") for f in re.split(r"[.\n]", doc.body) if len(f.strip(" -*#|>")) >= 30
        )
        texts.update({doc.title, doc.path.stem, doc.path.relative_to(CORPUS).as_posix()})
    markers = {m for c in access_cases for ms in c.forbidden_markers.values() for m in ms}
    words = {d.engagement for d in documents if d.engagement}
    words |= {c.principal for c in access_cases if c.principal}
    words |= {"c0001", "c0002"}  # the principals of the extra scenarios below
    return {"texts": texts, "markers": markers, "words": words}


def leaks(captured: str, forbidden: dict[str, Any]) -> list[str]:
    low = captured.lower()
    found = [f"text:{t[:60]!r}" for t in forbidden["texts"] if t.lower() in low]
    found += [f"marker:{m}" for m in forbidden["markers"] if m.lower() in low]
    found += [
        f"word:{w}"
        for w in forbidden["words"]
        if re.search(rf"(?<![a-z0-9]){re.escape(w.lower())}(?![a-z0-9])", low)
    ]
    return sorted(found)


# --- the mocked transport and payload decoding ---


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    captured: list[Any] = []

    def send(self: Any, request: Any, **kwargs: Any) -> Any:
        captured.append(request)
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.request = request
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    return captured


def payload_parts(request: Any) -> Iterator[tuple[str, Any]]:
    """Each multipart part of one request as (name, decoded JSON)."""
    body = request.body if isinstance(request.body, bytes) else (request.body or "").encode()
    content_type = request.headers.get("Content-Type", "")
    if not content_type.startswith("multipart/"):
        yield ("body", json.loads(body) if body else None)
        return
    message = BytesParser().parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
    for part in message.walk():
        if part.is_multipart():
            continue
        name = part.get_param("name", header="content-disposition") or ""
        raw = part.get_payload(decode=True) or b""
        try:
            yield (name, json.loads(raw))
        except ValueError:
            yield (name, raw.decode("utf-8", "replace"))


def all_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from all_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from all_strings(v)
    elif value is not None:
        yield str(value)


def captured_text(sent: list[Any]) -> str:
    """Every string in every captured payload (decoded, so escaped
    newlines and unicode can't hide a match), plus the raw bodies."""
    pieces: list[str] = []
    for request in sent:
        for _, value in payload_parts(request):
            pieces.extend(all_strings(value))
        raw = request.body if isinstance(request.body, bytes) else (request.body or "").encode()
        pieces.append(raw.decode("utf-8", "replace"))
    return "\n".join(pieces)


def runs_by_trace(sent: list[Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """trace_id -> run id -> merged fields (post, patch and their parts)."""
    runs: dict[str, dict[str, Any]] = {}
    for request in sent:
        for name, value in payload_parts(request):
            pieces = name.split(".")
            if len(pieces) < 2 or pieces[0] not in ("post", "patch"):
                continue
            run = runs.setdefault(pieces[1], {})
            if len(pieces) == 2 and isinstance(value, dict):
                run.update(value)
            elif len(pieces) == 3:
                run[pieces[2]] = value
    by_trace: dict[str, dict[str, dict[str, Any]]] = {}
    for run_id, run in runs.items():
        by_trace.setdefault(uuid.UUID(str(run.get("trace_id"))).hex, {})[run_id] = run
    return by_trace


# --- the leaky fake model ---


class LeakyLLM(LLMClient):
    """Routes every question as ``archetype``; answers by echoing the
    rendered prompt into every field a run has."""

    def __init__(self, archetype: str = "A", *, fail: bool = False) -> None:
        self.archetype = archetype
        self.fail = fail

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        if system == ROUTER_SYSTEM_PROMPT:
            return json.dumps({"archetype": self.archetype, "reasoning": f"echo: {user}"})
        run = ls.get_current_run_tree()
        if run is not None:
            run.metadata["leak"] = user
            run.add_event({"name": "new_token", "kwargs": {"token": user}})
            run.add_event({"name": "custom", "message": user})
        with ls.trace(name=f"echo {user[:300]}", inputs={"prompt": user}) as child:
            child.end(outputs={"echo": user})
        if self.fail:
            raise RuntimeError(f"model failed on: {user}")
        return f"ECHO {user}"


def make_tracer(documents: list[Document], **kw: Any) -> LangSmithTracer:
    return build_tracer(
        api_key="test-key",
        terms=corpus_taint(documents, CORPUS),
        project="tessera-test",
        api_url="https://langsmith.invalid",
        info=NO_INFO,
        **kw,
    )


def traced_pipeline(tracer: LangSmithTracer, store: Any, embedder: Any, llm: LLMClient) -> NativePipeline:
    return NativePipeline(
        tracer.llm(llm, "answer"),
        embedder,
        store,
        router_llm=tracer.llm(llm, "router"),
        **tracer.native_steps(),
    )


def run_scenarios(
    tracer: LangSmithTracer,
    store: ChromaVectorStore,
    embedder: LocalEmbedder,
    access_cases: list[EvalCase],
    walls: Walls,
    documents: list[Document],
) -> dict[str, Any]:
    """Every access case, a D refusal naming codenames, failures in a
    tainted and an untainted trace, and a store that leaks quarantined
    chunks. Returns what the assertions need."""
    pipeline = traced_pipeline(tracer, store, embedder, LeakyLLM())
    traces: dict[str, Any] = {}
    for case in access_cases:
        principal = case_principal(case, walls)
        trace_id = tracer.new_trace_id()
        run = tracer.run(pipeline, case.query, principal, trace_id)
        traces[trace_id] = (case, principal, run)

    refusal = traced_pipeline(tracer, store, embedder, LeakyLLM("D"))
    tracer.run(refusal, "Compare Project Halcyon with Project Kestrel and Lantern", None, tracer.new_trace_id())

    failing = traced_pipeline(tracer, store, embedder, LeakyLLM(fail=True))
    halcyon_partner = Principal("c0001", frozenset({"halcyon"}))
    for principal in (halcyon_partner, None):
        with pytest.raises(RuntimeError):
            tracer.run(failing, "What did Project Halcyon find on grocer price tiers?", principal, tracer.new_trace_id())

    quarantined = [d for d in documents if d.is_quarantined]
    assert quarantined, "the corpus has no quarantined document to leak"
    leaked_chunks = [
        SearchResult(
            chunk_id=c.chunk_id, text=c.text, score=0.9, document_path=str(c.document_path),
            document_title=c.document_title, doc_type=c.doc_type, industry=c.industry,
            topics=c.topics, date=c.date.isoformat(), heading_path=c.heading_path,
            sensitivity=c.sensitivity,
        )
        for c in chunk_corpus(quarantined)
    ]

    class LeakingStore(ChromaVectorStore):
        """Returns quarantined chunks as if they had been indexed."""

        def __init__(self) -> None:
            pass

        def query(self, embedding: Any, k: int, where: Any = None) -> list[SearchResult]:
            return leaked_chunks[:k]

        def count(self) -> int:
            return len(leaked_chunks)

    leaky_store = traced_pipeline(tracer, LeakingStore(), embedder, LeakyLLM())
    quarantine_trace = tracer.new_trace_id()
    tracer.run(leaky_store, "How did we reset own-label price tiers?", Principal("c0002"), quarantine_trace)
    return {"traces": traces, "quarantine_trace": quarantine_trace}


# --- the gate ---


def test_redacting_client_sends_no_restricted_or_quarantined_content(
    sent: list[Any],
    documents: list[Document],
    store: ChromaVectorStore,
    embedder: LocalEmbedder,
    access_cases: list[EvalCase],
    walls: Walls,
    forbidden: dict[str, Any],
) -> None:
    tracer = make_tracer(documents)
    result = run_scenarios(tracer, store, embedder, access_cases, walls, documents)
    assert sent, "nothing was sent — the transport mock captured no request"

    text = captured_text(sent)
    assert leaks(text, forbidden) == []

    by_trace = runs_by_trace(sent)
    assert len(by_trace) == len(access_cases) + 4  # + refusal, 2 failures, quarantine

    # The restricted-removal count never leaves (Phase 4 plan §3.5.4).
    for runs in by_trace.values():
        for run in runs.values():
            assert not _has_removed_restricted(run)

    cleared_any = False
    walled_removed = False
    for trace_id, (case, principal, run) in result["traces"].items():
        runs = by_trace[trace_id]
        touched = any(r.sensitivity != "internal" for r in run.retrievals[0].results)
        root = runs[str(uuid.UUID(trace_id))]
        if touched:
            cleared_any = True
            for r in runs.values():
                for field in ("inputs", "outputs"):
                    if r.get(field):
                        assert r[field] == {"withheld": HIDDEN}, (case.id, r.get("name"), field)
                assert not r.get("events")
        else:
            # Untainted: the answer is visible (redacted, not hidden).
            assert root["outputs"] != {"withheld": HIDDEN}
            assert root["outputs"]["archetype"] == "A"
            if run.answer.trace.removed.get("restricted"):
                walled_removed = True
        # Metadata: the stack, the archetype, an opaque principal — never
        # the person_id — and Tessera's trace id as the root run id.
        metadata = root["extra"]["metadata"]
        assert metadata["stack"] == "native"
        assert metadata["archetype"] == "A"
        assert metadata["tessera_trace_id"] == trace_id
        assert metadata["principal"].startswith("principal-") or principal is None
    assert cleared_any, "no access case retrieved restricted content — the hiding path went untested"
    assert walled_removed, "no walled case had restricted chunks removed — the count check went untested"

    quarantine_runs = by_trace[result["quarantine_trace"]]
    assert all(
        r.get("outputs") in (None, {}, {"withheld": HIDDEN}) for r in quarantine_runs.values()
    )


def test_a_plain_client_would_leak_through_the_same_capture(
    sent: list[Any],
    documents: list[Document],
    store: ChromaVectorStore,
    embedder: LocalEmbedder,
    access_cases: list[EvalCase],
    walls: Walls,
    forbidden: dict[str, Any],
) -> None:
    """The control: the same runs through a client with no hooks leak, so
    the gate's zero is a measurement, not a blind capture."""
    from langsmith import Client

    tracer = make_tracer(documents)
    tracer.client = Client(api_key="test-key", api_url="https://langsmith.invalid", info=NO_INFO)
    pipeline = traced_pipeline(tracer, store, embedder, LeakyLLM())
    cleared = next(c for c in access_cases if c.access == "authorized")
    for case in (cleared, access_cases[0]):
        with ls.tracing_context(enabled=True, client=tracer.client, project_name="control"):
            pipeline.run(case.query, case_principal(case, walls))
    tracer.client.flush()
    found = leaks(captured_text(sent), forbidden)
    assert any(f.startswith("text:") for f in found)
    assert any(f.startswith("marker:") for f in found)
    assert any(f.startswith("word:") for f in found)


def _has_removed_restricted(value: Any) -> bool:
    if isinstance(value, dict):
        removed = value.get("removed")
        if isinstance(removed, dict) and "restricted" in removed:
            return True
        return any(_has_removed_restricted(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_removed_restricted(v) for v in value)
    return False



# --- P5-5: the LangChain retrievers' own runs ---

# The run each retriever kind must produce through LangChain's callbacks —
# proof the runs exist and went through the redacting client.
LC_RETRIEVER_RUNS = {
    "vector": {"ArchetypeRetriever", "ScopedVectorRetriever"},
    "bm25": {"ArchetypeRetriever", "BM25Retriever"},
    "hybrid": {"ArchetypeRetriever", "EnsembleRetriever", "BM25Retriever", "ScopedVectorRetriever"},
    "multiquery": {"ArchetypeRetriever", "MultiQueryRetriever", "ScopedVectorRetriever"},
    "rerank": {"ArchetypeRetriever", "ContextualCompressionRetriever", "ScopedVectorRetriever"},
    "parent_doc": {"ArchetypeRetriever", "ParentDocumentRetriever"},
}


@pytest.fixture(scope="module")
def lc_store(documents: list[Document], embedder: LocalEmbedder, tmp_path_factory: pytest.TempPathFactory) -> Any:
    from tessera.ingestion.indexing import index_corpus
    from tessera.integrations.langchain import lc_chroma
    from tessera.lc.embeddings import as_langchain
    from tessera.lc.store import LangChainChromaStore

    built = LangChainChromaStore(lc_chroma(tmp_path_factory.mktemp("lc-index"), as_langchain(embedder)))
    index_corpus(documents, embedder, built)
    return built


@pytest.fixture(scope="module")
def retriever_parts(documents: list[Document]) -> dict[str, Any]:
    from tessera.integrations.langchain import hf_cross_encoder
    from tessera.lc.retrievers import BM25Cache, parent_docstore

    return {"bm25": BM25Cache(), "cross_encoder": hf_cross_encoder(), "parents": parent_docstore(documents)}


@pytest.mark.parametrize("kind", list(LC_RETRIEVER_RUNS))
def test_langchain_retriever_runs_are_redacted_too(
    kind: str,
    sent: list[Any],
    documents: list[Document],
    lc_store: Any,
    embedder: LocalEmbedder,
    access_cases: list[EvalCase],
    walls: Walls,
    forbidden: dict[str, Any],
    retriever_parts: dict[str, Any],
) -> None:
    """Every access case through the native pipeline with a LangChain
    retriever swapped in (the ``retriever`` switch): LangChain's callback
    runs for the retriever and its sub-retrievers nest under Tessera's
    traced retrieve step, go through the same redacting client, and send
    nothing restricted. Multi-query's rephrasing model is the leaky echo."""
    from tessera.lc.retrievers import lc_retrieve_fn

    tracer = make_tracer(documents)
    llm = LeakyLLM()
    retrieve = lc_retrieve_fn(lc_store, kind, llm=tracer.llm(llm, "rephrase"), **retriever_parts)
    pipeline = NativePipeline(
        tracer.llm(llm, "answer"),
        embedder,
        lc_store,
        router_llm=tracer.llm(llm, "router"),
        **tracer.native_steps(retrieve=retrieve),
    )
    for case in access_cases:
        tracer.run(pipeline, case.query, case_principal(case, walls), tracer.new_trace_id())

    assert leaks(captured_text(sent), forbidden) == []
    names = {run.get("name") for runs in runs_by_trace(sent).values() for run in runs.values()}
    assert LC_RETRIEVER_RUNS[kind] <= names, f"missing LangChain runs: {LC_RETRIEVER_RUNS[kind] - names}"
