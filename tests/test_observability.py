"""Taint-based redaction (Phase 5 plan §3.9.3) and the tracing wiring in
the composition roots (§3.9.2), on small fixed inputs. The end-to-end gate
over the real corpus is tests/test_langsmith_redaction.py.
"""

from __future__ import annotations

import json
import sys
import types
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tessera import cli
from tessera.ingestion.access_loader import Walls
from tessera.ingestion.loader import Document
from tessera.observability.taint import (
    SHINGLE_WORDS,
    Placeholders,
    Redactor,
    RequestTaint,
    TaintTerms,
    corpus_taint,
)
from tessera.pipeline import AnswerResult, PipelineRun
from tessera.retrieval.router import Archetype, RoutingDecision
from tessera.store.base import SearchResult

CORPUS = Path("/corpus")
SECRET_BODY = (
    "## Findings\n\n"
    "- Re-opening the value tier gap recovered an estimated £340m of margin at flat volume.\n"
    "- Convenience stores were over-discounted against their competitive set.\n"
)


def _doc(rel: str, body: str, **kw: Any) -> Document:
    return Document(
        path=CORPUS / rel,
        title=kw.pop("title", f"Title of {rel}"),
        doc_type=kw.pop("doc_type", "methodology"),
        industry="retail",
        topics=["pricing-strategy"],
        date=date(2025, 1, 1),
        body=body,
        **kw,
    )


@pytest.fixture
def documents() -> list[Document]:
    return [
        _doc(
            "engagements/halcyon-grocer-price-architecture.md",
            SECRET_BODY,
            title="Engagement Summary: Project Halcyon",
            doc_type="engagement",
            sensitivity="restricted",
            engagement="halcyon",
        ),
        _doc(
            "case_studies/grocer-price-tier-reset.md",
            "## Situation\n\nA leading UK grocer with around 2,800 stores saw its value tier drift.\n",
            doc_type="case_study",
            sensitivity="internal",
            review_status="pending",
        ),
        _doc(
            "methodology/pricing.md",
            "## Overview\n\nThe good/better/best ladder sets tier gaps by elasticity.\n",
            sensitivity="internal",
        ),
    ]


def _result(path: str, text: str, sensitivity: str = "internal", engagement: str | None = None) -> SearchResult:
    return SearchResult(
        chunk_id=f"{Path(path).stem}::0", text=text, score=0.8, document_path=path,
        document_title="T", doc_type="methodology", industry="retail", topics=["t"],
        date="2025-01-01", heading_path=("Overview",), sensitivity=sensitivity, engagement=engagement,
    )


# --- the taint set ---


def test_corpus_taint_holds_restricted_and_quarantined_content_only(documents: list[Document]) -> None:
    terms = corpus_taint(documents, CORPUS)

    assert "halcyon" in terms.words
    assert "halcyon-grocer-price-architecture" in terms.texts
    assert "engagements/halcyon-grocer-price-architecture.md" in terms.texts
    assert "Engagement Summary: Project Halcyon" in terms.texts
    assert any("£340m" in t for t in terms.texts)
    assert any("2,800 stores" in t for t in terms.texts)  # quarantined, though internal
    assert not any("good/better/best" in t for t in terms.texts)
    assert terms.tainted_paths == {
        str(CORPUS / "engagements/halcyon-grocer-price-architecture.md"),
        str(CORPUS / "case_studies/grocer-price-tier-reset.md"),
    }


def test_retrieving_restricted_or_quarantined_chunks_taints_a_request(documents: list[Document]) -> None:
    terms = corpus_taint(documents, CORPUS)
    internal = _result(str(CORPUS / "methodology/pricing.md"), "ladder")
    quarantined = _result(str(CORPUS / "case_studies/grocer-price-tier-reset.md"), "2,800 stores")
    restricted_new = _result("/elsewhere/new.md", "unseen restricted text", "restricted", "osprey")

    request = RequestTaint(terms)
    request.record_results([internal])
    assert not request.touched

    for tainted in (quarantined, restricted_new):
        request = RequestTaint(terms)
        request.record_results([internal, tainted])
        assert request.touched
    # A restricted chunk the corpus set didn't know joins the request's terms.
    assert "unseen restricted text" not in request.redactor().text(
        "says: unseen restricted text", request.placeholders
    )
    assert "osprey" not in request.redactor().text("Project Osprey", request.placeholders).lower()


# --- the redactor ---


def test_exact_terms_and_whole_words_are_replaced_case_insensitively() -> None:
    redactor = Redactor(TaintTerms(texts=frozenset({"£340m of margin"}), words=frozenset({"aurora", "c0014"})))
    ph = Placeholders()

    out = redactor.text("AURORA saved £340M of margin; auroral lights; asked by c0014 not c00145", ph)

    assert out == "[withheld-1] saved [withheld-2]; auroral lights; asked by [withheld-3] not c00145"


def test_the_same_secret_gets_the_same_placeholder_within_a_request() -> None:
    redactor = Redactor(TaintTerms(words=frozenset({"halcyon", "kestrel"})))
    ph = Placeholders()
    assert redactor.text("Halcyon, kestrel, HALCYON", ph) == "[withheld-1], [withheld-2], [withheld-1]"
    assert redactor.text("halcyon", Placeholders()) == "[withheld-1]"


def test_fragments_of_tainted_text_are_caught_however_they_are_cut() -> None:
    secret = "Re-opening the value tier gap recovered an estimated £340m of margin at flat volume."
    redactor = Redactor(TaintTerms(texts=frozenset({secret})))
    ph = Placeholders()

    truncated = "echo: the value tier gap recovered an estimated £34"
    quoted = "As noted, gap recovered an estimated £340m of margin — remarkable."

    assert "recovered an estimated" not in redactor.text(truncated, ph)
    assert "recovered an estimated" not in redactor.text(quoted, ph)
    # Fewer shared words than a fragment is left alone.
    short = "the value tier gap"
    assert len(short.split()) < SHINGLE_WORDS
    assert redactor.text(short, ph) == short


def test_redaction_reaches_every_string_including_keys() -> None:
    redactor = Redactor(TaintTerms(words=frozenset({"halcyon"})))
    value = {"halcyon": ["Halcyon", {"x": ("halcyon",)}], "n": 3, "ok": None}

    out = redactor.value(value, Placeholders())

    assert "halcyon" not in json.dumps(out).lower()
    assert out["n"] == 3 and out["ok"] is None


# --- the run redactor (needs langsmith) ---


def _run(trace_id: uuid.UUID, **fields: Any) -> dict[str, Any]:
    return {"id": str(uuid.uuid4()), "trace_id": str(trace_id), "dotted_order": "1", "name": "step", **fields}


def test_run_redactor_hides_tainted_traces_and_redacts_the_rest() -> None:
    pytest.importorskip("langsmith")
    from tessera.observability.langsmith_tracing import HIDDEN, HIDDEN_ERROR, RunRedactor

    terms = TaintTerms(words=frozenset({"halcyon"}))
    redactor = RunRedactor(terms)
    clean, tainted, unknown = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    clean_request, tainted_request = RequestTaint(terms), RequestTaint(terms)
    tainted_request.touched = True
    redactor.close(clean.hex, clean_request)
    redactor.close(tainted.hex, tainted_request)
    fields = {
        "inputs": {"q": "Project Halcyon"},
        "outputs": {"answer": "about Halcyon", "removed": {"restricted": 4, "superseded": 1}},
        "events": [{"name": "new_token", "kwargs": {"token": "Halcyon"}}],
        "error": "RuntimeError('Halcyon failed')",
        "extra": {"metadata": {"note": "halcyon"}},
        "attachments": {"a": ("text/plain", b"halcyon")},
        "start_time": "2026-10-07T00:00:00",
    }

    out = redactor.process([_run(clean, **fields), _run(tainted, **fields), _run(unknown, **fields)])

    visible, hidden, failed_closed = out
    assert visible["inputs"] == {"q": "Project [withheld-1]"}
    assert visible["outputs"] == {"answer": "about [withheld-1]", "removed": {"superseded": 1}}
    assert visible["events"] == [{"name": "new_token", "kwargs": {"token": "[withheld-1]"}}]
    assert visible["start_time"] == fields["start_time"] and visible["id"] == out[0]["id"]
    for run in (hidden, failed_closed):
        assert run["inputs"] == {"withheld": HIDDEN} and run["outputs"] == {"withheld": HIDDEN}
        assert run["events"] == []
        assert run["error"] == f"RuntimeError: {HIDDEN_ERROR}"
        assert "halcyon" not in json.dumps(run["extra"]).lower()
    assert all(run["attachments"] == {} for run in out)
    assert redactor.wait_processed(clean.hex, timeout=0)


def test_the_tracer_gives_opaque_principals_and_uuid7_trace_ids() -> None:
    pytest.importorskip("langsmith")
    from tessera.observability.langsmith_tracing import build_tracer
    from tessera.principal import Principal

    tracer = build_tracer(api_key="k", terms=TaintTerms(), project="p", info={"version": "0"})
    ref = tracer.principal_ref(Principal("c0014"))

    assert ref.startswith("principal-") and "c0014" not in ref
    assert ref == tracer.principal_ref(Principal("c0014"))
    assert ref != tracer.principal_ref(Principal("c0015"))
    assert tracer.principal_ref(None) == "internal-only"
    assert uuid.UUID(tracer.new_trace_id()).version == 7


# --- the composition roots ---

runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(cli, "_load_walls", lambda settings: Walls(cleared={}))

    class Store:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 1

    monkeypatch.setattr(cli, "ChromaVectorStore", Store)
    monkeypatch.setattr(cli, "ChromaExpertiseStore", Store)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())
    monkeypatch.setattr(cli, "NvidiaClient", lambda api_key, model, **kw: object())
    monkeypatch.setattr(cli, "load_corpus", lambda corpus_dir: [])


class FakeTracer:
    def __init__(self) -> None:
        self.runs: list[tuple[str, str]] = []

    def new_trace_id(self) -> str:
        return "0" * 31 + "7"

    def llm(self, inner: object, role: str) -> object:
        return inner

    def native_steps(self, retrieve: object = None) -> dict[str, object]:
        return {}

    def run(self, pipeline: object, query: str, principal: object, trace_id: str) -> PipelineRun:
        self.runs.append((query, trace_id))
        answer = AnswerResult(query=query, archetype=Archetype.LOOKUP, answer="traced answer", citations=[])
        return PipelineRun(answer=answer, decision=RoutingDecision(query, Archetype.LOOKUP, "r"))


def test_query_traces_through_the_tracer_under_the_jsonl_trace_id(
    cli_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracer = FakeTracer()
    built: dict[str, Any] = {}

    def build_tracer(**kw: Any) -> FakeTracer:
        built.update(kw)
        return tracer

    fake_module = types.ModuleType("tessera.observability.langsmith_tracing")
    fake_module.build_tracer = build_tracer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tessera.observability.langsmith_tracing", fake_module)
    monkeypatch.setenv("TESSERA_LANGSMITH_TRACING", "true")
    monkeypatch.setenv("TESSERA_LANGSMITH_API_KEY", "lsv2-test")
    monkeypatch.setenv("TESSERA_LANGSMITH_PROJECT", "tessera-dev")

    result = runner.invoke(cli.app, ["query", "do we have a pricing framework?"])

    assert result.exit_code == 0, result.output
    assert "traced answer" in result.output
    assert tracer.runs == [("do we have a pricing framework?", "0" * 31 + "7")]
    assert built["api_key"] == "lsv2-test" and built["project"] == "tessera-dev"
    assert built["api_url"] == "https://api.smith.langchain.com"
    record = json.loads(Path("data/traces/traces.jsonl").read_text().splitlines()[-1])
    assert record["trace_id"] == "0" * 31 + "7"


def test_tracing_on_without_a_key_or_without_langsmith_stops_with_instructions(
    cli_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TESSERA_LANGSMITH_TRACING", "true")
    monkeypatch.delenv("TESSERA_LANGSMITH_API_KEY", raising=False)
    result = runner.invoke(cli.app, ["query", "x"])
    assert result.exit_code == 1
    assert "TESSERA_LANGSMITH_API_KEY is empty" in result.output

    monkeypatch.setenv("TESSERA_LANGSMITH_API_KEY", "lsv2-test")
    monkeypatch.setitem(sys.modules, "tessera.observability.langsmith_tracing", None)
    result = runner.invoke(cli.app, ["query", "x"])
    assert result.exit_code == 1
    assert "uv sync --extra dev --extra lc" in result.output


def test_tracing_off_by_default_never_builds_a_tracer(
    cli_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TESSERA_LANGSMITH_TRACING", raising=False)
    monkeypatch.setitem(sys.modules, "tessera.observability.langsmith_tracing", None)
    canned = AnswerResult(query="x", archetype=Archetype.LOOKUP, answer="native answer", citations=[])
    monkeypatch.setattr(cli, "answer_query", lambda *a, **kw: canned)

    result = runner.invoke(cli.app, ["query", "x"])

    assert result.exit_code == 0, result.output
    assert "native answer" in result.output


def test_the_api_answers_through_the_given_answerer_under_the_response_trace_id() -> None:
    from fastapi.testclient import TestClient

    from tessera.api import create_app

    seen: list[tuple[str, str]] = []

    def answer(question: str, principal: object, trace_id: str) -> AnswerResult:
        seen.append((question, trace_id))
        return AnswerResult(query=question, archetype=Archetype.LOOKUP, answer="traced", citations=[])

    class NoStore:
        def count(self) -> int:
            return 0

    app = create_app(
        object(), object(), NoStore(), None,  # type: ignore[arg-type]
        llm_name="fake", answer=answer, new_trace_id=lambda: "trace-1",
    )
    body = TestClient(app).post("/api/ask", json={"question": "framework?"}).json()

    assert body["answer"] == "traced" and body["trace_id"] == "trace-1"
    assert seen == [("framework?", "trace-1")]
