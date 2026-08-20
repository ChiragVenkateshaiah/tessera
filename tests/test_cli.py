"""Unit tests for the CLI wiring (typer commands) — every external
constructor (LocalEmbedder, ChromaVectorStore, GeminiClient) and I/O call
(load_corpus, answer_query) is monkeypatched with a fake, so these tests
run with no model download, no disk index, and no network/LLM call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tessera import cli
from tessera.pipeline import AnswerResult
from tessera.retrieval.router import Archetype

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No .env in the sandbox and a valid GEMINI_API_KEY by default, so
    every test starts from a working config unless it deliberately
    unsets it.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_missing_config_reports_actionable_error_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 1
    assert "Missing or invalid configuration" in result.output


def test_ingest_wires_loader_chunker_embedder_and_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    fake_docs = ["doc1", "doc2"]
    fake_chunks = [type("C", (), {"text": "a"})(), type("C", (), {"text": "b"})()]

    monkeypatch.setattr(
        cli,
        "load_corpus",
        lambda corpus_dir: (calls.append("load_corpus"), fake_docs)[1],
    )
    monkeypatch.setattr(
        cli,
        "chunk_corpus",
        lambda docs: (calls.append("chunk_corpus"), fake_chunks)[1],
    )

    class FakeEmbedder:
        def __init__(self) -> None:
            calls.append("LocalEmbedder")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append("embed_documents")
            return [[0.0] for _ in texts]

    class FakeStore:
        def __init__(self, persist_dir: Path) -> None:
            calls.append("ChromaVectorStore")
            self._count = 0

        def add(self, chunks: object, embeddings: object) -> None:
            calls.append("add")
            self._count = len(fake_chunks)

        def count(self) -> int:
            return self._count

    monkeypatch.setattr(cli, "LocalEmbedder", FakeEmbedder)
    monkeypatch.setattr(cli, "ChromaVectorStore", FakeStore)

    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 0
    assert calls == [
        "load_corpus",
        "chunk_corpus",
        "LocalEmbedder",
        "embed_documents",
        "ChromaVectorStore",
        "add",
    ]
    assert "Indexed 2 chunks" in result.output


def test_query_requires_an_existing_index(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyStore:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 0

    monkeypatch.setattr(cli, "ChromaVectorStore", EmptyStore)

    result = runner.invoke(cli.app, ["query", "anything"])

    assert result.exit_code == 1
    assert "No index found" in result.output


def test_query_prints_answer_and_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    class NonEmptyStore:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 1

    monkeypatch.setattr(cli, "ChromaVectorStore", NonEmptyStore)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())
    monkeypatch.setattr(cli, "GeminiClient", lambda api_key, model: object())

    from tessera.generation.answer import Citation

    canned = AnswerResult(
        query="what's our framework?",
        archetype=Archetype.LOOKUP,
        answer="We have it, see [1].",
        citations=[
            Citation(
                marker=1,
                document_path="methodology/x.md",
                document_title="X",
                heading_path=("Overview",),
            )
        ],
    )
    monkeypatch.setattr(cli, "answer_query", lambda *a, **kw: canned)

    result = runner.invoke(cli.app, ["query", "what's our framework?"])

    assert result.exit_code == 0
    assert "[A] We have it, see [1]." in result.output
    assert "[1] X — Overview (methodology/x.md)" in result.output


def test_eval_requires_an_existing_index(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyStore:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 0

    monkeypatch.setattr(cli, "ChromaVectorStore", EmptyStore)

    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 1
    assert "No index found" in result.output


def test_eval_resolves_and_drives_the_evals_harness_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """eval_command() imports evals.harness lazily via a sys.path fixup
    (see cli.py's REPO_ROOT comment) since evals/ lives outside the
    installed package. This stubs that module in sys.modules to verify
    the command wires load_cases -> run_harness -> format_report without
    needing the real harness or a live LLM.
    """
    calls: list[str] = []

    class NonEmptyStore:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 1

    monkeypatch.setattr(cli, "ChromaVectorStore", NonEmptyStore)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())
    monkeypatch.setattr(cli, "GeminiClient", lambda api_key, model: object())

    fake_harness = type(sys)("evals.harness")
    fake_harness.load_cases = lambda cases_dir: (calls.append("load_cases"), [])[1]
    fake_harness.run_harness = lambda cases, llm, embedder, store, corpus_dir: (
        calls.append("run_harness"),
        "report-object",
    )[1]
    fake_harness.format_report = lambda report: (calls.append("format_report"), "REPORT TEXT")[1]

    fake_evals_pkg = type(sys)("evals")
    fake_evals_pkg.harness = fake_harness
    monkeypatch.setitem(sys.modules, "evals", fake_evals_pkg)
    monkeypatch.setitem(sys.modules, "evals.harness", fake_harness)

    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 0
    assert calls == ["load_cases", "run_harness", "format_report"]
    assert "REPORT TEXT" in result.output
