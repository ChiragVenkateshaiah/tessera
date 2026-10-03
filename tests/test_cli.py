"""Unit tests for the CLI wiring (typer commands) — every external
constructor (LocalEmbedder, ChromaVectorStore, NvidiaClient) and I/O call
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
    """No .env in the sandbox and a valid NVIDIA_API_KEY by default, so
    every test starts from a working config unless it deliberately
    unsets it.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")


def test_missing_config_reports_actionable_error_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 1
    assert "Missing or invalid configuration" in result.output


def test_ingest_wires_loader_chunker_embedder_and_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    fake_docs = [type("D", (), {"is_quarantined": False})() for _ in range(2)]
    fake_chunk_attrs = {"text": "a", "document_title": "Doc", "heading_path": ("H",)}
    fake_chunks = [
        type("C", (), fake_chunk_attrs)(),
        type("C", (), {**fake_chunk_attrs, "text": "b"})(),
    ]

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
    monkeypatch.setattr(cli, "NvidiaClient", lambda api_key, model, **kw: object())

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
    monkeypatch.setattr(cli, "ChromaExpertiseStore", NonEmptyStore)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())
    monkeypatch.setattr(cli, "NvidiaClient", lambda api_key, model, **kw: object())

    fake_harness = type(sys)("evals.harness")
    fake_harness.load_cases = lambda cases_dir: (calls.append("load_cases"), [])[1]

    def fake_run_harness(
        cases, llm, embedder, store, corpus_dir, expertise_store=None, on_case_complete=None, **kw
    ):
        calls.append("run_harness")
        assert expertise_store is not None  # B must be scored by `tessera eval`
        return "report-object"

    fake_harness.run_harness = fake_run_harness
    fake_harness.format_report = lambda report: (calls.append("format_report"), "REPORT TEXT")[1]

    fake_evals_pkg = type(sys)("evals")
    fake_evals_pkg.harness = fake_harness
    monkeypatch.setitem(sys.modules, "evals", fake_evals_pkg)
    monkeypatch.setitem(sys.modules, "evals.harness", fake_harness)

    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 0
    assert calls == ["load_cases", "run_harness", "format_report"]
    assert "REPORT TEXT" in result.output


def _stub_harness_for_check(
    monkeypatch: pytest.MonkeyPatch, bar_passed: bool
) -> None:
    class NonEmptyStore:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 1

    monkeypatch.setattr(cli, "ChromaVectorStore", NonEmptyStore)
    monkeypatch.setattr(cli, "ChromaExpertiseStore", NonEmptyStore)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())
    monkeypatch.setattr(cli, "NvidiaClient", lambda api_key, model, **kw: object())

    bar_result = type(
        "BarResult",
        (),
        {"passed": bar_passed, "gated_failures": [type("T", (), {"name": "Mean recall@k (A/C)"})()]},
    )()

    fake_harness = type(sys)("evals.harness")
    fake_harness.load_cases = lambda cases_dir: []
    fake_harness.run_harness = lambda *a, **k: "report-object"
    fake_harness.format_report = lambda report: "REPORT TEXT"
    fake_harness.evaluate_bar = lambda report: bar_result

    fake_evals_pkg = type(sys)("evals")
    fake_evals_pkg.harness = fake_harness
    monkeypatch.setitem(sys.modules, "evals", fake_evals_pkg)
    monkeypatch.setitem(sys.modules, "evals.harness", fake_harness)


def test_eval_check_flag_exits_zero_when_bar_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_harness_for_check(monkeypatch, bar_passed=True)

    result = runner.invoke(cli.app, ["eval", "--check"])

    assert result.exit_code == 0
    assert "REPORT TEXT" in result.output


def test_eval_check_flag_exits_nonzero_when_bar_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_harness_for_check(monkeypatch, bar_passed=False)

    result = runner.invoke(cli.app, ["eval", "--check"])

    assert result.exit_code == 1
    assert "REPORT TEXT" in result.output
    assert "Quality bar FAILED" in result.output


def test_eval_without_check_flag_ignores_bar_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_harness_for_check(monkeypatch, bar_passed=False)

    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 0


def test_index_people_wires_loader_summary_embedder_and_expertise_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_people = ["p1", "p2", "p3"]

    monkeypatch.setattr(
        cli,
        "load_expertise",
        lambda people_dir, corpus_dir=None: (calls.append("load_expertise"), fake_people)[1],
    )
    monkeypatch.setattr(cli, "profile_summary_text", lambda p: f"summary {p}")

    class FakeEmbedder:
        def __init__(self) -> None:
            calls.append("LocalEmbedder")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append("embed_documents")
            assert texts == ["summary p1", "summary p2", "summary p3"]
            return [[0.0] for _ in texts]

    class FakeStore:
        def __init__(self, persist_dir: Path) -> None:
            calls.append("ChromaExpertiseStore")

        def add(self, people: object, embeddings: object) -> None:
            calls.append("add")

        def count(self) -> int:
            return len(fake_people)

    monkeypatch.setattr(cli, "LocalEmbedder", FakeEmbedder)
    monkeypatch.setattr(cli, "ChromaExpertiseStore", FakeStore)

    result = runner.invoke(cli.app, ["index-people"])

    assert result.exit_code == 0
    assert calls == [
        "load_expertise",
        "LocalEmbedder",
        "embed_documents",
        "ChromaExpertiseStore",
        "add",
    ]
    assert "Indexed 3 people" in result.output


def _patch_query_deps(monkeypatch: pytest.MonkeyPatch, expertise_count: int, seen: dict) -> None:
    class NonEmptyStore:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 1

    class FakeExpertiseStore:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return expertise_count

    monkeypatch.setattr(cli, "ChromaVectorStore", NonEmptyStore)
    monkeypatch.setattr(cli, "ChromaExpertiseStore", FakeExpertiseStore)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())
    monkeypatch.setattr(cli, "NvidiaClient", lambda api_key, model, **kw: object())

    def fake_answer(text, llm, embedder, store, expertise_store=None, **kw):
        seen["expertise_store"] = expertise_store
        return seen["canned"]

    monkeypatch.setattr(cli, "answer_query", fake_answer)


def test_query_prints_people_for_an_expertise_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_expertise_generation import match

    seen: dict = {
        "canned": AnswerResult(
            query="who knows pricing?",
            archetype=Archetype.EXPERTISE,
            answer="Ask Person a [1].",
            citations=[],
            experts=[match("a", 2.0), match("b", 0.1, self_reported=True)],
        )
    }
    _patch_query_deps(monkeypatch, expertise_count=600, seen=seen)

    result = runner.invoke(cli.app, ["query", "who knows pricing?"])

    assert result.exit_code == 0
    assert "[B] Ask Person a [1]." in result.output
    assert "[1] Person a — Consultant, pricing, London" in result.output
    assert "[2] Person b" in result.output and "[self-reported only]" in result.output
    assert seen["expertise_store"] is not None


def test_query_passes_no_expertise_store_when_people_index_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {
        "canned": AnswerResult(
            query="q", archetype=Archetype.LOOKUP, answer="ok", citations=[]
        )
    }
    _patch_query_deps(monkeypatch, expertise_count=0, seen=seen)

    result = runner.invoke(cli.app, ["query", "q"])

    assert result.exit_code == 0
    assert seen["expertise_store"] is None


def test_eval_requires_a_people_index(monkeypatch: pytest.MonkeyPatch) -> None:
    class Store:
        def __init__(self, persist_dir: Path, empty: bool = False) -> None:
            pass

        def count(self) -> int:
            return 1

    class EmptyPeople(Store):
        def count(self) -> int:
            return 0

    monkeypatch.setattr(cli, "ChromaVectorStore", Store)
    monkeypatch.setattr(cli, "ChromaExpertiseStore", EmptyPeople)
    fake_harness = type(sys)("evals.harness")
    fake_harness.load_cases = lambda d: []
    fake_harness.run_harness = lambda *a, **k: pytest.fail("must not run without people")
    fake_harness.format_report = lambda r: ""
    fake_evals_pkg = type(sys)("evals")
    fake_evals_pkg.harness = fake_harness
    monkeypatch.setitem(sys.modules, "evals", fake_evals_pkg)
    monkeypatch.setitem(sys.modules, "evals.harness", fake_harness)

    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 1
    assert "index-people" in result.output


def test_eval_prints_progress_and_wraps_the_llm_with_retry_and_pacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tessera.generation.resilient import RetryingLLMClient

    captured: dict = {}

    class Store:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 1

    monkeypatch.setattr(cli, "ChromaVectorStore", Store)
    monkeypatch.setattr(cli, "ChromaExpertiseStore", Store)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())

    def fake_nvidia(api_key, model, **kw):
        captured["sdk_max_retries"] = kw.get("sdk_max_retries")
        return object()

    monkeypatch.setattr(cli, "NvidiaClient", fake_nvidia)

    fake_harness = type(sys)("evals.harness")
    fake_harness.load_cases = lambda d: []

    def fake_run_harness(cases, llm, embedder, store, corpus_dir, **kw):
        captured["llm"] = llm
        cb = kw["on_case_complete"]
        cb(1, 2, type("R", (), {"case_id": "q1", "error": None})())
        cb(2, 2, type("R", (), {"case_id": "q2", "error": "boom"})())
        return "report"

    fake_harness.run_harness = fake_run_harness
    fake_harness.format_report = lambda r: "REPORT TEXT"
    fake_evals_pkg = type(sys)("evals")
    fake_evals_pkg.harness = fake_harness
    monkeypatch.setitem(sys.modules, "evals", fake_evals_pkg)
    monkeypatch.setitem(sys.modules, "evals.harness", fake_harness)

    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 0
    assert "[1/2] q1 ok" in result.output and "[2/2] q2 ERROR" in result.output
    assert isinstance(captured["llm"], RetryingLLMClient)
    assert captured["llm"]._min_interval == cli.EVAL_MIN_CALL_INTERVAL_SECONDS
    assert captured["sdk_max_retries"] == 0  # SDK fast retries off; ours take over


def _patch_chat_deps(monkeypatch: pytest.MonkeyPatch, answer) -> list[str]:
    """Fake stores/embedder/LLM; ``answer(text)`` stands in for answer_query.
    Returns the list of questions the pipeline actually received.
    """

    class NonEmptyStore:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 3

    asked: list[str] = []

    def fake_answer(text, llm, embedder, store, expertise_store=None, **kw):
        asked.append(text)
        return answer(text)

    monkeypatch.setattr(cli, "ChromaVectorStore", NonEmptyStore)
    monkeypatch.setattr(cli, "ChromaExpertiseStore", NonEmptyStore)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())
    monkeypatch.setattr(cli, "NvidiaClient", lambda api_key, model, **kw: object())
    monkeypatch.setattr(cli, "answer_query", fake_answer)
    return asked


def _canned(text: str, archetype: Archetype = Archetype.LOOKUP) -> AnswerResult:
    return AnswerResult(query=text, archetype=archetype, answer=f"answer to {text}", citations=[])


def test_chat_answers_each_question_until_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _patch_chat_deps(monkeypatch, _canned)

    result = runner.invoke(cli.app, ["chat"], input="first q\n\n   \nsecond q\nexit\nnever asked\n")

    assert result.exit_code == 0
    assert asked == ["first q", "second q"]
    assert "[A] answer to first q" in result.output
    assert "A · lookup" in result.output
    assert "Answered 2 questions." in result.output


def test_chat_ends_cleanly_at_end_of_input(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _patch_chat_deps(monkeypatch, _canned)

    result = runner.invoke(cli.app, ["chat"], input="only q\n")

    assert result.exit_code == 0
    assert asked == ["only q"]
    assert "Answered 1 question." in result.output


def test_chat_keeps_going_after_a_failed_question(monkeypatch: pytest.MonkeyPatch) -> None:
    def answer(text: str) -> AnswerResult:
        if text == "bad":
            raise RuntimeError("LLM returned 503")
        return _canned(text)

    asked = _patch_chat_deps(monkeypatch, answer)

    result = runner.invoke(cli.app, ["chat"], input="bad\ngood\nquit\n")

    assert result.exit_code == 0
    assert asked == ["bad", "good"]
    assert "Couldn't answer that: LLM returned 503" in result.output
    assert "[A] answer to good" in result.output
    assert "Answered 1 question." in result.output


def test_chat_writes_a_transcript(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_chat_deps(monkeypatch, lambda t: _canned(t, Archetype.COMPARATIVE))
    out = tmp_path / "session.md"

    result = runner.invoke(cli.app, ["chat", "--transcript", str(out)], input="compare us\nexit\n")

    assert result.exit_code == 0
    text = out.read_text()
    assert text.startswith("# Tessera chat — ")
    assert "## Q1. compare us" in text
    assert "Archetype D (comparative — declined)" in text
    assert "[D] answer to compare us" in text


def test_chat_requires_an_existing_index(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyStore:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 0

    monkeypatch.setattr(cli, "ChromaVectorStore", EmptyStore)

    result = runner.invoke(cli.app, ["chat"], input="q\n")

    assert result.exit_code == 1
    assert "run `tessera ingest` first" in result.output


# --- Phase 4: provider selection ---


def _record_bedrock(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    built: list[dict] = []
    monkeypatch.setattr(cli, "_require_aws_profile", lambda profile: None)

    def fake_bedrock(model, **kw):
        built.append({"model": model, **kw})
        return object()

    monkeypatch.setattr(cli, "BedrockClient", fake_bedrock)
    return built


def test_bedrock_provider_builds_a_haiku_router_and_an_opus_answerer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TESSERA_LLM_PROVIDER", "bedrock")
    built = _record_bedrock(monkeypatch)
    monkeypatch.setattr(cli, "NvidiaClient", lambda *a, **kw: pytest.fail("NIM not expected"))

    llms = cli._build_llms(cli._load_settings())

    answer, router = built
    assert answer["model"] == "anthropic.claude-opus-5-5"
    assert answer["effort"] == "medium"
    assert router["model"] == "anthropic.claude-haiku-4-5"
    assert router["effort"] is None
    for client in built:
        assert client["aws_profile"] == "tessera"
        assert client["aws_region"] == "us-east-1"
        assert client["sdk_max_retries"] == 0
    assert llms.answer is not llms.router
    assert "opus-5-5" in llms.name and "haiku-4-5" in llms.name


def test_nvidia_provider_uses_one_client_for_routing_and_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "NvidiaClient", lambda api_key, model, **kw: object())
    monkeypatch.setattr(cli, "BedrockClient", lambda *a, **kw: pytest.fail("Bedrock not expected"))

    llms = cli._build_llms(cli._load_settings())

    assert llms.answer is llms.router
    assert llms.name.startswith("nvidia:")


def test_eval_on_bedrock_keeps_the_judge_on_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    from tessera.config import MODEL_PRICES
    from tessera.generation.resilient import RetryingLLMClient

    monkeypatch.setenv("TESSERA_LLM_PROVIDER", "bedrock")
    _record_bedrock(monkeypatch)
    nvidia_built: list[str] = []

    def fake_nvidia(api_key, model, **kw):
        nvidia_built.append(model)
        return object()

    monkeypatch.setattr(cli, "NvidiaClient", fake_nvidia)

    class Store:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 1

    monkeypatch.setattr(cli, "ChromaVectorStore", Store)
    monkeypatch.setattr(cli, "ChromaExpertiseStore", Store)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())

    captured: dict = {}
    fake_harness = type(sys)("evals.harness")
    fake_harness.load_cases = lambda d: []

    def fake_run_harness(cases, llm, embedder, store, corpus_dir, **kw):
        captured.update(kw, llm=llm)
        return "report"

    fake_harness.run_harness = fake_run_harness
    fake_harness.format_report = lambda r: "REPORT TEXT"
    fake_evals_pkg = type(sys)("evals")
    fake_evals_pkg.harness = fake_harness
    monkeypatch.setitem(sys.modules, "evals", fake_evals_pkg)
    monkeypatch.setitem(sys.modules, "evals.harness", fake_harness)

    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 0, result.output
    assert nvidia_built == ["nvidia/nemotron-3-ultra-550b-a55b"]  # the judge only
    judge = captured["judge_llm"]
    assert isinstance(judge, RetryingLLMClient)
    assert judge._min_interval == cli.EVAL_MIN_CALL_INTERVAL_SECONDS
    assert judge is not captured["llm"] and captured["router_llm"] is not captured["llm"]
    assert captured["prices"] is MODEL_PRICES
    assert "judge: nvidia:" in result.output


def test_query_prints_tokens_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    from tessera.generation.base import Usage
    from tessera.generation.usage import UsageSummary

    _patch_chat_deps(
        monkeypatch,
        lambda text: AnswerResult(
            query=text,
            archetype=Archetype.LOOKUP,
            answer="ok",
            citations=[],
            usage=UsageSummary(calls=(Usage("anthropic.claude-haiku-4-5", 1000, 100, 0.1),)),
        ),
    )

    result = runner.invoke(cli.app, ["query", "pricing?"])

    assert result.exit_code == 0, result.output
    assert "1,000 in / 100 out tokens · $0.0015" in result.output


def test_bedrock_provider_without_the_aws_profile_exits_with_instructions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Point botocore at empty files so the real ~/.aws is never read.
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TESSERA_LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_AWS_PROFILE", "tessera-missing")

    class Store:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 1

    monkeypatch.setattr(cli, "ChromaVectorStore", Store)
    monkeypatch.setattr(cli, "ChromaExpertiseStore", Store)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())

    result = runner.invoke(cli.app, ["query", "pricing?"])

    assert result.exit_code == 1
    assert "AWS profile 'tessera-missing' not found" in result.output
    assert "aws configure --profile tessera-missing" in result.output


# --- P4-3: traces and the feedback commands ---


def _query_once(monkeypatch: pytest.MonkeyPatch) -> str:
    """Run `tessera query` with fakes; return the trace id it printed."""
    import re

    _patch_chat_deps(monkeypatch, _canned)
    result = runner.invoke(cli.app, ["query", "pricing?"])
    assert result.exit_code == 0, result.output
    return re.search(r"trace ([0-9a-f]{32})", result.output).group(1)


def test_query_writes_a_trace_and_prints_its_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from tessera.feedback.local import JsonlTraceLog

    trace_id = _query_once(monkeypatch)

    record = JsonlTraceLog(Path("data/traces/traces.jsonl")).get(trace_id)
    assert record is not None and record["query"] == "pricing?"


def test_feedback_add_review_and_to_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml

    trace_id = _query_once(monkeypatch)

    added = runner.invoke(
        cli.app,
        ["feedback", "add", trace_id, "--rating", "down", "--reason", "wrong-source", "--comment", "too broad"],
    )
    assert added.exit_code == 0, added.output

    review = runner.invoke(cli.app, ["feedback", "review"])
    assert trace_id in review.output and "wrong-source — too broad" in review.output

    out = runner.invoke(cli.app, ["feedback", "to-cases"])
    assert "Wrote 1 candidate case(s)" in out.output
    (case,) = yaml.safe_load(Path("data/feedback/candidates.yaml").read_text())
    assert case["status"] == "candidate" and case["query"] == "pricing?"


def test_feedback_add_rejects_an_unknown_trace_or_rating(monkeypatch: pytest.MonkeyPatch) -> None:
    unknown = runner.invoke(cli.app, ["feedback", "add", "nope", "--rating", "down"])
    assert unknown.exit_code == 1 and "No answer with trace id" in unknown.output

    bad = runner.invoke(cli.app, ["feedback", "add", "nope", "--rating", "meh"])
    assert bad.exit_code == 1 and "--rating must be" in bad.output


def test_feedback_to_cases_refuses_to_write_into_evals_cases() -> None:
    result = runner.invoke(
        cli.app, ["feedback", "to-cases", "--out", str(cli.EVAL_CASES_DIR / "from_feedback.yaml")]
    )

    assert result.exit_code == 1
    assert "Refusing to write candidates into evals/cases/" in result.output
    assert not (cli.EVAL_CASES_DIR / "from_feedback.yaml").exists()


def test_feedback_review_with_nothing_to_review() -> None:
    result = runner.invoke(cli.app, ["feedback", "review"])

    assert result.exit_code == 0 and "No thumbs-down feedback." in result.output


def test_data_report_reads_the_corpus_and_prints_every_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "---\ntitle: A\ndoc_type: methodology\nindustry: x\ntopics: [t]\n"
        "date: 2018-01-01\nreview_status: pending\n---\n\n## Overview\n\nBody.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TESSERA_CORPUS_DIR", str(corpus))

    class FakeEmbedder:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

    monkeypatch.setattr(cli, "LocalEmbedder", FakeEmbedder)

    result = runner.invoke(cli.app, ["data-report"])

    assert result.exit_code == 0, result.output
    assert "Data-Quality Report" in result.output
    assert "Quarantined (awaiting human review; not embedded): 1" in result.output
    assert "0 chunks embedded" in result.output  # the quarantined doc is not embedded
