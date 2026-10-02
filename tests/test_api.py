"""Unit tests for the HTTP layer (`tessera.api`) — `answer_query` is
monkeypatched with canned results and the stores are fakes, so these run
with no model, no index, and no LLM call.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tessera import api, cli
from tessera.generation.answer import Citation
from tessera.pipeline import AnswerResult
from tessera.retrieval.router import Archetype
from tests.test_expertise_generation import match


class FakeStore:
    def __init__(self, n: int) -> None:
        self.n = n

    def count(self) -> int:
        return self.n


def _client(
    monkeypatch: pytest.MonkeyPatch,
    answer,
    *,
    people: FakeStore | None = FakeStore(600),
) -> tuple[TestClient, list[str]]:
    asked: list[str] = []

    def fake_answer(question, llm, embedder, store, expertise_store=None, **kw):
        asked.append(question)
        return answer(question)

    monkeypatch.setattr(api, "answer_query", fake_answer)
    app = api.create_app(object(), object(), FakeStore(336), people, llm_name="fake:model")
    return TestClient(app), asked


def _lookup(question: str) -> AnswerResult:
    return AnswerResult(
        query=question,
        archetype=Archetype.LOOKUP,
        answer="Use the four-stage framework [1].",
        citations=[
            Citation(
                marker=1,
                document_path="data/corpus/methodology/market-entry-overview.md",
                document_title="Market Entry Analysis: Overview",
                heading_path=("Framework", "Stages"),
            )
        ],
    )


def test_ask_returns_a_cited_lookup_answer_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    client, asked = _client(monkeypatch, _lookup)

    response = client.post("/api/ask", json={"question": "  market entry framework?  "})

    assert response.status_code == 200
    body = response.json()
    assert asked == ["market entry framework?"]  # trimmed before the pipeline sees it
    assert body["archetype"] == "A"
    assert body["archetype_label"] == "lookup"
    assert body["answer"] == "Use the four-stage framework [1]."
    assert body["citations"] == [
        {
            "marker": 1,
            "title": "Market Entry Analysis: Overview",
            "heading_path": ["Framework", "Stages"],
            "document_path": "data/corpus/methodology/market-entry-overview.md",
        }
    ]
    assert body["experts"] == []
    assert isinstance(body["latency_s"], float)


def test_ask_returns_people_with_their_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    def expertise(question: str) -> AnswerResult:
        return AnswerResult(
            query=question,
            archetype=Archetype.EXPERTISE,
            answer="Ask Person a [1].",
            citations=[],
            experts=[match("a", 2.0), match("b", 0.1, self_reported=True)],
        )

    client, _ = _client(monkeypatch, expertise)

    body = client.post("/api/ask", json={"question": "who knows pricing?"}).json()

    assert body["archetype"] == "B" and body["archetype_label"] == "expertise"
    first, second = body["experts"]
    assert first["rank"] == 1 and first["name"] == "Person a"
    assert first["evidenced"] is True
    assert first["evidence"] == [
        {
            "kind": "project",
            "description": "pricing project in energy, 2025 (workstream lead)",
            "self_reported": False,
        }
    ]
    assert second["evidenced"] is False
    assert second["evidence"][0]["self_reported"] is True
    assert set(first) >= {"person_id", "title", "practice", "office", "last_updated"}


def test_ask_returns_the_comparative_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = "I can't compare approaches across specific client engagements."
    client, _ = _client(
        monkeypatch,
        lambda q: AnswerResult(query=q, archetype=Archetype.COMPARATIVE, answer=refusal, citations=[]),
    )

    body = client.post("/api/ask", json={"question": "compare Acme vs Globex"}).json()

    assert body["archetype"] == "D"
    assert body["archetype_label"] == "comparative — declined"
    assert body["answer"] == refusal
    assert body["citations"] == [] and body["experts"] == []


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "   "}, {"question": "x" * (api.MAX_QUESTION_CHARS + 1)}],
)
def test_ask_rejects_missing_blank_or_oversized_questions(
    monkeypatch: pytest.MonkeyPatch, payload: dict
) -> None:
    client, asked = _client(monkeypatch, _lookup)

    response = client.post("/api/ask", json=payload)

    assert response.status_code == 422
    assert asked == []


def test_a_pipeline_failure_becomes_a_json_error_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(question: str) -> AnswerResult:
        raise RuntimeError("LLM returned 503 — secret internal detail")

    client, _ = _client(monkeypatch, boom)

    response = client.post("/api/ask", json={"question": "anything"})

    assert response.status_code == 502
    assert response.json() == {"error": api.ANSWER_FAILED_MESSAGE}
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text


def test_health_reports_index_counts_and_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(monkeypatch, _lookup)

    assert client.get("/api/health").json() == {
        "status": "ok",
        "document_chunks": 336,
        "people": 600,
        "people_search": True,
        "llm": "fake:model",
    }


def test_health_without_a_people_index(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(monkeypatch, _lookup, people=None)

    body = client.get("/api/health").json()

    assert body["people"] == 0 and body["people_search"] is False


def test_root_serves_a_page(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(monkeypatch, _lookup)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Tessera" in response.text


def test_serve_builds_dependencies_once_and_runs_uvicorn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    class NonEmptyStore:
        def __init__(self, persist_dir: Path) -> None:
            pass

        def count(self) -> int:
            return 3

    monkeypatch.setattr(cli, "ChromaVectorStore", NonEmptyStore)
    monkeypatch.setattr(cli, "ChromaExpertiseStore", NonEmptyStore)
    monkeypatch.setattr(cli, "LocalEmbedder", lambda: object())
    monkeypatch.setattr(cli, "NvidiaClient", lambda api_key, model, **kw: object())

    ran: dict = {}

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: ran.update(app=app, host=host, port=port))

    result = CliRunner().invoke(cli.app, ["serve", "--port", "8123"])

    assert result.exit_code == 0, result.output
    assert ran["host"] == "127.0.0.1" and ran["port"] == 8123
    health = TestClient(ran["app"]).get("/api/health").json()
    assert health["document_chunks"] == 3 and health["people_search"] is True
    assert health["llm"].startswith("nvidia:")


def test_ask_reports_tokens_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    from tessera.generation.base import Usage
    from tessera.generation.usage import ModelPrice, UsageSummary

    def priced(question: str) -> AnswerResult:
        base = _lookup(question)
        usage = UsageSummary(
            calls=(Usage("haiku", 500, 20, 0.4), Usage("opus", 3000, 400, 6.2)),
        )
        return AnswerResult(
            query=base.query,
            archetype=base.archetype,
            answer=base.answer,
            citations=base.citations,
            usage=usage,
        )

    seen: dict = {}

    def fake_answer(question, llm, embedder, store, expertise_store=None, **kw):
        seen.update(kw)
        return priced(question)

    monkeypatch.setattr(api, "answer_query", fake_answer)
    router = object()
    app = api.create_app(
        object(),
        object(),
        FakeStore(336),
        None,
        llm_name="bedrock:opus",
        router_llm=router,
        prices={"haiku": ModelPrice(1.0, 5.0), "opus": ModelPrice(4.0, 20.0)},
    )

    body = TestClient(app).post("/api/ask", json={"question": "pricing?"}).json()

    assert seen["router_llm"] is router
    assert body["usage"]["input_tokens"] == 3500
    assert body["usage"]["output_tokens"] == 420
    assert [c["model"] for c in body["usage"]["calls"]] == ["haiku", "opus"]
    assert body["cost_usd"] == pytest.approx((500 + 100 + 12_000 + 8_000) / 1e6)


def test_ask_with_no_llm_calls_reports_zero_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(monkeypatch, _lookup)

    body = client.post("/api/ask", json={"question": "pricing?"}).json()

    assert body["usage"] == {"input_tokens": 0, "output_tokens": 0, "calls": []}
    assert body["cost_usd"] == 0.0  # no LLM calls in the canned answer
