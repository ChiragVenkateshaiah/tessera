"""Unit tests for the end-to-end pipeline (fake LLM/Embedder/VectorStore,
no network) — proves route -> retrieve -> generate wiring, and that B/D
short-circuit before retrieval or generation ever run.
"""

import pytest

from tessera.embedding.base import Embedder
from tessera.generation.answer import NO_RESULTS_MESSAGE, RELEVANCE_THRESHOLD
from tessera.generation.base import LLMClient
from tessera.generation.prompts import ROUTER_SYSTEM_PROMPT
from tessera.pipeline import AnswerResult, answer_query
from tessera.retrieval.router import (
    COMPARATIVE_REFUSAL_MESSAGE,
    NOT_YET_SUPPORTED_MESSAGE,
    Archetype,
)
from tessera.store.base import SearchResult, VectorStore


class ScriptedLLMClient(LLMClient):
    """Returns a canned response keyed by exact system-prompt match — the
    pipeline calls the LLM twice per query (route, then generate) with
    different system prompts, so a single canned response isn't enough.
    """

    def __init__(self, responses_by_system: dict[str, str]) -> None:
        self._responses = responses_by_system
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        if system not in self._responses:
            raise AssertionError(f"no scripted response for system prompt: {system!r}")
        return self._responses[system]


class FakeEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 1

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def _result(document_path: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=f"{document_path}#1",
        text=f"text for {document_path}",
        score=score,
        document_path=document_path,
        document_title=document_path,
        doc_type="methodology",
        industry="cross-industry",
        topics=["t1"],
        date="2024-01-01",
        heading_path=("Overview",),
    )


class FakeVectorStore(VectorStore):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def add(self, chunks: object, embeddings: object) -> None:
        raise NotImplementedError

    def query(
        self, embedding: list[float], k: int, where: dict[str, object] | None = None
    ) -> list[SearchResult]:
        return self._results[:k]

    def count(self) -> int:
        return len(self._results)


def _router_response(archetype: str) -> str:
    return f'{{"archetype": "{archetype}", "reasoning": "test"}}'


def test_lookup_query_returns_generated_answer_with_citations() -> None:
    from tessera.generation.prompts import LOOKUP_ANSWER_SYSTEM_PROMPT

    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("A"),
            LOOKUP_ANSWER_SYSTEM_PROMPT: "We have it, see [1].",
        }
    )
    store = FakeVectorStore([_result("methodology/market-entry.md", 0.6)])

    result = answer_query("do we have a market entry template?", llm, FakeEmbedder(), store)

    assert isinstance(result, AnswerResult)
    assert result.archetype is Archetype.LOOKUP
    assert result.answer == "We have it, see [1]."
    assert len(result.citations) == 1
    assert result.citations[0].document_path == "methodology/market-entry.md"


def test_synthesis_query_returns_generated_answer() -> None:
    from tessera.generation.prompts import SYNTHESIS_ANSWER_SYSTEM_PROMPT

    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("C"),
            SYNTHESIS_ANSWER_SYSTEM_PROMPT: "Here's a briefing [1][2].",
        }
    )
    store = FakeVectorStore(
        [_result("a.md", 0.6), _result("b.md", 0.55), _result("c.md", 0.5)]
    )

    result = answer_query("get me up to speed on pricing", llm, FakeEmbedder(), store)

    assert result.archetype is Archetype.SYNTHESIS
    assert result.answer == "Here's a briefing [1][2]."


def test_expertise_query_short_circuits_before_retrieval_or_generation() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("B")})
    store = FakeVectorStore([_result("a.md", 0.9)])

    result = answer_query("who knows about pricing?", llm, FakeEmbedder(), store)

    assert result.archetype is Archetype.EXPERTISE
    assert result.answer == NOT_YET_SUPPORTED_MESSAGE
    assert result.citations == []
    # only the router call happened — generation never ran
    assert len(llm.calls) == 1


def test_comparative_query_short_circuits_before_retrieval_or_generation() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("D")})
    store = FakeVectorStore([_result("a.md", 0.9)])

    result = answer_query("compare Client X vs Client Y", llm, FakeEmbedder(), store)

    assert result.archetype is Archetype.COMPARATIVE
    assert result.answer == COMPARATIVE_REFUSAL_MESSAGE
    assert result.citations == []
    assert len(llm.calls) == 1


def test_off_corpus_lookup_query_produces_clean_refusal_without_generation_call() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("A")})
    store = FakeVectorStore([_result("a.md", RELEVANCE_THRESHOLD - 0.1)])

    result = answer_query("what's our policy on parental leave?", llm, FakeEmbedder(), store)

    assert result.answer == NO_RESULTS_MESSAGE
    assert result.citations == []
    # router call happened, but no generation call was scripted/needed
    assert len(llm.calls) == 1
