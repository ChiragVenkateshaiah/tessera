"""Unit tests for grounded-answer generation (fake LLM client, no
network), plus a real-corpus/real-embedder integration test proving
RELEVANCE_THRESHOLD actually separates the Task 6 acceptance-check
off-corpus example from genuine matches.
"""

from pathlib import Path

import pytest

from tessera.embedding.local import LocalEmbedder
from tessera.generation.answer import (
    NO_RESULTS_MESSAGE,
    RELEVANCE_THRESHOLD,
    Citation,
    GeneratedAnswer,
    generate_answer,
)
from tessera.generation.base import LLMClient
from tessera.generation.prompts import (
    LOOKUP_ANSWER_SYSTEM_PROMPT,
    SYNTHESIS_ANSWER_SYSTEM_PROMPT,
)
from tessera.ingestion.chunker import chunk_corpus
from tessera.ingestion.loader import load_corpus
from tessera.retrieval.retriever import retrieve
from tessera.retrieval.retriever import RetrievalResult
from tessera.retrieval.router import Archetype
from tessera.store.base import SearchResult
from tessera.store.chroma import ChromaVectorStore

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


class FakeLLMClient(LLMClient):
    """Returns a canned response and records the system/user prompts it
    was called with, so tests can assert on prompt shape without a real
    model.
    """

    def __init__(self, canned_response: str) -> None:
        self._response = canned_response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        return self._response


class ExplodingLLMClient(LLMClient):
    """Fails the test if called — used to prove the "nothing relevant"
    path never spends an LLM call.
    """

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise AssertionError("LLM should not be called when nothing is relevant")


def _result(
    document_path: str,
    document_title: str,
    score: float,
    heading_path: tuple[str, ...] = ("Overview",),
) -> SearchResult:
    return SearchResult(
        chunk_id=f"{document_path}#1",
        text=f"text for {document_title}",
        score=score,
        document_path=document_path,
        document_title=document_title,
        doc_type="methodology",
        industry="cross-industry",
        topics=["t1"],
        date="2024-01-01",
        heading_path=heading_path,
    )


def test_generate_answer_uses_lookup_prompt_for_archetype_a() -> None:
    llm = FakeLLMClient("Yes, we have [1].")
    retrieval = RetrievalResult(
        query="do we have a market entry template?",
        archetype=Archetype.LOOKUP,
        results=[_result("methodology/market-entry.md", "Market Entry", 0.6)],
    )

    generated = generate_answer(retrieval, llm)

    assert generated.answer == "Yes, we have [1]."
    system, user = llm.calls[0]
    assert system == LOOKUP_ANSWER_SYSTEM_PROMPT
    assert "Market Entry" in user
    assert "do we have a market entry template?" in user


def test_generate_answer_uses_synthesis_prompt_for_archetype_c() -> None:
    llm = FakeLLMClient("Here's a briefing [1][2].")
    retrieval = RetrievalResult(
        query="get me up to speed on pricing",
        archetype=Archetype.SYNTHESIS,
        results=[
            _result("methodology/pricing.md", "Pricing Strategy", 0.6),
            _result("thought-leadership/pricing-2.md", "Pricing in Practice", 0.55),
        ],
    )

    generated = generate_answer(retrieval, llm)

    system, _ = llm.calls[0]
    assert system == SYNTHESIS_ANSWER_SYSTEM_PROMPT


def test_generate_answer_returns_citations_for_every_source_used() -> None:
    llm = FakeLLMClient("answer text")
    results = [
        _result("a.md", "Doc A", 0.6, heading_path=("Overview",)),
        _result("b.md", "Doc B", 0.5, heading_path=("Details", "Sub")),
    ]
    retrieval = RetrievalResult(query="q", archetype=Archetype.LOOKUP, results=results)

    generated = generate_answer(retrieval, llm)

    assert generated.citations == [
        Citation(marker=1, document_path="a.md", document_title="Doc A", heading_path=("Overview",)),
        Citation(
            marker=2,
            document_path="b.md",
            document_title="Doc B",
            heading_path=("Details", "Sub"),
        ),
    ]


def test_generate_answer_drops_chunks_below_relevance_threshold() -> None:
    llm = FakeLLMClient("answer")
    results = [
        _result("a.md", "Doc A", RELEVANCE_THRESHOLD + 0.1),
        _result("b.md", "Doc B", RELEVANCE_THRESHOLD - 0.1),
    ]
    retrieval = RetrievalResult(query="q", archetype=Archetype.LOOKUP, results=results)

    generated = generate_answer(retrieval, llm)

    assert len(generated.citations) == 1
    assert generated.citations[0].document_path == "a.md"
    _, user = llm.calls[0]
    assert "Doc B" not in user


def test_generate_answer_returns_fixed_message_when_nothing_relevant_without_calling_llm() -> None:
    llm = ExplodingLLMClient()
    results = [_result("a.md", "Doc A", RELEVANCE_THRESHOLD - 0.01)]
    retrieval = RetrievalResult(query="q", archetype=Archetype.LOOKUP, results=results)

    generated = generate_answer(retrieval, llm)

    assert generated == GeneratedAnswer(
        query="q", archetype=Archetype.LOOKUP, answer=NO_RESULTS_MESSAGE, citations=[]
    )


def test_generate_answer_returns_fixed_message_when_no_results_at_all() -> None:
    llm = ExplodingLLMClient()
    retrieval = RetrievalResult(query="q", archetype=Archetype.SYNTHESIS, results=[])

    generated = generate_answer(retrieval, llm)

    assert generated.answer == NO_RESULTS_MESSAGE
    assert generated.citations == []


@pytest.mark.parametrize("archetype", [Archetype.EXPERTISE, Archetype.COMPARATIVE])
def test_generate_answer_rejects_archetypes_that_never_reach_it(
    archetype: Archetype,
) -> None:
    llm = ExplodingLLMClient()
    retrieval = RetrievalResult(query="q", archetype=archetype, results=[])

    with pytest.raises(ValueError):
        generate_answer(retrieval, llm)


# --- Real-corpus integration: proves the threshold against real scores ---


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder()


@pytest.fixture(scope="module")
def indexed_store(tmp_path_factory: pytest.TempPathFactory, embedder: LocalEmbedder):
    docs = load_corpus(CORPUS_DIR)
    chunks = chunk_corpus(docs)
    embeddings = embedder.embed_documents([c.text for c in chunks])
    persist_dir = tmp_path_factory.mktemp("vectorstore")
    store = ChromaVectorStore(persist_dir=persist_dir)
    store.add(chunks, embeddings)
    return store


def test_off_corpus_acceptance_check_query_produces_clean_refusal(
    indexed_store: ChromaVectorStore, embedder: LocalEmbedder
) -> None:
    """The Task 6 acceptance check, directly: an off-corpus question
    produces "we don't have anything on that" rather than an LLM call
    that might invent something.
    """
    llm = ExplodingLLMClient()
    retrieval = retrieve(
        "What's our policy on parental leave?",
        Archetype.LOOKUP,
        embedder,
        indexed_store,
    )

    generated = generate_answer(retrieval, llm)

    assert generated.answer == NO_RESULTS_MESSAGE
    assert generated.citations == []


def test_on_corpus_query_clears_the_relevance_bar(
    indexed_store: ChromaVectorStore, embedder: LocalEmbedder
) -> None:
    llm = FakeLLMClient("Yes, see [1].")
    retrieval = retrieve(
        "Do we have a framework for market entry analysis?",
        Archetype.LOOKUP,
        embedder,
        indexed_store,
    )

    generated = generate_answer(retrieval, llm)

    assert generated.answer == "Yes, see [1]."
    assert len(generated.citations) > 0
