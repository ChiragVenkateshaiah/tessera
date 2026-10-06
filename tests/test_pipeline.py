"""Unit tests for the end-to-end pipeline (fake LLM/Embedder/VectorStore,
no network) — proves route -> retrieve -> generate wiring, and that B/D
short-circuit before retrieval or generation ever run.
"""

import pytest

from tessera.embedding.base import Embedder
from tessera.generation.answer import NO_RESULTS_MESSAGE, RELEVANCE_THRESHOLD
from tessera.generation.base import LLMClient
from tessera.generation.prompts import ROUTER_SYSTEM_PROMPT
from tessera.pipeline import EXPERTISE_UNAVAILABLE_MESSAGE, AnswerResult, answer_query
from tessera.retrieval.router import (
    COMPARATIVE_REFUSAL_MESSAGE,
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


class PricingEmbedder(FakeEmbedder):
    """One-dimension embedder where 'pricing' text is similar to itself."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if "pricing" in t else 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0 if "pricing" in text else 0.0]


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


class ExplodingVectorStore(FakeVectorStore):
    """Proves the document retriever is never reached for archetype B."""

    def query(self, embedding, k, where=None):  # type: ignore[override]
        raise AssertionError("document store queried for an archetype B query")


def test_expertise_query_uses_people_path_and_never_touches_document_store() -> None:
    from tessera.generation.prompts import EXPERTISE_ANSWER_SYSTEM_PROMPT
    from tests.test_expertise_generation import (
        ScoredExpertiseStore,
        make_person,
        make_project,
    )

    expert = make_person(
        "c1", projects=[make_project("pricing"), make_project("pricing", 2026)]
    )
    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("B"),
            EXPERTISE_ANSWER_SYSTEM_PROMPT: "Ask Person c1 [1].",
        }
    )

    result = answer_query(
        "who knows about pricing?",
        llm,
        PricingEmbedder(),
        ExplodingVectorStore([]),
        ScoredExpertiseStore([(expert, 0.6)]),
    )

    assert result.archetype is Archetype.EXPERTISE
    assert result.answer == "Ask Person c1 [1]."
    assert result.citations == []
    assert [m.person.person_id for m in result.experts] == ["c1"]
    assert len(llm.calls) == 2  # route + generate


def test_expertise_query_with_no_qualifying_expert_skips_generation() -> None:
    from tessera.generation.expertise import NO_EXPERT_MESSAGE
    from tests.test_expertise_generation import ScoredExpertiseStore, make_person

    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("B")})
    nobody = make_person("c1")  # no evidence on any topic

    result = answer_query(
        "who knows about pricing?",
        llm,
        PricingEmbedder(),
        ExplodingVectorStore([]),
        ScoredExpertiseStore([(nobody, 0.3)]),
    )

    assert result.answer == NO_EXPERT_MESSAGE
    assert result.experts == []
    assert len(llm.calls) == 1  # router only — zero generation calls


def test_expertise_query_without_people_index_reports_unavailable() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("B")})

    result = answer_query(
        "who knows about pricing?", llm, FakeEmbedder(), ExplodingVectorStore([])
    )

    assert result.answer == EXPERTISE_UNAVAILABLE_MESSAGE
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


# --- Phase 4: separate router client, usage per answer ---


class MeteredScripted(ScriptedLLMClient):
    """ScriptedLLMClient that also reports a fixed usage for ``model``."""

    def __init__(self, responses_by_system: dict[str, str], model: str) -> None:
        super().__init__(responses_by_system)
        self.model = model

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0):
        from tessera.generation.base import Completion, Usage

        return Completion(self.complete(system, user, temperature), Usage(self.model, 100, 10, 0.1))


def test_router_llm_routes_and_llm_answers_with_usage_from_both() -> None:
    from tessera.generation.prompts import LOOKUP_ANSWER_SYSTEM_PROMPT

    router = MeteredScripted({ROUTER_SYSTEM_PROMPT: _router_response("A")}, "haiku")
    answerer = MeteredScripted({LOOKUP_ANSWER_SYSTEM_PROMPT: "See [1]."}, "opus")
    store = FakeVectorStore([_result("methodology/market-entry.md", 0.6)])

    result = answer_query(
        "market entry template?", answerer, FakeEmbedder(), store, router_llm=router
    )

    assert [s for s, _ in router.calls] == [ROUTER_SYSTEM_PROMPT]
    assert [s for s, _ in answerer.calls] == [LOOKUP_ANSWER_SYSTEM_PROMPT]
    assert [u.model for u in result.usage.calls] == ["haiku", "opus"]
    assert result.usage.input_tokens == 200


def test_terminal_answer_carries_the_routing_usage() -> None:
    llm = MeteredScripted({ROUTER_SYSTEM_PROMPT: _router_response("D")}, "haiku")

    result = answer_query("compare Acme and Globex", llm, FakeEmbedder(), FakeVectorStore([]))

    assert result.archetype is Archetype.COMPARATIVE
    assert [u.model for u in result.usage.calls] == ["haiku"]


def test_unmetered_client_leaves_usage_without_cost() -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("D")})

    result = answer_query("compare Acme and Globex", llm, FakeEmbedder(), FakeVectorStore([]))

    assert result.usage.calls == ()
    assert result.usage.unmetered_calls == 1
    assert result.usage.cost_usd({}) is None


# --- Phase 5 (P5-2): the Pipeline protocol and PipelineRun -----------------


def test_native_pipeline_run_carries_what_the_harness_scores() -> None:
    from tessera.generation.prompts import LOOKUP_ANSWER_SYSTEM_PROMPT
    from tessera.pipeline import NativePipeline, Pipeline, PipelineRun

    llm = ScriptedLLMClient(
        {
            ROUTER_SYSTEM_PROMPT: _router_response("A"),
            LOOKUP_ANSWER_SYSTEM_PROMPT: "See [1].",
        }
    )
    store = FakeVectorStore(
        [_result("over.md", RELEVANCE_THRESHOLD + 0.2), _result("under.md", RELEVANCE_THRESHOLD - 0.1)]
    )
    pipeline: Pipeline = NativePipeline(llm, FakeEmbedder(), store)

    run = pipeline.run("market entry template?")

    assert isinstance(run, PipelineRun)
    assert run.decision.archetype is Archetype.LOOKUP
    (retrieval,) = run.retrievals
    assert [r.document_path for r in retrieval.results] == ["over.md", "under.md"]
    assert [r.document_path for r in run.shown] == ["over.md"]  # cleared the floor
    assert run.generated is not None and run.generated.answer == run.answer.answer == "See [1]."
    assert run.expertise is None


def test_routing_and_generation_usage_are_kept_apart_even_with_one_client() -> None:
    from tessera.generation.prompts import LOOKUP_ANSWER_SYSTEM_PROMPT
    from tessera.pipeline import NativePipeline

    llm = ScriptedLLMClient(
        {ROUTER_SYSTEM_PROMPT: _router_response("A"), LOOKUP_ANSWER_SYSTEM_PROMPT: "See [1]."}
    )
    store = FakeVectorStore([_result("a.md", 0.6)])

    run = NativePipeline(llm, FakeEmbedder(), store).run("q")

    # The scripted client reports no usage, so each call is "unmetered".
    assert run.routing_usage.unmetered_calls == 1
    assert run.generation_calls == 1
    assert run.answer.usage.unmetered_calls == 2  # the answer's total covers both


def test_a_terminal_answer_makes_no_generation_call() -> None:
    from tessera.pipeline import NativePipeline

    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: _router_response("D")})

    run = NativePipeline(llm, FakeEmbedder(), FakeVectorStore([])).run("compare client X and Y")

    assert run.answer.answer == COMPARATIVE_REFUSAL_MESSAGE
    assert run.generation_calls == 0 and run.retrievals == () and run.generated is None


def test_steps_are_injectable_so_routing_can_be_forced_without_a_call() -> None:
    from tessera.generation.prompts import SYNTHESIS_ANSWER_SYSTEM_PROMPT
    from tessera.pipeline import NativePipeline
    from tessera.retrieval.router import RoutingDecision

    llm = ScriptedLLMClient({SYNTHESIS_ANSWER_SYSTEM_PROMPT: "Briefing [1]."})
    forced = lambda query, _llm: RoutingDecision(query, Archetype.SYNTHESIS, "forced")
    store = FakeVectorStore([_result("a.md", 0.6)])

    run = NativePipeline(llm, FakeEmbedder(), store, route_fn=forced).run("q")

    assert run.decision.reasoning == "forced"
    assert run.routing_usage.unmetered_calls == 0 and len(llm.calls) == 1
    assert run.answer.archetype is Archetype.SYNTHESIS


def test_answer_query_is_the_native_pipelines_answer() -> None:
    from tessera.generation.prompts import LOOKUP_ANSWER_SYSTEM_PROMPT
    from tessera.pipeline import NativePipeline

    responses = {ROUTER_SYSTEM_PROMPT: _router_response("A"), LOOKUP_ANSWER_SYSTEM_PROMPT: "See [1]."}
    store = FakeVectorStore([_result("a.md", 0.6)])

    via_wrapper = answer_query("q", ScriptedLLMClient(responses), FakeEmbedder(), store)
    via_pipeline = NativePipeline(ScriptedLLMClient(responses), FakeEmbedder(), store).run("q")

    assert via_wrapper == via_pipeline.answer
