"""The corrective loop (Phase 5 plan §3.7, P5-7): its graders, its "at
most once" rule, its metering, and that generation answers the original
question from the retried context. The scope-binding and leak checks on
the real index are in ``tests/test_lc_retrieval.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langgraph", reason="the lc extra isn't installed")

from tessera.generation.base import Completion, LLMClient, Usage  # noqa: E402
from tessera.generation.prompts import (  # noqa: E402
    LOOKUP_ANSWER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SYNTHESIS_ANSWER_SYSTEM_PROMPT,
)
from tessera.lc.corrective import (  # noqa: E402
    GRADER_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    WEAK_TOP_SCORE,
    parse_verdict,
    score_sufficient,
)
from tessera.lc.graph import LangGraphPipeline  # noqa: E402
from tessera.pipeline import NativePipeline  # noqa: E402
from tessera.retrieval.retriever import RetrievalResult  # noqa: E402
from tessera.retrieval.router import Archetype  # noqa: E402
from tests.test_pipeline import FakeVectorStore, PricingEmbedder, _result, _router_response  # noqa: E402


class Scripted(LLMClient):
    """Metered replies keyed by system prompt; records every call."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Completion:
        self.calls.append((system, user))
        return Completion(self.replies[system], Usage("m", 10, 1, 0.0))


def replies(archetype: str = "A", *, sufficient: bool, rewrite: str = "pricing methodology playbook") -> dict[str, str]:
    return {
        ROUTER_SYSTEM_PROMPT: _router_response(archetype),
        GRADER_SYSTEM_PROMPT: f'{{"sufficient": {str(sufficient).lower()}, "reasoning": "test"}}',
        REWRITE_SYSTEM_PROMPT: rewrite,
        LOOKUP_ANSWER_SYSTEM_PROMPT: "From [1].",
        SYNTHESIS_ANSWER_SYSTEM_PROMPT: "From [1] and [2].",
    }


class QueryStore(FakeVectorStore):
    """Returns weak results for the original question and strong ones for
    the rewritten query, and records each query's embedding."""

    def __init__(self) -> None:
        super().__init__([])
        self.queries = 0

    def query(self, embedding: list[float], k: int, where: Any = None) -> list[Any]:
        self.queries += 1
        return [_result("methodology/pricing.md", 0.9), _result("methodology/pricing-2.md", 0.8)][:k]


def run(corrective: str, llm: Scripted, query: str = "pricing framework?") -> Any:
    return LangGraphPipeline(llm, PricingEmbedder(), QueryStore(), corrective=corrective).run(query)


# --- graders ---


def _retrieval(archetype: Archetype, scores: list[float]) -> RetrievalResult:
    return RetrievalResult("q", archetype, [_result(f"d{i}.md", s) for i, s in enumerate(scores)])


@pytest.mark.parametrize(
    ("archetype", "scores", "sufficient"),
    [
        (Archetype.LOOKUP, [0.9], True),
        (Archetype.LOOKUP, [], False),  # nothing over the floor
        (Archetype.LOOKUP, [0.2], False),  # under the generation floor
        (Archetype.LOOKUP, [WEAK_TOP_SCORE - 0.01], False),  # weak best chunk
        (Archetype.SYNTHESIS, [0.9], False),  # C needs two
        (Archetype.SYNTHESIS, [0.9, 0.5], True),
    ],
)
def test_the_score_grader(archetype: Archetype, scores: list[float], sufficient: bool) -> None:
    assert score_sufficient(_retrieval(archetype, scores)) is sufficient


@pytest.mark.parametrize(
    ("text", "sufficient"),
    [
        ('{"sufficient": false, "reasoning": "x"}', False),
        ('```json\n{"sufficient": false, "reasoning": "x"}\n```', False),
        ('{"sufficient": true, "reasoning": "x"}', True),
        ("not json at all", True),  # unreadable: no retry
        ('{"reasoning": "missing verdict"}', True),
    ],
)
def test_the_llm_verdict_parser(text: str, sufficient: bool) -> None:
    assert parse_verdict(text) is sufficient


# --- the loop ---


def test_off_makes_no_corrective_call_and_equals_native() -> None:
    llm = Scripted(replies(sufficient=False))
    graph = run("off", llm)
    native = NativePipeline(llm, PricingEmbedder(), QueryStore()).run("pricing framework?")
    assert graph == native
    assert [s for s, _ in llm.calls].count(GRADER_SYSTEM_PROMPT) == 0


def test_a_sufficient_verdict_retrieves_once() -> None:
    llm = Scripted(replies(sufficient=True))
    result = run("llm", llm)
    assert len(result.retrievals) == 1
    systems = [s for s, _ in llm.calls]
    assert systems == [ROUTER_SYSTEM_PROMPT, GRADER_SYSTEM_PROMPT, LOOKUP_ANSWER_SYSTEM_PROMPT]
    # Routing 1 + grading 1 + generation 1, all in the answer's usage.
    assert len(result.answer.usage.calls) == 3
    assert result.generation_calls == 1


def test_an_insufficient_verdict_rewrites_and_retries_once() -> None:
    llm = Scripted(replies(sufficient=False, rewrite="pricing methodology playbook"))
    result = run("llm", llm)
    assert [r.query for r in result.retrievals] == ["pricing framework?", "pricing methodology playbook"]
    systems = [s for s, _ in llm.calls]
    # One grade only: the retry isn't graded again (at most once).
    assert systems == [ROUTER_SYSTEM_PROMPT, GRADER_SYSTEM_PROMPT, REWRITE_SYSTEM_PROMPT, LOOKUP_ANSWER_SYSTEM_PROMPT]
    # Generation answers the original question from the retried context.
    answer_prompt = llm.calls[-1][1]
    assert "Question: pricing framework?" in answer_prompt
    assert result.generated is not None and result.generated.query == "pricing framework?"
    assert len(result.answer.usage.calls) == 4 and result.generation_calls == 1
    assert len(result.routing_usage.calls) == 1


def test_an_empty_rewrite_does_not_retry() -> None:
    llm = Scripted(replies(sufficient=False, rewrite="   "))
    result = run("llm", llm)
    assert len(result.retrievals) == 1


def test_the_score_grader_retries_without_a_grading_call() -> None:
    class Weak(QueryStore):
        """Over the generation floor, under WEAK_TOP_SCORE."""

        def query(self, embedding: list[float], k: int, where: Any = None) -> list[Any]:
            return [_result("methodology/pricing.md", WEAK_TOP_SCORE - 0.05)][:k]

    llm = Scripted(replies(sufficient=True))
    result = LangGraphPipeline(llm, PricingEmbedder(), Weak(), corrective="score").run("pricing framework?")
    systems = [s for s, _ in llm.calls]
    assert GRADER_SYSTEM_PROMPT not in systems
    assert REWRITE_SYSTEM_PROMPT in systems
    assert len(result.retrievals) == 2


def test_the_grader_and_rewriter_use_the_routing_model() -> None:
    answer = Scripted(replies(sufficient=False))
    router = Scripted(replies(sufficient=False))
    LangGraphPipeline(answer, PricingEmbedder(), QueryStore(), router_llm=router, corrective="llm").run(
        "pricing framework?"
    )
    assert [s for s, _ in router.calls] == [ROUTER_SYSTEM_PROMPT, GRADER_SYSTEM_PROMPT, REWRITE_SYSTEM_PROMPT]
    assert [s for s, _ in answer.calls] == [LOOKUP_ANSWER_SYSTEM_PROMPT]


@pytest.mark.parametrize("archetype", ["B", "D"])
def test_the_loop_never_runs_outside_documents(archetype: str) -> None:
    llm = Scripted(replies(archetype, sufficient=False))
    LangGraphPipeline(llm, PricingEmbedder(), QueryStore(), corrective="llm").run("who knows pricing?")
    assert GRADER_SYSTEM_PROMPT not in [s for s, _ in llm.calls]
