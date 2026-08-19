"""Unit tests for evals/metrics.py — deterministic retrieval-metric
functions tested directly; the LLM-judge is tested only for
response-parsing (fake LLM client, no network), per CLAUDE.md's "do not
over-test LLM outputs" convention.
"""

import pytest

from evals.metrics import (
    JudgeError,
    JudgeScore,
    build_judge_user_prompt,
    judge_answer,
    mean,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from tessera.generation.base import LLMClient


class FakeLLMClient(LLMClient):
    def __init__(self, canned_response: str) -> None:
        self._response = canned_response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        return self._response


# --- recall_at_k / precision_at_k / reciprocal_rank / mean ---


def test_recall_at_k_counts_relevant_hits_within_top_k() -> None:
    retrieved = ["a.md", "b.md", "c.md", "d.md"]
    relevant = {"a.md", "c.md", "z.md"}  # z.md never retrieved

    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 3)


def test_recall_at_k_only_counts_top_k() -> None:
    retrieved = ["a.md", "b.md", "c.md"]
    relevant = {"c.md"}

    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_rejects_empty_relevant_set() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a.md"], set(), k=5)


def test_precision_at_k_counts_relevant_within_top_k() -> None:
    retrieved = ["a.md", "b.md", "c.md"]
    relevant = {"a.md", "c.md"}

    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_divides_by_actual_retrieved_when_fewer_than_k() -> None:
    retrieved = ["a.md"]
    relevant = {"a.md"}

    assert precision_at_k(retrieved, relevant, k=5) == 1.0


def test_precision_at_k_zero_when_nothing_retrieved() -> None:
    assert precision_at_k([], {"a.md"}, k=5) == 0.0


def test_reciprocal_rank_of_first_relevant_hit() -> None:
    retrieved = ["a.md", "b.md", "c.md"]
    relevant = {"c.md"}

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_reciprocal_rank_zero_when_nothing_relevant_retrieved() -> None:
    assert reciprocal_rank(["a.md", "b.md"], {"z.md"}) == 0.0


def test_mean_of_empty_list_is_zero() -> None:
    assert mean([]) == 0.0


def test_mean_averages_values() -> None:
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


# --- LLM-judge: response parsing only ---


def test_judge_answer_parses_valid_json_response() -> None:
    llm = FakeLLMClient(
        '{"groundedness": 4, "relevance": 5, "reasoning": "well cited"}'
    )

    score = judge_answer(
        "do we have X?", "should point to X", ["[1] Doc X"], "Yes, see [1].", llm
    )

    assert score == JudgeScore(groundedness=4, relevance=5, reasoning="well cited")


def test_judge_answer_strips_markdown_fence() -> None:
    llm = FakeLLMClient(
        '```json\n{"groundedness": 3, "relevance": 3, "reasoning": "ok"}\n```'
    )

    score = judge_answer("q", "ideal", ["[1] source"], "answer", llm)

    assert score.groundedness == 3
    assert score.relevance == 3


@pytest.mark.parametrize(
    "bad_response",
    [
        "not json",
        '{"groundedness": 4}',  # missing relevance/reasoning
        '{"relevance": 5, "reasoning": "x"}',  # missing groundedness
        "",
    ],
)
def test_judge_answer_raises_judge_error_on_malformed_response(
    bad_response: str,
) -> None:
    llm = FakeLLMClient(bad_response)

    with pytest.raises(JudgeError):
        judge_answer("q", "ideal", ["[1] source"], "answer", llm)


def test_build_judge_user_prompt_includes_all_inputs() -> None:
    prompt = build_judge_user_prompt(
        "do we have X?", "should point to X", ["[1] Doc X text"], "Yes, see [1]."
    )

    assert "do we have X?" in prompt
    assert "should point to X" in prompt
    assert "[1] Doc X text" in prompt
    assert "Yes, see [1]." in prompt


def test_build_judge_user_prompt_handles_no_sources() -> None:
    prompt = build_judge_user_prompt("q", "ideal", [], "we don't have anything")

    assert "(none)" in prompt
