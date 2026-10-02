"""Usage and cost accounting (`tessera.generation.usage`) — pure, no
network.
"""

from __future__ import annotations

import pytest

from tessera.generation.base import Completion, LLMClient, Usage
from tessera.generation.usage import (
    ModelPrice,
    UsageRecorder,
    UsageSummary,
    combine,
)

HAIKU = "anthropic.claude-haiku-4-5"
OPUS = "anthropic.claude-opus-5-5"
PRICES = {HAIKU: ModelPrice(1.00, 5.00), OPUS: ModelPrice(4.00, 20.00)}


class MeteredFake(LLMClient):
    """Answers "ok" and reports a fixed usage per call."""

    def __init__(self, model: str, input_tokens: int = 100, output_tokens: int = 10) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(
        self, system: str, user: str, temperature: float = 0.0
    ) -> Completion:
        return Completion("ok", Usage(self.model, self.input_tokens, self.output_tokens, 0.1))


class UnmeteredFake(LLMClient):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return "ok"


def test_model_price_cost_is_per_million_tokens() -> None:
    price = ModelPrice(4.00, 20.00)

    assert price.cost(Usage(OPUS, 1_000_000, 0, 0.0)) == pytest.approx(4.00)
    assert price.cost(Usage(OPUS, 2_000, 500, 0.0)) == pytest.approx(0.018)


def test_default_complete_with_usage_reports_no_usage() -> None:
    completion = UnmeteredFake().complete_with_usage("s", "u")

    assert completion == Completion("ok", None)


def test_recorder_returns_text_and_keeps_every_call() -> None:
    recorder = UsageRecorder(MeteredFake(OPUS, 1000, 100))

    assert recorder.complete("s", "u") == "ok"
    recorder.complete("s", "u")

    summary = recorder.summary()
    assert len(summary.calls) == 2
    assert summary.input_tokens == 2000
    assert summary.output_tokens == 200
    assert summary.cost_usd(PRICES) == pytest.approx(2 * (1000 * 4 + 100 * 20) / 1e6)


def test_cost_is_unknown_when_a_call_is_unmetered() -> None:
    recorder = UsageRecorder(UnmeteredFake())
    recorder.complete("s", "u")

    summary = recorder.summary()
    assert summary.unmetered_calls == 1
    assert summary.cost_usd(PRICES) is None


def test_cost_is_unknown_when_a_model_is_unpriced() -> None:
    recorder = UsageRecorder(MeteredFake("nvidia/nemotron-3-ultra-550b-a55b"))
    recorder.complete("s", "u")

    summary = recorder.summary()
    assert summary.input_tokens == 100  # tokens still reported
    assert summary.cost_usd(PRICES) is None


def test_empty_summary_costs_nothing() -> None:
    assert UsageSummary().cost_usd(PRICES) == 0.0


def test_combine_sums_recorders_and_counts_a_shared_one_once() -> None:
    router = UsageRecorder(MeteredFake(HAIKU, 500, 20))
    answer = UsageRecorder(MeteredFake(OPUS, 3000, 400))
    router.complete("s", "u")
    answer.complete("s", "u")

    both = combine([router, answer])
    assert [u.model for u in both.calls] == [HAIKU, OPUS]
    assert both.cost_usd(PRICES) == pytest.approx(
        (500 * 1 + 20 * 5 + 3000 * 4 + 400 * 20) / 1e6
    )

    assert len(combine([answer, answer]).calls) == 1
