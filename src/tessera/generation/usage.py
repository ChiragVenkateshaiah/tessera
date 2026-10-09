"""Token usage and cost per answer (Phase 4, plan §3.1.3).

`UsageRecorder` wraps an LLMClient and keeps the usage of every call made
through it. The pipeline makes one per answer, so usage comes back as
data on the answer — no recorder is threaded through the composition
roots, and nothing here prints or logs (CLAUDE.md constraint #6).
Turning tokens into dollars needs a price table, which the caller passes
in (config.MODEL_PRICES).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from tessera.generation.base import Completion, LLMClient, Usage


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class UsageSummary:
    """Every metered call behind one answer. ``unmetered_calls`` counts
    calls whose client couldn't report usage; any such call makes the cost
    unknown rather than understated.
    """

    calls: tuple[Usage, ...] = ()
    unmetered_calls: int = 0

    @property
    def input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.calls)

    def cost_usd(self, prices: Mapping[str, ModelPrice]) -> float | None:
        """Total cost, or None if a call was unmetered or its model has no
        price (e.g. the NIM free tier, which the table leaves out).
        """
        if self.unmetered_calls:
            return None
        total = 0.0
        for usage in self.calls:
            price = prices.get(usage.model)
            if price is None:
                return None
            total += price.cost(usage)
        return total


class UsageRecorder(LLMClient):
    """Delegates to ``inner`` and records the usage of every call."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._calls: list[Usage] = []
        self._unmetered = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(
        self, system: str, user: str, temperature: float = 0.0
    ) -> Completion:
        completion = self._inner.complete_with_usage(system, user, temperature)
        if completion.usage is None:
            self._unmetered += 1
        else:
            self._calls.append(completion.usage)
        return completion

    def add(self, usage: Usage | None) -> None:
        """Record a call made outside this recorder — a LangChain chain
        calling a chat model directly reports its usage here through a
        callback (Phase 5, P5-6), so it is metered with the rest."""
        if usage is None:
            self._unmetered += 1
        else:
            self._calls.append(usage)

    def summary(self) -> UsageSummary:
        return UsageSummary(calls=tuple(self._calls), unmetered_calls=self._unmetered)


def combine(recorders: Iterable[UsageRecorder]) -> UsageSummary:
    """One summary across several recorders (the same recorder passed
    twice is counted once).
    """
    calls: list[Usage] = []
    unmetered = 0
    seen: set[int] = set()
    for recorder in recorders:
        if id(recorder) in seen:
            continue
        seen.add(id(recorder))
        summary = recorder.summary()
        calls.extend(summary.calls)
        unmetered += summary.unmetered_calls
    return UsageSummary(calls=tuple(calls), unmetered_calls=unmetered)
