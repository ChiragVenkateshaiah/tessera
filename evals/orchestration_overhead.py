"""Orchestration overhead, measured with a fake LLM (Phase 5 plan §4.2,
P5-7): how much time the LangGraph orchestrators add per question over
``NativePipeline``, with every model call and store query faked so only
orchestration is timed.

    python -m evals.orchestration_overhead [--runs 2000]

Prints the median and p95 per run for each orchestrator on an A path (route,
retrieve, generate) and a D path (route, terminal), and the overhead over
native. Zero network calls.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

from tessera.embedding.base import Embedder
from tessera.generation.base import LLMClient
from tessera.generation.prompts import LOOKUP_ANSWER_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT
from tessera.pipeline import NativePipeline
from tessera.store.base import SearchResult, VectorStore


class _LLM(LLMClient):
    def __init__(self, archetype: str) -> None:
        self.archetype = archetype

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        if system == ROUTER_SYSTEM_PROMPT:
            return f'{{"archetype": "{self.archetype}", "reasoning": "x"}}'
        assert system == LOOKUP_ANSWER_SYSTEM_PROMPT
        return "The framework is in [1]."


class _Embedder(Embedder):
    @property
    def dimension(self) -> int:
        return 1

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class _Store(VectorStore):
    RESULT = SearchResult(
        chunk_id="methodology/pricing.md#1", text="text", score=0.9, document_path="methodology/pricing.md",
        document_title="Pricing", doc_type="methodology", industry="cross-industry", topics=["pricing"],
        date="2024-01-01", heading_path=("Overview",),
    )

    def add(self, chunks: object, embeddings: object) -> None:
        raise NotImplementedError

    def query(self, embedding: list[float], k: int, where: dict[str, object] | None = None) -> list[SearchResult]:
        return [self.RESULT]

    def count(self) -> int:
        return 1


def _time(run: Callable[[], object], runs: int) -> list[float]:
    for _ in range(min(50, runs)):  # warm-up
        run()
    out = []
    for _ in range(runs):
        start = time.perf_counter()
        run()
        out.append(time.perf_counter() - start)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Orchestration overhead with a fake LLM.")
    parser.add_argument("--runs", type=int, default=2000)
    args = parser.parse_args(argv)
    from tessera.lc.functional import FunctionalPipeline
    from tessera.lc.graph import LangGraphPipeline

    print(f"{'path':<5} {'orchestrator':<11} {'median ms':>10} {'p95 ms':>8} {'overhead (median)':>18}")
    for archetype in ("A", "D"):
        medians: dict[str, float] = {}
        for name, kind in (("native", NativePipeline), ("graph", LangGraphPipeline), ("functional", FunctionalPipeline)):
            pipeline = kind(_LLM(archetype), _Embedder(), _Store())
            times = sorted(_time(lambda: pipeline.run("pricing framework?"), args.runs))
            median = statistics.median(times) * 1000
            p95 = times[int(0.95 * len(times)) - 1] * 1000
            medians[name] = median
            over = "" if name == "native" else f"+{median - medians['native']:.2f} ms"
            print(f"{archetype:<5} {name:<11} {median:>10.3f} {p95:>8.3f} {over:>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
