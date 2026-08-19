"""recall@k, precision@k, MRR, groundedness, relevance, routing accuracy,
latency.

Retrieval metrics (recall@k/precision@k/reciprocal_rank) and aggregation
(mean) are pure and deterministic — unit-tested directly. Groundedness
and relevance require an LLM-as-judge call and are inherently
non-deterministic; per CLAUDE.md's working conventions ("do not
over-test LLM outputs; that is what the eval harness is for"), only the
judge's response-parsing logic is unit-tested (with a fake LLM client),
not its judgment quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from tessera.generation.base import LLMClient


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top k retrieved.

    retrieved/relevant are corpus-relative document paths (not chunk
    ids) — callers are expected to have already collapsed a ranked chunk
    list to unique documents (see harness.unique_documents_by_rank).
    """
    if not relevant:
        raise ValueError("relevant must be non-empty to compute recall@k")
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k retrieved documents that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = len(set(top_k) & relevant)
    return hits / len(top_k)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant document retrieved, 0.0 if none."""
    for rank, document in enumerate(retrieved, start=1):
        if document in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --- Groundedness/relevance via LLM-as-judge ---

JUDGE_SYSTEM_PROMPT = """You are grading answers from Tessera, an internal knowledge assistant for a consulting firm. You will be shown a user's question, a description of what an ideal answer should cover, the sources the assistant was allowed to use, and the assistant's actual answer.

Score two dimensions, each 1-5:

groundedness: does every factual claim in the answer trace back to the provided sources, with no invention or claims beyond what the sources support? 5 = fully grounded, no unsupported claims. 1 = largely invented or contradicts the sources.

relevance: does the answer actually address what the ideal-answer description says it should cover? 5 = fully addresses it. 1 = misses it entirely.

Respond with strict JSON only — no markdown fences, no other text — in exactly this shape:
{"groundedness": 4, "relevance": 5, "reasoning": "one sentence explaining both scores"}
"""


def build_judge_user_prompt(
    query: str, ideal_answer: str, sources: list[str], answer: str
) -> str:
    formatted_sources = (
        "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))
        if sources
        else "(none)"
    )
    return (
        f"Question: {query}\n\n"
        f"Ideal answer should cover: {ideal_answer}\n\n"
        f"Sources the assistant could use:\n\n{formatted_sources}\n\n"
        f"Assistant's actual answer:\n\n{answer}"
    )


@dataclass(frozen=True)
class JudgeScore:
    groundedness: int
    relevance: int
    reasoning: str


class JudgeError(ValueError):
    """The LLM judge's response couldn't be parsed into a valid score."""


def judge_answer(
    query: str,
    ideal_answer: str,
    sources: list[str],
    answer: str,
    llm: LLMClient,
) -> JudgeScore:
    """Score a generated answer's groundedness and relevance via LLM-as-judge.

    Pure with respect to infrastructure per CLAUDE.md constraint #6: the
    LLMClient is injected, not constructed.
    """
    raw = llm.complete(
        system=JUDGE_SYSTEM_PROMPT,
        user=build_judge_user_prompt(query, ideal_answer, sources, answer),
    )
    try:
        parsed = json.loads(_strip_markdown_fence(raw))
        return JudgeScore(
            groundedness=int(parsed["groundedness"]),
            relevance=int(parsed["relevance"]),
            reasoning=parsed["reasoning"],
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise JudgeError(f"could not parse judge response: {raw!r}") from exc


def _strip_markdown_fence(text: str) -> str:
    """Some models wrap JSON in ```json ... ``` even when told not to —
    strip it rather than fail the parse over formatting.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped
