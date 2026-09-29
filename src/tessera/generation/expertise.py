"""Turns ranked people into a grounded, evidence-cited answer (Phase 3,
archetype B). The people analogue of ``answer.py``: same shape — a
relevance floor that short-circuits to a fixed message with zero LLM
calls when nothing qualifies — but citing person records and their
evidence rather than document chunks.
"""

from __future__ import annotations

from tessera.generation.answer import GeneratedAnswer
from tessera.generation.base import LLMClient
from tessera.generation.prompts import (
    EXPERTISE_ANSWER_SYSTEM_PROMPT,
    build_expertise_user_prompt,
)
from tessera.retrieval.expertise import ExpertiseResult
from tessera.retrieval.router import Archetype
from tessera.store.base import PersonMatch

# Two floors on retrieval/expertise.py's evidence_score, calibrated
# 2026-09-29 against the real dataset/embedder (19 probe queries):
#  - present topics: best match 2.09–4.36, weakest of the top 5 ≥ 1.37;
#  - absent topics (quantum computing, pastry, …): best match 0.00;
#  - adjacent-but-absent (HR compensation, legal contracts): best match
#    0.66 / 0.19 — spurious partial-topic hits.
# EXPERTISE_QUERY_FLOOR gates the whole answer: the best match must clear
# it or nobody is recommended (1.0 sits in the 0.66→2.09 gap).
# EXPERTISE_PERSON_FLOOR then drops people with no topical backing at all
# (evidence 0 — surfaced on profile similarity alone) but deliberately
# keeps weak, self-reported-only people once the query has real experts,
# so the answer can show them flagged rather than hide them.
EXPERTISE_QUERY_FLOOR = 1.0
EXPERTISE_PERSON_FLOOR = 0.05

NO_EXPERT_MESSAGE = (
    "We don't have an obvious expert on that in Meridian's expertise data — "
    "nobody in the people index has project or authorship evidence relevant "
    "enough to recommend."
)


def filter_qualified(
    matches: list[PersonMatch],
    query_floor: float = EXPERTISE_QUERY_FLOOR,
    person_floor: float = EXPERTISE_PERSON_FLOOR,
) -> list[PersonMatch]:
    """The people the model may be shown, in ranked order — empty unless
    the best match clears query_floor.

    Factored out of generate_expertise_answer() for the same reason as
    ``answer.filter_relevant``: the eval harness needs to know exactly who
    the model was shown.
    """
    if not matches or max(m.evidence_score for m in matches) < query_floor:
        return []
    return [m for m in matches if m.evidence_score >= person_floor]


def generate_expertise_answer(result: ExpertiseResult, llm: LLMClient) -> GeneratedAnswer:
    """Generate an evidence-cited answer naming the right people.

    Pure with respect to infrastructure per CLAUDE.md constraint #6: the
    LLMClient is injected. If the best match doesn't clear EXPERTISE_QUERY_FLOOR this
    returns NO_EXPERT_MESSAGE without spending an LLM call, so the
    refusal is guaranteed rather than left to the model (constraint #2 —
    grounded generation only).
    """
    qualified = filter_qualified(result.matches)
    if not qualified:
        return GeneratedAnswer(
            query=result.query,
            archetype=Archetype.EXPERTISE,
            answer=NO_EXPERT_MESSAGE,
            citations=[],
            experts=[],
        )

    answer = llm.complete(
        system=EXPERTISE_ANSWER_SYSTEM_PROMPT,
        user=build_expertise_user_prompt(result.query, qualified),
    )
    return GeneratedAnswer(
        query=result.query,
        archetype=Archetype.EXPERTISE,
        answer=answer,
        citations=[],
        experts=qualified,
    )
