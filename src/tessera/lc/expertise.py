"""Archetype B on the LangChain stack (Phase 5 plan §3.6).

B is structured-evidence ranking, not document retrieval. Its LangChain
counterpart, ``PeopleRetriever``, is a ``BaseRetriever`` over the people
collection that returns ``Document``s carrying each ``PersonMatch``'s
fields as metadata — but the ranking inside is native's ``find_experts``:
the evidence scoring, the intent re-weighting and the generation floors
have **no framework equivalent**, so they are reused as plain functions.
Knowing which layers have no counterpart is part of understanding the
framework.

``find_experts_step`` puts it in ``NativePipeline``'s ``find_experts_fn``
slot: the retriever runs (and traces) as a LangChain retriever, and the
pipeline gets native's ``ExpertiseResult``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from tessera.retrieval.expertise import TOP_K, ExpertiseResult, find_experts
from tessera.store.base import ExpertiseStore


class PeopleRetriever(BaseRetriever):
    """``find_experts`` as a LangChain retriever."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedder: Any
    store: Any
    k: int = TOP_K
    _results: dict[str, ExpertiseResult] = PrivateAttr(default_factory=dict)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[LCDocument]:
        found = find_experts(query, self.embedder, self.store, k=self.k)
        self._results[str(run_manager.run_id)] = found
        return [
            LCDocument(
                page_content=f"{m.person.name} — {m.person.title}, {m.person.practice}",
                metadata={
                    "person_id": m.person.person_id,
                    "evidence_score": round(m.evidence_score, 4),
                    "rank_score": round(m.rank_score, 4),
                    "evidenced": m.is_evidenced,
                    "evidence": [e.description for e in m.evidence],
                },
            )
            for m in found.matches
        ]

    def experts(self, query: str) -> ExpertiseResult:
        """Run as a retriever (traced) and return native's result."""
        run_id = uuid.uuid4()
        self.invoke(query, run_id=run_id)
        return self._results.pop(str(run_id))


def find_experts_step() -> Callable[..., ExpertiseResult]:
    """``NativePipeline``'s ``find_experts_fn`` over a ``PeopleRetriever``."""

    def run(query: str, embedder: Any, store: ExpertiseStore, *, k: int = TOP_K) -> ExpertiseResult:
        return PeopleRetriever(embedder=embedder, store=store, k=k).experts(query)

    return run
