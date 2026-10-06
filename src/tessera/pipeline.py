"""query -> route -> retrieve -> generate.

The single entry point tying Tasks 4-6 together. Pure with respect to
infrastructure per CLAUDE.md constraint #6 — every dependency
(LLMClient, Embedder, VectorStore) is injected, nothing is constructed
here — so a future edge/routing layer can call this via a plain function
call, subprocess, or HTTP wrapper without this module changing.

Phase 5 (plan §3.1.2): the native composition is a ``Pipeline`` —
``NativePipeline.run(query, principal) -> PipelineRun`` — so the eval
harness can score any implementation through one interface. A
``PipelineRun`` is the ``AnswerResult`` plus what the harness scores (the
retrieval attempts, the chunks shown, the generator's output, routing and
generation usage kept apart). ``answer_query()`` is a thin wrapper over the
native pipeline with its signature unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

from tessera.embedding.base import Embedder
from tessera.generation.answer import (
    RELEVANCE_THRESHOLD,
    Citation,
    GeneratedAnswer,
    filter_relevant,
    generate_answer,
)
from tessera.generation.base import LLMClient
from tessera.generation.expertise import (
    EXPERTISE_PERSON_FLOOR,
    EXPERTISE_QUERY_FLOOR,
    filter_qualified,
    generate_expertise_answer,
)
from tessera.generation.usage import UsageRecorder, UsageSummary, combine
from tessera.principal import Principal
from tessera.retrieval.expertise import TOP_K as EXPERTISE_TOP_K
from tessera.retrieval.expertise import ExpertiseResult, find_experts
from tessera.retrieval.retriever import RetrievalResult, retrieve
from tessera.retrieval.router import (
    Archetype,
    RoutingDecision,
    route,
    terminal_response_for,
)
from tessera.store.base import ExpertiseStore, PersonMatch, SearchResult, VectorStore
from tessera.trace import RetrievedItem, Trace

EXPERTISE_UNAVAILABLE_MESSAGE = (
    "The expertise index hasn't been built, so I can't look up people yet."
)


@dataclass(frozen=True)
class AnswerResult:
    """The pipeline's single return shape, regardless of archetype.

    D carries a terminal message and no citations (never reaches
    retrieval/generation); A/C carry a generated answer with document
    citations; B carries a generated answer with the ranked people (and
    their evidence) it named in ``experts``. One shape means callers
    (CLI, eval harness) don't need to branch on archetype to know what
    they got back.
    """

    query: str
    archetype: Archetype
    answer: str
    citations: list[Citation]
    experts: list[PersonMatch] = field(default_factory=list)
    # Tokens of every LLM call behind this answer (routing included);
    # callers turn it into dollars with a price table (Phase 4).
    usage: UsageSummary = field(default_factory=UsageSummary)
    # How the answer was produced — the composition roots log it with a
    # trace_id (Phase 4, plan §3.2.1).
    trace: Trace = field(default_factory=lambda: Trace(route_reasoning=""))


@dataclass(frozen=True)
class PipelineRun:
    """One pipeline run: the ``AnswerResult`` plus what the eval harness
    scores (Phase 5 plan §3.1.2).

    ``retrievals`` is every document retrieval the run made, in order (the
    native stack makes at most one; a corrective loop or multi-query makes
    more), so leak checks see every attempt. ``shown`` is what cleared the
    relevance floor of the final attempt — what the answer model saw.
    ``generated`` is the document or expertise generator's output (None
    for a terminal answer), ``expertise`` the ranked shortlist behind a B
    answer. Routing and generation usage are always recorded apart, even
    when one client does both, so "zero generation calls" is readable.
    """

    answer: AnswerResult
    decision: RoutingDecision
    retrievals: tuple[RetrievalResult, ...] = ()
    shown: tuple[SearchResult, ...] = ()
    generated: GeneratedAnswer | None = None
    expertise: ExpertiseResult | None = None
    routing_usage: UsageSummary = field(default_factory=UsageSummary)
    generation_usage: UsageSummary = field(default_factory=UsageSummary)

    @property
    def generation_calls(self) -> int:
        """LLM calls made to generate the answer (routing excluded),
        metered or not."""
        return len(self.generation_usage.calls) + self.generation_usage.unmetered_calls


class Pipeline(Protocol):
    """Anything that answers a question as ``principal`` (None: internal
    documents only) and reports how. ``native`` is the first
    implementation; Phase 5 adds the LangChain stack.
    """

    def run(self, query: str, principal: Principal | None = None) -> PipelineRun: ...


# The steps a native run composes. Each is a module-level function by
# default, looked up when the run happens (not when the pipeline is
# built), so a test can monkeypatch this module and a composition root can
# pass a wrapped step (P5-3's @traceable) without editing core modules.
RouteFn = Callable[[str, LLMClient], RoutingDecision]
RetrieveFn = Callable[..., RetrievalResult]
GenerateFn = Callable[[RetrievalResult, LLMClient], GeneratedAnswer]
FindExpertsFn = Callable[..., ExpertiseResult]
GenerateExpertiseFn = Callable[[ExpertiseResult, LLMClient], GeneratedAnswer]


class NativePipeline:
    """The native stack: route, then a terminal response (D), people (B)
    or documents (A/C). Ports and step functions are constructor
    parameters; the steps default to the core functions.

    expertise_store is only needed for archetype B; without one a B query
    gets EXPERTISE_UNAVAILABLE_MESSAGE rather than an error. router_llm,
    when given, makes the routing call while ``llm`` writes the answer; by
    default one client does both (still metered apart).
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        store: VectorStore,
        expertise_store: ExpertiseStore | None = None,
        *,
        router_llm: LLMClient | None = None,
        expertise_k: int = EXPERTISE_TOP_K,
        route_fn: RouteFn | None = None,
        retrieve_fn: RetrieveFn | None = None,
        generate_fn: GenerateFn | None = None,
        find_experts_fn: FindExpertsFn | None = None,
        generate_expertise_fn: GenerateExpertiseFn | None = None,
    ) -> None:
        self._llm = llm
        self._router_llm = router_llm
        self._embedder = embedder
        self._store = store
        self._expertise_store = expertise_store
        self._expertise_k = expertise_k
        self._route = route_fn
        self._retrieve = retrieve_fn
        self._generate = generate_fn
        self._find_experts = find_experts_fn
        self._generate_expertise = generate_expertise_fn

    def run(self, query: str, principal: Principal | None = None) -> PipelineRun:
        routing = UsageRecorder(self._router_llm if self._router_llm is not None else self._llm)
        generation = UsageRecorder(self._llm)

        decision = (self._route or route)(query, routing)
        terminal = terminal_response_for(decision.archetype)
        if terminal is not None:
            return _finish(query, decision, principal, routing, generation,
                           answer=terminal, trace=Trace(decision.reasoning, fixed_response=True))

        if decision.archetype is Archetype.EXPERTISE:
            if self._expertise_store is None:
                return _finish(
                    query, decision, principal, routing, generation,
                    answer=EXPERTISE_UNAVAILABLE_MESSAGE,
                    trace=Trace(decision.reasoning, fixed_response=True),
                )
            found = (self._find_experts or find_experts)(
                query, self._embedder, self._expertise_store, k=self._expertise_k
            )
            generated = (self._generate_expertise or generate_expertise_answer)(found, generation)
            return _finish(
                query, decision, principal, routing, generation,
                answer=generated.answer,
                trace=expertise_trace(decision, found),
                experts=generated.experts,
                generated=generated,
                expertise=found,
            )

        retrieval = (self._retrieve or retrieve)(
            query, decision.archetype, self._embedder, self._store, principal=principal
        )
        generated = (self._generate or generate_answer)(retrieval, generation)
        shown = tuple(filter_relevant(retrieval.results))
        return _finish(
            query, decision, principal, routing, generation,
            answer=generated.answer,
            trace=document_trace(decision, retrieval, generated, shown),
            citations=generated.citations,
            generated=generated,
            retrievals=(retrieval,),
            shown=shown,
        )


def expertise_trace(decision: RoutingDecision, found: ExpertiseResult) -> Trace:
    """The trace of a B answer: every ranked person, and whether the
    generator was shown them (the same filter it applies)."""
    shown = {m.person.person_id for m in filter_qualified(found.matches)}
    return Trace(
        decision.reasoning,
        retrieved_kind="person",
        retrieved=tuple(
            RetrievedItem(m.person.person_id, m.evidence_score, m.person.person_id in shown)
            for m in found.matches
        ),
        floors={
            "expertise_query_floor": EXPERTISE_QUERY_FLOOR,
            "expertise_person_floor": EXPERTISE_PERSON_FLOOR,
        },
        fixed_response=not shown,
    )


def document_trace(
    decision: RoutingDecision,
    retrieval: RetrievalResult,
    generated: GeneratedAnswer,
    shown: tuple[SearchResult, ...],
) -> Trace:
    """The trace of an A/C answer: every retrieved chunk, whether it
    cleared the relevance floor, what the filters removed, and which
    superseded documents the answer's note points away from."""
    shown_ids = {r.chunk_id for r in shown}
    return Trace(
        decision.reasoning,
        retrieved_kind="chunk",
        retrieved=tuple(
            RetrievedItem(r.chunk_id, r.score, r.chunk_id in shown_ids, r.document_path)
            for r in retrieval.results
        ),
        floors={"relevance_threshold": RELEVANCE_THRESHOLD},
        fixed_response=not shown_ids,
        removed=dict(retrieval.removed),
        superseded=tuple(m.document_path for m in generated.superseded),
    )


def _finish(
    query: str,
    decision: RoutingDecision,
    principal: Principal | None,
    routing: UsageRecorder,
    generation: UsageRecorder,
    *,
    answer: str,
    trace: Trace,
    citations: list[Citation] | None = None,
    experts: list[PersonMatch] | None = None,
    generated: GeneratedAnswer | None = None,
    expertise: ExpertiseResult | None = None,
    retrievals: tuple[RetrievalResult, ...] = (),
    shown: tuple[SearchResult, ...] = (),
) -> PipelineRun:
    return PipelineRun(
        answer=AnswerResult(
            query=query,
            archetype=decision.archetype,
            answer=answer,
            citations=citations or [],
            experts=experts or [],
            usage=combine([routing, generation]),
            trace=replace(trace, principal=principal.person_id if principal else None),
        ),
        decision=decision,
        retrievals=retrievals,
        shown=shown,
        generated=generated,
        expertise=expertise,
        routing_usage=routing.summary(),
        generation_usage=generation.summary(),
    )


def answer_query(
    query: str,
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    expertise_store: ExpertiseStore | None = None,
    *,
    router_llm: LLMClient | None = None,
    principal: Principal | None = None,
) -> AnswerResult:
    """Run a query through the native pipeline: route, then return a
    terminal response (D), find and describe people (B), or retrieve and
    generate a grounded, cited answer from documents (A/C).

    expertise_store is only needed for archetype B; without one a B query
    gets EXPERTISE_UNAVAILABLE_MESSAGE rather than an error.

    router_llm, when given, makes the routing call (a cheap classification
    model) while ``llm`` writes the answer; by default one client does
    both.

    principal is who the question is asked as (a demo identity, resolved
    by the caller — Phase 4, plan §3.5.4): document retrieval returns
    internal documents plus restricted ones from engagements they are
    cleared for. None means internal documents only. The expertise path
    is unaffected — the people index holds no engagement or client data.
    """
    pipeline = NativePipeline(llm, embedder, store, expertise_store, router_llm=router_llm)
    return pipeline.run(query, principal).answer
