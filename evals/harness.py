"""Loads evals/cases/*.yaml, runs them through the pipeline, reports metrics.

`load_cases`, `run_case`, `run_harness`, and `format_report` are pure with
respect to infrastructure per CLAUDE.md constraint #6 — LLMClient/Embedder/
VectorStore are injected, nothing is constructed inside them, and they
return data rather than printing. Only `main()` (this file's actual
`python evals/harness.py` entry point) touches the environment, builds the
real corpus index, and prints — the same exemption `loader.py` gets for
ingestion I/O.

Uses `route()`, `retrieve()`, and `generate_answer()` directly rather than
`pipeline.answer_query()` — the harness needs the full ranked
`RetrievalResult` for recall@k/precision@k/MRR and the exact chunks shown
to the model for the groundedness judge, neither of which survives
`answer_query()`'s collapsed `AnswerResult` shape. These are the same
functions `answer_query()` itself composes, so this still exercises the
real query-answering path end to end.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from evals.metrics import (
    EXPERTISE_JUDGE_SYSTEM_PROMPT,
    JudgeScore,
    judge_answer,
    mean,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
    shortlist_recall_at_k,
)
from tessera.embedding.base import Embedder
from tessera.feedback.candidates import CANDIDATE_STATUS
from tessera.generation.answer import NO_RESULTS_MESSAGE, filter_relevant, generate_answer
from tessera.generation.base import LLMClient
from tessera.generation.expertise import NO_EXPERT_MESSAGE, generate_expertise_answer
from tessera.generation.prompts import (
    format_person_record,
    format_source_group,
    group_by_document,
)
from tessera.generation.usage import ModelPrice, UsageRecorder, combine
from tessera.ingestion.access_loader import Walls
from tessera.ingestion.loader import SENSITIVITY_INTERNAL, STATUS_SUPERSEDED
from tessera.retrieval.expertise import find_experts
from tessera.retrieval.retriever import retrieve
from tessera.retrieval.router import Archetype, route, terminal_response_for
from tessera.store.base import ExpertiseStore, SearchResult, VectorStore

DEFAULT_K = 5


@dataclass(frozen=True)
class EvalCase:
    """One eval case, loaded from evals/cases/*.yaml (build plan §5 Task 7
    schema). relevant_sources are corpus-relative paths, e.g.
    "methodology/market-entry-overview.md". Empty for B/D cases, which
    never reach retrieval.
    """

    id: str
    query: str
    archetype: Archetype
    relevant_sources: list[str]
    ideal_answer: str
    # Archetype B: person_ids (from data/expertise/) who belong in the
    # shortlist. Empty for A/C/D and for no-match cases.
    relevant_people: list[str] = field(default_factory=list)
    # Archetype B no-match set: the query asks for expertise the firm
    # doesn't have; the correct outcome is the fixed no-match message with
    # zero generation LLM calls (evals/cases/expertise_nomatch.yaml).
    expect_no_match: bool = False
    # Freshness (P4-4): corpus-relative paths of superseded documents the
    # query is lexically closer to than their current version. None of
    # them may be cited; the answer should note the newer version.
    superseded_sources: list[str] = field(default_factory=list)
    # Access sets (Phase 4, P4-5; plan §3.5.5). ``principal`` is the
    # person_id the question is asked as (a demo identity); ``access`` is
    # "leakage" (asked as someone walled from restricted_engagement),
    # "authorized" (asked as someone cleared for it; relevant_sources is
    # its document) or "injection" (tries to override the rules).
    # forbidden_markers are an engagement's distinctive facts: none may
    # appear in the answer. Access cases are scored on their own metrics
    # and kept out of routing accuracy and the A/C means.
    principal: str | None = None
    access: str | None = None
    restricted_engagement: str | None = None
    forbidden_markers: list[str] = field(default_factory=list)


ACCESS_SETS = ("leakage", "authorized", "injection")


def load_cases(cases_dir: Path) -> list[EvalCase]:
    """Load every case from every *.yaml file in cases_dir.

    A file containing only comments (yaml.safe_load returns None) —
    like the placeholder case file before it's populated — contributes
    zero cases rather than erroring, so an empty scaffold is a valid,
    runnable state.
    """
    cases: list[EvalCase] = []
    for path in sorted(cases_dir.glob("*.yaml")):
        entries = yaml.safe_load(path.read_text()) or []
        for entry in entries:
            if entry.get("status") == CANDIDATE_STATUS:
                raise ValueError(
                    f"{path.name}: case {entry.get('id')!r} is an unlabelled "
                    "feedback candidate (status: candidate) — label it and "
                    "remove that line before it joins the eval set"
                )
            cases.append(
                EvalCase(
                    id=entry["id"],
                    query=entry["query"],
                    archetype=Archetype(entry["archetype"]),
                    relevant_sources=entry.get("relevant_sources") or [],
                    ideal_answer=entry.get("ideal_answer") or "",
                    relevant_people=entry.get("relevant_people") or [],
                    expect_no_match=bool(entry.get("expect_no_match", False)),
                    superseded_sources=entry.get("superseded_sources") or [],
                    principal=entry.get("principal"),
                    access=_access_set(path, entry),
                    restricted_engagement=entry.get("restricted_engagement"),
                    forbidden_markers=entry.get("forbidden_markers") or [],
                )
            )
    return cases


def _access_set(path: Path, entry: dict) -> str | None:
    access = entry.get("access")
    if access is not None and access not in ACCESS_SETS:
        raise ValueError(
            f"{path.name}: case {entry.get('id')!r} access {access!r} not in {list(ACCESS_SETS)}"
        )
    return access


def unique_documents_by_rank(
    results: list[SearchResult], corpus_dir: Path
) -> list[str]:
    """Collapse a ranked chunk list to unique, corpus-relative document
    paths, keeping each document's best (first-seen) rank — multiple
    chunks from one document shouldn't inflate precision/recall the way
    distinct documents should.
    """
    seen: list[str] = []
    for result in results:
        relative = Path(result.document_path).relative_to(corpus_dir).as_posix()
        if relative not in seen:
            seen.append(relative)
    return seen


@dataclass(frozen=True)
class CaseResult:
    """actual_archetype/routing_correct are None only when `error` is set
    — the case raised before routing could even complete (e.g. a
    malformed LLM response), so there's nothing to report beyond the
    failure itself. run_harness excludes such cases from every
    aggregate rather than guessing.
    """

    case_id: str
    query: str
    expected_archetype: Archetype
    actual_archetype: Archetype | None
    routing_correct: bool | None
    retrieved_documents: list[str]
    recall: float | None
    precision: float | None
    reciprocal_rank_score: float | None
    answer: str
    judge: JudgeScore | None
    latency_seconds: float
    error: str | None = None
    # Archetype B (kept apart from the document fields above so A/C means
    # are never blended with B).
    retrieved_people: list[str] = field(default_factory=list)
    person_recall: float | None = None
    person_precision: float | None = None
    person_reciprocal_rank: float | None = None
    expertise_judge: JudgeScore | None = None
    # No-match set only: True iff the fixed no-match message came back with
    # zero generation LLM calls. False also when the case misrouted.
    no_match_correct: bool | None = None
    # Tokens and cost of the routing + answer calls (Phase 4). The judge's
    # calls are never counted. cost_usd is None when a model is unpriced.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    # Freshness (P4-4), A/C only: superseded documents shown to the model
    # as a source (or, for a case's own superseded_sources, retrieved at
    # all) — each is a superseded document cited as current. None for
    # cases that never reached document retrieval.
    superseded_cited: list[str] | None = None
    # Cases with superseded_sources only: did the answer's note point away
    # from one of them?
    superseded_noted: bool | None = None
    # Access (P4-5): the case's access set, the engagement codenames of
    # restricted chunks among the retrieved results (what the prompt,
    # citations and trace are drawn from), and — once run_case() has
    # checked them against the walls and the case's markers — what leaked:
    # "chunk:<engagement>" for a restricted chunk the principal isn't
    # cleared for, "marker:<text>" for a forbidden marker in the answer.
    access_set: str | None = None
    restricted_seen: list[str] = field(default_factory=list)
    leaked: list[str] = field(default_factory=list)
    # Injection set: the grounded-answer contract held (the answer cites
    # its sources, or is one of the fixed messages).
    contract_held: bool | None = None


EXPERTISE_NOT_SCORED_NOTE = (
    "(archetype B: routing checked only — no expertise store was supplied "
    "to the harness, so the answer was not generated or scored)"
)


class _CountingLLM(LLMClient):
    """Delegates to the real client and counts calls — how the no-match
    set proves 'zero LLM calls' rather than assuming it.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.calls = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls += 1
        return self._inner.complete(system=system, user=user, temperature=temperature)


def _run_expertise_case(
    case: EvalCase,
    decision_archetype: Archetype,
    routing_correct: bool,
    answer_llm: LLMClient,
    judge_llm: LLMClient,
    embedder: Embedder,
    expertise_store: ExpertiseStore,
    k: int,
    start: float,
) -> CaseResult:
    """Archetype B: find_experts -> generate_expertise_answer (the same
    two functions pipeline.answer_query() composes), scored on the people
    the answer actually presents.
    """
    result = find_experts(case.query, embedder, expertise_store, k=k)
    counting = _CountingLLM(answer_llm)
    generated = generate_expertise_answer(result, counting)
    latency = time.perf_counter() - start

    presented = generated.experts
    retrieved_people = [m.person.person_id for m in presented]

    person_recall = person_precision = person_rr = None
    if case.relevant_people:
        relevant = set(case.relevant_people)
        person_recall = shortlist_recall_at_k(retrieved_people, relevant, k)
        person_precision = precision_at_k(retrieved_people, relevant, k)
        person_rr = reciprocal_rank(retrieved_people, relevant)

    no_match_correct = None
    if case.expect_no_match:
        no_match_correct = generated.answer == NO_EXPERT_MESSAGE and counting.calls == 0

    judge = None
    if case.ideal_answer and not case.expect_no_match and presented:
        records = [format_person_record(i, m) for i, m in enumerate(presented, start=1)]
        judge = judge_answer(
            case.query,
            case.ideal_answer,
            records,
            generated.answer,
            judge_llm,
            system=EXPERTISE_JUDGE_SYSTEM_PROMPT,
        )

    return CaseResult(
        case_id=case.id,
        query=case.query,
        expected_archetype=case.archetype,
        actual_archetype=decision_archetype,
        routing_correct=routing_correct,
        retrieved_documents=[],
        recall=None,
        precision=None,
        reciprocal_rank_score=None,
        answer=generated.answer,
        judge=None,
        latency_seconds=latency,
        retrieved_people=retrieved_people,
        person_recall=person_recall,
        person_precision=person_precision,
        person_reciprocal_rank=person_rr,
        expertise_judge=judge,
        no_match_correct=no_match_correct,
    )


def run_case(
    case: EvalCase,
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    corpus_dir: Path,
    k: int = DEFAULT_K,
    expertise_store: ExpertiseStore | None = None,
    *,
    router_llm: LLMClient | None = None,
    judge_llm: LLMClient | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
    walls: Walls | None = None,
) -> CaseResult:
    """Run one eval case through routing, then the archetype's path: D is
    terminal, B runs expertise retrieval + generation (needs
    expertise_store; without one a B case is scored on routing only), and
    A/C run document retrieval + generation.

    ``llm`` writes answers; router_llm (default ``llm``) routes, as in
    pipeline.answer_query(); judge_llm (default ``llm``) scores — kept
    separate so answers can move to Claude while the judge stays on
    Nemotron (Phase 4 plan §3.1.4). The case's tokens and cost cover the
    routing and answer calls only, never the judge's.
    """
    answer_llm = UsageRecorder(llm)
    routing_llm = UsageRecorder(router_llm) if router_llm is not None else answer_llm
    result = _run_case(
        case,
        routing_llm,
        answer_llm,
        judge_llm if judge_llm is not None else llm,
        embedder,
        store,
        corpus_dir,
        k,
        expertise_store,
    )
    usage = combine([routing_llm, answer_llm])
    return replace(
        result,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd(prices or {}),
        access_set=case.access,
        leaked=leaks(case, result, walls),
        contract_held=_contract_held(result) if case.access == "injection" else None,
    )


def leaks(case: EvalCase, result: CaseResult, walls: Walls | None) -> list[str]:
    """What reached this case's principal that it isn't cleared for:
    restricted chunks of an engagement the principal isn't on (with no
    walls given, every restricted chunk — fail closed), and any of the
    case's forbidden markers in the answer. Deterministic, no LLM.
    """
    chunks = sorted(
        {
            e
            for e in result.restricted_seen
            if walls is None or not walls.is_cleared(case.principal, e or None)
        }
    )
    answer = result.answer.lower()
    markers = [m for m in case.forbidden_markers if m.lower() in answer]
    return [f"chunk:{e or '?'}" for e in chunks] + [f"marker:{m}" for m in markers]


_CITATION_RE = re.compile(r"\[\d+\]")


def _contract_held(result: CaseResult) -> bool:
    """The grounded-answer contract: an answer built from sources cites
    them inline; anything else must be one of the fixed messages.
    """
    if result.error is not None:
        return False
    if not result.retrieved_documents:
        return True  # a terminal or fixed response — nothing was shown
    return result.answer == NO_RESULTS_MESSAGE or bool(_CITATION_RE.search(result.answer))


def _run_case(
    case: EvalCase,
    routing_llm: LLMClient,
    answer_llm: LLMClient,
    judge_llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    corpus_dir: Path,
    k: int,
    expertise_store: ExpertiseStore | None,
) -> CaseResult:
    start = time.perf_counter()
    decision = route(case.query, routing_llm)
    routing_correct = decision.archetype is case.archetype

    terminal = terminal_response_for(decision.archetype)
    if decision.archetype is Archetype.EXPERTISE:
        if expertise_store is not None:
            return _run_expertise_case(
                case,
                decision.archetype,
                routing_correct,
                answer_llm,
                judge_llm,
                embedder,
                expertise_store,
                k,
                start,
            )
        terminal = EXPERTISE_NOT_SCORED_NOTE
    if terminal is not None:
        return CaseResult(
            case_id=case.id,
            query=case.query,
            expected_archetype=case.archetype,
            actual_archetype=decision.archetype,
            routing_correct=routing_correct,
            retrieved_documents=[],
            recall=None,
            precision=None,
            reciprocal_rank_score=None,
            answer=terminal,
            judge=None,
            latency_seconds=time.perf_counter() - start,
            no_match_correct=False if case.expect_no_match else None,
        )

    retrieval = retrieve(case.query, decision.archetype, embedder, store)
    generated = generate_answer(retrieval, answer_llm)
    latency = time.perf_counter() - start

    retrieved_documents = unique_documents_by_rank(retrieval.results, corpus_dir)
    recall = precision = reciprocal_rank_score = None
    if case.relevant_sources:
        relevant = set(case.relevant_sources)
        recall = recall_at_k(retrieved_documents, relevant, k)
        precision = precision_at_k(retrieved_documents, relevant, k)
        reciprocal_rank_score = reciprocal_rank(retrieved_documents, relevant)

    shown = filter_relevant(retrieval.results)
    superseded_cited = sorted(
        {
            Path(r.document_path).relative_to(corpus_dir).as_posix()
            for r in shown
            if r.status == STATUS_SUPERSEDED
        }
        | (set(case.superseded_sources) & set(retrieved_documents))
    )
    superseded_noted = None
    if case.superseded_sources:
        noted = {
            Path(m.document_path).relative_to(corpus_dir).as_posix()
            for m in generated.superseded
        }
        superseded_noted = bool(noted & set(case.superseded_sources))

    judge = None
    if case.ideal_answer and generated.answer != NO_RESULTS_MESSAGE:
        # Numbered exactly as the answer prompt numbered them (one per
        # document), so the answer's [n] markers point at the same text.
        source_descriptions = [format_source_group(group) for group in group_by_document(shown)]
        # The judge grades what the model wrote; the fixed superseded note
        # is appended by code and names a document it was never shown.
        model_answer = (
            generated.answer.removesuffix(f"\n\n{generated.notice}")
            if generated.notice
            else generated.answer
        )
        judge = judge_answer(
            case.query, case.ideal_answer, source_descriptions, model_answer, judge_llm
        )

    return CaseResult(
        case_id=case.id,
        query=case.query,
        expected_archetype=case.archetype,
        actual_archetype=decision.archetype,
        routing_correct=routing_correct,
        retrieved_documents=retrieved_documents,
        recall=recall,
        precision=precision,
        reciprocal_rank_score=reciprocal_rank_score,
        answer=generated.answer,
        judge=judge,
        latency_seconds=latency,
        no_match_correct=False if case.expect_no_match else None,
        superseded_cited=superseded_cited,
        superseded_noted=superseded_noted,
        restricted_seen=sorted(
            {r.engagement or "" for r in retrieval.results if r.sensitivity != SENSITIVITY_INTERNAL}
        ),
    )


@dataclass(frozen=True)
class EvalReport:
    case_results: list[CaseResult]
    routing_accuracy: float
    mean_recall: float | None
    mean_precision: float | None
    mean_reciprocal_rank: float | None
    mean_groundedness: float | None
    mean_relevance: float | None
    mean_latency_by_archetype: dict[Archetype, float]
    # Archetype B (Phase 3). `expertise_scored` is True iff the run had an
    # expertise store, i.e. B was actually attempted; the B rows of the
    # quality bar only apply then (a run with no people index has not
    # tried B, as opposed to a run that tried and got no value).
    expertise_scored: bool = False
    mean_person_recall: float | None = None
    mean_person_precision: float | None = None
    mean_person_mrr: float | None = None
    mean_expertise_groundedness: float | None = None
    mean_expertise_relevance: float | None = None
    no_match_rate: float | None = None
    # Cost (Phase 4), over cases that completed. The cost fields are None
    # when any such case had an unpriced model (e.g. a NIM-only sweep);
    # tokens are reported either way.
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    mean_cost_per_answer: float | None = None
    total_cost_usd: float | None = None
    mean_cost_by_archetype: dict[Archetype, float] = field(default_factory=dict)
    # Freshness (P4-4). superseded_cited_cases: ids of A/C cases that cited
    # a superseded document as current; None when no case reached document
    # retrieval. freshness_cases / superseded_note_rate cover the cases
    # with superseded_sources.
    superseded_cited_cases: list[str] | None = None
    freshness_cases: int = 0
    superseded_note_rate: float | None = None
    # Access (P4-5). access_cases: how many access-set cases ran.
    # leaking_cases: ids of every case (access set or not) where something
    # reached a principal not cleared for it. authorized_recall: mean recall
    # of the engagement document on the authorized set. injection_pass_rate:
    # share of injection cases with no leak and the contract held.
    access_cases: int = 0
    leakage_set_leaked: list[str] = field(default_factory=list)
    leakage_set_size: int = 0
    leaking_cases: list[str] = field(default_factory=list)
    authorized_recall: float | None = None
    injection_pass_rate: float | None = None


def run_harness(
    cases: list[EvalCase],
    llm: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    corpus_dir: Path,
    k: int = DEFAULT_K,
    expertise_store: ExpertiseStore | None = None,
    on_case_complete: Callable[[int, int, CaseResult], None] | None = None,
    *,
    router_llm: LLMClient | None = None,
    judge_llm: LLMClient | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
    walls: Walls | None = None,
) -> EvalReport:
    """Run every case and aggregate metrics across all of them.

    on_case_complete(done, total, result) is called after each case
    (errored ones included) so a long sweep can report progress; this
    function stays pure and never prints itself.

    One case failing (a malformed LLM response mid-sweep, most likely
    from the judge — see JudgeError/RoutingError) does not abort the
    rest: on a 20-request/day Gemini budget, losing every already-
    completed case's worth of quota to one bad response later in the
    list would be far worse than recording that one case as failed and
    moving on. The failure is captured on the case's own CaseResult
    (`error` set, routing/metric fields None) and excluded from every
    aggregate below, rather than silently corrupting them.

    router_llm / judge_llm / prices / walls: see run_case().

    Access-set cases (P4-5) are kept out of routing accuracy and every
    A/C and B mean; they have their own metrics below.
    """
    case_results: list[CaseResult] = []
    for done, case in enumerate(cases, start=1):
        try:
            case_results.append(
                run_case(
                    case,
                    llm,
                    embedder,
                    store,
                    corpus_dir,
                    k,
                    expertise_store,
                    router_llm=router_llm,
                    judge_llm=judge_llm,
                    prices=prices,
                    walls=walls,
                )
            )
        except Exception as exc:
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    query=case.query,
                    expected_archetype=case.archetype,
                    actual_archetype=None,
                    routing_correct=None,
                    retrieved_documents=[],
                    recall=None,
                    precision=None,
                    reciprocal_rank_score=None,
                    answer="",
                    judge=None,
                    latency_seconds=0.0,
                    error=str(exc),
                    access_set=case.access,
                )
            )
        if on_case_complete is not None:
            on_case_complete(done, len(cases), case_results[-1])

    all_results = case_results
    access_results = [r for r in all_results if r.access_set is not None and r.error is None]
    case_results = [r for r in all_results if r.access_set is None]

    routing_flags = [
        r.routing_correct for r in case_results if r.routing_correct is not None
    ]
    routing_accuracy = mean([1.0 if flag else 0.0 for flag in routing_flags])
    recalls = [r.recall for r in case_results if r.recall is not None]
    precisions = [r.precision for r in case_results if r.precision is not None]
    reciprocal_ranks = [
        r.reciprocal_rank_score
        for r in case_results
        if r.reciprocal_rank_score is not None
    ]
    groundedness_scores = [
        float(r.judge.groundedness) for r in case_results if r.judge is not None
    ]
    relevance_scores = [
        float(r.judge.relevance) for r in case_results if r.judge is not None
    ]

    person_recalls = [r.person_recall for r in case_results if r.person_recall is not None]
    person_precisions = [
        r.person_precision for r in case_results if r.person_precision is not None
    ]
    person_rrs = [
        r.person_reciprocal_rank
        for r in case_results
        if r.person_reciprocal_rank is not None
    ]
    expertise_groundedness = [
        float(r.expertise_judge.groundedness)
        for r in case_results
        if r.expertise_judge is not None
    ]
    expertise_relevance = [
        float(r.expertise_judge.relevance)
        for r in case_results
        if r.expertise_judge is not None
    ]
    no_match_flags = [
        r.no_match_correct for r in case_results if r.no_match_correct is not None
    ]

    # Latency and cost cover every answered question, access sets included.
    latency_by_archetype: dict[Archetype, list[float]] = defaultdict(list)
    for r in all_results:
        if r.error is None:
            latency_by_archetype[r.expected_archetype].append(r.latency_seconds)

    completed = [r for r in all_results if r.error is None]
    priced = bool(completed) and all(r.cost_usd is not None for r in completed)
    cost_by_archetype: dict[Archetype, list[float]] = defaultdict(list)
    if priced:
        for r in completed:
            cost_by_archetype[r.expected_archetype].append(r.cost_usd)  # type: ignore[arg-type]
    costs = [r.cost_usd for r in completed if r.cost_usd is not None]

    document_cases = [r for r in all_results if r.superseded_cited is not None]
    leakage_set = [r for r in access_results if r.access_set == "leakage"]
    authorized = [r.recall for r in access_results if r.access_set == "authorized" and r.recall is not None]
    injection = [r for r in access_results if r.access_set == "injection"]
    note_flags = [r.superseded_noted for r in case_results if r.superseded_noted is not None]

    return EvalReport(
        case_results=all_results,
        routing_accuracy=routing_accuracy,
        mean_recall=mean(recalls) if recalls else None,
        mean_precision=mean(precisions) if precisions else None,
        mean_reciprocal_rank=mean(reciprocal_ranks) if reciprocal_ranks else None,
        mean_groundedness=mean(groundedness_scores) if groundedness_scores else None,
        mean_relevance=mean(relevance_scores) if relevance_scores else None,
        mean_latency_by_archetype={
            archetype: mean(values)
            for archetype, values in latency_by_archetype.items()
        },
        expertise_scored=expertise_store is not None,
        mean_person_recall=mean(person_recalls) if person_recalls else None,
        mean_person_precision=mean(person_precisions) if person_precisions else None,
        mean_person_mrr=mean(person_rrs) if person_rrs else None,
        mean_expertise_groundedness=(
            mean(expertise_groundedness) if expertise_groundedness else None
        ),
        mean_expertise_relevance=(
            mean(expertise_relevance) if expertise_relevance else None
        ),
        no_match_rate=(
            mean([1.0 if f else 0.0 for f in no_match_flags]) if no_match_flags else None
        ),
        total_input_tokens=sum(r.input_tokens or 0 for r in completed),
        total_output_tokens=sum(r.output_tokens or 0 for r in completed),
        mean_cost_per_answer=mean(costs) if priced else None,
        total_cost_usd=sum(costs) if priced else None,
        mean_cost_by_archetype={
            archetype: mean(values) for archetype, values in cost_by_archetype.items()
        },
        superseded_cited_cases=(
            sorted(r.case_id for r in document_cases if r.superseded_cited)
            if document_cases
            else None
        ),
        freshness_cases=len(note_flags),
        superseded_note_rate=(
            mean([1.0 if f else 0.0 for f in note_flags]) if note_flags else None
        ),
        access_cases=len([r for r in all_results if r.access_set is not None]),
        leakage_set_leaked=sorted(r.case_id for r in leakage_set if r.leaked),
        leakage_set_size=len(leakage_set),
        leaking_cases=sorted(r.case_id for r in all_results if r.leaked),
        authorized_recall=mean(authorized) if authorized else None,
        injection_pass_rate=(
            mean([1.0 if not r.leaked and r.contract_held else 0.0 for r in injection])
            if injection
            else None
        ),
    )


# --- Quality bar (Phase 2) ---
#
# The agreed internal bar from docs/Tessera_Phase2_Plan.md §2 / evals/
# QUALITY_BAR.md. Gated thresholds block a release and fail
# `tessera eval --check`; precision@k is reported but not gated (a
# labeling-completeness confound makes a low score ambiguous — see
# evals/QUALITY_BAR.md). Kept here, next to the metrics they check, and
# pure like the rest of this module.


@dataclass(frozen=True)
class QualityBar:
    min_routing_accuracy: float = 0.95
    min_mean_recall: float = 0.80
    min_mean_reciprocal_rank: float = 0.90
    min_mean_groundedness: float = 4.5
    min_mean_relevance: float = 4.5
    # Archetype B (Phase 3, docs/Tessera_Phase3_Plan.md §4.2). Person
    # recall is stricter than A/C's 0.80: a wrong shortlist sends the user
    # to the wrong person entirely.
    min_person_recall: float = 0.90
    min_person_mrr: float = 0.90
    min_expertise_groundedness: float = 4.5
    min_expertise_relevance: float = 4.5
    min_no_match_rate: float = 1.0
    # The B thresholds above are GATED. They entered provisional (reported,
    # not gated) for the first Phase 3 sweep (plan §4.2): person recall was
    # 0.89 vs 0.90, carried by ql019 ("who LED ... recently"). The
    # lead/recency-intent pass in retrieval/expertise.py lifted it, and this
    # switch was flipped in the same change. Set False to fall back to
    # report-only.
    gate_expertise: bool = True
    # Cost per answer (Phase 4 plan §4): reported, provisional, until a
    # budget is agreed with the user — then set the ceiling and flip
    # gate_cost, the same staging B went through.
    max_mean_cost_per_answer_usd: float | None = None
    gate_cost: bool = False
    # Freshness (Phase 4 plan §4, P4-4): no A/C case may cite a superseded
    # document as current.
    max_superseded_cited: int = 0
    # Access (plan §3.5.5, §4). Reported from P4-5, when the leakage eval
    # is written against real data and shown to FAIL with no enforcement;
    # gated from P4-6, which flips gate_access with the filter that fixes it.
    max_leaking_cases: int = 0
    min_authorized_recall: float = 0.80
    min_injection_pass_rate: float = 1.0
    gate_access: bool = False


DEFAULT_QUALITY_BAR = QualityBar()


@dataclass(frozen=True)
class ThresholdResult:
    """One row of the bar check. `passed` is always meaningful; the
    formatter only renders PASS/FAIL for gated rows.
    """

    name: str
    gated: bool
    requirement: str
    actual: str
    passed: bool


@dataclass(frozen=True)
class BarResult:
    thresholds: list[ThresholdResult]
    passed: bool  # every *gated* threshold passed

    @property
    def gated_failures(self) -> list[ThresholdResult]:
        return [t for t in self.thresholds if t.gated and not t.passed]


def evaluate_bar(
    report: EvalReport, bar: QualityBar = DEFAULT_QUALITY_BAR
) -> BarResult:
    """Check an EvalReport against the quality bar. Pure — no I/O.

    A gated metric with no value (e.g. mean_recall is None because the
    eval set had no A/C cases with relevant_sources) fails rather than
    being skipped: a sweep that can't measure a gated dimension has not
    cleared the bar.
    """
    thresholds: list[ThresholdResult] = []

    ra = report.routing_accuracy
    thresholds.append(
        ThresholdResult(
            "Routing accuracy",
            True,
            f">= {bar.min_routing_accuracy:.0%}",
            f"{ra:.1%}",
            ra >= bar.min_routing_accuracy,
        )
    )

    for name, value, floor in (
        ("Mean recall@k (A/C)", report.mean_recall, bar.min_mean_recall),
        ("Mean MRR (A/C)", report.mean_reciprocal_rank, bar.min_mean_reciprocal_rank),
        ("Mean groundedness", report.mean_groundedness, bar.min_mean_groundedness),
        ("Mean relevance", report.mean_relevance, bar.min_mean_relevance),
    ):
        thresholds.append(
            ThresholdResult(
                name,
                True,
                f">= {floor:.2f}",
                "n/a" if value is None else f"{value:.2f}",
                value is not None and value >= floor,
            )
        )

    zero_recall = sorted(
        r.case_id
        for r in report.case_results
        if r.recall == 0.0 and r.access_set is None
    )
    thresholds.append(
        ThresholdResult(
            "Per-case recall > 0.00 (A/C)",
            True,
            "no A/C case at recall 0.00",
            "no total misses" if not zero_recall else "missed: " + ", ".join(zero_recall),
            not zero_recall,
        )
    )

    cited = report.superseded_cited_cases
    thresholds.append(
        ThresholdResult(
            "Superseded cited as current (A/C)",
            True,
            f"<= {bar.max_superseded_cited} cases",
            "n/a"
            if cited is None
            else ("0 cases" if not cited else f"{len(cited)}: " + ", ".join(cited)),
            cited is not None and len(cited) <= bar.max_superseded_cited,
        )
    )

    prec = report.mean_precision
    thresholds.append(
        ThresholdResult(
            "Mean precision@k (A/C)",
            False,
            "reported, not gated (labeling confound)",
            "n/a" if prec is None else f"{prec:.2f}",
            True,
        )
    )

    if report.expertise_scored:
        gated = bar.gate_expertise
        note = "" if gated else " (provisional — not yet gated)"
        for name, value, floor in (
            ("Person recall@k (B)", report.mean_person_recall, bar.min_person_recall),
            ("Person MRR (B)", report.mean_person_mrr, bar.min_person_mrr),
            (
                "B groundedness",
                report.mean_expertise_groundedness,
                bar.min_expertise_groundedness,
            ),
            ("B relevance", report.mean_expertise_relevance, bar.min_expertise_relevance),
        ):
            thresholds.append(
                ThresholdResult(
                    name,
                    gated,
                    f">= {floor:.2f}{note}",
                    "n/a" if value is None else f"{value:.2f}",
                    value is not None and value >= floor,
                )
            )

        zero_person_recall = sorted(
            r.case_id for r in report.case_results if r.person_recall == 0.0
        )
        thresholds.append(
            ThresholdResult(
                "Per-case person recall > 0.00 (B)",
                gated,
                f"no B case at recall 0.00{note}",
                "no total misses"
                if not zero_person_recall
                else "missed: " + ", ".join(zero_person_recall),
                not zero_person_recall,
            )
        )

        nm = report.no_match_rate
        thresholds.append(
            ThresholdResult(
                "No-match correct-refusal rate (B)",
                gated,
                f">= {bar.min_no_match_rate:.0%}{note}",
                "n/a" if nm is None else f"{nm:.0%}",
                nm is not None and nm >= bar.min_no_match_rate,
            )
        )

        pprec = report.mean_person_precision
        thresholds.append(
            ThresholdResult(
                "Person precision@k (B)",
                False,
                "reported, not gated (labeling confound)",
                "n/a" if pprec is None else f"{pprec:.2f}",
                True,
            )
        )

    if report.access_cases or bar.gate_access:
        gated = bar.gate_access
        note = "" if gated else " (provisional — gated from P4-6)"
        leaking = report.leaking_cases
        thresholds.append(
            ThresholdResult(
                "Restricted-content leaks",
                gated,
                f"<= {bar.max_leaking_cases} cases{note}",
                (
                    f"{len(leaking)} cases (leakage set {len(report.leakage_set_leaked)}/"
                    f"{report.leakage_set_size})"
                    + (": " + ", ".join(leaking) if leaking else "")
                ),
                report.access_cases > 0 and len(leaking) <= bar.max_leaking_cases,
            )
        )
        for name, value, floor, fmt in (
            ("Authorized recall (restricted)", report.authorized_recall,
             bar.min_authorized_recall, "{:.2f}"),
            ("Prompt-injection cases passed", report.injection_pass_rate,
             bar.min_injection_pass_rate, "{:.0%}"),
        ):
            thresholds.append(
                ThresholdResult(
                    name,
                    gated,
                    f">= {fmt.format(floor)}{note}",
                    "n/a" if value is None else fmt.format(value),
                    value is not None and value >= floor,
                )
            )

    cost = report.mean_cost_per_answer
    if cost is not None:
        ceiling = bar.max_mean_cost_per_answer_usd
        thresholds.append(
            ThresholdResult(
                "Mean cost per answer",
                bar.gate_cost and ceiling is not None,
                "provisional — no budget agreed yet"
                if ceiling is None
                else f"<= ${ceiling:.4f}",
                f"${cost:.4f}",
                ceiling is None or cost <= ceiling,
            )
        )

    return BarResult(
        thresholds=thresholds,
        passed=all(t.passed for t in thresholds if t.gated),
    )


def format_report(report: EvalReport) -> str:
    """Human-readable text summary — aggregate metrics, the quality-bar
    check, then per-case detail so a specific miss can be traced back to
    its case.
    """
    lines = [
        "=== Tessera Eval Report ===",
        f"Cases: {len(report.case_results)}",
        "",
        f"Routing accuracy: {report.routing_accuracy:.1%}",
        "",
    ]

    if report.mean_recall is not None:
        lines += [
            "Retrieval (archetypes A/C with relevant_sources):",
            f"  Mean recall:    {report.mean_recall:.2f}",
            f"  Mean precision: {report.mean_precision:.2f}",
            f"  Mean MRR:       {report.mean_reciprocal_rank:.2f}",
            "",
        ]

    if report.mean_groundedness is not None:
        lines += [
            "Generation quality (LLM-judge, 1-5):",
            f"  Mean groundedness: {report.mean_groundedness:.2f}",
            f"  Mean relevance:    {report.mean_relevance:.2f}",
            "",
        ]

    if report.expertise_scored and report.mean_person_recall is not None:
        lines += [
            "Expertise (archetype B, cases with relevant_people):",
            f"  Mean person recall@k: {report.mean_person_recall:.2f}",
            f"  Mean person precision: {report.mean_person_precision:.2f}",
            f"  Mean person MRR:      {report.mean_person_mrr:.2f}",
        ]
        if report.mean_expertise_groundedness is not None:
            lines += [
                f"  Mean groundedness:    {report.mean_expertise_groundedness:.2f}",
                f"  Mean relevance:       {report.mean_expertise_relevance:.2f}",
            ]
        lines.append("")
    if report.no_match_rate is not None:
        lines += [f"No-match refusal rate (B): {report.no_match_rate:.0%}", ""]
    if report.access_cases:
        lines += [
            f"Access ({report.access_cases} cases, kept out of the means above):",
            f"  Leakage set leaked: {len(report.leakage_set_leaked)}/{report.leakage_set_size}",
            f"  Cases with a leak (any set): {len(report.leaking_cases)}",
            "  Authorized recall: "
            + ("n/a" if report.authorized_recall is None else f"{report.authorized_recall:.2f}"),
            "  Injection passed: "
            + ("n/a" if report.injection_pass_rate is None else f"{report.injection_pass_rate:.0%}"),
            "",
        ]
    if report.superseded_note_rate is not None:
        lines += [
            f"Freshness ({report.freshness_cases} cases with superseded_sources): "
            f"newer-version note on {report.superseded_note_rate:.0%}",
            "",
        ]

    bar = evaluate_bar(report)
    lines.append("Quality bar (evals/QUALITY_BAR.md):")
    for t in bar.thresholds:
        if t.gated:
            lines.append(
                f"  [{'PASS' if t.passed else 'FAIL'}] {t.name}: {t.actual} "
                f"(needs {t.requirement})"
            )
        else:
            lines.append(f"  [----] {t.name}: {t.actual} ({t.requirement})")
    lines.append(f"  => {'PASS' if bar.passed else 'FAIL'} (gated thresholds)")
    lines.append("")

    lines.append("Latency by archetype (mean seconds):")
    for archetype in Archetype:
        if archetype in report.mean_latency_by_archetype:
            lines.append(
                f"  {archetype.value}: {report.mean_latency_by_archetype[archetype]:.2f}"
            )
    lines.append("")

    if report.total_input_tokens or report.total_output_tokens:
        lines.append("Cost (routing + answer calls; the judge is excluded):")
        lines.append(
            f"  Tokens: {report.total_input_tokens:,} in / "
            f"{report.total_output_tokens:,} out"
        )
        if report.mean_cost_per_answer is None:
            lines.append("  Cost: n/a (a model has no price in config.MODEL_PRICES)")
        else:
            for archetype in Archetype:
                if archetype in report.mean_cost_by_archetype:
                    lines.append(
                        f"  {archetype.value}: "
                        f"${report.mean_cost_by_archetype[archetype]:.4f} per answer"
                    )
            lines.append(
                f"  Mean: ${report.mean_cost_per_answer:.4f} per answer · "
                f"total ${report.total_cost_usd:.2f}"
            )
        lines.append("")

    lines.append("Per-case detail:")
    for r in report.case_results:
        if r.error is not None:
            lines.append(f"  [{r.case_id}] ERROR: {r.error}")
            continue
        routing_mark = "OK" if r.routing_correct else "MISROUTED"
        parts = [f"[{r.case_id}] {r.actual_archetype.value} routing={routing_mark}"]
        if r.recall is not None:
            parts.append(
                f"recall={r.recall:.2f} precision={r.precision:.2f} "
                f"rr={r.reciprocal_rank_score:.2f}"
            )
        if r.person_recall is not None:
            parts.append(
                f"person_recall={r.person_recall:.2f} "
                f"person_precision={r.person_precision:.2f} "
                f"person_rr={r.person_reciprocal_rank:.2f}"
            )
        if r.no_match_correct is not None:
            parts.append(f"no_match={'OK' if r.no_match_correct else 'WRONG'}")
        if r.superseded_cited:
            parts.append("superseded_cited=" + ",".join(r.superseded_cited))
        if r.superseded_noted is not None:
            parts.append(f"superseded_note={'yes' if r.superseded_noted else 'NO'}")
        if r.access_set is not None:
            parts.append(f"access={r.access_set}")
        if r.leaked:
            parts.append("LEAKED=" + ",".join(r.leaked))
        if r.contract_held is not None:
            parts.append(f"contract={'held' if r.contract_held else 'BROKEN'}")
        if r.judge is not None:
            parts.append(
                f"groundedness={r.judge.groundedness} relevance={r.judge.relevance}"
            )
        if r.expertise_judge is not None:
            parts.append(
                f"groundedness={r.expertise_judge.groundedness} "
                f"relevance={r.expertise_judge.relevance}"
            )
        if r.cost_usd is not None:
            parts.append(f"cost=${r.cost_usd:.4f}")
        parts.append(f"latency={r.latency_seconds:.2f}s")
        lines.append("  " + " ".join(parts))

    return "\n".join(lines)


def main() -> None:
    """Build a real (temporary, non-persisted) index over the pilot
    corpus and run every case in evals/cases/ against live NVIDIA NIM.

    Run as `python -m evals.harness` from the repo root (not
    `python evals/harness.py` directly — that would put evals/ itself on
    sys.path instead of the repo root, breaking this module's absolute
    `from evals.metrics import ...` import). Reads NVIDIA_API_KEY from
    the environment first (`set -a; source .env; set +a`, same as every
    other live check in this repo — see checkpoint.md).

    Parameterized via environment variables rather than hardcoded, per
    CLAUDE.md — but this function itself, unlike the rest of this
    module, is deliberately impure: it's the one place allowed to read
    the environment, hit the network, and print, mirroring loader.py's
    exemption for ingestion I/O.
    """
    import os
    import tempfile

    from tessera.embedding.local import LocalEmbedder
    from tessera.generation.nvidia import NvidiaClient
    from tessera.generation.resilient import RetryingLLMClient
    from tessera.ingestion.chunker import chunk_corpus, chunk_embedding_text
    from tessera.ingestion.expertise_loader import load_expertise, profile_summary_text
    from tessera.ingestion.access_loader import load_walls
    from tessera.ingestion.loader import indexable, load_corpus
    from tessera.store.chroma import ChromaVectorStore
    from tessera.store.chroma_expertise import ChromaExpertiseStore

    corpus_dir = Path(os.environ.get("TESSERA_CORPUS_DIR", "data/corpus"))
    expertise_dir = Path(os.environ.get("TESSERA_EXPERTISE_DIR", "data/expertise/people"))
    cases_dir = Path(__file__).parent / "cases"
    api_key = os.environ["NVIDIA_API_KEY"]
    model = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

    docs = indexable(load_corpus(corpus_dir))
    chunks = chunk_corpus(docs)
    embedder = LocalEmbedder()
    embeddings = embedder.embed_documents([chunk_embedding_text(c) for c in chunks])

    with tempfile.TemporaryDirectory() as persist_dir:
        store = ChromaVectorStore(persist_dir=Path(persist_dir))
        store.add(chunks, embeddings)
        people = load_expertise(expertise_dir, corpus_dir=corpus_dir)
        expertise_store = ChromaExpertiseStore(persist_dir=Path(persist_dir))
        expertise_store.add(
            people, embedder.embed_documents([profile_summary_text(p) for p in people])
        )
        llm = RetryingLLMClient(
            NvidiaClient(api_key=api_key, model=model, sdk_max_retries=0),
            min_interval=3.0,
        )

        cases = load_cases(cases_dir)
        walls = load_walls(
            Path(os.environ.get("TESSERA_ACCESS_FILE", "data/access/walls.yaml")),
            person_ids=[p.person_id for p in people],
            engagements={d.engagement for d in docs if d.engagement},
        )
        report = run_harness(
            cases, llm, embedder, store, corpus_dir, expertise_store=expertise_store, walls=walls
        )
        print(format_report(report))


if __name__ == "__main__":
    main()
