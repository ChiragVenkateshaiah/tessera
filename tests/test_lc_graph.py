"""LangGraph orchestration (Phase 5 plan §3.7, P5-7 acceptance).

**Equivalence**, with every component native and the corrective loop off:
the graph's ``PipelineRun`` must equal ``NativePipeline``'s (``==``),
checked two ways:
- **on fakes**, for every path (A, C, A with nothing over the floor, B,
  B with nobody qualifying, B without a people index, D, a routing error),
  with the native client and with metered router/answer clients apart;
- **on replay**: native NIM responses recorded once for all 92 eval cases
  (``evals/replay.py``) are replayed into both pipelines over exactly
  searched indexes, and every run must match.

**Isolation**: two invocations running at the same time on one compiled
graph keep their usage apart (usage is per-node state, not a shared
recorder).

**Rules** (plan §3.2.2): the graph is compiled once without a
checkpointer, and its state never holds a client, store or recorder.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph", reason="the lc extra isn't installed")

from evals.harness import case_principal, load_cases  # noqa: E402
from evals.replay import ReplayLLM, build_exact_indexes, load_recording, load_walls_for_repo  # noqa: E402
from tessera.generation.base import Completion, LLMClient, Usage  # noqa: E402
from tessera.generation.prompts import (  # noqa: E402
    EXPERTISE_ANSWER_SYSTEM_PROMPT,
    LOOKUP_ANSWER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SYNTHESIS_ANSWER_SYSTEM_PROMPT,
)
from tessera.lc.functional import FunctionalPipeline  # noqa: E402
from tessera.lc.graph import GRAPH, LangGraphPipeline  # noqa: E402
from tessera.pipeline import NativePipeline, PipelineRun  # noqa: E402
from tessera.principal import Principal  # noqa: E402
from tessera.retrieval.router import RoutingError  # noqa: E402
from tests.test_expertise_generation import ScoredExpertiseStore, make_person, make_project  # noqa: E402
from tests.test_pipeline import (  # noqa: E402
    ExplodingVectorStore,
    FakeVectorStore,
    MeteredScripted,
    PricingEmbedder,
    ScriptedLLMClient,
    _result,
    _router_response,
)

REPO = Path(__file__).resolve().parents[1]
RECORDING = REPO / "tests" / "fixtures" / "replay" / "native-nim.json"

ANSWERS = {
    LOOKUP_ANSWER_SYSTEM_PROMPT: "The pricing framework is in [1].",
    SYNTHESIS_ANSWER_SYSTEM_PROMPT: "Across [1] and [2], pricing is value-based.",
    EXPERTISE_ANSWER_SYSTEM_PROMPT: "Ask Person c1 [1].",
}


FLOWS = [LangGraphPipeline, FunctionalPipeline]


@pytest.fixture(params=FLOWS, ids=["graph", "functional"])
def flow(request: pytest.FixtureRequest) -> Any:
    """Each LangGraph programming model: the StateGraph and the Functional API."""
    return request.param


def both(make: Any, query: str, principal: Principal | None = None, flow: Any = LangGraphPipeline) -> tuple[PipelineRun, PipelineRun]:
    """The same pipeline arguments, run natively and through ``flow``."""
    return make(NativePipeline).run(query, principal), make(flow).run(query, principal)


def scripted(archetype: str, model: str | None = None) -> LLMClient:
    responses = {ROUTER_SYSTEM_PROMPT: _router_response(archetype), **ANSWERS}
    return MeteredScripted(responses, model) if model else ScriptedLLMClient(responses)


DOCS = [_result("methodology/pricing.md", 0.9), _result("thought_leadership/pricing-2025.md", 0.8)]
EXPERT = make_person("c1", projects=[make_project("pricing"), make_project("pricing", 2026)])


PATHS: dict[str, Any] = {
    "A": lambda kind, llm, **kw: kind(llm, PricingEmbedder(), FakeVectorStore(DOCS), **kw),
    "C": lambda kind, llm, **kw: kind(llm, PricingEmbedder(), FakeVectorStore(DOCS), **kw),
    "A-nothing-over-the-floor": lambda kind, llm, **kw: kind(
        llm, PricingEmbedder(), FakeVectorStore([_result("methodology/other.md", 0.1)]), **kw
    ),
    "B": lambda kind, llm, **kw: kind(
        llm, PricingEmbedder(), ExplodingVectorStore([]), ScoredExpertiseStore([(EXPERT, 0.6)]), **kw
    ),
    "B-nobody-qualifies": lambda kind, llm, **kw: kind(
        llm, PricingEmbedder(), ExplodingVectorStore([]), ScoredExpertiseStore([(make_person("c1"), 0.3)]), **kw
    ),
    "B-no-people-index": lambda kind, llm, **kw: kind(llm, PricingEmbedder(), ExplodingVectorStore([]), **kw),
    "D": lambda kind, llm, **kw: kind(llm, PricingEmbedder(), ExplodingVectorStore([]), **kw),
}
ROUTED = {"A": "A", "C": "C", "A-nothing-over-the-floor": "A", "B": "B", "B-nobody-qualifies": "B",
          "B-no-people-index": "B", "D": "D"}


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("principal", [None, Principal("c0014"), Principal("partner", frozenset({"halcyon"}))],
                         ids=["internal-only", "walled", "cleared"])
def test_the_graph_equals_native_on_every_path(path: str, principal: Principal | None, flow: Any) -> None:
    native, graph = both(lambda kind: PATHS[path](kind, scripted(ROUTED[path])), "pricing framework?", principal, flow)
    assert graph == native
    assert graph.answer.archetype.value == ROUTED[path]


@pytest.mark.parametrize("path", list(PATHS))
def test_metered_router_and_answer_clients_stay_apart(path: str, flow: Any) -> None:
    """A separate metered router (the Gemini shape: Flash routes, Pro answers)."""
    native, graph = both(
        lambda kind: PATHS[path](
            kind, scripted(ROUTED[path], "answer-model"), router_llm=scripted(ROUTED[path], "router-model")
        ),
        "pricing framework?",
        flow=flow,
    )
    assert graph == native
    assert {u.model for u in graph.routing_usage.calls} == {"router-model"}
    assert {u.model for u in graph.generation_usage.calls} <= {"answer-model"}


def test_a_routing_error_propagates_the_same_way(flow: Any) -> None:
    llm = ScriptedLLMClient({ROUTER_SYSTEM_PROMPT: "not json"})
    with pytest.raises(RoutingError):
        NativePipeline(llm, PricingEmbedder(), FakeVectorStore(DOCS)).run("pricing?")
    with pytest.raises(RoutingError):
        flow(llm, PricingEmbedder(), FakeVectorStore(DOCS)).run("pricing?")


def test_steps_are_injected_per_invocation_like_native(flow: Any) -> None:
    """A composition root's step replacements reach the graph's nodes
    (through the runtime context, not the compiled graph)."""
    seen: list[str] = []

    def retrieve_fn(query: str, archetype: Any, embedder: Any, store: Any, *, principal: Any = None) -> Any:
        from tessera.retrieval.retriever import retrieve

        seen.append(query)
        return retrieve(query, archetype, embedder, store, principal=principal)

    native, graph = both(
        lambda kind: kind(scripted("A"), PricingEmbedder(), FakeVectorStore(DOCS), retrieve_fn=retrieve_fn),
        "pricing framework?",
        flow=flow,
    )
    assert graph == native
    assert seen == ["pricing framework?", "pricing framework?"]


# --- isolation ---


class BarrierLLM(LLMClient):
    """A metered fake that waits at a barrier on every call, so two
    invocations are provably inside the graph at the same time."""

    def __init__(self, inner: LLMClient, model: str, barrier: threading.Barrier) -> None:
        self._inner = inner
        self._model = model
        self._barrier = barrier

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Completion:
        self._barrier.wait(timeout=10)
        return Completion(self._inner.complete(system, user, temperature), Usage(self._model, 1, 1, 0.0))


def test_two_concurrent_invocations_keep_their_usage_apart(flow: Any) -> None:
    barrier = threading.Barrier(2)
    lookup = flow(BarrierLLM(scripted("A"), "model-a", barrier), PricingEmbedder(), FakeVectorStore(DOCS))
    synthesis = flow(BarrierLLM(scripted("C"), "model-c", barrier), PricingEmbedder(), FakeVectorStore(DOCS))
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(lookup.run, "pricing framework?")
        c = pool.submit(synthesis.run, "brief me on pricing")
        runs = {"a": a.result(timeout=30), "c": c.result(timeout=30)}
    for name, run in runs.items():
        models = {u.model for u in (*run.routing_usage.calls, *run.generation_usage.calls)}
        assert models == {f"model-{name}"}
        assert len(run.routing_usage.calls) == 1
        assert len(run.generation_usage.calls) == 1
    assert runs["a"].answer.archetype.value == "A"
    assert runs["c"].answer.archetype.value == "C"


def test_one_pipeline_answers_two_questions_at_once(flow: Any) -> None:
    """The same pipeline (one context) from two threads."""
    barrier = threading.Barrier(2)
    pipeline = flow(BarrierLLM(scripted("A"), "m", barrier), PricingEmbedder(), FakeVectorStore(DOCS))
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = list(pool.map(pipeline.run, ["pricing framework?", "pricing playbook?"]))
    assert [r.answer.query for r in runs] == ["pricing framework?", "pricing playbook?"]
    assert all(len(r.answer.usage.calls) == 2 for r in runs)


# --- the rules ---


def test_neither_flow_has_a_checkpointer() -> None:
    from tessera.lc.functional import tessera_flow

    assert GRAPH.checkpointer is None
    assert tessera_flow.checkpointer is None


def test_state_holds_plain_data_only() -> None:
    pipeline = LangGraphPipeline(scripted("A"), PricingEmbedder(), FakeVectorStore(DOCS))
    state = GRAPH.invoke(pipeline.initial_state("pricing framework?", None), context=pipeline.context)
    forbidden = (LLMClient, type(pipeline.context.embedder), type(pipeline.context.store))
    assert not [k for k, v in state.items() if isinstance(v, forbidden)]


# --- replay ---


@pytest.fixture(scope="module")
def recording() -> dict[str, Any]:
    if not RECORDING.exists():
        pytest.skip(f"no recording at {RECORDING} — run `python -m evals.replay --out {RECORDING}`")
    return load_recording(RECORDING)


@pytest.fixture(scope="module")
def exact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from tessera.embedding.local import LocalEmbedder

    embedder = LocalEmbedder()
    store, experts = build_exact_indexes(tmp_path_factory.mktemp("replay"), embedder)
    return {"embedder": embedder, "store": store, "experts": experts, "walls": load_walls_for_repo()}


def test_replay_of_native_responses_is_exact_on_every_case(
    recording: dict[str, Any], exact: dict[str, Any], flow: Any
) -> None:
    cases = load_cases(REPO / "evals" / "cases")
    recorded_ok = [c for c in cases if c.id not in set(recording["meta"]["failed"])]
    assert len(recorded_ok) >= 90, "too few cases recorded — re-record"
    native_llm, graph_llm = ReplayLLM(recording["responses"]), ReplayLLM(recording["responses"])
    native = NativePipeline(native_llm, exact["embedder"], exact["store"], exact["experts"])
    graph = flow(graph_llm, exact["embedder"], exact["store"], exact["experts"])
    mismatched = []
    for case in recorded_ok:
        principal = case_principal(case, exact["walls"])
        if graph.run(case.query, principal) != native.run(case.query, principal):
            mismatched.append(case.id)
    assert mismatched == []
    assert graph_llm.served == native_llm.served > 0


# --- streaming: the four modes ---


def _token_pipeline(answer: str, corrective: str = "off") -> LangGraphPipeline:
    """The graph with the LCEL answer chain on a LangChain chat model that
    streams ``answer`` word by word (``model_client=lc`` + ``prompt_chain=lc``)."""
    from itertools import cycle

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from tessera.lc.generation import LcModel, generate_step

    chat = GenericFakeChatModel(messages=cycle([AIMessage(content=answer)]))
    return LangGraphPipeline(
        scripted("A"), PricingEmbedder(), FakeVectorStore(DOCS),
        generate_fn=generate_step(LcModel(chat_model=chat, model_id="fake", retries=False)),
        corrective=corrective,
    )


def test_streaming_gives_the_same_run_as_invoking() -> None:
    from tessera.lc.graph import stream_run

    pipeline = _token_pipeline("The pricing framework is in [1].")
    streamed = stream_run(pipeline, "pricing framework?")
    assert streamed.run == pipeline.run("pricing framework?")
    assert streamed.snapshots >= 4  # values: one full state per step


def test_updates_time_every_node_in_order() -> None:
    from tessera.lc.graph import stream_run

    ticks = iter(range(100))
    streamed = stream_run(_token_pipeline("Answer [1]."), "pricing framework?", clock=lambda: float(next(ticks)))
    assert [node for node, _ in streamed.node_seconds] == ["route", "documents", "generate", "assemble"]
    assert all(seconds > 0 for _, seconds in streamed.node_seconds)


def test_messages_stream_tokens_and_time_the_first_one() -> None:
    from tessera.lc.graph import stream_run

    answer = "The pricing framework is in source one [1]."
    streamed = stream_run(_token_pipeline(answer), "pricing framework?")
    assert streamed.tokens > 5
    assert streamed.token_text == answer
    assert streamed.first_token_s is not None and streamed.first_token_s <= streamed.total_s


def test_a_native_model_client_streams_no_tokens() -> None:
    from tessera.lc.graph import stream_run

    streamed = stream_run(LangGraphPipeline(scripted("A"), PricingEmbedder(), FakeVectorStore(DOCS)), "pricing?")
    assert streamed.tokens == 0 and streamed.first_token_s is None


def test_custom_progress_reports_stages_and_counts_only() -> None:
    from tessera.lc.graph import stream_run

    streamed = stream_run(_token_pipeline("Answer [1].", corrective="score"), "pricing framework?")
    assert [p["stage"] for p in streamed.progress] == ["routed", "retrieved", "graded", "answered"]
    allowed = {"stage", "archetype", "chunks", "sufficient", "citations"}
    assert all(set(p) <= allowed for p in streamed.progress)
    assert all(not isinstance(v, str) or k in ("stage", "archetype") for p in streamed.progress for k, v in p.items())
