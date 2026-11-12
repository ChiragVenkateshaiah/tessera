"""LangGraph's ``RetryPolicy`` as the ``retry`` switch's ``graph`` value
(Phase 5 plan §3.6, deferred from P5-6 to P5-7).

The third LangChain-family retry mechanism, set against the native
``RetryingLLMClient`` and ``.with_retry()`` with the same fault-injection
shape as ``tests/test_lc_generation.py``:
- a node that fails with a 429/5xx is re-run whole, and the run equals a
  clean one (a failed attempt's writes are dropped: no doubled usage, no
  extra retrieval);
- a 400 is raised at once;
- it gives up after six attempts;
- the corrective subgraph's nodes retry too;
- the default graph has no policy, so ``retry=lc`` is never retried twice.

All zero-call: scripted clients, a policy with no waits.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langgraph", reason="the lc extra isn't installed")

from langgraph.types import RetryPolicy  # noqa: E402

from tessera.generation.base import Completion, LLMClient, Usage  # noqa: E402
from tessera.generation.prompts import (  # noqa: E402
    EXPERTISE_ANSWER_SYSTEM_PROMPT,
    LOOKUP_ANSWER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SYNTHESIS_ANSWER_SYSTEM_PROMPT,
)
from tessera.lc.chat_models import LLMCallError  # noqa: E402
from tessera.lc.corrective import GRADER_SYSTEM_PROMPT, REWRITE_SYSTEM_PROMPT  # noqa: E402
from tessera.lc.graph import (  # noqa: E402
    GRAPH,
    GRAPH_RETRY_POLICY,
    GRAPH_WITH_RETRIES,
    LangGraphPipeline,
    build_graph,
    is_transient,
)
from tessera.pipeline import NativePipeline  # noqa: E402
from tessera.principal import Principal  # noqa: E402
from tests.test_expertise_generation import ScoredExpertiseStore, make_person, make_project  # noqa: E402
from tests.test_pipeline import FakeVectorStore, PricingEmbedder, _result, _router_response  # noqa: E402

# The production policy's shape, with no waits.
FAST = RetryPolicy(initial_interval=0.0, backoff_factor=2.0, max_interval=0.0, max_attempts=6, jitter=False,
                   retry_on=is_transient)
FAST_GRAPH = build_graph(FAST)

DOCS = [_result("methodology/pricing.md", 0.9), _result("thought_leadership/pricing-2025.md", 0.8)]
EXPERT = make_person("c1", projects=[make_project("pricing"), make_project("pricing", 2026)])


def replies(archetype: str) -> dict[str, str]:
    return {
        ROUTER_SYSTEM_PROMPT: _router_response(archetype),
        LOOKUP_ANSWER_SYSTEM_PROMPT: "The pricing framework is in [1].",
        SYNTHESIS_ANSWER_SYSTEM_PROMPT: "Across [1] and [2], pricing is value-based.",
        EXPERTISE_ANSWER_SYSTEM_PROMPT: "Ask Person c1 [1].",
        GRADER_SYSTEM_PROMPT: '{"sufficient": false, "reasoning": "test"}',
        REWRITE_SYSTEM_PROMPT: "pricing methodology playbook",
    }


class FlakyLLM(LLMClient):
    """Answers from ``replies``, but first raises the scripted errors for
    a system prompt (as ``LangChainLLMClient`` raises them: ``LLMCallError``
    with a status). Counts every call per prompt."""

    def __init__(self, replies: dict[str, str], failures: dict[str, list[int]] | None = None) -> None:
        self.replies = replies
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.calls: dict[str, int] = {}

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Completion:
        self.calls[system] = self.calls.get(system, 0) + 1
        pending = self.failures.get(system)
        if pending:
            raise LLMCallError(pending.pop(0), RuntimeError("scripted"))
        return Completion(self.replies[system], Usage("m", 10, 2, 0.0))


def a_path(kind: Any, llm: LLMClient, **kw: Any) -> Any:
    return kind(llm, PricingEmbedder(), FakeVectorStore(DOCS), **kw)


def b_path(kind: Any, llm: LLMClient, **kw: Any) -> Any:
    return kind(llm, PricingEmbedder(), FakeVectorStore([]), ScoredExpertiseStore([(EXPERT, 0.6)]), **kw)


PRINCIPAL = Principal("partner", frozenset({"halcyon"}))


@pytest.mark.parametrize(
    ("archetype", "make", "failing"),
    [
        ("A", a_path, LOOKUP_ANSWER_SYSTEM_PROMPT),
        ("C", a_path, SYNTHESIS_ANSWER_SYSTEM_PROMPT),
        ("B", b_path, EXPERTISE_ANSWER_SYSTEM_PROMPT),
    ],
)
def test_transient_errors_are_retried_and_the_run_equals_a_clean_one(archetype: str, make: Any, failing: str) -> None:
    clean = make(NativePipeline, FlakyLLM(replies(archetype))).run("pricing framework?", PRINCIPAL)
    flaky = FlakyLLM(replies(archetype), {ROUTER_SYSTEM_PROMPT: [503, 429], failing: [502]})
    run = make(LangGraphPipeline, flaky, graph=FAST_GRAPH).run("pricing framework?", PRINCIPAL)
    assert run == clean  # same answer, citations, trace and usage: failed attempts left nothing behind
    assert flaky.calls[ROUTER_SYSTEM_PROMPT] == 3 and flaky.calls[failing] == 2


def test_a_client_error_is_raised_at_once() -> None:
    flaky = FlakyLLM(replies("A"), {ROUTER_SYSTEM_PROMPT: [400]})
    with pytest.raises(LLMCallError) as err:
        a_path(LangGraphPipeline, flaky, graph=FAST_GRAPH).run("pricing framework?")
    assert err.value.status_code == 400 and flaky.calls[ROUTER_SYSTEM_PROMPT] == 1


def test_it_gives_up_after_six_attempts() -> None:
    flaky = FlakyLLM(replies("A"), {LOOKUP_ANSWER_SYSTEM_PROMPT: [503] * 6})
    with pytest.raises(LLMCallError) as err:
        a_path(LangGraphPipeline, flaky, graph=FAST_GRAPH).run("pricing framework?")
    assert err.value.status_code == 503 and flaky.calls[LOOKUP_ANSWER_SYSTEM_PROMPT] == 6


def test_the_corrective_nodes_retry_without_an_extra_retrieval() -> None:
    """The grader and the rewriter fail once each; the loop still makes
    exactly two retrieval attempts, as a clean run does."""
    clean = a_path(LangGraphPipeline, FlakyLLM(replies("A")), corrective="llm").run("pricing framework?")
    flaky = FlakyLLM(replies("A"), {GRADER_SYSTEM_PROMPT: [503], REWRITE_SYSTEM_PROMPT: [429]})
    run = a_path(LangGraphPipeline, flaky, corrective="llm", graph=FAST_GRAPH).run("pricing framework?")
    assert run == clean
    assert len(run.retrievals) == 2
    assert flaky.calls[GRADER_SYSTEM_PROMPT] == 2 and flaky.calls[REWRITE_SYSTEM_PROMPT] == 2


def test_without_the_policy_the_first_transient_error_is_raised() -> None:
    """The default graph has no RetryPolicy: under retry=native or lc the
    client layer has already retried, so the graph must not again."""
    flaky = FlakyLLM(replies("A"), {ROUTER_SYSTEM_PROMPT: [503]})
    with pytest.raises(LLMCallError):
        a_path(LangGraphPipeline, flaky).run("pricing framework?")
    assert flaky.calls[ROUTER_SYSTEM_PROMPT] == 1


def _policies(graph: Any) -> dict[str, bool]:
    nodes = graph.builder.nodes
    found = {name: spec.retry_policy is not None for name, spec in nodes.items() if name != "documents"}
    sub = nodes["documents"].runnable
    found.update({f"documents.{name}": spec.retry_policy is not None for name, spec in sub.builder.nodes.items()})
    return found


def test_the_policy_is_on_exactly_the_model_calling_nodes() -> None:
    assert not any(_policies(GRAPH).values())
    assert _policies(GRAPH_WITH_RETRIES) == {
        "route": True,
        "terminal": False,
        "experts": False,  # embeds and searches; no model call
        "generate_b": True,
        "generate": True,
        "assemble": False,
        "documents.retrieve": True,  # the Send multi-query rephrases
        "documents.grade": True,
        "documents.rewrite": True,
        "documents.retry": True,
    }


def test_the_policy_mirrors_the_native_5xx_schedule() -> None:
    p = GRAPH_RETRY_POLICY
    assert (p.initial_interval, p.backoff_factor, p.max_interval, p.max_attempts) == (15.0, 2.0, 120.0, 6)
    assert p.jitter is True


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (LLMCallError(429, RuntimeError("x")), True),
        (LLMCallError(503, RuntimeError("x")), True),
        (RuntimeError("[500] upstream"), True),
        (LLMCallError(400, RuntimeError("unknown field `guided_json`")), False),
        (ValueError("no status"), False),
    ],
)
def test_what_counts_as_transient(exc: BaseException, transient: bool) -> None:
    assert is_transient(exc) is transient


def test_a_chat_model_without_client_retries_still_translates_its_errors() -> None:
    """retry=graph builds ``LcModel(..., retries=False)``: the chat model's
    errors must still carry a status, or the policy couldn't read them."""
    from tests.test_lc_generation import FlakyChat

    from tessera.generation.usage import UsageRecorder
    from tessera.lc.generation import LcModel, TransientLLMError

    chat = FlakyChat(script=[Exception("[503] busy"), "never"])
    model, callbacks = LcModel(chat_model=chat, model_id="m", retries=False).model(UsageRecorder(FlakyLLM({})))
    with pytest.raises(TransientLLMError) as err:
        model.invoke("hi", config={"callbacks": callbacks})
    assert err.value.status_code == 503 and chat.calls == 1  # translated, not retried
