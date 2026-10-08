"""LangChain generation, routing and retries (Phase 5 plan §3.5–3.6, P5-6
acceptance) — all zero-call.

- **Prompt identity**: every ``ChatPromptTemplate`` renders exactly what
  ``prompts.py`` builds, braces in the inputs included.
- **Fake chat models**: the LCEL chain, the structured router and the B
  chain give the native generator's ``GeneratedAnswer`` / decision, on a
  LangChain fake chat model and on the native client through
  ``TesseraChatModel``; every call is metered into the pipeline's recorder.
- **Model parity**: ``ChatNVIDIA`` sends what ``NvidiaClient`` sends
  (captured HTTP body); the Gemini factory's parity fields.
- **Fault injection**: scripted 429/503/400 sequences through native
  ``RetryingLLMClient`` and LangChain ``.with_retry()``; provider errors
  translated to readable status codes.
- ``.batch`` and ``astream_events`` exercised.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("langchain_core", reason="the lc extra isn't installed")

import requests  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from tessera.generation.answer import generate_answer  # noqa: E402
from tessera.generation.base import Completion, LLMClient, Usage  # noqa: E402
from tessera.generation.expertise import generate_expertise_answer  # noqa: E402
from tessera.generation.prompts import (  # noqa: E402
    EXPERTISE_ANSWER_SYSTEM_PROMPT,
    LOOKUP_ANSWER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SYNTHESIS_ANSWER_SYSTEM_PROMPT,
    build_expertise_user_prompt,
    build_grounded_answer_user_prompt,
    build_router_user_prompt,
)
from tessera.generation.resilient import RetryingLLMClient, is_retryable  # noqa: E402
from tessera.generation.usage import UsageRecorder  # noqa: E402
from tessera.lc.chat_models import LangChainLLMClient, LLMCallError, TesseraChatModel, status_of  # noqa: E402
from tessera.lc.expertise import PeopleRetriever  # noqa: E402
from tessera.lc.generation import (  # noqa: E402
    EVAL_REQUESTS_PER_SECOND,
    LcModel,
    RecorderCallback,
    TransientLLMError,
    eval_rate_limiter,
    expertise_chain,
    generate_expertise_step,
    generate_step,
    grounded_answer_chain,
    route_step,
    with_lc_retries,
)
from tessera.retrieval.expertise import ExpertiseResult, find_experts  # noqa: E402
from tessera.retrieval.retriever import RetrievalResult, SupersededMatch  # noqa: E402
from tessera.retrieval.router import Archetype, RoutingError, route  # noqa: E402
from tessera.store.base import Evidence, PersonMatch, SearchResult  # noqa: E402
from tests.test_expertise_retrieval import BagEmbedder, FixedScoreStore, person, proj  # noqa: E402


class ScriptedLLM(LLMClient):
    """A native client that answers from a script and reports usage."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user).text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Completion:
        self.prompts.append((system, user))
        return Completion(self.replies.pop(0), Usage("fake-model", len(user), 7, 0.01))


def chunk(doc: str, index: int, text: str, score: float = 0.8) -> SearchResult:
    return SearchResult(
        chunk_id=f"{doc}::{index}", text=text, score=score, document_path=f"/c/methodology/{doc}.md",
        document_title=f"{doc.title()} Guide", doc_type="methodology", industry="retail",
        topics=["pricing"], date="2025-01-01", heading_path=("Approach", f"Step {index}"),
    )


def retrieval(archetype: Archetype = Archetype.LOOKUP, superseded: bool = False) -> RetrievalResult:
    results = [
        chunk("pricing", 0, "Set tiers by elasticity {not a template}."),
        chunk("pricing", 1, "Use 104 weeks of data."),
        chunk("market", 0, "Size the market first."),
        chunk("noise", 0, "Below the floor.", score=0.1),
    ]
    supersede = (
        (SupersededMatch("/c/methodology/old.md", "Old Pricing", "/c/methodology/pricing.md", 0.7),)
        if superseded
        else ()
    )
    return RetrievalResult(query="Do we have a {pricing} framework?", archetype=archetype, results=results,
                           superseded=supersede)


def matches() -> list[PersonMatch]:
    p = person("c0001", projects=[proj("pharma-pricing")])
    return [
        PersonMatch(person=p, score=0.8, evidence=(Evidence("project", "led pharma pricing in 2025", 1.5),),
                    evidence_score=1.5, rank_score=1.6)
    ]


# --- prompt identity ---


def capture() -> tuple[RunnableLambda, list[list[BaseMessage]]]:
    seen: list[list[BaseMessage]] = []

    def model(prompt: Any) -> AIMessage:
        seen.append(prompt.to_messages())
        return AIMessage(content="ok")

    return RunnableLambda(model), seen


@pytest.mark.parametrize("query", ["Do we have a pricing framework?", 'Weird {braces} and "quotes"\nnewline'])
def test_the_router_template_renders_the_native_prompt(query: str) -> None:
    from tessera.lc.generation import ROUTER_PROMPT

    system, human = ROUTER_PROMPT.format_messages(query=query)
    assert system.content == ROUTER_SYSTEM_PROMPT
    assert human.content == build_router_user_prompt(query)


@pytest.mark.parametrize(
    ("archetype", "system"),
    [(Archetype.LOOKUP, LOOKUP_ANSWER_SYSTEM_PROMPT), (Archetype.SYNTHESIS, SYNTHESIS_ANSWER_SYSTEM_PROMPT)],
)
def test_the_grounded_chain_renders_the_native_prompt(archetype: Archetype, system: str) -> None:
    model, seen = capture()
    r = retrieval(archetype)
    shown = r.results[:3]
    grounded_answer_chain(model).invoke({"question": r.query, "shown": shown, "archetype": archetype})
    rendered_system, rendered_user = seen[0]
    assert rendered_system.content == system
    assert rendered_user.content == build_grounded_answer_user_prompt(r.query, shown)


def test_the_expertise_chain_renders_the_native_prompt() -> None:
    model, seen = capture()
    expertise_chain(model).invoke({"question": "Who knows {pharma} pricing?", "matches": matches()})
    rendered_system, rendered_user = seen[0]
    assert rendered_system.content == EXPERTISE_ANSWER_SYSTEM_PROMPT
    assert rendered_user.content == build_expertise_user_prompt("Who knows {pharma} pricing?", matches())


# --- fake chat models: the same answers as native, metered ---


@pytest.mark.parametrize("superseded", [False, True])
@pytest.mark.parametrize("archetype", [Archetype.LOOKUP, Archetype.SYNTHESIS])
def test_the_lcel_chain_matches_native_generation(archetype: Archetype, superseded: bool) -> None:
    r = retrieval(archetype, superseded)
    native = generate_answer(r, ScriptedLLM("Tiers come from elasticity [1]."))

    fake = GenericFakeChatModel(messages=iter([AIMessage(content="Tiers come from elasticity [1].")]))
    recorder = UsageRecorder(ScriptedLLM())
    on_fake = generate_step(LcModel(chat_model=fake, model_id="m", retries=False))(r, recorder)
    on_native = generate_step(LcModel())(r, UsageRecorder(ScriptedLLM("Tiers come from elasticity [1].")))

    assert on_fake == native and on_native == native
    assert recorder.summary().unmetered_calls == 1  # the fake reports no usage: counted, not lost


def test_the_lcel_chain_meters_native_client_calls_into_the_recorder() -> None:
    recorder = UsageRecorder(ScriptedLLM("Answer [1]."))
    generate_step(LcModel())(retrieval(), recorder)
    (call,) = recorder.summary().calls
    assert call.model == "fake-model" and call.output_tokens == 7


def test_no_relevant_source_is_the_fixed_message_without_a_call() -> None:
    r = RetrievalResult(query="q", archetype=Archetype.LOOKUP, results=[chunk("noise", 0, "x", score=0.1)])
    llm = ScriptedLLM()
    assert generate_step(LcModel())(r, UsageRecorder(llm)) == generate_answer(r, ScriptedLLM())
    assert llm.prompts == []


def test_the_expertise_chain_matches_native() -> None:
    found = ExpertiseResult(query="Who knows pharma pricing?", matches=matches())
    native = generate_expertise_answer(found, ScriptedLLM("Asha [1]."))
    lc = generate_expertise_step(LcModel())(found, UsageRecorder(ScriptedLLM("Asha [1].")))
    assert lc == native


@pytest.mark.parametrize(
    "reply",
    ['{"archetype": "C", "reasoning": "a briefing"}', '```json\n{"archetype": "C", "reasoning": "a briefing"}\n```'],
)
def test_the_router_on_the_native_client_matches_native_routing(reply: str) -> None:
    lc = route_step(LcModel())("Staffed Monday — what to read?", UsageRecorder(ScriptedLLM(reply)))
    assert lc == route("Staffed Monday — what to read?", ScriptedLLM(reply))


def test_an_unparseable_route_is_a_routing_error_on_both() -> None:
    with pytest.raises(RoutingError):
        route("q", ScriptedLLM("not json"))
    with pytest.raises(RoutingError):
        route_step(LcModel())("q", UsageRecorder(ScriptedLLM("not json")))


# --- ChatNVIDIA: request parity, structured routing, metering ---


@pytest.fixture
def nim(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """ChatNVIDIA over a mocked transport; ``replies`` scripts the content
    (or an HTTP status code to fail with)."""
    state: dict[str, Any] = {"bodies": [], "replies": []}

    def send(self: Any, req: Any, **kw: Any) -> Any:
        r = requests.Response()
        r.request = req
        r.headers["content-type"] = "application/json"
        if req.body is None:  # the model listing
            r.status_code = 200
            r._content = json.dumps({"data": [{"id": "nvidia/nemotron-3-ultra-550b-a55b"}]}).encode()
            return r
        state["bodies"].append(json.loads(req.body))
        reply = state["replies"].pop(0)
        if isinstance(reply, BaseException):  # e.g. requests' ReadTimeout
            raise reply
        if isinstance(reply, int):
            r.status_code = reply
            r._content = json.dumps({"detail": "Service temporarily overloaded"}).encode()
            return r
        r.status_code = 200
        r._content = json.dumps({
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 9, "total_tokens": 129},
        }).encode()
        return r

    monkeypatch.setattr(requests.Session, "send", send)
    from tessera.integrations.langchain import chat_nvidia

    state["chat"] = chat_nvidia("nvapi-test", "nvidia/nemotron-3-ultra-550b-a55b")
    return state


def test_chatnvidia_sends_what_nvidia_client_sends(nim: dict[str, Any]) -> None:
    nim["replies"].append("OK")
    LangChainLLMClient(nim["chat"], "nvidia/nemotron-3-ultra-550b-a55b").complete("sys", "user")
    body = nim["bodies"][-1]
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]
    assert body["temperature"] == 0.0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "max_tokens" not in body and "max_completion_tokens" not in body


def test_structured_routing_on_chatnvidia_is_json_schema_and_metered(nim: dict[str, Any]) -> None:
    nim["replies"].append('{"archetype": "B", "reasoning": "a person"}')
    recorder = UsageRecorder(ScriptedLLM())
    lc_model = LcModel(chat_model=nim["chat"], model_id="nvidia/nemotron-3-ultra-550b-a55b", retries=False)
    decision = route_step(lc_model)("Who knows pharma pricing?", recorder)

    assert decision.archetype is Archetype.EXPERTISE and decision.reasoning == "a person"
    assert nim["bodies"][-1]["response_format"]["type"] == "json_schema"
    assert "tools" not in nim["bodies"][-1] and "tool_choice" not in nim["bodies"][-1]  # guided JSON, no forced call
    (call,) = recorder.summary().calls
    assert (call.model, call.input_tokens, call.output_tokens) == ("nvidia/nemotron-3-ultra-550b-a55b", 120, 9)
    assert call.latency_s >= 0


def test_chatnvidia_errors_carry_a_status_the_retry_layer_reads(nim: dict[str, Any]) -> None:
    nim["replies"].append(503)
    with pytest.raises(LLMCallError) as err:
        LangChainLLMClient(nim["chat"], "m").complete("s", "u")
    assert err.value.status_code == 503 and is_retryable(err.value)


def test_chatnvidia_waits_as_long_as_the_openai_sdk(nim: dict[str, Any]) -> None:
    """NvidiaClient's openai SDK reads for 600 s; ChatNVIDIA's default is 60 s,
    which cut off slow NIM answers in the P5-6 model_client sweep."""
    assert nim["chat"]._client.timeout == 600.0


def test_a_client_side_read_timeout_is_retried_on_both_paths(nim: dict[str, Any]) -> None:
    timeout = requests.exceptions.ReadTimeout("HTTPSConnectionPool(...): Read timed out. (read timeout=600)")
    assert status_of(timeout) == 504 and is_retryable(LLMCallError(504, timeout))

    nim["replies"].extend([timeout, "OK"])
    delays: list[float] = []
    native = RetryingLLMClient(LangChainLLMClient(nim["chat"], "m"), sleep=delays.append, clock=lambda: 0.0)
    assert native.complete("s", "u") == "OK" and delays == [15.0]  # the 5xx backoff

    nim["replies"].extend([requests.exceptions.ConnectionError("reset by peer"), "OK"])
    reply = with_lc_retries(nim["chat"], jitter=False).invoke("hi")
    assert reply.content == "OK"


def test_a_real_client_error_is_still_not_retried() -> None:
    assert status_of(ValueError("bad json")) is None
    assert status_of(RuntimeError("[400] bad request")) == 400


def test_the_gemini_factory_has_native_parity_fields() -> None:
    pytest.importorskip("langchain_google_genai")
    from tessera.integrations.langchain import chat_gemini

    model = chat_gemini("tessera-510716", "gemini-3.8-flash")
    assert (model.max_retries, model.max_output_tokens, model.location, model.vertexai) == (0, 16_000, "global", True)
    assert model.temperature is None  # Gemini 3: no temperature, as GeminiClient


# --- usage ---


def test_reasoning_tokens_are_counted_once() -> None:
    recorder = UsageRecorder(ScriptedLLM())
    callback = RecorderCallback("gemini-3.8-flash", recorder)
    from langchain_core.outputs import LLMResult

    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 10, "output_tokens": 87, "total_tokens": 97,
                        "output_token_details": {"reasoning": 86}},
    )
    import uuid

    run = uuid.uuid4()
    callback.on_chat_model_start({}, [], run_id=run)
    callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run)
    (call,) = recorder.summary().calls
    assert (call.model, call.input_tokens, call.output_tokens) == ("gemini-3.8-flash", 10, 87)


# --- fault injection ---


class ScriptedError(Exception):
    """A provider error; ``headers`` mimics an HTTP response's."""

    def __init__(self, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.response = type("R", (), {"headers": headers or {}})()


class FlakyChat(BaseChatModel):
    """Fails or answers per a script; counts calls."""

    script: list[Any]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "flaky"

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=step))])


def native_retrying(chat: FlakyChat, delays: list[float]) -> RetryingLLMClient:
    return RetryingLLMClient(LangChainLLMClient(chat, "m"), sleep=delays.append, clock=lambda: 0.0)


def test_native_retries_translate_and_honour_retry_after() -> None:
    chat = FlakyChat(script=[Exception("[503] busy"), ScriptedError("[429] slow", {"retry-after": "7"}), "ok"])
    delays: list[float] = []
    assert native_retrying(chat, delays).complete("s", "u") == "ok"
    assert chat.calls == 3
    assert delays[0] == 15.0  # 5xx base backoff
    assert delays[1] == 7.0  # Retry-After honoured


def test_lc_with_retry_recovers_from_transient_errors() -> None:
    chat = FlakyChat(script=[Exception("[503] busy"), Exception("[429] slow"), "ok"])
    reply = with_lc_retries(chat, jitter=False).invoke("hi")
    assert reply.content == "ok" and chat.calls == 3


@pytest.mark.parametrize("make", [lambda c: with_lc_retries(c, jitter=False), lambda c: native_retrying(c, [])])
def test_a_client_error_is_raised_at_once_by_both(make: Any) -> None:
    chat = FlakyChat(script=[Exception("[400] bad request"), "never"])
    runnable = make(chat)
    with pytest.raises(LLMCallError) as err:
        runnable.invoke("hi") if hasattr(runnable, "invoke") else runnable.complete("s", "u")
    assert err.value.status_code == 400 and chat.calls == 1


def test_lc_with_retry_gives_up_after_six_attempts() -> None:
    chat = FlakyChat(script=[Exception("[503] busy")] * 6)
    with pytest.raises(TransientLLMError):
        with_lc_retries(chat, jitter=False).invoke("hi")
    assert chat.calls == 6


def test_a_gemini_429_hidden_on_the_cause_is_found() -> None:
    cause = ScriptedError("quota")
    cause.code = 429  # type: ignore[attr-defined]
    try:
        try:
            raise cause
        except ScriptedError as inner:
            raise RuntimeError("GoogleRateLimitError") from inner
    except RuntimeError as outer:
        assert status_of(outer) == 429


def test_eval_pacing_is_one_request_per_three_seconds() -> None:
    limiter = eval_rate_limiter()
    assert EVAL_REQUESTS_PER_SECOND == pytest.approx(1 / 3)
    assert limiter.requests_per_second == pytest.approx(1 / 3) and limiter.max_bucket_size == 1


# --- batch, streaming, the native client as a chat model ---


def test_batch_and_astream_events() -> None:
    import asyncio

    chain = grounded_answer_chain(TesseraChatModel(client=ScriptedLLM("one [1].", "two [1].")))
    r = retrieval()
    inputs = [{"question": q, "shown": r.results[:3], "archetype": Archetype.LOOKUP} for q in ("a", "b")]
    assert sorted(chain.batch(inputs, config={"max_concurrency": 1})) == ["one [1].", "two [1]."]

    streaming = grounded_answer_chain(GenericFakeChatModel(messages=iter([AIMessage(content="tok en s")])))

    async def events() -> list[str]:
        return [e["event"] async for e in streaming.astream_events(inputs[0], version="v2")]

    seen = asyncio.run(events())
    assert "on_chat_model_stream" in seen and "on_prompt_start" in seen


# --- B: PeopleRetriever keeps native's evidence ranking ---


def test_people_retriever_returns_native_find_experts() -> None:
    doer = person("doer", projects=[proj("pharma-pricing")])
    claimer = person("claimer")
    store = FixedScoreStore([(doer, 0.6), (claimer, 0.7)])
    retriever = PeopleRetriever(embedder=BagEmbedder(), store=store, k=5)
    assert retriever.experts("pharma pricing") == find_experts("pharma pricing", BagEmbedder(), store, k=5)
    docs = retriever.invoke("pharma pricing")
    assert docs[0].metadata["person_id"] == "doer" and docs[0].metadata["evidenced"] is True
