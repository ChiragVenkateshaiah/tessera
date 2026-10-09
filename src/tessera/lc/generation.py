"""Generation and routing on LangChain (Phase 5 plan §3.5–3.6, P5-6): the
``router``, ``prompt_chain``, ``model_client`` and ``retry`` switches.

**Prompts.** ``ChatPromptTemplate``s with the same text as ``prompts.py``
(literal braces escaped). The prompt-identity test renders both.

**The grounded-answer chain**, built from Runnables on purpose::

    RunnableParallel(question, shown, archetype)
      | RunnablePassthrough.assign(sources=RunnableLambda(format_sources))
      | RunnableBranch(A -> lookup prompt, else synthesis prompt)
      | model | StrOutputParser()

``format_sources`` is the shared ``format_source_group``, so the judge
reads sources numbered exactly as the model did. The relevance floor,
the fixed "nothing on that" message, the citations and the superseded
note stay deterministic, outside the model (``finish_answer``).

**Structured routing.** ``with_structured_output(RouteDecision,
method="json_schema")`` on a LangChain chat model (``ChatNVIDIA`` sends
``response_format: json_schema``). On the native client (the
``model_client`` switch held native) the same Pydantic model parses the
prompt-instructed JSON. ``include_raw=True`` would keep the
``AIMessage`` for its usage, but ``ChatNVIDIA`` raises on it (P5-1 spike
item 6) — and isn't needed: usage is metered by callback.

**Usage.** An ``LcModel`` says what a chain calls. With no chat model it
is the native client, through the pipeline's own ``UsageRecorder``
(``TesseraChatModel``). With one, a ``RecorderCallback`` — a
``UsageCallback`` per invocation — adds every call's tokens and latency
to that recorder. Either way nothing is unmetered.

**Retries** (``retry`` switch). ``native``: the chat model behind the
native port inside ``RetryingLLMClient`` (``Retry-After`` honoured, 3 s
pacing in sweeps). ``lc``: the chat model's errors translated to
``TransientLLMError`` for 429/5xx, ``.with_retry()`` on those
(exponential jitter, no ``Retry-After``), and an ``InMemoryRateLimiter``
for pacing. LangGraph's ``RetryPolicy`` comes with P5-7.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field

from tessera.generation.answer import NO_RESULTS_MESSAGE, GeneratedAnswer, filter_relevant, finish_answer
from tessera.generation.base import LLMClient
from tessera.generation.expertise import NO_EXPERT_MESSAGE, filter_qualified
from tessera.generation.prompts import (
    EXPERTISE_ANSWER_SYSTEM_PROMPT,
    LOOKUP_ANSWER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SYNTHESIS_ANSWER_SYSTEM_PROMPT,
    format_person_record,
    format_source_group,
    group_by_document,
)
from tessera.generation.usage import UsageRecorder
from tessera.lc.chat_models import LLMCallError, TesseraChatModel, UsageCallback, status_of
from tessera.retrieval.expertise import ExpertiseResult
from tessera.retrieval.retriever import RetrievalResult
from tessera.retrieval.router import Archetype, RoutingDecision, RoutingError, _strip_markdown_fence

RETRY_ATTEMPTS = 6  # as RetryingLLMClient
EVAL_REQUESTS_PER_SECOND = 1 / 3  # the sweeps' 3 s pacing


def _literal(text: str) -> str:
    """Prompt text as template text: its own braces are literal."""
    return text.replace("{", "{{").replace("}", "}}")


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _literal(ROUTER_SYSTEM_PROMPT)), ("human", "Classify this query:\n\n{query}")]
)
LOOKUP_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _literal(LOOKUP_ANSWER_SYSTEM_PROMPT)), ("human", "Question: {question}\n\nSources:\n\n{sources}")]
)
SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _literal(SYNTHESIS_ANSWER_SYSTEM_PROMPT)),
        ("human", "Question: {question}\n\nSources:\n\n{sources}"),
    ]
)
EXPERTISE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _literal(EXPERTISE_ANSWER_SYSTEM_PROMPT)),
        ("human", "Question: {question}\n\nPerson records:\n\n{records}"),
    ]
)


class RouteDecision(BaseModel):
    """The router's structured output."""

    archetype: Literal["A", "B", "C", "D"] = Field(description="The query archetype.")
    reasoning: str = Field(description="One sentence explaining the classification.")


# --- retries ---


class TransientLLMError(LLMCallError):
    """A 429 or 5xx: worth retrying."""


def _raise_translated(exc: BaseException) -> None:
    code = status_of(exc)
    if code is not None and (code == 429 or 500 <= code < 600):
        raise TransientLLMError(code, exc) from exc
    if code is not None:
        raise LLMCallError(code, exc) from exc
    raise exc


def with_lc_retries(
    runnable: Runnable, *, attempts: int = RETRY_ATTEMPTS, jitter: bool = True
) -> Runnable:
    """``runnable`` whose provider errors are translated (status codes
    readable), retried on 429/5xx with ``.with_retry()``'s exponential
    jitter. Unlike ``RetryingLLMClient`` it never reads ``Retry-After``."""

    def call(value: Any, config: RunnableConfig) -> Any:
        try:
            return runnable.invoke(value, config)
        except Exception as exc:  # translated, then retried or raised
            _raise_translated(exc)

    return RunnableLambda(call, name=f"translated({runnable.get_name()})").with_retry(
        retry_if_exception_type=(TransientLLMError,),
        wait_exponential_jitter=jitter,  # off only in fault-injection tests
        stop_after_attempt=attempts,
    )


def eval_rate_limiter() -> InMemoryRateLimiter:
    """Pacing for sweeps under the ``lc`` retry value: one request per 3 s
    (the native sweeps' minimum interval)."""
    return InMemoryRateLimiter(
        requests_per_second=EVAL_REQUESTS_PER_SECOND, check_every_n_seconds=0.1, max_bucket_size=1
    )


# --- what a chain calls ---


class RecorderCallback(UsageCallback):
    """A ``UsageCallback`` that adds each call to a ``UsageRecorder``."""

    def __init__(self, model: str, recorder: UsageRecorder) -> None:
        super().__init__(model)
        self.recorder = recorder

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        before_calls, before_unmetered = len(self.calls), self.unmetered
        super().on_llm_end(response, **kwargs)
        if len(self.calls) > before_calls:
            self.recorder.add(self.calls[-1])
        elif self.unmetered > before_unmetered:
            self.recorder.add(None)


@dataclass(frozen=True)
class LcModel:
    """What an LCEL chain calls. ``chat_model`` None: the native client,
    through the pipeline's own recorder. Otherwise a LangChain chat model
    (``lc`` retry: wrapped per call), metered by callback."""

    chat_model: Any = None
    model_id: str = "tessera-native"
    retries: bool = True

    def model(self, recorder: UsageRecorder) -> tuple[Runnable, list[Any]]:
        if self.chat_model is None:
            return TesseraChatModel(client=recorder), []
        model: Runnable = self.chat_model
        if self.retries:
            model = with_lc_retries(model)
        return model, [RecorderCallback(self.model_id, recorder)]

    def router(self, recorder: UsageRecorder) -> tuple[Runnable, list[Any], bool]:
        """The routing model, and whether it returns a ``RouteDecision``
        itself (API-enforced JSON schema) or text to parse."""
        if self.chat_model is None:
            return TesseraChatModel(client=recorder), [], False
        structured: Runnable = self.chat_model.with_structured_output(RouteDecision, method="json_schema")
        if self.retries:
            structured = with_lc_retries(structured)
        return structured, [RecorderCallback(self.model_id, recorder)], True


# --- the chains ---


def format_sources(shown: list[Any]) -> str:
    """The numbered sources block, exactly as ``build_grounded_answer_user_prompt``."""
    return "\n\n".join(
        f"[{i}] {format_source_group(group)}" for i, group in enumerate(group_by_document(shown), start=1)
    )


def grounded_answer_chain(model: Runnable) -> Runnable:
    prepare = RunnableParallel(
        question=itemgetter("question"), shown=itemgetter("shown"), archetype=itemgetter("archetype")
    ) | RunnablePassthrough.assign(sources=RunnableLambda(lambda x: format_sources(x["shown"]), name="format_sources"))
    prompt = RunnableBranch(
        (lambda x: x["archetype"] is Archetype.LOOKUP, LOOKUP_PROMPT),
        SYNTHESIS_PROMPT,
    )
    return (prepare | prompt | model | StrOutputParser()).with_config(run_name="grounded_answer")


def expertise_chain(model: Runnable) -> Runnable:
    records = RunnableLambda(
        lambda x: "\n\n".join(format_person_record(i, m) for i, m in enumerate(x["matches"], start=1)),
        name="format_person_records",
    )
    return (
        RunnablePassthrough.assign(records=records) | EXPERTISE_PROMPT | model | StrOutputParser()
    ).with_config(run_name="expertise_answer")


# --- pipeline steps (NativePipeline's route_fn / generate_fn / ...) ---


def route_step(lc_model: LcModel) -> Callable[[str, LLMClient], RoutingDecision]:
    """The ``router`` switch: a LangChain routing chain."""

    def route(query: str, llm: LLMClient) -> RoutingDecision:
        assert isinstance(llm, UsageRecorder)
        model, callbacks, structured = lc_model.router(llm)
        if structured:
            chain = (ROUTER_PROMPT | model).with_config(run_name="route")
        else:
            parser = PydanticOutputParser(pydantic_object=RouteDecision)
            chain = (
                ROUTER_PROMPT | model | StrOutputParser() | RunnableLambda(_strip_markdown_fence) | parser
            ).with_config(run_name="route")
        try:
            decision = chain.invoke({"query": query}, config={"callbacks": callbacks})
        except (LLMCallError, TransientLLMError):
            raise
        except Exception as exc:  # a parse failure, as native's RoutingError
            raise RoutingError(f"could not parse routing response: {exc}") from exc
        if not isinstance(decision, RouteDecision):
            raise RoutingError(f"could not parse routing response: {decision!r}")
        return RoutingDecision(query=query, archetype=Archetype(decision.archetype), reasoning=decision.reasoning)

    return route


def generate_step(lc_model: LcModel) -> Callable[[RetrievalResult, LLMClient], GeneratedAnswer]:
    """The ``prompt_chain`` switch: the LCEL grounded-answer chain, with
    native's floor, fixed message, citations and note."""

    def generate(retrieval: RetrievalResult, llm: LLMClient) -> GeneratedAnswer:
        if retrieval.archetype not in (Archetype.LOOKUP, Archetype.SYNTHESIS):
            raise ValueError(f"generate() only supports A and C, got {retrieval.archetype.value!r}")
        relevant = filter_relevant(retrieval.results)
        if not relevant:
            return GeneratedAnswer(
                query=retrieval.query, archetype=retrieval.archetype, answer=NO_RESULTS_MESSAGE, citations=[]
            )
        assert isinstance(llm, UsageRecorder)
        model, callbacks = lc_model.model(llm)
        text = grounded_answer_chain(model).invoke(
            {"question": retrieval.query, "shown": relevant, "archetype": retrieval.archetype},
            config={"callbacks": callbacks},
        )
        return finish_answer(retrieval, relevant, text)

    return generate


def generate_expertise_step(lc_model: LcModel) -> Callable[[ExpertiseResult, LLMClient], GeneratedAnswer]:
    """B's generation as an LCEL chain, with native's two floors."""

    def generate(found: ExpertiseResult, llm: LLMClient) -> GeneratedAnswer:
        qualified = filter_qualified(found.matches)
        if not qualified:
            return GeneratedAnswer(
                query=found.query, archetype=Archetype.EXPERTISE, answer=NO_EXPERT_MESSAGE, citations=[], experts=[]
            )
        assert isinstance(llm, UsageRecorder)
        model, callbacks = lc_model.model(llm)
        text = expertise_chain(model).invoke(
            {"question": found.query, "matches": qualified}, config={"callbacks": callbacks}
        )
        return GeneratedAnswer(
            query=found.query, archetype=Archetype.EXPERTISE, answer=text, citations=[], experts=qualified
        )

    return generate
