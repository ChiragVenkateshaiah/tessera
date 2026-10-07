"""Chat models across the two stacks (Phase 5 plan §3.5, the
``model_client`` and ``retry`` switches).

- ``UsageCallback``: one instance per invocation, records each LLM call's
  tokens **and latency** (the stock ``UsageMetadataCallbackHandler`` has
  no latency) as Tessera ``Usage`` under the **configured** model id, so
  ``cost_usd`` and traces work unchanged. Gemini's thinking tokens are
  counted once: LangChain's ``output_tokens`` already includes them, as
  native's ``candidates + thoughts`` does.
- ``LangChainLLMClient``: a LangChain chat model (``ChatNVIDIA``,
  ``ChatGoogleGenerativeAI``) behind the native ``LLMClient`` port, so the
  native router, generator and judge can use it, and so native
  ``RetryingLLMClient`` can wrap it. Provider errors are translated to
  carry ``status_code`` (``ChatNVIDIA`` raises a bare
  ``Exception("[503] …")``; Gemini's 429 hides its code on ``__cause__``),
  so every retry mechanism sees status codes.
- ``TesseraChatModel``: the native ``LLMClient`` port as a LangChain chat
  model, so an LCEL chain can run on the native client (the
  ``prompt_chain`` switch with ``model_client`` held native). Calls go
  through the port, so the pipeline's ``UsageRecorder`` meters them.
"""

from __future__ import annotations

import re
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

from tessera.generation.base import Completion, LLMClient, Usage

_BRACKET_STATUS = re.compile(r"^\s*\[(\d{3})\]")


def status_of(exc: BaseException) -> int | None:
    """The HTTP status behind a provider error, wherever it hides."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("status_code", "code"):
            value = getattr(current, attr, None)
            if isinstance(value, int):
                return value
        match = _BRACKET_STATUS.match(str(current))
        if match:
            return int(match.group(1))
        current = current.__cause__ or current.__context__
    return None


def _response_of(exc: BaseException) -> Any:
    """The HTTP response a provider error carries, wherever it hides."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if response is not None:
            return response
        current = current.__cause__ or current.__context__
    return None


class LLMCallError(RuntimeError):
    """A provider error, re-raised with the ``status_code`` that
    ``resilient._status_code`` and the retry mechanisms read — and with the
    original ``response``, so ``RetryingLLMClient`` still honours a
    ``Retry-After`` header (dropping it made a 7 s server hint a 90 s
    backoff; the fault-injection test pins it)."""

    def __init__(self, status_code: int, cause: BaseException) -> None:
        super().__init__(f"[{status_code}] {cause}")
        self.status_code = status_code
        self.response = _response_of(cause)


def translate(exc: BaseException) -> BaseException:
    """``exc`` with a readable ``status_code`` (itself, if it has one)."""
    if isinstance(getattr(exc, "status_code", None), int):
        return exc
    code = status_of(exc)
    return LLMCallError(code, exc) if code is not None else exc


def message_text(message: BaseMessage) -> str:
    """A reply's text. Gemini's ``content`` is a list of blocks (with a
    thought signature); ``.text`` joins the text blocks for every model."""
    return message.text


class UsageCallback(BaseCallbackHandler):
    """Tokens and latency of every LLM call in one invocation."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[Usage] = []
        self.unmetered = 0
        self._started: dict[UUID, float] = {}

    def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._started[run_id] = time.perf_counter()

    def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        latency = time.perf_counter() - self._started.pop(run_id, time.perf_counter())
        usage = None
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or usage
        if not usage:
            self.unmetered += 1
            return
        self.calls.append(
            Usage(
                model=self.model,
                input_tokens=int(usage.get("input_tokens", 0)),
                # Includes reasoning tokens already (output_token_details
                # breaks them out); adding them again would double-count.
                output_tokens=int(usage.get("output_tokens", 0)),
                latency_s=latency,
            )
        )


class LangChainLLMClient(LLMClient):
    """A LangChain chat model as the native ``LLMClient`` port. ``model``
    is the configured id, used for pricing (never ``response_metadata``)."""

    def __init__(self, chat_model: Any, model: str) -> None:
        self.chat_model = chat_model
        self.model = model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Completion:
        handler = UsageCallback(self.model)
        try:
            reply = self.chat_model.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)],
                config={"callbacks": [handler]},
            )
        except Exception as exc:  # re-raised with a status the retry layer reads
            raise translate(exc) from exc
        text = message_text(reply)
        if not text:
            raise RuntimeError(f"{self.model} returned an empty completion")
        return Completion(text, handler.calls[-1] if handler.calls else None)


class TesseraChatModel(BaseChatModel):
    """The native ``LLMClient`` port as a LangChain chat model: the first
    system message is the system prompt, the human messages are the user
    turn. Usage rides back on ``usage_metadata``."""

    client: Any
    model_name: str = "tessera-native"

    @property
    def _llm_type(self) -> str:
        return "tessera-native"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = "\n\n".join(m.text for m in messages if isinstance(m, SystemMessage))
        user = "\n\n".join(m.text for m in messages if not isinstance(m, SystemMessage))
        completion: Completion = self.client.complete_with_usage(system, user)
        usage = completion.usage
        message = AIMessage(
            content=completion.text,
            usage_metadata={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
            }
            if usage
            else None,
            response_metadata={"model_name": usage.model if usage else self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
