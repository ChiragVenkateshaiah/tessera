"""BedrockClient against a fake Anthropic SDK client — the request it
builds per model, and how it reads the response. No AWS calls.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.resources.messages import Messages

from tessera.generation.base import Usage
from tessera.generation.bedrock import BedrockClient, accepts_temperature
from tessera.generation.resilient import RetryingLLMClient

HAIKU = "anthropic.claude-haiku-4-5"
OPUS = "anthropic.claude-opus-5-5"

SDK_CREATE_PARAMS = set(inspect.signature(Messages.create).parameters) - {"self"}


def _message(
    *blocks: SimpleNamespace,
    stop_reason: str = "end_turn",
    input_tokens: int = 1200,
    output_tokens: int = 85,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


class FakeSdk:
    """Stands in for AnthropicBedrockMantle: records each request, returns
    the scripted messages (or raises the scripted exceptions) in order.
    """

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **request: Any) -> SimpleNamespace:
        # Same keyword check as the real SDK, which raises TypeError on an
        # argument it doesn't know (1.x removed `temperature`).
        unknown = set(request) - SDK_CREATE_PARAMS
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


def _client(model: str, sdk: FakeSdk, **kw: Any) -> BedrockClient:
    return BedrockClient(model, aws_region="us-east-1", client=sdk, **kw)


def test_accepts_temperature_only_for_haiku() -> None:
    assert accepts_temperature(HAIKU)
    assert not accepts_temperature(OPUS)
    assert not accepts_temperature("anthropic.claude-opus-4-8")


def test_router_request_on_haiku_sends_temperature_and_no_effort() -> None:
    sdk = FakeSdk(_message(_text('{"archetype": "A"}')))

    _client(HAIKU, sdk).complete("route this", "a question")

    (request,) = sdk.requests
    assert request["model"] == HAIKU
    assert request["system"] == "route this"
    assert request["messages"] == [{"role": "user", "content": "a question"}]
    assert request["extra_body"] == {"temperature": 0.0}
    assert "output_config" not in request


def test_answer_request_on_opus_sends_effort_and_no_temperature() -> None:
    sdk = FakeSdk(_message(_text("answer [1]")))

    _client(OPUS, sdk, effort="medium").complete("answer this", "q")

    (request,) = sdk.requests
    assert request["output_config"] == {"effort": "medium"}
    assert "extra_body" not in request  # Opus 5.5 rejects any temperature


def test_returns_text_blocks_only_and_reports_usage() -> None:
    thinking = SimpleNamespace(type="thinking", thinking="")
    sdk = FakeSdk(_message(thinking, _text("Use the "), _text("pricing framework [1].")))

    completion = _client(OPUS, sdk).complete_with_usage("s", "u")

    assert completion.text == "Use the pricing framework [1]."
    assert isinstance(completion.usage, Usage)
    assert completion.usage.model == OPUS
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (1200, 85)
    assert completion.usage.latency_s >= 0.0


def test_refusal_raises() -> None:
    sdk = FakeSdk(_message(stop_reason="refusal"))

    with pytest.raises(RuntimeError, match="refusal"):
        _client(OPUS, sdk).complete("s", "u")


def test_no_text_raises_with_the_stop_reason() -> None:
    thinking = SimpleNamespace(type="thinking", thinking="")
    sdk = FakeSdk(_message(thinking, stop_reason="max_tokens"))

    with pytest.raises(RuntimeError, match="max_tokens"):
        _client(OPUS, sdk).complete("s", "u")


class Throttled(Exception):
    status_code = 429
    response = SimpleNamespace(headers={})


def test_retry_wrapper_retries_a_429_and_keeps_the_usage() -> None:
    sdk = FakeSdk(Throttled("slow down"), _message(_text("ok"), output_tokens=3))
    sleeps: list[float] = []
    client = RetryingLLMClient(_client(HAIKU, sdk), sleep=sleeps.append)

    completion = client.complete_with_usage("s", "u")

    assert completion.text == "ok"
    assert completion.usage is not None and completion.usage.output_tokens == 3
    assert len(sdk.requests) == 2 and len(sleeps) == 1
