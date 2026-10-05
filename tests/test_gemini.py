"""GeminiClient against a fake google-genai client — the request it builds
per model, and how it reads the response. Requests and responses are the
SDK's real pydantic types (`extra="forbid"`), so a misspelt config field
fails here, not on a sweep. No Google Cloud calls.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from google.genai import errors, types
from google.genai.models import Models

from tessera.generation.base import Usage
from tessera.generation.gemini import GeminiClient, honours_temperature
from tessera.generation.resilient import RetryingLLMClient

FLASH = "gemini-3.6-flash"
PRO = "gemini-3.1-pro"

SDK_PARAMS = set(inspect.signature(Models.generate_content).parameters) - {"self"}


def _response(
    *texts: str,
    finish: str = "STOP",
    prompt_tokens: int = 1200,
    output_tokens: int = 85,
    thought_tokens: int | None = None,
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=t) for t in texts]),
                finish_reason=finish,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=thought_tokens,
        ),
    )


class FakeSdk:
    """Stands in for genai.Client: records each request, returns the
    scripted responses (or raises the scripted exceptions) in order.
    """

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.models = self

    def generate_content(self, **request: Any) -> types.GenerateContentResponse:
        unknown = set(request) - SDK_PARAMS
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
        assert isinstance(request["config"], types.GenerateContentConfig)
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


def _client(model: str, sdk: FakeSdk, **kw: Any) -> GeminiClient:
    return GeminiClient(model, project="tessera-test", client=sdk, **kw)


def test_honours_temperature_only_before_gemini_3() -> None:
    assert not honours_temperature(FLASH)
    assert not honours_temperature(PRO)
    assert honours_temperature("gemini-2.5-flash")


def test_router_request_sets_minimal_thinking_and_leaves_temperature_alone() -> None:
    sdk = FakeSdk(_response('{"archetype": "A"}'))

    _client(FLASH, sdk, thinking_level="minimal").complete("route this", "a question")

    (request,) = sdk.requests
    config = request["config"]
    assert request["model"] == FLASH
    assert request["contents"] == "a question"
    assert config.system_instruction == "route this"
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
    # Gemini 3 is left at its default temperature (Google's guidance).
    assert config.temperature is None
    assert config.automatic_function_calling.disable is True


def test_older_model_gets_the_requested_temperature_and_default_thinking() -> None:
    sdk = FakeSdk(_response("ok"))

    _client("gemini-2.5-flash", sdk).complete("s", "u", temperature=0.0)

    config = sdk.requests[0]["config"]
    assert config.temperature == 0.0
    assert config.thinking_config is None


def test_returns_text_and_bills_thinking_as_output() -> None:
    sdk = FakeSdk(
        _response("Use the ", "pricing framework [1].", output_tokens=85, thought_tokens=40)
    )

    completion = _client(PRO, sdk, thinking_level="low").complete_with_usage("s", "u")

    assert completion.text == "Use the pricing framework [1]."
    assert isinstance(completion.usage, Usage)
    assert completion.usage.model == PRO
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (1200, 125)
    assert completion.usage.latency_s >= 0.0


@pytest.mark.parametrize("finish", ["SAFETY", "PROHIBITED_CONTENT", "RECITATION"])
def test_a_blocked_answer_raises(finish: str) -> None:
    sdk = FakeSdk(_response("partial", finish=finish))

    with pytest.raises(RuntimeError, match=finish):
        _client(PRO, sdk).complete("s", "u")


def test_a_blocked_prompt_raises() -> None:
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY")
    )
    sdk = FakeSdk(blocked)

    with pytest.raises(RuntimeError, match="blocked the prompt"):
        _client(PRO, sdk).complete("s", "u")


def test_no_text_raises_with_the_finish_reason() -> None:
    sdk = FakeSdk(_response(finish="MAX_TOKENS"))

    with pytest.raises(RuntimeError, match="MAX_TOKENS"):
        _client(PRO, sdk).complete("s", "u")


def test_retry_wrapper_retries_a_429_and_keeps_the_usage() -> None:
    throttled = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})
    sdk = FakeSdk(throttled, _response("ok", output_tokens=3))
    sleeps: list[float] = []
    client = RetryingLLMClient(_client(FLASH, sdk), sleep=sleeps.append)

    completion = client.complete_with_usage("s", "u")

    assert completion.text == "ok"
    assert completion.usage is not None and completion.usage.output_tokens == 3
    assert len(sdk.requests) == 2 and len(sleeps) == 1


def test_retry_wrapper_does_not_retry_a_400() -> None:
    bad = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
    sdk = FakeSdk(bad)
    client = RetryingLLMClient(_client(FLASH, sdk), sleep=lambda s: pytest.fail("no retry"))

    with pytest.raises(errors.ClientError):
        client.complete("s", "u")
