"""Claude on Amazon Bedrock — the Phase 4 implementation of the LLMClient
port (plan §3.1.1).

Uses the Anthropic SDK's `AnthropicBedrockMantle`, which speaks the
Messages API at `bedrock-mantle.{region}.api.aws` (Opus 4.7 and later,
plus Haiku 4.5). Credentials come from the AWS credential chain — a named
profile, never keys in code — and IAM authorizes the call as
`bedrock-mantle:CreateInference`. Like `NvidiaClient`, every setting is a
constructor parameter; the composition root reads config.
"""

from __future__ import annotations

import time
from typing import Any

from anthropic import AnthropicBedrockMantle

from tessera.generation.base import Completion, LLMClient, Usage

# Non-streaming requests stay well under the SDK's HTTP timeout at this
# size; answers here are a few hundred tokens, plus Opus 5.5's thinking.
DEFAULT_MAX_TOKENS = 16_000


def accepts_temperature(model: str) -> bool:
    """Opus 4.7+, Sonnet 5.x and Fable reject a temperature with a 400;
    Haiku 4.5 still takes it. Tessera asks for 0.0 (routing should be as
    repeatable as the model allows), so it's sent only where accepted.
    """
    return "haiku-4-5" in model


class BedrockClient(LLMClient):
    """One Claude model on Bedrock.

    effort sets `output_config.effort` (e.g. "medium" for answers on Opus
    5.5, whose thinking can't be turned off). Leave it None for Haiku 4.5,
    which rejects the field. ``client`` is for tests; normally the SDK
    client is built from the region and profile.
    """

    def __init__(
        self,
        model: str,
        *,
        aws_region: str,
        aws_profile: str | None = None,
        effort: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        sdk_max_retries: int = 2,
        client: Any = None,
    ) -> None:
        # As with NvidiaClient: pass sdk_max_retries=0 when wrapping this in
        # RetryingLLMClient so the two retry layers don't multiply.
        self._client = client or AnthropicBedrockMantle(
            aws_region=aws_region, aws_profile=aws_profile, max_retries=sdk_max_retries
        )
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(
        self, system: str, user: str, temperature: float = 0.0
    ) -> Completion:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if accepts_temperature(self._model):
            # SDK 1.x dropped `temperature` from messages.create(); the API
            # still honours it on Haiku 4.5, so it goes in the raw body.
            request["extra_body"] = {"temperature": temperature}
        if self._effort is not None:
            request["output_config"] = {"effort": self._effort}

        start = time.perf_counter()
        message = self._client.messages.create(**request)
        latency = time.perf_counter() - start

        if message.stop_reason == "refusal":
            raise RuntimeError(f"{self._model} declined the request (stop_reason=refusal)")
        # Opus 5.5 returns thinking blocks ahead of the text; only text is
        # the answer.
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        if not text:
            raise RuntimeError(
                f"{self._model} returned no text (stop_reason={message.stop_reason})"
            )
        usage = Usage(
            model=self._model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_s=latency,
        )
        return Completion(text, usage)
