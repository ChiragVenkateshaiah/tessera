"""Gemini on Google Cloud's Agent Platform (formerly Vertex AI) — the
Phase 4 production LLMClient. ADR 0007 records why it replaced Claude on
Bedrock.

Uses the `google-genai` SDK with `enterprise=True`, which calls
`aiplatform.googleapis.com` as a project in a location. Credentials come
from Application Default Credentials (`gcloud auth application-default
login` locally, the service account on Cloud Run), never keys in code.
Like `NvidiaClient` and `BedrockClient`, every setting is a constructor
parameter; the composition root reads config.
"""

from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import types

from tessera.generation.base import Completion, LLMClient, Usage

# Thinking tokens count against this limit, so it's generous; answers
# here are a few hundred tokens.
DEFAULT_MAX_OUTPUT_TOKENS = 16_000

# Finish reasons that mean the model (or its filters) declined, as
# opposed to running out of room.
_BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


def honours_temperature(model: str) -> bool:
    """Google advises leaving Gemini 3+ at its default temperature (1.0):
    lower values can cause looping or degraded output. Older models take
    Tessera's 0.0. The eval sweep, not this rule, is the final judge.
    """
    return not model.startswith("gemini-3")


class GeminiClient(LLMClient):
    """One Gemini model on Agent Platform.

    thinking_level sets `ThinkingConfig.thinking_level` ("minimal", "low",
    "medium", "high"; which ones a model accepts varies). None leaves the
    model's default. ``client`` is for tests; normally the SDK client is
    built from the project and location.
    """

    def __init__(
        self,
        model: str,
        *,
        project: str,
        location: str = "global",
        thinking_level: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        client: Any = None,
    ) -> None:
        # The SDK doesn't retry unless asked to (HttpOptions.retry_options),
        # so wrapping this in RetryingLLMClient gives a single retry layer.
        self._client = client or genai.Client(
            enterprise=True, project=project, location=location
        )
        self._model = model
        self._thinking_level = thinking_level
        self._max_output_tokens = max_output_tokens

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(
        self, system: str, user: str, temperature: float = 0.0
    ) -> Completion:
        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": self._max_output_tokens,
            # No tools are ever passed, so automatic function calling is
            # irrelevant; disabling it silences an SDK warning per call.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if honours_temperature(self._model):
            config["temperature"] = temperature
        if self._thinking_level is not None:
            config["thinking_config"] = types.ThinkingConfig(
                thinking_level=self._thinking_level.upper()
            )

        start = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(**config),
        )
        latency = time.perf_counter() - start

        block = getattr(response.prompt_feedback, "block_reason", None)
        if block:
            raise RuntimeError(f"{self._model} blocked the prompt ({_name(block)})")
        finish = _name(response.candidates[0].finish_reason) if response.candidates else None
        if finish in _BLOCKED:
            raise RuntimeError(f"{self._model} declined the request (finish_reason={finish})")
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError(f"{self._model} returned no text (finish_reason={finish})")

        meta = response.usage_metadata
        usage = Usage(
            model=self._model,
            input_tokens=meta.prompt_token_count or 0,
            # Thinking tokens are billed at the output rate.
            output_tokens=(meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0),
            latency_s=latency,
        )
        return Completion(text, usage)


def _name(value: object) -> str:
    """An SDK enum's name ("SAFETY"), or the string the API sent."""
    return getattr(value, "name", None) or str(value)
