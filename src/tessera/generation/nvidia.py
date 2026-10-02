"""NVIDIA NIM implementation of the LLMClient interface.

Targets NVIDIA's hosted, OpenAI-compatible endpoint (build.nvidia.com) —
swapping to a different catalog model (e.g. a smaller/faster one) is a
`model=` constructor argument, not a code change; swapping to a
self-hosted NIM container later is a `base_url=` argument for the same
reason.
"""

from __future__ import annotations

import time

from openai import OpenAI

from tessera.generation.base import Completion, LLMClient, Usage

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaClient(LLMClient):
    """Chat completion via NVIDIA's NIM API (OpenAI-compatible).

    Takes its API key/model as constructor parameters rather than reading
    environment variables itself — config-sourcing is the caller's job
    (constraint #6, CLAUDE.md), not this module's.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        sdk_max_retries: int = 2,
    ) -> None:
        # sdk_max_retries is the openai SDK's own fast retry count. A
        # composition root that wraps this client in RetryingLLMClient
        # should pass 0 so the two don't multiply — the SDK's retries come
        # back within seconds, which is exactly wrong for a 429.
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, max_retries=sdk_max_retries
        )
        self._model = model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(
        self, system: str, user: str, temperature: float = 0.0
    ) -> Completion:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            # Nemotron-3-Ultra is a reasoning-capable model with an
            # optional "thinking" pass exposed via this NIM-specific
            # extra_body field. Disabled here: nothing in this codebase's
            # single-shot completion contract (routing, grounded
            # generation, LLM-as-judge) needs multi-step reasoning, and
            # leaving it off keeps latency and quota spend comparable to
            # Phase 1's Gemini calls.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        latency = time.perf_counter() - start
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("NVIDIA NIM returned an empty completion")
        usage = None
        if response.usage is not None:
            usage = Usage(
                model=self._model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_s=latency,
            )
        return Completion(content, usage)
