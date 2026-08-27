"""NVIDIA NIM implementation of the LLMClient interface.

Targets NVIDIA's hosted, OpenAI-compatible endpoint (build.nvidia.com) —
swapping to a different catalog model (e.g. a smaller/faster one) is a
`model=` constructor argument, not a code change; swapping to a
self-hosted NIM container later is a `base_url=` argument for the same
reason.
"""

from __future__ import annotations

from openai import OpenAI

from tessera.generation.base import LLMClient

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
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
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
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("NVIDIA NIM returned an empty completion")
        return content
