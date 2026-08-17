"""Gemini implementation of the LLMClient interface."""

from __future__ import annotations

from google import genai
from google.genai import types

from tessera.generation.base import LLMClient

DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiClient(LLMClient):
    """Chat completion via the Gemini API.

    Takes its API key/model as constructor parameters rather than reading
    environment variables itself — config-sourcing is the caller's job
    (constraint #6, CLAUDE.md), not this module's.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                # No tools are ever passed, so automatic function calling
                # is irrelevant here — disabling it silences an unrelated
                # SDK warning on every call.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty completion")
        return response.text
