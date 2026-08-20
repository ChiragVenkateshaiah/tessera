"""Unit tests for Settings — env var loading, defaults, and the
TESSERA_-prefixed aliases matching .env.example's contract.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tessera.config import Settings


def test_settings_loads_required_and_default_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key == "test-key"
    assert settings.gemini_model == "gemini-3.6-flash"
    assert settings.corpus_dir == Path("data/corpus")
    assert settings.vectorstore_dir == Path("data/vectorstore")


def test_settings_honors_tessera_prefixed_path_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("TESSERA_CORPUS_DIR", "/tmp/other-corpus")
    monkeypatch.setenv("TESSERA_VECTORSTORE_DIR", "/tmp/other-store")

    settings = Settings(_env_file=None)

    assert settings.corpus_dir == Path("/tmp/other-corpus")
    assert settings.vectorstore_dir == Path("/tmp/other-store")


def test_settings_raises_when_gemini_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
