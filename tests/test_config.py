"""Unit tests for Settings — env var loading, defaults, and the
TESSERA_-prefixed aliases matching .env.example's contract.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tessera.config import Settings


def test_settings_loads_required_and_default_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.nvidia_api_key == "test-key"
    assert settings.nvidia_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert settings.corpus_dir == Path("data/corpus")
    assert settings.vectorstore_dir == Path("data/vectorstore")


def test_settings_honors_tessera_prefixed_path_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("TESSERA_CORPUS_DIR", "/tmp/other-corpus")
    monkeypatch.setenv("TESSERA_VECTORSTORE_DIR", "/tmp/other-store")

    settings = Settings(_env_file=None)

    assert settings.corpus_dir == Path("/tmp/other-corpus")
    assert settings.vectorstore_dir == Path("/tmp/other-store")


def test_settings_raises_when_nvidia_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_llm_provider_defaults_to_nvidia_with_bedrock_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    for var in ("TESSERA_LLM_PROVIDER", "BEDROCK_AWS_PROFILE", "BEDROCK_ANSWER_MODEL"):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "nvidia"
    assert settings.bedrock_aws_profile == "tessera"
    assert settings.bedrock_router_model == "anthropic.claude-haiku-4-5"
    assert settings.bedrock_answer_model == "anthropic.claude-opus-5-5"


def test_llm_provider_and_bedrock_models_come_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("TESSERA_LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_ANSWER_MODEL", "anthropic.claude-opus-4-8")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "bedrock"
    assert settings.bedrock_answer_model == "anthropic.claude-opus-4-8"


def test_unknown_llm_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("TESSERA_LLM_PROVIDER", "openai")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_every_default_bedrock_model_has_a_price() -> None:
    from tessera.config import MODEL_PRICES

    defaults = Settings.model_fields
    for name in ("bedrock_router_model", "bedrock_answer_model"):
        assert defaults[name].default in MODEL_PRICES
