"""Runtime configuration, loaded from environment / .env via
pydantic-settings.

Only `cli.py` reads this — the query path (`router.py`, `retriever.py`,
`generation/`, `pipeline.py`) takes every dependency as an injected
parameter per CLAUDE.md constraint #6, and never reaches into config for
something that should be passed in. `.env.example` documents every field
below with its exact env var name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from tessera.generation.usage import ModelPrice

# USD per million tokens, for cost per answer (Phase 4). These are
# Anthropic's published first-party rates (claude-api reference, cached
# 2026-09-25): AWS's Bedrock pricing page did not list these models when
# checked on 2026-10-02, so they are UNVERIFIED for Bedrock — confirm
# them before quoting a sweep's cost. Global endpoint; a regional
# endpoint adds 10%. Models left out (the NVIDIA NIM free tier) report
# tokens but no cost.
MODEL_PRICES: dict[str, ModelPrice] = {
    "anthropic.claude-haiku-4-5": ModelPrice(1.00, 5.00),
    "anthropic.claude-sonnet-5-5": ModelPrice(2.00, 10.00),
    "anthropic.claude-opus-4-8": ModelPrice(5.00, 25.00),
    "anthropic.claude-opus-5-5": ModelPrice(4.00, 20.00),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    nvidia_api_key: str
    nvidia_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    # Which provider answers questions. The eval judge stays on NVIDIA NIM
    # either way (plan §3.1.4), so NVIDIA_API_KEY is always required.
    llm_provider: Literal["nvidia", "bedrock"] = Field(
        default="nvidia", validation_alias="TESSERA_LLM_PROVIDER"
    )
    bedrock_region: str = "us-east-1"
    # A dedicated profile, so Tessera never picks up another project's
    # default AWS credentials.
    bedrock_aws_profile: str = "tessera"
    bedrock_router_model: str = "anthropic.claude-haiku-4-5"
    bedrock_answer_model: str = "anthropic.claude-opus-5-5"
    bedrock_answer_effort: str = "medium"
    corpus_dir: Path = Field(
        default=Path("data/corpus"), validation_alias="TESSERA_CORPUS_DIR"
    )
    vectorstore_dir: Path = Field(
        default=Path("data/vectorstore"), validation_alias="TESSERA_VECTORSTORE_DIR"
    )
    expertise_dir: Path = Field(
        default=Path("data/expertise/people"), validation_alias="TESSERA_EXPERTISE_DIR"
    )
    # Ethical walls (Phase 4, plan §3.5.2): engagement -> cleared people.
    access_file: Path = Field(
        default=Path("data/access/walls.yaml"), validation_alias="TESSERA_ACCESS_FILE"
    )
    # Phase 4 feedback loop. Runtime data, gitignored.
    trace_log: Path = Field(
        default=Path("data/traces/traces.jsonl"), validation_alias="TESSERA_TRACE_LOG"
    )
    feedback_file: Path = Field(
        default=Path("data/feedback/feedback.jsonl"), validation_alias="TESSERA_FEEDBACK_FILE"
    )
    feedback_candidates: Path = Field(
        default=Path("data/feedback/candidates.yaml"),
        validation_alias="TESSERA_FEEDBACK_CANDIDATES",
    )
