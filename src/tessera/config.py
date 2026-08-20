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

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: str
    gemini_model: str = "gemini-3.6-flash"
    corpus_dir: Path = Field(
        default=Path("data/corpus"), validation_alias="TESSERA_CORPUS_DIR"
    )
    vectorstore_dir: Path = Field(
        default=Path("data/vectorstore"), validation_alias="TESSERA_VECTORSTORE_DIR"
    )
