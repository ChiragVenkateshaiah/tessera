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
    # Gemini: Google's published paid-tier rates (ai.google.dev pricing,
    # checked 2026-10-05; output includes thinking tokens). The Agent
    # Platform pricing page didn't render, so reconcile against the GCP
    # billing report after the first sweep. Flash is an introductory rate
    # until 2026-12-31 and doubles to 1.50 / 7.50 from 2027-01-01.
    "gemini-3.8-flash": ModelPrice(0.75, 3.75),
    "gemini-3.1-pro-preview": ModelPrice(2.00, 12.00),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    nvidia_api_key: str
    nvidia_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    # Which provider answers questions. The eval judge stays on NVIDIA NIM
    # either way (plan §3.1.4), so NVIDIA_API_KEY is always required.
    llm_provider: Literal["nvidia", "gemini", "bedrock"] = Field(
        default="nvidia", validation_alias="TESSERA_LLM_PROVIDER"
    )
    # Which pipeline answers (Phase 5 plan §3.1.2): the native stack, or the
    # LangChain stack beside it (built in P5-4..P5-7). --stack overrides it.
    stack: Literal["native", "lc"] = Field(default="native", validation_alias="TESSERA_STACK")
    # The LangChain stack's per-layer switches (Phase 5 plan §3.2.1),
    # TESSERA_LC_<SWITCH>. Each is "native" or "lc", so any one framework
    # component can be measured with the rest held native. The defaults are
    # the "lc-defaults" profile: each layer at the value its evidence chose
    # (P5-4: docs/Tessera_Phase5_Plan.md §3.3, evals/reports/p5-4-ingestion.md).
    lc_loader: Literal["native", "lc"] = Field(default="lc", validation_alias="TESSERA_LC_LOADER")
    lc_splitter: Literal["native", "lc"] = Field(default="native", validation_alias="TESSERA_LC_SPLITTER")
    lc_chunk_size: int = Field(default=1600, validation_alias="TESSERA_LC_CHUNK_SIZE")
    lc_embed_prefix: bool = Field(default=True, validation_alias="TESSERA_LC_EMBED_PREFIX")
    lc_embeddings: Literal["native", "lc"] = Field(default="lc", validation_alias="TESSERA_LC_EMBEDDINGS")
    lc_store: Literal["native", "lc"] = Field(default="lc", validation_alias="TESSERA_LC_STORE")
    lc_indexing: Literal["native", "lc"] = Field(default="lc", validation_alias="TESSERA_LC_INDEXING")
    # Retrieval (P5-5): native retrieve(), or a LangChain retriever through
    # the shared contract — lc (vector), parent_doc, bm25, hybrid,
    # multiquery (calls the router model), rerank (local cross-encoder).
    # Evidence: evals/reports/p5-5-retrieval.md.
    lc_retriever: Literal["native", "lc", "parent_doc", "bm25", "hybrid", "multiquery", "rerank"] = Field(
        default="lc", validation_alias="TESSERA_LC_RETRIEVER"
    )
    lc_hybrid_bm25_weight: float = Field(default=0.25, validation_alias="TESSERA_LC_HYBRID_BM25_WEIGHT")
    # Gemini on Google Cloud's Agent Platform (formerly Vertex AI), the
    # production provider since ADR 0007. Auth is Application Default
    # Credentials; no key in config.
    gcp_project: str | None = Field(default=None, validation_alias="GOOGLE_CLOUD_PROJECT")
    gcp_location: str = Field(default="global", validation_alias="GOOGLE_CLOUD_LOCATION")
    # Ids checked against the live model list 2026-10-05. 3.8 Flash
    # rejects "minimal" thinking, and its default level took 28 s on a
    # one-word reply vs 3 s at "low". Pro exists only as a preview.
    gemini_router_model: str = "gemini-3.8-flash"
    gemini_router_thinking: str | None = "low"
    gemini_answer_model: str = "gemini-3.1-pro-preview"
    gemini_answer_thinking: str | None = "low"
    # Claude on Amazon Bedrock: built and unit-tested, kept as a dormant
    # provider (the AWS account can't pay for Marketplace models; ADR 0007).
    bedrock_region: str = "us-east-1"
    # A dedicated profile, so Tessera never picks up another project's
    # default AWS credentials.
    bedrock_aws_profile: str = "tessera"
    bedrock_router_model: str = "anthropic.claude-haiku-4-5"
    bedrock_answer_model: str = "anthropic.claude-opus-5-5"
    bedrock_answer_effort: str = "medium"
    # LangSmith tracing (Phase 5 plan §3.9): enabled ONLY here, never by
    # LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 (the composition roots refuse
    # those). Runs go through Tessera's redacting client; the key and
    # endpoint are passed to it explicitly. Needs the lc extra.
    langsmith_tracing: bool = Field(default=False, validation_alias="TESSERA_LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, validation_alias="TESSERA_LANGSMITH_API_KEY")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", validation_alias="TESSERA_LANGSMITH_ENDPOINT"
    )
    langsmith_project: str = Field(default="tessera", validation_alias="TESSERA_LANGSMITH_PROJECT")
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
