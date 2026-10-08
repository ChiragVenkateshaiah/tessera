"""LangChain object factories (plan §3.2.2). The composition roots import
this module lazily, only when the LangChain stack is selected.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import chromadb
from langchain_classic.indexes import SQLRecordManager
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from tessera.embedding.local import DEFAULT_MODEL_NAME
from tessera.lc.store import LC_COLLECTION_NAME, TesseraChroma


class SerializedEmbeddings(Embeddings):
    """One encode at a time. A Hugging Face fast tokenizer raises "Already
    borrowed" when two threads use it together (``tessera eval --workers``,
    FastAPI's thread pool); the lock costs nothing next to an encode."""

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return self.inner.embed_query(text)


def hf_embeddings(model_name: str = DEFAULT_MODEL_NAME) -> Embeddings:
    """``HuggingFaceEmbeddings`` over the native model (unnormalized, as
    ``LocalEmbedder`` encodes), safe to share across threads."""
    return SerializedEmbeddings(HuggingFaceEmbeddings(model_name=model_name))


def lc_chroma(
    persist_dir: Path,
    embeddings: Embeddings,
    *,
    embed_prefix: bool = True,
    collection_name: str = LC_COLLECTION_NAME,
    search_ef: int | None = None,
) -> TesseraChroma:
    """The LangChain collection in the same Chroma directory as native's
    (a different collection), in cosine space. ``search_ef`` as
    ``ChromaVectorStore``'s (exact search for the parity tests)."""
    metadata: dict[str, object] = {"hnsw:space": "cosine"}
    if search_ef is not None:
        metadata["hnsw:search_ef"] = search_ef
    return TesseraChroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=chromadb.PersistentClient(path=str(persist_dir)),
        collection_metadata=metadata,
        embed_prefix=embed_prefix,
    )


DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def hf_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER) -> Any:
    """A local cross-encoder for the ``rerank`` retriever (zero LLM calls;
    the model downloads once, ~90 MB)."""
    from langchain_community.cross_encoders import BaseCrossEncoder, HuggingFaceCrossEncoder

    class SerializedCrossEncoder(BaseCrossEncoder):
        """One ``score`` at a time, as ``SerializedEmbeddings``."""

        def __init__(self, inner: BaseCrossEncoder) -> None:
            self.inner = inner
            self._lock = threading.Lock()

        def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
            with self._lock:
                return self.inner.score(text_pairs)

    return SerializedCrossEncoder(HuggingFaceCrossEncoder(model_name=model_name))


def chat_nvidia(api_key: str, model: str, *, rate_limiter: Any = None) -> Any:
    """``ChatNVIDIA`` sending what ``NvidiaClient`` sends (plan §3.5): the
    model, ``temperature`` 0.0, thinking off, and **no** ``max_tokens``
    (``None`` omits it; the default sends 1024). The P5-6 parity test pins
    the request body."""
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=None,
        model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
        rate_limiter=rate_limiter,
    )


def chat_gemini(
    project: str,
    model: str,
    *,
    location: str = "global",
    thinking_level: str | None = "low",
    max_output_tokens: int = 16_000,
    rate_limiter: Any = None,
) -> Any:
    """``ChatGoogleGenerativeAI`` on Agent Platform (ADC, no key) sending
    what ``GeminiClient`` sends: the same model ids, ``thinking_level``,
    ``max_output_tokens`` 16,000, location ``global``, and **SDK retries
    off** — ``GeminiClient`` makes one attempt and leaves retries to the
    ``retry`` switch."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {}
    if thinking_level is not None:
        kwargs["thinking_level"] = thinking_level
    return ChatGoogleGenerativeAI(
        model=model,
        vertexai=True,
        project=project,
        location=location,
        max_output_tokens=max_output_tokens,
        max_retries=0,
        rate_limiter=rate_limiter,
        **kwargs,
    )


def sql_record_manager(path: Path, namespace: str = f"chroma/{LC_COLLECTION_NAME}") -> SQLRecordManager:
    """``index()``'s record manager, in a SQLite file."""
    manager = SQLRecordManager(namespace, db_url=f"sqlite:///{path}")
    manager.create_schema()
    return manager
