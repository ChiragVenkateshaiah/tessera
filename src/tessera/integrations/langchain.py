"""LangChain object factories (plan §3.2.2). The composition roots import
this module lazily, only when the LangChain stack is selected.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_classic.indexes import SQLRecordManager
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from tessera.embedding.local import DEFAULT_MODEL_NAME
from tessera.lc.store import LC_COLLECTION_NAME, TesseraChroma


def hf_embeddings(model_name: str = DEFAULT_MODEL_NAME) -> Embeddings:
    """``HuggingFaceEmbeddings`` over the native model (unnormalized, as
    ``LocalEmbedder`` encodes)."""
    return HuggingFaceEmbeddings(model_name=model_name)


def lc_chroma(
    persist_dir: Path,
    embeddings: Embeddings,
    *,
    embed_prefix: bool = True,
    collection_name: str = LC_COLLECTION_NAME,
) -> TesseraChroma:
    """The LangChain collection in the same Chroma directory as native's
    (a different collection), in cosine space."""
    return TesseraChroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=chromadb.PersistentClient(path=str(persist_dir)),
        collection_metadata={"hnsw:space": "cosine"},
        embed_prefix=embed_prefix,
    )


def sql_record_manager(path: Path, namespace: str = f"chroma/{LC_COLLECTION_NAME}") -> SQLRecordManager:
    """``index()``'s record manager, in a SQLite file."""
    manager = SQLRecordManager(namespace, db_url=f"sqlite:///{path}")
    manager.create_schema()
    return manager
