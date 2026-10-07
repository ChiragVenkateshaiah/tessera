"""Indexing with LangChain's ``index()`` API (plan §3.3, the ``indexing``
switch) — LangChain's answer to the stale-chunk problem native solves
with delete-then-add (``tessera.ingestion.indexing``).

``index()`` keeps a ``SQLRecordManager`` of which row ids it wrote for
which ``source``, hashes each document (content and metadata), skips the
unchanged ones and deletes what a run no longer produces. Row ids are
those content hashes, so the chunk id rides in metadata (see
``tessera.lc.store``) — the P5-1 spike's second option, chosen because
the first (id-keyed rows with ``force_update=True``) re-embeds every
chunk on every run and gives up what ``index()`` is for.

**Cleanup mode** (a deliberate departure from the plan's
``cleanup="incremental"``): ``incremental`` only cleans sources present
in this run's batch, so a document newly **quarantined** for review —
held out of the batch, hence absent — keeps its chunks in the index. The
stale-chunk test shows it (strict xfail). Every run here indexes the whole
corpus, so ``cleanup="full"`` is correct and removes them; it is the
default. ``full`` also removes a file deleted from the corpus, which
native delete-then-add does not.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from langchain_core.indexing import RecordManager, index

from tessera.ingestion.chunker import chunk_document
from tessera.ingestion.indexing import IndexingResult, Splitter
from tessera.ingestion.loader import Document, indexable
from tessera.lc.store import TesseraChroma, to_lc_document

Cleanup = Literal["full", "incremental"]


def index_with_record_manager(
    documents: Iterable[Document],
    chroma: TesseraChroma,
    record_manager: RecordManager,
    *,
    split: Splitter = chunk_document,
    cleanup: Cleanup = "full",
) -> IndexingResult:
    """Index the indexable documents' chunks through ``index()``."""
    documents = list(documents)
    kept = indexable(documents)
    chunks = [chunk for doc in kept for chunk in split(doc)]
    result = index(
        [to_lc_document(c) for c in chunks],
        record_manager,
        chroma,
        cleanup=cleanup,
        source_id_key="source",
        key_encoder="sha256",
    )
    return IndexingResult(
        documents=len(documents),
        quarantined=len(documents) - len(kept),
        chunks_added=result["num_added"],
        chunks_deleted=result["num_deleted"],
        chunks_unchanged=result["num_skipped"],
    )
