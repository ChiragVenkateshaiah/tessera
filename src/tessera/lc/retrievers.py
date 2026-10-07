"""LangChain retrieval (Phase 5 plan §3.4, P5-5): archetype-aware,
permission-safe, plus variants.

**Scope lives in the retriever, not the query.** A ``ScopedStore`` is
built per request, bound to one principal. It is the only way any
retriever here reaches the index, and every query, row read and vector
read through it carries that principal's ``permission_filter`` — the
multi-query rephrasings, the BM25 index, the reranker's pool, the parent
lookups, the superseded probe and the expansion alike. A test records
every ``where`` it sees. One consequence: the trace's count of withheld
restricted chunks is always 0 here, because counting them would mean
querying outside the scope.

**One contract for every retriever.** A candidate generator (vector,
BM25, the hybrid ``EnsembleRetriever``, ``MultiQueryRetriever``, a
cross-encoder reranker, ``ParentDocumentRetriever``) produces a ranked
list; ``ArchetypeRetriever`` then:
1. **re-scores every candidate with cosine similarity** against the
   stored vectors — fused, unioned and reranked lists carry no cosine
   score, but the generation floor and the superseded cutoff need one.
   A ranked generator's order is kept (a fused or reranked order is the
   point of it); a union with no ranking — ``MultiQueryRetriever`` returns
   its queries' hits in query order, the original question last — is
   ordered by cosine to the original question;
2. hands the list to the native ``retrieve_from_candidates``: the
   freshness and permission re-check, per-document diversification, the
   superseded probe and replacements, lookup's expansion.

So a variant changes only which chunks are candidates and in what order.
With the vector generator it reproduces native ``retrieve()`` exactly
(the parity test).

**BM25 and "filter before ranking".** ``BM25Retriever`` ranks an
in-memory corpus. One index over everything would rank restricted text
for a walled user before any post-filter ran (Phase 4 plan §3.5.4). So
each principal's BM25 ranks only over the union of their scopes —
internal, plus each engagement they are cleared for — read through their
``ScopedStore``. Indexes are cached per scope set and per corpus version;
``invalidate()`` drops them (the review workflow calls it, P5-8).

**Package reality.** ``EnsembleRetriever``, ``MultiQueryRetriever``,
``ParentDocumentRetriever``, ``ContextualCompressionRetriever`` and
``CrossEncoderReranker`` come from ``langchain-classic`` (legacy,
maintenance mode); ``BM25Retriever`` and ``HuggingFaceCrossEncoder`` from
``langchain-community``.
"""

from __future__ import annotations

import math
import re
import threading
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_query import DEFAULT_QUERY_PROMPT, LineListOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.stores import InMemoryStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore as LCVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict, PrivateAttr

from tessera.embedding.base import Embedder
from tessera.generation.answer import filter_relevant
from tessera.generation.base import LLMClient
from tessera.ingestion.loader import (
    SENSITIVITY_INTERNAL,
    SENSITIVITY_RESTRICTED,
    Document,
    indexable,
)
from tessera.lc.store import LangChainChromaStore
from tessera.principal import Principal
from tessera.retrieval.retriever import (
    CURRENT_ONLY,
    LOOKUP_CANDIDATE_K,
    SYNTHESIS_CANDIDATE_K,
    RetrievalResult,
    is_permitted,
    permission_filter,
    retrieve,
    retrieve_from_candidates,
)
from tessera.retrieval.router import Archetype
from tessera.store.base import SearchResult, VectorStore
from tessera.store.chroma import search_result_from_row

RETRIEVER_KINDS = ("vector", "parent_doc", "bm25", "hybrid", "multiquery", "rerank")
DEFAULT_HYBRID_BM25_WEIGHT = 0.25  # tuned on query_log.yaml (evals/reports/p5-5-retrieval.md)
RERANK_POOL = 50  # chunks the cross-encoder re-ranks, before selection
PARENT_SUFFIX = "::parent"


def conjoin(where: dict[str, object] | None, extra: dict[str, object] | None) -> dict[str, object] | None:
    if not where:
        return extra
    if not extra:
        return where
    return {"$and": [where, extra]}


def candidate_k(archetype: Archetype) -> int:
    return LOOKUP_CANDIDATE_K if archetype is Archetype.LOOKUP else SYNTHESIS_CANDIDATE_K


# --- the principal-bound store ---


class ScopedStore(VectorStore):
    """The LangChain index as one principal may see it. Every read ANDs
    their permission filter; every result is re-checked."""

    def __init__(self, store: LangChainChromaStore, principal: Principal | None) -> None:
        self.store = store
        self.principal = principal
        self.permission = permission_filter(principal)

    def scoped(self, where: dict[str, object] | None) -> dict[str, object]:
        out = conjoin(where, self.permission)
        assert out is not None
        return out

    def add(self, chunks: Any, embeddings: Any) -> None:
        raise NotImplementedError("a ScopedStore is read-only")

    def count(self) -> int:
        return self.store.count()

    def query(
        self, embedding: list[float], k: int, where: dict[str, object] | None = None
    ) -> list[SearchResult]:
        found = self.store.query(embedding, k, self.scoped(where))
        return [r for r in found if is_permitted(r, self.principal)]

    def rows(self, where: dict[str, object] | None = None) -> list[SearchResult]:
        return [r for r in self.store.rows(self.scoped(where)) if is_permitted(r, self.principal)]

    def vectors(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        return self.store.vectors(chunk_ids, self.scoped(None))

    @property
    def scope_names(self) -> frozenset[str]:
        engagements = self.principal.engagements if self.principal else frozenset()
        return frozenset({"internal", *engagements})


# --- SearchResult <-> LangChain Document ---


def to_document(r: SearchResult) -> LCDocument:
    return LCDocument(
        page_content=r.text,
        metadata={
            "chunk_id": r.chunk_id,
            "score": r.score,
            "document_path": r.document_path,
            "document_title": r.document_title,
            "doc_type": r.doc_type,
            "industry": r.industry,
            "topics": ",".join(r.topics),
            "date": r.date,
            "heading_path": " > ".join(r.heading_path),
            "status": r.status,
            "superseded_by": r.superseded_by or "",
            "sensitivity": r.sensitivity,
            "engagement": r.engagement or "",
        },
    )


def from_document(doc: LCDocument, score: float) -> SearchResult:
    return search_result_from_row(doc.metadata["chunk_id"], doc.page_content, doc.metadata, 1.0 - score)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def rescore(
    docs: list[LCDocument], query_embedding: list[float], scoped: ScopedStore
) -> list[SearchResult]:
    """Contract step 1: every candidate gets its cosine similarity to the
    query, from its stored vector (read in scope), in the generator's
    order. A whole parent document scores as its best chunk. Candidates
    out of scope, or repeated, are dropped."""
    vectors = scoped.vectors([d.metadata["chunk_id"] for d in docs if not d.metadata.get("parent")])
    out: list[SearchResult] = []
    seen: set[str] = set()
    for doc in docs:
        chunk_id = doc.metadata["chunk_id"]
        if chunk_id in seen:
            continue
        if doc.metadata.get("parent"):
            best = scoped.query(
                query_embedding,
                k=1,
                where=conjoin(CURRENT_ONLY, {"document_path": doc.metadata["document_path"]}),
            )
            if not best:
                continue
            score = best[0].score
        elif chunk_id in vectors:
            score = _cosine(query_embedding, vectors[chunk_id])
        else:
            continue
        seen.add(chunk_id)
        out.append(from_document(doc, score))
    return out


# --- candidate generators ---


class _Retriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ScopedVectorRetriever(_Retriever):
    """The vector candidate pool: current chunks in scope, nearest first."""

    scoped: Any
    embedder: Any
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[LCDocument]:
        found = self.scoped.query(self.embedder.embed_query(query), self.k, CURRENT_ONLY)
        return [to_document(r) for r in found]


def bm25_tokens(text: str) -> list[str]:
    """Lowercased word tokens. BM25Retriever's default is ``str.split``,
    which is case-sensitive and keeps punctuation ("Pricing," ≠ "pricing")."""
    return re.findall(r"\w+", text.lower())


class BM25Cache:
    """BM25 indexes, one per scope set (internal + the principal's
    engagements), built from that principal's in-scope rows and keyed by
    the corpus version, so a principal never ranks over text outside
    their scopes."""

    def __init__(self) -> None:
        self._indexes: dict[tuple[frozenset[str], int], list[LCDocument]] = {}
        self._lock = threading.Lock()

    def documents(self, scoped: ScopedStore) -> list[LCDocument]:
        key = (scoped.scope_names, scoped.count())
        with self._lock:
            if key not in self._indexes:
                self._indexes[key] = [to_document(r) for r in scoped.rows(CURRENT_ONLY)]
            return self._indexes[key]

    def retriever(self, scoped: ScopedStore, k: int) -> BM25Retriever:
        return BM25Retriever.from_documents(self.documents(scoped), k=k, preprocess_func=bm25_tokens)

    def invalidate(self) -> None:
        """Drop every index (the corpus changed under review)."""
        with self._lock:
            self._indexes.clear()


class _ScopedLCVectorStore(LCVectorStore):
    """Just enough of a LangChain ``VectorStore`` for
    ``ParentDocumentRetriever``'s child search, over the ScopedStore."""

    def __init__(self, scoped: ScopedStore, embedder: Embedder) -> None:
        self.scoped = scoped
        self.embedder = embedder

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[LCDocument]:
        found = self.scoped.query(self.embedder.embed_query(query), k, CURRENT_ONLY)
        return [to_document(r) for r in found]

    def add_texts(self, texts: Iterable[str], metadatas: Any = None, **kwargs: Any) -> list[str]:
        raise NotImplementedError("read-only")

    @classmethod
    def from_texts(cls, texts: list[str], embedding: Embeddings, metadatas: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError("read-only")


def parent_docstore(documents: Iterable[Document]) -> InMemoryStore:
    """Whole indexable documents keyed by path, for ParentDocumentRetriever.
    The docstore itself is unfiltered — parents are fetched only for child
    hits that passed the scope, and are re-checked."""
    store = InMemoryStore()
    entries = []
    for doc in indexable(list(documents)):
        meta = {
            "chunk_id": f"{doc.path.stem}{PARENT_SUFFIX}",
            "parent": True,
            "document_path": str(doc.path),
            "document_title": doc.title,
            "doc_type": doc.doc_type,
            "industry": doc.industry,
            "topics": ",".join(doc.topics),
            "date": doc.date.isoformat(),
            "heading_path": "",
            "status": doc.status,
            "superseded_by": str(doc.superseded_by_path) if doc.superseded_by_path else "",
            # The same rule as make_chunk: anything not positively internal.
            "sensitivity": SENSITIVITY_RESTRICTED if doc.is_restricted else SENSITIVITY_INTERNAL,
            "engagement": doc.engagement or "",
        }
        entries.append((str(doc.path), LCDocument(page_content=doc.body, metadata=meta)))
    store.mset(entries)
    return store


def multi_query_chain(llm: LLMClient) -> Any:
    """MultiQueryRetriever's query generator over a native LLM client
    (the ``model_client`` switch stays native): its default prompt, the
    native completion, its line parser."""

    def complete(prompt_value: Any) -> str:
        return llm.complete(
            "You rewrite a user's question into alternative search queries.",
            prompt_value.to_string(),
        )

    return DEFAULT_QUERY_PROMPT | RunnableLambda(complete) | LineListOutputParser()


# --- the archetype retriever: generator -> contract ---


class ArchetypeRetriever(_Retriever):
    """``LookupRetriever`` (A) or ``SynthesisRetriever`` (C): a candidate
    generator, then the shared contract (cosine re-score, then
    ``retrieve_from_candidates``)."""

    archetype: Archetype
    scoped: Any
    embedder: Any
    candidates: BaseRetriever
    expand: bool = True
    # A generator whose output order means nothing (MultiQueryRetriever's
    # union, in query order, the original question last) is ordered by
    # cosine to the original question; a fused or reranked order is kept.
    order_by_cosine: bool = False
    _results: dict[str, RetrievalResult] = PrivateAttr(default_factory=dict)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[LCDocument]:
        docs = self.candidates.invoke(query, config={"callbacks": run_manager.get_child()})
        query_embedding = self.embedder.embed_query(query)
        candidates = rescore(docs, query_embedding, self.scoped)
        if self.order_by_cosine:
            candidates.sort(key=lambda r: r.score, reverse=True)
        result = retrieve_from_candidates(
            query,
            self.archetype,
            query_embedding,
            candidates,
            self.scoped,
            principal=self.scoped.principal,
            expand=self.expand,
        )
        self._results[str(run_manager.run_id)] = result
        return [to_document(r) for r in result.results]

    def retrieve(self, query: str, config: RunnableConfig | None = None) -> RetrievalResult:
        """Run as a LangChain retriever (traced, with callbacks) and return
        the full ``RetrievalResult`` — superseded matches and removal
        counts included — for the pipeline and the harness."""
        # invoke() takes the run id as a keyword argument, not from config.
        run_id = uuid.uuid4()
        self.invoke(query, config=config, run_id=run_id)
        return self._results.pop(str(run_id))


def LookupRetriever(**kwargs: Any) -> ArchetypeRetriever:  # noqa: N802 - the plan's names
    return ArchetypeRetriever(archetype=Archetype.LOOKUP, **kwargs)


def SynthesisRetriever(**kwargs: Any) -> ArchetypeRetriever:  # noqa: N802
    return ArchetypeRetriever(archetype=Archetype.SYNTHESIS, **kwargs)


def build_retriever(
    kind: str,
    archetype: Archetype,
    scoped: ScopedStore,
    embedder: Embedder,
    *,
    bm25: BM25Cache | None = None,
    bm25_weight: float = DEFAULT_HYBRID_BM25_WEIGHT,
    llm: LLMClient | None = None,
    cross_encoder: Any = None,
    parents: InMemoryStore | None = None,
) -> ArchetypeRetriever:
    """One principal-bound retriever of ``kind`` (``RETRIEVER_KINDS``)."""
    k = candidate_k(archetype)
    vector = ScopedVectorRetriever(scoped=scoped, embedder=embedder, k=k)
    expand = True
    order_by_cosine = False
    candidates: BaseRetriever
    if kind == "vector":
        candidates = vector
    elif kind == "bm25":
        candidates = (bm25 or BM25Cache()).retriever(scoped, k)
    elif kind == "hybrid":
        candidates = EnsembleRetriever(
            retrievers=[(bm25 or BM25Cache()).retriever(scoped, k), vector],
            weights=[bm25_weight, 1.0 - bm25_weight],
            id_key="chunk_id",
        )
    elif kind == "multiquery":
        if llm is None:
            raise ValueError("the multiquery retriever needs an llm")
        candidates = MultiQueryRetriever(
            retriever=vector, llm_chain=multi_query_chain(llm), include_original=True
        )
        order_by_cosine = True
    elif kind == "rerank":
        if cross_encoder is None:
            raise ValueError("the rerank retriever needs a cross_encoder")
        candidates = ContextualCompressionRetriever(
            base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=k),
            base_retriever=ScopedVectorRetriever(scoped=scoped, embedder=embedder, k=RERANK_POOL),
        )
    elif kind == "parent_doc":
        if parents is None:
            raise ValueError("the parent_doc retriever needs a parent docstore")
        candidates = ParentDocumentRetriever(
            vectorstore=_ScopedLCVectorStore(scoped, embedder),
            docstore=parents,
            id_key="document_path",
            child_splitter=RecursiveCharacterTextSplitter(),  # unused: children are indexed already
            search_kwargs={"k": k},
        )
        expand = False  # parents are whole documents already
    else:
        raise ValueError(f"unknown retriever kind {kind!r}; one of {RETRIEVER_KINDS}")
    return ArchetypeRetriever(
        archetype=archetype,
        scoped=scoped,
        embedder=embedder,
        candidates=candidates,
        expand=expand,
        order_by_cosine=order_by_cosine,
    )


# --- the bridge: native retrieve() as a LangChain retriever ---


class TesseraRetriever(_Retriever):
    """Native ``retrieve()`` as a LangChain ``BaseRetriever``, bound to one
    principal and archetype, with the generation relevance floor applied
    by default — the bridge for third-party chains."""

    archetype: Archetype
    embedder: Any
    store: Any
    principal: Any = None
    apply_floor: bool = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[LCDocument]:
        result = retrieve(query, self.archetype, self.embedder, self.store, principal=self.principal)
        results = filter_relevant(result.results) if self.apply_floor else result.results
        return [to_document(r) for r in results]


def lc_retrieve_fn(
    store: LangChainChromaStore,
    kind: str,
    *,
    bm25: BM25Cache | None = None,
    bm25_weight: float = DEFAULT_HYBRID_BM25_WEIGHT,
    llm: LLMClient | None = None,
    cross_encoder: Any = None,
    parents: InMemoryStore | None = None,
) -> Callable[..., RetrievalResult]:
    """A ``retrieve_fn`` for ``NativePipeline`` (the ``retriever`` switch):
    builds the principal-bound retriever per request, over the LangChain
    index, and returns its ``RetrievalResult``."""
    cache = bm25 or BM25Cache()

    def run(
        query: str,
        archetype: Archetype,
        embedder: Embedder,
        _store: VectorStore,
        *,
        principal: Principal | None = None,
    ) -> RetrievalResult:
        scoped = ScopedStore(store, principal)
        retriever = build_retriever(
            kind,
            archetype,
            scoped,
            embedder,
            bm25=cache,
            bm25_weight=bm25_weight,
            llm=llm,
            cross_encoder=cross_encoder,
            parents=parents,
        )
        return retriever.retrieve(query)

    return run


__all__ = [
    "RETRIEVER_KINDS",
    "ArchetypeRetriever",
    "BM25Cache",
    "LookupRetriever",
    "ScopedStore",
    "SynthesisRetriever",
    "TesseraRetriever",
    "build_retriever",
    "lc_retrieve_fn",
    "parent_docstore",
]

