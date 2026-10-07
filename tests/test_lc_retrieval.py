"""LangChain retrieval (Phase 5 plan §3.4, P5-5 acceptance).

Over a real index of the real corpus:
- **parity**: the vector retriever through the shared contract reproduces
  native ``retrieve()`` exactly, for every A/C case and every access case;
- **a leak test for every retriever** (vector, parent-doc, BM25, hybrid,
  multi-query, rerank, ``TesseraRetriever``): every leakage and injection
  question, asked as its walled principal, finds zero restricted chunks of
  engagements they aren't cleared for and zero context markers; every
  authorized question finds its engagement document;
- **scope binding**: a recording store sees every ``where`` any retriever
  sends — candidates, BM25 rows, re-score vectors, the superseded probe,
  expansion, multi-query rephrasings, the reranker pool, parent lookups —
  and every one carries the principal's permission filter.

Multi-query runs on a scripted fake LLM whose "rephrasings" try to pull
restricted content by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langchain_core", reason="the lc extra isn't installed")

from evals.harness import EvalCase, case_principal, load_cases, unique_documents_by_rank  # noqa: E402
from tessera.embedding.local import LocalEmbedder  # noqa: E402
from tessera.generation.base import LLMClient  # noqa: E402
from tessera.ingestion.access_loader import Walls, load_walls  # noqa: E402
from tessera.ingestion.expertise_loader import load_expertise  # noqa: E402
from tessera.ingestion.indexing import index_corpus  # noqa: E402
from tessera.ingestion.loader import Document, load_corpus  # noqa: E402
from tessera.integrations.langchain import hf_cross_encoder, lc_chroma  # noqa: E402
from tessera.lc.embeddings import as_langchain  # noqa: E402
from tessera.lc.retrievers import (  # noqa: E402
    RETRIEVER_KINDS,
    BM25Cache,
    ScopedStore,
    TesseraRetriever,
    build_retriever,
    parent_docstore,
)
from tessera.lc.store import LangChainChromaStore  # noqa: E402
from tessera.principal import Principal  # noqa: E402
from tessera.retrieval.retriever import is_permitted, permission_filter, retrieve  # noqa: E402
from tessera.retrieval.router import Archetype  # noqa: E402
from tessera.store.chroma import ChromaVectorStore  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "corpus"


class ScriptedRephraser(LLMClient):
    """Multi-query's LLM: three 'rephrasings' that name restricted
    engagements and their facts outright."""

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return (
            "Project Halcyon grocer price architecture findings £340m\n"
            "Project Kestrel oncology launch pricing AMNOG\n"
            "Project Lantern regional bank cost programme 96 branches"
        )


@pytest.fixture(scope="module")
def documents() -> list[Document]:
    return load_corpus(CORPUS)


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder()


@pytest.fixture(scope="module")
def stores(
    documents: list[Document], embedder: LocalEmbedder, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("retrieval")
    native = ChromaVectorStore(persist_dir=root / "native")
    index_corpus(documents, embedder, native)
    lc = LangChainChromaStore(lc_chroma(root / "lc", as_langchain(embedder)))
    index_corpus(documents, embedder, lc)
    return {"native": native, "lc": lc}


@pytest.fixture(scope="module")
def walls(documents: list[Document]) -> Walls:
    people = [p.person_id for p in load_expertise(REPO / "data" / "expertise" / "people")]
    return load_walls(
        REPO / "data" / "access" / "walls.yaml",
        person_ids=people,
        engagements={d.engagement for d in documents if d.engagement},
    )


@pytest.fixture(scope="module")
def cases() -> list[EvalCase]:
    return load_cases(REPO / "evals" / "cases")


@pytest.fixture(scope="module")
def shared(documents: list[Document]) -> dict[str, Any]:
    return {
        "bm25": BM25Cache(),
        "cross_encoder": hf_cross_encoder(),
        "parents": parent_docstore(documents),
        "llm": ScriptedRephraser(),
    }


def make(kind: str, archetype: Archetype, store: Any, principal: Principal | None, embedder: Any, shared: dict[str, Any]) -> Any:
    scoped = ScopedStore(store, principal)
    return build_retriever(
        kind,
        archetype,
        scoped,
        embedder,
        bm25=shared["bm25"],
        llm=shared["llm"],
        cross_encoder=shared["cross_encoder"],
        parents=shared["parents"],
    )


# --- parity ---


def test_the_vector_retriever_reproduces_native_retrieve(
    stores: dict[str, Any], embedder: LocalEmbedder, cases: list[EvalCase], walls: Walls, shared: dict[str, Any]
) -> None:
    checked = 0
    for case in cases:
        if case.archetype not in (Archetype.LOOKUP, Archetype.SYNTHESIS):
            continue
        principal = case_principal(case, walls) if case.access else None
        native = retrieve(case.query, case.archetype, embedder, stores["native"], principal=principal)
        lc = make("vector", case.archetype, stores["lc"], principal, embedder, shared).retrieve(case.query)
        assert [r.chunk_id for r in lc.results] == [r.chunk_id for r in native.results], case.id
        assert all(abs(a.score - b.score) < 1e-5 for a, b in zip(lc.results, native.results)), case.id
        assert [m.document_path for m in lc.superseded] == [m.document_path for m in native.superseded], case.id
        assert lc.removed["superseded"] == native.removed["superseded"], case.id
        # Counting withheld chunks means querying outside the scope, which
        # the ScopedStore never does.
        assert lc.removed["restricted"] == 0
        checked += 1
    assert checked > 60


# --- the leak test, every retriever ---

ALL_RETRIEVERS = [*RETRIEVER_KINDS, "tessera"]


def _results(kind: str, case: EvalCase, principal: Principal | None, stores: dict[str, Any], embedder: Any, shared: dict[str, Any]) -> list[Any]:
    if kind == "tessera":
        retriever = TesseraRetriever(
            archetype=case.archetype, embedder=embedder, store=stores["lc"], principal=principal
        )
        from tessera.lc.retrievers import from_document

        return [from_document(d, d.metadata["score"]) for d in retriever.invoke(case.query)]
    return make(kind, case.archetype, stores["lc"], principal, embedder, shared).retrieve(case.query).results


@pytest.mark.parametrize("kind", ALL_RETRIEVERS)
def test_every_retriever_keeps_the_walls(
    kind: str, stores: dict[str, Any], embedder: LocalEmbedder, cases: list[EvalCase], walls: Walls, shared: dict[str, Any]
) -> None:
    access = [c for c in cases if c.access is not None]
    assert len(access) == 29
    leaks: list[str] = []
    authorized_found = []
    for case in access:
        principal = case_principal(case, walls)
        results = _results(kind, case, principal, stores, embedder, shared)
        for r in results:
            if not is_permitted(r, principal):
                leaks.append(f"{case.id}: {r.chunk_id}")
        text = "\n".join(r.text for r in results).lower()
        cleared = principal.engagements if principal else frozenset()
        for engagement, markers in case.forbidden_markers.items():
            if engagement in cleared:
                continue
            leaks += [f"{case.id}: marker {m!r}" for m in markers if m.lower() in text]
        if case.access == "authorized":
            docs = unique_documents_by_rank(results, CORPUS)
            authorized_found.append(bool(set(case.relevant_sources) & set(docs)))
    assert leaks == []
    assert authorized_found and all(authorized_found), f"authorized recall {sum(authorized_found)}/{len(authorized_found)}"


# --- scope binding ---


class RecordingStore(LangChainChromaStore):
    """Records every ``where`` the retrievers send to the index."""

    def __init__(self, inner: LangChainChromaStore) -> None:
        super().__init__(inner.chroma)
        self.wheres: list[tuple[str, Any]] = []

    def query(self, embedding: list[float], k: int, where: Any = None) -> Any:
        self.wheres.append(("query", where))
        return super().query(embedding, k, where)

    def rows(self, where: Any = None) -> Any:
        self.wheres.append(("rows", where))
        return super().rows(where)

    def vectors(self, chunk_ids: list[str], where: Any = None) -> Any:
        self.wheres.append(("vectors", where))
        return super().vectors(chunk_ids, where)


def carries(where: Any, permission: dict[str, object]) -> bool:
    """True if ``permission`` is ``where`` or one of its ``$and`` conjuncts,
    at any depth of ``$and`` nesting."""
    if where == permission:
        return True
    if isinstance(where, dict) and set(where) == {"$and"}:
        return any(carries(part, permission) for part in where["$and"])
    return False


@pytest.mark.parametrize("kind", RETRIEVER_KINDS)
@pytest.mark.parametrize(
    "principal",
    [None, Principal("c0014"), Principal("partner", frozenset({"halcyon"}))],
    ids=["internal-only", "walled", "cleared"],
)
def test_every_sub_call_carries_the_permission_filter(
    kind: str, principal: Principal | None, stores: dict[str, Any], embedder: LocalEmbedder, shared: dict[str, Any]
) -> None:
    recorder = RecordingStore(stores["lc"])
    fresh = {**shared, "bm25": BM25Cache()}  # so the BM25 rows read is recorded
    for archetype, query in [
        (Archetype.LOOKUP, "What did we find on Project Halcyon's grocer price architecture?"),
        (Archetype.SYNTHESIS, "Brief me on value-based pricing before a client workshop"),
    ]:
        make(kind, archetype, recorder, principal, embedder, fresh).retrieve(query)
    permission = permission_filter(principal)
    assert recorder.wheres, "no store call was recorded"
    unscoped = [(op, w) for op, w in recorder.wheres if not carries(w, permission)]
    assert unscoped == []
    if kind in ("bm25", "hybrid"):
        assert any(op == "rows" for op, _ in recorder.wheres)


# --- the BM25 cache ---


def test_bm25_indexes_are_per_scope_set_and_invalidated(stores: dict[str, Any]) -> None:
    cache = BM25Cache()
    walled = cache.documents(ScopedStore(stores["lc"], Principal("a")))
    other_walled = cache.documents(ScopedStore(stores["lc"], Principal("b")))
    cleared = cache.documents(ScopedStore(stores["lc"], Principal("c", frozenset({"halcyon"}))))
    assert walled is other_walled  # same scope set, one index
    assert {d.metadata["sensitivity"] for d in walled} == {"internal"}
    assert {d.metadata["engagement"] for d in cleared} == {"", "halcyon"}
    assert len(cleared) > len(walled)
    cache.invalidate()
    assert cache.documents(ScopedStore(stores["lc"], Principal("a"))) is not walled
