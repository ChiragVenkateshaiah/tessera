"""LangChain ingestion parity (Phase 5 plan §3.3, P5-4 acceptance).

Each LangChain layer is checked against its native counterpart over the
real corpus:
- **loader**: identical document metadata, body and order, and an exact
  round trip back to the native ``Document``; the same validation errors;
- **embeddings**: ``HuggingFaceEmbeddings`` vectors equal native's within
  tolerance;
- **embedding text**: the LangChain store's prefix equals
  ``chunk_embedding_text`` on every chunk;
- **store**: the native retriever returns the same chunks, in the same
  order, with the same scores, from the LangChain collection — written
  directly or through ``index()`` — as from the native one;
- **splitter**: ``<stem>::<n>`` ids and native labels.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langchain_core", reason="the lc extra isn't installed")

from evals.harness import load_cases  # noqa: E402
from tessera.embedding.local import LocalEmbedder  # noqa: E402
from tessera.ingestion.chunker import chunk_corpus, chunk_document, chunk_embedding_text, make_chunk  # noqa: E402
from tessera.ingestion.indexing import index_corpus  # noqa: E402
from tessera.ingestion.loader import CorpusError, Document, indexable, load_corpus  # noqa: E402
from tessera.integrations.langchain import hf_embeddings, lc_chroma, sql_record_manager  # noqa: E402
from tessera.lc.embeddings import LangChainEmbedder, as_langchain  # noqa: E402
from tessera.lc.indexing import index_with_record_manager  # noqa: E402
from tessera.lc.loader import TesseraCorpusLoader, native_document  # noqa: E402
from tessera.lc.splitter import split_document  # noqa: E402
from tessera.lc.store import LangChainChromaStore, embedding_text, lc_metadata  # noqa: E402
from tessera.retrieval.retriever import retrieve  # noqa: E402
from tessera.retrieval.router import Archetype  # noqa: E402
from tessera.store.chroma import ChromaVectorStore  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "corpus"


@pytest.fixture(scope="module")
def documents() -> list[Document]:
    return load_corpus(CORPUS)


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder()


# --- loader ---


def expected_metadata(doc: Document) -> dict[str, Any]:
    """Written out field by field, independently of lc/loader.py."""
    return {
        "source": str(doc.path),
        "relative_path": doc.path.relative_to(CORPUS).as_posix(),
        "title": doc.title,
        "doc_type": doc.doc_type,
        "industry": doc.industry,
        "topics": doc.topics,
        "date": doc.date.isoformat(),
        "status": doc.status,
        "superseded_by": doc.superseded_by,
        "superseded_by_path": None if doc.superseded_by_path is None else str(doc.superseded_by_path),
        "review_status": doc.review_status,
        "sensitivity": doc.sensitivity,
        "engagement": doc.engagement,
    }


def test_the_loader_matches_native_document_for_document(documents: list[Document]) -> None:
    loaded = list(TesseraCorpusLoader(CORPUS).lazy_load())
    native = indexable(documents)

    assert len(loaded) == len(native) and len(native) < len(documents)  # quarantine held out
    for lc_doc, doc in zip(loaded, native, strict=True):
        assert lc_doc.page_content == doc.body
        assert lc_doc.metadata == expected_metadata(doc)
        assert native_document(lc_doc) == doc


def test_the_loader_can_include_quarantined_documents(documents: list[Document]) -> None:
    loaded = TesseraCorpusLoader(CORPUS, include_quarantined=True).load()
    assert [native_document(d) for d in loaded] == documents


def test_the_loader_rejects_what_native_rejects(tmp_path: Path) -> None:
    bad = tmp_path / "methodology" / "x.md"
    bad.parent.mkdir()
    bad.write_text("---\ntitle: x\ndoc_type: memo\nindustry: retail\ntopics: [a]\ndate: 2025-01-01\n---\nbody\n")
    with pytest.raises(CorpusError) as native_error:
        load_corpus(tmp_path)
    with pytest.raises(CorpusError) as lc_error:
        list(TesseraCorpusLoader(tmp_path).lazy_load())
    assert str(lc_error.value) == str(native_error.value)


def test_the_native_chunker_gives_the_same_chunks_from_either_loader(documents: list[Document]) -> None:
    via_lc = [c for d in TesseraCorpusLoader(CORPUS).lazy_load() for c in chunk_document(native_document(d))]
    assert via_lc == chunk_corpus(indexable(documents))


# --- embeddings ---


def test_huggingface_embeddings_match_native_vectors(
    documents: list[Document], embedder: LocalEmbedder
) -> None:
    texts = [chunk_embedding_text(c) for c in chunk_corpus(indexable(documents))]
    lc = LangChainEmbedder(hf_embeddings(), embedder.dimension)
    native_vectors = embedder.embed_documents(texts)
    lc_vectors = lc.embed_documents(texts)
    worst = max(abs(a - b) for u, v in zip(native_vectors, lc_vectors, strict=True) for a, b in zip(u, v))
    assert worst < 1e-5
    query = "pricing framework for a new market"
    assert max(abs(a - b) for a, b in zip(embedder.embed_query(query), lc.embed_query(query))) < 1e-5


def test_the_store_embeds_exactly_the_native_embedding_text(documents: list[Document]) -> None:
    for chunk in chunk_corpus(indexable(documents)):
        assert embedding_text(chunk.text, lc_metadata(chunk)) == chunk_embedding_text(chunk)


# --- store: the native retriever over each collection ---


@pytest.fixture(scope="module")
def stores(
    documents: list[Document], embedder: LocalEmbedder, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("stores")
    native = ChromaVectorStore(persist_dir=root / "native")
    index_corpus(documents, embedder, native)
    direct = LangChainChromaStore(lc_chroma(root / "direct", as_langchain(embedder)))
    index_corpus(documents, embedder, direct)
    chroma = lc_chroma(root / "indexed", as_langchain(embedder))
    index_with_record_manager(documents, chroma, sql_record_manager(root / "records.sqlite"))
    return {"native": native, "lc direct": direct, "lc index()": LangChainChromaStore(chroma)}


def _retrievable_cases() -> list[Any]:
    cases = load_cases(REPO / "evals" / "cases")
    return [c for c in cases if c.archetype in (Archetype.LOOKUP, Archetype.SYNTHESIS)]


@pytest.mark.parametrize("name", ["lc direct", "lc index()"])
def test_the_native_retriever_sees_the_same_index_in_the_lc_collection(
    stores: dict[str, Any], embedder: LocalEmbedder, name: str
) -> None:
    native, lc = stores["native"], stores[name]
    assert lc.count() == native.count()
    cases = _retrievable_cases()
    assert len(cases) > 40
    for case in cases:
        a = retrieve(case.query, case.archetype, embedder, native)
        b = retrieve(case.query, case.archetype, embedder, lc)
        assert [r.chunk_id for r in b.results] == [r.chunk_id for r in a.results], case.id
        assert all(abs(x.score - y.score) < 1e-6 for x, y in zip(a.results, b.results)), case.id
        assert [r.text for r in b.results] == [r.text for r in a.results], case.id
        assert b.removed == a.removed, case.id
        # Superseded matches come from a second, filtered nearest-neighbour
        # query. Chroma's HNSW search is approximate and index() inserts
        # rows in a different order, so it can surface a document's
        # second-best chunk (ac-l03: ::2 at 0.3903 for ::0 at 0.3924).
        # Same documents; scores within that margin.
        assert [m.document_path for m in b.superseded] == [m.document_path for m in a.superseded], case.id
        assert all(abs(x.score - y.score) < 0.01 for x, y in zip(a.superseded, b.superseded)), case.id


# --- splitter ---


def test_the_lc_splitter_keeps_native_ids_and_labels(documents: list[Document]) -> None:
    for doc in indexable(documents):
        chunks = split_document(doc)
        assert [c.chunk_id for c in chunks] == [f"{doc.path.stem}::{i}" for i in range(len(chunks))]
        for c in chunks:
            assert c == make_chunk(doc, c.chunk_index, c.heading_path, c.text)
            # strip_headers=True: the heading line isn't in the text. (A
            # chunk can still start with "#": the character splitter cuts
            # fenced code, so a code comment can open a chunk.)
            assert c.text and not c.text.lstrip("#").startswith(f" {c.heading_path[-1]}" if c.heading_path else "\0")


def test_the_lc_splitter_keeps_the_header_path_and_respects_size() -> None:
    body = "## Approach\n\n### Step one\n\n" + ("word " * 400) + "\n\n## Outcome\n\nDone.\n"
    doc = Document(
        path=Path("/c/methodology/x.md"), title="X", doc_type="methodology", industry="retail",
        topics=["t"], date=date(2025, 1, 1), body=body, sensitivity="internal",
    )
    chunks = split_document(doc, chunk_size=500)
    assert {c.heading_path for c in chunks} == {("Approach", "Step one"), ("Outcome",)}
    assert all(len(c.text) <= 500 for c in chunks)
    assert len([c for c in chunks if c.heading_path == ("Approach", "Step one")]) >= 4
    with_headers = split_document(doc, strip_headers=False)
    # The empty "## Approach" section is folded into its child's text.
    assert with_headers[0].text.startswith("## Approach  \n### Step one")


# --- `tessera ingest --stack lc` ---


@pytest.fixture
def lc_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path]:
    from typer.testing import CliRunner

    from tessera import cli
    from tests.test_stale_chunks import HashEmbedder, write_corpus

    corpus = write_corpus(tmp_path / "corpus", pricing_parts=5, engagement="halcyon", review="reviewed")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("TESSERA_CORPUS_DIR", str(corpus))
    monkeypatch.setenv("TESSERA_VECTORSTORE_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("TESSERA_LC_EMBEDDINGS", "native")  # the hash embedder, no model load
    monkeypatch.setattr(cli, "LocalEmbedder", HashEmbedder)
    return CliRunner(), tmp_path


def _lc_rows(tmp_path: Path, path: str) -> list[str]:
    from tests.test_stale_chunks import HashEmbedder

    chroma = lc_chroma(tmp_path / "index", as_langchain(HashEmbedder()))
    store = LangChainChromaStore(chroma)
    found = store.query(HashEmbedder().embed_query("x"), k=50, where={"document_path": path})
    return sorted(r.chunk_id for r in found)


@pytest.mark.parametrize("indexing", ["lc", "native"])
def test_ingest_stack_lc_builds_and_rebuilds_the_lc_index(
    lc_cli: tuple[Any, Path], monkeypatch: pytest.MonkeyPatch, indexing: str
) -> None:
    from tessera import cli
    from tests.test_stale_chunks import write_corpus

    runner, tmp_path = lc_cli
    monkeypatch.setenv("TESSERA_LC_INDEXING", indexing)
    first = runner.invoke(cli.app, ["ingest", "--stack", "lc"])
    assert first.exit_code == 0, first.output
    assert f"indexing {indexing}" in first.output
    pricing = str(tmp_path / "corpus" / "methodology" / "pricing.md")
    assert len(_lc_rows(tmp_path, pricing)) == 5

    write_corpus(tmp_path / "corpus", pricing_parts=2, engagement="halcyon", review="pending")
    second = runner.invoke(cli.app, ["ingest", "--stack", "lc"])
    assert second.exit_code == 0, second.output
    assert _lc_rows(tmp_path, pricing) == ["pricing::0", "pricing::1"]
    assert _lc_rows(tmp_path, str(tmp_path / "corpus" / "case_studies" / "grocer-reset.md")) == []
    # The native collection was never touched.
    assert ChromaVectorStore(persist_dir=tmp_path / "index").count() == 0


def test_ingest_stack_lc_refuses_lc_indexing_over_a_native_store(
    lc_cli: tuple[Any, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tessera import cli

    runner, _ = lc_cli
    monkeypatch.setenv("TESSERA_LC_STORE", "native")
    result = runner.invoke(cli.app, ["ingest", "--stack", "lc"])
    assert result.exit_code == 2
    assert "TESSERA_LC_INDEXING=lc needs TESSERA_LC_STORE=lc" in result.output
