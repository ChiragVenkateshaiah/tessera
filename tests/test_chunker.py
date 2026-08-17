from datetime import date
from pathlib import Path

from tessera.ingestion.chunker import DEFAULT_MAX_CHUNK_WORDS, chunk_corpus, chunk_document
from tessera.ingestion.loader import Document, load_corpus

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


def _doc(body: str, **overrides: object) -> Document:
    defaults: dict[str, object] = dict(
        path=Path("fixture.md"),
        title="Fixture Doc",
        doc_type="methodology",
        industry="cross-industry",
        topics=["fixture"],
        date=date(2024, 1, 1),
    )
    defaults.update(overrides)
    return Document(body=body, **defaults)  # type: ignore[arg-type]


def test_splits_on_headings() -> None:
    doc = _doc(
        "## Overview\n\nFirst section text.\n\n## Details\n\nSecond section text.\n"
    )

    chunks = chunk_document(doc)

    assert [c.heading_path for c in chunks] == [("Overview",), ("Details",)]
    assert "First section" in chunks[0].text
    assert "Second section" in chunks[1].text


def test_nested_headings_produce_full_path() -> None:
    doc = _doc(
        "## Framework\n\n### Step One\n\nDo the first thing.\n\n"
        "### Step Two\n\nDo the second thing.\n"
    )

    chunks = chunk_document(doc)

    assert chunks[0].heading_path == ("Framework", "Step One")
    assert chunks[1].heading_path == ("Framework", "Step Two")


def test_heading_with_no_own_text_produces_no_chunk_but_children_do() -> None:
    # A heading that immediately hands off to children (no own paragraph)
    # should not produce an empty chunk itself.
    doc = _doc("## Parent\n\n### Child\n\nChild content.\n")

    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Parent", "Child")


def test_metadata_carried_onto_every_chunk() -> None:
    doc = _doc(
        "## Overview\n\nSome text.\n",
        title="My Title",
        doc_type="thought_leadership",
        industry="retail",
        topics=["a", "b"],
        date=date(2023, 5, 5),
    )

    chunks = chunk_document(doc)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.document_title == "My Title"
    assert c.doc_type == "thought_leadership"
    assert c.industry == "retail"
    assert c.topics == ["a", "b"]
    assert c.date == date(2023, 5, 5)
    assert c.document_path == doc.path


def test_fenced_code_block_never_split_even_when_oversized() -> None:
    fence_lines = "\n".join(f"line {i} of formula" for i in range(60))
    body = f"## Formula\n\n```\n{fence_lines}\n```\n"
    doc = _doc(body)

    chunks = chunk_document(doc, max_words=20)

    assert len(chunks) == 1
    assert chunks[0].text.count("```") == 2
    assert "line 0 of formula" in chunks[0].text
    assert "line 59 of formula" in chunks[0].text


def test_table_never_split_even_when_oversized() -> None:
    rows = "\n".join(f"| row {i} | value {i} |" for i in range(30))
    body = f"## Data\n\n| Col A | Col B |\n|---|---|\n{rows}\n"
    doc = _doc(body)

    chunks = chunk_document(doc, max_words=10)

    assert len(chunks) == 1
    assert chunks[0].text.count("\n") == body.count("\n") - 3  # heading + blanks stripped
    assert "row 0" in chunks[0].text
    assert "row 29" in chunks[0].text


def test_oversized_prose_section_subdivides_at_paragraph_boundaries() -> None:
    paragraphs = [f"Paragraph {i} has several words in it for counting." for i in range(10)]
    body = "## Long Section\n\n" + "\n\n".join(paragraphs) + "\n"
    doc = _doc(body)

    chunks = chunk_document(doc, max_words=20)

    assert len(chunks) > 1
    for c in chunks:
        assert c.heading_path == ("Long Section",)
    # every paragraph survives somewhere, none dropped
    rejoined = " ".join(c.text for c in chunks)
    for p in paragraphs:
        assert p in rejoined


def test_chunk_ids_unique_and_indices_sequential() -> None:
    doc = _doc("## A\n\ntext a\n\n## B\n\ntext b\n\n## C\n\ntext c\n")

    chunks = chunk_document(doc)

    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_preamble_before_first_heading_is_not_silently_dropped() -> None:
    doc = _doc("Some intro text before any heading.\n\n## Overview\n\nBody.\n")

    chunks = chunk_document(doc)

    assert any("intro text" in c.text for c in chunks)
    # preamble chunk has an empty heading path (no section owns it)
    assert chunks[0].heading_path == ()


# --- Integration: the real Task 1 corpus ---


def test_real_corpus_chunks_reasonably() -> None:
    docs = load_corpus(CORPUS_DIR)
    chunks = chunk_corpus(docs)

    # 52 documents, several headings each — a sane range, not a single
    # mega-chunk per doc and not thousands of one-line slivers.
    assert 150 <= len(chunks) <= 600

    for c in chunks:
        assert c.text.strip(), f"empty chunk from {c.document_path}"
        assert c.heading_path, f"chunk from {c.document_path} has no heading path"

    doc_paths = {c.document_path for c in chunks}
    assert len(doc_paths) == 52  # every document contributed at least one chunk


def test_real_corpus_no_chunk_orphaned_from_source_metadata() -> None:
    docs = load_corpus(CORPUS_DIR)
    doc_by_path = {d.path: d for d in docs}
    chunks = chunk_corpus(docs)

    for c in chunks:
        source = doc_by_path[c.document_path]
        assert c.document_title == source.title
        assert c.doc_type == source.doc_type
        assert c.industry == source.industry
        assert c.topics == source.topics
        assert c.date == source.date


def test_real_corpus_headings_not_split_mid_section() -> None:
    # Every chunk's heading_path must correspond to an actual heading
    # found verbatim in its source document body.
    docs = load_corpus(CORPUS_DIR)
    chunks = chunk_corpus(docs)
    doc_by_path = {d.path: d for d in docs}

    for c in chunks:
        if not c.heading_path:
            continue
        body = doc_by_path[c.document_path].body
        last_heading = c.heading_path[-1]
        assert last_heading in body, (
            f"{c.chunk_id}: heading {last_heading!r} not found verbatim "
            f"in source body"
        )


def test_real_corpus_structural_elements_survive_intact() -> None:
    docs = {d.path.name: d for d in load_corpus(CORPUS_DIR)}

    checkbox_doc = docs["due-diligence-financial-dd-information-request-list.md"]
    chunks = chunk_document(checkbox_doc)
    assert sum(c.text.count("- [ ]") for c in chunks) == checkbox_doc.body.count(
        "- [ ]"
    )

    fence_doc = docs["supply-chain-network-optimization-model-specification.md"]
    chunks = chunk_document(fence_doc)
    for c in chunks:
        assert c.text.count("```") % 2 == 0, f"unbalanced fence in {c.chunk_id}"
