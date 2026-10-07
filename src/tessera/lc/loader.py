"""``TesseraCorpusLoader``: the corpus as LangChain ``Document``s (plan §3.3).

One ``Document`` per markdown file: ``page_content`` is the body, and
``metadata`` is the validated, resolved front matter. Validation is the
native loader's (``load_corpus`` → ``front_matter_problems`` and the
supersession checks), so both stacks accept and reject exactly the same
corpus, and an unlabelled document resolves to the same sensitivity.
Quarantined documents are held out, as ``indexable()`` holds them out.

``lazy_load()`` is the primary method. The cross-document supersession
check needs every file's front matter before the first document can be
trusted, so the corpus is validated as a whole first and the documents
are then yielded one at a time.

``native_document()`` turns a loaded ``Document`` back into the native
dataclass, so the ``loader`` switch can feed either splitter; the parity
test checks the round trip is exact.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument

from tessera.ingestion.loader import Document, load_corpus


def document_metadata(doc: Document, corpus_dir: Path) -> dict[str, Any]:
    """A native document's fields as LangChain metadata (JSON types only)."""
    return {
        "source": str(doc.path),
        "relative_path": doc.path.relative_to(corpus_dir).as_posix(),
        "title": doc.title,
        "doc_type": doc.doc_type,
        "industry": doc.industry,
        "topics": list(doc.topics),
        "date": doc.date.isoformat(),
        "status": doc.status,
        "superseded_by": doc.superseded_by,
        "superseded_by_path": str(doc.superseded_by_path) if doc.superseded_by_path else None,
        "review_status": doc.review_status,
        "sensitivity": doc.sensitivity,
        "engagement": doc.engagement,
    }


def native_document(document: LCDocument) -> Document:
    """The native ``Document`` a ``TesseraCorpusLoader`` document came from."""
    m = document.metadata
    return Document(
        path=Path(m["source"]),
        title=m["title"],
        doc_type=m["doc_type"],
        industry=m["industry"],
        topics=list(m["topics"]),
        date=date.fromisoformat(m["date"]),
        body=document.page_content,
        status=m["status"],
        superseded_by=m["superseded_by"],
        superseded_by_path=Path(m["superseded_by_path"]) if m["superseded_by_path"] else None,
        review_status=m["review_status"],
        sensitivity=m["sensitivity"],
        engagement=m["engagement"],
    )


class TesseraCorpusLoader(BaseLoader):
    """Loads the corpus under ``corpus_dir``, sorted by path."""

    def __init__(self, corpus_dir: Path, *, include_quarantined: bool = False) -> None:
        self.corpus_dir = corpus_dir
        self.include_quarantined = include_quarantined

    def lazy_load(self) -> Iterator[LCDocument]:
        for doc in load_corpus(self.corpus_dir):
            if doc.is_quarantined and not self.include_quarantined:
                continue
            metadata = document_metadata(doc, self.corpus_dir)
            yield LCDocument(page_content=doc.body, metadata=metadata, id=metadata["relative_path"])
