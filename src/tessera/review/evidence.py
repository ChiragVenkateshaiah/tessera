"""What a reviewer sees for a pending document (Phase 5, plan §3.8.3–3.8.4).

Evidence, not a verdict:
- the front matter and the text;
- the data-quality checks, with the pending document chunked and embedded
  on the fly (it is quarantined, so it isn't in any index);
- the top-N most similar indexed documents **with their scores**. There is
  no threshold and no "identifiable" label: deciding that an anonymized
  case study still points at a client is the human's job (Discovery §4),
  and a similarity score can't do it — the bank case study is a
  paraphrase of an engagement that shares none of its markers.

Reviewer-scoped: the similarity search runs through the same
``permission_filter`` as retrieval, with the reviewer's review clearance,
so restricted evidence appears only for engagements they are cleared for.
A reviewer not cleared for every engagement gets one notice, identical on
every document, and can only leave documents pending: the notice depends on
who is asking, never on what the document resembles, so it reveals nothing.

The output is plain JSON (it lives in checkpointed graph state).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import frontmatter

from tessera.embedding.base import Embedder
from tessera.ingestion.access_loader import Walls
from tessera.ingestion.chunker import chunk_document, chunk_embedding_text
from tessera.ingestion.data_quality import NEAR_DUPLICATE_THRESHOLD, STALE_AFTER_YEARS
from tessera.ingestion.loader import Document, front_matter_problems
from tessera.principal import Principal
from tessera.retrieval.retriever import permission_filter
from tessera.store.base import VectorStore

TOP_N = 5
# Per pending chunk, how many neighbours to pull before grouping by document.
NEIGHBOURS_PER_CHUNK = 20

SIMILARITY_NOTE = (
    "Similarity is evidence, not clearance. A high score shows overlapping "
    "wording or subject, not that the document identifies a client; a low "
    "score doesn't show that it can't. The decision is yours."
)
NEEDS_FULLY_CLEARED_REVIEWER = (
    "This document needs a fully cleared reviewer. Only a reviewer cleared "
    "for every engagement can approve, reclassify or reject it; you can leave "
    "it pending."
)


def _json(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [_json(v) for v in value]
    return value


def prepare_evidence(
    document: Document,
    relative_path: str,
    text: str,
    *,
    reviewer: str,
    walls: Walls,
    embedder: Embedder,
    store: VectorStore,
    as_of: date,
    corpus_dir: Path | None = None,
    top_n: int = TOP_N,
) -> dict[str, Any]:
    """The reviewer's view of one pending document, as JSON-ready data.
    Paths are shown relative to ``corpus_dir`` when given."""
    clearance = walls.review_clearance(reviewer)
    scope = permission_filter(Principal(reviewer, clearance))
    chunks = chunk_document(document)
    vectors = embedder.embed_documents([chunk_embedding_text(c) for c in chunks]) if chunks else []

    own = str(document.path)
    best: dict[str, dict[str, Any]] = {}
    near_duplicates: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors):
        for hit in store.query(vector, k=NEIGHBOURS_PER_CHUNK, where=scope):
            if hit.document_path == own:
                continue
            score = round(hit.score, 4)
            entry = best.get(hit.document_path)
            if entry is None or score > entry["score"]:
                best[hit.document_path] = {
                    "document": _shown(hit.document_path, corpus_dir),
                    "title": hit.document_title,
                    "sensitivity": hit.sensitivity,
                    "engagement": hit.engagement,
                    "score": score,
                    "pending_chunk": chunk.chunk_id,
                    "matched_chunk": hit.chunk_id,
                }
            if score >= NEAR_DUPLICATE_THRESHOLD:
                near_duplicates.append({"pending_chunk": chunk.chunk_id, "chunk": hit.chunk_id, "score": score})

    similar = sorted(best.values(), key=lambda e: (-e["score"], e["document"]))[:top_n]
    metadata = frontmatter.loads(text).metadata
    age = (as_of - document.date).days / 365.25
    may_rule = walls.may_approve(reviewer)
    return {
        "document": relative_path,
        "title": document.title,
        "front_matter": {k: _json(v) for k, v in metadata.items()},
        "text": document.body,
        "checks": {
            "front_matter_problems": front_matter_problems(metadata),
            "chunks": len(chunks),
            "age_years": round(age, 1),
            "stale": age > STALE_AFTER_YEARS,
            "near_duplicates": sorted(near_duplicates, key=lambda d: (-d["score"], d["pending_chunk"], d["chunk"])),
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        },
        "similar": similar,
        "similarity_note": SIMILARITY_NOTE,
        "may_rule": may_rule,
        "notice": None if may_rule else NEEDS_FULLY_CLEARED_REVIEWER,
    }


def format_evidence(evidence: dict[str, Any]) -> str:
    """The evidence as the CLI prints it."""
    fm = evidence["front_matter"]
    checks = evidence["checks"]
    lines = [
        f"=== {evidence['title']} ===",
        f"{evidence['document']}",
        "Front matter: " + ", ".join(f"{k}={v}" for k, v in fm.items()),
        "",
        evidence["text"].strip(),
        "",
        "Data-quality checks:",
        f"  front-matter problems: {', '.join(checks['front_matter_problems']) or 'none'}",
        f"  {checks['chunks']} chunks · {checks['age_years']} years old{' (stale)' if checks['stale'] else ''}",
        f"  near-duplicate chunks (>= {checks['near_duplicate_threshold']}): {len(checks['near_duplicates'])}",
    ]
    lines += [f"    {d['pending_chunk']} ~ {d['chunk']}  {d['score']:.3f}" for d in checks["near_duplicates"]]
    lines += ["", f"Most similar documents you are cleared to see (top {len(evidence['similar'])}):"]
    for e in evidence["similar"]:
        label = e["engagement"] or e["sensitivity"]
        lines.append(f"  {e['score']:.3f}  [{label}] {e['title']} — {e['document']}")
    lines += ["", evidence["similarity_note"]]
    if evidence["notice"]:
        lines += ["", f"NOTICE: {evidence['notice']}"]
    return "\n".join(lines)


def _shown(document_path: str, corpus_dir: Path | None) -> str:
    if corpus_dir is None:
        return document_path
    try:
        return Path(document_path).resolve().relative_to(corpus_dir.resolve()).as_posix()
    except ValueError:
        return document_path


__all__ = [
    "NEEDS_FULLY_CLEARED_REVIEWER",
    "SIMILARITY_NOTE",
    "TOP_N",
    "format_evidence",
    "prepare_evidence",
]

