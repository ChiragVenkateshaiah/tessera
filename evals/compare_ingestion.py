"""Retrieval-only comparison of splitter and embedding-prefix variants
(Phase 5 plan §3.3, P5-4 evidence).

Each variant builds a fresh index of the real corpus — native delete-then-add,
the native store, the native embedder — so the splitter and the
``embed_prefix`` switch are the only things that change, and runs the
real ``retrieve()`` over every A/C case with its archetype **forced** (no
routing call, no generation, zero LLM calls). Scoring is the eval
harness's: unique documents by rank, recall@5 and reciprocal rank, as
``evals/tune_retrieval.py`` does.

**Tuned on ``query_log.yaml`` (the ``ql`` cases), reported on held-out
``placeholder.yaml``** (the ``q`` cases): the LangChain chunk size is
chosen on the tuning set only, and the held-out numbers are reported for
every variant beside it.

Chunk statistics come with each variant: count, size distribution
(words), and the structure the character splitter can break — chunks with
an unbalanced code fence, and table rows separated from their header row.

Run from the repo root (the lc extra installed):
    python -m evals.compare_ingestion --out evals/reports/p5-4-ingestion.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from evals.harness import EvalCase, load_cases, unique_documents_by_rank
from evals.metrics import mean, recall_at_k, reciprocal_rank
from tessera.embedding.base import Embedder
from tessera.ingestion.chunker import Chunk, chunk_document, chunk_embedding_text
from tessera.ingestion.indexing import Splitter, index_corpus
from tessera.ingestion.loader import Document, indexable, load_corpus
from tessera.retrieval.retriever import retrieve
from tessera.retrieval.router import Archetype
from tessera.store.chroma import ChromaVectorStore

METRIC_K = 5  # the eval report's recall@k
LC_CHUNK_SIZES = (400, 800, 1200, 1600)
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")


@dataclass(frozen=True)
class Variant:
    name: str
    splitter: str  # "native" | "lc"
    chunk_size: int | None  # characters, lc only
    strip_headers: bool | None  # lc only
    embed_prefix: bool


@dataclass(frozen=True)
class ChunkStats:
    chunks: int
    words_median: float
    words_p90: float
    words_max: int
    unbalanced_fences: int
    headerless_table_pieces: int


@dataclass(frozen=True)
class SetScore:
    cases: int
    mean_recall: float
    mean_mrr: float
    total_misses: int  # cases at recall 0.00


@dataclass(frozen=True)
class Row:
    variant: Variant
    stats: ChunkStats
    tuned: SetScore
    held_out: SetScore


def variants() -> list[Variant]:
    out = []
    for prefix in (True, False):
        out.append(Variant(f"native{'' if prefix else ' (no prefix)'}", "native", None, None, prefix))
        for size in LC_CHUNK_SIZES:
            out.append(
                Variant(f"lc {size}{'' if prefix else ' (no prefix)'}", "lc", size, True, prefix)
            )
        out.append(
            Variant(f"lc 800 keep headers{'' if prefix else ' (no prefix)'}", "lc", 800, False, prefix)
        )
    return out


def splitter_for(variant: Variant) -> Splitter:
    if variant.splitter == "native":
        return chunk_document
    from tessera.lc.splitter import split_document

    return partial(split_document, chunk_size=variant.chunk_size, strip_headers=variant.strip_headers)


def chunk_stats(chunks: list[Chunk]) -> ChunkStats:
    words = sorted(len(c.text.split()) for c in chunks)
    unbalanced = sum(
        1 for c in chunks if sum(1 for line in c.text.splitlines() if line.startswith("```")) % 2
    )
    headerless = 0
    for c in chunks:
        lines = [line for line in c.text.splitlines() if line.lstrip().startswith("|")]
        if lines and not any(_TABLE_SEPARATOR.match(line.strip()) for line in lines):
            headerless += 1
    return ChunkStats(
        chunks=len(chunks),
        words_median=statistics.median(words),
        words_p90=words[int(0.9 * (len(words) - 1))],
        words_max=words[-1],
        unbalanced_fences=unbalanced,
        headerless_table_pieces=headerless,
    )


class CachingEmbedder(Embedder):
    """Embeds each distinct text once across every variant."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        for text, vector in zip(missing, self._inner.embed_documents(missing)):
            self._cache[text] = vector
        return [self._cache[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def score_set(
    cases: list[EvalCase], embedder: Embedder, store: ChromaVectorStore, corpus_dir: Path
) -> SetScore:
    recalls, rrs = [], []
    for case in cases:
        result = retrieve(case.query, case.archetype, embedder, store)
        docs = unique_documents_by_rank(result.results, corpus_dir)
        relevant = set(case.relevant_sources)
        recalls.append(recall_at_k(docs, relevant, METRIC_K))
        rrs.append(reciprocal_rank(docs, relevant))
    return SetScore(
        cases=len(cases),
        mean_recall=round(mean(recalls), 4),
        mean_mrr=round(mean(rrs), 4),
        total_misses=sum(1 for r in recalls if r == 0.0),
    )


def run_variant(
    variant: Variant,
    documents: list[Document],
    embedder: Embedder,
    tuned: list[EvalCase],
    held_out: list[EvalCase],
    corpus_dir: Path,
) -> Row:
    split = splitter_for(variant)
    embedding_text: Callable[[Chunk], str] = (
        chunk_embedding_text if variant.embed_prefix else (lambda c: c.text)
    )
    chunks = [c for d in indexable(documents) for c in split(d)]
    with tempfile.TemporaryDirectory() as persist:
        store = ChromaVectorStore(persist_dir=Path(persist))
        index_corpus(documents, embedder, store, split=split, embedding_text=embedding_text)
        return Row(
            variant=variant,
            stats=chunk_stats(chunks),
            tuned=score_set(tuned, embedder, store, corpus_dir),
            held_out=score_set(held_out, embedder, store, corpus_dir),
        )


def choose_lc(rows: list[Row]) -> Row:
    """The LangChain variant with the best tuning-set recall, then MRR —
    chosen on query_log.yaml only."""
    lc = [r for r in rows if r.variant.splitter == "lc" and r.variant.embed_prefix]
    return max(lc, key=lambda r: (r.tuned.mean_recall, r.tuned.mean_mrr, -r.stats.chunks))


def format_report(rows: list[Row], chosen: Row) -> str:
    head = (
        "| Variant | Chunks | Words med / p90 / max | Unbalanced fences | Headerless table pieces "
        "| Tuned recall | Tuned MRR | Tuned misses | Held-out recall | Held-out MRR | Held-out misses |"
    )
    lines = [head, "|" + "---|" * 11]
    for r in rows:
        s, t, h = r.stats, r.tuned, r.held_out
        mark = " **(chosen lc)**" if r is chosen else ""
        lines.append(
            f"| {r.variant.name}{mark} | {s.chunks} | {s.words_median:g} / {s.words_p90} / {s.words_max} "
            f"| {s.unbalanced_fences} | {s.headerless_table_pieces} "
            f"| {t.mean_recall:.3f} | {t.mean_mrr:.3f} | {t.total_misses} "
            f"| {h.mean_recall:.3f} | {h.mean_mrr:.3f} | {h.total_misses} |"
        )
    t, h = rows[0].tuned, rows[0].held_out
    lines.append("")
    lines.append(
        f"Tuning set: {t.cases} A/C cases from query_log.yaml; held-out: {h.cases} "
        "from placeholder.yaml. Forced archetypes, real retrieve(), zero LLM calls."
    )
    return "\n".join(lines)


def main() -> None:
    import os

    from tessera.embedding.local import LocalEmbedder

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="also write the rows as JSON here")
    args = parser.parse_args()

    corpus_dir = Path(os.environ.get("TESSERA_CORPUS_DIR", "data/corpus"))
    documents = load_corpus(corpus_dir)
    cases = [
        c
        for c in load_cases(Path(__file__).parent / "cases")
        if c.archetype in (Archetype.LOOKUP, Archetype.SYNTHESIS)
        and c.relevant_sources
        and c.access is None
    ]
    tuned = [c for c in cases if c.id.startswith("ql")]
    held_out = [c for c in cases if re.fullmatch(r"q\d+", c.id)]
    embedder = CachingEmbedder(LocalEmbedder())

    rows = []
    for variant in variants():
        print(f"  {variant.name} ...", file=sys.stderr, flush=True)
        rows.append(run_variant(variant, documents, embedder, tuned, held_out, corpus_dir))
    chosen = choose_lc(rows)
    print(format_report(rows, chosen))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rows": [asdict(r) for r in rows], "chosen_lc": chosen.variant.name}
        args.out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
