"""Retrieval-only comparison of the LangChain retrievers (Phase 5 plan
§3.4, P5-5 evidence).

Every retriever runs over the same index (the LangChain collection, built
through native delete-then-add, so only the retriever changes) for every
A/C case with its archetype **forced** — no routing, no generation, zero
LLM calls. ``MultiQueryRetriever`` calls the LLM, so it is measured live
in a sweep instead (``tessera eval --stack lc`` with
``TESSERA_LC_RETRIEVER=multiquery``).

Scored as the eval harness scores: unique documents by rank, recall@5,
precision@5, reciprocal rank; plus the context size (characters of chunk
text the answer model would be shown, before the relevance floor) and
the mean retrieval latency.

**The hybrid's BM25 weight is tuned on ``query_log.yaml`` and reported
on held-out ``placeholder.yaml``.**

Run from the repo root (the lc extra installed):
    python -m evals.compare_retrievers --out evals/reports/p5-5-retrieval.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evals.harness import EvalCase, load_cases, unique_documents_by_rank
from evals.metrics import mean, precision_at_k, recall_at_k, reciprocal_rank
from tessera.retrieval.router import Archetype

METRIC_K = 5
HYBRID_WEIGHTS = (0.25, 0.5, 0.75)  # BM25's share; the vector gets the rest


@dataclass(frozen=True)
class SetScore:
    cases: int
    mean_recall: float
    mean_precision: float
    mean_mrr: float
    total_misses: int
    mean_context_chars: float
    mean_latency_ms: float


@dataclass(frozen=True)
class Row:
    name: str
    kind: str
    bm25_weight: float | None
    tuned: SetScore
    held_out: SetScore


def score_set(cases: list[EvalCase], make: Any, corpus_dir: Path) -> SetScore:
    recalls, precisions, rrs, chars, latencies = [], [], [], [], []
    for case in cases:
        start = time.perf_counter()
        result = make(case.archetype).retrieve(case.query)
        latencies.append((time.perf_counter() - start) * 1000)
        docs = unique_documents_by_rank(result.results, corpus_dir)
        relevant = set(case.relevant_sources)
        recalls.append(recall_at_k(docs, relevant, METRIC_K))
        precisions.append(precision_at_k(docs, relevant, METRIC_K))
        rrs.append(reciprocal_rank(docs, relevant))
        chars.append(sum(len(r.text) for r in result.results))
    return SetScore(
        cases=len(cases),
        mean_recall=round(mean(recalls), 4),
        mean_precision=round(mean(precisions), 4),
        mean_mrr=round(mean(rrs), 4),
        total_misses=sum(1 for r in recalls if r == 0.0),
        mean_context_chars=round(mean(chars)),
        mean_latency_ms=round(mean(latencies), 1),
    )


def format_report(rows: list[Row], chosen_weight: float) -> str:
    lines = [
        "| Retriever | Tuned recall | Tuned precision | Tuned MRR | Tuned misses | Held-out recall "
        "| Held-out precision | Held-out MRR | Held-out misses | Context chars (tuned) | Latency ms (tuned) |",
        "|" + "---|" * 11,
    ]
    for r in rows:
        t, h = r.tuned, r.held_out
        mark = " **(chosen weight)**" if r.kind == "hybrid" and r.bm25_weight == chosen_weight else ""
        lines.append(
            f"| {r.name}{mark} | {t.mean_recall:.3f} | {t.mean_precision:.3f} | {t.mean_mrr:.3f} "
            f"| {t.total_misses} | {h.mean_recall:.3f} | {h.mean_precision:.3f} | {h.mean_mrr:.3f} "
            f"| {h.total_misses} | {t.mean_context_chars:,.0f} | {t.mean_latency_ms:,.0f} |"
        )
    t, h = rows[0].tuned, rows[0].held_out
    lines += [
        "",
        f"Tuning set: {t.cases} A/C cases from query_log.yaml; held-out: {h.cases} from "
        "placeholder.yaml. Forced archetypes, internal-only principal, zero LLM calls. "
        "Latency is per retrieval on this machine, excluding model loading.",
    ]
    return "\n".join(lines)


def main() -> None:
    import os

    from tessera.embedding.local import LocalEmbedder
    from tessera.ingestion.indexing import index_corpus
    from tessera.ingestion.loader import load_corpus
    from tessera.integrations.langchain import hf_cross_encoder, lc_chroma
    from tessera.lc.embeddings import as_langchain
    from tessera.lc.retrievers import BM25Cache, ScopedStore, build_retriever, parent_docstore
    from tessera.lc.store import LangChainChromaStore

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

    embedder = LocalEmbedder()
    bm25 = BM25Cache()
    shared = {
        "bm25": bm25,
        "cross_encoder": hf_cross_encoder(),
        "parents": parent_docstore(documents),
    }
    with tempfile.TemporaryDirectory() as persist:
        store = LangChainChromaStore(lc_chroma(Path(persist), as_langchain(embedder)))
        index_corpus(documents, embedder, store)
        scoped = ScopedStore(store, None)
        bm25.documents(scoped)  # build the index once, outside the timings

        def maker(kind: str, weight: float | None = None) -> Any:
            def make(archetype: Archetype) -> Any:
                extra = {"bm25_weight": weight} if weight is not None else {}
                return build_retriever(kind, archetype, scoped, embedder, **shared, **extra)

            return make

        plan: list[tuple[str, str, float | None]] = [
            ("vector (= native)", "vector", None),
            ("parent_doc", "parent_doc", None),
            ("bm25", "bm25", None),
            *[(f"hybrid bm25={w}", "hybrid", w) for w in HYBRID_WEIGHTS],
            ("rerank (cross-encoder)", "rerank", None),
        ]
        rows = []
        for name, kind, weight in plan:
            print(f"  {name} ...", file=sys.stderr, flush=True)
            make = maker(kind, weight)
            rows.append(
                Row(name, kind, weight, score_set(tuned, make, corpus_dir), score_set(held_out, make, corpus_dir))
            )

    hybrids = [r for r in rows if r.kind == "hybrid"]
    chosen = max(hybrids, key=lambda r: (r.tuned.mean_recall, r.tuned.mean_mrr, r.tuned.mean_precision))
    print(format_report(rows, chosen.bm25_weight or 0.0))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rows": [asdict(r) for r in rows], "chosen_hybrid_bm25_weight": chosen.bm25_weight}
        args.out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
