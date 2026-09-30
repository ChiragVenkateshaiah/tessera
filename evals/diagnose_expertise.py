"""Retrieval-only (zero LLM) diagnostic for archetype B.

The P3 analogue of ``evals/tune_retrieval.py``: run ``find_experts`` over
one or more queries against the persisted people index and print the
ranked people with their evidence, for hand-checking rank quality and for
labelling ``relevant_people`` in P3-5.

    uv run python -m evals.diagnose_expertise "who knows pharma pricing"
    uv run python -m evals.diagnose_expertise --practice pricing -k 10 "retail pricing"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tessera.embedding.local import LocalEmbedder
from tessera.retrieval.expertise import ExpertiseResult, find_experts
from tessera.store.chroma_expertise import ChromaExpertiseStore


def format_result(result: ExpertiseResult) -> str:
    """Pure: render one result as text."""
    lines = [f"Q: {result.query}"]
    if not result.matches:
        lines.append("  (no candidates)")
    for rank, m in enumerate(result.matches, 1):
        p = m.person
        flag = "" if m.is_evidenced else "  [SELF-REPORTED ONLY]"
        lines.append(
            f"  {rank}. {p.name} ({p.person_id}) — {p.title}, {p.practice}, "
            f"{p.office}  sem={m.score:.2f} evidence={m.evidence_score:.2f} rank={m.rank_score:.2f}{flag}"
        )
        for e in m.evidence:
            lines.append(f"       - [{e.kind}] {e.description} (+{e.strength:.2f})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--persist-dir", type=Path, default=Path("data/vectorstore"))
    parser.add_argument("--practice")
    parser.add_argument("--office")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    where: dict[str, object] = {}
    if args.practice:
        where["practice"] = args.practice
    if args.office:
        where["office"] = args.office
    conditions = [{k: v} for k, v in where.items()]
    chroma_where = (
        None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}
    )

    store = ChromaExpertiseStore(persist_dir=args.persist_dir)
    if store.count() == 0:
        raise SystemExit("No people index — run `tessera index-people` first.")
    embedder = LocalEmbedder()
    for q in args.queries:
        print(format_result(find_experts(q, embedder, store, where=chroma_where, k=args.k)))
        print()


if __name__ == "__main__":
    main()
