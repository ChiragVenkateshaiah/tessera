"""The golden retrieval snapshot (Phase 5 plan §3.1.1, P5-0).

Every eval case is run with its archetype **forced** (no routing call) and
a scripted fake LLM (no real generation call), and everything that is
deterministic about the case is recorded: what was retrieved, with what
score, what cleared the relevance floor, what restricted or superseded
material was seen, the expertise shortlist, and how many generation calls
the path made. Zero network calls.

The snapshot pins retrieval behaviour while the query path is refactored
(P5-2) and a second stack is built beside it: a refactor is neutral only
if its snapshot equals the committed one field for field.

`snapshot_case` and `build_snapshot` are pure with respect to
infrastructure (CLAUDE.md constraint #6): the embedder, stores and walls
are injected, and they return data. Only `main()` builds a real index,
reads the environment and writes or compares a file — the same exemption
`harness.main()` has.

    python -m evals.snapshot --out evals/snapshots/v0.4.0.json
    python -m evals.snapshot --check evals/snapshots/v0.4.0.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.harness import DEFAULT_K, EvalCase, case_principal, unique_documents_by_rank
from tessera.embedding.base import Embedder
from tessera.generation.answer import filter_relevant, generate_answer
from tessera.generation.base import LLMClient
from tessera.generation.expertise import generate_expertise_answer
from tessera.ingestion.access_loader import Walls
from tessera.ingestion.loader import SENSITIVITY_INTERNAL, STATUS_SUPERSEDED
from tessera.retrieval.expertise import find_experts
from tessera.retrieval.retriever import retrieve
from tessera.retrieval.router import Archetype
from tessera.store.base import ExpertiseStore, VectorStore

SNAPSHOT_VERSION = 1
SCORE_DECIMALS = 4

# What the scripted LLM answers. It cites [1], so an A/C answer looks like
# a grounded one to anything downstream; its content is never recorded.
SCRIPTED_ANSWER = "Scripted snapshot answer [1]."


class ScriptedLLM(LLMClient):
    """Answers every call with SCRIPTED_ANSWER and counts the calls."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls += 1
        return SCRIPTED_ANSWER


def _relative(path: str, corpus_dir: Path) -> str:
    return Path(path).relative_to(corpus_dir).as_posix()


def _score(value: float) -> float:
    return round(value, SCORE_DECIMALS)


def snapshot_case(
    case: EvalCase,
    embedder: Embedder,
    store: VectorStore,
    expertise_store: ExpertiseStore,
    corpus_dir: Path,
    walls: Walls | None,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """One case's deterministic record, archetype forced to the labelled
    one. D is terminal, so it records only that no call was made.
    """
    llm = ScriptedLLM()
    record: dict[str, Any] = {"archetype": case.archetype.value}

    if case.archetype is Archetype.EXPERTISE:
        result = find_experts(case.query, embedder, expertise_store, k=k)
        generated = generate_expertise_answer(result, llm)
        record["shortlist"] = [
            {
                "person_id": m.person.person_id,
                "score": _score(m.score),
                "evidence_score": _score(m.evidence_score),
                "rank_score": _score(m.rank_score),
                "evidence": [[e.kind, e.self_reported] for e in m.evidence],
            }
            for m in result.matches
        ]
        record["presented"] = [m.person.person_id for m in generated.experts]
    elif case.archetype in (Archetype.LOOKUP, Archetype.SYNTHESIS):
        principal = case_principal(case, walls)
        retrieval = retrieve(case.query, case.archetype, embedder, store, principal=principal)
        generated = generate_answer(retrieval, llm)
        shown = filter_relevant(retrieval.results)
        record.update(
            documents=unique_documents_by_rank(retrieval.results, corpus_dir),
            chunks=[
                {"chunk_id": r.chunk_id, "score": _score(r.score)} for r in retrieval.results
            ],
            over_floor=[r.chunk_id for r in shown],
            restricted_seen=sorted(
                {
                    r.engagement or ""
                    for r in retrieval.results
                    if r.sensitivity != SENSITIVITY_INTERNAL
                }
            ),
            superseded_shown=sorted(
                {_relative(r.document_path, corpus_dir) for r in shown if r.status == STATUS_SUPERSEDED}
            ),
            superseded_excluded=[
                {
                    "document": _relative(m.document_path, corpus_dir),
                    "superseded_by": _relative(m.superseded_by, corpus_dir),
                    "score": _score(m.score),
                }
                for m in retrieval.superseded
            ],
            superseded_noted=sorted(_relative(m.document_path, corpus_dir) for m in generated.superseded),
            removed=dict(sorted(retrieval.removed.items())),
        )

    record["generation_calls"] = llm.calls
    return record


def build_snapshot(
    cases: list[EvalCase],
    embedder: Embedder,
    store: VectorStore,
    expertise_store: ExpertiseStore,
    corpus_dir: Path,
    walls: Walls | None,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Every case's record, keyed by case id."""
    return {
        "version": SNAPSHOT_VERSION,
        "k": k,
        "score_decimals": SCORE_DECIMALS,
        "cases": {
            case.id: snapshot_case(case, embedder, store, expertise_store, corpus_dir, walls, k)
            for case in sorted(cases, key=lambda c: c.id)
        },
    }


def diff_snapshots(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Human-readable differences, one line per changed field; empty when
    the two are equal.
    """
    lines: list[str] = []
    for key in ("version", "k", "score_decimals"):
        if expected.get(key) != actual.get(key):
            lines.append(f"{key}: {expected.get(key)!r} -> {actual.get(key)!r}")
    want, got = expected.get("cases", {}), actual.get("cases", {})
    for case_id in sorted(set(want) | set(got)):
        if case_id not in got:
            lines.append(f"{case_id}: missing")
        elif case_id not in want:
            lines.append(f"{case_id}: new case")
        else:
            for field_name in sorted(set(want[case_id]) | set(got[case_id])):
                before = want[case_id].get(field_name)
                after = got[case_id].get(field_name)
                if before != after:
                    lines.append(f"{case_id}.{field_name}: {before!r} -> {after!r}")
    return lines


def main() -> None:
    """Build a fresh, temporary index over the corpus (not the persisted
    one, so a stale local index can't pass for a code change), snapshot
    every case, then write it (--out) or compare it with a committed one
    (--check, exit 1 on any difference).
    """
    import argparse
    import json
    import os
    import sys
    import tempfile

    from evals.harness import load_cases
    from tessera.embedding.local import LocalEmbedder
    from tessera.ingestion.access_loader import load_walls
    from tessera.ingestion.chunker import chunk_corpus, chunk_embedding_text
    from tessera.ingestion.expertise_loader import load_expertise, profile_summary_text
    from tessera.ingestion.loader import indexable, load_corpus
    from tessera.store.chroma import ChromaVectorStore
    from tessera.store.chroma_expertise import ChromaExpertiseStore

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="write the snapshot here")
    mode.add_argument("--check", type=Path, help="compare with this snapshot")
    args = parser.parse_args()

    corpus_dir = Path(os.environ.get("TESSERA_CORPUS_DIR", "data/corpus"))
    expertise_dir = Path(os.environ.get("TESSERA_EXPERTISE_DIR", "data/expertise/people"))
    access_file = Path(os.environ.get("TESSERA_ACCESS_FILE", "data/access/walls.yaml"))

    loaded = load_corpus(corpus_dir)
    docs = indexable(loaded)
    chunks = chunk_corpus(docs)
    embedder = LocalEmbedder()
    people = load_expertise(expertise_dir, corpus_dir=corpus_dir)
    walls = load_walls(
        access_file,
        person_ids=[p.person_id for p in people],
        engagements={d.engagement for d in loaded if d.engagement},
    )

    with tempfile.TemporaryDirectory() as persist_dir:
        store = ChromaVectorStore(persist_dir=Path(persist_dir))
        store.add(chunks, embedder.embed_documents([chunk_embedding_text(c) for c in chunks]))
        expertise_store = ChromaExpertiseStore(persist_dir=Path(persist_dir))
        expertise_store.add(
            people, embedder.embed_documents([profile_summary_text(p) for p in people])
        )
        cases = load_cases(Path(__file__).parent / "cases")
        snapshot = build_snapshot(cases, embedder, store, expertise_store, corpus_dir, walls)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
        print(f"Wrote {len(snapshot['cases'])} cases to {args.out}")
        return

    differences = diff_snapshots(json.loads(args.check.read_text()), snapshot)
    if differences:
        print(f"{len(differences)} difference(s) from {args.check}:")
        print("\n".join(f"  {line}" for line in differences))
        sys.exit(1)
    print(f"Identical to {args.check} ({len(snapshot['cases'])} cases).")


if __name__ == "__main__":
    main()
