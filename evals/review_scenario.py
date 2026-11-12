"""The review scenarios (Phase 5 plan §3.8.6, P5-8).

Six scenarios drive the review graph (``tessera.review.graph``) with
scripted replies, on a **temporary copy** of the corpus, and then measure
retrieval on **both stacks**: native, the LangChain stack's vector
retriever (lc-defaults), and its **hybrid** retriever (BM25 + vector, the
one that could go stale after a review). Retrieval-only: archetypes are
forced and the LLM is scripted, so the whole run makes zero network calls.

1. The grocer case study approved as internal: context-marker leaks on
   exactly ``ac-l01`` and ``ac-l13`` (it repeats Project Halcyon's facts).
2. Reclassified to Halcyon: no leaks, and ``c0048`` (Halcyon's partner,
   ``ac-a01``) retrieves the case study itself.
3. Rejected: not indexed, not offered again, absent from BM25.
4. The bank case study approved as internal: **no detected leak**, though
   it is a paraphrase of Project Lantern that shares none of Lantern's
   markers. The honest-limits case: the reason the decision is a human's.
5. A partially cleared reviewer gets the uniform notice and can't approve.
6. Interrupted and resumed **in a new process** at the same document;
   each decision is applied once; ``update_state()`` before ``apply``
   changes the outcome.

    python -m evals.review_scenario --out evals/reports/p5-8-review-scenarios.json

The committed corpus is never touched: the case studies stay pending
(plan §10.3).
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from evals.harness import EvalCase, case_principal, context_marker_hits, load_cases
from evals.snapshot import ScriptedLLM, forced_route

REPO = Path(__file__).resolve().parents[1]
GROCER = "case_studies/grocer-price-tier-reset.md"
BANK = "case_studies/regional-bank-branch-consolidation.md"
STACKS = ("native", "lc", "lc-hybrid")
AS_OF = date(2026, 11, 12)
GENERAL_COUNSEL = "general_counsel"
LEGAL_ASSOCIATE = "legal_associate"


def scenario_cases() -> list[EvalCase]:
    """Every A/C case that carries forbidden markers (the context-marker
    check's population), plus ``ac-a01`` for scenario 2."""
    cases = load_cases(REPO / "evals" / "cases")
    return sorted(
        (c for c in cases if c.forbidden_markers or c.id == "ac-a01"),
        key=lambda c: c.id,
    )


# ---- a world: one temporary copy with both stacks' indexes --------------


def build_template(root: Path) -> None:
    """Copy the corpus under ``root`` and index it for both stacks, the way
    ``tessera ingest`` and ``tessera ingest --stack lc`` (lc-defaults) do.
    The expensive step (embedding), done once and copied per scenario."""
    from tessera.embedding.local import LocalEmbedder
    from tessera.ingestion.indexing import index_corpus
    from tessera.ingestion.loader import load_corpus
    from tessera.integrations.langchain import hf_embeddings, lc_chroma, sql_record_manager
    from tessera.lc.indexing import index_with_record_manager
    from tessera.store.chroma import ChromaVectorStore

    root.mkdir(parents=True)
    shutil.copytree(REPO / "data" / "corpus", root / "corpus")
    # Indexed by a corpus-relative path, as `tessera ingest` indexes
    # data/corpus, so a copy of the template is a working world.
    with contextlib.chdir(root):
        documents = load_corpus(Path("corpus"))
        index_corpus(documents, LocalEmbedder(), ChromaVectorStore(persist_dir=root / "vs"))
        chroma = lc_chroma(root / "vs", hf_embeddings())
        index_with_record_manager(documents, chroma, sql_record_manager(root / "vs" / "lc_record_manager.sqlite"))


@dataclass
class World:
    """One scenario's temporary copy: corpus, both stacks' indexes, the
    review's audit log and checkpoints. Work inside it with ``inside()``:
    the corpus path is relative to the root, as the indexes store it."""

    root: Path
    _cache: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def copy_of(cls, template: Path, root: Path) -> World:
        shutil.copytree(template, root)
        return cls(root)

    @property
    def corpus(self) -> Path:
        return Path("corpus")

    def inside(self) -> contextlib.chdir:
        return contextlib.chdir(self.root)

    def _get(self, key: str, make: Any) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    @property
    def walls(self) -> Any:
        from tessera.ingestion.access_loader import load_walls

        return self._get("walls", lambda: load_walls(REPO / "data" / "access" / "walls.yaml"))

    @property
    def embedder(self) -> Any:
        from tessera.embedding.local import LocalEmbedder

        return self._get("embedder", LocalEmbedder)

    @property
    def native(self) -> Any:
        from tessera.store.chroma import ChromaVectorStore

        return self._get("native", lambda: ChromaVectorStore(persist_dir=self.root / "vs"))

    @property
    def lc_embeddings(self) -> Any:
        from tessera.integrations.langchain import hf_embeddings

        return self._get("lc_embeddings", hf_embeddings)

    @property
    def chroma(self) -> Any:
        from tessera.integrations.langchain import lc_chroma

        return self._get("chroma", lambda: lc_chroma(self.root / "vs", self.lc_embeddings))

    @property
    def lc(self) -> Any:
        from tessera.lc.store import LangChainChromaStore

        return self._get("lc", lambda: LangChainChromaStore(self.chroma))

    @property
    def bm25(self) -> Any:
        from tessera.lc.retrievers import BM25Cache

        return self._get("bm25", BM25Cache)

    def context(self) -> Any:
        from tessera.ingestion.indexing import index_corpus
        from tessera.integrations.langchain import sql_record_manager
        from tessera.lc.indexing import index_with_record_manager
        from tessera.review.audit import AuditLog
        from tessera.review.graph import ReviewContext

        records = sql_record_manager(self.root / "vs" / "lc_record_manager.sqlite")

        def native(docs: list[Any]) -> dict[str, int]:
            r = index_corpus(docs, self.embedder, self.native)
            return {"added": r.chunks_added, "deleted": r.chunks_deleted}

        def lc(docs: list[Any]) -> dict[str, int]:
            r = index_with_record_manager(docs, self.chroma, records, cleanup="incremental")
            return {"added": r.chunks_added, "deleted": r.chunks_deleted, "unchanged": r.chunks_unchanged}

        return ReviewContext(
            corpus_dir=self.corpus,
            walls=self.walls,
            embedder=self.embedder,
            evidence_store=self.native,
            indexers={"native": native, "lc": lc},
            audit=AuditLog(self.root / "review" / "audit.jsonl"),
            as_of=AS_OF,
            clock=lambda: "2026-11-12T00:00:00+00:00",
            after_index=[self.bm25.invalidate],
        )

    # -- what the stacks hold and retrieve --

    def indexed(self, document: str) -> dict[str, list[dict[str, Any]]]:
        """Each stack's chunks of ``document``: (sensitivity, engagement)."""
        path = str(self.corpus / document)
        native = self.native._collection.get(where={"document_path": path}, include=["metadatas"])["metadatas"]
        lc = self.lc.chroma.get(where={"document_path": path}, include=["metadatas"])["metadatas"]
        labels = lambda metas: sorted({(m["sensitivity"], m.get("engagement") or "") for m in metas})  # noqa: E731
        return {
            "native": {"chunks": len(native), "labels": labels(native)},
            "lc": {"chunks": len(lc), "labels": labels(lc)},
        }

    def bm25_documents(self, engagements: frozenset[str]) -> set[str]:
        """The documents in the BM25 index a principal cleared for
        ``engagements`` ranks over."""
        from tessera.lc.retrievers import ScopedStore
        from tessera.principal import Principal

        scoped = ScopedStore(self.lc, Principal("probe", engagements))
        self.bm25.retriever(scoped, 4)  # builds (or reuses) the cached index
        return {self._relative(d.metadata["document_path"]) for d in self.bm25.documents(scoped)}

    def _relative(self, path: str) -> str:
        return Path(path).relative_to(self.corpus).as_posix()

    def pipeline(self, stack: str, case: EvalCase) -> Any:
        from tessera.lc.embeddings import LangChainEmbedder
        from tessera.lc.retrievers import lc_retrieve_fn
        from tessera.pipeline import NativePipeline

        route = forced_route(case.archetype)
        if stack == "native":
            return NativePipeline(ScriptedLLM(), self.embedder, self.native, route_fn=route)
        kind = "vector" if stack == "lc" else "hybrid"
        return NativePipeline(
            ScriptedLLM(),
            LangChainEmbedder(self.lc_embeddings),
            self.lc,
            route_fn=route,
            retrieve_fn=lc_retrieve_fn(self.lc, kind, bm25=self.bm25),
        )

    def measure(self, cases: list[EvalCase]) -> dict[str, dict[str, Any]]:
        """Per stack: the cases with context-marker hits, and which cases
        retrieved each case study."""
        out: dict[str, dict[str, Any]] = {}
        for stack in STACKS:
            hits: dict[str, list[str]] = {}
            retrieved: dict[str, list[str]] = {GROCER: [], BANK: []}
            for case in cases:
                run = self.pipeline(stack, case).run(case.query, case_principal(case, self.walls))
                found = context_marker_hits(case, run, self.walls)
                if found:
                    hits[case.id] = found
                docs = {self._relative(r.document_path) for att in run.retrievals for r in att.results}
                for study in retrieved:
                    if study in docs:
                        retrieved[study].append(case.id)
            out[stack] = {"hit_cases": sorted(hits), "hits": hits, "retrieved": retrieved}
        return out

    # -- driving a review --

    def review(self, reviewer: str, replies: dict[str, dict[str, Any]], *, thread: str | None = None) -> dict[str, Any]:
        """Run one review thread to its end: answer each document's pause
        from ``replies`` (default: close the review), confirm every
        breakpoint. Returns what the reviewer was shown."""
        from tessera.review import graph as rg

        context = self.context()
        reviewer_id = self.walls.reviewer_names.get(reviewer, reviewer)
        shown: list[dict[str, Any]] = []
        with rg.open_checkpointer(self.root / "review" / "checkpoints.sqlite") as cp:
            graph = rg.compile_review_graph(cp)
            tid = thread or rg.new_thread_id(cp, reviewer_id, AS_OF)
            rg.start(graph, tid, reviewer_id, context)
            tries: dict[str, int] = {}
            while (paused := rg.pause(graph.get_state(rg.config_for(tid)))) is not None:
                if "breakpoint" in paused:
                    rg.confirm(graph, tid, context)
                    continue
                document = paused["evidence"]["document"]
                shown.append({"document": document, "allowed": paused["allowed"], "error": paused["error"],
                              "evidence": paused["evidence"]})
                planned = replies.get(document, {"action": rg.STOP})
                # A list of replies is tried in order (e.g. a refused approve, then leave).
                if isinstance(planned, list):
                    reply = planned[min(tries.get(document, 0), len(planned) - 1)]
                    tries[document] = tries.get(document, 0) + 1
                else:
                    reply = planned
                rg.answer(graph, tid, reply, context)
        return {"thread": tid, "shown": shown}

    def pending_offered(self, reviewer: str) -> list[str]:
        """What a fresh review would offer, without ruling on anything."""
        return [s["document"] for s in self.review(reviewer, {})["shown"]]

    def audit(self) -> list[dict[str, Any]]:
        from tessera.review.audit import AuditLog

        return AuditLog(self.root / "review" / "audit.jsonl").records()

    def front_matter(self, document: str) -> dict[str, Any]:
        import frontmatter

        meta = frontmatter.load(self.corpus / document).metadata
        return {k: meta.get(k) for k in ("sensitivity", "engagement", "review_status")}


# ---- the scenarios --------------------------------------------------------


def _similar(shown: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {k: e[k] for k in ("document", "sensitivity", "engagement", "score")}
        for e in shown["evidence"]["similar"]
    ]


def scenario_1(world: World, cases: list[EvalCase]) -> dict[str, Any]:
    run = world.review(GENERAL_COUNSEL, {GROCER: {"action": "approve"}})
    measured = world.measure(cases)
    expected = ["ac-l01", "ac-l13"]
    return {
        "title": "Grocer case study approved as internal",
        "expected": {"hit_cases": expected},
        "front_matter": world.front_matter(GROCER),
        "indexed": world.indexed(GROCER),
        "evidence_top": _similar(run["shown"][0]),
        "measured": measured,
        "pass": all(m["hit_cases"] == expected for m in measured.values()),
    }


def scenario_2(world: World, cases: list[EvalCase]) -> dict[str, Any]:
    world.review(GENERAL_COUNSEL, {GROCER: {"action": "reclassify", "engagement": "halcyon"}})
    measured = world.measure(cases)
    return {
        "title": "Grocer case study reclassified to Halcyon",
        "expected": {"hit_cases": [], "ac-a01 retrieves": GROCER},
        "front_matter": world.front_matter(GROCER),
        "indexed": world.indexed(GROCER),
        "measured": measured,
        "pass": all(m["hit_cases"] == [] and "ac-a01" in m["retrieved"][GROCER] for m in measured.values()),
    }


def scenario_3(world: World, cases: list[EvalCase]) -> dict[str, Any]:
    everything = frozenset(world.walls.cleared)
    bm25_before = GROCER in world.bm25_documents(everything)
    world.review(GENERAL_COUNSEL, {GROCER: {"action": "reject", "note": "identifiable to an insider"}})
    indexed = world.indexed(GROCER)
    offered = world.pending_offered(GENERAL_COUNSEL)
    in_bm25 = GROCER in world.bm25_documents(everything)
    measured = world.measure(cases)
    rejected = [r for r in world.audit() if r["document"] == GROCER]
    return {
        "title": "Grocer case study rejected",
        "expected": {"indexed": 0, "offered_again": False, "in_bm25": False},
        "front_matter": world.front_matter(GROCER),
        "indexed": indexed,
        "offered_next_time": offered,
        "in_bm25_before": bm25_before,
        "in_bm25_after": in_bm25,
        "audit": [{k: r[k] for k in ("action", "outcome", "note")} for r in rejected],
        "measured": measured,
        "pass": indexed["native"]["chunks"] == 0
        and indexed["lc"]["chunks"] == 0
        and GROCER not in offered
        and not in_bm25
        and world.front_matter(GROCER)["review_status"] == "pending"
        and all(m["hit_cases"] == [] and not m["retrieved"][GROCER] for m in measured.values()),
    }


def scenario_4(world: World, cases: list[EvalCase]) -> dict[str, Any]:
    run = world.review(GENERAL_COUNSEL, {GROCER: {"action": "leave"}, BANK: {"action": "approve"}})
    bank_shown = next(s for s in run["shown"] if s["document"] == BANK)
    measured = world.measure(cases)
    return {
        "title": "Bank case study approved as internal (honest limits)",
        "expected": {"hit_cases": []},
        "front_matter": world.front_matter(BANK),
        "indexed": world.indexed(BANK),
        "evidence_top": _similar(bank_shown),
        "measured": measured,
        "pass": all(m["hit_cases"] == [] for m in measured.values()) and world.indexed(BANK)["native"]["chunks"] > 0,
    }


def scenario_5(world: World, cases: list[EvalCase]) -> dict[str, Any]:
    from tessera.review.evidence import NEEDS_FULLY_CLEARED_REVIEWER

    associate = world.walls.reviewer_names[LEGAL_ASSOCIATE]
    clearance = world.walls.review_clearance(associate)
    run = world.review(
        LEGAL_ASSOCIATE,
        {
            GROCER: [{"action": "approve"}, {"action": "leave"}],
            BANK: [{"action": "reject"}, {"action": "leave"}],
        },
    )
    notices = sorted({s["evidence"]["notice"] for s in run["shown"]})
    refused = [s["error"] for s in run["shown"] if s["error"]]
    shown_engagements = sorted({e["engagement"] for s in run["shown"] for e in s["evidence"]["similar"] if e["engagement"]})
    outcomes = sorted({r["outcome"] for r in world.audit()})
    still_pending = [world.front_matter(d)["review_status"] for d in (GROCER, BANK)]
    return {
        "title": "A partially cleared reviewer",
        "expected": {"notice": NEEDS_FULLY_CLEARED_REVIEWER, "approve": "refused", "outcome": "left pending"},
        "clearance": sorted(clearance),
        "notices": notices,
        "allowed": sorted({a for s in run["shown"] for a in s["allowed"]}),
        "refused": refused,
        "evidence_engagements": shown_engagements,
        "audit_outcomes": outcomes,
        "review_status": still_pending,
        "indexed": {d: world.indexed(d) for d in (GROCER, BANK)},
        "pass": notices == [NEEDS_FULLY_CLEARED_REVIEWER]
        and len(refused) == 2
        and all(r == NEEDS_FULLY_CLEARED_REVIEWER for r in refused)
        and set(shown_engagements) <= clearance
        and outcomes == ["left pending"]
        and still_pending == ["pending", "pending"],
    }


def _step(world_root: Path, step: str) -> dict[str, Any]:
    """Scenario 6's separate processes: ``python -m evals.review_scenario
    --step <start|resume> --world <dir>``."""
    proc = subprocess.run(
        [sys.executable, "-m", "evals.review_scenario", "--step", step, "--world", str(world_root)],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"step {step} failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_step(world_root: Path, step: str) -> dict[str, Any]:
    from tessera.review import graph as rg

    world = World(world_root.resolve())
    with world.inside():
        return _run_step(world, step)


def _run_step(world: World, step: str) -> dict[str, Any]:
    from tessera.review import graph as rg

    context = world.context()
    gc = world.walls.reviewer_names[GENERAL_COUNSEL]
    with rg.open_checkpointer(world.root / "review" / "checkpoints.sqlite") as cp:
        graph = rg.compile_review_graph(cp)
        if step == "start":
            tid = rg.new_thread_id(cp, gc, AS_OF)
            rg.start(graph, tid, gc, context)
            paused = rg.pause(graph.get_state(rg.config_for(tid)))
            return {"thread": tid, "paused_at": paused["evidence"]["document"]}
        # resume: a new process finds the open thread and its document
        tid = rg.open_thread(graph, cp, gc)
        assert tid is not None, "no open review thread"
        cfg = rg.config_for(tid)
        resumed_at = rg.pause(graph.get_state(cfg))["evidence"]["document"]
        rg.answer(graph, tid, {"action": "approve"}, context)
        waiting = rg.pause(graph.get_state(cfg))["decision"]
        # Time travel: correct the waiting decision before apply runs.
        rg.correct(graph, tid, {"action": "reclassify", "engagement": "halcyon"}, world.corpus)
        corrected = rg.pause(graph.get_state(cfg))["decision"]
        rg.confirm(graph, tid, context)
        rg.answer(graph, tid, {"action": rg.STOP}, context)  # the bank study: close the review
        # Replay apply → index → audit from the checkpoint before apply:
        # every step must find its own earlier work and change nothing.
        before_apply = next(
            s for s in graph.get_state_history(cfg)
            if s.next == ("apply",) and s.values["decision"]["decision_id"] == corrected["decision_id"]
        )
        updates = list(graph.stream(None, before_apply.config, context=context, stream_mode="updates"))
        replayed = {node: out for u in updates for node, out in u.items() if node in ("apply", "index", "audit")}
        rg.answer(graph, tid, {"action": rg.STOP}, context)  # the replayed branch reaches the bank again
        history = [list(s.next) for s in graph.get_state_history(cfg)]
        return {
            "thread": tid,
            "resumed_at": resumed_at,
            "waiting_decision": waiting["action"],
            "corrected_decision": [corrected["action"], corrected["engagement"]],
            "replayed_apply": replayed["apply"]["applied"],
            "replayed_steps": sorted(replayed),
            "history_steps": len(history),
        }


def scenario_6(world: World, cases: list[EvalCase]) -> dict[str, Any]:
    first = _step(world.root, "start")
    second = _step(world.root, "resume")
    grocer_lines = [r for r in world.audit() if r["document"] == GROCER]
    measured = world.measure(cases)
    return {
        "title": "Interrupted and resumed in a new process; update_state() before apply",
        "expected": {
            "resumed_at": first["paused_at"],
            "applied": "reclassify to halcyon (corrected from approve), once",
        },
        "process_1": first,
        "process_2": second,
        "audit_lines_for_grocer": [{k: r[k] for k in ("action", "engagement", "outcome")} for r in grocer_lines],
        "front_matter": world.front_matter(GROCER),
        "indexed": world.indexed(GROCER),
        "measured": measured,
        "pass": second["resumed_at"] == first["paused_at"] == GROCER
        and second["waiting_decision"] == "approve"
        and second["corrected_decision"] == ["reclassify", "halcyon"]
        and second["replayed_apply"] == {"outcome": "already applied"}
        and len(grocer_lines) == 1
        and grocer_lines[0]["action"] == "reclassify"
        and world.front_matter(GROCER) == {"sensitivity": "restricted", "engagement": "halcyon", "review_status": "reviewed"}
        and all(m["hit_cases"] == [] and "ac-a01" in m["retrieved"][GROCER] for m in measured.values()),
    }


SCENARIOS = {1: scenario_1, 2: scenario_2, 3: scenario_3, 4: scenario_4, 5: scenario_5, 6: scenario_6}


def run_scenarios(work: Path, which: list[int] | None = None) -> dict[str, Any]:
    template = work / "template"
    if not template.exists():
        build_template(template)
    cases = scenario_cases()
    base = World.copy_of(template, work / "baseline")
    with base.inside():
        baseline = base.measure(cases)
    results: dict[str, Any] = {"baseline": baseline, "scenarios": {}}
    for n in which or sorted(SCENARIOS):
        world = World.copy_of(template, work / f"s{n}")
        with world.inside():
            results["scenarios"][str(n)] = SCENARIOS[n](world, cases)
    results["pass"] = all(s["pass"] for s in results["scenarios"].values())
    return results


def main() -> None:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    parser.add_argument("--only", help="comma-separated scenario numbers")
    parser.add_argument("--step", choices=["start", "resume"], help=argparse.SUPPRESS)
    parser.add_argument("--world", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mermaid", type=Path, help="write the review graph's draw_mermaid() here")
    args = parser.parse_args()

    if args.step:
        print(json.dumps(run_step(args.world, args.step)))
        return
    if args.mermaid:
        from tessera.review.graph import build_review_graph

        args.mermaid.write_text(build_review_graph().compile().get_graph().draw_mermaid(), encoding="utf-8")
    which = [int(n) for n in args.only.split(",")] if args.only else None
    with tempfile.TemporaryDirectory() as work:
        results = run_scenarios(Path(work), which)
    for n, s in results["scenarios"].items():
        print(f"{n}. {s['title']}: {'PASS' if s['pass'] else 'FAIL'}")
        if "measured" in s:
            for stack, m in s["measured"].items():
                print(f"     {stack:<9} marker-hit cases {m['hit_cases'] or 'none'}"
                      f" · grocer retrieved by {m['retrieved'][GROCER] or 'none'}"
                      f" · bank retrieved by {m['retrieved'][BANK] or 'none'}")
    print(f"=> {'PASS' if results['pass'] else 'FAIL'} ({len(results['scenarios'])} scenarios)")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    sys.exit(0 if results["pass"] else 1)


if __name__ == "__main__":
    main()
