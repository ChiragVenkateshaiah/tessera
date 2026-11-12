"""The review graph's mechanics (Phase 5, plan §3.8.2): interrupt and
resume, the breakpoint before ``apply``, idempotence, stale versions,
threads, the strict checkpoint serializer and the audit log.

Fast: a tiny hashing embedder over a real Chroma store on a temporary
copy of the corpus. The six scenarios with the real embedders are in
``tests/test_review_scenarios.py``.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph", reason="the lc extra isn't installed")

from tessera.embedding.base import Embedder  # noqa: E402
from tessera.ingestion.access_loader import load_walls  # noqa: E402
from tessera.ingestion.indexing import index_corpus  # noqa: E402
from tessera.ingestion.loader import load_corpus  # noqa: E402
from tessera.review import graph as rg  # noqa: E402
from tessera.review.audit import AuditLog  # noqa: E402
from tessera.review.decisions import ReviewError  # noqa: E402
from tessera.review.evidence import NEEDS_FULLY_CLEARED_REVIEWER, SIMILARITY_NOTE  # noqa: E402
from tessera.store.chroma import ChromaVectorStore  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GROCER = "case_studies/grocer-price-tier-reset.md"
BANK = "case_studies/regional-bank-branch-consolidation.md"
TODAY = date(2026, 11, 12)


class HashEmbedder(Embedder):
    """Bag of hashed words: deterministic, instant, good enough to rank."""

    @property
    def dimension(self) -> int:
        return 64

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * 64
        for word in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class Rig:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.corpus = root / "corpus"
        shutil.copytree(REPO / "data" / "corpus", self.corpus)
        self.walls = load_walls(REPO / "data" / "access" / "walls.yaml")
        self.embedder = HashEmbedder()
        self.store = ChromaVectorStore(persist_dir=root / "vs")
        index_corpus(load_corpus(self.corpus), self.embedder, self.store)
        self.indexed: list[str] = []
        self.invalidated = 0
        self.audit = AuditLog(root / "review" / "audit.jsonl")

    def context(self) -> rg.ReviewContext:
        def native(docs: list[Any]) -> dict[str, int]:
            self.indexed.extend(d.path.relative_to(self.corpus).as_posix() for d in docs)
            r = index_corpus(docs, self.embedder, self.store)
            return {"added": r.chunks_added, "deleted": r.chunks_deleted}

        def bump() -> None:
            self.invalidated += 1

        return rg.ReviewContext(
            corpus_dir=self.corpus, walls=self.walls, embedder=self.embedder, evidence_store=self.store,
            indexers={"native": native}, audit=self.audit, as_of=TODAY, clock=lambda: "T", after_index=[bump],
        )

    def checkpointer(self) -> Any:
        return rg.open_checkpointer(self.root / "review" / "checkpoints.sqlite")

    def text(self, document: str) -> str:
        return (self.corpus / document).read_text(encoding="utf-8")


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


GC, ASSOCIATE = "legal-01", "legal-02"


def paused(graph: Any, tid: str) -> Any:
    return rg.pause(graph.get_state(rg.config_for(tid)))


def test_it_pauses_at_each_pending_document_with_evidence(rig: Rig) -> None:
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        rg.start(graph, "t", GC, rig.context())
        first = paused(graph, "t")

    evidence = first["evidence"]
    assert evidence["document"] == GROCER  # pending documents in path order
    assert evidence["front_matter"]["review_status"] == "pending"
    assert evidence["similar"] and all("score" in e for e in evidence["similar"])
    assert evidence["similarity_note"] == SIMILARITY_NOTE
    assert evidence["may_rule"] and evidence["notice"] is None
    assert first["allowed"] == ["approve", "reclassify", "reject", "leave", "stop"]
    # No threshold, no verdict: nothing in the evidence labels the document.
    assert not {"identifiable", "verdict", "threshold"} & set(evidence)


def test_approve_waits_at_the_breakpoint_then_applies_indexes_and_audits_once(rig: Rig) -> None:
    ctx = rig.context()
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        rg.start(graph, "t", GC, ctx)
        rg.answer(graph, "t", {"action": "approve"}, ctx)
        waiting = paused(graph, "t")
        assert waiting["breakpoint"] == "apply"
        assert "review_status: pending" in rig.text(GROCER)  # nothing written yet
        rg.confirm(graph, "t", ctx)
        assert paused(graph, "t")["evidence"]["document"] == BANK

    assert "review_status: reviewed" in rig.text(GROCER)
    assert rig.indexed == [GROCER] and rig.invalidated == 1
    (record,) = rig.audit.records()
    assert (record["action"], record["outcome"], record["thread_id"]) == ("approve", "applied", "t")


def test_reject_and_leave_skip_apply_and_write_nothing(rig: Rig) -> None:
    before = (rig.text(GROCER), rig.text(BANK))
    ctx = rig.context()
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        rg.start(graph, "t", GC, ctx)
        rg.answer(graph, "t", {"action": "reject"}, ctx)
        rg.answer(graph, "t", {"action": "leave"}, ctx)
        assert paused(graph, "t") is None  # finished: nothing left

    assert (rig.text(GROCER), rig.text(BANK)) == before
    assert rig.indexed == []
    assert [r["outcome"] for r in rig.audit.records()] == ["rejected", "left pending"]


def test_rejected_versions_are_not_offered_again_but_left_ones_are(rig: Rig) -> None:
    ctx = rig.context()
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        rg.start(graph, "t1", GC, ctx)
        rg.answer(graph, "t1", {"action": "reject"}, ctx)
        rg.answer(graph, "t1", {"action": "leave"}, ctx)
        rg.start(graph, "t2", GC, ctx)
        assert paused(graph, "t2")["evidence"]["document"] == BANK

    # An edited version is a new version: it comes back.
    (rig.corpus / GROCER).write_text(rig.text(GROCER) + "\nA new paragraph.\n", encoding="utf-8")
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        rg.start(graph, "t3", GC, ctx)
        assert paused(graph, "t3")["evidence"]["document"] == GROCER


def test_a_document_edited_before_apply_is_stale_and_not_written(rig: Rig) -> None:
    ctx = rig.context()
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        rg.start(graph, "t", GC, ctx)
        rg.answer(graph, "t", {"action": "approve"}, ctx)
        edited = rig.text(GROCER).replace("2,800 stores", "2,900 stores")
        (rig.corpus / GROCER).write_text(edited, encoding="utf-8")
        rg.confirm(graph, "t", ctx)

    assert rig.text(GROCER) == edited  # untouched
    assert rig.indexed == []
    assert rig.audit.records()[0]["outcome"] == "stale"


def test_a_partially_cleared_reviewer_gets_the_notice_and_cannot_rule(rig: Rig) -> None:
    ctx = rig.context()
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        rg.start(graph, "t", ASSOCIATE, ctx)
        first = paused(graph, "t")
        rg.answer(graph, "t", {"action": "approve"}, ctx)
        refused = paused(graph, "t")
        rg.answer(graph, "t", {"action": "leave"}, ctx)
        second = paused(graph, "t")

    assert first["allowed"] == ["leave", "stop"]
    assert first["evidence"]["notice"] == second["evidence"]["notice"] == NEEDS_FULLY_CLEARED_REVIEWER
    assert refused["evidence"]["document"] == GROCER and refused["error"] == NEEDS_FULLY_CLEARED_REVIEWER
    assert "review_status: pending" in rig.text(GROCER)
    clearance = rig.walls.review_clearance(ASSOCIATE)
    shown = {e["engagement"] for p in (first, second) for e in p["evidence"]["similar"] if e["engagement"]}
    assert shown <= clearance


def test_a_resume_in_a_new_connection_continues_at_the_same_document(rig: Rig) -> None:
    ctx = rig.context()
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        tid = rg.new_thread_id(cp, GC, TODAY)
        rg.start(graph, tid, GC, ctx)
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        assert rg.open_thread(graph, cp, GC) == tid == f"review-{GC}-2026-11-12"
        assert paused(graph, tid)["evidence"]["document"] == GROCER
        assert rg.open_thread(graph, cp, ASSOCIATE) is None


def test_stop_closes_the_thread_and_a_second_review_that_day_gets_a_new_id(rig: Rig) -> None:
    ctx = rig.context()
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        tid = rg.new_thread_id(cp, GC, TODAY)
        rg.start(graph, tid, GC, ctx)
        rg.answer(graph, tid, {"action": rg.STOP}, ctx)
        assert rg.open_thread(graph, cp, GC) is None
        assert rg.new_thread_id(cp, GC, TODAY) == f"{tid}-2"
    assert rig.audit.records() == []


def test_correct_replaces_the_decision_only_at_the_breakpoint(rig: Rig) -> None:
    ctx = rig.context()
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        rg.start(graph, "t", GC, ctx)
        with pytest.raises(ReviewError, match="only while it waits before apply"):
            rg.correct(graph, "t", {"action": "reject"}, rig.corpus)
        rg.answer(graph, "t", {"action": "approve"}, ctx)
        rg.correct(graph, "t", {"action": "reclassify", "engagement": "halcyon"}, rig.corpus)
        assert paused(graph, "t")["decision"]["engagement"] == "halcyon"
        rg.confirm(graph, "t", ctx)
        history = list(graph.get_state_history(rg.config_for("t")))

    assert "sensitivity: restricted" in rig.text(GROCER) and "engagement: halcyon" in rig.text(GROCER)
    (record,) = rig.audit.records()
    assert record["action"] == "reclassify"
    # The history keeps the approve that was corrected: the second witness.
    actions = {(s.values.get("decision") or {}).get("action") for s in history}
    assert {"approve", "reclassify"} <= actions


def test_replaying_apply_from_history_changes_nothing(rig: Rig) -> None:
    ctx = rig.context()
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        rg.start(graph, "t", GC, ctx)
        rg.answer(graph, "t", {"action": "approve"}, ctx)
        rg.confirm(graph, "t", ctx)
        after = rig.text(GROCER)
        before_apply = next(s for s in graph.get_state_history(rg.config_for("t")) if s.next == ("apply",))
        updates = list(graph.stream(None, before_apply.config, context=ctx, stream_mode="updates"))

    applied = next(u["apply"] for u in updates if "apply" in u)
    assert applied["applied"] == {"outcome": "already applied"}
    assert rig.text(GROCER) == after
    assert len(rig.audit.records()) == 1


def test_an_unknown_action_is_asked_again(rig: Rig) -> None:
    ctx = rig.context()
    with rig.checkpointer() as cp:
        graph = rg.compile_review_graph(cp)
        rg.start(graph, "t", GC, ctx)
        rg.answer(graph, "t", {"action": "publish"}, ctx)
        again = paused(graph, "t")
        rg.answer(graph, "t", {"action": "reclassify", "engagement": "Not Valid"}, ctx)
        invalid = paused(graph, "t")

    assert again["error"] == "unknown action 'publish'"
    assert "lowercase codename" in invalid["error"]
    assert again["evidence"]["document"] == invalid["evidence"]["document"] == GROCER


def test_the_checkpoint_serializer_refuses_non_json_types() -> None:
    """State holds JSON types only; the serializer is strict, so a
    dataclass in a checkpoint isn't reconstructed (plan §3.8.2)."""
    from dataclasses import dataclass

    @dataclass
    class NotJson:
        x: int = 1

    serde = rg.strict_serializer()
    plain = {"a": [1, "b", {"c": None}]}
    assert serde.loads_typed(serde.dumps_typed(plain)) == plain
    restored = serde.loads_typed(serde.dumps_typed({"v": NotJson()}))
    assert not isinstance(restored["v"], NotJson)


def test_the_audit_log_appends_a_decision_id_once(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a" / "audit.jsonl")
    record = {"decision_id": "d1", "document": GROCER, "content_hash": "h", "action": "reject"}
    assert log.append(record) is True
    assert log.append(record) is False
    assert log.rejected() == {(GROCER, "h")}


def test_the_committed_mermaid_matches_the_graph() -> None:
    drawn = rg.build_review_graph().compile().get_graph().draw_mermaid()
    assert (REPO / "src" / "tessera" / "review" / "graph.mmd").read_text(encoding="utf-8") == drawn
