"""The review graph (Phase 5, plan §3.8.2): LangGraph with ``interrupt()``
and a SQLite checkpointer.

    load_pending → next_document → prepare_evidence → review (interrupt)
        → apply (breakpoint before) → index → audit → load_pending
        → audit → load_pending                    (reject, leave pending)

- **Persistence and resume.** Every step checkpoints to SQLite. A review
  is one thread, ``review-<reviewer>-<UTC date>``; a process that stops
  at the ``review`` interrupt can exit, and a new process resumes the
  thread at the same document.
- **Idempotence.** LangGraph re-runs an interrupted node from its start,
  so ``review`` has no side effects. ``apply``, ``index`` and ``audit``
  are idempotent, keyed on the decision id plus the content hash:
  ``apply`` writes only while the file still has the hash the reviewer
  saw, recognises its own output on a re-run, and reports any other
  version as stale; ``index`` is delete-then-add on both stacks;
  ``audit`` appends a decision id once.
- **A breakpoint before ``apply``** (``interrupt_before``). A ruling that
  changes the corpus waits there, so it can be corrected with
  ``update_state()`` before anything is written (time travel), then
  confirmed by resuming. Reject and leave-pending skip it.
- **State holds JSON types only**, and the checkpointer's serializer is
  strict (``allowed_msgpack_modules=None``): a non-JSON type in state is
  refused, not unpickled.

The graph is built once; its dependencies (corpus path, walls, embedder,
evidence store, indexers, audit log, clock) arrive per run through the
runtime context, as in the query graph.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from tessera.embedding.base import Embedder
from tessera.ingestion.access_loader import Walls
from tessera.ingestion.loader import Document, load_document, scan_corpus
from tessera.review.audit import AuditLog
from tessera.review.decisions import (
    ACTIONS,
    RULINGS,
    ReviewError,
    applied_hash,
    apply_front_matter,
    content_hash,
    make_decision,
)
from tessera.review.evidence import NEEDS_FULLY_CLEARED_REVIEWER, prepare_evidence
from tessera.store.base import VectorStore

# Re-indexes one changed document in one stack; returns counts.
Indexer = Callable[[list[Document]], dict[str, int]]
STOP = "stop"


@dataclass
class ReviewContext:
    """Per-run dependencies (LangGraph runtime context)."""

    corpus_dir: Path
    walls: Walls
    embedder: Embedder
    evidence_store: VectorStore
    indexers: dict[str, Indexer]
    audit: AuditLog
    as_of: date
    clock: Callable[[], str]
    # Run after a successful index (e.g. dropping BM25 caches).
    after_index: list[Callable[[], None]] = field(default_factory=list)


class ReviewState(TypedDict, total=False):
    reviewer: str
    pending: list[dict[str, str]]  # [{"document", "content_hash"}]
    handled: list[str]  # documents already ruled on or left in this thread
    current: dict[str, str] | None
    evidence: dict[str, Any] | None
    decision: dict[str, Any] | None
    applied: dict[str, Any] | None
    indexed: dict[str, dict[str, int]] | None
    closed: bool


def _path(ctx: ReviewContext, relative: str) -> Path:
    return ctx.corpus_dir / relative


def load_pending(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    """Every quarantined document still awaiting a ruling: not rejected at
    this version, and not already handled in this thread."""
    ctx = runtime.context
    documents, _ = scan_corpus(ctx.corpus_dir)
    rejected = ctx.audit.rejected()
    handled = set(state.get("handled") or [])
    pending = []
    for doc in sorted(documents, key=lambda d: str(d.path)):
        if not doc.is_quarantined:
            continue
        relative = doc.path.relative_to(ctx.corpus_dir).as_posix()
        digest = content_hash(doc.path.read_text(encoding="utf-8"))
        if relative in handled or (relative, digest) in rejected:
            continue
        pending.append({"document": relative, "content_hash": digest})
    return {"pending": pending, "decision": None, "applied": None, "indexed": None, "evidence": None}


def next_document(state: ReviewState) -> dict[str, Any]:
    pending = state.get("pending") or []
    return {"current": pending[0] if pending else None}


def _after_next(state: ReviewState) -> str:
    return "prepare_evidence" if state.get("current") else END


def prepare_evidence_node(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    ctx = runtime.context
    current = state["current"]
    path = _path(ctx, current["document"])
    evidence = prepare_evidence(
        load_document(path),
        current["document"],
        path.read_text(encoding="utf-8"),
        reviewer=state["reviewer"],
        walls=ctx.walls,
        embedder=ctx.embedder,
        store=ctx.evidence_store,
        as_of=ctx.as_of,
        corpus_dir=ctx.corpus_dir,
    )
    return {"evidence": evidence}


def _with_applied_hash(decision: dict[str, Any], corpus_dir: Path) -> dict[str, Any]:
    """Add the hash the file will have once ``decision`` is applied — how
    a re-run of ``apply`` recognises its own write. Read-only. Computed
    only while the file is still the version the reviewer saw."""
    if decision["action"] in ("approve", "reclassify"):
        text = (corpus_dir / decision["document"]).read_text(encoding="utf-8")
        if content_hash(text) == decision["content_hash"]:
            decision["applied_hash"] = applied_hash(text, decision)
    return decision


def review(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    """Pause for the reviewer. No side effects: LangGraph re-runs this node
    from its start on resume. An action the reviewer may not take is
    refused and asked again."""
    ctx = runtime.context
    current, evidence = state["current"], state["evidence"]
    may_rule = ctx.walls.may_approve(state["reviewer"])
    allowed = [a for a in ACTIONS if may_rule or a not in RULINGS]
    error = None
    while True:
        answer = interrupt({"evidence": evidence, "allowed": [*allowed, STOP], "error": error})
        action = (answer or {}).get("action")
        if action == STOP:
            return {"closed": True, "current": None}
        if action not in allowed:
            error = NEEDS_FULLY_CLEARED_REVIEWER if action in RULINGS else f"unknown action {action!r}"
            continue
        try:
            decision = make_decision(
                current["document"],
                current["content_hash"],
                state["reviewer"],
                action,
                engagement=answer.get("engagement"),
                note=answer.get("note", ""),
            )
            _with_applied_hash(decision, ctx.corpus_dir)
        except ReviewError as exc:
            error = str(exc)
            continue
        return {"decision": decision}


def _after_review(state: ReviewState) -> str:
    if state.get("closed"):
        return END
    return "apply" if state["decision"]["action"] in ("approve", "reclassify") else "audit"


def apply(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    """Write the front matter once. Idempotent on the content hash."""
    decision = state["decision"]
    path = _path(runtime.context, decision["document"])
    text = path.read_text(encoding="utf-8")
    digest = content_hash(text)
    if digest == decision.get("applied_hash"):
        return {"applied": {"outcome": "already applied"}}
    if digest != decision["content_hash"]:
        # Edited since the reviewer saw it (or the decision was corrected
        # after an earlier write): never apply a ruling to a version the
        # reviewer didn't see.
        return {"applied": {"outcome": "stale", "found_hash": digest}}
    path.write_text(apply_front_matter(text, decision), encoding="utf-8")
    return {"applied": {"outcome": "applied"}}


def _after_apply(state: ReviewState) -> str:
    return "index" if state["applied"]["outcome"] != "stale" else "audit"


def index(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    """Re-index the document in every stack (delete-then-add: idempotent),
    then drop caches built from the old index."""
    ctx = runtime.context
    document = load_document(_path(ctx, state["decision"]["document"]))
    indexed = {name: indexer([document]) for name, indexer in sorted(ctx.indexers.items())}
    for hook in ctx.after_index:
        hook()
    return {"indexed": indexed}


def audit(state: ReviewState, runtime: Runtime[ReviewContext], config: RunnableConfig) -> dict[str, Any]:
    ctx = runtime.context
    decision = state["decision"]
    record = {
        **decision,
        "thread_id": config["configurable"]["thread_id"],
        "outcome": (state.get("applied") or {}).get("outcome")
        or {"reject": "rejected", "leave": "left pending"}[decision["action"]],
        "indexed": state.get("indexed"),
        "at": ctx.clock(),
    }
    ctx.audit.append(record)
    return {"handled": [*(state.get("handled") or []), decision["document"]]}


def build_review_graph() -> StateGraph:
    graph = StateGraph(ReviewState, context_schema=ReviewContext)
    graph.add_node("load_pending", load_pending)
    graph.add_node("next_document", next_document)
    graph.add_node("prepare_evidence", prepare_evidence_node)
    graph.add_node("review", review)
    graph.add_node("apply", apply)
    graph.add_node("index", index)
    graph.add_node("audit", audit)
    graph.add_edge(START, "load_pending")
    graph.add_edge("load_pending", "next_document")
    graph.add_conditional_edges("next_document", _after_next, ["prepare_evidence", END])
    graph.add_edge("prepare_evidence", "review")
    graph.add_conditional_edges("review", _after_review, ["apply", "audit", END])
    graph.add_conditional_edges("apply", _after_apply, ["index", "audit"])
    graph.add_edge("index", "audit")
    graph.add_edge("audit", "load_pending")
    return graph


def strict_serializer() -> JsonPlusSerializer:
    """No msgpack extension types at all: state is JSON, and anything else
    is refused rather than reconstructed (the checkpoint CVEs' class)."""
    return JsonPlusSerializer(allowed_msgpack_modules=None)


@contextmanager
def open_checkpointer(path: Path) -> Iterator[SqliteSaver]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=strict_serializer())
    finally:
        conn.close()


def compile_review_graph(checkpointer: SqliteSaver) -> Any:
    return build_review_graph().compile(checkpointer=checkpointer, interrupt_before=["apply"])


# ---- threads -----------------------------------------------------------


def thread_prefix(reviewer: str) -> str:
    return f"review-{reviewer}-"


def thread_ids(checkpointer: SqliteSaver, reviewer: str) -> list[str]:
    """The reviewer's threads, oldest first (by first checkpoint)."""
    seen: dict[str, str] = {}
    for item in checkpointer.list(None):
        tid = item.config["configurable"]["thread_id"]
        if tid.startswith(thread_prefix(reviewer)):
            ts = item.checkpoint["ts"]
            seen[tid] = min(seen.get(tid, ts), ts)
    return sorted(seen, key=lambda t: (seen[t], t))


def config_for(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def is_open(graph: Any, thread_id: str) -> bool:
    return bool(graph.get_state(config_for(thread_id)).next)


def open_thread(graph: Any, checkpointer: SqliteSaver, reviewer: str) -> str | None:
    """The reviewer's latest thread that is paused mid-review, if any."""
    for tid in reversed(thread_ids(checkpointer, reviewer)):
        if is_open(graph, tid):
            return tid
    return None


def new_thread_id(checkpointer: SqliteSaver, reviewer: str, today: date) -> str:
    """``review-<reviewer>-<UTC date>``, with ``-2``, ``-3``… for a second
    review the same day."""
    base = f"{thread_prefix(reviewer)}{today.isoformat()}"
    taken = set(thread_ids(checkpointer, reviewer))
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def start(graph: Any, thread_id: str, reviewer: str, context: ReviewContext) -> Any:
    return graph.invoke({"reviewer": reviewer, "handled": []}, config_for(thread_id), context=context)


def answer(graph: Any, thread_id: str, reply: dict[str, Any], context: ReviewContext) -> Any:
    """Resume the ``review`` interrupt with the reviewer's reply."""
    return graph.invoke(Command(resume=reply), config_for(thread_id), context=context)


def confirm(graph: Any, thread_id: str, context: ReviewContext) -> Any:
    """Continue past the breakpoint before ``apply``."""
    return graph.invoke(None, config_for(thread_id), context=context)


def correct(graph: Any, thread_id: str, reply: dict[str, Any], corpus_dir: Path) -> None:
    """Replace the decision waiting at the breakpoint before ``apply``
    (time travel: ``update_state()`` writes a new checkpoint as if
    ``review`` had returned it)."""
    snapshot = graph.get_state(config_for(thread_id))
    if snapshot.next != ("apply",):
        raise ReviewError("a decision can be corrected only while it waits before apply")
    state = snapshot.values
    current = state["current"]
    decision = make_decision(
        current["document"], current["content_hash"], state["reviewer"], reply["action"],
        engagement=reply.get("engagement"), note=reply.get("note", ""),
    )
    _with_applied_hash(decision, corpus_dir)
    graph.update_state(config_for(thread_id), {"decision": decision}, as_node="review")


def pause(snapshot: Any) -> dict[str, Any] | None:
    """What the thread is waiting for: the ``review`` interrupt's payload,
    ``{"breakpoint": "apply", "decision": …}``, or None when finished."""
    if not snapshot.next:
        return None
    if snapshot.next == ("apply",):
        return {"breakpoint": "apply", "decision": snapshot.values.get("decision")}
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


__all__ = [
    "ReviewContext",
    "ReviewState",
    "answer",
    "build_review_graph",
    "compile_review_graph",
    "confirm",
    "correct",
    "new_thread_id",
    "open_checkpointer",
    "open_thread",
    "pause",
    "start",
    "strict_serializer",
    "thread_ids",
]
