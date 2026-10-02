"""Thumbs-down feedback → *candidate* eval cases (plan §3.2.2).

A candidate carries what was observed (the router's archetype, the
sources or people the model was shown, the reviewer's reason) and leaves
the labels empty. It is marked ``status: candidate``, which
`evals.harness.load_cases` refuses to load: an unlabelled case in
`evals/cases/` would corrupt the bar, so a human has to label it and
delete that line first. Pure — the CLI reads the stores and writes the
file.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from tessera.feedback.base import Feedback

CANDIDATE_STATUS = "candidate"

CANDIDATES_HEADER = """\
# CANDIDATE eval cases from thumbs-down feedback — NOT LABELLED.
# Written by `tessera feedback to-cases`. To promote one into the eval set:
#   1. confirm or correct `archetype` (it is what the router chose);
#   2. fill `relevant_sources` (A/C, corpus-relative paths) or
#      `relevant_people` (B), and `ideal_answer`, by reading the corpus —
#      `observed` is what the system did, not what is right;
#   3. delete the `status: candidate` line (load_cases refuses it) and
#      `observed`, give it a stable id, and move it into evals/cases/;
#   4. run `tessera eval --check` and paste the report in the PR.
"""


def _relative(path: str, corpus_dir: Path) -> str:
    try:
        return Path(path).relative_to(corpus_dir).as_posix()
    except ValueError:
        return path


def candidate_cases(
    feedback: list[Feedback],
    get_trace: Callable[[str], dict[str, Any] | None],
    corpus_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """One candidate per trace whose latest rating is thumbs-down.

    Returns the candidates and the trace_ids that had no trace record
    (feedback for an answer whose trace was lost can't become a case).
    """
    latest: dict[str, Feedback] = {}
    for item in feedback:  # oldest first, so later ratings win
        latest[item.trace_id] = item

    cases: list[dict[str, Any]] = []
    missing: list[str] = []
    for trace_id, item in latest.items():
        if item.rating != "down":
            continue
        trace = get_trace(trace_id)
        if trace is None:
            missing.append(trace_id)
            continue
        used = [r for r in trace.get("retrieved", []) if r.get("used")]
        sources: list[str] = []
        for r in used:
            if r.get("document_path"):
                rel = _relative(r["document_path"], corpus_dir)
                if rel not in sources:
                    sources.append(rel)
        people = [r["id"] for r in used] if trace.get("retrieved_kind") == "person" else []
        cases.append(
            {
                "id": f"fb-{trace_id[:8]}",
                "status": CANDIDATE_STATUS,
                "query": trace["query"],
                "archetype": trace["archetype"],
                "relevant_sources": [],
                "relevant_people": [],
                "ideal_answer": "",
                "observed": {
                    "trace_id": trace_id,
                    "timestamp": trace.get("timestamp"),
                    "sources_shown": sources,
                    "people_shown": people,
                    "fixed_response": trace.get("fixed_response", False),
                    "reason": item.reason,
                    "comment": item.comment,
                },
            }
        )
    return cases, missing


def render_candidates(cases: list[dict[str, Any]]) -> str:
    """The staging file's text: the how-to-label header, then the cases."""
    body = yaml.safe_dump(cases, sort_keys=False, allow_unicode=True, width=88)
    return CANDIDATES_HEADER + "\n" + (body if cases else "[]\n")
