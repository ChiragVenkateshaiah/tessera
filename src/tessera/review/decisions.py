"""A review decision and what it does to a document's front matter
(Phase 5, plan §3.8.2, §3.8.5). Pure: text in, text out — no file I/O.

A decision is plain JSON (it lives in the review graph's checkpointed
state, which holds JSON types only). Four outcomes:

- ``approve``: indexed as ``internal`` (open to everyone);
- ``reclassify``: indexed as ``restricted`` to one engagement;
- ``reject``: not indexed and not offered again. ``review_status`` has no
  "rejected" value, so the front matter is untouched and the audit log
  records it;
- ``leave``: still pending; offered again next time.

Idempotence keys: the **decision id** names one ruling on one version of
one document, and the **content hash** is that version. ``apply`` writes
only when the file still has the hash the reviewer saw, and recognises
its own output on a re-run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import yaml

from tessera.ingestion.loader import (
    REVIEW_REVIEWED,
    SENSITIVITY_INTERNAL,
    SENSITIVITY_RESTRICTED,
    front_matter_problems,
)

Action = Literal["approve", "reclassify", "reject", "leave"]
ACTIONS: tuple[str, ...] = ("approve", "reclassify", "reject", "leave")
# Outcomes that change the corpus and need a fully cleared reviewer.
RULINGS: frozenset[str] = frozenset({"approve", "reclassify", "reject"})


class ReviewError(ValueError):
    """A decision that can't be applied as given."""


def content_hash(text: str) -> str:
    """The version of a document a reviewer saw: sha256 of the whole file."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def decision_id(document: str, digest: str, reviewer: str, action: str, engagement: str | None = None) -> str:
    """Deterministic, so the same ruling on the same version always has the
    same id: a resumed or re-run step finds its own earlier record."""
    key = json.dumps([document, digest, reviewer, action, engagement])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def make_decision(
    document: str,
    digest: str,
    reviewer: str,
    action: str,
    *,
    engagement: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """A validated decision as JSON-ready data."""
    if action not in ACTIONS:
        raise ReviewError(f"unknown action {action!r}: use one of {list(ACTIONS)}")
    if action == "reclassify" and not engagement:
        raise ReviewError("reclassify needs an engagement")
    if action != "reclassify" and engagement:
        raise ReviewError(f"{action} takes no engagement")
    return {
        "decision_id": decision_id(document, digest, reviewer, action, engagement),
        "document": document,
        "content_hash": digest,
        "reviewer": reviewer,
        "action": action,
        "engagement": engagement,
        "note": note,
    }


def _split(text: str) -> tuple[list[str], str]:
    """(front-matter lines, the rest verbatim) of a ``---``-fenced file."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ReviewError("the document has no front matter")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ReviewError("the front matter is not closed") from None
    return lines[1:end], "\n".join(lines[end:])


def _set(header: list[str], key: str, value: str | None) -> list[str]:
    """Set (or, with ``None``, remove) one top-level scalar key, in place
    if present, else appended. Other lines are kept byte for byte."""
    prefix = f"{key}:"
    out, found = [], False
    for line in header:
        if line.startswith(prefix):
            found = True
            if value is not None:
                out.append(f"{key}: {value}")
            continue
        out.append(line)
    if not found and value is not None:
        out.append(f"{key}: {value}")
    return out


def apply_front_matter(text: str, decision: dict[str, Any]) -> str:
    """The document text after ``decision``. Only ``approve`` and
    ``reclassify`` change it: ``review_status`` becomes ``reviewed`` and
    the access labels are set. The body and every other front-matter line
    are kept verbatim. The result must load cleanly."""
    action = decision["action"]
    if action not in ("approve", "reclassify"):
        return text
    header, rest = _split(text)
    header = _set(header, "review_status", REVIEW_REVIEWED)
    if action == "approve":
        header = _set(header, "sensitivity", SENSITIVITY_INTERNAL)
        header = _set(header, "engagement", None)
    else:
        header = _set(header, "sensitivity", SENSITIVITY_RESTRICTED)
        header = _set(header, "engagement", decision["engagement"])
    problems = front_matter_problems(yaml.safe_load("\n".join(header)) or {})
    if problems:
        raise ReviewError(f"{decision['document']}: {'; '.join(problems)}")
    return "\n".join(["---", *header, rest])


def applied_hash(text_before: str, decision: dict[str, Any]) -> str:
    """The content hash the document has once ``decision`` is applied."""
    return content_hash(apply_front_matter(text_before, decision))


__all__ = [
    "ACTIONS",
    "RULINGS",
    "ReviewError",
    "applied_hash",
    "apply_front_matter",
    "content_hash",
    "decision_id",
    "make_decision",
]
