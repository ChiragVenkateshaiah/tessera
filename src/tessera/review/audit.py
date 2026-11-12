"""The review audit log, ``data/review/audit.jsonl`` (Phase 5, plan §3.8.5).

One JSON line per decision, appended once: ``append`` is keyed on the
decision id, so a resumed or re-run ``audit`` step finds its own earlier
line and writes nothing. It is also where rejection lives —
``review_status`` has no "rejected" value — so ``rejected()`` is what keeps
a rejected version from being offered again. A rejection names a content
hash: if the document is edited, the new version comes back for review.

Local runtime data (gitignored, plan §10.3); the path is a parameter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def has(self, decision_id: str) -> bool:
        return any(r["decision_id"] == decision_id for r in self.records())

    def append(self, record: dict[str, Any]) -> bool:
        """Append ``record`` unless its decision id is already logged.
        Returns whether a line was written."""
        if self.has(record["decision_id"]):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return True

    def rejected(self) -> set[tuple[str, str]]:
        """(document, content hash) of every rejected version."""
        return {(r["document"], r["content_hash"]) for r in self.records() if r["action"] == "reject"}


__all__ = ["AuditLog"]
