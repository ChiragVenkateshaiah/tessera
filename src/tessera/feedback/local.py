"""Local JSONL implementations of the feedback-loop ports — one JSON object
per line, appended, so a crash mid-write loses at most the last line and
the files stay greppable. Paths are constructor parameters; the CLI reads
them from config.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from tessera.feedback.base import Feedback, FeedbackStore, TraceLog


def _append_line(path: Path, obj: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from an interrupted write
    return records


class JsonlTraceLog(TraceLog):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        if "trace_id" not in record:
            raise ValueError("a trace record needs a trace_id")
        _append_line(self._path, record, self._lock)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        # A linear scan: fine for a local log; the managed store indexes it.
        for record in _read_lines(self._path):
            if record.get("trace_id") == trace_id:
                return record
        return None


class JsonlFeedbackStore(FeedbackStore):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def add(self, feedback: Feedback) -> None:
        _append_line(
            self._path,
            {
                "trace_id": feedback.trace_id,
                "rating": feedback.rating,
                "created_at": feedback.created_at.isoformat(),
                "reason": feedback.reason,
                "comment": feedback.comment,
            },
            self._lock,
        )

    def list(self) -> list[Feedback]:
        return [
            Feedback(
                trace_id=r["trace_id"],
                rating=r["rating"],
                created_at=datetime.fromisoformat(r["created_at"]),
                reason=r.get("reason"),
                comment=r.get("comment"),
            )
            for r in _read_lines(self._path)
        ]
