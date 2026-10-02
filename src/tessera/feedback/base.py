"""Ports for the feedback loop (Phase 4, plan §3.2.2): where traces and
feedback are kept. Local JSONL implementations live in `local.py`; a
managed store (and CloudWatch for traces) replaces them later without the
callers changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

Rating = Literal["up", "down"]


@dataclass(frozen=True)
class Feedback:
    """One rating of one answer, tied to that answer's trace."""

    trace_id: str
    rating: Rating
    created_at: datetime
    reason: str | None = None
    comment: str | None = None


class TraceLog(ABC):
    """Append-only log of trace records (see `tessera.trace.trace_record`)."""

    @abstractmethod
    def append(self, record: dict[str, Any]) -> None:
        """Store one record; it must carry a ``trace_id``."""

    @abstractmethod
    def get(self, trace_id: str) -> dict[str, Any] | None:
        """The record with this trace_id, or None."""


class FeedbackStore(ABC):
    """Where ratings are kept."""

    @abstractmethod
    def add(self, feedback: Feedback) -> None:
        """Store one rating."""

    @abstractmethod
    def list(self) -> list[Feedback]:
        """Every rating, oldest first."""
