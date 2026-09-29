"""Retry/backoff and call pacing for any LLMClient.

A decorator over the LLMClient port rather than logic inside a concrete
client, so the Phase 4 Bedrock client gets the same behaviour for free.
Motivated by 2026-09-29: NVIDIA NIM throttled far below its documented
40 rpm, an unpaced 55-case eval sweep cascaded into instant 429s after
the first few calls, and nothing in the repo backed off (see
checkpoint.md Notes).

Pure with respect to infrastructure per CLAUDE.md constraint #6 in the
sense that matters here: the clock and sleep are injected (real ones by
default), nothing is read from the environment, and progress is reported
through a callback rather than printed.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from tessera.generation.base import LLMClient

# Backoff bases, in seconds, doubled per attempt and capped. 429 (rate
# limit) starts higher than a 5xx overload: a 429 means the window is
# spent, so retrying fast just extends the penalty. The 45 s / 15 s pair
# is the one that got a full 55-case sweep through on 2026-09-29.
RATE_LIMIT_BASE_SECONDS = 45.0
SERVER_ERROR_BASE_SECONDS = 15.0
MAX_BACKOFF_SECONDS = 120.0
DEFAULT_MAX_ATTEMPTS = 6


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def is_retryable(exc: BaseException) -> bool:
    """429 and 5xx are transient; everything else (400, 401, parse errors,
    connection errors with no status) is raised straight away.
    """
    code = _status_code(exc)
    return code is not None and (code == 429 or 500 <= code < 600)


def _retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


class RetryingLLMClient(LLMClient):
    """Wraps another LLMClient: spaces calls at least ``min_interval``
    seconds apart, and retries transient failures (429 / 5xx) with
    exponential backoff, honouring a Retry-After header when the server
    sends one. After ``max_attempts`` the last error is raised.

    on_retry(attempt, delay_seconds, error) is called before each sleep so
    a composition root can show progress; this class never prints.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        min_interval: float = 0.0,
        on_retry: Callable[[int, float, BaseException], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._inner = inner
        self._max_attempts = max_attempts
        self._min_interval = min_interval
        self._on_retry = on_retry
        self._sleep = sleep
        self._clock = clock
        self._last_call_at: float | None = None

    def _pace(self) -> None:
        if self._min_interval > 0 and self._last_call_at is not None:
            wait = self._last_call_at + self._min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_call_at = self._clock()

    def _backoff(self, attempt: int, exc: BaseException) -> float:
        advised = _retry_after_seconds(exc)
        if advised is not None:
            return min(advised, MAX_BACKOFF_SECONDS)
        base = (
            RATE_LIMIT_BASE_SECONDS
            if _status_code(exc) == 429
            else SERVER_ERROR_BASE_SECONDS
        )
        return min(base * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        for attempt in range(1, self._max_attempts + 1):
            self._pace()
            try:
                return self._inner.complete(system, user, temperature)
            except Exception as exc:
                if not is_retryable(exc) or attempt == self._max_attempts:
                    raise
                delay = self._backoff(attempt, exc)
                if self._on_retry is not None:
                    self._on_retry(attempt, delay, exc)
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover
