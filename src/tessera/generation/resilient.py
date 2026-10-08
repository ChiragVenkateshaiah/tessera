"""Retry/backoff and call pacing for any LLMClient.

A decorator over the LLMClient port rather than logic inside a concrete
client, so the Phase 4 Bedrock and Gemini clients get the same behaviour for free.
Motivated by 2026-09-29: NVIDIA NIM throttled far below its documented
40 rpm, an unpaced 55-case eval sweep cascaded into instant 429s after
the first few calls, and nothing in the repo backed off (see
checkpoint.md Notes).

Pure with respect to infrastructure per CLAUDE.md constraint #6 in the
sense that matters here: the clock and sleep are injected (real ones by
default), nothing is read from the environment, and progress is reported
through a callback rather than printed.

Thread-safe (2026-10-08): ``tessera eval --workers N`` runs cases on
threads that share one client. Each call reserves the next free slot
under a lock, so the spacing holds across threads, and a 429 pushes that
slot out by its backoff, so every thread (the one that got it included,
through the same ``_pace``) waits it out once instead of spending the same
exhausted window.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from tessera.generation.base import Completion, LLMClient

# Backoff bases, in seconds, doubled per attempt and capped. 429 (rate
# limit) starts higher than a 5xx overload: a 429 means the window is
# spent, so retrying fast just extends the penalty. The 45 s / 15 s pair
# is the one that got a full 55-case sweep through on 2026-09-29.
RATE_LIMIT_BASE_SECONDS = 45.0
SERVER_ERROR_BASE_SECONDS = 15.0
MAX_BACKOFF_SECONDS = 120.0
DEFAULT_MAX_ATTEMPTS = 6


def _status_code(exc: BaseException) -> int | None:
    # openai/anthropic errors carry `status_code`; google-genai's APIError
    # carries the HTTP status as `code`.
    for attr in ("status_code", "code"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    return None


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
        self._lock = threading.Lock()
        # The earliest moment the next call may start (spacing and any
        # 429 cool-down), shared by every thread using this client.
        self._next_slot: float | None = None

    def _pace(self) -> None:
        with self._lock:
            now = self._clock()
            start = now if self._next_slot is None else max(now, self._next_slot)
            self._next_slot = start + self._min_interval
        if start > now:
            self._sleep(start - now)

    def _cool_down(self, seconds: float) -> None:
        """A 429: no thread starts a call before this one's backoff ends."""
        with self._lock:
            until = self._clock() + seconds
            self._next_slot = until if self._next_slot is None else max(self._next_slot, until)

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
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(
        self, system: str, user: str, temperature: float = 0.0
    ) -> Completion:
        for attempt in range(1, self._max_attempts + 1):
            self._pace()
            try:
                return self._inner.complete_with_usage(system, user, temperature)
            except Exception as exc:
                if not is_retryable(exc) or attempt == self._max_attempts:
                    raise
                delay = self._backoff(attempt, exc)
                if self._on_retry is not None:
                    self._on_retry(attempt, delay, exc)
                if _status_code(exc) == 429:
                    # The window is spent for every thread: book the wait
                    # as the shared next slot; _pace sleeps it, once.
                    self._cool_down(delay)
                else:
                    self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover
