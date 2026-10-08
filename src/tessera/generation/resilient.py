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
threads that share clients, and the clients share one ``Pacer`` (below),
which can also adapt its interval to 429s and 5xx.
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


class Pacer:
    """When the next LLM call may start, shared by every client (and every
    thread) that holds the same instance.

    Each call reserves the next free slot under a lock, so the spacing
    holds across threads. A 429 books its backoff as the shared next slot
    (``cool_down``), so every caller waits it out once instead of spending
    the same exhausted window.

    **Adaptive** when ``max_interval`` is above ``min_interval``
    (2026-10-08): additive-increase / multiplicative-decrease, the rule TCP
    uses for congestion, run backwards on the interval. A throttled call
    (429 or 5xx) doubles the interval, up to ``max_interval``; every
    ``successes_to_ease`` successes in a row take ``ease_by`` seconds off
    it, down to ``min_interval``. A fixed interval can't find the
    provider's real capacity, which for NIM under load is far below its
    documented 40 rpm: four workers at a fixed 3 s made the 2026-10-08
    native sweep no faster and gave it 11 ERROR rows instead of 3.
    """

    def __init__(
        self,
        min_interval: float = 0.0,
        *,
        max_interval: float | None = None,
        ease_by: float = 0.5,
        successes_to_ease: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_interval is not None and max_interval < min_interval:
            raise ValueError("max_interval must be >= min_interval")
        self.min_interval = min_interval
        self.max_interval = max_interval if max_interval is not None else min_interval
        self._ease_by = ease_by
        self._successes_to_ease = successes_to_ease
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._interval = min_interval
        self._streak = 0
        self.throttled_calls = 0
        self.peak_interval = min_interval
        # The earliest moment the next call may start.
        self._next_slot: float | None = None

    @property
    def adaptive(self) -> bool:
        return self.max_interval > self.min_interval

    @property
    def interval(self) -> float:
        return self._interval

    def wait(self) -> None:
        """Block until this caller's reserved slot."""
        with self._lock:
            now = self._clock()
            start = now if self._next_slot is None else max(now, self._next_slot)
            self._next_slot = start + self._interval
        if start > now:
            self._sleep(start - now)

    def cool_down(self, seconds: float) -> None:
        """No caller starts before ``seconds`` from now (a 429's backoff)."""
        with self._lock:
            until = self._clock() + seconds
            self._next_slot = until if self._next_slot is None else max(self._next_slot, until)

    def throttled(self) -> None:
        """A 429 or 5xx: back off multiplicatively (adaptive only)."""
        with self._lock:
            self.throttled_calls += 1
            self._streak = 0
            if self.adaptive:
                self._interval = min(self.max_interval, max(self._interval, 1.0) * 2)
                self.peak_interval = max(self.peak_interval, self._interval)

    def succeeded(self) -> None:
        """A call went through: ease off additively after a streak."""
        with self._lock:
            self._streak += 1
            if self.adaptive and self._streak >= self._successes_to_ease:
                self._streak = 0
                self._interval = max(self.min_interval, self._interval - self._ease_by)


class RetryingLLMClient(LLMClient):
    """Wraps another LLMClient: spaces calls through a ``Pacer`` and
    retries transient failures (429 / 5xx) with exponential backoff,
    honouring a Retry-After header when the server sends one. After
    ``max_attempts`` the last error is raised.

    Pass one ``pacer`` to every client that shares a rate limit (an eval
    sweep's answer, router and judge clients, native or LangChain);
    without one, the client gets its own fixed ``min_interval`` pacer.

    on_retry(attempt, delay_seconds, error) is called before each sleep so
    a composition root can show progress; this class never prints.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        min_interval: float = 0.0,
        pacer: Pacer | None = None,
        on_retry: Callable[[int, float, BaseException], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._inner = inner
        self._max_attempts = max_attempts
        self._pacer = pacer if pacer is not None else Pacer(min_interval, sleep=sleep, clock=clock)
        self._on_retry = on_retry
        self._sleep = sleep

    @property
    def pacer(self) -> Pacer:
        return self._pacer

    @property
    def _min_interval(self) -> float:
        return self._pacer.min_interval

    def _cool_down(self, seconds: float) -> None:
        self._pacer.cool_down(seconds)

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
            self._pacer.wait()
            try:
                completion = self._inner.complete_with_usage(system, user, temperature)
            except Exception as exc:
                if not is_retryable(exc):
                    raise
                self._pacer.throttled()
                if attempt == self._max_attempts:
                    raise
                delay = self._backoff(attempt, exc)
                if self._on_retry is not None:
                    self._on_retry(attempt, delay, exc)
                if _status_code(exc) == 429:
                    # The window is spent for every caller: book the wait
                    # as the shared next slot; Pacer.wait sleeps it, once.
                    self._pacer.cool_down(delay)
                else:
                    self._sleep(delay)
            else:
                self._pacer.succeeded()
                return completion
        raise AssertionError("unreachable")  # pragma: no cover
