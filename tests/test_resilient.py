"""RetryingLLMClient — retry/backoff/pacing, with a fake clock and sleep so
nothing here actually waits.
"""

import pytest

from tessera.generation.base import LLMClient
from tessera.generation.resilient import (
    DEFAULT_MAX_ATTEMPTS,
    MAX_BACKOFF_SECONDS,
    RATE_LIMIT_BASE_SECONDS,
    SERVER_ERROR_BASE_SECONDS,
    Pacer,
    RetryingLLMClient,
    is_retryable,
)


class HttpError(Exception):
    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        headers = {"retry-after": retry_after} if retry_after is not None else {}
        self.response = type("R", (), {"headers": headers})()


class ScriptedInner(LLMClient):
    """Raises/returns the scripted outcomes in order."""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)


class FakeTime:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


def make(inner: LLMClient, t: FakeTime, **kw) -> RetryingLLMClient:
    return RetryingLLMClient(inner, sleep=t.sleep, clock=t.clock, **kw)


def test_success_passes_through_with_no_sleep() -> None:
    t = FakeTime()
    client = make(ScriptedInner(["hello"]), t)
    assert client.complete("s", "u") == "hello"
    assert t.sleeps == []


def test_retries_429_then_succeeds_with_a_long_first_backoff() -> None:
    t = FakeTime()
    inner = ScriptedInner([HttpError(429), "ok"])
    assert make(inner, t).complete("s", "u") == "ok"
    assert inner.calls == 2
    assert t.sleeps == [RATE_LIMIT_BASE_SECONDS]


def test_503_backs_off_less_than_429() -> None:
    t = FakeTime()
    make(ScriptedInner([HttpError(503), "ok"]), t).complete("s", "u")
    assert t.sleeps == [SERVER_ERROR_BASE_SECONDS]
    assert SERVER_ERROR_BASE_SECONDS < RATE_LIMIT_BASE_SECONDS


def test_backoff_doubles_and_is_capped() -> None:
    t = FakeTime()
    inner = ScriptedInner([HttpError(429)] * 4 + ["ok"])
    make(inner, t, max_attempts=5).complete("s", "u")
    assert t.sleeps == [45.0, 90.0, MAX_BACKOFF_SECONDS, MAX_BACKOFF_SECONDS]


def test_gives_up_after_max_attempts_and_raises_the_last_error() -> None:
    t = FakeTime()
    inner = ScriptedInner([HttpError(503)] * 3)
    with pytest.raises(HttpError):
        make(inner, t, max_attempts=3).complete("s", "u")
    assert inner.calls == 3
    assert len(t.sleeps) == 2  # no sleep after the final failure


@pytest.mark.parametrize("status", [400, 401, 404])
def test_non_transient_errors_are_raised_immediately(status: int) -> None:
    t = FakeTime()
    inner = ScriptedInner([HttpError(status), "never reached"])
    with pytest.raises(HttpError):
        make(inner, t).complete("s", "u")
    assert inner.calls == 1 and t.sleeps == []


def test_errors_without_a_status_code_are_not_retried() -> None:
    t = FakeTime()
    inner = ScriptedInner([ValueError("bad json"), "never reached"])
    with pytest.raises(ValueError):
        make(inner, t).complete("s", "u")
    assert inner.calls == 1


def test_retry_after_header_is_honoured_and_capped() -> None:
    t = FakeTime()
    make(ScriptedInner([HttpError(429, retry_after="7"), "ok"]), t).complete("s", "u")
    assert t.sleeps == [7.0]

    t2 = FakeTime()
    make(ScriptedInner([HttpError(429, retry_after="9999"), "ok"]), t2).complete("s", "u")
    assert t2.sleeps == [MAX_BACKOFF_SECONDS]


def test_on_retry_is_told_the_attempt_delay_and_error() -> None:
    t = FakeTime()
    seen: list[tuple[int, float, int]] = []
    client = make(
        ScriptedInner([HttpError(429), HttpError(503), "ok"]),
        t,
        on_retry=lambda attempt, delay, exc: seen.append((attempt, delay, exc.status_code)),
    )
    client.complete("s", "u")
    assert seen == [(1, 45.0, 429), (2, 30.0, 503)]


def test_min_interval_spaces_consecutive_calls() -> None:
    t = FakeTime()
    client = make(ScriptedInner(["a", "b", "c"]), t, min_interval=3.0)
    client.complete("s", "u")
    assert t.sleeps == []  # first call is never delayed
    client.complete("s", "u")
    assert t.sleeps == [3.0]
    t.now += 10  # a long gap already satisfies the interval
    client.complete("s", "u")
    assert t.sleeps == [3.0]


def test_pacing_holds_across_threads_sharing_one_client() -> None:
    """tessera eval --workers: every call, from any thread, starts at
    least min_interval after the one before it (real clock, 50 ms)."""
    import threading
    import time

    starts: list[float] = []
    lock = threading.Lock()

    class Recording(LLMClient):
        def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
            with lock:
                starts.append(time.monotonic())
            return "ok"

    client = RetryingLLMClient(Recording(), min_interval=0.05)
    threads = [threading.Thread(target=lambda: [client.complete("s", "u") for _ in range(3)]) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    starts.sort()
    assert len(starts) == 12
    assert min(b - a for a, b in zip(starts, starts[1:])) >= 0.045  # timer slack


def test_a_429_holds_back_every_caller_until_its_backoff_ends() -> None:
    t = FakeTime()
    client = make(ScriptedInner(["a"]), t)
    client._cool_down(45.0)  # another thread's 429, just now
    client.complete("s", "u")
    assert t.sleeps == [45.0]


def test_is_retryable_classification() -> None:
    assert is_retryable(HttpError(429)) and is_retryable(HttpError(500))
    assert is_retryable(HttpError(599))
    assert not is_retryable(HttpError(499)) and not is_retryable(RuntimeError("x"))


def test_rejects_zero_attempts_and_has_a_sane_default() -> None:
    with pytest.raises(ValueError):
        RetryingLLMClient(ScriptedInner([]), max_attempts=0)
    assert DEFAULT_MAX_ATTEMPTS >= 3


# --- the shared, adaptive Pacer (2026-10-08) ---


def test_two_clients_sharing_one_pacer_are_spaced_as_one() -> None:
    """An eval's answer client and judge client: one rate limit, one pacer."""
    t = FakeTime()
    pacer = Pacer(3.0, sleep=t.sleep, clock=t.clock)
    answers = RetryingLLMClient(ScriptedInner(["a", "b"]), pacer=pacer, sleep=t.sleep)
    judge = RetryingLLMClient(ScriptedInner(["j"]), pacer=pacer, sleep=t.sleep)

    answers.complete("s", "u")
    judge.complete("s", "u")
    answers.complete("s", "u")

    assert t.sleeps == [3.0, 3.0]
    assert answers.pacer is judge.pacer


def test_an_adaptive_pacer_doubles_on_throttling_and_eases_after_a_streak() -> None:
    t = FakeTime()
    pacer = Pacer(3.0, max_interval=30.0, ease_by=0.5, successes_to_ease=2, sleep=t.sleep, clock=t.clock)
    client = RetryingLLMClient(
        ScriptedInner([HttpError(503), HttpError(429), "a", "b", "c", "d"]), pacer=pacer, sleep=t.sleep
    )

    client.complete("s", "u")  # 503, 429, then ok
    assert pacer.interval == 12.0 and pacer.throttled_calls == 2
    client.complete("s", "u")  # two successes in a row: ease once
    assert pacer.interval == 11.5
    client.complete("s", "u")
    client.complete("s", "u")
    assert pacer.interval == 11.0 and pacer.peak_interval == 12.0


def test_an_adaptive_pacer_stays_between_its_bounds() -> None:
    pacer = Pacer(3.0, max_interval=10.0, ease_by=5.0, successes_to_ease=1)
    for _ in range(10):
        pacer.throttled()
    assert pacer.interval == 10.0
    for _ in range(10):
        pacer.succeeded()
    assert pacer.interval == 3.0


def test_a_fixed_pacer_never_changes_its_interval() -> None:
    pacer = Pacer(3.0)
    pacer.throttled()
    pacer.succeeded()
    assert not pacer.adaptive and pacer.interval == 3.0 and pacer.throttled_calls == 1
    with pytest.raises(ValueError):
        Pacer(3.0, max_interval=1.0)


def test_a_client_error_does_not_count_as_throttling() -> None:
    pacer = Pacer(3.0, max_interval=30.0)
    client = RetryingLLMClient(ScriptedInner([HttpError(400)]), pacer=pacer, sleep=lambda s: None)
    with pytest.raises(HttpError):
        client.complete("s", "u")
    assert pacer.throttled_calls == 0 and pacer.interval == 3.0
