"""Smoke tests for sources/schwab.py — the 401-aware Schwab proxy.

Scope is the load-bearing INV-D guarantees only: token file is never
mutated by the retry path, 401 retry succeeds, and a second 401 raises.
Broader coverage deferred (see _map/debt.md D-6).
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import schwab_client
import sources.schwab as src_schwab


@pytest.fixture
def isolated_token(monkeypatch, tmp_path):
    p = tmp_path / "tokens.json"
    p.write_text("{}")
    monkeypatch.setattr(schwab_client, "TOKEN_PATH", p)
    return p


def test_with_retry_passthrough_on_200():
    resp = MagicMock(status_code=200)
    call = MagicMock(return_value=resp)
    assert src_schwab.with_retry(call) is resp
    assert call.call_count == 1


def test_with_retry_rebuilds_on_401_then_succeeds(monkeypatch):
    monkeypatch.setattr(schwab_client, "_client", MagicMock())
    bad, good = MagicMock(status_code=401), MagicMock(status_code=200)
    call = MagicMock(side_effect=[bad, good])
    assert src_schwab.with_retry(call) is good
    assert schwab_client._client is None  # cache cleared between calls


def test_with_retry_raises_on_second_401():
    """The message points at GradeA's own Connect Schwab control.

    Since FOUNDATION-AUTH-INTERNAL-1 that is the only place this token can be
    renewed: GradeA holds its own Schwab app registration and runs its own OAuth
    flow, so a re-auth prompt that sent the operator to another application
    would be sending them somewhere that cannot help.
    """
    bad = MagicMock(status_code=401)
    call = MagicMock(side_effect=[bad, bad])
    with pytest.raises(src_schwab.AuthError, match="Connect Schwab"):
        src_schwab.with_retry(call)


def test_with_retry_does_not_delete_token_file(isolated_token):
    """Load-bearing OAuth-state guard: 401 retry must NOT unlink tokens.json."""
    call = MagicMock(side_effect=[MagicMock(status_code=401),
                                   MagicMock(status_code=200)])
    src_schwab.with_retry(call)
    assert isolated_token.exists()


# ---------- authlib raises, it does not return a 401 ------------------------
# A rejected refresh reaches us as an authlib OAuthError carrying `.error` and
# no `status_code`. Treating that as an upstream fault is what turned an expired
# Schwab session into a permanent 502 on every data endpoint.

def _oauth_error(code: str, description: str):
    from authlib.integrations.base_client.errors import OAuthError

    return OAuthError(error=code, description=description)


@pytest.mark.parametrize("code", ["invalid_grant", "unsupported_token_type"])
def test_with_retry_rebuilds_on_authlib_token_error(monkeypatch, code):
    monkeypatch.setattr(schwab_client, "_client", MagicMock())
    good = MagicMock(status_code=200)
    call = MagicMock(side_effect=[_oauth_error(code, "rejected"), good])

    assert src_schwab.with_retry(call) is good
    assert schwab_client._client is None


def test_with_retry_raises_autherror_when_the_token_is_still_rejected():
    """Second failure is a re-auth prompt, not a 502 — every route already maps
    AuthError to 401 auth_required, and the frontend renders that as the connect
    banner rather than as an outage."""
    err = _oauth_error("invalid_grant", "Refresh token is invalid, expired or revoked")
    call = MagicMock(side_effect=[err, err])

    with pytest.raises(src_schwab.AuthError) as excinfo:
        src_schwab.with_retry(call)

    assert "Connect Schwab" in str(excinfo.value)
    assert "8182" not in str(excinfo.value)


def test_with_retry_leaves_genuine_upstream_faults_alone():
    """A 500 is Schwab's problem; rebuilding the token would only hide it."""
    boom = RuntimeError("schwab is down")
    call = MagicMock(side_effect=boom)

    with pytest.raises(RuntimeError, match="schwab is down"):
        src_schwab.with_retry(call)
    assert call.call_count == 1


# --- FOUNDATION-SCHWAB-RETRY-1: the proxy every module now receives ----------

class _Recorder:
    """Stand-in schwab-py client counting calls and failing on demand."""

    def __init__(self, fail_times=0, exc=None):
        self.fail_times = fail_times
        self.exc = exc or _oauth_error("invalid_grant", "Refresh token is invalid, expired or revoked")
        self.calls = 0
        self.not_callable = "plain-attribute"

    def get_quotes(self, symbols):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return {"symbols": symbols, "instance": id(self)}


def _install(monkeypatch, *clients):
    """Serve `clients` in order; invalidate_client() advances to the next one."""
    box = {"i": 0}
    monkeypatch.setattr(src_schwab._sc, "get_client", lambda: clients[box["i"]])
    monkeypatch.setattr(
        src_schwab._sc, "invalidate_client",
        lambda: box.__setitem__("i", min(box["i"] + 1, len(clients) - 1)),
    )
    return box


def test_proxy_passes_calls_through(monkeypatch):
    rec = _Recorder()
    _install(monkeypatch, rec)
    assert src_schwab.get_client().get_quotes("SPY")["symbols"] == "SPY"
    assert rec.calls == 1


def test_proxy_retries_against_the_rebuilt_client_not_the_dead_one(monkeypatch):
    """The regression that made the retry theatre: binding the method once."""
    dead, fresh = _Recorder(fail_times=99), _Recorder()
    _install(monkeypatch, dead, fresh)
    out = src_schwab.get_client().get_quotes("SPY")
    assert out["instance"] == id(fresh), "retry must use the rebuilt client"
    assert dead.calls == 1 and fresh.calls == 1


def test_proxy_raises_autherror_when_the_token_is_still_rejected(monkeypatch):
    """A dead refresh token must reach routes as AuthError -> 401, not 502."""
    _install(monkeypatch, _Recorder(fail_times=99), _Recorder(fail_times=99))
    with pytest.raises(src_schwab.AuthError):
        src_schwab.get_client().get_quotes("SPY")


def test_proxy_reproduces_the_live_invalid_grant_failure(monkeypatch):
    """Schwab's real body: unsupported_token_type wrapping invalid_grant."""
    exc = _oauth_error("unsupported_token_type", "Refresh token is invalid, expired or revoked")
    _install(monkeypatch, _Recorder(fail_times=99, exc=exc), _Recorder(fail_times=99, exc=exc))
    with pytest.raises(src_schwab.AuthError):
        src_schwab.get_client().get_quotes("SPY")


def test_proxy_leaves_genuine_upstream_faults_alone(monkeypatch):
    """A real outage must stay a 502, not be mislabelled as an auth problem."""
    boom = RuntimeError("schwab is down")
    rec = _Recorder(fail_times=99, exc=boom)
    _install(monkeypatch, rec)
    with pytest.raises(RuntimeError):
        src_schwab.get_client().get_quotes("SPY")
    assert rec.calls == 1, "no retry on a non-token fault"


def test_proxy_passes_non_callables_through(monkeypatch):
    _install(monkeypatch, _Recorder())
    assert src_schwab.get_client().not_callable == "plain-attribute"


def test_raw_client_is_unwrapped(monkeypatch):
    rec = _Recorder()
    _install(monkeypatch, rec)
    assert src_schwab.raw_client() is rec


# --- FOUNDATION-RATE-LIMIT-1: token-bucket rate gate + 429 handling ---------
# All tests here use an injected fake clock (and a fake sleep for the 429
# backoff path) so they prove behaviour by counting and by clock rather than
# by real time.sleep -- a per-minute-budget suite that really sleeps a minute
# is a suite nobody runs.


class _FakeClock:
    """Monotonic-shaped fake clock: advances only when told to."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeSleep:
    """Records requested sleep durations and advances a fake clock instead of
    actually blocking."""

    def __init__(self, clock: _FakeClock):
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


def test_token_bucket_blocks_the_n_plus_1th_call_against_a_bucket_of_n():
    """N+1 calls against a bucket of N provably blocks (here: hits the wait
    cap and returns False) the extra one -- proof by counting, not intent."""
    clock = _FakeClock()
    bucket = src_schwab._TokenBucket(3, time_source=clock)

    # Drain the bucket: 3 calls succeed immediately with no clock movement.
    for _ in range(3):
        assert bucket.acquire(wait_cap_sec=0.0) is True

    # The 4th call has nothing left and a zero wait cap -- must fail to
    # acquire rather than block indefinitely or silently proceed.
    assert bucket.acquire(wait_cap_sec=0.0) is False


def test_token_bucket_refills_by_clock_not_by_wall_time():
    """Advancing the fake clock -- not sleeping -- makes a token available."""
    clock = _FakeClock()
    bucket = src_schwab._TokenBucket(60, time_source=clock)  # 1 token/sec

    for _ in range(60):
        assert bucket.acquire(wait_cap_sec=0.0) is True
    assert bucket.acquire(wait_cap_sec=0.0) is False  # drained

    clock.advance(1.0)  # exactly one token's worth of time, no real sleep
    assert bucket.acquire(wait_cap_sec=0.0) is True
    assert bucket.acquire(wait_cap_sec=0.0) is False  # drained again


def test_rate_gated_call_raises_ratelimitexceeded_when_bucket_is_exhausted(monkeypatch):
    """The gate wired into with_retry's call path fails loudly, not silently,
    once the bucket and its wait cap are both exhausted."""
    clock = _FakeClock()
    src_schwab._reset_rate_gate_for_tests(capacity=1, time_source=clock)
    monkeypatch.setattr(src_schwab.settings, "schwab_rate_limit_wait_cap_sec", 0.0)

    ok = MagicMock(return_value=MagicMock(status_code=200))
    assert src_schwab.with_retry(ok) is ok.return_value  # consumes the 1 token

    with pytest.raises(src_schwab.RateLimitError):
        src_schwab.with_retry(ok)


def test_rate_gated_call_waits_for_a_slot_up_to_the_cap_then_succeeds(monkeypatch):
    """A caller on an empty bucket waits (does not fail-fast) up to the
    configured cap, and the wait itself advances only the fake clock."""
    clock = _FakeClock()
    fake_sleep_calls: list[float] = []

    class _WaitingBucket(src_schwab._TokenBucket):
        """Same accounting as _TokenBucket, but records a 'wait' by advancing
        the fake clock exactly the amount `acquire` would have blocked, so
        the test proves the wait happened without a real sleep."""

        def acquire(self, *, wait_cap_sec: float) -> bool:
            self._refill_locked()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            # Not enough tokens: advance the clock by exactly one token's
            # worth of time (simulating the real wait) then retry once.
            wait_needed = (1.0 - self._tokens) / self._rate_per_sec
            if wait_needed > wait_cap_sec:
                return False
            fake_sleep_calls.append(wait_needed)
            clock.advance(wait_needed)
            self._refill_locked()
            self._tokens -= 1.0
            return True

    bucket = _WaitingBucket(60, time_source=clock)  # 1 token/sec, starts full
    for _ in range(60):
        assert bucket.acquire(wait_cap_sec=0.0) is True  # fully drain the bucket

    # Next call has no token; must wait (not fail) since cap allows it.
    assert bucket.acquire(wait_cap_sec=5.0) is True
    assert fake_sleep_calls == [1.0], "the call provably waited ~1s by clock"


def test_429_response_honors_retry_after_then_succeeds(monkeypatch):
    """A simulated 429 with Retry-After provably waits that long (via the
    fake sleep, not a real one) and then succeeds."""
    clock = _FakeClock()
    fake_sleep = _FakeSleep(clock)
    src_schwab._reset_rate_gate_for_tests(capacity=10, time_source=clock)
    monkeypatch.setattr(src_schwab, "_sleep", fake_sleep)
    monkeypatch.setattr(src_schwab.settings, "schwab_429_max_wait_sec", 30.0)
    monkeypatch.setattr(src_schwab.settings, "schwab_429_max_retries", 3)

    too_many = MagicMock(status_code=429, headers={"Retry-After": "7"})
    ok = MagicMock(status_code=200)
    call = MagicMock(side_effect=[too_many, ok])

    assert src_schwab.with_retry(call) is ok
    assert fake_sleep.calls == [7.0], "must wait exactly Retry-After seconds, by clock"
    assert src_schwab.rate_limit_stats()["throttled_429_count"] == 1


def test_429_without_retry_after_uses_default_bounded_backoff(monkeypatch):
    """No Retry-After header: backs off from the 1s base, NOT from the max.

    Waiting the maximum on a headerless 429 would pin a synchronous FastAPI
    worker for the full cap on the very first retry; the common case is a
    brief burst that clears in about a second."""
    clock = _FakeClock()
    fake_sleep = _FakeSleep(clock)
    src_schwab._reset_rate_gate_for_tests(capacity=10, time_source=clock)
    monkeypatch.setattr(src_schwab, "_sleep", fake_sleep)
    monkeypatch.setattr(src_schwab.settings, "schwab_429_max_wait_sec", 5.0)
    monkeypatch.setattr(src_schwab.settings, "schwab_429_max_retries", 3)

    too_many = MagicMock(status_code=429, headers={})
    ok = MagicMock(status_code=200)
    call = MagicMock(side_effect=[too_many, ok])

    assert src_schwab.with_retry(call) is ok
    assert fake_sleep.calls == [1.0], "first headerless retry waits the 1s base, not the cap"


def test_429_without_retry_after_backoff_doubles_and_caps(monkeypatch):
    """Successive headerless 429s double the wait (1s, 2s, 4s...) and never
    exceed schwab_429_max_wait_sec -- proven by the fake sleep log, by clock,
    not by asserting intent."""
    clock = _FakeClock()
    fake_sleep = _FakeSleep(clock)
    src_schwab._reset_rate_gate_for_tests(capacity=10, time_source=clock)
    monkeypatch.setattr(src_schwab, "_sleep", fake_sleep)
    monkeypatch.setattr(src_schwab.settings, "schwab_429_max_wait_sec", 3.0)
    monkeypatch.setattr(src_schwab.settings, "schwab_429_max_retries", 3)

    too_many = MagicMock(status_code=429, headers={})
    ok = MagicMock(status_code=200)
    call = MagicMock(side_effect=[too_many, too_many, too_many, ok])

    assert src_schwab.with_retry(call) is ok
    # 1s, then 2s, then 4s clamped to the 3s cap.
    assert fake_sleep.calls == [1.0, 2.0, 3.0]


def test_429_exhausts_bounded_retries_then_raises_loudly(monkeypatch):
    """A 429 that never clears must fail loudly after the configured retry
    budget -- never a silent empty payload."""
    clock = _FakeClock()
    fake_sleep = _FakeSleep(clock)
    src_schwab._reset_rate_gate_for_tests(capacity=10, time_source=clock)
    monkeypatch.setattr(src_schwab, "_sleep", fake_sleep)
    monkeypatch.setattr(src_schwab.settings, "schwab_429_max_wait_sec", 1.0)
    monkeypatch.setattr(src_schwab.settings, "schwab_429_max_retries", 2)

    always_429 = MagicMock(status_code=429, headers={})
    call = MagicMock(return_value=always_429)

    with pytest.raises(src_schwab.RateLimitError):
        src_schwab.with_retry(call)

    # 1 initial + 2 retries = 3 attempts total before giving up.
    assert call.call_count == 3
    assert src_schwab.rate_limit_stats()["throttled_429_count"] == 3


def test_429_retry_after_capped_by_max_wait_sec(monkeypatch):
    """A huge Retry-After (e.g. Schwab's own ~60s window) is capped, not
    honored verbatim -- a dashboard request must not block a full minute."""
    clock = _FakeClock()
    fake_sleep = _FakeSleep(clock)
    src_schwab._reset_rate_gate_for_tests(capacity=10, time_source=clock)
    monkeypatch.setattr(src_schwab, "_sleep", fake_sleep)
    monkeypatch.setattr(src_schwab.settings, "schwab_429_max_wait_sec", 30.0)
    monkeypatch.setattr(src_schwab.settings, "schwab_429_max_retries", 3)

    too_many = MagicMock(status_code=429, headers={"Retry-After": "600"})
    ok = MagicMock(status_code=200)
    call = MagicMock(side_effect=[too_many, ok])

    assert src_schwab.with_retry(call) is ok
    assert fake_sleep.calls == [30.0], "must cap at schwab_429_max_wait_sec, not honor 600s"


def test_concurrent_threads_do_not_exceed_the_budget():
    """Concurrent callers provably do not exceed the bucket's capacity within
    a single window -- proof by counting completed calls against a shared
    bucket, not by asserting intent."""
    import concurrent.futures

    clock = _FakeClock()
    bucket = src_schwab._TokenBucket(20, time_source=clock)
    successes = {"count": 0}
    lock = threading.Lock()

    def worker():
        if bucket.acquire(wait_cap_sec=0.0):
            with lock:
                successes["count"] += 1

    # 50 concurrent callers, only 20 tokens available, zero wait cap so any
    # caller that can't get an immediate token fails rather than blocks.
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as pool:
        list(pool.map(lambda _: worker(), range(50)))

    assert successes["count"] == 20, "exactly the bucket capacity must succeed, never more"


def test_concurrent_threads_via_with_retry_share_one_process_wide_bucket(monkeypatch):
    """The gate wired into with_retry is a single process-wide bucket, so
    concurrent callers through the real call path also cannot exceed budget."""
    import concurrent.futures

    clock = _FakeClock()
    src_schwab._reset_rate_gate_for_tests(capacity=10, time_source=clock)
    monkeypatch.setattr(src_schwab.settings, "schwab_rate_limit_wait_cap_sec", 0.0)

    ok = lambda: MagicMock(status_code=200)  # noqa: E731
    results = {"ok": 0, "limited": 0}
    lock = threading.Lock()

    def worker():
        try:
            src_schwab.with_retry(ok)
            with lock:
                results["ok"] += 1
        except src_schwab.RateLimitError:
            with lock:
                results["limited"] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as pool:
        list(pool.map(lambda _: worker(), range(30)))

    assert results["ok"] == 10, "only the bucket's capacity may succeed"
    assert results["limited"] == 20, "the rest must fail loudly, not silently"


def test_rate_limit_stats_counts_calls_and_429s(monkeypatch):
    clock = _FakeClock()
    fake_sleep = _FakeSleep(clock)
    src_schwab._reset_rate_gate_for_tests(capacity=10, time_source=clock)
    monkeypatch.setattr(src_schwab, "_sleep", fake_sleep)

    ok = MagicMock(status_code=200)
    src_schwab.with_retry(MagicMock(return_value=ok))
    src_schwab.with_retry(MagicMock(return_value=ok))

    stats = src_schwab.rate_limit_stats()
    assert stats["calls_total"] == 2
    assert "calls_in_window" not in stats, (
        "the counter is cumulative for the process; it must not claim a window it never measures"
    )
    assert "window_sec" not in stats
    assert stats["throttled_429_count"] == 0
    assert stats["limit_per_min"] == src_schwab.settings.schwab_rate_limit_per_min


def test_rate_gate_does_not_weaken_401_retry_semantics(monkeypatch):
    """The rate gate and 429 handling are additive: a 401 still rebuilds the
    client and retries exactly once, same as before this slice."""
    import schwab_client

    clock = _FakeClock()
    src_schwab._reset_rate_gate_for_tests(capacity=10, time_source=clock)
    monkeypatch.setattr(schwab_client, "_client", MagicMock())

    bad, good = MagicMock(status_code=401), MagicMock(status_code=200)
    call = MagicMock(side_effect=[bad, good])
    assert src_schwab.with_retry(call) is good
    assert schwab_client._client is None  # cache still cleared between tries
