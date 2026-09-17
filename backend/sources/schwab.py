"""Schwab source adapter — proxy over schwab_client (OAuth boundary, INV-C).

`with_retry(call)` is opt-in: on a token failure it drops the in-memory client
cache so the next call re-reads the stored token (letting a fresh login or
schwab-py's refresh-token flow take effect), then retries once. A second failure
raises AuthError. No token file is ever unlinked here — deleting the credential
is an explicit /api/auth/clear action, never a side effect of a failed call.

A token failure is either a 401 response or one of authlib's OAuth errors. The
latter matter because authlib *raises* rather than returning a response when a
refresh is rejected, and those exceptions carry no `status_code` — so a 401-only
check saw `invalid_grant` as an upstream fault and surfaced a 502 that no amount
of reconnecting would clear.

FOUNDATION-RATE-LIMIT-1 — every call also passes a token-bucket rate gate
before `call()` is invoked, and a 429 response is retried with backoff
(honoring `Retry-After` when Schwab sends it) before raising `RateLimitError`.
This lives here, not per-caller, for the same reason `with_retry` does: one
chokepoint every module already imports (INV-13) is the only place a
process-wide budget can actually be enforced — a per-call-site limiter cannot
see calls made by other call sites.
"""
from __future__ import annotations

import functools
import inspect
import threading
import time
from collections.abc import Callable
from typing import Any

import schwab_client as _sc
from schwab_client import (  # noqa: F401  (re-exports)
    AuthError,
    ConfigError,
    active_token_path,
    auth_diagnostic,
    auth_status,
    build_authorization_url,
    clear_token,
    complete_from_url,
    exchange_code,
    has_token,
    invalid_env,
    is_configured,
    login_context,
    missing_env,
    record_auth_event,
    token_source,
)
from settings import settings


class RateLimitError(Exception):
    """Raised when the Schwab rate gate cannot admit a call within its wait cap.

    Two distinct triggers both raise this, both loudly:
    - the token bucket had no slot available within
      `settings.schwab_rate_limit_wait_cap_sec`.
    - a 429 response's backoff (bounded by `settings.schwab_429_max_retries`
      and `settings.schwab_429_max_wait_sec`) was exhausted without success.

    A 429 that turned into a silently-empty payload was the outcome this
    slice exists to prevent -- every other error path in this module surfaces
    visibly (AuthError, or the original exception re-raised), and this must
    too.
    """


class _TokenBucket:
    """Thread-safe token bucket: `capacity` tokens, refilled continuously.

    Refill rate is `capacity` tokens per 60 seconds (i.e. the configured
    per-minute budget), so the bucket is really "N calls per rolling minute"
    rather than a once-a-minute reset -- a caller isn't punished for the
    accident of which wall-clock minute a burst lands in.

    `time_source` is injectable so tests drive the bucket with a fake clock
    instead of real `time.sleep` -- a per-minute budget proven only by
    sleeping a minute is a test nobody runs.
    """

    def __init__(self, capacity: int, *, time_source: Callable[[], float] = time.monotonic):
        self._capacity = max(1, capacity)
        self._rate_per_sec = self._capacity / 60.0
        self._time = time_source
        self._tokens = float(self._capacity)
        self._last_refill = self._time()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def _refill_locked(self) -> None:
        now = self._time()
        elapsed = max(0.0, now - self._last_refill)
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)
            self._last_refill = now

    def acquire(self, *, wait_cap_sec: float) -> bool:
        """Block until a token is available or `wait_cap_sec` elapses.

        Returns True once a token has been taken. Returns False if the wait
        cap was reached with no token available -- the caller decides how to
        surface that (this slice raises RateLimitError).
        """
        deadline = self._time() + max(0.0, wait_cap_sec)
        with self._condition:
            while True:
                self._refill_locked()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

                now = self._time()
                remaining = deadline - now
                if remaining <= 0:
                    return False

                # Time until at least one token accrues, capped by whatever
                # wait budget is left -- avoids a long, unresponsive sleep
                # when the deadline is closer than the next token.
                tokens_needed = 1.0 - self._tokens
                time_to_next_token = tokens_needed / self._rate_per_sec if self._rate_per_sec > 0 else remaining
                self._condition.wait(timeout=min(remaining, max(time_to_next_token, 0.001)))


class _RateLimitStats:
    """Thread-safe counters for FOUNDATION-RATE-LIMIT-1 observability.

    Deliberately not wired to any route in this slice -- see
    `rate_limit_stats()` for how a future slice reads this.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls_total = 0
        self._throttled_429_count = 0
        self._waited_calls_count = 0

    def record_call(self) -> None:
        with self._lock:
            self._calls_total += 1

    def record_429(self) -> None:
        with self._lock:
            self._throttled_429_count += 1

    def record_waited(self) -> None:
        with self._lock:
            self._waited_calls_count += 1

    def snapshot(self) -> dict[str, int]:
        # `calls_total` is cumulative for the life of the process, and is named
        # that way deliberately. An earlier draft called it `calls_in_window`
        # and shipped a `window_sec: 60` alongside it, which would have told a
        # future Settings tile that 4,000 calls had happened in the last
        # minute. The bucket knows the rolling budget; these counters do not,
        # and a counter must not imply a window it never measures.
        with self._lock:
            return {
                "calls_total": self._calls_total,
                "limit_per_min": settings.schwab_rate_limit_per_min,
                "throttled_429_count": self._throttled_429_count,
                "waited_calls_count": self._waited_calls_count,
            }


# Process-wide singletons -- one bucket, one set of counters, for the entire
# process, matching cache.py's per-process (not cross-process) scope.
_bucket = _TokenBucket(settings.schwab_rate_limit_per_min)
_stats = _RateLimitStats()


def rate_limit_stats() -> dict[str, int]:
    """Snapshot of the rate gate's counters.

    Not exposed over HTTP by this slice. A future slice wires this into a
    Settings/status route: import this function, wrap the dict in its own
    Pydantic response_model, and add a `_map/contracts.md` entry at that
    time -- see _map/specs/rate-limit-1.md Section D.
    """
    return _stats.snapshot()


def _reset_rate_gate_for_tests(capacity: int | None = None, *, time_source=None) -> None:
    """Test-only: replace the process-wide bucket/stats with fresh ones.

    Production code never calls this. Tests use it (via monkeypatch/direct
    call) to get a clean bucket sized and clocked for the scenario under
    test, without depending on real wall-clock time or leaking state between
    tests.
    """
    global _bucket, _stats
    cap = capacity if capacity is not None else settings.schwab_rate_limit_per_min
    kwargs = {"time_source": time_source} if time_source is not None else {}
    _bucket = _TokenBucket(cap, **kwargs)
    _stats = _RateLimitStats()


def _retry_after_seconds(resp: Any) -> float | None:
    """Parse a Retry-After header (seconds form) off a response, if present."""
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _is_429(resp: Any = None, exc: Exception | None = None) -> bool:
    if resp is not None and _status_code(resp) == 429:
        return True
    if exc is not None and _status_code(exc) == 429:
        return True
    return False



class _RetryingClient:
    """schwab-py client whose calls survive a token rotation.

    Every method goes through `with_retry`, so a call that fails because the
    stored token was rotated (a refresh mints a new refresh token and revokes
    the previous one) drops the cached client, re-reads the token file, and tries
    once more. Only if the *second* attempt still fails does the caller see an
    `AuthError` -- which routes render as 401 `auth_required`, the code the UI
    turns into "reconnect".

    Why this is a proxy and not a wrapper at each call site: `with_retry` was
    written, tested, and then called by nothing. All 35 Schwab call sites used
    the raw client, so authlib's `invalid_grant` sailed past the classifier into
    the generic `except Exception` and surfaced as 502 `upstream_error`. A dead
    refresh token therefore reported itself as a Schwab outage, and no auth
    banner ever appeared. Putting the retry behind the one function every module
    already imports fixes all of them at once and leaves nothing to remember.

    The method is re-resolved from `_sc.get_client()` on *each* attempt rather
    than bound once. `with_retry` rebuilds the client between tries, so a bound
    method would retry against the very instance that just failed and the retry
    would be theatre.
    """

    def __getattr__(self, name: str) -> Any:
        target = getattr(_sc.get_client(), name)
        if not callable(target) or inspect.iscoroutinefunction(target):
            # Attributes and coroutines pass through untouched: `with_retry` is
            # synchronous, and wrapping a coroutine function would hand back an
            # un-awaited coroutine with no retry semantics at all.
            return target

        @functools.wraps(target)
        def call_with_retry(*args: Any, **kwargs: Any) -> Any:
            return with_retry(lambda: getattr(_sc.get_client(), name)(*args, **kwargs))

        return call_with_retry

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<_RetryingClient wrapping {_sc.get_client()!r}>"


def get_client() -> Any:
    """The Schwab client every module uses. Retries once on a token rotation.

    Returns a proxy, not schwab-py's client. `raw_client()` is the escape hatch
    for the rare caller that needs the real object.
    """
    _sc.get_client()  # surface ConfigError/AuthError here, not on first method
    return _RetryingClient()


def raw_client() -> Any:
    """The unwrapped schwab-py client, with no retry behaviour.

    For callers that need the real object -- identity checks, or passing it to
    schwab-py internals. Prefer `get_client()`.
    """
    return _sc.get_client()


def sdk_client_cls() -> Any:
    """Lazy lookup of schwab.client.Client (for SDK enum access)."""
    from schwab.client import Client  # type: ignore  # noqa: PLC0415

    return Client


def sdk_stream_client_cls() -> Any:
    """Lazy lookup of schwab.streaming.StreamClient."""
    from schwab.streaming import StreamClient  # type: ignore  # noqa: PLC0415

    return StreamClient


def _status_code(obj: Any) -> int | None:
    sc = getattr(obj, "status_code", None)
    if isinstance(sc, int):
        return sc
    inner = getattr(obj, "response", None)
    sc = getattr(inner, "status_code", None) if inner is not None else None
    return sc if isinstance(sc, int) else None


# authlib's machine-readable codes for "this token will not work again", as
# opposed to a transient upstream fault. `unsupported_token_type` is in the list
# because authlib reports an unsignable token that way.
_TOKEN_ERROR_CODES = frozenset({
    "invalid_grant",
    "invalid_token",
    "token_expired",
    "token_invalid",
    "missing_token",
    "unsupported_token_type",
})


def _is_token_failure(exc: Exception) -> bool:
    """True when `exc` means the stored token is the problem."""
    if _status_code(exc) == 401:
        return True
    code = getattr(exc, "error", None)
    return isinstance(code, str) and code in _TOKEN_ERROR_CODES


def _reauth() -> AuthError:
    """The error a caller should see once a rebuilt token is still rejected."""
    return AuthError(
        "Schwab rejected the stored token after a rebuild. "
        "Connect Schwab from the dashboard to sign in again."
    )


def _rate_gated_call(call: Callable[[], Any]) -> Any:
    """Run one attempt of `call` behind the rate gate, with bounded 429 backoff.

    Every individual attempt `with_retry` makes (initial, and the one retry
    after a token rebuild) passes through here, so neither attempt can bypass
    the budget -- a retry that skipped the gate would defeat the whole point
    of metering at this chokepoint (INV-13).

    A 429 is distinct from a token failure: it means the *bucket accounting
    here* under-counted relative to Schwab's own view (a neighboring process,
    clock drift, or a burst that raced the refill math), not that the token is
    bad. So it is retried here, with backoff, rather than treated as a reason
    to rebuild the client.
    """
    attempts = 0
    while True:
        if not _bucket.acquire(wait_cap_sec=settings.schwab_rate_limit_wait_cap_sec):
            raise RateLimitError(
                f"Schwab rate gate: no slot available within "
                f"{settings.schwab_rate_limit_wait_cap_sec}s wait cap "
                f"(budget={settings.schwab_rate_limit_per_min}/min)."
            )
        _stats.record_call()

        try:
            resp = call()
        except Exception as exc:  # noqa: BLE001
            if not _is_429(exc=exc):
                raise
            _stats.record_429()
            attempts += 1
            if attempts > settings.schwab_429_max_retries:
                raise RateLimitError(
                    f"Schwab 429: exceeded {settings.schwab_429_max_retries} retries."
                ) from exc
            _backoff_for_429(attempt=attempts, exc=exc)
            continue

        if not _is_429(resp=resp):
            return resp

        _stats.record_429()
        attempts += 1
        if attempts > settings.schwab_429_max_retries:
            raise RateLimitError(
                f"Schwab 429: exceeded {settings.schwab_429_max_retries} retries."
            )
        _backoff_for_429(attempt=attempts, resp=resp)


# Indirection so tests can prove "waited N seconds" by recording calls to a
# fake sleep rather than by actually blocking the test process -- a per-
# minute-budget test suite that really sleeps is a test suite nobody runs.
_sleep: Callable[[float], None] = time.sleep


# First backoff step when Schwab sends a 429 with no Retry-After header.
# Doubles per attempt (1s, 2s, 4s...), capped by schwab_429_max_wait_sec.
_BACKOFF_BASE_SEC = 1.0


def _backoff_for_429(*, attempt: int, resp: Any = None, exc: Any = None) -> None:
    """Sleep before the next 429 retry, bounded by the configured max wait.

    Retry-After wins when Schwab sends it -- it is Schwab telling us exactly
    when it will serve us again, and guessing shorter just burns a retry.

    With no header we back off exponentially from `_BACKOFF_BASE_SEC`, NOT by
    waiting the maximum. An earlier draft defaulted the headerless case to
    `schwab_429_max_wait_sec` (30s), which at 3 retries could pin a FastAPI
    worker for 90 seconds on a single dashboard tile -- worse for the user
    than a fast, honest error, and worse for the process, since these calls
    serve synchronous requests. Exponential-from-1s means the common case
    (a brief burst) clears in about a second.
    """
    retry_after = _retry_after_seconds(resp) if resp is not None else _retry_after_seconds(exc)
    if retry_after is None:
        wait = _BACKOFF_BASE_SEC * (2 ** max(0, attempt - 1))
    else:
        wait = retry_after
    wait = min(max(0.0, wait), settings.schwab_429_max_wait_sec)
    if wait > 0:
        _stats.record_waited()
        _sleep(wait)


def with_retry(call: Callable[[], Any]) -> Any:
    """Run `call`; on a token failure, rebuild the client once and retry.

    Every attempt passes the rate gate and bounded 429 backoff in
    `_rate_gated_call` first -- the token-refresh retry semantics below are
    layered on top, unchanged from before FOUNDATION-RATE-LIMIT-1.
    """
    try:
        resp = _rate_gated_call(call)
        if _status_code(resp) != 401:
            return resp
    except Exception as exc:  # noqa: BLE001
        if not _is_token_failure(exc):
            raise
        _sc.invalidate_client()
        try:
            return _rate_gated_call(call)
        except Exception as exc2:  # noqa: BLE001
            if _is_token_failure(exc2):
                raise _reauth() from exc2
            raise

    _sc.invalidate_client()
    resp2 = _rate_gated_call(call)
    if _status_code(resp2) == 401:
        raise _reauth()
    return resp2
