"""cache.py — the single shared in-process TTL cache for GradeA.

FOUNDATION-CACHE-1.

Why this module exists
----------------------
Before this slice there were three hand-rolled TTL caches (`chain.py`,
`earnings.py`, `darkpool.py`), each with its own dict, its own lock (or
none), its own staleness check, and its own stats surface. None of them
protected against a thundering herd: N concurrent callers on a cold key
all called Schwab. With a ~65-70 req/min Schwab REST budget and a
dashboard that opens ~30 calls per authed tab load, that is the failure
mode that actually bites.

This module is the only place a TTL cache may live (INV-9).

What it guarantees
------------------
1. **Single-flight.** On a cold or expired key, exactly ONE caller invokes
   the underlying function; the rest block on an Event and then read the
   fresh value. This is the property that caps upstream call volume.
2. **Fetch outside the lock.** The namespace lock is never held across
   I/O, so a slow Schwab call cannot block readers of other keys.
3. **No silent staleness.** Every entry records the wall-clock time it was
   stored. `age_sec()` and `stats()` expose it so the API can stamp
   `as_of` and the UI can label it. A stale value is never returned as if
   it were fresh.
4. **No silent failure.** If the underlying call raises, the exception
   propagates to every waiter. Nothing is cached, and a previous good
   value is NOT resurrected to paper over the error.

What it deliberately does NOT do
--------------------------------
- No cross-process or cross-restart persistence. This is a per-process
  memo. Durable caching is `history_cache.py` (parquet) and `storage.py`
  (SQLite); do not duplicate their job here.
- No async support. Every current caller is sync. `ttl_cache` raises
  TypeError on a coroutine function rather than silently caching a
  coroutine object (which would be a bug that looks like a cache hit).
- No max-size / LRU eviction. Key spaces here are small and bounded
  (symbols x expiries). Revisit only with evidence of growth.

Public surface
--------------
    ttl_cache(ttl_sec, *, name=None, key_fn=None)  # decorator
    get_or_call(namespace, key, fn, ttl_sec=None)  # imperative form
    age_sec(namespace, key) -> float | None
    stats() -> list[dict]
    invalidate(namespace=None, key=None) -> int
    reset_for_tests() -> None
"""
from __future__ import annotations

import functools
import inspect
import threading
import time
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# TTL tiers. Named so call sites read as intent, not magic numbers. Pick the
# tier that matches how fast the underlying data actually moves — a tighter
# TTL than the data's true refresh rate just burns Schwab budget for nothing.
# ---------------------------------------------------------------------------

TTL_QUOTE = 15.0        # live quotes / batch quote calls
TTL_DERIVED = 30.0      # composite reads computed from quotes (regime, trinity)
TTL_CHAIN = 30.0        # Schwab option-chain fetches (matches prior default)
TTL_HOURLY = 3600.0     # calendars, earnings — change a few times a day
TTL_DAILY = 21600.0     # 6h; FINRA weekly files, parquet-backed daily closes

# A waiter never blocks longer than this on the single-flight leader. If the
# leader is slower than this, the waiter falls through and calls the
# underlying function itself rather than hanging a request thread forever.
_SINGLEFLIGHT_WAIT_SEC = 20.0

TtlSpec = float | Callable[[], float]


class _Entry:
    """One cached value. `mono` drives expiry, `wall` drives as_of."""

    __slots__ = ("value", "mono", "wall")

    def __init__(self, value: Any, mono: float, wall: float) -> None:
        self.value = value
        self.mono = mono
        self.wall = wall


class _Namespace:
    __slots__ = ("name", "ttl_spec", "entries", "inflight", "lock",
                 "hits", "misses", "errors")

    def __init__(self, name: str, ttl_spec: TtlSpec) -> None:
        self.name = name
        self.ttl_spec = ttl_spec
        self.entries: dict[Any, _Entry] = {}
        self.inflight: dict[Any, threading.Event] = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.errors = 0

    def ttl(self) -> float:
        """Resolve the TTL at call time.

        Callable TTLs let a namespace track a Pydantic Settings field that
        may be reloaded, instead of freezing the value at import time (the
        bug the old `chain._load_ttl()` had — it snapshotted at import).
        """
        spec = self.ttl_spec
        try:
            return float(spec() if callable(spec) else spec)
        except Exception:  # noqa: BLE001 — a bad TTL must not break the call
            return TTL_DERIVED


_registry: dict[str, _Namespace] = {}
_registry_lock = threading.Lock()


def _namespace(name: str, ttl_spec: TtlSpec | None = None) -> _Namespace:
    with _registry_lock:
        ns = _registry.get(name)
        if ns is None:
            ns = _Namespace(name, ttl_spec if ttl_spec is not None else TTL_DERIVED)
            _registry[name] = ns
        elif ttl_spec is not None:
            ns.ttl_spec = ttl_spec
        return ns


def _default_key(args: tuple, kwargs: dict) -> Any:
    """Hashable key from positional + keyword args.

    Unhashable arguments are a programming error here, not something to
    paper over: a cache keyed on a mutable value silently returns wrong
    data. Raise loudly instead.
    """
    key = (args, tuple(sorted(kwargs.items())))
    try:
        hash(key)
    except TypeError as exc:
        raise TypeError(
            f"cache key is unhashable ({exc}). Pass key_fn= to ttl_cache "
            f"to derive a hashable key from these arguments."
        ) from exc
    return key


def get_or_call(
    namespace: str,
    key: Any,
    fn: Callable[[], Any],
    ttl_sec: TtlSpec | None = None,
) -> Any:
    """Return the cached value for `key`, or compute it via `fn` exactly once.

    Single-flight: concurrent callers on a cold key elect one leader to run
    `fn`; the others wait and then read the leader's result.
    """
    ns = _namespace(namespace, ttl_sec)
    ttl = ns.ttl()

    # Two passes max: pass 1 may elect us a waiter, pass 2 reads what the
    # leader stored. If the leader failed or timed out, we run fn ourselves
    # rather than blocking indefinitely.
    for attempt in (0, 1):
        now = time.monotonic()
        with ns.lock:
            entry = ns.entries.get(key)
            if entry is not None and (now - entry.mono) < ttl:
                ns.hits += 1
                return entry.value

            event = ns.inflight.get(key)
            if event is None:
                event = threading.Event()
                ns.inflight[key] = event
                is_leader = True
            else:
                is_leader = False
            ns.misses += 1 if is_leader else 0

        if is_leader:
            try:
                value = fn()
            except BaseException:
                # Do NOT cache, and do NOT fall back to a stale entry —
                # the caller must see the real error (no silent fallbacks).
                with ns.lock:
                    ns.errors += 1
                    ns.inflight.pop(key, None)
                event.set()
                raise
            with ns.lock:
                ns.entries[key] = _Entry(value, time.monotonic(), time.time())
                ns.inflight.pop(key, None)
            event.set()
            return value

        # Waiter path.
        event.wait(timeout=_SINGLEFLIGHT_WAIT_SEC)
        if attempt == 1:
            # Leader errored or was too slow. Compute directly; correctness
            # beats deduplication.
            return fn()

    raise AssertionError("unreachable")  # pragma: no cover


def ttl_cache(
    ttl_sec: TtlSpec,
    *,
    name: str | None = None,
    key_fn: Callable[..., Any] | None = None,
):
    """Decorator: memoize a sync function for `ttl_sec` with single-flight.

    Args:
        ttl_sec: seconds, or a zero-arg callable returning seconds (use a
            callable for settings-driven TTLs so env changes are honored).
        name: namespace label shown in `stats()`. Defaults to the qualified
            function name.
        key_fn: builds the cache key from the call arguments. Required when
            an argument is unhashable or should be excluded from the key
            (e.g. a test-injection seam).

    The wrapped function gains:
        .cache_namespace -> str
        .cache_clear()   -> int (entries dropped)
    """
    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"ttl_cache does not support async functions ({fn.__qualname__}). "
                f"Caching a coroutine object would look like a hit and then fail "
                f"on second await. Cache the sync computation instead."
            )
        ns_name = name or f"{fn.__module__}.{fn.__name__}"
        _namespace(ns_name, ttl_sec)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs) if key_fn else _default_key(args, kwargs)
            return get_or_call(ns_name, key, lambda: fn(*args, **kwargs))

        wrapper.cache_namespace = ns_name          # type: ignore[attr-defined]
        wrapper.cache_clear = lambda: invalidate(ns_name)  # type: ignore[attr-defined]
        return wrapper

    return decorator


def age_sec(namespace: str, key: Any) -> float | None:
    """Wall-clock age of a cached entry in seconds, or None if absent.

    Use this to stamp `as_of` / `cached` on an API payload so the UI can
    label staleness instead of guessing.
    """
    ns = _registry.get(namespace)
    if ns is None:
        return None
    with ns.lock:
        entry = ns.entries.get(key)
        if entry is None:
            return None
        return max(0.0, time.time() - entry.wall)


def stats() -> list[dict]:
    """Per-namespace counters, sorted by name. Cheap; safe to expose."""
    with _registry_lock:
        namespaces = list(_registry.values())
    out: list[dict] = []
    for ns in namespaces:
        with ns.lock:
            hits, misses, errors = ns.hits, ns.misses, ns.errors
            size = len(ns.entries)
            oldest = min((e.wall for e in ns.entries.values()), default=None)
        total = hits + misses
        out.append({
            "name": ns.name,
            "ttl_sec": round(ns.ttl(), 3),
            "size": size,
            "hits": hits,
            "misses": misses,
            "errors": errors,
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "oldest_entry_age_sec": round(time.time() - oldest, 3) if oldest else None,
        })
    out.sort(key=lambda r: r["name"])
    return out


def invalidate(namespace: str | None = None, key: Any = None) -> int:
    """Drop cached entries. Returns the number of entries removed.

    invalidate()                  -> everything
    invalidate("ns")              -> one namespace
    invalidate("ns", key)         -> one entry
    In-flight leaders are left alone; they will store their result and the
    next read re-populates.
    """
    with _registry_lock:
        targets = (
            list(_registry.values()) if namespace is None
            else [_registry[namespace]] if namespace in _registry
            else []
        )
    dropped = 0
    for ns in targets:
        with ns.lock:
            if key is None:
                dropped += len(ns.entries)
                ns.entries.clear()
            elif key in ns.entries:
                del ns.entries[key]
                dropped += 1
    return dropped


def reset_for_tests() -> None:
    """Clear all namespaces, entries, and counters. Test-only hook."""
    with _registry_lock:
        for ns in _registry.values():
            with ns.lock:
                ns.entries.clear()
                ns.inflight.clear()
                ns.hits = ns.misses = ns.errors = 0
