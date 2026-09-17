"""Tests for the shared TTL cache (FOUNDATION-CACHE-1).

These assert behaviour, never internals — the whole point of consolidating
three hand-rolled caches was to stop tests from coupling to cache guts.
"""
from __future__ import annotations

import threading
import time

import pytest

import cache


@pytest.fixture(autouse=True)
def _clean_cache():
    cache.reset_for_tests()
    yield
    cache.reset_for_tests()


def _counter_fn(value="v"):
    calls = {"n": 0}

    def fn(*args, **kwargs):
        calls["n"] += 1
        return {"value": value, "call": calls["n"]}

    return fn, calls


class TestTtlCacheBasics:
    def test_miss_then_hit(self):
        """Second call inside the TTL must not touch the underlying function."""
        fn, calls = _counter_fn()
        cached = cache.ttl_cache(30, name="t.basic")(fn)

        first = cached("SPY")
        second = cached("SPY")

        assert calls["n"] == 1
        assert first is second, "a hit returns the same object, not a copy"

    def test_distinct_args_are_distinct_entries(self):
        fn, calls = _counter_fn()
        cached = cache.ttl_cache(30, name="t.args")(fn)

        cached("SPY")
        cached("QQQ")
        cached("SPY", strike_count=10)

        assert calls["n"] == 3
        row = _row("t.args")
        assert row["size"] == 3

    def test_expiry_recomputes(self):
        fn, calls = _counter_fn()
        cached = cache.ttl_cache(0.15, name="t.expiry")(fn)

        cached("SPY")
        time.sleep(0.25)
        cached("SPY")

        assert calls["n"] == 2

    def test_callable_ttl_is_resolved_per_call(self):
        """A settings-driven TTL must be read at call time, not at import.

        chain.py's old `_load_ttl()` snapshotted the setting at import, so
        changing the env var required a restart. Callable TTLs fix that.
        """
        ttl = {"v": 30.0}
        fn, calls = _counter_fn()
        cached = cache.ttl_cache(lambda: ttl["v"], name="t.dynttl")(fn)

        cached("SPY")
        cached("SPY")
        assert calls["n"] == 1

        ttl["v"] = 0.0  # everything is now stale
        cached("SPY")
        assert calls["n"] == 2


class TestSingleFlight:
    def test_concurrent_cold_callers_make_one_upstream_call(self):
        """The property that actually protects the Schwab rate budget.

        12 threads hitting a cold key must produce exactly ONE upstream call.
        Without this, a dashboard tab load fans out to N identical requests.
        """
        calls = {"n": 0}
        lock = threading.Lock()

        def slow():
            with lock:
                calls["n"] += 1
            time.sleep(0.25)
            return {"ok": True}

        cached = cache.ttl_cache(30, name="t.singleflight")(slow)
        results: list[dict] = []
        threads = [
            threading.Thread(target=lambda: results.append(cached()))
            for _ in range(12)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert calls["n"] == 1, f"expected 1 upstream call, got {calls['n']}"
        assert len(results) == 12
        assert all(r is results[0] for r in results), "all waiters share one result"


class TestErrorHandling:
    def test_exceptions_are_not_cached(self):
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise RuntimeError("upstream down")

        cached = cache.ttl_cache(30, name="t.boom")(boom)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cached()

        assert calls["n"] == 3, "a failure must not be cached as a result"

    def test_failure_does_not_resurrect_a_stale_value(self):
        """No silent fallbacks: an error surfaces even when a stale entry exists."""
        state = {"fail": False, "n": 0}

        def flaky():
            state["n"] += 1
            if state["fail"]:
                raise RuntimeError("upstream down")
            return {"good": True}

        cached = cache.ttl_cache(0.1, name="t.flaky")(flaky)
        assert cached() == {"good": True}

        state["fail"] = True
        time.sleep(0.2)  # let the good entry expire
        with pytest.raises(RuntimeError):
            cached()

    def test_error_counter_is_reported(self):
        def boom():
            raise ValueError("nope")

        cached = cache.ttl_cache(30, name="t.errcount")(boom)
        with pytest.raises(ValueError):
            cached()

        assert _row("t.errcount")["errors"] == 1


class TestGuardRails:
    def test_unhashable_argument_raises(self):
        """Silently mis-keying on a mutable arg would return wrong data."""
        cached = cache.ttl_cache(30, name="t.unhashable")(lambda lst: lst)
        with pytest.raises(TypeError, match="unhashable"):
            cached([1, 2, 3])

    def test_key_fn_normalises_unhashable_input(self):
        fn, calls = _counter_fn()
        cached = cache.ttl_cache(
            30,
            name="t.keyfn",
            key_fn=lambda symbols: tuple(sorted(s.upper() for s in symbols)),
        )(fn)

        cached(["SPY", "QQQ"])
        cached(["qqq", "spy"])  # same set, different order/case

        assert calls["n"] == 1, "normalised keys must collapse to one entry"

    def test_async_function_is_rejected(self):
        """Caching a coroutine object looks like a hit then fails on re-await."""
        with pytest.raises(TypeError, match="async"):
            @cache.ttl_cache(30, name="t.async")
            async def _coro():  # pragma: no cover
                return 1


class TestObservability:
    def test_stats_shape_and_counters(self):
        fn, _ = _counter_fn()
        cached = cache.ttl_cache(30, name="t.stats")(fn)
        cached("SPY")
        cached("SPY")

        row = _row("t.stats")
        expected = {"name", "ttl_sec", "size", "hits", "misses", "errors",
                    "hit_rate", "oldest_entry_age_sec"}
        assert expected <= row.keys()
        assert row["hits"] == 1 and row["misses"] == 1 and row["size"] == 1
        assert row["hit_rate"] == 0.5
        assert row["oldest_entry_age_sec"] >= 0

    def test_age_sec_tracks_wall_clock(self):
        cached = cache.ttl_cache(30, name="t.age")(lambda: {"x": 1})
        cached()
        age = cache.age_sec("t.age", ((), ()))
        assert age is not None and 0 <= age < 5

    def test_age_sec_none_for_unknown(self):
        assert cache.age_sec("t.nope", "k") is None

    def test_invalidate_namespace_and_key(self):
        fn, calls = _counter_fn()
        cached = cache.ttl_cache(30, name="t.inval")(fn)
        cached("SPY")
        cached("QQQ")
        assert calls["n"] == 2

        assert cache.invalidate("t.inval", (("SPY",), ())) == 1
        cached("SPY")
        assert calls["n"] == 3
        cached("QQQ")  # still cached
        assert calls["n"] == 3

        assert cache.invalidate("t.inval") == 2
        cached("QQQ")
        assert calls["n"] == 4

    def test_cache_clear_attribute_on_wrapper(self):
        fn, calls = _counter_fn()
        cached = cache.ttl_cache(30, name="t.clearattr")(fn)
        cached("SPY")
        assert cached.cache_namespace == "t.clearattr"
        cached.cache_clear()
        cached("SPY")
        assert calls["n"] == 2

    def test_reset_for_tests_clears_counters(self):
        cached = cache.ttl_cache(30, name="t.reset")(lambda: 1)
        cached()
        cached()
        cache.reset_for_tests()
        row = _row("t.reset")
        assert row["hits"] == 0 and row["misses"] == 0 and row["size"] == 0


def _row(name: str) -> dict:
    matches = [r for r in cache.stats() if r["name"] == name]
    assert matches, f"namespace {name!r} not registered"
    return matches[0]
