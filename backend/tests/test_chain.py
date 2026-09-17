"""Slice C1-chain-be — contract-pin tests for GET /api/options/chain.

Pins the response shape that options.chain.js (slice C2) will depend on:
symbol, spot, expiry, expiries[], strikes[].{strike, call, put}, atm_strike.
Mocks the Schwab option chain layer so tests run offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import chain as chain_mod  # noqa: E402


def _fake_chain_payload() -> dict:
    """Two expiries × two strikes synthetic chain.

    Spot 558.40. Front expiry 2026-05-30:14. Back expiry 2026-06-20:35.
    Strike 555.0: call delta 0.62 / iv 18.3 / vol 4200 / oi 12000;
                  put  delta -0.38 / iv 20.1 / vol 1200 / oi 8400.
    Strike 560.0: call delta 0.45 / iv 19.0 / vol 5800 / oi 15000;
                  put  delta -0.55 / iv 21.4 / vol 3200 / oi 11000.
    Back-month present but only one strike to confirm expiry filter works.
    """
    return {
        "underlying": {"last": 558.40},
        "underlyingPrice": 558.40,
        "callExpDateMap": {
            "2026-05-30:14": {
                "555.0": [{
                    "bid": 4.20, "ask": 4.40, "last": 4.30,
                    "delta": 0.62, "gamma": 0.018, "theta": -0.08, "vega": 0.45,
                    "volatility": 18.3, "totalVolume": 4200, "openInterest": 12000,
                }],
                "560.0": [{
                    "bid": 1.80, "ask": 1.95, "last": 1.88,
                    "delta": 0.45, "gamma": 0.022, "theta": -0.10, "vega": 0.42,
                    "volatility": 19.0, "totalVolume": 5800, "openInterest": 15000,
                }],
            },
            "2026-06-20:35": {
                "555.0": [{
                    "bid": 8.00, "ask": 8.20, "last": 8.10,
                    "delta": 0.58, "gamma": 0.012, "theta": -0.05, "vega": 0.75,
                    "volatility": 19.8, "totalVolume": 1200, "openInterest": 5000,
                }],
            },
        },
        "putExpDateMap": {
            "2026-05-30:14": {
                "555.0": [{
                    "bid": 1.10, "ask": 1.20, "last": 1.15,
                    "delta": -0.38, "gamma": 0.018, "theta": -0.07, "vega": 0.40,
                    "volatility": 20.1, "totalVolume": 1200, "openInterest": 8400,
                }],
                "560.0": [{
                    "bid": 3.40, "ask": 3.55, "last": 3.48,
                    "delta": -0.55, "gamma": 0.022, "theta": -0.09, "vega": 0.45,
                    "volatility": 21.4, "totalVolume": 3200, "openInterest": 11000,
                }],
            },
            "2026-06-20:35": {
                "555.0": [{
                    "bid": 4.80, "ask": 4.95, "last": 4.88,
                    "delta": -0.42, "gamma": 0.012, "theta": -0.04, "vega": 0.70,
                    "volatility": 21.0, "totalVolume": 800, "openInterest": 3200,
                }],
            },
        },
    }


def _fake_chain_empty() -> dict:
    return {"underlying": {"last": 0.0}, "callExpDateMap": {}, "putExpDateMap": {}}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

        class _Opts:
            class ContractType:
                ALL = "ALL"

            class StrikeRange:
                ALL = "ALL"

            class Strategy:
                SINGLE = "SINGLE"

        self.Options = _Opts

    def get_option_chain(self, symbol, **_kwargs):
        return _FakeResponse(self._payload)


def _patch_chain(monkeypatch, payload):
    monkeypatch.setattr(chain_mod, "get_client", lambda: _FakeClient(payload))


# ---------------------------------------------------------------------------
# DISPERSION-CHAIN-CACHE-1 — TTL cache wrapper tests
# ---------------------------------------------------------------------------


import chain as chain_cache_mod  # noqa: E402  (already imported above as chain_mod)


def _make_counter_fn(payload: dict):
    """Return a (fn, counter[]) pair; counter[0] increments on each call."""
    counter = [0]

    def _fn(symbol, expiry=None, strike_count=40):
        counter[0] += 1
        return payload

    return _fn, counter


class TestChainCache:
    """DISPERSION-CHAIN-CACHE-1 — unit tests for get_options_chain_cached."""

    def setup_method(self):
        """Reset cache state before each test."""
        chain_cache_mod.reset_cache_for_tests()

    def test_CACHE_1_miss_calls_underlying_and_returns_result(self):
        """CACHE-1: cache miss → underlying fn called, result returned correctly."""
        payload = {"symbol": "AAPL", "spot": 100.0, "expiry": "2026-06-20",
                   "expiries": ["2026-06-20"], "strikes": [], "atm_strike": 100.0}
        fn, counter = _make_counter_fn(payload)

        result = chain_cache_mod.get_options_chain_cached(
            "AAPL", expiry="2026-06-20", strike_count=4, _chain_fn=fn
        )

        assert counter[0] == 1, "underlying fn should be called exactly once on miss"
        assert result["symbol"] == "AAPL"
        assert result["spot"] == 100.0
        stats = chain_cache_mod.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0
        assert stats["size"] == 1

    def test_CACHE_2_hit_within_ttl_does_not_call_underlying(self):
        """CACHE-2: second call within TTL → underlying fn NOT called again."""
        payload = {"symbol": "MSFT", "spot": 200.0, "expiry": None,
                   "expiries": [], "strikes": [], "atm_strike": None}
        fn, counter = _make_counter_fn(payload)

        # First call — cache miss
        r1 = chain_cache_mod.get_options_chain_cached(
            "MSFT", expiry=None, strike_count=8, _chain_fn=fn
        )
        # Second call — should be a cache hit (TTL is 30s; no time has elapsed)
        r2 = chain_cache_mod.get_options_chain_cached(
            "MSFT", expiry=None, strike_count=8, _chain_fn=fn
        )

        assert counter[0] == 1, "underlying fn must only be called once across both calls"
        assert r1 is r2, "both calls must return the same cached object"
        stats = chain_cache_mod.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_CACHE_3_expiry_after_ttl_calls_underlying_again(self, monkeypatch):
        """CACHE-3: entry older than TTL → underlying fn called again on next request.

        FOUNDATION-CACHE-1 rewrote this to drive expiry through the real TTL
        setting instead of reaching into cache internals. The old version
        backdated `chain._cache[key]['ts']` directly, which coupled the test to
        one specific cache implementation — exactly the coupling that made the
        three hand-rolled caches expensive to consolidate.
        """
        from settings import settings

        payload = {"symbol": "NVDA", "spot": 500.0, "expiry": "2026-07-18",
                   "expiries": ["2026-07-18"], "strikes": [], "atm_strike": 500.0}
        fn, counter = _make_counter_fn(payload)

        # First call — populates the cache.
        chain_cache_mod.get_options_chain_cached(
            "NVDA", expiry="2026-07-18", strike_count=4, _chain_fn=fn
        )
        assert counter[0] == 1

        # Drop the TTL to zero. The chain namespace resolves its TTL per call,
        # so the existing entry is now unconditionally stale.
        monkeypatch.setattr(settings, "schwab_chain_cache_ttl_sec", 0)

        chain_cache_mod.get_options_chain_cached(
            "NVDA", expiry="2026-07-18", strike_count=4, _chain_fn=fn
        )
        assert counter[0] == 2, "underlying fn must be called again after TTL expiry"
        stats = chain_cache_mod.get_cache_stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 0

    def test_CACHE_4_stats_shape_and_reset(self):
        """CACHE-4: stats shape correct; reset_cache_for_tests clears everything."""
        payload = {"symbol": "AMD", "spot": 80.0, "expiry": None,
                   "expiries": [], "strikes": [], "atm_strike": None}
        fn, _ = _make_counter_fn(payload)
        chain_cache_mod.get_options_chain_cached("AMD", _chain_fn=fn)
        chain_cache_mod.get_options_chain_cached("AMD", _chain_fn=fn)  # hit

        stats = chain_cache_mod.get_cache_stats()
        assert {"hits", "misses", "size", "ttl_sec"} <= stats.keys()
        assert stats["hits"] == 1 and stats["misses"] == 1 and stats["size"] == 1
        assert stats["ttl_sec"] > 0

        chain_cache_mod.reset_cache_for_tests()
        after = chain_cache_mod.get_cache_stats()
        assert after["hits"] == 0 and after["misses"] == 0 and after["size"] == 0


# ---------------------------------------------------------------------------
# CHAIN-PAGE-1 -- proves the HTTP route itself is now cache-wired.
#
# The tests above (T_CHAIN_a-d) hit the route but each issues exactly one
# request, so they never would have caught the bug this slice fixes: the
# route was calling chain_mod.get_options_chain directly, bypassing
# get_options_chain_cached entirely, even though that wrapper (and its own
# unit tests, TestChainCache above) already existed. A unit test against
# get_options_chain_cached in isolation cannot detect "the route doesn't
# call this function" -- only a route-level counting test can.
# ---------------------------------------------------------------------------


class _CountingFakeClient(_FakeClient):
    """Same as _FakeClient, but counts get_option_chain calls."""

    def __init__(self, payload, counter):
        super().__init__(payload)
        self._counter = counter

    def get_option_chain(self, symbol, **kwargs):
        self._counter[0] += 1
        return super().get_option_chain(symbol, **kwargs)


