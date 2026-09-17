"""chain.py — options chain page data.

Slice C1-chain-be: serves the full options chain page with calls + puts
side-by-side per strike. Uses the canonical Schwab client (`get_client`)
via schwab-py — same pattern as signals.py and trace.py.

Public surface:
    get_options_chain(symbol, expiry=None, strike_count=40) -> dict

Response shape (frontend contract):
{
    "symbol": "SPY",
    "spot": 558.40,
    "expiry": "2026-06-20",            # selected expiry (front-month if None)
    "expiries": ["2026-05-30", ...],   # all available expiries
    "strikes": [
        {
            "strike": 555.0,
            "call": {"bid":3.20, "ask":3.30, "last":3.25, "mid":3.25,
                     "delta":0.52, "iv":0.183, "volume":4200, "open_interest":12000,
                     "gamma":0.018, "vega":0.45, "theta":-0.08},
            "put":  {"bid":2.80, "ask":2.90, "last":2.85, "mid":2.85,
                     "delta":-0.48, "iv":0.205, ...},
        },
        ...
    ],
    "atm_strike": 558.0,                # closest strike to spot
}
"""
from __future__ import annotations

from typing import Any

import cache as _cache_mod
from sources.schwab import get_client

# ---------------------------------------------------------------------------
# TTL cache for Schwab option-chain fetches.
# Originally DISPERSION-CHAIN-CACHE-1 (bespoke dict + Lock); migrated onto the
# shared primitive in cache.py by FOUNDATION-CACHE-1. Behaviour is unchanged
# except that concurrent cold-key callers now single-flight instead of each
# hitting Schwab, and the TTL is resolved per-call instead of at import.
# Key: (symbol_upper, expiry, strike_count)
# ---------------------------------------------------------------------------

CACHE_NS = "chain.options_chain"


def _chain_ttl() -> float:
    try:
        from settings import settings  # type: ignore
        return float(settings.schwab_chain_cache_ttl_sec)
    except Exception:  # noqa: BLE001
        return _cache_mod.TTL_CHAIN


def get_options_chain_cached(
    symbol: str,
    expiry: str | None = None,
    strike_count: int = 40,
    *,
    _chain_fn=None,
) -> dict[str, Any]:
    """TTL-cached wrapper around get_options_chain.

    Cache miss or expired entry → calls underlying and stores result.
    Cache hit within TTL → returns cached dict without calling Schwab.
    Concurrent misses on the same key → one Schwab call, shared result.
    _chain_fn: DI seam for tests; excluded from the cache key on purpose.
    """
    fn = _chain_fn if _chain_fn is not None else get_options_chain
    key = (symbol.upper(), expiry, strike_count)
    return _cache_mod.get_or_call(
        CACHE_NS,
        key,
        lambda: fn(symbol, expiry=expiry, strike_count=strike_count),
        ttl_sec=_chain_ttl,
    )


def get_cache_stats() -> dict[str, Any]:
    """Return {hits, misses, size, ttl_sec} for the chain TTL cache.

    Shape preserved for existing callers; the numbers now come from the
    shared cache registry.
    """
    for row in _cache_mod.stats():
        if row["name"] == CACHE_NS:
            return {"hits": row["hits"], "misses": row["misses"],
                    "size": row["size"], "ttl_sec": row["ttl_sec"]}
    return {"hits": 0, "misses": 0, "size": 0, "ttl_sec": _chain_ttl()}


def reset_cache_for_tests() -> None:
    """Clear chain cache state. Test-only hook."""
    _cache_mod.reset_for_tests()


def _safe(v, d=0.0) -> float:
    """Convert Schwab's mixed None/NaN-ish numerics to clean floats."""
    if v is None:
        return d
    try:
        f = float(v)
    except (TypeError, ValueError):
        return d
    # Schwab uses -999.0 / -999 as a sentinel for missing greeks/IV
    if f <= -999.0:
        return d
    return f


def _leg_dto(leg: dict[str, Any]) -> dict[str, Any]:
    """Map one raw Schwab leg dict to our flat per-side DTO."""
    bid = _safe(leg.get("bid"))
    ask = _safe(leg.get("ask"))
    mid = round((bid + ask) / 2.0, 4) if (bid and ask) else _safe(leg.get("last"))
    return {
        "bid": bid,
        "ask": ask,
        "last": _safe(leg.get("last")),
        "mid": mid,
        "volume": int(_safe(leg.get("totalVolume"))),
        "open_interest": int(_safe(leg.get("openInterest"))),
        "delta": _safe(leg.get("delta")),
        "gamma": _safe(leg.get("gamma")),
        "theta": _safe(leg.get("theta")),
        "vega": _safe(leg.get("vega")),
        "iv": _safe(leg.get("volatility")) / 100.0,  # Schwab returns IV as percent
    }


def get_options_chain(
    symbol: str,
    expiry: str | None = None,
    strike_count: int = 40,
) -> dict[str, Any]:
    """Build the options chain DTO for `symbol`.

    Args:
        symbol: underlying (e.g. "SPY", "SPX", "$SPX.X" — passed through to Schwab).
        expiry: ISO date "YYYY-MM-DD" to filter to one expiry; None → front-month.
        strike_count: ±count strikes around ATM (Schwab honors this server-side).
    """
    client = get_client()
    resp = client.get_option_chain(
        symbol,
        contract_type=client.Options.ContractType.ALL,
        strike_count=strike_count,
        include_underlying_quote=True,
    )
    resp.raise_for_status()
    body = resp.json() or {}

    underlying = body.get("underlying") or {}
    spot = _safe(underlying.get("last") or body.get("underlyingPrice"))

    call_map = body.get("callExpDateMap", {}) or {}
    put_map = body.get("putExpDateMap", {}) or {}

    # All available expiries — Schwab keys are "YYYY-MM-DD:DTE"; strip the DTE suffix
    all_expiries_raw = sorted(set(call_map.keys()) | set(put_map.keys()))
    all_expiries = [k.split(":", 1)[0] for k in all_expiries_raw]

    # Choose the target expiry — first matching prefix, else front-month
    if expiry:
        match = [k for k in all_expiries_raw if k.startswith(expiry)]
        target_raw = match[0] if match else (all_expiries_raw[0] if all_expiries_raw else None)
    else:
        target_raw = all_expiries_raw[0] if all_expiries_raw else None

    if not target_raw:
        return {
            "symbol": symbol.upper(),
            "spot": spot,
            "expiry": None,
            "expiries": all_expiries,
            "strikes": [],
            "atm_strike": None,
        }

    selected_expiry = target_raw.split(":", 1)[0]
    call_strikes = call_map.get(target_raw, {}) or {}
    put_strikes = put_map.get(target_raw, {}) or {}

    all_strikes = sorted(
        {float(s) for s in call_strikes.keys()} | {float(s) for s in put_strikes.keys()}
    )

    rows: list[dict[str, Any]] = []
    for strike in all_strikes:
        key = f"{strike:.1f}" if f"{strike:.1f}" in call_strikes or f"{strike:.1f}" in put_strikes else f"{strike}"
        # Schwab keys are typically "5800.0" — match by float lookup
        c_list = (
            call_strikes.get(f"{strike}")
            or call_strikes.get(f"{strike:.1f}")
            or call_strikes.get(f"{strike:.2f}")
            or []
        )
        p_list = (
            put_strikes.get(f"{strike}")
            or put_strikes.get(f"{strike:.1f}")
            or put_strikes.get(f"{strike:.2f}")
            or []
        )
        c_leg = c_list[0] if c_list else None
        p_leg = p_list[0] if p_list else None

        rows.append({
            "strike": strike,
            "call": _leg_dto(c_leg) if c_leg else None,
            "put": _leg_dto(p_leg) if p_leg else None,
        })

    atm_strike = (
        min(all_strikes, key=lambda s: abs(s - spot)) if all_strikes and spot else None
    )

    return {
        "symbol": symbol.upper(),
        "spot": spot,
        "expiry": selected_expiry,
        "expiries": all_expiries,
        "strikes": rows,
        "atm_strike": atm_strike,
    }


def get_chain_for_dte_band(
    symbol: str,
    dte_min: int = 0,
    dte_max: int = 365,
    strike_count: int = 40,
) -> dict[str, Any]:
    """Slice SCAN-2 support: return ALL legs across expiries in [dte_min, dte_max].

    Returns:
        {
          "symbol": str, "spot": float,
          "legs": [
              {"side":"Call"|"Put", "expiry":"YYYY-MM-DD", "dte":int,
               "strike":float, **leg_dto_fields},
              ...
          ]
        }

    Where leg_dto_fields are the same shape produced by _leg_dto (bid/ask/mid/
    last/volume/open_interest/delta/gamma/theta/vega/iv).
    """
    import datetime as _dt
    client = get_client()
    resp = client.get_option_chain(
        symbol,
        contract_type=client.Options.ContractType.ALL,
        strike_count=strike_count,
        include_underlying_quote=True,
    )
    resp.raise_for_status()
    body = resp.json() or {}

    underlying = body.get("underlying") or {}
    spot = _safe(underlying.get("last") or body.get("underlyingPrice"))
    today = _dt.date.today()

    legs: list[dict[str, Any]] = []

    def _walk(side_label: str, exp_map: dict[str, Any]):
        for raw_key, strikes_map in (exp_map or {}).items():
            expiry = raw_key.split(":", 1)[0]
            try:
                exp_date = _dt.date.fromisoformat(expiry)
            except ValueError:
                continue
            dte = (exp_date - today).days
            if dte < dte_min or dte > dte_max:
                continue
            for strike_str, leg_list in (strikes_map or {}).items():
                try:
                    strike = float(strike_str)
                except (TypeError, ValueError):
                    continue
                if not leg_list:
                    continue
                leg = _leg_dto(leg_list[0])
                leg.update({
                    "side": side_label,
                    "expiry": expiry,
                    "dte": dte,
                    "strike": strike,
                })
                legs.append(leg)

    _walk("Call", body.get("callExpDateMap") or {})
    _walk("Put", body.get("putExpDateMap") or {})

    return {"symbol": symbol.upper(), "spot": spot, "legs": legs}
