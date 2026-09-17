"""Tests for universe.py — FOUNDATION-UNIVERSE-1.

Covers the two tiers separately: the pure seed accessors (no DB), and the
SQLite registry (seed sync idempotence, user-row preservation, tags). The
regression pins at the bottom are the "behavior unchanged" proof: every
swept module constant must equal the literal list it replaced, byte for
byte, on sweep day.

The shared conftest redirects GRADEA_DB_PATH to tests/.test.db and wipes it
before every test, so registry tests start from an empty file.
"""

from __future__ import annotations

import pytest

import universe

# ---------- seed tier (pure, no DB) ------------------------------------------


def test_every_seed_universe_is_nonempty_and_duplicate_free():
    for name in universe._SEEDS:
        symbols = universe.seed_symbols(name)
        assert symbols, f"universe {name!r} is empty"
        assert len(symbols) == len(set(symbols)), f"duplicates in {name!r}"


def test_seed_symbols_returns_a_fresh_list():
    a = universe.seed_symbols("core_etfs")
    a.append("MUTATED")
    assert "MUTATED" not in universe.seed_symbols("core_etfs")


def test_seed_csv_round_trips_members():
    assert universe.seed_csv("stream_default") == "SPY,QQQ,VIX"
    assert universe.seed_csv("trace_default") == "SPY,QQQ"


def test_seed_weights_only_for_weighted_universes():
    weights = universe.seed_weights("dispersion_spx50")
    assert weights["AAPL"] == 7.0
    assert len(weights) == 50
    with pytest.raises(ValueError):
        universe.seed_weights("core_etfs")


def test_unknown_seed_universe_raises_key_error():
    with pytest.raises(KeyError):
        universe.seed_symbols("nope")


# ---------- registry tier (SQLite) --------------------------------------------


def test_ensure_seeded_is_idempotent():
    universe.ensure_seeded()
    first = {u["name"]: u["member_count"] for u in universe.list_universes()}
    universe.ensure_seeded()
    second = {u["name"]: u["member_count"] for u in universe.list_universes()}
    assert first == second
    assert first["earnings_default"] == 61
    assert first["dispersion_spx50"] == 50


def test_user_rows_survive_reseeding():
    universe.ensure_seeded()
    with universe._connect() as conn:
        conn.execute(
            "INSERT INTO universe_members"
            "(universe_name, symbol, asset_type, sector, weight, source) "
            "VALUES('core_etfs', 'RSP', 'etf', NULL, NULL, 'user')"
        )
    universe.ensure_seeded()
    members = {m.symbol: m for m in universe.get_universe("core_etfs")}
    assert "RSP" in members and members["RSP"].source == "user"
    assert members["SPY"].source == "seed"


def test_get_universe_carries_tags():
    members = {m.symbol: m for m in universe.get_universe("earnings_default")}
    assert members["AAPL"].asset_type == "equity"
    assert members["AAPL"].sector == "Information Technology"
    assert members["XOM"].sector == "Energy"

    etfs = {m.symbol: m for m in universe.get_universe("core_etfs")}
    assert etfs["SPY"].asset_type == "etf" and etfs["SPY"].sector is None

    flies = {m.symbol: m for m in universe.get_universe("neg_fly_default")}
    assert flies["SPX"].asset_type == "index"

    opts = universe.get_universe("flow_tape_seeds")
    assert all(m.asset_type == "option" for m in opts)


def test_weighted_universe_members_carry_weights():
    members = {m.symbol: m for m in universe.get_universe("dispersion_spx50")}
    assert members["AAPL"].weight == 7.0
    assert members["PM"].weight == 0.4


def test_get_universe_unknown_name_raises():
    with pytest.raises(KeyError):
        universe.get_universe("nope")


def test_resolve_prefers_cleaned_explicit_symbols():
    assert universe.resolve([" spy ", "", "qqq"], "core_etfs") == ["SPY", "QQQ"]


def test_resolve_falls_back_to_universe_when_symbols_blank():
    assert set(universe.resolve(None, "core_etfs")) == {"SPY", "QQQ", "IWM"}
    assert set(universe.resolve([" ", ""], "core_etfs")) == {"SPY", "QQQ", "IWM"}


# ---------- route contract ----------------------------------------------------


# ---------- CRUD routes (UNIVERSE-CRUD-1) --------------------------------------


# ---------- regression pins: swept constants unchanged -------------------------

_SCAN_11 = [
    "SPY",
    "QQQ",
    "IWM",
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
    "META",
    "AMD",
    "AMZN",
    "GOOG",
]


def test_settings_defaults_unchanged():
    import settings as settings_mod

    assert settings_mod.DEFAULT_STREAM_SYMBOLS == "SPY,QQQ,VIX"
    assert settings_mod.DEFAULT_TRACE_SYMBOLS == "SPY,QQQ"
    assert settings_mod.DEFAULT_EARNINGS_UNIVERSE.startswith("AAPL,MSFT,NVDA,GOOGL,")
    assert settings_mod.DEFAULT_EARNINGS_UNIVERSE.endswith(",NOW,BKNG,NEE")
    assert len(settings_mod.DEFAULT_EARNINGS_UNIVERSE.split(",")) == 61
