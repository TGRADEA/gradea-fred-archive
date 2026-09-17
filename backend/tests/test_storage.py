"""Tests for the SQLite-backed storage layer.

conftest points GRADEA_DB_PATH at a per-session temp file so these tests
don't touch the real gradea.db.
"""
from __future__ import annotations

import storage


def test_init_db_is_idempotent():
    storage.init_db()
    storage.init_db()  # second call must not raise


def test_upsert_signal_inserts_then_updates():
    storage.init_db()
    sid = storage.upsert_signal(
        id=None,
        name="rsi-oversold",
        universe=["AAPL", "MSFT"],
        rule="rsi(14) < 30",
        lookback=60,
        enabled=True,
    )
    assert isinstance(sid, int) and sid > 0

    # Update in place — same id, new name.
    sid2 = storage.upsert_signal(
        id=sid,
        name="rsi-oversold-renamed",
        universe=["AAPL"],
        rule="rsi(14) < 25",
    )
    assert sid2 == sid

    got = storage.get_signal(sid)
    assert got is not None
    assert got["name"] == "rsi-oversold-renamed"
    assert got["universe"] == ["AAPL"]


def test_list_and_delete_signals_round_trip():
    storage.init_db()
    sid = storage.upsert_signal(
        id=None, name="tmp", universe=["SPY"], rule="close > 0",
    )
    rows = storage.list_signals()
    assert any(r["id"] == sid for r in rows)
    storage.delete_signal(sid)
    assert storage.get_signal(sid) is None


def test_recent_backtests_returns_inserted_run():
    storage.init_db()
    run_id = storage.add_backtest_run(
        name="ma-cross",
        rule="close > sma(close, 21)",
        universe=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        stats={"trades": 12, "win_rate": 0.58},
    )
    assert run_id > 0
    recent = storage.recent_backtests(5)
    assert any(r["id"] == run_id for r in recent)
