"""Tests for chain_snapshots.py (FOUNDATION-CHAIN-SNAPSHOT-1).

All offline: Schwab is a fake client object injected at the module boundary,
the NYSE calendar is the real pandas-market-calendars data (local, no
network), and the store root is redirected to tmp_path per test. Fixed,
known dates are used for schedule tests so results never depend on when the
suite runs:

    2026-08-06 — a Thursday, normal NYSE session (9:30–16:00 ET, EDT=UTC-4)
    2026-08-08 — a Saturday (closed)
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import chain_snapshots as snap
import storage

# ---------- fixed clocks (see module docstring) ------------------------------

RTH = datetime(2026, 8, 6, 15, 0, tzinfo=UTC)  # Thu 11:00 ET — mid-session
PRE_MARKET = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)  # Thu 08:00 ET
NEAR_CLOSE = datetime(2026, 8, 6, 19, 50, tzinfo=UTC)  # Thu 15:50 ET
SATURDAY = datetime(2026, 8, 8, 15, 0, tzinfo=UTC)  # closed


# ---------- fake Schwab client ------------------------------------------------


class _FakeResp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _FakeClient:
    """Duck-types the retrying proxy: get_option_chain + Options enum attrs."""

    class Options:
        class ContractType:
            ALL = "ALL"

    def __init__(self, bodies: dict):
        self._bodies = bodies
        self.calls: list[str] = []

    def get_option_chain(self, symbol, **_kw):
        self.calls.append(symbol)
        body = self._bodies[symbol]
        if isinstance(body, Exception):
            raise body
        return _FakeResp(body)


def _leg(bid=1.0, ask=1.2, last=1.1, **extra):
    leg = {
        "bid": bid,
        "ask": ask,
        "last": last,
        "bidSize": 10,
        "askSize": 12,
        "totalVolume": 100,
        "openInterest": 500,
        "delta": 0.5,
        "gamma": 0.01,
        "theta": -0.05,
        "vega": 0.1,
        "rho": 0.02,
        "volatility": 22.5,
        "quoteTimeInLong": 1786388400000,
    }
    leg.update(extra)
    return leg


def _chain_body(spot=500.0, expiries=("2026-08-21:15",)):
    """Minimal Schwab chain body: one call + one put per expiry."""
    calls, puts = {}, {}
    for key in expiries:
        calls[key] = {"500.0": [_leg()]}
        puts[key] = {"500.0": [_leg(delta=-0.5)]}
    return {
        "underlying": {"last": spot},
        "callExpDateMap": calls,
        "putExpDateMap": puts,
    }


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Redirect the snapshot root to tmp_path and init the (clean) test DB."""
    monkeypatch.setattr(snap, "SNAPSHOT_ROOT", tmp_path / "chain_snapshots")
    storage.init_db()
    return tmp_path / "chain_snapshots"


# ---------- write path --------------------------------------------------------


def test_write_once_guard_raises(store):
    cl = _FakeClient({"SPY": _chain_body()})
    path, n = snap.capture_symbol("SPY", kind="manual", ts_utc=RTH, client=cl)
    assert path.exists() and n == 2
    before = path.read_bytes()
    with pytest.raises(snap.SnapshotExistsError):
        snap.capture_symbol("SPY", kind="manual", ts_utc=RTH, client=cl)
    assert path.read_bytes() == before  # file untouched
    assert len(cl.calls) == 1  # duplicate never spent a Schwab request


def test_capture_all_writes_files_and_manifest(store):
    syms = ["SPY", "QQQ", "IWM"]
    cl = _FakeClient({s: _chain_body() for s in syms})
    run_id = snap.capture_all("manual", now=RTH, client=cl, symbols=syms)
    run = storage.last_snapshot_run()
    assert run["run_id"] == run_id
    assert run["capture_kind"] == "manual"
    assert run["symbols_ok"] == syms
    assert run["symbols_failed"] == []
    assert run["rows_written"] == 6  # 2 rows per symbol
    assert len(run["files_written"]) == 3
    for f in run["files_written"]:
        assert Path(f).exists()
    assert run["finished_at"] is not None


def test_per_symbol_failure_isolation(store):
    boom = RuntimeError("token expired: refresh failed with 401")
    cl = _FakeClient(
        {
            "SPY": _chain_body(),
            "QQQ": boom,
            "IWM": _chain_body(),
            "$SPX.X": _chain_body(),  # SPX fetches its wire form
        }
    )
    snap.capture_all("manual", now=RTH, client=cl, symbols=["SPY", "QQQ", "IWM", "SPX"])
    run = storage.last_snapshot_run()
    assert run["symbols_ok"] == ["SPY", "IWM", "SPX"]  # run continued past QQQ
    assert len(run["files_written"]) == 3
    assert run["symbols_failed"] == [
        {"symbol": "QQQ", "error": "token expired: refresh failed with 401"}
    ]  # error recorded verbatim


def test_row_schema_v1(store):
    cl = _FakeClient({"SPY": _chain_body()})
    path, _ = snap.capture_symbol("SPY", kind="near_close", ts_utc=NEAR_CLOSE, client=cl)
    df = pd.read_parquet(path)
    assert list(df.columns) == snap.SCHEMA_V1_COLUMNS
    assert set(df["right"]) == {"C", "P"}
    assert (df["schema_version"] == 1).all()
    assert (df["capture_kind"] == "near_close").all()
    assert df["iv"].tolist() == pytest.approx([0.225, 0.225])  # Schwab percent → decimal
    assert (df["spot"] == 500.0).all()
    assert (df["dte"] == 15).all()  # 2026-08-21 minus ET session date 2026-08-06


def test_max_dte_filter(store):
    body = _chain_body(expiries=("2026-08-21:15", "2027-08-06:365"))
    cl = _FakeClient({"SPY": body})
    path, n = snap.capture_symbol("SPY", kind="manual", ts_utc=RTH, client=cl, max_dte=120)
    assert n == 2  # only the 15-DTE expiry survives the 120-DTE cap
    df = pd.read_parquet(path)
    assert set(df["expiry"]) == {"2026-08-21"}


def test_symbol_mapping_and_store_dir(store):
    cl = _FakeClient({"$SPX.X": _chain_body(spot=6400.0)})
    path, _ = snap.capture_symbol("SPX", kind="manual", ts_utc=RTH, client=cl)
    assert cl.calls == ["$SPX.X"]  # Schwab wire form at the fetcher boundary
    assert path.parent.parent.name == "SPX"  # config symbol names the directory
    assert path.parent.name == "2026-08-06"  # UTC date component
    assert path.name == "150000Z.parquet"  # UTC time component


def test_path_traversal_symbol_rejected(store):
    cl = _FakeClient({})
    for bad in ("../evil", "SPY/..", "a b", "$SPX.X", ""):
        with pytest.raises(snap.SnapshotError):
            snap.capture_symbol(bad, kind="manual", ts_utc=RTH, client=cl)
    assert cl.calls == []  # rejected before any network call


def test_naive_now_rejected(store):
    cl = _FakeClient({"SPY": _chain_body()})
    naive = datetime(2026, 8, 6, 15, 0)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        snap.capture_symbol("SPY", kind="manual", ts_utc=naive, client=cl)
    with pytest.raises(ValueError, match="timezone-aware"):
        snap.capture_all("manual", now=naive, client=cl, symbols=["SPY"])


def test_read_snapshots_naive_bounds_rejected(store):
    with pytest.raises(ValueError, match="timezone-aware"):
        snap.read_snapshots("SPY", datetime(2026, 8, 1), datetime(2026, 8, 7, tzinfo=UTC))
    with pytest.raises(ValueError, match="timezone-aware"):
        snap.read_snapshots("SPY", datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 8, 7))


# ---------- read path ---------------------------------------------------------


def test_empty_store_returns_schema_columns(store):
    df = snap.read_snapshots(
        "SPY", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 12, 31, tzinfo=UTC)
    )
    assert df.empty
    assert list(df.columns) == snap.SCHEMA_V1_COLUMNS


def test_read_snapshots_date_window(store):
    cl = _FakeClient({"SPY": _chain_body()})
    day1 = datetime(2026, 8, 5, 15, 0, tzinfo=UTC)
    day2 = datetime(2026, 8, 6, 15, 0, tzinfo=UTC)
    snap.capture_symbol("SPY", kind="manual", ts_utc=day1, client=cl)
    snap.capture_symbol("SPY", kind="manual", ts_utc=day2, client=cl)

    both = snap.read_snapshots("SPY", day1, day2)
    assert len(both) == 4
    only_day2 = snap.read_snapshots("SPY", datetime(2026, 8, 6, 0, 0, tzinfo=UTC), day2)
    assert len(only_day2) == 2
    assert (pd.to_datetime(only_day2["snapshot_ts_utc"], utc=True) == day2).all()


# ---------- scheduler gating --------------------------------------------------


def test_due_capture_rth_weekend_and_premarket():
    # Saturday: never, even with nothing captured yet
    assert (
        snap.due_capture(SATURDAY, last_capture_utc=None, near_close_done=False, interval_min=30)
        is None
    )
    # Pre-market on a trading day: no interval capture
    assert (
        snap.due_capture(PRE_MARKET, last_capture_utc=None, near_close_done=False, interval_min=30)
        is None
    )
    # Mid-session, store virgin: capture now
    assert (
        snap.due_capture(RTH, last_capture_utc=None, near_close_done=False, interval_min=30)
        == "interval"
    )


def test_due_capture_interval_gating(store):
    recent = datetime(2026, 8, 6, 14, 45, tzinfo=UTC)  # 15 min before RTH clock
    stale = datetime(2026, 8, 6, 14, 20, tzinfo=UTC)  # 40 min before
    assert (
        snap.due_capture(RTH, last_capture_utc=recent, near_close_done=False, interval_min=30)
        is None
    )
    assert (
        snap.due_capture(RTH, last_capture_utc=stale, near_close_done=False, interval_min=30)
        == "interval"
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        snap.due_capture(
            RTH,
            last_capture_utc=datetime(2026, 8, 6, 14, 20),
            near_close_done=False,
            interval_min=30,
        )


def test_due_capture_near_close_once_per_day():
    five_min_ago = datetime(2026, 8, 6, 19, 45, tzinfo=UTC)
    # In-window, not yet done: near_close wins even though interval hasn't elapsed
    assert (
        snap.due_capture(
            NEAR_CLOSE, last_capture_utc=five_min_ago, near_close_done=False, interval_min=30
        )
        == "near_close"
    )
    # Already done today: falls through to interval gating → not due
    assert (
        snap.due_capture(
            NEAR_CLOSE, last_capture_utc=five_min_ago, near_close_done=True, interval_min=30
        )
        is None
    )
    # Window bounds: 15:44 ET is too early, 16:00 ET is past both window and RTH
    before = datetime(2026, 8, 6, 19, 44, tzinfo=UTC)
    after = datetime(2026, 8, 6, 20, 0, tzinfo=UTC)
    assert (
        snap.due_capture(
            before, last_capture_utc=five_min_ago, near_close_done=False, interval_min=30
        )
        is None
    )
    assert (
        snap.due_capture(
            after, last_capture_utc=five_min_ago, near_close_done=False, interval_min=30
        )
        is None
    )


# ---------- status endpoint ---------------------------------------------------


# ---------- SNAPSHOT-CONFIG-1: runtime symbol config ---------------------------


@pytest.fixture
def cfg_db(tmp_path, monkeypatch):
    """Isolated DB so app_config writes cannot leak across tests."""
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "cfg.db"))
    monkeypatch.setattr(snap, "SNAPSHOT_ROOT", tmp_path / "chain_snapshots")
    storage.init_db()


def test_resolver_precedence_db_beats_env(cfg_db, monkeypatch):
    import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "snapshot_symbols_raw", "SPY,QQQ")
    assert snap.snapshot_symbols() == (["SPY", "QQQ"], "env_or_default", None)
    storage.set_config(snap.CONFIG_KEY, ["GLD", "TLT"])
    assert snap.snapshot_symbols() == (["GLD", "TLT"], "db", None)


def test_default_list_is_full_karsan_set(cfg_db):
    syms, source, _universe = snap.snapshot_symbols()
    assert syms == ["SPY", "QQQ", "IWM", "DIA", "SPX", "VIX",
                    "GLD", "USO", "TLT", "HYG"]
    assert source == "env_or_default"


def test_status_reflects_db_config(cfg_db):
    storage.set_config(snap.CONFIG_KEY, [])
    body = snap.status()
    assert body["enabled"] is False and body["symbols"] == []


def test_vix_wire_mapping_and_cleaner_edges(cfg_db):
    assert snap._SCHWAB_TICKER_MAP["VIX"] == "$VIX.X"
    assert snap.clean_snapshot_symbols([" spy", "SPY", "$vix"]) == ["SPY", "VIX"]
    with pytest.raises(ValueError, match="too many"):
        snap.clean_snapshot_symbols([f"S{i}" for i in range(65)])


# ---------- UNIVERSE-CONFIG-1: config can reference a named universe ---------


