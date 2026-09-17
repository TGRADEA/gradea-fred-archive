"""storage.py — SQLite-backed persistence for signals & backtests.

Tables:
  signals          — user-defined signal definitions
  signal_evals     — most-recent evaluation per signal
  backtest_runs    — completed backtest result rows (JSON blob of stats)
  chain_snapshot_runs — manifest of option-chain snapshot captures
                     (FOUNDATION-CHAIN-SNAPSHOT-1; see chain_snapshots.py)
  universes        — named symbol universes (FOUNDATION-UNIVERSE-1; seeds
  universe_members    and accessors live in universe.py)

Why SQLite: single-file, no daemon, fits the "self-hosted, friends/family"
deployment model. File lives at $GRADEA_DATA_DIR/gradea.db (defaults to
project root next to tokens.json — same convention).
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from settings import settings as _settings

DB_PATH = str(_settings.db_path)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    universe    TEXT NOT NULL,        -- JSON list of symbols
    rule        TEXT NOT NULL,        -- expression string evaluated by rules.py
    lookback    INTEGER NOT NULL DEFAULT 60,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS signal_evals (
    signal_id    INTEGER PRIMARY KEY,
    last_run_at  REAL NOT NULL,
    matches      TEXT NOT NULL,       -- JSON list of {symbol, value, ts}
    error        TEXT,
    FOREIGN KEY(signal_id) REFERENCES signals(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    rule        TEXT NOT NULL,
    universe    TEXT NOT NULL,
    start_date  TEXT NOT NULL,
    end_date    TEXT NOT NULL,
    stats       TEXT NOT NULL,         -- JSON stats blob
    created_at  REAL NOT NULL,
    -- FOUNDATION-STRATEGY-REGISTRY-1: id of the registered strategy
    -- (signals row) this run executed, NULL for ad-hoc runs. Plain
    -- INTEGER (no FK) so the ALTER-migrated schema and this one are
    -- identical — SQLite cannot add FK constraints via ALTER TABLE.
    strategy_id INTEGER,
    -- FOUNDATION-PIT-DATA-1 (BT-F5): JSON manifest of the exact bars the
    -- run consumed — per-symbol first_ts/last_ts/n_bars/sha256/source plus
    -- captured_at/period_years/engine/replay_of. NULL for rows created
    -- before the slice. The bars themselves live in write-once .npz
    -- snapshots under data/backtests/{id}/ (see bar_snapshots.py).
    data_manifest TEXT
);

CREATE INDEX IF NOT EXISTS idx_backtest_created ON backtest_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS chain_snapshot_runs (
    run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        REAL NOT NULL,
    finished_at       REAL,
    capture_kind      TEXT NOT NULL,   -- 'interval' | 'near_close' | 'manual'
    symbols_requested TEXT NOT NULL,   -- JSON list of config symbols
    symbols_ok        TEXT NOT NULL DEFAULT '[]',   -- JSON list
    symbols_failed    TEXT NOT NULL DEFAULT '[]',   -- JSON list of {symbol, error}
    rows_written      INTEGER NOT NULL DEFAULT 0,
    files_written     TEXT NOT NULL DEFAULT '[]'    -- JSON list of paths
);

CREATE INDEX IF NOT EXISTS idx_snapshot_runs_started ON chain_snapshot_runs(started_at DESC);

CREATE TABLE IF NOT EXISTS universes (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    kind        TEXT NOT NULL    -- 'symbols' | 'weighted' | 'option_contracts'
);

CREATE TABLE IF NOT EXISTS universe_members (
    universe_name TEXT NOT NULL REFERENCES universes(name),
    symbol        TEXT NOT NULL,
    asset_type    TEXT NOT NULL,           -- 'equity' | 'etf' | 'index' | 'option'
    sector        TEXT,                    -- GICS sector; NULL for etf/index/option
    weight        REAL,                    -- weighted universes only
    source        TEXT NOT NULL DEFAULT 'seed',  -- 'seed' | 'user'
    PRIMARY KEY (universe_name, symbol)
);

CREATE TABLE IF NOT EXISTS app_config (
    key        TEXT PRIMARY KEY,   -- e.g. 'karsan.symbols'
    value      TEXT NOT NULL,      -- JSON-encoded
    updated_at TEXT NOT NULL       -- ISO8601 UTC
);

CREATE TABLE IF NOT EXISTS karsan_daily (
    date              TEXT NOT NULL,     -- ET trading day, YYYY-MM-DD
    symbol            TEXT NOT NULL,
    as_of_utc         TEXT NOT NULL,     -- ISO8601 capture time
    score             REAL NOT NULL,
    label             TEXT NOT NULL,
    tone              TEXT NOT NULL,
    dg_tone           TEXT NOT NULL,
    net_gex           REAL,
    zero_gamma        REAL,
    spot              REAL,
    oc_tone           TEXT NOT NULL,
    primary_phase     TEXT NOT NULL,
    days_to_opex      INTEGER NOT NULL,
    in_wow            INTEGER NOT NULL,
    vs_tone           TEXT NOT NULL,
    term_ratio_3m_1m  REAL,
    atm_skew          REAL,
    vvix              REAL,
    source            TEXT NOT NULL DEFAULT 'live',  -- 'live' | 'replay' (KARSAN-BACKFILL-1)
    schema_version    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (date, symbol)           -- write-once: dup insert violates PK
);
"""


@contextmanager
def conn() -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    with conn() as c:
        c.executescript(_SCHEMA)
        # Additive migrations for DBs created before the column existed
        # (CREATE TABLE IF NOT EXISTS skips them). Cheap no-op otherwise.
        cols = {r[1] for r in c.execute("PRAGMA table_info(karsan_daily)")}
        if "source" not in cols:  # KARSAN-BACKFILL-1
            c.execute(
                "ALTER TABLE karsan_daily ADD COLUMN "
                "source TEXT NOT NULL DEFAULT 'live'"
            )
        bt_cols = {r[1] for r in c.execute("PRAGMA table_info(backtest_runs)")}
        if "strategy_id" not in bt_cols:  # FOUNDATION-STRATEGY-REGISTRY-1
            c.execute("ALTER TABLE backtest_runs ADD COLUMN strategy_id INTEGER")
        if "data_manifest" not in bt_cols:  # FOUNDATION-PIT-DATA-1 (BT-F5)
            c.execute("ALTER TABLE backtest_runs ADD COLUMN data_manifest TEXT")


# ---------- signals ---------------------------------------------------------

def list_signals() -> list[dict]:
    with conn() as c:
        rows = c.execute("""
            SELECT s.*, e.last_run_at, e.matches, e.error
            FROM signals s
            LEFT JOIN signal_evals e ON e.signal_id = s.id
            ORDER BY s.id
        """).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["universe"] = json.loads(d["universe"])
        d["matches"] = json.loads(d["matches"]) if d.get("matches") else []
        d["enabled"] = bool(d["enabled"])
        out.append(d)
    return out


def get_signal(signal_id: int) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["universe"] = json.loads(d["universe"])
    d["enabled"] = bool(d["enabled"])
    return d


def upsert_signal(*, id: int | None, name: str, universe: list[str],
                  rule: str, lookback: int = 60, enabled: bool = True) -> int:
    now = time.time()
    with conn() as c:
        if id:
            c.execute("""
                UPDATE signals
                   SET name=?, universe=?, rule=?, lookback=?, enabled=?, updated_at=?
                 WHERE id=?
            """, (name, json.dumps(universe), rule, lookback, int(enabled), now, id))
            return id
        cur = c.execute("""
            INSERT INTO signals (name, universe, rule, lookback, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, json.dumps(universe), rule, lookback, int(enabled), now, now))
        return int(cur.lastrowid)


def delete_signal(signal_id: int) -> None:
    with conn() as c:
        c.execute("DELETE FROM signals WHERE id = ?", (signal_id,))


def save_signal_eval(signal_id: int, *, matches: list[dict],
                     error: str | None = None) -> None:
    with conn() as c:
        c.execute("""
            INSERT INTO signal_evals (signal_id, last_run_at, matches, error)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(signal_id) DO UPDATE SET
                last_run_at = excluded.last_run_at,
                matches     = excluded.matches,
                error       = excluded.error
        """, (signal_id, time.time(), json.dumps(matches), error))


# ---------- backtests -------------------------------------------------------

def add_backtest_run(*, name: str, rule: str, universe: list[str],
                     start_date: str, end_date: str, stats: dict,
                     strategy_id: int | None = None,
                     data_manifest: dict | None = None) -> int:
    """Persist one run. ``strategy_id`` links the run to the registered
    strategy (signals row) it executed; NULL means an ad-hoc run.
    ``data_manifest`` records the exact bars the run consumed (BT-F5);
    NULL only for legacy rows — the engine always passes one."""
    with conn() as c:
        cur = c.execute("""
            INSERT INTO backtest_runs
                (name, rule, universe, start_date, end_date, stats,
                 created_at, strategy_id, data_manifest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, rule, json.dumps(universe), start_date, end_date,
              json.dumps(stats), time.time(), strategy_id,
              json.dumps(data_manifest) if data_manifest is not None else None))
        return int(cur.lastrowid)


def get_backtest_run(run_id: int) -> dict | None:
    """One run row by id, JSON columns decoded. None if absent (BT-F5:
    replay resolves its source row through this)."""
    with conn() as c:
        r = c.execute("SELECT * FROM backtest_runs WHERE id = ?",
                      (run_id,)).fetchone()
    if r is None:
        return None
    return _backtest_row(r)


def _backtest_row(r: sqlite3.Row) -> dict:
    """Decode a backtest_runs row (JSON columns → Python objects)."""
    d = dict(r)
    d["universe"] = json.loads(d["universe"])
    d["stats"] = json.loads(d["stats"])
    raw = d.get("data_manifest")
    d["data_manifest"] = json.loads(raw) if raw else None
    return d


def recent_backtests(n: int = 5) -> list[dict]:
    with conn() as c:
        rows = c.execute("""
            SELECT * FROM backtest_runs
            ORDER BY created_at DESC
            LIMIT ?
        """, (n,)).fetchall()
    return [_backtest_row(r) for r in rows]


# ---------- chain snapshot runs (FOUNDATION-CHAIN-SNAPSHOT-1) ---------------

def _snapshot_run_row(r: sqlite3.Row) -> dict:
    """Decode a chain_snapshot_runs row (JSON list columns → Python lists)."""
    d = dict(r)
    d["symbols_requested"] = json.loads(d["symbols_requested"])
    d["symbols_ok"] = json.loads(d["symbols_ok"])
    d["symbols_failed"] = json.loads(d["symbols_failed"])
    d["files_written"] = json.loads(d["files_written"])
    return d


def start_snapshot_run(*, capture_kind: str, symbols_requested: list[str]) -> int:
    """Open a capture run in the manifest; returns run_id."""
    with conn() as c:
        cur = c.execute("""
            INSERT INTO chain_snapshot_runs (started_at, capture_kind, symbols_requested)
            VALUES (?, ?, ?)
        """, (time.time(), capture_kind, json.dumps(symbols_requested)))
        return int(cur.lastrowid)


def finish_snapshot_run(run_id: int, *, symbols_ok: list[str],
                        symbols_failed: list[dict], rows_written: int,
                        files_written: list[str]) -> None:
    """Complete a capture run.

    This is the SOLE UPDATE touching chain_snapshot_runs, scoped to the
    run_id the caller just created (INV-SNAP-1 carve-out) — manifest rows,
    like snapshot files, are otherwise append-only.
    """
    with conn() as c:
        c.execute("""
            UPDATE chain_snapshot_runs
            SET finished_at = ?, symbols_ok = ?, symbols_failed = ?,
                rows_written = ?, files_written = ?
            WHERE run_id = ?
        """, (time.time(), json.dumps(symbols_ok), json.dumps(symbols_failed),
              rows_written, json.dumps(files_written), run_id))


def last_snapshot_run() -> dict | None:
    """Most recently started capture run, or None if the store is virgin."""
    with conn() as c:
        r = c.execute("""
            SELECT * FROM chain_snapshot_runs
            ORDER BY started_at DESC
            LIMIT 1
        """).fetchone()
    return _snapshot_run_row(r) if r else None


def snapshot_runs_today(*, day_start_ts: float) -> list[dict]:
    """Capture runs started at/after `day_start_ts` (ET-midnight epoch seconds)."""
    with conn() as c:
        rows = c.execute("""
            SELECT * FROM chain_snapshot_runs
            WHERE started_at >= ?
            ORDER BY started_at ASC
        """, (day_start_ts,)).fetchall()
    return [_snapshot_run_row(r) for r in rows]


def snapshot_rows_total() -> int:
    """Total option rows written across all capture runs (manifest estimate)."""
    with conn() as c:
        r = c.execute("SELECT COALESCE(SUM(rows_written), 0) FROM chain_snapshot_runs").fetchone()
    return int(r[0])


# ---------- karsan daily composite (KARSAN-HISTORY-1) ------------------------


def insert_karsan_daily(row: dict) -> None:
    """Write-once insert; raises sqlite3.IntegrityError on (date, symbol) dup."""
    cols = ("date", "symbol", "as_of_utc", "score", "label", "tone", "dg_tone",
            "net_gex", "zero_gamma", "spot", "oc_tone", "primary_phase",
            "days_to_opex", "in_wow", "vs_tone", "term_ratio_3m_1m",
            "atm_skew", "vvix", "source", "schema_version")
    with conn() as c:
        c.execute(
            f"INSERT INTO karsan_daily ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' * len(cols))})",
            tuple(row[k] for k in cols),
        )


def karsan_history(symbol: str, days: int) -> list[dict]:
    """Most recent `days` rows for symbol, oldest first (chart order)."""
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM (SELECT * FROM karsan_daily WHERE symbol = ? "
            "ORDER BY date DESC LIMIT ?) ORDER BY date ASC",
            (symbol, days),
        ).fetchall()
    return [dict(r) for r in rows]


def karsan_recorded_on(symbol: str, day: str) -> bool:
    with conn() as c:
        row = c.execute(
            "SELECT 1 FROM karsan_daily WHERE date = ? AND symbol = ?",
            (day, symbol),
        ).fetchone()
    return row is not None


def get_config(key: str) -> tuple[object, str] | None:
    """Runtime config value or None if unset. Returns (value, updated_at).

    Generic KV for UI-controllable settings (KARSAN-CONFIG-2). Values are
    JSON-encoded; precedence over env/code defaults is the caller's policy.
    """
    with conn() as c:
        row = c.execute(
            "SELECT value, updated_at FROM app_config WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0]), row[1]


def set_config(key: str, value: object) -> str:
    """Upsert a runtime config value (JSON-encoded). Returns updated_at."""
    now = datetime.now(UTC).isoformat()
    with conn() as c:
        c.execute(
            "INSERT INTO app_config (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (key, json.dumps(value), now),
        )
    return now
