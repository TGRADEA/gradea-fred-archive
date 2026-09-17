"""chain_snapshots.py — append-only, observed-basis option-chain snapshot store.

FOUNDATION-CHAIN-SNAPSHOT-1: establish a write-once option-chain snapshot
store so that every future backtest, calibration, and live-parity slice can
rely on point-in-time Schwab chain data existing from the day this shipped.

Why this store exists
---------------------
Options backtesting needs historical chains, and Schwab serves live chains
only. Purchased vendor history (ThetaData/ORATS) answers walk-forward depth,
but vendor NBBO is not the Schwab quote a live algo actually sees — only our
own snapshots measure the (Schwab quote − vendor NBBO) distribution at our
real poll times. Every trading day this module does not run is history lost
forever, which is why the capture layer ships before the engine that will
consume it. The store is *observed-basis by construction*: rows are written
once, at capture time, and never revised (same discipline as the macro
`basis=observed` daily writer).

Storage layout (write-once, one immutable file per capture)
-----------------------------------------------------------
    data/chain_snapshots/{SYMBOL}/{YYYY-MM-DD}/{HHMMSS}Z.parquet

Path date/time components are the UTC snapshot timestamp. The directory
symbol is the *config* symbol (``SPX``), never the Schwab wire symbol
(``$SPX.X``) — mapping happens at the fetcher boundary, per the
history_cache convention.

Row schema v1 (``schema_version`` column on every row)
------------------------------------------------------
    snapshot_ts_utc, capture_kind ('interval'|'near_close'|'manual'), symbol,
    spot, expiry, dte, strike, right ('C'|'P'), bid, ask, last, mid,
    bid_size, ask_size, volume, open_interest, delta, gamma, theta, vega,
    rho, iv, quote_time_utc, schema_version=1

Manifest
--------
Every capture run writes one row to the SQLite ``chain_snapshot_runs`` table
(storage.py): which symbols were requested, which succeeded, which failed and
why (verbatim), rows and files written. Per-symbol isolation — one symbol
failing records an error and continues; it never aborts the run and never
writes partial rows for that symbol.

Scheduler
---------
``schedule_loop()`` wakes every minute and captures when
  (a) inside NYSE regular trading hours (pandas-market-calendars, real
      holiday calendar) and >= ``settings.snapshot_interval_min`` since the
      last capture run, or
  (b) 15:45–15:59 ET on a trading day and no 'near_close' capture exists yet
      today — the near-close snapshot is the canonical EOD row for daily
      backtests (the same convention ORATS uses). On early-close sessions the
      window falls after the close; the captured quotes are the day's final
      marks, which is exactly what the EOD row wants.

Public surface (future slices bind to these names — Engine v2 and the
vendor-calibration slice import them):
    capture_all(kind, *, now=None, client=None, symbols=None) -> int  # run_id
    read_snapshots(symbol, start, end) -> pd.DataFrame
    status() -> dict            # served by GET /api/snapshots/chains/status
    schedule_loop() -> coroutine
    SCHEMA_V1_COLUMNS, SCHEMA_VERSION

Invariants
----------
INV-SNAP-1 — snapshot files are immutable: this module never deletes,
overwrites, or rewrites a snapshot file, and issues no UPDATE/DELETE on
snapshot rows (the manifest completion update lives in storage.py, scoped to
the run_id it created). INV-D — Schwab access only via
``sources.schwab.get_client()``; no third-party HTTP imports here, no
yfinance, no fallback source of any kind. This store is Schwab-parity by
definition: a failed capture is a recorded failure, never substituted data.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import calendar_svc
import storage
import universe as universe_mod
from chain import _safe
from settings import settings
from sources.schwab import get_client

ET = ZoneInfo("America/New_York")

SCHEMA_VERSION = 1

# Column order IS the schema. Every parquet file written by this module has
# exactly these columns, and read_snapshots() returns them even for an empty
# store, so downstream code can bind to the shape before data exists.
SCHEMA_V1_COLUMNS = [
    "snapshot_ts_utc",
    "capture_kind",
    "symbol",
    "spot",
    "expiry",
    "dte",
    "strike",
    "right",
    "bid",
    "ask",
    "last",
    "mid",
    "bid_size",
    "ask_size",
    "volume",
    "open_interest",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "iv",
    "quote_time_utc",
    "schema_version",
]

CAPTURE_KINDS = ("interval", "near_close", "manual")

# Near-close capture window, ET wall clock. See module docstring for why the
# window is fixed rather than derived from the session close.
NEAR_CLOSE_START = dtime(15, 45)
NEAR_CLOSE_END = dtime(15, 59, 59)

# Symbol mapping at the fetcher boundary (history_cache convention): config
# names the plain symbol, Schwab's chain endpoint wants the index form.
_SCHWAB_TICKER_MAP: dict[str, str] = {
    "SPX": "$SPX.X",
    "VIX": "$VIX.X",
}

# Directory-name whitelist. The symbol becomes a path component, so anything
# outside this set (separators, dots that could form '..', '$') is refused
# loudly rather than sanitized silently — a mangled name would create a store
# directory nothing ever reads back.
_SAFE_SYMBOL = re.compile(r"^[A-Z][A-Z0-9\-]{0,11}$")

# ── SNAPSHOT-CONFIG-1: runtime symbol config ─────────────────────────────────
# Same DB > env > code-default precedence as karsan.record_symbols (shares
# the storage.app_config KV). karsan.clean_symbols is deliberately NOT
# reused: its regex admits dots, and here the symbol becomes a store
# directory name — _SAFE_SYMBOL (no dots) is the gate that matters.

CONFIG_KEY = "snapshots.symbols"
MAX_SYMBOLS = 64


def clean_snapshot_symbols(raw: list[str]) -> list[str]:
    """Upcase, strip, $-strip, dedupe (order-preserving). Raises ValueError.

    Validates against _SAFE_SYMBOL — path-safety, stricter than quote-form
    validity (BRK.B is a real symbol but an unsafe directory name; refused
    loudly rather than sanitized).
    """
    out: list[str] = []
    for item in raw:
        sym = (item or "").strip().upper().lstrip("$")
        if not _SAFE_SYMBOL.match(sym):
            raise ValueError(
                f"invalid snapshot symbol {item!r}: must match {_SAFE_SYMBOL.pattern}"
                " (symbol becomes a store directory name; dots refused)"
            )
        if sym not in out:
            out.append(sym)
    if len(out) > MAX_SYMBOLS:
        raise ValueError(f"too many symbols: {len(out)} > {MAX_SYMBOLS}")
    return out


def snapshot_symbols() -> tuple[list[str], str, str | None]:
    """Resolved capture list: (symbols, source, universe_name | None).

    source: 'db' | 'env_or_default'. The DB value is either a plain list
    (legacy shape, still honored) or {"universe": name} — a universe ref
    resolves LIVE at read time via the registry (UNIVERSE-CONFIG-1).
    Members that fail _SAFE_SYMBOL at capture time degrade per-symbol via
    capture_all's existing isolation, never a crash. Empty list disables
    the scheduler; manual CLI captures still work with explicit symbols.
    """
    row = storage.get_config(CONFIG_KEY)
    if row is not None:
        value, _updated = row
        if isinstance(value, dict):
            name = str(value["universe"])
            members = [m.symbol for m in universe_mod.get_universe(name)]
            return members, "db", name
        return [str(s) for s in value], "db", None
    return list(settings.snapshot_symbols), "env_or_default", None


_BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_ROOT = _BASE_DIR / "data" / "chain_snapshots"


class SnapshotError(RuntimeError):
    """Base error for the snapshot store. Always loud, never swallowed."""


class SnapshotExistsError(SnapshotError):
    """Write-once guard: the target snapshot file already exists (INV-SNAP-1)."""


def _require_aware(ts: datetime, name: str) -> None:
    """Reject naive datetimes.

    Every timestamp in this store is UTC and tz-aware; session logic is
    America/New_York. A naive datetime is ambiguous by definition and would
    silently shift capture paths and DTE math by the host's UTC offset.
    """
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware (UTC); got naive datetime {ts!r}")


def _dir_symbol(symbol: str) -> str:
    """Validate and normalize a config symbol into a store directory name."""
    sym = (symbol or "").strip().upper()
    if not _SAFE_SYMBOL.match(sym):
        raise SnapshotError(
            f"refusing unsafe snapshot symbol {symbol!r}: symbols must match "
            f"{_SAFE_SYMBOL.pattern} (config symbol, not the Schwab wire form)"
        )
    return sym


def _snapshot_path(symbol: str, ts_utc: datetime) -> Path:
    """The one immutable file a (symbol, timestamp) capture may write."""
    _require_aware(ts_utc, "ts_utc")
    t = ts_utc.astimezone(UTC)
    return (
        SNAPSHOT_ROOT
        / _dir_symbol(symbol)
        / t.strftime("%Y-%m-%d")
        / f"{t.strftime('%H%M%S')}Z.parquet"
    )


def _quote_time_utc(leg: dict) -> datetime | None:
    """Schwab quoteTimeInLong is epoch ms; None/0 → None (never a fake time)."""
    raw = leg.get("quoteTimeInLong")
    if not raw:
        return None
    try:
        return datetime.fromtimestamp(float(raw) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OSError):
        return None


def _walk_legs(
    side: str,
    exp_map: dict,
    *,
    symbol: str,
    spot: float,
    ts_utc: datetime,
    capture_kind: str,
    session_date: date,
    max_dte: int,
) -> list[dict]:
    """Flatten one side of a Schwab exp-date map into schema-v1 row dicts.

    Reuses chain.py's numeric normalization (`_safe`) so a -999 sentinel or a
    None greek lands as 0.0 here exactly as it does on the live chain page.
    Expiries beyond `max_dte` (ET calendar days) are skipped — the store's
    budget is spent on the tenors strategies actually trade.
    """
    rows: list[dict] = []
    for raw_key, strikes_map in (exp_map or {}).items():
        expiry = str(raw_key).split(":", 1)[0]
        try:
            exp_date = date.fromisoformat(expiry)
        except ValueError:
            continue
        dte = (exp_date - session_date).days
        if dte < 0 or dte > max_dte:
            continue
        for strike_str, leg_list in (strikes_map or {}).items():
            try:
                strike = float(strike_str)
            except (TypeError, ValueError):
                continue
            if not leg_list:
                continue
            leg = leg_list[0]
            bid = _safe(leg.get("bid"))
            ask = _safe(leg.get("ask"))
            mid = round((bid + ask) / 2.0, 4) if (bid and ask) else _safe(leg.get("last"))
            rows.append(
                {
                    "snapshot_ts_utc": ts_utc.astimezone(UTC),
                    "capture_kind": capture_kind,
                    "symbol": symbol.upper(),
                    "spot": spot,
                    "expiry": expiry,
                    "dte": dte,
                    "strike": strike,
                    "right": side,
                    "bid": bid,
                    "ask": ask,
                    "last": _safe(leg.get("last")),
                    "mid": mid,
                    "bid_size": int(_safe(leg.get("bidSize"))),
                    "ask_size": int(_safe(leg.get("askSize"))),
                    "volume": int(_safe(leg.get("totalVolume"))),
                    "open_interest": int(_safe(leg.get("openInterest"))),
                    "delta": _safe(leg.get("delta")),
                    "gamma": _safe(leg.get("gamma")),
                    "theta": _safe(leg.get("theta")),
                    "vega": _safe(leg.get("vega")),
                    "rho": _safe(leg.get("rho")),
                    "iv": _safe(leg.get("volatility")) / 100.0,  # Schwab returns IV as percent
                    "quote_time_utc": _quote_time_utc(leg),
                    "schema_version": SCHEMA_VERSION,
                }
            )
    return rows


def capture_symbol(
    symbol: str,
    *,
    kind: str,
    ts_utc: datetime,
    client=None,
    max_dte: int | None = None,
) -> tuple[Path, int]:
    """Fetch one symbol's full chain and write one immutable parquet file.

    Returns (path, row_count). Raises on ANY failure — the caller
    (capture_all) records the error in the manifest; nothing here retries,
    substitutes, or writes a partial file. The write-once guard runs before
    the network call so a duplicate capture never spends a Schwab request.
    """
    if kind not in CAPTURE_KINDS:
        raise SnapshotError(f"unknown capture kind {kind!r}; expected one of {CAPTURE_KINDS}")
    _require_aware(ts_utc, "ts_utc")
    path = _snapshot_path(symbol, ts_utc)
    if path.exists():
        raise SnapshotExistsError(
            f"snapshot already exists: {path} — snapshot files are write-once (INV-SNAP-1)"
        )

    cl = client if client is not None else get_client()
    wire_symbol = _SCHWAB_TICKER_MAP.get(symbol.upper(), symbol.upper())
    resp = cl.get_option_chain(
        wire_symbol,
        contract_type=cl.Options.ContractType.ALL,
        include_underlying_quote=True,
    )
    resp.raise_for_status()
    body = resp.json() or {}

    underlying = body.get("underlying") or {}
    spot = _safe(underlying.get("last") or body.get("underlyingPrice"))
    session_date = ts_utc.astimezone(ET).date()
    dte_cap = max_dte if max_dte is not None else settings.snapshot_max_dte

    common = {
        "symbol": symbol,
        "spot": spot,
        "ts_utc": ts_utc,
        "capture_kind": kind,
        "session_date": session_date,
        "max_dte": dte_cap,
    }
    rows = _walk_legs("C", body.get("callExpDateMap") or {}, **common)
    rows += _walk_legs("P", body.get("putExpDateMap") or {}, **common)
    if not rows:
        raise SnapshotError(
            f"Schwab returned an empty chain for {symbol!r} "
            f"(wire symbol {wire_symbol!r}) — refusing to write an empty snapshot"
        )

    df = pd.DataFrame(rows, columns=SCHEMA_V1_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return path, len(df)


def capture_all(
    kind: str,
    *,
    now: datetime | None = None,
    client=None,
    symbols: list[str] | None = None,
) -> int:
    """Capture every configured symbol once; one immutable file each.

    Writes exactly one manifest row (chain_snapshot_runs) regardless of
    outcome. Per-symbol isolation: a failing symbol records {symbol, error}
    verbatim and the run continues. Returns the manifest run_id.
    """
    if kind not in CAPTURE_KINDS:
        raise SnapshotError(f"unknown capture kind {kind!r}; expected one of {CAPTURE_KINDS}")
    ts = now if now is not None else datetime.now(UTC)
    _require_aware(ts, "now")
    syms = [s.upper() for s in symbols] if symbols is not None else snapshot_symbols()[0]

    run_id = storage.start_snapshot_run(capture_kind=kind, symbols_requested=syms)
    ok: list[str] = []
    failed: list[dict] = []
    files: list[str] = []
    rows_total = 0
    for sym in syms:
        try:
            path, n = capture_symbol(sym, kind=kind, ts_utc=ts, client=client)
        except (
            Exception
        ) as exc:  # noqa: BLE001 — isolation is the contract; error is recorded verbatim
            failed.append({"symbol": sym, "error": str(exc)})
            continue
        ok.append(sym)
        files.append(str(path))
        rows_total += n
    storage.finish_snapshot_run(
        run_id,
        symbols_ok=ok,
        symbols_failed=failed,
        rows_written=rows_total,
        files_written=files,
    )
    return run_id


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA_V1_COLUMNS)


def read_snapshots(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Read schema-v1 rows for `symbol` with snapshot_ts_utc in [start, end].

    Read-only by construction — this function opens files and never writes.
    An empty store (or an empty window) returns an empty DataFrame that still
    carries SCHEMA_V1_COLUMNS, so consumers can bind to the shape today.
    Naive datetimes are rejected, same as everywhere else in this module.
    """
    _require_aware(start, "start")
    _require_aware(end, "end")
    root = SNAPSHOT_ROOT / _dir_symbol(symbol)
    if not root.is_dir():
        return _empty_frame()

    s_day = start.astimezone(UTC).date()
    e_day = end.astimezone(UTC).date()
    frames: list[pd.DataFrame] = []
    for day_dir in sorted(root.iterdir()):
        try:
            day = date.fromisoformat(day_dir.name)
        except ValueError:
            continue
        if day < s_day or day > e_day:
            continue
        for f in sorted(day_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(f))
    if not frames:
        return _empty_frame()

    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["snapshot_ts_utc"], utc=True)
    out = df[(ts >= start) & (ts <= end)].reset_index(drop=True)
    return out


# ---------- scheduler -------------------------------------------------------

# The NYSE calendar lives in calendar_svc (FOUNDATION-CALENDAR-1). The
# module-level alias keeps existing consumers working — karsan.py calls
# _snap._session_bounds at two sites; that seam now points at the shared
# service.
_session_bounds = calendar_svc.session_bounds


def due_capture(
    now: datetime,
    *,
    last_capture_utc: datetime | None,
    near_close_done: bool,
    interval_min: int,
) -> str | None:
    """Pure scheduling decision — the whole gate, frozen-clock testable.

    Returns 'near_close', 'interval', or None. Near-close wins when both
    apply: it is the canonical EOD row and must exist exactly once per
    trading day.
    """
    _require_aware(now, "now")
    et_now = now.astimezone(ET)
    bounds = _session_bounds(et_now.date())
    if bounds is None:
        return None  # weekend or NYSE holiday — never capture

    if not near_close_done and NEAR_CLOSE_START <= et_now.time() <= NEAR_CLOSE_END:
        return "near_close"

    open_utc, close_utc = bounds
    if not (open_utc <= now < close_utc):
        return None  # outside RTH
    if last_capture_utc is None:
        return "interval"
    _require_aware(last_capture_utc, "last_capture_utc")
    if now - last_capture_utc >= timedelta(minutes=interval_min):
        return "interval"
    return None


def _et_day_start_ts(now: datetime) -> float:
    """Epoch seconds at ET midnight of `now`'s ET date (manifest 'today' boundary)."""
    _require_aware(now, "now")
    et_now = now.astimezone(ET)
    midnight = datetime.combine(et_now.date(), dtime.min, tzinfo=ET)
    return midnight.timestamp()


def _tick(now: datetime | None = None) -> int | None:
    """One scheduler wake: consult the manifest, decide, capture if due.

    Interval gating keys off the last run's started_at regardless of kind or
    outcome — a run whose symbols all failed still counts, so a dead token
    produces one recorded failure per interval instead of one per minute
    (Schwab budget: len(symbols) chain calls per capture, ≤ ~120/min limit).
    """
    ts = now if now is not None else datetime.now(UTC)
    _require_aware(ts, "now")
    if not snapshot_symbols()[0]:
        return None  # disabled via config PUT [] or GRADEA_SNAPSHOT_SYMBOLS=""

    last = storage.last_snapshot_run()
    last_ts = datetime.fromtimestamp(last["started_at"], tz=UTC) if last else None
    todays = storage.snapshot_runs_today(day_start_ts=_et_day_start_ts(ts))
    near_done = any(r["capture_kind"] == "near_close" for r in todays)

    kind = due_capture(
        ts,
        last_capture_utc=last_ts,
        near_close_done=near_done,
        interval_min=settings.snapshot_interval_min,
    )
    if kind is None:
        return None
    return capture_all(kind, now=ts)


async def schedule_loop() -> None:
    """Wake every minute; capture when due. Runs forever, cancellation-aware.

    One bad run never tears the loop down: capture errors land in the
    manifest per symbol, and anything unexpected above that is swallowed so
    the next wake still happens — a stopped scheduler is silent data loss,
    which is the one failure mode this store exists to prevent.
    """
    while True:
        try:
            await asyncio.to_thread(_tick)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — see docstring; per-symbol errors are in the manifest
            pass
        await asyncio.sleep(60)


# ---------- status ----------------------------------------------------------


def status() -> dict:
    """Snapshot-store status for GET /api/snapshots/chains/status.

    Shape is pinned by schemas.ChainSnapshotStatusResponse and
    _map/contracts.md (dual enforcement, INV-A). `total_rows_estimate` sums
    manifest rows_written — an estimate because files written outside a
    manifest run (none today) would not be counted; file count and bytes are
    measured from disk.
    """
    now = datetime.now(UTC)
    last = storage.last_snapshot_run()
    todays = storage.snapshot_runs_today(day_start_ts=_et_day_start_ts(now))
    near_done = any(r["capture_kind"] == "near_close" for r in todays)

    total_files = 0
    disk_bytes = 0
    earliest: str | None = None
    if SNAPSHOT_ROOT.is_dir():
        for f in SNAPSHOT_ROOT.rglob("*.parquet"):
            total_files += 1
            disk_bytes += f.stat().st_size
            day = f.parent.name
            if earliest is None or day < earliest:
                earliest = day

    last_run = None
    if last:
        last_run = {
            "run_id": last["run_id"],
            "finished_at": last["finished_at"],
            "capture_kind": last["capture_kind"],
            "rows_written": last["rows_written"],
            "symbols_ok": last["symbols_ok"],
            "symbols_failed": last["symbols_failed"],
        }

    return {
        "enabled": bool(snapshot_symbols()[0]),
        "symbols": snapshot_symbols()[0],
        "interval_min": settings.snapshot_interval_min,
        "last_run": last_run,
        "today": {"captures": len(todays), "near_close_done": near_done},
        "store": {
            "total_files": total_files,
            "total_rows_estimate": storage.snapshot_rows_total(),
            "disk_bytes": disk_bytes,
            "earliest_date": earliest,
        },
    }
