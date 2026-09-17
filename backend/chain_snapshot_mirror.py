"""chain_snapshot_mirror.py — off-box mirror of the chain-snapshot store.

FOUNDATION-CHAIN-SNAPSHOT-MIRROR-1: push every immutable parquet file under
``data/chain_snapshots/`` into Supabase Postgres, so the point-in-time chain
history has a second copy that a dashboard can query (closes D-55).

Why a mirror and not a move
---------------------------
The parquet store stays the source of truth (INV-SNAP-1): capture never
touches the network beyond Schwab, and this module never touches capture.
The mirror is an independent catch-up worker — it lists local files, asks
the ledger which ones Supabase already holds, and uploads the difference.
That shape makes it resumable by construction: a run that dies mid-file
leaves no ledger row, so the next run re-sends the file; row inserts ignore
primary-key duplicates, so a re-send costs bandwidth and nothing else.

Direction is strictly local → Supabase. This module never deletes, renames,
or rewrites a local file, and never issues an UPDATE or DELETE upstream.

Target schema (``_map/migrations/chain_snapshot_mirror_schema.sql``)
--------------------------------------------------------------------
    public.option_chain_snapshots       one row per (symbol, capture,
                                        expiry, strike, side); monthly range
                                        partitions managed by pg_partman
    public.option_chain_snapshot_files  ledger: one row per parquet file
                                        fully uploaded (the resume point)
    public.option_chain_at(symbol, at)  dashboard read: the chain as
                                        observed at or before a timestamp

Columns are the parquet schema v1 (``chain_snapshots.SCHEMA_V1_COLUMNS``)
with one rename: parquet ``right`` → Postgres ``side``, because RIGHT is an
SQL reserved word and a dashboard query should not need to quote it.

Public surface:
    configured() -> bool
    local_files() -> list[LocalFile]
    mirror_pending(*, limit=None) -> dict      # one catch-up run
    read_mirrored(symbol, start, end) -> pd.DataFrame
    status() -> dict        # served by GET /api/snapshots/chains/mirror/status
    mirror_loop() -> coroutine

Invariants: INV-D (Supabase only via ``sources.supabase``), INV-SNAP-1 (read
only against the local store).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import chain_snapshots as _snap
from settings import settings
from sources import supabase as sb

ROWS_TABLE = "option_chain_snapshots"
LEDGER_TABLE = "option_chain_snapshot_files"

# Rows per PostgREST insert. 24 columns × 500 rows is ~100 KB of JSON — well
# inside any request-size limit and small enough that a failed batch is
# cheap to re-send.
BATCH_ROWS = 500

# Rows per ledger page. PostgREST silently caps a response at its configured
# max (1000 by default), so ledger reads page explicitly.
PAGE_ROWS = sb.POSTGREST_MAX_ROWS

# parquet column → Postgres column. Everything not listed keeps its name.
COLUMN_RENAMES: dict[str, str] = {"right": "side"}
_REVERSE_RENAMES = {v: k for k, v in COLUMN_RENAMES.items()}
MIRROR_COLUMNS = [COLUMN_RENAMES.get(c, c) for c in _snap.SCHEMA_V1_COLUMNS]

_FILE_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})Z\.parquet$")


class MirrorError(RuntimeError):
    """A mirror step failed. Always loud, never swallowed into a partial ledger."""


@dataclass(frozen=True)
class LocalFile:
    """One immutable snapshot file, identified by its path components."""

    symbol: str
    snapshot_ts_utc: datetime
    path: Path

    @property
    def key(self) -> tuple[str, str]:
        return (self.symbol, _ts_key(self.snapshot_ts_utc))

    def relative_to_store(self) -> str:
        try:
            return self.path.relative_to(_snap.SNAPSHOT_ROOT).as_posix()
        except ValueError:
            return self.path.as_posix()


# ---------- configuration ----------------------------------------------------


def configured() -> bool:
    """True when the mirror is switched on AND Supabase credentials exist."""
    return bool(settings.snapshot_mirror_enabled) and settings.supabase_configured


def _creds() -> tuple[str, str]:
    if not configured():
        raise sb.SupabaseNotConfigured(
            "Chain-snapshot mirror is not configured. Set "
            "GRADEA_SNAPSHOT_MIRROR_ENABLED=1, SUPABASE_URL and "
            "SUPABASE_SERVICE_KEY in gradea-backend/.env"
        )
    return settings.supabase_url, settings.supabase_service_key


# ---------- timestamps -------------------------------------------------------


def _ts_key(ts: datetime | pd.Timestamp) -> str:
    """Canonical second-resolution UTC key, the same for a path-derived
    datetime and a PostgREST timestamptz string."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(value: Any) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC")


# ---------- local store scan -------------------------------------------------


def local_files(root: Path | None = None) -> list[LocalFile]:
    """Every snapshot file in the store, in path order. Read-only.

    Anything that does not match the store layout
    (``{SYMBOL}/{YYYY-MM-DD}/{HHMMSS}Z.parquet``) is skipped, not raised: a
    stray file is not the mirror's to judge, and it must never block the
    files that do belong.
    """
    base = root if root is not None else _snap.SNAPSHOT_ROOT
    out: list[LocalFile] = []
    if not base.is_dir():
        return out
    for sym_dir in sorted(base.iterdir()):
        if not sym_dir.is_dir() or not _snap._SAFE_SYMBOL.match(sym_dir.name):
            continue
        for day_dir in sorted(sym_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            try:
                day = date.fromisoformat(day_dir.name)
            except ValueError:
                continue
            for f in sorted(day_dir.iterdir()):
                m = _FILE_RE.match(f.name)
                if not m or not f.is_file():
                    continue
                hh, mm, ss = (int(x) for x in m.groups())
                ts = datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=UTC)
                out.append(LocalFile(symbol=sym_dir.name, snapshot_ts_utc=ts, path=f))
    return out


# ---------- ledger -----------------------------------------------------------


def mirrored_keys(
    symbols: list[str], *, creds: tuple[str, str] | None = None
) -> set[tuple[str, str]]:
    """(symbol, ts_key) for every file the ledger says is fully mirrored."""
    url, key = creds if creds is not None else _creds()
    keys: set[tuple[str, str]] = set()
    for sym in symbols:
        offset = 0
        while True:
            rows = sb.select(
                LEDGER_TABLE,
                project_url=url,
                service_key=key,
                columns="symbol,snapshot_ts_utc",
                filters={"symbol": f"eq.{sym}"},
                order="snapshot_ts_utc.asc",
                limit=PAGE_ROWS,
                offset=offset,
            )
            for r in rows:
                keys.add((str(r["symbol"]), _ts_key(_parse_ts(r["snapshot_ts_utc"]))))
            if len(rows) < PAGE_ROWS:
                break
            offset += PAGE_ROWS
    return keys


# ---------- row coercion -----------------------------------------------------


def _json_safe(v: Any) -> Any:
    """One parquet cell → something PostgREST accepts.

    NaN and NaT become SQL NULL (JSON has no NaN, and Postgres would reject
    the string). Timestamps go out as ISO-8601 UTC. numpy scalars unwrap.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, datetime | pd.Timestamp):
        if pd.isna(v):
            return None
        return _parse_ts(v).isoformat()
    if hasattr(v, "item"):  # numpy scalar
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def file_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    """Read one snapshot file → (rows in mirror column names, sha256, capture_kind).

    A file missing any schema-v1 column is refused loudly: silently uploading
    a partial row would be exactly the quiet data loss the ledger exists to
    prevent.
    """
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    df = pd.read_parquet(path, engine="pyarrow")
    missing = [c for c in _snap.SCHEMA_V1_COLUMNS if c not in df.columns]
    if missing:
        raise MirrorError(f"{path}: missing schema-v1 columns {missing}")
    df = df[_snap.SCHEMA_V1_COLUMNS].rename(columns=COLUMN_RENAMES)
    rows = [{k: _json_safe(v) for k, v in rec.items()} for rec in df.to_dict("records")]
    if not rows:
        raise MirrorError(f"{path}: empty snapshot file — refusing to mirror nothing")
    kind = str(rows[0]["capture_kind"])
    return rows, digest, kind


# ---------- upload -----------------------------------------------------------


def mirror_file(lf: LocalFile, *, creds: tuple[str, str] | None = None) -> int:
    """Upload one file's rows, then its ledger row. Returns the row count.

    The ledger row is written LAST and only after every batch was accepted,
    so a ledger entry is a proof of completeness. Raises on any failure;
    ``mirror_pending`` records the error and moves on to the next file.
    """
    url, key = creds if creds is not None else _creds()
    rows, digest, kind = file_rows(lf.path)
    for i in range(0, len(rows), BATCH_ROWS):
        sb.insert_rows(
            ROWS_TABLE,
            rows[i : i + BATCH_ROWS],
            project_url=url,
            service_key=key,
            ignore_duplicates=True,
        )
    sb.insert_rows(
        LEDGER_TABLE,
        [
            {
                "symbol": lf.symbol,
                "snapshot_ts_utc": lf.snapshot_ts_utc.isoformat(),
                "capture_kind": kind,
                "source_path": lf.relative_to_store(),
                "row_count": len(rows),
                "sha256": digest,
            }
        ],
        project_url=url,
        service_key=key,
        ignore_duplicates=True,
    )
    return len(rows)


_RUN_LOCK = threading.Lock()
_LAST_RUN: dict[str, Any] | None = None


def mirror_pending(*, limit: int | None = None, now: datetime | None = None) -> dict[str, Any]:
    """One catch-up run: scan, diff against the ledger, upload the rest.

    Per-file isolation — a failing file records ``{path, error}`` verbatim
    and the run continues. A failure before any file is attempted (ledger
    unreachable, bad credentials) lands in ``error``. Never raises: the
    summary is the whole report, and the scheduler must survive a bad run.
    """
    global _LAST_RUN
    started = now if now is not None else datetime.now(UTC)
    summary: dict[str, Any] = {
        "started_at": started.timestamp(),
        "finished_at": None,
        "scanned": 0,
        "pending": 0,
        "uploaded_files": 0,
        "uploaded_rows": 0,
        "failed": [],
        "error": None,
    }
    with _RUN_LOCK:
        try:
            if not configured():
                raise sb.SupabaseNotConfigured(
                    "mirror disabled or Supabase credentials missing "
                    "(GRADEA_SNAPSHOT_MIRROR_ENABLED / SUPABASE_URL / SUPABASE_SERVICE_KEY)"
                )
            creds = _creds()
            files = local_files()
            summary["scanned"] = len(files)
            known = mirrored_keys(sorted({f.symbol for f in files}), creds=creds)
            pending = [f for f in files if f.key not in known]
            summary["pending"] = len(pending)
            todo = pending[:limit] if limit is not None else pending
            for lf in todo:
                try:
                    n = mirror_file(lf, creds=creds)
                except Exception as exc:  # noqa: BLE001 — isolation is the contract
                    summary["failed"].append({"path": lf.relative_to_store(), "error": str(exc)})
                    continue
                summary["uploaded_files"] += 1
                summary["uploaded_rows"] += n
        except Exception as exc:  # noqa: BLE001 — see docstring
            summary["error"] = str(exc)
        summary["finished_at"] = datetime.now(UTC).timestamp()
        _LAST_RUN = dict(summary)
    return summary


# ---------- read path (dashboard query from Python) --------------------------


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_snap.SCHEMA_V1_COLUMNS)


def read_mirrored(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Schema-v1 rows for `symbol` with snapshot_ts_utc in [start, end], from
    Supabase. Same column order as ``chain_snapshots.read_snapshots`` so the
    two are interchangeable to a consumer.

    SQL consumers should call ``public.option_chain_at(symbol, at)`` directly;
    this helper exists for in-process code and parity checks.
    """
    _snap._require_aware(start, "start")
    _snap._require_aware(end, "end")
    url, key = _creds()
    sym = _snap._dir_symbol(symbol)
    s, e = _parse_ts(start).isoformat(), _parse_ts(end).isoformat()
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = sb.select(
            ROWS_TABLE,
            project_url=url,
            service_key=key,
            columns=",".join(MIRROR_COLUMNS),
            filters={
                "symbol": f"eq.{sym}",
                "and": f"(snapshot_ts_utc.gte.{s},snapshot_ts_utc.lte.{e})",
            },
            order="snapshot_ts_utc.asc,expiry.asc,strike.asc,side.asc",
            limit=PAGE_ROWS,
            offset=offset,
        )
        rows.extend(page)
        if len(page) < PAGE_ROWS:
            break
        offset += PAGE_ROWS
    if not rows:
        return _empty_frame()
    df = pd.DataFrame(rows).rename(columns=_REVERSE_RENAMES)
    for col in ("snapshot_ts_utc", "quote_time_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df[_snap.SCHEMA_V1_COLUMNS].reset_index(drop=True)


# ---------- scheduler & status ----------------------------------------------


async def mirror_loop() -> None:
    """Catch up every ``snapshot_mirror_interval_min`` minutes. Runs forever.

    Nothing to do when the mirror is off — the loop keeps sleeping so a
    later ``.env`` change plus restart is the only enablement path, same as
    every other lifespan task. A failed run is already inside the summary;
    anything above that is swallowed so the next wake still happens.
    """
    while True:
        try:
            if configured():
                await asyncio.to_thread(mirror_pending)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — see docstring
            pass
        await asyncio.sleep(max(60, int(settings.snapshot_mirror_interval_min) * 60))


def status() -> dict[str, Any]:
    """Mirror status for GET /api/snapshots/chains/mirror/status.

    Filesystem and process memory only — never Supabase, never Schwab — so
    it is safe to poll. ``pending`` lives inside ``last_run`` because
    computing it fresh would be a network call.
    """
    return {
        "enabled": bool(settings.snapshot_mirror_enabled),
        "configured": configured(),
        "interval_min": int(settings.snapshot_mirror_interval_min),
        "local_files": len(local_files()),
        "last_run": dict(_LAST_RUN) if _LAST_RUN is not None else None,
    }


def reset_for_tests() -> None:
    """Forget the last run. Test-only hook."""
    global _LAST_RUN
    _LAST_RUN = None
