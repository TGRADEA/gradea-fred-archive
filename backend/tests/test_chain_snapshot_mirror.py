"""Tests for chain_snapshot_mirror.py (FOUNDATION-CHAIN-SNAPSHOT-MIRROR-1).

All offline. Real parquet files are written into a tmp store by the real
capture path (fake Schwab client, as in test_chain_snapshots.py); Supabase is
a fake at the `sources.supabase` boundary that answers PostgREST-shaped
select/insert calls out of memory and records every call it receives.

The fake enforces the two things the mirror relies on the real database for:
primary-key duplicate rows are ignored (not errors), and the ledger is a
separate table from the rows.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import chain_snapshot_mirror as mirror
import chain_snapshots as snap
import storage
from sources import supabase as sb
from tests.test_chain_snapshots import _chain_body, _FakeClient

T1 = datetime(2026, 8, 6, 15, 0, tzinfo=UTC)
T2 = datetime(2026, 8, 6, 19, 50, tzinfo=UTC)

ROWS_PK = ("symbol", "snapshot_ts_utc", "expiry", "strike", "side")
LEDGER_PK = ("symbol", "snapshot_ts_utc")


# ---------- fake Supabase -----------------------------------------------------


class FakeSupabase:
    """In-memory PostgREST: insert with PK-duplicate ignore, paged select."""

    def __init__(self, *, fail_on: set[str] | None = None):
        self.tables: dict[str, list[dict]] = {mirror.ROWS_TABLE: [], mirror.LEDGER_TABLE: []}
        self.inserts: list[tuple[str, int]] = []
        self.selects: list[tuple[str, dict, int | None]] = []
        self.fail_on = fail_on or set()  # symbols whose ROW inserts raise

    @staticmethod
    def _pk(table: str, row: dict) -> tuple:
        cols = ROWS_PK if table == mirror.ROWS_TABLE else LEDGER_PK
        return tuple(str(row[c]) for c in cols)

    def insert_rows(
        self,
        table,
        rows,
        *,
        project_url,
        service_key,
        schema="public",
        ignore_duplicates=True,
        timeout=None,
    ):
        rows = [dict(r) for r in rows]
        assert project_url and service_key
        if table == mirror.ROWS_TABLE and any(r["symbol"] in self.fail_on for r in rows):
            raise sb.SupabaseError("Supabase HTTP 503 on public.option_chain_snapshots: down")
        # Every value must survive JSON — the real client would choke on NaN.
        json.dumps(rows, allow_nan=False)
        self.inserts.append((table, len(rows)))
        seen = {self._pk(table, r) for r in self.tables[table]}
        for r in rows:
            k = self._pk(table, r)
            if k in seen:
                assert ignore_duplicates, "duplicate without ignore-duplicates"
                continue
            seen.add(k)
            self.tables[table].append(r)
        return len(rows)

    def select(
        self,
        table,
        *,
        project_url,
        service_key,
        schema="public",
        columns="*",
        filters=None,
        order=None,
        limit=None,
        offset=None,
        timeout=None,
    ):
        self.selects.append((table, dict(filters or {}), offset))
        rows = [dict(r) for r in self.tables[table]]
        for column, expr in (filters or {}).items():
            if column == "and":
                inner = expr.strip("()").split(",")
                for clause in inner:
                    col, op, value = clause.split(".", 2)
                    rows = [r for r in rows if _cmp(str(r[col]), op, value)]
                continue
            op, _, value = expr.partition(".")
            rows = [r for r in rows if _cmp(str(r.get(column)), op, value)]
        if order:
            for key_dir in reversed(order.split(",")):
                key, _, direction = key_dir.partition(".")
                rows.sort(key=lambda r: r.get(key), reverse=(direction == "desc"))
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        if columns != "*":
            want = columns.split(",")
            rows = [{c: r.get(c) for c in want} for r in rows]
        return rows


def _cmp(actual: str, op: str, value: str) -> bool:
    if op == "eq":
        return actual == value
    if op == "gte":
        return actual >= value
    if op == "lte":
        return actual <= value
    raise AssertionError(f"unhandled operator {op!r}")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(snap, "SNAPSHOT_ROOT", tmp_path / "chain_snapshots")
    storage.init_db()
    mirror.reset_for_tests()
    return tmp_path / "chain_snapshots"


@pytest.fixture
def fake(monkeypatch):
    fk = FakeSupabase()
    monkeypatch.setattr(mirror.sb, "select", fk.select)
    monkeypatch.setattr(mirror.sb, "insert_rows", fk.insert_rows)
    monkeypatch.setattr(mirror.settings, "snapshot_mirror_enabled", True)
    monkeypatch.setattr(mirror.settings, "supabase_url", "https://x.supabase.co")
    monkeypatch.setattr(mirror.settings, "supabase_service_key", "svc-key")
    return fk


def _capture(symbol: str, ts: datetime, kind: str = "manual") -> Path:
    cl = _FakeClient({snap._SCHWAB_TICKER_MAP.get(symbol, symbol): _chain_body()})
    path, _ = snap.capture_symbol(symbol, kind=kind, ts_utc=ts, client=cl)
    return path


# ---------- configuration gate -------------------------------------------------


def test_not_configured_is_a_recorded_noop(store, monkeypatch):
    monkeypatch.setattr(mirror.settings, "snapshot_mirror_enabled", False)
    _capture("SPY", T1)
    summary = mirror.mirror_pending()
    assert summary["uploaded_files"] == 0 and summary["scanned"] == 0
    assert "not configured" in summary["error"] or "disabled" in summary["error"]
    assert mirror.status()["last_run"]["error"] == summary["error"]


def test_enabled_without_credentials_is_not_configured(store, monkeypatch):
    monkeypatch.setattr(mirror.settings, "snapshot_mirror_enabled", True)
    monkeypatch.setattr(mirror.settings, "supabase_url", "")
    monkeypatch.setattr(mirror.settings, "supabase_service_key", "")
    assert mirror.configured() is False


# ---------- local scan ---------------------------------------------------------


def test_local_files_parses_store_layout_and_skips_strays(store):
    p1 = _capture("SPY", T1)
    p2 = _capture("SPX", T2)
    (store / "SPY" / "2026-08-06" / "notes.txt").write_text("stray")
    (store / "SPY" / "not-a-date").mkdir()
    (store / "bad symbol").mkdir()
    files = mirror.local_files()
    assert [(f.symbol, f.snapshot_ts_utc, f.path) for f in files] == [
        ("SPX", T2, p2),
        ("SPY", T1, p1),
    ]
    assert files[1].relative_to_store() == "SPY/2026-08-06/150000Z.parquet"


# ---------- upload path ---------------------------------------------------------


def test_mirror_uploads_rows_then_ledger(store, fake):
    path = _capture("SPY", T1, kind="near_close")
    before = path.read_bytes()

    summary = mirror.mirror_pending()

    assert summary["error"] is None and summary["failed"] == []
    assert summary["scanned"] == 1 and summary["pending"] == 1
    assert summary["uploaded_files"] == 1 and summary["uploaded_rows"] == 2
    # rows first, ledger last — the ledger row is a proof of completeness
    assert fake.inserts == [(mirror.ROWS_TABLE, 2), (mirror.LEDGER_TABLE, 1)]

    rows = fake.tables[mirror.ROWS_TABLE]
    assert {r["side"] for r in rows} == {"C", "P"}
    assert all("right" not in r for r in rows)
    assert set(rows[0]) == set(mirror.MIRROR_COLUMNS)
    assert rows[0]["snapshot_ts_utc"] == "2026-08-06T15:00:00+00:00"
    assert rows[0]["quote_time_utc"].startswith("2026-08-")
    assert rows[0]["iv"] == pytest.approx(0.225)
    assert rows[0]["capture_kind"] == "near_close"
    assert isinstance(rows[0]["volume"], int) and rows[0]["schema_version"] == 1

    ledger = fake.tables[mirror.LEDGER_TABLE]
    assert ledger == [
        {
            "symbol": "SPY",
            "snapshot_ts_utc": "2026-08-06T15:00:00+00:00",
            "capture_kind": "near_close",
            "source_path": "SPY/2026-08-06/150000Z.parquet",
            "row_count": 2,
            "sha256": hashlib.sha256(before).hexdigest(),
        }
    ]
    # INV-SNAP-1: the local store is untouched
    assert path.read_bytes() == before
    assert len(list(store.rglob("*.parquet"))) == 1


def test_second_run_uploads_nothing(store, fake):
    _capture("SPY", T1)
    mirror.mirror_pending()
    n_inserts = len(fake.inserts)
    summary = mirror.mirror_pending()
    assert summary["pending"] == 0 and summary["uploaded_files"] == 0
    assert len(fake.inserts) == n_inserts


def test_only_new_files_are_uploaded(store, fake):
    _capture("SPY", T1)
    mirror.mirror_pending()
    _capture("SPY", T2)
    _capture("QQQ", T2)
    summary = mirror.mirror_pending()
    assert summary["scanned"] == 3 and summary["pending"] == 2
    assert summary["uploaded_files"] == 2
    assert len(fake.tables[mirror.LEDGER_TABLE]) == 3


def test_batching_splits_rows_per_request(store, fake, monkeypatch):
    monkeypatch.setattr(mirror, "BATCH_ROWS", 1)
    _capture("SPY", T1)  # 2 rows
    mirror.mirror_pending()
    assert fake.inserts == [
        (mirror.ROWS_TABLE, 1),
        (mirror.ROWS_TABLE, 1),
        (mirror.LEDGER_TABLE, 1),
    ]


def test_limit_caps_files_per_run(store, fake):
    _capture("SPY", T1)
    _capture("SPY", T2)
    summary = mirror.mirror_pending(limit=1)
    assert summary["pending"] == 2 and summary["uploaded_files"] == 1
    assert mirror.mirror_pending()["uploaded_files"] == 1


def test_ledger_reads_are_paged(store, fake, monkeypatch):
    monkeypatch.setattr(mirror, "PAGE_ROWS", 1)
    _capture("SPY", T1)
    _capture("SPY", T2)
    mirror.mirror_pending()
    fake.selects.clear()
    summary = mirror.mirror_pending()
    assert summary["pending"] == 0  # both ledger rows were seen across pages
    offsets = [off for table, _f, off in fake.selects if table == mirror.LEDGER_TABLE]
    assert offsets == [0, 1, 2]


# ---------- failure isolation ---------------------------------------------------


def test_failed_file_gets_no_ledger_row_and_run_continues(store, fake):
    fake.fail_on = {"QQQ"}
    _capture("SPY", T1)
    qqq = _capture("QQQ", T1)
    before = qqq.read_bytes()

    summary = mirror.mirror_pending()

    assert summary["uploaded_files"] == 1
    assert summary["failed"] == [
        {
            "path": "QQQ/2026-08-06/150000Z.parquet",
            "error": "Supabase HTTP 503 on public.option_chain_snapshots: down",
        }
    ]  # verbatim
    assert [r["symbol"] for r in fake.tables[mirror.LEDGER_TABLE]] == ["SPY"]
    assert qqq.read_bytes() == before

    # next run retries exactly the failed file
    fake.fail_on = set()
    again = mirror.mirror_pending()
    assert again["pending"] == 1 and again["uploaded_files"] == 1


def test_resend_after_partial_batch_is_idempotent(store, fake, monkeypatch):
    """A run that died after some batches leaves rows but no ledger row; the
    re-send must land every row exactly once."""
    monkeypatch.setattr(mirror, "BATCH_ROWS", 1)
    _capture("SPY", T1)
    calls = {"n": 0}
    real_insert = fake.insert_rows

    def flaky(table, rows, **kw):
        calls["n"] += 1
        if calls["n"] == 2:  # second row batch dies
            raise sb.SupabaseError("connection reset")
        return real_insert(table, rows, **kw)

    monkeypatch.setattr(mirror.sb, "insert_rows", flaky)
    first = mirror.mirror_pending()
    assert first["failed"][0]["error"] == "connection reset"
    assert fake.tables[mirror.LEDGER_TABLE] == []
    assert len(fake.tables[mirror.ROWS_TABLE]) == 1

    monkeypatch.setattr(mirror.sb, "insert_rows", real_insert)
    second = mirror.mirror_pending()
    assert second["uploaded_files"] == 1
    assert len(fake.tables[mirror.ROWS_TABLE]) == 2  # no duplicate of the first row
    assert len(fake.tables[mirror.LEDGER_TABLE]) == 1


def test_ledger_unreachable_is_a_run_level_error(store, fake, monkeypatch):
    _capture("SPY", T1)

    def boom(*a, **k):
        raise sb.SupabaseError("Supabase HTTP 401 on public.option_chain_snapshot_files")

    monkeypatch.setattr(mirror.sb, "select", boom)
    summary = mirror.mirror_pending()
    assert summary["error"].startswith("Supabase HTTP 401")
    assert summary["uploaded_files"] == 0 and fake.inserts == []


# ---------- value coercion ------------------------------------------------------


def test_nan_and_nat_become_null_and_missing_column_is_refused(store, fake, tmp_path):
    path = _capture("SPY", T1)
    df = pd.read_parquet(path)
    df.loc[0, "spot"] = float("nan")
    df["quote_time_utc"] = pd.NaT
    df["quote_time_utc"] = pd.to_datetime(df["quote_time_utc"], utc=True)
    edited = tmp_path / "edited.parquet"
    df.to_parquet(edited, engine="pyarrow", index=False)

    rows, digest, kind = mirror.file_rows(edited)
    assert rows[0]["spot"] is None and rows[1]["spot"] == 500.0
    assert all(r["quote_time_utc"] is None for r in rows)
    assert digest == hashlib.sha256(edited.read_bytes()).hexdigest()
    assert kind == "manual"
    json.dumps(rows, allow_nan=False)  # nothing left that JSON cannot carry

    df.drop(columns=["vega"]).to_parquet(edited, engine="pyarrow", index=False)
    with pytest.raises(mirror.MirrorError, match="vega"):
        mirror.file_rows(edited)


# ---------- read path -----------------------------------------------------------


def test_read_mirrored_round_trips_schema_v1(store, fake):
    _capture("SPY", T1)
    _capture("SPY", T2)
    mirror.mirror_pending()

    both = mirror.read_mirrored("SPY", T1, T2)
    assert list(both.columns) == snap.SCHEMA_V1_COLUMNS
    assert len(both) == 4
    assert set(both["right"]) == {"C", "P"}
    assert both["snapshot_ts_utc"].dt.tz is not None

    only_t2 = mirror.read_mirrored("SPY", datetime(2026, 8, 6, 16, 0, tzinfo=UTC), T2)
    assert len(only_t2) == 2 and (only_t2["snapshot_ts_utc"] == T2).all()

    empty = mirror.read_mirrored("QQQ", T1, T2)
    assert empty.empty and list(empty.columns) == snap.SCHEMA_V1_COLUMNS

    with pytest.raises(ValueError, match="timezone-aware"):
        mirror.read_mirrored("SPY", datetime(2026, 8, 6), T2)


# ---------- status endpoint -----------------------------------------------------


def test_status_is_off_by_default(store):
    assert mirror.status()["enabled"] is False


# ---------- sources.supabase.insert_rows (HTTP boundary) -------------------------


def _mock_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(sb.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))


def test_insert_rows_sends_one_post_with_ignore_duplicates(monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201)

    _mock_client(monkeypatch, handler)
    n = sb.insert_rows(
        "option_chain_snapshots",
        [{"a": 1}, {"a": 2}],
        project_url="https://x.supabase.co/",
        service_key="svc",
    )
    assert n == 2
    (req,) = seen
    assert req.method == "POST"
    assert str(req.url) == "https://x.supabase.co/rest/v1/option_chain_snapshots"
    assert req.headers["Prefer"] == "return=minimal,resolution=ignore-duplicates"
    assert req.headers["Content-Profile"] == "public"
    assert req.headers["Authorization"] == "Bearer svc"
    assert json.loads(req.content) == [{"a": 1}, {"a": 2}]


def test_insert_rows_empty_is_free_and_errors_are_loud(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(409, text='{"code":"23505"}')

    _mock_client(monkeypatch, handler)
    assert sb.insert_rows("t", [], project_url="https://x", service_key="k") == 0
    assert calls["n"] == 0
    with pytest.raises(sb.SupabaseError, match="HTTP 409"):
        sb.insert_rows(
            "t", [{"a": 1}], project_url="https://x", service_key="k", ignore_duplicates=False
        )
    with pytest.raises(sb.SupabaseNotConfigured):
        sb.insert_rows("t", [{"a": 1}], project_url="", service_key="")
