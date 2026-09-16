"""Extend the panel with FRED series the archive does not yet hold.

**This module never writes to the canonical archive.** The five CSVs at the
repository root and ``manifest.json`` belong to ``mirror_fred.py``, and that
tool's checksum contract is the reason this repository exists. Everything
fetched here lands in ``data/extended/`` under its own manifest, and
:func:`load_extended_panel` merges the two at read time. If an ingest goes
wrong, deleting ``data/extended/`` restores the original state exactly.

Network note: fetching requires outbound access to ``fred.stlouisfed.org``.
Where the environment's egress policy blocks it, everything else in this
toolkit still runs against the archived series -- the panel simply stays at five
series. Run this module from a machine with FRED access and commit the result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import ARCHIVE_ROOT, SERIES, SeriesSpec, read_series, trading_calendar

EXTENDED_DIR = ARCHIVE_ROOT / "data" / "extended"
EXTENDED_MANIFEST = EXTENDED_DIR / "manifest.json"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"


@dataclass(frozen=True)
class Recommendation:
    """A series worth adding, and what it unlocks."""

    series_id: str
    label: str
    pub_lag_bdays: int
    starts: str
    priority: int
    unlocks: str


#: Ordered by how much each one widens what this toolkit can honestly say.
#: Priority 1 items remove a stated limitation of the current build; priority 2
#: items add new regime dimensions; priority 3 items are refinements.
RECOMMENDED: tuple[Recommendation, ...] = (
    Recommendation(
        "DFF", "Federal funds effective rate", 1, "1954-07",
        1,
        "A cash rate. Every Sharpe in this build is a total-return Sharpe computed "
        "against zero, which flatters any unlevered bond position over a sample "
        "that starts at double-digit yields. DFF turns them into excess-return "
        "Sharpes and is the single highest-value addition.",
    ),
    Recommendation(
        "BAA10Y", "Moody's Baa corporate minus 10y Treasury", 1, "1986-01",
        1,
        "Credit spread history back to 1986 -- through 1998, 2002, 2008 and 2020. "
        "The archived ICE OAS series start in Aug 2023 and contain no default "
        "cycle, which is why every credit number here carries a caveat. This "
        "series retires that caveat.",
    ),
    Recommendation(
        "USREC", "NBER recession indicator", 20, "1854-12",
        1,
        "Ground truth for regime labels. Lets the dashboard show whether a regime "
        "classifier actually identifies downturns or merely looks like it does. "
        "Note the long lag: NBER dates cycles well after the fact, so this is a "
        "validation series, never a signal.",
    ),
    Recommendation(
        "CPIAUCSL", "CPI, all urban consumers", 15, "1947-01",
        2,
        "With INDPRO, gives the growth-by-inflation quadrant that most macro "
        "regime frameworks actually mean. The current build classifies financial "
        "conditions and the curve, not the economy.",
    ),
    Recommendation(
        "INDPRO", "Industrial production index", 15, "1919-01",
        2,
        "The growth axis of the same quadrant framework, and monthly rather than "
        "quarterly so it can drive a daily panel.",
    ),
    Recommendation(
        "VIXCLS", "CBOE volatility index", 1, "1990-01",
        2,
        "An equity-implied volatility regime, independent of anything in the "
        "current panel and the natural risk-off cross-check on NFCI.",
    ),
    Recommendation(
        "SP500", "S&P 500 index", 1, "rolling 10y",
        2,
        "An equity return stream, so strategies stop being rates-only. FRED serves "
        "this on a rolling 10-year window -- the same trap as the OAS series, so "
        "mirror it early and often.",
    ),
    Recommendation(
        "DGS3MO", "3-month Treasury CMT", 1, "1981-09",
        3,
        "A third curve point, which turns the two-point slope into curvature and "
        "makes butterfly trades expressible.",
    ),
    Recommendation(
        "DTWEXBGS", "Trade-weighted US dollar index", 1, "2006-01",
        3,
        "A dollar regime. Correlated with financial conditions but not identical, "
        "and it leads credit spreads in EM-driven episodes.",
    ),
    Recommendation(
        "DCOILWTICO", "WTI crude oil spot", 1, "1986-01",
        3,
        "A supply-shock axis that separates inflation driven by demand from "
        "inflation driven by energy -- the two behave very differently for bonds.",
    ),
)


def fetch_series(series_id: str, timeout: int = 60) -> pd.Series:
    """Download one FRED series as a float Series indexed by observation date.

    Uses the API when ``FRED_API_KEY`` is set (it honours revisions and is
    rate-limited politely) and the public ``fredgraph.csv`` endpoint otherwise.
    Rows FRED returns as ``.`` -- no observation -- are dropped, matching the
    archive's own convention.
    """
    import requests  # imported lazily so the rest of the package needs no network stack

    api_key = os.environ.get("FRED_API_KEY")
    if api_key:
        resp = requests.get(
            FRED_API_URL,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": "1776-07-04",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        pairs = [(o["date"], o["value"]) for o in obs]
    else:
        resp = requests.get(FRED_CSV_URL, params={"id": series_id}, timeout=timeout)
        resp.raise_for_status()
        rows = list(csv.reader(resp.text.splitlines()))
        if not rows:
            raise ValueError(f"FRED returned an empty response for {series_id}")
        pairs = [(r[0], r[1]) for r in rows[1:] if len(r) >= 2]

    dates, values = [], []
    for date, value in pairs:
        if value in (".", "", None):
            continue
        dates.append(date)
        values.append(float(value))
    if not dates:
        raise ValueError(f"FRED returned no usable observations for {series_id}")

    s = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id, dtype="float64")
    return s.sort_index()


def write_extended(series_id: str, s: pd.Series, label: str = "", pub_lag_bdays: int = 1) -> Path:
    """Write a fetched series to ``data/extended/`` and update its manifest.

    The CSV is byte-identical in shape to the archive's own format -- same
    header, ISO dates, LF endings -- so the same tooling reads both.
    """
    EXTENDED_DIR.mkdir(parents=True, exist_ok=True)
    path = EXTENDED_DIR / f"{series_id}.csv"
    lines = ["observation_date,value\n"]
    lines += [f"{d.date().isoformat()},{v:g}\n" for d, v in s.items()]
    payload = "".join(lines).encode("utf-8")
    path.write_bytes(payload)

    manifest: dict = {"schema": 1, "series": {}}
    if EXTENDED_MANIFEST.exists():
        manifest = json.loads(EXTENDED_MANIFEST.read_text())
    manifest.setdefault("series", {})[series_id] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "obs_count": int(s.size),
        "first_date": str(s.index[0].date()),
        "last_date": str(s.index[-1].date()),
        "bytes": len(payload),
        "label": label or series_id,
        "pub_lag_bdays": pub_lag_bdays,
        "source": "fred.stlouisfed.org",
    }
    manifest["updated_at"] = pd.Timestamp.utcnow().isoformat()
    EXTENDED_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def extended_specs() -> dict[str, SeriesSpec]:
    """Series specs for everything previously written to ``data/extended/``."""
    if not EXTENDED_MANIFEST.exists():
        return {}
    manifest = json.loads(EXTENDED_MANIFEST.read_text())
    specs: dict[str, SeriesSpec] = {}
    for sid, meta in manifest.get("series", {}).items():
        if not (EXTENDED_DIR / f"{sid}.csv").exists():
            continue
        specs[sid] = SeriesSpec(
            series_id=sid,
            label=meta.get("label", sid),
            units="",
            frequency="",
            pub_lag_bdays=int(meta.get("pub_lag_bdays", 1)),
            note=f"Ingested from FRED into data/extended/ on {meta.get('source', 'fred')}.",
        )
    return specs


def ingest(series_ids: list[str] | None = None, timeout: int = 60) -> pd.DataFrame:
    """Fetch and store a set of series; defaults to the priority-1 recommendations.

    Returns a report frame rather than raising on the first failure -- a blocked
    egress policy or a retired series id should not abort the whole batch.
    """
    if series_ids is None:
        series_ids = [r.series_id for r in RECOMMENDED if r.priority == 1]
    known = {r.series_id: r for r in RECOMMENDED}

    rows = []
    for sid in series_ids:
        rec = known.get(sid)
        try:
            s = fetch_series(sid, timeout=timeout)
            write_extended(
                sid,
                s,
                label=rec.label if rec else sid,
                pub_lag_bdays=rec.pub_lag_bdays if rec else 1,
            )
            rows.append(
                {
                    "series_id": sid, "status": "ok", "obs": int(s.size),
                    "first": str(s.index[0].date()), "last": str(s.index[-1].date()),
                    "detail": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            rows.append(
                {
                    "series_id": sid, "status": "failed", "obs": 0,
                    "first": "", "last": "", "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows).set_index("series_id")
