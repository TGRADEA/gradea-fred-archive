"""Load the FRED archive into two aligned panels: ``raw`` and ``pit``.

The distinction between the two is the whole point of this module.

``raw``
    Each series aligned to the trading calendar on its *observation* date. This
    is what actually happened on day t, and it is the only thing that may be
    used to compute the return the portfolio earned on day t.

``pit``
    Each series shifted forward by its publication lag, so the value carried at
    day t is the most recent value a trader could actually have *seen* by the
    morning of day t. This is the only thing that may be used to compute a
    signal.

Mixing the two is how a backtest ends up with a Sharpe of 4. NFCI is the sharp
edge here: it is dated for the week ending Friday but is not published until the
following Wednesday, so a naive alignment hands the strategy five days of
hindsight on financial conditions -- precisely the days when conditions move.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SeriesSpec:
    """Metadata for one archived FRED series."""

    series_id: str
    label: str
    units: str
    frequency: str
    pub_lag_bdays: int
    note: str

    @property
    def path(self) -> Path:
        return ARCHIVE_ROOT / f"{self.series_id}.csv"


#: Publication lags are expressed in *business days after the observation date*,
#: and are deliberately conservative -- when in doubt the lag is rounded up, so
#: the backtest under-reports rather than over-reports what was knowable.
SERIES: dict[str, SeriesSpec] = {
    "DGS2": SeriesSpec(
        series_id="DGS2",
        label="2-Year Treasury CMT",
        units="percent",
        frequency="business-daily",
        pub_lag_bdays=1,
        note="H.15 posts after the 4:15pm ET close, so day t is tradeable at t+1.",
    ),
    "DGS10": SeriesSpec(
        series_id="DGS10",
        label="10-Year Treasury CMT",
        units="percent",
        frequency="business-daily",
        pub_lag_bdays=1,
        note="H.15 posts after the 4:15pm ET close, so day t is tradeable at t+1.",
    ),
    "NFCI": SeriesSpec(
        series_id="NFCI",
        label="Chicago Fed National Financial Conditions Index",
        units="index (0 = average conditions, + = tighter)",
        frequency="weekly (week ending Friday)",
        pub_lag_bdays=4,
        note=(
            "Released Wednesday 8:30am ET for the week ending the prior Friday "
            "(3 business days), plus one day so the signal trades on a full "
            "session. This 4-day lag is the single most important anti-lookahead "
            "guard in the panel."
        ),
    ),
    "BAMLC0A0CM": SeriesSpec(
        series_id="BAMLC0A0CM",
        label="ICE BofA US Corporate (IG) OAS",
        units="percent",
        frequency="business-daily",
        pub_lag_bdays=1,
        note="Index level struck end-of-day, disseminated the next business day.",
    ),
    "BAMLH0A0HYM2": SeriesSpec(
        series_id="BAMLH0A0HYM2",
        label="ICE BofA US High Yield OAS",
        units="percent",
        frequency="business-daily",
        pub_lag_bdays=1,
        note="Index level struck end-of-day, disseminated the next business day.",
    ),
}

#: Series whose history is capped by FRED's rolling ~3-year OAS window. Any
#: statistic computed over these is a statement about the last three years and
#: nothing more; the engine refuses to present them as long-run evidence.
SHORT_HISTORY = ("BAMLC0A0CM", "BAMLH0A0HYM2")

#: The trading calendar is taken from this series' observation dates. DGS10 is
#: the longest business-daily series in the archive and already excludes US
#: bond-market holidays, so it is a better calendar than any synthetic one.
CALENDAR_SERIES = "DGS10"


def read_series(series_id: str, root: Path | None = None) -> pd.Series:
    """Read one archive CSV into a float Series indexed by observation date."""
    spec = SERIES[series_id]
    path = spec.path if root is None else Path(root) / f"{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This package expects to run from inside a clone "
            f"of the gradea-fred-archive repository."
        )
    dates: list[str] = []
    values: list[float] = []
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["observation_date", "value"]:
            raise ValueError(f"{path} has unexpected header {header!r}")
        for row in reader:
            if len(row) != 2 or not row[1]:
                continue
            dates.append(row[0])
            values.append(float(row[1]))
    s = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id, dtype="float64")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s


def trading_calendar(root: Path | None = None) -> pd.DatetimeIndex:
    """US bond-market trading days, taken from the DGS10 observation dates."""
    return pd.DatetimeIndex(read_series(CALENDAR_SERIES, root).index)


def _available_from(s: pd.Series, pub_lag_bdays: int) -> pd.Series:
    """Re-index a series by the date its value first became knowable.

    Two observations can land on the same availability date (the OAS series
    carry occasional weekend stamps that roll onto the same Monday). Keeping the
    last is correct: by that morning a trader had seen both.
    """
    available = s.index + pd.offsets.BDay(pub_lag_bdays)
    out = pd.Series(s.to_numpy(), index=available, name=s.name)
    return out[~out.index.duplicated(keep="last")]


def _on_calendar(s: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Carry a series onto the trading calendar without losing off-calendar values.

    The obvious ``s.reindex(calendar).ffill()`` is wrong, and wrong in a way that
    is easy to miss: reindexing drops every observation whose date is not itself
    a calendar day *before* the fill runs, so the value is gone rather than
    carried. It bites whenever a publication date lands on a market holiday --
    a value published on Monday 5 July 1965 was silently replaced by the previous
    Thursday's. The direction of the error is conservative (staler data, never
    future data) which is exactly why it survives casual inspection.

    Filling on the union of both indices first, then restricting to the calendar,
    carries every observation to the first trading day on which it was knowable.
    """
    return s.reindex(s.index.union(calendar)).ffill().reindex(calendar)


@dataclass
class Panel:
    """Aligned view of the archive.

    Attributes
    ----------
    raw:
        Observation-dated values on the trading calendar. Use for returns.
    pit:
        Publication-lagged values on the trading calendar. Use for signals.
    calendar:
        The trading calendar both frames share.
    coverage:
        Per-series first/last observation date and count, carried through so
        reports can state honestly how much history stands behind each number.
    """

    raw: pd.DataFrame
    pit: pd.DataFrame
    calendar: pd.DatetimeIndex
    coverage: pd.DataFrame

    def first_valid_common(self, columns: list[str]) -> pd.Timestamp | None:
        """Earliest date on which every named column is populated in ``pit``."""
        sub = self.pit[columns].dropna()
        return None if sub.empty else sub.index[0]


def load_panel(root: Path | None = None, series: list[str] | None = None) -> Panel:
    """Build the aligned :class:`Panel` from the archive CSVs."""
    ids = list(SERIES) if series is None else list(series)
    calendar = trading_calendar(root)

    raw_cols: dict[str, pd.Series] = {}
    pit_cols: dict[str, pd.Series] = {}
    coverage_rows: list[dict[str, object]] = []

    for sid in ids:
        spec = SERIES[sid]
        s = read_series(sid, root)
        coverage_rows.append(
            {
                "series_id": sid,
                "label": spec.label,
                "frequency": spec.frequency,
                "pub_lag_bdays": spec.pub_lag_bdays,
                "obs_count": int(s.size),
                "first_date": s.index[0],
                "last_date": s.index[-1],
                "short_history": sid in SHORT_HISTORY,
            }
        )
        # Returns panel: the value stamped for day t, carried forward across any
        # day the series does not print (NFCI prints weekly).
        raw_cols[sid] = _on_calendar(s, calendar)
        # Signal panel: the value knowable by the morning of day t.
        pit_cols[sid] = _on_calendar(_available_from(s, spec.pub_lag_bdays), calendar)

    coverage = pd.DataFrame(coverage_rows).set_index("series_id")
    return Panel(
        raw=pd.DataFrame(raw_cols, index=calendar),
        pit=pd.DataFrame(pit_cols, index=calendar),
        calendar=calendar,
        coverage=coverage,
    )


def derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the spreads that every regime definition downstream leans on.

    Kept as a function over an arbitrary frame so it can be applied identically
    to ``raw`` and ``pit`` without either one leaking into the other.
    """
    out = frame.copy()
    if {"DGS10", "DGS2"} <= set(out.columns):
        out["SLOPE_10Y2Y"] = out["DGS10"] - out["DGS2"]
    if {"BAMLH0A0HYM2", "BAMLC0A0CM"} <= set(out.columns):
        # Quality spread: how much extra the market charges to step down the
        # credit ladder. Widens well before HY OAS peaks in most stress episodes.
        out["QUALITY_SPREAD"] = out["BAMLH0A0HYM2"] - out["BAMLC0A0CM"]
    return out


def load_full_panel(root: Path | None = None) -> Panel:
    """Load the archive plus anything previously ingested into ``data/extended/``.

    The archive's five series always come first and are never overridden: an
    ingested file carrying an archived series id is ignored, so a bad fetch
    cannot quietly shadow the canonical copy.
    """
    from .ingest import EXTENDED_DIR, extended_specs  # local import: ingest imports us

    panel = load_panel(root)
    extras = {sid: spec for sid, spec in extended_specs().items() if sid not in SERIES}
    if not extras:
        return panel

    raw, pit = panel.raw.copy(), panel.pit.copy()
    rows = []
    for sid, spec in extras.items():
        s = read_series_from(EXTENDED_DIR / f"{sid}.csv", sid)
        raw[sid] = _on_calendar(s, panel.calendar)
        pit[sid] = _on_calendar(_available_from(s, spec.pub_lag_bdays), panel.calendar)
        rows.append(
            {
                "series_id": sid, "label": spec.label, "frequency": spec.frequency or "ingested",
                "pub_lag_bdays": spec.pub_lag_bdays, "obs_count": int(s.size),
                "first_date": s.index[0], "last_date": s.index[-1], "short_history": False,
            }
        )
    coverage = pd.concat([panel.coverage, pd.DataFrame(rows).set_index("series_id")])
    return Panel(raw=raw, pit=pit, calendar=panel.calendar, coverage=coverage)


def read_series_from(path: Path, name: str) -> pd.Series:
    """Read an archive-format CSV from an explicit path."""
    dates: list[str] = []
    values: list[float] = []
    with Path(path).open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            if len(row) != 2 or not row[1]:
                continue
            dates.append(row[0])
            values.append(float(row[1]))
    return pd.Series(
        values, index=pd.DatetimeIndex(dates), name=name, dtype="float64"
    ).sort_index()
