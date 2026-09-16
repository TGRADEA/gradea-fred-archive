"""Compute every strategy and regime once, and serialise the result for the dashboard.

The dashboard is a static page. It does no arithmetic beyond formatting, which
means the numbers it shows and the numbers the CLI prints come from the same
code path and cannot drift apart.

Curves are sampled weekly for transport. Every statistic is computed on the full
daily series first and then carried across as a scalar, so the sampling affects
what the charts draw and nothing else.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Panel, derived_columns, load_full_panel
from .engine import BacktestResult, run_backtest
from .ingest import RECOMMENDED
from .metrics import compute_metrics
from .regimes import FAMILY_DESCRIPTIONS, LABEL_ORDER, build_regimes
from .returns import ASSET_LABELS, build_asset_returns
from .strategies import available_strategies

#: One in five trading days -- a weekly grid that keeps every chart under a few
#: thousand points without visibly changing any curve's shape.
SAMPLE_STRIDE = 5


def _clean(x) -> float | None:
    """JSON has no NaN or Infinity. Anything not finite becomes null."""
    if x is None:
        return None
    f = float(x)
    return None if (math.isnan(f) or math.isinf(f)) else f


def _round(x, digits: int = 6) -> float | None:
    c = _clean(x)
    return None if c is None else round(c, digits)


def _sample(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    sampled = index[::SAMPLE_STRIDE]
    # Always keep the final observation; a stride that misses it makes the chart
    # stop short of the data's actual end date.
    if len(index) and index[-1] not in sampled:
        sampled = sampled.append(pd.DatetimeIndex([index[-1]]))
    return sampled


def _series_on(grid: pd.DatetimeIndex, s: pd.Series, digits: int = 6) -> list[float | None]:
    aligned = s.reindex(grid)
    return [_round(v, digits) for v in aligned.to_numpy()]


def _metrics_payload(m) -> dict:
    return {k: _round(v, 8) for k, v in asdict(m).items() if not isinstance(v, str)} | {
        "start": m.start,
        "end": m.end,
        "n_days": m.n_days,
    }


def build_payload(
    panel: Panel | None = None,
    risk_free: pd.Series | None = None,
) -> dict:
    """Run everything and return the dashboard's data payload."""
    panel = panel if panel is not None else load_full_panel()
    raw = derived_columns(panel.raw)
    pit = derived_columns(panel.pit)
    asset_returns = build_asset_returns(panel.raw)
    regimes = build_regimes(panel.pit)

    grid = _sample(panel.calendar)
    dates = [d.date().isoformat() for d in grid]

    # ---- strategies -------------------------------------------------------
    results: list[BacktestResult] = []
    strategy_payload = []
    for spec in available_strategies(panel.pit):
        weights = spec.build(panel.pit, regimes, **spec.defaults())
        result = run_backtest(
            spec.key,
            weights,
            asset_returns,
            regimes,
            description=spec.description,
            risk_free=risk_free,
        )
        results.append(result)

        attribution = {}
        for family, table in result.attribution.items():
            order = LABEL_ORDER.get(family, list(table.index))
            rows = []
            for label in order:
                if label not in table.index:
                    continue
                r = table.loc[label]
                rows.append(
                    {
                        "label": label,
                        "n_days": int(r["n_days"]),
                        "share": _round(r["share_of_days"], 5),
                        "ann_return": _round(r["ann_return"], 6),
                        "ann_vol": _round(r["ann_vol"], 6),
                        "sharpe": _round(r["sharpe"], 4),
                        "max_drawdown": _round(r["max_drawdown"], 5),
                        "hit_rate": _round(r["hit_rate"], 4),
                        "thin": bool(r["thin"]),
                    }
                )
            if rows:
                attribution[family] = rows

        by_year = []
        if result.by_year is not None:
            for period, row in result.by_year.iterrows():
                by_year.append(
                    {
                        "period": period,
                        "ret": _round(row["ret"], 6),
                        "sharpe": _round(row["sharpe"], 4),
                        "max_drawdown": _round(row["max_drawdown"], 5),
                        "n_days": int(row["n_days"]),
                    }
                )

        strategy_payload.append(
            {
                "key": spec.key,
                "name": spec.name,
                "family": spec.family,
                "description": spec.description,
                "caveat": spec.caveat,
                "assets": result.assets,
                "params": [asdict(p) for p in spec.params],
                "metrics": _metrics_payload(result.metrics),
                "gross_cagr": _round(result.gross_metrics.cagr, 6),
                "equity": _series_on(grid, result.equity, 5),
                "drawdown": _series_on(grid, result.drawdown, 5),
                "exposure": _series_on(grid, result.weights.abs().sum(axis=1), 3),
                "attribution": attribution,
                "by_year": by_year,
                "warnings": result.warnings,
            }
        )

    # ---- reference assets -------------------------------------------------
    assets_payload = {}
    for col in asset_returns.columns:
        s = asset_returns[col].dropna()
        if s.empty:
            continue
        assets_payload[col] = {
            "label": ASSET_LABELS.get(col, col),
            "equity": _series_on(grid, (1 + s).cumprod(), 5),
            "metrics": _metrics_payload(compute_metrics(s, risk_free=risk_free)),
        }

    # ---- regimes ----------------------------------------------------------
    regime_payload = {}
    for family in regimes.columns:
        labels = [l for l in LABEL_ORDER.get(family, []) if l in set(regimes[family].dropna())]
        lookup = {label: i for i, label in enumerate(labels)}
        sampled = regimes[family].reindex(grid)
        timeline = [lookup.get(v) if isinstance(v, str) else None for v in sampled]
        counts = regimes[family].value_counts()
        total = int(counts.sum())
        regime_payload[family] = {
            "description": FAMILY_DESCRIPTIONS.get(family, ""),
            "labels": labels,
            "timeline": timeline,
            "shares": {l: _round(int(counts.get(l, 0)) / total, 4) if total else None for l in labels},
            "counts": {l: int(counts.get(l, 0)) for l in labels},
            "first_labelled": (
                regimes[family].first_valid_index().date().isoformat()
                if regimes[family].notna().any() else None
            ),
        }

    # ---- underlying series for context charts -----------------------------
    context_cols = [c for c in ["DGS10", "DGS2", "SLOPE_10Y2Y", "NFCI",
                                "BAMLH0A0HYM2", "BAMLC0A0CM", "QUALITY_SPREAD"] if c in raw.columns]
    context = {c: _series_on(grid, raw[c], 4) for c in context_cols}

    coverage = []
    for sid, row in panel.coverage.iterrows():
        coverage.append(
            {
                "series_id": sid,
                "label": row["label"],
                "frequency": row["frequency"],
                "pub_lag_bdays": int(row["pub_lag_bdays"]),
                "obs_count": int(row["obs_count"]),
                "first_date": pd.Timestamp(row["first_date"]).date().isoformat(),
                "last_date": pd.Timestamp(row["last_date"]).date().isoformat(),
                "short_history": bool(row["short_history"]),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_stride": SAMPLE_STRIDE,
        "dates": dates,
        "coverage": coverage,
        "strategies": strategy_payload,
        "assets": assets_payload,
        "regimes": regime_payload,
        "context": context,
        "recommendations": [
            {
                "series_id": r.series_id, "label": r.label, "starts": r.starts,
                "priority": r.priority, "unlocks": r.unlocks,
            }
            for r in sorted(RECOMMENDED, key=lambda x: (x.priority, x.series_id))
        ],
        "caveats": [
            "Sharpe ratios are computed against a zero cash rate because the archive "
            "holds no short-rate series. On an unlevered bond position over a sample "
            "beginning at double-digit yields this overstates risk-adjusted return "
            "substantially. Ingest DFF to correct it.",
            "Returns are modelled from yields and spreads, not observed from prices. "
            "Roll-down is omitted, which understates steep-curve returns; defaults are "
            "omitted, which overstates high-yield returns.",
            "The two ICE BofA OAS series begin 2023-08-01 and contain no default cycle. "
            "Every credit statistic here describes a three-year window.",
            "Signals are lagged by each series' publication delay and then held one "
            "further day before trading. NFCI carries a four-business-day lag.",
            "All percentile and z-score thresholds are expanding, never full-sample.",
        ],
    }


def write_payload(path: Path | str, payload: dict | None = None) -> Path:
    """Serialise the payload to JSON."""
    payload = payload if payload is not None else build_payload()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path
