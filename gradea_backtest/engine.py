"""The backtest loop.

Execution model, stated plainly:

1. A strategy sees ``pit`` and ``regimes`` for day t. Both are already
   publication-lagged, so nothing unpublished reaches it.
2. Its output weights for day t are then shifted one more day. The position
   held across day t was therefore decided using data published by the morning
   of day t-1. That is one full day more conservative than necessary, and it is
   deliberate: it costs a little return and removes any argument about whether
   a signal could have been acted on in time.
3. The portfolio earns ``sum(weight * asset_return)`` from the **raw** panel.
4. Turnover is charged at a fixed cost in basis points of notional traded.

Weights are in units of the asset's own return stream, so a weight of 1.0 in
``UST10Y`` is an unlevered long. Weights are not normalised or rescaled behind
your back; if a strategy asks for 3x, it gets 3x, and the metrics will say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import Metrics, compute_metrics, drawdown_curve, regime_attribution, subperiod_table

#: Round-trip cost charged per unit of notional turned over, in basis points.
#: 2bp is a reasonable all-in estimate for on-the-run Treasury futures or cash;
#: credit indices are wider and are charged more.
DEFAULT_COST_BPS = {
    "UST2Y": 1.0,
    "UST10Y": 1.5,
    "STEEPENER": 3.0,   # two legs
    "IG_XS": 4.0,
    "HY_XS": 8.0,
    "IG_TR": 4.0,
    "HY_TR": 8.0,
}


@dataclass
class BacktestResult:
    """Everything one strategy run produced."""

    name: str
    description: str
    weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    equity: pd.Series
    drawdown: pd.Series
    metrics: Metrics
    gross_metrics: Metrics
    attribution: dict[str, pd.DataFrame] = field(default_factory=dict)
    by_year: pd.DataFrame | None = None
    assets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _cost_series(weights: pd.DataFrame, cost_bps: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    """Per-day turnover and the cost it incurs.

    Turnover on the first day counts the full cost of putting the position on.
    Skipping it would hand every strategy a free entry.
    """
    prev = weights.shift(1).fillna(0.0)
    delta = (weights - prev).abs()
    turnover = delta.sum(axis=1)
    per_asset_cost = pd.DataFrame(
        {c: delta[c] * (cost_bps.get(c, 2.0) / 1e4) for c in weights.columns},
        index=weights.index,
    )
    return turnover, per_asset_cost.sum(axis=1)


def run_backtest(
    name: str,
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    regimes: pd.DataFrame | None = None,
    description: str = "",
    cost_bps: dict[str, float] | None = None,
    risk_free: pd.Series | None = None,
    execution_lag: int = 1,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """Run one strategy's weights against the asset return panel."""
    cost_bps = DEFAULT_COST_BPS if cost_bps is None else cost_bps
    assets = [c for c in weights.columns if c in asset_returns.columns]
    missing = [c for c in weights.columns if c not in asset_returns.columns]
    warnings: list[str] = []
    if missing:
        warnings.append(
            f"dropped {len(missing)} weight column(s) with no return stream: {', '.join(missing)}"
        )
    if not assets:
        raise ValueError(f"strategy '{name}' produced no weights matching any asset return")

    w = weights[assets].reindex(asset_returns.index)
    # The strategy may legitimately be flat (NaN) before its inputs exist; that
    # is zero exposure, not a missing value.
    w = w.fillna(0.0)

    # Hold the position that was decided `execution_lag` days ago.
    held = w.shift(execution_lag).fillna(0.0)

    rets = asset_returns[assets]
    # Only trade on days every held asset actually printed a return.
    tradeable = rets.notna().all(axis=1)
    gross = (held * rets.fillna(0.0)).sum(axis=1).where(tradeable)

    turnover, costs = _cost_series(held, cost_bps)
    net = gross - costs.where(tradeable)

    # Trim to the window where the strategy is actually invested. A long stretch
    # of leading zeros would otherwise drag CAGR toward zero and flatter the
    # drawdown, which is a quiet way to make a strategy look better than it is.
    active = held.abs().sum(axis=1) > 0
    if active.any():
        first, last = active.idxmax(), active[::-1].idxmax()
        net = net.loc[first:last]
        gross = gross.loc[first:last]
        turnover = turnover.loc[first:last]
        costs = costs.loc[first:last]
        held = held.loc[first:last]
    else:
        warnings.append("strategy never took a position over the available history")

    if start is not None:
        net, gross = net.loc[start:], gross.loc[start:]
        turnover, costs, held = turnover.loc[start:], costs.loc[start:], held.loc[start:]
    if end is not None:
        net, gross = net.loc[:end], gross.loc[:end]
        turnover, costs, held = turnover.loc[:end], costs.loc[:end], held.loc[:end]

    net = net.dropna()
    gross = gross.reindex(net.index)

    equity = (1.0 + net).cumprod()
    result = BacktestResult(
        name=name,
        description=description,
        weights=held,
        gross_returns=gross,
        net_returns=net,
        turnover=turnover.reindex(net.index),
        costs=costs.reindex(net.index),
        equity=equity,
        drawdown=drawdown_curve(net),
        metrics=compute_metrics(net, risk_free=risk_free, turnover=turnover),
        gross_metrics=compute_metrics(gross, risk_free=risk_free),
        by_year=subperiod_table(net, "YE", risk_free=risk_free),
        assets=assets,
        warnings=warnings,
    )

    if regimes is not None:
        for family in regimes.columns:
            table = regime_attribution(net, regimes[family].reindex(net.index), risk_free=risk_free)
            if not table.empty:
                result.attribution[family] = table

    return result


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    """Side-by-side summary of several runs, sorted by net Sharpe."""
    rows = []
    for r in results:
        m = r.metrics
        rows.append(
            {
                "strategy": r.name,
                "start": m.start,
                "end": m.end,
                "years": round(m.n_days / 252, 1),
                "cagr": m.cagr,
                "vol": m.ann_vol,
                "sharpe": m.sharpe,
                "sortino": m.sortino,
                "max_dd": m.max_drawdown,
                "calmar": m.calmar,
                "hit": m.hit_rate,
                "ann_turnover": m.ann_turnover,
                "cost_drag": (r.gross_metrics.cagr - m.cagr),
            }
        )
    return pd.DataFrame(rows).set_index("strategy").sort_values("sharpe", ascending=False)
