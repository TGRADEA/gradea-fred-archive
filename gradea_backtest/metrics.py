"""Performance statistics.

Conventions, stated once so no number below is ambiguous:

* Returns are daily **arithmetic** returns. Compounding is applied where a
  statistic is defined on compounded terms (CAGR, drawdown) and not where it is
  not (volatility, skew).
* ``periods_per_year`` is 252 throughout: the archive's calendar is the US bond
  market's trading calendar.
* **Sharpe is computed against an explicit risk-free series when one is passed
  and against zero otherwise.** A zero-rate Sharpe on an unlevered bond position
  is a statement about total return, not about skill, and it will look far too
  good over a sample that begins at 16% yields. Pass a cash rate.
* Drawdown is on the compounded equity curve, peak-to-trough, in fractional
  terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Metrics:
    """Summary statistics for one return stream."""

    n_days: int
    start: str
    end: str
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    best_day: float
    worst_day: float
    skew: float
    kurtosis: float
    ann_turnover: float

    def to_dict(self) -> dict:
        return asdict(self)


def drawdown_curve(returns: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak of the compounded curve."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def compute_metrics(
    returns: pd.Series,
    risk_free: pd.Series | None = None,
    turnover: pd.Series | None = None,
) -> Metrics:
    """Summarise a daily return series.

    ``risk_free`` is a daily (not annualised) cash return aligned to
    ``returns``; when supplied, Sharpe and Sortino are computed on the excess
    series.
    """
    r = returns.dropna()
    if r.empty:
        return Metrics(
            0, "", "", *(float("nan"),) * 11, 0.0
        )

    n = int(r.size)
    years = n / TRADING_DAYS
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    if risk_free is not None:
        excess = (r - risk_free.reindex(r.index).fillna(0.0)).dropna()
    else:
        excess = r

    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ex_mean_ann = float(excess.mean() * TRADING_DAYS)
    ex_vol = float(excess.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ex_mean_ann / ex_vol if ex_vol > 0 else float("nan")

    downside = excess[excess < 0]
    dvol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if downside.size > 1 else float("nan")
    sortino = ex_mean_ann / dvol if dvol and dvol > 0 else float("nan")

    dd = float(drawdown_curve(r).min())
    calmar = cagr / abs(dd) if dd < 0 else float("nan")

    nonzero = r[r != 0.0]
    hit = float((nonzero > 0).mean()) if nonzero.size else float("nan")

    ann_to = 0.0
    if turnover is not None:
        t = turnover.reindex(r.index).fillna(0.0)
        ann_to = float(t.sum() / years) if years > 0 else 0.0

    return Metrics(
        n_days=n,
        start=str(r.index[0].date()),
        end=str(r.index[-1].date()),
        total_return=total,
        cagr=cagr,
        ann_vol=vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=dd,
        calmar=calmar,
        hit_rate=hit,
        best_day=float(r.max()),
        worst_day=float(r.min()),
        skew=float(r.skew()),
        kurtosis=float(r.kurtosis()),
        ann_turnover=ann_to,
    )


def regime_attribution(
    returns: pd.Series,
    labels: pd.Series,
    risk_free: pd.Series | None = None,
    min_days: int = 60,
) -> pd.DataFrame:
    """Break a return stream down by regime label.

    Rows with fewer than ``min_days`` observations are still returned but are
    flagged ``thin=True``: an annualised Sharpe computed on 30 days of data is
    not a Sharpe, and the dashboard greys those cells rather than deleting them.
    """
    joined = pd.DataFrame({"r": returns, "label": labels}).dropna()
    rows = []
    for label, grp in joined.groupby("label", observed=True):
        m = compute_metrics(grp["r"], risk_free=risk_free)
        rows.append(
            {
                "label": str(label),
                "n_days": m.n_days,
                "share_of_days": m.n_days / max(len(joined), 1),
                "ann_return": m.cagr,
                "ann_vol": m.ann_vol,
                "sharpe": m.sharpe,
                "max_drawdown": m.max_drawdown,
                "hit_rate": m.hit_rate,
                "thin": m.n_days < min_days,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "label", "n_days", "share_of_days", "ann_return",
                "ann_vol", "sharpe", "max_drawdown", "hit_rate", "thin",
            ]
        )
    return pd.DataFrame(rows).set_index("label")


def subperiod_table(returns: pd.Series, freq: str = "YE", risk_free: pd.Series | None = None) -> pd.DataFrame:
    """Performance per calendar sub-period -- the cheapest stability check there is.

    A strategy whose entire edge sits in two years will show it here, and no
    amount of full-sample Sharpe will hide it.
    """
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame(columns=["period", "ret", "vol", "sharpe", "max_drawdown", "n_days"])
    rows = []
    for period, grp in r.groupby(pd.Grouper(freq=freq)):
        if grp.empty:
            continue
        m = compute_metrics(grp, risk_free=risk_free)
        rows.append(
            {
                "period": str(period.date())[:4] if freq.startswith("Y") else str(period.date()),
                "ret": float((1 + grp).prod() - 1),
                "vol": m.ann_vol,
                "sharpe": m.sharpe,
                "max_drawdown": m.max_drawdown,
                "n_days": m.n_days,
            }
        )
    return pd.DataFrame(rows).set_index("period")
