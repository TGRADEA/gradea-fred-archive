"""Turn yields and spreads into the return streams a strategy can hold.

The archive contains no price series, so there is nothing to backtest until
returns are constructed. They are constructed here, explicitly, rather than
being smuggled in as an assumption somewhere downstream.

A constant-maturity Treasury quote is a *par yield*: the coupon a fresh bond of
that maturity would carry to price at 100. So each day we can reprice a
hypothetical par bond, and a position that holds the on-the-run bond and rolls
it daily earns

    r_t  =  y_{t-1} / 252  -  D_mod(y_{t-1}) * dy_t  +  0.5 * C(y_{t-1}) * dy_t^2

carry, duration, convexity. Duration and convexity are recomputed from the
day's own yield rather than pinned to a constant, which matters a great deal
over a sample that runs from 16% yields in 1981 to 0.5% in 2020: a fixed
duration of 8.0 would overstate 1981 bond volatility by roughly a third.

What this model leaves out, and what it therefore costs you:

* **Roll-down.** A real 10-year position ages into a 9.97-year bond each day and
  picks up the slope between those points. Omitting it understates returns in a
  steep curve -- by order 20-40bp a year at a 100bp 10s2s slope.
* **Financing.** These are unlevered total returns, not excess-over-cash.
  Sharpe ratios below are computed against an explicit cash rate where one is
  supplied, and against zero otherwise.
* **Bid-offer inside the index.** Handled as an explicit cost in the engine, not
  buried in the return series.

None of these are fatal for *relative* comparisons between strategies holding
the same instruments, which is what this toolkit is for. They do mean the
absolute level of a curve should not be read as a live track record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

#: Coupon frequency assumed for the par-bond repricing. US Treasuries pay
#: semiannually.
COUPONS_PER_YEAR = 2

#: Spread duration assumed for the credit indices, in years. These are stable
#: characteristics of the ICE BofA indices over the archived window; they are
#: constants rather than fitted values because the archive holds no index
#: duration series to fit against.
SPREAD_DURATION = {
    "BAMLC0A0CM": 7.0,   # US Corporate Master, long-run effective duration ~7
    "BAMLH0A0HYM2": 3.8,  # US High Yield Master II, shorter and more convex
}


def _par_bond_price(yields: np.ndarray, maturity: float, coupon_rate: np.ndarray) -> np.ndarray:
    """Price a bond of ``maturity`` years and ``coupon_rate`` at ``yields``.

    Vectorised over the yield array. Both rates are decimals (0.045 = 4.5%).
    """
    m = COUPONS_PER_YEAR
    n_periods = int(round(maturity * m))
    periods = np.arange(1, n_periods + 1, dtype="float64")
    disc = (1.0 + yields[:, None] / m) ** (-periods[None, :])
    coupon = (coupon_rate[:, None] / m) * 100.0
    return (coupon * disc).sum(axis=1) + 100.0 * disc[:, -1]


def duration_convexity(yields: pd.Series, maturity: float) -> tuple[pd.Series, pd.Series]:
    """Modified duration and convexity of a par bond at each day's own yield.

    Computed by a symmetric 1bp bump of the pricing function rather than a
    closed form. The bump is exact to the model being bumped, which is what we
    want -- a closed form would be exact to a *different* model.
    """
    y = yields.to_numpy(dtype="float64") / 100.0
    valid = np.isfinite(y) & (y > 0)
    d_mod = np.full(y.shape, np.nan)
    convex = np.full(y.shape, np.nan)

    if valid.any():
        yv = y[valid]
        h = 1e-4  # 1 basis point
        p0 = _par_bond_price(yv, maturity, yv)
        p_up = _par_bond_price(yv + h, maturity, yv)
        p_dn = _par_bond_price(yv - h, maturity, yv)
        d_mod[valid] = (p_dn - p_up) / (2.0 * p0 * h)
        convex[valid] = (p_dn + p_up - 2.0 * p0) / (p0 * h * h)

    idx = yields.index
    return (
        pd.Series(d_mod, index=idx, name=f"d_mod_{maturity:g}y"),
        pd.Series(convex, index=idx, name=f"convexity_{maturity:g}y"),
    )


def treasury_total_return(yields: pd.Series, maturity: float) -> pd.Series:
    """Daily total return of a rolling par-bond position at ``maturity``.

    ``yields`` must come from the *raw* panel: the return earned on day t is a
    function of what the market actually did on day t, not of what a trader
    could see. Signals are the only thing that gets lagged.
    """
    y_prev = yields.shift(1)
    d_mod, convex = duration_convexity(y_prev, maturity)
    dy = (yields - y_prev) / 100.0  # decimal change

    carry = (y_prev / 100.0) / TRADING_DAYS
    price = -d_mod * dy + 0.5 * convex * dy * dy
    out = carry + price
    out.name = f"UST{maturity:g}Y"
    return out


def credit_excess_return(oas: pd.Series, series_id: str) -> pd.Series:
    """Daily excess-over-Treasury return of a credit index from its OAS.

    Spread carry less the mark-to-market of the spread move. This is an *excess*
    return: add the matching Treasury return to get a total return. Default
    losses are not modelled, which biases high-yield results upward in exactly
    the periods that matter -- the archived HY window (Aug 2023 onward) contains
    no default wave, so the bias is small here, but it would not stay small over
    2008 or 2020.
    """
    sd = SPREAD_DURATION[series_id]
    s_prev = oas.shift(1)
    d_oas = (oas - s_prev) / 100.0
    carry = (s_prev / 100.0) / TRADING_DAYS
    out = carry - sd * d_oas
    out.name = f"{series_id}_XS"
    return out


def build_asset_returns(raw: pd.DataFrame) -> pd.DataFrame:
    """Assemble every tradeable return stream derivable from the archive.

    Parameters
    ----------
    raw:
        The ``Panel.raw`` frame -- observation-dated, *not* publication-lagged.

    Returns
    -------
    A frame of daily arithmetic returns. Columns present depend on which series
    the panel was loaded with.

    Assets
    ------
    ``UST2Y`` / ``UST10Y``
        Unlevered rolling par-bond total returns.
    ``STEEPENER``
        Long 2y, short 10y, scaled to equal duration exposure so the position is
        (close to) first-order neutral to a parallel shift and expresses the
        slope alone. Sized in 10-year equivalents, so a unit of STEEPENER is a
        unit of DV01 risk, comparable with a unit of UST10Y.
    ``IG_XS`` / ``HY_XS``
        Credit excess returns.
    ``IG_TR`` / ``HY_TR``
        Credit total returns, excess plus a duration-matched Treasury leg.
    """
    out: dict[str, pd.Series] = {}

    have_2 = "DGS2" in raw.columns
    have_10 = "DGS10" in raw.columns

    if have_2:
        out["UST2Y"] = treasury_total_return(raw["DGS2"], 2.0)
    if have_10:
        out["UST10Y"] = treasury_total_return(raw["DGS10"], 10.0)

    if have_2 and have_10:
        d2, _ = duration_convexity(raw["DGS2"].shift(1), 2.0)
        d10, _ = duration_convexity(raw["DGS10"].shift(1), 10.0)
        # Long h notional of the 2y against 1 notional short of the 10y, with
        # h chosen so the two legs carry equal DV01 and the pair is neutral to a
        # parallel shift.
        hedge = (d10 / d2).replace([np.inf, -np.inf], np.nan)
        # The net long notional (h - 1) has to be financed, and leaving that out
        # is not a rounding error -- it inverts the sign of the trade's carry.
        # Unfinanced, the position appears to earn ~80% of the 2y yield for taking
        # no duration risk, which is not a trade that exists. Financed at the
        # front end (the 2y yield standing in for term repo, the closest proxy the
        # archive contains), the carry collapses to the textbook result:
        #
        #     return  =  -(y10 - y2)/252  +  D10 * d(slope)
        #
        # a steepener pays away the slope every day it is held and is repaid only
        # if the curve steepens. That negative carry is the entire reason the
        # trade is hard, and a backtest that omits it is not testing the trade.
        financing = (raw["DGS2"].shift(1) / 100.0) / TRADING_DAYS
        out["STEEPENER"] = out["UST2Y"] * hedge - out["UST10Y"] - (hedge - 1.0) * financing

    if "BAMLC0A0CM" in raw.columns:
        out["IG_XS"] = credit_excess_return(raw["BAMLC0A0CM"], "BAMLC0A0CM")
    if "BAMLH0A0HYM2" in raw.columns:
        out["HY_XS"] = credit_excess_return(raw["BAMLH0A0HYM2"], "BAMLH0A0HYM2")

    # Total returns need a Treasury leg of roughly the index's own duration.
    if "IG_XS" in out and have_10:
        ig_ust = treasury_total_return(raw["DGS10"], 7.0)
        out["IG_TR"] = out["IG_XS"] + ig_ust
    if "HY_XS" in out and have_2 and have_10:
        # HY duration ~3.8y sits between the 2y and 10y points; interpolate.
        w = (3.8 - 2.0) / (10.0 - 2.0)
        hy_ust = (1 - w) * treasury_total_return(raw["DGS2"], 2.0) + w * treasury_total_return(
            raw["DGS10"], 10.0
        )
        out["HY_TR"] = out["HY_XS"] + hy_ust

    frame = pd.DataFrame(out, index=raw.index)
    return frame


#: Human-readable labels, used by the reporting layer and the dashboard.
ASSET_LABELS = {
    "UST2Y": "2y Treasury (total return)",
    "UST10Y": "10y Treasury (total return)",
    "STEEPENER": "10s2s steepener (duration-neutral)",
    "IG_XS": "IG credit (excess return)",
    "HY_XS": "HY credit (excess return)",
    "IG_TR": "IG credit (total return)",
    "HY_TR": "HY credit (total return)",
}
