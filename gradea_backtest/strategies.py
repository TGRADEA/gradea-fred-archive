"""Strategy library.

Each strategy is a function from the point-in-time panel (and the regime labels
derived from it) to a frame of target weights. None of them read ``raw``; none
of them can, because they are never handed it.

The library is deliberately small and mostly consists of ideas that were
well-known before this archive existed -- trend in yields, mean reversion in the
curve, credit carry gated on stress, flight-to-quality on tightening financial
conditions. That is the point. A strategy invented by searching this dataset for
what worked in it would look better here and worse everywhere else, and the
engine has no way to tell you which one you are looking at. Reusing priors that
were formed elsewhere is the only real defence available at this sample size.

Every parameter carries a default that was *not* tuned on the archive: 200-day
trend, 20-day momentum, 13-week NFCI change, +/-0.2 NFCI band. Change them by all
means -- but each sweep across a parameter spends some of the sample's ability to
tell you anything, and the dashboard's parameter controls exist to show you how
flat or sharp a strategy's response is, not to help you find the peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class ParamSpec:
    """One tunable knob, with the range the dashboard will offer."""

    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float
    help: str = ""


@dataclass(frozen=True)
class Strategy:
    """A named, parameterised weight generator."""

    key: str
    name: str
    family: str
    description: str
    assets: list[str]
    build: Callable[..., pd.DataFrame]
    params: list[ParamSpec] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    caveat: str = ""

    def defaults(self) -> dict[str, float]:
        return {p.key: p.default for p in self.params}


REGISTRY: dict[str, Strategy] = {}


def register(strategy: Strategy) -> Strategy:
    REGISTRY[strategy.key] = strategy
    return strategy


def _frame(index: pd.Index, **columns: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({k: v for k, v in columns.items()}, index=index).fillna(0.0)


def vol_target(
    weight: pd.Series,
    proxy_returns: pd.Series,
    target_vol: float,
    window: int = 60,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Scale a weight so realised vol sits near ``target_vol`` annualised.

    The vol estimate is a trailing window of the *proxy* return series and is
    shifted one day, so the scaling applied on day t uses only volatility
    realised through day t-1. An unshifted estimate would let the position know
    how turbulent today was before deciding how much of today to take.
    """
    realised = proxy_returns.rolling(window, min_periods=window // 2).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / realised.shift(1)).clip(upper=max_leverage)
    return (weight * scale).fillna(0.0)


# --------------------------------------------------------------------------
# Baselines. Every comparison needs something to beat, and for rates the thing
# to beat is "own the bond".
# --------------------------------------------------------------------------

register(Strategy(
    key="hold_10y",
    name="Hold 10y Treasury",
    family="Baseline",
    description=(
        "Unlevered long the rolling 10-year par bond, always. The benchmark any "
        "duration strategy has to justify itself against."
    ),
    assets=["UST10Y"],
    build=lambda pit, regimes, **_: _frame(pit.index, UST10Y=pd.Series(1.0, index=pit.index)),
))

register(Strategy(
    key="hold_2y",
    name="Hold 2y Treasury",
    family="Baseline",
    description="Unlevered long the rolling 2-year par bond. Front-end carry, little duration.",
    assets=["UST2Y"],
    build=lambda pit, regimes, **_: _frame(pit.index, UST2Y=pd.Series(1.0, index=pit.index)),
))

register(Strategy(
    key="hold_steepener",
    name="Hold 10s2s steepener",
    family="Baseline",
    description=(
        "Duration-neutral long 2y / short 10y, always on. Self-financing, so its "
        "Sharpe needs no cash-rate deduction to be meaningful."
    ),
    assets=["STEEPENER"],
    build=lambda pit, regimes, **_: _frame(pit.index, STEEPENER=pd.Series(1.0, index=pit.index)),
))


# --------------------------------------------------------------------------
# Trend.
# --------------------------------------------------------------------------

def _duration_trend(pit, regimes, window=200.0, short_side=1.0, **_):
    w = int(window)
    ma = pit["DGS10"].rolling(w, min_periods=w).mean()
    signal = pd.Series(np.nan, index=pit.index)
    signal[pit["DGS10"] < ma] = 1.0
    signal[pit["DGS10"] >= ma] = -1.0 * float(short_side)
    return _frame(pit.index, UST10Y=signal.where(ma.notna()))


register(Strategy(
    key="duration_trend",
    name="Duration trend (10y vs MA)",
    family="Trend",
    description=(
        "Long duration while the 10y yield sits below its moving average, short "
        "(or flat) while above. Yields trend at multi-month horizons far more "
        "reliably than equity prices do, which is why this is the oldest idea in "
        "systematic macro."
    ),
    assets=["UST10Y"],
    build=_duration_trend,
    params=[
        ParamSpec("window", "Moving-average window (days)", 200, 20, 400, 10,
                  "Longer windows trade less and catch fewer turns."),
        ParamSpec("short_side", "Short-side participation", 1.0, 0.0, 1.0, 0.25,
                  "0 = go flat instead of short when yields are above the average."),
    ],
))


def _duration_trend_vt(pit, regimes, window=200.0, target_vol=5.0, **_):
    base = _duration_trend(pit, regimes, window=window)["UST10Y"]
    # Vol is proxied from yield changes scaled by a nominal duration -- the
    # strategy is not allowed to see the realised return series it is about to
    # earn, so it estimates risk from the published yield path instead.
    proxy = -(pit["DGS10"].diff() / 100.0) * 8.0
    return _frame(pit.index, UST10Y=vol_target(base, proxy, float(target_vol) / 100.0))


register(Strategy(
    key="duration_trend_vt",
    name="Duration trend, vol-targeted",
    family="Trend",
    description=(
        "The same trend signal, sized to a constant risk budget. Bond volatility "
        "moved by a factor of four across this sample; a fixed notional means the "
        "1980s dominate the P&L and the 2010s barely register."
    ),
    assets=["UST10Y"],
    build=_duration_trend_vt,
    params=[
        ParamSpec("window", "Moving-average window (days)", 200, 20, 400, 10),
        ParamSpec("target_vol", "Target volatility (% ann.)", 5.0, 1.0, 15.0, 0.5,
                  "Leverage is capped at 3x regardless."),
    ],
))


# --------------------------------------------------------------------------
# Curve.
# --------------------------------------------------------------------------

def _curve_meanrev(pit, regimes, entry=0.0, exit_level=1.5, **_):
    slope = pit["DGS10"] - pit["DGS2"]
    signal = pd.Series(np.nan, index=pit.index)
    state = 0.0
    out = []
    for value in slope.to_numpy():
        if np.isnan(value):
            out.append(np.nan)
            continue
        if state == 0.0 and value < float(entry):
            state = 1.0            # curve inverted: own the steepener
        elif state == 1.0 and value > float(exit_level):
            state = 0.0            # normalised: stand down
        out.append(state)
    signal = pd.Series(out, index=pit.index)
    return _frame(pit.index, STEEPENER=signal)


register(Strategy(
    key="curve_meanrev",
    name="Inversion steepener",
    family="Curve",
    description=(
        "Put on the steepener when the curve inverts and hold it until the slope "
        "has normalised past the exit level. Inversions have always ended, and "
        "they have always ended by steepening -- but the holding period has run to "
        "years, and the position bleeds carry for most of it."
    ),
    assets=["STEEPENER"],
    build=_curve_meanrev,
    params=[
        ParamSpec("entry", "Entry: slope below (%)", 0.0, -1.0, 0.5, 0.05),
        ParamSpec("exit_level", "Exit: slope above (%)", 1.5, 0.25, 3.0, 0.25),
    ],
    caveat=(
        "Few independent episodes. The 1976-2026 sample contains roughly six "
        "distinct inversion cycles, so this strategy's statistics rest on about "
        "six observations however many days they span."
    ),
))


# --------------------------------------------------------------------------
# Financial conditions.
# --------------------------------------------------------------------------

def _nfci_duration(pit, regimes, window=65.0, threshold=0.05, **_):
    chg = pit["NFCI"].diff(int(window))
    signal = pd.Series(np.nan, index=pit.index)
    signal[chg > float(threshold)] = 1.0     # conditions tightening -> flight to quality
    signal[chg <= float(threshold)] = 0.0
    return _frame(pit.index, UST10Y=signal.where(chg.notna()))


register(Strategy(
    key="nfci_duration",
    name="Flight-to-quality on tightening conditions",
    family="Conditions",
    description=(
        "Own duration only while financial conditions are tightening. The "
        "mechanism is the one thing here with a clear causal story: tightening "
        "conditions are when the Fed is expected to ease and when investors "
        "reach for Treasuries, and both push the same way."
    ),
    assets=["UST10Y"],
    build=_nfci_duration,
    params=[
        ParamSpec("window", "NFCI change window (days)", 65, 10, 130, 5, "65 days is ~13 weeks."),
        ParamSpec("threshold", "Tightening threshold (index pts)", 0.05, 0.0, 0.4, 0.01),
    ],
    requires=["NFCI"],
))


def _nfci_curve(pit, regimes, window=65.0, **_):
    chg = pit["NFCI"].diff(int(window))
    signal = pd.Series(np.nan, index=pit.index)
    signal[chg > 0] = 1.0      # tightening: front end rallies harder, curve steepens
    signal[chg <= 0] = -1.0
    return _frame(pit.index, STEEPENER=signal.where(chg.notna()))


register(Strategy(
    key="nfci_curve",
    name="Conditions-driven curve",
    family="Conditions",
    description=(
        "Steepener while conditions tighten, flattener while they ease. Expresses "
        "the same flight-to-quality idea in a self-financing form, so it does not "
        "depend on the level of yields to look good."
    ),
    assets=["STEEPENER"],
    build=_nfci_curve,
    params=[ParamSpec("window", "NFCI change window (days)", 65, 10, 130, 5)],
    requires=["NFCI"],
))


# --------------------------------------------------------------------------
# Credit. Short history -- see the caveats.
# --------------------------------------------------------------------------

def _credit_carry(pit, regimes, **_):
    stress = regimes["credit_stress"] if "credit_stress" in regimes.columns else None
    signal = pd.Series(np.nan, index=pit.index)
    if stress is None:
        return _frame(pit.index, HY_XS=signal)
    signal[stress.isin(["Calm", "Normal"])] = 1.0
    signal[stress == "Stress"] = 0.0
    return _frame(pit.index, HY_XS=signal.where(stress.notna()))


register(Strategy(
    key="credit_carry",
    name="HY carry, gated on stress",
    family="Credit",
    description=(
        "Collect high-yield spread carry except when HY OAS is in the top fifth "
        "of its own history, then stand aside."
    ),
    assets=["HY_XS"],
    build=_credit_carry,
    requires=["BAMLH0A0HYM2"],
    caveat=(
        "The HY archive begins Aug 2023 and contains no default cycle. Both the "
        "return model (no defaults) and the sample (no stress) push this "
        "strategy's numbers up. Treat them as an illustration of the mechanism, "
        "not as evidence it works."
    ),
))


def _credit_momentum(pit, regimes, window=20.0, **_):
    chg = pit["BAMLH0A0HYM2"].diff(int(window))
    signal = pd.Series(np.nan, index=pit.index)
    signal[chg < 0] = 1.0       # spreads compressing
    signal[chg >= 0] = 0.0
    return _frame(pit.index, HY_XS=signal.where(chg.notna()))


register(Strategy(
    key="credit_momentum",
    name="HY spread momentum",
    family="Credit",
    description="Own high-yield only while spreads have been compressing over the window.",
    assets=["HY_XS"],
    build=_credit_momentum,
    params=[ParamSpec("window", "Spread change window (days)", 20, 5, 90, 5)],
    requires=["BAMLH0A0HYM2"],
    caveat="Same three-year, no-default-cycle limitation as the carry strategy.",
))


# --------------------------------------------------------------------------
# Composite.
# --------------------------------------------------------------------------

def _regime_barbell(pit, regimes, duration_weight=1.0, steepener_weight=1.0, **_):
    """Own duration when the macro state is risk-off, the steepener when risk-on."""
    state = regimes.get("macro_state")
    idx = pit.index
    dur = pd.Series(0.0, index=idx)
    steep = pd.Series(0.0, index=idx)
    if state is None:
        return _frame(idx, UST10Y=dur, STEEPENER=steep)
    dur[state == "Risk-off"] = float(duration_weight)
    steep[state == "Risk-on"] = float(steepener_weight)
    # Neutral sits half-invested in both rather than flat, so the strategy is not
    # simply a market-timing bet on the 12% of days labelled risk-off.
    dur[state == "Neutral"] = 0.5 * float(duration_weight)
    steep[state == "Neutral"] = 0.5 * float(steepener_weight)
    mask = state.notna()
    return _frame(idx, UST10Y=dur.where(mask), STEEPENER=steep.where(mask))


register(Strategy(
    key="regime_barbell",
    name="Regime barbell",
    family="Composite",
    description=(
        "Rotate between duration and the steepener on the composite macro state. "
        "The two legs have historically drawn down at different times, which is "
        "the entire argument for holding them in one book."
    ),
    assets=["UST10Y", "STEEPENER"],
    build=_regime_barbell,
    params=[
        ParamSpec("duration_weight", "Duration leg weight", 1.0, 0.0, 2.0, 0.25),
        ParamSpec("steepener_weight", "Steepener leg weight", 1.0, 0.0, 2.0, 0.25),
    ],
))


def _markov_risk_budget(pit, regimes, low=1.0, medium=0.6, high=0.2, **_):
    """Scale duration exposure down as the learned volatility regime escalates."""
    state = regimes.get("markov_vol")
    idx = pit.index
    if state is None:
        return _frame(idx, UST10Y=pd.Series(0.0, index=idx))
    weight = pd.Series(np.nan, index=idx)
    weight[state == "Low vol"] = float(low)
    weight[state == "Medium vol"] = float(medium)
    weight[state == "High vol"] = float(high)
    return _frame(idx, UST10Y=weight.where(state.notna()))


register(Strategy(
    key="markov_risk_budget",
    name="Markov volatility risk budget",
    family="Regime",
    description=(
        "Hold the 10-year, sized by the learned volatility regime: full size when "
        "the filter says calm, a fifth when it says turbulent. This is a risk rule "
        "rather than a forecast -- it makes no claim about direction, only that a "
        "constant notional means the turbulent periods dominate the P&L."
    ),
    assets=["UST10Y"],
    build=_markov_risk_budget,
    params=[
        ParamSpec("low", "Weight in low-vol regime", 1.0, 0.0, 2.0, 0.1),
        ParamSpec("medium", "Weight in medium-vol regime", 0.6, 0.0, 2.0, 0.1),
        ParamSpec("high", "Weight in high-vol regime", 0.2, 0.0, 2.0, 0.1),
    ],
    caveat=(
        "The weights were set by the shape of the idea (less size when it is "
        "rougher), not fitted. Fitting them on this sample would make the "
        "deflated Sharpe meaningless, since the deflation counts strategies, not "
        "the parameter settings tried within them. Note the turnover: switching "
        "on the hard argmax label makes this resize roughly 65 times a year and "
        "costs about 100bp annually. Sizing on the filter's belief vector "
        "instead of its argmax would plainly churn less -- that variant is "
        "deliberately not swapped in here, because changing the strategy after "
        "seeing its result is the second search that the deflation cannot see."
    ),
))


def available_strategies(pit: pd.DataFrame) -> list[Strategy]:
    """Strategies whose required inputs are present in the loaded panel."""
    return [s for s in REGISTRY.values() if all(r in pit.columns for r in s.requires)]
