"""Regime classifiers.

Every classifier here obeys two rules, and both exist because breaking either
one produces a backtest that looks wonderful and is worthless.

**Rule 1: read only from the point-in-time panel.** A regime label for day t may
depend only on data published by the morning of day t.

**Rule 2: no full-sample statistics.** A threshold like "HY OAS in its bottom
quartile" is a lookahead if the quartile is computed over the whole sample -- it
encodes knowledge of the 2008 peak into a 2005 label. Every percentile and
z-score below is *expanding*: at day t it sees days 0..t and nothing else, and
returns NaN until a warmup window has passed.

Fixed numeric thresholds (an inverted curve is a negative curve; a positive NFCI
is tighter-than-average financial conditions) carry no lookahead, because they
come from the definition of the series rather than from the sample. Those are
used in preference to estimated thresholds wherever a meaningful one exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Days of history required before an expanding statistic is trusted. Three
#: years -- short enough to leave most of the sample usable, long enough that an
#: early percentile is not being read off a handful of points.
DEFAULT_WARMUP = 756

#: Order in which labels are presented, per regime family. Bar charts and legends
#: read these so that "Inverted" always sits at the same end of the axis.
LABEL_ORDER: dict[str, list[str]] = {
    "curve_slope": ["Inverted", "Flat", "Normal", "Steep"],
    "curve_dynamics": ["Bull steepening", "Bull flattening", "Bear steepening", "Bear flattening"],
    "nfci_level": ["Loose", "Neutral", "Tight"],
    "nfci_momentum": ["Easing", "Stable", "Tightening"],
    "rate_trend": ["Falling yields", "Rising yields"],
    "credit_stress": ["Calm", "Normal", "Stress"],
    "macro_state": ["Risk-on", "Neutral", "Risk-off"],
}

#: What each family is actually asking about, surfaced in the dashboard so a
#: reader is never guessing what a label means.
FAMILY_DESCRIPTIONS: dict[str, str] = {
    "curve_slope": (
        "Level of the 10s2s slope. Inversion has preceded every US recession in "
        "the sample, with a lead time between 6 and 24 months that is far too "
        "variable to trade directly -- but it conditions everything else."
    ),
    "curve_dynamics": (
        "Joint direction of yield level and slope over 20 sessions. Bull = yields "
        "falling. The four cells map cleanly onto the cycle: bull steepening is "
        "easing-into-weakness, bear flattening is a tightening scare."
    ),
    "nfci_level": (
        "Chicago Fed financial conditions against their own history. Zero is "
        "average conditions by construction; positive is tighter than average."
    ),
    "nfci_momentum": (
        "13-week change in NFCI. The direction of financial conditions leads the "
        "level for risk assets -- conditions tightening from a loose level has "
        "historically hurt more than conditions merely being tight."
    ),
    "rate_trend": (
        "10y yield against its own 200-day average. A crude but durable trend "
        "filter, and the one signal here with no publication lag worth worrying "
        "about."
    ),
    "credit_stress": (
        "High-yield OAS against its expanding percentile history. Limited by the "
        "archive to the post-Aug-2023 window, so it labels the recent past only."
    ),
    "macro_state": (
        "Composite of financial conditions, curve dynamics and credit. A coarse "
        "risk-on / risk-off read intended as a conditioning lens, not a signal."
    ),
}


def expanding_percentile(s: pd.Series, warmup: int = DEFAULT_WARMUP) -> pd.Series:
    """Percentile rank of each value within the history available up to it.

    Returns values in [0, 1], NaN until ``warmup`` observations have accrued.
    Ties are ranked by the fraction of prior observations strictly below, which
    keeps the result monotone in the input.
    """
    values = s.to_numpy(dtype="float64")
    n = values.size
    out = np.full(n, np.nan)
    seen: list[float] = []
    for i in range(n):
        v = values[i]
        if np.isnan(v):
            continue
        if len(seen) >= warmup:
            # bisect over the sorted history is O(log n); the insert is O(n) but
            # n here is ~16k, which is nothing.
            lo, hi = 0, len(seen)
            while lo < hi:
                mid = (lo + hi) // 2
                if seen[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            out[i] = lo / len(seen)
        idx = np.searchsorted(seen, v)
        seen.insert(int(idx), float(v))
    return pd.Series(out, index=s.index, name=f"{s.name}_pct")


def expanding_zscore(s: pd.Series, warmup: int = DEFAULT_WARMUP) -> pd.Series:
    """Z-score of each value against the mean and sd of its own past."""
    mean = s.expanding(min_periods=warmup).mean()
    sd = s.expanding(min_periods=warmup).std()
    return ((s - mean) / sd.where(sd > 0)).rename(f"{s.name}_z")


def _cut(s: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    out = pd.cut(s, bins=bins, labels=labels, right=False)
    return pd.Series(out, index=s.index).astype("object").where(s.notna())


def curve_slope_regime(pit: pd.DataFrame) -> pd.Series:
    """Inverted / Flat / Normal / Steep, on the 10s2s slope in percent."""
    slope = pit["DGS10"] - pit["DGS2"]
    return _cut(
        slope,
        [-np.inf, 0.0, 1.0, 2.0, np.inf],
        ["Inverted", "Flat", "Normal", "Steep"],
    ).rename("curve_slope")


def curve_dynamics_regime(pit: pd.DataFrame, window: int = 20) -> pd.Series:
    """Bull/bear crossed with steepening/flattening over ``window`` sessions."""
    slope = pit["DGS10"] - pit["DGS2"]
    d_level = pit["DGS10"].diff(window)
    d_slope = slope.diff(window)
    bull = d_level < 0
    steep = d_slope > 0
    out = pd.Series(np.nan, index=pit.index, dtype="object")
    out[bull & steep] = "Bull steepening"
    out[bull & ~steep] = "Bull flattening"
    out[~bull & steep] = "Bear steepening"
    out[~bull & ~steep] = "Bear flattening"
    return out.where(d_level.notna() & d_slope.notna()).rename("curve_dynamics")


def nfci_level_regime(pit: pd.DataFrame) -> pd.Series:
    """Loose / Neutral / Tight around the index's own zero point.

    The +/-0.2 band is a judgement call: it keeps the roughly one-third of days
    that sit near average conditions out of both tails, where they would
    otherwise dominate whichever bucket they fell into.
    """
    return _cut(
        pit["NFCI"],
        [-np.inf, -0.2, 0.2, np.inf],
        ["Loose", "Neutral", "Tight"],
    ).rename("nfci_level")


def nfci_momentum_regime(pit: pd.DataFrame, window: int = 65) -> pd.Series:
    """Easing / Stable / Tightening from the ~13-week change in NFCI."""
    chg = pit["NFCI"].diff(window)
    return _cut(
        chg,
        [-np.inf, -0.05, 0.05, np.inf],
        ["Easing", "Stable", "Tightening"],
    ).rename("nfci_momentum")


def rate_trend_regime(pit: pd.DataFrame, window: int = 200) -> pd.Series:
    """10y yield below (Falling) or above (Rising) its 200-day average."""
    ma = pit["DGS10"].rolling(window, min_periods=window).mean()
    out = pd.Series(np.nan, index=pit.index, dtype="object")
    out[pit["DGS10"] < ma] = "Falling yields"
    out[pit["DGS10"] >= ma] = "Rising yields"
    return out.where(ma.notna()).rename("rate_trend")


def credit_stress_regime(pit: pd.DataFrame, warmup: int = 120) -> pd.Series:
    """Calm / Normal / Stress from the expanding percentile of HY OAS.

    The warmup is cut to 120 days rather than the usual 756 because the HY
    archive is only ~780 days long; a 3-year warmup would leave nothing behind.
    That is a real weakening of the guarantee, not a free pass -- the earliest
    labels here rest on six months of history and should be read as provisional.
    """
    if "BAMLH0A0HYM2" not in pit.columns:
        return pd.Series(np.nan, index=pit.index, dtype="object", name="credit_stress")
    pct = expanding_percentile(pit["BAMLH0A0HYM2"], warmup=warmup)
    return _cut(
        pct,
        [-np.inf, 0.33, 0.80, np.inf],
        ["Calm", "Normal", "Stress"],
    ).rename("credit_stress")


def macro_state_regime(pit: pd.DataFrame) -> pd.Series:
    """Composite risk-on / neutral / risk-off score.

    A deliberately simple additive score over three independent reads. It is a
    lens for slicing other strategies' returns, not a signal -- an equal-weighted
    sum of three indicators is not an alpha model and is not presented as one.
    """
    score = pd.Series(0.0, index=pit.index)
    contributors = pd.Series(0, index=pit.index)

    nfci = pit.get("NFCI")
    if nfci is not None:
        score = score.add(np.sign(-nfci).fillna(0.0), fill_value=0.0)
        contributors = contributors.add(nfci.notna().astype(int), fill_value=0)

    slope = pit["DGS10"] - pit["DGS2"]
    d_slope = slope.diff(20)
    score = score.add(np.sign(d_slope).fillna(0.0), fill_value=0.0)
    contributors = contributors.add(d_slope.notna().astype(int), fill_value=0)

    hy = pit.get("BAMLH0A0HYM2")
    if hy is not None:
        d_hy = hy.diff(20)
        score = score.add(np.sign(-d_hy).fillna(0.0), fill_value=0.0)
        contributors = contributors.add(d_hy.notna().astype(int), fill_value=0)

    out = pd.Series(np.nan, index=pit.index, dtype="object")
    enough = contributors >= 2
    out[enough & (score >= 2)] = "Risk-on"
    out[enough & (score <= -2)] = "Risk-off"
    out[enough & (score > -2) & (score < 2)] = "Neutral"
    return out.rename("macro_state")


#: Registry consumed by the engine, the CLI and the dashboard export.
REGIME_BUILDERS = {
    "curve_slope": curve_slope_regime,
    "curve_dynamics": curve_dynamics_regime,
    "nfci_level": nfci_level_regime,
    "nfci_momentum": nfci_momentum_regime,
    "rate_trend": rate_trend_regime,
    "credit_stress": credit_stress_regime,
    "macro_state": macro_state_regime,
}


def build_regimes(pit: pd.DataFrame, families: list[str] | None = None) -> pd.DataFrame:
    """Compute every regime family as a column of categorical labels."""
    names = list(REGIME_BUILDERS) if families is None else list(families)
    cols = {}
    for name in names:
        try:
            cols[name] = REGIME_BUILDERS[name](pit)
        except KeyError:
            # A family whose inputs were not loaded is skipped, not faked.
            continue
    return pd.DataFrame(cols, index=pit.index)
