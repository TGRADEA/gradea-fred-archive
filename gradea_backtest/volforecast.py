"""Volatility forecast benchmarking: QLIKE, the DM matrix, and the Model Confidence Set.

A port of the evaluation machinery from the GradeA "SPY Volatility Forecast
Benchmark — MCS + DM Matrix (17 models)" research entry (Trading Research,
Status: Verified, 2026-06-30). The method is that entry's; what runs here is a
different asset and a much longer window, and the results are not comparable to
its SPY leaderboard.

**Read this before comparing anything to that page.** That benchmark forecasts
SPY realized variance over 200 aligned observations. This one forecasts 10-year
Treasury return variance over roughly sixty years, because SPY is not in the
archive and neither FRED's index nor the Schwab pipeline is reachable from here.
Rates volatility and equity volatility are different processes -- rates have a
policy reaction function sitting on top of them -- so nothing below confirms or
contradicts the SPY findings. What transfers is the machinery, which is
asset-agnostic, and which accepts a SPY series unchanged the moment one exists.

Two model families from that benchmark are simply absent here rather than
reimplemented badly: everything requiring implied volatility (HAR-VRP, VIX-Q-HAR,
raw VIX^2/252) needs an options surface the archive does not contain. That
matters for the conclusions -- HAR-VRP was the spike-day winner there, 10 out of
10 top-RV days -- so the tail-event question this file cannot answer is exactly
the one their VRP models were kept for.

One deliberate divergence: the original is pure-stdlib Python, honouring an
in-house pipeline constraint. This package already depends on numpy, pandas and
scipy, so a stdlib-only reimplementation inside it would mean maintaining a
second, less-tested numeric stack for no benefit. The constraint is noted and
not inherited.

References, as cited by the original:
  Corsi (2009), J. Financial Econometrics 7(2) -- HAR-RV
  Patton (2011), J. Econometrics 160(1) -- QLIKE under imperfect proxies
  Diebold & Mariano (1995), JBES 13(3); Harvey, Leybourne & Newbold (1997)
  Hansen, Lunde & Nason (2011), Econometrica 79(2) -- Model Confidence Set
  Politis & Romano (1994), JASA 89(428) -- stationary bootstrap
  Patton & Sheppard (2015), RES 97(3) -- realized semivariance (SHAR)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252

#: HAR's three horizons: yesterday, last week, last month. Corsi's insight is
#: that a cascade of these three reproduces long-memory behaviour without any
#: fractional integration, and it has outlasted most of what was built to beat it.
HAR_LAGS = (1, 5, 22)


# ---------------------------------------------------------------------------
# Targets and features
# ---------------------------------------------------------------------------

def realized_variance(returns: pd.Series) -> pd.Series:
    """Daily realized variance from squared returns.

    The weakest link in the whole exercise, and the original says so plainly:
    a daily-close proxy is "good for model ranking, weak as ground truth". A
    single squared return is an unbiased but extremely noisy estimator of the
    day's variance. It is adequate for *ranking* models, since the noise is
    common to every model being ranked, and inadequate for anything that needs
    the level. Five-minute RV fixes this and needs intraday data.
    """
    return (returns**2).rename("rv")


def semivariances(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Downside and upside realized semivariance (Patton & Sheppard 2015)."""
    neg = (returns.where(returns < 0, 0.0) ** 2).rename("rv_neg")
    pos = (returns.where(returns > 0, 0.0) ** 2).rename("rv_pos")
    return neg, pos


def har_design(rv: pd.Series, lags: tuple[int, ...] = HAR_LAGS) -> pd.DataFrame:
    """Lagged daily / weekly / monthly averages of realized variance.

    Every column is shifted by one day, so the row aligned to date t contains
    only information available through t-1 and predicts the variance realized on
    t. Getting this shift wrong produces a model that forecasts today using
    today, which looks like a breakthrough.
    """
    out = {}
    for lag in lags:
        out[f"rv_{lag}"] = rv.rolling(lag, min_periods=lag).mean().shift(1)
    return pd.DataFrame(out, index=rv.index)


# ---------------------------------------------------------------------------
# Loss and tests
# ---------------------------------------------------------------------------

def qlike(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """QLIKE loss (Patton 2011): a/p - log(a/p) - 1.

    Chosen over MSE because it is robust to a noisy volatility proxy: its
    ranking of models is unchanged in expectation when the target is measured
    with error, which is precisely the situation created by a daily-close RV
    proxy. It is also asymmetric in the right direction, penalising
    under-prediction of variance harder than over-prediction.

    Non-positive forecasts are undefined here, not floored. Flooring them would
    quietly hand a broken model a finite loss and let it into the confidence
    set; the original drops those dates for the same reason (13 of 213).
    """
    a = np.asarray(actual, dtype="float64")
    p = np.asarray(predicted, dtype="float64")
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & np.isfinite(p) & (a > 0) & (p > 0)
    ratio = a[ok] / p[ok]
    out[ok] = ratio - np.log(ratio) - 1.0
    return out


def _newey_west_var(d: np.ndarray, horizon: int = 1) -> float:
    """HAC long-run variance of a loss differential (Newey-West, Bartlett)."""
    n = d.size
    d = d - d.mean()
    lag_max = max(horizon - 1, int(np.floor(1.5 * n ** (1 / 3))))
    gamma0 = float(d @ d) / n
    total = gamma0
    for lag in range(1, min(lag_max, n - 1) + 1):
        cov = float(d[lag:] @ d[:-lag]) / n
        total += 2.0 * (1.0 - lag / (lag_max + 1)) * cov
    return max(total, 1e-18)


@dataclass
class DMResult:
    stat: float
    p_value: float
    mean_diff: float
    n: int
    better: str


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1, names: tuple[str, str] = ("a", "b")
) -> DMResult:
    """Two-sided DM test of equal predictive accuracy, HLN-corrected.

    The Harvey-Leybourne-Newbold correction matters: the raw DM statistic
    over-rejects in small samples, and a benchmark's whole job is to avoid
    declaring a winner that isn't one. The corrected statistic is referenced to
    N(0,1) here, following the original.
    """
    a = np.asarray(loss_a, dtype="float64")
    b = np.asarray(loss_b, dtype="float64")
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    n = d.size
    if n < 30:
        return DMResult(float("nan"), float("nan"), float("nan"), n, "")

    var = _newey_west_var(d, horizon)
    stat = d.mean() / np.sqrt(var / n)
    # HLN small-sample correction.
    correction = np.sqrt(
        (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    )
    stat *= correction
    p = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    better = names[0] if d.mean() < 0 else names[1]
    return DMResult(float(stat), float(p), float(d.mean()), n, better)


def dm_matrix(losses: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Pairwise DM statistics. Negative entry [i, j] means row i forecasts better."""
    names = list(losses.columns)
    out = pd.DataFrame(np.nan, index=names, columns=names, dtype="float64")
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            r = diebold_mariano(losses[a].to_numpy(), losses[b].to_numpy(), horizon, (a, b))
            out.loc[a, b] = r.stat
            out.loc[b, a] = -r.stat
    return out


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------

def _stationary_block_indices(n: int, n_boot: int, block: int, rng) -> np.ndarray:
    """Stationary bootstrap index paths (Politis-Romano), geometric block length."""
    p_restart = 1.0 / block
    starts = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < p_restart
    restart[:, 0] = True
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    block: int = 5,
    n_boot: int = 2000,
    seed: int = 20260916,
    chunk: int = 200,
) -> pd.DataFrame:
    """Hansen-Lunde-Nason Model Confidence Set, T_max range statistic.

    Answers the question a leaderboard cannot: given the noise in the data, which
    models can be *distinguished* from the best one? Everything that survives is
    statistically indistinguishable from the winner, and a benchmark that reports
    a rank without reporting this is overstating what it knows.

    Returns one row per model with its mean loss, its MCS p-value, and whether it
    survives at ``alpha``. The p-value is the running maximum across elimination
    rounds, so it is monotone in elimination order, as the original specifies.

    Bootstrap paths are generated in chunks: the index matrix is n_boot x T, and
    at sixty years of daily data a single allocation would be hundreds of
    megabytes.
    """
    clean = losses.dropna()
    names = list(clean.columns)
    L = clean.to_numpy(dtype="float64")
    n, m = L.shape
    if m < 2 or n < 50:
        raise ValueError(f"MCS needs >=2 models and >=50 aligned observations, got {m} and {n}")

    rng = np.random.default_rng(seed)
    # Centred bootstrap means for every model, reused across elimination rounds:
    # the resampling scheme does not depend on which models remain.
    boot_means = np.empty((n_boot, m))
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = _stationary_block_indices(n, size, block, rng)
        boot_means[done : done + size] = L[idx].mean(axis=1)
        done += size

    sample_mean = L.mean(axis=0)
    alive = list(range(m))
    p_values: dict[int, float] = {}
    running_max = 0.0

    while len(alive) > 1:
        sub = np.array(alive)
        k = sub.size
        # d_i. : model i's mean loss relative to the average of the surviving set.
        mean_alive = sample_mean[sub]
        d_bar = mean_alive - mean_alive.mean()
        boot_alive = boot_means[:, sub]
        d_boot = boot_alive - boot_alive.mean(axis=1, keepdims=True)
        # Bootstrap variance of each d_i. around its sample value.
        var = ((d_boot - d_bar) ** 2).mean(axis=0)
        var = np.maximum(var, 1e-18)

        t_stat = d_bar / np.sqrt(var)
        t_boot = (d_boot - d_bar) / np.sqrt(var)
        T_max = float(t_stat.max())
        T_max_boot = t_boot.max(axis=1)
        p = float((T_max_boot >= T_max).mean())

        running_max = max(running_max, p)
        if p >= alpha:
            for i in sub:
                p_values.setdefault(int(i), running_max)
            break
        # Eliminate the worst model in the surviving set.
        worst = int(sub[int(np.argmax(t_stat))])
        p_values[worst] = running_max
        alive.remove(worst)
    else:
        p_values.setdefault(int(alive[0]), 1.0)

    for i in alive:
        p_values.setdefault(int(i), 1.0)

    rows = [
        {
            "model": names[i],
            "mean_qlike": float(sample_mean[i]),
            "p_mcs": float(p_values.get(i, np.nan)),
            "in_set": bool(p_values.get(i, 0.0) >= alpha),
        }
        for i in range(m)
    ]
    return pd.DataFrame(rows).set_index("model").sort_values("mean_qlike")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept, returning NaNs rather than raising."""
    A = np.column_stack([np.ones(X.shape[0]), X])
    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1] + 1, np.nan)
    return beta


def _apply(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X]) @ beta


#: Forecasts are produced on an expanding window with coefficients refreshed
#: quarterly. Fitting once over the whole sample would let a 2008 forecast use
#: coefficients estimated partly from 2020.
DEFAULT_MIN_TRAIN = 1260   # five years
DEFAULT_REFIT_EVERY = 63   # one quarter

#: Forecast horizon in days. **This is a forced divergence from the original,
#: which forecasts one-day-ahead SPY RV.**
#:
#: DGS10 is quoted to the basis point, so 11.8% of days in the archive show a
#: yield change of exactly zero and the left tail of daily realized variance runs
#: three orders of magnitude below its mean. QLIKE contains a -log(actual/pred)
#: term, so a near-zero actual against an ordinary forecast produces an enormous
#: loss: on one-day targets the random walk scores a mean QLIKE of 238 against
#: HAR's 1.4, which measures quotation granularity rather than forecast skill.
#: Aggregating to a week cuts the mean-to-1st-percentile ratio from 2653x to
#: 115x and makes the loss function informative again.
#:
#: The original's own "Verify Before Coding" note flags daily-close RV as "good
#: for model ranking, weak as ground truth" and asks for 5-minute RV. On a
#: 1bp-quantised yield series it is weaker still, and weekly aggregation is the
#: only fix available without intraday data.
DEFAULT_HORIZON = 5


def build_targets_and_features(
    returns: pd.Series, horizon: int = DEFAULT_HORIZON
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Forward realized variance, HAR features, and SHAR features.

    The row indexed at date t holds only information observable through t, and
    the target is the variance realized over t+1 .. t+horizon. Overlapping
    targets induce MA(horizon-1) dependence in the loss differentials, which the
    DM test and the MCS bootstrap are told about rather than left to discover.
    """
    rv1 = realized_variance(returns)
    target = rv1.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    target = target.rename("y")

    har = pd.DataFrame(
        {f"rv_{lag}": rv1.rolling(lag, min_periods=lag).mean() for lag in HAR_LAGS},
        index=returns.index,
    )
    # Semivariance needs the sign of the return, which squaring destroys. Taking
    # it from `returns` rather than reconstructing it from RV is the whole point
    # of the Patton-Sheppard decomposition: bad volatility and good volatility
    # forecast differently.
    neg, pos = semivariances(returns)
    shar = pd.DataFrame(
        {
            "rvneg_1": neg.rolling(1).mean(),
            "rvpos_1": pos.rolling(1).mean(),
            "rvneg_5": neg.rolling(5, min_periods=5).mean(),
            "rv_22": har["rv_22"],
        },
        index=returns.index,
    )
    return target, har, shar


def walk_forward_forecasts(
    returns: pd.Series,
    regimes: pd.Series | None = None,
    horizon: int = DEFAULT_HORIZON,
    min_train: int = DEFAULT_MIN_TRAIN,
    refit_every: int = DEFAULT_REFIT_EVERY,
) -> pd.DataFrame:
    """Out-of-sample forward-variance forecasts from every model family.

    ``RW``            the last ``horizon`` days of realized variance.
    ``AR1-RV``        regression on yesterday's RV alone.
    ``HAR-RV``        Corsi's daily/weekly/monthly cascade.
    ``SHAR``          HAR with the daily term split into down- and up-semivariance.
    ``MS-HAR``        HAR fitted separately within each learned volatility regime.
    ``BMA-Equal``     equal weights on HAR-RV, SHAR and AR1.
    ``BMA-QLIKE-Opt`` simplex-searched weights minimising training QLIKE.

    Fitted on levels, as in the original, so a model may emit a non-positive
    forecast on a quiet stretch. Those are dropped by QLIKE rather than floored.
    """
    target, har, shar = build_targets_and_features(returns, horizon)
    frame = pd.concat([target, har, shar], axis=1)
    valid = frame.dropna().index
    if len(valid) < min_train + 50:
        raise ValueError("not enough history to walk forward")

    y = frame.loc[valid, "y"].to_numpy()
    X_har = frame.loc[valid, list(har.columns)].to_numpy()
    X_shar = frame.loc[valid, list(shar.columns)].to_numpy()
    X_ar1 = frame.loc[valid, ["rv_1"]].to_numpy()
    reg = regimes.reindex(valid) if regimes is not None else None

    names = ["RW", "AR1-RV", "HAR-RV", "SHAR", "MS-HAR", "BMA-Equal", "BMA-QLIKE-Opt"]
    out = {n: np.full(len(valid), np.nan) for n in names}
    # The random walk carries the last `horizon` days of variance forward, which
    # is the like-for-like naive forecast of a `horizon`-day target.
    out["RW"] = X_har[:, 0] * horizon

    beta_ar = beta_har = beta_shar = None
    beta_regime: dict[str, np.ndarray] = {}
    weights = np.array([1 / 3, 1 / 3, 1 / 3])

    for i in range(min_train, len(valid)):
        if (i - min_train) % refit_every == 0:
            # Stop the training window `horizon` days short of today: the targets
            # for the most recent rows overlap days that have not happened yet.
            tr = slice(0, max(i - horizon, 1))
            beta_ar = _ols(X_ar1[tr], y[tr])
            beta_har = _ols(X_har[tr], y[tr])
            beta_shar = _ols(X_shar[tr], y[tr])

            beta_regime = {}
            if reg is not None:
                labels_tr = reg.iloc[tr]
                for label in labels_tr.dropna().unique():
                    mask = (labels_tr == label).to_numpy()
                    if mask.sum() >= 250:
                        beta_regime[label] = _ols(X_har[tr][mask], y[tr][mask])

            preds_tr = np.column_stack(
                [
                    _apply(beta_har, X_har[tr]),
                    _apply(beta_shar, X_shar[tr]),
                    _apply(beta_ar, X_ar1[tr]),
                ]
            )
            best, best_loss = weights, np.inf
            # Coarse simplex grid on purpose: the original flags that these
            # weights need to be stable under refit, and a finely-tuned weight
            # vector is exactly what fails that check.
            for w1 in np.arange(0, 1.01, 0.1):
                for w2 in np.arange(0, 1.01 - w1 + 1e-9, 0.1):
                    w = np.array([w1, w2, 1.0 - w1 - w2])
                    mean_loss = np.nanmean(qlike(y[tr], preds_tr @ w))
                    if np.isfinite(mean_loss) and mean_loss < best_loss:
                        best, best_loss = w, mean_loss
            weights = best

        x_har = X_har[i : i + 1]
        p_har = float(_apply(beta_har, x_har)[0])
        p_shar = float(_apply(beta_shar, X_shar[i : i + 1])[0])
        p_ar = float(_apply(beta_ar, X_ar1[i : i + 1])[0])

        out["HAR-RV"][i] = p_har
        out["SHAR"][i] = p_shar
        out["AR1-RV"][i] = p_ar

        label = reg.iloc[i] if reg is not None else None
        beta_ms = beta_regime.get(label) if isinstance(label, str) else None
        out["MS-HAR"][i] = float(_apply(beta_ms, x_har)[0]) if beta_ms is not None else p_har

        trio = np.array([p_har, p_shar, p_ar])
        out["BMA-Equal"][i] = float(trio.mean())
        out["BMA-QLIKE-Opt"][i] = float(trio @ weights)

    forecasts = pd.DataFrame(out, index=valid)
    forecasts.iloc[:min_train] = np.nan
    forecasts["actual"] = y
    forecasts.attrs["horizon"] = horizon
    return forecasts


def benchmark(
    forecasts: pd.DataFrame,
    alpha: float = 0.10,
    n_boot: int = 2000,
    horizon: int | None = None,
) -> dict:
    """Score a forecast frame: QLIKE losses, the MCS leaderboard, and the DM matrix.

    The bootstrap block length and the DM lag are both set from the forecast
    horizon, because overlapping targets make neighbouring losses dependent by
    construction. Ignoring that would shrink the standard errors and let models
    into the confidence set that do not belong there.
    """
    h = horizon if horizon is not None else int(forecasts.attrs.get("horizon", 1))
    actual = forecasts["actual"].to_numpy()
    models = [c for c in forecasts.columns if c != "actual"]
    losses = pd.DataFrame(
        {m: qlike(actual, forecasts[m].to_numpy()) for m in models}, index=forecasts.index
    )
    aligned = losses.dropna()
    scored = losses.dropna(how="all")
    return {
        "losses": aligned,
        "mcs": model_confidence_set(aligned, alpha=alpha, n_boot=n_boot, block=max(5, 2 * h)),
        "dm": dm_matrix(aligned, horizon=h),
        "n_aligned": len(aligned),
        "n_dropped": int(len(scored) - len(aligned)),
        "horizon": h,
    }
