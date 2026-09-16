"""Is the backtest result skill, or is it the best of eleven coin flips?

The QuantGuild corpus returns to this question in a dozen lectures -- "Is Trading
Luck or Skill", "I Bet You've Never Found Alpha", "Stop Using the Sharpe Ratio",
"3 Backtesting Pitfalls", "CAGR: The Quantitative BS Test". It is also the
largest hole in the rest of this package, which until now reported a Sharpe ratio
for eleven strategies and let the reader assume the best one meant something.

It usually does not. Two effects, both of which inflate a headline Sharpe:

**Multiple testing.** Search eleven strategies and the best one is drawn from the
maximum of eleven random variables, not from one. Under a null where every
strategy is worthless, that maximum is comfortably positive. The expected best
Sharpe under the null scales with the spread of Sharpes across the trials and
with the logarithm of how many you ran, and it is subtracted here rather than
discussed.

**Non-normal returns.** The Sharpe ratio's own sampling distribution depends on
the skew and kurtosis of what it measures. Bond strategies are negatively skewed
and fat-tailed -- long stretches of carry punctuated by a sharp loss -- and that
combination makes a Sharpe ratio *less* reliable than the normal case, not more.
A strategy earning carry quietly looks good right up until it doesn't.

Implemented from Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014) and
"The Sharpe Ratio Efficiency Frontier" (2012), plus a stationary block bootstrap
(Politis & Romano, 1994) for a confidence interval that survives the fact that
daily returns are autocorrelated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252

#: Euler-Mascheroni constant, in the expected-maximum expression below.
EULER_GAMMA = 0.5772156649015329


@dataclass
class SignificanceResult:
    """Everything the significance layer can say about one return stream."""

    sharpe_ann: float
    n_days: int
    skew: float
    kurtosis: float
    psr: float
    deflated_sharpe: float
    expected_max_sharpe_ann: float
    n_trials: int
    min_track_record_years: float
    bootstrap_lo: float
    bootstrap_hi: float
    bootstrap_p_value: float
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


def _moments(returns: np.ndarray) -> tuple[float, float, float]:
    """Per-period Sharpe, skew, and non-excess kurtosis."""
    sd = returns.std(ddof=1)
    sharpe = float(returns.mean() / sd) if sd > 0 else float("nan")
    # Non-excess kurtosis: 3.0 for a normal distribution, which is the
    # convention the PSR expression below is written in.
    return sharpe, float(stats.skew(returns)), float(stats.kurtosis(returns) + 3.0)


def probabilistic_sharpe(
    returns: pd.Series | np.ndarray, benchmark_sharpe_ann: float = 0.0
) -> float:
    """P(true Sharpe > benchmark), correcting for skew and kurtosis.

    ``benchmark_sharpe_ann`` is annualised for the caller's convenience and
    converted internally; everything else here works in per-period units.
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype="float64")
    if r.size < 30:
        return float("nan")
    sharpe, skew, kurt = _moments(r)
    if not np.isfinite(sharpe):
        return float("nan")
    target = benchmark_sharpe_ann / np.sqrt(TRADING_DAYS)

    # The denominator is the standard error of the Sharpe estimator. Negative
    # skew raises it and excess kurtosis raises it: both make the estimate less
    # trustworthy, which is the whole point of using this instead of a t-stat.
    variance = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe**2
    if variance <= 0:
        return float("nan")
    z = (sharpe - target) * np.sqrt(r.size - 1) / np.sqrt(variance)
    return float(stats.norm.cdf(z))


def expected_maximum_sharpe(trial_sharpes_ann: np.ndarray | list[float]) -> float:
    """Expected best annualised Sharpe when every strategy tried is worthless.

    This is the bar a strategy has to clear to have said anything at all. It
    rises with the number of trials and with how widely their Sharpes are
    scattered -- a library of near-identical strategies is barely a search, while
    a library of wildly different ones is a wide one.
    """
    s = np.asarray([x for x in trial_sharpes_ann if np.isfinite(x)], dtype="float64")
    n = s.size
    if n < 2:
        return 0.0
    variance = float(s.var(ddof=1)) / TRADING_DAYS  # per-period
    if variance <= 0:
        return 0.0
    # Expected maximum of n draws from a standard normal, to the usual
    # two-term approximation.
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    expected_max = np.sqrt(variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    return float(expected_max * np.sqrt(TRADING_DAYS))


def minimum_track_record_years(
    returns: pd.Series | np.ndarray, benchmark_sharpe_ann: float = 0.0, confidence: float = 0.95
) -> float:
    """Years of data needed before this Sharpe could clear the benchmark.

    When it exceeds the history actually available, the strategy has not yet
    earned the right to its own number -- regardless of how good that number is.
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype="float64")
    if r.size < 30:
        return float("nan")
    sharpe, skew, kurt = _moments(r)
    target = benchmark_sharpe_ann / np.sqrt(TRADING_DAYS)
    if not np.isfinite(sharpe) or sharpe <= target:
        return float("inf")
    variance = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe**2
    if variance <= 0:
        return float("nan")
    z = stats.norm.ppf(confidence)
    n = 1.0 + variance * (z / (sharpe - target)) ** 2
    return float(n / TRADING_DAYS)


def stationary_bootstrap_sharpe(
    returns: pd.Series | np.ndarray,
    n_boot: int = 2000,
    mean_block: int = 21,
    seed: int = 20260916,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval and one-sided p-value for the Sharpe ratio.

    Blocks of geometrically-distributed length are resampled with wraparound, so
    the resamples preserve the autocorrelation and volatility clustering that an
    i.i.d. bootstrap would destroy -- and destroying it would narrow the interval
    and overstate the significance.

    Returns ``(lo, hi, p_value)``: the 5th and 95th percentiles of the
    bootstrapped annualised Sharpe, and the fraction of resamples at or below
    zero, which is the probability of seeing this result if the strategy has no
    edge and the observed dependence structure is real.
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype="float64")
    n = r.size
    if n < 100:
        return (float("nan"),) * 3

    rng = np.random.default_rng(seed)
    p_restart = 1.0 / mean_block

    # Build all resample index paths at once: start a new random block whenever
    # the coin says so, otherwise step forward one position with wraparound.
    starts = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < p_restart
    restart[:, 0] = True
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)

    samples = r[idx]
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(sds > 0, means / sds, np.nan) * np.sqrt(TRADING_DAYS)
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size == 0:
        return (float("nan"),) * 3

    lo, hi = np.percentile(sharpes, [5, 95])
    return float(lo), float(hi), float((sharpes <= 0).mean())


def searched_sharpes(results: dict, specs: dict) -> list[float]:
    """Annualised Sharpes of the genuinely searched strategies.

    Anything in the ``Baseline`` family is a benchmark rather than a trial and is
    excluded; see :func:`assess` for why that distinction changes the answer.
    """
    return [
        r.metrics.sharpe
        for key, r in results.items()
        if getattr(specs.get(key), "family", None) != "Baseline"
    ]


def assess(
    returns: pd.Series,
    trial_sharpes_ann: list[float],
    n_boot: int = 2000,
) -> SignificanceResult:
    """Full significance assessment of one strategy within a search of many.

    ``trial_sharpes_ann`` must be every strategy *searched*, not the survivors.
    A deflation computed over the ones that worked is not a deflation.

    It must also exclude benchmarks. A buy-and-hold baseline was never a trial:
    nobody searched for "own the 2-year note", it is the thing the search is
    measured against. Including the baselines here is not conservative, it is
    wrong in a specific way -- deflation scales with the *spread* of Sharpes
    across trials, and an unlevered 2y position scoring 2.2 against a zero cash
    rate widens that spread enormously. On this library it moves the null's
    expected best Sharpe from 0.45 to 1.09, which would bury every real result
    under an artefact of how the benchmark is financed. Use
    :func:`searched_sharpes` to build the list.
    """
    r = returns.dropna()
    n = int(r.size)
    arr = np.asarray(r, dtype="float64")

    if n < 100:
        return SignificanceResult(
            float("nan"), n, *(float("nan"),) * 9, 0, "insufficient history"
        )

    sharpe_per, skew, kurt = _moments(arr)
    sharpe_ann = sharpe_per * np.sqrt(TRADING_DAYS)
    expected_max = expected_maximum_sharpe(trial_sharpes_ann)

    psr = probabilistic_sharpe(r, 0.0)
    dsr = probabilistic_sharpe(r, expected_max)
    mintrl = minimum_track_record_years(r, expected_max)
    lo, hi, pval = stationary_bootstrap_sharpe(r, n_boot=n_boot)

    # One sentence a reader can act on. The thresholds are the conventional 95%
    # and 90% levels; nothing here is tuned.
    if not np.isfinite(dsr):
        verdict = "not assessable"
    elif dsr > 0.95 and lo > 0:
        verdict = "survives deflation and the bootstrap"
    elif dsr > 0.90:
        verdict = "suggestive, short of conventional significance"
    elif sharpe_ann > expected_max:
        verdict = "beats the null's expected best, but not significantly"
    else:
        verdict = "indistinguishable from the best of a random search"

    return SignificanceResult(
        sharpe_ann=float(sharpe_ann),
        n_days=n,
        skew=skew,
        kurtosis=kurt,
        psr=float(psr),
        deflated_sharpe=float(dsr),
        expected_max_sharpe_ann=float(expected_max),
        n_trials=len([x for x in trial_sharpes_ann if np.isfinite(x)]),
        min_track_record_years=float(mintrl),
        bootstrap_lo=lo,
        bootstrap_hi=hi,
        bootstrap_p_value=pval,
        verdict=verdict,
    )
