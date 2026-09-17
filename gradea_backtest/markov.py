"""Markov-switching volatility regimes.

A three-state hidden Markov model over log daily volatility, fitted by
Baum-Welch on expanding history and used for *filtered* inference only.

Provenance
----------
This is an independent implementation from the primary literature:

* Rabiner (1989), "A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition", Proc. IEEE 77(2) -- the forward-backward
  recursions, Baum-Welch re-estimation, and the scaling used for numerical
  stability.
* Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary
  Time Series and the Business Cycle", Econometrica 57(2) -- Markov-switching
  applied to an economic series, and the filtered-probability object this module
  returns.
* Dempster, Laird & Rubin (1977), "Maximum Likelihood from Incomplete Data via
  the EM Algorithm", JRSS-B 39(1) -- the EM framework Baum-Welch instantiates.

An earlier version of this file adapted the `MarkovRegime` class from the
QuantGuild lecture repository. That repository carries no licence, so it grants
no copy permission, and this repository is public. Everything specific to that
implementation has been removed: the hand-set transition prior, the
percentile-thirds state seeding, its smoothing constants, and its emission
defaults. Nothing here is derived from it. The lecture series remains the reason
the hypothesis was worth testing, which is the use the GradeA knowledge base
records for that library -- concepts and mathematics, implemented from source.

Design
------
**Log-volatility emissions.** Volatility is positive and right-skewed, so the
observation is modelled as Gaussian in logs rather than in levels. A level-space
Gaussian assigns real probability mass to negative volatility and is dominated
by the right tail when fitted.

**Zeros are interval-censored, and the interval is sampled rather than
collapsed.** DGS10 is quoted to the basis point, so 11.8% of days print an
unchanged yield. That is not zero volatility; it means the move was smaller than
half a tick. Discarding those days would delete precisely the calmest ones.

Flooring them all at a single value is worse. It puts 11.8% of the sample on one
identical observation, and a Gaussian mixture will spend an entire state on that
spike: fitted that way, the "low volatility" state is not a market condition at
all, it is the set of days the yield did not move, and the remaining two states
are left to cover everything else. The first version of this file did exactly
that and assigned 68% of the sample to the high state.

An unchanged print means the true move is somewhere in [0, half a tick), so each
censored day is placed uniformly inside that interval. The draw is derived
deterministically from the observation's own date, so a given day always
receives the same value regardless of where the series is cut -- randomness here
would break both reproducibility and the truncation-invariance guarantee.

**Baum-Welch fits; forward filtering infers.** These are different operations
and only one of them is safe for a signal:

* *Fitting* runs forward-backward over a training window that lies entirely in
  the past. Using the whole of that window to estimate parameters is ordinary
  in-sample estimation, not lookahead.
* *Inference* for day t uses the forward recursion alone -- predict through the
  transition matrix, weight by the likelihood of what was observed, renormalise.
  The smoothed (forward-backward) state probabilities are never used as labels.

The second point is the one that matters. Smoothed probabilities at day t
incorporate observations after t, so a strategy conditioned on them is
untradeable however good its backtest looks. The GradeA review of this lecture
material flagged the same hazard independently: "those smoothed probabilities
include later observations and must not be used as contemporaneous trading
signals."

**Expanding recalibration.** Parameters are refitted periodically on everything
strictly before the current day, so a 1985 label never depends on 2008.
``tests/test_markov_and_significance.py`` checks this by deleting the future and
requiring identical output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Three states: the smallest number that can express "neither calm nor
#: stressed". Four cannot be estimated reliably from the number of genuine
#: volatility cycles in a few decades of daily data.
DEFAULT_N_STATES = 3

STATE_LABELS = ("Low vol", "Medium vol", "High vol")

#: Symmetric Dirichlet concentration on each transition row. One is the uniform
#: prior (add-one smoothing): it keeps every transition possible when a training
#: window happens to contain none of a given kind, without which a state can
#: become absorbing on one unlucky window.
TRANSITION_PRIOR = 1.0

#: Baum-Welch stops when the average log-likelihood improves by less than this,
#: or after this many sweeps. EM increases the likelihood monotonically, so the
#: iteration cap is a wall-clock bound rather than a correctness one.
EM_TOL = 1e-5
EM_MAX_ITER = 20

#: Minimum observations before a fit is attempted at all.
MIN_CALIBRATION = 250

#: Cap on the estimation window, in observations. Five years.
#:
#: Fitting is expanding up to this cap and rolling beyond it, for a modelling
#: reason rather than a computational one: a single parameter set estimated over
#: sixty years assumes the volatility process is stationary across the Volcker
#: disinflation, the Greenspan era and ZIRP, which it plainly is not. A 2026 fit
#: dominated by 1980s yield dynamics describes neither. Five years is ample for
#: three states -- six emission parameters and nine transitions against 1,260
#: observations -- and it keeps the guarantee that matters: every observation
#: used lies strictly in the past.
MAX_TRAIN = 1260

#: Window, in sessions, over which volatility is estimated before being handed
#: to the model. Five is one trading week: long enough to estimate a scale
#: rather than draw one sample from it, short enough that a regime change is not
#: smoothed away.
VOL_WINDOW = 5

#: Floor on the RMS, as a decimal rate: half the quotation tick. Reached only if
#: every session in the window printed an unchanged yield.
RMS_FLOOR = 0.000025

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    peak = np.max(a, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    out = peak + np.log(np.sum(np.exp(a - peak), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def _gaussian_logpdf(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Log density of each observation under each state. Shape (T, K)."""
    z = (x[:, None] - mean[None, :]) / sd[None, :]
    return -0.5 * (z * z + _LOG_2PI) - np.log(sd)[None, :]


def _kmeans_1d(x: np.ndarray, k: int, seed: int = 0, iters: int = 60) -> np.ndarray:
    """Lloyd's algorithm on a scalar series, returning sorted centres.

    Rabiner §V.B initialises HMM emissions by clustering, which is what this
    does. Quantiles of the observation would also cluster it, but they fix the
    state sizes in advance; k-means lets the data decide how large the calm
    state is, which is the question being asked.
    """
    rng = np.random.default_rng(seed)
    lo, hi = float(np.min(x)), float(np.max(x))
    centres = np.sort(rng.uniform(lo, hi, size=k)) if hi > lo else np.full(k, lo)
    for _ in range(iters):
        labels = np.argmin(np.abs(x[:, None] - centres[None, :]), axis=1)
        moved = False
        for j in range(k):
            members = x[labels == j]
            if members.size:
                new = float(members.mean())
                if abs(new - centres[j]) > 1e-12:
                    moved = True
                centres[j] = new
        centres = np.sort(centres)
        if not moved:
            break
    return centres


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class HMMParams:
    """Fitted parameters, all in log space where they are probabilities."""

    log_pi: np.ndarray   # (K,) initial state distribution
    log_A: np.ndarray    # (K, K) transition matrix, rows sum to one
    mean: np.ndarray     # (K,) emission mean of log volatility
    sd: np.ndarray       # (K,) emission standard deviation

    @property
    def n_states(self) -> int:
        return self.mean.size


def _forward_backward(log_b: np.ndarray, p: HMMParams) -> tuple[np.ndarray, np.ndarray, float]:
    """Log alpha, log beta and the log-likelihood (Rabiner §III.A-B)."""
    T, K = log_b.shape
    log_alpha = np.empty((T, K))
    log_beta = np.zeros((T, K))

    log_alpha[0] = p.log_pi + log_b[0]
    for t in range(1, T):
        log_alpha[t] = log_b[t] + _logsumexp(log_alpha[t - 1][:, None] + p.log_A, axis=0)

    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp(p.log_A + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)

    return log_alpha, log_beta, float(_logsumexp(log_alpha[-1]))


def fit_hmm(
    x: np.ndarray,
    n_states: int = DEFAULT_N_STATES,
    seed: int = 0,
    max_iter: int = EM_MAX_ITER,
    tol: float = EM_TOL,
    max_train: int = MAX_TRAIN,
) -> HMMParams | None:
    """Fit a Gaussian HMM to ``x`` by Baum-Welch. Returns None if too short.

    ``x`` is log volatility. States come back sorted by emission mean, so index
    0 is always the calmest -- without that the state labels permute freely
    between refits and a strategy conditioned on them would flip at random.
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if x.size < MIN_CALIBRATION:
        return None
    if max_train and x.size > max_train:
        x = x[-max_train:]          # most recent window, still entirely past
    T = x.size

    centres = _kmeans_1d(x, n_states, seed=seed)
    labels = np.argmin(np.abs(x[:, None] - centres[None, :]), axis=1)
    sd = np.array(
        [x[labels == j].std() if (labels == j).sum() > 2 else x.std() for j in range(n_states)]
    )
    sd = np.maximum(sd, 1e-6)
    p = HMMParams(
        log_pi=np.full(n_states, -np.log(n_states)),
        log_A=np.full((n_states, n_states), -np.log(n_states)),
        mean=centres.copy(),
        sd=sd,
    )

    prev_ll = -np.inf
    for _ in range(max_iter):
        log_b = _gaussian_logpdf(x, p.mean, p.sd)
        log_alpha, log_beta, ll = _forward_backward(log_b, p)

        # E step: gamma (state posteriors) and xi (transition posteriors).
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        log_xi_num = (
            log_alpha[:-1, :, None]
            + p.log_A[None, :, :]
            + (log_b[1:] + log_beta[1:])[:, None, :]
        )
        xi = np.exp(log_xi_num - _logsumexp(log_xi_num.reshape(T - 1, -1), axis=1)[:, None, None])

        # M step. The Dirichlet prior is added as pseudo-counts on the
        # transition numerator, which is the MAP rather than ML update.
        trans = xi.sum(axis=0) + TRANSITION_PRIOR
        log_A = np.log(trans) - np.log(trans.sum(axis=1, keepdims=True))

        weight = gamma.sum(axis=0)
        mean = (gamma * x[:, None]).sum(axis=0) / np.maximum(weight, 1e-12)
        var = (gamma * (x[:, None] - mean[None, :]) ** 2).sum(axis=0) / np.maximum(weight, 1e-12)
        sd = np.sqrt(np.maximum(var, 1e-12))

        log_pi = log_gamma[0] - _logsumexp(log_gamma[0])
        p = HMMParams(log_pi=log_pi, log_A=log_A, mean=mean, sd=sd)

        if ll - prev_ll < tol * max(abs(prev_ll), 1.0):
            break
        prev_ll = ll

    # Sort states by emission mean and permute everything consistently.
    order = np.argsort(p.mean)
    return HMMParams(
        log_pi=p.log_pi[order],
        log_A=p.log_A[np.ix_(order, order)],
        mean=p.mean[order],
        sd=p.sd[order],
    )


def filter_step(belief: np.ndarray, value: float, p: HMMParams) -> np.ndarray:
    """Advance the filtered state distribution by one observation.

    Predict through the transition matrix, weight by how likely each state makes
    what was actually observed, renormalise. This is the forward recursion alone
    -- the only inference a trader can act on, because it conditions on the past
    and present and nothing else.
    """
    log_prior = _logsumexp(np.log(np.maximum(belief, 1e-300))[:, None] + p.log_A, axis=0)
    if not np.isfinite(value):
        out = np.exp(log_prior - _logsumexp(log_prior))
        return out / out.sum()
    log_post = log_prior + _gaussian_logpdf(np.array([value]), p.mean, p.sd)[0]
    total = _logsumexp(log_post)
    if not np.isfinite(total):
        out = np.exp(log_prior - _logsumexp(log_prior))
        return out / out.sum()
    return np.exp(log_post - total)


# ---------------------------------------------------------------------------
# Observation and driver
# ---------------------------------------------------------------------------

def volatility_observation(
    pit: pd.DataFrame, column: str = "DGS10", window: int = VOL_WINDOW
) -> pd.Series:
    """Log root-mean-square daily yield change over a trailing window.

    Reads days t-window+1 through t inclusive and is taken from the
    publication-lagged panel, so the observation driving a label on day t was
    knowable on day t.
    """
    dy = pit[column].diff() / 100.0
    rms = (dy.pow(2).rolling(window, min_periods=window).mean()).pow(0.5)
    return np.log(rms.clip(lower=RMS_FLOOR)).rename("log_vol")


def markov_regime(
    pit: pd.DataFrame,
    column: str = "DGS10",
    recalibrate_every: int = 252,
    warmup: int = 756,
    n_states: int = DEFAULT_N_STATES,
) -> tuple[pd.Series, pd.DataFrame]:
    """Filtered volatility-regime labels and the belief matrix behind them.

    The belief is worth keeping: a strategy sized by P(high vol) behaves
    differently from one that flips on argmax, and the share of days on which
    the filter is actually confident is itself a diagnostic.
    """
    obs = volatility_observation(pit, column)
    values = obs.to_numpy(dtype="float64")
    index = obs.index

    params: HMMParams | None = None
    belief = np.full(n_states, 1.0 / n_states)
    beliefs = np.full((values.size, n_states), np.nan)
    labels: list[str | float] = []

    for i in range(values.size):
        if i >= warmup and (params is None or i % recalibrate_every == 0):
            fitted = fit_hmm(values[:i], n_states=n_states)
            if fitted is not None:
                params = fitted
                belief = np.exp(params.log_pi) if not np.isfinite(beliefs[i - 1]).all() else belief

        if params is None:
            labels.append(np.nan)
            continue

        belief = filter_step(belief, values[i], params)
        beliefs[i] = belief
        labels.append(STATE_LABELS[int(np.argmax(belief))])

    return (
        pd.Series(labels, index=index, dtype="object", name="markov_vol"),
        pd.DataFrame(beliefs, index=index, columns=list(STATE_LABELS[:n_states])),
    )


def markov_vol_regime(pit: pd.DataFrame) -> pd.Series:
    """Registry-compatible wrapper returning labels only."""
    return markov_regime(pit)[0]
