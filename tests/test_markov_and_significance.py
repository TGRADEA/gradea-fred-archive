"""Tests for the two modules ported from the QuantGuild corpus.

The Markov classifier gets the same treatment as every other regime family: its
labels must be reproducible from data that existed when they were assigned. That
matters more here than elsewhere, because a learned classifier has somewhere to
hide a lookahead that a fixed threshold does not -- the fitted emission means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from gradea_backtest.data import load_panel
from gradea_backtest.markov import (
    MIN_CALIBRATION,
    RMS_FLOOR,
    HMMParams,
    filter_step,
    fit_hmm,
    markov_regime,
    volatility_observation,
)
from gradea_backtest.significance import (
    assess,
    expected_maximum_sharpe,
    minimum_track_record_years,
    probabilistic_sharpe,
    stationary_bootstrap_sharpe,
)


@pytest.fixture(scope="module")
def panel():
    return load_panel()


@pytest.fixture(scope="module")
def regimes(panel):
    """Fitting the HMM across sixty years is the expensive part of this file;
    every test that only needs the output shares one run."""
    return markov_regime(panel.pit)


# ---------------------------------------------------------------------------
# Markov regime
# ---------------------------------------------------------------------------

def test_markov_labels_survive_deleting_the_future(panel, regimes):
    """The load-bearing test. Refit on expanding history is what makes this
    classifier honest; fitting once on the whole sample would put the 2008 and
    2020 volatility spikes into the model's 1985 labels."""
    cut = pd.Timestamp("2005-01-03")
    full, _ = regimes
    early, _ = markov_regime(panel.pit.loc[:cut])

    common = full.loc[:cut].index.intersection(early.index)
    assert len(common) > 5000
    a, b = full.loc[common].fillna("~"), early.loc[common].fillna("~")
    mismatch = int((a != b).sum())
    assert mismatch == 0, f"{mismatch} Markov labels changed once the future was removed"


def test_belief_is_a_probability_distribution(regimes):
    _, beliefs = regimes
    rows = beliefs.dropna()
    assert len(rows) > 5000
    assert np.allclose(rows.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    assert (rows.to_numpy() >= 0).all()


def test_learned_states_are_ordered_by_volatility(panel, regimes):
    """Low must actually be calmer than High once the labels reach real data --
    the sort inside fit_hmm orders the parameters, this checks the ordering
    survives filtering and relabelling."""
    labels, _ = regimes
    obs = volatility_observation(panel.pit)
    means = {s: obs[labels == s].mean() for s in ("Low vol", "Medium vol", "High vol")}
    assert means["Low vol"] < means["Medium vol"] < means["High vol"]


def test_known_turmoil_is_classified_as_high_volatility(regimes):
    """Volcker, the GFC, the COVID crash and the 2022 inflation shock. A regime
    model that misses all four is not modelling regimes."""
    labels, _ = regimes
    for window in ("1980-03", "2008-10", "2020-03", "2022-09"):
        counts = labels[window].value_counts()
        assert counts.get("High vol", 0) > counts.get("Low vol", 0), (
            f"{window} was not predominantly high-volatility: {counts.to_dict()}"
        )


def test_fit_refuses_a_window_too_short_to_estimate():
    assert fit_hmm(np.random.default_rng(0).normal(size=MIN_CALIBRATION - 1)) is None


def test_fit_returns_states_sorted_by_emission_mean():
    """Unsorted states permute freely between refits, and a strategy keyed to
    state 0 would flip meaning at every recalibration."""
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(-9, 0.4, 500), rng.normal(-7, 0.4, 500), rng.normal(-5, 0.4, 500)])
    p = fit_hmm(x)
    assert p is not None
    assert np.all(np.diff(p.mean) > 0)
    assert np.allclose(np.exp(p.log_A).sum(axis=1), 1.0)


def test_fit_recovers_known_states():
    """Three well-separated regimes should come back close to where they were."""
    rng = np.random.default_rng(5)
    truth = [-9.0, -7.0, -5.0]
    blocks = [rng.normal(m, 0.3, 400) for m in truth]
    p = fit_hmm(np.concatenate(blocks))
    assert p is not None
    for got, want in zip(p.mean, truth):
        assert abs(got - want) < 0.6, f"recovered {p.mean}, expected near {truth}"


def test_baum_welch_does_not_reduce_the_likelihood():
    """EM is monotone in the likelihood by construction; a drop means the M step
    is wrong. Checked by fitting with successively more iterations."""
    from gradea_backtest.markov import _forward_backward, _gaussian_logpdf

    rng = np.random.default_rng(8)
    x = np.concatenate([rng.normal(-8, 0.5, 600), rng.normal(-6, 0.5, 600)])
    lls = []
    for iters in (1, 3, 8, 20):
        p = fit_hmm(x, max_iter=iters, tol=0.0)
        lls.append(_forward_backward(_gaussian_logpdf(x, p.mean, p.sd), p)[2])
    assert all(b >= a - 1e-6 for a, b in zip(lls, lls[1:])), lls


def test_filter_handles_an_impossible_observation():
    """An observation far outside every state makes all likelihoods underflow.
    The filter must fall back to its prediction rather than emit NaN for the
    rest of the sample."""
    k = 3
    p = HMMParams(
        log_pi=np.full(k, -np.log(k)),
        log_A=np.full((k, k), -np.log(k)),
        mean=np.array([-9.0, -7.0, -5.0]),
        sd=np.array([0.3, 0.3, 0.3]),
    )
    belief = filter_step(np.full(k, 1 / k), -1e6, p)
    assert np.isfinite(belief).all()
    assert belief.sum() == pytest.approx(1.0)


def test_observation_is_deterministic_and_rarely_floored():
    """The RMS window exists partly to keep the quotation floor from binding; if
    it bound often, a state would fit the quotation grid instead of the market."""
    panel = load_panel()
    a = volatility_observation(panel.pit)
    b = volatility_observation(panel.pit)
    pd.testing.assert_series_equal(a, b)
    at_floor = (a.dropna() <= np.log(RMS_FLOOR) + 1e-9).mean()
    assert at_floor < 0.02, f"{at_floor:.1%} of observations sit on the floor"


def test_a_quiet_year_is_not_labelled_turbulent(regimes):
    """The mirror of the turmoil test. A model that calls everything high
    volatility would pass that one and fail this."""
    labels, _ = regimes
    counts = labels["2017"].value_counts()
    assert counts.get("Low vol", 0) > counts.get("High vol", 0)


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def test_psr_is_calibrated_under_the_null():
    """With no edge, PSR should be uniform on [0,1] -- so it crosses 0.95 about
    5% of the time. A significance test that is not calibrated is decoration."""
    values = [
        probabilistic_sharpe(pd.Series(np.random.default_rng(s).normal(0, 0.01, 2520)))
        for s in range(200)
    ]
    values = np.array(values)
    assert 0.42 < values.mean() < 0.58
    assert (values > 0.95).mean() < 0.11
    assert stats.kstest(values, "uniform").pvalue > 0.01


def test_psr_detects_a_real_edge():
    rng = np.random.default_rng(7)
    edge = pd.Series(rng.normal(1.0 / np.sqrt(252) * 0.01, 0.01, 5000))
    assert probabilistic_sharpe(edge) > 0.99


def test_expected_max_sharpe_rises_with_trials_and_spread():
    narrow = [0.5, 0.52, 0.48, 0.51, 0.49]
    wide = [0.0, 0.5, 1.0, 1.5, 2.0]
    assert expected_maximum_sharpe(wide) > expected_maximum_sharpe(narrow)

    spread = list(np.linspace(0, 1.5, 5))
    many = list(np.linspace(0, 1.5, 50))
    assert expected_maximum_sharpe(many) > expected_maximum_sharpe(spread)

    assert expected_maximum_sharpe([0.7]) == 0.0  # a single trial is not a search


def test_deflation_is_harsher_than_the_raw_test():
    """Deflated Sharpe must never exceed the undeflated PSR: correcting for
    multiple testing can only lower confidence."""
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.6 / np.sqrt(252) * 0.01, 0.01, 4000))
    result = assess(r, [0.2, 0.4, 0.6, 0.8, 1.0], n_boot=400)
    assert result.deflated_sharpe <= result.psr + 1e-12
    assert result.expected_max_sharpe_ann > 0


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.9 / np.sqrt(252) * 0.01, 0.01, 3000))
    point = float(r.mean() / r.std(ddof=1) * np.sqrt(252))
    lo, hi, p = stationary_bootstrap_sharpe(r, n_boot=800)
    assert lo < point < hi
    assert p < 0.05


def test_bootstrap_does_not_reject_a_true_null():
    rng = np.random.default_rng(5)
    lo, hi, p = stationary_bootstrap_sharpe(pd.Series(rng.normal(0, 0.01, 3000)), n_boot=800)
    assert lo < 0 < hi
    assert 0.05 < p < 0.95


def test_min_track_record_grows_as_the_bar_rises():
    rng = np.random.default_rng(13)
    r = pd.Series(rng.normal(0.8 / np.sqrt(252) * 0.01, 0.01, 4000))
    assert minimum_track_record_years(r, 0.0) < minimum_track_record_years(r, 0.5)


def test_short_history_is_reported_not_guessed():
    result = assess(pd.Series(np.random.default_rng(1).normal(0, 0.01, 50)), [0.5, 0.7])
    assert result.verdict == "insufficient history"
    assert not np.isfinite(result.deflated_sharpe)
