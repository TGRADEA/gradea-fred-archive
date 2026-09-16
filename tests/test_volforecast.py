"""Tests for the ported volatility-forecast benchmark.

The evaluation machinery is only worth having if it can be trusted to *not*
declare a winner. Most of these tests check that: that QLIKE is zero at a
perfect forecast and undefined at an impossible one, that the DM test does not
reject when two models are the same, and that the Model Confidence Set keeps
everything it cannot distinguish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gradea_backtest.data import load_panel
from gradea_backtest.markov import markov_regime
from gradea_backtest.returns import treasury_total_return
from gradea_backtest.volforecast import (
    benchmark,
    build_targets_and_features,
    diebold_mariano,
    dm_matrix,
    model_confidence_set,
    qlike,
    realized_variance,
    semivariances,
    walk_forward_forecasts,
)


@pytest.fixture(scope="module")
def returns():
    panel = load_panel()
    return treasury_total_return(panel.pit["DGS10"], 10.0).dropna()


@pytest.fixture(scope="module")
def forecasts(returns):
    panel = load_panel()
    labels, _ = markov_regime(panel.pit)
    return walk_forward_forecasts(returns, labels)


# ---------------------------------------------------------------------------
# QLIKE
# ---------------------------------------------------------------------------

def test_qlike_is_zero_only_at_a_perfect_forecast():
    assert qlike([1.0], [1.0])[0] == pytest.approx(0.0, abs=1e-12)
    assert qlike([2.5], [2.5])[0] == pytest.approx(0.0, abs=1e-12)
    assert qlike([1.0], [1.2])[0] > 0
    assert qlike([1.0], [0.8])[0] > 0


def test_qlike_punishes_under_prediction_harder():
    """The asymmetry is the reason for using QLIKE on variance: being short
    volatility when it arrives costs more than being long when it doesn't."""
    under = qlike([1.0], [0.5])[0]
    over = qlike([1.0], [2.0])[0]
    assert under > over


def test_qlike_is_undefined_not_floored_for_impossible_forecasts():
    """Flooring a non-positive forecast hands a broken model a finite loss and
    can sneak it into the confidence set."""
    out = qlike([1.0, 1.0, 1.0], [0.0, -1.0, np.nan])
    assert np.isnan(out).all()


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------

def test_dm_does_not_reject_for_identical_forecasts():
    rng = np.random.default_rng(4)
    loss = rng.gamma(2, 0.4, 1500)
    result = diebold_mariano(loss, loss.copy())
    assert not np.isfinite(result.stat) or result.p_value > 0.99


def test_dm_rejects_a_clear_difference_and_names_the_winner():
    rng = np.random.default_rng(5)
    good = rng.gamma(2, 0.3, 1500)
    bad = good + rng.gamma(2, 0.3, 1500)
    result = diebold_mariano(good, bad, names=("good", "bad"))
    assert result.p_value < 0.01
    assert result.better == "good"


def test_dm_matrix_is_antisymmetric():
    rng = np.random.default_rng(6)
    losses = pd.DataFrame({k: rng.gamma(2, s, 800) for k, s in [("a", 0.3), ("b", 0.4), ("c", 0.5)]})
    m = dm_matrix(losses)
    for i in m.index:
        for j in m.columns:
            if i != j:
                assert m.loc[i, j] == pytest.approx(-m.loc[j, i], rel=1e-9)


def test_dm_widens_its_errors_for_overlapping_forecasts():
    """A longer horizon means overlapping targets and dependent losses, so the
    statistic must shrink toward zero rather than stay put."""
    rng = np.random.default_rng(8)
    a = rng.gamma(2, 0.3, 2000)
    b = a + rng.gamma(2, 0.05, 2000)
    assert abs(diebold_mariano(a, b, horizon=10).stat) < abs(diebold_mariano(a, b, horizon=1).stat)


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------

def test_mcs_keeps_every_model_it_cannot_distinguish():
    rng = np.random.default_rng(9)
    losses = pd.DataFrame({f"m{i}": rng.gamma(2, 0.4, 1000) for i in range(4)})
    table = model_confidence_set(losses, alpha=0.10, n_boot=500)
    assert table["in_set"].all(), "identical models were eliminated from the confidence set"


def test_mcs_isolates_a_clear_winner():
    rng = np.random.default_rng(10)
    losses = pd.DataFrame(
        {"winner": rng.gamma(2, 0.15, 1200), **{f"m{i}": rng.gamma(2, 0.5, 1200) for i in range(3)}}
    )
    table = model_confidence_set(losses, alpha=0.10, n_boot=500)
    assert table.loc["winner", "in_set"]
    assert not table.drop(index="winner")["in_set"].any()


def test_mcs_refuses_a_sample_too_small_to_mean_anything():
    losses = pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4]})
    with pytest.raises(ValueError):
        model_confidence_set(losses)


# ---------------------------------------------------------------------------
# Features and targets
# ---------------------------------------------------------------------------

def test_semivariance_actually_splits_on_sign():
    """The bug this catches: squaring first destroys the sign, leaving the
    downside series identically zero and SHAR a duplicate of HAR."""
    r = pd.Series([-0.02, 0.01, -0.005, 0.03])
    neg, pos = semivariances(r)
    assert neg.gt(0).sum() == 2
    assert pos.gt(0).sum() == 2
    assert (neg + pos).round(12).equals(realized_variance(r).round(12))


def test_features_never_look_past_their_own_date(returns):
    """Every feature at date t must be computable from returns through t, and
    the target must lie strictly in the future."""
    target, har, shar = build_targets_and_features(returns, horizon=5)
    cut = returns.index[3000]
    trimmed_target, trimmed_har, _ = build_targets_and_features(returns.loc[:cut], horizon=5)
    common = har.dropna().index.intersection(trimmed_har.dropna().index)
    assert len(common) > 1000
    pd.testing.assert_frame_equal(har.loc[common], trimmed_har.loc[common], check_names=False)
    # The last `horizon` targets of the trimmed series need days beyond the cut.
    assert trimmed_target.dropna().index[-1] < cut


def test_forecasts_are_reproducible_from_the_past_alone(returns):
    """Truncate the future and the past forecasts must not move. An expanding
    refit that quietly used the whole sample would fail here."""
    panel = load_panel()
    labels, _ = markov_regime(panel.pit)
    cut = returns.index[6000]

    full = walk_forward_forecasts(returns, labels, min_train=1260, refit_every=63)
    part = walk_forward_forecasts(returns.loc[:cut], labels.loc[:cut], min_train=1260, refit_every=63)

    common = full.dropna().index.intersection(part.dropna().index)
    assert len(common) > 2000
    cols = [c for c in full.columns if c != "actual"]
    pd.testing.assert_frame_equal(
        full.loc[common, cols], part.loc[common, cols], check_names=False, rtol=1e-9, atol=1e-12
    )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_benchmark_runs_and_ranks_sensibly(forecasts):
    result = benchmark(forecasts, n_boot=400)
    mcs = result["mcs"]
    assert result["n_aligned"] > 5000
    assert mcs["in_set"].any(), "the confidence set cannot be empty"
    # HAR has outlasted a great deal of cleverness; if a random walk beats it
    # something upstream is broken.
    assert mcs.loc["HAR-RV", "mean_qlike"] < mcs.loc["RW", "mean_qlike"]
    assert mcs.loc["HAR-RV", "mean_qlike"] < mcs.loc["AR1-RV", "mean_qlike"]


def test_every_model_produces_forecasts(forecasts):
    for col in forecasts.columns:
        assert forecasts[col].notna().sum() > 5000, f"{col} produced almost nothing"
