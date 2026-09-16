"""Tests for the properties that decide whether any number in this repo means anything.

A backtest can be wrong in two ways. It can have a bug, which shows up as an
absurd number and gets noticed. Or it can leak future information, which shows up
as an *attractive* number and does not get noticed. These tests target the second
kind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gradea_backtest.data import SERIES, _available_from, load_panel, read_series
from gradea_backtest.engine import run_backtest
from gradea_backtest.regimes import build_regimes, expanding_percentile, expanding_zscore
from gradea_backtest.returns import build_asset_returns, duration_convexity, treasury_total_return
from gradea_backtest.strategies import REGISTRY, available_strategies


@pytest.fixture(scope="module")
def panel():
    return load_panel()


@pytest.fixture(scope="module")
def returns(panel):
    return build_asset_returns(panel.raw)


@pytest.fixture(scope="module")
def regimes(panel):
    return build_regimes(panel.pit)


# ---------------------------------------------------------------------------
# The publication-lag guarantee.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("series_id", list(SERIES))
def test_pit_value_was_published_before_it_is_used(panel, series_id):
    """Every value in `pit` at day t must appear in the raw series at a date whose
    publication date is on or before t. This is the guarantee the whole toolkit
    rests on, checked directly rather than assumed."""
    spec = SERIES[series_id]
    raw_obs = read_series(series_id)
    available = _available_from(raw_obs, spec.pub_lag_bdays)

    pit = panel.pit[series_id].dropna()
    # Sample across the whole span rather than testing 16k points one by one.
    for t in pit.index[::97]:
        published = available.loc[:t]
        assert not published.empty, f"{series_id}: value carried at {t} with nothing published yet"
        assert pit.loc[t] == published.iloc[-1], (
            f"{series_id} at {t}: pit carries {pit.loc[t]}, "
            f"latest published is {published.iloc[-1]} (from {published.index[-1]})"
        )


def test_nfci_pit_never_leads_raw(panel):
    """NFCI is the dangerous one: weekly, with a release lag that spans the days
    financial conditions actually move. The lagged panel must never carry a value
    the observation panel has not yet reached."""
    both = pd.DataFrame({"raw": panel.raw["NFCI"], "pit": panel.pit["NFCI"]}).dropna()
    raw_obs = read_series("NFCI")

    for t, row in both.iloc[::53].iterrows():
        # The pit value must be an observation dated at least 4 business days back.
        matches = raw_obs[raw_obs == row["pit"]]
        assert not matches.empty
        assert matches.index.min() <= t - pd.offsets.BDay(SERIES["NFCI"].pub_lag_bdays)


def test_pit_differs_from_raw_often_enough_to_matter(panel):
    """A lag that never bites would mean the shift silently failed."""
    for series_id in ("NFCI", "DGS10"):
        both = pd.DataFrame({"raw": panel.raw[series_id], "pit": panel.pit[series_id]}).dropna()
        differ = (both["raw"] != both["pit"]).mean()
        assert differ > 0.20, f"{series_id}: pit and raw differ on only {differ:.1%} of days"


# ---------------------------------------------------------------------------
# Expanding statistics must not see their own future.
# ---------------------------------------------------------------------------

def test_expanding_percentile_matches_truncated_recomputation():
    """The value at day t must be identical whether or not days after t exist.
    This is the test that catches a full-sample percentile masquerading as an
    expanding one."""
    rng = np.random.default_rng(20260916)
    s = pd.Series(rng.normal(size=1500), index=pd.bdate_range("2015-01-01", periods=1500))

    full = expanding_percentile(s, warmup=250)
    for cut in (400, 700, 1100):
        truncated = expanding_percentile(s.iloc[:cut], warmup=250)
        pd.testing.assert_series_equal(
            full.iloc[:cut], truncated, check_names=False,
            obj=f"expanding percentile changed when data after position {cut} was removed",
        )


def test_expanding_percentile_respects_warmup():
    s = pd.Series(np.arange(500, dtype="float64"))
    out = expanding_percentile(s, warmup=100)
    assert out.iloc[:100].isna().all()
    assert out.iloc[100:].notna().all()
    # A strictly increasing input must rank at the top of its own history throughout.
    assert (out.iloc[100:] == 1.0).all()


def test_expanding_zscore_matches_truncated_recomputation():
    rng = np.random.default_rng(7)
    s = pd.Series(rng.normal(size=900), index=pd.bdate_range("2018-01-01", periods=900))
    full = expanding_zscore(s, warmup=200)
    truncated = expanding_zscore(s.iloc[:600], warmup=200)
    pd.testing.assert_series_equal(full.iloc[:600], truncated, check_names=False)


def test_regime_labels_are_stable_under_truncation(panel):
    """Regime labels for the past must not change when the future arrives."""
    cut = panel.pit.index.get_loc(pd.Timestamp("2015-01-02"))
    full = build_regimes(panel.pit)
    early = build_regimes(panel.pit.iloc[: cut + 1])

    for family in early.columns:
        a = full[family].iloc[: cut + 1]
        b = early[family]
        # credit_stress uses an expanding percentile with a short warmup; the rest
        # use fixed thresholds. All of them must be truncation-invariant.
        mismatch = (a.fillna("~") != b.fillna("~")).sum()
        assert mismatch == 0, f"{family}: {mismatch} labels changed when the future was removed"


# ---------------------------------------------------------------------------
# The return model.
# ---------------------------------------------------------------------------

def test_duration_falls_as_yields_rise():
    """Modified duration of a par bond is monotonically decreasing in yield --
    the property that makes a fixed duration assumption wrong across this sample."""
    yields = pd.Series([0.5, 1.0, 2.0, 5.0, 10.0, 16.0])
    d10, c10 = duration_convexity(yields, 10.0)
    assert (d10.diff().dropna() < 0).all()
    assert (c10.diff().dropna() < 0).all()
    # Sanity anchors against textbook values.
    assert 7.5 < d10.iloc[3] < 8.0     # 10y par bond at 5% -> ~7.8
    assert 9.5 < d10.iloc[0] < 10.0    # at 0.5% -> just under its maturity


def test_two_year_duration_is_near_two():
    d2, _ = duration_convexity(pd.Series([4.0]), 2.0)
    assert 1.85 < d2.iloc[0] < 1.95


def test_bond_returns_match_known_years(returns):
    """The 10y total return has documented values for years everyone remembers.
    If the model cannot reproduce 2022, nothing built on it is worth reading."""
    r = returns["UST10Y"]
    y2022 = (1 + r["2022"]).prod() - 1
    y2008 = (1 + r["2008"]).prod() - 1
    assert -0.20 < y2022 < -0.12, f"2022 10y return {y2022:.1%} outside the known -16% ballpark"
    assert 0.14 < y2008 < 0.27, f"2008 10y return {y2008:.1%} outside the known +20% ballpark"


def test_steepener_matches_textbook_identity(panel, returns):
    """A duration-neutral steepener earns -(slope)/252 + D10 * d(slope).

    Getting this wrong is not a small error: an unfinanced steepener appears to
    earn most of the yield level for taking no duration risk, which inverts the
    sign of the trade's carry and turns a hard trade into a free one."""
    raw = panel.raw
    slope = raw["DGS10"] - raw["DGS2"]
    d10, _ = duration_convexity(raw["DGS10"].shift(1), 10.0)
    textbook = -(slope.shift(1) / 100.0) / 252 + d10 * (slope.diff() / 100.0)

    both = pd.DataFrame({"model": returns["STEEPENER"], "textbook": textbook}).dropna()
    assert both["model"].corr(both["textbook"]) > 0.999
    assert (both["model"] - both["textbook"]).abs().mean() < 1e-4


def test_steepener_carry_sign_flips_with_the_curve(panel, returns):
    """Positive slope must cost carry to hold; inversion must pay it."""
    slope = (panel.raw["DGS10"] - panel.raw["DGS2"]).shift(1)
    r = returns["STEEPENER"]
    inverted = r[slope < 0].mean()
    positive = r[slope > 0.5].mean()
    assert inverted > 0 > positive


def test_returns_use_raw_not_lagged_data(panel):
    """Returns must respond to the day's own yield move. Building them off the
    lagged panel would shift every return forward and quietly decorrelate the
    strategy from the thing it is supposed to be trading."""
    from_raw = treasury_total_return(panel.raw["DGS10"], 10.0)
    from_pit = treasury_total_return(panel.pit["DGS10"], 10.0)
    assert not np.allclose(
        from_raw.dropna().to_numpy()[-2000:], from_pit.dropna().to_numpy()[-2000:]
    )


# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------

def test_buy_and_hold_reproduces_the_asset(panel, returns, regimes):
    """A constant weight of 1.0 with no turnover must return the asset itself."""
    spec = REGISTRY["hold_10y"]
    result = run_backtest("t", spec.build(panel.pit, regimes), returns, regimes)
    expected = returns["UST10Y"].reindex(result.gross_returns.index)
    pd.testing.assert_series_equal(
        result.gross_returns, expected, check_names=False, rtol=1e-12, atol=1e-12
    )
    # Net differs from gross on exactly one day: putting the position on costs
    # 1.5bp, and nothing trades thereafter.
    diff = (result.net_returns - result.gross_returns).abs()
    assert (diff > 1e-12).sum() == 1
    assert diff.iloc[0] == pytest.approx(1.5e-4, rel=1e-9)
    # One entry trade and no rebalancing thereafter.
    assert result.turnover.sum() == pytest.approx(1.0, rel=1e-12)


def test_execution_lag_shifts_the_position(returns, regimes, panel):
    """The weight held on day t must be the weight decided on day t-1."""
    idx = returns.index
    weights = pd.DataFrame({"UST10Y": np.zeros(len(idx))}, index=idx)
    mark = idx[-500]
    weights.loc[mark, "UST10Y"] = 1.0

    result = run_backtest("pulse", weights, returns, regimes, execution_lag=1)
    held = result.weights["UST10Y"]
    next_day = idx[idx.get_loc(mark) + 1]
    # The engine trims to the window where the position is actually on, which for
    # a single-day pulse is the single day after the signal -- not the signal day.
    assert list(held.index) == [next_day]
    assert held.loc[next_day] == 1.0

    unshifted = run_backtest("pulse0", weights, returns, regimes, execution_lag=0)
    assert list(unshifted.weights.index) == [mark]


def test_costs_reduce_returns(panel, returns, regimes):
    spec = REGISTRY["duration_trend"]
    weights = spec.build(panel.pit, regimes, **spec.defaults())
    cheap = run_backtest("cheap", weights, returns, regimes, cost_bps={"UST10Y": 0.0})
    dear = run_backtest("dear", weights, returns, regimes, cost_bps={"UST10Y": 20.0})
    assert dear.metrics.cagr < cheap.metrics.cagr
    assert dear.metrics.ann_turnover == pytest.approx(cheap.metrics.ann_turnover, rel=1e-9)


def test_entry_cost_is_charged(returns, regimes):
    """Putting the initial position on is a trade and must be paid for."""
    idx = returns.index[-1000:]
    weights = pd.DataFrame({"UST10Y": np.ones(len(idx))}, index=idx)
    result = run_backtest("hold", weights, returns.loc[idx], regimes, cost_bps={"UST10Y": 100.0})
    assert result.costs.iloc[0] > 0


def test_every_strategy_runs_and_stays_finite(panel, returns, regimes):
    for spec in available_strategies(panel.pit):
        result = run_backtest(spec.key, spec.build(panel.pit, regimes, **spec.defaults()),
                              returns, regimes, description=spec.description)
        assert result.net_returns.notna().all()
        assert np.isfinite(result.net_returns).all()
        assert result.metrics.n_days > 100, f"{spec.key} produced almost no history"
        # A daily return beyond +/-25% on unlevered rates exposure is a bug, not a market move.
        assert result.net_returns.abs().max() < 0.25, f"{spec.key} produced an implausible daily return"


def test_strategies_never_touch_the_raw_panel(panel, regimes):
    """Hand every strategy a pit panel whose future is blanked out. The weights it
    produces for the surviving dates must be unchanged -- proof by construction
    that nothing downstream of `pit` is reading ahead."""
    cut = pd.Timestamp("2010-01-04")
    truncated_pit = panel.pit.loc[:cut]
    truncated_regimes = build_regimes(truncated_pit)

    for spec in available_strategies(panel.pit):
        full = spec.build(panel.pit, regimes, **spec.defaults()).loc[:cut]
        part = spec.build(truncated_pit, truncated_regimes, **spec.defaults())
        common = full.index.intersection(part.index)
        if len(common) < 100:
            continue
        pd.testing.assert_frame_equal(
            full.loc[common].fillna(0.0), part.loc[common].fillna(0.0),
            check_names=False, rtol=1e-9, atol=1e-9,
            obj=f"{spec.key} weights changed when future data was removed",
        )
