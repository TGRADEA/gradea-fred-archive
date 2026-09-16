# Regime Desk — backtesting toolkit

A strategy and regime backtester built on the five FRED series this repository
mirrors, plus a self-contained HTML dashboard.

Nothing here writes to the archive. The five CSVs and `manifest.json` belong to
`mirror_fred.py`; this toolkit only reads them, and `python -m gradea_backtest
verify` will confirm their checksums still match before you trust anything below.

```bash
pip install -r requirements.txt

python -m gradea_backtest verify          # archive checksums, read-only
python -m gradea_backtest list            # strategies and regime families
python -m gradea_backtest compare         # every strategy, side by side
python -m gradea_backtest run nfci_duration --attribution
python -m gradea_backtest dashboard       # -> dashboard/regime-desk.html
```

## What it actually does

The archive holds no prices, so there is nothing to backtest until returns are
constructed. `returns.py` constructs them:

| Asset | Construction |
|---|---|
| `UST2Y`, `UST10Y` | Rolling par-bond total return. Duration and convexity are recomputed from each day's own yield, because a fixed duration of 8.0 overstates 1981 bond volatility by about a third. |
| `STEEPENER` | Duration-neutral long 2y / short 10y, **financed at the front end**. Reduces to the textbook `-(slope)/252 + D10 × d(slope)`. |
| `IG_XS`, `HY_XS` | Credit excess return: spread carry less spread duration times the OAS move. |
| `IG_TR`, `HY_TR` | The above plus a duration-matched Treasury leg. |

Seven regime families are then computed — curve slope and dynamics, NFCI level
and momentum, rate trend, credit stress, and a composite macro state — and every
strategy's returns are broken down within each.

## The two panels

This is the design decision everything else depends on.

**`raw`** carries each series on its observation date. It is the only thing used
to compute the return the portfolio earned.

**`pit`** carries each series shifted by its publication lag, so the value at day
*t* is the most recent one a trader could have seen by that morning. It is the
only thing used to compute a signal.

NFCI is the sharp edge: it is dated for the week ending Friday and published the
following Wednesday, so a naive alignment hands the strategy five days of
hindsight on financial conditions — precisely the days when conditions move. Its
lag here is four business days, and signals are then held one *further* day
before trading.

`tests/test_no_lookahead.py` checks these properties directly rather than
assuming them. The two that matter most:

- Every regime label and every strategy weight is recomputed on a panel whose
  future has been deleted, and must come out identical. A full-sample percentile
  masquerading as an expanding one fails this immediately.
- The steepener is checked against its textbook identity, because an unfinanced
  steepener appears to earn most of the yield level for taking no duration risk —
  which inverts the sign of the trade's carry and turns a hard trade into a free
  one.

## Ported from QuantGuild

Two modules come from Roman Paolucci's QuantGuild material
(github.com/romanmichaelpaolucci). That corpus is options- and equity-focused
teaching code — Black-Scholes, Greeks, stochastic processes, portfolio
construction — and most of it has no counterpart in a rates-and-credit panel
built on five FRED series. Two things did, and both closed a real gap here.

### `markov.py` — learned volatility regimes

A port of the `MarkovRegime` class from the "Markov Chain Regime Switching Bot"
lectures: three hidden states, Gaussian emissions, a sticky transition matrix,
and Bayesian forward filtering. It is the only regime family in this package
that is *learned* rather than declared — every other one encodes a threshold
someone chose.

Three adaptations were needed:

| | Original | Here | Why |
|---|---|---|---|
| Observation | intraday `(high-low)/close` from IB bars | absolute daily change in the 10y yield | the archive is daily |
| Calibration | once, on a block of history | refit every 252 days on expanding history | fitting once over a backtest sample puts 2008's volatility into 1985's labels |
| Inference | forward filter | forward filter, unchanged | Baum-Welch or Viterbi over the whole series would label beautifully and be untradeable |

The second is the load-bearing one. The original calibrates once and then runs
live, which is correct for a live bot — everything it was fitted on really is in
its past. The same code pointed at history is a lookahead.

The learned states separate cleanly: mean daily 10y moves of 1.4bp, 2.6bp and
7.8bp, and the filter independently flags March 1980, October 2008, March 2020
and September 2022 as high-volatility without being told those dates matter.

### `significance.py` — is it skill, or the best of eleven coin flips?

The QuantGuild corpus returns to this question in a dozen lectures. It was also
the largest hole here: this package reported a Sharpe ratio for each strategy and
let the reader assume the best one meant something.

It mostly does not. Deflating for multiple testing (Bailey & López de Prado)
against the nine genuinely searched strategies, the null's expected best Sharpe
is **0.45** — and only `nfci_duration`, at 0.65 with a deflated Sharpe of 0.93,
comes close to clearing it. Everything else is indistinguishable from a lucky
search.

Implemented: Probabilistic and Deflated Sharpe ratios, minimum track record
length, and a stationary block bootstrap (Politis & Romano) whose resamples keep
the autocorrelation of daily returns intact. PSR is verified calibrated against a
true null — uniform on [0,1], 4.3% false positives at the 95% threshold.

One methodological choice worth stating: **benchmarks are excluded from the trial
count.** Nobody searched for "own the 2-year note"; it is what the search is
measured against. Including the baselines moves the null's expected best from
0.45 to 1.09, because deflation scales with the spread of trial Sharpes and an
unlevered 2y position scoring 2.2 against a zero cash rate widens that spread
enormously. That would bury every real result under an artefact of how the
benchmark is financed.

## Known limitations

These are stated on the dashboard too, because they change how the numbers should
be read:

1. **Sharpe is computed against a zero cash rate.** The archive holds no
   short-rate series. On unlevered bond positions over a sample starting at
   double-digit yields, this overstates risk-adjusted return substantially.
2. **Roll-down is omitted** (understates steep-curve returns by roughly
   20–40bp/yr at a 100bp slope) and **defaults are omitted** (overstates high
   yield).
3. **Credit history is three years.** `BAMLC0A0CM` and `BAMLH0A0HYM2` start
   2023-08-01 and contain no default cycle.
4. **Strategy defaults were not tuned on this archive**, deliberately. The
   dashboard's parameter ranges exist to show how flat or sharp a strategy's
   response is, not to help you find the peak — each sweep spends some of the
   sample's ability to tell you anything.
5. **The deflation counts strategies, not parameter settings.** Sweeping a
   strategy's parameters and keeping the best is a second search that the
   deflated Sharpe here does not see, and it will overstate significance if you
   do it. Treat the parameter controls as diagnostics, not as an optimiser.

## Extending the panel

`python -m gradea_backtest ingest --list` ranks the FRED series worth adding by
how much each one removes a limitation above. The first three retire limitations
1 and 3 outright:

```bash
python -m gradea_backtest ingest DFF BAA10Y USREC
```

Fetched series land in `data/extended/` under their own manifest and are merged
at read time by `load_full_panel()`. The canonical archive is never touched, so
deleting that directory restores the original state exactly.

Ingestion needs outbound access to `fred.stlouisfed.org`. Where an egress policy
blocks it, everything else here still runs against the archived five.

## Layout

```
gradea_backtest/
  data.py        panel loading, publication lags, calendar alignment
  returns.py     yields and spreads -> tradeable return streams
  regimes.py     seven regime classifiers, all expanding-window
  strategies.py  strategy registry
  engine.py      execution, costs, attribution
  metrics.py     performance statistics
  markov.py      learned volatility regimes (ported, see above)
  significance.py  deflated Sharpe, PSR, block bootstrap
  ingest.py      FRED fetcher (writes only to data/extended/)
  export.py      payload for the dashboard
  cli.py         command line
dashboard/
  template.html  the dashboard, with a data placeholder
  regime-desk.html  built, self-contained
tests/
```
