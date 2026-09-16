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
  ingest.py      FRED fetcher (writes only to data/extended/)
  export.py      payload for the dashboard
  cli.py         command line
dashboard/
  template.html  the dashboard, with a data placeholder
  regime-desk.html  built, self-contained
tests/
```
