# gradea_v4

The GradeA v4 line. Three things live here:

- **FRED archive** (repo root) — the off-box mirror of the GradeA FRED series
  archive (`gradea-backend/data/fred/`), established by
  **FOUNDATION-FRED-MIRROR-1**. Everything below this section is about it.
- **Backtesting toolkit** (`gradea_backtest/`, `dashboard/`) — see
  [BACKTESTING.md](BACKTESTING.md).
- **Backend pipeline** (`backend/`) — Schwab option-chain snapshot capture and
  the Supabase mirror, copied file-for-file from `gradea-trading-platform`.
  See [backend/README.md](backend/README.md).

This repository was `gradea-fred-archive` until 2026-09-17; GitHub redirects
the old name. The archive rules below (append-only, never force-push) are
unchanged by the rename.

## Why this repo exists

FRED serves the two ICE BofA OAS series on a rolling **~3-year window**.
Observations older than that window are unreachable at any `cosd`, so once they
pass out of it, the GradeA archive is the only copy in existence. That archive
directory is gitignored and lives on one machine. This repo is the second copy.

`DGS2`, `DGS10`, and `NFCI` carry full history at every request and are freely
re-fetchable; they are mirrored here for convenience, not necessity. The
irreplaceable content is `BAMLH0A0HYM2` and `BAMLC0A0CM`, accreting at roughly
**262 observations per year each**.

## Contents

One CSV per series plus `manifest.json` holding a SHA-256, observation count,
and date span for each.

```
observation_date,value
1962-01-02,4.06
```

Dates are ISO and strictly ascending. There are no gaps encoded as `.` and no
null values — FRED's no-observation rows are dropped at the network boundary.

## Restoring

```bash
git clone git@github.com:TGRADEA/gradea_v4.git ~/gradea_v4
export GRADEA_FRED_MIRROR_PATH="$HOME/gradea_v4"
cd /path/to/gradea-trading-platform/gradea-backend
python scripts/mirror_fred.py --verify-all     # must exit 0 before restoring
python scripts/mirror_fred.py --restore-all
```

`--restore-all` refuses to run if verification fails, and refuses to shrink an
archive that holds more observations than this mirror. Pass `--allow-shrink`
only if you intend a rollback.

## Daily maintenance

```bash
python scripts/refresh_fred.py --all \
  && python scripts/mirror_fred.py --export-all --commit --push
```

A day with no new observations produces no commit. Commits are named for the
series that changed, so `git log -p DGS10.csv` is a usable revision history:
FRED revises modeled and preliminary series after first publication, and each
revision appears here as a diff.

## Backtesting toolkit

`gradea_backtest/` is a regime-aware backtester built on these five series, with
a self-contained dashboard in `dashboard/`. It is strictly additive: it reads the
archive and never writes to it, and `python -m gradea_backtest verify` checks the
CSVs against `manifest.json` without modifying anything. See
[BACKTESTING.md](BACKTESTING.md).

## Do not force-push

This repo is the only backup. History is the vintage record; rewriting it
destroys data that FRED will not serve again.
