# gradea-fred-archive

Off-box mirror of the GradeA FRED series archive
(`gradea-backend/data/fred/`). Established by **FOUNDATION-FRED-MIRROR-1**.

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
git clone git@github.com:TGRADEA/gradea-fred-archive.git ~/gradea-fred-archive
export GRADEA_FRED_MIRROR_PATH="$HOME/gradea-fred-archive"
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

## Do not force-push

This repo is the only backup. History is the vintage record; rewriting it
destroys data that FRED will not serve again.
