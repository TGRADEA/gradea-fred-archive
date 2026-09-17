# backend/ — Schwab chain-snapshot capture and Supabase mirror

The data pipeline behind the weekly options work: capture full Schwab option
chains on a schedule, keep them as immutable point-in-time parquet files, and
mirror those files into Supabase Postgres where a dashboard can query them.

## Provenance

Every module here is a copy from `TGRADEA/gradea-trading-platform`
(`gradea-backend/`, main @ `9619e13`) plus the FOUNDATION-CHAIN-SNAPSHOT-MIRROR-1
slice, kept file-for-file and name-for-name so the two trees stay diffable.
The flat layout is deliberate (that repo's INV-1). What is **not** copied is the
FastAPI app: the loops that ran inside its lifespan run here as
`scripts/snapshot_service.py`, and the browser OAuth routes become
`scripts/schwab_login.py`.

| Module | Role |
|---|---|
| `schwab_client.py`, `sources/schwab.py` | Schwab token custody, OAuth, rate-gated retrying client |
| `chain.py` | Schwab chain normalisation (`-999` sentinel, IV percent → fraction) |
| `chain_snapshots.py` | Append-only parquet store + NYSE-hours scheduler |
| `chain_snapshot_mirror.py`, `sources/supabase.py` | Ledger-driven upload to Supabase |
| `storage.py`, `universe.py`, `settings.py`, `calendar_svc.py`, `cache.py` | SQLite manifest, symbol universes, config, NYSE calendar, TTL cache |

## Setup

```bash
cd backend
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp env.example .env                              # fill in SCHWAB_* (and SUPABASE_* for the mirror)
python scripts/schwab_login.py                   # one-time; re-run every 7 days (Schwab's cap)
```

## Run

```bash
python scripts/snapshot_service.py               # capture loop + mirror loop, until Ctrl-C
python scripts/snapshot_chains.py SPY QQQ        # one manual capture, now
python scripts/mirror_chain_snapshots.py         # one mirror catch-up run (backfill)
python scripts/mirror_chain_snapshots.py --status
```

Captures land in `data/chain_snapshots/{SYMBOL}/{YYYY-MM-DD}/{HHMMSS}Z.parquet`
(gitignored) with a run manifest in `gradea.db`. The mirror never deletes or
rewrites a local file and never issues an UPDATE or DELETE upstream.

## What Schwab returns for Greeks

Per contract in the chain response: `delta`, `gamma`, `theta`, `vega`, `rho`,
and `volatility` (implied vol as a percent; `18.3` means 18.3%). Missing values
arrive as `-999`. Alongside: `bid`, `ask`, `last`, `mark`, `bidSize`, `askSize`,
`totalVolume`, `openInterest`, `theoreticalOptionValue`, `timeValue`,
`intrinsicValue`, `daysToExpiration`, `quoteTimeInLong`, `inTheMoney`.
`chain.py` maps `-999` to `0.0` and stores IV as a fraction; the parquet schema
v1 keeps `delta gamma theta vega rho iv`.

## Supabase

`migrations/chain_snapshot_mirror_schema.sql` is the applied DDL:
`public.option_chain_snapshots` (monthly pg_partman partitions; Supabase has no
TimescaleDB), the ledger `public.option_chain_snapshot_files`, the as-of read
`public.option_chain_at(symbol, at)` and the view
`public.option_chain_snapshot_latest`. RLS is enabled with no policies, so only
the service-role key (server-side) can read or write; a browser dashboard needs
either read policies or a backend route in front of it.

Dashboard query, through PostgREST:

```http
POST /rest/v1/rpc/option_chain_at
{"p_symbol": "SPY"}                                   # newest capture
{"p_symbol": "SPY", "p_at": "2026-09-17T19:50:00Z"}   # as observed at/before
```

## Tests

```bash
cd backend && pytest
```

Tests are the platform's, minus the ones that exercised its HTTP routes or
files outside this directory. Root `pytest` runs the backtesting toolkit only.
