-- Applied to Supabase project rzabvzdsbwrjzloqvrnw on 2026-09-17 as two
-- migrations, chain_snapshot_mirror_extensions and
-- chain_snapshot_mirror_schema (FOUNDATION-CHAIN-SNAPSHOT-MIRROR-1). Recorded
-- here so the mirror schema is reviewable in git rather than only living in
-- the Supabase dashboard.
--
-- The parquet files under gradea-backend/data/chain_snapshots/ remain the
-- source of truth (INV-SNAP-1); this is a read mirror pushed by
-- gradea-backend/chain_snapshot_mirror.py. Supabase does not offer
-- TimescaleDB, so time partitioning is native range partitioning managed by
-- pg_partman, with pg_cron running its maintenance.

-- ── migration 1: extensions ────────────────────────────────────────────────
create schema if not exists partman;
create extension if not exists pg_partman schema partman;
create extension if not exists pg_cron;

-- ── migration 2: schema ────────────────────────────────────────────────────
create table public.option_chain_snapshots (
  snapshot_ts_utc timestamptz not null,
  capture_kind    text        not null check (capture_kind in ('interval', 'near_close', 'manual')),
  symbol          text        not null,
  spot            double precision,
  expiry          date        not null,
  dte             smallint    not null,
  strike          double precision not null,
  side            char(1)     not null check (side in ('C', 'P')),
  bid             double precision,
  ask             double precision,
  last            double precision,
  mid             double precision,
  bid_size        integer,
  ask_size        integer,
  volume          integer,
  open_interest   integer,
  delta           double precision,
  gamma           double precision,
  theta           double precision,
  vega            double precision,
  rho             double precision,
  iv              double precision,
  quote_time_utc  timestamptz,
  schema_version  smallint    not null default 1,
  primary key (symbol, snapshot_ts_utc, expiry, strike, side)
) partition by range (snapshot_ts_utc);

comment on table public.option_chain_snapshots is
  'Point-in-time Schwab option-chain observations mirrored from gradea-backend/data/chain_snapshots parquet files (schema v1). Append-only: rows are never revised. Column "side" is the parquet column "right" (C/P), renamed because RIGHT is an SQL reserved word. iv is a fraction (Schwab percent / 100); a Schwab -999 sentinel is stored as 0.0, matching the parquet.';

-- Track one expiry (or one contract) through time.
create index option_chain_snapshots_expiry_idx
  on public.option_chain_snapshots (symbol, expiry, snapshot_ts_utc);

-- Mirror ledger: one row per parquet file fully uploaded. The mirror resumes
-- from this table, and dashboard "latest chain" lookups resolve the newest
-- capture here instead of scanning the partitioned table.
create table public.option_chain_snapshot_files (
  symbol          text        not null,
  snapshot_ts_utc timestamptz not null,
  capture_kind    text        not null,
  source_path     text        not null,
  row_count       integer     not null,
  sha256          text        not null,
  mirrored_at     timestamptz not null default now(),
  primary key (symbol, snapshot_ts_utc)
);

comment on table public.option_chain_snapshot_files is
  'Ledger of parquet snapshot files fully mirrored into option_chain_snapshots. A file is listed only after every one of its rows was accepted.';

-- Monthly partitions, two months pre-created, default partition catches
-- anything outside the managed range so an insert never fails on a gap.
select partman.create_parent(
  p_parent_table => 'public.option_chain_snapshots',
  p_control      => 'snapshot_ts_utc',
  p_interval     => '1 month',
  p_premake      => 2
);

update partman.part_config
   set infinite_time_partitions = true
 where parent_table = 'public.option_chain_snapshots';

select cron.schedule(
  'partman-maintenance',
  '7 * * * *',
  $$call partman.run_maintenance_proc()$$
);

-- Dashboard read: the chain for one symbol as observed at or before p_at
-- (default: the newest capture). Callable through PostgREST as
-- POST /rest/v1/rpc/option_chain_at {"p_symbol": "SPY"}.
create function public.option_chain_at(p_symbol text, p_at timestamptz default now())
returns setof public.option_chain_snapshots
language sql
stable
as $$
  select s.*
    from public.option_chain_snapshots s
   where s.symbol = upper(p_symbol)
     and s.snapshot_ts_utc = (
           select max(f.snapshot_ts_utc)
             from public.option_chain_snapshot_files f
            where f.symbol = upper(p_symbol)
              and f.snapshot_ts_utc <= p_at)
   order by s.expiry, s.strike, s.side;
$$;

-- Dashboard read: newest mirrored capture per symbol.
create view public.option_chain_snapshot_latest as
  select symbol,
         max(snapshot_ts_utc) as snapshot_ts_utc,
         count(*)             as captures,
         sum(row_count)       as rows_mirrored
    from public.option_chain_snapshot_files
   group by symbol;
