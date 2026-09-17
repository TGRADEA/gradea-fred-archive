#!/usr/bin/env python
"""snapshot_chains.py — CLI: run one manual option-chain snapshot capture.

Usage:
    python scripts/snapshot_chains.py              # configured symbols
    python scripts/snapshot_chains.py SPY QQQ      # explicit symbols

Writes one immutable parquet file per symbol under
data/chain_snapshots/{SYMBOL}/{YYYY-MM-DD}/{HHMMSS}Z.parquet and one
manifest row (capture_kind='manual') in SQLite. Useful for a Windows Task
Scheduler fallback or a by-hand capture around an event — the backend's
scheduler and this CLI share the same write-once store, so they can never
overwrite each other (a same-second collision raises).

Exit code 0 if every symbol captured, 1 if any symbol failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make backend modules importable when run as a script
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import chain_snapshots
import storage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Symbols to capture (default: GRADEA_SNAPSHOT_SYMBOLS)",
    )
    args = parser.parse_args()

    storage.init_db()
    chain_snapshots.capture_all("manual", symbols=args.symbols or None)
    run = storage.last_snapshot_run()

    for sym in run["symbols_ok"]:
        print(f"[{sym}] ok")
    for item in run["symbols_failed"]:
        print(f"[{item['symbol']}] FAILED: {item['error']}", file=sys.stderr)
    print(
        f"run {run['run_id']}: {len(run['symbols_ok'])} ok, "
        f"{len(run['symbols_failed'])} failed, {run['rows_written']:,} rows, "
        f"{len(run['files_written'])} files"
    )
    return 1 if run["symbols_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
