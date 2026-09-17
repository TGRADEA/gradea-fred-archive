#!/usr/bin/env python
"""mirror_chain_snapshots.py — CLI: push local chain snapshots to Supabase.

Usage:
    python scripts/mirror_chain_snapshots.py             # mirror everything pending
    python scripts/mirror_chain_snapshots.py --limit 20  # at most 20 files this run
    python scripts/mirror_chain_snapshots.py --status    # no network; local + last run

Requires GRADEA_SNAPSHOT_MIRROR_ENABLED=1, SUPABASE_URL and
SUPABASE_SERVICE_KEY in gradea-backend/.env. The backend's lifespan loop and
this CLI share the same ledger, so running both is safe: a file the other
already finished is skipped, and a file both attempt lands once (row inserts
ignore duplicates; the ledger row is keyed by symbol + timestamp).

Exit code 0 if every pending file mirrored, 1 if any file failed or the run
could not start, 2 if the mirror is not configured.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make backend modules importable when run as a script
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import chain_snapshot_mirror as mirror


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="max files to upload this run")
    parser.add_argument("--status", action="store_true", help="print status and exit")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(mirror.status(), indent=2, default=str))
        return 0

    if not mirror.configured():
        print(
            "mirror not configured: set GRADEA_SNAPSHOT_MIRROR_ENABLED=1, "
            "SUPABASE_URL and SUPABASE_SERVICE_KEY in gradea-backend/.env",
            file=sys.stderr,
        )
        return 2

    summary = mirror.mirror_pending(limit=args.limit)
    for item in summary["failed"]:
        print(f"[{item['path']}] FAILED: {item['error']}", file=sys.stderr)
    if summary["error"]:
        print(f"run failed: {summary['error']}", file=sys.stderr)
        return 1
    print(
        f"scanned {summary['scanned']} files, {summary['pending']} pending, "
        f"uploaded {summary['uploaded_files']} files / {summary['uploaded_rows']:,} rows, "
        f"{len(summary['failed'])} failed"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
