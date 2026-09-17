#!/usr/bin/env python
"""snapshot_service.py — run the chain-snapshot capture and Supabase mirror loops.

Usage:
    python scripts/snapshot_service.py

What gradea-trading-platform starts inside its FastAPI lifespan, as a
standalone process: `chain_snapshots.schedule_loop()` captures every
configured symbol's option chain each GRADEA_SNAPSHOT_INTERVAL_MIN minutes
during NYSE regular hours plus one near-close capture per session, and
`chain_snapshot_mirror.mirror_loop()` pushes new parquet files to Supabase
every GRADEA_SNAPSHOT_MIRROR_INTERVAL_MIN minutes (only when
GRADEA_SNAPSHOT_MIRROR_ENABLED=1 and the Supabase service key are set).

Needs a Schwab token (scripts/schwab_login.py) for capture. Runs until
interrupted; Ctrl-C stops both loops. Nothing here ever deletes or rewrites a
snapshot file (INV-SNAP-1).
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import chain_snapshot_mirror
import chain_snapshots
import storage
from settings import settings

log = logging.getLogger("snapshot_service")


async def _main() -> None:
    storage.init_db()
    symbols, source, universe = chain_snapshots.snapshot_symbols()
    log.info(
        "capture: %d symbols from %s%s, every %d min, max DTE %d, store %s",
        len(symbols),
        source,
        f" (universe {universe})" if universe else "",
        settings.snapshot_interval_min,
        settings.snapshot_max_dte,
        chain_snapshots.SNAPSHOT_ROOT,
    )
    if not symbols:
        log.warning("capture disabled: GRADEA_SNAPSHOT_SYMBOLS is empty")
    if chain_snapshot_mirror.configured():
        log.info("mirror: enabled, every %d min", settings.snapshot_mirror_interval_min)
    else:
        log.warning(
            "mirror: off (set GRADEA_SNAPSHOT_MIRROR_ENABLED=1, SUPABASE_URL, "
            "SUPABASE_SERVICE_KEY to enable)"
        )
    await asyncio.gather(
        chain_snapshots.schedule_loop(),
        chain_snapshot_mirror.mirror_loop(),
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
