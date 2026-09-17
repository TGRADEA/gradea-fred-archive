"""calendar_svc — the single owner of the NYSE trading calendar.

FOUNDATION-CALENDAR-1. Before this module, two files each built their own
``pandas_market_calendars`` NYSE calendar (cycle.py eagerly at import,
chain_snapshots.py lazily), and every future consumer — Engine v2, seasonal
layers, point-in-time data (BT-F5) — would have added a third. One calendar,
built once, consumed everywhere.

Rules this module enforces (INV-CAL-1 in _map/invariants.md):

- ``mcal.get_calendar`` is called exactly once in the tree: here.
- ``import pandas_market_calendars`` appears nowhere else in gradea-backend.
- Consumers import the helpers below; they never touch the raw calendar
  object unless they truly need the schedule DataFrame (then: ``get_nyse()``).

Timezone note: NYSE sessions are defined in America/New_York and
pandas-market-calendars returns tz-aware UTC timestamps for opens/closes.
``session_bounds`` passes those through unchanged (UTC). App-boundary date
logic elsewhere stays America/Chicago per repo convention — this module
deals only in exchange sessions, where ET/UTC are the honest units.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pandas_market_calendars as mcal

_NYSE = None


def get_nyse():
    """Lazily-built NYSE calendar singleton (local data, no network).

    Lazy so that importing this module — or anything that imports it — never
    pays the calendar construction cost until a session question is actually
    asked. cycle.py used to build this eagerly at import; that eagerness was
    an accident of history, not a requirement.
    """
    global _NYSE
    if _NYSE is None:
        _NYSE = mcal.get_calendar("NYSE")
    return _NYSE


def trading_days(start: date, end: date) -> list[date]:
    """NYSE trading days in [start, end], inclusive, as plain dates.

    Lifted verbatim from cycle._trading_days (FOUNDATION-CALENDAR-1).
    pandas 4 deprecated slicing with datetime.date — cast to Timestamp
    explicitly.
    """
    sched = get_nyse().schedule(
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(end),
    )
    return [d.date() for d in sched.index]


def session_bounds(day: date) -> tuple[datetime, datetime] | None:
    """(open_utc, close_utc) for a trading day, or None on holidays/weekends.

    Lifted verbatim from chain_snapshots._session_bounds
    (FOUNDATION-CALENDAR-1). Timestamps are tz-aware UTC, exactly as
    pandas-market-calendars returns them.
    """
    sched = get_nyse().schedule(start_date=day, end_date=day)
    if sched.empty:
        return None
    row = sched.iloc[0]
    return (row["market_open"].to_pydatetime(), row["market_close"].to_pydatetime())


def session_bounds_ms(day: date) -> tuple[int, int] | None:
    """NYSE open/close epoch milliseconds for ``day``, or None when closed.

    This is the cash-session seam for price history.  The service remains the
    sole owner of pandas-market-calendars; callers receive neutral integers.
    """
    bounds = session_bounds(day)
    if bounds is None:
        return None
    return tuple(int(value.astimezone(UTC).timestamp() * 1000) for value in bounds)


def is_trading_day(day: date) -> bool:
    """True when NYSE holds a regular session on ``day``."""
    return session_bounds(day) is not None
