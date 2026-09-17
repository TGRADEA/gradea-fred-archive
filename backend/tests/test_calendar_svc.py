"""FOUNDATION-CALENDAR-1 — the shared NYSE calendar service.

All offline: pandas-market-calendars ships its holiday data locally, so the
real calendar is used with fixed historical dates whose answers can never
change.
"""

from __future__ import annotations

from datetime import UTC, date

import calendar_svc
import chain_snapshots


def test_singleton_is_lazy_and_cached(monkeypatch):
    monkeypatch.setattr(calendar_svc, "_NYSE", None)
    first = calendar_svc.get_nyse()
    assert first is not None
    assert calendar_svc.get_nyse() is first


def test_trading_days_known_week():
    # Week of 2026-01-05 (Mon-Fri), no holidays: exactly 5 sessions.
    days = calendar_svc.trading_days(date(2026, 1, 5), date(2026, 1, 9))
    assert days == [
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 1, 9),
    ]


def test_trading_days_excludes_holiday():
    # 2026-01-01 (New Year's Day, Thursday) is a NYSE holiday.
    days = calendar_svc.trading_days(date(2025, 12, 31), date(2026, 1, 2))
    assert date(2026, 1, 1) not in days
    assert date(2025, 12, 31) in days
    assert date(2026, 1, 2) in days


def test_session_bounds_regular_day_utc():
    bounds = calendar_svc.session_bounds(date(2026, 1, 5))
    assert bounds is not None
    open_utc, close_utc = bounds
    assert open_utc.tzinfo is not None and close_utc.tzinfo is not None
    # 09:30 ET / 16:00 ET on a winter day == 14:30 / 21:00 UTC.
    assert open_utc.astimezone(UTC).hour == 14
    assert open_utc.astimezone(UTC).minute == 30
    assert close_utc.astimezone(UTC).hour == 21


def test_session_bounds_none_on_weekend_and_holiday():
    assert calendar_svc.session_bounds(date(2026, 1, 3)) is None  # Saturday
    assert calendar_svc.session_bounds(date(2026, 1, 1)) is None  # holiday


def test_is_trading_day():
    assert calendar_svc.is_trading_day(date(2026, 1, 5)) is True
    assert calendar_svc.is_trading_day(date(2026, 1, 3)) is False


def test_chain_snapshots_alias_points_at_service():
    # The seam karsan.py depends on (_snap._session_bounds) is the shared fn.
    assert chain_snapshots._session_bounds is calendar_svc.session_bounds


def test_session_bounds_ms_matches_utc_bounds():
    bounds = calendar_svc.session_bounds_ms(date(2026, 1, 5))
    assert bounds is not None
    assert bounds[1] - bounds[0] == 390 * 60_000
