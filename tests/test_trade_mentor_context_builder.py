"""Builder-owned edge checks for Trade Mentor context time bounds."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from zoneinfo import ZoneInfo


def _sessions_ending(last: date, count: int) -> list[date]:
    import market_calendar

    rows: list[date] = []
    cursor = last
    while len(rows) < count:
        if market_calendar.is_session(cursor):
            rows.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(rows))


def _daily(days: list[date]) -> list[dict]:
    return [
        {"dt": day.isoformat(), "open": 99 + index, "high": 101 + index,
         "low": 98 + index, "close": 100 + index}
        for index, day in enumerate(days)
    ]


def _vxx(context):
    return next(row for row in context["readings"] if row["symbol"] == "VXX")


def test_daily_context_uses_the_actual_early_close_and_refuses_a_missing_session():
    """Friday 14:00 ET is complete on the day after Thanksgiving, not 16:00."""
    from trade_mentor_context import build_context

    market = ZoneInfo("America/New_York")
    now = datetime(2026, 11, 27, 14, 0, tzinfo=market)
    days = _sessions_ending(date(2026, 11, 27), 21)
    measured = build_context(now=now, m5_bars={}, d1_bars={"VXX": _daily(days)})
    assert _vxx(measured)["d1_status"] == "measured"
    assert _vxx(measured)["d1_as_of"] == "2026-11-27"

    gappy = build_context(now=now, m5_bars={}, d1_bars={"VXX": _daily(days[:9] + days[10:])})
    assert _vxx(gappy)["d1_status"] == "unavailable"
    assert "gap" in _vxx(gappy)["d1_reason"]
