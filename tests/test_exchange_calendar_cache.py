"""The XNYS calendar is memoized, and memoizing it changed no answer.

Packet F1 item 1, 2026-09-03. A py-spy trace of the desk's post-scan warehouse
build put **84% of that thread's samples** inside
``research_warehouse/exchange_calendar.py``: ``session_for`` ->
``trading_session`` -> ``is_trading_day`` -> ``holidays(year)``, recomputing
Easter and five nth-weekday walks once per M5 bar per occurrence. Benchmarked
in the desk venv: 20,000 ``session_for`` calls took 0.25 s uncached and 0.012 s
with ``lru_cache`` on ``holidays`` / ``half_days`` / ``trading_session`` (21x).

Two kinds of assertion live here and both are needed:

* the IDENTITY tests, which fail on the un-memoized code and are the
  fail-before-fix proof - a cache that is not there returns a new dict and a
  new ``TradingSession`` every call;
* the ANSWER tests, which must pass on both, because a cache that changes an
  answer is a re-dating of history, not a speed-up. The existing calendar
  tests in ``tests/test_warehouse_aggregate.py`` are the wider version of
  this and stay untouched.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_warehouse import exchange_calendar as xcal  # noqa: E402

UTC = timezone.utc


def test_holidays_and_half_days_are_computed_once_per_year():
    assert xcal.holidays(2026) is xcal.holidays(2026)
    assert xcal.half_days(2026) is xcal.half_days(2026)
    # Different years are still different answers.
    assert xcal.holidays(2026) is not xcal.holidays(2027)


def test_a_session_is_built_once_per_day():
    first = xcal.trading_session(date(2026, 9, 3))
    second = xcal.trading_session(date(2026, 9, 3))
    assert first is not None
    assert first is second
    # And a closed day's None is just as cached as a session.
    assert xcal.trading_session(date(2026, 11, 26)) is None


def test_the_cached_calendar_answers_a_holiday_a_half_day_and_a_weekend():
    # Thanksgiving 2026 - a full closure, so no session at all.
    assert xcal.is_trading_day(date(2026, 11, 26)) is False
    assert xcal.trading_session(date(2026, 11, 26)) is None

    # The Friday after Thanksgiving - a session that closes at 13:00 ET, and
    # whose extended session ends there too (there is no post-market).
    half = xcal.trading_session(date(2026, 11, 27))
    assert half is not None
    assert half.is_half_day is True
    assert xcal.is_half_day(date(2026, 11, 27)) is True
    assert half.rth_close_at == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)
    assert half.eth_close_at == half.rth_close_at

    # A weekend is closed without being a holiday.
    saturday = date(2026, 9, 5)
    assert saturday.weekday() == 5
    assert xcal.is_trading_day(saturday) is False
    assert xcal.trading_session(saturday) is None

    # And an ordinary session still runs 09:30-16:00 ET with a 04:00-20:00 ETH
    # wrap, in UTC, on the calendar version it was built under.
    plain = xcal.trading_session(date(2026, 9, 3))
    assert plain is not None
    assert plain.is_half_day is False
    assert plain.rth_open_at == datetime(2026, 9, 3, 13, 30, tzinfo=UTC)
    assert plain.rth_close_at == datetime(2026, 9, 3, 20, 0, tzinfo=UTC)
    assert plain.eth_open_at == datetime(2026, 9, 3, 8, 0, tzinfo=UTC)
    assert plain.eth_close_at == datetime(2026, 9, 4, 0, 0, tzinfo=UTC)
    assert plain.calendar_version == xcal.CALENDAR_VERSION


def test_session_for_a_moment_reuses_the_days_session():
    open_moment = datetime(2026, 9, 3, 14, 0, tzinfo=UTC)
    later = datetime(2026, 9, 3, 19, 0, tzinfo=UTC)
    assert xcal.session_for(open_moment) is xcal.session_for(later)
    assert xcal.session_for(open_moment) is xcal.trading_session(date(2026, 9, 3))


def test_a_named_calendar_is_cached_separately_from_the_default():
    default = xcal.trading_session(date(2026, 9, 3))
    named = xcal.trading_session(date(2026, 9, 3), calendar="XNAS")
    assert default is not None and named is not None
    assert named is not default
    assert named.exchange_calendar == "XNAS"
    assert named is xcal.trading_session(date(2026, 9, 3), calendar="XNAS")
