"""NYSE session calendar.

Session identity is the key every overnight artifact and ledger row is filed
under. Before this module there was no calendar at all: the runner did weekday
arithmetic, so a Saturday run filed its work under Saturday, and three `ok`
rows sat in the ledger claiming coverage of 2026-08-08 -- a date on which the
exchange never opened (Sol 5.6 verification review, item 2).

The holiday sets below are checked against the exchange's published closures,
not against the implementation, so a rule that drifts is caught rather than
confirmed.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")


#: Published NYSE full-day closures. Independent of the implementation.
PUBLISHED = {
    2024: [
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
        "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    ],
    2025: [
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
        "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    ],
    2026: [
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
        "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    ],
    2027: [
        "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
        "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    ],
}


@pytest.mark.parametrize("year", sorted(PUBLISHED))
def test_holidays_match_the_published_nyse_calendar(year):
    from market_calendar import holidays_for_year

    assert sorted(d.isoformat() for d in holidays_for_year(year)) == PUBLISHED[year]


def test_observance_rules_move_the_right_way():
    from market_calendar import holidays_for_year

    # 2026-07-04 is a Saturday, so the exchange closes Friday the 3rd.
    assert date(2026, 7, 3) in holidays_for_year(2026)
    assert date(2026, 7, 4) not in holidays_for_year(2026)
    # 2027-07-04 is a Sunday, so it moves to Monday the 5th.
    assert date(2027, 7, 5) in holidays_for_year(2027)
    # A Saturday New Year's Day is simply not observed -- moving it would
    # close the previous trading year a day early.
    assert date(2022, 1, 1).weekday() == 5
    assert date(2021, 12, 31) not in holidays_for_year(2022)


def test_weekends_and_holidays_are_not_sessions():
    from market_calendar import is_session

    assert is_session(date(2026, 8, 7)) is True      # Friday
    assert is_session(date(2026, 8, 8)) is False     # Saturday
    assert is_session(date(2026, 8, 9)) is False     # Sunday
    assert is_session(date(2026, 8, 10)) is True     # Monday
    assert is_session(date(2026, 11, 26)) is False   # Thanksgiving
    assert is_session(date(2026, 11, 27)) is True    # the half-day is a session


def test_the_calendar_refuses_to_extrapolate_past_its_horizon():
    from market_calendar import SessionCalendarError, VALID_THROUGH, is_session

    with pytest.raises(SessionCalendarError, match="refusing to extrapolate"):
        is_session(date(VALID_THROUGH.year + 1, 6, 1))
    with pytest.raises(SessionCalendarError):
        is_session(date(1999, 6, 1))


def test_last_completed_session_waits_for_the_close():
    from market_calendar import last_completed_session

    friday = date(2026, 8, 7)
    # Mid-session Friday: Friday is not complete, so Thursday is the answer.
    assert last_completed_session(datetime(2026, 8, 7, 15, 0, tzinfo=ET)) == date(2026, 8, 6)
    # Exactly at the close counts as complete.
    assert last_completed_session(datetime(2026, 8, 7, 16, 0, tzinfo=ET)) == friday
    # Friday evening, all weekend, and Monday pre-close all answer Friday.
    assert last_completed_session(datetime(2026, 8, 7, 22, 0, tzinfo=ET)) == friday
    assert last_completed_session(datetime(2026, 8, 8, 21, 0, tzinfo=ET)) == friday
    assert last_completed_session(datetime(2026, 8, 9, 12, 0, tzinfo=ET)) == friday
    assert last_completed_session(datetime(2026, 8, 10, 9, 0, tzinfo=ET)) == friday


def test_an_overnight_run_is_attributed_to_the_session_it_reads():
    from market_calendar import last_completed_session

    # 01:00 ET Wednesday processes Tuesday's evidence.
    assert last_completed_session(datetime(2026, 8, 12, 1, 0, tzinfo=ET)) == date(2026, 8, 11)
    # The desk clock is Pacific; the answer must not depend on which clock asks.
    assert last_completed_session(datetime(2026, 8, 11, 22, 0, tzinfo=PACIFIC)) == date(2026, 8, 11)


def test_a_holiday_weekend_walks_back_to_the_last_real_session():
    from market_calendar import last_completed_session

    # Thanksgiving 2026 is Thursday 11-26; a Thursday-night run must answer
    # Wednesday, not Thursday.
    assert last_completed_session(datetime(2026, 11, 26, 22, 0, tzinfo=ET)) == date(2026, 11, 25)
    # Independence Day 2026 is observed Friday 07-03, so a Saturday run
    # answers Thursday.
    assert last_completed_session(datetime(2026, 7, 4, 20, 0, tzinfo=ET)) == date(2026, 7, 2)


def test_previous_session_skips_weekends_and_holidays():
    from market_calendar import previous_session

    assert previous_session(date(2026, 8, 10)) == date(2026, 8, 7)
    assert previous_session(date(2026, 11, 27)) == date(2026, 11, 25)


def test_describe_names_why_a_day_is_not_a_session():
    from market_calendar import describe

    assert "weekend" in describe(date(2026, 8, 8))
    assert "holiday" in describe(date(2026, 11, 26))
    assert "regular NYSE session" in describe(date(2026, 8, 7))


# ---------------------------------------------------------------------------
# trading_days_between: the clock every armed-alert expiry and Focus fade runs
# on (Phase 0.12). Weekday arithmetic would expire a 5-day watch armed on a
# Friday before Wednesday, and would count Thanksgiving as a session.
# ---------------------------------------------------------------------------
def test_trading_days_between_counts_sessions_after_the_start():
    import market_calendar

    # Mon 2026-08-03 .. Fri 2026-08-07: four sessions after the start.
    assert market_calendar.trading_days_between(date(2026, 8, 3), date(2026, 8, 7)) == 4


def test_trading_days_between_skips_the_weekend():
    import market_calendar

    # Fri 2026-08-07 -> Mon 2026-08-10 is ONE session, not three days.
    assert market_calendar.trading_days_between(date(2026, 8, 7), date(2026, 8, 10)) == 1


def test_trading_days_between_skips_a_holiday():
    import market_calendar

    # Thanksgiving 2026 is Thu 2026-11-26. Wed 25th -> Fri 27th is ONE session.
    assert market_calendar.trading_days_between(date(2026, 11, 25), date(2026, 11, 27)) == 1


def test_trading_days_between_is_zero_on_the_same_day_and_never_negative():
    import market_calendar

    assert market_calendar.trading_days_between(date(2026, 8, 5), date(2026, 8, 5)) == 0
    assert market_calendar.trading_days_between(date(2026, 8, 7), date(2026, 8, 5)) == 0


def test_trading_days_between_refuses_outside_the_validated_range():
    import market_calendar

    with pytest.raises(market_calendar.SessionCalendarError):
        market_calendar.trading_days_between(date(2026, 8, 3), date(2099, 1, 4))
