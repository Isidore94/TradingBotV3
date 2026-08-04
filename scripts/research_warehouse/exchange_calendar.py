"""A versioned XNYS session calendar (plan sec 5.4, Phase 4).

``trading_session`` needs real exchange sessions - which days traded, which
closed early, and the exact RTH/ETH boundaries in UTC - and nothing in the
repository modelled holidays before this: ``scripts/market_session.py`` returns
09:30-16:00 for whatever date it is handed, weekend or Christmas alike. That is
fine for the live desk (it only ever asks about today, while the market is
open) and wrong for a research archive, where "no bars on 2026-11-26" must be
distinguishable from "the exchange was closed".

So the calendar is stated here as **rules**, with an explicit
``calendar_version`` recorded on every session row. When the rules are revised
the version is bumped and old rows keep saying which calendar produced them;
history is never silently re-dated. This module is research-only - no champion
path imports it.

Rules implemented (NYSE/NASDAQ US equities):

* full closures: New Year's Day, MLK, Washington's Birthday, Good Friday,
  Memorial Day, Juneteenth (from 2022), Independence Day, Labor Day,
  Thanksgiving, Christmas - with the NYSE observance rule that a Saturday
  holiday moves to the preceding Friday and a Sunday holiday to the following
  Monday, except New Year's Day, which is never observed on the preceding
  Friday;
* 13:00 ET early closes: July 3 when Independence Day falls Tue-Fri, the Friday
  after Thanksgiving, and December 24 when it falls Mon-Thu.

Anything outside those rules is a normal weekday session. A day the exchange
actually closed for an unscheduled reason (weather, a national day of mourning)
is not knowable from rules; it shows up as a session with no bars, which is an
honest gap, not a silent one.
"""

from __future__ import annotations

import zoneinfo
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

EXCHANGE_CALENDAR = "XNYS"
CALENDAR_VERSION = "xnys_rules_v1"
EXCHANGE_TZ = zoneinfo.ZoneInfo("America/New_York")

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
HALF_DAY_CLOSE = time(13, 0)
ETH_OPEN = time(4, 0)
ETH_CLOSE = time(20, 0)

M5_MINUTES = 5
M1_MINUTES = 1

#: Juneteenth became an NYSE holiday in 2022; before that it was a normal day.
JUNETEENTH_FIRST_YEAR = 2022


@dataclass(frozen=True)
class TradingSession:
    session_id: str
    session_date: date
    exchange_calendar: str
    calendar_version: str
    rth_open_at: datetime
    rth_close_at: datetime
    eth_open_at: datetime
    eth_close_at: datetime
    is_half_day: bool

    @property
    def rth_minutes(self) -> int:
        return int((self.rth_close_at - self.rth_open_at).total_seconds() // 60)

    @property
    def expected_m5_bars_rth(self) -> int:
        return self.rth_minutes // M5_MINUTES

    @property
    def expected_m1_bars_rth(self) -> int:
        return self.rth_minutes // M1_MINUTES

    def phase_of(self, moment: datetime) -> str:
        if moment < self.rth_open_at:
            return "PRE"
        if moment >= self.rth_close_at:
            return "POST"
        return "RTH"


def session_id_for(day: date, calendar: str = EXCHANGE_CALENDAR) -> str:
    return f"{calendar}-{day.isoformat()}"


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    day = date(year, month, 1)
    offset = (weekday - day.weekday()) % 7
    return day + timedelta(days=offset + 7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    day = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return day - timedelta(days=(day.weekday() - weekday) % 7)


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian computus - Good Friday is two days earlier."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lunar = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lunar) // 451
    month, day = divmod(h + lunar - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(day: date, *, allow_friday: bool = True) -> date:
    """NYSE observance: Saturday -> preceding Friday, Sunday -> next Monday."""
    if day.weekday() == 5:  # Saturday
        return day - timedelta(days=1) if allow_friday else day
    if day.weekday() == 6:  # Sunday
        return day + timedelta(days=1)
    return day


def holidays(year: int) -> dict:
    """Full-closure dates for one year, keyed by date with their names."""
    dates: dict[date, str] = {}

    def add(day: date, name: str) -> None:
        if day.weekday() < 5:
            dates[day] = name

    # New Year's Day is never pulled back to the preceding 31 December.
    add(_observed(date(year, 1, 1), allow_friday=False), "New Year's Day")
    add(_nth_weekday(year, 1, 0, 3), "Martin Luther King Jr. Day")
    add(_nth_weekday(year, 2, 0, 3), "Washington's Birthday")
    add(easter_sunday(year) - timedelta(days=2), "Good Friday")
    add(_last_weekday(year, 5, 0), "Memorial Day")
    if year >= JUNETEENTH_FIRST_YEAR:
        add(_observed(date(year, 6, 19)), "Juneteenth National Independence Day")
    add(_observed(date(year, 7, 4)), "Independence Day")
    add(_nth_weekday(year, 9, 0, 1), "Labor Day")
    add(_nth_weekday(year, 11, 3, 4), "Thanksgiving Day")
    add(_observed(date(year, 12, 25)), "Christmas Day")
    return dates


def half_days(year: int) -> dict:
    """13:00 ET early closes for one year."""
    dates: dict[date, str] = {}
    closures = holidays(year)

    def add(day: date, name: str) -> None:
        if day.weekday() < 5 and day not in closures:
            dates[day] = name

    independence = date(year, 7, 4)
    if independence.weekday() < 5:  # Mon-Fri; the day before closes at 13:00
        add(independence - timedelta(days=1), "Independence Day eve")
    add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1), "Day after Thanksgiving")
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() <= 3:  # Mon-Thu
        add(christmas_eve, "Christmas Eve")
    return dates


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in holidays(day.year)


def is_half_day(day: date) -> bool:
    return day in half_days(day.year)


def _at(day: date, moment: time) -> datetime:
    return datetime.combine(day, moment, tzinfo=EXCHANGE_TZ).astimezone(timezone.utc)


def trading_session(day: date, *, calendar: str = EXCHANGE_CALENDAR) -> TradingSession | None:
    """The session for ``day``, or None when the exchange was closed.

    Boundaries are computed in exchange time and returned in UTC, so DST is
    handled by the zone rather than by an offset constant: the 09:30 open is
    13:30 UTC in summer and 14:30 UTC in winter.
    """
    if not is_trading_day(day):
        return None
    half = is_half_day(day)
    close = HALF_DAY_CLOSE if half else RTH_CLOSE
    return TradingSession(
        session_id=session_id_for(day, calendar),
        session_date=day,
        exchange_calendar=calendar,
        calendar_version=CALENDAR_VERSION,
        rth_open_at=_at(day, RTH_OPEN),
        rth_close_at=_at(day, close),
        eth_open_at=_at(day, ETH_OPEN),
        # A half day's extended session ends with the early close; there is no
        # post-market on those days.
        eth_close_at=_at(day, close) if half else _at(day, ETH_CLOSE),
        is_half_day=half,
    )


def sessions_between(start: date, end: date, *, calendar: str = EXCHANGE_CALENDAR):
    day = start
    while day <= end:
        session = trading_session(day, calendar=calendar)
        if session is not None:
            yield session
        day += timedelta(days=1)


def session_for(moment: datetime, *, calendar: str = EXCHANGE_CALENDAR) -> TradingSession | None:
    """The session that owns an instant, by its exchange-local date."""
    return trading_session(moment.astimezone(EXCHANGE_TZ).date(), calendar=calendar)


def exchange_week(day: date) -> tuple[date, date]:
    """The Monday-Sunday exchange week containing ``day``."""
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=6)


def week_sessions(day: date, *, calendar: str = EXCHANGE_CALENDAR) -> list:
    start, end = exchange_week(day)
    return list(sessions_between(start, end, calendar=calendar))


__all__ = [
    "CALENDAR_VERSION",
    "EXCHANGE_CALENDAR",
    "EXCHANGE_TZ",
    "HALF_DAY_CLOSE",
    "JUNETEENTH_FIRST_YEAR",
    "RTH_CLOSE",
    "RTH_OPEN",
    "TradingSession",
    "easter_sunday",
    "exchange_week",
    "half_days",
    "holidays",
    "is_half_day",
    "is_trading_day",
    "session_for",
    "session_id_for",
    "sessions_between",
    "trading_session",
    "week_sessions",
]
