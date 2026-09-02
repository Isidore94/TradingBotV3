"""NYSE session calendar: is this date a trading session, and which was last?

The repo had no exchange calendar. Several modules say so in their own
comments and degrade accordingly -- `chart_snapshot` names a session that may
never have happened, `autopilot_core` accepts one harmless extra rebuild. That
is tolerable when the cost of being wrong is a wasted fetch. It is not
tolerable for **session identity**: the overnight AI layer keys artifacts and
ledger rows to a session date, and without a calendar it was keying them to
whatever weekday arithmetic produced -- which is how three `ok` rows came to
sit in the ledger claiming coverage of Saturday 2026-08-08, a date on which
the exchange never opened (Sol 5.6 verification review, item 2).

This module answers two questions and refuses to guess at either:

* :func:`is_session` -- is this ET date a regular trading session?
* :func:`last_completed_session` -- the most recent session whose close is at
  or before a given moment.

**What it knows.** Weekends, and the ten scheduled NYSE holidays with their
observance rules. Good Friday is computed from the Gregorian Easter algorithm.
Juneteenth is honoured from 2022, when the exchange first observed it.

**What it cannot know, and says so.** Unscheduled closures -- a national day
of mourning, a hurricane, an infrastructure failure -- are not predictable
from rules. A date the exchange closed unexpectedly will be reported as a
session. That is a known, bounded limitation, and it is the reason
:data:`VALID_THROUGH` exists: past its horizon the rules stop being a
statement about anything, so the calendar raises rather than extrapolating.

**Early closes are deliberately not modelled.** Half-days (1 pm ET) change
when a session ends, not whether it happened. Treating every close as 16:00 ET
is conservative in the only direction that matters: a session is never
declared complete before it actually is. The overnight window opens at 18:30
ET, hours after even a regular close, so the distinction never reaches a
caller. Modelling it would add failure modes to buy nothing.

Nothing here imports pandas, a network client, or a new dependency, so the
headless `ai_jobs` package can use it under `requirements-core.txt` alone.

Other modules keep their existing "holidays not modelled" behaviour; adopting
this calendar in them is a separate packet, not a side effect of this one.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")

#: Regular-hours close. See the module docstring on early closes.
REGULAR_CLOSE = time(16, 0)

#: The span these rules are a statement about. Outside it the calendar raises
#: rather than extrapolating -- an answer nobody has checked is worse than no
#: answer, because callers act on it.
VALID_FROM = date(2000, 1, 1)
VALID_THROUGH = date(2032, 12, 31)

#: The exchange first observed Juneteenth in 2022.
JUNETEENTH_FIRST_YEAR = 2022


class SessionCalendarError(RuntimeError):
    """The calendar cannot answer for this date. Callers must fail closed."""


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    """The ``nth`` ``weekday`` of a month (Monday=0)."""
    cursor = date(year, month, 1)
    offset = (weekday - cursor.weekday()) % 7
    return cursor + timedelta(days=offset + 7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The final ``weekday`` of a month (Monday=0)."""
    cursor = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return cursor - timedelta(days=(cursor.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (the anonymous algorithm)."""
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


def _observed(day: date) -> date | None:
    """NYSE observance for a fixed-date holiday, or None when not observed.

    Saturday holidays move to the preceding Friday; Sunday holidays move to
    the following Monday. New Year's Day is the exception the exchange makes:
    a Saturday 1 January is simply not observed, because moving it would close
    the previous trading year a day early.
    """
    if day.weekday() == 5:  # Saturday
        if (day.month, day.day) == (1, 1):
            return None
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday
        return day + timedelta(days=1)
    return day


def holidays_for_year(year: int) -> set[date]:
    """Every observed NYSE full-day closure in ``year``."""
    observed: set[date] = set()

    for fixed in (date(year, 1, 1), date(year, 7, 4), date(year, 12, 25)):
        moved = _observed(fixed)
        if moved is not None:
            observed.add(moved)

    if year >= JUNETEENTH_FIRST_YEAR:
        moved = _observed(date(year, 6, 19))
        if moved is not None:
            observed.add(moved)

    observed.add(_nth_weekday(year, 1, 0, 3))    # Martin Luther King Jr. Day
    observed.add(_nth_weekday(year, 2, 0, 3))    # Washington's Birthday
    observed.add(_easter(year) - timedelta(days=2))  # Good Friday
    observed.add(_last_weekday(year, 5, 0))      # Memorial Day
    observed.add(_nth_weekday(year, 9, 0, 1))    # Labor Day
    observed.add(_nth_weekday(year, 11, 3, 4))   # Thanksgiving

    return observed


def _check_range(day: date) -> None:
    if not (VALID_FROM <= day <= VALID_THROUGH):
        raise SessionCalendarError(
            f"{day.isoformat()} is outside the range these NYSE rules are "
            f"validated for ({VALID_FROM.isoformat()}..{VALID_THROUGH.isoformat()}); "
            "refusing to extrapolate a session calendar"
        )


def is_session(day: date) -> bool:
    """Is ``day`` (an ET calendar date) a regular NYSE trading session?

    Raises :class:`SessionCalendarError` outside the validated range. See the
    module docstring for the unscheduled-closure limitation.
    """
    if not isinstance(day, date) or isinstance(day, datetime):
        day = day.date() if isinstance(day, datetime) else day
    _check_range(day)
    if day.weekday() >= 5:
        return False
    return day not in holidays_for_year(day.year)


def previous_session(day: date) -> date:
    """The latest session strictly before ``day``."""
    cursor = day - timedelta(days=1)
    # A week of weekend plus the longest holiday cluster is far inside this;
    # the bound exists so a calendar bug cannot become an infinite loop.
    for _ in range(30):
        if is_session(cursor):
            return cursor
        cursor -= timedelta(days=1)
    raise SessionCalendarError(
        f"no NYSE session found in the 30 days before {day.isoformat()}"
    )


def trading_days_between(start: date, end: date) -> int:
    """How many trading sessions fall AFTER ``start``, up to and including ``end``.

    The clock every "expire this N trading days from now" rule runs on
    (Phase 0.12: armed-alert expiry, Focus fade). Weekday arithmetic gets this
    wrong twice - it counts Thanksgiving as a session, and a five-session watch
    armed on a Friday would come due on the following Friday rather than the
    Friday after.

    ``start`` itself is never counted: a watch armed today has zero sessions
    behind it, whatever time of day it was armed. An ``end`` at or before
    ``start`` is 0, never a negative - "not yet due" is the only meaningful
    reading of a clock that has not started.

    Raises :class:`SessionCalendarError` when either endpoint is outside the
    validated range, so a caller fails CLOSED. That matters here: every caller
    of this function deletes something when it answers, and uncertainty must
    never delete.
    """
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()
    _check_range(start)
    _check_range(end)
    if end <= start:
        return 0
    sessions = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if is_session(cursor):
            sessions += 1
        cursor += timedelta(days=1)
    return sessions


def session_close(day: date) -> datetime:
    """Regular close of ``day``, as an aware ET datetime."""
    return datetime.combine(day, REGULAR_CLOSE, tzinfo=MARKET_TZ)


def last_completed_session(now: datetime) -> date:
    """The most recent session whose close is at or before ``now``.

    This is the session an overnight run is *about*. A run at 01:00 ET
    Wednesday is processing Tuesday; a run at 21:00 ET Saturday is still
    processing Friday, because Saturday was never a session at all.

    Raises :class:`SessionCalendarError` if the calendar cannot answer, so
    callers fail closed rather than keying artifacts to a guessed date.
    """
    moment = now if now.tzinfo else now.astimezone()
    moment = moment.astimezone(MARKET_TZ)
    cursor = moment.date()
    for _ in range(30):
        if is_session(cursor) and session_close(cursor) <= moment:
            return cursor
        cursor -= timedelta(days=1)
    raise SessionCalendarError(
        f"no completed NYSE session found in the 30 days before {moment.isoformat()}"
    )


def describe(day: date) -> str:
    """One-line explanation, for ledger reasons and logs."""
    if day.weekday() >= 5:
        return f"{day.isoformat()} is a weekend"
    if day in holidays_for_year(day.year):
        return f"{day.isoformat()} is an NYSE holiday"
    return f"{day.isoformat()} is a regular NYSE session"
