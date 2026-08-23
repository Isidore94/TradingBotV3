"""The authoritative session close, early closes included (R10.A scheduling).

`market_calendar` deliberately models every close as 16:00 ET, and its reasons
are good ones: a session is then never declared complete before it actually is,
and the overnight window it feeds opens hours later, so the distinction never
reached a caller. **That is still true for every existing caller, and none of
them changes here.**

The after-close outcome sweep is the first caller for which the distinction
matters in the other direction. It has to wait until the scan thread has stopped
finalizing - close + 30 minutes of scan wind-down + a margin - and on a half day
the real close is 13:00 ET. Using 16:00 there would not be *wrong*, but it would
park the sweep three hours after the desk stopped trading and, on the shortest
sessions of the year, past the point the trader has gone. Using it in the other
direction would be worse: firing at 13:35 ET on a REGULAR day would put the
sweep inside the scan window, which is the race this whole packet exists to
close.

So this module answers one question - *when did this session actually close?* -
and only the scheduler asks it. Nothing here changes a detector, a scanner
window, `market_calendar`, or `market_session`.

**What is modelled.** The three scheduled NYSE half days (13:00 ET):

* the **day after Thanksgiving**;
* **24 December**, when it is a session;
* **3 July**, when it is a session and 4 July falls on a weekday.

**What is not.** An unscheduled early close - a day of mourning, a hurricane,
an infrastructure failure - is not predictable from rules, and this module says
16:00 for one. That is the same bounded limitation `market_calendar` documents
for unscheduled *closures*, and it fails in the safe direction: the sweep waits
longer than it needed to, and never runs early.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from market_calendar import (
    MARKET_TZ,
    REGULAR_CLOSE,
    SessionCalendarError,
    is_session,
)

#: The scheduled half-day close.
EARLY_CLOSE = time(13, 0)

#: How the answer was arrived at, so a caller can log it rather than assert it.
REASON_REGULAR = "regular_close"
REASON_DAY_AFTER_THANKSGIVING = "day_after_thanksgiving"
REASON_CHRISTMAS_EVE = "christmas_eve"
REASON_JULY_THIRD = "july_3"
REASON_UNKNOWN = "unknown_calendar"


def _thanksgiving(year: int) -> date:
    """Fourth Thursday in November."""
    day = date(year, 11, 1)
    thursdays = 0
    while True:
        if day.weekday() == 3:
            thursdays += 1
            if thursdays == 4:
                return day
        day += timedelta(days=1)


def early_close_reason(day: date) -> str | None:
    """Why `day` closes early, or None. Never raises.

    A day that is not a session has no close at all and answers None - asking
    "when did Saturday close?" is a question about nothing.
    """
    try:
        if not is_session(day):
            return None
    except SessionCalendarError:
        # Outside the calendar's horizon. Saying "regular" here is the
        # conservative answer: the sweep waits longer, never less.
        return None
    except Exception:
        return None

    if day == _thanksgiving(day.year) + timedelta(days=1):
        return REASON_DAY_AFTER_THANKSGIVING
    if (day.month, day.day) == (12, 24):
        return REASON_CHRISTMAS_EVE
    if (day.month, day.day) == (7, 3):
        # Only when the 4th is itself a weekday holiday. When 4 July falls on a
        # Saturday the exchange observes it on Friday the 3rd and closes for the
        # whole day, so the 3rd is not a session and never reaches here; when it
        # falls on a Sunday the observed holiday is Monday and the preceding
        # Friday is a full session.
        if date(day.year, 7, 4).weekday() < 5:
            return REASON_JULY_THIRD
    return None


def session_close_time(day: date) -> time:
    """13:00 ET on a scheduled half day, 16:00 ET otherwise."""
    return EARLY_CLOSE if early_close_reason(day) else REGULAR_CLOSE


def session_close(day: date) -> datetime:
    """The aware market-local close of `day`. Regular close for a non-session.

    A non-session answers with the regular close rather than None so a caller
    doing arithmetic cannot get a `TypeError` on a Saturday; callers that care
    whether the day traded at all ask `market_calendar.is_session`.
    """
    return datetime.combine(day, session_close_time(day), tzinfo=MARKET_TZ)


def describe(day: date) -> str:
    """One line for a log: the close and why."""
    reason = early_close_reason(day) or REASON_REGULAR
    return f"{day.isoformat()} closes {session_close_time(day).strftime('%H:%M')} ET ({reason})"
