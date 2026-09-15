"""When the Daily Recap fills itself in - trader request, 2026-09-14.

The trader asked for the Daily Recap to "auto populate at 12pm PST each day so
at end of day I can review it". Until now the page read a session only when
the trader opened it or pressed Refresh, and it opened on the LAST COMPLETED
session, so a visit at the end of the day showed yesterday until the picker was
moved by hand.

This module answers one question PURELY - *is the automatic read for this
session due right now?* - with no Qt, no I/O, no clock read and no state of its
own. The panel owns the timer and the "already fired" memory; because the
answer is a function of the moment, a timer that fired late, a desk that was
started after noon, or a clock the OS corrected all resolve the same way.

Three rules decide the answer.

* **Due from the configured Pacific wall-clock time until midnight, once per
  session, then once after the exchange-owned close.** A desk started at 15:00
  still reads today on its first tick; the separate close read follows on the
  next tick and never repeats. Pacific is `America/Los_Angeles` - a clock WITH
  daylight saving, exactly as the Trade Mentor's slots are
  (`trade_mentor_schedule.PACIFIC`).
* **A day that is not an exchange session is never due.** Saturday at noon
  has no session to read; the page keeps whatever it was showing.
* **Uncertainty asks nothing.** A date the calendar refuses to answer for, or
  a time the trader mis-typed in settings, is `None`: the page stays as it is
  and the manual controls still work. Nothing here starts a scan, pushes to
  the phone or writes a store, so it is not an automatic STARTER under the
  quiet-hours rule; the amendment in
  `docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md` records why.

Note the market's own close: 12:00 Pacific is 15:00 ET, one hour BEFORE the
regular close. The configured read is therefore PROVISIONAL by the reader's
own labelling (`daily_recap_reader._is_provisional`); the scheduler then makes
one closed-and-measured read after `market_early_close.session_close`, including
an early close. The time is a setting (`SETTING_KEY`) so the trader can move it
without a code change.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

#: The trader's wall clock. DST-aware by construction.
PACIFIC = ZoneInfo("America/Los_Angeles")

#: `local_settings.json` key holding the Pacific wall-clock time as "HH:MM".
#: An empty string or "off" disables the automatic read.
SETTING_KEY = "daily_recap_auto_time"

#: What the trader asked for, in as many words.
DEFAULT_AUTO_TIME = "12:00"

#: Spellings that mean "never fire", so a trader can switch it off from the
#: settings file without deleting the key.
DISABLED_VALUES = frozenset({"", "off", "none", "never"})


def parse_auto_time(value: object) -> time | None:
    """The configured wall-clock time, or `None` when disabled or unreadable.

    A mis-typed value is `None` rather than a guessed time, and the caller
    treats `None` as "never due": a wrong time would silently fire at an hour
    the trader never chose.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in DISABLED_VALUES:
        return None
    try:
        parsed = time.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return None
    return parsed.replace(second=0, microsecond=0)


def auto_time_from_settings(get_setting=None) -> time | None:
    """Read `SETTING_KEY` through `project_paths.get_local_setting`.

    `get_setting` is injectable for tests; the default reads the desk's own
    per-machine settings file. An unreadable settings file disables the read
    for this tick rather than raising into a timer slot.
    """
    if get_setting is None:
        try:
            import project_paths

            get_setting = project_paths.get_local_setting
        except Exception:  # noqa: BLE001 - settings unreadable: not due
            return None
    try:
        raw = get_setting(SETTING_KEY, DEFAULT_AUTO_TIME)
    except Exception:  # noqa: BLE001
        return None
    return parse_auto_time(raw)


def _pacific(now: datetime) -> datetime:
    """`now` on the trader's wall clock. A naive stamp is taken as local time
    and CONVERTED, never re-labelled (CLAUDE.md, the one completed-bar rule)."""
    moment = now if now.tzinfo is not None else now.astimezone()
    return moment.astimezone(PACIFIC)


def due_session(
    now: datetime,
    *,
    auto_time: time | None,
    last_fired_session: str | None,
) -> str | None:
    """The session whose automatic read is due at `now`, else `None`.

    Due means: today (Pacific) is an exchange session, the wall clock has
    reached `auto_time`, and `last_fired_session` is not already today. The
    answer is today's ISO date so the caller can both select it and remember
    it. A calendar refusal is `None`.
    """
    if auto_time is None:
        return None
    moment = _pacific(now)
    today: date = moment.date()
    stamp = today.isoformat()
    if last_fired_session == stamp:
        return None
    if moment.timetz().replace(tzinfo=None) < auto_time:
        return None
    try:
        import market_calendar

        if not market_calendar.is_session(today):
            return None
    except Exception:  # noqa: BLE001 - uncertainty asks nothing
        return None
    return stamp


def post_close_due_session(
    now: datetime,
    *,
    configured_session: str | None,
    last_post_close_session: str | None,
) -> str | None:
    """The one second read due after the exchange-owned close.

    The configured read must already have happened in this process.  On a
    restart after the close, the configured slot is read first and this helper
    permits the closed read on the next tick.  That makes restarts safe without
    persisting timer state or ever looping.
    """
    moment = _pacific(now)
    today = moment.date()
    stamp = today.isoformat()
    if configured_session != stamp or last_post_close_session == stamp:
        return None
    try:
        import market_calendar
        import market_early_close

        if not market_calendar.is_session(today):
            return None
        close = market_early_close.session_close(today).astimezone(PACIFIC)
    except Exception:  # noqa: BLE001 - uncertainty asks nothing
        return None
    return stamp if moment >= close else None


def next_fire_at(now: datetime, *, auto_time: time | None) -> datetime | None:
    """The next Pacific instant the read would be due, for a label or a log.

    Walks forward at most 30 days for the next exchange session at or after
    `now`; `None` when disabled or when the calendar cannot answer. Purely
    informational - `due_session` is the decision.
    """
    if auto_time is None:
        return None
    try:
        import market_calendar

        moment = _pacific(now)
        cursor = moment.date()
        for _ in range(30):
            candidate = datetime.combine(cursor, auto_time, tzinfo=PACIFIC)
            if market_calendar.is_session(cursor) and candidate > moment:
                return candidate
            cursor = date.fromordinal(cursor.toordinal() + 1)
    except Exception:  # noqa: BLE001
        return None
    return None


__all__ = [
    "DEFAULT_AUTO_TIME",
    "DISABLED_VALUES",
    "PACIFIC",
    "SETTING_KEY",
    "auto_time_from_settings",
    "due_session",
    "next_fire_at",
    "parse_auto_time",
    "post_close_due_session",
]
