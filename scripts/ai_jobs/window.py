"""Off-hours launch window (plan sec 2 and 6.1).

Two independent gates, deliberately not one:

1. **The configured window.** ``ai_offhours_start`` / ``ai_offhours_end``, ET
   wall clock, trader-adjustable. Weekends are open all day. Holidays are
   treated as normal weekdays, which is the conservative choice: the window
   still applies rather than opening up a day nobody validated.
2. **The market-session block.** "No local inference during market hours" is a
   plan sec 2 hard rule, not a preference -- during the session the desk runs
   the full trading complement and only ~10GB of RAM is free. So the session
   itself is refused *regardless of what the window says*. A fat-fingered
   window cannot put a 14GB model load in front of the open.

Gate 2 is the invariant; gate 1 is the preference. Both must pass to launch.

The window bounds are stored in ET because every other time-reasoning surface
in this system is market-local, and a desk that moves timezone should not
silently move its inference schedule. On a Pacific desk, 22:00-06:00 local is
01:00-09:00 ET.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")

OFFHOURS_START_SETTING = "ai_offhours_start"
OFFHOURS_END_SETTING = "ai_offhours_end"
PREOPEN_GUARD_SETTING = "ai_preopen_guard_minutes"

DEFAULT_OFFHOURS_START = "18:30"
DEFAULT_OFFHOURS_END = "08:00"
#: Extra minutes before the opening bell during which no job may launch.
#:
#: Was 0, on the reasoning that the session block already protects the session
#: itself. It does not protect the run-up to it: the desk's own launch task
#: fires at 06:00 Pacific and pre-market prep is competing for the box well
#: before the bell, so a job started at 09:29 ET with a 14 GB model still to
#: load lands squarely on top of it. 15 minutes is the reserve (Sol 5.6
#: verification review, item 9). It is part of the session block, so --force
#: does not spend it either.
DEFAULT_PREOPEN_GUARD_MINUTES = 15


def _paths():
    try:
        from scripts import project_paths
    except ImportError:
        import project_paths
    return project_paths


def _parse_clock(value: str, fallback: str) -> time:
    text = str(value or "").strip() or fallback
    for fmt in ("%H:%M", "%H%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.time().replace(second=0, microsecond=0)
    return datetime.strptime(fallback, "%H:%M").time()


def offhours_bounds() -> tuple[time, time]:
    """Configured ET window bounds, falling back to the documented defaults."""
    settings = _paths()
    start = _parse_clock(
        settings.get_local_setting(OFFHOURS_START_SETTING, DEFAULT_OFFHOURS_START),
        DEFAULT_OFFHOURS_START,
    )
    end = _parse_clock(
        settings.get_local_setting(OFFHOURS_END_SETTING, DEFAULT_OFFHOURS_END),
        DEFAULT_OFFHOURS_END,
    )
    return start, end


def preopen_guard_minutes() -> int:
    raw = _paths().get_local_setting(PREOPEN_GUARD_SETTING, DEFAULT_PREOPEN_GUARD_MINUTES)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_PREOPEN_GUARD_MINUTES


def market_now(now: datetime | None = None) -> datetime:
    """``now`` as ET. A naive value is read as local time, then converted."""
    moment = now or datetime.now()
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(MARKET_TZ)


def is_weekend(moment: datetime) -> bool:
    return moment.weekday() >= 5  # Saturday=5, Sunday=6


def in_offhours_window(now: datetime | None = None) -> bool:
    """Is the configured window open? Weekends are open all day."""
    moment = market_now(now)
    if is_weekend(moment):
        return True
    start, end = offhours_bounds()
    current = moment.time()
    if start <= end:
        return start <= current < end
    # Wraps past midnight (the 18:30-08:00 default shape).
    return current >= start or current < end


def window_close_at(now: datetime | None = None) -> datetime | None:
    """When the current window closes, or None if it is not open."""
    moment = market_now(now)
    if not in_offhours_window(moment):
        return None
    if is_weekend(moment):
        # Open until the window's weekday end on the next weekday morning.
        _start, end = offhours_bounds()
        cursor = moment
        while is_weekend(cursor) or cursor.time() >= end:
            cursor = (cursor + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        return cursor.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    start, end = offhours_bounds()
    close = moment.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    if start > end and moment.time() >= start:
        close += timedelta(days=1)  # wrapped window: the end is tomorrow
    return close


def minutes_until_window_close(now: datetime | None = None) -> float | None:
    close = window_close_at(now)
    if close is None:
        return None
    return (close - market_now(now)).total_seconds() / 60.0


class SessionLookupFailed(RuntimeError):
    """The calendar could not answer whether this day is a session."""


def _session_bounds(session_day: date) -> tuple[datetime, datetime]:
    """Regular-hours open/close for one ET date.

    Raises :class:`SessionLookupFailed` when the calendar cannot answer.

    This **fails closed** (checkpoint review 2026-08-08 second review). It used
    to return ``None`` on an unavailable calendar or a failed lookup, and the
    caller read ``None`` as "not a session day" -- so a broken import or a
    raising calendar silently unlocked local inference for the whole trading
    day. "No local inference during market hours" is a plan sec 2 hard rule;
    a rule that evaporates when its input fails is not a rule. Missing data is
    uncertainty, never confirmation, so an unanswerable day is treated as a
    market session and the launch is refused.
    """
    try:
        from market_session import get_market_session_window, normalize_market_local_datetime
    except ImportError as exc:
        raise SessionLookupFailed(f"market_session is unavailable: {exc}") from exc
    probe = datetime(session_day.year, session_day.month, session_day.day, 12, 0, tzinfo=MARKET_TZ)
    try:
        window = get_market_session_window(normalize_market_local_datetime(probe))
    except Exception as exc:
        raise SessionLookupFailed(
            f"market session lookup for {session_day.isoformat()} failed: {exc}"
        ) from exc
    return window.open_local, window.close_local


def market_session_block(now: datetime | None = None) -> str:
    """"" when inference is allowed, else why the market blocks it.

    This is the sec 2 hard rule and it ignores the configured window entirely.
    An unanswerable calendar blocks too -- see :func:`_session_bounds`.
    """
    moment = market_now(now)
    if is_weekend(moment):
        return ""
    try:
        open_local, close_local = _session_bounds(moment.date())
    except SessionLookupFailed as exc:
        return (
            f"cannot determine whether {moment.date().isoformat()} is a market "
            f"session ({exc}); treating it as one. Plan sec 2 forbids local "
            "inference during market hours, so an unanswerable calendar "
            "refuses the launch rather than assuming the day is free."
        )
    guard = timedelta(minutes=preopen_guard_minutes())
    if open_local - guard <= moment <= close_local:
        return (
            f"market session is live or imminent "
            f"({open_local:%H:%M}-{close_local:%H:%M} ET"
            + (f", {preopen_guard_minutes()}m pre-open guard" if guard else "")
            + "); plan sec 2 forbids local inference during market hours"
        )
    return ""


def launch_allowed(
    now: datetime | None = None,
    *,
    reserve_minutes: float = 0.0,
) -> tuple[bool, str]:
    """May a job *launch* right now?

    ``reserve_minutes`` refuses a launch that could not plausibly finish before
    the window closes. Jobs already running are governed by the sec 6.1 rule
    instead: finish the current model call, then stop gracefully.
    """
    moment = market_now(now)
    blocked = market_session_block(moment)
    if blocked:
        return False, blocked
    if not in_offhours_window(moment):
        start, end = offhours_bounds()
        return False, (
            f"outside the off-hours window ({start:%H:%M}-{end:%H:%M} ET); "
            f"now is {moment:%H:%M} ET"
        )
    remaining = minutes_until_window_close(moment)
    if reserve_minutes and remaining is not None and remaining < reserve_minutes:
        return False, (
            f"only {remaining:.0f} min left in the window, job reserves "
            f"{reserve_minutes:.0f} min; skipping rather than running into the open"
        )
    return True, f"window open, {remaining:.0f} min remaining" if remaining is not None else "window open"


def describe_window() -> dict[str, str]:
    """Health/Settings payload: what the window is, in both clocks."""
    start, end = offhours_bounds()
    moment = market_now()
    local_start = datetime.combine(moment.date(), start, tzinfo=MARKET_TZ).astimezone()
    local_end = datetime.combine(moment.date(), end, tzinfo=MARKET_TZ).astimezone()
    allowed, reason = launch_allowed(moment)
    return {
        "window_et": f"{start:%H:%M}-{end:%H:%M}",
        "window_desk_local": f"{local_start:%H:%M}-{local_end:%H:%M}",
        "preopen_guard_minutes": str(preopen_guard_minutes()),
        "now_et": f"{moment:%Y-%m-%d %H:%M}",
        "launch_allowed": "yes" if allowed else "no",
        "reason": reason,
    }
