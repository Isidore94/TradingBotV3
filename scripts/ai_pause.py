"""Pause AI: one machine-local switch that stops every local-AI use of the GPU host.

The switch is ``ai_paused_until`` in local settings (ISO-8601 with an offset, Pacific).
The Trade Mentor app, the desk, the night runner and ``run_ai_jobs.ps1`` all read it;
a missing, unreadable, naive or past value means "not paused". Qt-free and cheap:
``get_local_setting`` re-stats the file at most once a second.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import project_paths

PAUSE_KEY = "ai_paused_until"
#: A value must end in an explicit offset; a naive time is rejected on both sides.
OFFSET_SUFFIX = re.compile(r"(Z|[+-]\d{2}:\d{2})$")
#: Malformed values already logged (once each).
_warned: set[str] = set()
PT = ZoneInfo("America/Los_Angeles")
#: "tonight" pauses until the next 06:00 PT, when the night AI's window closes.
RESUME_AT_PT = time(6, 0)
#: "until_resumed" is stored as this far date, so every reader needs only one rule.
UNTIL_RESUMED = datetime(2999, 1, 1, tzinfo=PT)
TONIGHT = "tonight"
FOREVER = "until_resumed"
PRESETS: dict[str, Any] = {
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    TONIGHT: TONIGHT,
    FOREVER: FOREVER,
}


def _moment(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo is not None else moment.astimezone()


def _parse(raw: Any) -> datetime | None:
    """The saved end, or None. Only ISO-8601 ending in ``Z`` or ``+HH:MM`` counts
    (the same rule ``run_ai_jobs.ps1`` applies); anything else is logged once."""
    text = str(raw or "").strip()
    if not text:
        return None
    value = None
    if OFFSET_SUFFIX.search(text):
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            value = None
    if value is None or value.tzinfo is None:
        if text not in _warned:
            _warned.add(text)
            logging.warning("Pause AI: %s=%r is not an ISO time with an offset; AI is not paused", PAUSE_KEY, text)
        return None
    return value


def paused_until(now: datetime | None = None) -> datetime | None:
    """When the pause ends (PT), or None when AI is not paused."""
    until = _parse(project_paths.get_local_setting(PAUSE_KEY, ""))
    if until is None or until <= _moment(now):
        return None
    return until.astimezone(PT)


def is_paused(now: datetime | None = None) -> bool:
    return paused_until(now) is not None


def resume_time(duration: Any, now: datetime | None = None) -> datetime:
    """The end of a pause of ``duration``: a timedelta, a preset name, "tonight" or "until_resumed"."""
    moment = _moment(now).astimezone(PT)
    choice = PRESETS.get(duration, duration) if isinstance(duration, str) else duration
    if choice == FOREVER:
        return UNTIL_RESUMED
    if choice == TONIGHT:
        # The next 06:00 PT after now (tomorrow's once it is past 06:00 today).
        at = datetime.combine(moment.date(), RESUME_AT_PT, tzinfo=PT)
        return at if at > moment else datetime.combine(moment.date() + timedelta(days=1), RESUME_AT_PT, tzinfo=PT)
    if isinstance(choice, timedelta) and choice > timedelta(0):
        return moment + choice
    raise ValueError(f"unknown pause length: {duration!r}")


def pause_for(duration: Any, now: datetime | None = None) -> datetime:
    """Pause every local-AI use; returns when the pause ends (PT)."""
    until = resume_time(duration, now)
    project_paths.save_local_setting(PAUSE_KEY, until.isoformat(timespec="seconds"))
    return until


def resume() -> None:
    """AI may run again (the next check in each process picks it up)."""
    project_paths.save_local_setting(PAUSE_KEY, "")


def until_text(until: datetime | None, now: datetime | None = None) -> str:
    """``until 14:30``, ``until Thu 06:00`` (another day) or ``until you resume``."""
    if until is None:
        return ""
    if until >= UNTIL_RESUMED:
        return "until you resume"
    local = until.astimezone(PT)
    today = _moment(now).astimezone(PT).date()
    return f"until {local:%H:%M}" if local.date() == today else f"until {local:%a %H:%M}"


def reason(now: datetime | None = None) -> str:
    """``AI paused until 14:30`` while paused, else ""."""
    until = paused_until(now)
    return f"AI paused {until_text(until, now)}" if until is not None else ""
