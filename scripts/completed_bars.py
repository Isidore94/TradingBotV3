"""The one intraday completed-bar rule (R5 section 5).

`plan.md` sec 5: **completed bars only for state transitions; a forming bar is
preview.** That rule was written down once and then implemented three times.

- `weekend_strength._completed_intraday` had it right, and now delegates here.
- BounceBot re-implements it ad hoc at each detector call site
  (`bounce_bot_lib/legacy.py:4384-4386` and `4533-4535`) as
  ``cutoff = get_market_local_now().replace(tzinfo=None)`` and then a compare.
  **That spelling is wrong for a tz-aware stamp**: ``replace(tzinfo=None)``
  discards the offset rather than converting through it, so a bar carrying a
  zone is judged against a wall-clock number that never meant the same instant.
  Those call sites migrate opportunistically, never as a silent behaviour
  change to a shipped detector (R5 section 5 says so explicitly) - so this
  module exists first, and the new engines are its first users.
- `strength_scan` does not need it: it documents that it receives bars already
  completed and in ascending order.

The rule, stated once::

    a bar is COMPLETE when   bar_start + bar_minutes <= now

Inclusive at the boundary, deliberately: a strict ``<`` discards the bar that
just closed, which on a 5-minute engine is the single most important bar there
is.

Timezones are converted with ``astimezone``, never stripped. A bar whose
timestamp cannot be read at all is dropped rather than assumed complete -
missing data is uncertainty, never confirmation.

Pure: no clock of its own (``now`` is always passed in), no I/O, no imports
from any detector.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

#: Field names bar producers actually use. ``dt`` is first because that is what
#: ``autopilot_core._frame_rows`` emits, and that is the real fetch path -
#: omitting it once meant every live-downloaded bar read as "no readable
#: timestamp" and was silently dropped, while every hand-built unit test passed.
_TIME_KEYS = ("dt", "timestamp", "time", "date")


def bar_time(bar: Mapping[str, Any]) -> datetime | None:
    """A bar's start time, whatever the producer called the field."""
    value = None
    for key in _TIME_KEYS:
        value = bar.get(key)
        if value:
            break
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.fromisoformat(text[:10])
        except ValueError:
            return None


def align_to(stamp: datetime, now: datetime) -> datetime:
    """Put ``stamp`` on the same awareness footing as ``now``.

    Converting with ``astimezone`` is the whole point: a stamp carrying a zone
    is moved to the local wall clock, not stripped of its offset.
    """
    if stamp.tzinfo is not None and now.tzinfo is None:
        return stamp.astimezone().replace(tzinfo=None)
    if stamp.tzinfo is None and now.tzinfo is not None:
        return stamp.replace(tzinfo=now.tzinfo)
    return stamp


def is_completed_bar(
    bar: Mapping[str, Any], bar_minutes: int, *, now: datetime
) -> bool:
    """True when this intraday bar has finished. Undateable bars are False."""
    stamp = bar_time(bar)
    if stamp is None:
        return False
    span = timedelta(minutes=max(1, int(bar_minutes)))
    return align_to(stamp, now) + span <= now


def completed_intraday_bars(
    bars: Sequence[Mapping[str, Any]], bar_minutes: int, *, now: datetime
) -> list[Mapping[str, Any]]:
    """Every finished bar, in the order given. Forming bars are dropped."""
    return [bar for bar in bars or () if is_completed_bar(bar, bar_minutes, now=now)]


def completed_m5_bars(
    bars: Sequence[Mapping[str, Any]], *, now: datetime
) -> list[Mapping[str, Any]]:
    """The M5 case, which is what every R5 engine wants."""
    return completed_intraday_bars(bars, 5, now=now)
