"""The machine's half of the day's record — R10.G (C2, season).

The trader writes what they saw. Nothing writes what the machine saw, so a
question as ordinary as "what was the tape doing the week those setups failed?"
has no answer that does not involve re-deriving the regime from bars months
later — which is a different measurement from the one the desk actually acted
on at the time.

Two stores, one purpose:

**`daily_market_context.jsonl` (`daily_market_context_v1`)** — one row per
session at close+grace. If the desk was off at close, the row is completed at
the next launch and flagged `completed_late`, because a row written on Tuesday
about Monday is a different kind of evidence from one written while Monday was
still happening. It is **never fabricated**: a session nobody measured gets no
row, and the gap is what a reader sees.

**Auto-regime shifts** — audit C2 found `market_environment_annotations.jsonl`
**does not exist at all**, so the regime the desk was operating under was
unrecorded and unrecoverable. Every automatic shift is now a row. Manual
overrides are recorded too and marked as such: the difference between what the
machine thought and what the trader forced is the agreement rate R10.H's
environment timeline is built to show, and it needs both halves.

Nothing here reaches a detector, a score, a gate, an alert, a watchlist or
Focus. It records what the desk already decided; it never participates in
deciding it.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping

#: Schema NAMES (ground rule 5).
SCHEMA_DAILY_MARKET_CONTEXT = "daily_market_context_v1"
SCHEMA_MARKET_REGIME_SHIFT = "market_regime_shift_v1"

STREAM_CONTEXT = "daily_market_context"
STREAM_REGIME = "market_regime_shifts"

SOURCE_AUTO = "auto"
SOURCE_USER = "user"

#: Minutes after the close before the day's row may be written. The session is
#: not over at 16:00 for our purposes - late prints and the outcome sweep still
#: move numbers - so the row waits.
CLOSE_GRACE_MINUTES = 35


def regime_shift_event(
    *,
    from_regime: str,
    to_regime: str,
    source: str,
    session_date: str,
    detail: str = "",
    spy_day_pct: Any = None,
) -> dict[str, Any]:
    """One regime change, recorded as it happens.

    `source` distinguishes the machine's own read from a trader override, and
    keeping both is the point: R10.H's environment timeline shows the agreement
    rate between them, which needs the disagreements.
    """
    return {
        "event_type": "regime_shift",
        "from_regime": str(from_regime or ""),
        "to_regime": str(to_regime or ""),
        "source": str(source or ""),
        "session_date": str(session_date or ""),
        "detail": str(detail or ""),
        "spy_day_pct": _number(spy_day_pct),
    }


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def daily_context_row(
    *,
    session_date: str,
    measured: Mapping[str, Any],
    completed_late: bool = False,
    completed_at: str = "",
) -> dict[str, Any]:
    """The day's machine-side context. Only what was measured.

    Every value comes from the caller, which measured it. Nothing here fills a
    blank with a plausible number: an absent field is absent, and a reader can
    tell "the desk did not measure this" from "the desk measured zero"
    (ground rule 6).
    """
    row = {
        "event_type": "daily_context",
        "session_date": str(session_date or ""),
        "completed_late": bool(completed_late),
        "completed_at": str(completed_at or ""),
    }
    for name, value in (measured or {}).items():
        row[str(name)] = value
    if completed_late:
        row["late_note"] = (
            "written after the session it describes, because the desk was not "
            "running at close+grace; a row written later is weaker evidence "
            "about intraday state than one written at the time"
        )
    return row


def context_due(now: datetime, close_at: datetime, *, grace_minutes: int = CLOSE_GRACE_MINUTES) -> bool:
    """Has this session's row become writable?

    Both sides are normalized by ATTACHING the close's zone to a naive `now`,
    never by stripping the aware side (CLAUDE.md, `_gate_moment`): stripping
    ends the comparison error and keeps the wrong answer.
    """
    from datetime import timedelta

    if now.tzinfo is None and close_at.tzinfo is not None:
        now = now.replace(tzinfo=close_at.tzinfo)
    elif close_at.tzinfo is None and now.tzinfo is not None:
        close_at = close_at.replace(tzinfo=now.tzinfo)
    return now >= close_at + timedelta(minutes=int(grace_minutes))


def missing_sessions(
    written: set[str] | frozenset[str], sessions: list[date], *, through: date
) -> list[str]:
    """Sessions that have no row yet, up to and including `through`.

    This is what makes a row `completed_late` rather than fabricated: the
    caller asks what is missing, measures those sessions from real data, and
    writes them flagged. A session it cannot measure stays missing, and the
    gap is the evidence.
    """
    return [
        day.isoformat()
        for day in sessions
        if day <= through and day.isoformat() not in written
    ]


# ---------------------------------------------------------------------------
# the calendar overlay
# ---------------------------------------------------------------------------
#: Where the trader-editable overlay lives.
CALENDAR_FILENAME = "market_calendar.json"

CALENDAR_SCHEMA = "market_calendar_overlay_v1"

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_ABSENT = "absent"


def calendar_path() -> Path:
    from project_paths import ROOT_DIR

    return Path(ROOT_DIR) / "config" / CALENDAR_FILENAME


def load_calendar_overlay(path: Path | str | None = None) -> dict[str, Any]:
    """The overlay, and an honest verdict about what it covers.

    The computed calendar in `market_calendar.py` is the base and remains the
    fallback; this adds the years a rules engine cannot know - an unscheduled
    close, a newly declared holiday, an early close nobody predicted.

    An overlay that does not cover the ACTIVE year is **degraded and says so**.
    Silently falling through to the computed rules would be the same failure
    the daily-bar store had: a value that looks right until the year it isn't.
    """
    target = Path(path) if path is not None else calendar_path()
    if not target.exists():
        return {
            "schema": CALENDAR_SCHEMA,
            "status": STATUS_ABSENT,
            "years": [],
            "holidays": {},
            "early_closes": {},
            "note": (
                "no calendar overlay on this machine; the computed rules in "
                "market_calendar.py are the only source, which cannot know about "
                "an unscheduled close"
            ),
            "path": str(target),
        }
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "schema": CALENDAR_SCHEMA,
            "status": STATUS_DEGRADED,
            "years": [],
            "holidays": {},
            "early_closes": {},
            "note": f"the calendar overlay could not be read ({exc}), so its coverage is UNKNOWN",
            "path": str(target),
        }
    years = sorted(str(year) for year in (payload.get("years") or []))
    return {
        "schema": str(payload.get("schema") or CALENDAR_SCHEMA),
        "status": STATUS_OK,
        "years": years,
        "holidays": dict(payload.get("holidays") or {}),
        "early_closes": dict(payload.get("early_closes") or {}),
        "note": "",
        "path": str(target),
    }


def calendar_coverage(overlay: Mapping[str, Any], *, today: date | None = None) -> dict[str, Any]:
    """Does the overlay cover the year the desk is trading?

    DEGRADED, visibly, when it does not. The point is not that the desk stops -
    the computed rules still work - but that a year nobody confirmed is a year
    nobody confirmed, and the health surface should say so rather than let the
    silence read as coverage.
    """
    year = str((today or date.today()).year)
    covered = year in set(overlay.get("years") or [])
    if overlay.get("status") == STATUS_ABSENT:
        status, note = STATUS_DEGRADED, overlay.get("note") or "no overlay"
    elif overlay.get("status") == STATUS_DEGRADED:
        status, note = STATUS_DEGRADED, overlay.get("note") or "overlay unreadable"
    elif covered:
        status, note = STATUS_OK, f"the overlay covers {year}"
    else:
        status, note = (
            STATUS_DEGRADED,
            f"the overlay does not cover {year} (it covers "
            f"{', '.join(overlay.get('years') or []) or 'no year'}), so this year's "
            "closes rest on the computed rules alone",
        )
    return {
        "active_year": year,
        "covered": covered,
        "status": status,
        "years": list(overlay.get("years") or []),
        "note": note,
    }


def overlay_holidays(overlay: Mapping[str, Any], year: int) -> set[date]:
    """Extra holidays the overlay declares for one year."""
    out: set[date] = set()
    for stamp in (overlay.get("holidays") or {}).get(str(year), []) or []:
        try:
            out.add(datetime.strptime(str(stamp)[:10], "%Y-%m-%d").date())
        except ValueError:
            continue
    return out


def overlay_early_closes(overlay: Mapping[str, Any], year: int) -> dict[date, time]:
    """Early closes the overlay declares for one year, as {date: local time}."""
    out: dict[date, time] = {}
    for stamp, clock in ((overlay.get("early_closes") or {}).get(str(year), {}) or {}).items():
        try:
            day = datetime.strptime(str(stamp)[:10], "%Y-%m-%d").date()
            hour, minute = str(clock).split(":")[:2]
            out[day] = time(int(hour), int(minute))
        except (ValueError, TypeError):
            continue
    return out
