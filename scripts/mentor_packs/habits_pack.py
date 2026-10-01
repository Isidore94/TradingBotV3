"""Habits pack: what the trader keeps saying and feeling, as the night counted it. Read-only.

Reads the night's ``mentor_habits.json`` (``ai_jobs.mentor_habits``, in the ai_store digests folder): the habits seen
on 3+ days of the last 30 (a mood tag or a 3-word phrase), each with first/last seen, days seen, example ids and
when it shows up (after a loss, before the open, late day, regime), plus the last session's mood-tag counts. Ids:
``habits:asof``, ``habit:<key>`` (the night's own ids), ``habits:day``. Counts and observations, never rules. A
missing file is "not counted yet", never "no habits".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from mentor_packs.registry import Pack, make_pack

NAME = "habits_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The trader's habits as the night counted them over the last 30 days: mood tags and phrases he said on "
            "3+ days, when (after a loss, before the open, late day, regime), how often the rest of that day was "
            "red, with example ids; plus the last session's mood-tag counts. Observations, never rules."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
HABITS_FILE = "mentor_habits.json"


@dataclass(frozen=True)
class Sources:
    registry: Callable[[], Path | None]


def _live_registry() -> Path | None:
    from mentor_app.memory import _digests_root

    root = _digests_root()
    return None if root is None else Path(root) / HABITS_FILE


def live_sources() -> Sources:
    return Sources(registry=_live_registry)


def _context_text(context: Mapping[str, Any]) -> str:
    parts = [f"after a loss {context.get('after_loss', 0)}", f"before the open {context.get('before_open', 0)}",
             f"late day {context.get('late_day', 0)}"]
    regimes = context.get("regimes") or {}
    if regimes:
        parts.append("regimes " + ", ".join(f"{k} {v}" for k, v in regimes.items()))
    return "; ".join(parts)


def build(*, sources: Sources | None = None) -> Pack:
    """Build the habits pack (one file read: call it on a worker)."""
    src = sources or live_sources()
    path = src.registry()
    if path is None:
        return make_pack(NAME, (), empty_text="no ai_store is configured, so the night's habit counts are unknown")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return make_pack(NAME, (), empty_text="the night has not counted your habits yet; unknown")
    except (OSError, ValueError) as exc:
        return make_pack(NAME, (), empty_text=f"the habit counts could not be read ({type(exc).__name__}); unknown")
    habits = [h for h in payload.get("habits") or () if isinstance(h, Mapping) and h.get("id")]
    session = str(payload.get("session_date") or "?")
    rows: list[dict[str, Any]] = [{
        "id": "habits:asof", "kind": "asof", "session": session,
        "text": (f"Habits counted by the night of {session} over the last {payload.get('window_days', 30)} days "
                 f"(seen on {payload.get('min_days', 3)}+ days): {len(habits)} found. Observations, never rules."),
    }]
    for habit in habits:
        quotes = "; ".join(f"\"{q}\"" for q in habit.get("quotes") or ())
        rows.append({
            "id": str(habit["id"]), "kind": "habit", "key": str(habit.get("key") or ""),
            "days_seen": habit.get("days_seen"), "inbox_ok": bool(habit.get("inbox_ok")),
            "text": (f"{habit.get('text', '')}. First seen {habit.get('first_seen', '?')}, last "
                     f"{habit.get('last_seen', '?')}; when: {_context_text(habit.get('context') or {})}; examples "
                     f"{', '.join(habit.get('examples') or ())}" + (f": {quotes}" if quotes else "")),
        })
    day = payload.get("day") or {}
    if day:
        tags = ", ".join(f"{k} {v}" for k, v in (day.get("mood_tags") or {}).items()) or "none"
        rows.append({"id": "habits:day", "kind": "day",
                     "text": (f"{day.get('session_date', session)}: {day.get('journal_lines', 0)} journal line(s), "
                              f"{day.get('turns', 0)} question(s); mood tags {tags}; lines after a loss "
                              f"{day.get('after_loss', 0)}, after a win {day.get('after_win', 0)}")})
    return make_pack(NAME, rows)


#: app_state key: the ISO week (``YYYY-Www``, PT) the one habit Inbox item was posted.
INBOX_WEEK_KEY = "habits:inbox:week"


def inbox_line(pack: Pack, posted_week: str | None, now: Any) -> tuple[str, str] | None:
    """(week, line) for the week's ONE habit Inbox item: the first habit the night marked ``inbox_ok`` (seen on
    3+ days, each followed by a red rest of day); None when none, or one was already posted this week."""
    from zoneinfo import ZoneInfo

    local = now.astimezone(ZoneInfo("America/Los_Angeles"))
    year, week, _ = local.isocalendar()
    key = f"{year}-W{week:02d}"
    if posted_week == key:
        return None
    habit = next((row for row in pack.rows if row.get("kind") == "habit" and row.get("inbox_ok")), None)
    if habit is None:
        return None
    return key, f"A habit with a red rest of day after it: {habit['text'][:200]} [{habit['id']}]"


# ---------------------------------------------------------------- fixture
FIXTURE_REGISTRY: dict[str, Any] = {
    "schema": "mentor_habits_v1", "session_date": "2026-09-30", "window_days": 30, "min_days": 3,
    "day": {"session_date": "2026-09-30", "journal_lines": 4, "turns": 6, "mood_tags": {"fomo": 2, "tilted": 1},
            "after_loss": 2, "after_win": 0, "hours_et": {"10": 3}},
    "habits": [
        {"key": "tag:fomo", "id": "habit:tag_fomo", "first_seen": "2026-09-22", "last_seen": "2026-09-30",
         "days_seen": 5, "count": 7, "examples": ["journal:12", "journal:15", "journal:19"],
         "quotes": ["I chased TWLO again", "chasing the open"], "inbox_ok": True,
         "context": {"after_loss": 4, "before_open": 0, "late_day": 1, "regimes": {"chop": 5}},
         "text": "You said mood tag 'fomo' on 5 days (7 times) since 2026-09-22"},
        {"key": "phrase:chased the open", "id": "habit:phrase_chased_the_open", "first_seen": "2026-09-24",
         "last_seen": "2026-09-29", "days_seen": 3, "count": 3, "examples": ["turn:40", "journal:15"],
         "quotes": ["I chased the open"], "inbox_ok": False,
         "context": {"after_loss": 0, "before_open": 1, "late_day": 0, "regimes": {}},
         "text": "You said the phrase \"chased the open\" on 3 days (3 times) since 2026-09-24"},
    ],
}


def fixture_sources(root: Path) -> Sources:
    path = Path(root) / HABITS_FILE
    path.write_text(json.dumps(FIXTURE_REGISTRY), encoding="utf-8")
    return Sources(registry=lambda: path)


def fixture() -> Pack:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        return build(sources=fixture_sources(Path(tmp)))
