"""When the trader last saw each night slot's output on screen.

A small machine-local JSON registry, `{slot: "YYYY-MM-DD"}`. Panels call
`note_slot_output_read(slot)` when they render a fresh output; the digest facts
report slots unread for 14+ days. Nothing is deleted or disabled from this:
killing a slot is the trader's call.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any

#: The slots whose output a desk page shows, in report order.
TRACKED_SLOTS = (
    "day_review_narration",
    "econ_brief",
    "improvement_ideas",
    "week_questions",
)
#: A slot unread this many days is named in the digest facts.
UNREAD_DAYS = 14

#: Registry key: the day stamping began. A never-read slot is only called
#: unread once the registry is that old; before then it is unknown.
SINCE_KEY = "_since"

_LOCK = threading.Lock()
#: Slots already stamped today in this process, so a re-render writes nothing.
_NOTED_TODAY: dict[str, str] = {}


def _registry_path() -> Path:
    from project_paths import SLOT_OUTPUT_READS_FILE

    return Path(SLOT_OUTPUT_READS_FILE)


def _read(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items() if isinstance(value, str)}


def note_slot_output_read(slot: str, *, today: date | None = None, path: Path | None = None) -> None:
    """Stamp `slot` as read today. At most one small write per slot per day; never raises."""
    day = (today or date.today()).isoformat()
    name = str(slot or "").strip()
    if not name:
        return
    with _LOCK:
        if path is None and _NOTED_TODAY.get(name) == day:
            return
        target = Path(path) if path is not None else _registry_path()
        try:
            payload = _read(target)
            if payload.get(name) != day:
                payload.setdefault(SINCE_KEY, day)
                payload[name] = day
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + ".tmp")
                temp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(temp, target)
            if path is None:
                _NOTED_TODAY[name] = day
        except OSError:
            logging.debug("Slot read stamp for %s not saved.", name, exc_info=True)


def days_since_read(slot: str, *, today: date | None = None, path: Path | None = None) -> int | None:
    """Whole days since `slot`'s output was last shown; None when never recorded."""
    target = Path(path) if path is not None else _registry_path()
    stamp = _read(target).get(str(slot or "").strip())
    if not stamp:
        return None
    try:
        seen = date.fromisoformat(stamp[:10])
    except ValueError:
        return None
    return max(0, ((today or date.today()) - seen).days)


def unread_line(*, today: date | None = None, path: Path | None = None) -> str:
    """"unread 14+ days: a, b" or "... none"; unknown before any read was recorded."""
    target = Path(path) if path is not None else _registry_path()
    day = today or date.today()
    since = days_since_read(SINCE_KEY, today=day, path=target)
    if since is None:
        return f"unread {UNREAD_DAYS}+ days: unknown (no reads recorded yet)"
    stale: list[Any] = []
    for slot in TRACKED_SLOTS:
        days = days_since_read(slot, today=day, path=target)
        if days is None:
            days = since
        if days >= UNREAD_DAYS:
            stale.append(slot)
    return f"unread {UNREAD_DAYS}+ days: " + (", ".join(stale) if stale else "none")
