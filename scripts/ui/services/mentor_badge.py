"""The desk's Trade Mentor button badge: due, unanswered slots, read from the app's slots file.

Read-only. The Trade Mentor app owns ``trade_mentor_slots.json`` when
``mentor_app_enabled`` is on; the desk only stats it (at most once a second, like
``get_local_setting``) and re-parses it when its mtime or size moved.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

RESTAT_SECONDS = 1.0


def due_unanswered(payload: Mapping[str, Any], now: datetime) -> int:
    """Slots delivered, not answered, not skipped, and still inside their hour."""
    from trade_mentor_schedule import PACIFIC, slots_for_session

    slots = payload.get("slots") if isinstance(payload, Mapping) else None
    if not isinstance(slots, Mapping) or now.tzinfo is None:
        return 0
    local = now.astimezone(PACIFIC)
    count = 0
    for slot in slots_for_session(local.date()):
        record = slots.get(slot.slot_id)
        if not isinstance(record, Mapping) or not record.get("delivered_at"):
            continue
        if record.get("answered_at") or record.get("skipped_reason"):
            continue
        if slot.scheduled_at <= local < slot.expires_at:
            count += 1
    return count


class MentorSlotsBadge:
    """Cached reader: ``count(now)`` costs one stat a second and a parse per change."""

    def __init__(self, path: Path | None = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        if path is None:
            from project_paths import TRADE_MENTOR_SLOTS_FILE

            path = TRADE_MENTOR_SLOTS_FILE
        self.path = Path(path)
        self._clock = clock
        self._checked = float("-inf")
        self._stamp: tuple[int, int] | None = None
        self._payload: Mapping[str, Any] = {}

    def _refresh(self) -> None:
        now = self._clock()
        if now - self._checked < RESTAT_SECONDS:
            return
        self._checked = now
        try:
            stat = self.path.stat()
        except OSError:
            self._stamp, self._payload = None, {}
            return
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp == self._stamp:
            return
        self._stamp = stamp
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        self._payload = payload if isinstance(payload, dict) else {}

    def count(self, now: datetime) -> int:
        self._refresh()
        return due_unanswered(self._payload, now)
