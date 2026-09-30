"""The quiet Inbox: where every proactive thing lands. Nothing ever pops, beeps or pushes.

A daily cap (``mentor_proactive_per_day``, 6), quiet hours around the open
(06:30-07:00 PT) and ``/quiet <dur>`` all refuse new items; refused items are simply
not shown. Items expire. The window shows a badge count; the transcript never moves
on its own.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
QUIET_START = time(6, 30)
QUIET_END = time(7, 0)
DEFAULT_PER_DAY = 6


@dataclass
class InboxItem:
    id: int
    kind: str
    text: str
    priority: int
    created_utc: datetime
    expires_utc: datetime | None = None
    read: bool = False


@dataclass
class Inbox:
    per_day_cap: int = DEFAULT_PER_DAY
    now: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    muted_until: datetime | None = None
    last_refusal: str = ""
    _items: list[InboxItem] = field(default_factory=list)
    _added_by_day: dict[str, int] = field(default_factory=dict)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def _clock(self) -> datetime:
        moment = self.now()
        return moment if moment.tzinfo else moment.astimezone()

    def refusal(self, moment: datetime | None = None) -> str:
        """"" when a new item may land; otherwise why not."""
        moment = moment or self._clock()
        local = moment.astimezone(PT)
        if QUIET_START <= local.time() < QUIET_END:
            return "quiet hours around the open (06:30-07:00 PT)"
        if self.muted_until is not None and moment < self.muted_until:
            return f"muted until {self.muted_until.astimezone(PT):%H:%M} PT"
        if self._added_by_day.get(local.date().isoformat(), 0) >= max(0, int(self.per_day_cap)):
            return f"today's cap of {self.per_day_cap} is used"
        return ""

    def add(self, kind: str, text: str, *, priority: int = 1, ttl: timedelta | None = None) -> InboxItem | None:
        moment = self._clock()
        self.last_refusal = self.refusal(moment)
        if self.last_refusal:
            return None
        day = moment.astimezone(PT).date().isoformat()
        self._added_by_day[day] = self._added_by_day.get(day, 0) + 1
        item = InboxItem(
            id=next(self._ids),
            kind=str(kind),
            text=str(text),
            priority=int(priority),
            created_utc=moment.astimezone(timezone.utc),
            expires_utc=(moment + ttl).astimezone(timezone.utc) if ttl else None,
        )
        self._items.append(item)
        return item

    def mute(self, duration: timedelta) -> datetime:
        self.muted_until = self._clock() + duration
        return self.muted_until

    def items(self) -> list[InboxItem]:
        moment = self._clock()
        live = [item for item in self._items if item.expires_utc is None or item.expires_utc > moment]
        return sorted(live, key=lambda item: (item.priority, item.created_utc))

    def badge(self) -> int:
        return sum(1 for item in self.items() if not item.read)

    def mark_read(self, item_id: int) -> None:
        for item in self._items:
            if item.id == item_id:
                item.read = True

    def dismiss(self, item_id: int) -> None:
        self._items = [item for item in self._items if item.id != item_id]
