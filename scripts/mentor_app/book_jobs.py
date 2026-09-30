"""Book jobs (P8): when the app reads Questrade, and the ``/book`` card. Qt-free; no model.

On demand only: ``/book``, ``/check`` and once at 06:20 PT on weekdays. A snapshot at most
15 min old is reused; after a failed read, no new one for 1 hour (the refresh token is
single-use and rotates, so the chain is never hammered); no token = no request; with the
desk closed, no fetch. Never a periodic poll. The fetch runs on the app's news thread
(network, no GPU), so Pause AI does not block it. The snapshot and the last failure go
to the app's own chat store (``app_state``); nothing else is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
MORNING_FETCH = time(6, 20)
MORNING_END = time(13, 30)
FOOTER = "*read-only: positions and cash only; it never orders*"


def _aware(now: datetime) -> datetime:
    return now if now.tzinfo else now.astimezone()


def stored_snapshot(store: Any) -> Any:
    from mentor_packs.book_pack import SNAPSHOT_KEY
    from questrade_positions import BookSnapshot

    return BookSnapshot.from_json(store.get_state(SNAPSHOT_KEY))


def stored_status(store: Any) -> dict[str, Any] | None:
    from mentor_packs.book_pack import STATUS_KEY

    try:
        value = json.loads(store.get_state(STATUS_KEY) or "null")
    except ValueError:
        return None
    return value if isinstance(value, dict) and value else None


def skip_reason(store: Any, now: datetime) -> str:
    """Why no fetch now ("" = fetch): a fresh snapshot, or the 1-hour backoff after a failure."""
    from questrade_positions import backoff_left

    moment = _aware(now)
    snap = stored_snapshot(store)
    if snap is not None and snap.is_fresh(moment):
        return "fresh"
    status = stored_status(store) or {}
    if status.get("reason") and status.get("reason") != "no token":
        left = backoff_left(status.get("at_utc"), moment)
        if left is not None:
            return f"backing off after a failed read ({int(left.total_seconds() // 60) + 1} min left)"
    return ""


def ensure_book(store: Any, now: datetime, *, fetch: Callable[..., Any] | None = None,
                desk_closed: Callable[[], bool | None] | None = None) -> dict[str, Any]:
    """Fetch the Questrade book when due. Never raises; returns ``{fetched, reason}``."""
    from mentor_packs.book_pack import SNAPSHOT_KEY, STATUS_KEY

    moment = _aware(now)
    why = skip_reason(store, moment)
    if why:
        return {"fetched": False, "reason": why}
    if desk_closed is not None:
        try:
            closed = desk_closed() is True
        except Exception:  # noqa: BLE001 - a broken probe is "unknown", never "closed"
            closed = False
        if closed:
            return {"fetched": False, "reason": "the desk is closed"}
    if fetch is None:
        from questrade_positions import fetch_book as fetch
    try:
        snap, reason = fetch(moment)
    except Exception as exc:  # noqa: BLE001 - the fetcher never raises; a stub might
        snap, reason = None, f"{type(exc).__name__}: {exc}"[:200]
    stamp = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    if snap is None:
        store.set_state(STATUS_KEY, json.dumps({"reason": str(reason or "unknown"), "at_utc": stamp}, sort_keys=True))
        return {"fetched": False, "reason": str(reason or "unknown"), "failed": True}
    store.set_state(SNAPSHOT_KEY, snap.as_json())
    store.set_state(STATUS_KEY, "{}")
    return {"fetched": True, "reason": ""}


def store_sources(store: Any, base: Any = None) -> Any:
    """``book_pack`` sources that read the snapshot from this app's store (the journal stays ``mode=ro``)."""
    import dataclasses

    from mentor_packs import book_pack

    src = base or book_pack.live_sources()
    return dataclasses.replace(src, snapshot=lambda: _as_dict(stored_snapshot(store)),
                               status=lambda: stored_status(store))


def _as_dict(snap: Any) -> dict[str, Any] | None:
    return snap.as_dict() if snap is not None else None


@dataclass
class BookSchedule:
    """Once a weekday, from 06:20 PT (until 13:30). Marked when queued."""

    done_day: Any = None

    def due(self, now: datetime) -> bool:
        local = _aware(now).astimezone(PT)
        if local.weekday() >= 5 or not (MORNING_FETCH <= local.time() < MORNING_END):
            return False
        return self.done_day != local.date()

    def mark(self, now: datetime) -> None:
        self.done_day = _aware(now).astimezone(PT).date()


def card_markdown(pack: Any, *, note: str = "") -> str:
    """The /book card: the source first, then every row with its id; the footer says it never orders."""
    rows = list(getattr(pack, "rows", ()) or ())
    if not rows:
        return f"**Book**: {getattr(pack, 'empty_text', '') or 'nothing'}\n\n{FOOTER}"
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(str(row.get("kind")), []).append(row)
    source = (by_kind.get("source") or [{}])[0]
    lines = [f"**Book**: {source.get('text', 'source unknown')} `[{source.get('id', 'book:source')}]`", ""]
    sections = (("Accounts", ("account",)), ("Positions", ("position", "position_none")),
                ("Exposure", ("side", "industry", "industry_unknown")), ("Hints", ("hint", "hint_room")))
    for title, kinds in sections:
        section = [row for kind in kinds for row in by_kind.get(kind, ())]
        if section:
            lines.append(f"*{title}*")
            lines += [f"- {row.get('text', '')} `[{row['id']}]`" for row in section]
            lines.append("")
    if note:
        lines += [f"*({note})*", ""]
    lines.append(FOOTER)
    return "\n".join(lines)
