"""Book jobs (P8, P12): when the app reads Questrade and IBKR, and the ``/book`` card. Qt-free; no model.

On demand only: ``/book``, ``/check`` and once at 06:20 PT on weekdays. Each broker is read
in turn (Questrade, then IBKR over its own short-lived TWS client, id 9155), each with its
own cache: a snapshot at most 15 min old is reused; after a failed read, no new one for 1
hour (the Questrade refresh token is single-use; TWS is not hammered); no Questrade token =
no request; with the desk closed, no fetch. Never a periodic poll. The fetch runs on the
app's news thread (network, no GPU), so Pause AI does not block it. Snapshots and last
failures go to the app's own chat store (``app_state``); nothing else is written.
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


def fetch_note(out: dict[str, Any]) -> str:
    """The card's note for a broker not read now ("" when read or fresh)."""
    notes = []
    for name, part in (("Questrade", out), ("IBKR", out.get("ibkr"))):
        reason = str((part or {}).get("reason") or "")
        if part is not None and not part.get("fetched") and reason not in ("", "fresh"):
            notes.append(f"not read from {name} now: {reason}")
    return "; ".join(notes)


def _aware(now: datetime) -> datetime:
    return now if now.tzinfo else now.astimezone()


def _keys(broker: str) -> tuple[str, str]:
    from mentor_packs import book_pack

    if broker == "IBKR":
        return book_pack.IBKR_SNAPSHOT_KEY, book_pack.IBKR_STATUS_KEY
    return book_pack.SNAPSHOT_KEY, book_pack.STATUS_KEY


def stored_snapshot(store: Any, broker: str = "QUESTRADE") -> Any:
    from questrade_positions import BookSnapshot

    return BookSnapshot.from_json(store.get_state(_keys(broker)[0]))


def stored_status(store: Any, broker: str = "QUESTRADE") -> dict[str, Any] | None:
    try:
        value = json.loads(store.get_state(_keys(broker)[1]) or "null")
    except ValueError:
        return None
    return value if isinstance(value, dict) and value else None


def skip_reason(store: Any, now: datetime, broker: str = "QUESTRADE") -> str:
    """Why no fetch now ("" = fetch): a fresh snapshot, or the 1-hour backoff after a failure."""
    from questrade_positions import backoff_left

    moment = _aware(now)
    snap = stored_snapshot(store, broker)
    if snap is not None and snap.is_fresh(moment):
        return "fresh"
    status = stored_status(store, broker) or {}
    if status.get("reason") and status.get("reason") != "no token":
        left = backoff_left(status.get("at_utc"), moment)
        if left is not None:
            return f"backing off after a failed read ({int(left.total_seconds() // 60) + 1} min left)"
    return ""


def ensure_book(store: Any, now: datetime, *, fetch: Callable[..., Any] | None = None,
                desk_closed: Callable[[], bool | None] | None = None,
                ibkr_fetch: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Read each broker when due, in turn. Never raises.

    Returns Questrade's ``{fetched, reason}`` plus ``ibkr`` (the same for IBKR) when an IBKR
    reader is given; ``ibkr_fetch=None`` reads no IBKR (the app passes the real reader).
    """
    moment = _aware(now)
    probed: list[bool] = []

    def closed() -> bool:
        if not probed:
            try:
                probed.append(desk_closed is not None and desk_closed() is True)
            except Exception:  # noqa: BLE001 - a broken probe is "unknown", never "closed"
                probed.append(False)
        return probed[0]

    if fetch is None:
        from questrade_positions import fetch_book as fetch
    out = _ensure_one(store, moment, fetch, "QUESTRADE", closed)
    if ibkr_fetch is not None:
        out["ibkr"] = _ensure_one(store, moment, ibkr_fetch, "IBKR", closed)
    return out


def _ensure_one(store: Any, moment: datetime, fetch: Callable[..., Any], broker: str,
                closed: Callable[[], bool]) -> dict[str, Any]:
    why = skip_reason(store, moment, broker)
    if why:
        return {"fetched": False, "reason": why}
    if closed():
        return {"fetched": False, "reason": "the desk is closed"}
    snapshot_key, status_key = _keys(broker)
    try:
        snap, reason = fetch(moment)
    except Exception as exc:  # noqa: BLE001 - the fetchers never raise; a stub might
        snap, reason = None, f"{type(exc).__name__}: {exc}"[:200]
    stamp = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    if snap is None:
        store.set_state(status_key, json.dumps({"reason": str(reason or "unknown"), "at_utc": stamp}, sort_keys=True))
        return {"fetched": False, "reason": str(reason or "unknown"), "failed": True}
    store.set_state(snapshot_key, snap.as_json())
    store.set_state(status_key, "{}")
    return {"fetched": True, "reason": ""}


def store_sources(store: Any, base: Any = None) -> Any:
    """``book_pack`` sources that read the snapshot from this app's store (the journal stays ``mode=ro``)."""
    import dataclasses

    from mentor_packs import book_pack

    src = base or book_pack.live_sources()
    return dataclasses.replace(src, snapshot=lambda: _as_dict(stored_snapshot(store)),
                               status=lambda: stored_status(store),
                               ibkr_snapshot=lambda: _as_dict(stored_snapshot(store, "IBKR")),
                               ibkr_status=lambda: stored_status(store, "IBKR"))


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
