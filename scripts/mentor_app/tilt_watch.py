"""Tilt watch: the in-session run of the tilt pack, its challenge rows and their day-end grade. Qt-free.

Every 2 min in 06:30-13:00 PT on weekdays the window runs :func:`run_watch` on the news
thread (no model, no GPU; it works while AI is paused). A leg signature kept in
``app_state`` skips the build when nothing new reached the journal. Each observation not
seen before today becomes one ``challenges`` row (kind ``tilt``, evidence = its id and its
leg ids) and is marked seen, so a restart never repeats it. The window posts at most ONE
Inbox item per :data:`POST_EVERY`; observations wait through the spacing, quiet hours or a
mute and are posted together as that one item; a used daily cap or a new day drops them
(``/tilt`` still shows them). Nothing ever pops or moves the transcript. :func:`grade_open` grades each row after the day's close: rest-of-day realized
PnL after the observation vs before it; hit = a red rest of day.

P15b: the same pass finds trades CLOSED since the last one; each gets ONE "how did it feel?"
Inbox item, under the same cap, quiet hours and 30-min spacing as the tilt items.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")
KIND = "tilt"
CHECK_MS = 2 * 60 * 1000
SESSION_START = time(6, 30)
SESSION_END = time(13, 0)
#: At most one tilt item in the Inbox per half hour.
POST_EVERY = timedelta(minutes=30)
#: A row is graded once the day's regular session is over (16:15 ET leaves room for the last closes).
GRADE_AFTER = time(16, 15)
LAST_LEG_KEY = "tilt:last_leg"
SEEN_KEY = "tilt:seen:{day}"
LAST_POST_KEY = "tilt:last_post_utc"
#: P15b: the trades already asked "how did it feel" (one question per closed trade, ever).
FEEL_SEEN_KEY = "feel:seen:{day}"
#: P15b: closes held for their question until the Inbox accepts it (survives a restart).
FEEL_WAIT_KEY = "feel:waiting:{day}"


def _aware(now: datetime) -> datetime:
    return now if now.tzinfo else now.astimezone()


@dataclass
class TiltSchedule:
    """Due inside 06:30-13:00 PT on weekdays; the 2-min cadence is the window's timer."""

    def due(self, now: datetime) -> bool:
        local = _aware(now).astimezone(PT)
        return local.weekday() < 5 and SESSION_START <= local.time() < SESSION_END


def _json(text: str | None, default: Any) -> Any:
    try:
        value = json.loads(text) if text else default
    except ValueError:
        return default
    return value if isinstance(value, type(default)) else default


def challenge_id(day: str, observation_id: str) -> str:
    return f"tilt:{day}:{observation_id.removeprefix('tilt:')}"


def closed_today(journal: Path | str, day: str) -> list[dict[str, Any]]:
    """P15b: the trades CLOSED on ``day`` (ET), read-only: ``{trade_id, symbol, side}`` in close order."""
    from mentor_packs import journal_read

    out = []
    for trade in journal_read.read_trades(journal, since=day):
        closed = journal_read.parse_time(trade.get("closed_at"))
        if str(trade.get("status") or "").upper() != "CLOSED" or closed is None or closed.date().isoformat() != day:
            continue
        symbol = (journal_read.underlying(trade.get("symbol")) if journal_read.is_option(trade)
                  else str(trade.get("symbol") or "").upper())
        out.append({"trade_id": str(trade.get("trade_id") or ""), "symbol": symbol,
                    "side": str(trade.get("direction") or "").upper(), "closed_at": closed.isoformat()})
    return sorted((row for row in out if row["trade_id"]), key=lambda row: (row["closed_at"], row["trade_id"]))


def waiting_closes(store: Any, day: str) -> list[dict[str, Any]]:
    """The closes held for their question (quiet hours, a mute, the spacing), kept in ``app_state``."""
    return [dict(row) for row in _json(store.get_state(FEEL_WAIT_KEY.format(day=day)), []) if isinstance(row, dict)]


def pending_closes(store: Any, day: str, closed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every close still owed its question today: the held ones plus the new ones, persisted so a restart keeps
    them. A trade is marked asked only when its Inbox item is accepted (:func:`mark_asked`)."""
    seen = set(_json(store.get_state(FEEL_SEEN_KEY.format(day=day)), []))
    waiting = [row for row in waiting_closes(store, day) if row.get("trade_id") not in seen]
    held = {row["trade_id"] for row in waiting}
    added = [{**row, "day": day} for row in closed if row["trade_id"] not in seen and row["trade_id"] not in held]
    if added:
        waiting += added
        store.set_state(FEEL_WAIT_KEY.format(day=day), json.dumps(waiting, sort_keys=True))
    return waiting


def mark_asked(store: Any, day: str, trade_ids: Iterable[str]) -> None:
    """The Inbox took the question (or the cap dropped it): never asked again, no longer held."""
    ids = {str(item) for item in trade_ids}
    seen = set(_json(store.get_state(FEEL_SEEN_KEY.format(day=day)), [])) | ids
    store.set_state(FEEL_SEEN_KEY.format(day=day), json.dumps(sorted(seen)))
    left = [row for row in waiting_closes(store, day) if row.get("trade_id") not in seen]
    store.set_state(FEEL_WAIT_KEY.format(day=day), json.dumps(left, sort_keys=True))


FEEL_LOOKBACK_DAYS = 30


def resolve_trade(journal: Path | str, ref: str, now: datetime) -> dict[str, Any] | None:
    """``/feel <ref>``: a trade id as typed, else the symbol's latest CLOSED trade in the last 30 days."""
    from mentor_packs import journal_read

    since = (_aware(now).astimezone(ET).date() - timedelta(days=FEEL_LOOKBACK_DAYS)).isoformat()
    trades = journal_read.read_trades(journal, since=since)
    wanted = str(ref or "").strip()
    by_id = next((trade for trade in trades if str(trade.get("trade_id") or "") == wanted), None)
    if by_id is None:
        sym = wanted.lstrip("$").upper()
        closed = [trade for trade in trades if str(trade.get("status") or "").upper() == "CLOSED"
                  and sym in (str(trade.get("symbol") or "").upper(), journal_read.underlying(trade.get("symbol")))]
        closed.sort(key=lambda trade: str(trade.get("closed_at") or ""))
        by_id = closed[-1] if closed else None
    if by_id is None:
        return None
    symbol = (journal_read.underlying(by_id.get("symbol")) if journal_read.is_option(by_id)
              else str(by_id.get("symbol") or "").upper())
    return {"trade_id": str(by_id.get("trade_id") or ""), "symbol": symbol,
            "side": str(by_id.get("direction") or "").upper()}


def feeling_note(trade: Mapping[str, Any], words: str) -> str:
    """The stored note: the trader's words, labelled with the trade so recall finds it."""
    side = f" {str(trade.get('side') or '').lower()}" if trade.get("side") else ""
    return f"How the {trade.get('symbol')}{side} ({trade.get('trade_id')}) felt: {' '.join(str(words).split())}"


def feel_text(trade: Mapping[str, Any]) -> str:
    """The one feelings question for a closed trade."""
    side = f" {str(trade.get('side') or '').lower()}" if trade.get("side") else ""
    return f"How did the {trade.get('symbol') or 'last'}{side} feel? One word or a sentence."


def run_watch(store: Any, now: datetime, *, build: Callable[[], Pack],
              signature: Callable[[], Any] | None = None,
              closed: Callable[[str], list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """One watch pass. Returns ``{skipped, new, last_post, closed}``; ``new`` = observations first seen now,
    ``closed`` (P15b) = trades closed since the last pass, each to be asked how it felt once."""
    from mentor_packs import tilt_pack

    moment = _aware(now)
    day = moment.astimezone(ET).date().isoformat()
    sig = signature() if signature is not None else None
    marker = {"day": day, "signature": list(sig)} if sig is not None else None
    if marker is not None and _json(store.get_state(LAST_LEG_KEY), {}) == marker:
        # Nothing new in the journal; the held questions still come back (a restart keeps them).
        held = waiting_closes(store, day) if closed is not None else []
        return {"skipped": True, "new": [], "closed": held, "last_post": store.get_state(LAST_POST_KEY)}
    closes = pending_closes(store, day, closed(day)) if closed is not None else []
    pack = build()
    if not pack.rows:
        return {"skipped": False, "new": [], "closed": closes, "last_post": store.get_state(LAST_POST_KEY),
                "error": pack.empty_text}
    seen = set(_json(store.get_state(SEEN_KEY.format(day=day)), []))
    new = [row for row in tilt_pack.observations(pack) if row["id"] not in seen]
    issued = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    for row in new:
        store.add_challenge(
            challenge_id(day, row["id"]), kind=KIND, symbol=str(row.get("symbol") or ""), claim=str(row["text"]),
            evidence_ids=[row["id"]] + [f"leg:{leg}" for leg in row.get("legs") or ()], issued_utc=issued,
            outcome={"status": "open", "day": day, "at": row["at"], "pattern": row["kind"],
                     "before_pnl": row.get("before_pnl"), "legs": list(row.get("legs") or ())},
        )
    if new:
        store.set_state(SEEN_KEY.format(day=day), json.dumps(sorted(seen | {row["id"] for row in new})))
    if marker is not None:
        store.set_state(LAST_LEG_KEY, json.dumps(marker, sort_keys=True))
    return {"skipped": False, "new": new, "closed": closes, "last_post": store.get_state(LAST_POST_KEY)}


def may_post(last_post: str | None, now: datetime) -> bool:
    """True when no tilt item reached the Inbox in the last POST_EVERY."""
    if not last_post:
        return True
    try:
        then = datetime.fromisoformat(str(last_post))
    except ValueError:
        return True
    then = then if then.tzinfo else then.replace(tzinfo=timezone.utc)
    return _aware(now) - then >= POST_EVERY


def inbox_text(new: list[dict[str, Any]]) -> str:
    """One item: the first new observation, and how many more `/tilt` shows."""
    text = str(new[0]["text"])
    return text + (f" (+{len(new) - 1} more: /tilt)" if len(new) > 1 else "")


def card_markdown(pack: Pack) -> str:
    """The /tilt card: today's observations with their ids, then the base rates."""
    if not pack.rows:
        return f"**Tilt**: {pack.empty_text or 'nothing to show'}"
    lines = ["**Tilt watch: today**", ""]
    for row in pack.rows:
        if row.get("kind") == "base":
            continue
        lines.append(f"- {row['text']} [{row['id']}]")
    lines += ["", "*Base rates (journal history)*"]
    lines += [f"- {row['text']} [{row['id']}]" for row in pack.rows if row.get("kind") == "base"]
    lines += ["", "*Observations, not rules. The thresholds are yours to overrule.*"]
    return "\n".join(lines)


# ---------------------------------------------------------------- grading (no model)
def grade_open(store: Any, now: datetime, *, journal: Path | str | None = None) -> int:
    """Grade each open tilt row whose day is over: rest-of-day PnL after the observation vs before it."""
    from mentor_packs import journal_read, tilt_pack

    moment = _aware(now).astimezone(ET)
    path = Path(journal) if journal is not None else tilt_pack.live_journal()
    updated = 0
    cache: dict[str, list] = {}
    for row in store.challenges(kind=KIND, open_only=True):
        outcome = _json(row.get("outcome_json"), {})
        try:
            day = date.fromisoformat(str(outcome.get("day") or ""))
            at = datetime.fromisoformat(str(outcome.get("at") or ""))
        except ValueError:
            continue
        if moment < datetime.combine(day, GRADE_AFTER, tzinfo=ET):
            continue
        at = at if at.tzinfo else at.replace(tzinfo=ET)
        key = day.isoformat()
        if key not in cache:
            trades = journal_read.read_trades(path, since=key)
            cache[key] = tilt_pack.day_events([], trades, day)[1]
        closes = cache[key]
        before = round(tilt_pack.realized(closes, before=at), 2)
        rest = round(tilt_pack.realized(closes, after=at), 2)
        after = sum(1 for c in closes if c.at > at)
        new = {**outcome, "status": "graded", "before_pnl": before, "rest_pnl": rest, "closes_after": after}
        if after:
            new["hit"] = rest < 0
        else:  # nothing closed after it: no rest of day to judge, so never a hit or a miss (out of n)
            new["result"] = "no trades after"
            new.pop("hit", None)
        graded = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
        if store.update_challenge(row["id"], outcome=new, graded_utc=graded):
            updated += 1
    if updated:
        logging.info("Trade Mentor tilt: graded %d observation(s)", updated)
    return updated
