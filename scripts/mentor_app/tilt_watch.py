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
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
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


def run_watch(store: Any, now: datetime, *, build: Callable[[], Pack],
              signature: Callable[[], Any] | None = None) -> dict[str, Any]:
    """One watch pass. Returns ``{skipped, new, last_post}``; ``new`` = observations first seen now."""
    from mentor_packs import tilt_pack

    moment = _aware(now)
    day = moment.astimezone(ET).date().isoformat()
    sig = signature() if signature is not None else None
    marker = {"day": day, "signature": list(sig)} if sig is not None else None
    if marker is not None and _json(store.get_state(LAST_LEG_KEY), {}) == marker:
        return {"skipped": True, "new": [], "last_post": store.get_state(LAST_POST_KEY)}
    pack = build()
    if not pack.rows:
        return {"skipped": False, "new": [], "last_post": store.get_state(LAST_POST_KEY), "error": pack.empty_text}
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
    return {"skipped": False, "new": new, "last_post": store.get_state(LAST_POST_KEY)}


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
