"""Focus membership as episodes, not snapshots — R10.E (F1–F6).

`human_focus_daily_picks.csv` is a **snapshot**: a row per symbol per day it was
seen on a list. Audit F5 measured what that cannot answer. 244 of 499
(symbol, side) pairs — **49%** — appear on two or more distinct sessions, DOCN
SHORT on seven. Is that a name that survived the day roll, or one the trader
re-added each morning? The snapshot cannot say, and the two mean opposite
things: the first is a bug in `expire_m5_if_new_day`, the second is conviction.

The caveat travels with the number in the audit precisely because the store
cannot resolve it. So membership becomes an **episode**: a name joins a list,
stays for a while, and leaves, and each of those is an event with a
`membership_episode_id` tying them together. A re-add after a departure is a
NEW episode, which is exactly the distinction F5 could not draw.

Three more things this store refuses to do:

* **It never reconstructs membership from current state.** If a snapshot was
  missed, that is an `observation_gap` row saying so. Inferring "it must have
  been there" would manufacture the very history the episodes exist to
  establish, and F4 already showed what that costs: `focus_auto_picks.json`
  does not exist for any historical date, so no owner is recoverable for any
  past pick — which is why `unknown_legacy` is a first-class owner here rather
  than a default of `trader`.
* **The pick key includes the category** (F3). `human_focus_tracking._pick_key`
  returns `(trade_date, symbol, side)` and the callers build dicts keyed on it,
  so a name on both the swing and the M5 list silently loses one row. The CSV
  shows **0** multi-source keys, and that absence is the signature of the
  collision rather than evidence against it.
* **It never blocks the Focus write.** A failed event append costs the event,
  never the pick. The trader's list is the product; this is evidence about it.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any, Mapping

#: Schema NAME (ground rule 5).
SCHEMA_FOCUS_MEMBERSHIP_EVENT = "focus_membership_event_v1"
STREAM = "focus_membership_events"

EVENT_JOINED = "joined"
EVENT_LEFT = "left"
EVENT_EXPIRED = "expired"
EVENT_OBSERVATION_GAP = "observation_gap"
EVENT_ENRICHED = "enriched"

#: Who put this name on the list.
OWNER_TRADER = "trader"
OWNER_MACHINE = "machine"
OWNER_UNKNOWN_LEGACY = "unknown_legacy"

#: Age buckets for an episode's length, in sessions on the list.
AGE_BUCKETS = ((1, "same_day"), (3, "1-2_sessions"), (6, "3-5_sessions"), (11, "6-10_sessions"))
AGE_BUCKET_LONG = "over_10_sessions"


def membership_key(symbol: str, side: str, category: str) -> str:
    """The identity of a membership. **Category is part of it** (F3).

    Without it a name on both the swing and the M5 list is one key, and the
    later row silently wins - which is why the picks CSV shows zero multi-source
    keys while the collision is happening on every such name.
    """
    return f"{str(symbol or '').strip().upper()}|{str(side or '').strip().lower()}|{str(category or '').strip().lower()}"


def episode_id(key: str, joined_at: str) -> str:
    """Stable id for one continuous stay on a list.

    Derived from the key and the join instant, so a re-add after a departure
    gets a DIFFERENT id - which is the whole point. Two episodes of the same
    name are two rows, and F5's 49% becomes answerable.
    """
    digest = hashlib.sha256(f"{key}|{joined_at}".encode("utf-8")).hexdigest()[:12]
    return f"{key.split('|')[0]}-{digest}"


def owner_for(marker: Mapping[str, Any] | None, *, markers_present: bool) -> str:
    """Who owns this pick.

    A marker means the machine adopted it. No marker, in a store that HAS
    markers, means the trader typed it - which is what makes "user-entered
    names are never auto-removed" structural rather than aspirational.

    No marker in a store that has NO markers at all is `unknown_legacy`, never
    `trader`: F4 measured that `focus_auto_picks.json` does not exist for any
    historical date, so attributing those picks to the trader would be
    inventing provenance the system never recorded.
    """
    if marker:
        return OWNER_MACHINE
    return OWNER_TRADER if markers_present else OWNER_UNKNOWN_LEGACY


def age_bucket(sessions: int | None) -> str:
    """An episode's length, bucketed. `None` in, `unknown` out."""
    if sessions is None:
        return "unknown"
    for bound, label in AGE_BUCKETS:
        if sessions < bound:
            return label
    return AGE_BUCKET_LONG


def joined_event(
    *,
    symbol: str,
    side: str,
    category: str,
    owner: str,
    joined_at: str,
    origin: str = "",
    context: str = "",
) -> dict[str, Any]:
    key = membership_key(symbol, side, category)
    return {
        "event_type": EVENT_JOINED,
        "membership_key": key,
        "membership_episode_id": episode_id(key, joined_at),
        "symbol": str(symbol or "").strip().upper(),
        "side": str(side or "").strip().lower(),
        "category": str(category or "").strip().lower(),
        "owner": owner,
        "joined_at": joined_at,
        "origin": str(origin or ""),
        "context": str(context or ""),
    }


def left_event(
    *,
    symbol: str,
    side: str,
    category: str,
    owner: str,
    episode: str,
    joined_at: str,
    left_at: str,
    reason: str,
    sessions_on_list: int | None = None,
) -> dict[str, Any]:
    key = membership_key(symbol, side, category)
    return {
        "event_type": EVENT_LEFT,
        "membership_key": key,
        "membership_episode_id": episode or episode_id(key, joined_at),
        "symbol": str(symbol or "").strip().upper(),
        "side": str(side or "").strip().lower(),
        "category": str(category or "").strip().lower(),
        "owner": owner,
        "joined_at": joined_at,
        "left_at": left_at,
        "reason": str(reason or ""),
        "days_on_list": sessions_on_list,
        "age_bucket": age_bucket(sessions_on_list),
    }


def expired_event(
    *, symbol: str, side: str, category: str, owner: str, episode: str, joined_at: str, at: str
) -> dict[str, Any]:
    """One row PER NAME the day roll clears (F5).

    `expire_m5_if_new_day` clears the whole M5 list at once, and a single
    "cleared N" row would leave a survivor invisible: a name still on the list
    tomorrow would look like a name that was never cleared. One row per name
    makes a survivor a test failure AND a visible gap.
    """
    row = left_event(
        symbol=symbol,
        side=side,
        category=category,
        owner=owner,
        episode=episode,
        joined_at=joined_at,
        left_at=at,
        reason="day_roll",
    )
    row["event_type"] = EVENT_EXPIRED
    return row


def observation_gap_event(*, expected_session: str, reason: str, seen_session: str = "") -> dict[str, Any]:
    """A snapshot that did not happen, recorded as a hole rather than filled.

    Membership is never reconstructed from current state. A gap says the
    evidence is missing; inferring the membership would manufacture the history
    this store exists to establish.
    """
    return {
        "event_type": EVENT_OBSERVATION_GAP,
        "expected_session": str(expected_session or ""),
        "last_seen_session": str(seen_session or ""),
        "reason": str(reason or ""),
        "note": (
            "no snapshot covers this session, so membership across it is UNKNOWN "
            "and is deliberately not reconstructed from current state"
        ),
    }


def enriched_event(*, episode: str, membership_key: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """A later revision carrying what the write path could not afford to compute.

    Append-only stores exist so a fact learned later is recorded later rather
    than retro-fitted onto a row that predates it. This runs on a worker; the
    Focus write never waits for it (ground rule 9).
    """
    return {
        "event_type": EVENT_ENRICHED,
        "membership_episode_id": episode,
        "membership_key": membership_key,
        **{f"enriched_{name}": value for name, value in (fields or {}).items()},
    }


def sessions_between(joined_at: str, left_at: str) -> int | None:
    """Business days between two ISO stamps, or None if either is unreadable."""
    def _parse(value: str) -> date | None:
        text = str(value or "")[:10]
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            return None

    first, last = _parse(joined_at), _parse(left_at)
    if first is None or last is None:
        return None
    import numpy as np

    return int(np.busday_count(first, last)) + 1
