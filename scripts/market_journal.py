"""The trader's written record of the day — R10.H.

R10.G records what the machine saw. This is the other half: what the trader
saw, in their own words. The two together are what lets a later reader — human
or model — understand a session rather than re-derive it, which is the whole
point of the two-tier design.

**One writer, two surfaces.** A "Journal" tab on the Trading Desk after
"Capture" for a note written mid-session, and a left-nav "Market Journal" page
for the sit-down review. Both go through this module, so an entry means the
same thing whichever surface it came from, and there is exactly one store to
reason about.

The existing left-nav "Journal" page stays the **trade/tax** journal. The label
collision is deliberate and recorded: one is a record of what you traded, the
other of what you thought, and merging them would make the tax journal a diary.

**After-the-fact entries are first class, and never backdated** (decision
record §5a). On an AWAY day the trader comes in that evening and writes about
the session. That entry carries `session_date` = the session it is ABOUT and
`created_at` = when it was actually written, both tz-aware. An entry about
Friday written on Saturday says so, because a reader weighing "what did you
think at the time?" needs to know it was not written at the time.

Append-only, schema-NAMED, corrections supersede (ground rule 5). An edit is a
new entry naming the one it replaces; nothing is rewritten, because a journal
that can be quietly rewritten is not evidence about what anyone believed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

#: Schema NAME (ground rule 5).
SCHEMA_MARKET_JOURNAL_ENTRY = "market_journal_entry_v1"
STREAM = "market_journal"

#: What a timeframe label may be. M5 is the in-session default because the desk
#: tab is used while the tape is moving; D1 is the review default.
TIMEFRAME_M5 = "M5"
TIMEFRAME_D1 = "D1"
TIMEFRAMES = (TIMEFRAME_M5, "M15", "H1", TIMEFRAME_D1, "W1")

#: How an entry reached the store, kept because it changes how it should be read.
ORIGIN_DESK_TAB = "desk_tab"
ORIGIN_JOURNAL_PAGE = "journal_page"
ORIGIN_AWAY_RECAP = "away_recap"

#: The journal-only RVOL floor. It is an OVERLAY on this page's charts and
#: never touches the canonical D1 level store (trader decision, plan.md L1118).
RVOL_OVERLAY_FLOOR = 1.2


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment


def entry_id(session_date: str, created_at: str, text: str) -> str:
    """Stable id for one entry, from what it is about and when it was written."""
    digest = hashlib.sha256(
        f"{session_date}|{created_at}|{text}".encode("utf-8")
    ).hexdigest()[:12]
    return f"mj-{session_date}-{digest}"


def build_entry(
    *,
    text: str,
    session_date: str,
    timeframe: str = TIMEFRAME_M5,
    symbols: Iterable[str] = (),
    origin: str = ORIGIN_DESK_TAB,
    now: datetime | None = None,
    supersedes: str = "",
) -> dict[str, Any]:
    """One journal entry.

    `session_date` is what the entry is ABOUT. `created_at` is when it was
    written. They are separate fields precisely so an evening write-up of an
    AWAY day can be honest about both - and `written_after_the_session` is
    computed rather than asserted, so it cannot be set wrongly by a caller.
    """
    moment = _now(now)
    created_at = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    body = str(text or "").strip()
    session = str(session_date or "").strip()
    written_session = moment.astimezone().date().isoformat()
    return {
        "event_type": "entry",
        "entry_id": entry_id(session, created_at, body),
        "session_date": session,
        "created_at": created_at,
        "created_local_date": written_session,
        # Computed, never claimed. An entry about Friday written on Saturday
        # is weaker evidence about what the trader thought at the time, and a
        # reader must not have to work that out from two timestamps.
        "written_after_the_session": bool(session and written_session > session),
        "timeframe": _normalize_timeframe(timeframe),
        "symbols": [str(item).strip().upper() for item in (symbols or ()) if str(item).strip()],
        "origin": str(origin or ""),
        "text": body,
        "supersedes": str(supersedes or ""),
    }


def _normalize_timeframe(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in TIMEFRAMES else TIMEFRAME_M5


def is_publishable(entry: Mapping[str, Any]) -> tuple[bool, str]:
    """May this entry be written?

    An empty entry is refused rather than stored: a journal full of blanks is
    worse than a shorter one, because it makes the record look denser than the
    thinking behind it.
    """
    if not str(entry.get("text") or "").strip():
        return False, "an empty entry is not a thought; nothing is stored"
    if not str(entry.get("session_date") or "").strip():
        return False, "an entry with no session is unfiled and could never be read back"
    return True, ""


def supersede(entry: Mapping[str, Any], *, text: str, now: datetime | None = None) -> dict[str, Any]:
    """A correction: a NEW entry naming the one it replaces (ground rule 5).

    The original stays exactly as written. A journal that can be quietly
    rewritten is not evidence about what anyone believed - it is evidence about
    what they are willing to admit now.
    """
    return build_entry(
        text=text,
        session_date=str(entry.get("session_date") or ""),
        timeframe=str(entry.get("timeframe") or TIMEFRAME_M5),
        symbols=entry.get("symbols") or (),
        origin=str(entry.get("origin") or ORIGIN_JOURNAL_PAGE),
        now=now,
        supersedes=str(entry.get("entry_id") or ""),
    )


def resolve_entries(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The current view: superseded entries hidden, originals still on disk.

    Hiding is a READ-side decision. Every row remains in the ledger, so the
    history of a correction is recoverable even though the page shows only the
    latest text.
    """
    entries = [dict(row) for row in rows if str(row.get("event_type") or "") == "entry"]
    replaced = {str(row.get("supersedes") or "") for row in entries if row.get("supersedes")}
    current = [row for row in entries if str(row.get("entry_id") or "") not in replaced]
    current.sort(key=lambda row: (str(row.get("session_date") or ""), str(row.get("created_at") or "")))
    return current


def agreement_rate(shifts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """How often the machine's auto regime matched what the trader forced.

    Measured over regime-shift rows (R10.G). A session where the trader never
    overrode is AGREEMENT by silence and is counted as such; a session with no
    auto read at all is not counted either way, because there was nothing to
    agree with. Returns `None` rather than a rate when nothing is measurable -
    an agreement rate over zero comparable sessions is not 100%.
    """
    by_session: dict[str, dict[str, str]] = {}
    for row in shifts or ():
        session = str(row.get("session_date") or "")
        source = str(row.get("source") or "")
        target = str(row.get("to_regime") or "")
        if not session or not target:
            continue
        by_session.setdefault(session, {})[source] = target

    compared = 0
    agreed = 0
    for session, sources in by_session.items():
        auto = sources.get("auto")
        if not auto:
            continue
        compared += 1
        user = sources.get("user")
        if user is None or user == auto:
            agreed += 1
    return {
        "sessions_compared": compared,
        "sessions_agreed": agreed,
        "rate": round(agreed / compared, 4) if compared else None,
        "note": (
            "a session the trader never overrode counts as agreement; a session "
            "with no auto read is not counted either way"
            if compared
            else "no session carried an auto regime read, so the rate is UNMEASURED"
        ),
    }
