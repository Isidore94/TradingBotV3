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
#: The desk's own hand. An auto-mode flip writes a row so the journal reads as
#: one timeline - what the trader thought AND what the machine did, in order -
#: rather than requiring two stores to be merged by eye. It is marked as
#: machine-written because a reader weighing "what did you think?" must never
#: count a row nobody thought.
ORIGIN_AUTO_MODE_FLIP = "auto_mode_flip"
MACHINE_ORIGINS = (ORIGIN_AUTO_MODE_FLIP,)

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
        #
        # R4 A17: measured against the session's CLOSE, not against its date.
        # A note typed at 21:00 Pacific is written five hours after the market
        # shut, and under the date rule it claimed to have been written during
        # the session - which is the one thing this field exists to deny. The
        # date rule is kept as the fallback for a session the calendar cannot
        # place, because a slightly coarse answer is better than none.
        "written_after_the_session": _written_after_the_session(session, moment, written_session),
        "timeframe": _normalize_timeframe(timeframe),
        "symbols": [str(item).strip().upper() for item in (symbols or ()) if str(item).strip()],
        "origin": str(origin or ""),
        "text": body,
        "supersedes": str(supersedes or ""),
    }


def _written_after_the_session(session: str, moment: datetime, written_session: str) -> bool:
    """Had the trader already seen how that session finished?

    The honest reading of the question, which is what the field's name has
    always claimed and what `session_date_for`'s own docstring describes. At or
    after the session's regular close, the answer is yes - the same calendar the
    rest of the desk uses decides when that was.

    Falls back to the calendar-date comparison when the close cannot be read: an
    entry about Friday written on Saturday is still, unambiguously, after it.
    """
    if not session:
        return False
    try:
        from datetime import date as _date

        from market_calendar import session_close

        parts = [int(part) for part in session.split("-")]
        close = session_close(_date(parts[0], parts[1], parts[2]))
        stamp = moment if moment.tzinfo else moment.astimezone()
        return stamp >= close
    except Exception:  # noqa: BLE001 - a coarse answer beats no answer
        return bool(written_session > session)


def session_date_for(now: datetime | None = None) -> str:
    """The session a note typed NOW is about - V2 item 4, decision 0016 answer 11.

    Today's session until the close; then STILL today's session, right up to the
    next session's open. A thought written at 18:00 is about the day that just
    ended, and dating it tomorrow would file it against a session that has not
    happened yet. On a weekend or a holiday it is the last session that traded.

    **The roll is the OPEN, not midnight** (R4 A17). This read the calendar date
    in New York, so a Pacific trader typing at 21:00 PT - which is 00:00 ET the
    next day - filed their note against TOMORROW'S session, on a day that had
    not opened. Worse, `written_after_the_session` then computed False, so the
    row claimed the note was written during a session that had not started. The
    trader's own rule: the session ends at the close, and the note is about that
    session until the next one opens.

    This does NOT touch `written_after_the_session`, which `build_entry` still
    COMPUTES from `created_at`. The two answer different questions - which day
    the note is about, and whether the trader had already seen how that day
    finished - and the second is the one a reader needs to discount the first.

    Falls back to the local date if the calendar cannot be read. A note filed
    against today is a small error; a note that could not be written is a lost
    thought, and this function must never be the reason one is.
    """
    moment = _now(now)
    local = moment.astimezone()
    try:
        from market_calendar import is_session, previous_session
        from market_session import get_market_session_window

        window = get_market_session_window(reference=local)
        market_date = window.market_date
        if is_session(market_date) and local >= window.open_local:
            # Today TRADES and it has OPENED. Whether the close has passed or
            # not, the note is about today - before the close it is the running
            # session, after it the one that just finished.
            return market_date.isoformat()
        # Either today never trades (a weekend or a holiday) or it has not
        # opened yet - the small hours in New York, which is the evening on the
        # trader's own clock. Both mean the same thing: the note is about the
        # last session that actually traded.
        return previous_session(market_date).isoformat()
    except Exception:  # noqa: BLE001 - never the reason a thought is lost
        pass
    return local.date().isoformat()


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


def is_machine_entry(entry: Mapping[str, Any]) -> bool:
    """Did the desk write this row, rather than the trader?

    Asked at READ time off ``origin``, so no row needs a second field that
    could disagree with the first.
    """
    return str(entry.get("origin") or "") in MACHINE_ORIGINS


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
