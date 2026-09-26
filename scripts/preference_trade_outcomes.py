"""What I said, what I did, what happened — P6.

Four stores already answer one third of this each. The annotation log knows what
the trader SAID about a name; the journal knows what they TRADED; the cohort
rollups know what the name then DID on paper. Nothing put the three on one row,
so the question the whole capture programme exists to answer —

    *of the setups I liked, which did I actually take, and how did the ones I
    skipped do?*

— could only be answered by opening three files and joining them by eye.

This is that join, and it is deliberately a REPORT rather than a link.

**Every row renders its match confidence, or says "no match".** Nothing here
mints an identifier: `plan.md` P5.3/P5.4 own the canonical opportunity id, and a
second one invented in a nightly report would compete with it while being
weaker. What a row carries instead is a `trade_id` when a trade was found, a
stated confidence, and `match_basis` naming what the match rested on. A reader
can always see how firm the link is.

**Read-only, and it never writes into the journal.** `trade_annotations` are
trader-owned; this module reads `list_trades` and writes one CSV of its own.
Following `journal_walkaway`'s pattern: pure computation over stores something
else fills, publishing a file nothing scores from.

**A missing half is stated, never filled.** A statement with no trade is the
most interesting row in the file — it is the skip — so it is written with an
empty `trade_id` and an explicit `traded` of "no". A trade whose paper grade has
not matured yet carries a blank forward return, never a zero.

**The record is SYMMETRIC (WS-5B, WISHLIST 5, block B).** Until 2026-09-13 this
module asked the stores for likes, favorites and passes and never asked for a
rejection at all, so the report could say *"you liked it and did not take it"*
and could never say *"you vetoed it and took it anyway"* — which is the half of
the record that costs money. Every explicit verdict is now a statement in its
own channel, `verdict_family` keeps the two halves from ever being pooled, and
`unfavorite` is still absent BY DECISION: taking a name out of Focus is
housekeeping, not a judgement (CLAUDE.md P5, "`unfavorite` is never graded").
"""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import market_calendar
from journal_analytics import trade_r_multiple
from project_paths import OUTPUT_DIR

_log = logging.getLogger(__name__)

#: v2 adds the reject half of the record and the three columns that describe it
#: (WS-5B). The first nineteen columns did not move: `ai_summary.
#: preference_to_trade_section` reads `match_basis`, `trade_id` and
#: `session_date` out of this file by name, and a reader of the published 19
#: sees exactly what it saw before.
SCHEMA = "preference_trade_outcomes_v2"

#: Where the report lands. Beside the other read-only journal reports.
REPORT_FILE = OUTPUT_DIR / "preference_trade_outcomes.csv"

#: How many days back a nightly run considers. Long enough to cover a swing
#: idea's life, short enough that the file stays readable.
DEFAULT_WINDOW_DAYS = 45

#: How many SESSIONS after the statement a trade may open and still count as
#: acting on it. A trade three weeks later is a different decision.
#:
#: ST5.1: the constant was named ``_DAYS`` and the arithmetic was
#: ``timedelta(days=10)`` while every docstring in the module said "sessions".
#: Over a Labor Day week that is five real sessions thrown away - a statement on
#: Friday 2026-09-04 reached only 2026-09-14 when ten sessions run to
#: 2026-09-21, so a trade taken on the 18th read as "never taken".
TRADE_WINDOW_SESSIONS = 10

#: Deprecated alias kept for one release in case a caller outside this repo
#: imported the old name. Nothing in `scripts/` reads it (grep, 2026-09-06).
TRADE_WINDOW_DAYS = TRADE_WINDOW_SESSIONS

#: The window, spelled the way the report has to name it. Units carry their name.
TRADE_WINDOW_NOTE = f"{TRADE_WINDOW_SESSIONS} sessions"

COLUMNS = [
    "schema",
    "generated_at",
    "session_date",
    "symbol",
    "side",
    # WHAT YOU SAID
    "channel",
    "statement",
    "statement_detail",
    "statement_id",
    # WHAT YOU DID
    "traded",
    "trade_id",
    "trade_opened_at",
    "match_confidence",
    "match_basis",
    # WHAT HAPPENED
    "journal_r",
    "journal_net_pnl",
    "paper_forward_return_h3",
    "paper_forward_return_h5",
    "paper_cohort",
    # WS-5B, AT THE END. Everything above is the published contract.
    "like_mode",
    "verdict_family",
    "match_state",
]

#: The two halves of the record. They are never added into one number: "I said
#: take it" and "I said leave it" are two answers to two different questions.
FAMILY_ENDORSE = "endorse"
FAMILY_REJECT = "reject"

#: Every channel that carries a REFUSAL. One map, because a channel that lands
#: in the wrong family pools a veto with a like and no downstream reader could
#: tell. A channel absent from this set is an endorsement — which is what every
#: channel that existed before WS-5B was, so an old row read back with a blank
#: `verdict_family` reads `endorse` and is right.
REJECT_CHANNELS = frozenset(
    {
        "annotation:veto",
        "annotation:pass",
        "pick_feedback:dislike",
        "pick_feedback:not_today",
        "review_event:m5_click_away",
    }
)

#: What the join could say. `matched` is a trade found (however weak the basis);
#: the three misses are DIFFERENT answers and a single blank cannot carry them.
MATCH_STATE_MATCHED = "matched"
MATCH_STATE_WINDOW_OPEN = "window_open"
MATCH_STATE_NO_MATCH_AFTER_WINDOW = "no_match_after_window"
MATCH_STATE_JOURNAL_UNAVAILABLE = "journal_unavailable"
#: RESERVED and never emitted today (lead ruling, WS-5B): no path in this module
#: can currently tell "the matcher could not run" apart from "the journal could
#: not be read", and inventing a path to produce it would be inventing evidence.
#: The name is in the vocabulary so a later packet that CAN tell them apart does
#: not have to rename `journal_unavailable` out from under a shipped reader.
MATCH_STATE_MATCHING_UNAVAILABLE = "matching_unavailable"
MATCH_STATES = (
    MATCH_STATE_MATCHED,
    MATCH_STATE_WINDOW_OPEN,
    MATCH_STATE_NO_MATCH_AFTER_WINDOW,
    MATCH_STATE_JOURNAL_UNAVAILABLE,
    MATCH_STATE_MATCHING_UNAVAILABLE,
)


def verdict_family_for(channel: Any) -> str:
    """`endorse` or `reject` for one channel name. The ONE place it is decided."""
    return FAMILY_REJECT if str(channel or "") in REJECT_CHANNELS else FAMILY_ENDORSE


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("SHORT") or text in {"SELL", "S"}:
        return "SHORT"
    if text.startswith("LONG") or text in {"BUY", "B"}:
        return "LONG"
    return ""


def _canonical_r(trade) -> str:
    """The journal's one R (`journal_analytics.trade_r_multiple`, native), as text; blank when unknown."""
    value = trade_r_multiple(dict(trade))
    return "" if value is None else f"{value:.4f}"


def _float_or_blank(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if number != number else f"{number:.4f}"


# ---------------------------------------------------------------------------
# what you said
# ---------------------------------------------------------------------------
def collect_statements(
    *,
    since: date,
    until: date,
    annotations_path: Path | None = None,
    feedback_path: Path | None = None,
    favorites_path: Path | None = None,
    events_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Every explicit verdict the trader made about a name in the window.

    Four stores, each read through the module that owns it rather than a second
    parser here, and BOTH families of verdict:

    * ``like_claim`` (with its P9 mode), ``veto`` (with its reason code and the
      vocabulary version that coded it) and ``pass`` from the annotation log;
    * ``swing_favorite`` from the swing favorites store - the trader's own
      end-of-day list, resolved per session so a retraction is honoured;
    * ``like``, ``dislike`` and ``not_today`` from `pick_feedback`;
    * the M5 click-away from the review-event store, because a click away from
      an M5 alert IS a pass (trader, 2026-09-01).

    ``unfavorite`` is NOT here and never will be: taking a name out of Focus is
    housekeeping, and reading it as a negative judgement would teach the loop a
    lesson the trader never gave it (CLAUDE.md P5).

    A statement with no side is KEPT and marked, unlike the cohorts which
    refuse to grade one: this report is about what was said and whether it was
    acted on, and a sideless statement was still made.

    **One thing said once is ONE statement.** The annotation log heals torn
    tails rather than claiming atomicity, so a row can reach the file twice; a
    duplicate would double its family count. Identity is the store's own event
    id where there is one, and the statement itself where there is not.
    """
    statements: list[dict[str, Any]] = []
    statements.extend(_annotation_statements(since, until, annotations_path))
    statements.extend(_favorite_statements(since, until, favorites_path))
    statements.extend(_feedback_statements(since, until, feedback_path))
    statements.extend(_review_event_statements(since, until, events_path))
    statements = _deduplicate(statements)
    statements.sort(key=lambda row: (row["session_date"], row["symbol"], row["channel"]))
    return statements


def _statement_identity(row: Mapping[str, Any]) -> tuple:
    """What makes two rows the SAME statement.

    The store's own id when it has one - that is the strongest identity there
    is - and otherwise the statement itself, because a store with no id (swing
    favorites, `pick_feedback`) cannot distinguish a torn duplicate from a
    second identical click, and one of those two readings is always safe.
    """
    ident = str(row.get("statement_id") or "").strip()
    if ident:
        return (str(row.get("channel") or ""), ident)
    return (
        str(row.get("channel") or ""),
        row.get("session_date"),
        str(row.get("symbol") or ""),
        str(row.get("side") or ""),
        str(row.get("statement") or ""),
        str(row.get("statement_detail") or ""),
    )


def _deduplicate(statements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for row in statements:
        key = _statement_identity(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _veto_detail(row: Mapping[str, Any]) -> str:
    """The reason code AND the vocabulary version that coded it.

    Cohort identity on write is `(vocab_version, reason_code)`, so a reject
    statement that dropped the version could not be joined back to the cohort
    it was graded in. An UNCODED veto is legal and is not a coded one (P10 A1):
    it carries no code and no version, and it says so rather than borrowing the
    current vocabulary's number.
    """
    code = str(row.get("reason_code") or "").strip()
    version = str(row.get("vocab_version") or "").strip()
    if code and version:
        return f"{code} (v{version})"
    if code:
        return code
    return "uncoded"


def _annotation_statements(since: date, until: date, path: Path | None) -> list[dict[str, Any]]:
    try:
        from project_paths import TRADER_ANNOTATIONS_FILE
        from ui.annotations.store import (
            EVENT_LIKE_CLAIM,
            EVENT_PASS,
            EVENT_VETO,
            like_mode_of,
            load_annotations,
        )

        rows = load_annotations(
            Path(path or TRADER_ANNOTATIONS_FILE),
            # WS-5B: the veto joins the two that were already asked for. The
            # reject half was missing because nothing ever ASKED for it.
            event_types=(EVENT_LIKE_CLAIM, EVENT_PASS, EVENT_VETO),
        )
    except Exception as exc:  # noqa: BLE001 - a channel is never worth the report
        _log.debug("Annotation statements unavailable: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        session = _as_date(row.get("session_date"))
        if session is None or not (since <= session <= until):
            continue
        kind = str(row.get("event_type") or "")
        like_mode = ""
        if kind == "like_claim":
            statement = "liked"
            detail = str(row.get("claimed_setup_id") or "")
            # P9: a quick like names no setup and a claimed one must. Absence
            # reads `claimed`, because a claim was REQUIRED until P9 - and that
            # rule lives in `like_mode_of`, never copied here.
            like_mode = like_mode_of(row)
        elif kind == "veto":
            statement = "vetoed"
            detail = _veto_detail(row)
        else:
            statement = "passed"
            codes = [str(code or "").strip() for code in (row.get("reason_codes") or []) if code]
            detail = ", ".join(codes)
        out.append(
            {
                "session_date": session,
                "symbol": _symbol(row.get("symbol")),
                "side": _side(row.get("side")),
                "channel": f"annotation:{kind}",
                "statement": statement,
                "statement_detail": detail,
                "statement_id": str(row.get("event_id") or ""),
                "like_mode": like_mode,
                "verdict_family": verdict_family_for(f"annotation:{kind}"),
            }
        )
    return out


def _favorite_statements(since: date, until: date, path: Path | None) -> list[dict[str, Any]]:
    try:
        import swing_favorites

        raw = (
            swing_favorites.load_rows(path) if path else swing_favorites.load_rows()
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug("Swing favorites unavailable: %s", exc)
        return []

    # Resolved PER SESSION through the store's own `favorites_for_session`, so a
    # name the trader added and then retracted is not reported as a pick they
    # never took. The append-only log keeps the retraction; the live list for
    # that session is what they actually stood behind.
    sessions = sorted(
        {
            str(row.get("session_date") or "").strip()
            for row in raw
            if str(row.get("session_date") or "").strip()
        }
    )
    out: list[dict[str, Any]] = []
    for session_text in sessions:
        session = _as_date(session_text)
        if session is None or not (since <= session <= until):
            continue
        for row in swing_favorites.favorites_for_session(session_text, rows=raw):
            out.append(
                {
                    "session_date": session,
                    "symbol": _symbol(row.get("symbol")),
                    "side": _side(row.get("side")),
                    "channel": "swing_favorite",
                    "statement": "picked",
                    "statement_detail": str(row.get("origin") or "today's swing list"),
                    "statement_id": "",
                    # A favorite is not a rail like, and it never had a mode.
                    "like_mode": "",
                    "verdict_family": FAMILY_ENDORSE,
                }
            )
    return out


#: The `pick_feedback` verdicts that are STATEMENTS, and what each one says.
#:
#: `unfavorite` is deliberately not a key. It is in neither of `pick_feedback`'s
#: own LIKE/REJECT maps either, for the same reason: removing a name from Focus
#: is housekeeping and the trader never passed judgement on it.
#:
#: `not_today` is NARROWER than `dislike` (packet R2) - one session thrown back,
#: not the name itself - so the two ride in separate channels and are never
#: combined into one verdict.
_FEEDBACK_STATEMENTS: dict[str, tuple[str, str]] = {
    "like": ("pick_feedback:like", "liked"),
    "dislike": ("pick_feedback:dislike", "disliked"),
    "not_today": ("pick_feedback:not_today", "not today"),
}


def _feedback_statements(since: date, until: date, path: Path | None) -> list[dict[str, Any]]:
    try:
        import json

        from project_paths import PICK_FEEDBACK_FILE

        target = Path(path or PICK_FEEDBACK_FILE)
        if not target.exists():
            return []
        raw: list[dict[str, Any]] = []
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and str(row.get("verdict") or "") in _FEEDBACK_STATEMENTS:
                    raw.append(row)
    except OSError as exc:
        _log.debug("Pick feedback unavailable: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for row in raw:
        session = _as_date(row.get("trade_date"))
        if session is None or not (since <= session <= until):
            continue
        verdict = str(row.get("verdict") or "")
        channel, statement = _FEEDBACK_STATEMENTS[verdict]
        origin = str(row.get("origin") or "")
        if verdict == "like":
            detail = origin
        else:
            # The trader's OWN WORDS, carried so a reader can see why. It is
            # never machine-coded into a reason cohort (CLAUDE.md P5) and it is
            # not one of `ai_summary.PREFERENCE_EXAMPLE_COLUMNS`, so it reaches
            # a person and not a model.
            detail = str(row.get("reason") or "").strip() or origin
        out.append(
            {
                "session_date": session,
                "symbol": _symbol(row.get("symbol")),
                "side": _side(row.get("side")),
                "channel": channel,
                "statement": statement,
                "statement_detail": detail,
                "statement_id": "",
                # A ★ on a board carries no P9 mode: it was neither the quick
                # key nor the claim dialog, and naming either would be a claim
                # about a keypress that never happened.
                "like_mode": "",
                "verdict_family": verdict_family_for(channel),
            }
        )
    return out


def _review_event_statements(since: date, until: date, path: Path | None) -> list[dict[str, Any]]:
    """The M5 click-away. A click away from an M5 alert IS a pass.

    Trader, 2026-09-01 - never "fixed", and `clicked_away_from_m5_alert` is
    never renamed, because `review_learning` and `pick_feedback` both key on
    it. It has no verb of its own: it is an `action: "skip"` review event whose
    detail reason is that string.
    """
    try:
        import project_paths
        from pick_feedback import M5_CLICK_AWAY_REASON
        from review_events import load_review_events

        target = Path(path) if path is not None else Path(project_paths.ALERT_REVIEW_EVENTS_FILE)
        rows = load_review_events(target)
    except Exception as exc:  # noqa: BLE001 - a channel is never worth the report
        _log.debug("Review-event statements unavailable: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("action") or "").strip().lower() != "skip":
            continue
        detail = row.get("detail")
        reason = ""
        if isinstance(detail, dict):
            reason = str(detail.get("reason") or "").strip().lower()
        if reason != M5_CLICK_AWAY_REASON:
            continue
        session = _as_date(row.get("trade_date")) or _as_date(row.get("ts"))
        if session is None or not (since <= session <= until):
            continue
        out.append(
            {
                "session_date": session,
                "symbol": _symbol(row.get("symbol")),
                "side": _side(row.get("side")),
                "channel": "review_event:m5_click_away",
                "statement": "clicked away",
                "statement_detail": M5_CLICK_AWAY_REASON,
                "statement_id": str(row.get("review_record_id") or ""),
                "like_mode": "",
                "verdict_family": FAMILY_REJECT,
            }
        )
    return out


# ---------------------------------------------------------------------------
# what you did
# ---------------------------------------------------------------------------
def statement_window_end(said_on: date, sessions: int = TRADE_WINDOW_SESSIONS) -> date:
    """The last calendar date a trade may OPEN on and still count as acting.

    ST5.1. Ten SESSIONS walked on the exchange calendar, never ten calendar
    days: ``market_calendar.trading_days_between(said_on, result) == sessions``
    by construction, so a Labor Day inside the window pushes the end out rather
    than eating a session. ``said_on`` itself is never counted (the calendar's
    own convention), which also answers the non-session case for free - a
    statement made on a Saturday starts counting at the next session.

    A calendar that refuses (outside its validated 2000-2032 range) is
    uncertainty, and uncertainty here must not silently widen the window into a
    match that was never made: the fallback is the OLD calendar-day arithmetic,
    which is strictly narrower, and it is logged.
    """
    wanted = max(1, int(sessions))
    try:
        cursor = said_on
        counted = 0
        # A ten-session window cannot exceed a fortnight of weekends plus the
        # longest holiday cluster; the bound exists so a calendar bug cannot
        # become an infinite loop.
        for _ in range(wanted * 3 + 30):
            cursor += timedelta(days=1)
            if market_calendar.is_session(cursor):
                counted += 1
                if counted >= wanted:
                    return cursor
    except market_calendar.SessionCalendarError as exc:
        _log.debug("Session window fell back to calendar days for %s: %s", said_on, exc)
    else:
        _log.debug("Session window for %s never reached %d sessions", said_on, wanted)
    return said_on + timedelta(days=wanted)


def match_trade(statement: Mapping[str, Any], trades: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The trade that acted on this statement, with a stated confidence.

    Never a hard link. The best available evidence is (symbol, side, a trade
    opened on or within ``TRADE_WINDOW_SESSIONS`` sessions after the statement),
    and that is a JUDGEMENT: the trader could have taken the name for an
    unrelated reason the same week. So the row carries what the match rested on
    and how firm it is, and a reader can discount it.

    * symbol + side + same day -> 0.9, "symbol+side+same_session"
    * symbol + side + inside the window -> 0.7, "symbol+side+window"
    * symbol only (a sideless statement, or the trader took it the other way)
      -> 0.5, "symbol+window_side_unknown"
    * nothing -> confidence 0.0 and "no match", which is the interesting row.
    """
    symbol = statement.get("symbol")
    said_on = statement.get("session_date")
    side = statement.get("side") or ""
    if not symbol or not isinstance(said_on, date):
        return {"trade": None, "confidence": 0.0, "basis": "no match"}

    window_end = statement_window_end(said_on)
    best: tuple[float, str, Mapping[str, Any]] | None = None
    for trade in trades:
        if _symbol(trade.get("symbol")) != symbol:
            continue
        opened = _as_date(trade.get("opened_at")) or _as_date(trade.get("trade_date"))
        if opened is None or not (said_on <= opened <= window_end):
            continue
        trade_side = _side(trade.get("direction"))
        if side and trade_side and trade_side == side:
            confidence, basis = (
                (0.9, "symbol+side+same_session") if opened == said_on else (0.7, "symbol+side+window")
            )
        elif side and trade_side and trade_side != side:
            # Taken the other way round. Still an action on the name, and the
            # row says so rather than claiming the statement was followed.
            confidence, basis = 0.35, "symbol+window_opposite_side"
        else:
            confidence, basis = 0.5, "symbol+window_side_unknown"
        if best is None or confidence > best[0]:
            best = (confidence, basis, trade)
    if best is None:
        return {"trade": None, "confidence": 0.0, "basis": "no match"}
    return {"trade": best[2], "confidence": best[0], "basis": best[1]}


# ---------------------------------------------------------------------------
# what happened
# ---------------------------------------------------------------------------
def load_paper_grades() -> dict[tuple[str, str, str], dict[str, Any]]:
    """(session, symbol, side) -> the cohort's forward return for that pick.

    Read from the cohort OUTCOME files that already exist - the like, pass and
    rejection trios and the focus rollup - so this report never recomputes a
    forward return. Ground rule 6: reformatted, never derived.

    Absent files are absent grades, and a row with no grade renders blank.
    """
    grades: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        import project_paths
    except Exception:  # noqa: BLE001
        return grades

    for attribute in (
        "LIKE_COHORT_OUTCOMES_FILE",
        "HUMAN_FOCUS_OUTCOMES_FILE",
        "VETO_COHORT_OUTCOMES_FILE",
    ):
        path = getattr(project_paths, attribute, None)
        if path is None or not Path(path).is_file():
            continue
        try:
            with Path(path).open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = (
                        str(row.get("trade_date") or "").strip(),
                        _symbol(row.get("symbol")),
                        _side(row.get("side")),
                    )
                    if not key[0] or not key[1]:
                        continue
                    grades.setdefault(
                        key,
                        {
                            "h3": row.get("h3_return"),
                            "h5": row.get("h5_return"),
                            "cohort": str(row.get("source") or ""),
                        },
                    )
        except OSError as exc:
            _log.debug("Paper grade file unreadable (%s): %s", attribute, exc)
    return grades


def match_state_for(
    said_on: date | None,
    *,
    matched: bool,
    reference: date,
    journal_available: bool = True,
) -> str:
    """Which of the four answers this row is. WS-5B.

    A blank ``trade_id`` was carrying three different facts at once, and they
    are not the same answer:

    * ``matched``               -- a trade was found. How firm the link is stays
      in ``match_confidence`` / ``match_basis``; this column never upgrades it.
    * ``window_open``           -- the 10-SESSION window has not closed yet.
      Not a miss; not yet an answer.
    * ``no_match_after_window`` -- the window closed with no trade. The real
      "said it, did not do it".
    * ``journal_unavailable``   -- the journal could not be read, so the second
      half of the row was never measured.

    The arithmetic is `statement_window_end(said_on) > reference`, which is the
    SAME comparison `ai_summary._preference_window_open` already makes from
    `match_basis` + `session_date`. Two readers, one answer: if they disagreed
    the desk would hold two truths about one row.
    """
    if not journal_available:
        return MATCH_STATE_JOURNAL_UNAVAILABLE
    if matched:
        return MATCH_STATE_MATCHED
    if not isinstance(said_on, date):
        # Uncertainty reads CLOSED, exactly as the AI section reads it: a date
        # nobody can parse must not count as "still waiting", which would
        # quietly shrink the number of real misses.
        return MATCH_STATE_NO_MATCH_AFTER_WINDOW
    return (
        MATCH_STATE_WINDOW_OPEN
        if statement_window_end(said_on) > reference
        else MATCH_STATE_NO_MATCH_AFTER_WINDOW
    )


def build_rows(
    statements: list[dict[str, Any]],
    trades: list[Mapping[str, Any]],
    *,
    grades: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    now: datetime | None = None,
    journal_available: bool = True,
) -> list[dict[str, Any]]:
    """One row per statement: what was said, whether it was taken, what it did.

    ``journal_available=False`` is the honest shape of an unreadable journal:
    the statements are still published, ``match_basis`` stays EMPTY (which is
    exactly how `ai_summary.preference_to_trade_section` already recognises the
    bucket) and every row says ``journal_unavailable`` in its own column. The
    trader still SAID it; a gap in one half of the row is not a reason to
    publish nothing.
    """
    moment = now or datetime.now()
    stamp = moment.isoformat(timespec="seconds")
    reference = moment.date()
    grades = grades if grades is not None else {}
    rows: list[dict[str, Any]] = []
    for statement in statements:
        match = (
            match_trade(statement, trades)
            if journal_available
            else {"trade": None, "confidence": 0.0, "basis": ""}
        )
        trade = match["trade"]
        session_text = statement["session_date"].isoformat()
        grade = grades.get((session_text, statement["symbol"], statement["side"])) or {}
        rows.append(
            {
                "schema": SCHEMA,
                "generated_at": stamp,
                "session_date": session_text,
                "symbol": statement["symbol"],
                "side": statement["side"],
                "channel": statement["channel"],
                "statement": statement["statement"],
                "statement_detail": statement["statement_detail"],
                "statement_id": statement["statement_id"],
                # The plainest column in the file, and the one the whole report
                # exists for.
                "traded": "yes" if trade is not None else "no",
                "trade_id": str(trade.get("trade_id") or "") if trade else "",
                "trade_opened_at": str(trade.get("opened_at") or "") if trade else "",
                "match_confidence": f"{float(match['confidence']):.2f}" if trade else "",
                "match_basis": match["basis"],
                # The journal's one R, native currency; blank when no risk was typed.
                "journal_r": _canonical_r(trade) if trade else "",
                # A money column, so CAD: it is summed across trades of mixed currency.
                "journal_net_pnl": _float_or_blank(trade.get("net_pnl_cad")) if trade else "",
                "paper_forward_return_h3": _float_or_blank(grade.get("h3")),
                "paper_forward_return_h5": _float_or_blank(grade.get("h5")),
                "paper_cohort": str(grade.get("cohort") or ""),
                # WS-5B. A statement that arrived without a family (a caller
                # building rows by hand, or a channel added later) is read from
                # its channel rather than guessed.
                "like_mode": str(statement.get("like_mode") or ""),
                "verdict_family": str(statement.get("verdict_family") or "")
                or verdict_family_for(statement.get("channel")),
                "match_state": match_state_for(
                    statement.get("session_date"),
                    matched=trade is not None,
                    reference=reference,
                    journal_available=journal_available,
                ),
            }
        )
    return rows


def trade_level_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and money BY TRADE, over a file that is one row per STATEMENT.

    ST5.2. Both grains are real and neither replaces the other: a statement is
    what the trader said, and three statements about one name are three
    statements. But the P&L belongs to the TRADE, and summing the statement rows
    counted it once per thing the trader said about it. Live on 2026-09-06:
    **13 rows with ``traded=yes`` over 10 distinct ``trade_id``s**, so a
    statement-grain total was three trades' P&L too large.

    Every count here says its grain, and ``duplicate_statement_rows`` is the
    difference between the two - the number a reader needs to see that the file
    has more rows than trades ON PURPOSE.

    ``planned_risk_recorded`` counts distinct matched trades whose ``journal_r``
    is present. Blank R means the trader never typed a plan; it is silence, not
    a zero, and nothing here fills it (ST5.5's invariant).
    """
    statements_per_trade: dict[str, int] = {}
    net_by_trade: dict[str, float] = {}
    risk_by_trade: dict[str, bool] = {}
    matched = 0
    all_rows = list(rows)
    # WS-5B: the two families, ALWAYS both keys. A missing key would read as a
    # family with no rows and a family that does not exist, and those are
    # different facts. A row with no family at all is an endorsement, because
    # every channel that could write one before WS-5B was.
    by_family: dict[str, int] = {FAMILY_ENDORSE: 0, FAMILY_REJECT: 0}
    for row in all_rows:
        family = str((row or {}).get("verdict_family") or "").strip() or FAMILY_ENDORSE
        by_family[family] = by_family.get(family, 0) + 1
        trade_id = str((row or {}).get("trade_id") or "").strip()
        if not trade_id:
            continue
        matched += 1
        statements_per_trade[trade_id] = statements_per_trade.get(trade_id, 0) + 1
        # ONCE per trade. `setdefault` is the whole fix: the second statement
        # about a trade contributes a row and no money.
        if trade_id not in net_by_trade:
            try:
                net_by_trade[trade_id] = float((row or {}).get("journal_net_pnl"))
            except (TypeError, ValueError):
                net_by_trade[trade_id] = 0.0
        if trade_id not in risk_by_trade:
            risk_by_trade[trade_id] = bool(str((row or {}).get("journal_r") or "").strip())

    n_trades = len(statements_per_trade)
    recorded = sum(1 for present in risk_by_trade.values() if present)
    return {
        "n_statements": len(all_rows),
        "n_statements_by_family": by_family,
        "n_statements_matched": matched,
        "n_trades_matched": n_trades,
        "net_pnl": sum(net_by_trade.values()),
        "duplicate_statement_rows": matched - n_trades,
        "statements_per_trade": statements_per_trade,
        "planned_risk_recorded": recorded,
        "planned_risk_note": (
            f"planned risk recorded on {recorded} of {n_trades} matched trades"
        ),
        "window_note": TRADE_WINDOW_NOTE,
    }


def summary_note(rows: Iterable[Mapping[str, Any]], summary: Mapping[str, Any] | None = None) -> str:
    """The report's summary block, in one sentence per grain (ST5.2).

    Both denominators are named, because "13 statements" and "10 trades" are
    two answers to two different questions and a reader given one of them will
    read it as the other.
    """
    rows = list(rows)
    summary = dict(summary) if summary is not None else trade_level_summary(rows)
    families = summary.get("n_statements_by_family") or {}
    return (
        f"{families.get(FAMILY_ENDORSE, 0)} endorsement(s) and "
        f"{families.get(FAMILY_REJECT, 0)} refusal(s), never pooled. "
        f"{len(rows)} statement(s) inside a {TRADE_WINDOW_NOTE} window; "
        f"{summary['n_statements_matched']} matched a trade over "
        f"{summary['n_trades_matched']} distinct trade(s) "
        f"({summary['duplicate_statement_rows']} extra statement row(s) about a "
        f"trade already counted); P&L is summed once per trade. "
        f"{summary['planned_risk_note']}."
    )


def write_rows(rows: list[dict[str, Any]], path: Path | None = None) -> bool:
    """Publish the report atomically. Returns whether it was written."""
    import os

    target = Path(path or REPORT_FILE)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in COLUMNS})
        os.replace(tmp, target)
        return True
    except OSError as exc:
        _log.debug("Preference/trade report could not be written: %s", exc)
        return False


def run_preference_trade_outcomes(
    *,
    now: datetime | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    report_path: Path | None = None,
    trades: list[Mapping[str, Any]] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """The nightly slot. Deterministic, read-only, and no model is called."""
    moment = now or datetime.now()
    until = moment.date()
    since = until - timedelta(days=max(1, int(window_days)))

    journal_available = True
    journal_note = ""
    if trades is None:
        try:
            import journal_store

            trades = list(journal_store.JournalStore().list_trades())
        except Exception as exc:  # noqa: BLE001
            # WS-5B: the trader still SAID it. Publishing nothing threw away the
            # whole left-hand half of the record because the right-hand half was
            # unreadable - and it left the report's last good copy describing a
            # different night. The statements are written, `match_basis` stays
            # empty and every row says `journal_unavailable` out loud.
            trades = []
            journal_available = False
            journal_note = f"journal unavailable: {exc}"

    statements = collect_statements(since=since, until=until)
    if not statements:
        return {
            "status": "skipped",
            "reason": (
                f"no statements recorded between {since} and {until} - an absent "
                "record, not a window without opinions"
                + (f". {journal_note}" if journal_note else "")
            ),
            "rows": 0,
        }

    rows = build_rows(
        statements,
        trades,
        grades=load_paper_grades(),
        now=moment,
        journal_available=journal_available,
    )
    written = write_rows(rows, report_path)
    taken = sum(1 for row in rows if row["traded"] == "yes")
    # ST5.2 / live gate #79: BOTH grains travel out of the slot, so the ledger
    # line and the Weekend Prep note can print `n_trades` beside `n_statements`
    # instead of leaving a reader to assume they are the same number.
    summary = trade_level_summary(rows)
    return {
        "status": "ok" if (written and journal_available) else "degraded",
        "rows": len(rows),
        "taken": taken,
        "not_taken": len(rows) - taken,
        "n_statements_by_family": summary["n_statements_by_family"],
        "n_statements_matched": summary["n_statements_matched"],
        "n_trades_matched": summary["n_trades_matched"],
        "duplicate_statement_rows": summary["duplicate_statement_rows"],
        "net_pnl_by_trade": summary["net_pnl"],
        "planned_risk_recorded": summary["planned_risk_recorded"],
        "window": TRADE_WINDOW_NOTE,
        "reason": (
            f"{len(rows)} statement(s) between {since} and {until} matched inside a "
            f"{TRADE_WINDOW_NOTE} window; {taken} were traded, "
            f"{len(rows) - taken} were not. " + summary_note(rows, summary)
            + (f" {journal_note}." if journal_note else "")
            + ("" if written else " The report could not be written.")
        ),
    }


__all__ = [
    "COLUMNS",
    "DEFAULT_WINDOW_DAYS",
    "FAMILY_ENDORSE",
    "FAMILY_REJECT",
    "MATCH_STATES",
    "MATCH_STATE_JOURNAL_UNAVAILABLE",
    "MATCH_STATE_MATCHED",
    "MATCH_STATE_MATCHING_UNAVAILABLE",
    "MATCH_STATE_NO_MATCH_AFTER_WINDOW",
    "MATCH_STATE_WINDOW_OPEN",
    "REJECT_CHANNELS",
    "REPORT_FILE",
    "SCHEMA",
    "TRADE_WINDOW_DAYS",
    "TRADE_WINDOW_NOTE",
    "TRADE_WINDOW_SESSIONS",
    "build_rows",
    "collect_statements",
    "load_paper_grades",
    "match_state_for",
    "match_trade",
    "run_preference_trade_outcomes",
    "statement_window_end",
    "summary_note",
    "trade_level_summary",
    "verdict_family_for",
    "write_rows",
]
