"""Where a trade came from, and when its label was made - TJ-9 items 4 and 5.

Two questions, one rule, no I/O. Both answers are REPORTED and nothing here
reaches a detector, a score, a gate, an alert, a watchlist, Focus, the review
queue or ``review_policy.json`` (plan.md sec 5).

**Planned vs unplanned** (item 5, decision 0021 answer 25). A trade is
``planned`` when the trader said something about that name and that side BEFORE
its first fill - a like, a claimed pick, a Focus add or an armed alert. Anything
said afterwards is a different fact: liking a trade at 11:00 does not make the
07:31 entry a plan. Four lanes are passed in already loaded, each keyed by the
stamp ITS OWN store writes:

=============  ===================  =====================================
lane           stamp key            store
=============  ===================  =====================================
``decisions``  ``created_at``       ``ui/annotations/store.py``
``claims``     ``claim_at_utc``     ``claimed_picks``
``focus_adds`` ``joined_at``        ``focus_picks``
``armed``      ``armed_at``         ``armed_alert_expiry``
=============  ===================  =====================================

A reader that understands only one of them would silently answer ``unplanned``
for three quarters of the evidence, so every key is accepted on every lane.

**Uncertainty is never a verdict.** A trade whose opening stamp cannot be read,
and a trade whose opening stamp is MIDNIGHT, are ``unmeasured`` - never
``unplanned``. A broker file is authoritative for money and blind to time
(``journal_trade_shape.is_date_only``): it writes every fill at midnight
market-local, and midnight is not a time a fill happens at, so "before the first
fill" has no meaning for that trade even when a claim sits in front of it.

**A label knows when it was made** (item 4). A setup confirmed the next morning
knows how the trade ended; one claimed before the entry did not. Three values,
never pooled without saying so:

* ``claimed_before_entry`` - the confirmed setup IS the setup of a claim or like
  the trader stamped before the first fill;
* ``same_session`` - confirmed on the session the trade was opened (TJ-14 item 4
  makes this reachable from a card; the value is defined and tested now);
* ``recalled_after`` - everything else, which is the 09:00 card's own honest
  case.

**Aware stamps are compared with ``astimezone``, never by stripping** (plan.md
sec 5). The cases this module exists for are exactly the ones where the naive
strings compare the wrong way round: a like written ``14:15+00:00`` is SIXTEEN
MINUTES BEFORE a fill written ``07:31-07:00``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

#: The exchange's own zone. A naive stamp is ATTACHED to it rather than
#: stripped from the aware side (CLAUDE.md, the adoption-gate rule); it is also
#: the zone `journal_trade_shape` resolves a fill in, so "what session was this"
#: has one answer across both modules.
MARKET_TZ = ZoneInfo("America/New_York")

#: Item 5's three answers.
PLANNED = "planned"
UNPLANNED = "unplanned"
UNMEASURED = "unmeasured"
PLANNED_STATES = (PLANNED, UNPLANNED, UNMEASURED)

#: Item 4's three answers.
CLAIMED_BEFORE_ENTRY = "claimed_before_entry"
SAME_SESSION = "same_session"
RECALLED_AFTER = "recalled_after"
LABEL_PROVENANCES = (CLAIMED_BEFORE_ENTRY, SAME_SESSION, RECALLED_AFTER)

#: Every stamp key the four lanes write, newest-store-first. One tuple rather
#: than a per-lane map: a lane row that carries two of them is still one moment,
#: and a caller that hands `planned_state` the lanes in a different order is
#: answered correctly rather than silently.
_STAMP_KEYS = (
    "claim_at_utc",
    "created_at",
    "joined_at",
    "armed_at",
    "claimed_at",
    "occurred_at",
)

#: Where a lane row names the setup it claimed.
_SETUP_KEYS = ("claimed_setup_id", "setup_id", "setup")

#: Where a trade names the moment of its first fill, in order of authority.
_FILL_KEYS = ("opened_at", "first_fill_at", "entry_at")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _side(value: Any) -> str:
    """``LONG`` / ``SHORT`` / ``""``. The same normalisation the report uses."""
    text = _text(value).upper()
    if text.startswith("SHORT") or text in {"SELL", "S"}:
        return "SHORT"
    if text.startswith("LONG") or text in {"BUY", "B"}:
        return "LONG"
    return ""


def _symbol(value: Any) -> str:
    return _text(value).upper()


def _setup(value: Any) -> str:
    return _text(value).lower()


def _moment(value: Any) -> datetime | None:
    """An AWARE datetime, or ``None`` when the value is not a moment.

    A naive stamp is given the market's zone rather than the machine's: an
    agent's laptop offset is not evidence about a New York fill. Nothing is
    ever stripped from the aware side.
    """
    if isinstance(value, datetime):
        moment = value
    else:
        text = _text(value)
        if len(text) < 10:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=MARKET_TZ)
    return moment


def _stamp_of(row: Mapping[str, Any]) -> datetime | None:
    for key in _STAMP_KEYS:
        if key in row:
            moment = _moment(row.get(key))
            if moment is not None:
                return moment
    return None


def _is_midnight(moment: datetime) -> bool:
    """A stamp that is midnight, in its OWN offset or market-local.

    ``journal_trade_shape.is_date_only`` asks the market-local question, which
    is the one the statement importer writes (it stores every date-only fill at
    midnight ``America/New_York``). A row that reached the journal already
    carrying a different offset - a manual entry, a fixture, an importer that
    normalised before storing - states the same "no clock time" fact in its own
    zone, so both are read as date-only. Both directions are the safe one here:
    the answer is ``unmeasured``, never a guess.

    **This is deliberately WIDER than ``journal_trade_shape.is_date_only``**,
    and the live journal is why: the DRAM assignment row is stored
    ``2026-07-16T00:00:00-07:00``, which is 03:00 in New York and which that
    function therefore reads as a real fill at three in the morning. Midnight
    in the stamp's own zone is the same statement - "the time is not known" -
    and a plan the desk invented for a time that never happened would be worse
    than saying ``unmeasured``.
    """
    if (moment.hour, moment.minute, moment.second, moment.microsecond) == (0, 0, 0, 0):
        return True
    local = moment.astimezone(MARKET_TZ)
    return (local.hour, local.minute, local.second, local.microsecond) == (0, 0, 0, 0)


def first_fill_at(trade: Mapping[str, Any]) -> datetime | None:
    """The trade's first fill as an aware moment, or ``None``.

    ``None`` means "there is no time here to compare against": an unreadable
    stamp, or a date-only broker stamp. Both are uncertainty and both end in
    ``unmeasured``.
    """
    if not isinstance(trade, Mapping):
        return None
    for key in _FILL_KEYS:
        moment = _moment(trade.get(key))
        if moment is None:
            continue
        return None if _is_midnight(moment) else moment
    return None


def trade_session(trade: Mapping[str, Any]) -> date | None:
    """The exchange session of the trade's FIRST FILL, market-local.

    Deliberately NOT ``trade_date``. ``journal_store.rebuild_trades`` writes
    ``trade_date = closed_at or opened_at``, so on a position held across
    sessions that column names the day it was CLOSED. A live example: SMPL
    opened 2026-08-27 and closed 2026-09-18, and 116 of the journal's 216
    trades have an opened date that differs from ``trade_date``. Reading it
    here would have called a label written on the closing day ``same_session``
    - the one thing `label_provenance` exists to make impossible - and a label
    written on the opening day ``recalled_after``.

    Falls back to ``trade_date`` only when no fill stamp can be read at all, so
    a trade with nothing but a date still answers something rather than
    nothing. A date-only OPEN stamp is still a real session even though it is
    not a real time; refusing to name a TIME is :func:`first_fill_at`'s
    separate job.
    """
    if not isinstance(trade, Mapping):
        return None
    for key in _FILL_KEYS:
        moment = _moment(trade.get(key))
        if moment is not None:
            return moment.astimezone(MARKET_TZ).date()
    text = _text(trade.get("trade_date"))
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


def _rows_before(
    trade: Mapping[str, Any],
    lanes: Iterable[Sequence[Mapping[str, Any]] | None],
    fill: datetime,
) -> list[Mapping[str, Any]]:
    """Every lane row about this name and side, stamped before the first fill."""
    symbol = _symbol(trade.get("symbol"))
    side = _side(trade.get("direction") or trade.get("side"))
    found: list[Mapping[str, Any]] = []
    for lane in lanes:
        for row in lane or ():
            if not isinstance(row, Mapping):
                continue
            if _symbol(row.get("symbol")) != symbol:
                continue
            # A LONG like says nothing about a SHORT entry. A row with NO side
            # is not read as agreement either: it cannot say which way round
            # the trader meant it, and a plan the desk invented is worse than
            # no plan at all.
            if _side(row.get("side") or row.get("direction")) != side:
                continue
            stamp = _stamp_of(row)
            if stamp is None or stamp.astimezone(MARKET_TZ) >= fill.astimezone(MARKET_TZ):
                continue
            found.append(row)
    return found


def stamp_of(row: Mapping[str, Any]) -> datetime | None:
    """The moment one lane row was written, as an aware datetime.

    Public because the Trade Mentor's setup suggestion has to pick the LATEST
    claim before a fill, and a second copy of this key list somewhere else
    would be a second opinion about when a claim happened.
    """
    return _stamp_of(row) if isinstance(row, Mapping) else None


def statements_before_entry(
    trade: Mapping[str, Any],
    *lanes: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Every lane row about this name and side, stamped before the first fill.

    Empty when the first fill is unreadable or date-only - the same refusal
    :func:`planned_state` turns into ``unmeasured``.
    """
    fill = first_fill_at(trade)
    if fill is None:
        return []
    return _rows_before(trade, lanes, fill)


def planned_state(
    trade: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]] | None = (),
    claims: Sequence[Mapping[str, Any]] | None = (),
    focus_adds: Sequence[Mapping[str, Any]] | None = (),
    armed: Sequence[Mapping[str, Any]] | None = (),
) -> str:
    """``planned`` / ``unplanned`` / ``unmeasured`` for one trade.

    Pure: the four lanes arrive already loaded, and nothing here opens a store,
    reads a clock or touches Qt.
    """
    fill = first_fill_at(trade)
    if fill is None:
        return UNMEASURED
    if _rows_before(trade, (decisions, claims, focus_adds, armed), fill):
        return PLANNED
    return UNPLANNED


def label_provenance(
    trade: Mapping[str, Any],
    setup: Any,
    claims: Sequence[Mapping[str, Any]] | None = (),
    confirmed_at: datetime | None = None,
) -> str:
    """Which of the three ages this confirmed label has.

    ``claims`` is any lane that NAMES a setup - the annotation log's claimed
    likes, the claimed-pick store - and only a claim naming THIS setup counts:
    a claim about something else, however early, is somebody else's evidence.
    """
    wanted = _setup(setup)
    fill = first_fill_at(trade)
    if wanted and fill is not None:
        for row in _rows_before(trade, (claims,), fill):
            for key in _SETUP_KEYS:
                if key in row and _setup(row.get(key)) == wanted:
                    return CLAIMED_BEFORE_ENTRY

    session = trade_session(trade)
    moment = _moment(confirmed_at)
    if session is not None and moment is not None:
        if moment.astimezone(MARKET_TZ).date() == session:
            return SAME_SESSION
    return RECALLED_AFTER


__all__ = [
    "CLAIMED_BEFORE_ENTRY",
    "LABEL_PROVENANCES",
    "PLANNED",
    "PLANNED_STATES",
    "RECALLED_AFTER",
    "SAME_SESSION",
    "UNMEASURED",
    "UNPLANNED",
    "first_fill_at",
    "label_provenance",
    "planned_state",
    "stamp_of",
    "statements_before_entry",
    "trade_session",
]
