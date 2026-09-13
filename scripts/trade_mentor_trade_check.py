"""The 10:00 question: what is missing from yesterday's trades - WS-TM item 4.

The Trade Mentor's 10:00 Pacific card has two sections. The first is an ordinary
market read. The second is this: the PREVIOUS exchange session's trades, and for
each one only the material fields it is actually missing.

**Only what is missing.** "All the support data" is not an ever-growing
compulsory questionnaire. A trade that already carries its thesis, its setup
claim and its stop is asked ONE question, not four - otherwise the morning task
becomes something the trader stops doing, and a task nobody does records nothing.

**Four answer states, kept distinct.**

* `not supplied` - it was never written down, and here is what it was;
* `no fixed target` - there WAS no target, which is a complete answer and must
  stop being asked;
* `not remembered` - it existed, and a day later it is gone;
* `not applicable` - the question does not apply to this trade (a scale-out with
  no single stop, an option assignment).

Collapsing any two of them destroys the only thing this task exists to record.
"I had no plan" and "I had a plan I cannot recall" are different facts about a
trader, and a reader who cannot tell them apart cannot say anything useful about
either.

**"No stop" is never a zero.** A `planned_stop` of `0.0` reads downstream as a
stop AT zero and an infinite R. So this module writes an ANNOTATION row and
never touches `planned_stop`, `planned_entry` or `planned_risk`: the trader owns
`trade_annotations` (I7), and a value remembered the next morning is not the
documented pre-entry plan those columns mean.

**Recalled is LABELLED.** Every row carries `recalled_after_session = True` and
the actual write time. Remembered risk must never be presented as a documented
pre-entry plan - that is the same rule `written_after_the_session` enforces on
the Market Journal, for the same reason.

**Capped, and the remainder counted.** Three incomplete trades at a time
(`TRADE_CAP_DEFAULT`); what is left over is a NUMBER the Journal's completeness
view shows, never a fourth question and never silence.

**No coverage, no questionnaire.** If the broker statement for that session has
not landed, an empty list of trades is a lie about the session. It says
`journal not ready` and asks nothing.

Storage is `journal_store`'s existing append-only `opportunity_events` table
under the `RECALLED` event type - an annotation kind, not a schema migration.
Nothing here reaches a detector, a score, a gate, an alert, Focus or the review
queue (plan.md sec 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping

#: The four material fields, in the order they are asked. `missing` is built in
#: this order so a trade with nothing recorded reads the same way every morning.
MATERIAL_FIELDS = ("thesis", "stop", "target", "setup")

#: How each field is answered today, when it is answered at all. `target` has no
#: column anywhere on the journal schema, which is WHY the four states exist:
#: the answer to "what was your target?" is evidence in itself.
_FIELD_SOURCES = {
    "thesis": "notes",
    "stop": "planned_stop",
    "setup": "setup_tags",
    "target": "",
}

ANSWER_NOT_SUPPLIED = "not_supplied"
ANSWER_NO_FIXED_TARGET = "no_fixed_target"
ANSWER_NOT_REMEMBERED = "not_remembered"
ANSWER_NOT_APPLICABLE = "not_applicable"
ANSWER_STATES = (
    ANSWER_NOT_SUPPLIED,
    ANSWER_NO_FIXED_TARGET,
    ANSWER_NOT_REMEMBERED,
    ANSWER_NOT_APPLICABLE,
)

#: Five minutes of questions, measured in trades. Configurable per call.
TRADE_CAP_DEFAULT = 3

#: The append-only event type the answers are stored as.
EVENT_RECALLED = "RECALLED"

#: The status values that mean the trade touched the session.
_SESSION_STATUSES = ("CLOSED", "OPEN", "PARTIALLY_CLOSED")

REASON_NOT_READY = "journal not ready"


@dataclass(frozen=True)
class TradeQuestion:
    """One trade, and only the material fields it is missing."""

    trade_id: str
    symbol: str
    direction: str
    missing: tuple[str, ...]


@dataclass(frozen=True)
class TradeCheckTask:
    """The 10:00 card's second section.

    `trades` is capped; `remaining` and `incomplete_total` are the whole truth
    behind the cap, so the Journal's completeness view can say "N trades still
    missing fields" rather than leaving the rest unmentioned.
    """

    reviewed_session: str
    journal_ready: bool = False
    trades: tuple[TradeQuestion, ...] = ()
    remaining: int = 0
    incomplete_total: int = 0
    reason: str = ""
    #: Every incomplete trade id, capped or not - the completeness view's list.
    incomplete_trade_ids: tuple[str, ...] = field(default=())


def previous_exchange_session(session: date) -> str:
    """The session before `session`, walked on the exchange calendar.

    Not "yesterday": Monday asks about Friday, and the Monday after
    Thanksgiving asks about the Friday half day.
    """
    if isinstance(session, datetime):
        session = session.date()
    try:
        from market_calendar import previous_session

        return previous_session(session).isoformat()
    except Exception:  # noqa: BLE001 - a coarse answer beats none
        from datetime import timedelta

        return (session - timedelta(days=1)).isoformat()


def _journal_ready(store: Any, day: str) -> bool:
    """Did a broker statement for `day` actually land?

    Any COVERED row for the day is enough. A day nobody covered is a day whose
    trade list cannot be trusted to be complete, and an empty questionnaire
    drawn from an incomplete list is worse than no questionnaire.
    """
    try:
        import journal_coverage

        rows = journal_coverage.coverage_rows(store, start=day, end=day)
    except Exception:  # noqa: BLE001 - an unreadable ledger is not readiness
        logging.debug("Journal coverage unreadable.", exc_info=True)
        return False
    return any(str(row.get("status") or "").upper() == journal_coverage.COVERED for row in rows)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def missing_fields(trade: Mapping[str, Any], answered: set[str]) -> tuple[str, ...]:
    """Which material fields this trade still cannot answer.

    A field counts as answered either because the trade RECORDS it or because
    the trader already answered it in a recalled row - including "there was no
    target", which is a complete answer and must stop being asked.
    """
    result: list[str] = []
    for name in MATERIAL_FIELDS:
        if name in answered:
            continue
        source = _FIELD_SOURCES.get(name) or ""
        if source and _has_value(trade.get(source)):
            continue
        result.append(name)
    return tuple(result)


def recalled_fields(store: Any, trade_id: str) -> list[dict[str, Any]]:
    """Every next-morning answer stored for one trade, oldest first."""
    try:
        rows = store.list_opportunity_events(
            trade_id=str(trade_id), event_type=EVENT_RECALLED, limit=10000
        )
    except Exception:  # noqa: BLE001 - a missing store is no answers
        logging.debug("Recalled fields unreadable.", exc_info=True)
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") or {}
        result.append(
            {
                "trade_id": str(row.get("trade_id") or ""),
                "field": str(payload.get("field") or ""),
                "state": str(payload.get("state") or ""),
                "text": str(payload.get("text") or ""),
                # None, never 0.0. "I had no stop" is not a stop at zero.
                "value": payload.get("value"),
                "recalled_after_session": bool(payload.get("recalled_after_session")),
                "recorded_at": str(row.get("occurred_at") or ""),
                "trade_date": str(payload.get("trade_date") or ""),
            }
        )
    return result


def answered_fields(store: Any, trade_id: str) -> set[str]:
    """The material fields this trade has already been asked about and answered."""
    return {
        row["field"]
        for row in recalled_fields(store, trade_id)
        if row["field"] in MATERIAL_FIELDS and row["state"] in ANSWER_STATES
    }


def build_task(store: Any, session: date, *, cap: int = TRADE_CAP_DEFAULT) -> TradeCheckTask:
    """The 10:00 question for the session before `session`."""
    reviewed = previous_exchange_session(session)
    if not _journal_ready(store, reviewed):
        return TradeCheckTask(
            reviewed_session=reviewed, journal_ready=False, reason=REASON_NOT_READY
        )

    try:
        trades = store.list_trades(trade_date=reviewed)
    except Exception:  # noqa: BLE001
        logging.debug("Trade list unreadable.", exc_info=True)
        return TradeCheckTask(
            reviewed_session=reviewed, journal_ready=False, reason=REASON_NOT_READY
        )

    questions: list[TradeQuestion] = []
    for trade in trades:
        status = str(trade.get("status") or "").upper()
        if status and status not in _SESSION_STATUSES:
            continue
        trade_id = str(trade.get("trade_id") or "")
        if not trade_id:
            continue
        gaps = missing_fields(trade, answered_fields(store, trade_id))
        if not gaps:
            continue
        questions.append(
            TradeQuestion(
                trade_id=trade_id,
                symbol=str(trade.get("symbol") or ""),
                direction=str(trade.get("direction") or ""),
                missing=gaps,
            )
        )

    limit = max(0, int(cap))
    asked = tuple(questions[:limit])
    return TradeCheckTask(
        reviewed_session=reviewed,
        journal_ready=True,
        trades=asked,
        remaining=max(0, len(questions) - len(asked)),
        incomplete_total=len(questions),
        reason="",
        incomplete_trade_ids=tuple(question.trade_id for question in questions),
    )


def save_answers(
    store: Any,
    trade_id: str,
    answers: Mapping[str, Mapping[str, Any]],
    *,
    now: datetime | None = None,
    trade_date: str = "",
) -> list[dict[str, Any]]:
    """Store one morning's recalled answers for one trade.

    ANNOTATION rows only. `planned_stop` / `planned_entry` / `planned_risk` are
    never written from here: those columns mean "the plan the trader typed
    before the trade", and a number remembered the next morning is a different
    claim that must not be able to masquerade as one. A state this module does
    not recognise is refused rather than stored - four states that mean four
    things stop meaning anything the moment a fifth can be invented by a caller.
    """
    moment = now or datetime.now().astimezone()
    if not trade_date:
        try:
            trade = store.get_trade(str(trade_id)) or {}
            trade_date = str(trade.get("trade_date") or "")
        except Exception:  # noqa: BLE001
            trade_date = ""
    written: list[dict[str, Any]] = []
    for name, answer in (answers or {}).items():
        if name not in MATERIAL_FIELDS:
            raise ValueError(f"{name!r} is not a material field")
        state = str((answer or {}).get("state") or "")
        if state not in ANSWER_STATES:
            raise ValueError(f"{state!r} is not one of the four answer states")
        value = (answer or {}).get("value")
        payload = {
            "field": name,
            "state": state,
            "text": str((answer or {}).get("text") or ""),
            # Never coerced to a float: an absent number stays absent.
            "value": value if value is not None else None,
            "recalled_after_session": True,
            "trade_date": str(trade_date or ""),
        }
        row = store.record_opportunity_event(
            opportunity_id=f"trade:{trade_id}",
            event_type=EVENT_RECALLED,
            trade_id=str(trade_id),
            occurred_at=moment,
            reason=f"{name}:{state}",
            payload=payload,
            source="trade_mentor",
        )
        written.append(row)
    return written
