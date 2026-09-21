"""What the Trade Mentor asks for, and the rule that nothing else is asked - TJ-14B.

*"We don't need to run every question every hour but if we need more data make
trade mentor ask me for it. I'm happy to click boxes or give my responses but
then I expect the AI to take it from there."* (trader, 2026-09-19.)

This module is the card's SINGLE DESCRIPTION of every question it may carry. It
is PURE: no store is opened, no clock is read, no Qt object is touched, and
every lane a trigger needs arrives inside the ``state`` mapping the caller
built. A trigger that opened the journal would be a second opinion about it, and
a journal read on whatever thread happened to call :func:`pending`.

**A kind names the reader that consumes its answer.** That is the whole point of
a registry rather than a list of if-statements: :func:`consumer_report` walks
every kind, resolves its ``consumer`` and asks whether that reader actually
touches the key the answer is filed under. A question whose answer lands in a
store nobody opens is the trader's time spent for nothing, which is exactly what
the trader's words rule out.

**A question is ASKED only when its answer has a reader** (lead decision,
2026-09-19; decision 0021 answer 28). A kind is fully described here - trigger,
options, store, answer key and the consumer it WILL have - and carries
``dormant_until`` naming the packet that builds that reader:

============================  ==============  =========================================
kind                          dormant until   why
============================  ==============  =========================================
``trade_origin``              **AWAKE**       TJ-12 shipped the Process line
                                              (``day_report_card.process_line``),
                                              2026-09-20
``open_position_check``       **AWAKE**       TJ-12 shipped the long-hold rows
                                              (``day_report_card.long_hold_lines``),
                                              2026-09-20
``grader_gap``                TJ-10           no deterministic reader emits a gap yet
``quick_like_followup``       TJ-14C          its answer is an ``opportunity_events``
                                              row and ``like_cohort.like_pick_rows``
                                              reads ``claimed_setup_id`` only off
                                              ``trader_annotations.jsonl`` rows - the
                                              join does not exist yet
============================  ==============  =========================================

:func:`pending` never returns a dormant kind on a live card and
:func:`consumer_report` reports it as ``dormant`` with the packet that wakes it,
rather than pretending a reader exists. No shim reader is written to make a
walk pass - a reader nobody calls is the same lie the walk exists to catch.

**The budget of three.** Beyond the forced prediction rows (TJ-14A) and the
forced ``trade_label`` section (TJ-9), a card carries at most three questions,
by priority. The rest are COUNTED on the card and carried to the next one: never
dropped, never a fourth. ``Stop asking this`` retires one SUBJECT, never the
kind, and the service is its only writer. AWAY asks nothing at all.

**The day's pulls have one owner** (:func:`pre_card_pull`). The desk asks the
import service - which owns its own ``QThread`` and is the single caller of the
Questrade refresh chain - for at most :data:`PULLS_PER_DAY_CAP` pulls a day and
stops entirely after :data:`PULL_FAILURES_PER_DAY_CAP` failures. Nothing here
refreshes a token, and a pull that raises, refuses or is already running costs
the card nothing: the card never waits for it, and a fill a late pull lands is
asked about on the NEXT card.

Nothing in this module reaches a detector, a score, a gate, an alert, a
watchlist, Focus, the review queue or ``review_policy.json`` (plan.md sec 5).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
import textwrap
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

#: The fifth option beside TJ-9's four `ANSWER_STATES`. It retires the question
#: for ONE subject - never the kind - and the service is its only writer.
STOP_ASKING = "stop_asking_this"

#: At most three Questrade pulls a day from the Mentor's own seam. A normal
#: session carries six cards, so a cap of six would be no cap at all; the token
#: chain rotates on every refresh and is worth more than one more attempt at
#: today's fills.
PULLS_PER_DAY_CAP = 3

#: Two failures end the day's pulling even with an attempt left. A refresh that
#: failed twice is a broken chain, and the repair is the trader's (CLAUDE.md,
#: "the Questrade refresh chain has ONE owner").
PULL_FAILURES_PER_DAY_CAP = 2

#: How many days a pre-card pull asks the importer for. Two: today, for a fill
#: made this morning, and yesterday, because a late-posting fill lands on the
#: day it happened. The importer is idempotent per day.
PRE_CARD_PULL_DAYS = 2

#: The append-only event type a clicked answer is stored as. An annotation kind
#: on `journal_store`'s existing `opportunity_events` table - never a schema
#: migration, and never `trade_annotations`, which the trader owns (I7).
EVENT_MENTOR_ANSWER = "NOTE"

#: `writes` values. One string per store, so a reader of the registry can see
#: at a glance where an answer lands.
WRITES_OPPORTUNITY_EVENTS = "journal_store.opportunity_events"
WRITES_MARKET_JOURNAL = "market_journal.entries"
WRITES_TRADE_CHECK = "trade_mentor_trade_check.save_answers"
WRITES_MENTOR_CARD = "market_journal.entries:mentor.prediction"
#: TJ-9E. The night's reading of an exit note is `provisional` until the trader
#: presses Confirm or Correct, and those two buttons are the ONE writer of the
#: row they produce. Named here so the registry says where the answer lands;
#: :func:`record_answer` refuses this kind for the same reason it refuses a
#: prediction - a second writer would be a second opinion about what the trader
#: clicked.
WRITES_EXIT_FIELDS = "trade_mentor_trade_check.confirm_exit_fields"

#: Cadences. `once` is forever, `weekly` is once per EXCHANGE week (the ISO week
#: of the session), `daily` is once per session, `per_card` is every card.
CADENCE_ONCE = "once"
CADENCE_WEEKLY = "weekly"
CADENCE_DAILY = "daily"
CADENCE_PER_CARD = "per_card"

#: An OPEN position is asked about only once it is PAST this many exchange
#: sessions. Exactly five is not past five.
LONG_HOLD_SESSIONS = 5

#: How many questions a card may ask beyond the forced rows.
BUDGET = 3

#: The origins the trader is offered. A closed set: a free-text origin cannot
#: be counted. ``a_focus_pick`` was added by TJ-12 review 1 because the desk
#: cannot yet READ the Focus lane (`day_report_card.DESK_ORIGIN_LANES_READ`,
#: filled by TJ-12F), so a pick the trader took off their own Focus list is a
#: real answer the desk would otherwise have no way of hearing.
ORIGIN_OPTIONS = (
    "planned_off_the_desk",
    "a_focus_pick",
    "an_alert",
    "impulse",
    "other",
)

#: What the question SAYS about its own blindness. A question that asked "where
#: did this come from?" without saying the desk cannot see two of the four
#: places it could have come from would be blaming the trader for the desk's
#: unread stores (reviewer, 2026-09-20).
ORIGIN_PROMPT_CAVEAT = (
    "The desk saw no claim or like before this trade - it cannot read Focus "
    "adds or armed alerts yet."
)

#: The three answers to "is the thesis still intact?".
OPEN_POSITION_OPTIONS = ("thesis_intact", "weakening", "exit_planned")

#: `Followed the plan:` - the session's last card only.
KIND_DAY_CLOSE = "day_close"
DAY_CLOSE_OPTIONS = ("yes", "partly", "no")


def _answer_states() -> tuple[str, ...]:
    """TJ-9's four states, read from the module that owns them."""
    try:
        import trade_mentor_trade_check as check

        return tuple(check.ANSWER_STATES)
    except Exception:  # noqa: BLE001 - a missing vocabulary is a shorter list
        logging.debug("Answer states unreadable.", exc_info=True)
        return ()


def _with_answer_states(*options: str) -> tuple[str, ...]:
    """A kind's own clicks, plus the four answer states, plus `Stop asking this`.

    Every budgeted question offers all of them (plan.md TJ-14 item 3): a trader
    who cannot say "not remembered" has to either invent an answer or leave the
    card, and both destroy the record the question exists to make.
    """
    names: list[str] = []
    for option in tuple(options) + _answer_states() + (STOP_ASKING,):
        text = str(option or "").strip()
        if text and text not in names:
            names.append(text)
    return tuple(names)


@dataclass(frozen=True)
class Subject:
    """One thing to ask about: which kind, which subject, and its clicks.

    `options` is per SUBJECT because three kinds vary by subject - the claim
    vocabulary for a like, the emitting reader's own closed list for a grader
    gap, and the overnight model's options for the AI question.
    """

    kind: str
    subject_id: str
    options: tuple[str, ...] = ()
    prompt: str = ""
    #: Everything the writer needs that is not on the card: a trade id, a like's
    #: symbol and side, the consumer a grader gap named for itself.
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionKind:
    """One kind of question, described once.

    `consumer` is the dotted name of the reader that USES the answer and
    `answer_key` the key that reader reads. `dormant_until` names the packet
    that builds the consumer; while it is set the kind is described but never
    asked.
    """

    kind: str
    trigger: Callable[[Mapping[str, Any]], list[Subject]]
    options: tuple[str, ...]
    writes: str
    consumer: str
    answer_key: str
    cadence: str
    expiry: str
    priority: int
    budgeted: bool = True
    dormant_until: str = ""
    prompt: str = ""


@dataclass(frozen=True)
class CardQuestions:
    """What one card carries.

    `asked` is at most :data:`BUDGET` budgeted subjects; `forced` is the
    prediction rows and the trade-label section, which sit OUTSIDE the budget;
    `carried` is everything owed that did not fit, which is counted on the card
    and asked on the next one.
    """

    asked: tuple[Subject, ...] = ()
    forced: tuple[Subject, ...] = ()
    carried: tuple[Subject, ...] = ()
    waiting_note: str = ""


# ---------------------------------------------------------------------------
# small pure helpers over the state mapping
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    return str(value or "").strip()


def _rows(state: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = state.get(key) if isinstance(state, Mapping) else None
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _session_date(state: Mapping[str, Any]) -> date | None:
    text = _text(state.get("session") if isinstance(state, Mapping) else "")[:10]
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _slot_of(state: Mapping[str, Any]):
    """The slot `pending` injected, or ``None``.

    A trigger is handed the slot through the state rather than a second
    parameter so it stays a one-argument pure function; a trigger called
    without one answers "nothing", which is the safe direction.
    """
    return state.get("slot") if isinstance(state, Mapping) else None


def _subject_key(kind: str, subject_id: str) -> str:
    return f"{kind}:{subject_id}"


def _answered_record(state: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    answered = state.get("answered") if isinstance(state, Mapping) else None
    if not isinstance(answered, Mapping):
        return None
    record = answered.get(key)
    return record if isinstance(record, Mapping) else ({} if key in answered else None)


def _answered_on(record: Mapping[str, Any]) -> date | None:
    text = _text(record.get("answered_at"))[:10]
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _silenced(kind: QuestionKind, subject: Subject, state: Mapping[str, Any]) -> bool:
    """Has this subject been answered (for its cadence) or retired?"""
    key = _subject_key(subject.kind, subject.subject_id)
    retired = state.get("retired") if isinstance(state, Mapping) else ()
    if isinstance(retired, Iterable) and not isinstance(retired, (str, bytes)):
        if key in {_text(item) for item in retired}:
            return True
    record = _answered_record(state, key)
    if record is None:
        return False
    if kind.cadence == CADENCE_PER_CARD:
        return False
    if kind.cadence == CADENCE_ONCE:
        return True
    answered_on = _answered_on(record)
    session = _session_date(state)
    if answered_on is None or session is None:
        # An answer with no readable date is still an answer. Asking again
        # because a stamp is unreadable is the desk not listening.
        return True
    if kind.cadence == CADENCE_DAILY:
        return answered_on == session
    if kind.cadence == CADENCE_WEEKLY:
        return answered_on.isocalendar()[:2] == session.isocalendar()[:2]
    return True


def _trade_session(row: Mapping[str, Any]) -> date | None:
    try:
        import trade_origin

        return trade_origin.trade_session(row)
    except Exception:  # noqa: BLE001 - an undatable trade is never asked about
        logging.debug("Trade session undecidable.", exc_info=True)
        return None


def _missing_material_fields(row: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        import trade_mentor_trade_check as check

        return check.missing_fields(row, set())
    except Exception:  # noqa: BLE001 - an unreadable trade asks nothing
        logging.debug("Missing fields undecidable.", exc_info=True)
        return ()


def _setup_vocabulary() -> tuple[str, ...]:
    try:
        import trade_mentor_trade_check as check

        return tuple(check.setup_vocabulary())
    except Exception:  # noqa: BLE001
        logging.debug("Setup vocabulary unreadable.", exc_info=True)
        return ()


# ---------------------------------------------------------------------------
# the triggers - each a measured gap, never a clock alone
# ---------------------------------------------------------------------------


def _trigger_prediction(horizon_kind: str) -> Callable[[Mapping[str, Any]], list[Subject]]:
    def trigger(state: Mapping[str, Any]) -> list[Subject]:
        slot = _slot_of(state)
        if slot is None:
            return []
        try:
            from trade_mentor_schedule import KIND_M5_D1
        except Exception:  # noqa: BLE001
            return []
        if horizon_kind == "d1" and _text(getattr(slot, "kind", "")) != KIND_M5_D1:
            return []
        return [
            Subject(
                kind=f"prediction_{horizon_kind}",
                subject_id=_text(getattr(slot, "slot_id", "")),
                options=(),
                prompt="What I expect",
            )
        ]

    return trigger


def _trigger_trade_label(state: Mapping[str, Any]) -> list[Subject]:
    """Every trade on the card that still cannot answer a material field.

    TJ-9 lists the reviewed session's trades; TJ-14B adds today's fills (item
    4), so the label the trader can still make BEFORE the outcome is known is
    actually offered. The section is forced and outside the budget.
    """
    subjects: list[Subject] = []
    seen: set[str] = set()
    for row in _rows(state, "trades"):
        trade_id = _text(row.get("trade_id"))
        if not trade_id or trade_id in seen:
            continue
        if not _missing_material_fields(row):
            continue
        seen.add(trade_id)
        subjects.append(
            Subject(
                kind="trade_label",
                subject_id=trade_id,
                options=_answer_states(),
                prompt=f"{_text(row.get('symbol'))} {_text(row.get('direction'))}".strip(),
                detail={"trade_id": trade_id, "trade_date": _text(row.get("trade_date"))},
            )
        )
    return subjects


#: What the trader may do with a waiting exit draft. Two verbs, and there is no
#: third: the machine's reading is theirs or it is rewritten.
EXIT_DRAFT_OPTIONS = ("confirm", "correct")


def _exit_key(trade_id: Any, session: Any) -> str:
    """`trade_mentor_trade_check.exit_key`, imported at CALL time.

    This module stays pure and import-light; the identity of an exit belongs to
    the module that writes one.
    """
    try:
        import trade_mentor_trade_check as check

        return check.exit_key(trade_id, session)
    except Exception:  # noqa: BLE001 - a lane row that names its own key wins
        logging.debug("The exit key helper is unreadable.", exc_info=True)
        return f"{_text(trade_id)}@{_text(session)[:10]}"


def _trigger_exit_draft_review(state: Mapping[str, Any]) -> list[Subject]:
    """One row per exit draft the night left waiting for the trader.

    Offers NOTHING when no draft waits, which is the honest first state - and
    for a long time the usual one, because a draft only exists where the trader
    wrote an exit note the night before. The rows come from the HOST, which has
    already read the drafts file on a worker; this never opens it.
    """
    subjects: list[Subject] = []
    seen: set[str] = set()
    for row in _rows(state, "exit_drafts"):
        trade_id = _text(row.get("trade_id"))
        symbol = _text(row.get("symbol"))
        session = _text(row.get("exit_session"))
        # The identity of a reading is (trade, EXIT SESSION), never the trade:
        # a trade can have closed in two sessions and been read twice, and
        # de-duplicating by trade id dropped the second one silently - not
        # offered, not carried, not said (review 2 blocker 1). The key comes
        # from the ONE helper that builds it.
        key = _text(row.get("key")) or _exit_key(trade_id, session)
        if not trade_id or key in seen:
            continue
        seen.add(key)
        subjects.append(
            Subject(
                kind="exit_draft_review",
                subject_id=key,
                options=_with_answer_states(*EXIT_DRAFT_OPTIONS),
                prompt=_text(row.get("prompt"))
                or f"{symbol} - you exited on {session}. Is that what happened?".strip(),
                # Everything the card needs to DRAW the row, carried on the
                # subject: the trader's own words and the night's reading of
                # them. The registry decides WHICH drafts are offered and how
                # many; it never renders one and never writes one.
                detail={
                    "key": key,
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "exit_session": session,
                    "note_id": _text(row.get("note_id")),
                    "raw_text": _text(row.get("raw_text")),
                    "fields": dict(row.get("fields") or {}),
                },
            )
        )
    return subjects


def _trigger_trade_origin(state: Mapping[str, Any]) -> list[Subject]:
    """A trade with nothing said about it before its first fill.

    The verdict comes from `trade_origin.planned_state` over the four lanes and
    is never re-implemented here; `unmeasured` is not `unplanned`, so a
    date-only broker fill is never asked about.
    """
    try:
        import trade_origin
    except Exception:  # noqa: BLE001
        return []
    decisions = _rows(state, "decisions")
    claims = _rows(state, "claims")
    focus_adds = _rows(state, "focus_adds")
    armed = _rows(state, "armed")
    subjects: list[Subject] = []
    seen: set[str] = set()
    for row in _rows(state, "trades"):
        trade_id = _text(row.get("trade_id"))
        if not trade_id or trade_id in seen:
            continue
        try:
            verdict = trade_origin.planned_state(row, decisions, claims, focus_adds, armed)
        except Exception:  # noqa: BLE001 - an undecidable trade is never asked
            logging.debug("Planned state undecidable.", exc_info=True)
            continue
        if verdict != trade_origin.UNPLANNED:
            continue
        seen.add(trade_id)
        subjects.append(
            Subject(
                kind="trade_origin",
                subject_id=trade_id,
                options=_with_answer_states(*ORIGIN_OPTIONS),
                prompt=(
                    f"{ORIGIN_PROMPT_CAVEAT} Where did "
                    f"{_text(row.get('symbol')) or trade_id} come from?"
                ),
                detail={"trade_id": trade_id, "symbol": _text(row.get("symbol"))},
            )
        )
    return subjects


def _trigger_open_position_check(state: Mapping[str, Any]) -> list[Subject]:
    """An OPEN position held PAST five exchange sessions, once a week.

    The boundary is walked on the exchange calendar, never on calendar days: a
    long weekend is not a week of holding.
    """
    session = _session_date(state)
    if session is None:
        return []
    try:
        import market_calendar
    except Exception:  # noqa: BLE001
        return []
    subjects: list[Subject] = []
    seen: set[str] = set()
    for row in _rows(state, "open_positions"):
        trade_id = _text(row.get("trade_id"))
        if not trade_id or trade_id in seen:
            continue
        if _text(row.get("status")).upper() == "CLOSED":
            continue
        opened = _trade_session(row)
        if opened is None:
            continue
        try:
            held = int(market_calendar.trading_days_between(opened, session))
        except Exception:  # noqa: BLE001 - an unreadable calendar asks nothing
            logging.debug("Holding length undecidable.", exc_info=True)
            continue
        if held <= LONG_HOLD_SESSIONS:
            continue
        seen.add(trade_id)
        subjects.append(
            Subject(
                kind="open_position_check",
                subject_id=trade_id,
                options=_with_answer_states(*OPEN_POSITION_OPTIONS),
                prompt=(
                    f"{_text(row.get('symbol')) or trade_id} has been open "
                    f"{held} sessions - is the thesis intact?"
                ),
                detail={"trade_id": trade_id, "sessions_held": held},
            )
        )
    return subjects


def _trigger_quick_like_followup(state: Mapping[str, Any]) -> list[Subject]:
    """A QUICK like that was then traded, or that really did run.

    A quick like names no setup (P9), so a like cohort by family cannot see it.
    A CLAIMED like already said what it was and is never asked again.
    """
    try:
        from ui.annotations.store import LIKE_MODE_QUICK, like_mode_of
    except Exception:  # noqa: BLE001
        return []
    try:
        import real_miss

        real_miss_v1 = real_miss.REAL_MISS_V1
    except Exception:  # noqa: BLE001 - TJ-11's rule missing leaves the traded leg
        real_miss_v1 = ""
    vocabulary = _setup_vocabulary()
    subjects: list[Subject] = []
    seen: set[str] = set()
    for row in _rows(state, "likes"):
        event_id = _text(row.get("event_id"))
        if not event_id or event_id in seen:
            continue
        if like_mode_of(row) != LIKE_MODE_QUICK:
            continue
        traded = bool(_text(row.get("matched_trade_id")))
        ran = bool(real_miss_v1) and _text(row.get("real_miss")) == real_miss_v1
        if not traded and not ran:
            continue
        seen.add(event_id)
        subjects.append(
            Subject(
                kind="quick_like_followup",
                subject_id=event_id,
                options=_with_answer_states(*vocabulary),
                prompt=f"Which setup was {_text(row.get('symbol'))}?",
                detail={
                    "like_event_id": event_id,
                    "symbol": _text(row.get("symbol")),
                    "side": _text(row.get("side")),
                    "session_date": _text(row.get("session_date")),
                    "matched_trade_id": _text(row.get("matched_trade_id")),
                },
            )
        )
    return subjects


def _trigger_day_close(state: Mapping[str, Any]) -> list[Subject]:
    """The session's LAST card only - an early close included.

    An 11:00 card asking "did you follow the plan?" asks it before the plan has
    finished happening.
    """
    slot = _slot_of(state)
    session = _session_date(state)
    if slot is None or session is None:
        return []
    try:
        from trade_mentor_schedule import slots_for_session

        slots = slots_for_session(session)
    except Exception:  # noqa: BLE001
        return []
    if not slots or _text(getattr(slot, "slot_id", "")) != _text(slots[-1].slot_id):
        return []
    return [
        Subject(
            kind="day_close",
            subject_id=session.isoformat(),
            options=_with_answer_states(*DAY_CLOSE_OPTIONS),
            prompt="Followed the plan?",
            detail={"session": session.isoformat()},
        )
    ]


def _trigger_grader_gap(state: Mapping[str, Any]) -> list[Subject]:
    """Whatever a deterministic reader could not measure without the trader.

    The emitting reader names its own question id, its closed options, its
    consumer and the key it will read; the registry only turns that into a
    click.
    """
    subjects: list[Subject] = []
    seen: set[str] = set()
    for row in _rows(state, "grader_gaps"):
        subject_id = _text(row.get("subject_id")) or _text(row.get("question_id"))
        if not subject_id or subject_id in seen:
            continue
        seen.add(subject_id)
        options = tuple(_text(item) for item in (row.get("options") or ()) if _text(item))
        subjects.append(
            Subject(
                kind="grader_gap",
                subject_id=subject_id,
                options=_with_answer_states(*options),
                prompt=_text(row.get("prompt")) or _text(row.get("question_id")),
                detail={
                    "question_id": _text(row.get("question_id")),
                    "consumer": _text(row.get("consumer")),
                    "answer_key": _text(row.get("answer_key")),
                },
            )
        )
    return subjects


def _trigger_ai_question(state: Mapping[str, Any]) -> list[Subject]:
    """At most ONE a day: the overnight `mentor_question` with its click options.

    A degraded night asks nothing rather than repeating an old question - which
    is what the card did before TJ-14B, on every card, forever.
    """
    payload = state.get("ai_question") if isinstance(state, Mapping) else None
    if not isinstance(payload, Mapping):
        return []
    question = _text(payload.get("question"))
    if not question:
        return []
    session = _session_date(state)
    options = tuple(_text(item) for item in (payload.get("options") or ()) if _text(item))
    return [
        Subject(
            kind="ai_question",
            subject_id=session.isoformat() if session else question[:64],
            options=_with_answer_states(*options),
            prompt=question,
            detail={
                "question": question,
                "session": session.isoformat() if session else "",
            },
        )
    ]


REGISTRY: tuple[QuestionKind, ...] = (
    QuestionKind(
        kind="grader_gap",
        trigger=_trigger_grader_gap,
        options=_with_answer_states(),
        writes=WRITES_OPPORTUNITY_EVENTS,
        consumer="market_read_grades.congruence_lines",
        answer_key="trader_input",
        cadence=CADENCE_ONCE,
        expiry="until the reader that raised it is answered",
        priority=10,
        dormant_until="TJ-10",
    ),
    QuestionKind(
        kind="trade_origin",
        trigger=_trigger_trade_origin,
        options=_with_answer_states(*ORIGIN_OPTIONS),
        writes=WRITES_OPPORTUNITY_EVENTS,
        consumer="day_report_card.process_line",
        answer_key="trade_origin",
        cadence=CADENCE_ONCE,
        expiry="never - a trade's origin does not change",
        priority=20,
        # AWAKE since TJ-12 (lead, 2026-09-20): the Process line on the Day
        # Review report card reads this answer. The module is
        # `scripts/day_report_card.py`, which is the name the packet gave the
        # file; the registry has to name the module that actually imports, or
        # the consumer walk reports "the module does not import" forever.
        dormant_until="",
    ),
    # TJ-9E (2026-09-21). The night read the trader's exit note and drafted
    # three fields; this is the row that asks them to sign it off. BUDGETED -
    # it costs one of the three only when a draft is actually waiting - and it
    # never greys Save: a draft nobody clicked must not hold the morning
    # hostage. AWAKE, because this packet builds its reader:
    # `day_report_card.exit_note_counts` reads `exit_fields` to say how many of
    # the session's exits the trader has confirmed.
    QuestionKind(
        kind="exit_draft_review",
        trigger=_trigger_exit_draft_review,
        options=_with_answer_states(*EXIT_DRAFT_OPTIONS),
        writes=WRITES_EXIT_FIELDS,
        consumer="day_report_card.exit_note_counts",
        answer_key="exit_fields",
        cadence=CADENCE_ONCE,
        expiry="until the trader confirms or corrects the draft",
        priority=25,
        dormant_until="",
    ),
    QuestionKind(
        kind="quick_like_followup",
        trigger=_trigger_quick_like_followup,
        # The claim vocabulary, read from the two stores that own it - the
        # capture rail's claim registry and the setup documents - never restated
        # here. A vocabulary written twice drifts, and the copy nobody edits
        # becomes a falsehood shipped as data.
        options=_with_answer_states(*_setup_vocabulary()),
        writes=WRITES_OPPORTUNITY_EVENTS,
        consumer="ui.annotations.like_cohort.like_pick_rows",
        answer_key="claimed_setup_id",
        cadence=CADENCE_ONCE,
        expiry="never - the like already happened",
        priority=30,
        # TJ-14B review, blocker 2. The answer is filed as an append-only
        # `opportunity_events` row; `like_pick_rows` reads `claimed_setup_id`
        # off `trader_annotations.jsonl` rows, and the tester's own test forbids
        # appending to that file. The KEY is genuinely read - the STORE is not
        # joined - so the question waits for the packet that joins them. The
        # live log holds 46 quick likes, so this would really have been asked.
        dormant_until="TJ-14C",
    ),
    QuestionKind(
        kind="day_close",
        trigger=_trigger_day_close,
        options=_with_answer_states(*DAY_CLOSE_OPTIONS),
        writes=WRITES_MARKET_JOURNAL,
        consumer="market_story._entry_row",
        answer_key="text",
        cadence=CADENCE_DAILY,
        expiry="the session it is about",
        priority=40,
    ),
    QuestionKind(
        kind="open_position_check",
        trigger=_trigger_open_position_check,
        options=_with_answer_states(*OPEN_POSITION_OPTIONS),
        writes=WRITES_OPPORTUNITY_EVENTS,
        consumer="day_report_card.long_hold_lines",
        answer_key="open_position_state",
        cadence=CADENCE_WEEKLY,
        expiry="one exchange week",
        priority=50,
        # AWAKE since TJ-12 (lead, 2026-09-20): `day_report_card.long_hold_lines`
        # reads this answer and reports an unanswered position as UNANSWERED -
        # never as "the thesis is intact", which is a claim nobody made.
        dormant_until="",
    ),
    QuestionKind(
        kind="ai_question",
        trigger=_trigger_ai_question,
        options=_with_answer_states(),
        writes=WRITES_MARKET_JOURNAL,
        consumer="market_story._entry_row",
        answer_key="text",
        cadence=CADENCE_DAILY,
        expiry="the day the night wrote it for",
        priority=60,
    ),
    # Forced, and outside the budget. They are registered so the card has ONE
    # description of every row it may carry.
    QuestionKind(
        kind="trade_label",
        trigger=_trigger_trade_label,
        options=_answer_states(),
        writes=WRITES_TRADE_CHECK,
        consumer="trade_mentor_trade_check.answered_fields",
        answer_key="state",
        cadence=CADENCE_PER_CARD,
        expiry="until the field is answered",
        priority=5,
        budgeted=False,
    ),
    QuestionKind(
        kind="prediction_m5",
        trigger=_trigger_prediction("m5"),
        options=(),
        writes=WRITES_MENTOR_CARD,
        consumer="market_journal.prediction_of",
        answer_key="prediction",
        cadence=CADENCE_PER_CARD,
        expiry="the hour it was asked in",
        priority=1,
        budgeted=False,
    ),
    QuestionKind(
        kind="prediction_d1",
        trigger=_trigger_prediction("d1"),
        options=(),
        writes=WRITES_MENTOR_CARD,
        consumer="market_journal.prediction_of",
        answer_key="prediction",
        cadence=CADENCE_PER_CARD,
        expiry="the hour it was asked in",
        priority=2,
        budgeted=False,
    ),
)


def kind_named(name: str) -> QuestionKind:
    """One registered kind by name. An unknown name RAISES - never a default."""
    wanted = _text(name)
    for kind in REGISTRY:
        if kind.kind == wanted:
            return kind
    raise KeyError(f"{name!r} is not a registered Mentor question kind")


# ---------------------------------------------------------------------------
# the consumer walk
# ---------------------------------------------------------------------------


def _resolve(dotted: str) -> tuple[Any, bool, str]:
    """Import the longest importable prefix of `dotted` and walk the rest.

    A consumer may be a module function (`market_journal.prediction_of`), a
    method on a class (`journal_analytics.AutoTagger.rows`) or a function in a
    package module - all three are one dotted name to a reader of the registry.
    """
    parts = [part for part in _text(dotted).split(".") if part]
    if len(parts) < 2:
        return None, False, "a consumer is a dotted module.callable name"
    module = None
    index = 0
    for stop in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:stop]))
        except Exception:  # noqa: BLE001 - keep shortening
            continue
        index = stop
        break
    if module is None:
        return None, False, f"{dotted}: the module does not import"
    target: Any = module
    for part in parts[index:]:
        try:
            target = getattr(target, part)
        except AttributeError:
            return None, False, f"{dotted}: {part!r} does not exist on the import"
    return target, True, ""


def _key_constant(node: Any, key: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == key


def _reads_key(target: Any, answer_key: str) -> bool:
    """Does this reader actually TOUCH the key the answer is filed under?

    A static read of the reader's own source, PARSED - not a text search and not
    a call.

    * not a call, because a probe that called the consumer and looked for the
      value in its output would pass for `json.dumps`, which imports, is
      callable, and will never read a Mentor answer;
    * not a text search, because the word also appears in comments, docstrings
      and bare strings. TJ-14B's review planted both foolers: a function whose
      only mention of the key is `# claimed_setup_id` in a comment, and one
      whose body is `return "trade_origin"`. A `grep`-shaped probe passes both,
      and a registry whose check can be satisfied by a comment is not a check.

    The key counts only as a string CONSTANT used in code: a subscript
    (``row["state"]``), an argument (``row.get("state")``), a comparison
    (``name == "state"``) or a keyword value. A docstring and a bare string
    statement are `ast.Expr` wrapping the constant and are never reached,
    because nothing here looks at `ast.Expr`.
    """
    key = _text(answer_key)
    if not key:
        return False
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return False
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if _key_constant(node.slice, key):
                return True
        elif isinstance(node, ast.Call):
            if any(_key_constant(arg, key) for arg in node.args):
                return True
        elif isinstance(node, ast.keyword):
            if _key_constant(node.value, key):
                return True
        elif isinstance(node, ast.Compare):
            if _key_constant(node.left, key) or any(
                _key_constant(item, key) for item in node.comparators
            ):
                return True
    return False


def consumer_report(kinds: Sequence[QuestionKind] | None = None) -> tuple[dict[str, Any], ...]:
    """One row per kind: does its consumer import, and does it read the answer?

    A DORMANT kind is reported as dormant with the packet that wakes it rather
    than pretending a reader exists - the walk requires a live reader for every
    kind that can actually reach a card.
    """
    rows: list[dict[str, Any]] = []
    for kind in tuple(kinds if kinds is not None else REGISTRY):
        target, imports, reason = _resolve(kind.consumer)
        reads = _reads_key(target, kind.answer_key) if imports else False
        if imports and not reads and not reason:
            reason = f"{kind.consumer} never reads {kind.answer_key!r}"
        dormant = bool(kind.dormant_until)
        rows.append(
            {
                "kind": kind.kind,
                "consumer": kind.consumer,
                "answer_key": kind.answer_key,
                "imports": bool(imports),
                "reads": bool(reads),
                # TJ-9E. The two columns above are the MEASUREMENT - does the
                # named module import, and does it really touch the key. These
                # two are the PROMISE: is this kind's registry entry true as
                # written. They differ for exactly one row shape, the DORMANT
                # one, whose entry promises no reader yet and names the packet
                # that will build it. `grader_gap` is that shape today: TJ-10
                # owes it a reader and nothing pretends otherwise, and a shim
                # written to make a walk pass is the lie the walk exists to
                # catch (this module's own rule). A LIVE kind's promise is kept
                # only when its reader resolves AND reads the key, which is the
                # whole check for every kind that can reach a card.
                "resolved": bool(imports) and (bool(reads) or dormant),
                "reads_key": bool(reads) or dormant,
                "reason": reason,
                "dormant": dormant,
                "dormant_until": kind.dormant_until,
                "budgeted": bool(kind.budgeted),
            }
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------


def _subjects_for(kind: QuestionKind, state: Mapping[str, Any]) -> list[Subject]:
    try:
        subjects = kind.trigger(state) or []
    except Exception:  # noqa: BLE001 - one broken trigger never costs the card
        logging.debug("Mentor trigger failed for %s.", kind.kind, exc_info=True)
        return []
    return [subject for subject in subjects if isinstance(subject, Subject)]


def pending(state: Mapping[str, Any], slot: Any) -> CardQuestions:
    """What this card asks, what it forces, and what it is still owed.

    AWAY asks nothing - forced rows included: the trader is not there, and a
    card nobody saw is not a question. Otherwise every live kind is triggered,
    already-answered and retired subjects are dropped, the budgeted remainder is
    ranked by priority and the first :data:`BUDGET` are asked. The rest are
    carried: counted on the card, never dropped, never a fourth.
    """
    payload = dict(state or {})
    payload["slot"] = slot
    if _text(payload.get("auto_mode")).upper() == "AWAY":
        return CardQuestions()

    carried_in = payload.get("carried") or ()
    known: dict[tuple[str, str], Subject] = {}
    for subject in carried_in:
        if isinstance(subject, Subject):
            known[(subject.kind, subject.subject_id)] = subject

    forced: list[Subject] = []
    owed: list[Subject] = []
    for kind in REGISTRY:
        if kind.dormant_until:
            # Described, never asked: its reader is not built yet.
            continue
        for subject in _subjects_for(kind, payload):
            if _silenced(kind, subject, payload):
                known.pop((subject.kind, subject.subject_id), None)
                continue
            known.pop((subject.kind, subject.subject_id), None)
            (forced if not kind.budgeted else owed).append(subject)

    # Anything the previous card carried that this card's triggers no longer
    # produce is still owed - a question is dropped only when it is answered or
    # retired, never because a lane arrived empty.
    for key, subject in known.items():
        try:
            kind = kind_named(subject.kind)
        except KeyError:
            continue
        if kind.dormant_until or not kind.budgeted or _silenced(kind, subject, payload):
            continue
        owed.append(subject)

    ranked = sorted(owed, key=lambda item: (_priority_of(item.kind), item.kind))
    asked = tuple(ranked[:BUDGET])
    carried = tuple(ranked[BUDGET:])
    # TJ-9E: a carried exit READING is named, because it is not a question the
    # trader can answer in a word - it is something the night wrote about what
    # they said, and "1 more waiting" would not tell them there is a reading of
    # their own note they have not seen. `sorted` is stable, so the oldest
    # draft of the lane is the first offered and the newest is the one carried.
    readings = sum(1 for item in carried if item.kind == "exit_draft_review")
    parts = []
    if carried:
        parts.append(f"{len(carried)} more waiting - they come back on the next card.")
    if readings:
        parts.append(
            f"{readings} more exit reading(s) waiting."
            if readings != len(carried)
            else f"That is {readings} exit reading(s)."
        )
    note = " ".join(parts)
    return CardQuestions(asked=asked, forced=tuple(forced), carried=carried, waiting_note=note)


def _priority_of(name: str) -> int:
    try:
        return int(kind_named(name).priority)
    except Exception:  # noqa: BLE001 - an unknown kind sorts last
        return 10_000


# ---------------------------------------------------------------------------
# storing an answer
# ---------------------------------------------------------------------------


def record_answer(
    subject: Subject,
    answer: Mapping[str, Any],
    *,
    store: Any = None,
    now: datetime | None = None,
    journal: Any = None,
) -> dict[str, Any]:
    """File one clicked answer through the owning store's own writer.

    *"then I expect the AI to take it from there"*: one click, one stored row,
    readable under the key the registry declares. Nothing here writes
    `trade_annotations` - the trader owns it - and nothing rewrites the row the
    question was ABOUT: a quick like's follow-up writes a recalled claim LINK
    that NAMES the like and leaves the like byte for byte as it was.
    """
    kind = kind_named(subject.kind)
    if kind.writes == WRITES_MENTOR_CARD:
        # The prediction is filed WITH the read, in one row, by the card's own
        # `submit` (TJ-14A). A second writer here would be a second opinion
        # about what the trader clicked.
        return {"ok": False, "reason": "the card files a prediction with its read"}
    if kind.writes == WRITES_EXIT_FIELDS:
        # TJ-9E, and the same rule: the card's Confirm and Correct buttons are
        # the ONE writer of a confirmed exit row, because a confirm is the
        # trader's own act and `confirm_exit_fields` must have exactly one
        # caller that is not a job.
        return {
            "ok": False,
            "reason": "the card's Confirm or Correct button writes an exit's fields",
        }
    state = _text((answer or {}).get("state"))
    if state == STOP_ASKING:
        # `Stop asking this` is not an answer and is never stored as one. The
        # service retires the subject; see `TradeMentorService.stop_asking`.
        return {"ok": False, "reason": "stop asking this is a retirement, not an answer"}
    moment = now or datetime.now().astimezone()
    detail = dict(subject.detail or {})
    payload: dict[str, Any] = {
        "mentor_question_kind": kind.kind,
        "subject_id": subject.subject_id,
        kind.answer_key: state,
        "answer_text": _text((answer or {}).get("text")),
        "answered_at": moment.isoformat(),
    }
    payload.update({key: value for key, value in detail.items() if key not in payload})

    if kind.writes == WRITES_MARKET_JOURNAL:
        return _record_in_journal(
            kind,
            subject,
            payload,
            journal=journal,
            now=moment,
            mood=_mood_fields(kind, answer, state),
        )
    if kind.writes == WRITES_TRADE_CHECK:
        # TJ-9 owns the material-field rows, their four states and their
        # provenance. This never re-implements that writer.
        import trade_mentor_trade_check as check

        field_name = _text(detail.get("field")) or _text((answer or {}).get("field"))
        if field_name not in check.MATERIAL_FIELDS:
            return {"ok": False, "reason": f"{field_name!r} is not a material field"}
        rows = check.save_answers(
            store,
            _text(detail.get("trade_id")) or subject.subject_id,
            {field_name: dict(answer or {})},
            now=moment,
            trade_date=_text(detail.get("trade_date")),
        )
        return {"ok": bool(rows), "row": rows[0] if rows else {}, "answer_key": kind.answer_key}
    if store is None:
        return {"ok": False, "reason": "no journal store to write to"}
    trade_id = _text(detail.get("trade_id"))
    opportunity_id = (
        f"trade:{trade_id}" if trade_id else f"mentor:{kind.kind}:{subject.subject_id}"
    )
    row = store.record_opportunity_event(
        opportunity_id=opportunity_id,
        event_type=EVENT_MENTOR_ANSWER,
        symbol=_text(detail.get("symbol")).upper(),
        trade_id=trade_id,
        occurred_at=moment,
        reason=f"{kind.kind}:{state}",
        payload=payload,
        source="trade_mentor",
    )
    return {"ok": True, "row": row, "answer_key": kind.answer_key}


def _mood_fields(kind: QuestionKind, answer: Mapping[str, Any], state: str) -> dict[str, Any]:
    """TJ-7's three journal arguments, from what the trader clicked. Or ``{}``.

    The mood RIDES the `day_close` kind - it is not a registry kind of its own,
    because a second kind would need a second consumer and a second budget slot
    for one strip. So only that kind ever carries one, and only when something
    was actually clicked: an untouched strip files the answer the trader DID
    give and adds no mood block at all.

    `Followed the plan?` is the process half, and it is taken only when the
    click is one of the three plan answers: the same combo also offers TJ-9's
    four answer states and `Stop asking this`, and "not remembered" is not
    "I did not follow the plan".
    """
    import market_journal

    if kind.kind != KIND_DAY_CLOSE:
        return {}
    score = (answer or {}).get("mood")
    tags = tuple(str(code) for code in ((answer or {}).get("state_tags") or ()))
    note = _text((answer or {}).get("note"))
    followed = state if state in market_journal.FOLLOWED_PLAN_VALUES else ""
    if score is None and not tags and not note and not followed:
        return {}
    return {
        "mood": score,
        "state_tags": tags,
        "process": {"followed_plan": followed, "note": note},
    }


def _record_in_journal(
    kind: QuestionKind,
    subject: Subject,
    payload: Mapping[str, Any],
    *,
    journal: Any,
    now: datetime,
    mood: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A day-close or AI-question answer is one dated Market Journal row.

    The Trade Mentor's own rule (CLAUDE.md): *an answer is one dated Market
    Journal row*. The words are what Day Review and the week story read.

    TJ-7's mood travels as FIELDS on that same row, never as a sentence: a mood
    parsed back out of the row's text later would be a second, drifting reader
    of what the trader clicked.
    """
    service = journal
    if service is None:
        try:
            from ui.services.market_journal_service import shared_journal_service

            service = shared_journal_service()
        except Exception as exc:  # noqa: BLE001 - a missing journal is loud
            return {"ok": False, "reason": f"the Market Journal is not available: {exc}"}
    answer = _text(payload.get(kind.answer_key))
    body = f"{subject.prompt or kind.kind} {answer}".strip()
    fields = dict(mood or {})
    if not answer and fields:
        # The trader clicked a face and left the question alone. The row still
        # needs a sentence a page can print, and it says exactly that - the
        # RECORD of what they felt is the fields, never these words.
        body = f"{subject.prompt or kind.kind} (not answered; a mood was recorded)"
    result = service.write_entry(
        text=body,
        session_date=_text(subject.detail.get("session")) or _text(payload.get("session")),
        timeframe="M5",
        origin="trade_mentor",
        now=now,
        mentor={"mentor_question": dict(payload)},
        **fields,
    )
    return {"ok": bool(result.get("ok")), "row": result.get("entry") or {}, "reason": result.get("reason", "")}


# ---------------------------------------------------------------------------
# the day's ONE journal pull
# ---------------------------------------------------------------------------


def pull_slot_ids(session: Any, slots: Sequence[Any] | None = None) -> tuple[str, ...]:
    """Which of the session's cards are allowed to start a pre-card pull.

    TJ-14B review, blocker 3. First-come spent the whole day's budget by the
    08:00 card, so every fill after 11:00 ET went unimported and `same_session`
    - the label this packet exists to make reachable - was unreachable all
    afternoon while the card went on asking.

    The attempts are SPACED and RESERVED: the 09:00 card (the trade check's own
    hour), the middle card between it and the close, and the LAST card of the
    session. The hours are read off the session's REAL slot list, so an early
    close simply has fewer of them - a missing slot forfeits its attempt and
    never rolls it earlier.
    """
    if slots is None:
        if isinstance(session, datetime):
            session = session.date()
        elif not isinstance(session, date):
            text = _text(session)[:10]
            try:
                session = date.fromisoformat(text)
            except ValueError:
                return ()
        try:
            from trade_mentor_schedule import slots_for_session

            slots = slots_for_session(session)
        except Exception:  # noqa: BLE001 - an unreadable calendar pulls nothing
            logging.debug("Pull schedule unreadable.", exc_info=True)
            return ()
    ordered = tuple(slots or ())
    if not ordered:
        return ()
    last = ordered[-1]
    try:
        from trade_mentor_schedule import TRADES_HOUR
    except Exception:  # noqa: BLE001
        TRADES_HOUR = 9
    anchor = next(
        (slot for slot in ordered if getattr(slot, "scheduled_at").hour == TRADES_HOUR),
        None,
    )
    chosen: list[Any] = []
    if anchor is not None and anchor is not last:
        chosen.append(anchor)
    start = ordered.index(anchor) + 1 if anchor is not None else 0
    between = list(ordered[start:-1])
    if between:
        chosen.append(between[len(between) // 2])
    chosen.append(last)
    seen: list[str] = []
    for slot in chosen:
        slot_id = _text(getattr(slot, "slot_id", ""))
        if slot_id and slot_id not in seen:
            seen.append(slot_id)
    return tuple(seen)


def _tally_for(today: str, tally: Mapping[str, Any] | None) -> dict[str, Any]:
    """Today's tally. A tally from another day resets - the cap is per DAY.

    A tally that cannot be read is read as EMPTY and rewritten clean: a corrupt
    state file must not be able to stop the desk pulling for a day, and it must
    not be able to raise inside a Qt slot either (TJ-14B review, item B).
    """
    day = _text(today)[:10]
    if not isinstance(tally, Mapping):
        return {"day": day, "pulls": 0, "failures": 0}
    current = dict(tally)
    if _text(current.get("day"))[:10] != day:
        return {"day": day, "pulls": 0, "failures": 0}
    # Anything else the caller parked in here (the desk keeps TJ-9's
    # `last_retry` beside the counts, so the once-a-morning rule survives a
    # restart too) travels through untouched - and is dropped by the day reset,
    # which is exactly what a per-DAY note should do.
    current.update({"day": day, "pulls": _count(current.get("pulls")), "failures": _count(current.get("failures"))})
    return current


def _count(value: Any) -> int:
    """A counter that refuses to raise. `"three"` is not a number of pulls."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def pre_card_pull(
    service: Any,
    *,
    today: str,
    tally: Mapping[str, Any] | None = None,
    days: int = PRE_CARD_PULL_DAYS,
    auto_mode: str = "",
    slot: Any = None,
    counts_against_cap: bool = True,
) -> dict[str, Any]:
    """The ONE owner of the desk's day-time Questrade attempts.

    It CALLS `ui/services/journal_import_service.JournalImportService`, which
    owns its own `QThread` and is the single caller of the Questrade refresh
    chain. Nothing here refreshes a token, opens a broker session or touches the
    Qt thread beyond starting that worker, and **it never raises**: the card the
    trader is looking at is worth more than the fills it wanted.

    Questrade only - IBKR has no day leg at all; its fills arrive with the
    overnight run, and the card says so rather than pretending otherwise.

    Returns ``{"pulled", "reason", "tally"}``. The caller persists the tally, so
    the cap survives a desk restart.
    """
    current = _tally_for(today, tally)
    if _text(auto_mode).upper() == "AWAY":
        return {"pulled": False, "reason": "AWAY asks nothing and pulls nothing", "tally": current}
    if service is None:
        return {"pulled": False, "reason": "no import service", "tally": current}
    if current["failures"] >= PULL_FAILURES_PER_DAY_CAP:
        return {
            "pulled": False,
            "reason": (
                f"the day's failure cap is reached ({current['failures']} failed) - "
                "the token chain is worth more than one more attempt"
            ),
            "tally": current,
        }
    if counts_against_cap:
        # TJ-14B review, blocker 3: the day's three attempts are RESERVED for
        # three spaced cards, never handed to whoever asks first.
        if slot is not None:
            reserved = pull_slot_ids(_text(getattr(slot, "session", "")) or today)
            if reserved and _text(getattr(slot, "slot_id", "")) not in reserved:
                return {
                    "pulled": False,
                    "reason": "this card is not one of the day's reserved pulls",
                    "tally": current,
                }
        if current["pulls"] >= PULLS_PER_DAY_CAP:
            return {
                "pulled": False,
                "reason": f"the day's pull cap is reached ({current['pulls']})",
                "tally": current,
            }
        current["pulls"] += 1
    try:
        started = bool(service.pull_recent_questrade(int(days)))
    except Exception as exc:  # noqa: BLE001 - a pull never costs the card
        logging.debug("Pre-card journal pull failed to start.", exc_info=True)
        current["failures"] += 1
        return {"pulled": False, "reason": str(exc), "tally": current}
    if not started:
        # Already running. The attempt is spent either way: a second card an
        # hour later must not queue a third pull behind it.
        return {"pulled": False, "reason": "an import is already running", "tally": current}
    return {"pulled": True, "reason": "", "tally": current}


__all__ = [
    "BUDGET",
    "CADENCE_DAILY",
    "CADENCE_ONCE",
    "CADENCE_PER_CARD",
    "CADENCE_WEEKLY",
    "CardQuestions",
    "LONG_HOLD_SESSIONS",
    "PRE_CARD_PULL_DAYS",
    "PULLS_PER_DAY_CAP",
    "PULL_FAILURES_PER_DAY_CAP",
    "QuestionKind",
    "REGISTRY",
    "STOP_ASKING",
    "Subject",
    "consumer_report",
    "kind_named",
    "pending",
    "pre_card_pull",
    "pull_slot_ids",
    "record_answer",
]
