"""The 09:00 question: what is missing from yesterday's trades - WS-TM item 4.

The Trade Mentor's 09:00 Pacific card has two sections. The first is an ordinary
market read. The second is this: the PREVIOUS exchange session's trades, and for
each one only the material fields it is actually missing.

**Only what is missing.** "All the support data" is not an ever-growing
compulsory questionnaire. A trade that already carries its thesis, its setup
claim and its stop is asked ONE question, not four - otherwise the morning task
becomes something the trader stops doing, and a task nobody does records nothing.

**Forced, at 09:00** (TJ-9, trader 2026-09-19: *"I want to be forced to label my
trades around 0900 as per trade mentor"*). Every trade of the reviewed session
is listed - the cap of three was five minutes of questions and the trader asked
for all of them - and the card's Save stays grey until each listed field holds a
value or one of the four explicit answer states.

**A machine guess is not an answer.** 33 live trades carry a `provisional` tag
and exactly one carries a confirmed one. A provisional tag used to retire the
setup question silently, which is how 215 trades reached one confirmed label.
The setup is asked until the TRADER confirms it; the guess rides on the question
as a suggestion, in lane order - the setup of a CLAIMED like stamped before the
first fill, else the provisional tag - and a guess nobody clicked writes
NOTHING.

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

**The cap survives, for the backlog only.** `TRADE_CAP_DEFAULT` is still three
and is still the rule for an OLDER backlog ("Tag this week"); it no longer caps
the session the card is forcing, and no caller builds a backlog task yet.

**No coverage, no questionnaire.** If the broker statement for that session has
not landed, an empty list of trades is a lie about the session. It says
`journal not ready` and NAMES THE DATE the fills are current to, and the card
keeps the section up for the rest of the session instead of asking nothing all
day.

Storage is `journal_store`'s existing append-only `opportunity_events` table
under the `RECALLED` event type - an annotation kind, not a schema migration.
Nothing here reaches a detector, a score, a gate, an alert, Focus or the review
queue (plan.md sec 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import journal_store

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
EVENT_RECALLED_RAW = "RECALLED_RAW"

#: TJ-9E. The trader's own words about one EXIT, and the three fields they
#: confirmed or corrected afterwards. Two types, because they are two different
#: facts with two different authors: the first is what the trader wrote, the
#: second is what they signed off. Nothing machine-written ever lands in either
#: - the night's reading is a `provisional` draft in a pack file.
EVENT_EXIT_NOTE_RAW = "EXIT_NOTE_RAW"
EVENT_EXIT_FIELDS = "EXIT_NOTE_FIELDS"

#: The exit ask, in the trader's own three questions (2026-09-21). ONE box and
#: nothing to pick from: *"ideally we can just write it out and the AI fills
#: this stuff in overnight"*.
EXIT_PROMPT = "Why did you exit? What did you feel? What were you watching?"

#: How many exact quotes an exit's `watching` may carry. A bound shared with
#: the night slot, which re-checks it against this number.
MAX_EXIT_WATCHING = 3

#: How long the night's reading of an exit note stays ON OFFER, in exchange
#: sessions walked on the calendar (review 1 blocker: draft reachability).
#:
#: A draft is offered on its OWN, independent of the session a card happens to
#: be reviewing, because the timeline never lines up: the trade exits Monday,
#: the trader types the note on TUESDAY's 09:00 card (which reviews Monday),
#: Tuesday NIGHT drafts it - and Wednesday's card reviews TUESDAY, where
#: Monday's trade is not a row at all. Without a window of its own the Confirm
#: click would never appear, which is half of what the trader asked for:
#: *"ideally we can just write it out and the AI fills this stuff in
#: overnight"*.
#:
#: Five, the desk's own week (`evidence_stats.WEEK_SESSIONS`), because a
#: reading of what the trader wrote is worth confirming while they still
#: remember the trade and stops being worth interrupting them for after that.
#: Past the window a draft is no longer OFFERED; it stays on disk as
#: `provisional`, it is still counted as unexplained-but-drafted in the report
#: card's clause, and NOTHING is ever confirmed by age.
EXIT_DRAFT_OFFER_SESSIONS = 5

#: Why an exit note could not claim `same_session`. Two reasons, and neither is
#: ever a guess: a date-only fill has no moment for the note to be before, and
#: an unreadable leg list is uncertainty.
REASON_EXIT_LEGS_UNREADABLE = "the exit's fills could not be read"

#: The status values that mean the trade touched the session, READ FROM THE
#: WRITER (`journal_store.TRADE_STATUSES`) rather than spelled here.
#:
#: This tuple used to hold `PARTIALLY_CLOSED`, which nothing on the desk has
#: ever written: the assembler stamps `CLOSED_PARTIAL`
#: (`journal_store._finalize_trade_state`). Seven live trades - every one the
#: trader had half exited - were therefore dropped by the filter below and were
#: never asked about at 09:00. A second spelling of somebody else's vocabulary
#: is a defect waiting for a rename, so this is the import and not a copy.
_SESSION_STATUSES = journal_store.TRADE_STATUSES

REASON_NOT_READY = "journal not ready"

#: Why a label made on the fill's own DATE is still not `same_session`: a broker
#: file is authoritative for money and BLIND TO TIME, so a date-only fill has no
#: moment for a label to have been made before.
REASON_DATE_ONLY_FILL = "the fill is date-only; there is no time to be before"

#: Which lane produced the machine's setup suggestion. Lane order, never
#: confidence: what the trader NAMED when they claimed the like outranks what
#: the bulk tagger guessed afterwards.
LANE_CLAIMED_LIKE = "claimed_like"
LANE_PROVISIONAL = "provisional"
GUESS_LANES = (LANE_CLAIMED_LIKE, LANE_PROVISIONAL)

#: How many days the ONE morning import retry asks for. Three, because a long
#: weekend is three sessions and the importer is idempotent per day.
MORNING_RETRY_DAYS = 3


@dataclass(frozen=True)
class TradeQuestion:
    """One trade, the material fields it is missing, and the machine's guess.

    `setup_guess` is a SUGGESTION and never an answer: it is offered as a
    confirm button beside the vocabulary list, and until the trader clicks it
    nothing is written. `opened_at` and `trade_date` travel with the question so
    the provenance of a confirmed label can be decided from the trade's own
    stamps without a second query.
    """

    trade_id: str
    symbol: str
    direction: str
    missing: tuple[str, ...]
    setup_guess: str = ""
    setup_guess_lane: str = ""
    opened_at: str = ""
    trade_date: str = ""
    #: TJ-9E. The session this row's EXIT box is about, or ``""`` when the
    #: trade had no closing fill in the session being reviewed. It is a
    #: SESSION and not a boolean because a trade can exit in two of them - 13
    #: live trades do - and each half is its own question on its own morning.
    exit_session: str = ""
    #: Has that exit already been answered - words OR an answer state? An
    #: answered exit is never asked again (review 1 blocker 5: the box came
    #: back EMPTY on the next card of the same morning and greyed Save, so the
    #: trader had to retype what they had written an hour before). The row can
    #: still be on the card for an entry gap, and then it shows the words the
    #: trader wrote, read-only.
    exit_answered: bool = False
    #: What they wrote, so the card can show it back without a second read.
    exit_note: str = ""
    #: Or the answer state they gave instead of words.
    exit_answer_state: str = ""


@dataclass(frozen=True)
class TradeCheckTask:
    """The 09:00 card's second section.

    `remaining` and `incomplete_total` are the whole truth behind any cap, so
    the Journal's completeness view can say "N trades still missing fields"
    rather than leaving the rest unmentioned. Since TJ-9 the reviewed session is
    never capped, so `remaining` is 0 for it.

    `fills_current_to` is the last session with verified import coverage, as an
    ISO date or "". The TASK knows it - not the widget - so the card, the
    Journal and the AWAY digest print one line built once (decision 0021
    answer 27: the report says how fresh it is).
    """

    reviewed_session: str
    journal_ready: bool = False
    trades: tuple[TradeQuestion, ...] = ()
    remaining: int = 0
    incomplete_total: int = 0
    reason: str = ""
    #: Every incomplete trade id, capped or not - the completeness view's list.
    incomplete_trade_ids: tuple[str, ...] = field(default=())
    fills_current_to: str = ""
    #: TJ-14B item 2/4: the trade ids filled on the card's OWN session. They sit
    #: in `trades` beside the reviewed session's, because the label made before
    #: the outcome is known is the one worth the most - and the three label ages
    #: are only worth reporting apart if the desk actually produces the middle
    #: one. Coverage is deliberately NOT required for these: today's statement
    #: never lands mid-session, and a fill the desk has already SEEN is a fill
    #: the trader can label.
    same_session_trade_ids: tuple[str, ...] = field(default=())


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


def fills_current_to(store: Any) -> date | None:
    """The last session the desk has VERIFIED import coverage for, or ``None``.

    The newest COVERED day, never the newest ROW: a `FAILED` day and a
    `NO_SESSION` day are both present in the ledger and neither one is a day
    whose fills the trader has. ``None`` means no ledger and no claim - an
    absence is not a date.
    """
    try:
        import journal_coverage

        rows = journal_coverage.coverage_rows(store)
    except Exception:  # noqa: BLE001 - an unreadable ledger names no date
        logging.debug("Journal coverage unreadable.", exc_info=True)
        return None
    best: date | None = None
    for row in rows:
        if str(row.get("status") or "").upper() != journal_coverage.COVERED:
            continue
        text = str(row.get("day") or "")[:10]
        if len(text) < 10:
            continue
        try:
            day = date.fromisoformat(text)
        except ValueError:
            continue
        if best is None or day > best:
            best = day
    return best


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _setup_is_answered(trade: Mapping[str, Any]) -> bool:
    """Is the setup the TRADER's, rather than a machine's parked guess?

    TJ-9. `list_trades` joins `setup_tags` whatever its `tag_status`, so before
    this a `provisional` tag - written by `journal_bulk_tag`, clicked by nobody
    - silently retired the one question this task exists to ask. The row
    already carries `tag_status`, so telling the two apart costs no extra read.
    A trade with no annotation row at all reads `confirmed` with empty tags,
    which is correctly "still missing".
    """
    if not _has_value(trade.get("setup_tags")):
        return False
    status = str(trade.get("tag_status") or "").strip().lower()
    # An absent status is the pre-P6a reading: nothing machine-written is on
    # the row, so what is there is the trader's.
    return status in ("", "confirmed")


def missing_fields(trade: Mapping[str, Any], answered: set[str]) -> tuple[str, ...]:
    """Which material fields this trade still cannot answer.

    A field counts as answered either because the trade RECORDS it or because
    the trader already answered it in a recalled row - including "there was no
    target", which is a complete answer and must stop being asked. The setup is
    recorded only when the trader CONFIRMED it; a provisional tag is a
    suggestion, not an answer.
    """
    result: list[str] = []
    for name in MATERIAL_FIELDS:
        if name in answered:
            continue
        if name == "setup":
            if _setup_is_answered(trade):
                continue
            result.append(name)
            continue
        source = _FIELD_SOURCES.get(name) or ""
        if source and _has_value(trade.get(source)):
            continue
        result.append(name)
    return tuple(result)


def claimed_setup_rows(session: str, path: Any = None) -> list[dict[str, Any]]:
    """Every CLAIMED like in the annotation log that could name this session's
    trades, read through the store that owns the file.

    Bounded by the same window the preference report uses to decide that a
    statement was acted on (`preference_trade_outcomes.statement_window_end`,
    ten exchange sessions): a claim from three months ago is not what the
    trader had in mind this morning. A quick like names no setup (P9) and is
    skipped here for that reason, never as a judgement about it.
    """
    try:
        from project_paths import TRADER_ANNOTATIONS_FILE
        from ui.annotations.store import EVENT_LIKE_CLAIM, load_annotations

        rows = load_annotations(
            Path(path or TRADER_ANNOTATIONS_FILE), event_types=(EVENT_LIKE_CLAIM,)
        )
    except Exception:  # noqa: BLE001 - a missing log is no suggestion
        logging.debug("Claimed likes unreadable.", exc_info=True)
        return []

    try:
        reviewed = date.fromisoformat(str(session)[:10])
    except ValueError:
        return []

    kept: list[dict[str, Any]] = []
    for row in rows:
        if not str(row.get("claimed_setup_id") or "").strip():
            continue
        text = str(row.get("session_date") or "")[:10]
        if len(text) < 10:
            continue
        try:
            said_on = date.fromisoformat(text)
        except ValueError:
            continue
        if said_on > reviewed:
            continue
        try:
            from preference_trade_outcomes import statement_window_end

            if statement_window_end(said_on) < reviewed:
                continue
        except Exception:  # noqa: BLE001 - a calendar that refuses keeps the row
            logging.debug("Statement window unreadable.", exc_info=True)
        kept.append(dict(row))
    return kept


def setup_vocabulary() -> tuple[str, ...]:
    """The setup names the card LISTS beside the confirm button.

    The capture rail's own claim registry (`ui.annotations.setup_claims`, the
    ids a `claimed_setup_id` may legally carry) plus the family names the setup
    documents state (`ai_jobs.enrichment.setup_vocabulary`). Read from those
    two, never restated here: a vocabulary written twice drifts, and the copy
    nobody edits becomes a falsehood shipped as data.
    """
    names: list[str] = []
    for loader in (_registry_claim_ids, _document_family_names):
        try:
            for name in loader():
                slug = str(name or "").strip().lower()
                if slug and slug not in names and not is_rejection_or_link(slug):
                    names.append(slug)
        except Exception:  # noqa: BLE001 - a missing vocabulary is a short list
            logging.debug("Setup vocabulary partly unreadable.", exc_info=True)
    return tuple(sorted(names))


def _registry_claim_ids():
    from ui.annotations.setup_claims import valid_setup_claim_ids

    return valid_setup_claim_ids()


def _document_family_names():
    from ai_jobs.enrichment import setup_vocabulary as document_vocabulary

    return document_vocabulary()


def is_rejection_or_link(tag: Any) -> bool:
    """Is this tag something that may NEVER be offered as a setup?

    Two families, each decided by the module that owns it: a REJECTION
    (`vetoed:<code>` / `passed:<codes>` - `journal_analytics.is_rejection_tag`)
    and a LINK (`link:<kind>` - `journal_analytics.is_link_candidate`, a
    pointer that is explicitly never a tag). An unreadable rule refuses rather
    than offers: the cost of a missing suggestion is a click, and the cost of a
    wrong one is a rejection counted forever as a setup.
    """
    text = str(tag or "").strip()
    if not text:
        return True
    try:
        from journal_analytics import is_link_candidate, is_rejection_tag

        return bool(is_rejection_tag(text) or is_link_candidate(text))
    except Exception:  # noqa: BLE001
        logging.debug("Rejection/link rule unreadable.", exc_info=True)
        return True


def eligible_setup_names(tags: Any) -> tuple[str, ...]:
    """The parts of a tag string that could honestly be offered as a setup.

    A trade's `setup_tags` is a LIST in one column, so `;` - the journal's
    top-level separator - is split first. Each top-level tag is then tested
    WHOLE, because a pass writes all of its reason codes inside one tag as
    `passed:thin,extended` and `split_tags` splits on the comma: filtering
    after that split would leave `extended` standing alone as an eligible setup
    name, and `extended` is a reason the trader stayed OUT.

    Two refusals, and no third:

    * a rejection or a link is never a setup (:func:`is_rejection_or_link`);
    * a part carrying a `:` is a `<prefix>:<code>` shape from some store, not a
      setup name, so it is refused on SHAPE rather than against a second list.

    Membership of :func:`setup_vocabulary` is offered to the trader as the list
    beside the button, not enforced here: the provisional lane's names come
    from the scanner's own `setup_family` values, which that vocabulary does
    not contain, and refusing them would silently delete the lane.
    """
    whole = str(tags or "").strip()
    if not whole:
        return ()
    try:
        from journal_analytics import split_tags
    except Exception:  # noqa: BLE001 - an unreadable splitter offers nothing
        logging.debug("Tag splitter unreadable.", exc_info=True)
        return ()

    kept: list[str] = []
    for tag in (part.strip() for part in whole.split(";")):
        if not tag or is_rejection_or_link(tag):
            continue
        for part in split_tags(tag):
            slug = str(part or "").strip()
            if not slug or ":" in slug or is_rejection_or_link(slug):
                continue
            if slug not in kept:
                kept.append(slug)
    return tuple(kept)


def setup_guess_for(
    trade: Mapping[str, Any], claims: Any = ()
) -> tuple[str, str]:
    """The machine's best setup suggestion for one trade, and its lane.

    Lane order, never confidence:

    1. the setup of a CLAIMED like on that name and side stamped BEFORE the
       first fill - what the trader themselves named at the time;
    2. the `provisional` tag the bulk tagger parked on the row.

    ``("", "")`` when there is nothing to suggest - including for a trade whose
    setup the trader already confirmed, which is never offered a guess at all.
    """
    if _setup_is_answered(trade):
        return "", ""
    try:
        import trade_origin

        best: tuple[Any, str] | None = None
        for row in trade_origin.statements_before_entry(trade, claims or ()):
            # The claim lane gets the same filter as the tag lane: the guess
            # must not depend on a validation two stores away.
            eligible = eligible_setup_names(row.get("claimed_setup_id"))
            if not eligible:
                continue
            stamp = trade_origin.stamp_of(row)
            if stamp is None:
                continue
            # The LATEST claim before the fill: a trader who renamed the
            # setup twice meant the second one.
            if best is None or stamp > best[0]:
                best = (stamp, eligible[0])
        if best is not None:
            return best[1], LANE_CLAIMED_LIKE
    except Exception:  # noqa: BLE001 - a suggestion never costs the question
        logging.debug("Claimed-like suggestion failed.", exc_info=True)

    status = str(trade.get("tag_status") or "").strip().lower()
    if status == "provisional":
        # The column holds a LIST, and on the live journal it holds rejections:
        # APTV carried `vetoed:too_extended_from_base`, which this used to
        # offer as a one-click confirmed SETUP.
        eligible = eligible_setup_names(trade.get("setup_tags"))
        if eligible:
            return eligible[0], LANE_PROVISIONAL
    return "", ""


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
                "unit": str(payload.get("unit") or ""),
                "source_span": str(payload.get("source_span") or ""),
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


def questions_for_session(store: Any, reviewed: str) -> list[TradeQuestion] | None:
    """One question per trade of `reviewed` that still cannot answer a field.

    ``None`` - not ``[]`` - when the trade list could not be read: an empty
    questionnaire drawn from an unreadable list is a lie about the session, and
    the two have to stay distinguishable.

    The annotation log is read ONCE for the whole session rather than once per
    trade: it is a small append-only file, but a per-trade read would turn a
    four-trade morning into four full-file walks on the Qt thread.
    """
    try:
        trades = store.list_trades(trade_date=reviewed)
    except Exception:  # noqa: BLE001
        logging.debug("Trade list unreadable.", exc_info=True)
        return None

    claims = claimed_setup_rows(reviewed)
    # ONE session-wide read of the exit notes, before the loop: an exit the
    # trader has already explained is ANSWERED and is never asked again, and
    # asking that per trade would be a walk of the append-only table per row.
    answered_exits = exit_notes_for_session(store, reviewed) if trades else {}
    questions: list[TradeQuestion] = []
    for trade in trades:
        status = str(trade.get("status") or "").upper()
        if status and status not in _SESSION_STATUSES:
            continue
        trade_id = str(trade.get("trade_id") or "")
        if not trade_id:
            continue
        gaps = missing_fields(trade, answered_fields(store, trade_id))
        # TJ-9E. An EXIT in the reviewed session is a question in its own
        # right, so the row survives an EMPTY `missing`. A swing opened last
        # week and closed yesterday has every entry field answered already -
        # it was asked the morning after it opened - and `if not gaps:
        # continue` is what meant nothing ever asked about its exit. Live, 116
        # of 180 closed trades exit on a day other than the one they opened.
        exit_session = reviewed if reviewed in exit_sessions(store, trade_id) else ""
        answered = dict(answered_exits.get(trade_id) or {}) if exit_session else {}
        # An answered exit closes its question the way an answered field does.
        # The row survives only while something on it is still OPEN.
        if not gaps and (not exit_session or answered):
            continue
        guess, lane = setup_guess_for(trade, claims) if "setup" in gaps else ("", "")
        questions.append(
            TradeQuestion(
                trade_id=trade_id,
                symbol=str(trade.get("symbol") or ""),
                direction=str(trade.get("direction") or ""),
                missing=gaps,
                setup_guess=guess,
                setup_guess_lane=lane,
                opened_at=str(trade.get("opened_at") or ""),
                trade_date=str(trade.get("trade_date") or reviewed),
                exit_session=exit_session,
                exit_answered=bool(answered),
                exit_note=str(answered.get("raw_text") or ""),
                exit_answer_state=str(answered.get("answer_state") or ""),
            )
        )
    return questions


def unexplained_exit_count(store: Any, session: str) -> int:
    """How many exits ON `session` the trader has not explained yet.

    The other half of :func:`unlabelled_trade_count`, and a SEPARATE number on
    purpose (review 1 blocker 4). The count a badge prints and the predicate
    that decides whether the 09:00 section RIDES were briefly the same
    function: narrowing it to material fields made a swing whose entries were
    answered long ago count zero, so an exit nobody explained was asked once at
    09:00 and never again if that card was missed. Both numbers are owed, and
    the desk asks again while EITHER is above zero.

    An unreadable list answers 0 rather than a guess: uncertainty is never a
    count.
    """
    return sum(
        1
        for question in questions_for_session(store, str(session)[:10]) or ()
        if question.exit_session and not question.exit_answered
    )


def build_task(store: Any, session: date, *, cap: int = TRADE_CAP_DEFAULT) -> TradeCheckTask:
    """The 09:00 question for the session before `session`.

    Every trade of the reviewed session is listed - TJ-9 item 2, the trader's
    own word. `cap` is kept because `TRADE_CAP_DEFAULT` is still the rule for
    an older backlog, which no caller builds yet; it does not cap the session
    being forced, so `remaining` is 0 here by construction.
    """
    if isinstance(session, datetime):
        session = session.date()
    reviewed = previous_exchange_session(session)
    current = fills_current_to(store)
    fresh_to = current.isoformat() if current else ""
    # TJ-14B item 4. Today's own fills, whether or not last night's statement
    # landed: the desk asks about what it has SEEN. Never gated on coverage -
    # today's statement does not exist yet, and waiting for it is exactly how
    # every live label came to be `recalled_after`.
    today = tuple(questions_for_session(store, session.isoformat()) or ())
    today_ids = tuple(question.trade_id for question in today)

    if not _journal_ready(store, reviewed):
        return TradeCheckTask(
            reviewed_session=reviewed,
            journal_ready=False,
            trades=today,
            incomplete_total=len(today),
            reason=REASON_NOT_READY,
            incomplete_trade_ids=today_ids,
            fills_current_to=fresh_to,
            same_session_trade_ids=today_ids,
        )

    questions = questions_for_session(store, reviewed)
    if questions is None:
        return TradeCheckTask(
            reviewed_session=reviewed,
            journal_ready=False,
            trades=today,
            incomplete_total=len(today),
            reason=REASON_NOT_READY,
            incomplete_trade_ids=today_ids,
            fills_current_to=fresh_to,
            same_session_trade_ids=today_ids,
        )

    # The reviewed session first, today's fills after it. A trade that somehow
    # answers to both dates is listed ONCE: two sections for one trade is two
    # Save gates on the same fields.
    seen = {question.trade_id for question in questions}
    asked = tuple(questions) + tuple(q for q in today if q.trade_id not in seen)
    return TradeCheckTask(
        reviewed_session=reviewed,
        journal_ready=True,
        trades=asked,
        remaining=0,
        incomplete_total=len(asked),
        reason="",
        incomplete_trade_ids=tuple(question.trade_id for question in asked),
        fills_current_to=fresh_to,
        same_session_trade_ids=tuple(q.trade_id for q in today if q.trade_id not in seen),
    )


def unlabelled_trade_count(store: Any, session: str) -> int:
    """How many trades ON `session` still cannot answer a material field.

    The number a later reader prints ("N trade(s) unlabelled"). It asks about
    the session's OWN trades, so a caller does not have to know which morning
    the card would have reviewed them on. An unreadable list answers 0 rather
    than a guess: uncertainty is never a count.

    MATERIAL FIELDS ONLY. Since TJ-9E a row also comes back for an unexplained
    EXIT, which is a different question with a different answer; counting one
    as "unlabelled" would make a fully labelled trade read as unlabelled
    forever in the Journal's completeness view. The exits have their own count
    (:func:`unexplained_exit_count`) and BOTH are owed - see that docstring for
    why they are two numbers and not one.
    """
    return sum(
        1
        for question in questions_for_session(store, str(session)[:10]) or ()
        if question.missing
    )


def confirm_setup(
    store: Any,
    question: TradeQuestion,
    *,
    now: datetime | None = None,
    setup: str = "",
) -> dict[str, Any]:
    """The trader's one click: the suggested setup becomes THEIR confirmed tag.

    Written through the Journal's own writer (`save_trade_annotation`, which
    sets `tag_status='confirmed'`), which is what makes it the trader's act -
    the machine never confirms anything. The provenance comes from
    `trade_origin.label_provenance`, a pure function of stamps; the button
    decides nothing. An existing note is carried through unchanged: a confirm
    is about the setup, and a note is the trader's prose.
    """
    chosen = str(setup or question.setup_guess or "").strip()
    if not chosen:
        return {"ok": False, "reason": "there is nothing to confirm"}
    # The SECOND line of defence, after `eligible_setup_names` filtered the
    # guess. `setup` also arrives from the card's vocabulary list, so the
    # refusal lives at the WRITER too: a rejection confirmed as a setup is
    # counted by "My setups" forever, and `trade_annotations` is trader-owned.
    if eligible_setup_names(chosen) != (chosen,):
        return {
            "ok": False,
            "reason": f"{chosen!r} is not a setup name - a rejection is never confirmed as one",
        }
    trade_id = str(question.trade_id)
    state = {}
    try:
        state = store.annotation_state(trade_id)
    except Exception:  # noqa: BLE001 - a missing row is an empty one
        logging.debug("Annotation state unreadable.", exc_info=True)
    if str(state.get("tag_status") or "") == "confirmed" and str(
        state.get("setup_tags") or ""
    ).strip():
        # The trader already answered. Never offered, never overwritten.
        return {"ok": False, "reason": "this trade already carries a confirmed tag"}

    trade = {
        "trade_id": trade_id,
        "symbol": question.symbol,
        "direction": question.direction,
        "opened_at": question.opened_at,
        "trade_date": question.trade_date,
    }
    moment = now or datetime.now().astimezone()
    try:
        import trade_origin

        provenance = trade_origin.label_provenance(
            trade, chosen, claimed_setup_rows(question.trade_date), moment
        )
    except Exception:  # noqa: BLE001 - an undecidable age is left unrecorded
        logging.debug("Label provenance undecidable.", exc_info=True)
        provenance = ""

    store.save_trade_annotation(
        trade_id,
        setup_tags=chosen,
        notes=str(state.get("notes") or ""),
        label_provenance=provenance,
    )
    return {"ok": True, "setup": chosen, "label_provenance": provenance}


def morning_import_retry(
    service: Any,
    task: TradeCheckTask,
    *,
    today: str,
    last_retry: str = "",
    tally: Mapping[str, Any] | None = None,
    auto_mode: str = "",
) -> dict[str, Any]:
    """The ONE deterministic import retry the desk makes before the 09:00 card.

    TJ-9 item 6. When last night ended without an OK import for the session the
    card is about, the desk pulls once more before asking - seconds, no model.
    Everything about it is a refusal to do more than that:

    * it CALLS `ui/services/journal_import_service.JournalImportService`, which
      owns its own `QThread` and is the desk's single Questrade refresh-chain
      owner. Nothing here refreshes a token, opens a session or touches the
      Qt thread beyond starting that worker;
    * **Questrade only.** IBKR has no day leg at all - its fills arrive with
      the overnight run - and the card says so rather than pretending a retry
      could find them;
    * at most ONCE per morning, keyed on the date the caller passes. A ready
      task retries nothing, and a second card the same day retries nothing;
    * a service that is already running, or that refuses, is not an error: the
      pull it is already doing is the pull this wanted.

    Returns ``{"retried", "reason", "last_retry"}``; the caller persists
    ``last_retry`` so the once-a-morning rule survives a card being rebuilt.
    """
    day = str(today or "")[:10]
    if getattr(task, "journal_ready", False):
        return {"retried": False, "reason": "the journal is ready", "last_retry": last_retry}
    if str(last_retry or "")[:10] == day and day:
        return {"retried": False, "reason": "already retried today", "last_retry": last_retry}
    if str(auto_mode or "").upper() == "AWAY":
        # Defence in depth: `pre_card_pull` refuses AWAY too, and both seams
        # say so, because the trader who is not there cannot be interrupted by
        # a broker call made on their behalf either.
        return {
            "retried": False,
            "reason": "AWAY asks nothing and pulls nothing",
            "last_retry": last_retry,
        }
    if service is None:
        return {"retried": False, "reason": "no import service", "last_retry": last_retry}
    # TJ-14B lead decision 3: the desk's day-time Questrade attempts have ONE
    # owner, and this retry goes THROUGH it rather than beside it. The
    # once-a-morning rule above is still this function's; the failure cap and
    # the tally are `pre_card_pull`'s. It does NOT spend the pre-card cap
    # (review blocker 1): the three spaced pre-card attempts and the ONE
    # morning catch-up answer different questions, and the catch-up reaches
    # further back (`MORNING_RETRY_DAYS`), so a day that spent one on the other
    # would lose Friday's fills on a Monday.
    import mentor_questions

    outcome = mentor_questions.pre_card_pull(
        service,
        today=day,
        tally=tally,
        days=MORNING_RETRY_DAYS,
        auto_mode=auto_mode,
        counts_against_cap=False,
    )
    reason = str(outcome.get("reason") or "")
    if outcome.get("pulled"):
        return {"retried": True, "reason": "", "last_retry": day, "tally": outcome["tally"]}
    # `last_retry` is stamped ONLY when an import actually STARTED (review
    # blocker 1). A service that was already busy, or that refused, did not do
    # the pull this wanted: marking the morning spent there is how a Monday
    # whose Friday-night import failed lost its three-day catch-up entirely.
    return {
        "retried": False,
        "reason": reason,
        "last_retry": last_retry,
        "tally": outcome["tally"],
    }


def _answer_provenance(
    trade: Mapping[str, Any],
    trade_id: str,
    trade_date: str,
    moment: datetime,
) -> tuple[str, bool, str]:
    """How OLD this answer is, decided by `trade_origin.label_provenance`.

    TJ-14B lead decision 4: the flag comes from the pure rule over the trade's
    own stamps, never from a constant. `label_provenance` is asked with NO
    setup, so only its two reachable answers here are possible - `same_session`
    for an answer given on the session of the first fill, `recalled_after` for
    everything else. `claimed_before_entry` belongs to a CONFIRMED SETUP and is
    :func:`confirm_setup`'s to decide, not a remembered stop's.

    An unreadable rule keeps the old, conservative claim (`recalled_after`):
    calling a next-morning answer a same-session one would present remembered
    risk as a documented pre-entry plan, which is the one thing this module
    exists to make impossible.
    """
    row = dict(trade or {})
    if not row:
        row = {"trade_id": str(trade_id), "trade_date": str(trade_date or "")}
    try:
        import trade_origin

        provenance = trade_origin.label_provenance(row, "", (), moment)
        if provenance == trade_origin.SAME_SESSION and trade_origin.first_fill_at(row) is None:
            # A BROKER FILE IS AUTHORITATIVE FOR MONEY AND BLIND TO TIME. The
            # statement importer writes every date-only fill at MIDNIGHT
            # market-local, and `trade_session` still names a real DATE for it -
            # so a label typed at 11:00 on the day a date-only fill is dated
            # would read `same_session`, which claims the trader labelled it
            # before the outcome was known. It cannot be known: there is no
            # time to be before. `recalled_after`, with the reason recorded.
            return trade_origin.RECALLED_AFTER, True, REASON_DATE_ONLY_FILL
        return provenance, provenance != trade_origin.SAME_SESSION, ""
    except Exception:  # noqa: BLE001 - an undecidable age keeps the old claim
        logging.debug("Answer provenance undecidable.", exc_info=True)
        return "", True, "provenance undecidable"


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
    trade: Mapping[str, Any] = {}
    try:
        trade = store.get_trade(str(trade_id)) or {}
    except Exception:  # noqa: BLE001 - a missing row still stores the answer
        logging.debug("Trade row unreadable for a recalled answer.", exc_info=True)
        trade = {}
    if not trade_date:
        trade_date = str(trade.get("trade_date") or "")
    provenance, recalled_after, provenance_reason = _answer_provenance(
        trade, trade_id, trade_date, moment
    )
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
            "unit": str((answer or {}).get("unit") or ""),
            "source_span": str((answer or {}).get("source_span") or ""),
            # TJ-14B. The BOOLEAN that matches the provenance, never the
            # hard-coded `True` this used to stamp on every row: since the card
            # lists today's fills, a label made on the fill's own session is
            # reachable, and "remembered the next morning" would be a false
            # claim about it. The provenance itself travels beside it.
            "recalled_after_session": recalled_after,
            "label_provenance": provenance,
            # Why a same-session label was REFUSED, when it was. Empty when
            # nothing was refused: an absence is never a reason.
            "label_provenance_reason": provenance_reason,
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


def save_raw_reply(
    store: Any,
    trade_id: str,
    raw_text: str,
    *,
    missing: tuple[str, ...] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Save the trader's exact words before any local model sees them."""
    body = str(raw_text or "")
    if not body.strip():
        raise ValueError("the raw answer is empty")
    moment = now or datetime.now().astimezone()
    payload = {
        "raw_text": body,
        "missing_fields": [name for name in missing if name in MATERIAL_FIELDS],
        "recalled_after_session": True,
    }
    return store.record_opportunity_event(
        opportunity_id=f"trade:{trade_id}",
        event_type=EVENT_RECALLED_RAW,
        trade_id=str(trade_id),
        occurred_at=moment,
        reason="raw_next_morning_reply",
        payload=payload,
        source="trade_mentor",
    )


# ---------------------------------------------------------------------------
# TJ-9E - the EXIT: one box, the words first, and three fields the NIGHT drafts
# ---------------------------------------------------------------------------
def _close_legs(store: Any, trade_id: str) -> tuple[list[datetime], bool]:
    """``(the trade's closing fills as aware moments, readable)``.

    The moments keep their OWN offset - they are converted to market-local only
    where a SESSION is wanted. A fill stored ``2026-09-11T00:00:00-07:00`` is a
    broker row that carries no clock time, and converting it first would read
    it as three in the morning in New York (`trade_origin._is_midnight`, which
    records exactly that live row).

    ``readable`` is False when the leg list could not be opened at all, which
    is uncertainty and never an empty session.
    """
    try:
        legs = store.list_trade_legs(str(trade_id))
    except Exception:  # noqa: BLE001 - an unreadable leg list is uncertainty
        logging.debug("Trade legs unreadable for an exit.", exc_info=True)
        return [], False
    import trade_origin

    moments: list[datetime] = []
    for leg in legs or ():
        if str((leg or {}).get("role") or "").upper() != "CLOSE":
            continue
        moment = trade_origin._moment((leg or {}).get("timestamp"))
        if moment is not None:
            moments.append(moment)
    return moments, True


def exit_sessions(store: Any, trade_id: str) -> tuple[str, ...]:
    """Every session this trade left part of its position in, oldest first.

    A SCALE-OUT inside one session is ONE session here, however many fills it
    took: the trader took a position off once and explains it once. Two closing
    legs on two dates are two sessions, and each is asked on its own morning -
    13 live trades are that shape.
    """
    import trade_origin

    moments, _readable = _close_legs(store, trade_id)
    days: list[str] = []
    for moment in sorted(moments):
        day = moment.astimezone(trade_origin.MARKET_TZ).date().isoformat()
        if day not in days:
            days.append(day)
    return tuple(days)


def _exit_note_provenance(
    store: Any, trade_id: str, exit_session: str, moment: datetime
) -> tuple[str, bool, str]:
    """How old this exit note is, measured against the EXIT's own session.

    NOT `trade_origin.trade_session`, which is the session of the FIRST FILL:
    that is the right ruler for an entry field and the wrong one for an exit.
    116 of the journal's 180 closed trades exit on a day other than the one
    they opened, so reusing it would call a note typed on the exit's own
    afternoon `recalled_after` - and, on a trade opened and exited in one
    session, would be right by accident.

    A DATE-ONLY exit is never `same_session`, for the reason
    :func:`_answer_provenance` already records: a broker file is authoritative
    for money and blind to time, the statement importer writes a date-only fill
    at midnight, and there is no moment for the note to be before. The midnight
    test is `trade_origin`'s WIDER one - midnight in the stamp's own offset OR
    market-local - rather than `journal_trade_shape.is_date_only`, which asks
    the market-local question alone. That module's own docstring records why:
    the live DRAM row is stored ``2026-07-16T00:00:00-07:00`` and reads as a
    fill at three in the morning in New York, which is a time that never
    happened. An unreadable leg list is the same answer for the same reason -
    uncertainty keeps the conservative claim.
    """
    import trade_origin

    session = str(exit_session or "")[:10]
    moments, readable = _close_legs(store, trade_id)
    if not readable:
        return trade_origin.RECALLED_AFTER, True, REASON_EXIT_LEGS_UNREADABLE

    on_the_day = [
        row
        for row in moments
        if row.astimezone(trade_origin.MARKET_TZ).date().isoformat() == session
    ]
    if on_the_day and all(trade_origin._is_midnight(row) for row in on_the_day):
        return trade_origin.RECALLED_AFTER, True, REASON_DATE_ONLY_FILL

    said_on = moment.astimezone(trade_origin.MARKET_TZ).date().isoformat()
    if session and said_on == session:
        return trade_origin.SAME_SESSION, False, ""
    return trade_origin.RECALLED_AFTER, True, ""


def save_exit_note(
    store: Any,
    trade_id: str,
    raw_text: str,
    *,
    exit_session: str,
    state: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """The trader's own words about one exit, on disk, BEFORE anything reads them.

    RAW FIRST. Nothing is parsed, coded, scored or summarised on the way in;
    the night's reading happens hours later against the stored row, which is
    the whole reason the row is written first.

    **A failed write is LOUD.** A journal write is the one evidence store
    CLAUDE.md allows to fail the thing it records, and this is the trader's own
    sentence about their own trade: swallowing it into a log would lose it. A
    blank note is refused for the same reason - it is not a note, and storing
    one would retire a question nobody answered.

    A SECOND note on the same exit APPENDS (`opportunity_events` is immutable
    by design), so the first row stays byte-identical and the fact that the
    trader changed their mind survives.

    ``state`` is the answer the trader gives INSTEAD of words - `not
    remembered` or `not applicable`, from TJ-9's own four. It is a complete
    answer and must stop being asked, which is why it is stored; it is not an
    explanation, so nothing counts it as one and the night never reads it.
    """
    body = str(raw_text or "")
    answer_state = str(state or "").strip()
    if answer_state and answer_state not in ANSWER_STATES:
        raise ValueError(f"{answer_state!r} is not one of the four answer states")
    if not body.strip() and not answer_state:
        raise ValueError("the exit note is empty")
    session = str(exit_session or "")[:10]
    moment = now or datetime.now().astimezone()
    trade: Mapping[str, Any] = {}
    try:
        trade = store.get_trade(str(trade_id)) or {}
    except Exception:  # noqa: BLE001 - a missing row still stores the words
        logging.debug("Trade row unreadable for an exit note.", exc_info=True)
        trade = {}
    provenance, written_after, reason = _exit_note_provenance(
        store, trade_id, session, moment
    )
    payload = {
        "raw_text": body,
        "answer_state": answer_state,
        "exit_session": session,
        "label_provenance": provenance,
        # Why a same-session claim was REFUSED, when it was. Empty when nothing
        # was refused: an absence is never a reason.
        "label_provenance_reason": reason,
        # COMPUTED, never backdated - the Market Journal's own rule.
        "written_after_the_session": written_after,
        "trade_date": str(trade.get("trade_date") or ""),
    }
    return store.record_opportunity_event(
        opportunity_id=f"trade:{trade_id}",
        event_type=EVENT_EXIT_NOTE_RAW,
        trade_id=str(trade_id),
        symbol=str(trade.get("symbol") or ""),
        side=str(trade.get("direction") or ""),
        occurred_at=moment,
        reason="exit_note",
        payload=payload,
        source="trade_mentor",
    )


def _note_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """One stored raw note, as every reader of it sees it."""
    payload = row.get("payload") or {}
    return {
        "note_id": str(row.get("event_id") or ""),
        "trade_id": str(row.get("trade_id") or ""),
        "symbol": str(row.get("symbol") or ""),
        "side": str(row.get("side") or ""),
        "raw_text": str(payload.get("raw_text") or ""),
        "answer_state": str(payload.get("answer_state") or ""),
        "exit_session": str(payload.get("exit_session") or ""),
        "label_provenance": str(payload.get("label_provenance") or ""),
        "label_provenance_reason": str(payload.get("label_provenance_reason") or ""),
        "written_after_the_session": bool(payload.get("written_after_the_session")),
        "occurred_at": str(row.get("occurred_at") or ""),
    }


def exit_notes(store: Any, trade_id: str) -> list[dict[str, Any]]:
    """Every exit note stored for one trade, OLDEST FIRST.

    Oldest first because the store is append-only: "the note" is the LAST row,
    and a reader that wants the history has it in the order it was written.
    """
    try:
        rows = store.list_opportunity_events(
            trade_id=str(trade_id), event_type=EVENT_EXIT_NOTE_RAW, limit=10000
        )
    except Exception:  # noqa: BLE001 - a missing store is no notes
        logging.debug("Exit notes unreadable.", exc_info=True)
        return []
    return [_note_row(row) for row in rows]


def exit_notes_by_session(
    store: Any, sessions: Any
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{session: {trade_id: note}}`` for several sessions in TWO reads.

    The append-only table is walked ONCE for the words and once for what the
    trader signed off, however many sessions are asked for: a five-session
    window is a card-build cost and five separate session reads would be five
    walks of the same table on the Qt thread.

    The LAST note for a (trade, session) wins - a superseding note is the
    trader's latest word - and the trade's CONFIRMED fields travel on the same
    row, so a caller that wants both does not open the table twice. A trade
    that exited in two sessions appears under each session with its own note.
    """
    wanted = [str(day or "")[:10] for day in (sessions or ()) if str(day or "")[:10]]
    out: dict[str, dict[str, dict[str, Any]]] = {day: {} for day in wanted}
    if not wanted:
        return out
    days = set(wanted)
    try:
        rows = store.list_opportunity_events(
            event_type=EVENT_EXIT_NOTE_RAW, limit=10000
        )
    except Exception:  # noqa: BLE001 - an unreadable table is no notes
        logging.debug("Session exit notes unreadable.", exc_info=True)
        return out
    for row in rows:
        note = _note_row(row)
        if note["exit_session"] not in days or not note["trade_id"]:
            continue
        out[note["exit_session"]][note["trade_id"]] = note
    if not any(out.values()):
        return out
    confirmed = _confirmed_exit_rows(store, days)
    for day, notes in out.items():
        for trade_id, note in notes.items():
            note["exit_fields"] = dict((confirmed.get(day) or {}).get(trade_id) or {})
    return out


def exit_notes_for_session(store: Any, session: str) -> dict[str, dict[str, Any]]:
    """``{trade_id: the session's exit note}`` in ONE read of the whole table.

    Session-wide rather than per trade: a six-trade morning would otherwise be
    six walks of an append-only table on a Qt worker, and the Day Review page
    reads this once for the whole day.
    """
    wanted = str(session or "")[:10]
    if not wanted:
        return {}
    return exit_notes_by_session(store, (wanted,)).get(wanted, {})


def _confirmed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("payload") or {}
    return {
        "status": journal_store.TAG_STATUS_CONFIRMED,
        "fields": dict(payload.get("fields") or {}),
        "exit_session": str(payload.get("exit_session") or ""),
        "note_id": str(payload.get("note_id") or ""),
        "source": str(payload.get("source") or ""),
        "confirmed_at": str(row.get("occurred_at") or ""),
    }


def _confirmed_exit_rows(store: Any, sessions: Any) -> dict[str, dict[str, Any]]:
    """``{session: {trade_id: the trader's confirmed fields}}``, in ONE read."""
    days = {str(day or "")[:10] for day in (sessions or ()) if str(day or "")[:10]}
    out: dict[str, dict[str, Any]] = {day: {} for day in days}
    if not days:
        return out
    try:
        rows = store.list_opportunity_events(event_type=EVENT_EXIT_FIELDS, limit=10000)
    except Exception:  # noqa: BLE001 - an unreadable table is nothing confirmed
        logging.debug("Confirmed exit fields unreadable.", exc_info=True)
        return out
    for row in rows:
        record = _confirmed_row(row)
        trade_id = str(row.get("trade_id") or "")
        if not trade_id or record["exit_session"] not in days:
            continue
        out[record["exit_session"]][trade_id] = record
    return out


def exit_fields(store: Any, trade_id: str) -> dict[str, Any]:
    """What the TRADER confirmed about one exit, or ``{}``.

    ``{}`` is the honest answer while a draft is waiting: a machine reading is
    never counted as the trader's, which is the rule 33 provisional tags and
    exactly one confirmed one already taught this desk.
    """
    try:
        rows = store.list_opportunity_events(
            trade_id=str(trade_id), event_type=EVENT_EXIT_FIELDS, limit=10000
        )
    except Exception:  # noqa: BLE001 - a missing store is nothing confirmed
        logging.debug("Confirmed exit fields unreadable.", exc_info=True)
        return {}
    if not rows:
        return {}
    return _confirmed_row(rows[-1])


def offer_window(session: Any, sessions: int = EXIT_DRAFT_OFFER_SESSIONS) -> tuple[str, ...]:
    """The `sessions` exchange sessions ending at `session`, OLDEST first.

    Walked on the exchange calendar, never in calendar days: a Monday card
    reaching back five days would reach back three sessions. A calendar that
    cannot answer falls back to days, which is WIDER than the truth and so
    never silently narrows the window.
    """
    try:
        cursor = date.fromisoformat(str(session or "")[:10])
    except ValueError:
        return ()
    days = [cursor]
    for _step in range(max(0, int(sessions) - 1)):
        try:
            from market_calendar import previous_session

            cursor = previous_session(cursor)
        except Exception:  # noqa: BLE001 - an unanswerable calendar walks days
            logging.debug("The exchange calendar could not walk back.", exc_info=True)
            from datetime import timedelta

            cursor = cursor - timedelta(days=1)
        days.append(cursor)
    return tuple(day.isoformat() for day in reversed(days))


def waiting_exit_drafts(
    store: Any,
    session: Any,
    *,
    sessions: int = EXIT_DRAFT_OFFER_SESSIONS,
    root: Any = None,
) -> list[dict[str, Any]]:
    """Every UNCONFIRMED exit draft still on offer, OLDEST FIRST.

    The lane behind the `exit_draft_review` question. It is keyed to the
    trader's CLOCK and not to whatever session the card is reviewing - see
    :data:`EXIT_DRAFT_OFFER_SESSIONS` for why the two never line up - and it
    reads the window in TWO store queries plus one small pack read per session.

    A draft the trader has already Confirmed or Corrected is GONE: the
    signed-off row is the fact and nothing is offered twice. A draft whose note
    has no words behind it any more, or which names a trade this store cannot
    see, is skipped rather than shown as an empty card row.

    Never raises: an unreadable pack is no drafts, which is also the honest
    state of a desk whose night has never run.
    """
    window = offer_window(session, sessions)
    if not window:
        return []
    notes = exit_notes_by_session(store, window)
    out: list[dict[str, Any]] = []
    for day in window:
        try:
            from ai_jobs import exit_note_fields

            stored = exit_note_fields.read_latest(day, root=root) or {}
        except Exception:  # noqa: BLE001 - a missing pack is simply no draft
            logging.debug("Exit drafts unreadable for %s.", day, exc_info=True)
            continue
        for draft in stored.get("drafts") or ():
            if not isinstance(draft, Mapping):
                continue
            trade_id = str(draft.get("trade_id") or "")
            note = (notes.get(day) or {}).get(trade_id) or {}
            if not trade_id or not note or note.get("exit_fields"):
                continue
            out.append(
                {
                    "trade_id": trade_id,
                    "symbol": str(draft.get("symbol") or note.get("symbol") or ""),
                    "exit_session": day,
                    "note_id": str(draft.get("note_id") or ""),
                    "raw_text": str(note.get("raw_text") or ""),
                    "fields": dict(draft.get("fields") or {}),
                    "status": str(draft.get("status") or ""),
                    "drafted_at": str(draft.get("drafted_at") or ""),
                }
            )
    return out


def _exit_vocabularies() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(why codes, felt codes)`` - each read from the module that OWNS it.

    `trader_state_tags` is TJ-7's feelings vocabulary and there is exactly one
    on this desk: an exit's feeling and a session's mood are different GRAINS
    of the same words, and a second list would let the two drift.
    """
    import exit_reasons
    import trader_state_tags

    return tuple(exit_reasons.codes()), tuple(trader_state_tags.codes())


def _stored_value(value: Any, *, coded: bool) -> dict[str, Any]:
    """One field value as it is STORED on the trader's confirmed row.

    `span` and `quote` are PRESENT and EMPTY for a value the trader typed
    themselves: a reader that has to tell "the trader wrote this" from "this
    build did not record it" is reading two different absences as one.
    """
    row = dict(value or {}) if isinstance(value, Mapping) else {}
    out: dict[str, Any] = {
        "span": list(row.get("span") or ()),
        "quote": str(row.get("quote") or ""),
    }
    if coded:
        out["code"] = str(row.get("code") or "")
    return out


def _write_exit_fields(
    store: Any,
    trade_id: str,
    fields: Mapping[str, Any],
    *,
    exit_session: str,
    note_id: str,
    source: str,
    moment: datetime,
) -> dict[str, Any]:
    """The ONE writer of a confirmed exit row. The trader's act, never a job's."""
    payload = {
        "fields": dict(fields),
        "exit_session": str(exit_session or "")[:10],
        "note_id": str(note_id or ""),
        # `confirm` or `correct` - what the trader DID, so a later reader can
        # tell a signed-off draft from one they rewrote.
        "source": str(source or ""),
        "status": journal_store.TAG_STATUS_CONFIRMED,
    }
    row = store.record_opportunity_event(
        opportunity_id=f"trade:{trade_id}",
        event_type=EVENT_EXIT_FIELDS,
        trade_id=str(trade_id),
        occurred_at=moment,
        reason=f"exit_fields:{source}",
        payload=payload,
        source="trade_mentor",
    )
    return {"ok": True, "fields": dict(fields), "event_id": str(row.get("event_id") or "")}


def confirm_exit_fields(
    store: Any,
    trade_id: str,
    draft: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The trader's one click: the night's reading becomes THEIR record.

    A confirm is an APPEND. The draft file is not touched and not rewritten -
    what the machine said stays readable beside what the trader signed - and
    the codes are re-checked against the two closed vocabularies here, because
    a row a later reader has to interpret must carry a code that list still
    holds.
    """
    body = dict(draft or {})
    fields = body.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        return {"ok": False, "reason": "there is nothing to confirm"}
    why_codes, felt_codes = _exit_vocabularies()

    stored: dict[str, Any] = {}
    why = fields.get("why")
    if isinstance(why, Mapping):
        code = str(why.get("code") or "")
        if code not in why_codes:
            return {"ok": False, "reason": f"{code!r} is not an exit reason this desk knows"}
        stored["why"] = {**_stored_value(why, coded=True), "code": code}
    felt_rows = fields.get("felt")
    if isinstance(felt_rows, (list, tuple)):
        kept = []
        for value in felt_rows:
            code = str((value or {}).get("code") or "") if isinstance(value, Mapping) else ""
            if code not in felt_codes:
                return {"ok": False, "reason": f"{code!r} is not a feeling this desk knows"}
            kept.append({**_stored_value(value, coded=True), "code": code})
        stored["felt"] = kept
    watching = fields.get("watching")
    if isinstance(watching, (list, tuple)):
        # Stored as the QUOTES themselves: the span was the night's proof that
        # it did not invent them, and the trader's own correction has no span.
        stored["watching"] = [
            str((value or {}).get("quote") or "") if isinstance(value, Mapping) else str(value)
            for value in watching
        ][:MAX_EXIT_WATCHING]
    if not stored:
        return {"ok": False, "reason": "there is nothing to confirm"}
    return _write_exit_fields(
        store,
        trade_id,
        stored,
        exit_session=str(body.get("exit_session") or body.get("session_date") or ""),
        note_id=str(body.get("note_id") or ""),
        source="confirm",
        moment=now or datetime.now().astimezone(),
    )


def correct_exit_fields(
    store: Any,
    trade_id: str,
    *,
    why: str = "",
    felt: Any = (),
    watching: Any = (),
    now: datetime | None = None,
    exit_session: str = "",
    note_id: str = "",
) -> dict[str, Any]:
    """The trader rewrites the three values. Still the two closed vocabularies.

    A code outside them RAISES rather than being stored: a corrected row is
    still a row a later reader has to interpret, and the one thing worse than a
    machine's wrong code is a code nobody can look up at all. The quotes are
    free text - they are the trader's own words about what they were watching,
    so they carry `span: []` and `quote: ""`, and `source: "correct"` on the
    row is what tells a grounded value from a free one.

    `exit_session` and `note_id` are RESOLVED from the trade's own latest exit
    note when a caller does not pass them, and a trade with no note at all
    RAISES (review 1 advisory 5). They used to default to ``""``, which wrote a
    row that joined to nothing: both readers key on the session, so a row
    without one shows no confirmed fields anywhere and looks like a confirm
    that never happened.
    """
    why_codes, felt_codes = _exit_vocabularies()
    chosen = str(why or "").strip()
    if chosen and chosen not in why_codes:
        raise ValueError(f"{chosen!r} is not an exit reason this desk knows")
    feelings = [str(code or "").strip() for code in (felt or ()) if str(code or "").strip()]
    for code in feelings:
        if code not in felt_codes:
            raise ValueError(f"{code!r} is not a feeling this desk knows")
    quotes = [str(text or "").strip() for text in (watching or ()) if str(text or "").strip()]
    if len(quotes) > MAX_EXIT_WATCHING:
        raise ValueError(f"an exit carries at most {MAX_EXIT_WATCHING} things watched")
    session = str(exit_session or "")[:10]
    note = str(note_id or "")
    if not session or not note:
        latest = (exit_notes(store, trade_id) or [None])[-1]
        if latest is None:
            raise ValueError(
                "there is no exit note on this trade to correct; a corrected row "
                "with no session joins to nothing"
            )
        session = session or str(latest.get("exit_session") or "")
        note = note or str(latest.get("note_id") or "")
    if not session:
        raise ValueError("a corrected exit row must name the session its exit is in")
    stored: dict[str, Any] = {
        "felt": [{"code": code, "span": [], "quote": ""} for code in feelings],
        "watching": quotes,
    }
    if chosen:
        stored["why"] = {"code": chosen, "span": [], "quote": ""}
    return _write_exit_fields(
        store,
        trade_id,
        stored,
        exit_session=session,
        note_id=note,
        source="correct",
        moment=now or datetime.now().astimezone(),
    )
