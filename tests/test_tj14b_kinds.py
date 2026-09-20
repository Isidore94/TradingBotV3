"""TJ-14B item 2 - what each kind fires on, and what answering one writes.

RED BEFORE THE FIX. On `e8c04f88` `scripts/mentor_questions.py` does not exist.

THE CONTRACT THESE PIN (plan.md 12.4 TJ-14 item 2)
--------------------------------------------------
* `trade_origin` fires on a trade with `planned_state == "unplanned"`, ONCE,
  and never on a planned one. The state comes from `trade_origin.planned_state`
  over the four lanes - never re-implemented here.
* `open_position_check` fires on an OPEN position past FIVE exchange sessions,
  once per exchange week per position. The five-session boundary is asserted
  against `market_calendar.trading_days_between` in the test itself, so a
  calendar surprise names itself instead of hiding in a literal.
* `quick_like_followup` fires on a QUICK like that was traded or that became a
  `REAL_MISS_V1` run - never on a CLAIMED like, which already named its setup.
  Answering it writes a LINK that names the like and leaves the like row BYTE
  IDENTICAL, and writes no tag: `trade_annotations` is the trader's.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_lift_dormancy import lift_dormancy  # noqa: E402,F401
from tj14b_support import (  # noqa: E402
    MID_WEEK,
    NEXT_WEEK,
    SESSION,
    keys_of,
    like_row,
    new_store,
    pacific,
    slot_at,
    state,
    trade_row,
)


def _owed(result) -> set[tuple[str, str]]:
    return keys_of(result.asked) | keys_of(result.carried)


# ---------------------------------------------------------------------------
# trade_origin
# ---------------------------------------------------------------------------


def test_trade_origin_is_asked_once_and_never_for_a_planned_trade():
    """A trade with a claim on that name and side before its first fill came
    from somewhere. A trade from nowhere is the one worth asking about."""
    import mentor_questions

    unplanned = trade_row("T-NOWHERE", symbol="AAPL", day="2026-09-11")
    planned = trade_row("T-PLANNED", symbol="MSFT", day="2026-09-11")

    payload = state(
        trades=[unplanned, planned],
        # One claim on MSFT LONG, stamped an hour before the 07:31 fill.
        claims=[
            {
                "symbol": "MSFT",
                "side": "long",
                "created_at": "2026-09-11T06:30:00-04:00",
                "claimed_setup_id": "avwap_reclaim",
            }
        ],
    )

    result = mentor_questions.pending(payload, slot_at(SESSION, 11))

    assert ("trade_origin", "T-NOWHERE") in _owed(result)
    assert ("trade_origin", "T-PLANNED") not in _owed(result)


def test_trade_origin_offers_the_four_origins_the_trader_was_given():
    """The closed set from the packet. A free-text origin cannot be counted."""
    import mentor_questions

    options = set(mentor_questions.kind_named("trade_origin").options)

    assert {"planned_off_the_desk", "an_alert", "impulse", "other"} <= options


def test_trade_origin_is_never_asked_twice_about_one_trade():
    """*"once"*. The trade's origin does not change after the trader says it."""
    import mentor_questions

    payload = state(trades=[trade_row("T-NOWHERE", symbol="AAPL", day="2026-09-11")])
    assert ("trade_origin", "T-NOWHERE") in _owed(
        mentor_questions.pending(payload, slot_at(SESSION, 11))
    )

    payload["answered"] = {
        "trade_origin:T-NOWHERE": {"answered_at": "2026-09-14", "state": "impulse"}
    }

    assert ("trade_origin", "T-NOWHERE") not in _owed(
        mentor_questions.pending(payload, slot_at(SESSION, 11))
    )


# ---------------------------------------------------------------------------
# open_position_check
# ---------------------------------------------------------------------------


def test_the_five_session_boundary_is_the_exchange_calendars_and_not_a_week():
    """The fixture's own arithmetic, asserted before anything leans on it.

    Counting back from Monday 2026-09-14: 09-11, 09-10, 09-09, 09-08, 09-04
    (2026-09-07 is Labor Day). So 2026-09-04 is exactly five sessions back and
    2026-09-03 is six.
    """
    import market_calendar

    assert market_calendar.trading_days_between(date(2026, 9, 4), SESSION) == 5
    assert market_calendar.trading_days_between(date(2026, 9, 3), SESSION) == 6


def test_open_position_check_fires_only_past_five_sessions():
    """*"an OPEN position past five sessions"*. Exactly five is not past five."""
    import mentor_questions

    payload = state(
        open_positions=[
            trade_row("P-FIVE", symbol="TLT", day="2026-09-04", status="OPEN"),
            trade_row("P-SIX", symbol="GLD", day="2026-09-03", status="OPEN"),
        ]
    )

    owed = _owed(mentor_questions.pending(payload, slot_at(SESSION, 11)))

    assert ("open_position_check", "P-SIX") in owed
    assert ("open_position_check", "P-FIVE") not in owed


def test_open_position_check_asks_once_per_exchange_week_per_position():
    """Answered on Monday, silent on Wednesday, back the following Monday.

    A daily "is the thesis intact?" on a position held for a month is the desk
    nagging; a weekly one is the desk keeping a record.
    """
    import mentor_questions

    held = trade_row("P-HELD", symbol="GLD", day="2026-08-17", status="OPEN")
    answered = {
        "open_position_check:P-HELD": {
            "answered_at": SESSION.isoformat(),
            "state": "thesis_intact",
        }
    }

    same_week = state(
        session=MID_WEEK,
        now=pacific(MID_WEEK, 11),
        open_positions=[held],
        answered=answered,
    )
    assert ("open_position_check", "P-HELD") not in _owed(
        mentor_questions.pending(same_week, slot_at(MID_WEEK, 11))
    )

    next_week = state(
        session=NEXT_WEEK,
        now=pacific(NEXT_WEEK, 11),
        open_positions=[held],
        answered=answered,
    )
    assert ("open_position_check", "P-HELD") in _owed(
        mentor_questions.pending(next_week, slot_at(NEXT_WEEK, 11))
    )


def test_a_closed_trade_is_never_an_open_position_check():
    """The question is *"is the thesis intact"*. A closed trade has no thesis
    left to hold."""
    import mentor_questions

    payload = state(
        trades=[trade_row("T-CLOSED", symbol="GLD", day="2026-08-17")],
        open_positions=[],
    )

    assert ("open_position_check", "T-CLOSED") not in _owed(
        mentor_questions.pending(payload, slot_at(SESSION, 11))
    )


# ---------------------------------------------------------------------------
# quick_like_followup
# ---------------------------------------------------------------------------


def test_a_quick_like_that_was_traded_is_followed_up_and_a_claimed_like_is_not():
    """A QUICK like names no setup (P9), so a like cohort by family cannot see
    it. A CLAIMED like already said what it was and is never asked again."""
    import mentor_questions

    payload = state(
        likes=[
            like_row(
                "like-quick-traded",
                symbol="AMD",
                day="2026-09-11",
                like_mode="quick",
                matched_trade_id="T-1",
            ),
            like_row(
                "like-quick-quiet",
                symbol="INTC",
                day="2026-09-11",
                like_mode="quick",
            ),
            like_row(
                "like-claimed",
                symbol="NVDA",
                day="2026-09-11",
                like_mode="claimed",
                claimed_setup_id="avwap_reclaim",
                matched_trade_id="T-2",
            ),
        ]
    )

    owed = _owed(mentor_questions.pending(payload, slot_at(SESSION, 11)))

    assert ("quick_like_followup", "like-quick-traded") in owed
    assert ("quick_like_followup", "like-claimed") not in owed
    assert ("quick_like_followup", "like-quick-quiet") not in owed


def test_a_quick_like_that_became_a_real_miss_is_followed_up():
    """TJ-11 is merged, so `REAL_MISS_V1` is the second trigger the packet
    names: a name the trader liked, did not take, and that really did run."""
    import mentor_questions
    import real_miss

    payload = state(
        likes=[
            like_row(
                "like-quick-ran",
                symbol="SMCI",
                day="2026-09-11",
                like_mode="quick",
                real_miss=real_miss.REAL_MISS_V1,
            )
        ]
    )

    assert ("quick_like_followup", "like-quick-ran") in _owed(
        mentor_questions.pending(payload, slot_at(SESSION, 11))
    )


def test_a_quick_like_answer_writes_a_link_that_names_the_like(tmp_path):
    """The answer is a RECALLED CLAIM LINK, not a rewrite of the like.

    The like row is read back byte for byte after the write, and the trade's
    annotation row is untouched: a machine writes only `provisional` /
    `needs_review`, and a like carries zero privileges (CLAUDE.md, P9).
    """
    import mentor_questions

    store = new_store(tmp_path)
    annotations = tmp_path / "trader_annotations.jsonl"
    row = like_row(
        "like-quick-traded",
        symbol="AMD",
        day="2026-09-11",
        like_mode="quick",
        matched_trade_id="T-1",
    )
    annotations.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    before = annotations.read_bytes()

    payload = state(likes=[row], annotations_path=annotations)
    result = mentor_questions.pending(payload, slot_at(SESSION, 11))
    subject = next(
        item for item in list(result.asked) + list(result.carried)
        if item.kind == "quick_like_followup"
    )

    written = mentor_questions.record_answer(
        subject,
        {"state": "avwap_reclaim"},
        store=store,
        now=pacific(SESSION, 11),
    )

    assert annotations.read_bytes() == before, "the like row was rewritten"

    events = store.list_opportunity_events(limit=100)
    links = [
        event
        for event in events
        if "like-quick-traded" in json.dumps(event.get("payload") or {}, default=str)
    ]
    assert links, f"no link row names the like: {written}"

    # Nothing the trader did not click reaches `trade_annotations`.
    assert str(store.annotation_state("T-1").get("tag_status") or "") != "confirmed"


def test_the_quick_like_options_come_from_the_claim_vocabulary():
    """*"`Which setup was it?` from the claim vocabulary"* - the same closed
    list the confirm button sits beside, never a free-text setup name."""
    import mentor_questions
    import trade_mentor_trade_check as check

    options = set(mentor_questions.kind_named("quick_like_followup").options)
    vocabulary = set(check.setup_vocabulary())

    assert vocabulary, "the claim vocabulary is empty"
    assert vocabulary <= options


def test_answering_a_trade_origin_question_stores_it_with_no_manual_step(tmp_path):
    """*"then I expect the AI to take it from there"* - one click, one stored
    row, and the answer is readable under the key the registry declares."""
    import mentor_questions

    store = new_store(tmp_path)
    payload = state(trades=[trade_row("T-NOWHERE", symbol="AAPL", day="2026-09-11")])
    result = mentor_questions.pending(payload, slot_at(SESSION, 11))
    subject = next(
        item for item in list(result.asked) + list(result.carried)
        if item.kind == "trade_origin"
    )

    mentor_questions.record_answer(
        subject, {"state": "impulse"}, store=store, now=pacific(SESSION, 11)
    )

    key = mentor_questions.kind_named("trade_origin").answer_key
    rows = [
        event
        for event in store.list_opportunity_events(trade_id="T-NOWHERE", limit=100)
        if key in json.dumps(event.get("payload") or {}, default=str)
    ]
    assert rows, f"no stored row carries the {key!r} answer"
