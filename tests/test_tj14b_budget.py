"""TJ-14B item 3 - the budget of three, the carry, and `Stop asking this`.

RED BEFORE THE FIX. On `e8c04f88` there is no `scripts/mentor_questions.py` and
`TradeMentorService` has no `stop_asking`, so every test here dies on the
import or on `AttributeError`.

THE CONTRACT THESE PIN (plan.md 12.4 TJ-14 item 3)
--------------------------------------------------
* Beyond the predictions and the forced `trade_label` section a card carries at
  most THREE questions, by priority.
* The rest are COUNTED on the card and CARRIED to the next card. Never dropped,
  never a fourth.
* `Stop asking this` retires that kind FOR THAT SUBJECT only - the kind keeps
  asking about every other subject. It is persisted beside the slot state and
  the SERVICE is its only writer.
* AWAY asks nothing.
* `day_close` is the session's LAST card only, early close included.

The priorities themselves are never re-typed here: each test reads them off the
registry, so a builder may re-rank the kinds without touching these tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_lift_dormancy import lift_dormancy  # noqa: E402,F401
from tj14b_support import (  # noqa: E402
    EARLY_CLOSE,
    SESSION,
    keys_of,
    kinds_of,
    last_slot,
    like_row,
    pacific,
    slot_at,
    state,
    trade_row,
)


def _seven_budgeted_subjects() -> dict:
    """A card with SEVEN budgeted questions owed and one forced section.

    Two unplanned trades, two long-held open positions, one traded quick like,
    one grader gap and one AI question = 7 budgeted subjects; one unlabelled
    trade = the forced `trade_label` section, which is outside the budget.
    """
    return state(
        trades=[
            trade_row("T-UNPLANNED-1", symbol="AAPL", day="2026-09-11"),
            trade_row("T-UNPLANNED-2", symbol="MSFT", day="2026-09-11"),
            trade_row("T-UNLABELLED", symbol="NVDA", day="2026-09-11"),
        ],
        open_positions=[
            trade_row(
                "P-OLD-1", symbol="TLT", day="2026-08-17", status="OPEN"
            ),
            trade_row(
                "P-OLD-2", symbol="GLD", day="2026-08-17", status="OPEN"
            ),
        ],
        likes=[
            like_row(
                "like-quick-1",
                symbol="AMD",
                day="2026-09-11",
                like_mode="quick",
                matched_trade_id="T-UNPLANNED-1",
            )
        ],
        grader_gaps=[
            {
                "question_id": "gap-1",
                "options": ("yes", "no"),
                "consumer": "market_read_grades.congruence_lines",
                "answer_key": "trader_input",
                "subject_id": "gap-1",
            }
        ],
        ai_question={"question": "Did the open drive hold?", "options": ("Yes", "No")},
    )


def test_a_card_asks_three_questions_and_counts_the_rest():
    """Seven owed, three asked, four counted. Never a fourth."""
    import mentor_questions

    result = mentor_questions.pending(_seven_budgeted_subjects(), slot_at(SESSION, 11))

    assert len(result.asked) == 3, kinds_of(result.asked)
    assert len(result.carried) == 4, kinds_of(result.carried)
    assert "4" in result.waiting_note
    assert "waiting" in result.waiting_note.lower()


def test_the_three_asked_are_the_highest_priority_of_the_seven():
    """By PRIORITY, not by arrival. The numbers come off the registry, so a
    re-rank is free and a card that asked in load order still fails."""
    import mentor_questions

    payload = _seven_budgeted_subjects()
    result = mentor_questions.pending(payload, slot_at(SESSION, 11))

    owed = list(result.asked) + list(result.carried)
    ranked = sorted(
        owed, key=lambda item: (mentor_questions.kind_named(item.kind).priority, item.kind)
    )

    assert keys_of(result.asked) == keys_of(ranked[:3])


def test_the_questions_over_budget_are_carried_to_the_next_card_and_never_dropped():
    """*"carried, never dropped"*. The four that waited are asked next hour."""
    import mentor_questions

    payload = _seven_budgeted_subjects()
    first = mentor_questions.pending(payload, slot_at(SESSION, 11))

    answered = {
        f"{item.kind}:{item.subject_id}": {"answered_at": SESSION.isoformat()}
        for item in first.asked
    }
    second_state = state(
        **{
            key: value
            for key, value in payload.items()
            if key not in ("answered", "carried", "session", "now", "auto_mode")
        },
        answered=answered,
        carried=first.carried,
        now=pacific(SESSION, 12),
    )

    second = mentor_questions.pending(second_state, last_slot(SESSION))

    assert keys_of(first.carried) <= keys_of(second.asked) | keys_of(second.carried)
    assert len(second.asked) == 3


def test_the_forced_trade_label_section_sits_outside_the_budget_of_three():
    """A card that carries yesterday's trades still asks three OTHER questions.

    TJ-9's section is not one of the three: *"on a card that carries it the
    budget covers the OTHER kinds only"*.
    """
    import mentor_questions

    result = mentor_questions.pending(_seven_budgeted_subjects(), slot_at(SESSION, 9))

    assert len(result.asked) == 3
    assert "trade_label" not in kinds_of(result.asked)
    assert "trade_label" in kinds_of(result.forced)
    assert ("trade_label", "T-UNLABELLED") in keys_of(result.forced)


def test_the_prediction_click_is_forced_and_the_d1_row_only_on_a_d1_card():
    """TJ-14A's rows are described by the registry and carried as FORCED."""
    import mentor_questions

    payload = _seven_budgeted_subjects()

    hourly = mentor_questions.pending(payload, slot_at(SESSION, 11))
    assert "prediction_m5" in kinds_of(hourly.forced)
    assert "prediction_d1" not in kinds_of(hourly.forced)

    d1 = mentor_questions.pending(payload, slot_at(SESSION, 8))
    assert "prediction_d1" in kinds_of(d1.forced)


def test_stop_asking_this_retires_one_subject_and_keeps_asking_about_the_others():
    """*"retires that kind FOR THAT SUBJECT only"*. GLD goes quiet; TLT does not."""
    import mentor_questions

    payload = _seven_budgeted_subjects()
    before = mentor_questions.pending(payload, slot_at(SESSION, 11))
    owed_before = keys_of(before.asked) | keys_of(before.carried)
    assert ("open_position_check", "P-OLD-1") in owed_before
    assert ("open_position_check", "P-OLD-2") in owed_before

    payload["retired"] = ["open_position_check:P-OLD-2"]
    after = mentor_questions.pending(payload, slot_at(SESSION, 11))
    owed_after = keys_of(after.asked) | keys_of(after.carried)

    assert ("open_position_check", "P-OLD-2") not in owed_after
    assert ("open_position_check", "P-OLD-1") in owed_after


def test_a_question_already_answered_is_never_asked_again():
    """The trader answered it. Asking twice is the desk not listening."""
    import mentor_questions

    payload = _seven_budgeted_subjects()
    before = mentor_questions.pending(payload, slot_at(SESSION, 11))
    owed_before = keys_of(before.asked) | keys_of(before.carried)
    assert ("trade_origin", "T-UNPLANNED-1") in owed_before

    payload["answered"] = {
        "trade_origin:T-UNPLANNED-1": {
            "answered_at": SESSION.isoformat(),
            "state": "impulse",
        }
    }
    after = mentor_questions.pending(payload, slot_at(SESSION, 11))

    assert ("trade_origin", "T-UNPLANNED-1") not in (
        keys_of(after.asked) | keys_of(after.carried)
    )
    assert ("trade_origin", "T-UNPLANNED-2") in (keys_of(after.asked) | keys_of(after.carried))


def test_away_asks_nothing():
    """AWAY means the trader is not there. Not one question, forced or not, and
    nothing is pushed anywhere."""
    import mentor_questions

    payload = _seven_budgeted_subjects()
    payload["auto_mode"] = "AWAY"

    result = mentor_questions.pending(payload, slot_at(SESSION, 9))

    assert result.asked == ()
    assert result.forced == ()
    assert result.carried == ()


def test_day_close_is_asked_only_on_the_last_card_of_the_session():
    """*"the session's last card only"*. An 11:00 card asking "did you follow
    the plan?" asks it before the plan has finished happening."""
    import mentor_questions

    payload = _seven_budgeted_subjects()

    early = mentor_questions.pending(payload, slot_at(SESSION, 11))
    assert "day_close" not in kinds_of(early.asked) + kinds_of(early.carried)

    closing = mentor_questions.pending(payload, last_slot(SESSION))
    assert "day_close" in kinds_of(closing.asked) + kinds_of(closing.carried)


def test_day_close_still_waits_for_the_last_card_on_an_early_close():
    """2026-11-27 closes at 10:00 Pacific and still carries the 12:00 D1 slot,
    so the LAST card is not the last card before the bell."""
    import mentor_questions
    from trade_mentor_schedule import slots_for_session

    slots = slots_for_session(EARLY_CLOSE)
    assert len(slots) >= 2, "the early close should still carry several slots"
    assert slots[-1].post_close is True

    payload = _seven_budgeted_subjects()
    payload["session"] = EARLY_CLOSE.isoformat()
    payload["now"] = pacific(EARLY_CLOSE, 9)

    not_last = mentor_questions.pending(payload, slots[-2])
    assert "day_close" not in kinds_of(not_last.asked) + kinds_of(not_last.carried)

    payload["now"] = pacific(EARLY_CLOSE, 12)
    is_last = mentor_questions.pending(payload, slots[-1])
    assert "day_close" in kinds_of(is_last.asked) + kinds_of(is_last.carried)


def test_the_service_is_the_single_writer_of_a_retired_subject(tmp_path):
    """*"persisted beside the slot state, single writer = the service"*.

    A retirement survives a desk restart, and the kind is still live for every
    other subject.
    """
    pytest.importorskip("PySide6", reason="the Mentor service is Qt")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.services.trade_mentor_service import TradeMentorService

    path = tmp_path / "slots.json"
    service = TradeMentorService(state_path=path)
    service.stop_asking("open_position_check", "P-OLD-2")

    reopened = TradeMentorService(state_path=path)

    assert "open_position_check:P-OLD-2" in set(reopened.retired_subjects())
    assert "open_position_check:P-OLD-1" not in set(reopened.retired_subjects())
