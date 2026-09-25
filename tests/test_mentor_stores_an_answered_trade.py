"""An answered trade is STORED, and it is not asked about again.

Trader, 2026-09-21: *"trade mentor asks me about my trades over and over despite
me already answering. if i answer the questions about a trade please then dont
ask for it again just store that info"*.

What the live journal showed that morning: ONE `RECALLED` row, ever, and no
confirmed setup from the card - the answers were never reaching the store. Three
causes, each RED before this fix:

* the card had ONE Save over every field of every trade, so a morning with five
  trades needed twenty dropdowns before a single answer was filed;
* words typed beside a dropdown left on "-" were thrown away, and so was a raw
  note nobody sent to the local AI;
* an answer to a question ABOUT a trade was looked up by the day it was given
  on. A trade's `trade_date` moves when the position closes, so a `once`
  question came back on the closing day's card.

Everything here runs offscreen over a scratch journal under `tmp_path`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_support import (  # noqa: E402
    REVIEWED,
    SESSION,
    add_round_trip,
    mark_covered,
    new_store,
    pacific,
)

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture()
def two_trades(tmp_path):
    """A card asking about TWO trades of the reviewed session."""
    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    QApplication.instance() or QApplication([])
    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    first = add_round_trip(store, "AAA", day=REVIEWED, entry_hour=7)
    second = add_round_trip(store, "BBB", day=REVIEWED, entry_hour=8)
    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card._clock = lambda: pacific(SESSION, 9)
    card.set_trade_check(check.build_task(store, SESSION), store=store)
    assert set(card._answer_inputs) == {first, second}
    # ASKED ONCE (2026-09-23): an exit state is an answer in its own right and
    # opens Save, so the fixture leaves both exits untouched; each test says
    # what it answers.
    return store, card, first, second


def _answer_every_field(card, trade_id: str) -> None:
    import trade_mentor_trade_check as check

    for combo, _text in card._answer_inputs[trade_id].values():
        combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))
    _pick_a_setup(card, trade_id)


def _pick_a_setup(card, trade_id: str) -> str:
    """The trader picks a name in the setup list, as a click would."""
    box = card.setup_choice_box(trade_id)
    index = next(i for i in range(box.count()) if box.itemData(i))
    box.setCurrentIndex(index)
    card._setup_hand_picked(trade_id)
    return str(box.itemData(index))


def test_one_answered_trade_is_stored_while_the_other_is_still_open(two_trades):
    import trade_mentor_trade_check as check

    store, card, first, second = two_trades
    _answer_every_field(card, first)

    assert card._trade_save_buttons[first].isEnabled() is True
    assert card._trade_save_buttons[second].isEnabled() is False
    assert card.save_answers_button.isEnabled() is True, "one answered trade is enough"

    card._trade_save_buttons[first].click()

    assert check.answered_fields(store, first) == set(tuple(name for name in check.MATERIAL_FIELDS if name != "setup"))
    assert store.annotation_state(first)["tag_status"] == "confirmed", "Save kept the pick"
    assert check.answered_fields(store, second) == set()
    assert list(card._answer_inputs) == [second], "the stored trade left the card alone"
    asked = [question.trade_id for question in check.build_task(store, SESSION).trades]
    assert asked == [second], "and the next card does not ask about it again"


def test_a_half_answered_trade_is_filed_and_its_blanks_stay_blank(two_trades):
    """ASKED ONCE (trader 2026-09-23): ANY answer opens a trade's Save, a blank
    field writes no row, and the filed trade is never asked again."""
    import trade_mentor_trade_check as check

    store, card, first, second = two_trades
    _answer_every_field(card, first)
    name = sorted(card._answer_inputs[second])[0]
    combo, _text = card._answer_inputs[second][name]
    combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_APPLICABLE))

    assert card._trade_save_buttons[second].isEnabled() is True
    result = card.save_trade_check()

    assert result["ok"] is True and set(result["trades"]) == {first, second}
    assert check.answered_fields(store, second) == {name}, "blanks are not answers"
    assert check.unlabelled_trade_count(store, check.previous_exchange_session(SESSION)) == 1
    assert check.build_task(store, SESSION).trades == (), "neither is asked again"


def test_words_typed_beside_a_blank_dropdown_are_an_answer(two_trades):
    import trade_mentor_trade_check as check

    store, card, first, _second = two_trades
    for name, (_combo, text_input) in card._answer_inputs[first].items():
        text_input.setText(f"my {name}")

    assert card._trade_save_buttons[first].isEnabled() is True
    assert card.save_trade_check(first)["ok"] is True

    rows = {row["field"]: row for row in check.recalled_fields(store, first)}
    assert set(rows) == set(tuple(name for name in check.MATERIAL_FIELDS if name != "setup"))
    assert {row["state"] for row in rows.values()} == {check.ANSWER_NOT_SUPPLIED}
    assert rows["stop"]["text"] == "my stop"
    assert rows["stop"]["value"] is None, "typed words are never coerced to a number"


def test_each_trade_asks_setup_thesis_stop_target_once(two_trades):
    """Trader 2026-09-25: no catch-all note and no second setup row - the setup
    list, then thesis (optional), stop and target, each asked once."""
    from PySide6.QtWidgets import QPlainTextEdit, QPushButton

    _store, card, first, _second = two_trades
    block = card._trade_blocks[first]

    assert list(card._answer_inputs[first]) == ["thesis", "stop", "target"]
    assert card.setup_choice_box(first) is not None
    texts = [button.text() for button in block.findChildren(QPushButton)]
    assert not any("local AI" in text for text in texts)
    boxes = block.findChildren(QPlainTextEdit)
    assert all(box is card.exit_note_box(first) for box in boxes), "only the exit box"


def test_an_untouched_card_still_files_nothing(two_trades):
    store, card, first, second = two_trades

    assert card.save_answers_button.isEnabled() is False
    assert card.save_trade_check()["ok"] is False
    assert set(card._answer_inputs) == {first, second}


def test_an_answer_about_a_trade_is_found_after_its_trade_date_moved(tmp_path):
    """Answered on Monday while OPEN; closed on Thursday. Thursday's card reads
    Thursday and Wednesday - and must still see Monday's answer."""
    import mentor_questions
    from ui.app import MainWindow

    store = new_store(tmp_path)
    trade_id = add_round_trip(store, "DKS", day=SESSION.isoformat(), entry_hour=7)
    answered_on = datetime(2026, 9, 7, 7, 14, tzinfo=pacific(SESSION, 9).tzinfo)
    store.record_opportunity_event(
        opportunity_id=f"trade:{trade_id}",
        event_type=mentor_questions.EVENT_MENTOR_ANSWER,
        symbol="DKS",
        trade_id=trade_id,
        occurred_at=answered_on,
        reason="trade_origin:an_alert",
        payload={
            "mentor_question_kind": "trade_origin",
            "subject_id": trade_id,
            "trade_origin": "an_alert",
        },
        source="trade_mentor",
    )
    days = (SESSION.isoformat(), REVIEWED)

    by_day_only = MainWindow._mentor_answered(store, days)
    assert f"trade_origin:{trade_id}" not in by_day_only, "the old read could not see it"

    answered = MainWindow._mentor_answered(store, days, trade_ids=[trade_id])
    assert answered[f"trade_origin:{trade_id}"] == {"answered_at": "2026-09-07"}
