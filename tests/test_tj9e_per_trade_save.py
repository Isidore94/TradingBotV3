"""LEAD INTEGRATION TEST, 2026-09-21: TJ-9E's forced exit box under the per-trade Save.

Two pieces of the trader's own requests landed on one Save gate the same day:

* TJ-9E - an EXIT is ONE forced free-text box, and its words go to the journal
  FIRST, before the entry answers;
* `claude/mentor-stores-answers-2026-09-21` - the gate is PER TRADE: an answered
  trade is stored on its own click and leaves the card.

Neither branch could test the other. This file pins the seam where they meet:
the exit box is one more field of ITS trade (`_open_fields` asks
`_exit_is_open`), a trade with an open exit is not filed, and filing a trade
writes its exit note first and leaves the OTHER trade's half-typed words alone.
Hand-counted throughout: 2 round trips, both closed in the reviewed session.
"""

from __future__ import annotations

import os
import sys
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

WORDS = "I took it off when the 50 day cracked and I felt rushed."


@pytest.fixture
def card_with_two_exits(tmp_path):
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
    assert card.exit_note_box(first) is not None, "a closed round trip carries the exit box"
    assert card.exit_note_box(second) is not None
    return store, card, first, second


def _answer_entry_fields(card, trade_id: str) -> None:
    import trade_mentor_trade_check as check

    for combo, _text in card._answer_inputs[trade_id].values():
        combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))


def test_exit_words_open_their_own_trades_save_and_no_other(card_with_two_exits):
    """ASKED ONCE (2026-09-23): any answer opens a trade's Save - the exit
    words alone are enough - and only that trade's."""
    _store, card, first, second = card_with_two_exits

    assert card._trade_save_buttons[first].isEnabled() is False, "nothing said yet"
    assert card._trade_save_buttons[second].isEnabled() is False

    card.exit_note_box(first).setPlainText(WORDS)

    assert card._trade_save_buttons[first].isEnabled() is True
    assert card._trade_save_buttons[second].isEnabled() is False, (
        "one trade's exit words must not open another trade's gate"
    )


def test_filing_one_trade_writes_its_exit_note_and_leaves_the_other_alone(card_with_two_exits):
    import trade_mentor_trade_check as check

    store, card, first, second = card_with_two_exits
    _answer_entry_fields(card, first)
    card.exit_note_box(first).setPlainText(WORDS)
    assert card._trade_save_buttons[second].isEnabled() is False

    card._trade_save_buttons[first].click()

    notes = check.exit_notes(store, first)
    assert len(notes) == 1, notes
    assert WORDS in str(notes[0]), "the trader's own words are what was stored"
    assert check.answered_fields(store, first) == set(check.MATERIAL_FIELDS)
    assert check.exit_notes(store, second) == [], "an untouched trade files nothing"
    assert first not in card._answer_inputs, "the filed trade left the card"
    assert card.exit_note_box(first) is None, "and took its exit box with it"
    assert card.exit_note_box(second) is not None, "the other trade is still asked"


def test_the_exit_note_is_on_disk_before_the_entry_answers(card_with_two_exits, monkeypatch):
    """RAW FIRST survives the per-trade gate: if the entry half fails, the
    trader's own sentence is already in the journal."""
    import trade_mentor_trade_check as check

    store, card, first, _second = card_with_two_exits
    _answer_entry_fields(card, first)
    card.exit_note_box(first).setPlainText(WORDS)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("the entry half failed")

    monkeypatch.setattr(check, "save_answers", _boom)
    result = card.save_trade_check(first)

    assert result["ok"] is False
    assert len(check.exit_notes(store, first)) == 1, "the words were saved first"
