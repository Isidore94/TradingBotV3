"""P8 P2 A: the Mentor asks the STOP first (trader's go 2026-09-25).

Live, 0 of 219 trades carry a planned stop. So the per-trade block draws stop,
then target, then setup, then thesis; a trade with no stop is listed before the
other trades, and it brings the trade section onto a card that would otherwise
carry only the budgeted questions. A missing stop never blocks Save and stays
unknown.
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
    slot_at,
    state,
    trade_row,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CARD_ORDER = ("stop", "target", "setup", "thesis")


def test_the_material_fields_are_asked_stop_first():
    import trade_mentor_trade_check as check

    assert check.MATERIAL_FIELDS == CARD_ORDER
    row = trade_row("T1", symbol="AAA", day=REVIEWED)
    assert check.missing_fields(row, set()) == CARD_ORDER


def test_a_trade_with_no_stop_is_listed_first(tmp_path):
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    with_stop = add_round_trip(store, "AAA", day=REVIEWED, entry_hour=7)
    no_stop = add_round_trip(store, "BBB", day=REVIEWED, entry_hour=8)
    store.save_risk_fields(with_stop, planned_stop=9.5)

    task = check.build_task(store, SESSION)

    assert [q.trade_id for q in task.trades] == [no_stop, with_stop]
    assert check.stop_owed(task) is True


def test_pending_forces_the_no_stop_trade_first():
    import mentor_questions

    has_stop = dict(trade_row("T1", symbol="AAA", day=REVIEWED), planned_stop=9.5)
    no_stop = trade_row("T2", symbol="BBB", day=REVIEWED, opened_hour=8)
    result = mentor_questions.pending(state(trades=[has_stop, no_stop]), slot_at(SESSION, 11))

    labels = [s.subject_id for s in result.forced if s.kind == "trade_label"]
    assert labels == ["T2", "T1"]
    assert mentor_questions.kind_named("trade_label").budgeted is False
    assert mentor_questions.BUDGET == 3


def test_a_no_stop_trade_brings_the_trade_section_onto_any_card(monkeypatch):
    """Nothing else owed (statement landed, both counts zero): a trade with no
    stop still rides, so it is asked before the day's budgeted questions."""
    from datetime import date as _date

    import trade_mentor_trade_check as check
    from ui.app import MainWindow

    class _Slot:
        scheduled_at = datetime(2026, 9, 15, 11, 0)

    class _Service:
        def unlabelled_trades(self, *_a, **_k):
            return 0

        def unexplained_exits(self, *_a, **_k):
            return 0

    reviewed = check.previous_exchange_session(_Slot.scheduled_at.date())
    monkeypatch.setattr(check, "fills_current_to", lambda *_a, **_k: _date.fromisoformat(reviewed))
    window = MainWindow.__new__(MainWindow)
    window.trade_mentor_service = _Service()

    stopless = check.TradeCheckTask(
        reviewed_session=reviewed,
        journal_ready=True,
        trades=(check.TradeQuestion(trade_id="T9", symbol="ZZZ", direction="LONG", missing=("stop",)),),
    )
    stopped = check.TradeCheckTask(
        reviewed_session=reviewed,
        journal_ready=True,
        trades=(check.TradeQuestion(trade_id="T8", symbol="YYY", direction="LONG", missing=("thesis",)),),
    )
    assert MainWindow._trade_check_is_owed(window, check, _Slot(), stopless) is True
    assert MainWindow._trade_check_is_owed(window, check, _Slot(), stopped) is False
    assert MainWindow._trade_check_is_owed(window, check, _Slot()) is False


@pytest.mark.qt
def test_the_card_draws_stop_target_setup_thesis_and_save_never_needs_a_stop(tmp_path):
    pytest.importorskip("PySide6", reason="the Mentor card is Qt")
    from PySide6.QtWidgets import QApplication, QLabel

    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    QApplication.instance() or QApplication([])
    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAA", day=REVIEWED, entry_hour=7)
    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card._clock = lambda: pacific(SESSION, 9)
    card.set_trade_check(check.build_task(store, SESSION), store=store)

    # The trade section sits above the budgeted questions.
    outer = card.layout()
    assert outer.indexOf(card.trade_check_box) < outer.indexOf(card.questions_box)

    block = card._trade_blocks[trade_id]
    drawn = []
    layout = block.layout()
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if widget is None:
            continue
        labels = [widget] if isinstance(widget, QLabel) else widget.findChildren(QLabel)
        for label in labels:
            if label.text() in CARD_ORDER and label.text() not in drawn:
                drawn.append(label.text())
    assert tuple(drawn) == CARD_ORDER

    # Only the thesis is answered: Save opens, and the stop stays unknown.
    card._answer_inputs[trade_id]["thesis"][1].setText("bounce off the 20")
    assert card._trade_save_buttons[trade_id].isEnabled() is True
    card._trade_save_buttons[trade_id].click()
    assert "stop" not in check.answered_fields(store, trade_id)
    assert store.get_trade(trade_id).get("planned_stop") in (None, "")
