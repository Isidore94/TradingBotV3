"""P8 B4: the status-bar "Inputs: N trades missing stop/setup" chip.

Hidden at 0, the count at N, and one click opens the Mentor on the OLDEST such
trade at its first missing question; the chip re-reads after the Mentor saves.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tj14b_support import add_round_trip, new_store, pacific  # noqa: E402

pytestmark = pytest.mark.qt

TODAY = date(2026, 9, 25)


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6", reason="the chip is Qt")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(qapp, predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _two_trades(tmp_path):
    """OLD (setup confirmed, no stop) and NEW (no stop, no setup)."""
    store = new_store(tmp_path)
    old = add_round_trip(store, "AAA", day="2026-09-10", entry_hour=7)
    new = add_round_trip(store, "BBB", day="2026-09-11", entry_hour=7)
    store.save_trade_annotation(old, setup_tags="vwap_bounce", notes="")
    return store, old, new


def _result(count):
    import journal_missing_inputs as mi

    trades = [
        {"trade_id": f"T{i}", "symbol": f"S{i}", "opened_at": f"2026-09-1{i}T07:31:00"}
        for i in range(count)
    ]
    return mi.missing_inputs(trades, {}, today=TODAY)


def test_the_chip_is_hidden_at_zero_and_shows_the_count_at_n(qapp):
    from ui.widgets.missing_inputs_chip import MissingInputsChip

    chip = MissingInputsChip(loader=lambda: _result(0))
    chip.apply(_result(0))
    assert chip.isHidden()
    assert chip.text() == ""

    chip.apply(_result(3))
    assert not chip.isHidden()
    assert chip.text() == "Inputs: 3 trades missing stop/setup"
    assert "S0" in chip.toolTip()

    chip.apply(_result(0))
    assert chip.isHidden()


def test_the_chip_reads_off_the_qt_thread(qapp, tmp_path):
    import journal_missing_inputs as mi
    from ui.widgets.missing_inputs_chip import MissingInputsChip

    store, old, _new = _two_trades(tmp_path)
    seen = []

    def loader():
        seen.append(threading.get_ident())
        return mi.load_chip(store, today=TODAY)

    chip = MissingInputsChip(loader=loader)
    chip.refresh()
    assert _wait(qapp, lambda: bool(chip.text()))
    chip.shutdown()
    assert seen and seen[0] != threading.get_ident()
    assert chip.text() == "Inputs: 2 trades missing stop/setup"
    assert chip.question().trade_id == old


def test_a_click_opens_the_mentor_on_the_oldest_trade_at_its_first_gap(qapp, tmp_path):
    import journal_missing_inputs as mi
    from ui.widgets.missing_inputs_chip import MissingInputsChip
    from ui.widgets.trade_mentor_card import TradeMentorCard

    store, old, new = _two_trades(tmp_path)
    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card._clock = lambda: pacific(TODAY, 10)
    chip = MissingInputsChip(loader=lambda: mi.load_chip(store, today=TODAY))
    chip.openTradeRequested.connect(lambda question: card.open_on_trade(question, store=store))
    chip.apply(mi.load_chip(store, today=TODAY))

    chip.click()

    assert list(card._trade_blocks) == [old]
    assert new not in card._trade_blocks
    assert card.positioned_on() == (old, "stop")
    assert card.positioned_widget() is card._answer_inputs[old]["stop"][1]
    assert not card.isHidden()
    # The Mentor's questions and order are its own: stop first.
    assert tuple(card._trade_questions[old].missing)[0] == "stop"


def test_a_click_adds_the_trade_beside_rows_already_on_the_card(qapp, tmp_path):
    """A card that is already up keeps its slot and its rows."""
    import journal_missing_inputs as mi
    from ui.widgets.trade_mentor_card import TradeMentorCard

    store, old, new = _two_trades(tmp_path)
    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card._clock = lambda: pacific(TODAY, 10)
    first = mi.question_for(store, {"trade_id": new, "trade_date": "2026-09-11"})
    card.open_on_trade(first, store=store)
    slot = card._slot
    card._answer_inputs[new]["thesis"][1].setText("typed before the click")

    card.open_on_trade(mi.load_chip(store, today=TODAY)["question"], store=store)

    assert card._slot is slot
    assert set(card._trade_blocks) == {old, new}
    assert card._answer_inputs[new]["thesis"][1].text() == "typed before the click"
    assert card.positioned_on() == (old, "stop")


def test_the_chip_updates_after_the_mentor_saves(qapp, tmp_path):
    import journal_missing_inputs as mi
    from ui.widgets.missing_inputs_chip import MissingInputsChip
    from ui.widgets.trade_mentor_card import TradeMentorCard

    store, old, _new = _two_trades(tmp_path)
    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card._clock = lambda: pacific(TODAY, 10)
    chip = MissingInputsChip(loader=lambda: mi.load_chip(store, today=TODAY))
    chip.openTradeRequested.connect(lambda question: card.open_on_trade(question, store=store))
    card.inputsFiled.connect(chip.request_refresh)
    chip.apply(mi.load_chip(store, today=TODAY))
    assert chip.text() == "Inputs: 2 trades missing stop/setup"

    chip.click()
    card._answer_inputs[old]["stop"][1].setText("9.50 under the low")
    card._trade_save_buttons[old].click()

    assert _wait(qapp, lambda: chip.text() == "Inputs: 1 trade missing stop/setup")
    chip.shutdown()
    assert chip.question().trade_id != old


def test_the_desk_puts_the_chip_on_the_status_bar_and_a_click_opens_its_mentor(
    qapp, tmp_path, monkeypatch
):
    """The real window: the chip sits on the status bar, hidden until read; the
    journal and Mentor signals ask it to re-read (coalesced); a click opens the
    desk's own Mentor popup on the oldest trade at its stop question."""
    import journal_missing_inputs as mi
    import journal_store
    from ui.app import MainWindow
    from ui.state import UiState
    from ui.widgets.missing_inputs_chip import MissingInputsChip

    store, old, _new = _two_trades(tmp_path)
    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        chip = window.missing_inputs_chip
        assert chip in window.statusBar().findChildren(MissingInputsChip)
        assert chip.isHidden()

        review = window.trading_panel.alert_center.chart_review
        review.mentor_card._clock = lambda: pacific(TODAY, 10)
        review.mentor_card.inputsFiled.emit()
        assert chip._coalescer.is_pending()
        chip._coalescer.cancel()
        window.journal_panel.trades_tab.dataChanged.emit()
        assert chip._coalescer.is_pending()
        chip._coalescer.cancel()

        monkeypatch.setattr(journal_store, "JournalStore", lambda *_a, **_k: store)
        chip.apply(mi.load_chip(store, today=TODAY))
        assert not chip.isHidden()
        chip.click()

        assert list(review.mentor_card._trade_blocks) == [old]
        assert review.mentor_card.positioned_on() == (old, "stop")
        assert review.mentor_popup.isVisible()
    finally:
        window.close()
