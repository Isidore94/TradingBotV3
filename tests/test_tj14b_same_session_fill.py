"""TJ-14B items 2 and 4 - a fill seen TODAY is asked about TODAY.

RED BEFORE THE FIX. On `e8c04f88` `trade_mentor_trade_check.build_task` always
reviews `previous_exchange_session(session)`, so a trade filled this morning is
invisible to the card until tomorrow; and `save_answers` stamps every payload
`recalled_after_session: True` with no `label_provenance` at all.

THE CONTRACT THESE PIN (plan.md TJ-9 AMENDED block; TJ-14 item 2)
-----------------------------------------------------------------
*"`same_session` (answered on a card the day of the fill, TJ-14 item 4)"*. The
label made before the outcome is known is worth more than the one made the next
morning, and the three ages are only worth reporting apart if the desk actually
produces the middle one.

The first test drives the REAL Qt slot `MainWindow._show_trade_mentor_prompt`
that `TradeMentorService.promptDue` is connected to - not the widget directly.
Nothing here touches a broker, the live journal or the desk.
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
    slot_at,
)

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402


class _FakeImportService:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def pull_recent_questrade(self, days: int) -> bool:
        self.calls.append(int(days))
        return True

    def shutdown(self) -> None:
        pass


@pytest.fixture()
def desk(tmp_path, monkeypatch):
    """A real `MainWindow` over a scratch journal. Never the live one."""
    import journal_store as journal_store_module
    from ui.app import MainWindow
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    mark_covered(store, SESSION.isoformat())
    monkeypatch.setattr(journal_store_module, "JournalStore", lambda *a, **k: store)

    window = MainWindow(UiState(workspace_mode="workspace"))
    fake = _FakeImportService()
    monkeypatch.setattr(window, "_journal_import_service", lambda: fake)
    window._journal_importer = fake
    card = window.trading_panel.alert_center.chart_review.mentor_card
    # The card's own clock, so "the day of the fill" is decidable rather than
    # whatever wall clock the test happens to run on.
    card._clock = lambda: pacific(SESSION, 11)
    try:
        yield window, store, card
    finally:
        window.close()


def test_a_fill_seen_today_is_asked_about_on_todays_card(desk):
    """The trade closed at 09:05 this morning. The card at 11:00 asks about it.

    Before TJ-14B the card only ever listed the PREVIOUS session's trades, so
    the label the trader could still make before the outcome was known was
    never offered - and every live label was `recalled_after`.
    """
    window, store, card = desk
    today = SESSION.isoformat()
    trade_id = add_round_trip(store, "AAPL", day=today, entry_hour=7)

    window._show_trade_mentor_prompt(slot_at(SESSION, 11))

    asked = set(card._answer_inputs) | {
        key for key in getattr(card, "_trade_questions", {})
    }
    assert trade_id in asked, f"today's fill was not asked about: {sorted(asked)}"


def test_yesterdays_trades_are_still_asked_beside_todays(desk):
    """The same-session question is ADDITIVE. TJ-9's forced list does not
    shrink because a fill landed this morning."""
    window, store, card = desk
    yesterday = add_round_trip(store, "MSFT", day=REVIEWED, entry_hour=7)
    today = add_round_trip(store, "AAPL", day=SESSION.isoformat(), entry_hour=7)

    window._show_trade_mentor_prompt(slot_at(SESSION, 9))

    asked = set(card._answer_inputs) | set(getattr(card, "_trade_questions", {}))
    assert {yesterday, today} <= asked


def test_an_answer_given_the_day_of_the_fill_is_labelled_same_session(tmp_path):
    """The WRITER decides the provenance, from the stamps.

    `trade_origin.label_provenance` already answers `same_session` for a
    confirm made on the fill's own session; the recalled-field rows written by
    `save_answers` still say `recalled_after_session: True` for every one of
    them, so a same-session answer is indistinguishable from a next-morning one
    the moment it is stored.
    """
    import trade_mentor_trade_check as check
    import trade_origin

    store = new_store(tmp_path)
    today = SESSION.isoformat()
    mark_covered(store, today)
    trade_id = add_round_trip(store, "AAPL", day=today, entry_hour=7)

    rows = check.save_answers(
        store,
        trade_id,
        {"stop": {"state": check.ANSWER_NOT_REMEMBERED, "text": ""}},
        now=pacific(SESSION, 11),
    )

    assert rows, "nothing was written"
    payload = rows[0].get("payload") or {}
    assert payload.get("label_provenance") == trade_origin.SAME_SESSION
    assert payload.get("recalled_after_session") is False


def test_an_answer_given_the_next_morning_is_still_labelled_recalled_after(tmp_path):
    """The other side of the same rule. A label made the morning after already
    knows how the trade ended, and must keep saying so."""
    import trade_mentor_trade_check as check
    import trade_origin

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAPL", day=REVIEWED, entry_hour=7)

    rows = check.save_answers(
        store,
        trade_id,
        {"stop": {"state": check.ANSWER_NOT_REMEMBERED, "text": ""}},
        now=pacific(SESSION, 9),
    )

    payload = rows[0].get("payload") or {}
    assert payload.get("label_provenance") == trade_origin.RECALLED_AFTER
    assert payload.get("recalled_after_session") is True
