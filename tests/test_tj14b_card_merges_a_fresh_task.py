"""TJ-14B fix round - the card MERGES a fresh task; it never chooses.

RED BEFORE THE FIX (`05ec3d06`). The 09:00 not-ready card started drawing
answer widgets for the fills the desk had already SEEN today, so from that
moment `TradeMentorCard.open_answers_session()` answered with the session - and
`MainWindow._show_trade_mentor_prompt` returned before `set_trade_check` on
every later slot of the day, because only the 09:00 slot carries kind
`m5_trades`. The consequences on the live card:

* the REVIEWED session's trades were never asked about that day, however long
  after the statement landed;
* the heading went on saying `journal not ready` and naming a freshness date
  that was no longer true.

The early return protected the trader's half-set widgets, and nothing else was
allowed to move because of it. Now every delivered slot builds the fresh task
and the CARD merges it: a row already there keeps its exact widget objects and
their values, a trade the task names and the card does not is added, a row that
is no longer owed goes, the heading is always rewritten, and the Save gate is
recomputed over every row now on the card.

Every test here drives the REAL Qt slot `MainWindow._show_trade_mentor_prompt`
that `TradeMentorService.promptDue` is connected to, offscreen, over a scratch
journal under `tmp_path`. Nothing here touches a broker, the live journal or
the desk.
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

# TJ-14B gives the desk's day-time journal pulls ONE persisted per-day tally;
# this gives each test in this module its own day of it. See the module.
from tj14b_desk_isolation import fresh_mentor_pull_tally  # noqa: E402,F401
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

#: The day before the reviewed session - the freshness date a not-ready card
#: prints, and the one it must STOP printing once the statement lands.
EARLIER = "2026-09-10"


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
    """A real `MainWindow` over a scratch journal, the statement NOT landed.

    Only `EARLIER` is covered, so the reviewed session (2026-09-11) is not
    ready and the 09:00 card is the not-ready one. A test that wants the ready
    day covers the reviewed session itself.
    """
    import journal_store as journal_store_module
    from ui.app import MainWindow
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    store = new_store(tmp_path)
    mark_covered(store, EARLIER)
    monkeypatch.setattr(journal_store_module, "JournalStore", lambda *a, **k: store)

    window = MainWindow(UiState(workspace_mode="workspace"))
    fake = _FakeImportService()
    monkeypatch.setattr(window, "_journal_import_service", lambda: fake)
    window._journal_importer = fake
    card = window.trading_panel.alert_center.chart_review.mentor_card
    card._clock = lambda: pacific(SESSION, 11)
    try:
        yield window, store, card
    finally:
        window.close()


def _answer_every_field(card, trade_id: str) -> None:
    """Give every open field of ONE trade a complete answer state."""
    import trade_mentor_trade_check as check

    for combo, _text in card._answer_inputs[trade_id].values():
        combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))


# ---------------------------------------------------------------------------
# (a) the reviewer's exact sequence
# ---------------------------------------------------------------------------


def test_the_reviewed_sessions_trades_are_added_once_the_statement_lands(desk):
    """09:00 asks about today's AAA alone; the statement lands; 11:00 adds BBB.

    Before the fix the 11:00 slot returned early - the card held answer widgets
    - so BBB was never asked about that day and the heading went on saying
    `journal not ready - fills current to 2026-09-10`.
    """
    window, store, card = desk
    today = add_round_trip(store, "AAA", day=SESSION.isoformat(), entry_hour=7)
    yesterday = add_round_trip(store, "BBB", day=REVIEWED, entry_hour=7)

    window._show_trade_mentor_prompt(slot_at(SESSION, 9))

    assert list(card._answer_inputs) == [today], "the not-ready card asks today's fill"
    nine = card.trade_check_label.text()
    assert "journal not ready" in nine
    assert f"fills current to {EARLIER}" in nine
    assert "seen TODAY" in nine

    # The morning retry lands the reviewed session's statement.
    mark_covered(store, REVIEWED)

    window._show_trade_mentor_prompt(slot_at(SESSION, 11))

    assert list(card._answer_inputs) == [today, yesterday], (
        "the reviewed session's trade joins the row already on the card"
    )
    eleven = card.trade_check_label.text()
    assert "journal not ready" not in eleven, "the not-ready line is gone"
    assert EARLIER not in eleven, "and so is the freshness date that went stale"
    assert REVIEWED in eleven
    assert card.save_answers_button.isVisibleTo(card)
    assert card.save_answers_button.isEnabled() is False, "nothing is answered yet"


# ---------------------------------------------------------------------------
# (b) the trader's half-set widgets survive the merge
# ---------------------------------------------------------------------------


def test_a_half_answered_row_keeps_its_widgets_when_another_trade_is_added(desk):
    """The whole reason the host used to return early. The merge has to give
    the same protection WITHOUT freezing the words above the rows."""
    import trade_mentor_trade_check as check

    window, store, card = desk
    today = add_round_trip(store, "AAA", day=SESSION.isoformat(), entry_hour=7)
    yesterday = add_round_trip(store, "BBB", day=REVIEWED, entry_hour=7)

    window._show_trade_mentor_prompt(slot_at(SESSION, 9))

    field = sorted(card._answer_inputs[today])[0]
    combo, text_input = card._answer_inputs[today][field]
    combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))
    text_input.setText("I was watching the open")

    mark_covered(store, REVIEWED)
    window._show_trade_mentor_prompt(slot_at(SESSION, 11))

    assert card._answer_inputs[today][field][0] is combo, "the SAME combo object"
    assert card._answer_inputs[today][field][1] is text_input
    assert combo.currentData() == check.ANSWER_NOT_REMEMBERED
    assert text_input.text() == "I was watching the open"

    # The gate is recomputed over ALL the rows now on the card.
    _answer_every_field(card, today)
    assert card.save_answers_button.isEnabled() is False, "the added trade is still open"
    # LEAD AMENDMENT 2026-09-21 (TJ-9E, the trader's own request): a round-trip
    # trade now also carries ONE forced EXIT box - "Why did you exit? What did
    # you feel? What were you watching?" - so Save waits on it as it waits on
    # the entry fields. One click answers it; what this test pins is unchanged.
    for _trade in (today, yesterday):
        if card.exit_note_box(_trade) is not None:
            card.set_exit_answer_state(_trade, check.ANSWER_NOT_REMEMBERED)
    _answer_every_field(card, yesterday)
    assert card.save_answers_button.isEnabled() is True


# ---------------------------------------------------------------------------
# (c) a fill seen later in the day joins a READY card
# ---------------------------------------------------------------------------


def test_a_fill_seen_later_in_the_day_joins_the_rows_already_there(desk):
    """The other direction: the statement had landed at 09:00 and the trader
    traded afterwards. The new fill is asked about on the next card."""
    import trade_mentor_trade_check as check

    window, store, card = desk
    mark_covered(store, REVIEWED)
    yesterday = add_round_trip(store, "BBB", day=REVIEWED, entry_hour=7)

    window._show_trade_mentor_prompt(slot_at(SESSION, 9))
    assert list(card._answer_inputs) == [yesterday]
    field = sorted(card._answer_inputs[yesterday])[0]
    combo = card._answer_inputs[yesterday][field][0]
    combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))

    # The trader trades at 10:40 and the desk has SEEN the fill by 11:00.
    today = add_round_trip(store, "CCC", day=SESSION.isoformat(), entry_hour=8)

    window._show_trade_mentor_prompt(slot_at(SESSION, 11))

    assert list(card._answer_inputs) == [yesterday, today]
    assert card._answer_inputs[yesterday][field][0] is combo
    assert combo.currentData() == check.ANSWER_NOT_REMEMBERED


# ---------------------------------------------------------------------------
# (d) every block says which session it is from
# ---------------------------------------------------------------------------


def test_every_block_names_the_session_its_trade_is_from(desk):
    """`same_session` versus `recalled_after` is the whole point of asking on
    the day, and a row that only says the symbol cannot show it."""
    window, store, card = desk
    today = add_round_trip(store, "AAA", day=SESSION.isoformat(), entry_hour=7)
    yesterday = add_round_trip(store, "BBB", day=REVIEWED, entry_hour=7)
    mark_covered(store, REVIEWED)

    window._show_trade_mentor_prompt(slot_at(SESSION, 9))

    today_heading = card.trade_heading_text(today)
    yesterday_heading = card.trade_heading_text(yesterday)
    assert today_heading.startswith("AAA LONG")
    assert "today" in today_heading
    assert SESSION.isoformat() in today_heading
    assert yesterday_heading.startswith("BBB LONG")
    assert REVIEWED in yesterday_heading
    assert "today" not in yesterday_heading


# ---------------------------------------------------------------------------
# (e) the nothing-seen not-ready card is untouched
# ---------------------------------------------------------------------------


def test_a_not_ready_card_with_nothing_seen_today_still_asks_nothing(desk):
    """No box, no Save, and the line that has always been there. The reworded
    heading applies only to the card that DRAWS today's fills."""
    import trade_mentor_trade_check as check

    window, store, card = desk
    add_round_trip(store, "BBB", day=REVIEWED, entry_hour=7)

    window._show_trade_mentor_prompt(slot_at(SESSION, 9))

    text = card.trade_check_label.text()
    assert card._answer_inputs == {}
    assert text == (
        f"Yesterday's trades ({REVIEWED}): {check.REASON_NOT_READY} - "
        f"fills current to {EARLIER}. The broker statement has not landed, so "
        "nothing is asked yet; this comes back on the next card. The day pull "
        "is Questrade only - IBKR has no day leg."
    )
    assert card.trade_check_label.isVisibleTo(card)
    assert card.trade_check_box.isVisibleTo(card) is False
    assert card.save_answers_button.isVisibleTo(card) is False
