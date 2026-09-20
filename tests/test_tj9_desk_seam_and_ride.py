"""TJ-9 review round: the LIVE seam, and the section that has to ride.

Reviewer blockers 1 and 4, both driven through `MainWindow._show_trade_mentor_prompt`
- the slot connected to `TradeMentorService.promptDue` - rather than through the
widget the earlier tests reached directly. `grep -rn "_show_trade_mentor_prompt"
tests/` returned NOTHING before this file, which is how a dead live path shipped
green.

BLOCKER 1. `app.py` read `check.KIND_M5_TRADES` where `check` was
`trade_mentor_trade_check`; the constant lives in `trade_mentor_schedule`, and
the line sat OUTSIDE the guard - so EVERY Trade Mentor prompt raised
`AttributeError` inside a Qt slot before the section was ever reached. The
tests below call the slot itself: an exception fails them.

BLOCKER 4. The section was built only for kind `m5_trades`, and only the 09:00
slot carries that kind. A 09:00 that was away, idle, locked or expired took the
whole day's questions with it. Item 2 says the first DESK slot after an AWAY
carries the section, so the rule is now "any delivered slot of the session, if
the check is still owed, once per slot" - and an ANSWERED check brings nothing
back.

Nothing here touches a broker, the live journal or the desk: the store is built
under `tmp_path` and handed to the window by patching `journal_store.JournalStore`,
and the import service is a fake that records its calls.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj9_support import (  # noqa: E402
    REVIEWED,
    SESSION_TODAY,
    add_round_trip,
    mark_covered,
    new_store,
    slot_at,
)

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the desk seam is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")


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
    monkeypatch.setattr(journal_store_module, "JournalStore", lambda *a, **k: store)

    window = MainWindow(UiState(workspace_mode="workspace"))
    fake = _FakeImportService()
    monkeypatch.setattr(window, "_journal_import_service", lambda: fake)
    window._journal_importer = fake
    try:
        yield window, store, fake
    finally:
        window.close()


def _card(window):
    return window.trading_panel.alert_center.chart_review.mentor_card


# ---------------------------------------------------------------------------
# Blocker 1 - the live path
# ---------------------------------------------------------------------------


def test_the_nine_oclock_prompt_reaches_the_card_instead_of_raising(desk):
    """The slot `promptDue` is connected to, called for real. Before the fix
    this raised `AttributeError: module 'trade_mentor_trade_check' has no
    attribute 'KIND_M5_TRADES'` before `set_trade_check` was reached, so no
    trade section had ever appeared on a running desk."""
    window, store, _fake = desk
    add_round_trip(store, "AAPL")

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 9))

    card = _card(window)
    assert card.isVisible() or card.isVisibleTo(card.parentWidget() or card)
    assert card.trade_check_session() == SESSION_TODAY.isoformat()
    assert card._answer_inputs, "the 09:00 card carries the trade section"


def test_an_ordinary_prompt_with_nothing_owed_still_shows_and_raises_nothing(desk):
    """The plain card is the read the trader is being interrupted for. It goes
    up whether or not there is anything to ask about yesterday."""
    window, _store, _fake = desk

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 11))

    card = _card(window)
    assert card._answer_inputs == {}
    assert card.trade_check_session() == ""


def test_a_failure_building_the_trade_check_never_costs_the_prompt(desk, monkeypatch):
    """The whole section is inside one guard. A journal that cannot be read is
    a logged line, not a Qt slot that throws over the trader's chart."""
    import trade_mentor_trade_check as check

    window, store, _fake = desk
    add_round_trip(store, "AAPL")
    monkeypatch.setattr(
        check, "build_task", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("journal gone"))
    )

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 9))

    card = _card(window)
    assert card.prompt_label.text(), "the read still stands"
    assert card._answer_inputs == {}


# ---------------------------------------------------------------------------
# Blocker 4 - the ride after an absence
# ---------------------------------------------------------------------------


def test_an_away_nine_oclock_is_carried_by_the_next_desk_slot(desk):
    """The 09:00 card was never delivered (AWAY records the absence and emits
    nothing). The 10:00 slot has kind `m5`, and before the fix that meant the
    day's trades were never asked about at all."""
    window, store, _fake = desk
    add_round_trip(store, "AAPL")

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 10))

    card = _card(window)
    assert card.trade_check_session() == SESSION_TODAY.isoformat()
    assert card._answer_inputs, "the first DESK slot after the absence carries it"


def test_a_trader_who_sits_down_at_eleven_is_still_asked(desk):
    window, store, _fake = desk
    add_round_trip(store, "AAPL")

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 11))

    assert _card(window)._answer_inputs


def test_an_answered_check_does_not_come_back_on_a_later_slot(desk):
    """"Once per slot" is not "every hour forever". A session whose trades are
    all labelled brings nothing back."""
    import trade_mentor_trade_check as check

    window, store, _fake = desk
    trade_id = add_round_trip(store, "AAPL")
    store.save_trade_annotation(trade_id, setup_tags="avwap_breakout", notes="held it")
    store.save_risk_fields(trade_id, planned_stop=9.5, risk_source="manual")
    check.save_answers(
        store, trade_id, {"target": {"state": check.ANSWER_NO_FIXED_TARGET, "text": ""}}
    )

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 10))

    assert _card(window)._answer_inputs == {}
    assert _card(window).trade_check_session() == ""


def test_a_half_answered_section_is_not_rebuilt_by_the_next_slot(desk):
    """The ride is the widget staying as it is. Rebuilding it every hour would
    throw away the combos the trader had already set."""
    window, store, _fake = desk
    trade_id = add_round_trip(store, "AAPL")

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 9))
    card = _card(window)
    combo = card._answer_inputs[trade_id]["thesis"][0]
    import trade_mentor_trade_check as check

    combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 10))

    assert card._answer_inputs[trade_id]["thesis"][0] is combo
    assert combo.currentData() == check.ANSWER_NOT_REMEMBERED


def test_the_real_service_stays_silent_in_away_and_the_desk_slot_after_it_asks(
    desk, tmp_path, monkeypatch
):
    """The absence half, through the REAL service. AWAY records `away` and
    emits nothing; the 10:00 DESK poll emits a plain `m5` slot, and that slot
    is what has to carry the section."""
    import autopilot_core
    from ui.services.trade_mentor_service import TradeMentorService

    window, store, _fake = desk
    add_round_trip(store, "AAPL")

    moment = {"now": datetime(2026, 9, 14, 9, 5, tzinfo=PACIFIC)}
    service = TradeMentorService(
        clock=lambda: moment["now"],
        idle_seconds=lambda: 0.0,
        session_locked=lambda: False,
        state_path=tmp_path / "slots.json",
    )
    monkeypatch.setattr(service, "enabled", lambda: True)

    monkeypatch.setattr(autopilot_core, "read_auto_pilot_mode", lambda: "AWAY")
    assert service.poll() is None, "AWAY prompts nothing"
    assert service.slot_state("2026-09-14-0900-m5_trades")["skipped_reason"] == "away"

    monkeypatch.setattr(autopilot_core, "read_auto_pilot_mode", lambda: "DESK")
    moment["now"] = datetime(2026, 9, 14, 10, 5, tzinfo=PACIFIC)
    delivered = service.poll()
    assert delivered is not None and delivered.kind == "m5"

    window._show_trade_mentor_prompt(delivered)

    assert _card(window)._answer_inputs, "the first DESK slot after AWAY asks"


def test_a_not_ready_journal_rides_to_a_later_slot_too(desk):
    """Item 6's line is part of the section, so it rides for the same reason.
    Nothing was covered for the reviewed session here."""
    window, store, fake = desk
    with store.connection() as conn:
        conn.execute("DELETE FROM import_coverage")
    mark_covered(store, "2026-09-10")

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 11))

    card = _card(window)
    assert "fills current to 2026-09-10" in card.trade_check_label.text()
    assert card.trade_check_label.isVisibleTo(card)
    assert fake.calls, "and the one morning Questrade retry was asked for"


def test_a_journal_that_becomes_ready_after_the_nine_oclock_card_is_still_asked_about(desk):
    """Re-review blocker. The 09:00 card printed `journal not ready` and asked
    the morning retry for the fills; the statement then landed. Every later
    slot returned early because the card still had a VISIBLE LABEL on it, so
    the section never became the questions that day - and the card went on
    printing a freshness date that was no longer true.

    Only a section with ANSWER WIDGETS is protected from a rebuild. A
    one-line state has nothing to lose."""
    window, store, fake = desk
    with store.connection() as conn:
        conn.execute("DELETE FROM import_coverage")
    mark_covered(store, "2026-09-10")
    add_round_trip(store, "AAPL")

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 9))
    card = _card(window)
    assert card._answer_inputs == {}
    assert "journal not ready" in card.trade_check_label.text()
    # TJ-14B: every card now also makes ONE light pre-card pull
    # (`mentor_questions.PRE_CARD_PULL_DAYS`), so the total number of pulls
    # is no longer the number of morning RETRIES. The retry is counted by the
    # days it asks for - `MORNING_RETRY_DAYS`, which the pre-card pull never
    # uses - so this assertion says exactly what it always meant, and now
    # tells the two pulls apart instead of adding them up.
    assert fake.calls.count(check.MORNING_RETRY_DAYS) == 1, "and the one morning retry was asked for"

    # The retry lands the statement.
    mark_covered(store, REVIEWED)

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 10))

    assert card._answer_inputs, "the 10:00 card carries the QUESTIONS"
    assert "journal not ready" not in card.trade_check_label.text()
    assert card.save_answers_button.isVisibleTo(card)
    assert card.save_answers_button.isEnabled() is False, "and Save is still grey"


def test_a_journal_still_not_ready_reprints_the_current_freshness_date(desk):
    """The date is re-read every slot. A line that keeps yesterday's answer is
    a report about the desk's memory, not about the journal."""
    window, store, fake = desk
    with store.connection() as conn:
        conn.execute("DELETE FROM import_coverage")
    mark_covered(store, "2026-09-09")
    add_round_trip(store, "AAPL")

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 9))
    card = _card(window)
    assert "fills current to 2026-09-09" in card.trade_check_label.text()

    # A partial import lands one more day, and the session under review is
    # still not covered.
    mark_covered(store, "2026-09-10")
    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 10))

    assert "journal not ready" in card.trade_check_label.text()
    assert "fills current to 2026-09-10" in card.trade_check_label.text()
    assert "2026-09-09" not in card.trade_check_label.text()
    # TJ-14B: every card now also makes ONE light pre-card pull
    # (`mentor_questions.PRE_CARD_PULL_DAYS`), so the total number of pulls
    # is no longer the number of morning RETRIES. The retry is counted by the
    # days it asks for - `MORNING_RETRY_DAYS`, which the pre-card pull never
    # uses - so this assertion says exactly what it always meant, and now
    # tells the two pulls apart instead of adding them up.
    assert fake.calls.count(check.MORNING_RETRY_DAYS) == 1, "still one retry a morning"


def test_the_morning_retry_is_asked_for_once_across_two_slots(desk):
    window, store, fake = desk
    with store.connection() as conn:
        conn.execute("DELETE FROM import_coverage")

    window._show_trade_mentor_prompt(slot_at(SESSION_TODAY, 9))
    window._show_trade_mentor_prompt(slot_at(date(2026, 9, 14), 10))

    # TJ-14B: every card now also makes ONE light pre-card pull
    # (`mentor_questions.PRE_CARD_PULL_DAYS`), so the total number of pulls
    # is no longer the number of morning RETRIES. The retry is counted by the
    # days it asks for - `MORNING_RETRY_DAYS`, which the pre-card pull never
    # uses - so this assertion says exactly what it always meant, and now
    # tells the two pulls apart instead of adding them up.
    assert fake.calls.count(check.MORNING_RETRY_DAYS) == 1
