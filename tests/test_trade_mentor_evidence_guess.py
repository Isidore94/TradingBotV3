"""The 09:00 Trade Mentor offers a setup for every trade that has any evidence.

A third guess lane after the claimed like and the provisional tag: the best
stored auto-tag candidate that names a setup. A guess nobody clicks writes
nothing; the click writes a CONFIRMED tag through the Journal's own writer.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

REVIEWED = "2026-09-11"


def _store(tmp_path):
    from journal_store import JournalStore

    return JournalStore(tmp_path / "journal.sqlite3")


def _trade(store, trade_id, symbol="AAA"):
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO trades(trade_id, broker, account_number, symbol, direction, status,
                opened_at, closed_at, trade_date, updated_at)
            VALUES(?, 'QUESTRADE', '1', ?, 'LONG', 'OPEN', ?, '', ?, ?)
            """,
            (trade_id, symbol, f"{REVIEWED}T10:30:00-04:00", REVIEWED, REVIEWED),
        )


def _candidate(store, trade_id, tag, confidence, source, rationale="seeded"):
    with store.connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO auto_tag_candidates(trade_id, tag, confidence, source,
                rationale, created_at)
            VALUES(?, ?, ?, ?, ?, '2026-09-11T00:00:00')
            """,
            (trade_id, tag, float(confidence), source, rationale),
        )


def _question(store, trade_id):
    import trade_mentor_trade_check as check

    questions = check.questions_for_session(store, REVIEWED)
    return next(item for item in questions if item.trade_id == trade_id)


def test_a_trade_with_only_below_threshold_evidence_still_gets_a_guess(tmp_path):
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _trade(store, "T1")
    store.mark_tags_needing_review("T1")
    _candidate(
        store,
        "T1",
        "new_5d_high",
        0.62,
        "evidence:alert_fired",
        "alert fired new_5d_high on AAA 2 day(s) before entry",
    )
    _candidate(store, "T1", "midday", 1.0, "trade_shape:entry_time")

    question = _question(store, "T1")

    assert "setup" in question.missing
    assert question.setup_guess == "new_5d_high"
    assert question.setup_guess_lane == check.LANE_EVIDENCE
    assert "2 day(s) before entry" in question.setup_guess_evidence
    assert "(0.62)" in question.setup_guess_evidence


def test_a_scanner_row_from_the_entry_day_is_never_offered(tmp_path):
    """It has a date and no time, so it may have been logged after the fill."""
    store = _store(tmp_path)
    _trade(store, "T1")
    store.mark_tags_needing_review("T1")
    _candidate(
        store, "T1", "pullback_long", 0.5, "setup_tracker",
        f"setup_tracker; AAA; context {REVIEWED}; pullback_long",
    )

    question = _question(store, "T1")

    assert question.setup_guess == ""


def test_shape_link_and_rejection_candidates_are_never_offered(tmp_path):
    store = _store(tmp_path)
    _trade(store, "T1")
    _candidate(store, "T1", "midday", 1.0, "trade_shape:entry_time")
    _candidate(store, "T1", "link:review:add_focus", 0.95, "trader_capture:review:add_focus")
    _candidate(store, "T1", "vetoed:too_extended_from_base", 0.95, "trader_capture:veto")

    question = _question(store, "T1")

    assert question.setup_guess == ""
    assert question.setup_guess_lane == ""


def test_the_provisional_tag_still_leads_and_carries_its_evidence(tmp_path):
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _trade(store, "T1")
    _candidate(store, "T1", "vwap_bounce", 0.78, "evidence:alert_fired", "alert fired vwap_bounce")
    _candidate(store, "T1", "new_5d_high", 0.62, "evidence:alert_fired", "older flag")
    assert store.apply_provisional_tags("T1", "vwap_bounce")

    question = _question(store, "T1")

    assert question.setup_guess == "vwap_bounce"
    assert question.setup_guess_lane == check.LANE_PROVISIONAL
    assert question.setup_guess_evidence == "alert fired vwap_bounce (0.78)"


def test_a_guess_nobody_clicks_writes_nothing_and_the_click_confirms(tmp_path):
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _trade(store, "T1")
    store.mark_tags_needing_review("T1")
    _candidate(store, "T1", "new_5d_high", 0.62, "evidence:alert_fired")

    question = _question(store, "T1")
    before = store.annotation_state("T1")
    assert before["setup_tags"] == ""
    assert before["tag_status"] == "needs_review"

    result = check.confirm_setup(store, question)

    assert result["ok"] is True
    after = store.annotation_state("T1")
    assert after["setup_tags"] == "new_5d_high"
    assert after["tag_status"] == "confirmed"
    # Confirmed means answered: the next card offers no guess for it.
    assert "setup" not in _question_or_none_missing(store, "T1")


def _question_or_none_missing(store, trade_id):
    import trade_mentor_trade_check as check

    for item in check.questions_for_session(store, REVIEWED) or ():
        if item.trade_id == trade_id:
            return item.missing
    return ()


def test_the_card_tooltip_says_why(tmp_path):
    import os

    import pytest

    pytest.importorskip("PySide6", reason="the Mentor card is Qt")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    QApplication.instance() or QApplication([])
    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    question = check.TradeQuestion(
        "T1",
        "AAA",
        "LONG",
        ("setup",),
        setup_guess="new_5d_high",
        setup_guess_lane=check.LANE_EVIDENCE,
        setup_guess_evidence="alert fired new_5d_high on AAA 2 day(s) before entry (0.62)",
    )
    card._add_setup_confirm(question)

    button = card.setup_confirm_button("T1")
    assert button is not None
    assert "2 day(s) before entry" in button.toolTip()
    assert "from evidence" in button.toolTip()
    assert card.setup_choice_box("T1").currentText() == "new_5d_high"


def test_a_trade_with_no_guess_still_gets_the_setup_list(tmp_path):
    """Trader 2026-09-25: every trade missing a setup gets the list, not only
    the ones the machine could guess. With no guess it opens on a blank pick
    and the button stays off until a real name is chosen."""
    import os

    import pytest

    pytest.importorskip("PySide6", reason="the Mentor card is Qt")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    QApplication.instance() or QApplication([])
    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    question = check.TradeQuestion("T1", "AAA", "LONG", ("setup",))
    card._add_setup_confirm(question)

    box = card.setup_choice_box("T1")
    button = card.setup_confirm_button("T1")
    assert box is not None and button is not None
    assert box.currentData() == ""
    assert not button.isEnabled()
    box.setCurrentIndex(1)
    assert box.currentData()
    assert button.isEnabled()
    box.setCurrentIndex(0)
    assert not button.isEnabled()


def test_a_trade_with_its_setup_answered_gets_no_list(tmp_path):
    import os

    import pytest

    pytest.importorskip("PySide6", reason="the Mentor card is Qt")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    QApplication.instance() or QApplication([])
    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card._add_setup_confirm(check.TradeQuestion("T1", "AAA", "LONG", ("thesis",)))

    assert card.setup_choice_box("T1") is None
