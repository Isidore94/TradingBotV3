"""S16 item 1: the trader-authored structural regime journal.

The trader: "The market has different regimes ... what's important is KNOWING the
market regime and then having setups you KNOW work in it." The regime is typed by
the trader, append-only, never inferred and never back-edited.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 9, 28, 9, 0, tzinfo=PACIFIC)


def _store(tmp_path):
    from journal_store import JournalStore

    return JournalStore(tmp_path / "trade_journal.sqlite3")


def _three_past(store):
    rows = []
    for start, regime in (
        ("2026-03-01", "bull_run"),
        ("2026-06-01", "weekly_hh_then_compression"),
        ("2026-08-01", "bear_channel_lower_highs"),
    ):
        rows.append(store.append_structural_regime(start_date=start, regime=regime, entered_at=NOW))
    return rows


# -- the store ---------------------------------------------------------------
def test_the_table_sits_beside_regimes_in_the_journal(tmp_path):
    store = _store(tmp_path)
    with store.connection() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"regimes", "structural_regime"} <= tables


def test_segments_are_append_only(tmp_path):
    store = _store(tmp_path)
    first = store.append_structural_regime(
        start_date="2026-08-01", regime="bear_channel_lower_highs",
        structure_note="weekly HH, daily LH/LL channel", entered_at=NOW,
    )
    assert first["source"] == "trader"
    assert first["entered_at"] == NOW.isoformat()
    assert datetime.fromisoformat(first["entered_at"]).tzinfo is not None
    with store.connection() as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE structural_regime SET regime = 'range'")
    with store.connection() as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("DELETE FROM structural_regime")
    assert store.list_structural_regime() == [first]


def test_a_correction_is_a_new_row_that_supersedes(tmp_path):
    import structural_regime

    store = _store(tmp_path)
    wrong = store.append_structural_regime(start_date="2026-08-01", regime="range", entered_at=NOW)
    fixed = store.append_structural_regime(
        start_date="2026-08-03", regime="bear_channel_lower_highs",
        entered_at=NOW, supersedes=wrong["segment_id"],
    )
    rows = store.list_structural_regime()
    assert [row["segment_id"] for row in rows] == [wrong["segment_id"], fixed["segment_id"]]
    assert rows[0]["regime"] == "range", "the old row is never edited"
    timeline = structural_regime.effective_segments(rows)
    assert [(seg["start_date"], seg["regime"]) for seg in timeline] == [("2026-08-03", "bear_channel_lower_highs")]
    assert structural_regime.regime_at(rows, "2026-08-02") is None


def test_a_later_row_on_the_same_start_date_replaces_the_earlier(tmp_path):
    import structural_regime

    store = _store(tmp_path)
    store.append_structural_regime(start_date="2026-08-01", regime="range", entered_at=NOW)
    store.append_structural_regime(start_date="2026-08-01", regime="bear_channel_lower_highs", entered_at=NOW)
    found = structural_regime.regime_at(store.list_structural_regime(), "2026-08-10")
    assert found["regime"] == "bear_channel_lower_highs"


def test_the_store_refuses_a_bad_regime_date_or_naive_stamp(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.append_structural_regime(start_date="2026-08-01", regime="melting_up", entered_at=NOW)
    with pytest.raises(ValueError):
        store.append_structural_regime(start_date="not a date", regime="range", entered_at=NOW)
    with pytest.raises(ValueError):
        store.append_structural_regime(start_date="2026-08-01", regime="range", entered_at=datetime(2026, 9, 28))
    with pytest.raises(ValueError):
        store.append_structural_regime(start_date="2026-08-01", regime="range", entered_at=NOW, supersedes=999)
    assert store.list_structural_regime() == []


def test_a_failed_write_raises(tmp_path):
    store = _store(tmp_path)
    path = tmp_path / "trade_journal.sqlite3"
    os.chmod(path, stat.S_IREAD)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            store.append_structural_regime(start_date="2026-08-01", regime="range", entered_at=NOW)
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


# -- the pure readers --------------------------------------------------------
def test_regime_at_boundaries_and_day_count(tmp_path):
    import structural_regime

    rows = _three_past(_store(tmp_path))
    assert structural_regime.regime_at(rows, "2026-05-31")["regime"] == "bull_run"
    assert structural_regime.regime_at(rows, "2026-05-31")["end_date"] == "2026-05-31"
    june = structural_regime.regime_at(rows, date(2026, 6, 1))
    assert june["regime"] == "weekly_hh_then_compression"
    assert june["day_count"] == 1
    assert structural_regime.regime_at(rows, "2026-07-31")["regime"] == "weekly_hh_then_compression"
    august = structural_regime.regime_at(rows, "2026-08-01")
    assert august["regime"] == "bear_channel_lower_highs"
    assert august["end_date"] == ""
    current = structural_regime.current_regime(rows, "2026-09-26")
    assert current["regime"] == "bear_channel_lower_highs"
    assert current["day_count"] == 57
    # 2026-08-01 is a Saturday; the first session is Monday 08-03.
    assert structural_regime.regime_at(rows, "2026-08-03")["session_count"] == 1


def test_the_regime_is_unknown_before_the_first_segment(tmp_path):
    import structural_regime

    assert structural_regime.current_regime([], "2026-09-26") is None
    rows = _three_past(_store(tmp_path))
    assert structural_regime.regime_at(rows, "2026-02-28") is None


def test_back_to_back_segments_of_one_regime_are_one_run(tmp_path):
    import structural_regime

    store = _store(tmp_path)
    store.append_structural_regime(start_date="2026-09-28", regime="bear_channel_lower_highs", entered_at=NOW)
    store.append_structural_regime(
        start_date="2026-08-01", regime="bear_channel_lower_highs", structure_note="LH/LL", entered_at=NOW,
    )
    found = structural_regime.regime_at(store.list_structural_regime(), "2026-09-28")
    assert found["start_date"] == "2026-08-01"
    assert found["day_count"] == 59


# -- the prefills --------------------------------------------------------------
def test_the_three_prefills_wait_until_confirmed(tmp_path):
    import structural_regime

    store = _store(tmp_path)
    pending = structural_regime.pending_prefills(store.list_structural_regime())
    assert [(p.regime, p.start_date) for p in pending] == [
        ("bull_run", "2026-03-01"),
        ("weekly_hh_then_compression", "2026-06-01"),
        ("bear_channel_lower_highs", "2026-08-01"),
    ]
    store.append_structural_regime(start_date="2026-03-02", regime="bull_run", entered_at=NOW)
    assert [p.regime for p in structural_regime.pending_prefills(store.list_structural_regime())] == [
        "weekly_hh_then_compression", "bear_channel_lower_highs",
    ]


def _qt():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _drain(app):
    from PySide6.QtCore import QThreadPool

    for _ in range(4):
        QThreadPool.globalInstance().waitForDone(5000)
        for _ in range(20):
            app.processEvents()


def _card(tmp_path, store):
    from ui.widgets.trade_mentor_card import TradeMentorCard

    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json", clock=lambda: NOW)
    card.refresh_regime_lane(store)
    return card


def test_a_prefill_writes_nothing_until_the_trader_confirms(tmp_path):
    from PySide6.QtCore import QDate

    app = _qt()
    store = _store(tmp_path)
    card = _card(tmp_path, store)
    _drain(app)
    dialog = card.open_regime_dialog()
    _drain(app)
    assert dialog.past_box.isVisibleTo(dialog)
    assert store.list_structural_regime() == [], "showing the prefills writes nothing"

    key = "2026-08_bear_channel_lower_highs"
    dialog.prefill_start_edit(key).setDate(QDate(2026, 8, 3))
    dialog.prefill_confirm_button(key).click()
    _drain(app)

    rows = store.list_structural_regime()
    assert [(row["start_date"], row["regime"], row["source"]) for row in rows] == [
        ("2026-08-03", "bear_channel_lower_highs", "trader"),
    ]
    assert dialog.prefill_confirm_button(key) is None, "a confirmed prefill leaves the list"
    assert card.regime_lane()["current"]["regime"] == "bear_channel_lower_highs"
    assert "bear channel" in dialog.current_label.text()


def test_the_card_reports_a_failed_regime_write(tmp_path):
    app = _qt()
    store = _store(tmp_path)
    card = _card(tmp_path, store)
    _drain(app)
    dialog = card.open_regime_dialog()
    path = tmp_path / "trade_journal.sqlite3"
    os.chmod(path, stat.S_IREAD)
    try:
        dialog.prefill_confirm_button("2026-03_bull_run").click()
        _drain(app)
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    assert "NOT stored" in card.status_label.text()
    assert dialog.prefill_confirm_button("2026-03_bull_run").isEnabled()
    assert store.list_structural_regime() == []


# -- the Mentor question -------------------------------------------------------
def _state(lane, session="2026-09-28", answered=None):
    return {"session": session, "structural_regime": lane, "answered": answered or {}}


def test_no_regime_question_before_the_lane_is_read():
    import mentor_questions

    asked = mentor_questions.pending(_state(None), slot=None).asked
    assert not [s for s in asked if s.kind == "structural_regime"]


def test_the_weekly_question_asks_still_current_and_a_still_keeps_the_run(tmp_path):
    import mentor_questions
    import structural_regime

    store = _store(tmp_path)
    _three_past(store)
    lane = structural_regime.load_lane(store, "2026-09-28")
    asked = [s for s in mentor_questions.pending(_state(lane), slot=None).asked if s.kind == "structural_regime"]
    assert len(asked) == 1
    subject = asked[0]
    assert subject.subject_id == "2026-W40"
    assert subject.options[0] == "still_bear_channel_lower_highs"
    assert "range" in subject.options and mentor_questions.STOP_ASKING in subject.options
    assert "Still bear channel" in subject.prompt

    later = datetime(2026, 9, 28, 10, 0, tzinfo=PACIFIC)
    outcome = mentor_questions.record_answer(
        subject, {"state": "still_bear_channel_lower_highs"}, store=store, now=later
    )
    assert outcome["ok"] is True
    rows = store.list_structural_regime()
    assert len(rows) == 4
    assert rows[-1]["supersedes"] == rows[2]["segment_id"]
    current = structural_regime.current_regime(rows, "2026-09-28")
    assert (current["start_date"], current["day_count"]) == ("2026-08-01", 59)

    lane = structural_regime.load_lane(store, "2026-09-30")
    same_week = mentor_questions.pending(_state(lane, "2026-09-30", lane["answered"]), slot=None).asked
    assert not [s for s in same_week if s.kind == "structural_regime"]
    lane = structural_regime.load_lane(store, "2026-10-05")
    next_week = mentor_questions.pending(_state(lane, "2026-10-05", lane["answered"]), slot=None).asked
    assert [s.subject_id for s in next_week if s.kind == "structural_regime"] == ["2026-W41"]


def test_a_changed_regime_answer_appends_a_segment_starting_on_the_card_session(tmp_path):
    import mentor_questions
    import structural_regime

    store = _store(tmp_path)
    lane = structural_regime.load_lane(store, "2026-09-28")
    assert lane["current"] is None
    subject = [s for s in mentor_questions.pending(_state(lane), slot=None).asked if s.kind == "structural_regime"][0]
    assert "past regimes" in subject.prompt
    mentor_questions.record_answer(subject, {"state": "range"}, store=store, now=NOW)
    current = structural_regime.current_regime(store.list_structural_regime(), "2026-09-28")
    assert (current["regime"], current["start_date"], current["day_count"]) == ("range", "2026-09-28", 1)


def test_a_failed_regime_answer_raises_through_the_mentor(tmp_path):
    import mentor_questions
    import structural_regime

    store = _store(tmp_path)
    lane = structural_regime.load_lane(store, "2026-09-28")
    subject = [s for s in mentor_questions.pending(_state(lane), slot=None).asked if s.kind == "structural_regime"][0]
    path = tmp_path / "trade_journal.sqlite3"
    os.chmod(path, stat.S_IREAD)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            mentor_questions.record_answer(subject, {"state": "range"}, store=store, now=NOW)
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def test_the_card_files_the_weekly_answer_off_the_qt_thread(tmp_path):
    import threading

    import mentor_questions
    import structural_regime

    app = _qt()
    store = _store(tmp_path)
    card = _card(tmp_path, store)
    _drain(app)
    result = mentor_questions.pending(_state(card.regime_lane()), slot=None)
    card.set_questions(result, store=store, service=None)
    combo = card.question_box("structural_regime", "2026-W40")
    combo.setCurrentIndex(combo.findData("bear_channel_lower_highs"))

    gui = threading.get_ident()
    seen: list = []
    real = store.append_structural_regime

    def spy(**kwargs):
        seen.append(threading.get_ident())
        return real(**kwargs)

    store.append_structural_regime = spy
    outcome = card.save_questions()
    _drain(app)

    assert outcome["queued"] == 1
    assert seen and seen[0] != gui
    assert structural_regime.current_regime(store.list_structural_regime(), "2026-09-28")["regime"] == (
        "bear_channel_lower_highs"
    )


def test_the_regime_consumer_reads_the_answer_key():
    import mentor_questions

    row = [r for r in mentor_questions.consumer_report() if r["kind"] == "structural_regime"][0]
    assert row["imports"] and row["reads"], row["reason"]
