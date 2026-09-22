"""A late answer stays attached to the trade and keeps its own timestamp."""

from __future__ import annotations

import sys
import os
from types import SimpleNamespace
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_review_projection_keeps_raw_answers_and_selected_exit(monkeypatch):
    from ui.services import journal_feed

    events = [
        {"trade_id": "t1", "event_type": "RECALLED_RAW", "occurred_at": "2026-09-22T09:00:00-04:00", "payload": {"raw_text": "I chased the open", "recalled_after_session": True}},
        {"trade_id": "t1", "event_type": "RECALLED", "occurred_at": "2026-09-22T09:01:00-04:00", "payload": {"field": "thesis", "state": "", "text": "breakout"}},
    ]

    class Store:
        def list_opportunity_events(self, **kwargs):
            return [row for row in events if row["event_type"] == kwargs["event_type"]]

    monkeypatch.setattr(journal_feed, "_store", lambda: Store())
    trade = {"trade_id": "t1", "opened_at": "2026-09-18T09:30:00-04:00", "closed_at": "2026-09-21T15:00:00-04:00", "symbol": "ABC", "currency": "USD", "net_pnl": 42.0}
    notes = {"t1": {"raw_text": "Sold into the close", "recorded_at": "2026-09-22T09:10:00-04:00", "exit_session": "2026-09-21", "exit_fields": {"status": "confirmed", "fields": {"why": "target"}, "confirmed_at": "2026-09-22T10:00:00-04:00"}}}
    result = journal_feed.trade_reviews_on("2026-09-21", [trade], notes)
    row = result[0]
    assert row["entry_raw"]["text"] == "I chased the open"
    assert row["entry_answers"]["thesis"]["recorded_at"] == "2026-09-22T09:01:00-04:00"
    assert row["exit_raw"]["text"] == "Sold into the close"
    assert row["exit_fields"]["status"] == "confirmed"
    assert row["currency"] == "USD" and row["net_pnl"] == 42.0


def test_empty_trade_projection_does_not_open_store(monkeypatch):
    from ui.services import journal_feed

    monkeypatch.setattr(journal_feed, "_store", lambda: (_ for _ in ()).throw(AssertionError("opened")))
    assert journal_feed.trade_reviews_on("2026-09-21", [], {}) == []


def test_unread_answer_source_does_not_look_like_no_answer(monkeypatch):
    import pytest
    from ui.services import journal_feed

    class BrokenStore:
        def list_opportunity_events(self, **_kwargs):
            raise OSError("answer table unreadable")

    monkeypatch.setattr(journal_feed, "_store", lambda: BrokenStore())
    with pytest.raises(OSError, match="answer table unreadable"):
        journal_feed.trade_reviews_on("2026-09-21", [{"trade_id": "t1"}], {})


def test_interim_scale_out_stays_on_its_own_day(monkeypatch):
    from ui.services import journal_feed

    trade = {
        "trade_id": "scale-1", "opened_at": "2026-09-14T09:30:00-04:00",
        "closed_at": "2026-09-21T15:00:00-04:00", "trade_date": "2026-09-21",
        "net_pnl": 100.0, "realized_pnl": 100.0,
    }

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args):
            return self

        def fetchall(self):
            return [{"trade_id": "scale-1", "timestamp": "2026-09-18T14:00:00-04:00"}]

    class Store:
        def list_trades(self, *, trade_date=None):
            return [] if trade_date else [trade]

        def connection(self):
            return Connection()

    monkeypatch.setattr(journal_feed, "_store", lambda: Store())
    selected = journal_feed.trades_on("2026-09-18")
    assert [row["trade_id"] for row in selected] == ["scale-1"]
    assert selected[0]["net_pnl"] is None and selected[0]["realized_pnl"] is None


def test_real_journal_store_keeps_interim_and_final_exit_money_distinct(monkeypatch, tmp_path):
    from journal_store import JournalStore
    from ui.services import journal_feed

    store = JournalStore(tmp_path / "journal.sqlite3")
    with store.connection() as conn:
        conn.execute(
            """INSERT INTO trades(
                trade_id, broker, account_number, account_label, symbol, security_type,
                currency, direction, status, opened_at, closed_at, trade_date,
                quantity_opened, quantity_closed, average_entry_price, average_exit_price,
                gross_pnl, commission, fees, net_pnl, pnl_usd, updated_at
            ) VALUES('scale-1','IBKR','U1','MAIN','ABC','STOCK','USD','LONG','CLOSED',
                '2026-09-14T09:30:00-04:00','2026-09-21T15:00:00-04:00','2026-09-21',
                100,100,10,11,103,2,1,100,100,'2026-09-22T12:00:00-04:00')"""
        )
        conn.execute(
            """INSERT INTO trade_legs(
                trade_id, execution_uid, side, role, quantity, price, timestamp, commission, fees
            ) VALUES('scale-1','leg-1','SELL','CLOSE',50,11,
                '2026-09-18T14:00:00-04:00',1,0)"""
        )
    monkeypatch.setattr(journal_feed, "_store", lambda: store)
    interim = journal_feed.trades_on("2026-09-18")
    final = journal_feed.trades_on("2026-09-21")
    assert len(interim) == len(final) == 1
    assert interim[0]["net_pnl"] is None
    assert final[0]["net_pnl"] == 100


def test_real_trade_row_shows_late_words_and_opens_exact_journal_trade():
    from PySide6.QtWidgets import QApplication
    from ui.panels.day_review_panel import DayReviewPanel
    from ui.services.day_review_service import empty_payload

    app = QApplication.instance() or QApplication([])
    panel = DayReviewPanel(service=object())
    opened: list[str] = []
    panel.openTradeRequested.connect(opened.append)
    try:
        payload = empty_payload("2026-09-21")
        payload["trades"] = [{"trade_id": "exact-7", "symbol": "ABC", "net_pnl": 42, "currency": "USD"}]
        payload["trade_reviews"] = [{
            "trade_id": "exact-7", "symbol": "ABC", "net_pnl": 42, "currency": "USD",
            "entry_raw": {"text": "I chased the open", "recorded_at": "2026-09-22T09:00:00-04:00"},
            "entry_answers": {"thesis": {"text": "breakout", "recorded_at": "2026-09-22T09:01:00-04:00"}},
            "exit_raw": {"text": "Sold into the close", "recorded_at": "2026-09-22T10:00:00-04:00"},
            "exit_fields": {"status": "confirmed", "fields": {"why": "target"}},
        }]
        payload["entries"] = [{"entry_id": "entry-1", "text": "Market should rise"}]
        payload["reads"] = [{
            "entry_id": "entry-1", "stamp": "2026-09-21T10:00:00-04:00",
            "horizon": "rest_of_day", "direction": "up", "confidence": "medium",
            "verdict": "right",
        }]
        panel.render(payload)
        assert panel.calls_table.item(0, 1).text() == "rest_of_day"
        panel.calls_table.cellDoubleClicked.emit(0, 0)
        assert panel.entries.currentRow() == 0
        assert "I chased the open" in panel.trade_detail.toPlainText()
        assert "Sold into the close" in panel.trade_detail.toPlainText()
        assert "Confirmed exit" in panel.trade_detail.toPlainText()
        panel.trades_table.cellDoubleClicked.emit(0, 0)
        assert opened == ["exact-7"]
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()


def test_week_day_link_opens_exact_session_without_old_reload():
    from ui import app

    events: list[object] = []
    host = SimpleNamespace()
    host._select_page_by_title = lambda title: events.append((title, host._opening_latest_day_review)) or True
    host.day_review_panel = SimpleNamespace(show_session=lambda session: events.append(session))
    app.MainWindow._open_day_review_session(host, "2026-09-18")
    assert events == [(app.DAY_REVIEW_PAGE_TITLE, True), "2026-09-18"]
    assert host._opening_latest_day_review is False


def test_stale_story_never_shows_old_words_on_current_day():
    from PySide6.QtWidgets import QApplication
    from ui.panels.day_review_panel import DayReviewPanel
    from ui.services.day_review_service import empty_payload

    app = QApplication.instance() or QApplication([])
    panel = DayReviewPanel(service=object())
    try:
        payload = empty_payload("2026-09-21")
        payload["day_story"] = {"session_date": "2026-09-21", "narration": {"headline": "Old certainty"}}
        payload["story_freshness"] = {"state": "stale", "reason": "late trade answer"}
        payload["report_card"] = {"lines": [{"key": "how_fresh", "night_status": "failed", "slots_failed": ["day_review_narration"]}]}
        panel.render(payload)
        assert "Old certainty" not in panel.story_note.text()
        assert "late trade answer" in panel.story_note.text()
        assert "failed its checks" in panel.story_note.text()
        assert panel.story_body.text() == ""
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()
