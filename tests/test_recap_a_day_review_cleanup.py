"""Day Recap step A: the Day Review page cleaned up (trader, 2026-09-23).

One clock, plain words, a glance strip, day navigation, one miss table, trade
charts, no Qt-thread parquet reads, and a no-trade day that is not an error.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

SESSION = "2026-09-22"
NOW = datetime(2026, 9, 23, 7, 30)
PACIFIC = ZoneInfo("America/Los_Angeles")


class _Service:
    """Reads nothing: every test here renders a payload by hand."""

    def read_day(self, session_date, **_kwargs):
        from ui.services.day_review_service import empty_payload

        return empty_payload(session_date)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app, monkeypatch):
    from ui.panels import day_review_panel

    monkeypatch.setattr(day_review_panel, "_display_zone", lambda: PACIFIC, raising=False)
    widget = day_review_panel.DayReviewPanel(service=_Service(), clock=lambda: NOW)
    widget.show_session(SESSION)
    yield widget
    widget.shutdown()
    widget.deleteLater()
    app.processEvents()


def _payload(**extra):
    from ui.services.day_review_service import empty_payload

    payload = empty_payload(SESSION)
    payload.update(extra)
    return payload


# -- 1. one clock ------------------------------------------------------------
def test_a_utc_note_stamp_is_shown_in_the_desk_zone(panel):
    """The live bug: a 07:43 PT call stored as 14:43 UTC read "14:43"."""
    panel.render(_payload(entries=[{
        "entry_id": "e1", "created_at": "2026-09-22T14:43:24+00:00",
        "timeframe": "M5", "text": "choppy open",
    }]))
    assert panel.entries.item(0).text().startswith("07:43"), panel.entries.item(0).text()
    panel.entries.setCurrentRow(0)
    assert "written 07:43" in panel.entry_meta.text()


def test_the_calls_table_shows_a_clock_not_raw_iso(panel):
    panel.render(_payload(reads=[{
        "entry_id": "e1", "stamp": "2026-09-22T07:43:24.344534-07:00",
        "horizon": "rest_of_day", "direction": "chop", "confidence": "medium",
        "verdict": "right",
    }]))
    assert panel.calls_table.item(0, 0).text() == "07:43"


def test_trade_times_are_in_the_desk_zone_with_a_date_only_off_session(panel):
    panel.render(_payload(
        trades=[{"trade_id": "t1", "symbol": "DRAM", "opened_at": "2026-09-22T12:33:41.781000-04:00"}],
        trade_reviews=[{
            "trade_id": "t1", "symbol": "DRAM",
            "opened_at": "2026-09-22T12:33:41.781000-04:00",
            "closed_at": "2026-09-23T10:07:28.588000-04:00",
            "net_pnl": -17.78, "currency": "USD",
        }],
    ))
    assert panel.trades_table.item(0, 0).text() == "09:33"
    detail = panel.trade_detail.toPlainText()
    assert "Opened 09:33" in detail
    assert "Closed Sep 23 07:07" in detail
    assert "-04:00" not in detail


def test_the_zone_is_named_once_in_the_header(panel):
    panel.render(_payload())
    assert panel.zone_note.text() == "Times in PDT"


# -- 8. a no-trade day is not an error ---------------------------------------
def test_a_no_trade_day_is_not_a_failure_in_the_status_line(panel):
    from ui.services.day_review_service import NO_TRADES_EXIT_NOTE

    panel.render(_payload(error=NO_TRADES_EXIT_NOTE))
    assert panel.status.text() == f"Day Review: {SESSION}"


def test_a_real_failure_still_shows_beside_the_no_trade_fact(panel):
    from ui.services.day_review_service import NO_TRADES_EXIT_NOTE

    panel.render(_payload(error=f"the open theses could not be read: boom · {NO_TRADES_EXIT_NOTE}"))
    assert panel.status.text() == "the open theses could not be read: boom"


# -- 7. no Qt-thread parquet reads -------------------------------------------
def test_no_bars_file_is_read_on_the_qt_thread(app, monkeypatch):
    """`_backfill_bars_for` and the per-exit loop in `render` read on a worker."""
    import threading

    import day_review_bars
    from ui.panels.day_review_panel import DayReviewPanel

    reads: list[tuple[str, int]] = []

    def _read(session, **_kwargs):
        reads.append((str(session), threading.get_ident()))
        return {"SPY": []}

    class _Service(_BarsService):
        pass

    monkeypatch.setattr(day_review_bars, "read_session_bars", _read)
    monkeypatch.setattr(day_review_bars, "session_is_backfillable", lambda *_a, **_k: True)
    panel = DayReviewPanel(service=_Service(), clock=lambda: NOW)
    try:
        gui = threading.get_ident()
        panel._backfill_bars_for("2026-09-18")
        panel.render(_payload(walkaway_backfill_sessions=("2026-09-17",)))
        for _ in range(200):
            worker = panel._bars_worker
            if worker is not None:
                worker.wait(2000)
            app.processEvents()
            if len(reads) >= 2 and (panel._bars_worker is None or not panel._bars_worker.isRunning()):
                break
        assert {session for session, _ in reads} == {"2026-09-18", "2026-09-17"}
        assert all(thread != gui for _session, thread in reads), reads
        # The file was there, so nothing was fetched, and it is not asked twice.
        assert panel.service.fetched == []
        before = len(reads)
        panel._backfill_bars_for("2026-09-18")
        assert len(reads) == before
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()


class _BarsService(_Service):
    def __init__(self):
        self.fetched: list[str] = []

    def backfill_session_bars_for(self, session_date, **_kwargs):
        self.fetched.append(str(session_date))


def test_the_forecast_dialog_is_non_modal_and_prefills_no_source(panel, app, monkeypatch):
    from PySide6.QtWidgets import QDialog, QDialogButtonBox

    imported: list[dict] = []
    monkeypatch.setattr(panel, "_import_forecast", lambda values: imported.append(values) or {})
    monkeypatch.setattr(
        QDialog, "exec", lambda *_a: pytest.fail("the forecast dialog blocked the desk")
    )
    panel._paste_daily_forecast()
    dialog = panel._forecast_dialog
    assert dialog.isVisible() and not dialog.isModal()
    assert dialog.model_box.text() == ""
    dialog.text_box.setPlainText("# Market Morning Brief")
    dialog.buttons.button(QDialogButtonBox.Ok).click()
    assert imported and imported[0]["text"] == "# Market Morning Brief"
    assert imported[0]["source_model"] == ""
