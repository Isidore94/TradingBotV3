"""TJ-1 review advisories (c) to (f) - four small holes, each with its reason.

(c) `theses_for()` with NO session filtered nothing, so the ONE machine-row
    filter had a way round it: `entries_about` applies it and `entries_for` does
    not, and only the first path was used.
(d) `_render_chart` handed every bar to the chart, including one with no `dt`.
    A bar that cannot be placed on a time axis must be dropped and COUNTED, not
    drawn at an invented moment (the candle invariant's cousin).
(e) `forecast_brief.headline` was public, used by the page, and missing from
    `__all__`.
(f) `ResearchResultsPanel.__init__` set `_report_loaded_once` AFTER `refresh()`,
    so a `showEvent` reached during that call would have raised `AttributeError`
    inside a Qt slot.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6", reason="two of these are Qt panels")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SESSION = "2026-09-16"
TRADER_TEXT = "SPY faded the open and the semis never confirmed."
MACHINE_TEXT = "Auto mode DESK -> AWAY. Written by the desk, not the trader."


# ---------------------------------------------------------------------------
# (c) theses_for() over EVERY session
# ---------------------------------------------------------------------------
@pytest.fixture()
def journal(tmp_path, monkeypatch):
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(
        project_paths, "MARKET_THESES_FILE", tmp_path / "market_theses.jsonl", raising=False
    )

    import market_journal
    from ui.services.market_journal_service import MarketJournalService

    service = MarketJournalService()
    ids = {
        "trader": service.write_entry(
            text=TRADER_TEXT, session_date=SESSION,
            timeframe=market_journal.TIMEFRAME_D1,
            origin=market_journal.ORIGIN_JOURNAL_PAGE,
        )["entry"]["entry_id"],
        "machine": service.write_entry(
            text=MACHINE_TEXT, session_date=SESSION,
            timeframe=market_journal.TIMEFRAME_M5, symbols=["SPY"],
            origin=market_journal.ORIGIN_AUTO_MODE_FLIP,
        )["entry"]["entry_id"],
    }
    return service, ids


def test_theses_for_every_session_drops_the_machine_rows_too(journal):
    service, ids = journal
    entry_ids = {str(row.get("entry_id") or "") for row in service.theses_for()}
    assert ids["trader"] in entry_ids
    assert ids["machine"] not in entry_ids


def test_theses_for_one_session_is_unchanged(journal):
    service, ids = journal
    entry_ids = {str(row.get("entry_id") or "") for row in service.theses_for(SESSION)}
    assert entry_ids == {ids["trader"]}


# ---------------------------------------------------------------------------
# (d) a bar with no timestamp
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _bar(index: int, *, dt=True) -> dict:
    row = {"open": 100.0 + index, "high": 101.0 + index, "low": 99.0 + index,
           "close": 100.5 + index, "volume": 1000}
    if dt:
        row["dt"] = datetime(2026, 9, 11, 6, 30) + timedelta(minutes=5 * index)
    return row


@pytest.fixture()
def panel(qapp):
    from ui.panels.day_review_panel import DayReviewPanel

    class _Service:
        def read_day(self, session_date, **_kwargs):
            return {"session_date": session_date}

    widget = DayReviewPanel(service=_Service(), clock=lambda: datetime(2026, 9, 11, 7, 30))
    yield widget
    try:
        widget.shutdown()
    except Exception:
        pass
    widget.deleteLater()
    qapp.processEvents()


def test_a_bar_with_no_timestamp_is_dropped_and_counted(panel, caplog):
    from ui.widgets.candle_chart import CandleChart

    with caplog.at_level("INFO"):
        panel._render_chart([_bar(0), _bar(1, dt=False), _bar(2)])

    charts = panel.findChildren(CandleChart)
    assert len(charts) == 1, "the drawable bars still drew"
    assert "2 completed bar(s)" in panel.spy_note.text()
    assert "no timestamp" in panel.spy_note.text()
    assert any("carried no timestamp" in record.message for record in caplog.records)


def test_bars_that_are_ALL_timestampless_draw_nothing_and_say_so(qapp, panel):
    from ui.widgets.candle_chart import CandleChart
    from ui.panels.day_review_panel import NO_CHART_NOTE

    panel._render_chart([_bar(0, dt=False), _bar(1, dt=False)])

    assert panel.findChildren(CandleChart) == [], "a chart was built for nothing"
    assert NO_CHART_NOTE in panel.spy_note.text()
    assert "no timestamp" in panel.spy_note.text()


def test_the_ordinary_case_says_nothing_about_timestamps(panel):
    panel._render_chart([_bar(0), _bar(1)])
    assert "2 completed bar(s)" in panel.spy_note.text()
    assert "no timestamp" not in panel.spy_note.text()


# ---------------------------------------------------------------------------
# (e) and (f)
# ---------------------------------------------------------------------------
def test_the_forecast_readers_public_helper_is_exported():
    import forecast_brief

    assert "headline" in forecast_brief.__all__
    assert set(forecast_brief.__all__) <= set(dir(forecast_brief))


def test_the_report_flag_exists_before_the_first_read_is_started():
    """Source-level, because the failure it prevents is an `AttributeError`
    inside a Qt slot on a path no test reaches on purpose: `refresh()` can lead
    to a `showEvent`, and `showEvent` reads this flag."""
    source = (
        SCRIPTS_DIR / "ui" / "panels" / "research_results_panel.py"
    ).read_text(encoding="utf-8")
    body = source[source.index("def __init__(self, parent=None)") :]
    body = body[: body.index("\n    def ")]

    assert "self._report_loaded_once = False" in body
    assert body.index("self._report_loaded_once = False") < body.index("self.refresh()")


def test_the_page_still_builds_and_shows_with_the_flag_in_that_order(qapp):
    from ui.panels.research_results_panel import ResearchResultsPanel

    widget = ResearchResultsPanel()
    try:
        widget.show()
        qapp.processEvents()
        assert widget._report_loaded_once is True
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()
