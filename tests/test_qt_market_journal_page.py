"""The Market Journal page loads, and the desk tab reaches it (2026-08-27).

Two defects behind one screenshot. The trader wrote five entries through the
Desk "Journal" tab, opened the left-nav Market Journal page, and saw an empty
entries list, an empty session picker and an empty everything else:

* nothing ever called ``reload()`` - not at construction, not on show - so the
  page was blank until "Refresh" was clicked;
* the Desk tab built its OWN ``MarketJournalService``, so its ``entryWritten``
  signal was emitted by an object the page had never heard of. Both wrote the
  same file correctly; what was lost was the refresh.

These are wiring tests: they construct the real widgets against a temporary
store and assert what reaches the screen. Source-level assertions are used only
where the alternative would be to start a background thread in the suite.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import market_journal_capture as mjc  # noqa: E402

PANEL_SOURCE = SCRIPTS_DIR / "ui" / "panels" / "alert_center_panel.py"
APP_SOURCE = SCRIPTS_DIR / "ui" / "app.py"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path, monkeypatch):
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)
    return tmp_path


@pytest.fixture
def panel(qapp, store):
    from ui.panels.market_journal_panel import MarketJournalPanel
    from ui.services.market_journal_service import MarketJournalService

    widget = MarketJournalPanel(service=MarketJournalService())
    yield widget
    widget.shutdown()
    widget.deleteLater()


def _render(widget, entries, digests=None):
    """Hand the page a worker payload directly - the worker itself is a thread."""
    widget._render(
        {
            "entries": entries,
            "sessions": sorted({row["session_date"] for row in entries}),
            "timeline": {"shifts": [], "agreement": {"rate": None, "note": "none"}},
            "context": {"measured": False, "reason": "not measured"},
            "digests": digests or {},
        }
    )


def _entry(entry_id, text, *, symbols=("DT",), origin="desk_tab"):
    return {
        "entry_id": entry_id,
        "session_date": date.today().isoformat(),
        "created_at": "2026-08-27T13:36:35+00:00",
        "timeframe": "M5",
        "symbols": list(symbols),
        "origin": origin,
        "text": text,
        "written_after_the_session": False,
    }


# ==========================================================================
# the page loads
# ==========================================================================
def test_the_page_loads_the_first_time_it_is_shown(panel, monkeypatch):
    """The whole defect: a day with five entries in it rendered as an empty
    journal, because `reload` had no caller at all."""
    calls = []
    monkeypatch.setattr(panel, "reload", lambda: calls.append(1))

    panel.show()
    assert calls == [1]

    # And only once - showing it again is a page switch, not a re-read.
    panel.hide()
    panel.show()
    assert calls == [1]


def test_the_entries_reach_the_list(panel):
    _render(panel, [_entry("mj-1", "Gap up open on strong NVDA earnings.")])

    assert panel.entries.count() == 1
    assert "Gap up open" in panel.entries.item(0).text()


def test_an_empty_session_says_so_and_clears_the_charts(panel):
    _render(panel, [_entry("mj-1", "something")])
    _render(panel, [])

    assert "No entries for this session yet." in panel.entries.item(0).text()
    assert "nothing to chart" in panel.charts_note.text()


def test_a_desk_written_row_is_marked_in_the_list(panel):
    """The journal is one timeline; a reader must be able to tell the desk's
    own rows from the trader's without reading the sentence."""
    _render(panel, [_entry("mj-1", "Auto mode DESK -> AWAY.", origin="auto_mode_flip")])

    assert "[desk]" in panel.entries.item(0).text()


def test_the_session_picker_never_silently_repoints_the_page(panel):
    """Adding the first item to an editable combo selects it, which used to
    change WHICH session was being read as a side effect of loading."""
    today = date.today().isoformat()
    _render(panel, [_entry("mj-1", "x")])

    assert panel.session_date() == today


# ==========================================================================
# the captured charts
# ==========================================================================
def test_an_entry_with_no_capture_says_it_was_never_taken(panel):
    _render(panel, [_entry("mj-1", "x")])

    assert "No charts were captured" in panel.charts_note.text()
    assert not any(holder.isVisible() for holder in panel.chart_holders.values())


def test_a_captured_entry_draws_its_stored_tape(panel, store):
    from datetime import datetime, timedelta

    bars = [
        {
            "dt": datetime(2026, 8, 27, 9, 30) + timedelta(minutes=5 * index),
            "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.2, "volume": 1_000,
        }
        for index in range(20)
    ]
    capture = mjc.build_capture(entry_id="mj-1", symbol="DT", m5_bars=bars, d1_bars=bars)
    mjc.record_capture(capture)
    _render(panel, [_entry("mj-1", "x")], digests={"mj-1": {"digest": capture["digest"]}})

    panel._render_capture("mj-1", capture)

    assert panel.charts["symbol_m5"]._bars
    assert "20 bars" in panel.chart_titles["symbol_m5"].text()
    assert "DT M5" in panel.digest_label.text()


def test_a_pane_that_was_never_captured_is_hidden_not_drawn_empty(panel):
    """An auto-mode flip captures SPY alone. Two empty axes read as two failed
    charts."""
    from datetime import datetime, timedelta

    bars = [
        {
            "dt": datetime(2026, 8, 27, 9, 30) + timedelta(minutes=5 * index),
            "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.2, "volume": 1_000,
        }
        for index in range(10)
    ]
    capture = mjc.build_capture(
        entry_id="mj-2", symbol="SPY", m5_bars=bars, reason=mjc.REASON_MODE_FLIP
    )
    _render(panel, [_entry("mj-2", "flip", symbols=("SPY",))], digests={"mj-2": {"digest": ""}})
    panel._render_capture("mj-2", capture)

    assert panel.chart_holders["symbol_m5"].isVisibleTo(panel)
    assert not panel.chart_holders["symbol_d1"].isVisibleTo(panel)
    assert not panel.chart_holders["benchmark_m5"].isVisibleTo(panel)


def test_a_late_capture_never_lands_under_another_entrys_words(panel):
    _render(panel, [_entry("mj-1", "x"), _entry("mj-2", "y")], digests={"mj-1": {"digest": ""}})
    before = panel.charts_note.text()

    panel._render_capture("mj-1", {"symbol": "DT", "series": {}})

    assert panel.charts_note.text() == before


def test_a_capture_row_whose_bars_vanished_is_said_out_loud(panel):
    _render(panel, [_entry("mj-1", "x")], digests={"mj-1": {"digest": "d"}})
    panel._render_capture("mj-1", {})

    assert "stored bars could not be read" in panel.charts_note.text()


# ==========================================================================
# one writer
# ==========================================================================
def test_the_service_is_a_process_wide_singleton():
    from ui.services.market_journal_service import shared_journal_service

    assert shared_journal_service() is shared_journal_service()


def test_the_desk_tab_uses_the_shared_service_rather_than_its_own():
    """Source-level, because constructing the Alert Center means constructing
    the desk. The defect was a second instance, so this asserts the
    constructor call is gone."""
    source = PANEL_SOURCE.read_text(encoding="utf-8")

    assert "shared_journal_service()" in source
    assert "MarketJournalService(self)" not in source


def test_the_desk_tab_captures_the_charts_after_the_entry_is_written():
    """Order is the rule: a note must never wait on a chart."""
    source = PANEL_SOURCE.read_text(encoding="utf-8")
    write_at = source.index("origin=market_journal.ORIGIN_DESK_TAB")
    capture_at = source.index("_capture_journal_charts(result.get(\"entry\")")

    assert write_at < capture_at


def test_the_auto_mode_flip_is_journalled_with_spy_attached():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "autoModeChanged.connect(self._record_auto_mode_flip)" in source
    assert "ORIGIN_AUTO_MODE_FLIP" in source
    assert "REASON_MODE_FLIP" in source
