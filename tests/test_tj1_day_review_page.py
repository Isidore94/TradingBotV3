"""TJ-1 item 3 - the Day Review page, offscreen.

Trader, 2026-09-17: *"there's just too much shit in these tabs and it's laggy as
all hell. this should be a simple 'what worked what didn't and what was your
process'."* Decision 0021 answer 12: the lag is on OPENING the tab and CLICKING
an entry. So this page is built to two rules and both are asserted here:

* **Nothing expensive at construction.** The old Market Journal built four
  `CandleChart`s on the first entry click (299 ms measured, G0). This page builds
  ONE, only when the SPY section has bars to draw, and reuses it.
* **One worker, one payload.** Every read happens on one `QThread` and the page
  paints only from what that worker hands back, so no section can start a read of
  its own on the Qt thread.

The contract these tests pin, so the builder has nothing to guess:

``scripts/ui/services/day_review_service.py``
    ``class DayReviewService`` with
    ``read_day(session_date, *, lookback_sessions=3, now=None) -> dict`` returning
    the keys in :data:`PAYLOAD_KEYS` (plus an optional ``error`` naming whatever
    could not be read), and thin forwards ``write_entry(...)`` /
    ``import_daily_forecast(...)`` onto the shared Market Journal writer (ONE
    writer, ground rule 8).

``scripts/ui/panels/day_review_panel.py``
    ``PICKER_SESSIONS = 15``; ``NO_STORY_YET``, ``NO_CHART_NOTE``,
    ``NO_IDEAS_YET``, ``WALKAWAY_TITLE``; ``class DayReviewPanel(QFrame)`` with
    ``__init__(self, service=None, parent=None, *, clock=None,
    auto_time_reader=None)``, signals ``statusChanged(str)`` and
    ``chartRequested(str, str)``, and the members
    ``session_picker``, ``refresh_button``, ``story_note``, ``theses``,
    ``rejected_that_worked_table``, ``walkaway_tables``, ``entries``, ``entry_reader``,
    ``entry_text``, ``save_button``, ``timeframe_picker``,
    ``paste_forecast_button``, ``forecast_box``, ``trades_table``, ``spy_note``,
    ``ideas_note``, plus ``render(payload)``, ``reload()``, ``start()``,
    ``poll_auto_read()``, ``session_date()``, ``shutdown()``.
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

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

SESSION = "2026-09-10"
#: Friday morning, mid-session: 2026-09-10 is the last COMPLETED session and
#: 2026-09-11 is today, offered as provisional.
NOW = datetime(2026, 9, 11, 7, 30)
TODAY = "2026-09-11"

#: What one worker read hands the page. Every section paints from this and
#: nothing else.
PAYLOAD_KEYS = (
    "session_date",
    "provisional",
    "story",
    "theses",
    "entries",
    "rejected_that_worked",
    "walkaway",
    "trades",
    "forecast",
    "spy_m5_bars",
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _StubService:
    """A DayReviewService stand-in. Records; reads nothing, writes nothing."""

    def __init__(self, payload=None) -> None:
        self.payload = payload or {}
        self.writes: list[dict] = []
        self.forecasts: list[dict] = []
        self.read_calls: list[tuple] = []

    def read_day(self, session_date, **kwargs):
        self.read_calls.append((session_date, kwargs))
        return dict(self.payload)

    def write_entry(self, **kwargs):
        self.writes.append(dict(kwargs))
        return {"ok": True, "entry": {"entry_id": "mj-stub-0001"}}

    def import_daily_forecast(self, **kwargs):
        self.forecasts.append(dict(kwargs))
        return {"ok": True, "entry": {"entry_id": "mj-stub-0002"}}


def _entry(entry_id, text, *, created_at, origin="journal_page", timeframe="M5"):
    return {
        "entry_id": entry_id,
        "session_date": SESSION,
        "created_at": created_at,
        "timeframe": timeframe,
        "symbols": [],
        "origin": origin,
        "text": text,
        "written_after_the_session": False,
    }


TRADER_EARLY = _entry(
    "mj-1",
    "Gap up and the semis led. I waited for the pullback.",
    created_at="2026-09-10T06:40:00-07:00",
)
MENTOR_LATER = _entry(
    "mj-2",
    "10:00 read: breadth turned, I am flat and comfortable.",
    created_at="2026-09-10T10:00:00-07:00",
    origin="trade_mentor",
)
MACHINE_ROW = _entry(
    "mj-3",
    "Auto mode DESK -> AWAY. Written by the desk, not the trader.",
    created_at="2026-09-10T07:05:00-07:00",
    origin="auto_mode_flip",
)
LONG_ENTRY = _entry(
    "mj-4",
    "First line of a long thought that runs well past the excerpt limit because "
    "the trader kept typing about the tape and never pressed Enter.\nSecond line.",
    created_at="2026-09-10T12:00:00-07:00",
)

SPY_BARS = [
    {
        "dt": datetime(2026, 9, 10, 6, 30) + timedelta(minutes=5 * i),
        "open": 100.0 + i,
        "high": 101.0 + i,
        "low": 99.5 + i,
        "close": 100.5 + i,
        "volume": 1000 + i,
    }
    for i in range(12)
]


def _payload(**overrides):
    base = {
        "session_date": SESSION,
        "provisional": False,
        "story": None,
        "theses": [],
        "entries": [],
        "rejected_that_worked": [],
        "walkaway": None,
        "trades": [],
        "forecast": {},
        "spy_m5_bars": [],
    }
    base.update(overrides)
    return base


@pytest.fixture()
def panel(qapp, monkeypatch):
    from ui.panels.day_review_panel import DayReviewPanel

    service = _StubService()
    widget = DayReviewPanel(service=service, clock=lambda: NOW)
    # No worker, ever, in this file: `reload` is the thread starter and every
    # test here paints through `render` instead.
    monkeypatch.setattr(widget, "reload", lambda: None)
    widget._stub_service = service
    yield widget
    try:
        widget.shutdown()
    except Exception:
        pass
    widget.deleteLater()


def _charts(widget):
    from ui.widgets.candle_chart import CandleChart

    return widget.findChildren(CandleChart)


# ==========================================================================
# 1. nothing expensive at construction
# ==========================================================================
def test_the_page_constructs_with_no_candle_chart_at_all(panel):
    """G0/G7: a `CandleChart` is a pyqtgraph plot. The page opens without one."""
    assert _charts(panel) == []


def test_a_session_with_no_spy_bars_still_builds_no_chart_and_says_why(panel):
    from ui.panels.day_review_panel import NO_CHART_NOTE

    panel.render(_payload(spy_m5_bars=[]))
    assert _charts(panel) == []
    assert NO_CHART_NOTE in panel.spy_note.text()
    assert "after the close" in NO_CHART_NOTE


def test_the_spy_chart_is_built_once_on_first_need_and_then_reused(panel):
    panel.render(_payload(session_date=TODAY, provisional=True, spy_m5_bars=SPY_BARS))
    charts = _charts(panel)
    assert len(charts) == 1, f"one SPY chart, got {len(charts)}"
    first = charts[0]

    panel.render(_payload(session_date=TODAY, provisional=True, spy_m5_bars=SPY_BARS[:6]))
    charts_again = _charts(panel)
    assert len(charts_again) == 1
    assert charts_again[0] is first, "a chart is reused, never rebuilt (ground rule 9)"


def test_the_retired_capture_panes_are_not_on_this_page(panel):
    """The old page's FOUR panes are what the 299 ms click paid for."""
    panel.render(_payload(session_date=TODAY, spy_m5_bars=SPY_BARS))
    assert len(_charts(panel)) == 1


# ==========================================================================
# 2. one payload paints every section
# ==========================================================================
def test_render_tolerates_a_payload_with_nothing_in_it(panel):
    """A first paint before any read, and a failed read, are the same shape."""
    panel.render({})
    panel.render(_payload())


def test_every_section_is_painted_from_the_one_payload(panel):
    from ui.panels.day_review_panel import NO_IDEAS_YET, NO_STORY_YET

    panel.render(
        _payload(
            entries=[TRADER_EARLY, MENTOR_LATER],
            theses=[{"entry_id": "mj-1", "text": "Dip buys work while the 10-year is calm."}],
            trades=[
                {
                    "trade_id": "t-1",
                    "symbol": "NVDA",
                    "direction": "LONG",
                    "quantity": 100,
                    "net_pnl": 240.0,
                    "status": "closed",
                    "opened_at": f"{SESSION}T07:05:00-07:00",
                }
            ],
            rejected_that_worked=[],
            forecast={"text": "Rates first, then oil.", "source_model": "chatgpt"},
        )
    )

    assert panel.entries.count() == 2
    assert panel.theses.count() == 1
    assert panel.trades_table.rowCount() == 1
    # TJ-1 says so plainly rather than pretending a story exists.
    assert NO_STORY_YET in panel.story_note.text()
    assert "written overnight" in NO_STORY_YET
    # TJ-6 brings the ideas; until then the card says nothing is there.
    assert NO_IDEAS_YET in panel.ideas_note.text()
    assert "Nothing yet" in NO_IDEAS_YET


def test_the_walkaway_is_four_ten_column_tables_under_its_fixed_titles(panel):
    """TJ-2B replaces TJ-1's one recap table and three placeholders."""
    from ui.panels.day_review_panel import WALKAWAY_TITLE

    assert WALKAWAY_TITLE == "Passed, and it ran"
    expected = [
        "Time",
        "Symbol",
        "Side",
        "What you did",
        "Ran after %",
        "Held at close %",
        "Traded?",
        "You made",
        "Left on the table %",
        "State",
    ]
    assert tuple(panel.walkaway_tables) == ("liked_not_traded", "rejected", "traded_left_early", "claimed_d1")
    for table in panel.walkaway_tables.values():
        assert table.columnCount() == 10
        assert [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())] == expected


def test_the_trades_line_is_read_only_and_names_the_money_once(panel):
    """ "What you traded" refers; the Journal page is still where trades are
    tagged and corrected (decision 0021 consequences)."""
    from PySide6.QtWidgets import QTableWidget

    panel.render(
        _payload(
            trades=[
                {
                    "trade_id": "t-1",
                    "symbol": "NVDA",
                    "direction": "LONG",
                    "quantity": 100,
                    "net_pnl": 240.0,
                    "status": "closed",
                    "opened_at": f"{SESSION}T07:05:00-07:00",
                }
            ]
        )
    )
    assert panel.trades_table.editTriggers() == QTableWidget.EditTrigger.NoEditTriggers
    row_text = " ".join(
        panel.trades_table.item(0, column).text()
        for column in range(panel.trades_table.columnCount())
        if panel.trades_table.item(0, column) is not None
    )
    assert "NVDA" in row_text
    assert "240" in row_text


def test_the_external_forecast_block_shows_the_session_forecast_when_one_exists(panel):
    panel.render(_payload(forecast={"text": "Rates first, then oil.", "source_model": "chatgpt"}))
    assert "Rates first, then oil." in panel.forecast_box.toPlainText()

    panel.render(_payload(forecast={}))
    assert panel.forecast_box.toPlainText().strip() == ""


# ==========================================================================
# 3. what you said: order, excerpts, and never a machine row
# ==========================================================================
def test_the_days_entries_read_oldest_first(panel):
    panel.render(_payload(entries=[MENTOR_LATER, TRADER_EARLY]))
    assert panel.entries.count() == 2
    assert "Gap up" in panel.entries.item(0).text()
    assert "10:00 read" in panel.entries.item(1).text()


def test_a_trade_mentor_answer_is_on_the_page_because_the_trader_wrote_it(panel):
    panel.render(_payload(entries=[MENTOR_LATER]))
    assert panel.entries.count() == 1
    assert "10:00 read" in panel.entries.item(0).text()


def test_a_machine_row_handed_to_the_page_is_still_not_in_the_list(panel):
    """Belt and braces on top of the service filter: the page is the surface the
    trader complained about, so it refuses the row itself."""
    panel.render(_payload(entries=[TRADER_EARLY, MACHINE_ROW, MENTOR_LATER]))
    assert panel.entries.count() == 2
    shown = " ".join(panel.entries.item(i).text() for i in range(panel.entries.count()))
    assert "Auto mode" not in shown


def test_the_list_shows_an_excerpt_and_the_reader_shows_the_whole_thought(panel):
    """The G3 rule: the list is one line per thought, the reader is the text."""
    panel.render(_payload(entries=[LONG_ENTRY]))
    item_text = panel.entries.item(0).text()
    assert "Second line." not in item_text, "the list is an excerpt, not the entry"
    assert item_text.rstrip().endswith("…"), "the ellipsis CLAIMS there is more"

    panel.entries.setCurrentRow(0)
    reader = panel.entry_reader.toPlainText()
    assert "Second line." in reader
    assert LONG_ENTRY["text"] in reader


def test_a_short_single_line_entry_is_shown_whole_with_no_ellipsis(panel):
    panel.render(_payload(entries=[TRADER_EARLY]))
    assert "…" not in panel.entries.item(0).text()


# ==========================================================================
# 4. the New entry box
# ==========================================================================
def test_saving_a_note_writes_through_the_one_writer_as_a_journal_page_entry(panel):
    panel.render(_payload())
    panel.entry_text.setPlainText("Stood down after 10:00; the tape was heavy.")
    panel.save_button.click()

    writes = panel._stub_service.writes
    assert len(writes) == 1, writes
    assert writes[0]["origin"] == "journal_page"
    assert writes[0]["session_date"] == panel.session_date()
    assert writes[0]["text"] == "Stood down after 10:00; the tape was heavy."
    assert panel.entry_text.toPlainText() == "", "a saved box is cleared"


def test_enter_saves_and_shift_enter_makes_a_newline(panel, qapp):
    panel.render(_payload())
    panel.entry_text.setPlainText("One Enter saves.")
    QApplication.sendEvent(
        panel.entry_text,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier),
    )
    assert len(panel._stub_service.writes) == 1

    panel.entry_text.setPlainText("Shift+Enter does not.")
    QApplication.sendEvent(
        panel.entry_text,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier),
    )
    assert len(panel._stub_service.writes) == 1


def test_the_forecast_button_is_the_daily_one_and_files_against_the_page_session(panel):
    """ "rename paste weekly forecast to paste daily forecast" (trader, 2026-09-17)."""
    label = panel.paste_forecast_button.text()
    assert "daily forecast" in label.lower()
    assert "weekly" not in label.lower()

    panel._import_forecast(
        {
            "text": "# Market Morning Brief — Thursday, September 17, 2026\n\nRates first.",
            "target_session": "2026-09-17",
            "source_model": "chatgpt",
        }
    )
    assert panel._stub_service.forecasts, "the page asked the service to import it"
    assert panel._stub_service.forecasts[0]["target_session"] == "2026-09-17"
    assert panel._stub_service.forecasts[0]["source_model"] == "chatgpt"


def test_an_empty_paste_imports_nothing(panel):
    panel._import_forecast({"text": "   ", "target_session": SESSION})
    assert panel._stub_service.forecasts == []


# ==========================================================================
# 5. the session picker and the two schedule functions
# ==========================================================================
def test_the_picker_offers_fifteen_completed_sessions_plus_today_as_provisional(panel):
    from ui.panels.day_review_panel import PICKER_SESSIONS

    assert PICKER_SESSIONS == 15
    assert panel.session_picker.count() == PICKER_SESSIONS + 1
    assert panel.session_picker.itemData(0) == SESSION
    assert panel.session_date() == SESSION

    last = panel.session_picker.count() - 1
    assert panel.session_picker.itemData(last) == TODAY
    assert "provisional" in panel.session_picker.itemText(last).lower()


def test_the_automatic_read_asks_both_schedule_functions_and_decides_nothing(panel, monkeypatch):
    """`daily_recap_schedule` stays the decision; the page only obeys it."""
    import daily_recap_schedule

    asked: list[str] = []
    monkeypatch.setattr(
        daily_recap_schedule,
        "due_session",
        lambda *_a, **_k: asked.append("noon") or None,
    )
    monkeypatch.setattr(
        daily_recap_schedule,
        "post_close_due_session",
        lambda *_a, **_k: asked.append("post_close") or SESSION,
    )
    shown: list[str] = []
    monkeypatch.setattr(panel, "show_session", lambda session: shown.append(session))

    assert panel.poll_auto_read() == SESSION
    assert asked == ["noon", "post_close"]
    assert shown == [SESSION]


def test_the_noon_read_wins_when_it_is_due_and_remembers_it_fired(panel, monkeypatch):
    """The memory is this process's and it is passed BACK to the decider, so a
    second tick in the same hour is not a second read."""
    import daily_recap_schedule

    calls: list[dict] = []

    def _due(_now, **kwargs):
        calls.append(dict(kwargs))
        return SESSION

    monkeypatch.setattr(daily_recap_schedule, "due_session", _due)
    monkeypatch.setattr(
        daily_recap_schedule,
        "post_close_due_session",
        lambda *_a, **_k: pytest.fail("the post-close check runs only when noon is not due"),
    )
    shown: list[str] = []
    monkeypatch.setattr(panel, "show_session", lambda session: shown.append(session))

    assert panel.poll_auto_read() == SESSION
    assert shown == [SESSION]
    assert calls[0]["last_fired_session"] is None

    panel.poll_auto_read()
    assert calls[1]["last_fired_session"] == SESSION


def test_the_timer_is_a_once_a_minute_check_started_by_the_host_not_the_constructor(panel):
    """A timer started during construction runs while a test is still patching
    what it reads (the `daily_recap_panel.start` rule, kept)."""
    assert not panel._auto_timer.isActive()
    panel.start()
    assert panel._auto_timer.isActive()
    assert panel._auto_timer.interval() == 60_000
    panel._auto_timer.stop()


# ==========================================================================
# 6. the service behind it
# ==========================================================================
def test_the_service_hands_back_one_payload_with_every_section_in_it(monkeypatch, tmp_path):
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)
    from ui.services.day_review_service import DayReviewService

    payload = DayReviewService().read_day(SESSION, now=NOW)
    assert set(PAYLOAD_KEYS) <= set(payload)
    assert payload["session_date"] == SESSION


def test_the_service_reads_a_stored_index_instead_of_streaming(monkeypatch, tmp_path):
    """Item 4's consumer: the page's read uses the index when there is one."""
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)

    import daily_recap_reader
    import day_review_index
    from ui.services.day_review_service import DayReviewService

    sentinel = {"schema": "day_review_index_v1", "session_date": SESSION, "pending": False}
    monkeypatch.setattr(day_review_index, "read_index", lambda *_a, **_k: sentinel)
    monkeypatch.setattr(day_review_index, "is_stale", lambda *_a, **_k: False)
    seen: list[dict] = []
    original = daily_recap_reader.read_session

    def _spy(session_date, **kwargs):
        seen.append(dict(kwargs))
        return original(session_date, **{k: v for k, v in kwargs.items() if k != "index"})

    monkeypatch.setattr(daily_recap_reader, "read_session", _spy)
    DayReviewService().read_day(SESSION, now=NOW)
    assert seen, "the service read the session"
    assert seen[0].get("index") is sentinel


def test_the_service_builds_and_writes_an_index_for_a_session_that_has_none(monkeypatch, tmp_path):
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)

    import day_review_index
    from ui.services.day_review_service import DayReviewService

    monkeypatch.setattr(day_review_index, "read_index", lambda *_a, **_k: None)
    written: list[object] = []
    monkeypatch.setattr(day_review_index, "write_index", lambda index, **_k: written.append(index))
    DayReviewService().read_day(SESSION, now=NOW)
    assert written, "an opened session leaves an index behind for next time"


def test_a_store_that_cannot_be_read_never_costs_the_page_its_payload(monkeypatch, tmp_path):
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)

    import daily_recap_reader
    from ui.services.day_review_service import DayReviewService

    def _refuse(*_a, **_k):
        raise OSError("the home folder is not mounted")

    monkeypatch.setattr(daily_recap_reader, "read_session", _refuse)
    payload = DayReviewService().read_day(SESSION, now=NOW)
    assert set(PAYLOAD_KEYS) <= set(payload)
    assert not payload["rejected_that_worked"]
    # Uncertainty is REPORTED, never hidden: the page says what it could not read.
    assert "not mounted" in str(payload.get("error") or "")


# ==========================================================================
# 7. the desk builds it, and builds neither old page
# ==========================================================================
@pytest.fixture(scope="module")
def qt_desk(qapp):
    from ui.app import MainWindow
    from ui.state import UiState

    window = MainWindow(UiState(workspace_mode="workspace"))
    yield window
    try:
        window.close()
    except Exception:
        pass


def test_the_desk_owns_a_day_review_panel(qt_desk):
    from ui.panels.day_review_panel import DayReviewPanel

    assert isinstance(qt_desk.day_review_panel, DayReviewPanel)


def test_the_desk_no_longer_constructs_either_retired_panel(qt_desk):
    """Item 7: the modules stay on disk; the WINDOW stops building them, which
    is where their construction cost and their timers lived."""
    assert not hasattr(qt_desk, "market_journal_panel")
    assert not hasattr(qt_desk, "daily_recap_panel")
