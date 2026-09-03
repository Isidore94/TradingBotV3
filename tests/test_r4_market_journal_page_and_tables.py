"""R4 A16, A17 and A18 - the journal page, the session roll, and ten rows.

A16: decision 0016 answer 11 is "one box, one Enter". V2 built that on the Desk
tab and left the left-nav page exactly as it was - a session picker, a Refresh, a
timeframe picker, a Save button and an after-the-fact caption, showing one
session at a time. Reading back "what did I think last week" meant knowing the
date first.

A17: `session_date_for` read the calendar date in New York, so a Pacific note at
21:00 PT - which is 00:00 ET the next day - filed against TOMORROW'S session, on
a day that had not opened, and `written_after_the_session` then said False. The
session ends at the CLOSE and the note is about it until the next one opens.

A18: only `tag_week` had a minimum height. Every other table on Weekend Prep sat
at whatever the layout gave it, which is the complaint V2 was answering.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PACIFIC = ZoneInfo("America/Los_Angeles")
EASTERN = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# A17 - the session a note is about
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hour", [21, 22])
def test_an_evening_pacific_note_is_about_the_session_that_just_traded(hour):
    """00:00 and 01:00 in New York - a day that has not opened."""
    import market_journal

    now = datetime(2026, 9, 2, hour, 0, tzinfo=PACIFIC)  # a Wednesday

    session = market_journal.session_date_for(now)
    entry = market_journal.build_entry(text="late", session_date=session, now=now)

    assert session == "2026-09-02"
    assert entry["written_after_the_session"] is True


def test_a_note_typed_during_the_session_is_about_today_and_not_flagged():
    import market_journal

    now = datetime(2026, 9, 2, 11, 15, tzinfo=EASTERN)

    session = market_journal.session_date_for(now)
    entry = market_journal.build_entry(text="live", session_date=session, now=now)

    assert session == "2026-09-02"
    assert entry["written_after_the_session"] is False


def test_the_weekend_still_files_against_the_last_session_that_traded():
    """The existing rule, kept: a Saturday note is about Friday."""
    import market_journal

    saturday = datetime(2026, 9, 5, 10, 0, tzinfo=PACIFIC)
    sunday = datetime(2026, 9, 6, 19, 0, tzinfo=PACIFIC)

    assert market_journal.session_date_for(saturday) == "2026-09-04"
    assert market_journal.session_date_for(sunday) == "2026-09-04"


def test_a_note_before_the_open_is_about_yesterday_not_today():
    """05:00 Pacific is 08:00 in New York - the session has not started."""
    import market_journal

    now = datetime(2026, 9, 2, 5, 0, tzinfo=PACIFIC)

    assert market_journal.session_date_for(now) == "2026-09-01"


# ---------------------------------------------------------------------------
# A16 - the page
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


class _StubService:
    """Only what the panel asks of it."""

    from PySide6.QtCore import QObject, Signal  # noqa: N815 - Qt naming

    def __init__(self, entries):
        self._entries = entries

    def entries_for(self, session_date: str = ""):
        rows = list(self._entries)
        if session_date:
            rows = [row for row in rows if row.get("session_date") == session_date]
        return rows

    def sessions_with_entries(self):
        return sorted({str(row.get("session_date") or "") for row in self._entries})

    def regime_timeline(self, **_kwargs):
        return {}

    def day_context(self, _session):
        return {}

    def chart_digests(self):
        return {}


def _entry(entry_id, session, created_at, text):
    return {
        "entry_id": entry_id,
        "session_date": session,
        "created_at": created_at,
        "timeframe": "D1",
        "text": text,
        "symbols": [],
    }


@pytest.fixture()
def panel(qapp):
    from ui.panels.market_journal_panel import MarketJournalPanel
    from ui.services.market_journal_service import MarketJournalService

    return MarketJournalPanel(MarketJournalService())


def test_the_picker_the_refresh_the_timeframe_and_save_left_the_layout(panel):
    """Out of the layout, NOT deleted - `reload()` and `_save()` still read them."""
    for name in ("session_picker", "refresh_button", "timeframe_picker", "save_button", "after_the_fact"):
        widget = getattr(panel, name)
        assert widget is not None, name
        assert widget.parent() is None, f"{name} is still parented into the page"


def test_the_schema_fields_are_empty_and_never_dropped(panel):
    """Nothing leaves the SCHEMA - only the surface."""
    import market_journal

    assert panel.timeframe_picker.count() == len(market_journal.TIMEFRAMES)
    entry = market_journal.build_entry(text="x", session_date="2026-09-02")
    for field in ("timeframe", "symbols", "session_date", "created_at", "origin"):
        assert field in entry, field


def test_the_session_is_computed_and_agrees_with_the_desk_tab(panel):
    import market_journal

    assert panel.session_date() == market_journal.session_date_for()


def test_the_entries_list_is_dated_and_newest_first(panel):
    panel._render_entries(
        [
            _entry("a", "2026-08-31", "2026-08-31T20:00:00+00:00", "oldest"),
            _entry("c", "2026-09-02", "2026-09-02T20:00:00+00:00", "newest"),
            _entry("b", "2026-09-01", "2026-09-01T20:00:00+00:00", "middle"),
        ]
    )

    labels = [panel.entries.item(i).text() for i in range(panel.entries.count())]
    assert [label.split("  ")[0] for label in labels] == [
        "2026-09-02",
        "2026-09-01",
        "2026-08-31",
    ]
    assert "newest" in labels[0]
    # And the newest is the one selected, because that is what the trader
    # opened the page to see.
    assert panel.entries.currentRow() == 0


def test_the_page_reads_every_session_rather_than_one(panel):
    """The picker answered "which session"; the list answers it by being dated."""
    source = (ROOT / "scripts" / "ui" / "panels" / "market_journal_panel.py").read_text(
        encoding="utf-8"
    )
    body = source.split("class _EntriesWorker", 1)[1].split("class _CaptureWorker", 1)[0]

    assert "self._service.entries_for()" in body
    assert "entries_for(self._session)" not in body


def test_enter_saves_and_shift_enter_does_not(panel, monkeypatch):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    saved: list[int] = []
    monkeypatch.setattr(panel, "_save", lambda: saved.append(1))

    shift = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier
    )
    plain = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
    )

    assert panel.eventFilter(panel.entry_text, shift) is False
    assert saved == []
    assert panel.eventFilter(panel.entry_text, plain) is True
    assert saved == [1]


# ---------------------------------------------------------------------------
# A18 - ten rows
# ---------------------------------------------------------------------------


def test_every_weekend_prep_table_shows_ten_rows(qapp):
    from PySide6.QtWidgets import QTableWidget
    from ui.panels import weekend_prep_panel as panel_module

    weekend = panel_module.WeekendPrepPanel()
    tables = weekend.findChildren(QTableWidget)

    assert tables, "the tab must have tables to be about"
    thin = [
        table
        for table in tables
        if table.minimumHeight() < panel_module.TABLE_TEN_ROWS_PX
    ]
    assert not thin, [table.objectName() or table.columnCount() for table in thin]
    weekend.shutdown()


def test_the_ten_row_floor_is_one_constant():
    """It was a bare 260 on one table and nothing on the other eleven."""
    from ui.panels import weekend_prep_panel as panel_module

    source = (ROOT / "scripts" / "ui" / "panels" / "weekend_prep_panel.py").read_text(
        encoding="utf-8"
    )

    assert panel_module.TABLE_TEN_ROWS_PX == 260
    assert source.count("setMinimumHeight(") == 1, "one owner for the floor"
    assert "setMinimumHeight(TABLE_TEN_ROWS_PX)" in source
