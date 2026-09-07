"""Packet G3 - the Market Journal's full thought is readable.

The GUI review's finding is a functional reading defect, not a taste one: the
left-nav Market Journal page puts the WHOLE text of an entry into a
``QListWidgetItem`` in a narrow list (`_render_entries`, the label is
``f"{session}  {timeframe}{hand}{marker}{camera} {[symbols] }{entry['text']}"``),
where the list elides it to one clipped line - and selecting the row repaints
the four capture charts and nothing else. A 1,200-character thought is
therefore shown NOWHERE on the page.

G3 answers it with two additive widgets - ``thought_meta`` (a ``QLabel``) and
``thought_view`` (a read-only ``QTextBrowser``) - filled synchronously from
``self._entries[row]`` at the head of ``_on_entry_selected``, before the
capture worker is started. The list keeps its dated, newest-first contract and
gains an EXCERPT.

These tests drive the real ``MarketJournalPanel`` offscreen through the real
seams the desk uses (``_render`` with a worker payload, ``_render_entries``,
``setCurrentRow`` -> ``currentRowChanged`` -> ``_on_entry_selected``). Nothing
here touches the live journal store: the service is a stub and
``project_paths.RUNTIME_DATA_DIR`` is pointed at ``tmp_path`` regardless.

Layout lane: no store, identity, timestamp or write behaviour is asserted or
changed here.
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

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

SESSION = "2026-09-02"
OTHER_SESSION = "2026-09-01"

#: A real thought's length. The reader must show every character of it, and the
#: list label must show a fraction. 1,200 characters, built so the first 60 are
#: distinctive and the tail is unmistakably far from the head.
LONG_HEAD = "SPY opened weak, faded the first push, and then reclaimed VWAP"
LONG_TEXT = LONG_HEAD + " " + ("and I kept adding to the same losing read. " * 30)
LONG_TEXT = LONG_TEXT[:1199] + "Z"

SHORT_TEXT = "NVDA held its anchored VWAP all day; nothing to do."


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _StubService(QObject):
    """Only the seams ``MarketJournalPanel`` binds in ``__init__``.

    The panel connects ``statusChanged`` and ``entryWritten`` at construction
    and asks ``chartCaptured`` for optionally, so the stub is a real ``QObject``
    with real signals. It never reads or writes a store.
    """

    statusChanged = Signal(str)
    entryWritten = Signal(dict)
    chartCaptured = Signal(dict)

    def __init__(self, entries=None, digests=None, captures=None):
        super().__init__()
        self._entries = list(entries or [])
        self._digests = dict(digests or {})
        self._captures = dict(captures or {})

    # -- what the workers call --------------------------------------------
    def entries_for(self, session_date: str = ""):
        rows = list(self._entries)
        if session_date:
            rows = [row for row in rows if row.get("session_date") == session_date]
        return rows

    def sessions_with_entries(self):
        return sorted({str(row.get("session_date") or "") for row in self._entries})

    def regime_timeline(self, **_kwargs):
        return {"shifts": [], "agreement": {"rate": None, "note": "none"}}

    def day_context(self, _session):
        return {"measured": False, "reason": "not measured"}

    def chart_digests(self):
        return dict(self._digests)

    def chart_capture(self, entry_id: str):
        return self._captures.get(entry_id)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The live journal is READ-ONLY to this suite; point the runtime dir away."""
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)
    return tmp_path


@pytest.fixture
def panel(qapp, store):
    from ui.panels.market_journal_panel import MarketJournalPanel

    widget = MarketJournalPanel(service=_StubService())
    yield widget
    widget.shutdown()
    widget.deleteLater()


def _entry(entry_id, text, *, session=SESSION, created_at=None, symbols=("SPY",)):
    """A row shaped exactly like a stored one - every key PRESENT."""
    return {
        "entry_id": entry_id,
        "session_date": session,
        "created_at": created_at or f"{session}T13:36:35-07:00",
        "timeframe": "D1",
        "symbols": list(symbols),
        "origin": "journal_page",
        "text": text,
        "written_after_the_session": False,
    }


def _render(widget, entries, digests=None):
    """Hand the page a worker payload directly - the worker itself is a thread."""
    widget.service._digests = dict(digests or {})
    widget._render(
        {
            "entries": entries,
            "sessions": sorted({row["session_date"] for row in entries}),
            "timeline": {"shifts": [], "agreement": {"rate": None, "note": "none"}},
            "context": {"measured": False, "reason": "not measured"},
            "digests": digests or {},
        }
    )


# ==========================================================================
# G3.1 - the list shows an excerpt, dated
# ==========================================================================
def test_a_long_thought_is_an_excerpt_in_the_list_not_the_whole_text(panel):
    """1,200 characters in a narrow list is one clipped line; 90 is a sentence.

    The dated contract is unchanged (`test_the_entries_list_is_dated_and_newest_
    first` splits the label on the two spaces), so the excerpt goes where the
    text was and the date stays first.
    """
    assert len(LONG_TEXT) == 1200

    _render(panel, [_entry("mj-long", LONG_TEXT)])

    label = panel.entries.item(0).text()
    assert len(label) < 140, f"the label is {len(label)} characters: {label[:160]!r}"
    assert label.startswith(SESSION), label[:40]
    assert LONG_TEXT[:60] in label
    assert label.endswith("…"), repr(label[-20:])
    # The tooltip keeps `written <created_at>` - the excerpt does not replace it.
    assert panel.entries.item(0).toolTip().startswith("written ")


# ==========================================================================
# G3.2 / G3.3 - a reader pane, filled by the selection
# ==========================================================================
def test_selecting_a_long_thought_shows_every_character_in_the_reader(panel):
    """The whole point: the full 1,200 characters are readable SOMEWHERE."""
    _render(panel, [_entry("mj-long", LONG_TEXT)])

    assert panel.thought_view.toPlainText() == LONG_TEXT

    meta = panel.thought_meta.text()
    assert SESSION in meta, meta
    assert "D1" in meta, meta
    assert "written" in meta, meta


def test_selecting_a_different_entry_changes_the_reader_to_that_entrys_text(panel):
    """Newest first, so the long one is row 0 and the older short one is row 1."""
    _render(
        panel,
        [
            _entry("mj-long", LONG_TEXT, session=SESSION),
            _entry("mj-short", SHORT_TEXT, session=OTHER_SESSION),
        ],
    )
    assert panel.thought_view.toPlainText() == LONG_TEXT

    panel.entries.setCurrentRow(1)

    assert panel.thought_view.toPlainText() == SHORT_TEXT
    assert OTHER_SESSION in panel.thought_meta.text()


def test_an_entry_with_no_capture_still_fills_the_reader(panel):
    """The words are the record; the charts are the follow-on. An entry written
    before the capture existed must still be readable, and the charts area must
    keep saying exactly what it says today."""
    _render(panel, [_entry("mj-long", LONG_TEXT)], digests={})

    assert panel.thought_view.toPlainText() == LONG_TEXT
    assert "No charts were captured" in panel.charts_note.text()


def test_an_empty_session_render_clears_the_reader(panel):
    """Otherwise yesterday's thought sits under today's silence - the same
    defect `_clear_charts` already fixes for the charts."""
    _render(panel, [_entry("mj-long", LONG_TEXT)])
    assert panel.thought_view.toPlainText() == LONG_TEXT

    _render(panel, [])

    assert panel.thought_view.toPlainText() == ""
    assert panel.thought_meta.text() == ""


def test_the_reader_is_filled_before_the_capture_worker_is_started(panel, monkeypatch):
    """Synchronously, from `self._entries[row]` - never from the worker.

    The late-capture guard (`test_a_late_capture_never_lands_under_another_
    entrys_words`) proves the worker's payload can arrive for a row the trader
    has left. The words must therefore never depend on it, and the way to prove
    that is to read the reader at the instant the worker is CONSTRUCTED.
    """
    import ui.panels.market_journal_panel as mjp

    seen: list[str] = []

    class _StubCaptureWorker(QObject):
        loaded = Signal(str, dict)

        def __init__(self, _service, entry_id, parent=None):
            super().__init__(parent)
            self._entry_id = entry_id
            seen.append(panel.thought_view.toPlainText())

        def isRunning(self):  # noqa: N802 - QThread's API
            return False

        def start(self):
            return None

        def wait(self, _ms=0):
            return True

    monkeypatch.setattr(mjp, "_CaptureWorker", _StubCaptureWorker)

    _render(panel, [_entry("mj-long", LONG_TEXT)], digests={"mj-long": {"digest": "d"}})

    assert seen, "the capture worker was never constructed - the seam moved"
    assert seen[0] == LONG_TEXT


# ==========================================================================
# Added by the BUILDER (nothing above is weakened): the excerpt's ellipsis is
# a claim, the reader sits above the charts, and the meta line keeps the zone
# the stamp carries. Each of these fails on the pre-G3 file too - `_excerpt`
# and `thought_meta` do not exist there.
# ==========================================================================
def test_the_ellipsis_means_there_is_more_and_is_not_printed_when_there_is_not():
    """A short thought is shown WHOLE in the list; `…` is never decoration."""
    from ui.panels.market_journal_panel import EXCERPT_LIMIT, _excerpt

    assert _excerpt(SHORT_TEXT) == SHORT_TEXT
    assert not _excerpt(SHORT_TEXT).endswith("…")
    # A first line that fits but is followed by more IS truncated - the rest
    # of the thought is real and the list must say so.
    assert _excerpt("one line\nand a second") == "one line…"
    assert _excerpt("") == ""
    long_one = _excerpt("x" * 400)
    assert len(long_one) == EXCERPT_LIMIT + 1
    assert long_one.endswith("…")


def test_the_reader_sits_above_the_charts_on_its_own_vertical_splitter(panel):
    """G3.2 - the right half is reader OVER charts, 2 to 3.

    Not the `lower` splitter's (2, 3), which is LEFT vs RIGHT: this is the new
    vertical one inside the right half.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter

    reader_holder = panel.thought_view.parent()
    charts_holder = panel.charts_note.parent()
    right = reader_holder.parent()

    assert isinstance(right, QSplitter), type(right).__name__
    assert right.orientation() == Qt.Vertical
    assert charts_holder.parent() is right, "the charts are not in the reader's splitter"
    assert right.indexOf(reader_holder) == 0, "the charts are above the words"
    # QSplitter has no `stretchFactor` getter: `setStretchFactor` writes the
    # child's size policy along the splitter's orientation, so that is where
    # the 2 and the 3 are read back from.
    assert reader_holder.sizePolicy().verticalStretch() == 2
    assert charts_holder.sizePolicy().verticalStretch() == 3


def test_the_meta_line_keeps_the_zone_the_stored_stamp_carries(panel, monkeypatch):
    """A time printed without its zone is a quiet backdating.

    G3b converts an aware stamp to the DESK zone, so the desk zone is pinned to
    -07:00 here: on a PST day or another machine the same fixture would
    otherwise print 12:36 -08:00 and this test would read the season, not the
    code (the G3b builder's own warning).
    """
    from datetime import timedelta, timezone

    from ui.panels.market_journal_panel import _written_line

    monkeypatch.setattr(
        "ui.annotations.pass_bars.desk_zone", lambda: timezone(timedelta(hours=-7))
    )
    _render(panel, [_entry("mj-long", LONG_TEXT, created_at=f"{SESSION}T13:36:35-07:00")])

    meta = panel.thought_meta.text()
    assert "written 13:36" in meta, meta
    assert "07:00" in meta, meta
    # A naive stamp SAYS it is naive rather than being handed a zone.
    assert "no zone recorded" in _written_line("2026-09-02T13:36:35")
    assert _written_line("") == "written at an unrecorded time"


# ==========================================================================
# G3 FIX ROUND (2026-09-06) - the reader must be READABLE, which is the whole
# point of the packet. The first cut put the words on the page and left them
# unreadable: at 3456 x 2160 the thought was set across the full 2,779 px of
# the right half (a 68-character sentence as one 400-character line), and the
# entries column opened at 655 px, where the new 90-character excerpt clipped
# after about 35 characters and the list read as nothing. Both are measured
# at the trader's own screen size, offscreen.
# ==========================================================================

#: The trader's screen. Every number below is measured at it.
DESK_SIZE = (3456, 2160)

#: A line of running text is readable at about 45-100 characters. 100
#: characters of this page's font measured 1,200 px; the bound is that plus
#: slack for a frame, a scrollbar and a fatter font - anything near the pane's
#: own 2,779 px is the defect.
READABLE_MEASURE_MAX = 1250

#: The excerpt is 90 characters. A column that cannot show most of one is a
#: list of clipped fragments, which is what 655 px gave.
ENTRIES_MIN_WIDTH = 1000


def _at_desk_size(qapp, widget):
    """Lay the page out at the trader's screen size, offscreen."""
    widget.resize(*DESK_SIZE)
    widget.show()
    qapp.processEvents()
    return widget


def test_the_reader_is_capped_to_a_readable_measure_at_the_desks_own_size(qapp, panel):
    """The words get a MEASURE, left-aligned, with the slack on the right.

    The pane keeps its full width - the charts below it need it - so this is
    read off the text widget inside the pane, not off the pane.
    """
    _at_desk_size(qapp, panel)
    _render(panel, [_entry("mj-long", LONG_TEXT)])

    pane = panel.thought_view.parent()
    assert pane.width() > READABLE_MEASURE_MAX, (
        "the right half is not wide enough for this measurement to mean anything: "
        f"{pane.width()} px"
    )

    viewport = panel.thought_view.viewport().width()
    assert viewport <= READABLE_MEASURE_MAX, (
        f"the thought is set {viewport} px wide inside a {pane.width()} px pane - "
        "one sentence becomes one very long line"
    )
    assert panel.thought_meta.width() <= READABLE_MEASURE_MAX, panel.thought_meta.width()
    # ...and the reading starts at the LEFT of the pane, with the slack to the
    # right. Centred text with a gutter on each side is not what was asked.
    assert panel.thought_view.x() <= 24, panel.thought_view.x()
    assert panel.thought_meta.x() <= 24, panel.thought_meta.x()
    # A cap is not a hiding place: the column is still a real measure wide.
    assert viewport >= 400, viewport


def test_the_entries_column_opens_wide_enough_to_read_an_excerpt(qapp, panel):
    """One third of the page, and the handle still moves."""
    from PySide6.QtWidgets import QSplitter

    from ui.panels.market_journal_panel import EXCERPT_LIMIT

    _at_desk_size(qapp, panel)
    _render(panel, [_entry("mj-long", LONG_TEXT)])

    assert panel.entries.width() >= ENTRIES_MIN_WIDTH, (
        f"the entries column opens at {panel.entries.width()} px, so a "
        f"{EXCERPT_LIMIT}-character excerpt is clipped to a fragment"
    )

    # Still a splitter, still draggable: move it the other way and the column
    # follows. A fixed third would be a different (and worse) answer.
    lower = panel.entries.parent().parent()
    assert isinstance(lower, QSplitter), type(lower).__name__
    opened_at = panel.entries.width()
    lower.setSizes([2400, 1000])
    qapp.processEvents()
    assert panel.entries.width() > opened_at, "the entries/charts handle no longer moves"


# ==========================================================================
# Packet G3b (2026-09-07) - three reviewer advisories on the reader.
# ==========================================================================
def test_written_line_converts_the_stored_utc_stamp_to_the_desk_zone(monkeypatch):
    """G3b item 1: every live row stores `created_at` in UTC
    (`market_journal.py`'s `moment.astimezone(timezone.utc).isoformat()`), so
    the OLD `_written_line` - which printed the zone the stamp CARRIES - read
    `written 13:36 UTC` for a note typed at 06:36 Pacific, on every live row.
    `astimezone` converts the AWARE side to `pass_bars.desk_zone()`; the zone
    is pinned so this says the same thing on every machine.
    """
    from zoneinfo import ZoneInfo

    import ui.annotations.pass_bars as pass_bars
    from ui.panels.market_journal_panel import _written_line

    monkeypatch.setattr(pass_bars, "desk_zone", lambda: ZoneInfo("America/Los_Angeles"))

    # 13:36 UTC is 06:36 Pacific (PDT, -07:00) in September.
    line = _written_line("2026-09-06T13:36:00+00:00")

    assert "06:36" in line, line
    assert "13:36" not in line, line
    assert "UTC-07:00" in line, line


def test_written_line_never_converts_a_naive_stamp(monkeypatch):
    """A naive stamp carries no zone to convert FROM - guessing one would be
    exactly the quiet backdating this page's whole contract forbids."""
    from zoneinfo import ZoneInfo

    import ui.annotations.pass_bars as pass_bars
    from ui.panels.market_journal_panel import _written_line

    monkeypatch.setattr(pass_bars, "desk_zone", lambda: ZoneInfo("America/Los_Angeles"))

    line = _written_line("2026-09-06T13:36:00")

    assert "no zone recorded" in line, line
    assert "13:36" in line, line


def test_reader_selection_is_keyed_by_entry_id_not_row_index(panel):
    """G3b item 2: `_on_entry_selected` used to look the entry up by the
    QListWidget's ROW INDEX (`_entry_for_row`), which only lines up with
    `self._entries` because every row today IS an entry. Insert a row with no
    `entry_id` at the top - standing in for a future header/grouping row -
    and prove the reader still shows the SELECTED entry's words, not
    whatever `self._entries` happens to hold at that row index.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QListWidgetItem

    _render(
        panel,
        [
            _entry("mj-long", LONG_TEXT, session=SESSION),
            _entry("mj-short", SHORT_TEXT, session=OTHER_SESSION),
        ],
    )
    # Sanity: newest first, so row 0 is the long entry with no header yet.
    assert panel.entries.item(0).data(Qt.UserRole) == "mj-long"

    blocked = panel.entries.blockSignals(True)
    try:
        header = QListWidgetItem("--- a future header row ---")
        header.setData(Qt.UserRole, "")
        panel.entries.insertItem(0, header)
        panel.entries.setCurrentRow(1)
    finally:
        panel.entries.blockSignals(blocked)

    # Row index 1 now HOLDS the long entry (pushed down by the header), while
    # `self._entries[1]` is still the SHORT entry - a row-index lookup would
    # read the wrong one. Signals were blocked above (an insert must not
    # itself repaint anything), so the handler is exercised directly, exactly
    # as `currentRowChanged` would have called it.
    assert panel.entries.currentItem().data(Qt.UserRole) == "mj-long"
    panel._on_entry_selected(1)

    assert panel.thought_view.toPlainText() == LONG_TEXT, panel.thought_view.toPlainText()


def test_refresh_reader_measure_recomputes_the_cap_from_the_current_font(panel):
    """G3b item 3: the reader's pixel cap (`_reader_measure`) was computed
    ONCE in `__init__` and never again, so a scale change left the old cap on
    the page - `MainWindow._apply_scaled_metrics` exists for exactly these
    Python-side pixel budgets and had no seam to call here. A much larger
    font must produce a much wider (still-capped) measure once
    `refresh_reader_measure()` is called.
    """
    small = panel.thought_view.font()
    small.setPointSize(6)
    panel.thought_view.setFont(small)
    panel.thought_meta.setFont(small)
    panel.refresh_reader_measure()
    before = panel.thought_view.maximumWidth()

    large = panel.thought_view.font()
    large.setPointSize(24)
    panel.thought_view.setFont(large)
    panel.thought_meta.setFont(large)
    panel.refresh_reader_measure()
    after = panel.thought_view.maximumWidth()

    assert after != before, (before, after)
    assert panel.thought_meta.maximumWidth() == after, "the meta label keeps the same cap"
