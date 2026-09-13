"""Packet WS-10D, builder-added: the Story pane's source LINKS, and its order.

Two things the tester's file could not pin, added here rather than edited into
it (a tester's assertion is never rewritten by the builder):

1. **The sources are links and a click selects what they name.** The packet
   asks for "sources as links that select the entry/capture"; nothing in
   `tests/test_ws_10d_market_story.py` drives the anchor, so a Story pane that
   printed the ids as dead text would have passed every test there.
2. **Every benchmark is named between "The market did" and "Sources"** - the
   assertion `test_the_story_pane_shows_you_said_then_the_market_did_then_the_sources`
   was reaching for with ``measured < body.index("SPY") < sources``. That one
   cannot pass as written: the same test requires the trader's note - which is
   `"SPY reclaimed the anchor and I finally sized up."` - to appear under "You
   said", so the FIRST "SPY" in the document is always inside the note, before
   the measured heading. `str.index` takes the first occurrence, so the two
   assertions contradict each other. Searching FROM the measured heading is the
   check that was meant, and it is the one below. The tester's line is left red
   rather than weakened (`docs/AGENT_TEAM.md`: the builder may add, never
   rewrite).

Nothing here touches a live store: the service is a stub and every path is
`tmp_path`.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")
SESSION = "2026-09-11"
NOTE = "SPY reclaimed the anchor and I finally sized up."
SECOND_NOTE = "QQQ never confirmed it, so I stayed small."


def _flat_bars() -> list[dict]:
    from tests.test_ws_10d_market_story import _flat_bars as bars

    return bars()


def _entry(text: str, *, hour: int):
    import market_journal

    return market_journal.build_entry(
        text=text,
        session_date=SESSION,
        timeframe="D1",
        symbols=("SPY",),
        origin=market_journal.ORIGIN_JOURNAL_PAGE,
        now=datetime(2026, 9, 11, hour, 0, tzinfo=PACIFIC),
    )


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6", reason="the Market Journal page is Qt")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp):
    import market_story
    from PySide6.QtCore import QObject, Signal

    from ui.panels.market_journal_panel import MarketJournalPanel

    first = _entry(NOTE, hour=7)
    second = _entry(SECOND_NOTE, hour=11)
    story = market_story.build_daily_story(
        SESSION, entries=(first, second), index_bars={"SPY": _flat_bars()}
    )

    class _Stub(QObject):
        statusChanged = Signal(str)
        entryWritten = Signal(dict)
        chartCaptured = Signal(dict)

        def entries_for(self, session_date: str = ""):
            return [dict(row) for row in story.trader_said]

        def sessions_with_entries(self):
            return [SESSION]

        def regime_timeline(self, **_kwargs):
            return {"shifts": [], "agreement": {"rate": None, "note": "none"}}

        def day_context(self, _session):
            return {"measured": False, "reason": "not measured"}

        def chart_digests(self):
            return {}

        def chart_capture(self, entry_id: str):
            return None

        def daily_story(self, session_date: str, **_kwargs):
            return story

        def theses_for(self, session_date: str = ""):
            return []

        def save_interpretation(self, **_kwargs):
            return {"ok": True}

    widget = MarketJournalPanel(service=_Stub())
    widget._render(
        {
            "session_date": SESSION,
            "entries": [dict(row) for row in story.trader_said],
            "sessions": [SESSION],
            "timeline": {"shifts": [], "agreement": {"rate": None, "note": "none"}},
            "context": {"measured": False, "reason": "not measured"},
            "digests": {},
            "story": story,
            "theses": [],
        }
    )
    widget._story = story
    yield widget
    widget.shutdown()
    widget.deleteLater()


@pytest.mark.qt
def test_every_benchmark_is_named_between_the_measured_and_sources_headings(panel):
    """The measured part names all six, including the five with no bars.

    Searched FROM the measured heading, because the trader's own note contains
    a ticker and `str.index` would find that one first. An unmeasured benchmark
    that is simply left out reads as "nothing happened there", which is a claim
    nobody measured.
    """
    from ui.panels import market_journal_panel as page

    body = panel.story_view.toPlainText()
    measured = body.index(page.STORY_MEASURED_HEADING)
    sources = body.index(page.STORY_SOURCES_HEADING)
    assert measured < sources

    block = body[measured:sources]
    for symbol in ("SPY", "QQQ", "IWM", "VXX", "TLT", "USO"):
        assert symbol in block, (symbol, block)
    assert "unmeasured" in block, block


@pytest.mark.qt
def test_a_source_link_selects_the_entry_it_names(panel):
    """Clicking a source moves the selection to that entry, and its charts.

    The story lists its sources by id so the trader can get from a sentence
    back to the thing it was built from; ids printed as dead text would pass
    every other test in this packet and answer nothing.
    """
    from PySide6.QtCore import QUrl
    from ui.panels import market_journal_panel as page

    ids = [str(row["entry_id"]) for row in panel._story.trader_said]
    assert len(ids) == 2

    body = panel.story_view.toPlainText()
    for entry_id in ids:
        assert entry_id in body[body.index(page.STORY_SOURCES_HEADING):]

    # The list opens on the newest entry (row 0 is newest-first), so clicking
    # the OLDER source is a real move rather than a no-op.
    panel._on_story_anchor(QUrl(f"{page.STORY_ENTRY_SCHEME}:{ids[0]}"))
    assert panel._selected_entry_id() == ids[0]
    assert panel.thought_view.toPlainText() == NOTE

    panel._on_story_anchor(QUrl(f"{page.STORY_ENTRY_SCHEME}:{ids[1]}"))
    assert panel._selected_entry_id() == ids[1]
    assert panel.thought_view.toPlainText() == SECOND_NOTE


@pytest.mark.qt
def test_the_paste_action_imports_a_forecast_and_invents_no_provenance(qapp):
    """WISHLIST 10K's button: the text goes through, the blanks stay blank.

    The dialog is not driven here (a modal `exec` in a test is a hang); its
    write half is, because that is where a field could be quietly filled in.
    An empty "written at" must reach the service EMPTY - it is
    `market_thesis.record_forecast` that turns it into `unknown`, in one place.
    """
    from PySide6.QtCore import QObject, Signal

    from ui.panels.market_journal_panel import MarketJournalPanel

    class _Stub(QObject):
        statusChanged = Signal(str)
        entryWritten = Signal(dict)
        chartCaptured = Signal(dict)

        def __init__(self):
            super().__init__()
            self.imported: list[dict] = []

        def entries_for(self, session_date: str = ""):
            return []

        def import_weekly_forecast(self, **kwargs):
            self.imported.append(kwargs)
            return {"ok": True, "entry": {"entry_id": "mj-x"}, "forecast": {}}

    widget = MarketJournalPanel(service=_Stub())
    try:
        result = widget._import_forecast(
            {
                "text": "Week of Sept 14: base case SPY grinds to 5,500.",
                "source_model": "",
                "created_at_claimed": "",
                "target_week": "2026-W38",
                "scenarios": ["base: 5,500"],
            }
        )
        assert result["ok"] is True
        assert len(widget.service.imported) == 1
        sent = widget.service.imported[0]
        assert sent["text"].startswith("Week of Sept 14")
        assert sent["source_model"] == ""
        assert sent["created_at_claimed"] == ""
        assert sent["target_week"] == "2026-W38"
        assert tuple(sent["scenarios"]) == ("base: 5,500",)

        # Nothing pasted, nothing written.
        assert widget._import_forecast({"text": "   "})["ok"] is False
        assert len(widget.service.imported) == 1
    finally:
        widget.shutdown()
        widget.deleteLater()


@pytest.mark.qt
def test_an_unknown_source_link_says_so_and_moves_nothing(panel):
    """A link to an entry the list does not hold never silently changes the row."""
    from PySide6.QtCore import QUrl
    from ui.panels import market_journal_panel as page

    before = panel._selected_entry_id()
    panel._on_story_anchor(QUrl(f"{page.STORY_ENTRY_SCHEME}:mj-nope"))

    assert panel._selected_entry_id() == before
    assert "mj-nope" in panel.status.text()
