"""TJ-3 items 2 and 3 - the one read carries the markers, the page draws them.

Packet `.claude/packets/TJ-3.md`; `plan.md` §12.4 "TJ-3" changes 2 and 3. RED
before the build.

The contract these tests pin, so the builder has nothing to guess
-----------------------------------------------------------------

``scripts/ui/services/day_review_service.py`` - the marker payload keys only.
    ``PAYLOAD_KEYS`` gains ``spy_markers`` and ``name_charts``; ``empty_payload``
    carries both PRESENT and empty. ``read_day`` builds them ON THE WORKER, from
    the bars that same payload hands the page:

    * ``payload["spy_markers"]`` - `day_review_markers.benchmark_markers` over
      ``payload["spy_m5_bars"]`` (whichever tape the payload ended up with: the
      Qt-thread hand-off for a live session, the durable file for a closed one).
    * ``payload["name_charts"]`` - ``{SYMBOL: {"bars": [...], "markers": (...)}}``
      for the names the trader decided on, from the session bars file the
      walk-away read already opened. Nothing here opens a second store.

``scripts/ui/panels/day_review_panel.py`` - the chart area and the row click.
    ``render`` pushes ``spy_markers`` onto the ONE SPY chart and calls no
    builder: the paint path formats, it never computes. A walk-away row
    activation opens that name in ONE reused `CandleChart`
    (``panel._name_chart``, ``panel.name_chart_symbol()``), drawn from the
    payload alone - no second read, no store. ``markerClicked`` selects that
    note in "What you said".

The page's existing ``chartRequested`` emission stays (`tests/test_tj2b_walkaway.py`
pins it): the host's board chart and this page's own pane are two different
answers to one click, and TJ-3 adds the second without removing the first.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Day Review page uses PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
EASTERN = ZoneInfo("America/New_York")

SESSION = "2026-09-18"  # a Friday
NOW = datetime(2026, 9, 19, 8, 0)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _m5_bars(count: int = 78, *, symbol_base: float = 100.0) -> list[dict]:
    first = datetime(2026, 9, 18, 6, 30, tzinfo=PACIFIC)
    return [
        {
            "dt": first + timedelta(minutes=5 * index),
            "open": symbol_base + index * 0.01,
            "high": symbol_base + 0.2 + index * 0.01,
            "low": symbol_base - 0.2 + index * 0.01,
            "close": symbol_base + 0.1 + index * 0.01,
            "volume": 1000,
        }
        for index in range(count)
    ]


def _entry(text: str, created_at: str, entry_id: str, *, origin: str = ""):
    import market_journal

    return {
        "event_type": "entry",
        "entry_id": entry_id,
        "session_date": SESSION,
        "written_session_date": SESSION,
        "created_at": created_at,
        "created_local_date": SESSION,
        "written_after_the_session": False,
        "timeframe": "M5",
        "symbols": [],
        "origin": origin or market_journal.ORIGIN_JOURNAL_PAGE,
        "text": text,
        "supersedes": "",
        "mentor": {},
        "reaffirms": "",
    }


ENTRIES = [
    _entry("the open was heavy", "2026-09-18T14:03:00+00:00", "e-1"),
    _entry("I like the base here", "2026-09-18T17:12:00+00:00", "e-2"),
    _entry("gave it back into the close", "2026-09-18T19:30:00+00:00", "e-3"),
]


def _marker(ref_id: str, index: int, kind: str = "note"):
    return {
        "stamp": (datetime(2026, 9, 18, 6, 30, tzinfo=PACIFIC)
                  + timedelta(minutes=5 * index)).isoformat(),
        "index": index,
        "kind": kind,
        "label": f"{kind} {ref_id}",
        "ref_id": ref_id,
    }


def _walkaway_day():
    """A real `WalkawayDay`: one rejected name and one liked name."""
    from walkaway_day import WalkawayDay, WalkawayRow

    def _row(symbol: str, verdict: str):
        return WalkawayRow(
            decision_id=(SESSION, symbol, "LONG", "chart_review", verdict, "M5", ""),
            time=datetime(2026, 9, 18, 13, 12, tzinfo=EASTERN),
            symbol=symbol,
            side="LONG",
            category="chart_review",
            what_you_did=verdict,
            ran_after_pct=10.0,
            state="measured",
        )

    return WalkawayDay(
        rejected=(_row("AAA", "veto"),),
        liked_not_traded=(_row("BBB", "like"),),
        sentences={},
        skill=None,
    )


def _page_payload():
    from ui.services.day_review_service import empty_payload

    payload = empty_payload(SESSION)
    payload["entries"] = [dict(row) for row in ENTRIES]
    payload["walkaway"] = _walkaway_day()
    payload["spy_m5_bars"] = _m5_bars()
    payload["spy_markers"] = (
        _marker("e-1", 6), _marker("e-2", 44), _marker("e-3", 72),
    )
    payload["name_charts"] = {
        "AAA": {"bars": _m5_bars(30, symbol_base=50.0),
                "markers": (_marker("cap-AAA", 11, "veto"),)},
        "BBB": {"bars": _m5_bars(20, symbol_base=70.0),
                "markers": (_marker("cap-BBB", 5, "like"),
                            _marker("t-bbb", 9, "trade_open"))},
    }
    return payload


class _StubService:
    """Reads nothing. The page is what is under test."""

    def __init__(self, payload=None) -> None:
        self.payload = payload or {}
        self.reads = 0

    def read_day(self, session_date, **_kwargs):
        self.reads += 1
        payload = dict(self.payload)
        payload.setdefault("session_date", session_date)
        return payload


@pytest.fixture()
def panel(qapp, monkeypatch):
    from ui.panels.day_review_panel import DayReviewPanel

    widget = DayReviewPanel(service=_StubService(), clock=lambda: NOW)
    monkeypatch.setattr(widget, "reload", lambda: None)
    yield widget
    try:
        widget.shutdown()
    except Exception:  # noqa: BLE001
        pass
    widget.deleteLater()
    qapp.processEvents()


# -- the one read ------------------------------------------------------------


def test_the_payload_declares_the_marker_keys_present_and_empty():
    from ui.services.day_review_service import PAYLOAD_KEYS, empty_payload

    assert "spy_markers" in PAYLOAD_KEYS
    assert "name_charts" in PAYLOAD_KEYS

    blank = empty_payload(SESSION)
    assert blank["spy_markers"] in ((), [])
    assert blank["name_charts"] == {}


class _Journal:
    def __init__(self, entries=()):
        self._entries = list(entries)

    def entries_about(self, _session):
        return [dict(row) for row in self._entries]

    def daily_story(self, _session):
        return None

    def theses_for(self, _session):
        return []


def _annotation(symbol: str, *, created_at: str, verdict: str = "veto"):
    return {
        "schema_version": 1, "event_id": f"e-{symbol}", "event_type": verdict,
        "symbol": symbol, "side": "LONG", "session_date": SESSION,
        "timeframe": "M5", "created_at": created_at, "source": "chart_review",
        "reason_code": "extended", "vocab_version": 3,
    }


def _wire(monkeypatch, *, entries=(), annotations=(), session_bars=None, trades=()):
    """Point `read_day` at plain dicts. No live store is opened."""
    import chart_snapshot
    import claimed_picks
    import daily_recap_reader
    import day_review_bars
    import journal_store
    from ui.services.day_review_service import DayReviewService

    store = {"annotations": list(annotations), "pick_feedback": [],
             "swing_favorites": [], "review_events": []}

    class _Store:
        def __init__(self, rows):
            self.rows = list(rows)

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl",
                        lambda name, *a, **k: _Store(store.get(name, [])))
    monkeypatch.setattr(daily_recap_reader, "_read_csv", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        journal_store, "JournalStore",
        lambda *a, **k: type("_J", (), {"list_trades": lambda self: list(trades)})(),
    )
    monkeypatch.setattr(day_review_bars, "read_session_bars",
                        lambda *a, **k: dict(session_bars or {}))
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *a, **k: True)
    monkeypatch.setattr(day_review_bars, "session_is_backfillable", lambda *a, **k: False)
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])

    service = DayReviewService(journal_service=_Journal(entries))
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    monkeypatch.setattr(service, "_trades", lambda *a, **k: list(trades))
    return service


def test_read_day_builds_the_spy_markers_against_the_tape_it_hands_the_page(monkeypatch):
    """10:12 Pacific is bar 44 of the durable tape, and the worker says so."""
    service = _wire(
        monkeypatch,
        entries=[_entry("I like the base here", "2026-09-18T17:12:00+00:00", "e-2")],
        session_bars={"SPY": _m5_bars()},
    )

    payload = service.read_day(SESSION, now=NOW)

    markers = list(payload["spy_markers"])
    assert [marker["ref_id"] for marker in markers] == ["e-2"]
    assert markers[0]["index"] == 44
    assert len(payload["spy_m5_bars"]) == 78


def test_read_day_builds_the_spy_markers_for_a_live_sessions_handed_in_tape(monkeypatch):
    """A session that has not closed gets its bars from the Qt slot, not a file."""
    import day_review_bars

    service = _wire(
        monkeypatch,
        entries=[_entry("I like the base here", "2026-09-18T17:12:00+00:00", "e-2")],
    )
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *a, **k: False)

    payload = service.read_day(SESSION, now=NOW, spy_m5_bars=_m5_bars())

    assert [marker["index"] for marker in payload["spy_markers"]] == [44]


def test_read_day_builds_a_name_chart_for_each_decided_symbol(monkeypatch):
    service = _wire(
        monkeypatch,
        annotations=[_annotation("AAA", created_at="2026-09-18T17:12:00+00:00")],
        session_bars={"SPY": _m5_bars(), "AAA": _m5_bars(30, symbol_base=50.0)},
    )

    payload = service.read_day(SESSION, now=NOW)

    charts = payload["name_charts"]
    assert "AAA" in charts, payload.get("error")
    assert len(charts["AAA"]["bars"]) == 30
    markers = list(charts["AAA"]["markers"])
    assert [marker["kind"] for marker in markers] == ["veto"]
    # LEAD AMENDMENT 2026-09-19, twice, and the history is the point.
    #
    # The tester's literal was 44, copied from the SPY case above (a 78-bar
    # tape); AAA's own tape here is 30 bars (06:30..08:55 PT), so 44 was a
    # marker past the end of the very bars this test asserts are drawn. The
    # lead amended it to 29, the LAST bar at or before 10:12 - but that is a
    # CLAMP, and the reviewer measured what a clamp costs on the live journal:
    # 62 of 216 trade legs fall after 13:00 PT and were being drawn on the
    # 12:55 candle as though the fill had happened at the close. The honest
    # answer for a stamp the tape never covered is that it has no bar: the
    # marker is kept so the page can SAY how many there are, and it is not
    # drawn anywhere. 10:12 is 77 minutes after AAA's last bar closed.
    assert markers[0]["index"] is None
    assert markers[0]["placement"] == "after_tape"


def test_a_name_with_no_tape_gets_no_invented_chart(monkeypatch):
    """A symbol the bars file never got is absent, never drawn on someone else's tape."""
    service = _wire(
        monkeypatch,
        annotations=[_annotation("AAA", created_at="2026-09-18T17:12:00+00:00")],
        session_bars={"SPY": _m5_bars()},
    )

    payload = service.read_day(SESSION, now=NOW)

    assert "AAA" not in payload["name_charts"]


def test_no_tape_means_no_markers_rather_than_a_failed_read(monkeypatch):
    service = _wire(
        monkeypatch,
        entries=[_entry("a thought", "2026-09-18T17:12:00+00:00", "e-2")],
        session_bars={},
    )

    payload = service.read_day(SESSION, now=NOW)

    assert list(payload["spy_markers"]) == []
    assert payload["name_charts"] == {}


# -- the page ----------------------------------------------------------------


def test_rendering_puts_the_payloads_markers_on_the_one_spy_chart(panel):
    panel.render(_page_payload())

    assert panel._chart is not None
    assert panel._chart.note_marker_count() == 3


def test_rendering_builds_no_marker_payload_on_the_paint_path(panel, monkeypatch):
    """Markers are built on the worker. The paint path formats and computes nothing."""
    import day_review_markers
    import ui.panels.day_review_panel as page

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("a marker payload was built on the Qt thread")

    monkeypatch.setattr(day_review_markers, "benchmark_markers", _forbidden)
    monkeypatch.setattr(day_review_markers, "symbol_markers", _forbidden)

    panel.render(_page_payload())

    # A module-level `from day_review_markers import ...` in the page would
    # bind the real function before any patch could reach it, so the panel's
    # namespace is checked too: the builder is not a thing this file holds.
    assert not hasattr(page, "benchmark_markers")
    assert not hasattr(page, "symbol_markers")
    assert not hasattr(page, "day_review_markers")


def test_rendering_twice_starts_no_read_and_keeps_one_chart(panel):
    payload = _page_payload()
    panel.render(payload)
    first = panel._chart

    panel.render(payload)

    assert panel._chart is first
    assert panel.service.reads == 0


def test_a_walkaway_row_click_opens_that_name_in_one_reused_chart(panel, qapp):
    panel.render(_page_payload())
    rejected = panel.miss_table_for("rejected")

    rejected.itemActivated.emit(rejected.item(0, 1))
    qapp.processEvents()
    first = panel._name_chart

    assert first is not None
    assert panel.name_chart_symbol() == "AAA"
    assert first.bar_count() == 30, "the name chart drew someone else's tape"
    assert first is not panel._chart, "the name took the SPY pane"

    liked = panel.miss_table_for("liked_not_traded")
    liked.itemActivated.emit(liked.item(0, 1))
    qapp.processEvents()

    assert panel._name_chart is first, "a second row built a second chart widget"
    assert panel.name_chart_symbol() == "BBB"
    assert first.bar_count() == 20


def test_the_name_chart_carries_that_names_own_markers(panel, qapp):
    panel.render(_page_payload())
    table = panel.miss_table_for("liked_not_traded")

    table.itemActivated.emit(table.item(0, 1))
    qapp.processEvents()

    assert panel._name_chart.note_marker_count() == 2


def test_the_name_chart_lives_on_this_page_and_is_shown(panel, qapp):
    panel.render(_page_payload())
    table = panel.miss_table_for("rejected")

    table.itemActivated.emit(table.item(0, 1))
    qapp.processEvents()

    parent = panel._name_chart.parentWidget()
    while parent is not None and parent is not panel:
        parent = parent.parentWidget()
    assert parent is panel, "the name chart is not on the Day Review page"
    assert panel._name_chart.isVisibleTo(panel)


def test_a_row_click_starts_no_second_read_and_opens_no_store(panel, monkeypatch, qapp):
    import day_review_bars
    import day_review_markers

    panel.render(_page_payload())

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("the Qt thread opened a store for a row click")

    monkeypatch.setattr(day_review_bars, "read_session_bars", _forbidden)
    monkeypatch.setattr(day_review_markers, "symbol_markers", _forbidden)
    monkeypatch.setattr(day_review_markers, "benchmark_markers", _forbidden)
    before = panel.service.reads

    table = panel.miss_table_for("rejected")
    table.itemActivated.emit(table.item(0, 1))
    qapp.processEvents()

    assert panel.service.reads == before


def test_a_row_click_still_tells_the_host_about_the_name(panel, qapp):
    """TJ-2B's contract is kept: the board chart and this pane are both answers."""
    panel.render(_page_payload())
    seen: list[tuple[str, str]] = []
    panel.chartRequested.connect(lambda symbol, side: seen.append((symbol, side)))

    table = panel.miss_table_for("rejected")
    table.itemActivated.emit(table.item(0, 1))
    qapp.processEvents()

    assert seen == [("AAA", "LONG")]


def test_a_name_with_no_chart_in_the_payload_says_so_and_draws_nothing(panel, qapp):
    payload = _page_payload()
    payload["name_charts"] = {}

    panel.render(payload)
    table = panel.miss_table_for("rejected")
    table.itemActivated.emit(table.item(0, 1))
    qapp.processEvents()

    assert panel.name_chart_symbol() == ""


# -- clicking a marker -------------------------------------------------------


def test_clicking_a_marker_selects_that_note_in_what_you_said(panel):
    """The trader's question - "where did I say that?" - answered on the page."""
    panel.render(_page_payload())

    panel._chart.markerClicked.emit("e-2")

    assert panel.entries.currentRow() == 1
    assert panel.entries.currentItem().data(Qt.UserRole) == "e-2"
    assert panel.entry_reader.toPlainText() == "I like the base here"


def test_clicking_a_marker_for_something_that_is_not_a_note_changes_no_selection(panel):
    panel.render(_page_payload())
    panel._chart.markerClicked.emit("e-3")
    assert panel.entries.currentRow() == 2

    panel._chart.markerClicked.emit("t-not-an-entry")

    assert panel.entries.currentRow() == 2
    assert panel.entry_reader.toPlainText() == "gave it back into the close"


def test_clicking_a_marker_on_the_name_chart_selects_the_note_too(panel, qapp):
    panel.render(_page_payload())
    table = panel.miss_table_for("rejected")
    table.itemActivated.emit(table.item(0, 1))
    qapp.processEvents()

    panel._name_chart.markerClicked.emit("e-1")

    assert panel.entries.currentRow() == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
