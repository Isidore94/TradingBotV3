"""TJ-3 review round: a mark the tape cannot carry is SAID, and a leg is findable.

Two things the reviewer asked for after measuring the live journal:

* **The page says how many marks were not drawn.** A stamp past the end of the
  tape has no bar (`placement: "after_tape"`), the count is made on the WORKER
  (`day_review_markers.placement_counts`, carried in the payload), and the SPY
  caption and the name pane's caption state it. A session with no tape at all
  says that too, instead of the silence it used to answer with.
* **Both legs of a trade can be found on the chart.** They share one `trade_id`
  - which is what the page SELECTS with, unchanged - so the payload carries a
  `marker_id` (`<trade_id>:in` / `:out`) and `note_marker_position` reaches the
  exit glyph. Clicking a leg still selects nothing in "What you said", because a
  trade is not a note, and it must not raise.
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

from PySide6.QtWidgets import QApplication  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
SESSION = "2026-09-18"
NOW = datetime(2026, 9, 19, 8, 0)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _m5_bars(count: int = 78) -> list[dict]:
    first = datetime(2026, 9, 18, 6, 30, tzinfo=PACIFIC)
    return [
        {
            "dt": first + timedelta(minutes=5 * index),
            "open": 100.0, "high": 100.2, "low": 99.8, "close": 100.1, "volume": 1000,
        }
        for index in range(count)
    ]


def _entry(text: str, created_at: str, entry_id: str):
    import market_journal

    return {
        "event_type": "entry", "entry_id": entry_id, "session_date": SESSION,
        "written_session_date": SESSION, "created_at": created_at,
        "created_local_date": SESSION, "written_after_the_session": False,
        "timeframe": "M5", "symbols": [], "origin": market_journal.ORIGIN_JOURNAL_PAGE,
        "text": text, "supersedes": "", "mentor": {}, "reaffirms": "",
    }


def _marker(ref_id: str, index, kind: str = "note", *, marker_id: str = "",
            placement: str = "on_bar"):
    return {
        "stamp": "", "index": index, "placement": placement, "kind": kind,
        "label": f"{kind} {ref_id}", "ref_id": ref_id,
        "marker_id": marker_id or ref_id,
    }


class _StubService:
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


def _payload(**overrides):
    from ui.services.day_review_service import empty_payload

    payload = empty_payload(SESSION)
    payload["entries"] = [
        _entry("the open was heavy", "2026-09-18T14:03:00+00:00", "e-1"),
        _entry("I like the base here", "2026-09-18T17:12:00+00:00", "e-2"),
    ]
    payload["spy_m5_bars"] = _m5_bars()
    payload["spy_markers"] = (
        _marker("e-1", 6),
        _marker("t-1", None, "trade_close", marker_id="t-1:out", placement="after_tape"),
        _marker("t-2", None, "trade_close", marker_id="t-2:out", placement="after_tape"),
    )
    payload["spy_marker_placements"] = {"on_bar": 1, "after_tape": 2, "between_bars": 0}
    payload.update(overrides)
    return payload


# -- the page says it --------------------------------------------------------


def test_the_spy_caption_says_how_many_marks_the_tape_could_not_carry(panel):
    panel.render(_payload())

    text = panel.spy_note.text()
    assert "2 marks after the tape" in text, text
    assert "not drawn" in text
    # The drawn one is still drawn, and the two unplaced ones are not.
    assert panel._chart.note_marker_count() == 1


def test_one_unplaced_mark_is_singular(panel):
    panel.render(_payload(
        spy_marker_placements={"on_bar": 1, "after_tape": 1, "between_bars": 0}
    ))

    assert "1 mark after the tape" in panel.spy_note.text()


def test_a_session_with_no_tape_says_the_marks_were_not_drawn(panel):
    """It used to answer with silence."""
    from ui.panels.day_review_panel import NO_TAPE_MARKER_NOTE

    panel.render(_payload(spy_m5_bars=[], spy_markers=(), spy_marker_placements={}))

    assert NO_TAPE_MARKER_NOTE in panel.spy_note.text()


def test_a_tape_that_carried_everything_says_nothing_extra(panel):
    panel.render(_payload(
        spy_markers=(_marker("e-1", 6),),
        spy_marker_placements={"on_bar": 1, "after_tape": 0, "between_bars": 0},
    ))

    assert "after the tape" not in panel.spy_note.text()
    assert "not drawn" not in panel.spy_note.text()


def test_the_name_pane_says_it_too(panel, qapp):
    from walkaway_day import WalkawayDay, WalkawayRow

    row = WalkawayRow(
        decision_id=(SESSION, "AAA", "LONG", "chart_review", "veto", "M5", ""),
        time=datetime(2026, 9, 18, 10, 12, tzinfo=PACIFIC),
        symbol="AAA", side="LONG", category="chart_review",
        what_you_did="veto", ran_after_pct=10.0, state="measured",
    )
    payload = _payload(
        walkaway=WalkawayDay(rejected=(row,), sentences={}, skill=None),
        name_charts={
            "AAA": {
                "bars": _m5_bars(30),
                "markers": (_marker("cap-AAA", None, "veto", placement="after_tape"),),
                "placements": {"on_bar": 0, "after_tape": 1, "between_bars": 0},
            }
        },
    )
    panel.render(payload)

    table = panel.miss_table_for("rejected")
    table.itemActivated.emit(table.item(0, 1))
    qapp.processEvents()

    assert panel.name_chart_symbol() == "AAA"
    assert "1 mark after the tape" in panel.name_chart_note.text()
    assert panel._name_chart.note_marker_count() == 0


# -- a leg is findable, and clicking one selects nothing ----------------------


def test_both_legs_of_a_trade_can_be_found_on_the_chart(qapp):
    from ui.widgets.candle_chart import CandleChart

    chart = CandleChart()
    chart.resize(900, 600)
    try:
        chart.set_data(_m5_bars(), timeframe="m5")
        chart.set_note_markers([
            _marker("t-42", 54, "trade_open", marker_id="t-42:in"),
            _marker("t-42", 63, "trade_close", marker_id="t-42:out"),
        ])
        chart.show()
        qapp.processEvents()

        entry_leg = chart.note_marker_position("t-42:in")
        exit_leg = chart.note_marker_position("t-42:out")

        assert entry_leg is not None and exit_leg is not None
        assert entry_leg != exit_leg, "the exit leg cannot be reached by its own name"
        assert entry_leg[0] == 54.0
        assert exit_leg[0] == 63.0
        # The bare selector still answers, with the first leg drawn.
        assert chart.note_marker_position("t-42") == entry_leg
    finally:
        chart.hide()
        chart.deleteLater()
        qapp.processEvents()


def test_clicking_a_trade_leg_selects_no_note_and_does_not_raise(panel):
    panel.render(_payload())
    panel._chart.markerClicked.emit("e-1")
    assert panel.entries.currentRow() == 0

    panel._chart.markerClicked.emit("t-1")
    panel._chart.markerClicked.emit("t-1:out")

    assert panel.entries.currentRow() == 0
    assert panel.entry_reader.toPlainText() == "the open was heavy"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
