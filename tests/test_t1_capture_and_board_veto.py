"""Packet T1.1 - a rail veto retires the chart with NO box and NO second row.

Trader, 2026-09-04:

    "when i double tap something in the capture window (either veto or
    like+claim) i shouldnt get a pop up note box. the point of the capture
    window is to quickly enter 'WHY' I like or dislike something. ... not today
    can continue to go to the next chart with a pop up note box."

What is true on `main` @ 6e05878 and must stop being true: a coded veto on the
capture rail emits `captured(EVENT_VETO)`, `AlertChartReview._on_captured`
forwards it as `removeTodayRequested` - the SAME signal the "✕ Not today"
button emits - and the panel's `_remove_review_alert_for_today` then writes a
SECOND, uncoded veto row through `verdicts.record_not_today` and opens the note
box. One click, two rows, one dialog the trader did not ask for.

The new contract:

* the pane grows `vetoRetireRequested`, and `_on_captured` emits THAT for a
  plain veto; `removeTodayRequested` goes back to being the "✕ Not today"
  BUTTON's signal only;
* the panel's `_retire_after_veto` does everything the "Not today" verb does
  EXCEPT the uncoded row and the box: the `remove_today` review event keeps its
  name (`review_learning.REJECT_ACTIONS` keys on the string), the symbol is
  still parked, the chart still advances;
* the "✕ Not today" button is UNCHANGED - uncoded row, box, advance;
* the day-trade veto is untouched.

Hermetic by construction. Every store is a tmp path, the cohort merges are
neutered (on `main` they default to `C:\\TradingBotData\\data\\runtime\\
veto_cohort_picks.csv` and a capture test really does append to it), and
`verdicts.record_not_today` is replaced by a spy - it writes to the LIVE
`TRADER_ANNOTATIONS_FILE`, because `_record_not_today_annotation` passes no
`path`.
"""

from __future__ import annotations

import json
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

from PySide6.QtWidgets import QApplication, QInputDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _reason_role():
    from ui.widgets import capture_rail as capture_rail_module

    return capture_rail_module._REASON_ROLE


def _neuter_cohort_merges(rail) -> None:
    """The cohort CSVs live in the trader's home folder; tests never write it."""
    rail._merge_veto_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_like_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_pass_cohort = lambda **_kwargs: {"written": True, "added": 0}


def _pick_note_free_reason(rail) -> str:
    """Select the first veto reason that commits on its own (no note required)."""
    role = _reason_role()
    for row in range(rail.reason_list.count()):
        rail.reason_list.setCurrentRow(row)
        if not rail._selected_reason_requires_note():
            return str(rail.reason_list.item(row).data(role))
    raise AssertionError("no note-free veto reason in the loaded vocabulary")


def _d1_alert(symbol: str, side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="08:25:00",
        symbol=symbol,
        side=side,
        trigger=f"({side.lower()}) zone1 reject at AVWAPE",
        timeframe="D1",
        tag=f"d1_flag_{side.lower()}",
        raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} ({side.lower()}) zone1 reject",
        is_d1=True,
    )


@pytest.fixture
def pane(tmp_path, monkeypatch):
    """The review pane alone - the widget that decides which signal a veto is."""
    import pick_feedback
    from ui.widgets.alert_chart_review import AlertChartReview
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    pick_feedback.clear_reviewed_today_cache()
    widget = AlertChartReview(annotations_path=tmp_path / "trader_annotations.jsonl")
    monkeypatch.setattr(widget, "_reviewed_symbols", lambda: set())
    _neuter_cohort_merges(widget.capture_rail)
    widget.set_alert(_d1_alert("AAPL"))
    yield widget
    widget.deleteLater()


@pytest.fixture
def boxes(monkeypatch):
    """Every note box the desk could open on this path, as a recorder.

    `open_note_prompt` is imported INSIDE `_prompt_for_not_today_note`, under a
    bare `except Exception: return` - so a stub that raises would be swallowed
    and prove nothing. A recorder is the only honest probe here.
    """
    import ui.widgets.note_prompt as note_prompt

    opened: list[str] = []
    monkeypatch.setattr(
        note_prompt,
        "open_note_prompt",
        lambda *a, **k: opened.append(str(k.get("title", ""))) or None,
    )
    monkeypatch.setattr(
        QInputDialog,
        "getMultiLineText",
        lambda *a, **k: (opened.append("QInputDialog"), ("", False))[1],
    )
    return opened


@pytest.fixture
def not_today_rows(monkeypatch):
    """`verdicts.record_not_today` writes to the LIVE annotations file.

    `_record_not_today_annotation` calls it with no `path`, so the uncoded row
    never lands in the tmp file the rail writes to. Spying is both the correct
    assertion and the only way to keep this test off `C:\\TradingBotData`.
    """
    from ui.annotations import verdicts

    written: list[dict] = []

    def _spy(*, symbol, side="", session_date=None, timeframe=""):
        row = {
            "symbol": symbol,
            "side": side,
            "timeframe": timeframe,
            "event_type": "veto",
            "reason_code": "",
        }
        written.append(row)
        return row

    monkeypatch.setattr(verdicts, "record_not_today", _spy)
    return written


@pytest.fixture
def panel(tmp_path, monkeypatch):
    """A bare Alert Center with tmp stores and one D1 chart up, one queued."""
    import pick_feedback
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    pick_feedback.clear_reviewed_today_cache()
    made = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    monkeypatch.setattr(made, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(made, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(made, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(made.chart_review, "_reviewed_symbols", lambda: set())
    monkeypatch.setattr(
        made.chart_review.capture_rail,
        "_annotations_path",
        tmp_path / "trader_annotations.jsonl",
    )
    _neuter_cohort_merges(made.chart_review.capture_rail)
    made.add_alert(_d1_alert("AAPL"))
    made.add_alert(_d1_alert("NVDA", "SHORT"))
    assert made._current_review_alert is not None
    assert made._current_review_alert.symbol == "AAPL"
    yield made
    made.close()
    made.deleteLater()


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# the pane: which signal a veto is
# ---------------------------------------------------------------------------
def test_a_reason_double_click_asks_to_retire_and_never_takes_the_not_today_verb(
    pane,
):
    """The trader's double-tap is `reason_list.itemActivated` -> `commit_veto`.

    A veto and a "Not today" are two different verbs with two different rows
    and only one of them has a box, so they may not share a signal.
    """
    retire: list = []
    not_today: list = []
    pane.vetoRetireRequested.connect(retire.append)
    pane.removeTodayRequested.connect(not_today.append)

    _pick_note_free_reason(pane.capture_rail)
    pane.capture_rail.reason_list.itemActivated.emit(
        pane.capture_rail.reason_list.currentItem()
    )

    assert [alert.symbol for alert in retire] == ["AAPL"]
    assert not_today == [], "a rail veto must not fire the Not-today button's signal"


def test_the_day_trade_veto_still_asks_for_placement_and_never_retires(pane):
    """`veto_keeps_chart()` is True for one commit; the new signal obeys it too."""
    retire: list = []
    not_today: list = []
    day_trade: list = []
    pane.vetoRetireRequested.connect(retire.append)
    pane.removeTodayRequested.connect(not_today.append)
    pane.vetoDayTradeRequested.connect(day_trade.append)

    _pick_note_free_reason(pane.capture_rail)
    assert pane.capture_rail.commit_veto_day_trade() is not None

    assert [alert.symbol for alert in day_trade] == ["AAPL"]
    assert retire == [], "the host places the name BEFORE the chart is retired"
    assert not_today == []


# ---------------------------------------------------------------------------
# the panel: one row, no box, everything else unchanged
# ---------------------------------------------------------------------------
def test_a_rail_veto_writes_exactly_one_annotation_row_and_no_uncoded_second(
    panel, boxes, not_today_rows, tmp_path
):
    """One veto click is ONE veto row, and it is the coded one."""
    rail = panel.chart_review.capture_rail
    code = _pick_note_free_reason(rail)
    rail.reason_list.itemActivated.emit(rail.reason_list.currentItem())

    coded = _rows(tmp_path / "trader_annotations.jsonl")
    assert [row["event_type"] for row in coded] == ["veto"]
    assert coded[0]["symbol"] == "AAPL"
    assert coded[0]["reason_code"] == code
    assert not_today_rows == [], "no uncoded 'Not today' row may be written too"


def test_a_rail_veto_opens_no_note_box(panel, boxes, not_today_rows):
    """The capture window IS the why. A box after it is the thing the trader cut."""
    rail = panel.chart_review.capture_rail
    _pick_note_free_reason(rail)
    rail.reason_list.itemActivated.emit(rail.reason_list.currentItem())

    assert boxes == [], f"a rail veto opened {boxes}"


def test_a_rail_veto_still_records_remove_today_parks_the_symbol_and_advances(
    panel, boxes, not_today_rows, monkeypatch
):
    """Everything the retirement already did keeps happening.

    `remove_today` is NOT renamed - `review_learning.REJECT_ACTIONS` keys on
    that exact string, and renaming it would silently drop every historical
    rejection out of the scoreboard.
    """
    recorded: list[str] = []
    monkeypatch.setattr(
        panel, "_record_review_event", lambda action, **kw: recorded.append(action)
    )
    rail = panel.chart_review.capture_rail
    _pick_note_free_reason(rail)
    rail.reason_list.itemActivated.emit(rail.reason_list.currentItem())

    assert "remove_today" in recorded
    assert "AAPL" in panel._ignored_symbols
    assert panel._current_review_alert is not None
    assert panel._current_review_alert.symbol == "NVDA", "the chart moved on"


def test_the_not_today_button_still_writes_the_uncoded_row_and_opens_the_box(
    panel, boxes, not_today_rows, tmp_path
):
    """The trader kept this one in so many words: "not today can continue to
    go to the next chart with a pop up note box"."""
    panel.chart_review.remove_today_button.click()

    assert [row["symbol"] for row in not_today_rows] == ["AAPL"]
    assert not_today_rows[0]["reason_code"] == "", "the button has no picklist"
    assert len(boxes) == 1, f"the Not-today box must still open; saw {boxes}"
    assert _rows(tmp_path / "trader_annotations.jsonl") == [], (
        "the button writes no CODED row - it has no reason list"
    )
    assert "AAPL" in panel._ignored_symbols
    assert panel._current_review_alert.symbol == "NVDA"


def test_the_two_verbs_share_one_body_rather_than_two_branch_ladders(
    panel, monkeypatch
):
    """Both retirement verbs reach ONE body, and differ only in the flag.

    The packet's rule is one private body with a flag (or two thin wrappers),
    never a copied branch ladder - the auto-pick / faded / Focus-review
    branches each return early, and a copy that lost one would silently start
    parking symbols that must not be parked.

    Rewritten in fix round 1: the earlier version asserted only that the two
    names existed and were different objects, which is true of two copied
    ladders as well - it could not fail for the thing it was named after.
    """
    assert callable(getattr(panel, "_retire_after_veto", None))
    assert callable(getattr(panel, "_remove_review_alert_for_today", None))
    assert panel._retire_after_veto is not panel._remove_review_alert_for_today

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        panel,
        "_retire_review_alert",
        lambda alert, *, write_not_today_annotation: calls.append(
            (alert.symbol, write_not_today_annotation)
        ),
    )

    alert = _d1_alert("AAPL")
    panel._retire_after_veto(alert)
    panel._remove_review_alert_for_today(alert)

    assert calls == [("AAPL", False), ("AAPL", True)], (
        "both verbs must reach the shared body, and the FLAG is the whole "
        "difference between them"
    )
