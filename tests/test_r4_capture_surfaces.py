"""R4 A5 - every screen that carries a verdict writes one, and names itself.

P10 declared five `surface` values and wired three. `focus_panel` and
`m5_alert_bar` were constants with no writer anywhere in the tree, so two of the
five columns of "which screen is the trader a better judge from?" could never be
populated - and the chart-review hosts never called the `surface` override that
`CaptureRail.set_scan_context` has carried since P10 B1, so a verdict passed on a
review chart filed as `rail`, indistinguishable from one typed on the rail
itself.

One test per REAL click handler. Every one of them drives the handler the trader's
click reaches, against a temp annotation file, and reads the row off disk.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    existing = QApplication.instance()
    yield existing or QApplication([])


# ---------------------------------------------------------------------------
# The Focus board's two verbs
# ---------------------------------------------------------------------------


class _StubFocusService:
    """Only what the editor asks of it, and it remembers what was asked."""

    def __init__(self, *, auto_adopted: set[str] | None = None) -> None:
        self.auto_adopted = set(auto_adopted or ())
        self.removed: list[tuple] = []

    def focus_symbols(self, side, category="m5"):
        return []

    def remove_if_auto_adopted(self, symbol, side, category="m5"):
        if str(symbol).upper() not in self.auto_adopted:
            return False
        self.removed.append((str(symbol).upper(), side, category))
        return True


def _editor(tmp_path, *, side="long", category="m5", service=None):
    from ui.panels.focus_picks_panel import FocusSideEditor

    return FocusSideEditor(
        "Test - Longs",
        side,
        category,
        service or _StubFocusService(),
        lambda symbol, editor_side="": {"bounce": {}, "rrs": {}, "mover": ""},
        tone="long",
        annotations_path=tmp_path / "trader_annotations.jsonl",
    )


def test_a_like_on_the_focus_board_writes_a_row_naming_the_focus_panel(app, tmp_path):
    editor = _editor(tmp_path)

    editor._like("NVDA")

    rows = _rows(tmp_path / "trader_annotations.jsonl")
    assert len(rows) == 1
    assert rows[0]["surface"] == "focus_panel"
    assert rows[0]["event_type"] == "like_claim"
    assert rows[0]["symbol"] == "NVDA"
    assert rows[0]["side"] == "LONG"
    assert rows[0]["timeframe"] == "M5"
    assert rows[0]["like_mode"] == "quick"


def test_a_swing_focus_like_says_D1_and_a_short_editor_says_SHORT(app, tmp_path):
    """The row's side and timeframe come from the board it was clicked on."""
    editor = _editor(tmp_path, side="short", category="swing")

    editor._like("AMD")

    row = _rows(tmp_path / "trader_annotations.jsonl")[0]
    assert row["side"] == "SHORT"
    assert row["timeframe"] == "D1"


def test_not_today_on_the_focus_board_records_and_drops_only_an_auto_pick(app, tmp_path):
    service = _StubFocusService(auto_adopted={"NVDA"})
    editor = _editor(tmp_path, service=service)

    editor._not_today("NVDA")

    row = _rows(tmp_path / "trader_annotations.jsonl")[0]
    assert row["event_type"] == "veto"
    assert row["surface"] == "focus_panel"
    # An uncoded veto carries no vocabulary version - a version without a code
    # would pool it into somebody else's cohort forever.
    assert "reason_code" not in row and "vocab_version" not in row
    assert service.removed == [("NVDA", "long", "m5")]


def test_not_today_on_a_name_the_trader_typed_records_it_and_removes_nothing(
    app, tmp_path
):
    """User-entered watchlist names are never automatically removed (sec 5)."""
    service = _StubFocusService(auto_adopted=set())
    editor = _editor(tmp_path, service=service)

    editor._not_today("HAND")

    assert len(_rows(tmp_path / "trader_annotations.jsonl")) == 1
    assert service.removed == []
    assert "stays on the list" in editor.status_label.text()


def test_the_focus_chips_menu_verbs_reach_the_editors_writers(app, tmp_path):
    """The signal path a right-click actually travels, end to end."""
    from ui.panels.focus_picks_panel import FocusStatusChip

    editor = _editor(tmp_path)
    chip = FocusStatusChip("TSLA", tone="long", state={"bounce": {}, "rrs": {}, "mover": ""})
    chip.likeRequested.connect(editor._like)
    chip.notTodayRequested.connect(editor._not_today)

    chip.likeRequested.emit(chip.symbol)
    chip.notTodayRequested.emit(chip.symbol)

    rows = _rows(tmp_path / "trader_annotations.jsonl")
    assert [row["event_type"] for row in rows] == ["like_claim", "veto"]
    assert {row["surface"] for row in rows} == {"focus_panel"}


# ---------------------------------------------------------------------------
# The M5 alert bar
# ---------------------------------------------------------------------------


class _Alert:
    def __init__(self, symbol="NVDA", side="LONG"):
        self.symbol = symbol
        self.side = side
        self.timeframe = "M5"
        self.raw_text = ""
        self.alert_type = "vwap_bounce"
        self.trigger = "vwap bounce"


def test_a_quick_like_on_the_m5_bar_writes_a_row_naming_the_bar(app, tmp_path):
    from ui.widgets.m5_alert_bar import M5AlertBar

    path = tmp_path / "trader_annotations.jsonl"
    bar = M5AlertBar(annotations_path=path)
    bar.post(_Alert())

    written = bar.quick_like(bar.list.item(0))

    assert written is not None
    row = _rows(path)[0]
    assert row["surface"] == "m5_alert_bar"
    assert row["event_type"] == "like_claim"
    assert row["like_mode"] == "quick"
    assert row["symbol"] == "NVDA"


def test_a_quick_like_leaves_the_bar_exactly_as_it_was(app, tmp_path):
    """A capture is not a control: the row stays, the click-through still works."""
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar(annotations_path=tmp_path / "trader_annotations.jsonl")
    bar.post(_Alert())
    bar.post(_Alert(symbol="AMD", side="SHORT"))
    before = bar.symbols()

    bar.quick_like(bar.list.item(0))

    assert bar.symbols() == before
    assert bar.count() == 2


def test_a_quick_like_with_no_row_under_it_writes_nothing(app, tmp_path):
    from ui.widgets.m5_alert_bar import M5AlertBar

    path = tmp_path / "trader_annotations.jsonl"
    bar = M5AlertBar(annotations_path=path)

    assert bar.quick_like(None) is None
    assert _rows(path) == []


# ---------------------------------------------------------------------------
# The rail, hosted on a chart-review screen
# ---------------------------------------------------------------------------


def test_the_review_panes_rail_reports_the_chart_review_screen(app, tmp_path):
    """A verdict on a review chart is not a verdict typed on the rail."""
    from ui.widgets.alert_chart_review import AlertChartReview

    pane = AlertChartReview(annotations_path=tmp_path / "trader_annotations.jsonl")

    assert pane.capture_rail._surface == "chart_review"


def test_the_chart_review_workspaces_rail_reports_the_same_screen(app, tmp_path):
    from ui.panels.chart_review_panel import ChartReviewPanel

    panel = ChartReviewPanel(annotations_path=tmp_path / "trader_annotations.jsonl")

    assert panel.capture_rail._surface == "chart_review"


def test_a_veto_from_a_review_pane_lands_with_that_screen_on_the_row(app, tmp_path):
    """The whole point: the row on disk says which screen it came from."""
    from ui.widgets.alert_chart_review import AlertChartReview

    path = tmp_path / "trader_annotations.jsonl"
    pane = AlertChartReview(annotations_path=path)
    rail = pane.capture_rail
    rail.set_context(symbol="NVDA", side="LONG", last_price=100.0, timeframe="D1")
    rail.reason_list.setCurrentRow(0)
    assert rail.selected_reason_code(), "the vocabulary gave the rail no reasons"
    rail.veto_note_input.setText("because")
    rail.commit_veto()

    rows = _rows(path)
    assert rows, rail.status_label.text()
    assert rows[0]["event_type"] == "veto"
    assert rows[0]["surface"] == "chart_review"


def test_a_bare_rail_still_calls_itself_the_rail(app, tmp_path):
    """The default is honest for a host that IS the rail; only a screen overrides."""
    from ui.widgets.capture_rail import CaptureRail

    rail = CaptureRail(annotations_path=tmp_path / "trader_annotations.jsonl")

    assert rail._surface == "rail"
