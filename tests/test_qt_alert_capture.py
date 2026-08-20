"""R4 sections 2.3 and 5: capture and the badge on the Alert Center pane.

Section 2.3's whole risk is that LIKE blurs into placement. The spec is
explicit: CaptureRail LIKE is analysis-only and *never* writes Focus
membership; "Add to Focus Picks" stays the one explicit placement verb. Several
of these tests exist only to keep that boundary from eroding -- an earlier draft
of the rail did route likes through FocusService.add, and it had to be removed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


@pytest.fixture
def pane(tmp_path):
    import pick_feedback
    from ui.widgets.alert_chart_review import AlertChartReview

    pick_feedback.clear_reviewed_today_cache()
    widget = AlertChartReview(annotations_path=tmp_path / "trader_annotations.jsonl")
    yield widget
    widget.deleteLater()


def _alert(symbol: str = "AAPL", side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="09:31:00",
        symbol=symbol,
        side=side,
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text=f"[B-TIER] {symbol}: Bounce confirmed",
    )


def _show(pane, monkeypatch, symbol: str = "AAPL", side: str = "LONG"):
    monkeypatch.setattr(pane.snapshot, "set_symbol", lambda *a, **k: None)
    pane.set_alert(_alert(symbol, side))


# --------------------------------------------------------------------------
# the rail exists and follows the alert
# --------------------------------------------------------------------------
def test_the_alert_pane_carries_a_capture_rail(pane):
    from ui.widgets.capture_rail import CaptureRail

    assert isinstance(pane.capture_rail, CaptureRail)


def test_setting_an_alert_points_the_rail_at_its_symbol(pane, monkeypatch):
    _show(pane, monkeypatch, "NVDA", "SHORT")
    assert pane.capture_rail._symbol == "NVDA"
    assert pane.capture_rail._side == "SHORT"


def test_a_new_alert_clears_the_previous_level_reference(pane, monkeypatch):
    _show(pane, monkeypatch, "NVDA")
    pane.snapshot.d1LevelSelected.emit("NVDA", "d1_horizontal:2026-06-01:100.00", "d1_horizontal", 100.0)
    assert pane.capture_rail._ref_level_id
    _show(pane, monkeypatch, "AMD")
    assert pane.capture_rail._ref_level_id == ""


def test_a_selected_level_becomes_the_capture_reference(pane, monkeypatch):
    _show(pane, monkeypatch, "NVDA")
    pane.snapshot.d1LevelSelected.emit("NVDA", "d1_horizontal:2026-06-01:100.00", "d1_horizontal", 100.0)
    assert pane.capture_rail._ref_level_family == "d1_horizontal"


# --------------------------------------------------------------------------
# LIKE is analysis-only - the boundary section 2.3 exists to protect
# --------------------------------------------------------------------------
def test_like_writes_one_annotation_row(pane, monkeypatch, tmp_path):
    _show(pane, monkeypatch, "AAPL")
    pane.capture_rail.setup_input.setCurrentIndex(0)
    row = pane.capture_rail.commit_like()
    assert row is not None
    lines = (tmp_path / "trader_annotations.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["symbol"] == "AAPL"


def test_like_never_writes_focus_membership(pane, monkeypatch, tmp_path):
    """The explicit placement verb stays the only thing that places."""
    _show(pane, monkeypatch, "AAPL")
    placed: list = []
    pane.focusRequested.connect(placed.append)
    pane.capture_rail.setup_input.setCurrentIndex(0)
    pane.capture_rail.commit_like()
    assert placed == []
    # And nothing but the annotation file appeared.
    assert {path.name for path in tmp_path.iterdir()} == {"trader_annotations.jsonl"}


def test_the_focus_verb_still_places(pane, monkeypatch):
    """The other half of the same boundary: placement still works, and is
    still a different button from LIKE."""
    _show(pane, monkeypatch, "AAPL")
    placed: list = []
    pane.focusRequested.connect(placed.append)
    pane.focus_button.click()
    assert len(placed) == 1
    assert placed[0].symbol == "AAPL"


def test_capture_does_not_advance_the_review_queue(pane, monkeypatch):
    """Capture is a recorder. Only the three queue verbs move the queue."""
    _show(pane, monkeypatch, "AAPL")
    moved: list = []
    pane.skipRequested.connect(moved.append)
    pane.removeTodayRequested.connect(moved.append)
    pane.focusRequested.connect(moved.append)
    pane.capture_rail.note_input.setText("heavy into the 50")
    pane.capture_rail.commit_note()
    assert moved == []


# --------------------------------------------------------------------------
# section 5 badge
# --------------------------------------------------------------------------
def test_a_symbol_decided_today_shows_the_badge(pane, monkeypatch, tmp_path):
    import pick_feedback

    feedback = tmp_path / "pick_feedback.jsonl"
    today = datetime.now().date().isoformat()
    feedback.write_text(
        json.dumps(
            {
                "ts": f"{today}T09:31:00",
                "trade_date": today,
                "symbol": "AAPL",
                "side": "LONG",
                "verdict": "dislike",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pick_feedback.clear_reviewed_today_cache()
    monkeypatch.setattr(
        pane,
        "_reviewed_symbols",
        lambda: pick_feedback.reviewed_symbols_today(
            market_date=today,
            pick_feedback_path=feedback,
            review_events_path=tmp_path / "none.jsonl",
            annotations_path=tmp_path / "none2.jsonl",
        ),
    )
    _show(pane, monkeypatch, "AAPL")
    assert "Reviewed today" in pane.reviewed_badge.text()
    _show(pane, monkeypatch, "TSLA")
    assert pane.reviewed_badge.text() == ""


def test_capturing_makes_the_badge_appear_immediately(pane, monkeypatch):
    seen: set[str] = set()
    monkeypatch.setattr(pane, "_reviewed_symbols", lambda: set(seen))
    _show(pane, monkeypatch, "AAPL")
    assert pane.reviewed_badge.text() == ""
    seen.add("AAPL")
    pane.capture_rail.note_input.setText("checked")
    pane.capture_rail.commit_note()
    assert "Reviewed today" in pane.reviewed_badge.text()


def test_a_badge_lookup_failure_never_takes_down_the_pane(pane, monkeypatch):
    def boom():
        raise OSError("home folder went away")

    monkeypatch.setattr(pane, "_reviewed_symbols", boom)
    _show(pane, monkeypatch, "AAPL")
    assert pane.reviewed_badge.text() == ""


# --------------------------------------------------------------------------
# 2026-08-20: the rail moved onto a tab, and the keyboard contract came with it
#
# The trader could not read the charts at all: title -> setup text -> charts ->
# two arm rows -> a ~600px rail -> the verb row, in one column. The rail and the
# arm bar became tabs. The founding contract of the rail (capture_rail module
# docstring) is that every capture is under five seconds without the mouse, so
# these pin that the keys still reach it from a tab that is not on screen.
# --------------------------------------------------------------------------
@pytest.fixture
def panel(tmp_path, monkeypatch):
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    widget = AlertCenterPanel()
    monkeypatch.setattr(
        widget.chart_review.capture_rail, "_annotations_path", tmp_path / "ann.jsonl"
    )
    yield widget
    widget.close()
    widget.deleteLater()


def _tab_labels(panel) -> list[str]:
    return [panel.tabs.tabText(index) for index in range(panel.tabs.count())]


def test_the_charts_own_the_pane_with_one_control_row_under_them(panel):
    """Between the charts and the tab strip: the verb row, and nothing else."""
    review = panel.chart_review
    layout = review.layout()
    rows = [layout.itemAt(i) for i in range(layout.count())]
    # ... the charts, then exactly one trailing item, and it is a layout (the
    # verb row), not another docked control widget.
    chart_index = next(
        i for i, item in enumerate(rows) if item.widget() is review.snapshot
    )
    assert chart_index == layout.count() - 2, "something is stacked under the charts"
    assert rows[-1].layout() is not None
    # The two docks are elsewhere, and neither is a child of the review pane.
    assert not review.isAncestorOf(review.capture_rail)
    assert not review.isAncestorOf(review.arm_bar)


def test_the_rail_and_the_arm_bar_are_reachable_as_tabs(panel):
    assert "Capture" in _tab_labels(panel)
    assert panel.isAncestorOf(panel.chart_review.capture_rail)
    assert panel.isAncestorOf(panel.chart_review.arm_bar)
    # The arm bar joins the inventory it fills rather than becoming a sixth tab.
    assert _tab_labels(panel)[panel._armed_tab_index].startswith("Armed")


@pytest.mark.parametrize(
    "sequence, widget_name",
    [
        ("Alt+V", "reason_list"),
        ("Alt+K", "setup_input"),
        ("Alt+S", "stop_input"),
        ("Alt+N", "note_input"),
    ],
)
def test_every_capture_key_raises_the_tab_and_focuses_its_input(
    panel, sequence, widget_name
):
    panel.tabs.setCurrentIndex(0)
    shortcut = panel._capture_shortcuts[sequence]
    from PySide6.QtCore import Qt

    # Panel scope, not rail scope: a shortcut bound inside a hidden tab page
    # never fires, which is exactly how this contract would have died quietly.
    assert shortcut.parent() is panel
    assert shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut

    shortcut.activated.emit()
    assert panel.tabs.currentIndex() == panel._capture_tab_index
    rail = panel.chart_review.capture_rail
    assert rail.focusWidget() is getattr(rail, widget_name)


def test_alt_v_arms_the_veto_flow_not_just_the_tab(panel, monkeypatch):
    """Alt+V, then a digit, is the whole veto. The digit needs a selection."""
    rail = panel.chart_review.capture_rail
    rail.reason_list.setCurrentRow(-1)
    panel.tabs.setCurrentIndex(0)
    panel._capture_shortcuts["Alt+V"].activated.emit()
    assert panel.tabs.currentIndex() == panel._capture_tab_index
    assert rail.reason_list.currentRow() == 0


def test_the_rail_binds_no_duplicate_of_a_key_its_host_owns(panel):
    """Two live bindings for one sequence is an ambiguous shortcut, and Qt
    fires NEITHER - the failure mode is the keys going dead, silently."""
    from PySide6.QtGui import QShortcut

    rail = panel.chart_review.capture_rail
    owned = {
        shortcut.key().toString()
        for shortcut in rail.findChildren(QShortcut)
        if shortcut.parent() is rail
    }
    assert not owned & {"Alt+V", "Alt+K", "Alt+S", "Alt+N"}


def test_the_rail_still_writes_through_record_annotation_after_reparenting(
    panel, monkeypatch, tmp_path
):
    """The move is layout. The recorder is untouched, and still the only
    thing a capture reaches."""
    from ui.widgets import capture_rail as capture_rail_module

    seen: list = []
    real = capture_rail_module.record_annotation

    def _spy(*args, **kwargs):
        seen.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(capture_rail_module, "record_annotation", _spy)
    panel.chart_symbol("NVDA")
    rail = panel.chart_review.capture_rail
    assert rail.symbol == "NVDA"
    rail.note_input.setText("held the 50 all morning")
    row = rail.commit_note()

    assert row is not None and len(seen) == 1
    assert row["symbol"] == "NVDA"
    written = (tmp_path / "ann.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(written) == 1
    assert json.loads(written[0])["note"] == "held the 50 all morning"
    # Still a recorder: nothing else on disk, and the queue did not move.
    assert {path.name for path in tmp_path.iterdir()} == {"ann.jsonl"}


def test_the_armed_state_is_legible_without_opening_the_tab(panel):
    """The arm bar's own "Nothing armed" line went onto the tab with it."""
    review = panel.chart_review
    assert review.armed_summary.isVisibleTo(review)
    assert review.armed_summary.text() == "Nothing armed"
    assert panel.tabs.tabText(panel._armed_tab_index) == "Armed"

    panel.chart_symbol("NVDA")
    review.set_armed_kinds(("hod_avwap",))
    review.set_armed_d1_events(("new_5d_high",))
    assert review.armed_count() == 2
    assert review.armed_summary.text() == "⚡ 2 armed"
    assert panel.tabs.tabText(panel._armed_tab_index) == "Armed (2)"

    review.clear()
    assert review.armed_summary.text() == "Nothing armed"
    assert panel.tabs.tabText(panel._armed_tab_index) == "Armed"


def test_a_docked_host_keeps_the_rail_in_its_own_stack(tmp_path):
    """Placement is the HOST's decision: the snapshot popup and the Chart
    Review workspace must not inherit a missing rail."""
    from ui.widgets.alert_chart_review import AlertChartReview

    docked = AlertChartReview(annotations_path=tmp_path / "a.jsonl")
    try:
        assert docked.isAncestorOf(docked.capture_rail)
        assert docked.isAncestorOf(docked.arm_bar)
        # And it keeps its own keys, because nothing above it took them.
        from PySide6.QtGui import QShortcut

        owned = {
            shortcut.key().toString()
            for shortcut in docked.capture_rail.findChildren(QShortcut)
            if shortcut.parent() is docked.capture_rail
        }
        assert {"Alt+V", "Alt+K", "Alt+S", "Alt+N"} <= owned
        # The duplicated armed line is the undocked host's affordance only.
        assert not docked.armed_summary.isVisibleTo(docked)
    finally:
        docked.deleteLater()
