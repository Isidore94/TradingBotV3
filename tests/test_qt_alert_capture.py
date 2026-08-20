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


def test_a_note_is_still_only_a_recorder(pane, monkeypatch):
    """Veto and like retire the chart now; a note remains pure capture."""
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


def test_the_capture_rail_is_off_the_pane_and_the_arm_bar_is_not(panel):
    """Trader, 2026-08-20, in two passes.

    First: "I cannot see the charts at all." Then, once the rail was gone:
    "I also need my m5 and D1 alert hotbuttons back on the bottom of the
    visual chart... I also need the ability to input a ticker manually."

    Measured at this column's width the rail is ~697px and the arm bar ~131px,
    so the split is not a compromise - it drops 84% of the height and keeps
    every control the trader reaches for per-chart.
    """
    review = panel.chart_review
    layout = review.layout()
    rows = [layout.itemAt(i) for i in range(layout.count())]
    widgets = [item.widget() for item in rows]
    # The chart slot is the snapshot and its placeholder - mutually exclusive,
    # so they count as one row of the stack.
    slot_end = max(widgets.index(review.snapshot), widgets.index(review.empty_state))
    # Under it: the arm bar, then the verb row. Nothing else.
    assert widgets[slot_end + 1] is review.arm_bar
    assert slot_end + 2 == layout.count() - 1, "something is stacked under the charts"
    assert rows[-1].layout() is not None, "the verb row must stay a layout row"

    assert review.isAncestorOf(review.arm_bar), "the hotbuttons come back"
    assert not review.isAncestorOf(review.capture_rail), "the rail does not"


def test_the_named_controls_are_the_ones_that_came_back(panel):
    """The three things the trader asked for by name, under the chart."""
    review = panel.chart_review
    bar = review.arm_bar
    assert review.isAncestorOf(bar.symbol_input), "type-a-ticker"
    assert bar.watch_buttons and all(
        review.isAncestorOf(button) for button in bar.watch_buttons.values()
    ), "M5 hotbuttons"
    assert bar.d1_event_buttons and all(
        review.isAncestorOf(button) for button in bar.d1_event_buttons.values()
    ), "D1 hotbuttons"


def test_the_rail_is_reachable_as_a_tab(panel):
    assert "Capture" in _tab_labels(panel)
    assert panel.isAncestorOf(panel.chart_review.capture_rail)
    # The Armed tab keeps the cross-symbol inventory; the controls that fill
    # it are under the chart, on the symbol being looked at.
    assert panel.tabs.widget(panel._armed_tab_index) is panel.armed_list


@pytest.mark.parametrize(
    "sequence, widget_name",
    [
        ("Alt+V", "reason_list"),
        ("Alt+K", "setup_input"),
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
    assert not owned & {"Alt+V", "Alt+K", "Alt+N"}


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


def test_the_armed_count_rides_the_tab_title(panel):
    """The tab still carries the count, in peripheral vision.

    The duplicate line on the verb row is OFF here: with the arm bar back
    under the chart its own armed text and chips are right there, and two
    copies of one state is noise.
    """
    review = panel.chart_review
    assert not review.armed_summary.isVisibleTo(review)
    assert panel.tabs.tabText(panel._armed_tab_index) == "Armed"

    panel.chart_symbol("NVDA")
    review.set_armed_kinds(("hod_avwap",))
    review.set_armed_d1_events(("new_5d_high",))
    assert review.armed_count() == 2
    assert panel.tabs.tabText(panel._armed_tab_index) == "Armed (2)"

    review.clear()
    assert panel.tabs.tabText(panel._armed_tab_index) == "Armed"


def test_an_undocked_arm_bar_still_surfaces_the_armed_line(tmp_path):
    """The verb-row line is the affordance for a host that TOOK the bar."""
    from ui.widgets.alert_chart_review import AlertChartReview

    review = AlertChartReview(
        annotations_path=tmp_path / "a.jsonl", dock_arm_bar=False
    )
    try:
        assert review.armed_summary.isVisibleTo(review)
        assert review.armed_summary.text() == "Nothing armed"
        review.set_armed_kinds(("hod_avwap",))
        assert review.armed_summary.text() == "⚡ 1 armed"
    finally:
        review.deleteLater()


def test_a_docked_host_keeps_the_rail_in_its_own_stack(tmp_path):
    """Placement is the HOST's decision: the snapshot popup and the Chart
    Review workspace must not inherit a missing rail."""
    from ui.widgets.alert_chart_review import AlertChartReview

    docked = AlertChartReview(annotations_path=tmp_path / "a.jsonl")
    # Fully docked is still the default: both controls in its own stack.
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
        assert {"Alt+V", "Alt+K", "Alt+N"} <= owned
        # The duplicated armed line is the undocked host's affordance only.
        assert not docked.armed_summary.isVisibleTo(docked)
    finally:
        docked.deleteLater()


# --------------------------------------------------------------------------
# 2026-08-20, second pass: veto IS a queue verb, and the day-trade exception
#
# Trader: "when I click veto it should just disappear as 'not for today'. the
# only exception is I want an option to hit 'veto but add to M5 focus' because
# it may be a shit D1 chart but its a good daytrade."
# --------------------------------------------------------------------------
def _pick_reason(rail) -> str:
    """Select the first veto reason that does NOT require a note.

    A note-required reason is refused at the schema, not the button, so a
    bare commit against one would fail for the wrong reason and these tests
    would stop measuring the queue behaviour they exist for.
    """
    from ui.widgets.capture_rail import _REASON_ROLE

    for row in range(rail.reason_list.count()):
        rail.reason_list.setCurrentRow(row)
        if not rail._selected_reason_requires_note():
            return rail.reason_list.item(row).data(_REASON_ROLE)
    raise AssertionError("no note-free veto reason in the vocabulary")


def test_a_veto_retires_the_chart_as_not_today(pane, monkeypatch):
    _show(pane, monkeypatch, "AAPL")
    retired: list = []
    pane.removeTodayRequested.connect(retired.append)
    _pick_reason(pane.capture_rail)
    assert pane.capture_rail.commit_veto() is not None
    assert [alert.symbol for alert in retired] == ["AAPL"]


def test_a_like_also_retires_the_chart(pane, monkeypatch):
    """Trader, 2026-08-20: "when I pick a like and claim setup reason, we
    should just move onto the next chart"."""
    _show(pane, monkeypatch, "AAPL")
    retired: list = []
    pane.removeTodayRequested.connect(retired.append)
    pane.capture_rail.setup_input.setCurrentIndex(0)
    assert pane.capture_rail.commit_like() is not None
    assert [alert.symbol for alert in retired] == ["AAPL"]


def test_a_note_still_holds_the_chart(pane, monkeypatch):
    """The one capture that must not move the queue: a note is written ABOUT
    the chart in front of you."""
    _show(pane, monkeypatch, "AAPL")
    moved: list = []
    pane.removeTodayRequested.connect(moved.append)
    pane.skipRequested.connect(moved.append)
    pane.focusRequested.connect(moved.append)
    pane.capture_rail.note_input.setText("watching the 50")
    pane.capture_rail.commit_note()
    assert moved == []


def test_the_hypothetical_stop_control_is_gone(pane):
    """Trader: "get rid of hypothetical stop for now its not useful."

    The CONTROL only. `ui.annotations.store` still validates hypo_stop rows,
    because the stream is append-only evidence and rows already written have
    to stay readable."""
    rail = pane.capture_rail
    assert not hasattr(rail, "stop_input")
    assert not hasattr(rail, "commit_hypo_stop")
    assert "Alt+S" not in dict(rail.action_shortcuts())

    from ui.annotations.store import EVENT_HYPO_STOP, build_annotation

    row = build_annotation(EVENT_HYPO_STOP, symbol="NVDA", stop_price=10.5, side="LONG")
    assert row["event_type"] == "hypo_stop", "history must stay readable"


def test_a_refused_veto_retires_nothing(pane, monkeypatch):
    """No reason picked -> no row, no queue move. The chart stays put."""
    _show(pane, monkeypatch, "AAPL")
    retired: list = []
    pane.removeTodayRequested.connect(retired.append)
    pane.capture_rail.reason_list.setCurrentRow(-1)
    assert pane.capture_rail.commit_veto() is None
    assert retired == []


def test_the_day_trade_veto_asks_for_placement_instead_of_retiring(pane, monkeypatch):
    """It must NOT take the plain-veto path: the host needs the alert object
    to place the name BEFORE the chart is retired."""
    _show(pane, monkeypatch, "AAPL")
    retired: list = []
    day_traded: list = []
    pane.removeTodayRequested.connect(retired.append)
    pane.vetoDayTradeRequested.connect(day_traded.append)

    _pick_reason(pane.capture_rail)
    row = pane.capture_rail.commit_veto_day_trade()

    assert row is not None, "the veto is still recorded, identically"
    assert [alert.symbol for alert in day_traded] == ["AAPL"]
    assert retired == [], "the pane must not retire it behind the host's back"


def test_the_day_trade_veto_writes_an_ordinary_veto_row(pane, monkeypatch, tmp_path):
    """No schema change, no new field, no second row. The annotation is the
    same D1 judgement it would have been."""
    _show(pane, monkeypatch, "AAPL")
    _pick_reason(pane.capture_rail)
    pane.capture_rail.commit_veto_day_trade()
    lines = (tmp_path / "trader_annotations.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written["event_type"] == "veto"
    assert written["symbol"] == "AAPL"
    assert written["schema_version"] == 1


def test_the_rail_still_places_nothing_itself(pane, monkeypatch, tmp_path):
    """The boundary the rail exists behind: it asks, the host writes."""
    _show(pane, monkeypatch, "AAPL")
    placed: list = []
    pane.focusRequested.connect(placed.append)
    _pick_reason(pane.capture_rail)
    pane.capture_rail.commit_veto_day_trade()
    assert placed == [], "a request is not a placement verb"
    # Nothing but the annotation file exists - no watchlist, no focus store.
    assert {path.name for path in tmp_path.iterdir()} == {"trader_annotations.jsonl"}


def test_the_panel_places_on_m5_focus_then_retires_the_chart(panel, monkeypatch):
    """End to end, in the order that matters."""
    from ui.models.bounce import BounceAlert

    calls: list = []

    class _Focus:
        def add(self, symbol, side, category, **kwargs):
            calls.append(("add", symbol, side, category, kwargs.get("origin")))
            return True

    monkeypatch.setattr(panel, "focus_service", _Focus())
    retired: list = []
    monkeypatch.setattr(
        panel,
        "_remove_review_alert_for_today",
        lambda alert: (calls.append(("retire", alert.symbol)), retired.append(alert)),
    )
    alert = BounceAlert(
        time_text="09:31:00",
        symbol="NVDA",
        side="LONG",
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text="[B-TIER] NVDA: Bounce confirmed",
    )
    panel._veto_but_day_trade(alert)

    assert [step[0] for step in calls] == ["add", "retire"], "place, THEN retire"
    assert calls[0] == ("add", "NVDA", "long", "m5", "veto_day_trade")
    assert [a.symbol for a in retired] == ["NVDA"]


def test_a_failed_placement_still_retires_the_chart(panel, monkeypatch):
    """The veto is already on disk. Leaving the name up invites a second one."""
    from ui.models.bounce import BounceAlert

    class _Broken:
        def add(self, *_a, **_k):
            raise OSError("focus store went away")

    monkeypatch.setattr(panel, "focus_service", _Broken())
    retired: list = []
    monkeypatch.setattr(
        panel, "_remove_review_alert_for_today", lambda alert: retired.append(alert)
    )
    alert = BounceAlert(
        time_text="09:31:00",
        symbol="NVDA",
        side="SHORT",
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text="[B-TIER] NVDA",
    )
    panel._veto_but_day_trade(alert)
    assert [a.symbol for a in retired] == ["NVDA"]


# --------------------------------------------------------------------------
# 2026-08-20, third pass: the pane stops wasting a 4K monitor
#
# Trader, with a screenshot of the desk: "look at how inefficient this GUI is.
# this bot basically gets an entire 4k monitor and we cant fit everything in
# cleanly?" The measurement behind the fix is in the test below.
# --------------------------------------------------------------------------
def test_an_empty_pane_does_not_smear_its_slack_into_the_labels(panel):
    """The measured fault, pinned.

    The snapshot carries this pane's only expanding stretch. HIDING it left Qt
    with four Preferred widgets and a column of slack, which it split equally:
    at 2000x1900 the one-line title got 346px, the setup line 346px, the arm
    bar 346px and the verb row 346px - about 1240px of a 4K screen spent on
    label padding, in the state the desk sits in whenever the queue is clear.
    """
    review = panel.chart_review
    panel.resize(2000, 1900)
    panel.show()
    review.clear()
    for _ in range(4):
        _QT.QApplication.instance().processEvents()

    assert review.empty_state.isVisibleTo(review)
    assert not review.snapshot.isVisibleTo(review)
    # The slack lands in ONE place, and it is the placeholder.
    assert review.empty_state.height() > review.height() * 0.7
    # Every other row in the stack sits at its size hint. Hidden rows are
    # excluded because they hold no layout space and report a stale geometry.
    layout = review.layout()
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if widget is None or widget is review.empty_state:
            continue
        if not widget.isVisibleTo(review):
            continue
        assert widget.height() <= widget.sizeHint().height() + 40, (
            f"{widget.objectName() or type(widget).__name__} inflated to "
            f"{widget.height()}px against a {widget.sizeHint().height()}px hint"
        )


def test_charting_gives_every_reclaimed_pixel_to_the_candles(panel, monkeypatch):
    review = panel.chart_review
    panel.resize(2000, 1900)
    panel.show()
    panel.chart_symbol("NVDA")
    for _ in range(4):
        _QT.QApplication.instance().processEvents()

    assert review.snapshot.isVisibleTo(review)
    assert not review.empty_state.isVisibleTo(review)
    assert review.snapshot.height() > review.height() * 0.7


def test_the_placeholder_says_how_to_get_a_chart(panel):
    """Dead space that explains itself beats dead space that does not."""
    review = panel.chart_review
    review.clear()
    text = review.empty_state.message_label.text().lower()
    assert "ticker" in text and "alert" in text
