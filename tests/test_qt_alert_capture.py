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
    pane.capture_rail.setup_list.setCurrentRow(0)
    pane.capture_rail.like_note_input.setText("clean base")  # R9.2: the why is required
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
    pane.capture_rail.setup_list.setCurrentRow(0)
    pane.capture_rail.like_note_input.setText("clean base")  # R9.2: the why is required
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
    """A VETO retires the chart; a LIKE and a NOTE never do (packet T1, trader
    2026-09-04: "I still need time to enter alerts"). A note was always on the
    quiet side of that line and still is."""
    _show(pane, monkeypatch, "AAPL")
    moved: list = []
    pane.skipRequested.connect(moved.append)
    pane.removeTodayRequested.connect(moved.append)
    pane.vetoRetireRequested.connect(moved.append)
    pane.likeRecorded.connect(moved.append)
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
# The day-trade pass (trader, 2026-08-31)
#
# "Many times I really like this stock for a daytrade but it has this ONE issue"
# - and they pass on it. The pass is a NOTE-shaped decision, not a veto: it
# records why, and the chart stays exactly where it was.
# --------------------------------------------------------------------------
def _pass_codes(pane, count: int = 1) -> list:
    return list(pane.capture_rail._pass_vocabulary.codes[:count])


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _tick(pane, codes) -> None:
    for code in codes:
        pane.capture_rail.pass_checkboxes[code].setChecked(True)


def _m5(day: int = 31, count: int = 3) -> list:
    return [
        {
            "dt": datetime(2026, 8, day, 9, 30 + 5 * index),
            "open": 10.0,
            "high": 10.4,
            "low": 9.8,
            "close": 10.1,
            "volume": 500.0,
        }
        for index in range(count)
    ]


def test_the_rail_offers_the_five_pass_reasons_the_trader_listed(pane):
    labels = [check.text() for check in pane.capture_rail.pass_checkboxes.values()]
    assert [label.split("  ", 1)[-1] for label in labels] == [
        "Poor market conditions",
        "Low rvol",
        "LRSI/SMI incongruency",
        "Incoming Horizontal",
        "Other incoming S/R",
    ]


def test_the_pass_block_sits_under_the_note_field(pane):
    """Trader: "a section under the existing note area". Under, and inside it,
    so "under" survives a host wide enough to flow sections side by side."""
    rail = pane.capture_rail
    section = rail.note_input.parentWidget()
    assert section.isAncestorOf(rail.pass_reason_box), "the pass is in the Note section"
    inner = section.layout()
    order = [inner.itemAt(i).widget() for i in range(inner.count())]
    assert order.index(rail.note_input) < order.index(rail.pass_reason_box.parentWidget())


def test_a_pass_writes_one_row_carrying_every_ticked_reason(pane, monkeypatch, tmp_path):
    _show(pane, monkeypatch, "NVDA", "LONG")
    codes = _pass_codes(pane, 2)
    _tick(pane, codes)
    pane.capture_rail.note_input.setText("liked it, rvol never came")
    assert pane.capture_rail.commit_pass() is not None

    rows = _rows(tmp_path / "trader_annotations.jsonl")
    assert len(rows) == 1
    assert rows[0]["event_type"] == "pass"
    assert rows[0]["symbol"] == "NVDA"
    assert rows[0]["reason_codes"] == codes
    assert rows[0]["note"] == "liked it, rvol never came"
    # Never a literal: the stamp is whatever the loaded vocabulary declares.
    assert rows[0]["vocab_version"] == pane.capture_rail._pass_vocabulary.vocab_version


def test_a_pass_never_retires_the_chart(pane, monkeypatch):
    """A VETO retires the chart; a LIKE, a NOTE and a PASS never do (packet T1,
    2026-09-04). A pass was always on the quiet side of that line."""
    _show(pane, monkeypatch)
    retired: list = []
    liked: list = []
    pane.removeTodayRequested.connect(retired.append)
    pane.vetoRetireRequested.connect(retired.append)
    pane.likeRecorded.connect(liked.append)

    _tick(pane, _pass_codes(pane))
    assert pane.capture_rail.commit_pass() is not None

    assert retired == [], "a pass must not retire the chart"
    assert liked == [], "a pass is not a like"
    assert pane.alert is not None


def test_a_pass_places_nothing_and_writes_no_watchlist(pane, monkeypatch, tmp_path):
    """The rail is a recorder. A pass is evidence, never a list membership."""
    from ui.widgets import capture_rail as capture_rail_module

    monkeypatch.setattr(
        capture_rail_module,
        "record_annotation",
        lambda *a, **k: pytest.fail("a pass must not route through the veto/note path"),
    )
    _show(pane, monkeypatch)
    _tick(pane, _pass_codes(pane))
    assert pane.capture_rail.commit_pass() is not None
    assert not any(tmp_path.glob("*.txt")), "no watchlist file is written"


def test_a_pass_attaches_the_m5_bars_the_pane_already_drew(pane, monkeypatch, tmp_path):
    from ui.annotations import pass_bars

    _show(pane, monkeypatch, "AMD")
    pane.snapshot._m5 = {"bars": _m5(31, 3), "overlays": []}
    _tick(pane, _pass_codes(pane))
    row = pane.capture_rail.commit_pass()

    assert row["m5_bar_count"] == 3
    stored = pass_bars.read_pass_bars(
        row, annotations_path=tmp_path / "trader_annotations.jsonl"
    )
    assert len(stored["bars"]) == 3
    assert stored["symbol"] == "AMD"


def test_a_pass_with_nothing_cached_writes_the_timestamp_and_fetches_nothing(
    pane, monkeypatch, tmp_path
):
    """The trader's own fallback: "just store the exact timestamp".

    The stronger half is that reaching for bars must never reach for a FEED.
    Every fetch seam the desk owns is made to explode; the capture still lands.
    """
    import ui.services.chart_bar_refresh as refresh
    import ui.services.chart_data_service as data_service

    def _boom(*_args, **_kwargs):
        raise AssertionError("a capture click must never fetch")

    monkeypatch.setattr(data_service, "shared_service", _boom)
    monkeypatch.setattr(refresh, "shared_refresh_service", _boom)

    class _ExplodingBot:
        def m5_chart_bars(self, *_a, **_k):
            raise AssertionError("a capture click must never read the bot")

        fetch_m5_chart_bars = m5_chart_bars

    _show(pane, monkeypatch, "TSLA")
    pane.snapshot._bot = _ExplodingBot()
    pane.snapshot._m5 = {}
    _tick(pane, _pass_codes(pane))
    row = pane.capture_rail.commit_pass()

    assert row is not None
    assert "m5_bars_ref" not in row
    assert datetime.fromisoformat(row["created_at"]).tzinfo is not None
    assert _rows(tmp_path / "trader_annotations.jsonl")[0]["event_type"] == "pass"


def test_a_provider_that_throws_costs_the_bars_and_never_the_row(pane, monkeypatch):
    def _angry():
        raise RuntimeError("chart is mid-rebuild")

    _show(pane, monkeypatch)
    pane.capture_rail.set_m5_bars_provider(_angry)
    _tick(pane, _pass_codes(pane))
    row = pane.capture_rail.commit_pass()
    assert row is not None and "m5_bars_ref" not in row


def test_a_pass_with_no_reason_ticked_writes_nothing_and_says_so(
    pane, monkeypatch, tmp_path
):
    _show(pane, monkeypatch)
    assert pane.capture_rail.commit_pass() is None
    assert "reason" in pane.capture_rail.status_text().lower()
    assert not (tmp_path / "trader_annotations.jsonl").exists()


def test_a_failed_append_is_reported_and_leaves_the_review_flow_alone(
    pane, monkeypatch, tmp_path
):
    """An evidence store is never allowed to cost the thing it records."""
    blocked = tmp_path / "unwritable"
    blocked.mkdir()
    pane.capture_rail._annotations_path = blocked
    retired: list = []
    pane.removeTodayRequested.connect(retired.append)

    _show(pane, monkeypatch)
    _tick(pane, _pass_codes(pane))
    assert pane.capture_rail.commit_pass() is None
    assert "NOT SAVED" in pane.capture_rail.status_text()
    assert retired == []
    assert pane.alert is not None


def test_a_digit_toggles_a_pass_reason_and_never_commits_on_its_own(
    pane, monkeypatch, tmp_path
):
    """Unlike the veto digit, which commits: a pass is multi-select, so the
    trader has to be able to press 2 and 4 before anything is written."""
    _show(pane, monkeypatch)
    first, second = _pass_codes(pane, 2)
    pane.capture_rail.toggle_pass_reason(first)
    pane.capture_rail.toggle_pass_reason(second)
    assert pane.capture_rail.selected_pass_codes() == [first, second]
    assert not (tmp_path / "trader_annotations.jsonl").exists()

    pane.capture_rail.toggle_pass_reason(first)
    assert pane.capture_rail.selected_pass_codes() == [second]


def test_committing_a_pass_clears_the_ticks_and_the_note(pane, monkeypatch):
    _show(pane, monkeypatch)
    _tick(pane, _pass_codes(pane))
    pane.capture_rail.note_input.setText("one issue")
    pane.capture_rail.commit_pass()
    assert pane.capture_rail.selected_pass_codes() == []
    assert pane.capture_rail.note_input.text() == ""


def test_alt_p_is_offered_to_a_host_that_rebinds_the_keys(pane):
    assert "Alt+P" in dict(pane.capture_rail.action_shortcuts())


def test_alt_p_raises_the_tab_and_lands_on_the_pass_reasons(panel):
    panel.tabs.setCurrentIndex(0)
    panel._capture_shortcuts["Alt+P"].activated.emit()
    rail = panel.chart_review.capture_rail
    assert panel.tabs.currentIndex() == panel._capture_tab_index
    assert rail.pass_reason_box.isAncestorOf(rail.focusWidget())


def test_the_pass_digits_are_scoped_to_their_own_box(panel):
    """A 3 typed into the note above has to stay a 3. The digit shortcuts live
    on the box that holds ONLY the checkboxes, so the note field is outside
    their context - and so are the veto and claim lists' identical digits."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QShortcut

    rail = panel.chart_review.capture_rail
    digits = [
        shortcut
        for shortcut in rail.pass_reason_box.findChildren(QShortcut)
        if shortcut.parent() is rail.pass_reason_box
    ]
    assert {s.key().toString() for s in digits} == set(rail._pass_hotkeys)
    assert all(
        s.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut for s in digits
    )
    assert not rail.pass_reason_box.isAncestorOf(rail.note_input)




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
        ("Alt+K", "setup_list"),
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
    assert not owned & {"Alt+V", "Alt+K", "Alt+N", "Alt+P"}


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


def test_a_veto_retires_the_chart_by_its_own_verb(pane, monkeypatch):
    """Packet T1.1: the rail's veto retires through `vetoRetireRequested`, NOT
    through `removeTodayRequested` - that signal is the "✕ Not today" BUTTON's
    only, because that verb writes a second, uncoded row and opens a note box
    the trader asked to be rid of on the capture window."""
    _show(pane, monkeypatch, "AAPL")
    retired: list = []
    not_today: list = []
    pane.vetoRetireRequested.connect(retired.append)
    pane.removeTodayRequested.connect(not_today.append)
    _pick_reason(pane.capture_rail)
    assert pane.capture_rail.commit_veto() is not None
    assert [alert.symbol for alert in retired] == ["AAPL"]
    assert not_today == []


def test_a_note_still_holds_the_chart(pane, monkeypatch):
    """The one capture that must not move the queue: a note is written ABOUT
    the chart in front of you."""
    _show(pane, monkeypatch, "AAPL")
    moved: list = []
    pane.removeTodayRequested.connect(moved.append)
    pane.vetoRetireRequested.connect(moved.append)
    pane.likeRecorded.connect(moved.append)
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
    pane.vetoRetireRequested.connect(retired.append)
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
    pane.vetoRetireRequested.connect(retired.append)
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
    # Lead ruling 2026-09-04: the day-trade veto retires through the BOX-FREE
    # verb now. It is a capture-rail veto with its reason code already on
    # disk, so the trader's "either veto or like+claim ... no pop up note box"
    # covers it too. The ORDER - place, then retire - is what this test is for
    # and is unchanged.
    monkeypatch.setattr(
        panel,
        "_retire_after_veto",
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
    # Lead ruling 2026-09-04: the box-free verb, as above.
    monkeypatch.setattr(
        panel, "_retire_after_veto", lambda alert: retired.append(alert)
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


# --------------------------------------------------------------------------
# Like + claim becomes a keyed list (trader, 2026-08-20): "layout the like
# + claim similiar to the veto but only do the main setups for now." Widened
# 2026-08-21: "add my post earnings setups and 2nd stdev breakout".
# --------------------------------------------------------------------------
def _offered_ids(rail) -> set:
    from ui.widgets.capture_rail import _CLAIM_ROLE

    return {
        rail.setup_list.item(row).data(_CLAIM_ROLE)
        for row in range(rail.setup_list.count())
    }


def test_the_claim_list_offers_main_swing_plus_the_named_extras(pane):
    from ui.annotations.setup_claims import setup_claim_groups
    from ui.widgets.capture_rail import EXTRA_CLAIM_IDS, MAIN_CLAIM_GROUP

    rail = pane.capture_rail
    groups = dict(setup_claim_groups())
    expected = {claim.setup_id for claim in groups[MAIN_CLAIM_GROUP]}
    expected |= set(EXTRA_CLAIM_IDS)
    assert _offered_ids(rail) == expected


def test_the_rail_offers_every_claim_the_trader_asked_for(pane):
    """A typo in EXTRA_CLAIM_IDS would silently cost a claim; this is the
    guard that stops it being silent."""
    from ui.annotations.setup_claims import valid_setup_claim_ids
    from ui.widgets.capture_rail import EXTRA_CLAIM_IDS

    known = valid_setup_claim_ids()
    missing = [setup_id for setup_id in EXTRA_CLAIM_IDS if setup_id not in known]
    assert missing == [], f"the registry does not name {missing}"
    assert set(EXTRA_CLAIM_IDS) <= _offered_ids(pane.capture_rail)


def test_the_families_the_trader_did_not_ask_for_stay_out(pane):
    """Mid-earnings retests and the rest of the study/playbook shelf are still
    unreachable from this rail - the ask was specific."""
    rail = pane.capture_rail
    offered = _offered_ids(rail)
    assert "mid_earnings_ema15_retest" not in offered
    assert "first_dev_breakout" not in offered
    assert "playbook_volume_thrust" not in offered


def test_the_claims_are_keyed_like_the_veto_reasons(pane):
    from ui.widgets.capture_rail import CLAIM_HOTKEYS

    rail = pane.capture_rail
    labels = [rail.setup_list.item(row).text() for row in range(rail.setup_list.count())]
    # The nine main-swing digits are the ones already in the trader's fingers;
    # a widening that renumbered them would be a regression, not a feature.
    assert labels[0].startswith("1 ")
    assert labels[8].startswith("9 ")
    expected_keys = set(CLAIM_HOTKEYS[: len(labels)])
    assert set(rail._claim_hotkeys) == expected_keys
    for index, label in enumerate(labels):
        assert label.startswith(f"{CLAIM_HOTKEYS[index]} ")


def test_a_letter_key_picks_a_post_earnings_claim(pane, monkeypatch, tmp_path):
    """There is no tenth digit, so the extras continue on letters.

    R9.2 changed what the key DOES: it picks the claim and asks for the why
    rather than committing on its own. The key-to-claim mapping is unchanged,
    which is what this pins.
    """
    import json

    _show(pane, monkeypatch, "AAPL")
    rail = pane.capture_rail
    rail.select_setup(rail._claim_hotkeys["q"])
    assert rail.selected_setup_id() == "post_earnings_candle_break"
    rail.like_note_input.setText("gap held the earnings candle")
    rail.commit_like()
    lines = (tmp_path / "trader_annotations.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[-1])["claimed_setup_id"] == "post_earnings_candle_break"


def test_a_digit_then_a_why_is_the_whole_like(pane, monkeypatch, tmp_path):
    """Alt+K, a digit, the why, Enter.

    Before R9.2 the digit alone committed and the chart was retired as "Not
    today". The trader asked to be prompted for the why every time
    (2026-08-22), so the digit now picks and the Enter commits.

    Packet T1.2 (2026-09-04) took the last of the movement away: the like is
    RECORDED and the chart STAYS, because the trader still has alerts to arm on
    it. The signal is `likeRecorded`; nothing retires.
    """
    import json

    _show(pane, monkeypatch, "AAPL")
    retired: list = []
    recorded: list = []
    pane.removeTodayRequested.connect(retired.append)
    pane.vetoRetireRequested.connect(retired.append)
    pane.likeRecorded.connect(recorded.append)
    rail = pane.capture_rail
    rail.select_setup(rail._claim_hotkeys["2"])
    assert not (tmp_path / "trader_annotations.jsonl").exists(), "the digit alone is not a like"
    rail.like_note_input.setText("reclaimed the band")
    rail.commit_like()

    lines = (tmp_path / "trader_annotations.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["claimed_setup_id"] == rail._claim_hotkeys["2"]
    assert json.loads(lines[0])["note"] == "reclaimed the band"
    assert retired == []
    assert [alert.symbol for alert in recorded] == ["AAPL"]
    assert pane.alert is not None, "the chart stays up (packet T1.2)"


def test_committing_with_nothing_picked_says_so_and_writes_nothing(pane, monkeypatch, tmp_path):
    _show(pane, monkeypatch, "AAPL")
    rail = pane.capture_rail
    rail.setup_list.setCurrentRow(-1)
    assert rail.commit_like() is None
    assert not (tmp_path / "trader_annotations.jsonl").exists()


def test_the_compressed_reason_replaced_the_cluttered_one(pane):
    rail = pane.capture_rail
    labels = [rail.reason_list.item(row).text() for row in range(rail.reason_list.count())]
    assert any("Compressed" in label for label in labels)
    assert not any("cluttered" in label.lower() for label in labels)


# --------------------------------------------------------------------------
# R9.2 (2026-08-22): the LIKE always asks why, and stops parking the symbol
#
# Measured on the window's two capture sessions: 40 of 52 `like_claim` rows
# retired the chart AND put the symbol on `alert_center_ignored_symbols.txt`
# for the rest of the day, because the like was routed through the same
# `remove_today` verb as a veto. Three consequences, all unintended:
# a parked symbol stops emitting `d1EventRecorded`, so on an AWAY day a LIKE
# silently dropped the name from the hourly D1 phone push; the other queued
# alerts for that symbol went with it; and `review_learning.REJECT_ACTIONS`
# counted every LIKE as a dismissal.
#
# On 2026-08-21 the trader liked AEP short at 10:37:15 ET - the best day trade
# of that week - and the system's response to recognising it was to file a
# research row and take the chart away.
# --------------------------------------------------------------------------
def test_a_like_with_no_why_writes_nothing_and_holds_the_chart(pane, monkeypatch, tmp_path):
    """Trader, 2026-08-22: "if I like a chart I should always be prompted with why".

    An ignorable prompt would recreate the empty-`dislike_reason` failure - 31
    rows of the most information-dense prose in the store, discarded because
    nothing required them. So the why is required, and a like without one is
    not a like at all.
    """
    _show(pane, monkeypatch, "AAPL")
    moved: list = []
    pane.removeTodayRequested.connect(moved.append)
    pane.vetoRetireRequested.connect(moved.append)
    pane.likeRecorded.connect(moved.append)
    pane.capture_rail.setup_list.setCurrentRow(0)
    pane.capture_rail.like_note_input.setText("")

    assert pane.capture_rail.commit_like() is None
    assert not (tmp_path / "trader_annotations.jsonl").exists()
    assert moved == [], "the chart stays until the why is given"
    assert "why" in pane.capture_rail.status_text().lower()


def test_a_like_with_a_why_writes_it_into_the_existing_note_field(pane, monkeypatch, tmp_path):
    """No schema change: the why is the row's `note`, which already exists."""
    _show(pane, monkeypatch, "AAPL")
    pane.capture_rail.setup_list.setCurrentRow(0)
    pane.capture_rail.like_note_input.setText("2nd stdev breakout")
    row = pane.capture_rail.commit_like()
    assert row is not None
    written = json.loads(
        (tmp_path / "trader_annotations.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert written["note"] == "2nd stdev breakout"
    assert written["event_type"] == "like_claim"
    assert written["claimed_setup_id"]


def test_a_whitespace_why_is_not_a_why(pane, monkeypatch, tmp_path):
    _show(pane, monkeypatch, "AAPL")
    pane.capture_rail.setup_list.setCurrentRow(0)
    pane.capture_rail.like_note_input.setText("   \t ")
    assert pane.capture_rail.commit_like() is None
    assert not (tmp_path / "trader_annotations.jsonl").exists()


def test_the_claim_digit_moves_focus_to_the_why_instead_of_committing(pane, monkeypatch, tmp_path):
    """Same mechanic as the veto vocabulary's `note_required`: the key selects,
    the why is typed, Enter commits."""
    _show(pane, monkeypatch, "AAPL")
    rail = pane.capture_rail
    rail.setup_list.setCurrentRow(0)
    first = rail.selected_setup_id()
    rail.setup_list.setCurrentRow(-1)
    rail.select_setup(first)
    assert rail.selected_setup_id() == first, "the digit still picks the claim"
    assert not (tmp_path / "trader_annotations.jsonl").exists(), "but it does not commit"
    assert "why" in rail.status_text().lower()


def test_picking_a_claim_moves_focus_to_the_why_field(pane, monkeypatch):
    """The stated mechanic, pinned by observation rather than by hasFocus().

    A widget that was never shown reports no focus, so the assertion is on the
    call the rail makes - which is the part that has to survive a refactor.
    """
    _show(pane, monkeypatch, "AAPL")
    rail = pane.capture_rail
    focused: list = []
    monkeypatch.setattr(rail.like_note_input, "setFocus", lambda *a: focused.append(True))
    rail.setup_list.setCurrentRow(0)
    rail.select_setup(rail.selected_setup_id())
    assert focused == [True]

    # And the same for a double-click, which must not behave differently.
    focused.clear()
    rail._claim_picked(rail.setup_list.item(0))
    assert focused == [True]


def test_the_why_field_says_it_is_required(pane, monkeypatch):
    _show(pane, monkeypatch, "AAPL")
    assert "required" in pane.capture_rail.like_note_input.placeholderText().lower()


# --------------------------------------------------------------------------
# Trader, 2026-08-27: "i want to be able to double click the like and claim the
# same way i can double click the veto."
#
# The veto's gesture ATTEMPTS the commit - `select_reason` calls `commit_veto`
# and only diverts to the note field when that reason's `note_required` is
# unmet. The like's gesture went straight to the prompt and could never commit,
# so a trader who had already typed the why was told to type it again.
#
# The fix is the veto's own wiring: the gesture calls `commit_like`, which
# already carries the required-why guard. Nothing about the 2026-08-22 rule
# moves - the two tests above still pass unchanged - and the new capability is
# exactly "why typed, then the gesture commits".
# --------------------------------------------------------------------------
def test_double_clicking_a_claim_commits_the_like_once_the_why_is_there(
    pane, monkeypatch, tmp_path
):
    _show(pane, monkeypatch, "AAPL")
    rail = pane.capture_rail
    rail.like_note_input.setText("held the 2nd dev and reclaimed")

    rail.setup_list.itemActivated.emit(rail.setup_list.item(0))

    path = tmp_path / "trader_annotations.jsonl"
    assert path.exists(), "the double-click must commit, as it does on a veto"
    written = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert written["event_type"] == "like_claim"
    assert written["note"] == "held the 2nd dev and reclaimed"
    from ui.widgets.capture_rail import _CLAIM_ROLE

    assert written["claimed_setup_id"] == rail.setup_list.item(0).data(_CLAIM_ROLE)


def test_double_clicking_a_claim_with_no_why_still_refuses_and_prompts(
    pane, monkeypatch, tmp_path
):
    """The 2026-08-22 rule is untouched: the gesture attempts, it does not
    override. A like without a why is still not a like."""
    _show(pane, monkeypatch, "AAPL")
    rail = pane.capture_rail
    rail.like_note_input.setText("")

    rail.setup_list.itemActivated.emit(rail.setup_list.item(0))

    assert not (tmp_path / "trader_annotations.jsonl").exists()
    assert "why" in rail.status_text().lower()


def test_the_digit_commits_too_once_the_why_is_there(pane, monkeypatch, tmp_path):
    """The veto's digit and double-click behave identically; the like's must
    too, or the rail is internally inconsistent in a way the veto is not."""
    _show(pane, monkeypatch, "AAPL")
    rail = pane.capture_rail
    rail.setup_list.setCurrentRow(0)
    target = rail.selected_setup_id()
    rail.setup_list.setCurrentRow(-1)
    rail.like_note_input.setText("post-earnings drift, day 3")

    rail.select_setup(target)

    path = tmp_path / "trader_annotations.jsonl"
    assert path.exists(), "digit + a typed why must commit"
    written = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert written["claimed_setup_id"] == target
    assert written["note"] == "post-earnings drift, day 3"


def test_a_committed_like_clears_the_why_so_the_next_chart_starts_empty(
    pane, monkeypatch, tmp_path
):
    """Otherwise the next double-click would silently reuse the previous
    chart's reasoning - the worst possible failure for this dataset."""
    _show(pane, monkeypatch, "AAPL")
    rail = pane.capture_rail
    rail.like_note_input.setText("first chart's why")
    rail.setup_list.itemActivated.emit(rail.setup_list.item(0))
    assert rail.like_note_input.text() == "", "the why must not carry over"


def test_the_like_gesture_is_wired_the_same_shape_as_the_veto(pane, monkeypatch):
    """Both lists' activation goes through the commit, not around it."""
    _show(pane, monkeypatch, "AAPL")
    rail = pane.capture_rail

    liked, vetoed = [], []
    monkeypatch.setattr(rail, "commit_like", lambda: liked.append(True))
    monkeypatch.setattr(rail, "commit_veto", lambda: vetoed.append(True))

    rail.setup_list.itemActivated.emit(rail.setup_list.item(0))
    assert liked == [True], "the claim list must call commit_like"

    if rail.reason_list.count():
        rail.reason_list.itemActivated.emit(rail.reason_list.item(0))
        assert vetoed == [True], "and the reason list still calls commit_veto"


def test_a_like_is_recorded_and_leaves_the_chart_where_it_is(pane, monkeypatch):
    """R9.2(b) took the like off the "Not today" verb. Packet T1.2 (trader,
    2026-09-04) took the movement away as well: "the 'like' button in the visual
    chart review should NOT advance the char to the next page because i still
    need time to enter alerts etc."

    So the pane REPORTS the like (`likeRecorded`) and asks for nothing. The
    review event behind it is still called `like_advance` - historical, because
    `review_learning.TAKE_ACTIONS` keys on that string.
    """
    _show(pane, monkeypatch, "AAPL")
    retired: list = []
    recorded: list = []
    pane.removeTodayRequested.connect(retired.append)
    pane.vetoRetireRequested.connect(retired.append)
    pane.likeRecorded.connect(recorded.append)
    pane.capture_rail.setup_list.setCurrentRow(0)
    pane.capture_rail.like_note_input.setText("clean base")
    assert pane.capture_rail.commit_like() is not None
    assert retired == [], "a like must never take a retirement path"
    assert [alert.symbol for alert in recorded] == ["AAPL"]
    assert pane.alert is not None, "the chart the trader is arming stays"


def test_a_veto_still_retires_and_a_like_is_not_a_retirement(pane, monkeypatch):
    _show(pane, monkeypatch, "AAPL")
    retired: list = []
    recorded: list = []
    pane.vetoRetireRequested.connect(retired.append)
    pane.likeRecorded.connect(recorded.append)
    _pick_reason(pane.capture_rail)
    assert pane.capture_rail.commit_veto() is not None
    assert [alert.symbol for alert in retired] == ["AAPL"]
    assert recorded == []


def test_a_like_never_reaches_the_ignore_set_and_the_symbol_still_alerts(panel, monkeypatch):
    """The measured harm, pinned at the panel: parking is what stopped the
    hourly D1 phone push from ever naming a liked symbol again that day."""
    from ui.models.bounce import BounceAlert

    alert = BounceAlert(
        time_text="09:31:00",
        symbol="AEP",
        side="SHORT",
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text="[B-TIER] AEP: Bounce confirmed",
    )
    panel._current_review_alert = alert
    panel._after_like(alert)

    assert "AEP" not in panel._ignored_symbols


def test_a_like_leaves_the_symbols_other_queued_alerts_alone(panel, monkeypatch):
    """Parking swept every queued alert for the symbol. Packet T1.2 takes even
    the single advance away: the chart stays and the waiting list is untouched,
    so the trader can arm an alert on the name they just liked."""
    from ui.models.bounce import BounceAlert

    def _alert_for(symbol, trigger):
        return BounceAlert(
            time_text="09:31:00",
            symbol=symbol,
            side="SHORT",
            trigger=trigger,
            timeframe="M5",
            tag="green",
            raw_text=f"[B-TIER] {symbol}: {trigger}",
        )

    current = _alert_for("AEP", "first")
    queued_same = _alert_for("AEP", "second")
    queued_other = _alert_for("NVDA", "third")
    panel._current_review_alert = current
    panel._review_queue = [queued_same, queued_other]

    panel._after_like(current)

    # No advance at all now: the liked chart is still up, and nothing was
    # dropped for sharing a symbol with it.
    assert panel._current_review_alert is current
    assert panel._review_queue == [queued_same, queued_other]
    assert "AEP" not in panel._ignored_symbols


def test_the_panel_records_like_advance_not_remove_today(panel, monkeypatch):
    from ui.models.bounce import BounceAlert

    recorded: list = []
    monkeypatch.setattr(
        panel, "_record_review_event", lambda action, **kw: recorded.append(action)
    )
    alert = BounceAlert(
        time_text="09:31:00",
        symbol="AEP",
        side="SHORT",
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text="[B-TIER] AEP",
    )
    panel._current_review_alert = alert
    panel._after_like(alert)
    # The NAME is historical and must not change - `review_learning.TAKE_ACTIONS`
    # keys on the string. Since packet T1.2 it means "liked; the symbol keeps
    # alerting and the chart stays".
    assert recorded == ["like_advance"]
    assert "remove_today" not in recorded
    assert panel._current_review_alert is alert
