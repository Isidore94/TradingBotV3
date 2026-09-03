"""S1.3 - one Strength surface, all of it open, and a boundary that can be dragged.

Trader, 2026-09-03: *"lets revamp the strength tab to just include all of that
great information into one tab. also make that tab resizable horizontally so I
can compress the capture/journal tab a bit and see more of the strength tab."*

What is true on `main` at 080495b, measured rather than read:

* the Strength column holds TWO `CollapsibleSection`s (RS/RW open, M5 Strength
  closed) plus the `FocusStrengthBoard` above them, which is not in a section at
  all;
* the RS Window is NOT on the BounceBot panel - `BouncePanel` has no tabs. It is
  `ui.panels.rs_window_panel.RsWindowPanel`, constructed in
  `TradingDeskPanel.__init__` as `RsWindowPanel(self.bounce_panel.service)` and
  added as a tab of `MasterAvwapWorkspace` (`trading_desk.py:550`), i.e. it lives
  in the SETUPS column;
* `tabs_row` is draggable today: at 1904 px, after the `splitterMoved` a real
  drag emits, `setSizes` to 30/70 reads back 30/70 and survives the next resize.
  What overrides a split is `desk_layout._PresetTracker`, which re-applies the
  60/40 `ALERT_TABS_ROW_WEIGHTS` on EVERY resize until the first drag - so a
  programmatic split, or any split set before a drag, is silently undone.

So the drag tests below are regression guards on a thing that already works, and
the red ones are the surface itself.
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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QScrollArea,
    QTabWidget,
    QWidget,
)

_QT_UNLIMITED = 16777215


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def desk(_qapp):
    """The real desk, with BounceBot's autostart disabled.

    `BouncePanel.__init__` ends in `QTimer.singleShot(0, self.start)`, so the
    first `processEvents` in a test connects to the live TWS on 7496. Nothing
    below is about BounceBot, and a test must not reach a broker.
    """
    from ui.panels.bounce_panel import BouncePanel

    started = BouncePanel.start
    BouncePanel.start = lambda self: None
    try:
        from ui.panels.trading_desk import TradingDeskPanel

        panel = TradingDeskPanel(workspace_mode="workspace")
        panel.resize(1920, 1080)
        panel.show()
        for _ in range(6):
            QApplication.processEvents()
        yield panel
        panel.shutdown()
        panel.close()
    finally:
        BouncePanel.start = started


def _sections(column: QWidget) -> list:
    from ui.widgets.collapsible_section import CollapsibleSection

    return column.findChildren(CollapsibleSection)


# ---------------------------------------------------------------------------
# one surface
# ---------------------------------------------------------------------------
def test_the_rs_window_is_no_longer_a_tab_in_the_setups_column(desk):
    """It moves; it is not copied. The old host must lose the page."""
    workspace = desk.master_workspace if hasattr(desk, "master_workspace") else None
    tab_widgets = [
        widget
        for widget in desk.findChildren(QTabWidget)
        if any(widget.tabText(i) == "RS Window" for i in range(widget.count()))
    ]
    assert tab_widgets == [], (
        "the RS Window page must move out of the setups column, found it on "
        f"{[w.objectName() or w.__class__.__name__ for w in tab_widgets]}"
        + (f" (workspace={workspace!r})" if workspace is not None else "")
    )


def test_the_strength_column_holds_three_sections_and_every_one_is_open(desk):
    from ui.panels.rs_window_panel import RsWindowPanel

    column = desk.alert_center.strength_column
    sections = _sections(column)
    titles = [section.header.text() for section in sections]
    assert len(sections) == 4, f"four sections, one Strength surface (Focus Strength joined - lead ruling 3, 2026-09-03); got {titles}"
    assert all(section.is_expanded() for section in sections), (
        f"every section starts OPEN; got {[(t, s.is_expanded()) for t, s in zip(titles, sections)]}"
    )

    # ...and they hold the three boards, not three empty headers.
    reached = [
        widget
        for section in sections
        for widget in [section.content()] + section.findChildren(QWidget)
        if widget is not None
    ]
    assert desk.alert_center.rrs_board_tab in reached, "the RS/RW board"
    assert desk.alert_center.strength_board_section in sections, "the M5 Strength Board"
    assert any(isinstance(widget, RsWindowPanel) for widget in reached), (
        "the RS Window widget, moved here from the setups tabs"
    )


def test_the_strength_column_scrolls_vertically_and_never_horizontally(desk):
    """ONE scroll area over the whole column, per the 452 px lesson."""
    from PySide6.QtCore import Qt

    column = desk.alert_center.strength_column
    sections = _sections(column)
    assert sections, "no sections to host"
    hosts = [
        area
        for area in column.findChildren(QScrollArea) + _ancestor_scrolls(column)
        if all(_is_ancestor(area, section) for section in sections)
    ]
    assert hosts, "all three sections must sit inside ONE scroll area"
    area = hosts[0]
    assert area.widgetResizable(), "the column stretches to the width it is given"
    assert (
        area.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    ), "it scrolls vertically and never horizontally"


def _ancestor_scrolls(widget: QWidget) -> list:
    found = []
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea):
            found.append(parent)
        parent = parent.parentWidget()
    return found


def _is_ancestor(area: QScrollArea, widget: QWidget) -> bool:
    parent = widget.parentWidget()
    while parent is not None:
        if parent is area:
            return True
        parent = parent.parentWidget()
    return False


# ---------------------------------------------------------------------------
# the boundary - regression guards, measured green on 080495b
# ---------------------------------------------------------------------------
@pytest.fixture
def alert_panel(_qapp):
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    panel.resize(1920, 1000)
    panel.show()
    for _ in range(4):
        QApplication.processEvents()
    yield panel
    panel.close()
    panel.deleteLater()


def test_the_capture_journal_tabs_can_be_dragged_down_to_three_tenths(alert_panel):
    """At 1920 the trader may hand 70% of the row to Strength, and it holds."""
    row = alert_panel.tabs_row
    row.splitterMoved.emit(1, 1)  # what a real drag tells the preset tracker
    total = sum(row.sizes())
    row.setSizes([int(total * 0.30), total - int(total * 0.30)])
    QApplication.processEvents()

    sizes = row.sizes()
    assert sizes[1] / sum(sizes) >= 0.70, sizes

    row.resize(1920, max(row.height(), 400))
    QApplication.processEvents()
    after = row.sizes()
    assert after[1] / sum(after) >= 0.70, (
        f"the split must survive the next relayout, {sizes} -> {after}"
    )


def test_the_dragged_split_round_trips_through_desk_layout(alert_panel, monkeypatch):
    """Saved and restored by the module that owns every desk split."""
    from ui.panels import desk_layout

    store: dict = {}
    monkeypatch.setattr(desk_layout, "save_local_setting", lambda key, value: store.__setitem__(key, value))
    monkeypatch.setattr(desk_layout, "get_local_setting", lambda key, default=None: store.get(key, default))

    row = alert_panel.tabs_row
    total = sum(row.sizes())
    dragged = [int(total * 0.30), total - int(total * 0.30)]
    desk_layout.save_local_setting("qt_alert_tabs_row_split_sizes_v1", dragged)

    restored = desk_layout.load_sizes("qt_alert_tabs_row_split_sizes_v1", 2)
    assert restored == dragged, restored
    desk_layout.apply_saved_sizes(
        row, "qt_alert_tabs_row_split_sizes_v1", desk_layout.ALERT_TABS_ROW_WEIGHTS
    )
    QApplication.processEvents()
    sizes = row.sizes()
    assert sizes[1] / sum(sizes) >= 0.70, sizes


def test_nothing_in_the_strength_column_carries_a_fixed_width(alert_panel):
    column = alert_panel.strength_column
    pinned = [
        (widget.__class__.__name__, widget.objectName(), widget.maximumWidth())
        for widget in [column] + column.findChildren(QWidget)
        if widget.maximumWidth() < _QT_UNLIMITED
        # Qt's own scroll-area corner containers carry a 14 px cap.
        and not widget.objectName().startswith("qt_scrollarea")
    ]
    assert pinned == [], pinned


def test_the_alert_column_floor_is_not_raised_by_the_new_surface(alert_panel):
    """Measured 932 px on 080495b (`chart_review.minimumSizeHint()` is 878 of
    it). The floor stays a FLOOR: this surface may not push it up."""
    assert alert_panel.minimumSizeHint().width() <= 932, (
        alert_panel.minimumSizeHint().width()
    )
    assert alert_panel.tabs.minimumWidth() <= 170


def test_the_arm_bar_is_still_under_the_chart(alert_panel):
    """Never moved without asking (CLAUDE.md). Its host is the chart pane."""
    review = alert_panel.chart_review
    assert review.arm_bar.parentWidget() is not None
    parent = review.arm_bar.parentWidget()
    while parent is not None and parent is not review:
        parent = parent.parentWidget()
    assert parent is review, "the arm bar stays inside AlertChartReview"
