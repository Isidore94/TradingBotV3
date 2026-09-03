"""S1 - the three behaviours the tester's packet left without a test.

Added by the builder, never in place of one of the tester's: each of these
pins something the packet asks for that no red test covered.

* S1.4's "in tabs mode the Alert Center tab is raised" - the packet names it
  and the tester's handoff says it is not pinned, because it needs a second
  desk built in `tabs` mode.
* S1.3's actual trader symptom - *"the width survives a restart"* - which is a
  RESTORED split not being snapped back by the next Resize. `_PresetTracker`
  already treats a saved split as user-set; nothing asserted it, so nothing
  would have caught its removal.
* S1.3's Strength surface as it SHIPPED: four open sections, not three. The
  tester wrote `== 3` before the lead ruled that `FocusStrengthBoard` becomes
  the first section; the lead has since tightened that assertion to `== 4`, and
  this one states the ORDER and the hosting the other does not.
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

from PySide6.QtWidgets import QApplication, QTabWidget  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _no_broker_from_a_test(monkeypatch):
    """`BouncePanel.__init__` ends in `QTimer.singleShot(0, self.start)`."""
    from ui.panels.bounce_panel import BouncePanel

    monkeypatch.setattr(BouncePanel, "start", lambda self: None)


def _row():
    from ui.models.setup import SetupRow

    return SetupRow(
        symbol="LNG",
        side="SHORT",
        score=245.0,
        bucket="favorite_setup",
        setup_tags=["AVWAP_BREAKOUT"],
        expected_r=0.85,
        raw={"setup_family": "avwap_breakout"},
    )


def _cell(panel, key: str):
    column = next(
        index for index, (name, _label) in enumerate(panel.model.COLUMNS) if name == key
    )
    return panel.proxy.index(0, column)


# ---------------------------------------------------------------------------
# S1.4 - the chart has to be SEEN, not just drawn
# ---------------------------------------------------------------------------
def test_in_tabs_mode_a_setups_ticker_click_raises_the_alert_center(monkeypatch):
    from ui.panels.alert_center_panel import AlertCenterPanel

    charted: list = []
    monkeypatch.setattr(
        AlertCenterPanel,
        "chart_symbol",
        lambda self, symbol, *, side="", origin="": charted.append(symbol) or True,
    )

    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="tabs")
    try:
        tabs = desk._mode_widget
        assert isinstance(tabs, QTabWidget)
        tabs.setCurrentWidget(desk.master_workspace)
        desk.master_panel.set_rows([_row()])
        QApplication.processEvents()

        desk.master_panel._on_table_clicked(_cell(desk.master_panel, "symbol"))
        QApplication.processEvents()

        assert charted == ["LNG"]
        assert tabs.currentWidget() is desk.alert_center, (
            "a chart the trader cannot see is the same as no chart at all"
        )
    finally:
        desk.shutdown()
        desk.close()


def test_in_workspace_mode_nothing_is_raised_and_the_chart_still_happens(monkeypatch):
    """Both columns are already on screen, so there is nothing to bring forward."""
    from ui.panels.alert_center_panel import AlertCenterPanel

    charted: list = []
    monkeypatch.setattr(
        AlertCenterPanel,
        "chart_symbol",
        lambda self, symbol, *, side="", origin="": charted.append((symbol, side))
        or True,
    )

    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        desk.master_panel.set_rows([_row()])
        QApplication.processEvents()
        desk.master_panel._on_table_clicked(_cell(desk.master_panel, "symbol"))
        assert charted == [("LNG", "SHORT")]
        assert not isinstance(desk._mode_widget, QTabWidget)
    finally:
        desk.shutdown()
        desk.close()


def test_one_double_click_charts_the_ticker_once(monkeypatch):
    """Qt sends `clicked` and then `doubleClicked` for one gesture."""
    from focus_picks import FocusPickStore
    from ui.panels.master_avwap_panel import MasterAvwapPanel
    from ui.services.focus_service import FocusService
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as raw:
        tmp = Path(raw)
        service = FocusService(
            FocusPickStore(
                focus_longs_path=tmp / "focus_longs.txt",
                focus_shorts_path=tmp / "focus_shorts.txt",
                longs_path=tmp / "longs.txt",
                shorts_path=tmp / "shorts.txt",
                membership_path=tmp / "focus_pick_membership.json",
            )
        )
        panel = MasterAvwapPanel(service, review_events_path=tmp / "events.jsonl")
        panel.set_rows([_row()])
        seen: list = []
        panel.symbolActivated.connect(lambda symbol, side: seen.append(symbol))

        index = _cell(panel, "symbol")
        panel._on_table_clicked(index)
        panel._open_symbol_snapshot_from_double_click(index)

        assert seen == ["LNG"], seen
        panel.deleteLater()


# ---------------------------------------------------------------------------
# S1.3 - the width survives a restart
# ---------------------------------------------------------------------------
def test_a_restored_saved_split_is_not_snapped_back_by_the_next_resize(monkeypatch):
    """THE TRADER'S ACTUAL SYMPTOM: "it does not survive a restart".

    `apply_saved_sizes` restores the split at construction and `track_preset`
    then re-applies the preset on every Resize until the trader drags - so if a
    restored split did not already count as user-set, the first paint after
    launch would scale 60/40 straight over the top of it and the drag would be
    lost every single time.
    """
    from PySide6.QtWidgets import QSplitter, QWidget

    from ui.panels import desk_layout

    store: dict = {}
    monkeypatch.setattr(
        desk_layout, "get_local_setting", lambda key, default=None: store.get(key, default)
    )
    monkeypatch.setattr(
        desk_layout, "save_local_setting", lambda key, value: store.__setitem__(key, value)
    )

    owner = QWidget()
    owner.resize(1920, 400)
    splitter = QSplitter(owner)
    for _ in range(2):
        splitter.addWidget(QWidget())
    splitter.setChildrenCollapsible(False)
    splitter.resize(1920, 300)
    splitter.show()
    QApplication.processEvents()

    key = "test_restart_split"
    store[key] = [576, 1344]  # 30/70: what the trader dragged yesterday

    desk_layout.apply_saved_sizes(splitter, key, desk_layout.ALERT_TABS_ROW_WEIGHTS)
    desk_layout.track_preset(
        owner, splitter, key, lambda _extent: desk_layout.ALERT_TABS_ROW_WEIGHTS
    )
    QApplication.processEvents()

    splitter.resize(1920, 320)  # the first real layout pass after launch
    QApplication.processEvents()

    sizes = splitter.sizes()
    # Within a pixel or two of 30/70 - Qt spends the handle's width out of the
    # total - and nowhere near the 60/40 preset, which is the failure mode.
    share = sizes[1] / sum(sizes)
    assert abs(share - 0.70) < 0.01, (
        f"the saved split must survive the first resize, got {sizes}"
    )
    owner.close()
    owner.deleteLater()


def test_with_nothing_saved_the_preset_still_applies(monkeypatch):
    """The other half of the same rule: the preset is for a virgin splitter."""
    from PySide6.QtWidgets import QSplitter, QWidget

    from ui.panels import desk_layout

    monkeypatch.setattr(desk_layout, "get_local_setting", lambda key, default=None: default)

    owner = QWidget()
    owner.resize(1920, 400)
    splitter = QSplitter(owner)
    for _ in range(2):
        splitter.addWidget(QWidget())
    splitter.setChildrenCollapsible(False)
    splitter.resize(1920, 300)
    splitter.show()

    key = "test_virgin_split"
    desk_layout.apply_saved_sizes(splitter, key, desk_layout.ALERT_TABS_ROW_WEIGHTS)
    desk_layout.track_preset(
        owner, splitter, key, lambda _extent: desk_layout.ALERT_TABS_ROW_WEIGHTS
    )
    splitter.resize(1920, 320)
    QApplication.processEvents()

    sizes = splitter.sizes()
    assert abs(sizes[0] / sum(sizes) - 0.60) < 0.02, sizes
    owner.close()
    owner.deleteLater()


def test_a_failed_split_save_is_reported_rather_than_swallowed(monkeypatch, caplog):
    """A layout that will not survive a restart must say so somewhere."""
    from PySide6.QtWidgets import QSplitter, QWidget

    from ui.panels import desk_layout

    def _refuse(key, value):
        raise OSError("settings file is read-only")

    monkeypatch.setattr(desk_layout, "save_local_setting", _refuse)

    owner = QWidget()
    splitter = QSplitter(owner)
    for _ in range(2):
        splitter.addWidget(QWidget())
    desk_layout.persist_sizes(owner, splitter, "test_unwritable_split")
    timer = owner._split_save_timers["test_unwritable_split"]

    with caplog.at_level("WARNING"):
        timer.timeout.emit()

    assert any(
        "test_unwritable_split" in record.getMessage() for record in caplog.records
    ), caplog.records
    owner.deleteLater()


# ---------------------------------------------------------------------------
# S1.3 - the surface as it shipped
# ---------------------------------------------------------------------------
def test_the_strength_surface_is_four_open_sections_in_one_scroll_area():
    """What the lead ruled and what the code does.

    The tester's sibling counts the sections and checks they are open; this one
    pins their ORDER and that all four sit inside the one column scroll area -
    the two facts that make it a single surface rather than four widgets that
    happen to be stacked.
    """
    from ui.panels.rs_window_panel import RsWindowPanel
    from ui.widgets.collapsible_section import CollapsibleSection
    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        desk.resize(1920, 1080)
        desk.show()
        for _ in range(4):
            QApplication.processEvents()
        column = desk.alert_center.strength_column
        sections = column.findChildren(CollapsibleSection)

        assert [section.header.text() for section in sections] == [
            "Focus Strength",
            "RS/RW Board",
            "M5 Strength Board (TC2000)",
            "RS Window",
        ]
        assert all(section.is_expanded() for section in sections)
        assert isinstance(desk.alert_center.rs_window_panel, RsWindowPanel)

        # And every one of them is inside the ONE column scroll area.
        area = desk.alert_center.strength_scroll
        for section in sections:
            parent = section.parentWidget()
            while parent is not None and parent is not area:
                parent = parent.parentWidget()
            assert parent is area, section.header.text()
    finally:
        desk.shutdown()
        desk.close()


def test_the_boundary_handle_is_wide_enough_to_grab():
    """Qt's default was 4 px - measured 2026-09-03 - for a drag the trader
    asked to make routine."""
    from ui.panels.alert_center_panel import AlertCenterPanel, TABS_ROW_HANDLE_PX
    from ui import theme

    panel = AlertCenterPanel()
    try:
        assert TABS_ROW_HANDLE_PX == 8
        assert panel.tabs_row.handleWidth() == theme.px(TABS_ROW_HANDLE_PX)
        assert panel.tabs_row.handleWidth() > 4
    finally:
        panel.close()
        panel.deleteLater()


# ---------------------------------------------------------------------------
# S1.5 - no test writes the trader's own decision stream
# ---------------------------------------------------------------------------
def test_the_review_event_tests_redirect_every_verdict_writer(tmp_path):
    """The leak S1.5 closed, pinned so it cannot reopen quietly.

    `ui.annotations.verdicts` defaults `path` to the LIVE
    `trader_annotations.jsonl`, and `tests/test_review_events.py` drove two
    panel hooks that reach it. Its autouse fixture now rebinds every writer;
    this asserts the fixture exists and names all four.
    """
    source = (ROOT_DIR / "tests" / "test_review_events.py").read_text(encoding="utf-8")
    assert "_annotations_go_to_a_temp_file" in source
    for name in ("record_like", "record_dislike", "record_not_today", "record_note_on"):
        assert name in source, name
