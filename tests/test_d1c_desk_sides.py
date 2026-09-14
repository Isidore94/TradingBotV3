"""D1C-L: M5 trades on the LEFT, D1 picks and their management on the RIGHT.

Trader, 2026-09-14: *"Keep M5 trades and entries on the left. Put D1 picks and
their management on the right. Inspect the existing swing-favorites strip,
which currently sits below the left M5 list, and reconcile it with this layout
without losing its actions."*

That supersedes the trader's own earlier word (2026-08-31, *"put it at the very
bottom of the M5 alerts tab"*), which is what
`tests/test_qt_swing_favorites.py::TestWhereItLives` pinned until today.

What these defend:

* the left column is the M5 alert bar ALONE - no swing favorites strip in it -
  and the Working-lately line is still the first thing inside the bar (ST6.4);
* the right column is a vertical split: the Master AVWAP workspace on top, the
  swing favorites strip under it, its own settings key, neither pane able to
  collapse;
* both desk modes mount it - in tabs mode the strip rides the "Master AVWAP"
  tab, not the "M5 alerts" tab - and a mode round trip costs the strip neither
  its chips nor one of its signals;
* the strip hides and shows WITH the setups column, because the trader opens
  that column to look at D1;
* the desk no longer writes the M5 column's split key.
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

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QSplitter, QTabWidget  # noqa: E402

#: The retired left-column key, spelled out rather than imported: after D1C-L
#: the desk neither applies nor writes it, and whether the constant survives in
#: `trading_desk` is the builder's call. What must stay true is that this
#: string is never written again, so the trader's saved drag is left alone in
#: `local_settings.json`.
RETIRED_M5_SPLIT_KEY = "qt_m5_column_split_sizes_v1"


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _pump(times: int = 6) -> None:
    app = QApplication.instance()
    for _ in range(times):
        app.processEvents()


def _ancestors(widget) -> list:
    """Every parent above `widget`, nearest first."""
    chain = []
    node = widget.parent() if widget is not None else None
    while node is not None:
        chain.append(node)
        node = node.parent()
    return chain


def _desk(*, shown: bool = False, setups_visible: bool | None = None):
    """A Trading Desk in workspace mode, built offscreen.

    Modelled on `tests/test_qt_desk_layout.py::_desk` and
    `tests/test_qt_swing_favorites.py::TestWhereItLives`.
    """
    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="workspace")
    if shown:
        desk.resize(1640, 980)
        desk.show()
    if setups_visible is not None:
        desk.set_setups_visible(setups_visible)
    _pump()
    return desk


def _close(desk) -> None:
    desk.shutdown()
    desk.close()


# ---------------------------------------------------------------------------
# Item 1 - the left column is M5 only
# ---------------------------------------------------------------------------
def test_the_left_column_is_the_m5_bar_alone_with_working_lately_still_on_top():
    """M5 trades and entries on the left: the alert bar and nothing else.

    The Working-lately line stays mounted INSIDE the bar (ST6.4), so moving the
    strip out must not take it with it.
    """
    from ui.widgets.m5_alert_bar import M5AlertBar
    from ui.widgets.swing_favorites_bar import SwingFavoritesBar

    desk = _desk()
    try:
        left = desk.m5_column
        assert desk.desk_splitter.widget(0) is left, "the M5 host is still the left column"

        # The bar is the column, or the column's one child - the packet allows
        # a one-child splitter so the settings code stays simple.
        assert left is desk.m5_alert_bar or desk.m5_alert_bar in left.findChildren(M5AlertBar)

        assert left is not desk.swing_favorites_bar
        assert left.findChildren(SwingFavoritesBar) == [], (
            "the swing favorites strip is D1 management and belongs on the right"
        )
        assert desk.swing_favorites_bar not in left.findChildren(SwingFavoritesBar)

        # ST6.4 is untouched: the Working-lately line is the first thing in the bar.
        assert desk.m5_alert_bar.layout().itemAt(0).widget() is desk.working_lately_strip
    finally:
        _close(desk)


# ---------------------------------------------------------------------------
# Item 2 - the right column is D1: setups above, the strip below
# ---------------------------------------------------------------------------
def test_workspace_mode_puts_the_setups_over_the_strip_in_the_right_column():
    desk = _desk()
    try:
        splitter = desk.desk_splitter
        assert splitter is not None
        assert splitter.count() == 3
        assert splitter.widget(0) is desk.m5_column
        assert splitter.widget(1) is desk.alert_center
        assert splitter.widget(2) is desk.d1_column, "D1 picks live on the right"

        assert desk.d1_column.count() == 2
        assert desk.d1_column.widget(0) is desk.master_workspace, "setups on top"
        assert desk.d1_column.widget(1) is desk.swing_favorites_bar, "the strip under them"
    finally:
        _close(desk)


def test_the_right_column_split_has_its_own_key_that_neither_pane_can_collapse():
    """A new key, so the M5 column's saved drag is never replayed onto it, and
    a strip dragged to nothing is one the trader cannot find again."""
    from ui.panels import trading_desk

    assert trading_desk.D1_COLUMN_SPLIT_KEY == "qt_d1_column_split_sizes_v1"
    # The tester wrote (4, 1); the reviewer measured 399 px of chip area for
    # ~150 px of content at the trader's 3456 x 2160 and the lead took (6, 1)
    # on 2026-09-14. The setups lead either way and the drag still decides.
    assert trading_desk.D1_COLUMN_WEIGHTS == (6, 1)
    assert trading_desk.D1_COLUMN_WEIGHTS[0] > trading_desk.D1_COLUMN_WEIGHTS[1]
    assert trading_desk.D1_COLUMN_SPLIT_KEY != trading_desk.DESK_SPLIT_KEY
    assert trading_desk.D1_COLUMN_SPLIT_KEY != RETIRED_M5_SPLIT_KEY

    desk = _desk()
    try:
        column = desk.d1_column
        assert isinstance(column, QSplitter)
        assert column.orientation() == Qt.Orientation.Vertical
        assert column.childrenCollapsible() is False
        column.resize(600, 900)
        _pump()
        column.setSizes([900, 0])
        _pump()
        assert min(column.sizes()) > 0, f"neither pane collapses, got {column.sizes()}"
    finally:
        _close(desk)


def test_the_desk_no_longer_writes_the_m5_column_split_key(monkeypatch):
    """Item 1: the M5 key stops being applied. Its stored value is left alone
    in `local_settings.json`; nothing here reads or rewrites it."""
    from PySide6.QtTest import QTest

    from ui.panels import desk_layout, trading_desk

    applied: list[str] = []
    for name in ("apply_saved_sizes", "track_preset", "persist_sizes"):
        real = getattr(desk_layout, name)

        def _spy(*args, _real=real, **kwargs):
            applied.append(next((arg for arg in args if isinstance(arg, str)), ""))
            return _real(*args, **kwargs)

        monkeypatch.setattr(desk_layout, name, _spy)

    saved: list[str] = []
    monkeypatch.setattr(
        desk_layout, "save_local_setting", lambda key, value: saved.append(key)
    )

    desk = _desk()
    try:
        assert RETIRED_M5_SPLIT_KEY not in applied, (
            "the left column no longer carries the M5 bar/strip split"
        )
        assert applied.count(trading_desk.D1_COLUMN_SPLIT_KEY) >= 3, (
            "the right column goes through apply_saved_sizes/track_preset/persist_sizes"
        )
        assert RETIRED_M5_SPLIT_KEY not in getattr(desk, "_split_save_timers", {})
        assert trading_desk.D1_COLUMN_SPLIT_KEY in getattr(desk, "_split_save_timers", {})

        # A real drag of each column, then past the 400 ms save debounce.
        if isinstance(desk.m5_column, QSplitter):
            desk.m5_column.splitterMoved.emit(120, 1)
        desk.d1_column.splitterMoved.emit(400, 1)
        QTest.qWait(800)
        _pump()

        assert RETIRED_M5_SPLIT_KEY not in saved, (
            f"the desk must not write the M5 column key any more, wrote {saved}"
        )
        assert trading_desk.D1_COLUMN_SPLIT_KEY in saved, (
            f"dragging the right column saves its own key, wrote {saved}"
        )
    finally:
        _close(desk)


def test_the_setups_toggle_hides_and_shows_the_strip_with_the_column():
    """The strip is D1 management, so it comes and goes with the D1 column -
    the trader opens that column to look at D1. Out of sight while the column
    is hidden is the intended behaviour, not a failure."""
    desk = _desk(shown=True, setups_visible=False)
    try:
        assert desk.master_workspace.isVisibleTo(desk) is False
        assert desk.swing_favorites_bar.isVisibleTo(desk) is False, (
            "the strip hides with the setups column"
        )

        # F9: give the setups the desk. It reveals a hidden column first.
        assert desk.toggle_setups_expanded() is True
        _pump()
        assert desk.master_workspace.isVisibleTo(desk) is True
        assert desk.swing_favorites_bar.isVisibleTo(desk) is True, (
            "opening the column brings the strip with it"
        )

        desk.set_setups_visible(False)
        _pump()
        assert desk.master_workspace.isVisibleTo(desk) is False
        assert desk.swing_favorites_bar.isVisibleTo(desk) is False
    finally:
        _close(desk)


# ---------------------------------------------------------------------------
# Item 2 - tabs mode mounts the same column
# ---------------------------------------------------------------------------
def test_the_master_avwap_tab_hosts_the_strip_and_the_m5_alerts_tab_does_not():
    from ui.widgets.swing_favorites_bar import SwingFavoritesBar

    desk = _desk()
    try:
        desk.set_mode("tabs")
        _pump()
        tabs = desk._mode_widget
        assert isinstance(tabs, QTabWidget)

        d1_index = tabs.indexOf(desk.d1_column)
        assert d1_index >= 0 and tabs.tabText(d1_index) == "Master AVWAP"
        m5_index = tabs.indexOf(desk.m5_column)
        assert m5_index >= 0 and tabs.tabText(m5_index) == "M5 alerts"

        assert desk.d1_column in _ancestors(desk.swing_favorites_bar)
        assert desk.m5_column not in _ancestors(desk.swing_favorites_bar)
        assert tabs.widget(m5_index).findChildren(SwingFavoritesBar) == []
        assert desk.swing_favorites_bar in tabs.widget(d1_index).findChildren(
            SwingFavoritesBar
        )
    finally:
        _close(desk)


# ---------------------------------------------------------------------------
# Item 3 - nothing the strip does is lost by the move
# ---------------------------------------------------------------------------
def test_a_mode_round_trip_keeps_the_chips_and_every_strip_action_still_lands():
    """"...without losing its actions": text + Enter, the side toggle, Add,
    Paste, Copy, a chip's X, `firstShown`, `set_taken`, `set_status` and the
    fade retraction all still reach what they reached before."""
    from PySide6.QtTest import QTest

    desk = _desk()
    bar = desk.swing_favorites_bar
    try:
        bar.set_favorites(
            [{"symbol": "NVDA", "side": "long"}, {"symbol": "AMD", "side": "short"}]
        )
        desk.set_mode("tabs")
        _pump()
        desk.set_mode("workspace")
        _pump()

        assert desk.d1_column in _ancestors(bar), "the strip rides the right column"
        assert bar.symbols() == [("NVDA", "long"), ("AMD", "short")], (
            "a mode switch must not cost the strip its chips"
        )

        service = desk.swing_favorites_service
        added: list[tuple[str, str]] = []
        removed: list[tuple[str, str]] = []
        retracted: list[list] = []
        service.add = lambda text, side: (added.append((text, side)) or ["TSLA"])
        service.remove = lambda symbol, side: (removed.append((symbol, side)) or True)
        service.favorites = lambda *_a, **_kw: [{"symbol": "SPY", "side": "long"}]
        service.retract_faded_picks = lambda faded: (retracted.append(list(faded)) or 1)

        # Text + Enter, on the selected side.
        bar.short_button.click()
        assert bar.side() == "short"
        bar.input.setText("tsla")
        QTest.keyClick(bar.input, Qt.Key.Key_Return)
        assert added == [("tsla", "short")]
        assert "TSLA" in bar.status_label.text()

        # The Add button.
        bar.long_button.click()
        bar.input.setText("meta")
        bar.add_button.click()
        assert added[-1] == ("meta", "long")

        # Paste.
        QApplication.clipboard().setText("amzn\ngoog")
        bar.paste_button.click()
        assert added[-1] == ("amzn\ngoog", "long")

        # Copy.
        bar.set_favorites(
            [{"symbol": "NVDA", "side": "long"}, {"symbol": "AMD", "side": "short"}]
        )
        bar.copy_button.click()
        assert QApplication.clipboard().text() == "NVDA\nAMD"

        # A chip's X.
        chip = bar._current_chips()[0]
        chip.removed.emit(chip.symbol, chip.side)
        assert removed == [("NVDA", "long")]
        assert "NVDA" in bar.status_label.text()

        # The service's three signals still reach the strip.
        service.takenChanged.emit({("NVDA", "long"): "T-1"})
        assert bar._current_chips()[0].taken_label.isVisibleTo(bar) is True
        service.statusChanged.emit("hello from the service")
        assert bar.status_label.text() == "hello from the service"

        # `firstShown` still re-derives the day's list from the service.
        bar.firstShown.emit()
        assert bar.symbols() == [("SPY", "long")]

        # A3: a faded pick still appends its retraction.
        desk.focus_service.picksFaded.emit([{"symbol": "NVDA", "side": "long"}])
        assert retracted == [[{"symbol": "NVDA", "side": "long"}]]
    finally:
        _close(desk)


# ---------------------------------------------------------------------------
# Item 2 - the doors that pointed at the workspace still open (builder-added)
# ---------------------------------------------------------------------------
def test_show_watchlist_still_raises_the_master_avwap_tab_in_tabs_mode():
    """WS-WL's door: the Journal's "Positions on the Watchlist" is a NAV call.

    In tabs mode it has to raise the tab the workspace lives on. That tab now
    holds the D1 COLUMN, so a `setCurrentWidget(master_workspace)` would find
    no tab, raise nothing, and the button would silently do half its job.
    """
    desk = _desk()
    try:
        desk.set_mode("tabs")
        _pump()
        tabs = desk._mode_widget
        assert isinstance(tabs, QTabWidget)
        tabs.setCurrentWidget(desk.bounce_panel)
        _pump()
        assert tabs.tabText(tabs.currentIndex()) == "BounceBot"

        assert desk.show_watchlist("positions") is True
        _pump()
        assert tabs.tabText(tabs.currentIndex()) == "Master AVWAP", (
            "the nav call must raise the tab the workspace is on"
        )
        assert desk.master_workspace.tabs.currentWidget() is desk.watchlist_tab
    finally:
        _close(desk)


def test_the_strip_keeps_its_chips_when_a_mode_switch_mounts_the_column():
    """A mode switch must not fire the strip's one-shot `firstShown`.

    `_detach_mode_panels` leaves the D1 column parentless for a moment; showing
    it there would show it as a top-level WINDOW and hand every child a real
    showEvent - and the strip answers its first showEvent by re-deriving the
    day's list from the store, which throws away whatever is on it.
    """
    desk = _desk()
    bar = desk.swing_favorites_bar
    try:
        shown: list[int] = []
        bar.firstShown.connect(lambda: shown.append(1))
        bar.set_favorites([{"symbol": "NVDA", "side": "long"}])
        desk.set_mode("tabs")
        _pump()
        assert shown == [], "a mode switch is not the trader seeing the strip"
        assert bar.symbols() == [("NVDA", "long")]
        assert bar.isVisible() is False, "nothing was shown as a stray window"
    finally:
        _close(desk)
