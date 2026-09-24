"""The compact desk layout (trader, 2026-09-23: "Build it. Make a setting in the
settings tab that lets me select old vs new UI").

Presentation only. Compact hides the left menu, the tape and the BounceBot
strip, adds page tabs, moves the Movers column right, turns the Alert Center
tabs into a drawer and folds the arm bar into one row. Classic must come back
exactly as it was, and every proxy must drive the REAL control.
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

pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLayout,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QWidget,
)

from ui.app import PAGE_SPECS  # noqa: E402
from ui.state import UiState  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _pump(times: int = 20) -> None:
    for _ in range(times):
        _app.processEvents()


@pytest.fixture(scope="module")
def window():
    from ui.app import MainWindow

    made = MainWindow(UiState(workspace_mode="workspace", desk_layout="compact"))
    made.resize(2560, 1400)
    made.show()
    _pump(40)
    yield made
    try:
        made.close()
    except Exception:
        pass


def _switch(window, layout: str) -> None:
    window.state.desk_layout = layout
    window._apply_state_changes()
    _pump(30)


# ------------------------------------------------------------------ setting
def test_desk_layout_setting_round_trips_and_defaults_to_compact():
    assert UiState().desk_layout == "compact"
    state = UiState(desk_layout="classic")
    state.save()
    assert UiState.load().desk_layout == "classic"
    state.desk_layout = "compact"
    state.save()
    assert UiState.load().desk_layout == "compact"
    from project_paths import save_local_setting

    save_local_setting("qt_desk_layout", "bogus")
    assert UiState.load().desk_layout == "compact"


def test_settings_combo_saves_the_layout_through_state_changed():
    from ui.panels.settings_panel import SettingsPanel

    state = UiState(desk_layout="compact")
    panel = SettingsPanel(state)
    fired = []
    panel.stateChanged.connect(lambda: fired.append(True))
    assert panel.desk_layout_input.currentText() == "New (compact)"
    panel.desk_layout_input.setCurrentText("Old (classic)")
    assert state.desk_layout == "classic"
    assert fired
    assert UiState.load().desk_layout == "classic"
    panel.shutdown()


# ------------------------------------------------------------- the shell
def test_compact_hides_nav_tape_and_bounce_strip_and_shows_the_tab_row(window):
    _switch(window, "compact")
    desk = window.trading_panel
    assert window.desk_layout() == "compact"
    assert not window.nav_rail.isVisible()
    assert not window.top_bar.isVisible()
    assert window.page_tab_row.isVisible()
    assert not desk.tape_host.isVisible()
    assert not desk.bounce_panel.isVisible()
    assert window.bounce_status_proxy.isVisible()
    # The setups toggle and mode buttons ride the tab row on the Desk page.
    assert desk.setups_toggle.isVisible()
    assert window.page_tab_row.isAncestorOf(desk.setups_toggle)
    assert window.page_tab_row.isAncestorOf(window.workspace_button)
    # Movers stand in their own desk column, right of the Alert Center.
    splitter = desk.desk_splitter
    order = [splitter.widget(i) for i in range(splitter.count())]
    assert order == [
        desk.m5_column,
        desk.alert_center,
        desk.alert_center.movers_column,
        desk.d1_column,
    ]


def test_the_page_tab_row_reaches_every_page_exactly_once(window):
    row = window.page_tab_row
    reached = row.page_indices()
    assert sorted(reached) == list(range(len(PAGE_SPECS)))
    assert len(reached) == len(set(reached))
    tab_titles = [PAGE_SPECS[index].title for index in row.tab_buttons]
    assert tab_titles == ["Trading Desk", "Journal", "Day Review", "Research"]
    assert row.tab_buttons[0].text() == "Desk"
    # Each entry routes through _select_page.
    for index in reached:
        if index in row.tab_buttons:
            row.tab_buttons[index].click()
        else:
            row.more_actions[index].trigger()
        _pump(2)
        assert window.pages.currentIndex() == index
        assert window.nav_buttons[index].isChecked()
        assert row.more_button.isChecked() == (index in row.more_actions)
    window._select_page(0)
    _pump(5)


def test_universe_hidden_in_the_more_menu_like_its_nav_button(window):
    index = [spec.title for spec in PAGE_SPECS].index("Universe")
    window.apply_unused_surface_visibility()
    assert window.page_tab_row.more_actions[index].isVisible() == (
        not window.nav_buttons[index].isHidden()
    )


def test_review_badges_mirror_onto_the_tab_row(window):
    journal = [spec.title for spec in PAGE_SPECS].index("Journal")
    window._apply_tag_review_badge(3)
    assert window.nav_buttons[journal].text() == "Journal (3 to review)"
    assert window.page_tab_row.label(journal) == "Journal (3 to review)"
    window._apply_tag_review_badge(0)
    assert window.page_tab_row.label(journal) == "Journal"


def test_setups_toggle_only_shows_on_the_desk_page(window):
    toggle = window.trading_panel.setups_toggle
    window._select_page(1)
    _pump(2)
    assert not toggle.isVisible()
    window._select_page(0)
    _pump(2)
    assert toggle.isVisible()


def test_the_tape_service_is_paused_in_compact_and_resumed_in_classic(window):
    service = window.trading_panel.group_tape_service
    assert service.paused
    _switch(window, "classic")
    assert not service.paused
    _switch(window, "compact")
    assert service.paused


def test_a_paused_tape_service_makes_no_fetch():
    from ui.services.group_tape_service import GroupTapeService

    calls = []
    service = GroupTapeService(downloader=lambda *a, **k: calls.append(1))
    service.pause()
    service._tick()
    assert not service.running
    assert service._last_attempt is None
    assert calls == []
    service.shutdown()


# ------------------------------------------------------------- the drawer
def test_drawer_opens_collapsed_and_a_tab_click_toggles_it(window):
    _switch(window, "compact")
    center = window.trading_panel.alert_center
    drawer = center._drawer
    assert drawer.is_active()
    assert not drawer.is_expanded()
    collapsed = drawer.collapsed_height()
    assert center.tabs_row.height() <= collapsed + 4
    current = center.tabs.currentIndex()
    center.tabs.tabBarClicked.emit(current)
    _pump(5)
    assert drawer.is_expanded()
    assert center.tabs_row.height() > collapsed + 100
    center.tabs.tabBarClicked.emit(current)
    _pump(5)
    assert not drawer.is_expanded()
    assert center.tabs_row.height() <= collapsed + 4


def test_a_capture_hotkey_opens_the_drawer_on_capture(window):
    center = window.trading_panel.alert_center
    center._drawer.collapse()
    shortcut = next(iter(center._capture_shortcuts.values()))
    shortcut.activated.emit()
    _pump(5)
    assert center._drawer.is_expanded()
    assert center.tabs.currentIndex() == center._capture_tab_index
    center._drawer.collapse()
    center._journal_route_shortcut.activated.emit()
    _pump(5)
    assert center._drawer.is_expanded()
    assert center.tabs.currentIndex() == center._journal_tab_index
    center._drawer.collapse()


# ------------------------------------------------------------ the arm row
def test_compact_arm_menus_click_the_real_buttons():
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    bar.set_compact(True)
    bar.set_enabled_for_symbol(True)
    watched, d1, anyb, ext = [], [], [], []
    bar.watchToggled.connect(watched.append)
    bar.d1EventToggled.connect(d1.append)
    bar.anyBounceToggled.connect(lambda: anyb.append(1))
    bar.externalChartRequested.connect(ext.append)
    m5_kind = next(iter(bar.m5_actions))
    bar.m5_actions[m5_kind].trigger()
    assert watched == [m5_kind]
    d1_kind = next(iter(bar.d1_event_buttons))
    bar.d1_actions[d1_kind].trigger()
    assert d1 == [d1_kind]
    bar.d1_actions["any_bounce"].trigger()
    assert anyb == [1]
    bar.d1_actions["external_chart"].trigger()
    assert len(ext) == 1


def test_compact_arm_menus_mirror_armed_enabled_and_tooltip():
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    bar.set_compact(True)
    kind = next(iter(bar.watch_buttons))
    assert not bar.m5_actions[kind].isEnabled()
    assert not bar.m5_menu_button.isEnabled()
    bar.set_enabled_for_symbol(True)
    assert bar.m5_actions[kind].isEnabled()
    bar.set_armed_kinds([kind])
    assert bar.m5_actions[kind].isChecked()
    assert bar.m5_actions[kind].text() == bar.watch_buttons[kind].text()
    assert bar.m5_menu_button.text() == "M5 alert (1) ▾"
    bar.set_pending_arms(d1_kinds=[next(iter(bar.d1_event_buttons))])
    assert bar.d1_menu_button.text() == "D1 alert (1) ▾"
    bar.set_watch_availability(False, "no bars yet")
    assert "no bars yet" in bar.m5_actions[kind].toolTip()


def test_compact_puts_the_arm_bar_and_verbs_on_one_flow(window):
    review = window.trading_panel.alert_center.chart_review
    assert review.compact_controls_enabled()
    assert review.arm_bar.is_compact()
    assert review.compact_controls.isAncestorOf(review.arm_bar)
    assert review.compact_controls.isAncestorOf(review.focus_button)
    assert not review.arm_bar.watch_buttons[next(iter(review.arm_bar.watch_buttons))].isVisible()


# ------------------------------------------------------- the status proxies
def test_status_proxies_click_the_real_bounce_controls(window):
    panel = window.trading_panel.bounce_panel
    proxy = window.bounce_status_proxy
    clicked = []
    panel.start_scanning_button.clicked.connect(lambda *_: clicked.append("start"))
    panel.stop_scanning_button.clicked.connect(lambda *_: clicked.append("stop"))
    panel.start_scanning_button.setEnabled(True)
    panel.stop_scanning_button.setEnabled(False)
    assert proxy.start_button.isEnabled() and not proxy.stop_button.isEnabled()
    proxy.start_button.click()
    assert clicked == ["start"]
    panel.stop_scanning_button.setEnabled(True)
    proxy.stop_button.click()
    assert clicked == ["start", "stop"]
    fired = []
    button = next(iter(panel.entry_assist_buttons.values()))
    button.setEnabled(True)
    button.clicked.connect(lambda *_: fired.append(1))
    proxy.sync_entry_actions()
    next(iter(proxy.entry_actions.values())).trigger()
    assert fired == [1]
    combo = panel.environment_input
    if combo.count() > 1:
        combo.blockSignals(True)
        proxy.mode_actions[1].trigger()
        combo.blockSignals(False)
        assert combo.currentIndex() == 1
        combo.blockSignals(True)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)


# ------------------------------------------------------------ round trip
def _tree(widget: QWidget) -> list:
    """Visible widget structure in layout order: (class, objectName, children)."""

    def children(w: QWidget) -> list:
        found: list = []
        if isinstance(w, QSplitter):
            kids = [w.widget(i) for i in range(w.count())]
        elif isinstance(w, (QTabWidget, QStackedWidget)):
            kids = [w.widget(i) for i in range(w.count())]
        elif isinstance(w, QScrollArea):
            kids = [w.widget()] if w.widget() is not None else []
        else:
            kids = []
            layout = w.layout()
            if layout is not None:
                _layout_widgets(layout, kids)
        for kid in kids:
            if kid is not None and not kid.isHidden():
                found.append(_tree(kid))
        return found

    return [type(widget).__name__, widget.objectName(), children(widget)]


def _layout_widgets(layout: QLayout, out: list) -> None:
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is not None:
            out.append(item.widget())
        elif item.layout() is not None:
            _layout_widgets(item.layout(), out)


def test_classic_compact_classic_restores_the_classic_tree(window):
    _switch(window, "classic")
    desk = window.trading_panel
    center = desk.alert_center
    before = _tree(window.centralWidget())
    before_status = _tree(window.statusBar())
    assert window.nav_rail.isVisible() and window.top_bar.isVisible()
    assert desk.tape_host.isVisible() and desk.bounce_panel.isVisible()
    assert center.movers_column.parent() is center.tabs_row
    assert center.chart_review.parent() is center.splitter
    _switch(window, "compact")
    assert _tree(window.centralWidget()) != before
    _switch(window, "classic")
    assert _tree(window.centralWidget()) == before
    assert _tree(window.statusBar()) == before_status
    assert center.movers_column.parent() is center.tabs_row
    assert center.chart_review.parent() is center.splitter
    assert center.tabs_row.widget(1) is center.movers_column
    assert not center.chart_review.arm_bar.is_compact()
    assert desk.setups_toggle.parent() is desk.tape_host
    assert window.workspace_button.parent() is window.top_bar
    _switch(window, "compact")


def test_tabs_mode_keeps_the_classic_alert_center_inside(window):
    _switch(window, "compact")
    window._set_workspace_mode("tabs")
    _pump(20)
    desk = window.trading_panel
    center = desk.alert_center
    assert not center.is_compact_layout()
    assert center.movers_column.parent() is center.tabs_row
    tabs = desk._mode_widget
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert "BounceBot" not in titles  # the strip is proxied in the status bar
    assert window.page_tab_row.isVisible()
    window._set_workspace_mode("workspace")
    _pump(20)
    assert center.is_compact_layout()
