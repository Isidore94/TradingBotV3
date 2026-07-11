#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import threading
from datetime import datetime

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from project_paths import get_shared_watchlist_details, get_tracker_storage_details
from ui.panels.autopilot_panel import AutopilotPanel
from ui.panels.journal_panel import JournalPanel
from ui.panels.research_panel import ResearchPanel
from ui.panels.settings_panel import SettingsPanel
from ui.panels.trading_desk import TradingDeskPanel
from ui.panels.universe_panel import UniversePanel
from ui.state import UiState
from ui.theme import apply_theme


class MainWindow(QMainWindow):
    def __init__(self, state: UiState) -> None:
        super().__init__()
        self.state = state
        self.setWindowTitle("TradingBotV3 Trading Desk")
        self.resize(1640, 980)
        self.setMinimumSize(1180, 760)

        self.trading_panel = TradingDeskPanel(workspace_mode=self.state.workspace_mode)
        self.journal_panel = JournalPanel()
        self.universe_panel = UniversePanel()
        self.research_panel = ResearchPanel()
        self.autopilot_panel = AutopilotPanel(bounce_service=self.trading_panel.bounce_panel.service)
        self.settings_panel = SettingsPanel(self.state, bounce_service=self.trading_panel.bounce_panel.service)
        self.settings_panel.stateChanged.connect(self._apply_state_changes)

        self.pages = QStackedWidget()
        self.pages.addWidget(self.trading_panel)
        self.pages.addWidget(self.trading_panel.focus_picks_panel)
        self.pages.addWidget(self.journal_panel)
        self.pages.addWidget(self.universe_panel)
        self.pages.addWidget(self.research_panel)
        self.pages.addWidget(self.autopilot_panel)
        self.pages.addWidget(self.settings_panel)

        self.title_label = QLabel("Trading Desk")
        self.title_label.setObjectName("TitleLabel")

        self.mode_group = QButtonGroup(self)
        self.workspace_button = _mode_button("Workspace")
        self.tabs_button = _mode_button("Tabs")
        self.mode_group.addButton(self.workspace_button)
        self.mode_group.addButton(self.tabs_button)
        self.workspace_button.clicked.connect(lambda: self._set_workspace_mode("workspace"))
        self.tabs_button.clicked.connect(lambda: self._set_workspace_mode("tabs"))

        self.nav_buttons: list[QPushButton] = []
        self._build_shell()
        self._build_status_bar()
        self._bind_shortcuts()
        self._sync_mode_buttons()

        self.trading_panel.statusChanged.connect(self._set_scan_status)
        self.trading_panel.rowsChanged.connect(self._set_setup_counts)
        self.trading_panel.connectionChanged.connect(self._set_ib_status)

        # Self-heal a stale universe on launch AND on a recurring check (the
        # app often stays open across sessions, so launch-only healing left
        # the universe stale all day whenever the one launch attempt failed
        # or the close simply passed while running). The swing scans fold
        # universe_longs/shorts into every run, so a stale pool quietly
        # degrades manual scans too. yfinance-only, in a background thread -
        # IB and the UI are untouched; the rebuild lock dedupes callers.
        QTimer.singleShot(2500, self._self_heal_universe)
        self._universe_heal_timer = QTimer(self)
        self._universe_heal_timer.setInterval(30 * 60_000)
        self._universe_heal_timer.timeout.connect(self._self_heal_universe)
        self._universe_heal_timer.start()

    def _build_shell(self) -> None:
        nav = QFrame()
        nav.setObjectName("NavRail")
        nav.setMinimumWidth(178)
        nav.setMaximumWidth(220)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(10, 10, 10, 10)
        nav_layout.setSpacing(8)

        brand = QLabel("TradingBotV3")
        brand.setObjectName("SectionTitle")
        nav_layout.addWidget(brand)
        nav_layout.addSpacing(8)

        nav_items = (
            ("Trading Desk", "mdi.chart-timeline-variant"),
            ("Focus Picks", "mdi.star-outline"),
            ("Journal", "mdi.notebook-outline"),
            ("Universe", "mdi.earth"),
            ("Research", "mdi.flask-outline"),
            ("Auto Pilot", "mdi.robot-outline"),
            ("Settings", "mdi.cog-outline"),
        )
        for index, (label, icon_name) in enumerate(nav_items):
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            icon = _nav_icon(icon_name)
            if icon is not None:
                button.setIcon(icon)
                button.setIconSize(QSize(18, 18))
            button.clicked.connect(lambda _checked=False, page=index: self._select_page(page))
            self.nav_buttons.append(button)
            nav_layout.addWidget(button)
        nav_layout.addStretch(1)
        self.nav_buttons[0].setChecked(True)

        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 10, 12, 10)
        top_layout.addWidget(self.title_label)
        top_layout.addStretch(1)
        top_layout.addWidget(self.workspace_button)
        top_layout.addWidget(self.tabs_button)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(top_bar)
        right_layout.addWidget(self.pages, 1)

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(8, 8, 8, 0)
        central_layout.setSpacing(8)
        central_layout.addWidget(nav)
        central_layout.addWidget(right, 1)
        self.setCentralWidget(central)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        # Persistent Auto Mode control (plan.md sec 15.2): visible and
        # clickable from every page - OFF -> AUTO-DESK -> AUTO-AWAY -> OFF.
        self.auto_mode_button = QPushButton()
        self.auto_mode_button.setObjectName("AutoModeButton")
        self.auto_mode_button.setToolTip(
            "Click to cycle Auto Mode: OFF -> AUTO-DESK -> AUTO-AWAY -> OFF. "
            "Desk/Away change presentation only - never trading decisions."
        )
        self.auto_mode_button.clicked.connect(self._cycle_auto_mode)
        self.autopilot_panel.service.enabledChanged.connect(lambda *_: self._sync_auto_mode_button())
        self._sync_auto_mode_button()
        status.addWidget(self.auto_mode_button)
        self.ib_status = QLabel("IB/TWS: unknown")
        self.scan_status = QLabel("Scan: idle")
        self.setup_status = QLabel("Setups: 0")
        self.watchlist_status = QLabel(_watchlist_status_text())
        self.universe_status = QLabel(_universe_status_text())
        self.data_status = QLabel(_data_status_text())
        status.addWidget(self.ib_status)
        status.addWidget(self.scan_status, 1)
        status.addPermanentWidget(self.setup_status)
        status.addPermanentWidget(self.watchlist_status)
        status.addPermanentWidget(self.universe_status)
        status.addPermanentWidget(self.data_status)

    def _cycle_auto_mode(self) -> None:
        service = self.autopilot_panel.service
        mode = service.auto_mode
        if mode == "OFF":
            service.set_profile("DESK")
            service.set_enabled(True)
        elif mode == "DESK":
            service.set_profile("AWAY")
        else:
            service.set_enabled(False)
        self._sync_auto_mode_button()

    def _sync_auto_mode_button(self) -> None:
        mode = self.autopilot_panel.service.auto_mode
        text = "Auto: OFF" if mode == "OFF" else f"Auto: {mode}"
        self.auto_mode_button.setText(text)
        color = {"OFF": "#8b8fa3", "DESK": "#3fb950", "AWAY": "#d29922"}.get(mode, "#8b8fa3")
        self.auto_mode_button.setStyleSheet(
            f"QPushButton#AutoModeButton {{ color: {color}; font-weight: 600; padding: 1px 10px; }}"
        )

    def _bind_shortcuts(self) -> None:
        run_action = QAction("Run Shared Scan", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(self.trading_panel.master_panel.run_shared_scan)
        self.addAction(run_action)

        focus_action = QAction("Focus Setup Filter", self)
        focus_action.setShortcut(QKeySequence("Ctrl+F"))
        focus_action.triggered.connect(lambda _checked=False: self.trading_panel.master_panel.search_input.setFocus())
        self.addAction(focus_action)

    def _select_page(self, index: int) -> None:
        titles = ("Trading Desk", "Focus Picks", "Journal", "Universe", "Research", "Auto Pilot", "Settings")
        self.pages.setCurrentIndex(index)
        self.title_label.setText(titles[index])
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        mode_visible = index == 0
        self.workspace_button.setVisible(mode_visible)
        self.tabs_button.setVisible(mode_visible)

    def _set_workspace_mode(self, mode: str) -> None:
        self.state.workspace_mode = mode
        self.state.save()
        self.trading_panel.set_mode(mode)
        self.settings_panel.mode_input.blockSignals(True)
        self.settings_panel.mode_input.setCurrentText(mode)
        self.settings_panel.mode_input.blockSignals(False)
        self._sync_mode_buttons()

    def _sync_mode_buttons(self) -> None:
        self.workspace_button.setChecked(self.state.workspace_mode == "workspace")
        self.tabs_button.setChecked(self.state.workspace_mode == "tabs")

    def _apply_state_changes(self) -> None:
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, self.state.theme_name, self.state.compact_density)
        self.trading_panel.set_mode(self.state.workspace_mode)
        self._sync_mode_buttons()

    def _set_scan_status(self, message: str) -> None:
        self.scan_status.setText(f"Scan: {message}")

    def _set_ib_status(self, message: str) -> None:
        self.ib_status.setText(message if message.lower().startswith("ib") else f"IB/TWS: {message}")

    def _set_setup_counts(self, total: int, favorites: int, near: int) -> None:
        self.setup_status.setText(f"Setups: {total} | Favorites: {favorites} | Near: {near}")

    def _self_heal_universe(self) -> None:
        import autopilot_core as core

        poll = getattr(self, "_universe_poll", None)
        if poll is not None and poll.isActive():
            return  # a heal attempt is already being tracked
        if not core.universe_is_stale(datetime.now()):
            self.universe_status.setText(_universe_status_text())
            return
        self.universe_status.setText("Universe: stale - rebuilding...")
        self.universe_status.setStyleSheet("color: #E5C07B;")
        threading.Thread(
            target=core.rebuild_universe_if_stale,
            kwargs={"force": False},
            name="universe-self-heal",
            daemon=True,
        ).start()
        self._universe_poll_ticks = 0
        self._universe_poll = QTimer(self)
        self._universe_poll.setInterval(10_000)
        self._universe_poll.timeout.connect(self._poll_universe_heal)
        self._universe_poll.start()

    def _poll_universe_heal(self) -> None:
        import autopilot_core as core

        self._universe_poll_ticks += 1
        done = not core.universe_is_stale(datetime.now())
        if done or self._universe_poll_ticks > 120:  # give up after ~20 min
            self.universe_status.setText(_universe_status_text())
            self.universe_status.setStyleSheet("" if done else "color: #E06C75;")
            self._universe_poll.stop()

    def closeEvent(self, event) -> None:
        for panel in (self.trading_panel, self.journal_panel, self.universe_panel, self.research_panel, self.autopilot_panel, self.settings_panel):
            try:
                panel.shutdown()
            except Exception:
                pass
        super().closeEvent(event)


def _mode_button(label: str) -> QPushButton:
    button = QPushButton(label)
    button.setCheckable(True)
    return button


def _nav_icon(name: str) -> QIcon | None:
    """Material Design icon for the nav rail; degrade gracefully if qtawesome
    (or its font cache) is unavailable so the shell still launches."""
    try:
        import qtawesome as qta

        return qta.icon(name, color="#8A93A1")
    except Exception:
        return None


def _watchlist_status_text() -> str:
    details = get_shared_watchlist_details()
    longs = "yes" if details.get("longs_exists") == "yes" else "missing"
    shorts = "yes" if details.get("shorts_exists") == "yes" else "missing"
    return f"Watchlists: longs {longs} | shorts {shorts}"


def _data_status_text() -> str:
    details = get_tracker_storage_details()
    return f"Data: {details.get('source_label', details.get('source', 'unknown'))}"


def _universe_status_text() -> str:
    import autopilot_core as core

    built_at = core.universe_built_at()
    if built_at is None:
        return "Universe: missing"
    state = "stale" if core.universe_is_stale(datetime.now(), built_at) else "fresh"
    return f"Universe: {state} ({built_at:%b %d %H:%M})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the PySide6 TradingBotV3 UI.")
    parser.add_argument(
        "--mode",
        choices=("workspace", "tabs", "full", "simple"),
        default=None,
        help="Trading Desk layout mode. full maps to workspace; simple maps to tabs.",
    )
    parser.add_argument(
        "--theme",
        choices=("dark", "light"),
        default=None,
        help="GUI color theme. Saved as the default for future launches.",
    )
    args = parser.parse_args(argv)

    state = UiState.load()
    if args.mode:
        state.workspace_mode = {"full": "workspace", "simple": "tabs"}.get(args.mode, args.mode)
        state.save()
    if args.theme:
        state.theme_name = args.theme
        state.save()

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("TradingBotV3")
    app.setOrganizationName("TradingBotV3")
    apply_theme(app, state.theme_name, state.compact_density)

    window = MainWindow(state)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
