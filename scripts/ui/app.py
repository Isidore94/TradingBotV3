#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from PySide6.QtCore import QSize, Qt
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
from ui.panels.journal_panel import JournalPanel
from ui.panels.research_panel import ResearchPanel
from ui.panels.settings_panel import SettingsPanel
from ui.panels.trading_desk import TradingDeskPanel
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
        self.research_panel = ResearchPanel()
        self.settings_panel = SettingsPanel(self.state)
        self.settings_panel.stateChanged.connect(self._apply_state_changes)

        self.pages = QStackedWidget()
        self.pages.addWidget(self.trading_panel)
        self.pages.addWidget(self.journal_panel)
        self.pages.addWidget(self.research_panel)
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
            ("Journal", "mdi.notebook-outline"),
            ("Research", "mdi.flask-outline"),
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
        self.ib_status = QLabel("IB/TWS: unknown")
        self.scan_status = QLabel("Scan: idle")
        self.setup_status = QLabel("Setups: 0")
        self.watchlist_status = QLabel(_watchlist_status_text())
        self.data_status = QLabel(_data_status_text())
        status.addWidget(self.ib_status)
        status.addWidget(self.scan_status, 1)
        status.addPermanentWidget(self.setup_status)
        status.addPermanentWidget(self.watchlist_status)
        status.addPermanentWidget(self.data_status)

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
        titles = ("Trading Desk", "Journal", "Research", "Settings")
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

    def closeEvent(self, event) -> None:
        for panel in (self.trading_panel, self.research_panel, self.settings_panel):
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
