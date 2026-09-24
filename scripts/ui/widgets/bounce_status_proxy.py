"""The BounceBot strip's controls, as status-bar proxies for the compact desk.

The compact layout hides the BounceBot strip (its service keeps running). The
controls the trader still needs sit in the status bar instead. Every proxy
clicks or sets the REAL control on the hidden strip, so each signal path is the
one the strip already uses; nothing here talks to the service directly.
IB, regime and technicals are already in the status bar and are not repeated.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QToolButton, QWidget

from ui import theme


class _EnabledMirror(QObject):
    """Keep a proxy's enabled state equal to its source button's."""

    def __init__(self, source, proxy, parent=None) -> None:
        super().__init__(parent)
        self._source = source
        self._proxy = proxy
        source.installEventFilter(self)
        proxy.setEnabled(source.isEnabled())

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        if watched is self._source and event.type() == QEvent.Type.EnabledChange:
            self._proxy.setEnabled(self._source.isEnabled())
        return False


class BounceStatusProxy(QWidget):
    """Start/Stop scanning, active bounces, Entry assist and Mode menus."""

    def __init__(self, bounce_panel, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("BounceStatusProxy")
        self._panel = bounce_panel

        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("StatusProxyButton")
        self.start_button.setToolTip("Start BounceBot scanning.")
        self.start_button.clicked.connect(bounce_panel.start_scanning_button.click)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("StatusProxyButton")
        self.stop_button.setToolTip("Stop BounceBot scanning.")
        self.stop_button.clicked.connect(bounce_panel.stop_scanning_button.click)
        self._mirrors = [
            _EnabledMirror(bounce_panel.start_scanning_button, self.start_button, self),
            _EnabledMirror(bounce_panel.stop_scanning_button, self.stop_button, self),
        ]

        self.active_label = QLabel(bounce_panel.active_label.text())
        self.active_label.setObjectName("MutedLabel")
        bounce_panel.service.activeBouncesChanged.connect(
            lambda count: self.active_label.setText(f"active bounces: {count}")
        )

        self.entry_menu = QMenu(self)
        self.entry_menu.setToolTipsVisible(True)
        self.entry_actions: dict[str, QAction] = {}
        for command, button in bounce_panel.entry_assist_buttons.items():
            action = QAction(button.text(), self)
            action.triggered.connect(lambda _checked=False, b=button: b.click())
            self.entry_menu.addAction(action)
            self.entry_actions[command] = action
        self.entry_menu.addSeparator()
        self.manual_tools_action = QAction("Manual window tools", self)
        self.manual_tools_action.setCheckable(True)
        self.manual_tools_action.triggered.connect(
            lambda _checked=False: bounce_panel.entry_assist_advanced_button.click()
        )
        self.entry_menu.addAction(self.manual_tools_action)
        self.entry_menu.aboutToShow.connect(self.sync_entry_actions)
        self.entry_button = _menu_button("Entry assist ▾", self.entry_menu)
        self.entry_button.setToolTip(bounce_panel.entry_assist_auto_label.text())

        combo = bounce_panel.environment_input
        self.mode_menu = QMenu(self)
        self.mode_actions: list[QAction] = []
        for index in range(combo.count()):
            action = QAction(combo.itemText(index), self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked=False, i=index: combo.setCurrentIndex(i))
            self.mode_menu.addAction(action)
            self.mode_actions.append(action)
        self.mode_menu.aboutToShow.connect(self.sync_mode_actions)
        combo.currentIndexChanged.connect(lambda *_args: self.sync_mode_actions())
        self.mode_button = _menu_button("Mode ▾", self.mode_menu)
        self.mode_button.setToolTip(combo.toolTip())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.px(4))
        for widget in (
            self.start_button,
            self.stop_button,
            self.active_label,
            self.entry_button,
            self.mode_button,
        ):
            layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)
        self.sync_entry_actions()
        self.sync_mode_actions()

    def sync_entry_actions(self) -> None:
        """Copy each entry-assist button's label, tooltip, enabled and shown state."""
        panel = self._panel
        for command, action in self.entry_actions.items():
            button = panel.entry_assist_buttons[command]
            action.setText(button.text())
            action.setToolTip(button.toolTip())
            action.setEnabled(button.isEnabled())
            action.setVisible(not button.isHidden())
        self.manual_tools_action.setChecked(panel.entry_assist_advanced_button.isChecked())
        self.entry_button.setToolTip(panel.entry_assist_auto_label.text())

    def sync_mode_actions(self) -> None:
        current = self._panel.environment_input.currentIndex()
        for index, action in enumerate(self.mode_actions):
            action.setChecked(index == current)


def _menu_button(text: str, menu: QMenu) -> QToolButton:
    button = QToolButton()
    button.setObjectName("CompactMenuButton")
    button.setProperty("statusProxy", True)
    button.setText(text)
    button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    button.setMenu(menu)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return button
