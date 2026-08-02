"""Desk Link satellite window: a view-only mirror of the main desk.

Tier 1 (docs/MULTI_MACHINE_DESK_PROPOSAL.md): connects to the main over
the LAN, shows connection state and a rolling feed of relayed alerts, and
opens the alert chart popup — the same SymbolSnapshotWidget the main desk
uses, fed from the relayed payload instead of local stores. No TWS, no
scanners, no shared-state writes happen here.

DeskLinkClient callbacks arrive on its connection thread; _ClientBridge
turns them into Qt signals so all widget work stays on the GUI thread.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from desk_link import protocol
from desk_link.client import DeskLinkClient
from desk_link.popup_payload import restore_alert_popup_payload
from desk_link.server import DEFAULT_PORT
from project_paths import get_local_setting, save_local_setting
from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

log = logging.getLogger(__name__)

_MAX_FEED_ROWS = 200
_MAX_OPEN_POPUPS = 6

HOST_SETTING = "desk_link_host"
TOKEN_SETTING = "desk_link_token"


def load_saved_connection() -> tuple[str, int, str]:
    """(host, port, token) from local settings; host/token may be empty."""
    saved = str(get_local_setting(HOST_SETTING, "") or "").strip()
    host, _, port_text = saved.partition(":")
    try:
        port = int(port_text) if port_text.strip() else DEFAULT_PORT
    except ValueError:
        port = DEFAULT_PORT
    token = str(get_local_setting(TOKEN_SETTING, "") or "").strip()
    return host.strip(), port, token


def save_connection(host: str, port: int, token: str) -> None:
    save_local_setting(HOST_SETTING, f"{host.strip()}:{int(port)}")
    save_local_setting(TOKEN_SETTING, token.strip())


class ConnectDialog(QDialog):
    """Main-desk address + token entry, prefilled from the saved values.

    This is the whole satellite pairing UX: the trader reads the token off
    the main's Settings page (Copy token) and types/pastes the main PC's
    name or IP here once. Everything is remembered for next launch.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect to main desk")
        host, port, token = load_saved_connection()

        self.host_input = QLineEdit(host)
        self.host_input.setPlaceholderText("Main PC name or IP, e.g. 192.168.1.20")
        self.port_input = QSpinBox()
        self.port_input.setRange(1024, 65535)
        self.port_input.setValue(port)
        self.token_input = QLineEdit(token)
        self.token_input.setPlaceholderText("Link token — Settings → Desk Link → Copy token on the main PC")

        form = QFormLayout()
        form.setSpacing(8)
        form.addRow("Main desk", self.host_input)
        form.addRow("Port", self.port_input)
        form.addRow("Link token", self.token_input)

        hint = QLabel(
            "Find these on the main PC: Settings page → Desk Link section. "
            "The port there must match this one (default 47600)."
        )
        hint.setObjectName("MutedLabel")
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Connect")
        buttons.accepted.connect(self._accept_if_complete)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)
        self.resize(460, self.sizeHint().height())

    def _accept_if_complete(self) -> None:
        if not self.host_input.text().strip():
            self.host_input.setFocus()
            return
        if not self.token_input.text().strip():
            self.token_input.setFocus()
            return
        self.accept()

    def connection(self) -> tuple[str, int, str]:
        return (
            self.host_input.text().strip(),
            int(self.port_input.value()),
            self.token_input.text().strip(),
        )


class _ClientBridge(QObject):
    messageReceived = Signal(object)
    statusChanged = Signal(str, str)


class SatellitePopupDialog(QDialog):
    """One relayed alert chart. Non-activating, like the main's popup."""

    def __init__(self, restored: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        alert = restored["alert"]
        symbol = str(alert.get("symbol") or "").upper()
        side = str(alert.get("side") or "")
        trigger = str(alert.get("trigger") or "")
        self.setWindowTitle(f"{symbol} · {side} · {trigger}".strip(" ·"))
        # Same gentle-raise contract as the main desk's snapshot dialog: the
        # popup must never steal the keyboard from whatever the trader is on.
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(1180, 760)

        header_bits = [bit for bit in (alert.get("time_text"), side, trigger, alert.get("context")) if bit]
        header = QLabel(" · ".join(str(bit) for bit in header_bits))
        header.setWordWrap(True)

        guidance = QLabel(str(restored.get("guidance_text") or ""))
        guidance.setObjectName("MutedLabel")
        guidance.setWordWrap(True)
        guidance.setVisible(bool(restored.get("guidance_text")))

        armed = restored.get("armed") or {}
        armed_bits = list(armed.get("kinds") or []) + [
            f"{level.get('direction', '')} {level.get('level', level.get('price', ''))}".strip()
            for level in (armed.get("levels") or [])
        ] + [event.get("kind", "") for event in (armed.get("d1_events") or [])]
        armed_label = QLabel("Armed on main: " + ", ".join(bit for bit in armed_bits if bit))
        armed_label.setObjectName("MutedLabel")
        armed_label.setWordWrap(True)
        armed_label.setVisible(bool(any(armed_bits)))

        snapshot = SymbolSnapshotWidget(self)
        snapshot.show_payload_snapshots(symbol, restored["d1"], restored["m5"])

        layout = QVBoxLayout(self)
        layout.addWidget(header)
        layout.addWidget(guidance)
        layout.addWidget(armed_label)
        layout.addWidget(snapshot, 1)


class SatelliteWindow(QMainWindow):
    def __init__(self, *, machine_name: str, host: str = "", port: int = DEFAULT_PORT, token: str = "") -> None:
        super().__init__()
        self._machine_name = machine_name
        self.setWindowTitle("TradingBotV3 Satellite")
        self.resize(560, 640)

        self.status_label = QLabel("Not connected.")
        self.status_label.setWordWrap(True)
        self.connect_button = QPushButton("Connect / change main desk…")
        self.connect_button.clicked.connect(self.open_connect_dialog)
        self.snapshot_label = QLabel("")
        self.snapshot_label.setObjectName("MutedLabel")
        self.snapshot_label.setWordWrap(True)
        self.feed = QListWidget()
        self.feed.setWordWrap(True)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(self.status_label, 1)
        top_row.addWidget(self.connect_button)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(top_row)
        layout.addWidget(self.snapshot_label)
        layout.addWidget(self.feed, 1)
        self.setCentralWidget(central)

        self._popups: list[SatellitePopupDialog] = []
        self._client: DeskLinkClient | None = None
        self._bridge = _ClientBridge(self)
        self._bridge.messageReceived.connect(self._on_message)
        self._bridge.statusChanged.connect(self._on_status)

        cli_host = host
        saved_host, saved_port, saved_token = load_saved_connection()
        host = host or saved_host
        token = token or saved_token
        if not cli_host:
            port = saved_port  # a CLI host carries its own port; otherwise the saved one wins
        self._pairing_prompted = False
        if host and token:
            # CLI-provided details are remembered too, so a first launch via
            # flags makes every later bare launch just work.
            save_connection(host, port, token)
            self._start_client(host, port, token)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        # First run with nothing saved: go straight to the pairing dialog once
        # the window is actually on screen, instead of a dead "not connected"
        # screen. Deliberately tied to showEvent (not __init__): a modal
        # queued from the constructor would fire in ANY later event loop,
        # including headless test runs that never show the window.
        if self._client is None and not self._pairing_prompted:
            self._pairing_prompted = True
            QTimer.singleShot(0, self.open_connect_dialog)

    def open_connect_dialog(self) -> None:
        dialog = ConnectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        host, port, token = dialog.connection()
        save_connection(host, port, token)
        self._start_client(host, port, token)

    def _start_client(self, host: str, port: int, token: str) -> None:
        old_client, self._client = self._client, None
        if old_client is not None:
            old_client.stop()
        self.setWindowTitle(f"TradingBotV3 Satellite — {host}")
        self.status_label.setText(f"Connecting to {host}:{port}…")
        self._client = DeskLinkClient(
            host=host,
            port=port,
            token=token,
            machine_name=self._machine_name,
            on_message=self._bridge.messageReceived.emit,
            on_status=self._bridge.statusChanged.emit,
        )
        self._client.start()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._client is not None:
            self._client.stop()
        super().closeEvent(event)

    # -- GUI-thread handlers -------------------------------------------------

    def _on_status(self, state: str, detail: str) -> None:
        labels = {
            "connecting": f"Connecting… ({detail})",
            "connected": f"Connected to {detail} — view-only satellite. Alerts pop here live.",
            "disconnected": f"Link lost — {detail}.",
            "rejected": (
                f"Link rejected: {detail}. Click \"Connect / change main desk…\" and paste the "
                "current token from the main PC's Settings page."
            ),
            "stopped": "Stopped.",
        }
        self.status_label.setText(labels.get(state, f"{state}: {detail}"))

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == protocol.TYPE_ALERT_POPUP:
            self._show_alert(message)
        elif kind == protocol.TYPE_STATE_SNAPSHOT:
            self._show_snapshot(message["payload"])

    def _show_snapshot(self, payload: dict[str, Any]) -> None:
        watchlists = payload.get("watchlists") or {}
        focus = payload.get("focus") or {}
        bits = [f"{name} {len(symbols)}" for name, symbols in watchlists.items()]
        focus_names = sorted(set((focus.get("longs") or []) + (focus.get("shorts") or [])))
        text = "Desk mirror — watchlists: " + ", ".join(bits) if bits else ""
        if focus_names:
            text += " · Focus: " + ", ".join(focus_names)
        self.snapshot_label.setText(text)

    def _show_alert(self, message: dict[str, Any]) -> None:
        try:
            restored = restore_alert_popup_payload(message["payload"])
        except ValueError as exc:
            self.feed.insertItem(0, f"⚠ Incompatible alert payload: {exc} (update this machine)")
            return
        alert = restored["alert"]
        row = " · ".join(
            str(bit)
            for bit in (alert.get("time_text"), alert.get("symbol"), alert.get("side"), alert.get("trigger"))
            if bit
        )
        self.feed.insertItem(0, row)
        while self.feed.count() > _MAX_FEED_ROWS:
            self.feed.takeItem(self.feed.count() - 1)
        QApplication.beep()

        self._popups = [popup for popup in self._popups if popup.isVisible()]
        while len(self._popups) >= _MAX_OPEN_POPUPS:
            self._popups.pop(0).close()
        popup = SatellitePopupDialog(restored, self)
        self._popups.append(popup)
        popup.show()
        popup.raise_()
