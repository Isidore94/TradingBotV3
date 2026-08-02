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

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from desk_link import protocol
from desk_link.client import DeskLinkClient
from desk_link.popup_payload import restore_alert_popup_payload
from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

log = logging.getLogger(__name__)

_MAX_FEED_ROWS = 200
_MAX_OPEN_POPUPS = 6


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
    def __init__(self, *, host: str, port: int, token: str, machine_name: str) -> None:
        super().__init__()
        self.setWindowTitle(f"TradingBotV3 Satellite — {host}")
        self.resize(560, 640)

        self.status_label = QLabel("Connecting…")
        self.status_label.setWordWrap(True)
        self.snapshot_label = QLabel("")
        self.snapshot_label.setObjectName("MutedLabel")
        self.snapshot_label.setWordWrap(True)
        self.feed = QListWidget()
        self.feed.setWordWrap(True)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.status_label)
        layout.addWidget(self.snapshot_label)
        layout.addWidget(self.feed, 1)
        self.setCentralWidget(central)

        self._popups: list[SatellitePopupDialog] = []
        self._bridge = _ClientBridge(self)
        self._bridge.messageReceived.connect(self._on_message)
        self._bridge.statusChanged.connect(self._on_status)
        self._client = DeskLinkClient(
            host=host,
            port=port,
            token=token,
            machine_name=machine_name,
            on_message=self._bridge.messageReceived.emit,
            on_status=self._bridge.statusChanged.emit,
        )
        self._client.start()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._client.stop()
        super().closeEvent(event)

    # -- GUI-thread handlers -------------------------------------------------

    def _on_status(self, state: str, detail: str) -> None:
        labels = {
            "connecting": f"Connecting… ({detail})",
            "connected": f"Connected to {detail} — view-only satellite. Alerts pop here live.",
            "disconnected": f"Link lost — {detail}.",
            "rejected": f"Link REJECTED: {detail}. Fix desk_link_token in local settings and relaunch.",
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
