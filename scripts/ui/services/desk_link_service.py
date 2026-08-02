"""Qt-side owner of the Desk Link relay server on the main desk.

Owns the DeskLinkServer (which owns its own threads) plus one QTimer for
the periodic state snapshot — no other component touches either. Enabled
by the machine-local setting ``desk_link_enabled`` (see
docs/MULTI_MACHINE_DESK_PROPOSAL.md); the link token is generated once and
stored in the same local settings, never in the shared store.

Server callbacks arrive on Desk Link threads; they are bridged to Qt
signals here (cross-thread signal emission is queued by Qt), so panels can
connect without thread care.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from desk_link.protocol import generate_link_token
from desk_link.server import DEFAULT_PORT, DeskLinkServer
from project_paths import get_local_setting, save_local_setting

log = logging.getLogger(__name__)

_SNAPSHOT_INTERVAL_MS = 60_000

ENABLED_SETTING = "desk_link_enabled"
PORT_SETTING = "desk_link_port"
TOKEN_SETTING = "desk_link_token"


def desk_link_enabled() -> bool:
    return bool(get_local_setting(ENABLED_SETTING, False))


def ensure_link_token() -> str:
    token = str(get_local_setting(TOKEN_SETTING, "") or "").strip()
    if not token:
        token = generate_link_token()
        save_local_setting(TOKEN_SETTING, token)
    return token


class DeskLinkService(QObject):
    """Starts/stops the relay server and publishes desk state to satellites."""

    satellitesChanged = Signal(list)  # sorted machine names, may be empty
    runningChanged = Signal(bool)

    def __init__(self, parent: QObject | None = None, *, machine_name: str = "main-desk") -> None:
        super().__init__(parent)
        self._server: DeskLinkServer | None = None
        self._machine_name = machine_name
        self._snapshot_timer = QTimer(self)
        self._snapshot_timer.setInterval(_SNAPSHOT_INTERVAL_MS)
        self._snapshot_timer.timeout.connect(self.publish_state_snapshot)

    # -- lifecycle -----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._server is not None

    @property
    def has_satellites(self) -> bool:
        return self._server is not None and self._server.client_count > 0

    def start(self) -> bool:
        if self._server is not None:
            return True
        port = int(get_local_setting(PORT_SETTING, DEFAULT_PORT) or DEFAULT_PORT)
        server = DeskLinkServer(
            token=ensure_link_token(),
            machine_name=self._machine_name,
            port=port,
            on_client_connected=self._on_client_change,
            on_client_disconnected=self._on_client_change,
        )
        try:
            server.start()
        except OSError:
            log.exception("Desk Link server failed to start on port %s.", port)
            return False
        self._server = server
        self.publish_state_snapshot()
        self._snapshot_timer.start()
        log.info("Desk Link serving satellites on port %s.", port)
        self.runningChanged.emit(True)
        return True

    def stop(self) -> None:
        self._snapshot_timer.stop()
        server, self._server = self._server, None
        if server is not None:
            server.stop()
            self.satellitesChanged.emit([])
            self.runningChanged.emit(False)

    # -- settings-page controls ----------------------------------------------

    def set_enabled(self, enabled: bool) -> bool:
        """Persist the toggle and apply it now. Returns False if serving failed."""
        save_local_setting(ENABLED_SETTING, bool(enabled))
        if enabled:
            return self.start()
        self.stop()
        return True

    def configured_port(self) -> int:
        return int(get_local_setting(PORT_SETTING, DEFAULT_PORT) or DEFAULT_PORT)

    def set_port(self, port: int) -> bool:
        save_local_setting(PORT_SETTING, int(port))
        if self.running:
            self.stop()
            return self.start()
        return True

    def current_token(self) -> str:
        return str(get_local_setting(TOKEN_SETTING, "") or "").strip()

    def ensure_token(self) -> str:
        return ensure_link_token()

    def regenerate_token(self) -> str:
        """Mint a new link token; connected satellites must re-pair.

        Restarting the server drops existing connections, so a leaked or
        mistyped token is fully invalidated the moment this returns.
        """
        token = generate_link_token()
        save_local_setting(TOKEN_SETTING, token)
        if self.running:
            self.stop()
            self.start()
        return token

    def connected_machines(self) -> list[str]:
        server = self._server
        return server.connected_machines() if server is not None else []

    def _on_client_change(self, machine: str, address) -> None:
        server = self._server
        if server is not None:
            # Emitting from the Desk Link thread is safe: cross-thread signal
            # delivery is queued onto the GUI thread by Qt.
            self.satellitesChanged.emit(server.connected_machines())

    # -- publishing ----------------------------------------------------------

    def publish_alert_popup(self, payload: dict[str, Any]) -> None:
        server = self._server
        if server is not None:
            server.send_alert_popup(payload)

    def publish_state_snapshot(self) -> None:
        server = self._server
        if server is None:
            return
        try:
            server.set_state_snapshot(self._build_state_snapshot())
        except Exception:
            log.exception("Desk Link state snapshot publish failed.")

    def _build_state_snapshot(self) -> dict[str, Any]:
        """Light desk mirror: the shared watchlists and Focus picks.

        These files also sync through Drive; publishing them over the link
        just makes a freshly connected satellite current immediately instead
        of waiting on cloud sync.
        """
        from project_paths import (
            FOCUS_LONGS_FILE,
            FOCUS_SHORTS_FILE,
            LONGS_FILE,
            SHORTS_FILE,
            SWING_LONGS_FILE,
            SWING_SHORTS_FILE,
        )

        def read_list(path) -> list[str]:
            try:
                if not path.exists():
                    return []
                return [
                    line.strip().upper()
                    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
            except OSError:
                return []

        return {
            "machine": self._machine_name,
            "watchlists": {
                "longs": read_list(LONGS_FILE),
                "shorts": read_list(SHORTS_FILE),
                "swing_longs": read_list(SWING_LONGS_FILE),
                "swing_shorts": read_list(SWING_SHORTS_FILE),
            },
            "focus": {
                "longs": read_list(FOCUS_LONGS_FILE),
                "shorts": read_list(FOCUS_SHORTS_FILE),
            },
        }
