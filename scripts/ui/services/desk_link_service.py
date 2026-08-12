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
import threading
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

import push_notify

from desk_link import protocol
from desk_link.protocol import generate_link_token
from desk_link.server import DEFAULT_PORT, DeskLinkServer
from project_paths import get_local_setting, save_local_setting

log = logging.getLogger(__name__)

_SNAPSHOT_INTERVAL_MS = 60_000
# Live M5 stream cadence matches the desk's own 30s chart-refresh timers;
# the symbol cap bounds a worst-case burst (40 symbols x ~5KB is trivial on
# a LAN and invisible next to the scans themselves).
_LIVE_CHART_INTERVAL_MS = 30_000
_LIVE_CHART_MAX_SYMBOLS = 40

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
    # Control lease (Tier 2): the machine name holding control, "" = main.
    controlChanged = Signal(str)
    # An intent from the CURRENT controller, to be applied by the desk and
    # acked via send_intent_result. (machine, intent payload)
    intentReceived = Signal(str, dict)

    # Server-thread -> GUI-thread bridges (Qt queues cross-thread signals).
    _clientMessage = Signal(str, dict)
    _clientGone = Signal(str)

    def __init__(self, parent: QObject | None = None, *, machine_name: str = "main-desk") -> None:
        super().__init__(parent)
        self._server: DeskLinkServer | None = None
        self._machine_name = machine_name
        self._controller = ""
        self._snapshot_timer = QTimer(self)
        self._snapshot_timer.setInterval(_SNAPSHOT_INTERVAL_MS)
        self._snapshot_timer.timeout.connect(self.publish_state_snapshot)
        self._bot_provider = None
        self._chart_symbols_provider = None
        self._auto_mode_provider = None
        self._live_chart_timer = QTimer(self)
        self._live_chart_timer.setInterval(_LIVE_CHART_INTERVAL_MS)
        self._live_chart_timer.timeout.connect(self._publish_live_charts)
        self._clientMessage.connect(self._handle_client_message)
        self._clientGone.connect(self._handle_client_gone)

    # -- lifecycle -----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._server is not None

    @property
    def has_satellites(self) -> bool:
        return bool(self.connected_machines())

    def start(self) -> bool:
        if self._server is not None:
            return True
        port = int(get_local_setting(PORT_SETTING, DEFAULT_PORT) or DEFAULT_PORT)
        server = DeskLinkServer(
            token=ensure_link_token(),
            machine_name=self._machine_name,
            port=port,
            on_client_connected=self._on_client_change,
            on_client_disconnected=self._on_client_disconnected,
            on_client_message=lambda machine, message: self._clientMessage.emit(machine, message),
        )
        try:
            server.start()
        except OSError:
            log.exception("Desk Link server failed to start on port %s.", port)
            return False
        self._server = server
        self.publish_state_snapshot()
        self._snapshot_timer.start()
        self._live_chart_timer.start()
        log.info("Desk Link serving satellites on port %s.", port)
        self.runningChanged.emit(True)
        return True

    def stop(self) -> None:
        self._snapshot_timer.stop()
        self._live_chart_timer.stop()
        server, self._server = self._server, None
        if server is not None:
            server.stop()
            self.satellitesChanged.emit([])
            self.runningChanged.emit(False)
        if self._controller:
            self._controller = ""
            self.controlChanged.emit("")

    # -- control lease (Tier 2) ----------------------------------------------

    @property
    def controller(self) -> str:
        """Machine name currently holding control; "" when the main has it."""
        return self._controller

    def take_back_control(self) -> None:
        """The main's override: immediate, always available (trader decision)."""
        controller, self._controller = self._controller, ""
        if not controller:
            return
        server = self._server
        if server is not None:
            server.send_to_machine(
                controller, protocol.TYPE_LEASE_REVOKED, {"reason": "main took back control"}
            )
        log.info("Desk Link control taken back from %s.", controller)
        self.controlChanged.emit("")

    def send_intent_result(self, machine: str, seq, ok: bool, detail: str = "") -> None:
        server = self._server
        if server is not None:
            server.send_to_machine(
                machine, protocol.TYPE_INTENT_RESULT, {"seq": seq, "ok": bool(ok), "detail": detail}
            )

    def _handle_client_message(self, machine: str, message: dict) -> None:
        kind = message.get("type")
        payload = message.get("payload") or {}
        if kind == protocol.TYPE_LEASE_REQUEST:
            if self._controller and self._controller != machine:
                self._send_to(machine, protocol.TYPE_LEASE_DENIED, {"holder": self._controller})
                return
            self._controller = machine
            self._send_to(machine, protocol.TYPE_LEASE_GRANT, {})
            log.info("Desk Link control granted to %s.", machine)
            self.controlChanged.emit(machine)
        elif kind == protocol.TYPE_LEASE_RELEASE:
            if self._controller == machine:
                self._controller = ""
                log.info("Desk Link control released by %s.", machine)
                self.controlChanged.emit("")
        elif kind == protocol.TYPE_INTENT:
            if self._controller == machine:
                self.intentReceived.emit(machine, dict(payload))
                # Direct-connection slots have applied the intent by the time
                # emit returns; republish desk state right away so the
                # satellite's mirror reflects its own action without waiting
                # out the 60s snapshot timer.
                self.publish_state_snapshot()
            else:
                self.send_intent_result(machine, payload.get("seq"), False, "not in control")

    def _handle_client_gone(self, machine: str) -> None:
        """Lease auto-reclaim: control dies with the connection.

        The server's idle timeout is the grace window - a satellite that
        sleeps or drops off Wi-Fi is reaped there, and the desk resumes
        primary on its own instead of sitting headless.
        """
        server = self._server
        still_connected = server is not None and machine in server.connected_machines()
        if self._controller == machine and not still_connected:
            self._controller = ""
            log.warning("Desk Link controller %s disconnected - main resumed control.", machine)
            self.controlChanged.emit("")
            self._push_reclaim_notice(machine)

    def _push_reclaim_notice(self, machine: str) -> None:
        """Phone push when control auto-reclaims (design promise).

        The trader may be away from both screens when the satellite dies;
        without this, the only sign their actions stopped applying is a
        banner that quietly disappeared on the main. Runs on a one-shot
        daemon thread - a slow ntfy round-trip must not stall the GUI - and
        never raises.
        """

        def _send() -> None:
            try:
                # AWAY is the only mode allowed to push (trader rule
                # 2026-08-11). A control hand-back while the trader is at the
                # desk is already on screen.
                auto_mode = ""
                if self._auto_mode_provider is not None:
                    auto_mode = str(self._auto_mode_provider() or "")
                if auto_mode.upper() != "AWAY":
                    return
                if push_notify.push_configured():
                    push_notify.send_push(
                        "Desk Link: main resumed control",
                        f"Satellite '{machine}' dropped off - the main desk took back control.",
                        tags="desk_link",
                    )
            except Exception:
                log.exception("Desk Link reclaim push failed.")

        threading.Thread(target=_send, name="desk-link-reclaim-push", daemon=True).start()

    def _send_to(self, machine: str, message_type: str, payload: dict) -> None:
        server = self._server
        if server is not None:
            server.send_to_machine(machine, message_type, payload)

    def _on_client_disconnected(self, machine: str, address) -> None:
        self._on_client_change(machine, address)
        self._clientGone.emit(machine)

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

    def send_test_popup(self, symbol: str = "SPY") -> bool:
        """Fire a synthetic popup so the link can be tested without a live
        market: real chart payload from the local daily store, clearly
        labeled as a test, sent through the exact relay path alerts use.
        Returns False when nothing is listening."""
        if not self.has_satellites:
            return False
        try:
            from datetime import datetime

            from desk_link.popup_payload import capture_alert_popup

            payload = capture_alert_popup(
                {
                    "time_text": datetime.now().strftime("%H:%M:%S"),
                    "symbol": str(symbol or "SPY").strip().upper(),
                    "side": "TEST",
                    "trigger": "Desk Link test popup",
                    "timeframe": "D1",
                    "context": "connection check - not a signal",
                    "tag": "desk_link_test",
                },
                bot=None,
                guidance_text="Test popup: confirms relay, chart payload, and rendering end to end.",
            )
            self.publish_alert_popup(payload)
            return True
        except Exception:
            log.exception("Desk Link test popup failed.")
            return False

    def publish_stream(self, stream: str, data: Any) -> None:
        """Relay one live desk surface (Tier 3 full relay).

        Everything rides the generic desk_stream envelope, so wiring a new
        surface is one signal connection here and one on the satellite.
        No-op without connected satellites - zero cost on a lone desk.
        """
        server = self._server
        if server is None or not server.connected_machines():
            return
        try:
            server.send_desk_stream({"stream": str(stream), "data": data})
        except Exception:
            log.exception("Desk Link stream %r publish failed.", stream)

    def set_auto_mode_source(self, provider) -> None:
        """Getter for the main's Auto mode (OFF/DESK/AWAY/EVENING), included in
        every state snapshot so satellites can mirror and change it."""
        self._auto_mode_provider = provider

    def set_live_chart_source(self, bot_provider, symbols_provider) -> None:
        """Feed the 30s M5 stream: a live-bot getter and a symbols getter
        (the Alert Center's current review/feed names)."""
        self._bot_provider = bot_provider
        self._chart_symbols_provider = symbols_provider

    def _publish_live_charts(self) -> None:
        server = self._server
        if server is None or not server.connected_machines():
            return
        if self._bot_provider is None or self._chart_symbols_provider is None:
            return
        try:
            bot = self._bot_provider()
            symbols = list(self._chart_symbols_provider() or [])[:_LIVE_CHART_MAX_SYMBOLS]
        except Exception:
            log.exception("Desk Link live-chart source failed.")
            return
        if bot is None or not symbols:
            return
        from desk_link.popup_payload import bars_to_wire

        for symbol in symbols:
            try:
                bars = bot.m5_chart_bars(symbol, max_sessions=2)
            except Exception:
                continue
            if bars:
                self.publish_stream("m5_bars", {"symbol": symbol, "bars": bars_to_wire(bars)})

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

        auto_mode = ""
        if self._auto_mode_provider is not None:
            try:
                auto_mode = str(self._auto_mode_provider() or "")
            except Exception:
                log.exception("Desk Link auto-mode source failed.")

        import price_alerts

        fired_price_alerts = []
        for trigger in price_alerts.todays_triggers():
            payload = dict(trigger)
            try:
                payload["message"] = price_alerts.format_trigger_message(payload)
            except (TypeError, ValueError):
                payload["message"] = f"{payload.get('symbol') or 'Price'} alert fired"
            payload["priority"] = "urgent"
            fired_price_alerts.append(payload)

        return {
            "machine": self._machine_name,
            "auto_mode": auto_mode,
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
            "price_alerts": fired_price_alerts,
        }
