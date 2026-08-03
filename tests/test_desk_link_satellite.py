"""Desk Link Tier 1 end-to-end: capture on the main → wire → satellite render.

Proves the critical path with the real components: the payload built from
real chart_snapshot output travels through the real server/client pair and
renders in the same SymbolSnapshotWidget the main desk uses — offscreen,
no TWS, no local stores on the receiving side.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import chart_snapshot
from desk_link import protocol
from desk_link.popup_payload import build_alert_popup_payload, restore_alert_popup_payload
from ui.models.bounce import BounceAlert

EASTERN = ZoneInfo("America/New_York")
WAIT = 5.0


def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _m5_bars(count: int = 24) -> list[dict]:
    start = datetime(2026, 7, 30, 9, 30, tzinfo=EASTERN)
    price = 250.0
    bars = []
    for index in range(count):
        price += 0.5 if index % 2 else -0.3
        bars.append(
            {
                "dt": start + timedelta(minutes=5 * index),
                "open": price,
                "high": price + 0.6,
                "low": price - 0.6,
                "close": price + 0.2,
                "volume": 20_000 + index,
            }
        )
    return bars


def _wire_payload() -> dict:
    bars = _m5_bars()
    payload = build_alert_popup_payload(
        BounceAlert(
            time_text="07:10:00",
            symbol="NVDA",
            side="LONG",
            trigger="VWAP bounce",
            timeframe="M5",
            context="RS leader",
        ),
        d1_snapshot={"symbol": "NVDA", "timeframe": "D1", "bars": bars[-6:], "overlays": [], "note": ""},
        m5_snapshot=chart_snapshot.build_m5_snapshot("NVDA", bars),
        armed_kinds=["vwap_bounce"],
        guidance_text="Focus name.",
    )
    return json.loads(json.dumps(payload))  # exactly what crosses the wire


def test_satellite_popup_renders_from_wire_payload():
    _qapp()
    from ui.satellite import SatellitePopupDialog

    dialog = SatellitePopupDialog(restore_alert_popup_payload(_wire_payload()))
    try:
        widget = dialog.findChildren(object.__class__, "")  # noqa: F841 (keep dialog alive)
        snapshot = next(
            child for child in dialog.children() if child.__class__.__name__ == "SymbolSnapshotWidget"
        )
        assert snapshot._symbol == "NVDA"
        assert len(snapshot._m5["bars"]) == 24
        assert snapshot._m5["overlays"], "VWAP/EMA overlays must survive the wire"
        assert isinstance(snapshot._m5["bars"][-1]["dt"], datetime)
        assert snapshot.m5_chart.isVisible() or not dialog.isVisible()  # offscreen: structure, not paint
        assert "NVDA" in dialog.windowTitle()
    finally:
        dialog.close()


def test_payload_snapshot_widget_tolerates_empty_snapshots():
    _qapp()
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    widget = SymbolSnapshotWidget()
    widget.show_payload_snapshots("AMD", {}, {})  # no bars at all: notes, not KeyError
    assert widget._symbol == "AMD"
    assert widget._d1["bars"] == []


def test_desk_link_service_relays_to_live_satellite_client(monkeypatch):
    _qapp()
    import ui.services.desk_link_service as service_module
    from desk_link.client import DeskLinkClient

    settings: dict = {"desk_link_port": 0}

    monkeypatch.setattr(service_module, "get_local_setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(
        service_module, "save_local_setting", lambda key, value: settings.__setitem__(key, value)
    )

    service = service_module.DeskLinkService(machine_name="main-under-test")
    assert service.start()
    try:
        token = settings["desk_link_token"]  # generated and persisted on first start
        port = service._server.address[1]

        received: list[dict] = []
        got_popup = threading.Event()

        def on_message(message: dict) -> None:
            received.append(message)
            if message["type"] == protocol.TYPE_ALERT_POPUP:
                got_popup.set()

        client = DeskLinkClient(
            host="127.0.0.1",
            port=port,
            token=token,
            machine_name="test-satellite",
            on_message=on_message,
        )
        client.start()
        try:
            deadline = threading.Event()
            for _ in range(50):
                if service.has_satellites:
                    break
                deadline.wait(0.1)
            assert service.has_satellites

            service.publish_alert_popup(_wire_payload())
            assert got_popup.wait(timeout=WAIT)
            popup = next(m for m in received if m["type"] == protocol.TYPE_ALERT_POPUP)
            assert restore_alert_popup_payload(popup["payload"])["alert"]["symbol"] == "NVDA"
            # The sticky state snapshot arrived on connect, before any change.
            assert any(m["type"] == protocol.TYPE_STATE_SNAPSHOT for m in received)
        finally:
            client.stop()
    finally:
        service.stop()


def test_send_test_popup_reaches_a_live_satellite(monkeypatch):
    _qapp()
    import ui.services.desk_link_service as service_module
    from desk_link.client import DeskLinkClient

    settings: dict = {"desk_link_port": 0}
    monkeypatch.setattr(service_module, "get_local_setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(service_module, "save_local_setting", lambda key, value: settings.__setitem__(key, value))

    service = service_module.DeskLinkService(machine_name="main-under-test")
    assert service.start()
    try:
        assert service.send_test_popup() is False  # nobody listening yet

        # A TCP accept is not a satellite until its authenticated hello has
        # completed.  The old client_count gate returned a false success here
        # even though broadcasts deliberately exclude unnamed connections.
        raw_client = socket.create_connection(("127.0.0.1", service._server.address[1]))
        try:
            waiter = threading.Event()
            for _ in range(50):
                if service._server.client_count == 1:
                    break
                waiter.wait(0.01)
            assert service._server.client_count == 1
            assert service.connected_machines() == []
            assert service.send_test_popup() is False
        finally:
            raw_client.close()

        received: list[dict] = []
        got_popup = threading.Event()

        def on_message(message: dict) -> None:
            received.append(message)
            if message["type"] == protocol.TYPE_ALERT_POPUP:
                got_popup.set()

        client = DeskLinkClient(
            host="127.0.0.1",
            port=service._server.address[1],
            token=settings["desk_link_token"],
            machine_name="test-satellite",
            on_message=on_message,
        )
        client.start()
        try:
            waiter = threading.Event()
            for _ in range(50):
                if service.has_satellites:
                    break
                waiter.wait(0.1)
            assert service.send_test_popup() is True
            assert got_popup.wait(timeout=WAIT)
            popup = next(m for m in received if m["type"] == protocol.TYPE_ALERT_POPUP)
            restored = restore_alert_popup_payload(popup["payload"])
            assert restored["alert"]["symbol"] == "SPY"
            assert restored["alert"]["side"] == "TEST"
        finally:
            client.stop()
    finally:
        service.stop()


def test_service_stays_off_without_the_local_setting(monkeypatch):
    import ui.services.desk_link_service as service_module

    monkeypatch.setattr(service_module, "get_local_setting", lambda key, default=None: default)
    assert service_module.desk_link_enabled() is False


def _patched_service(monkeypatch, settings: dict):
    import ui.services.desk_link_service as service_module

    monkeypatch.setattr(service_module, "get_local_setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(
        service_module, "save_local_setting", lambda key, value: settings.__setitem__(key, value)
    )
    return service_module.DeskLinkService(machine_name="main-under-test")


def test_settings_page_toggle_controls_the_service(monkeypatch):
    _qapp()
    from ui.panels.settings_panel import SettingsPanel
    from ui.state import UiState

    settings: dict = {"desk_link_port": 0}  # ephemeral port for the test bind
    service = _patched_service(monkeypatch, settings)
    panel = SettingsPanel(UiState(), desk_link_service=service)
    try:
        assert not service.running
        assert "Not serving" in panel.desk_link_status.text()

        panel.desk_link_enable_input.setChecked(True)  # user flips the toggle
        assert service.running
        assert settings["desk_link_enabled"] is True
        token = settings["desk_link_token"]
        assert token and panel.desk_link_token_view.text() == token
        assert "Serving on port" in panel.desk_link_status.text()

        panel.desk_link_enable_input.setChecked(False)
        assert not service.running
        assert settings["desk_link_enabled"] is False
        assert "Not serving" in panel.desk_link_status.text()
    finally:
        service.stop()


def _patched_satellite_settings(monkeypatch, settings: dict):
    import ui.satellite as satellite_module

    monkeypatch.setattr(
        satellite_module, "get_local_setting", lambda key, default=None: settings.get(key, default)
    )
    monkeypatch.setattr(
        satellite_module, "save_local_setting", lambda key, value: settings.__setitem__(key, value)
    )
    return satellite_module


def _patched_desk_role_settings(monkeypatch, settings: dict):
    import ui.desk_role as role_module

    monkeypatch.setattr(
        role_module, "get_local_setting", lambda key, default=None: settings.get(key, default)
    )
    monkeypatch.setattr(
        role_module, "save_local_setting", lambda key, value: settings.__setitem__(key, value)
    )
    return role_module


def test_desk_role_is_machine_local_persistent_and_fail_safe(monkeypatch):
    settings: dict = {}
    role_module = _patched_desk_role_settings(monkeypatch, settings)

    assert role_module.saved_desk_role() == "main"
    assert role_module.save_desk_role("satellite") == "satellite"
    assert settings["trading_desk_role"] == "satellite"
    assert role_module.saved_desk_role() == "satellite"

    settings["trading_desk_role"] = "unexpected-value"
    assert role_module.saved_desk_role() == "main"

    settings["trading_desk_role"] = "satellite"
    assert role_module.startup_desk_role() == "satellite"
    assert role_module.startup_desk_role(explicit="main") == "main"
    assert settings["trading_desk_role"] == "main"
    assert role_module.startup_desk_role(legacy_satellite=True) == "satellite"


def test_satellite_window_without_pairing_waits_instead_of_crashing(monkeypatch):
    _qapp()
    satellite_module = _patched_satellite_settings(monkeypatch, {})
    window = satellite_module.SatelliteWindow(machine_name="test-sat")
    try:
        # Nothing saved and nothing passed: no client yet; the pairing dialog
        # is queued for the event loop (not exec'd here) and the button is
        # always available.
        assert window._client is None
        assert window.connect_button.isEnabled()
    finally:
        window.close()


def test_satellite_window_autoconnects_from_saved_settings(monkeypatch):
    _qapp()
    settings = {"desk_link_host": "192.168.1.20:47601", "desk_link_token": "saved-token"}
    satellite_module = _patched_satellite_settings(monkeypatch, settings)
    window = satellite_module.SatelliteWindow(machine_name="test-sat")
    try:
        assert settings["desk_link_client_token"] == "saved-token"
        assert window._client is not None
        assert window._client._host == "192.168.1.20"
        assert window._client._port == 47601
        assert window._client._token == "saved-token"
        assert "192.168.1.20" in window.windowTitle()
    finally:
        window.close()


def test_connect_dialog_applies_and_persists_new_connection(monkeypatch):
    _qapp()
    from PySide6.QtWidgets import QDialog

    settings = {"desk_link_host": "old-host:47600", "desk_link_token": "old-token"}
    satellite_module = _patched_satellite_settings(monkeypatch, settings)
    window = satellite_module.SatelliteWindow(machine_name="test-sat")
    try:
        first_client = window._client

        def fake_exec(dialog_self):
            dialog_self.host_input.setText("10.0.0.5")
            dialog_self.port_input.setValue(50000)
            dialog_self.token_input.setText("new-token")
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(satellite_module.ConnectDialog, "exec", fake_exec)
        window.open_connect_dialog()

        assert settings["desk_link_host"] == "10.0.0.5:50000"
        assert settings["desk_link_client_token"] == "new-token"
        assert settings["desk_link_token"] == "old-token"  # legacy/server token is untouched
        assert window._client is not first_client  # old client replaced live
        assert window._client._host == "10.0.0.5"
        assert window._client._port == 50000
    finally:
        window.close()


def test_connect_dialog_prefills_from_saved_settings(monkeypatch):
    _qapp()
    settings = {"desk_link_host": "main-pc:48000", "desk_link_token": "tok"}
    satellite_module = _patched_satellite_settings(monkeypatch, settings)
    dialog = satellite_module.ConnectDialog()
    try:
        assert dialog.host_input.text() == "main-pc"
        assert dialog.port_input.value() == 48000
        assert dialog.token_input.text() == "tok"
    finally:
        dialog.close()


class _StubFeed:
    """Stands in for DeskLinkFeedService: records start/stop without a socket."""

    def __init__(self) -> None:
        from PySide6.QtCore import QObject, Signal

        class _Signals(QObject):
            linkStatusChanged = Signal(str, str)

        self._signals = _Signals()
        self.linkStatusChanged = self._signals.linkStatusChanged
        self.running = False
        self.link_status = ("stopped", "stopped")
        self.started: list[dict] = []
        self.stops = 0

    def current_link_status(self) -> tuple[str, str]:
        return self.link_status

    def start(self, **kwargs) -> None:
        self.started.append(kwargs)
        self.running = True
        self.link_status = ("connecting", f"connecting to {kwargs['host']}:{kwargs['port']}")

    def stop(self) -> None:
        self.stops += 1
        self.running = False
        self.link_status = ("stopped", "stopped")


def _pairing_panel(monkeypatch, settings: dict, feed=None):
    from ui.panels.settings_panel import SettingsPanel
    from ui.state import UiState

    _qapp()
    _patched_satellite_settings(monkeypatch, settings)
    _patched_desk_role_settings(monkeypatch, settings)
    settings.setdefault("desk_link_port", 0)
    # Production server and satellite helpers share one local-settings file;
    # keep one dict here so a credential-key collision cannot hide in tests.
    service = _patched_service(monkeypatch, settings)
    panel = SettingsPanel(UiState(), desk_link_service=service, desk_link_feed=feed)
    return panel, service


def test_settings_page_pairs_this_desk_with_a_main(monkeypatch):
    """The connect dialog's job, done on the Settings page instead."""
    settings: dict = {}
    feed = _StubFeed()
    panel, service = _pairing_panel(monkeypatch, settings, feed=feed)
    try:
        panel.main_desk_host_input.setText("192.168.0.223")
        panel.main_desk_port_input.setValue(47600)
        panel.main_desk_token_input.setText("relay-token")
        panel._connect_to_main_desk()

        assert settings["desk_link_host"] == "192.168.0.223:47600"
        assert settings["desk_link_client_token"] == "relay-token"
        assert feed.stops == 1  # old link dropped so the new one is not a no-op
        assert feed.started[-1]["host"] == "192.168.0.223"
        assert feed.started[-1]["port"] == 47600
        assert feed.started[-1]["token"] == "relay-token"
        assert "192.168.0.223" in panel.main_desk_link_status.text()

        # Live status from the client thread lands in the same label.
        feed.linkStatusChanged.emit("connected", "main-desk")
        assert "Connected to main-desk" in panel.main_desk_link_status.text()
        feed.linkStatusChanged.emit("rejected", "bad token")
        assert "Token rejected" in panel.main_desk_link_status.text()
    finally:
        service.stop()


def test_settings_page_prefills_pairing_and_refuses_half_filled(monkeypatch):
    settings = {"desk_link_host": "main-pc:48000", "desk_link_token": "tok"}
    feed = _StubFeed()
    panel, service = _pairing_panel(monkeypatch, settings, feed=feed)
    try:
        assert panel.main_desk_host_input.text() == "main-pc"
        assert panel.main_desk_port_input.value() == 48000
        assert panel.main_desk_token_input.text() == "tok"

        panel.main_desk_token_input.setText("   ")
        panel._connect_to_main_desk()
        assert not feed.started  # nothing sent, nothing overwritten
        assert settings["desk_link_token"] == "tok"
        assert settings["desk_link_client_token"] == "tok"  # migrated from the legacy pairing
        assert "link token" in panel.main_desk_link_status.text()

        panel.main_desk_token_input.setText("tok")
        panel._forget_main_desk()
        assert settings["desk_link_host"] == ":48000"
        assert settings["desk_link_client_token"] == ""
        assert settings["desk_link_token"] == "tok"
        assert feed.stops == 1
        assert panel.main_desk_host_input.text() == ""
    finally:
        service.stop()


def test_settings_page_pairing_on_a_normal_desk_only_saves(monkeypatch):
    """No feed (not launched with --satellite-desk): save, do not pretend to link."""
    settings: dict = {"desk_link_token": "server-token"}
    panel, service = _pairing_panel(monkeypatch, settings, feed=None)
    try:
        assert "Not paired" in panel.main_desk_link_status.text()
        panel.main_desk_host_input.setText("10.0.0.5")
        panel.main_desk_port_input.setValue(50000)
        panel.main_desk_token_input.setText("later-token")
        panel._connect_to_main_desk()

        assert settings["desk_link_host"] == "10.0.0.5:50000"
        assert settings["desk_link_client_token"] == "later-token"
        assert settings["desk_link_token"] == "server-token"
        assert service.current_token() == "server-token"
        assert "satellite" in panel.main_desk_link_status.text().lower()

        panel._forget_main_desk()
        assert settings["desk_link_client_token"] == ""
        assert settings["desk_link_token"] == "server-token"
    finally:
        service.stop()


def test_settings_page_switches_role_and_requests_safe_restart(monkeypatch):
    settings: dict = {"desk_link_token": "server-token"}
    panel, service = _pairing_panel(monkeypatch, settings, feed=None)
    requested: list[str] = []
    panel.deskRoleRestartRequested.connect(requested.append)
    try:
        assert panel.desk_role_input.currentData() == "main"
        assert not panel.desk_role_button.isEnabled()

        panel.desk_role_input.setCurrentIndex(panel.desk_role_input.findData("satellite"))
        assert panel.desk_role_button.isEnabled()
        assert "Restart as satellite" in panel.desk_role_button.text()
        panel._apply_desk_role()

        assert settings["trading_desk_role"] == "satellite"
        assert requested == ["satellite"]
        assert "Restarting" in panel.desk_role_status.text()
    finally:
        service.stop()


def test_settings_page_recovers_terminal_feed_status_on_construction(monkeypatch):
    settings = {
        "desk_link_host": "main-pc:48000",
        "desk_link_client_token": "bad-token",
    }
    feed = _StubFeed()
    feed.running = True  # a rejected DeskLinkClient object still exists
    feed.link_status = ("rejected", "bad token")

    panel, service = _pairing_panel(monkeypatch, settings, feed=feed)
    try:
        assert "Token rejected" in panel.main_desk_link_status.text()
        assert "bad token" in panel.main_desk_link_status.text()
    finally:
        service.stop()


def test_settings_page_regenerate_revokes_the_old_token(monkeypatch):
    _qapp()
    from ui.panels.settings_panel import SettingsPanel
    from ui.state import UiState

    settings: dict = {"desk_link_port": 0}
    service = _patched_service(monkeypatch, settings)
    panel = SettingsPanel(UiState(), desk_link_service=service)
    try:
        panel.desk_link_enable_input.setChecked(True)
        first = settings["desk_link_token"]
        panel._regenerate_desk_link_token()
        second = settings["desk_link_token"]
        assert second and second != first
        assert panel.desk_link_token_view.text() == second
        assert service.running  # regeneration restarts serving with the new token
    finally:
        service.stop()
