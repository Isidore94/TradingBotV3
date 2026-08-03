"""Desk Link Tier 1 end-to-end: capture on the main → wire → satellite render.

Proves the critical path with the real components: the payload built from
real chart_snapshot output travels through the real server/client pair and
renders in the same SymbolSnapshotWidget the main desk uses — offscreen,
no TWS, no local stores on the receiving side.
"""

from __future__ import annotations

import json
import os
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
        assert settings["desk_link_token"] == "new-token"
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


def test_satellite_mode_selector_mirrors_and_gates_on_control(monkeypatch):
    """The satellite shows the main's Auto mode from the snapshot and can only
    change it while holding the control lease (Tier 2 rule)."""
    _qapp()
    satellite_module = _patched_satellite_settings(monkeypatch, {})
    window = satellite_module.SatelliteWindow(machine_name="test-sat")
    try:
        # Snapshot mirroring never sends anything and never enables the combo.
        window._show_snapshot({"auto_mode": "AWAY", "watchlists": {}, "focus": {}})
        assert "AWAY" in window.mode_label.text()
        assert window.mode_selector.currentText() == "AWAY"
        assert not window.mode_selector.isEnabled()

        sent: list[tuple[str, dict]] = []

        class _FakeClient:
            def send(self, message_type, payload=None):
                sent.append((message_type, payload or {}))
                return True

            def stop(self):
                pass

        window._client = _FakeClient()

        # Not in control: the pick is refused locally, nothing hits the wire.
        window._on_mode_selected(list(satellite_module.AUTO_MODES).index("EVENING"))
        assert not any(kind == protocol.TYPE_INTENT for kind, _ in sent)
        assert window.mode_selector.currentText() == "AWAY"  # reverted to the main's truth

        window._set_in_control(True, "in control")
        assert window.mode_selector.isEnabled()
        window._on_mode_selected(list(satellite_module.AUTO_MODES).index("EVENING"))
        intents = [payload for kind, payload in sent if kind == protocol.TYPE_INTENT]
        assert intents and intents[-1]["action"] == "set_auto_mode"
        assert intents[-1]["mode"] == "EVENING"
    finally:
        window.close()


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
