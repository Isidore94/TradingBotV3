"""Desk Link Tier 2: control lease, take-back, and intent round-trips.

Runs the real service + real clients over localhost. The service's lease
logic lives on the GUI thread behind queued signals, so waits pump the Qt
event loop instead of sleeping blind.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from desk_link import protocol
from desk_link.client import DeskLinkClient
from desk_link.outbox import IntentOutbox

WAIT = 5.0


def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _pump_until(qapp, condition, timeout: float = WAIT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


# -- outbox -----------------------------------------------------------------


def test_outbox_journals_before_send_and_survives_restart(tmp_path):
    journal = tmp_path / "journal.jsonl"
    outbox = IntentOutbox(journal)
    first = outbox.create("ignore_for_day", "nvda")
    second = outbox.create("focus_add", "AMD", side="long")
    assert first["seq"] == 1 and first["symbol"] == "NVDA"
    assert second["seq"] == 2 and second["side"] == "long"
    outbox.mark_acked(1)

    # A crash/restart reloads only the unacked decisions, seq keeps rising.
    reloaded = IntentOutbox(journal)
    assert [intent["seq"] for intent in reloaded.unacked()] == [2]
    assert reloaded.create("focus_remove", "TSLA")["seq"] == 3


def test_outbox_tolerates_missing_and_corrupt_journal(tmp_path):
    assert IntentOutbox(tmp_path / "absent.jsonl").unacked() == []
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('not json\n{"kind": "intent", "seq": 7, "action": "x", "symbol": "A"}\n')
    outbox = IntentOutbox(corrupt)
    assert [intent["seq"] for intent in outbox.unacked()] == [7]
    assert outbox.create("y", "B")["seq"] == 8


# -- lease over the live wire ------------------------------------------------


class _Sat:
    def __init__(self, port: int, name: str):
        self.messages: list[dict] = []
        self.client = DeskLinkClient(
            host="127.0.0.1",
            port=port,
            token="tkn",
            machine_name=name,
            on_message=self.messages.append,
        )
        self.client.start()

    def got(self, message_type: str) -> bool:
        return any(m["type"] == message_type for m in self.messages)

    def last_payload(self, message_type: str) -> dict:
        return next(m["payload"] for m in reversed(self.messages) if m["type"] == message_type)


@pytest.fixture
def service(monkeypatch):
    qapp = _qapp()
    import ui.services.desk_link_service as service_module

    settings = {"desk_link_port": 0, "desk_link_token": "tkn"}
    monkeypatch.setattr(service_module, "get_local_setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(service_module, "save_local_setting", lambda key, value: settings.__setitem__(key, value))
    instance = service_module.DeskLinkService(machine_name="main-under-test")
    assert instance.start()
    yield qapp, instance
    instance.stop()


def test_lease_grant_single_holder_release_and_handoff(service):
    qapp, main = service
    port = main._server.address[1]
    control_log: list[str] = []
    main.controlChanged.connect(control_log.append)

    first, second = _Sat(port, "sat-a"), _Sat(port, "sat-b")
    try:
        assert _pump_until(qapp, lambda: main.has_satellites and len(main.connected_machines()) == 2)

        first.client.send(protocol.TYPE_LEASE_REQUEST)
        assert _pump_until(qapp, lambda: first.got(protocol.TYPE_LEASE_GRANT))
        assert main.controller == "sat-a"
        assert control_log[-1] == "sat-a"

        # Exclusive: the second satellite is denied while the first holds it.
        second.client.send(protocol.TYPE_LEASE_REQUEST)
        assert _pump_until(qapp, lambda: second.got(protocol.TYPE_LEASE_DENIED))
        assert second.last_payload(protocol.TYPE_LEASE_DENIED)["holder"] == "sat-a"

        # Release, then the second can take it.
        first.client.send(protocol.TYPE_LEASE_RELEASE)
        assert _pump_until(qapp, lambda: main.controller == "")
        second.client.send(protocol.TYPE_LEASE_REQUEST)
        assert _pump_until(qapp, lambda: second.got(protocol.TYPE_LEASE_GRANT))
        assert main.controller == "sat-b"
    finally:
        first.client.stop()
        second.client.stop()


def test_controller_disconnect_auto_reclaims_control(service):
    qapp, main = service
    port = main._server.address[1]
    satellite = _Sat(port, "sat-a")
    try:
        assert _pump_until(qapp, lambda: main.has_satellites)
        satellite.client.send(protocol.TYPE_LEASE_REQUEST)
        assert _pump_until(qapp, lambda: main.controller == "sat-a")

        satellite.client.stop()  # laptop sleep / Wi-Fi drop
        assert _pump_until(qapp, lambda: main.controller == "")
    finally:
        satellite.client.stop()


def test_take_back_control_is_immediate_and_notifies_satellite(service):
    qapp, main = service
    satellite = _Sat(main._server.address[1], "sat-a")
    try:
        assert _pump_until(qapp, lambda: main.has_satellites)
        satellite.client.send(protocol.TYPE_LEASE_REQUEST)
        assert _pump_until(qapp, lambda: main.controller == "sat-a")

        main.take_back_control()
        assert main.controller == ""
        assert _pump_until(qapp, lambda: satellite.got(protocol.TYPE_LEASE_REVOKED))
    finally:
        satellite.client.stop()


def test_intents_from_controller_only_and_acked(service):
    qapp, main = service
    port = main._server.address[1]
    received: list[tuple[str, dict]] = []

    def apply_and_ack(machine: str, intent: dict) -> None:
        received.append((machine, intent))
        main.send_intent_result(machine, intent.get("seq"), True, "applied")

    main.intentReceived.connect(apply_and_ack)
    holder, bystander = _Sat(port, "sat-a"), _Sat(port, "sat-b")
    try:
        assert _pump_until(qapp, lambda: len(main.connected_machines()) == 2)
        holder.client.send(protocol.TYPE_LEASE_REQUEST)
        assert _pump_until(qapp, lambda: main.controller == "sat-a")

        # Non-controller intents are refused without touching the desk.
        bystander.client.send(protocol.TYPE_INTENT, {"seq": 9, "action": "ignore_for_day", "symbol": "NVDA"})
        assert _pump_until(qapp, lambda: bystander.got(protocol.TYPE_INTENT_RESULT))
        refusal = bystander.last_payload(protocol.TYPE_INTENT_RESULT)
        assert refusal["ok"] is False and refusal["seq"] == 9
        assert received == []

        holder.client.send(protocol.TYPE_INTENT, {"seq": 1, "action": "focus_add", "symbol": "AMD", "side": "long"})
        assert _pump_until(qapp, lambda: holder.got(protocol.TYPE_INTENT_RESULT))
        assert received[0][0] == "sat-a" and received[0][1]["symbol"] == "AMD"
        ack = holder.last_payload(protocol.TYPE_INTENT_RESULT)
        assert ack["ok"] is True and ack["seq"] == 1
    finally:
        holder.client.stop()
        bystander.client.stop()


def test_auto_reclaim_sends_phone_push(service, monkeypatch):
    qapp, main = service
    import ui.services.desk_link_service as service_module

    pushes: list[tuple[str, str]] = []
    monkeypatch.setattr(service_module.push_notify, "push_configured", lambda: True)
    monkeypatch.setattr(
        service_module.push_notify,
        "send_push",
        lambda title, message, **kwargs: pushes.append((title, message)),
    )

    satellite = _Sat(main._server.address[1], "sat-a")
    try:
        assert _pump_until(qapp, lambda: main.has_satellites)
        satellite.client.send(protocol.TYPE_LEASE_REQUEST)
        assert _pump_until(qapp, lambda: main.controller == "sat-a")

        satellite.client.stop()  # unplanned drop -> auto-reclaim -> phone push
        assert _pump_until(qapp, lambda: main.controller == "" and len(pushes) > 0)
        assert "sat-a" in pushes[0][1]

        # A deliberate take-back must NOT page the trader's phone.
        pushes.clear()
        main.take_back_control()
        qapp.processEvents()
        assert pushes == []
    finally:
        satellite.client.stop()


def test_applied_intent_republishes_the_desk_snapshot_immediately(service):
    qapp, main = service
    main.intentReceived.connect(
        lambda machine, intent: main.send_intent_result(machine, intent.get("seq"), True, "applied")
    )
    satellite = _Sat(main._server.address[1], "sat-a")
    try:
        assert _pump_until(qapp, lambda: main.has_satellites)
        satellite.client.send(protocol.TYPE_LEASE_REQUEST)
        assert _pump_until(qapp, lambda: satellite.got(protocol.TYPE_LEASE_GRANT))

        def snapshots() -> int:
            return sum(m["type"] == protocol.TYPE_STATE_SNAPSHOT for m in satellite.messages)

        baseline = snapshots()  # the sticky snapshot from connect
        satellite.client.send(
            protocol.TYPE_INTENT, {"seq": 1, "action": "focus_add", "symbol": "AMD", "side": "long"}
        )
        # The mirror refresh must not wait out the 60s snapshot timer.
        assert _pump_until(qapp, lambda: snapshots() > baseline)
    finally:
        satellite.client.stop()


# -- intent application on the main ------------------------------------------


def _focus_stub():
    from PySide6.QtCore import QObject, Signal

    class _FocusStub(QObject):
        focusChanged = Signal()

        def __init__(self):
            super().__init__()
            self.calls: list[tuple] = []

        def add(self, symbol, side, category="m5", *, origin="", context=""):
            self.calls.append(("add", symbol, side, origin, context))
            return True

        def remove_everywhere(self, symbol, *, origin="", context=""):
            self.calls.append(("remove", symbol, origin, context))
            return 1

        def is_focus(self, symbol, side=None, category=None):
            return False

        def focus_side(self, symbol, category=None):
            return None

        def focus_category(self, symbol):
            return ""

        def all_focus(self, category=None):
            return {"long": [], "short": []}

    return _FocusStub()


def test_panel_applies_intents_through_local_paths(tmp_path):
    _qapp()
    from ui.panels.alert_center_panel import AlertCenterPanel

    focus = _focus_stub()
    panel = AlertCenterPanel(
        focus,
        ignored_symbols_path=tmp_path / "ignored.txt",
        review_events_path=tmp_path / "events.jsonl",
    )

    ok, detail = panel.apply_desk_link_intent("macbook", {"action": "ignore_for_day", "symbol": "nvda"})
    assert ok and "NVDA" in detail
    assert "NVDA" in panel._ignored_symbols

    ok, _ = panel.apply_desk_link_intent("macbook", {"action": "focus_add", "symbol": "AMD", "side": "long"})
    assert ok
    assert focus.calls[-1] == ("add", "AMD", "long", "desk_link", "desk_link:macbook")

    ok, _ = panel.apply_desk_link_intent("macbook", {"action": "focus_remove", "symbol": "AMD"})
    assert ok
    assert focus.calls[-1] == ("remove", "AMD", "desk_link", "desk_link:macbook")

    ok, detail = panel.apply_desk_link_intent("macbook", {"action": "focus_add", "symbol": "X", "side": "sideways"})
    assert not ok and "side" in detail
    ok, detail = panel.apply_desk_link_intent("macbook", {"action": "warp_drive", "symbol": "X"})
    assert not ok
    ok, detail = panel.apply_desk_link_intent("macbook", {"action": "ignore_for_day"})
    assert not ok and "symbol" in detail
