"""Desk Link transport tests: protocol contract + live localhost sessions.

Everything runs against 127.0.0.1 with ephemeral ports and event-based
waits (no sleeps as assertions), so the suite stays fast and deterministic
on any platform.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from desk_link import protocol
from desk_link.client import DeskLinkClient
from desk_link.framing import LineReader
from desk_link.server import DeskLinkServer

# Real client/server sockets on loopback ARE the point of this file: Desk Link
# is a wire protocol, and a test that mocked the transport would be testing a
# mock. Marked `network` so the offline tripwire lets it through - the marker
# permits, it does not deselect, so these still run in every suite.
#
# This is a re-marking, not a weakening: nothing here reaches the internet or a
# broker, and every assertion is unchanged.
pytestmark = pytest.mark.network


WAIT = 5.0


# -- protocol ---------------------------------------------------------------


def test_message_roundtrip_and_envelope_validation():
    message = protocol.make_message(protocol.TYPE_ALERT_POPUP, {"symbol": "NVDA"})
    decoded = protocol.decode_message(protocol.encode_message(message).rstrip(b"\n"))
    assert decoded["type"] == protocol.TYPE_ALERT_POPUP
    assert decoded["payload"] == {"symbol": "NVDA"}
    assert decoded["v"] == protocol.PROTOCOL_VERSION
    assert "+00:00" in decoded["ts"] or decoded["ts"].endswith("Z")

    with pytest.raises(protocol.DeskLinkProtocolError):
        protocol.decode_message(b"not json")
    with pytest.raises(protocol.DeskLinkProtocolError):
        protocol.decode_message(b"[1, 2]")
    with pytest.raises(protocol.DeskLinkProtocolError):
        protocol.decode_message(b'{"v": 99, "type": "x", "payload": {}}')
    with pytest.raises(protocol.DeskLinkProtocolError):
        protocol.decode_message(b'{"v": 1, "payload": {}}')


def test_hello_validation_rejects_bad_token_constant_time_path():
    hello = protocol.make_hello("right-token", "macbook")
    assert protocol.validate_hello(hello, "right-token") == "macbook"
    with pytest.raises(protocol.DeskLinkAuthError):
        protocol.validate_hello(protocol.make_hello("wrong", "macbook"), "right-token")
    with pytest.raises(protocol.DeskLinkAuthError):
        protocol.validate_hello(protocol.make_message(protocol.TYPE_PING), "right-token")


def test_server_refuses_to_start_without_token():
    with pytest.raises(ValueError):
        DeskLinkServer(token="   ", machine_name="main")


# -- framing ----------------------------------------------------------------


def test_line_reader_caps_oversized_frames():
    server_sock, client_sock = socket.socketpair()
    try:
        reader = LineReader(server_sock, max_line_bytes=64)
        client_sock.sendall(b"x" * 200)
        with pytest.raises(protocol.DeskLinkProtocolError):
            reader.read_line()
    finally:
        server_sock.close()
        client_sock.close()


# -- live sessions ----------------------------------------------------------


class _Satellite:
    """Test double built on the real DeskLinkClient with event-based waits."""

    def __init__(self, port: int, token: str, name: str = "sat", last_popup_seq: int = 0) -> None:
        self.messages: list[dict] = []
        self.statuses: list[tuple[str, str]] = []
        self.last_popup_seq = last_popup_seq
        self._message_seen = threading.Condition()
        self._status_seen = threading.Condition()
        self.client = DeskLinkClient(
            host="127.0.0.1",
            port=port,
            token=token,
            machine_name=name,
            on_message=self._on_message,
            on_status=self._on_status,
            hello_extra=lambda: {"last_popup_seq": self.last_popup_seq},
        )

    def _on_message(self, message: dict) -> None:
        with self._message_seen:
            self.messages.append(message)
            self._message_seen.notify_all()

    def _on_status(self, state: str, detail: str) -> None:
        with self._status_seen:
            self.statuses.append((state, detail))
            self._status_seen.notify_all()

    def got_types(self, message_type: str) -> bool:
        with self._message_seen:
            return any(m["type"] == message_type for m in self.messages)

    def wait_for_message(self, message_type: str, timeout: float = WAIT) -> dict:
        with self._message_seen:
            ok = self._message_seen.wait_for(
                lambda: any(m["type"] == message_type for m in self.messages), timeout=timeout
            )
            assert ok, f"no {message_type!r} within {timeout}s; got {[m['type'] for m in self.messages]}"
            return next(m for m in self.messages if m["type"] == message_type)

    def wait_for_state(self, state: str, timeout: float = WAIT) -> None:
        with self._status_seen:
            ok = self._status_seen.wait_for(
                lambda: any(s == state for s, _ in self.statuses), timeout=timeout
            )
            assert ok, f"state {state!r} not reached within {timeout}s; got {self.statuses}"


@pytest.fixture
def server():
    started: list[DeskLinkServer] = []

    def _make(**kwargs) -> DeskLinkServer:
        kwargs.setdefault("token", "test-token")
        kwargs.setdefault("machine_name", "main-desk")
        kwargs.setdefault("host", "127.0.0.1")
        kwargs.setdefault("port", 0)
        instance = DeskLinkServer(**kwargs)
        instance.start()
        started.append(instance)
        return instance

    yield _make
    for instance in started:
        instance.stop()


def test_handshake_snapshot_on_connect_and_live_broadcast(server):
    connected = threading.Event()
    main = server(on_client_connected=lambda machine, addr: connected.set())
    main.set_state_snapshot({"watchlists": {"longs": ["NVDA"]}})

    satellite = _Satellite(main.address[1], "test-token", name="macbook")
    satellite.client.start()
    try:
        satellite.wait_for_state("connected")
        assert connected.wait(timeout=WAIT)
        # The sticky snapshot arrives without waiting for the next change.
        snapshot = satellite.wait_for_message(protocol.TYPE_STATE_SNAPSHOT)
        assert snapshot["payload"]["watchlists"]["longs"] == ["NVDA"]

        main.send_alert_popup({"symbol": "AMD", "timeframe": "M5"})
        popup = satellite.wait_for_message(protocol.TYPE_ALERT_POPUP)
        assert popup["payload"]["symbol"] == "AMD"
        assert main.connected_machines() == ["macbook"]
    finally:
        satellite.client.stop()


def test_bad_token_is_rejected_and_client_stops_retrying(server):
    main = server()
    satellite = _Satellite(main.address[1], "wrong-token")
    satellite.client.start()
    try:
        satellite.wait_for_state("rejected")
        deadline = threading.Event()
        for _ in range(50):  # server-side reap races the client's rejected state
            if main.client_count == 0:
                break
            deadline.wait(0.1)
        assert main.client_count == 0
        # The client thread must have exited: rejection is terminal.
        thread = satellite.client._thread
        if thread is not None:
            thread.join(timeout=WAIT)
            assert not thread.is_alive()
    finally:
        satellite.client.stop()


def test_disconnect_notifies_server_and_client_reconnects(server):
    events: list[str] = []
    seen = threading.Condition()

    def note(kind):
        def _cb(machine, addr):
            with seen:
                events.append(kind)
                seen.notify_all()
        return _cb

    main = server(
        on_client_connected=note("connect"),
        on_client_disconnected=note("disconnect"),
    )
    satellite = _Satellite(main.address[1], "test-token")
    satellite.client.start()
    try:
        satellite.wait_for_state("connected")
        # Kill the transport out from under the client: it must notice and
        # reconnect on its own (Wi-Fi blip semantics).
        with satellite.client._sock_lock:
            live_sock = satellite.client._sock
        live_sock.shutdown(socket.SHUT_RDWR)
        with seen:
            ok = seen.wait_for(lambda: events.count("connect") >= 2, timeout=30.0)
        assert ok, f"no reconnect within 30s; events={events}"
        assert "disconnect" in events
    finally:
        satellite.client.stop()


def test_multiple_satellites_all_receive_broadcasts(server):
    main = server()
    first = _Satellite(main.address[1], "test-token", name="sat-a")
    second = _Satellite(main.address[1], "test-token", name="sat-b")
    first.client.start()
    second.client.start()
    try:
        first.wait_for_state("connected")
        second.wait_for_state("connected")
        main.send_alert_popup({"symbol": "TSLA"})
        assert first.wait_for_message(protocol.TYPE_ALERT_POPUP)["payload"]["symbol"] == "TSLA"
        assert second.wait_for_message(protocol.TYPE_ALERT_POPUP)["payload"]["symbol"] == "TSLA"
        assert main.connected_machines() == ["sat-a", "sat-b"]
    finally:
        first.client.stop()
        second.client.stop()


def test_missed_popups_replay_on_reconnect_from_last_seen(server):
    main = server()
    first = _Satellite(main.address[1], "test-token", name="sat-a")
    first.client.start()
    try:
        first.wait_for_state("connected")
        main.send_alert_popup({"symbol": "A"})
        seen = first.wait_for_message(protocol.TYPE_ALERT_POPUP)
        assert seen["payload"]["relay_seq"] == 1
        assert not seen["payload"].get("replayed")
    finally:
        first.client.stop()

    # Popups fired while no satellite is connected must not be lost.
    main.send_alert_popup({"symbol": "B"})
    main.send_alert_popup({"symbol": "C"})

    second = _Satellite(main.address[1], "test-token", name="sat-a", last_popup_seq=1)
    second.client.start()
    try:
        second.wait_for_state("connected")
        second.wait_for_message(protocol.TYPE_ALERT_POPUP)
        deadline = threading.Event()
        for _ in range(50):
            if sum(m["type"] == protocol.TYPE_ALERT_POPUP for m in second.messages) >= 2:
                break
            deadline.wait(0.1)
        replayed = [m["payload"] for m in second.messages if m["type"] == protocol.TYPE_ALERT_POPUP]
        assert [p["symbol"] for p in replayed] == ["B", "C"]
        assert all(p["replayed"] is True for p in replayed)
        assert [p["relay_seq"] for p in replayed] == [2, 3]
    finally:
        second.client.stop()


def test_fresh_session_and_stale_popups_do_not_replay(server):
    main = server()
    main.send_alert_popup({"symbol": "OLD"})
    main.send_alert_popup({"symbol": "STALE"})
    # Age out the second entry artificially: too old to interrupt with.
    with main._popup_lock:
        seq, stamp, payload = main._popup_buffer[-1]
        main._popup_buffer[-1] = (seq, stamp - 3600.0, payload)

    fresh = _Satellite(main.address[1], "test-token", name="fresh-sat")  # last_popup_seq=0
    behind = _Satellite(main.address[1], "test-token", name="behind-sat", last_popup_seq=1)
    fresh.client.start()
    behind.client.start()
    try:
        fresh.wait_for_state("connected")
        behind.wait_for_state("connected")
        # `behind` would be owed seq 2, but it aged out; `fresh` gets nothing.
        main.set_state_snapshot({"marker": True})  # ordered after any replay
        fresh.wait_for_message(protocol.TYPE_STATE_SNAPSHOT)
        behind.wait_for_message(protocol.TYPE_STATE_SNAPSHOT)
        assert not fresh.got_types(protocol.TYPE_ALERT_POPUP)
        assert not behind.got_types(protocol.TYPE_ALERT_POPUP)
    finally:
        fresh.client.stop()
        behind.client.stop()


def test_server_stop_is_clean_and_idempotent(server):
    main = server()
    satellite = _Satellite(main.address[1], "test-token")
    satellite.client.start()
    try:
        satellite.wait_for_state("connected")
        main.stop()
        main.stop()  # second stop must be a no-op, not an error
        # stop() snapshots _clients, drops them, then joins the accept thread.
        # An accept that is mid-registration when the snapshot is taken lands in
        # _clients *after* the drop, so under load the count settles a moment
        # later. That ordering is a real (small) race in DeskLinkServer.stop,
        # but Desk Link is retired (plan.md 7a note, 2026-08-08) and its code is
        # frozen pending the cleanup packet -- so this waits for the count to
        # settle rather than changing a server that must stay unused.
        deadline = time.monotonic() + 5.0
        while main.client_count and time.monotonic() < deadline:
            time.sleep(0.02)
        assert main.client_count == 0
    finally:
        satellite.client.stop()
