"""Desk Link satellite client: connects to the main desk and mirrors it.

Owns a single connection thread (reconnect loop included). Messages are
dispatched to the ``on_message`` callback on that thread — the GUI layer is
responsible for hopping to the Qt main thread (signal emission) before
touching widgets. Pings double as the keepalive both ways: the client sends
one whenever the line has been quiet for a ping interval, and the server's
idle timeout reaps connections that stop pinging.

A ``rejected`` handshake (bad token) stops the retry loop — a wrong token
never fixes itself, and hammering the server would just fill its log.
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Any, Callable

from desk_link import protocol
from desk_link.framing import LineReader

log = logging.getLogger(__name__)

_CONNECT_TIMEOUT_SECONDS = 5.0
_PING_INTERVAL_SECONDS = 5.0
_RECONNECT_DELAYS_SECONDS = (2.0, 4.0, 8.0, 15.0, 30.0)

STATE_CONNECTING = "connecting"
STATE_CONNECTED = "connected"
STATE_DISCONNECTED = "disconnected"
STATE_REJECTED = "rejected"
STATE_STOPPED = "stopped"


class DeskLinkClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        token: str,
        machine_name: str,
        on_message: Callable[[dict[str, Any]], None],
        on_status: Callable[[str, str], None] | None = None,
    ) -> None:
        self._host = str(host)
        self._port = int(port)
        self._token = str(token)
        self._machine_name = str(machine_name or "satellite")
        self._on_message = on_message
        self._on_status = on_status

        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._sock: socket.socket | None = None
        self._sock_lock = threading.Lock()
        self._state = STATE_STOPPED

    @property
    def state(self) -> str:
        return self._state

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, name="desk-link-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._close_socket()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)
        self._set_state(STATE_STOPPED, "stopped")

    # -- internals -----------------------------------------------------------

    def _set_state(self, state: str, detail: str) -> None:
        self._state = state
        if self._on_status is not None:
            try:
                self._on_status(state, detail)
            except Exception:
                log.exception("Desk Link client on_status callback failed.")

    def _close_socket(self) -> None:
        with self._sock_lock:
            sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def _run(self) -> None:
        attempt = 0
        while not self._stopping.is_set():
            self._set_state(STATE_CONNECTING, f"connecting to {self._host}:{self._port}")
            try:
                self._connect_and_serve()
                attempt = 0  # a successful session resets the backoff ladder
            except _HandshakeRejected as exc:
                self._set_state(STATE_REJECTED, str(exc))
                log.warning("Desk Link handshake rejected; not retrying: %s", exc)
                return
            except (protocol.DeskLinkProtocolError, OSError) as exc:
                if self._stopping.is_set():
                    return
                log.info("Desk Link connection lost: %s", exc)
            except Exception:
                log.exception("Desk Link client failed unexpectedly.")
            finally:
                self._close_socket()
            if self._stopping.is_set():
                return
            delay = _RECONNECT_DELAYS_SECONDS[min(attempt, len(_RECONNECT_DELAYS_SECONDS) - 1)]
            attempt += 1
            self._set_state(STATE_DISCONNECTED, f"retrying in {delay:.0f}s")
            if self._stopping.wait(timeout=delay):
                return

    def _connect_and_serve(self) -> None:
        sock = socket.create_connection((self._host, self._port), timeout=_CONNECT_TIMEOUT_SECONDS)
        with self._sock_lock:
            if self._stopping.is_set():
                sock.close()
                return
            self._sock = sock

        reader = LineReader(sock)
        sock.sendall(protocol.encode_message(protocol.make_hello(self._token, self._machine_name)))
        line = reader.read_line()
        if line is None:
            raise OSError("server closed during handshake")
        reply = protocol.decode_message(line)
        if reply["type"] == protocol.TYPE_REJECTED:
            raise _HandshakeRejected(str(reply["payload"].get("reason") or "rejected"))
        if reply["type"] != protocol.TYPE_WELCOME:
            raise protocol.DeskLinkProtocolError(f"expected welcome, got {reply['type']!r}")
        self._set_state(STATE_CONNECTED, str(reply["payload"].get("machine") or "main"))

        sock.settimeout(_PING_INTERVAL_SECONDS)
        while not self._stopping.is_set():
            try:
                line = reader.read_line()
            except TimeoutError:
                # Quiet line: keepalive. (socket.timeout is TimeoutError.)
                sock.sendall(protocol.encode_message(protocol.make_message(protocol.TYPE_PING)))
                continue
            if line is None:
                raise OSError("server closed the connection")
            message = protocol.decode_message(line)
            if message["type"] == protocol.TYPE_PONG:
                continue
            try:
                self._on_message(message)
            except Exception:
                log.exception("Desk Link on_message handler failed for %r.", message["type"])


class _HandshakeRejected(Exception):
    pass
