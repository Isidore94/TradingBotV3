"""Desk Link relay server: runs on the main desk, feeds satellites.

Threading contract (plan.md sec 5 — one component owns each thread): this
class owns one accept thread plus a reader and a writer thread per
connected satellite, all daemon, all torn down by ``stop()``. Nothing here
may ever block or crash the desk: socket errors are contained per client,
and a slow satellite (full outbound queue) is disconnected rather than
allowed to apply backpressure to the engine.

Tier 1 satellites are view-only, so the server's inbound handling is just
the hello handshake and ping keepalives; everything else a client sends is
ignored (logged once per connection).
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from desk_link import protocol
from desk_link.framing import LineReader

log = logging.getLogger(__name__)

DEFAULT_PORT = 47600
_HANDSHAKE_TIMEOUT_SECONDS = 10.0
# Client pings every ~5 s; missing several in a row means the satellite is
# gone (sleep/Wi-Fi drop) and the connection should be reaped.
_CLIENT_IDLE_TIMEOUT_SECONDS = 30.0
_OUTBOUND_QUEUE_MESSAGES = 128
_SENTINEL = object()


@dataclass
class _ClientConnection:
    sock: socket.socket
    address: tuple[str, int]
    machine: str = ""
    outbound: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=_OUTBOUND_QUEUE_MESSAGES))
    closed: threading.Event = field(default_factory=threading.Event)


class DeskLinkServer:
    """Accepts satellite connections, authenticates them, relays messages."""

    def __init__(
        self,
        *,
        token: str,
        machine_name: str,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        on_client_connected: Callable[[str, tuple[str, int]], None] | None = None,
        on_client_disconnected: Callable[[str, tuple[str, int]], None] | None = None,
    ) -> None:
        if not str(token or "").strip():
            raise ValueError("Desk Link refuses to serve without a link token")
        self._token = str(token)
        self._machine_name = str(machine_name or "main")
        self._host = host
        self._port = int(port)
        self._on_client_connected = on_client_connected
        self._on_client_disconnected = on_client_disconnected

        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._clients: dict[int, _ClientConnection] = {}
        self._clients_lock = threading.Lock()
        self._closing = threading.Event()
        self._last_snapshot: dict[str, Any] | None = None
        self._snapshot_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    @property
    def address(self) -> tuple[str, int] | None:
        if self._listener is None:
            return None
        try:
            return self._listener.getsockname()
        except OSError:
            return None

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def connected_machines(self) -> list[str]:
        with self._clients_lock:
            return sorted(client.machine for client in self._clients.values() if client.machine)

    def start(self) -> None:
        if self._accept_thread is not None:
            return
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._host, self._port))
        listener.listen(4)
        # A blocking accept() holds an io-ref on the socket, which defers the
        # real close in stop() and leaves the port accepting while the accept
        # thread is joined. A short timeout keeps the loop checking _closing.
        listener.settimeout(0.5)
        self._listener = listener
        self._closing.clear()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="desk-link-accept", daemon=True
        )
        self._accept_thread.start()

    def stop(self) -> None:
        self._closing.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self._clients_lock:
            clients = list(self._clients.values())
        for client in clients:
            self._drop_client(client, reason="server stopping")
        thread, self._accept_thread = self._accept_thread, None
        if thread is not None:
            thread.join(timeout=5.0)

    # -- relaying ------------------------------------------------------------

    def set_state_snapshot(self, payload: dict[str, Any]) -> None:
        """Store the sticky snapshot and push it to connected satellites.

        New satellites receive the stored snapshot right after welcome, so a
        reconnect always resyncs to current state without waiting for the
        next change.
        """
        message = protocol.make_message(protocol.TYPE_STATE_SNAPSHOT, payload)
        with self._snapshot_lock:
            self._last_snapshot = message
        self._broadcast(message)

    def send_alert_popup(self, payload: dict[str, Any]) -> None:
        self._broadcast(protocol.make_message(protocol.TYPE_ALERT_POPUP, payload))

    def _broadcast(self, message: dict[str, Any]) -> None:
        try:
            raw = protocol.encode_message(message)
        except protocol.DeskLinkProtocolError:
            log.exception("Desk Link refused to broadcast an oversized message.")
            return
        with self._clients_lock:
            clients = [client for client in self._clients.values() if client.machine]
        for client in clients:
            try:
                client.outbound.put_nowait(raw)
            except queue.Full:
                log.warning(
                    "Desk Link dropping slow satellite %s (%s): outbound queue full.",
                    client.machine,
                    client.address,
                )
                self._drop_client(client, reason="outbound queue overflow")

    # -- threads -------------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._closing.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                sock, address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return  # listener closed by stop()
            if self._closing.is_set():
                try:
                    sock.close()
                except OSError:
                    pass
                return
            client = _ClientConnection(sock=sock, address=address)
            with self._clients_lock:
                self._clients[id(client)] = client
            threading.Thread(
                target=self._client_reader,
                args=(client,),
                name=f"desk-link-reader-{address[0]}",
                daemon=True,
            ).start()

    def _client_reader(self, client: _ClientConnection) -> None:
        try:
            client.sock.settimeout(_HANDSHAKE_TIMEOUT_SECONDS)
            reader = LineReader(client.sock)
            line = reader.read_line()
            if line is None:
                return
            hello = protocol.decode_message(line)
            try:
                client.machine = protocol.validate_hello(hello, self._token)
            except protocol.DeskLinkAuthError as exc:
                log.warning("Desk Link rejected %s: %s", client.address, exc)
                try:
                    client.sock.sendall(protocol.encode_message(protocol.make_rejected(str(exc))))
                except OSError:
                    pass
                return

            threading.Thread(
                target=self._client_writer,
                args=(client,),
                name=f"desk-link-writer-{client.address[0]}",
                daemon=True,
            ).start()
            client.outbound.put(protocol.encode_message(protocol.make_welcome(self._machine_name)))
            with self._snapshot_lock:
                snapshot = self._last_snapshot
            if snapshot is not None:
                client.outbound.put(protocol.encode_message(snapshot))
            log.info("Desk Link satellite connected: %s from %s", client.machine, client.address)
            if self._on_client_connected is not None:
                try:
                    self._on_client_connected(client.machine, client.address)
                except Exception:
                    log.exception("Desk Link on_client_connected callback failed.")

            client.sock.settimeout(_CLIENT_IDLE_TIMEOUT_SECONDS)
            warned_unexpected = False
            while not self._closing.is_set() and not client.closed.is_set():
                line = reader.read_line()
                if line is None:
                    return
                message = protocol.decode_message(line)
                if message["type"] == protocol.TYPE_PING:
                    try:
                        client.outbound.put_nowait(
                            protocol.encode_message(protocol.make_message(protocol.TYPE_PONG))
                        )
                    except queue.Full:
                        return
                elif not warned_unexpected:
                    warned_unexpected = True
                    log.info(
                        "Desk Link ignoring unexpected %r from view-only satellite %s.",
                        message["type"],
                        client.machine,
                    )
        except (protocol.DeskLinkProtocolError, socket.timeout, OSError) as exc:
            if not self._closing.is_set() and not client.closed.is_set():
                log.info("Desk Link connection %s ended: %s", client.address, exc)
        except Exception:
            log.exception("Desk Link reader for %s failed unexpectedly.", client.address)
        finally:
            self._drop_client(client, reason="connection ended")

    def _client_writer(self, client: _ClientConnection) -> None:
        try:
            while not client.closed.is_set():
                item = client.outbound.get()
                if item is _SENTINEL:
                    return
                client.sock.sendall(item)
        except OSError:
            pass
        except Exception:
            log.exception("Desk Link writer for %s failed unexpectedly.", client.address)
        finally:
            self._drop_client(client, reason="writer ended")

    def _drop_client(self, client: _ClientConnection, *, reason: str) -> None:
        if client.closed.is_set():
            return
        client.closed.set()
        try:
            client.outbound.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        try:
            client.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            client.sock.close()
        except OSError:
            pass
        with self._clients_lock:
            removed = self._clients.pop(id(client), None)
        if removed is not None and client.machine:
            log.info("Desk Link satellite disconnected: %s (%s)", client.machine, reason)
            if self._on_client_disconnected is not None:
                try:
                    self._on_client_disconnected(client.machine, client.address)
                except Exception:
                    log.exception("Desk Link on_client_disconnected callback failed.")
