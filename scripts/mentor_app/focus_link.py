"""A tiny local socket that brings the running Trade Mentor window to the front.

The app listens on a per-user QLocalServer name; a second launch (or the desk's
button) connects and writes one line. ``send_focus_ping`` blocks for at most
``timeout_ms`` and is meant for a worker thread, never the desk's Qt thread.
"""

from __future__ import annotations

import getpass
import logging
import re

FOCUS_MESSAGE = b"focus\n"


def server_name() -> str:
    """One name per Windows user, so two users on one box never focus each other."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - an unnamed user still gets one server
        user = "user"
    return "tradingbotv3-mentor-focus-" + re.sub(r"[^A-Za-z0-9_.-]", "_", user)


def send_focus_ping(timeout_ms: int = 500, *, name: str | None = None) -> bool:
    """True when a running app received the ping. Safe from any thread."""
    try:
        from PySide6.QtNetwork import QLocalSocket
    except Exception:  # noqa: BLE001
        return False
    socket = QLocalSocket()
    try:
        socket.connectToServer(name or server_name())
        if not socket.waitForConnected(int(timeout_ms)):
            return False
        socket.write(FOCUS_MESSAGE)
        socket.flush()
        socket.waitForBytesWritten(int(timeout_ms))
        return True
    except Exception as exc:  # noqa: BLE001
        logging.info("Trade Mentor focus ping failed: %s", exc)
        return False
    finally:
        socket.abort()


def make_focus_server(parent, on_focus, *, name: str | None = None):
    """Listen for focus pings on the Qt thread; returns the server or None."""
    from PySide6.QtNetwork import QLocalServer

    server = QLocalServer(parent)
    target = name or server_name()
    if not server.listen(target):
        # A dead app can leave a stale name behind on some platforms.
        QLocalServer.removeServer(target)
        if not server.listen(target):
            logging.warning("Trade Mentor focus server could not listen: %s", server.errorString())
            return None

    def _accept() -> None:
        while server.hasPendingConnections():
            conn = server.nextPendingConnection()
            if conn is None:
                break
            conn.readyRead.connect(conn.readAll)
            conn.disconnected.connect(conn.deleteLater)
            on_focus()

    server.newConnection.connect(_accept)
    return server
