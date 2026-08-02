"""Bounded line framing for Desk Link sockets.

``socket.makefile().readline()`` has no hard cap, so a corrupt or hostile
peer could grow the buffer without bound. This reader caps the in-flight
line at the protocol's message limit and surfaces overruns as protocol
errors so the connection gets dropped instead of the process bloating.
"""

from __future__ import annotations

import socket

from desk_link.protocol import MAX_MESSAGE_BYTES, DeskLinkProtocolError

_RECV_CHUNK = 65536


class LineReader:
    """Reads newline-terminated frames from a socket with a size cap."""

    def __init__(self, sock: socket.socket, max_line_bytes: int = MAX_MESSAGE_BYTES) -> None:
        self._sock = sock
        self._max = int(max_line_bytes)
        self._buffer = bytearray()

    def read_line(self) -> bytes | None:
        """Return the next frame without its newline, or None on EOF.

        Raises DeskLinkProtocolError if a frame exceeds the cap, and lets
        socket.timeout / OSError from the underlying socket propagate.
        """
        while True:
            newline = self._buffer.find(b"\n")
            if newline != -1:
                line = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                return line
            if len(self._buffer) > self._max:
                raise DeskLinkProtocolError("peer sent an oversized frame")
            chunk = self._sock.recv(_RECV_CHUNK)
            if not chunk:
                if self._buffer:
                    raise DeskLinkProtocolError("peer closed mid-frame")
                return None
            self._buffer.extend(chunk)
