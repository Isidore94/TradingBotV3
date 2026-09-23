"""Close a peewee thread's SQLite connection when that thread ends.

yfinance keeps its timezone and cookie caches in SQLite through peewee, and
peewee gives every thread its own connection. ``yf.download(threads=True)``
runs each ticker on a fresh thread, so every ticker leaves two connections
behind when its thread exits. A ``sqlite3.Connection`` sits in a reference
cycle with its own statement cache, so reference counting never frees it:
only the cyclic collector does, and on the desk that runs on the GUI thread.
Closing a connection releases the interpreter lock several times, and while
another Python thread is busy each release costs the GUI thread about one
switch interval to get the lock back. Measured: 80 leaked connections cost
11 ms to collect on a quiet process and 5.1 s beside one busy thread.

:func:`install` makes each peewee thread-local connection carry a small
guard object. The guard is not in a cycle, so it is freed on the exiting
thread when that thread's locals are dropped, and it closes the connection
there - an explicit close also breaks the statement-cache cycle, so nothing
is left for the GUI collector. Queries, results and caches are unchanged.
"""

from __future__ import annotations

import sqlite3
from typing import Any

_MARKER = "_tbv3_close_with_thread"
_originals: dict[str, Any] = {}


class _CloseWithThread:
    """Closes one connection when the owning thread's locals are dropped."""

    __slots__ = ("conn",)

    def __init__(self, conn) -> None:
        self.conn = conn

    def __del__(self) -> None:
        conn, self.conn = self.conn, None
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            pass  # a close that fails leaves the old behaviour: gc frees it


def install() -> bool:
    """Patch peewee's per-thread connection state. Idempotent; never raises."""
    try:
        import peewee
    except Exception:
        return False
    local_cls = getattr(peewee, "_ConnectionLocal", None)
    if local_cls is None:
        return False
    if getattr(local_cls, _MARKER, False):
        return True
    base_set = local_cls.set_connection
    base_reset = local_cls.reset

    def set_connection(self, conn):
        base_set(self, conn)
        self._tbv3_closer = _CloseWithThread(conn) if isinstance(conn, sqlite3.Connection) else None

    def reset(self):
        base_reset(self)
        # peewee resets after its own close, or before a connect when the
        # previous connection is already closed; the guard has nothing to do.
        self._tbv3_closer = None

    _originals["set_connection"] = local_cls.__dict__.get("set_connection")
    _originals["reset"] = local_cls.__dict__.get("reset")
    local_cls.set_connection = set_connection
    local_cls.reset = reset
    setattr(local_cls, _MARKER, True)
    return True


def uninstall() -> None:
    """Undo :func:`install` (tests only)."""
    try:
        import peewee
    except Exception:
        return
    local_cls = getattr(peewee, "_ConnectionLocal", None)
    if local_cls is None or not getattr(local_cls, _MARKER, False):
        return
    for name in ("set_connection", "reset"):
        original = _originals.pop(name, None)
        if original is None:
            delattr(local_cls, name)
        else:
            setattr(local_cls, name, original)
    delattr(local_cls, _MARKER)
