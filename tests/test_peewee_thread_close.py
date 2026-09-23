"""A peewee thread's SQLite connection is closed when that thread exits.

2026-09-23: the desk's GUI-thread young gc sweeps froze the UI for 20-44 s per
10 minutes, in bursts that matched the strength-board refresh. yfinance's
threaded download opens a peewee SQLite connection per ticker thread (tz and
cookie caches); a sqlite3.Connection is in a cycle with its statement cache,
so only the GUI-thread collector freed them, and each close released the GIL
several times while the busy download thread held it. These tests pin that a
finished thread leaves no connection for the collector.
"""

from __future__ import annotations

import gc
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

peewee = pytest.importorskip("peewee")

import peewee_thread_close  # noqa: E402


@pytest.fixture
def installed():
    assert peewee_thread_close.install() is True
    try:
        yield
    finally:
        peewee_thread_close.uninstall()


def _live_connections() -> int:
    return sum(1 for obj in gc.get_objects() if isinstance(obj, sqlite3.Connection))


def _db(tmp_path):
    db = peewee.SqliteDatabase(str(tmp_path / "cache.db"), pragmas={"journal_mode": "wal"})

    class Row(peewee.Model):
        key = peewee.CharField(primary_key=True)
        value = peewee.CharField()

        class Meta:
            database = db

    db.connect()
    db.create_tables([Row])
    Row.create(key="AAPL", value="America/New_York")
    return db, Row


def test_finished_threads_leave_no_connection_for_the_gui_collector(tmp_path, installed):
    db, Row = _db(tmp_path)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        gc.collect()
        before = _live_connections()
        seen = []

        def lookup():
            seen.append(Row.get(Row.key == "AAPL").value)

        threads = [threading.Thread(target=lookup) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert seen == ["America/New_York"] * 20, "the lookups themselves are unchanged"
        assert _live_connections() - before == 0, (
            "each exiting thread closes its own connection; none waits for the GUI gc sweep"
        )
    finally:
        if was_enabled:
            gc.enable()
        db.close()


def test_the_owning_thread_can_still_close_and_reconnect(tmp_path, installed):
    db, Row = _db(tmp_path)
    assert db.close() is True
    db.connect()
    Row.create(key="MSFT", value="America/New_York")
    assert Row.get(Row.key == "MSFT").value == "America/New_York"
    db.close()
    assert db.is_closed()


def test_install_is_idempotent_and_uninstall_restores_peewee(installed):
    assert peewee_thread_close.install() is True
    assert peewee._ConnectionLocal.set_connection is not peewee._ConnectionState.set_connection
    peewee_thread_close.uninstall()
    assert peewee._ConnectionLocal.set_connection is peewee._ConnectionState.set_connection
    assert peewee._ConnectionLocal.reset is peewee._ConnectionState.reset
    peewee_thread_close.install()  # the fixture's uninstall runs next
