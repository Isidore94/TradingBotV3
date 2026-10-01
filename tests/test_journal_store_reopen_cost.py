"""Reopening an up-to-date journal never rescans every execution.

2026-09-30: every `JournalStore()` ran the v2->v3 data passes (SELECT * over
raw_executions, twice) on the Qt thread; the Mentor prompt alone cost the desk
40 s at 07:32 and 26 s at 12:00. The schema is still checked on every open.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_migrate  # noqa: E402
from journal_store import JournalStore  # noqa: E402


def _count_scans(monkeypatch):
    calls: list[str] = []
    for name in ("_collapse_execution_uids", "_backfill_sources_and_multipliers"):
        real = getattr(journal_migrate, name)

        def spy(conn, report, _real=real, _name=name):
            calls.append(_name)
            return _real(conn, report)

        monkeypatch.setattr(journal_migrate, name, spy)
    return calls


def test_a_current_journal_reopens_without_the_execution_scans(tmp_path, monkeypatch):
    path = tmp_path / "trade_journal.sqlite3"
    JournalStore(path)  # first open: creates and stamps the schema
    calls = _count_scans(monkeypatch)
    store = JournalStore(path)
    assert calls == []
    assert store.last_migration is None


def test_an_old_journal_still_gets_the_full_migration(tmp_path, monkeypatch):
    path = tmp_path / "trade_journal.sqlite3"
    first = JournalStore(path)
    with first.connection() as conn:
        conn.execute("UPDATE meta SET value = '2' WHERE key = 'schema_version'")
    calls = _count_scans(monkeypatch)
    store = JournalStore(path)
    assert calls == ["_collapse_execution_uids", "_backfill_sources_and_multipliers"]
    assert store.last_migration is not None
    with store.connection() as conn:
        assert journal_migrate.read_schema_version(conn) == 3


def test_the_schema_is_still_checked_on_every_open(tmp_path):
    path = tmp_path / "trade_journal.sqlite3"
    first = JournalStore(path)
    with first.connection() as conn:
        conn.execute("DROP TABLE IF EXISTS trade_annotations")
    JournalStore(path)
    with first.connection() as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "trade_annotations" in names
