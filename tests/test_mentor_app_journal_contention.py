"""Three writer processes on the journal now (desk, mentor app, night): WAL + busy_timeout,
and the two Market Journal appenders share one machine-local lock."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from journal_store import JournalStore, persisted_journal_schema_version  # noqa: E402

ROWS_PER_PROCESS = 40


def test_the_journal_runs_in_wal_with_a_busy_timeout(tmp_path):
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    conn = store.connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert int(conn.execute("PRAGMA busy_timeout").fetchone()[0]) == 5000
    finally:
        conn.close()


def test_a_wal_journal_still_opens_read_only(tmp_path):
    db = tmp_path / "trade_journal.sqlite3"
    store = JournalStore(db)
    held = store.connect()  # a writer keeps the WAL open, as the desk does
    try:
        assert persisted_journal_schema_version(db) is not None
        ro = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        try:
            assert ro.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
        finally:
            ro.close()
    finally:
        held.close()


_JOURNAL_WRITER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {scripts!r})
    from journal_store import JournalStore
    store = JournalStore({db!r})
    for n in range({rows}):
        store.record_opportunity_event(
            opportunity_id=f"{{sys.argv[1]}}-{{n}}", event_type="MENTOR_ASKED",
            trade_id=f"T-{{sys.argv[1]}}-{{n}}", payload={{"n": n}}, source="contention-test",
        )
    """
)


def test_two_processes_append_journal_events_without_losing_one(tmp_path):
    db = tmp_path / "trade_journal.sqlite3"
    JournalStore(db)  # schema first, so neither child migrates
    script = tmp_path / "writer.py"
    script.write_text(_JOURNAL_WRITER.format(scripts=str(SCRIPTS_DIR), db=str(db), rows=ROWS_PER_PROCESS))
    children = [subprocess.Popen([sys.executable, str(script), name]) for name in ("desk", "app")]
    assert [child.wait(120) for child in children] == [0, 0]
    rows = JournalStore(db).list_opportunity_events(event_type="MENTOR_ASKED", limit=10_000)
    assert len(rows) == 2 * ROWS_PER_PROCESS
    assert len({row["opportunity_id"] for row in rows}) == 2 * ROWS_PER_PROCESS


def test_the_market_journal_append_runs_inside_the_machine_local_lock(monkeypatch, tmp_path):
    import market_journal
    from evidence_ledger import EvidenceLedger
    from ui.services import market_journal_service as mjs

    held: list[str] = []
    seen_inside: list[bool] = []

    @contextmanager
    def spy_lock(key, **kwargs):
        held.append(key)
        yield
        held.pop()

    ledger = EvidenceLedger(
        stream=market_journal.STREAM, schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY, directory=tmp_path
    )
    real_append = ledger.append

    def append(*args, **kwargs):
        seen_inside.append(bool(held))
        return real_append(*args, **kwargs)

    monkeypatch.setattr(ledger, "append", append)
    monkeypatch.setattr(mjs, "local_writer_lock", spy_lock, raising=False)
    service = mjs.MarketJournalService()
    service._ledger = ledger
    result = service.write_entry(text="Tape is weak; fading pops.", session_date="2026-09-29")
    assert result["ok"] is True
    assert seen_inside == [True], "the append must hold the market-journal lock"
    from local_writer_lock import lock_key_for_path

    assert mjs.market_journal_lock_key(ledger) == lock_key_for_path(Path(tmp_path) / "market_journal.jsonl")


_MARKET_WRITER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {scripts!r})
    import market_journal
    from evidence_ledger import EvidenceLedger
    from ui.services.market_journal_service import MarketJournalService
    service = MarketJournalService()
    service._ledger = EvidenceLedger(
        stream=market_journal.STREAM, schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY, directory={folder!r}
    )
    for n in range({rows}):
        out = service.write_entry(text=f"{{sys.argv[1]}} read number {{n}} " + "x" * 400, session_date="2026-09-29")
        assert out["ok"], out
    """
)


def test_two_processes_append_market_journal_rows_whole(tmp_path):
    import market_journal
    from evidence_ledger import EvidenceLedger

    script = tmp_path / "market_writer.py"
    script.write_text(_MARKET_WRITER.format(scripts=str(SCRIPTS_DIR), folder=str(tmp_path / "ledger"), rows=ROWS_PER_PROCESS))
    children = [subprocess.Popen([sys.executable, str(script), name]) for name in ("desk", "mentor")]
    assert [child.wait(120) for child in children] == [0, 0]
    ledger = EvidenceLedger(
        stream=market_journal.STREAM, schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY, directory=tmp_path / "ledger"
    )
    result = ledger.read()
    assert result.unreadable == 0, "no torn row"
    assert len(result.rows) == 2 * ROWS_PER_PROCESS, "no lost row"
