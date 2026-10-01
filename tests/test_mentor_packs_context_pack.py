"""Trade Mentor context pack: unique ids, tz-aware clock, unknown on a failed source."""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import context_pack  # noqa: E402

NOW = datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc)


def test_fixture_ids_are_unique_and_every_row_has_text():
    pack = context_pack.fixture()
    assert len(pack.ids) == len(set(pack.ids)) == len(pack.rows)
    assert all(str(row.get("text") or "").strip() for row in pack.rows)
    text = pack.as_text()
    assert "Auto mode: DESK" in text and "NVDA" in text and "Nonfarm payrolls" in text


def test_the_clock_is_tz_aware_and_the_pack_stamp_is_utc():
    pack = context_pack.fixture()
    clock = next(row for row in pack.rows if row["id"] == "ctx:clock")
    assert datetime.fromisoformat(clock["at_utc"]).tzinfo is not None
    assert "07:00 PT (10:00 ET)" in clock["text"]
    assert datetime.fromisoformat(pack.built_utc).tzinfo is not None


def test_a_failing_source_is_unknown_never_a_guess():
    broken = replace(
        context_pack.fixture_sources(),
        auto_mode=lambda: (_ for _ in ()).throw(OSError("gone")),
        open_positions=lambda: (_ for _ in ()).throw(RuntimeError("locked")),
    )
    pack = context_pack.build(now=NOW, sources=broken)
    rows = {row["id"]: row for row in pack.rows}
    assert rows["ctx:auto_mode"]["kind"] == "unknown" and "unknown" in rows["ctx:auto_mode"]["text"]
    assert rows["ctx:positions"]["kind"] == "unknown"
    assert rows["ctx:d1_env"]["kind"] == "d1_env", "one broken source must not blank the others"


def test_position_ids_are_the_journal_trade_id_not_the_row_number():
    two = replace(
        context_pack.fixture_sources(),
        open_positions=lambda: [
            {"trade_id": "T-77", "symbol": "AMD", "direction": "LONG", "quantity_opened": 10,
             "quantity_closed": 0, "average_entry_price": 150, "opened_at": "2026-09-29T06:40:00-07:00"},
            {"trade_id": "T-42", "symbol": "NVDA", "direction": "SHORT", "quantity_opened": 100,
             "quantity_closed": 0, "average_entry_price": 120.5, "opened_at": "2026-09-29T07:05:00-07:00"},
        ],
    )
    ids = [row["id"] for row in context_pack.build(now=NOW, sources=two).rows if row["kind"] == "position"]
    assert ids == ["ctx:pos:T-77", "ctx:pos:T-42"]
    # The same trade keeps its id when an earlier one closes.
    one = replace(two, open_positions=lambda: two.open_positions()[1:])
    ids = [row["id"] for row in context_pack.build(now=NOW, sources=one).rows if row["kind"] == "position"]
    assert ids == ["ctx:pos:T-42"]


def test_econ_events_past_seven_days_are_left_out():
    far = replace(
        context_pack.fixture_sources(),
        econ=lambda session: {"today": [], "week": [{"id": "w9", "date": "2026-10-20", "time_et": "", "label": "CPI"}]},
    )
    text = context_pack.build(now=NOW, sources=far).as_text()
    assert "CPI" not in text and "[ctx:econ]" in text


def test_no_positions_and_no_regime_read_as_none():
    empty = replace(context_pack.fixture_sources(), open_positions=lambda: [], regime_rows=lambda: [])
    text = context_pack.build(now=NOW, sources=empty).as_text()
    assert "Open journal positions: none" in text and "none typed yet" in text


def test_the_live_journal_read_is_read_only(tmp_path, monkeypatch):
    import project_paths

    db = tmp_path / "trade_journal.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE trades (trade_id TEXT, symbol TEXT, direction TEXT, status TEXT, quantity_opened REAL, "
        "quantity_closed REAL, average_entry_price REAL, opened_at TEXT, account_label TEXT)"
    )
    conn.execute("INSERT INTO trades VALUES ('T-1','AMD','SHORT','OPEN',50,0,150.0,'2026-09-29T07:00','TFSA')")
    conn.commit()
    conn.close()
    before = db.stat().st_mtime_ns
    monkeypatch.setattr(project_paths, "JOURNAL_DB_FILE", db)
    rows = context_pack._live_open_positions()
    assert rows and rows[0]["symbol"] == "AMD" and rows[0]["trade_id"] == "T-1"
    assert db.stat().st_mtime_ns == before
    assert not (tmp_path / "trade_journal.sqlite3-wal").exists()


def test_the_live_focus_read_never_constructs_the_focus_store():
    source = (SCRIPTS_DIR / "mentor_packs" / "context_pack.py").read_text(encoding="utf-8")
    # FocusPickStore() expires and rewrites the m5 lists on construction; JournalStore() migrates.
    assert "FocusPickStore(" not in source and "JournalStore(" not in source


def test_the_context_pack_carries_today_so_far():
    rows = {row["id"]: row for row in context_pack.fixture().rows}
    assert rows["ctx:today"]["text"].startswith("Today so far: 2 closed trade(s)")
    broken = replace(context_pack.fixture_sources(), today=lambda moment: (_ for _ in ()).throw(OSError("locked")))
    rows = {row["id"]: row for row in context_pack.build(now=NOW, sources=broken).rows}
    assert rows["ctx:today"]["kind"] == "unknown"


def test_the_live_today_row_reads_the_journal_pack(tmp_path, monkeypatch):
    import project_paths
    from mentor_packs import journal_pack

    db = journal_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")
    monkeypatch.setattr(project_paths, "JOURNAL_DB_FILE", db)
    assert context_pack.live_sources().today(journal_pack.FIXTURE_NOW) == (
        "Today so far: 3 closed trade(s), 2 win(s), net +30.00 $, 1 open")


# ---------------------------------------------------------------- 2026-10-01: stale journal opens (book ghosts)
GHOST_REPORT = {
    "checked_at": "2026-09-28T22:04:00",
    "brokers": ["IBKR", "QUESTRADE"],
    "agreed": [{"broker": "IBKR", "account_number": "U1", "symbol": "DRAM", "trade_ids": ["T-DRAM"]}],
    "mismatched": [
        {"kind": "JOURNAL_OPEN_BROKER_FLAT", "broker": "QUESTRADE", "account_number": "293", "symbol": "NVDA",
         "journal_quantity": -100, "broker_quantity": 0, "trade_ids": ["T-NVDA-OLD"]},
        {"kind": "JOURNAL_OPEN_BROKER_FLAT", "broker": "QUESTRADE", "account_number": "293", "symbol": "AAL",
         "journal_quantity": 50, "broker_quantity": 0, "trade_ids": ["gone-after-rebuild"]},
        {"kind": "BROKER_OPEN_JOURNAL_FLAT", "broker": "QUESTRADE", "account_number": "518", "symbol": "QTUM",
         "journal_quantity": 0, "broker_quantity": 20, "trade_ids": []},
    ],
}
GHOST_TRADES = [
    {"trade_id": "T-NVDA-OLD", "broker": "QUESTRADE", "account_number": "293", "symbol": "NVDA", "direction": "SHORT",
     "quantity_opened": 100, "quantity_closed": 0, "average_entry_price": 120.5, "opened_at": "2026-06-10T07:05:00-07:00"},
    {"trade_id": "T-AAL", "broker": "QUESTRADE", "account_number": "293", "symbol": "AAL", "direction": "LONG",
     "quantity_opened": 50, "quantity_closed": 0, "average_entry_price": 12.0, "opened_at": "2026-07-01T07:05:00-07:00"},
    {"trade_id": "T-DRAM", "broker": "IBKR", "account_number": "U1", "symbol": "DRAM", "direction": "LONG",
     "quantity_opened": 300, "quantity_closed": 0, "average_entry_price": 30.0, "opened_at": "2026-09-20T07:05:00-07:00"},
    # Opened after the check: never hidden by an older "broker flat" for the same name.
    {"trade_id": "T-NVDA-NEW", "broker": "QUESTRADE", "account_number": "293", "symbol": "NVDA", "direction": "LONG",
     "quantity_opened": 10, "quantity_closed": 0, "average_entry_price": 180.0, "opened_at": "2026-09-29T06:40:00-07:00"},
]


def _ghost_rows(report):
    src = replace(context_pack.fixture_sources(), open_positions=lambda: GHOST_TRADES,
                  reconciliation=lambda: report)
    return context_pack.build(now=NOW, sources=src).rows


def test_journal_opens_the_broker_reports_flat_are_not_positions():
    rows = _ghost_rows(GHOST_REPORT)
    positions = {row["id"]: row for row in rows if row["kind"] == "position"}
    assert set(positions) == {"ctx:pos:T-DRAM", "ctx:pos:T-NVDA-NEW", "ctx:pos:broker:518:QTUM"}
    stale = next(row for row in rows if row["id"] == "ctx:positions:stale")
    assert stale["kind"] == "positions_stale" and stale["symbols"] == ["NVDA", "AAL"]
    assert stale["text"].startswith("Journal shows 2 stale open trade(s) the broker reports flat as of ")
    assert stale["text"].endswith(": NVDA, AAL - not positions")
    qtum = positions["ctx:pos:broker:518:QTUM"]
    assert qtum["symbol"] == "QTUM" and qtum["side"] == "LONG" and "(broker only, not in journal" in qtum["text"]
    assert "broker agreed as of" in positions["ctx:pos:T-DRAM"]["text"]
    assert "journal (not checked against broker)" in positions["ctx:pos:T-NVDA-NEW"]["text"]


def test_no_reconciliation_keeps_every_open_trade_labelled_unchecked():
    for report in (None, {}):
        rows = _ghost_rows(report)
        positions = [row for row in rows if row["kind"] == "position"]
        assert [row["id"] for row in positions] == [f"ctx:pos:{t['trade_id']}" for t in GHOST_TRADES]
        assert all(row["text"].endswith("; source: journal (not checked against broker)") for row in positions)
        assert not any(row["id"] == "ctx:positions:stale" for row in rows)


def test_an_unreadable_reconciliation_never_breaks_the_positions():
    rows = _ghost_rows(None)
    broken = replace(context_pack.fixture_sources(), open_positions=lambda: GHOST_TRADES,
                     reconciliation=lambda: (_ for _ in ()).throw(OSError("locked")))
    got = context_pack.build(now=NOW, sources=broken).rows
    assert [r["text"] for r in got if r["kind"] == "position"] == [r["text"] for r in rows if r["kind"] == "position"]


def test_today_line_says_how_many_journal_opens_are_stale():
    rows = {row["id"]: row for row in _ghost_rows(GHOST_REPORT)}
    assert rows["ctx:today"]["text"].endswith("; 2 of the journal's open trades are stale (broker flat)")


def test_the_live_reconciliation_read_is_read_only(tmp_path, monkeypatch):
    import json

    import project_paths

    db = tmp_path / "trade_journal.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO meta VALUES ('last_reconciliation', ?)", (json.dumps(GHOST_REPORT),))
    conn.commit()
    conn.close()
    before = db.stat().st_mtime_ns
    monkeypatch.setattr(project_paths, "JOURNAL_DB_FILE", db)
    assert context_pack.live_sources().reconciliation() == GHOST_REPORT
    assert db.stat().st_mtime_ns == before
    monkeypatch.setattr(project_paths, "JOURNAL_DB_FILE", tmp_path / "missing.sqlite3")
    assert context_pack.live_sources().reconciliation() is None
