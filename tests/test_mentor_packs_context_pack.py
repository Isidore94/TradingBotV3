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
