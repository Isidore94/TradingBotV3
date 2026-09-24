"""The P&L repair CLI, on a journal shaped like the live 2026-06-12 CVNA spread."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_pnl_repair  # noqa: E402
from journal_store import JournalStore  # noqa: E402


def _flex_raw(exec_id, side, qty, price, strike, symbol):
    return json.dumps({
        "accountId": "U5102524", "assetCategory": "OPT", "buySell": side, "dateTime": "20260612;145118",
        "ibExecID": exec_id, "ibOrderID": "393387315", "multiplier": "100", "quantity": qty,
        "strike": strike, "symbol": symbol, "tradePrice": price, "putCall": "P",
        "expiry": "20260618", "underlyingSymbol": "CVNA", "currency": "USD",
    }, sort_keys=True)


def _socket_raw(exec_id, sec_type, symbol, side, price):
    return json.dumps({
        "commission": {}, "contract": {"currency": "USD", "exchange": "SMART", "secType": sec_type,
                                       "symbol": symbol},
        "execution": {"acctNumber": "U5102524", "execId": exec_id, "orderId": "-982", "price": price,
                      "shares": 1.0, "side": side, "time": "20260612  11:51:18"},
    }, sort_keys=True)


def _row(uid, source, symbol, sec_type, side, price, stamp, exec_id, raw, multiplier):
    return {
        "execution_uid": uid, "broker": "IBKR", "account_number": "U5102524",
        "account_label": "U5102524", "account_type": "", "symbol": symbol, "security_type": sec_type,
        "currency": "USD", "side": side, "quantity": 1.0, "price": price, "timestamp": stamp,
        "trade_date": stamp[:10], "commission": 0.7, "fees": 0.0, "gross_amount": None,
        "net_amount": None, "order_id": "-982", "exchange_exec_id": exec_id, "raw_json": raw,
        "source": source, "multiplier": multiplier,
    }


def _live_shaped_journal(tmp_path: Path) -> Path:
    """The stored rows exactly as the live journal holds them before the fix."""
    db = tmp_path / "trade_journal.sqlite3"
    store = JournalStore(db)
    store.initialize_schema()
    pacific = "2026-06-12T11:51:18-07:00"
    wrong_flex = "2026-06-12T14:51:18-07:00"
    store.upsert_executions([
        _row("IBKR:U5102524:0001640e.6a2b8893.01.01", "IBKR_SOCKET", "CVNA", "BAG", "BUY", 0.66, pacific,
             "0001640e.6a2b8893.01.01", _socket_raw("0001640e.6a2b8893.01.01", "BAG", "CVNA", "BOT", 0.66), 1.0),
        _row("IBKR:U5102524:00021ab9.6a2c031e.02.01.01", "IBKR_SOCKET", "CVNA  260618P00064500", "OPT",
             "BUY", 2.17, pacific, "00021ab9.6a2c031e.02.01.01",
             _socket_raw("00021ab9.6a2c031e.02.01.01", "OPT", "CVNA  260618P00064500", "BOT", 2.17), 100.0),
        _row("IBKR:U5102524:00021ab9.6a2c031e.03.01.01", "IBKR_SOCKET", "CVNA  260618P00063000", "OPT",
             "SELL", 1.51, pacific, "00021ab9.6a2c031e.03.01.01",
             _socket_raw("00021ab9.6a2c031e.03.01.01", "OPT", "CVNA  260618P00063000", "SLD", 1.51), 100.0),
        _row("IBKR:U5102524:00021ab9.6a2c031e.02.01", "IBKR_FLEX", "CVNA260618P00064500", "OPT", "BUY",
             2.17, wrong_flex, "00021ab9.6a2c031e.02.01",
             _flex_raw("00021ab9.6a2c031e.02.01", "BUY", "1", "2.17", "64.5", "CVNA  260618P00064500"), 100.0),
        _row("IBKR:U5102524:00021ab9.6a2c031e.03.01", "IBKR_FLEX", "CVNA260618P00063000", "OPT", "SELL",
             1.51, wrong_flex, "00021ab9.6a2c031e.03.01",
             _flex_raw("00021ab9.6a2c031e.03.01", "SELL", "-1", "1.51", "63", "CVNA  260618P00063000"), 100.0),
    ])
    store.rebuild_trades(refresh_tags=False)
    return db


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _uids(db: Path) -> set[str]:
    with JournalStore(db).connection() as conn:
        return {row[0] for row in conn.execute("SELECT execution_uid FROM raw_executions")}


def test_the_default_run_is_a_dry_run_and_writes_nothing(tmp_path, capsys):
    db = _live_shaped_journal(tmp_path)
    before = _sha(db)

    assert journal_pnl_repair.main(["--db", str(db)]) == 0

    assert _sha(db) == before
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "Flex fills re-timed to New York time: 2" in out
    assert "Socket duplicates dropped (Flex kept): 2" in out
    assert "Combo (BAG) parent rows dropped: 1" in out
    assert "Nothing was written" in out


def test_apply_backs_up_then_repairs_and_a_second_run_changes_nothing(tmp_path, capsys):
    db = _live_shaped_journal(tmp_path)
    before_trades = JournalStore(db).list_trades()
    # Each leg counted twice, plus a fake BAG position.
    assert sorted((t["symbol"], t["quantity_opened"]) for t in before_trades) == [
        ("CVNA", 1.0), ("CVNA260618P00063000", 2.0), ("CVNA260618P00064500", 2.0),
    ]

    assert journal_pnl_repair.main(["--db", str(db), "--apply"]) == 0

    backups = list(tmp_path.glob("*.pre-pnl-repair-*.bak"))
    assert len(backups) == 1
    assert _uids(db) == {"IBKR:U5102524:00021ab9.6a2c031e.02.01", "IBKR:U5102524:00021ab9.6a2c031e.03.01"}
    with JournalStore(db).connection() as conn:
        stamps = [row[0] for row in conn.execute("SELECT timestamp FROM raw_executions")]
    assert all(datetime.fromisoformat(s).utcoffset().total_seconds() == -4 * 3600 for s in stamps)
    trades = JournalStore(db).list_trades()
    assert sorted((t["symbol"], t["status"], t["quantity_opened"]) for t in trades) == [
        ("CVNA260618P00063000", "OPEN", 1.0),
        ("CVNA260618P00064500", "OPEN", 1.0),
    ]

    capsys.readouterr()
    assert journal_pnl_repair.main(["--db", str(db), "--apply"]) == 0
    out = capsys.readouterr().out
    assert "Flex fills re-timed to New York time: 0" in out
    assert "Socket duplicates dropped (Flex kept): 0" in out


def test_a_bag_row_with_no_legs_is_kept_and_named(tmp_path):
    db = _live_shaped_journal(tmp_path)
    with JournalStore(db).connection() as conn:
        conn.execute("DELETE FROM raw_executions WHERE security_type = 'OPT'")
    report = journal_pnl_repair._repair_file(db)

    assert report["bag_parents_dropped"] == []
    assert report["bag_parents_kept"][0]["execution_uid"] == "IBKR:U5102524:0001640e.6a2b8893.01.01"


def test_apply_refuses_the_live_folder_without_the_traders_flag(tmp_path, monkeypatch, capsys):
    db = _live_shaped_journal(tmp_path)
    import journal_reclassify

    monkeypatch.setattr(journal_reclassify, "_is_live_store", lambda _path: True)
    before = _sha(db)

    assert journal_pnl_repair.main(["--db", str(db), "--apply"]) == journal_pnl_repair.EXIT_REFUSED_TO_START
    assert _sha(db) == before


def test_two_backups_in_the_same_second_never_overwrite(tmp_path):
    db = tmp_path / "trade_journal.sqlite3"
    first = journal_pnl_repair._backup_path(db)
    first.write_bytes(b"pre-repair")
    second = journal_pnl_repair._backup_path(db)
    assert second != first
    assert first.read_bytes() == b"pre-repair"
