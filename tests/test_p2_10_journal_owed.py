"""P2-10: the journal's owed items (made-up entries out of totals, one store owner)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SESSION = "2026-09-22"

# One real trade, one made-up entry (SYNTHETIC_OPEN), one TFSA stock short.
# Before P2-10 the totals were 100 - 40 + 341.57 + 25 = 426.57; after, only 60.00.
REAL = {"trade_id": "r1", "symbol": "ABC", "status": "CLOSED", "trade_date": SESSION,
        "tag_status": "confirmed", "setup_tags": "x", "net_pnl": 100.0, "net_pnl_cad": 100.0}
REAL_LOSS = {"trade_id": "r2", "symbol": "DEF", "status": "CLOSED", "trade_date": SESSION,
             "tag_status": "confirmed", "setup_tags": "x", "net_pnl": -40.0, "net_pnl_cad": -40.0}
MADE_UP = {"trade_id": "m1", "symbol": "SMH", "status": "CLOSED", "trade_date": SESSION,
           "tag_status": "confirmed", "setup_tags": "x", "net_pnl": 341.57,
           "net_pnl_cad": 341.57, "synthetic_entry": True}
TFSA_SHORT = {"trade_id": "m2", "symbol": "CRDO", "status": "CLOSED", "trade_date": SESSION,
              "tag_status": "confirmed", "setup_tags": "x", "net_pnl": 25.0, "net_pnl_cad": 25.0,
              "direction": "SHORT", "security_type": "STK", "account_tax_status": "TAX_FREE"}
ROWS = [REAL, REAL_LOSS, MADE_UP, TFSA_SHORT]


def _rows():
    return [dict(row) for row in ROWS]


def test_day_review_glance_keeps_made_up_trades_but_leaves_them_out_of_pnl():
    import day_report_card

    glance = day_report_card.glance({"trades": _rows()})

    assert glance["trades"] == 4  # kept
    assert glance["pnl"] == 60.0  # before: 426.57
    assert glance["pnl_counted"] == 2
    assert glance["wins"] == 1 and glance["losses"] == 1
    assert glance["pnl_not_counted"] == 2


def test_day_review_sparkline_leaves_made_up_trades_out():
    from ui.services.day_review_service import _pnl_by_session

    assert _pnl_by_session((SESSION,), _rows()) == ((SESSION, 60.0),)
    only_made_up = [dict(MADE_UP)]
    assert _pnl_by_session((SESSION,), only_made_up) == ((SESSION, None),)


def test_weekend_prep_week_line_leaves_made_up_trades_out():
    import weekend_verdict

    line = weekend_verdict.journal_week_line(_rows())

    assert line.n == 2
    assert "+60.00" in line.text


def test_recap_day_record_total_leaves_made_up_trades_out_but_keeps_the_rows():
    import day_session_record

    section = day_session_record._trades({"payload": {"trades": _rows()}}, [])

    assert section["n"] == 4
    assert len(section["rows"]) == 4
    assert section["net_pnl"] == 60.0  # before: 426.57
    made_up = next(row for row in section["rows"] if row["trade_id"] == "m1")
    assert made_up["net_pnl"] == 341.57  # the trade's own number stays visible
    assert made_up["counted_in_pnl"] is False


# ---------------------------------------------------------------------------
# step 2: the Mentor rule lane never reads the journal on the Qt thread
# ---------------------------------------------------------------------------
def _size_rule():
    return {"for_date": SESSION, "rule_id": "rc-1", "set_on": "2026-09-21",
            "text": "size down in chop", "tag": "size_down_in_chop", "streak": 1}


def _sized_trade(trade_id, day, qty, *, opened="09:50"):
    return {
        "trade_id": trade_id, "symbol": "ZETA", "direction": "LONG", "status": "CLOSED",
        "opened_at": f"{day}T{opened}:00-04:00", "closed_at": f"{day}T11:00:00-04:00",
        "trade_date": day, "net_pnl": 10.0, "average_entry_price": 10.0,
        "average_exit_price": 10.1, "quantity_opened": qty,
    }


class _CountingStore:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls = 0

    def list_trades(self, **_kwargs):
        self.calls += 1
        return list(self.rows)


def test_rule_lane_reads_no_journal_on_the_qt_thread():
    import types

    import pytest

    pytest.importorskip("PySide6")
    from ui.app import MainWindow

    store = _CountingStore()
    host = types.SimpleNamespace(
        rule_chip=types.SimpleNamespace(info=_size_rule), _regime_timeline=[],
    )
    MainWindow._mentor_rule_lane(host, store, SESSION, [_sized_trade("t1", SESSION, 500)])
    assert store.calls == 0  # before: one list_trades call on the Qt thread


def test_rule_lane_uses_the_worker_baseline_for_its_own_session_only():
    import types
    from datetime import datetime, timedelta, timezone

    import pytest

    pytest.importorskip("PySide6")
    from ui.app import MainWindow

    earlier = [_sized_trade(f"e{i}", f"2026-09-{10 + i:02d}", 100) for i in range(6)]
    store = _CountingStore(earlier)
    # The worker's read: one bounded journal query, the median entry notional.
    baseline = MainWindow._read_rule_size_baseline(SESSION, store=store)
    assert baseline == {"session": SESSION, "median": 1000.0}
    assert store.calls == 1

    chop = datetime(2026, 9, 22, 9, 35, tzinfo=timezone(timedelta(hours=-4)))
    host = types.SimpleNamespace(
        rule_chip=types.SimpleNamespace(info=_size_rule),
        _regime_timeline=[(chop, "neutral_chop")],
        _rule_size_baseline=baseline,
    )
    big = [_sized_trade("t1", SESSION, 500)]
    rows = MainWindow._mentor_rule_lane(host, None, SESSION, big)
    assert [row["trade_id"] for row in rows] == ["t1"]
    # A baseline for another session is not used: the size check says nothing.
    host._rule_size_baseline = {"session": "2026-09-21", "median": 1000.0}
    assert MainWindow._mentor_rule_lane(host, None, SESSION, big) == []
    # Another rule clears the baseline without starting a read.
    MainWindow._refresh_rule_size_baseline(host, {"tag": "hold_winners"})
    assert host._rule_size_baseline is None


# ---------------------------------------------------------------------------
# step 3: Questrade gap days are importable from a statement, through the CLI
# ---------------------------------------------------------------------------
GAP_ACCOUNT = "51830546"
STATEMENT_COLUMNS = [
    "Transaction Date", "Settlement Date", "Action", "Symbol", "Description", "Quantity",
    "Price", "Gross Amount", "Commission", "Net Amount", "Currency", "Account #",
    "Activity Type", "Account Type",
]


def _statement_row(day, action, qty, price, gross, net):
    return [f"{day} 12:00:00 AM", f"{day} 12:00:00 AM", action, "AAPL", "APPLE INC",
            qty, price, gross, "0.00", net, "USD", GAP_ACCOUNT, "Trades", "Individual margin"]


def _gap_statement(tmp_path):
    rows = [
        # 2026-06-10 is a FAILED gap day; 2026-06-12 is not.
        _statement_row("2026-06-10", "Buy", "10", "100", "-1000.00", "-1000.00"),
        _statement_row("2026-06-10", "Sell", "-10", "101", "1010.00", "1010.00"),
        _statement_row("2026-06-12", "Buy", "5", "50", "-250.00", "-250.00"),
    ]
    path = tmp_path / "statement.csv"
    lines = [",".join(f'"{value}"' for value in row) for row in [STATEMENT_COLUMNS, *rows]]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _gap_db(tmp_path):
    import journal_coverage
    import journal_questrade_gaps
    from journal_store import JournalStore

    db = tmp_path / "trade_journal.sqlite3"
    store = JournalStore(db)
    store.initialize_schema()
    for day in ("2026-06-10", "2026-06-11"):
        journal_coverage.mark_coverage(
            store, broker="QUESTRADE", account_number=GAP_ACCOUNT, day=day,
            status=journal_coverage.FAILED, source="QT_API",
            message=journal_questrade_gaps.EXECUTIONS_MISSING_REASON,
        )
    return db


def _statement_days(db):
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        return sorted(
            (row[0], row[1]) for row in conn.execute(
                "SELECT trade_date, COUNT(*) FROM raw_executions "
                "WHERE source = 'QT_STATEMENT' GROUP BY trade_date"
            )
        )
    finally:
        conn.close()


def _sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gap_statement_dry_run_names_the_gap_days_it_would_fill_and_writes_nothing(tmp_path, capsys):
    import journal_questrade_gaps

    db = _gap_db(tmp_path)
    statement = _gap_statement(tmp_path)
    before = _sha(db)

    code = journal_questrade_gaps.main(["--db", str(db), "--statement", str(statement)])

    assert code == journal_questrade_gaps.EXIT_OK
    assert _sha(db) == before
    out = capsys.readouterr().out
    assert "statement has trades for 1 of 2 FAILED account-day(s)" in out
    assert f"{GAP_ACCOUNT} 2026-06-10" in out
    assert "Nothing was written" in out


def test_gap_statement_apply_writes_only_the_gap_days_and_a_rerun_adds_nothing(tmp_path, capsys):
    import journal_questrade_gaps

    db = _gap_db(tmp_path)
    statement = _gap_statement(tmp_path)
    args = ["--db", str(db), "--statement", str(statement), "--apply"]

    assert journal_questrade_gaps.main(args) == journal_questrade_gaps.EXIT_OK
    assert _statement_days(db) == [("2026-06-10", 2)]  # 2026-06-12 is not a gap day
    assert list(tmp_path.glob("*.pre-qt-statement-*.bak"))
    failed = journal_questrade_gaps.gap_report(db)["failed_days"]
    assert [row["day"] for row in failed] == ["2026-06-11"]  # 06-10 is now COVERED
    assert "1 gap day(s) imported" in capsys.readouterr().out

    # The same file again: the deterministic uid collapses every row.
    assert journal_questrade_gaps.main(args) == journal_questrade_gaps.EXIT_OK
    assert _statement_days(db) == [("2026-06-10", 2)]


def test_gap_statement_apply_refuses_the_live_folder_without_the_traders_flag(tmp_path, monkeypatch):
    import journal_questrade_gaps
    import journal_reclassify

    db = _gap_db(tmp_path)
    statement = _gap_statement(tmp_path)
    monkeypatch.setattr(journal_reclassify, "_is_live_store", lambda _path: True)
    before = _sha(db)

    code = journal_questrade_gaps.main(["--db", str(db), "--statement", str(statement), "--apply"])

    assert code == journal_questrade_gaps.EXIT_REFUSED_TO_START
    assert _sha(db) == before


def test_gap_statement_apply_says_how_many_days_the_file_took_over_from_the_api(tmp_path, capsys):
    import journal_questrade_gaps
    from journal_store import JournalStore

    db = _gap_db(tmp_path)
    # The API saw only the buy on the gap day, so the day's money disagrees with the file.
    JournalStore(db).upsert_executions([{
        "execution_uid": f"QUESTRADE:{GAP_ACCOUNT}:api-1", "broker": "QUESTRADE",
        "account_number": GAP_ACCOUNT, "account_label": "Individual margin",
        "account_type": "Individual margin", "symbol": "AAPL", "security_type": "STK",
        "currency": "USD", "side": "BUY", "quantity": 10.0, "price": 100.0,
        "timestamp": "2026-06-10T09:45:00-04:00", "trade_date": "2026-06-10",
        "commission": 0.0, "fees": 0.0, "gross_amount": None, "net_amount": None,
        "order_id": "", "exchange_exec_id": "", "raw_json": "{}", "source": "QT_API",
    }])

    code = journal_questrade_gaps.main(
        ["--db", str(db), "--statement", str(_gap_statement(tmp_path)), "--apply"]
    )

    assert code == journal_questrade_gaps.EXIT_OK
    assert "1 taken over from QT_API rows by the file" in capsys.readouterr().out


def test_statement_import_only_days_leaves_other_days_alone(tmp_path):
    from datetime import date

    import journal_statement_import as statement
    from journal_store import JournalStore

    store = JournalStore(tmp_path / "journal.sqlite3")
    summary = statement.import_questrade_statement(
        store, _gap_statement(tmp_path), only_days={(GAP_ACCOUNT, date(2026, 6, 10))},
    )
    assert summary["days_written"] == 1
    assert summary["days_outside_scope"] == 1


# ---------------------------------------------------------------------------
# step 4: the half-exited status is read by the name the journal writes
# ---------------------------------------------------------------------------
def test_setup_evidence_reads_the_status_the_journal_writes_and_the_old_spellings():
    import journal_store
    import setup_environment_evidence as evidence

    written = journal_store.TRADE_STATUS_CLOSED_PARTIAL
    assert evidence._status_of({"status": written}) == evidence.STATUS_PARTLY_CLOSED  # before: open
    for old in ("PARTIAL", "PARTLY_CLOSED", "PARTIALLY_CLOSED", "closed_partial"):
        assert evidence._status_of({"status": old}) == evidence.STATUS_PARTLY_CLOSED
    assert evidence._status_of({"status": "CLOSED"}) == evidence.STATUS_COMPLETE
    assert evidence._status_of({"status": "OPEN"}) == evidence.STATUS_OPEN


def test_journal_store_names_every_half_exited_spelling_readers_accept():
    import journal_store

    assert journal_store.is_partly_closed("CLOSED_PARTIAL")
    assert journal_store.is_partly_closed(" partially_closed ")
    assert not journal_store.is_partly_closed("CLOSED")
    assert not journal_store.is_partly_closed(None)


# ---------------------------------------------------------------------------
# step 5: one owner for the journal feed's shared store
# ---------------------------------------------------------------------------
def test_two_threads_asking_at_once_open_exactly_one_store(tmp_path, monkeypatch):
    import threading
    import time

    import journal_store
    from ui.services import journal_feed

    opened = []

    class SlowStore:
        def __init__(self, path):
            opened.append(path)
            time.sleep(0.2)  # a migration in progress
            self.db_path = path

    monkeypatch.setattr(journal_store, "JournalStore", SlowStore)
    monkeypatch.setattr(journal_feed, "_STORE", None)
    monkeypatch.setattr(journal_feed, "journal_db_path", lambda: tmp_path / "j.sqlite3")

    got = []
    threads = [threading.Thread(target=lambda: got.append(journal_feed._store())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(opened) == 1  # before: two stores, two migrations at once
    assert got[0] is got[1]


def test_the_store_owner_opens_installs_and_resets(tmp_path, monkeypatch):
    import journal_store
    from ui.services import journal_feed

    class FakeStore:
        def __init__(self, path):
            self.db_path = path

    monkeypatch.setattr(journal_store, "JournalStore", FakeStore)
    monkeypatch.setattr(journal_feed, "_STORE", None)
    monkeypatch.setattr(journal_feed, "journal_db_path", lambda: tmp_path / "j.sqlite3")
    owner = journal_feed.STORE_OWNER

    assert owner.current() is None and not journal_feed.store_is_initialized()
    first = owner.open()
    assert owner.open() is first is journal_feed._store()
    assert journal_feed.store_is_initialized()

    replacement = FakeStore(tmp_path / "other.sqlite3")
    owner.install(replacement)
    assert journal_feed._store() is replacement

    owner.reset()
    assert owner.current() is None and journal_feed._STORE is None
