"""R7 §9 step 2 - the schema v2 -> v3 migration and its rehearsal CLI.

The migration deletes rows. Everything here exists because of that sentence:
the backup has to be taken before anything destructive runs, the dry run has to
leave the live file byte-identical, a uid it cannot read confidently has to be
left alone and reported, and re-running has to be a no-op.

The v2 fixture is built by writing v2 rows directly, not by calling the current
importers - the importers now emit v3 uids, so using them would test the
migration against input it will never see.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_migrate as jm  # noqa: E402
from journal_characterization import CHARACTERIZATION_EXECUTIONS  # noqa: E402
from journal_importers import QuestradeImporter  # noqa: E402
from journal_store import JOURNAL_SCHEMA_VERSION, JournalStore  # noqa: E402

V2_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE accounts (
    broker TEXT NOT NULL, account_number TEXT NOT NULL, account_label TEXT NOT NULL DEFAULT '',
    account_type TEXT NOT NULL DEFAULT '', currency TEXT NOT NULL DEFAULT '',
    raw_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL,
    PRIMARY KEY (broker, account_number));
CREATE TABLE import_runs (
    import_run_id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, status TEXT NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT, imported_executions INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '');
CREATE TABLE raw_executions (
    execution_uid TEXT PRIMARY KEY, broker TEXT NOT NULL, account_number TEXT NOT NULL,
    account_label TEXT NOT NULL DEFAULT '', account_type TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL, security_type TEXT NOT NULL DEFAULT '', currency TEXT NOT NULL DEFAULT 'USD',
    side TEXT NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL, timestamp TEXT NOT NULL,
    trade_date TEXT NOT NULL, commission REAL NOT NULL DEFAULT 0, fees REAL NOT NULL DEFAULT 0,
    gross_amount REAL, net_amount REAL, order_id TEXT NOT NULL DEFAULT '',
    exchange_exec_id TEXT NOT NULL DEFAULT '', raw_json TEXT NOT NULL DEFAULT '{}',
    imported_at TEXT NOT NULL);
CREATE TABLE trades (
    trade_id TEXT PRIMARY KEY, broker TEXT NOT NULL, account_number TEXT NOT NULL,
    account_label TEXT NOT NULL DEFAULT '', symbol TEXT NOT NULL, security_type TEXT NOT NULL DEFAULT '',
    currency TEXT NOT NULL DEFAULT 'USD', direction TEXT NOT NULL, status TEXT NOT NULL,
    opened_at TEXT NOT NULL, closed_at TEXT NOT NULL DEFAULT '', trade_date TEXT NOT NULL,
    quantity_opened REAL NOT NULL DEFAULT 0, quantity_closed REAL NOT NULL DEFAULT 0,
    average_entry_price REAL NOT NULL DEFAULT 0, average_exit_price REAL NOT NULL DEFAULT 0,
    gross_pnl REAL NOT NULL DEFAULT 0, commission REAL NOT NULL DEFAULT 0, fees REAL NOT NULL DEFAULT 0,
    net_pnl REAL NOT NULL DEFAULT 0, pnl_usd REAL, auto_tag_summary TEXT NOT NULL DEFAULT '',
    tag_confidence REAL, updated_at TEXT NOT NULL);
CREATE TABLE trade_legs (
    leg_id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id TEXT NOT NULL, execution_uid TEXT NOT NULL,
    side TEXT NOT NULL, role TEXT NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL,
    timestamp TEXT NOT NULL, commission REAL NOT NULL DEFAULT 0, fees REAL NOT NULL DEFAULT 0);
CREATE TABLE trade_annotations (
    trade_id TEXT PRIMARY KEY, setup_tags TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL);
CREATE TABLE auto_tag_candidates (
    trade_id TEXT NOT NULL, tag TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL DEFAULT '',
    rationale TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, PRIMARY KEY (trade_id, tag));
CREATE TABLE tag_corrections (
    correction_id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, setup_tag TEXT NOT NULL,
    source_trade_id TEXT NOT NULL DEFAULT '', confidence_boost REAL NOT NULL DEFAULT 0.12,
    created_at TEXT NOT NULL);
CREATE TABLE regimes (
    trade_date TEXT PRIMARY KEY, mid_term_regime TEXT NOT NULL DEFAULT '',
    short_term_regime TEXT NOT NULL DEFAULT '', intraday_regime TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL);
CREATE TABLE opportunity_events (
    event_id TEXT PRIMARY KEY, opportunity_id TEXT NOT NULL, lifecycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL DEFAULT '', side TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL, trade_id TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}', source TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
"""

_EXEC_COLUMNS = (
    "execution_uid broker account_number account_label account_type symbol security_type currency "
    "side quantity price timestamp trade_date commission fees gross_amount net_amount order_id "
    "exchange_exec_id raw_json imported_at"
).split()


def _write_v2(
    path: Path, executions, *, accounts=(), annotations=(), trades=(), legs=()
) -> Path:
    """A real v2 database: v2 DDL, v2 rows, and the v2 version marker."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(V2_SCHEMA)
        conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', '2')")
        for row in executions:
            filled = {key: row.get(key) for key in _EXEC_COLUMNS}
            filled.setdefault("imported_at", "2026-08-05T20:00:00")
            filled["imported_at"] = row.get("imported_at") or "2026-08-05T20:00:00"
            conn.execute(
                f"INSERT INTO raw_executions({', '.join(_EXEC_COLUMNS)}) "
                f"VALUES({', '.join('?' * len(_EXEC_COLUMNS))})",
                [filled[key] for key in _EXEC_COLUMNS],
            )
        for account in accounts:
            conn.execute(
                "INSERT INTO accounts(broker, account_number, account_label, account_type, currency, "
                "raw_json, updated_at) VALUES(?, ?, ?, ?, ?, '{}', '2026-08-05T20:00:00')",
                account,
            )
        for trade in trades:
            conn.execute(
                "INSERT INTO trades(trade_id, broker, account_number, account_label, symbol, "
                "security_type, currency, direction, status, opened_at, closed_at, trade_date, "
                "quantity_opened, quantity_closed, average_entry_price, average_exit_price, "
                "gross_pnl, commission, fees, net_pnl, pnl_usd, auto_tag_summary, tag_confidence, "
                "updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, '', NULL, '2026-08-05T20:00:00')",
                trade,
            )
        for leg in legs:
            conn.execute(
                "INSERT INTO trade_legs(trade_id, execution_uid, side, role, quantity, price, "
                "timestamp, commission, fees) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                leg,
            )
        for trade_id, tags in annotations:
            conn.execute(
                "INSERT INTO trade_annotations(trade_id, setup_tags, notes, updated_at) "
                "VALUES(?, ?, '', '2026-08-05T20:00:00')",
                (trade_id, tags),
            )
        conn.commit()
    finally:
        conn.close()
    return path


def _socket_row(exec_id: str, timestamp: str, **overrides):
    row = {
        "execution_uid": f"IBKR:U1234567:{exec_id}:NVDA:{timestamp}",
        "broker": "IBKR",
        "account_number": "U1234567",
        "account_label": "U1234567",
        "account_type": "",
        "symbol": "NVDA",
        "security_type": "STK",
        "currency": "USD",
        "side": "BUY",
        "quantity": 10.0,
        "price": 100.0,
        "timestamp": timestamp,
        "trade_date": timestamp[:10],
        "commission": 1.0,
        "fees": 0.0,
        "gross_amount": None,
        "net_amount": None,
        "order_id": "",
        "exchange_exec_id": exec_id,
        "raw_json": json.dumps({"contract": {"symbol": "NVDA"}, "execution": {"execId": exec_id}}),
        "imported_at": "2026-08-04T14:00:00",
    }
    row.update(overrides)
    return row


def _flex_row(exec_id: str, timestamp: str, **overrides):
    row = _socket_row(exec_id, timestamp)
    row.update(
        {
            "execution_uid": f"IBKR:U1234567:{exec_id}:NVDA:{timestamp}",
            "commission": 1.15,
            "net_amount": -1001.15,
            "raw_json": json.dumps(
                {"ibExecID": exec_id, "tradePrice": "100", "assetCategory": "STK", "accountId": "U1234567"}
            ),
            "imported_at": "2026-08-04T22:00:00",
        }
    )
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_the_symbol_and_timestamp_are_stripped_from_the_uid():
    row = _socket_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00-07:00")
    new_uid, reason = jm.canonical_execution_uid(row)
    assert reason == ""
    assert new_uid == "IBKR:U1234567:0000e0d5.68a3.01.01"


def test_an_ibkr_exec_id_full_of_dots_and_a_timestamp_full_of_colons_survive():
    """Why the old suffix is reconstructed instead of split off by counting colons."""
    row = _socket_row("0000e0d5.68a1b2c3.01.01", "2026-08-04T06:31:00-07:00")
    assert jm.canonical_execution_uid(row)[0] == "IBKR:U1234567:0000e0d5.68a1b2c3.01.01"


def test_an_already_canonical_uid_is_left_alone():
    row = _socket_row("x", "2026-08-04T06:31:00-07:00", execution_uid="MANUAL:MANUAL:abc123")
    assert jm.canonical_execution_uid(row)[0] == "MANUAL:MANUAL:abc123"


def test_a_uid_that_cannot_be_read_is_refused_rather_than_guessed():
    """Merging two different fills is worse than leaving a duplicate uid."""
    row = _socket_row("e1", "2026-08-04T06:31:00-07:00", execution_uid="IBKR:U1:e1:NVDA:SOME-OTHER-TIME")
    new_uid, reason = jm.canonical_execution_uid(row)
    assert new_uid is None
    assert "timestamp" in reason


@pytest.mark.parametrize(
    ("raw_json", "broker", "expected"),
    [
        (json.dumps({"contract": {}, "execution": {}}), "IBKR", "IBKR_SOCKET"),
        (json.dumps({"ibExecID": "x", "tradePrice": "1"}), "IBKR", "IBKR_FLEX"),
        (json.dumps({"id": 1, "symbolId": 2}), "QUESTRADE", "QT_API"),
        ("{}", "MANUAL", "MANUAL"),
        ("{}", "IBKR", ""),
        ("not json", "IBKR", ""),
    ],
)
def test_the_source_is_read_from_the_shape_of_the_raw_payload(raw_json, broker, expected):
    assert jm.classify_execution_source({"broker": broker, "raw_json": raw_json}) == expected


def test_a_missing_exec_id_gets_a_deterministic_surrogate_not_a_uuid():
    """Dropping symbol+timestamp removed the accidental uniqueness they provided."""
    first = jm.stable_execution_uid("QT", "51234567", "", "AAPL", "2026-08-03T09:31:00-07:00", 100, 150.0)
    second = jm.stable_execution_uid("QT", "51234567", "", "AAPL", "2026-08-03T09:31:00-07:00", 100, 150.0)
    third = jm.stable_execution_uid("QT", "51234567", "", "AAPL", "2026-08-03T09:31:00-07:00", 100, 151.0)
    assert first == second, "the same fill must not re-import as a new execution"
    assert first != third
    assert first.startswith("QT:51234567:auto-")


# ---------------------------------------------------------------------------
# The collapse
# ---------------------------------------------------------------------------


def test_the_socket_and_flex_copies_of_one_fill_become_one_execution(tmp_path):
    """Root cause B4, end to end: the position stops doubling."""
    db = _write_v2(
        tmp_path / "trade_journal.sqlite3",
        [
            _socket_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00-07:00"),
            _flex_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00"),
            _socket_row("0000e0d5.68a3.01.09", "2026-08-04T10:02:00-07:00", side="SELL", quantity=10.0, price=110.0),
        ],
    )
    store = JournalStore(db)
    report = store.last_migration
    assert report is not None
    assert report.executions_before == 3 and report.executions_after == 2
    assert len(report.collapsed) == 1

    kept = report.collapsed[0]
    assert kept["kept_source"] == "IBKR_FLEX" and kept["dropped_source"] == "IBKR_SOCKET"
    assert "richer source" in kept["reason"]

    with store.connection() as conn:
        row = conn.execute(
            "SELECT commission, source FROM raw_executions WHERE execution_uid = ?",
            ("IBKR:U1234567:0000e0d5.68a3.01.01",),
        ).fetchone()
        trades = conn.execute("SELECT status, quantity_opened, quantity_closed FROM trades").fetchall()
    assert row["commission"] == 1.15, "the Flex row's commission is the one that survives"
    assert row["source"] == "IBKR_FLEX"
    # 10 opened and 10 closed, not 20 opened and half of it stranded.
    assert [tuple(item) for item in trades] == [("CLOSED", 10.0, 10.0)]


def test_questrade_duplicates_keep_the_newer_import(tmp_path):
    """Two v2 rows for one Questrade execution, keyed apart by the timestamp.

    This is how a Questrade duplicate could exist under v2 at all: the uid
    embedded the timestamp, so a re-import that carried a corrected execution
    time landed as a second row rather than updating the first. Same execution
    id, two rows, doubled position - B4 by a different route than the IBKR one.
    """

    def _questrade(timestamp: str, price: float, imported_at: str):
        return {
            **_socket_row("qt-1", timestamp),
            "execution_uid": f"QT:51234567:qt-1:AAPL:{timestamp}",
            "broker": "QUESTRADE",
            "account_number": "51234567",
            "symbol": "AAPL",
            "price": price,
            "raw_json": json.dumps({"id": "qt-1"}),
            "imported_at": imported_at,
        }

    older = _questrade("2026-08-03T09:31:00-07:00", 150.0, "2026-08-03T20:00:00")
    newer = _questrade("2026-08-03T09:31:02-07:00", 150.25, "2026-08-04T20:00:00")
    db = _write_v2(tmp_path / "trade_journal.sqlite3", [older, newer])

    store = JournalStore(db)
    report = store.last_migration
    assert report.executions_before == 2 and report.executions_after == 1
    assert report.collapsed[0]["reason"].endswith("kept the newer import")
    with store.connection() as conn:
        row = conn.execute("SELECT execution_uid, price FROM raw_executions").fetchone()
    assert row["execution_uid"] == "QT:51234567:qt-1"
    assert row["price"] == 150.25


def test_idless_questrade_partials_are_rekeyed_before_the_v3_collapse(tmp_path):
    """An order id identifies the order, not each fill within that order."""
    account = "51234567"
    order_id = "4242"

    def _partial(timestamp: str, quantity: float, price: float):
        return {
            **_socket_row(order_id, timestamp),
            "execution_uid": f"QT:{account}:{order_id}:AAPL:{timestamp}",
            "broker": "QUESTRADE",
            "account_number": account,
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "exchange_exec_id": "",
            "raw_json": json.dumps(
                {
                    "orderId": int(order_id), "symbol": "AAPL", "side": "Buy",
                    "quantity": quantity, "price": price, "timestamp": timestamp,
                }
            ),
        }

    rows = [
        _partial("2026-08-05T09:31:00-07:00", 3.0, 150.0),
        _partial("2026-08-05T09:31:01-07:00", 2.0, 150.1),
    ]
    store = JournalStore(_write_v2(tmp_path / "trade_journal.sqlite3", rows))
    report = store.last_migration
    assert report is not None
    assert report.order_id_uid_groups == 1
    assert report.order_id_uid_rows == 2
    assert report.executions_before == report.executions_after == 2
    assert report.collapsed == []
    assert "1 legacy group(s)" in report.render()

    expected = {
        jm.stable_execution_uid(
            "QT", account, "", order_id, row["symbol"], row["timestamp"],
            row["side"], row["quantity"], row["price"],
        )
        for row in rows
    }
    with store.connection() as conn:
        actual = {row[0] for row in conn.execute("SELECT execution_uid FROM raw_executions")}
    assert actual == expected

    importer = QuestradeImporter.__new__(QuestradeImporter)
    account_row = {"number": account, "type": "TFSA"}
    re_pulled = [importer.normalize_execution(json.loads(row["raw_json"]), account_row) for row in rows]
    store.upsert_executions(re_pulled)
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == 2


def test_re_running_the_migration_changes_nothing(tmp_path):
    db = _write_v2(
        tmp_path / "trade_journal.sqlite3",
        [
            _socket_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00-07:00"),
            _flex_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00"),
        ],
    )
    first = JournalStore(db)
    assert first.last_migration is not None

    def _snapshot():
        conn = sqlite3.connect(db)
        try:
            return sorted(tuple(row) for row in conn.execute("SELECT * FROM raw_executions"))
        finally:
            conn.close()

    before = _snapshot()
    second = JournalStore(db)
    assert second.last_migration is None, "an already-v3 database is not migrated again"
    assert _snapshot() == before


def test_the_backup_is_taken_before_anything_is_deleted(tmp_path):
    db = _write_v2(
        tmp_path / "trade_journal.sqlite3",
        [
            _socket_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00-07:00"),
            _flex_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00"),
        ],
    )
    store = JournalStore(db)
    backups = list(tmp_path.glob("trade_journal.sqlite3.bak-v2-*"))
    assert len(backups) == 1
    assert str(backups[0]) == store.last_migration.backup_path

    # The backup still holds both rows and still says v2 - it is the state
    # before the migration, not a copy taken after it.
    conn = sqlite3.connect(backups[0])
    try:
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == 2
        assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "2"
    finally:
        conn.close()


def test_the_tax_status_seed_never_overwrites_the_trader(tmp_path):
    db = _write_v2(
        tmp_path / "trade_journal.sqlite3",
        [],
        accounts=[
            ("QUESTRADE", "51234567", "Margin", "Margin", "USD"),
            ("QUESTRADE", "52222222", "TFSA", "TFSA", "CAD"),
            ("QUESTRADE", "53333333", "Weird", "SomethingNew", "CAD"),
        ],
    )
    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE accounts ADD COLUMN tax_status TEXT NOT NULL DEFAULT ''")
    conn.execute("ALTER TABLE accounts ADD COLUMN tax_status_source TEXT NOT NULL DEFAULT ''")
    conn.execute(
        "UPDATE accounts SET tax_status = 'TAXABLE', tax_status_source = 'trader' "
        "WHERE account_number = '52222222'"
    )
    conn.commit()
    conn.close()

    store = JournalStore(db)
    with store.connection() as connection:
        rows = {
            row["account_number"]: (row["tax_status"], row["tax_status_source"])
            for row in connection.execute("SELECT * FROM accounts")
        }
    assert rows["51234567"] == ("TAXABLE", "auto")
    # I7: the trader said TAXABLE for a TFSA-typed account. Wrong-looking, and
    # still not ours to correct.
    assert rows["52222222"] == ("TAXABLE", "trader")
    # An account type nobody has mapped is left blank rather than guessed.
    assert rows["53333333"] == ("", "")


def test_machine_local_tax_status_seeds_without_account_ids_in_source(tmp_path, monkeypatch):
    import journal_migrate as jm

    db = _write_v2(
        tmp_path / "trade_journal.sqlite3", [],
        accounts=[("IBKR", "LOCAL-ONLY", "Brokerage", "", "USD")],
    )
    monkeypatch.setattr(
        jm, "get_local_setting",
        lambda key, default=None: {"IBKR:LOCAL-ONLY": "TAX_FREE"}
        if key == jm.TRADER_TAX_STATUS_SETTING else default,
    )

    store = JournalStore(db)

    assert store.list_accounts()[0]["tax_status"] == "TAX_FREE"
    source = (SCRIPTS_DIR / "journal_migrate.py").read_text(encoding="utf-8")
    for former_literal in ("51830546", "29347316", "U4867396", "U5102524"):
        assert former_literal not in source


def test_the_annotation_orphan_count_is_recorded_before_step_4_claims_to_fix_it(tmp_path):
    db = _write_v2(
        tmp_path / "trade_journal.sqlite3",
        [_socket_row("e1", "2026-08-04T06:31:00-07:00")],
        annotations=[("trade-that-no-longer-exists", "avwap-reclaim")],
    )
    store = JournalStore(db)
    assert store.last_migration.annotations_total == 1
    assert store.last_migration.annotations_orphaned_now == 1


def test_v2_trade_legs_bridge_uid_rewrites_so_annotations_rekey_to_the_rebuilt_trade(tmp_path):
    buy = _socket_row("e1", "2026-08-04T06:31:00-07:00")
    sell = _socket_row(
        "e2", "2026-08-04T10:31:00-07:00", side="SELL", quantity=10.0, price=110.0
    )
    old_trade_id = "v2-annotated-trade"
    db = _write_v2(
        tmp_path / "trade_journal.sqlite3",
        [buy, sell],
        trades=[
            (
                old_trade_id, "IBKR", "U1234567", "U1234567", "NVDA", "STK", "USD",
                "LONG", "CLOSED", buy["timestamp"], sell["timestamp"], "2026-08-04",
                10.0, 10.0, 100.0, 110.0, 100.0, 0.0, 0.0, 100.0, 100.0,
            )
        ],
        legs=[
            (old_trade_id, buy["execution_uid"], "BUY", "OPEN", 10.0, 100.0,
             buy["timestamp"], 0.0, 0.0),
            (old_trade_id, sell["execution_uid"], "SELL", "CLOSE", 10.0, 110.0,
             sell["timestamp"], 0.0, 0.0),
        ],
        annotations=[(old_trade_id, "avwap-reclaim")],
    )

    store = JournalStore(db)

    trades = store.list_trades()
    assert len(trades) == 1 and trades[0]["trade_id"] != old_trade_id
    with store.connection() as conn:
        annotation = conn.execute(
            "SELECT trade_id, setup_tags FROM trade_annotations"
        ).fetchone()
        orphan_count = conn.execute(
            "SELECT COUNT(*) FROM trade_annotations a "
            "LEFT JOIN trades t ON t.trade_id = a.trade_id WHERE t.trade_id IS NULL"
        ).fetchone()[0]
    assert tuple(annotation) == (trades[0]["trade_id"], "avwap-reclaim")
    assert orphan_count == 0
    assert store.last_rekey["orphaned"] == []


# ---------------------------------------------------------------------------
# The upsert's source precedence
# ---------------------------------------------------------------------------


def test_a_socket_import_cannot_overwrite_a_flex_row(tmp_path):
    """The same question the migration answers, asked at import time instead."""
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    flex = {**_flex_row("e1", "2026-08-04T06:31:00"), "execution_uid": "IBKR:U1:e1", "source": "IBKR_FLEX"}
    socket = {**_socket_row("e1", "2026-08-04T06:31:00-07:00"), "execution_uid": "IBKR:U1:e1", "source": "IBKR_SOCKET"}
    store.upsert_executions([flex])
    store.upsert_executions([socket])
    with store.connection() as conn:
        row = conn.execute("SELECT commission, source FROM raw_executions").fetchone()
    assert (row["commission"], row["source"]) == (1.15, "IBKR_FLEX")


def test_a_flex_import_does_overwrite_a_socket_row(tmp_path):
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    socket = {**_socket_row("e1", "2026-08-04T06:31:00-07:00"), "execution_uid": "IBKR:U1:e1", "source": "IBKR_SOCKET"}
    flex = {**_flex_row("e1", "2026-08-04T06:31:00"), "execution_uid": "IBKR:U1:e1", "source": "IBKR_FLEX"}
    store.upsert_executions([socket])
    store.upsert_executions([flex])
    with store.connection() as conn:
        row = conn.execute("SELECT commission, source FROM raw_executions").fetchone()
    assert (row["commission"], row["source"]) == (1.15, "IBKR_FLEX")


def test_a_row_of_the_same_source_still_refreshes(tmp_path):
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    first = {**_flex_row("e1", "2026-08-04T06:31:00"), "execution_uid": "IBKR:U1:e1", "source": "IBKR_FLEX"}
    corrected = {**first, "commission": 2.5}
    store.upsert_executions([first])
    store.upsert_executions([corrected])
    with store.connection() as conn:
        assert conn.execute("SELECT commission FROM raw_executions").fetchone()[0] == 2.5


def test_the_option_multiplier_is_stored_on_the_execution(tmp_path):
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    store.upsert_executions([row for row in CHARACTERIZATION_EXECUTIONS if row["security_type"] == "OPT"])
    with store.connection() as conn:
        multipliers = {row[0] for row in conn.execute("SELECT multiplier FROM raw_executions")}
    assert multipliers == {100.0}


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def test_a_dry_run_leaves_the_live_file_byte_identical(tmp_path):
    db = _write_v2(
        tmp_path / "trade_journal.sqlite3",
        [
            _socket_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00-07:00"),
            _flex_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00"),
        ],
    )
    before = db.read_bytes()
    report = jm.run_migration(db, dry_run=True)
    assert report.dry_run is True
    assert len(report.collapsed) == 1, "the dry run still reports what would happen"
    assert db.read_bytes() == before, "a dry run that writes is not a dry run"
    assert list(tmp_path.glob("*.bak-*")) == [], "and it takes no backup, because it changes nothing"


def test_the_cli_defaults_to_a_dry_run(tmp_path, capsys):
    db = _write_v2(tmp_path / "trade_journal.sqlite3", [_socket_row("e1", "2026-08-04T06:31:00-07:00")])
    before = db.read_bytes()
    assert jm.main(["--db", str(db)]) == 0
    output = capsys.readouterr().out
    assert "DRY RUN" in output and "Re-run with --apply" in output
    assert db.read_bytes() == before


def test_the_cli_applies_and_backs_up_when_told_to(tmp_path, capsys):
    db = _write_v2(
        tmp_path / "trade_journal.sqlite3",
        [
            _socket_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00-07:00"),
            _flex_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00"),
        ],
    )
    assert jm.main(["--db", str(db), "--apply"]) == 0
    output = capsys.readouterr().out
    assert "APPLIED" in output
    assert len(list(tmp_path.glob("*.bak-v2-*"))) == 1

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == 1
        assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(
            JOURNAL_SCHEMA_VERSION
        )
    finally:
        conn.close()


def test_the_cli_says_so_when_there_is_no_database(tmp_path, capsys):
    assert jm.main(["--db", str(tmp_path / "nope.sqlite3")]) == 2
    assert "no journal database" in capsys.readouterr().err


def test_the_report_names_every_kept_and_dropped_uid(tmp_path):
    db = _write_v2(
        tmp_path / "trade_journal.sqlite3",
        [
            _socket_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00-07:00"),
            _flex_row("0000e0d5.68a3.01.01", "2026-08-04T06:31:00"),
        ],
    )
    rendered = jm.run_migration(db, dry_run=True).render()
    assert "IBKR:U1234567:0000e0d5.68a3.01.01:NVDA:2026-08-04T06:31:00-07:00" in rendered
    assert "IBKR:U1234567:0000e0d5.68a3.01.01:NVDA:2026-08-04T06:31:00" in rendered
    assert "keep" in rendered and "drop" in rendered
