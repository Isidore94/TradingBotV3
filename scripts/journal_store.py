from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from journal_analytics import AutoTagger
from journal_identity import (
    contract_multiplier as _contract_multiplier_shared,
    group_key,
    group_key_text,
)
from journal_migrate import (
    SOURCE_RANK,
    MigrationReport,
    backup_database,
    migrate_to_v3,
    read_schema_version,
)
from project_paths import JOURNAL_DB_FILE, JOURNAL_EXPORT_DIR


JOURNAL_SCHEMA_VERSION = 3
EPSILON = 0.0000001

#: ``SOURCE_RANK`` as a SQL expression over the stored row, so the upsert can
#: refuse to let a poorer source overwrite a richer one. Built from the map
#: rather than typed out twice, so the two can never disagree.
_SOURCE_RANK_SQL = "(CASE raw_executions.source " + " ".join(
    f"WHEN '{name}' THEN {rank}" for name, rank in sorted(SOURCE_RANK.items()) if name
) + " ELSE 0 END)"

OPPORTUNITY_EVENT_TYPES = {
    "SEEN",
    "PLANNED",
    "TAKEN",
    "SKIPPED",
    "INVALIDATED",
    "TARGET_HIT",
    "STOPPED",
    "CLOSED",
    "REVIEWED",
    "NOTE",
}


REGIME_PRESETS = {
    "mid_term": [
        "Bull trend",
        "Bear trend",
        "Range",
        "Distribution",
        "Accumulation",
        "Volatile transition",
        "Custom",
    ],
    "short_term": [
        "Risk on",
        "Risk off",
        "Chop",
        "Pullback",
        "Breakout",
        "Breakdown",
        "Custom",
    ],
    "intraday": [
        "Trend up",
        "Trend down",
        "Two-way range",
        "Opening drive",
        "Fade day",
        "Low liquidity",
        "Custom",
    ],
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _date_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    if len(text) >= 10:
        return text[:10]
    return text


def _row_to_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    return {key: row[key] for key in row.keys()}


def _signed_quantity(row: dict[str, Any]) -> float:
    side = str(row.get("side") or "").strip().upper()
    qty = abs(_coerce_float(row.get("quantity")))
    if side in {"SELL", "SLD", "STC", "SSHORT", "SHORT"}:
        return -qty
    return qty


def _contract_multiplier(row: dict[str, Any]) -> float:
    """The contract multiplier. One definition, shared with the migration.

    This used to be a second copy of the rule that read ``security_type``
    verbatim, so it recognised "OPT"/"OPTION"/"OPTIONS" and missed "Option",
    "EquityOption" and every futures option. It now normalizes first, like
    everything else that reads a security type since R7 §9 step 3.
    """
    return _contract_multiplier_shared(row)


def _execution_assembly_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Normalized position identity first, then chronological fill order."""
    return (*group_key(row), str(row.get("timestamp") or ""), str(row.get("execution_uid") or ""))


def _hash_id(*parts: Any) -> str:
    blob = "|".join(str(part or "") for part in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def persisted_journal_schema_version(db_path: Path = JOURNAL_DB_FILE) -> int | None:
    """Read the schema marker without creating or migrating the database."""
    path = Path(db_path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return read_schema_version(conn)
    finally:
        conn.close()


def journal_database_needs_preparation(db_path: Path = JOURNAL_DB_FILE) -> bool:
    """Whether opening the Journal would create or migrate its persisted schema."""
    path = Path(db_path)
    if not path.is_file() or path.stat().st_size == 0:
        return True
    version = persisted_journal_schema_version(path)
    return version is None or version < JOURNAL_SCHEMA_VERSION


def existing_journal_requires_migration(db_path: Path = JOURNAL_DB_FILE) -> bool:
    """True only for a non-empty existing database below the current schema."""
    path = Path(db_path)
    return (
        path.is_file()
        and path.stat().st_size > 0
        and journal_database_needs_preparation(path)
    )


class JournalStore:
    def __init__(self, db_path: Path = JOURNAL_DB_FILE) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        #: The v2 -> v3 report, when this construction performed the migration.
        self.last_migration: MigrationReport | None = None
        #: What the last rebuild did to annotated trade ids (I4).
        self.last_rekey: dict[str, list[Any]] = {"remapped": [], "ambiguous": [], "orphaned": []}
        self.initialize_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connection(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Create the base tables, then migrate to the current schema version.

        The version is read **before** anything is created, because that read is
        the only way to tell a pre-existing v2 database (which must be backed up
        before its execution uids are collapsed) from a database this call is
        about to create for the first time (which has nothing to lose).
        """
        pre_existing = self.db_path.is_file() and self.db_path.stat().st_size > 0
        prior_version: int | None = None
        if pre_existing:
            with self.connection() as conn:
                prior_version = read_schema_version(conn)
            # A non-empty file with no recorded version is a v2 database from
            # before the version row existed. Treat it as v2 and back it up.
            if prior_version is None:
                prior_version = 2

        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS accounts (
                    broker TEXT NOT NULL,
                    account_number TEXT NOT NULL,
                    account_label TEXT NOT NULL DEFAULT '',
                    account_type TEXT NOT NULL DEFAULT '',
                    currency TEXT NOT NULL DEFAULT '',
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (broker, account_number)
                );

                CREATE TABLE IF NOT EXISTS import_runs (
                    import_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    imported_executions INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS raw_executions (
                    execution_uid TEXT PRIMARY KEY,
                    broker TEXT NOT NULL,
                    account_number TEXT NOT NULL,
                    account_label TEXT NOT NULL DEFAULT '',
                    account_type TEXT NOT NULL DEFAULT '',
                    symbol TEXT NOT NULL,
                    security_type TEXT NOT NULL DEFAULT '',
                    currency TEXT NOT NULL DEFAULT 'USD',
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    commission REAL NOT NULL DEFAULT 0,
                    fees REAL NOT NULL DEFAULT 0,
                    gross_amount REAL,
                    net_amount REAL,
                    order_id TEXT NOT NULL DEFAULT '',
                    exchange_exec_id TEXT NOT NULL DEFAULT '',
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    imported_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    broker TEXT NOT NULL,
                    account_number TEXT NOT NULL,
                    account_label TEXT NOT NULL DEFAULT '',
                    symbol TEXT NOT NULL,
                    security_type TEXT NOT NULL DEFAULT '',
                    currency TEXT NOT NULL DEFAULT 'USD',
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL DEFAULT '',
                    trade_date TEXT NOT NULL,
                    quantity_opened REAL NOT NULL DEFAULT 0,
                    quantity_closed REAL NOT NULL DEFAULT 0,
                    average_entry_price REAL NOT NULL DEFAULT 0,
                    average_exit_price REAL NOT NULL DEFAULT 0,
                    gross_pnl REAL NOT NULL DEFAULT 0,
                    commission REAL NOT NULL DEFAULT 0,
                    fees REAL NOT NULL DEFAULT 0,
                    net_pnl REAL NOT NULL DEFAULT 0,
                    pnl_usd REAL,
                    auto_tag_summary TEXT NOT NULL DEFAULT '',
                    tag_confidence REAL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_legs (
                    leg_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    execution_uid TEXT NOT NULL,
                    side TEXT NOT NULL,
                    role TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    commission REAL NOT NULL DEFAULT 0,
                    fees REAL NOT NULL DEFAULT 0,
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS trade_annotations (
                    trade_id TEXT PRIMARY KEY,
                    setup_tags TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auto_tag_candidates (
                    trade_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    rationale TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (trade_id, tag)
                );

                CREATE TABLE IF NOT EXISTS tag_corrections (
                    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    setup_tag TEXT NOT NULL,
                    source_trade_id TEXT NOT NULL DEFAULT '',
                    confidence_boost REAL NOT NULL DEFAULT 0.12,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS regimes (
                    trade_date TEXT PRIMARY KEY,
                    mid_term_regime TEXT NOT NULL DEFAULT '',
                    short_term_regime TEXT NOT NULL DEFAULT '',
                    intraday_regime TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS opportunity_events (
                    event_id TEXT PRIMARY KEY,
                    opportunity_id TEXT NOT NULL,
                    lifecycle_id TEXT NOT NULL,
                    symbol TEXT NOT NULL DEFAULT '',
                    side TEXT NOT NULL DEFAULT '',
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    trade_id TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    source TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_raw_exec_date ON raw_executions(trade_date);
                CREATE INDEX IF NOT EXISTS idx_raw_exec_symbol ON raw_executions(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_opportunity_events_opportunity
                    ON opportunity_events(opportunity_id, occurred_at);
                CREATE INDEX IF NOT EXISTS idx_opportunity_events_trade
                    ON opportunity_events(trade_id, occurred_at);
                CREATE INDEX IF NOT EXISTS idx_opportunity_events_date
                    ON opportunity_events(substr(occurred_at, 1, 10), event_type);
                """
            )

        needs_migration = prior_version is None or int(prior_version) < JOURNAL_SCHEMA_VERSION
        report = MigrationReport(db_path=str(self.db_path), from_version=prior_version)
        if pre_existing and prior_version is not None and int(prior_version) < JOURNAL_SCHEMA_VERSION:
            report.backup_path = str(backup_database(self.db_path, prior_version))

        with self.connection() as conn:
            migrate_to_v3(conn, report=report)

        if needs_migration:
            self.last_migration = report
            with self.connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("last_migration_report", _json_dumps(report.as_dict())),
                )
            if report.rebuild_required:
                # Collapsing duplicate executions invalidates every derived
                # trade and leg. Rebuilding here rather than leaving it to the
                # next import is what stops the Journal tab showing the doubled
                # positions the migration just fixed.
                self.rebuild_trades(refresh_tags=False)

    def start_import_run(
        self,
        source: str,
        *,
        account_number: str = "",
        trigger: str = "",
        coverage_start: Any = "",
        coverage_end: Any = "",
    ) -> int:
        """Open an import run, saying **which days** it intends to cover.

        The span is the A6 fix. A run that recorded only "42 executions" could
        not tell a day with no trades from a day nobody looked at, so a gap was
        structurally undetectable and the failed EOD slot left holes nothing
        went back for.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO import_runs(
                    source, status, started_at, account_number, trigger, coverage_start, coverage_end
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    "RUNNING",
                    _now_iso(),
                    str(account_number or ""),
                    str(trigger or ""),
                    _date_text(coverage_start) if coverage_start else "",
                    _date_text(coverage_end) if coverage_end else "",
                ),
            )
            return int(cursor.lastrowid)

    def finish_import_run(self, import_run_id: int, *, status: str, imported_executions: int, message: str = "") -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE import_runs
                SET status = ?, finished_at = ?, imported_executions = ?, message = ?
                WHERE import_run_id = ?
                """,
                (status, _now_iso(), int(imported_executions), str(message or ""), int(import_run_id)),
            )

    def upsert_accounts(self, broker: str, accounts: Iterable[dict[str, Any]]) -> int:
        rows = list(accounts)
        with self.connection() as conn:
            for account in rows:
                account_number = str(account.get("number") or account.get("accountNumber") or account.get("account_number") or "").strip()
                if not account_number:
                    continue
                conn.execute(
                    """
                    INSERT INTO accounts(
                        broker, account_number, account_label, account_type, currency, raw_json, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(broker, account_number) DO UPDATE SET
                        account_label = excluded.account_label,
                        account_type = excluded.account_type,
                        currency = excluded.currency,
                        raw_json = excluded.raw_json,
                        updated_at = excluded.updated_at
                        -- I7: `tax_status` and `tax_status_source` are absent
                        -- from this statement on purpose. They are trader-owned,
                        -- so an import refreshing an account's label from the
                        -- broker must leave them exactly as the trader set them.
                    """,
                    (
                        str(broker or "").upper(),
                        account_number,
                        str(account.get("name") or account.get("description") or account.get("account_label") or account_number),
                        str(account.get("type") or account.get("accountType") or account.get("account_type") or ""),
                        str(account.get("currency") or ""),
                        _json_dumps(account),
                        _now_iso(),
                    ),
                )
        return len(rows)

    def upsert_executions(self, executions: Iterable[Any]) -> int:
        """Insert or refresh executions, keyed by ``execution_uid``.

        Since v3 the uid no longer embeds the symbol and timestamp, so the same
        broker fill arriving over the IBKR socket and again in that night's Flex
        statement lands on one row instead of two (B4). That makes the question
        "which one wins?" real, and the answer is the same one the migration
        uses: the richer source. A desk-hours socket import may not overwrite the
        commissions, fees and netCash a Flex row already carries, and the
        ``WHERE`` clause below is what enforces it in the database rather than in
        whichever caller happens to run last.
        """
        rows = [item.as_row() if hasattr(item, "as_row") else dict(item) for item in executions]
        with self.connection() as conn:
            for row in rows:
                source = str(row.get("source") or "").upper()
                multiplier = row.get("multiplier")
                if multiplier is None:
                    multiplier = _contract_multiplier(row)
                conn.execute(
                    f"""
                    INSERT INTO raw_executions(
                        execution_uid, broker, account_number, account_label, account_type, symbol,
                        security_type, currency, side, quantity, price, timestamp, trade_date,
                        commission, fees, gross_amount, net_amount, order_id, exchange_exec_id,
                        raw_json, imported_at, source, multiplier
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(execution_uid) DO UPDATE SET
                        broker = excluded.broker,
                        account_number = excluded.account_number,
                        account_label = excluded.account_label,
                        account_type = excluded.account_type,
                        symbol = excluded.symbol,
                        security_type = excluded.security_type,
                        currency = excluded.currency,
                        side = excluded.side,
                        quantity = excluded.quantity,
                        price = excluded.price,
                        timestamp = excluded.timestamp,
                        trade_date = excluded.trade_date,
                        commission = excluded.commission,
                        fees = excluded.fees,
                        gross_amount = excluded.gross_amount,
                        net_amount = excluded.net_amount,
                        order_id = excluded.order_id,
                        exchange_exec_id = excluded.exchange_exec_id,
                        raw_json = excluded.raw_json,
                        imported_at = excluded.imported_at,
                        source = excluded.source,
                        multiplier = excluded.multiplier
                    WHERE {_SOURCE_RANK_SQL} <= ?
                    """,
                    (
                        row.get("execution_uid"),
                        str(row.get("broker") or "").upper(),
                        str(row.get("account_number") or ""),
                        str(row.get("account_label") or ""),
                        str(row.get("account_type") or ""),
                        str(row.get("symbol") or "").upper(),
                        str(row.get("security_type") or "").upper(),
                        str(row.get("currency") or "USD").upper(),
                        str(row.get("side") or "").upper(),
                        _coerce_float(row.get("quantity")),
                        _coerce_float(row.get("price")),
                        str(row.get("timestamp") or _now_iso()),
                        _date_text(row.get("trade_date") or row.get("timestamp")),
                        abs(_coerce_float(row.get("commission"))),
                        abs(_coerce_float(row.get("fees"))),
                        row.get("gross_amount"),
                        row.get("net_amount"),
                        str(row.get("order_id") or ""),
                        str(row.get("exchange_exec_id") or ""),
                        str(row.get("raw_json") or "{}"),
                        _now_iso(),
                        source,
                        float(multiplier),
                        SOURCE_RANK.get(source, 0),
                    ),
                )
        return len(rows)

    TAX_STATUSES = ("TAXABLE", "TAX_FREE", "TAX_DEFERRED")

    def list_accounts(self) -> list[dict[str, Any]]:
        """Every known account with its tax status, for the account tree (I6)."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM accounts ORDER BY broker, account_number"
            ).fetchall()
        accounts = []
        for raw in rows:
            row = _row_to_dict(raw)
            row["tax_status"] = str(row.get("tax_status") or "").upper()
            row["tax_status_source"] = str(row.get("tax_status_source") or "").lower()
            accounts.append(row)
        return accounts

    def set_account_tax_status(
        self, broker: str, account_number: str, tax_status: str, *, source: str = "trader"
    ) -> None:
        """Label an account's tax treatment. Trader-owned, so nothing else writes it.

        A blank status is allowed and means "unlabeled", which is the honest
        state for an account nobody has decided about - the account tree shows
        it in its own group rather than guessing, because a guessed tax status
        is a wrong number in a tax record.
        """
        normalized = str(tax_status or "").strip().upper()
        if normalized and normalized not in self.TAX_STATUSES:
            raise ValueError(f"unsupported tax status: {tax_status!r}")
        with self.connection() as conn:
            updated = conn.execute(
                "UPDATE accounts SET tax_status = ?, tax_status_source = ?, updated_at = ? "
                "WHERE broker = ? AND account_number = ?",
                (normalized, str(source or "").lower(), _now_iso(),
                 str(broker or "").upper(), str(account_number or "")),
            ).rowcount
            if not updated:
                # An account the trader labels before its first import still has
                # to keep the label - the import upsert never clobbers a
                # trader-sourced value (I7).
                conn.execute(
                    """
                    INSERT INTO accounts(
                        broker, account_number, account_label, account_type, currency,
                        raw_json, updated_at, tax_status, tax_status_source
                    ) VALUES(?, ?, ?, '', '', '{}', ?, ?, ?)
                    """,
                    (str(broker or "").upper(), str(account_number or ""), str(account_number or ""),
                     _now_iso(), normalized, str(source or "").lower()),
                )
        from project_paths import JOURNAL_DB_FILE

        if (
            str(source or "trader").lower() == "trader"
            and self.db_path.resolve() == Path(JOURNAL_DB_FILE).resolve()
        ):
            from journal_migrate import TRADER_TAX_STATUS_SETTING
            from project_paths import get_local_setting, save_local_setting

            saved = get_local_setting(TRADER_TAX_STATUS_SETTING, {})
            saved = dict(saved) if isinstance(saved, dict) else {}
            saved[f"{str(broker).upper()}:{str(account_number)}"] = normalized
            save_local_setting(TRADER_TAX_STATUS_SETTING, saved)

    def upsert_cash_transactions(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Store fees, dividends, interest and FX - the money that is not a trade.

        Kept out of ``raw_executions`` on purpose. These rows move cash but do
        not open or close a position, and letting one into assembly would invent
        a trade out of a dividend. The Fees view and the tax totals read them
        from here (§9 step 13).
        """
        stored = [dict(row) for row in rows]
        with self.connection() as conn:
            for row in stored:
                conn.execute(
                    """
                    INSERT INTO cash_transactions(
                        txn_uid, broker, account_number, txn_date, activity_type, description,
                        symbol, amount, currency, raw_json, imported_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(txn_uid) DO UPDATE SET
                        broker = excluded.broker,
                        account_number = excluded.account_number,
                        txn_date = excluded.txn_date,
                        activity_type = excluded.activity_type,
                        description = excluded.description,
                        symbol = excluded.symbol,
                        amount = excluded.amount,
                        currency = excluded.currency,
                        raw_json = excluded.raw_json,
                        imported_at = excluded.imported_at
                    """,
                    (
                        str(row.get("txn_uid") or ""),
                        str(row.get("broker") or "").upper(),
                        str(row.get("account_number") or ""),
                        _date_text(row.get("txn_date")),
                        str(row.get("activity_type") or "OTHER").upper(),
                        str(row.get("description") or ""),
                        str(row.get("symbol") or "").upper(),
                        _coerce_float(row.get("amount")),
                        str(row.get("currency") or "USD").upper(),
                        str(row.get("raw_json") or "{}"),
                        _now_iso(),
                    ),
                )
        return len(stored)

    def list_cash_transactions(
        self,
        *,
        broker: str = "",
        account_number: str = "",
        date_from: Any = None,
        date_to: Any = None,
        activity_type: str = "",
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if str(broker or "").strip():
            clauses.append("broker = ?")
            params.append(str(broker).upper())
        if str(account_number or "").strip():
            clauses.append("account_number = ?")
            params.append(str(account_number))
        if date_from is not None:
            clauses.append("txn_date >= ?")
            params.append(_date_text(date_from))
        if date_to is not None:
            clauses.append("txn_date <= ?")
            params.append(_date_text(date_to))
        if str(activity_type or "").strip():
            clauses.append("activity_type = ?")
            params.append(str(activity_type).upper())
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM cash_transactions {where} ORDER BY txn_date, txn_uid", params
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def _load_raw_executions(self) -> list[dict[str, Any]]:
        """Every execution, ordered so each position's fills arrive in time order.

        SQL cannot express the full normalized identity (especially OCC option
        spellings), so the rows are sorted with the exact same ``group_key``
        used by assembly. Within each normalized group, timestamp and uid are
        the deterministic fill order.
        """
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM raw_executions").fetchall()
        return sorted((_row_to_dict(row) for row in rows), key=_execution_assembly_sort_key)

    def rebuild_trades(self, *, refresh_tags: bool = True) -> int:
        """Reassemble every trade from the raw executions and the adjustments.

        Four things changed here in R7 sec 9 step 4, each a defect from the
        spec's sec 3 register rather than a preference:

        * ``CLOSED_PARTIAL`` exists (B1). A position that has been half exited
          used to be indistinguishable from one nobody had touched.
        * A fill that closes more than the journal knows is open no longer
          fabricates an inverse position out of the leftover (B2). The leftover
          still opens a trade - the shares were really sold - but its opening leg
          is marked ``SYNTHETIC_OPEN`` and the trade is stamped ``NEEDS_REVIEW``,
          because the honest statement is "an opening fill is missing", not
          "you are short".
        * Trader corrections in ``trade_adjustments`` are applied here, so a
          correction survives every future rebuild instead of being a one-time
          edit the next import undoes (B7, I3).
        * ``trade_id`` is anchored to the opening execution's uid instead of a
          per-group sequence number, and annotations are re-keyed onto the
          rebuilt trades (B6, I4). Inserting an earlier fill used to renumber
          every trade in the group and strand every tag and note on it.
        """
        executions = self._apply_adjustments(self._load_raw_executions())
        # Adjustments may change an identity or timestamp, or inject a fill.
        # Re-establish the same ordering contract after applying them.
        executions.sort(key=_execution_assembly_sort_key)
        group_actions = self._group_adjustments()
        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for row in executions:
            grouped.setdefault(group_key(row), []).append(row)

        trade_payloads = []
        leg_payloads = []
        for key, rows in grouped.items():
            current = None
            anchor_counts: dict[str, int] = {}

            def _open_trade(row, signed, commission, fees, *, synthetic, _key=key, _counts=anchor_counts):
                anchor = str(row.get("execution_uid") or "")
                occurrence = _counts.get(anchor, 0)
                _counts[anchor] = occurrence + 1
                state = self._new_trade_state(_key, row, signed, anchor, occurrence)
                state["synthetic_open"] = bool(synthetic)
                self._add_open_quantity(state, row, signed, commission, fees, synthetic=synthetic)
                return state

            for row in rows:
                signed_qty = _signed_quantity(row)
                if abs(signed_qty) <= EPSILON:
                    continue
                remaining_signed = signed_qty
                remaining_commission = abs(_coerce_float(row.get("commission")))
                remaining_fees = abs(_coerce_float(row.get("fees")))
                # Set once this single execution has already closed a position.
                # Anything left over after that is the B2 case: the journal was
                # asked to close more than it knows was ever opened.
                closed_from_this_row = False
                while abs(remaining_signed) > EPSILON:
                    if current is None:
                        current = _open_trade(
                            row,
                            remaining_signed,
                            remaining_commission,
                            remaining_fees,
                            synthetic=closed_from_this_row,
                        )
                        break

                    pos = float(current["position_qty"])
                    if pos * remaining_signed > 0:
                        self._add_open_quantity(current, row, remaining_signed, remaining_commission, remaining_fees)
                        break

                    close_abs = min(abs(pos), abs(remaining_signed))
                    ratio = close_abs / abs(remaining_signed)
                    leg_signed = -math.copysign(close_abs, pos)
                    leg_commission = remaining_commission * ratio
                    leg_fees = remaining_fees * ratio
                    self._add_close_quantity(current, row, leg_signed, leg_commission, leg_fees)
                    remaining_abs = abs(remaining_signed) - close_abs
                    remaining_signed = math.copysign(remaining_abs, remaining_signed) if remaining_abs > EPSILON else 0.0
                    remaining_commission = max(0.0, remaining_commission - leg_commission)
                    remaining_fees = max(0.0, remaining_fees - leg_fees)
                    closed_from_this_row = True

                    if abs(float(current["position_qty"])) <= EPSILON:
                        trade_payloads.append(self._finalize_trade_state(current))
                        leg_payloads.extend(current["legs"])
                        current = None

            if current is not None:
                forced = group_actions.get(group_key_text(key))
                if forced is not None:
                    self._force_close_trade_state(current, forced)
                trade_payloads.append(self._finalize_trade_state(current))
                leg_payloads.extend(current["legs"])

        with self.connection() as conn:
            carried = self._snapshot_referenced_trades(conn)
            conn.execute("DELETE FROM trade_legs")
            conn.execute("DELETE FROM auto_tag_candidates")
            conn.execute("DELETE FROM trades")
            for trade in trade_payloads:
                conn.execute(
                    """
                    INSERT INTO trades(
                        trade_id, broker, account_number, account_label, symbol, security_type, currency,
                        direction, status, opened_at, closed_at, trade_date, quantity_opened, quantity_closed,
                        average_entry_price, average_exit_price, gross_pnl, commission, fees, net_pnl,
                        pnl_usd, auto_tag_summary, tag_confidence, updated_at,
                        reconcile_status, anchor_execution_uid
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade["trade_id"],
                        trade["broker"],
                        trade["account_number"],
                        trade["account_label"],
                        trade["symbol"],
                        trade["security_type"],
                        trade["currency"],
                        trade["direction"],
                        trade["status"],
                        trade["opened_at"],
                        trade["closed_at"],
                        trade["trade_date"],
                        trade["quantity_opened"],
                        trade["quantity_closed"],
                        trade["average_entry_price"],
                        trade["average_exit_price"],
                        trade["gross_pnl"],
                        trade["commission"],
                        trade["fees"],
                        trade["net_pnl"],
                        trade["pnl_usd"],
                        "",
                        None,
                        _now_iso(),
                        trade["reconcile_status"],
                        trade["anchor_execution_uid"],
                    ),
                )
            for leg in leg_payloads:
                conn.execute(
                    """
                    INSERT INTO trade_legs(
                        trade_id, execution_uid, side, role, quantity, price, timestamp, commission, fees
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        leg["trade_id"],
                        leg["execution_uid"],
                        leg["side"],
                        leg["role"],
                        leg["quantity"],
                        leg["price"],
                        leg["timestamp"],
                        leg["commission"],
                        leg["fees"],
                    ),
                )
            # Deterministic import-derived lifecycle events are append-only.
            # Rebuilding the trade table cannot duplicate or rewrite them.
            for trade in trade_payloads:
                self._record_imported_trade_events(conn, trade)

            self.last_rekey = self._rekey_annotations(conn, carried, leg_payloads)

        self.book_currency_values()

        if refresh_tags:
            self.refresh_auto_tags()
        return len(trade_payloads)

    @staticmethod
    def _record_imported_trade_events(conn: sqlite3.Connection, trade: dict[str, Any]) -> None:
        trade_id = str(trade.get("trade_id") or "")
        opportunity_id = f"trade:{trade_id}"
        common = (
            opportunity_id,
            opportunity_id,
            str(trade.get("symbol") or "").upper(),
            str(trade.get("direction") or "").upper(),
            trade_id,
        )
        payload = _json_dumps(
            {
                "broker": trade.get("broker"),
                "account_number": trade.get("account_number"),
                "quantity_opened": trade.get("quantity_opened"),
                "average_entry_price": trade.get("average_entry_price"),
            }
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO opportunity_events(
                event_id, opportunity_id, lifecycle_id, symbol, side, event_type,
                occurred_at, trade_id, reason, payload_json, source, created_at
            ) VALUES(?, ?, ?, ?, ?, 'TAKEN', ?, ?, '', ?, 'broker_import', ?)
            """,
            (f"trade:{trade_id}:taken", *common[:4], trade.get("opened_at") or _now_iso(), common[4], payload, _now_iso()),
        )
        if str(trade.get("status") or "").upper() == "CLOSED" and trade.get("closed_at"):
            close_payload = _json_dumps(
                {
                    "average_exit_price": trade.get("average_exit_price"),
                    "net_pnl": trade.get("net_pnl"),
                    "pnl_usd": trade.get("pnl_usd"),
                }
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO opportunity_events(
                    event_id, opportunity_id, lifecycle_id, symbol, side, event_type,
                    occurred_at, trade_id, reason, payload_json, source, created_at
                ) VALUES(?, ?, ?, ?, ?, 'CLOSED', ?, ?, '', ?, 'broker_import', ?)
                """,
                (
                    f"trade:{trade_id}:closed",
                    *common[:4],
                    trade.get("closed_at"),
                    common[4],
                    close_payload,
                    _now_iso(),
                ),
            )

    # ------------------------------------------------------------------
    # Adjustments (spec sec 4 `trade_adjustments`). Applied here; the write API
    # arrives in sec 9 step 5. A correction lives as an append-only record and
    # is re-applied on every rebuild, so it survives the next import instead of
    # being a one-off edit that import silently undoes (I3, B7).
    # ------------------------------------------------------------------

    #: Actions that rewrite the execution list before assembly sees it.
    EXECUTION_ADJUSTMENT_ACTIONS = frozenset(
        {"VOID_EXECUTION", "EDIT_EXECUTION", "ADD_EXECUTION", "REASSIGN_GROUP"}
    )

    #: What an EDIT_EXECUTION or REASSIGN_GROUP may overlay. Deliberately not
    #: "whatever the payload contains": an adjustment must not rewrite
    #: ``execution_uid`` (that is identity, not data), and it must not touch a
    #: trader-owned field, which lives on the annotation and not here (I7).
    ADJUSTABLE_EXECUTION_FIELDS = frozenset(
        {
            "broker", "account_number", "account_label", "account_type", "symbol",
            "security_type", "currency", "side", "quantity", "price", "timestamp",
            "trade_date", "commission", "fees", "gross_amount", "net_amount",
            "order_id", "exchange_exec_id", "multiplier", "raw_json", "source",
        }
    )

    #: Every action the store will accept. An unknown action is refused at write
    #: time rather than silently ignored at rebuild time - a correction the
    #: trader believes they made, and which quietly does nothing, is worse than
    #: one that was never accepted.
    ADJUSTMENT_ACTIONS = frozenset(
        {"VOID_EXECUTION", "EDIT_EXECUTION", "ADD_EXECUTION", "FORCE_CLOSE", "REASSIGN_GROUP", "SUPERSEDE"}
    )

    #: The one action assembly deliberately ignores. It exists because an undo
    #: has to be a record - and a record spelled as one of the real actions is
    #: not inert. Undoing a FORCE_CLOSE by appending another FORCE_CLOSE with an
    #: empty payload closes the position all over again, which is exactly what
    #: the first version of this code did and what its test caught.
    INERT_ADJUSTMENT_ACTION = "SUPERSEDE"
    ADJUSTMENT_TARGET_KINDS = frozenset({"EXECUTION", "TRADE_GROUP"})

    #: Which target kind each action addresses. FORCE_CLOSE closes a position;
    #: everything else edits a row.
    ADJUSTMENT_TARGET_BY_ACTION = {
        "VOID_EXECUTION": "EXECUTION",
        "EDIT_EXECUTION": "EXECUTION",
        "ADD_EXECUTION": "EXECUTION",
        "REASSIGN_GROUP": "EXECUTION",
        "FORCE_CLOSE": "TRADE_GROUP",
        # SUPERSEDE inherits the kind of whatever it retires, so it is absent
        # here and handled explicitly below.
    }

    def record_adjustment(
        self,
        *,
        action: str,
        target_uid: str,
        reason: str,
        payload: Mapping[str, Any] | None = None,
        target_kind: str = "",
        source: str = "gui",
        supersedes: str = "",
    ) -> dict[str, Any]:
        """Append one correction. Nothing here is ever edited or deleted (I3).

        ``reason`` is mandatory and may not be blank. A correction with no stated
        reason is indistinguishable from a mistake six months later, and this
        table is the audit trail behind a tax filing.

        ``supersedes`` is how undo works: the old record stays and is marked as
        superseded by this one, so it stops applying at the next rebuild without
        ever leaving the history. Undoing an undo is another record.
        """
        normalized_action = str(action or "").strip().upper()
        if normalized_action not in self.ADJUSTMENT_ACTIONS:
            raise ValueError(f"unsupported adjustment action: {action!r}")
        expected_kind = self.ADJUSTMENT_TARGET_BY_ACTION.get(normalized_action, "")
        kind = str(target_kind or expected_kind).strip().upper()
        if kind not in self.ADJUSTMENT_TARGET_KINDS:
            raise ValueError(f"unsupported adjustment target kind: {target_kind!r}")
        if expected_kind and kind != expected_kind:
            raise ValueError(f"{normalized_action} targets {expected_kind}, not {kind}")
        target = str(target_uid or "").strip()
        if not target:
            raise ValueError("target_uid is required")
        cleaned_reason = str(reason or "").strip()
        if not cleaned_reason:
            raise ValueError("reason is required: an unexplained correction is not an audit trail")

        row = {
            "adjustment_id": uuid.uuid4().hex,
            "target_kind": kind,
            "target_uid": target,
            "action": normalized_action,
            "payload_json": _json_dumps(dict(payload or {})),
            "reason": cleaned_reason,
            "source": str(source or "").strip(),
            "superseded_by": "",
            "created_at": _now_iso(),
        }
        with self.connection() as conn:
            if supersedes:
                existing = conn.execute(
                    "SELECT adjustment_id FROM trade_adjustments WHERE adjustment_id = ?",
                    (str(supersedes),),
                ).fetchone()
                if existing is None:
                    raise ValueError(f"cannot supersede unknown adjustment {supersedes!r}")
            conn.execute(
                """
                INSERT INTO trade_adjustments(
                    adjustment_id, target_kind, target_uid, action, payload_json, reason, source,
                    superseded_by, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(
                    row[key]
                    for key in (
                        "adjustment_id", "target_kind", "target_uid", "action", "payload_json",
                        "reason", "source", "superseded_by", "created_at",
                    )
                ),
            )
            if supersedes:
                conn.execute(
                    "UPDATE trade_adjustments SET superseded_by = ? WHERE adjustment_id = ?",
                    (row["adjustment_id"], str(supersedes)),
                )
        result = dict(row)
        result["payload"] = json.loads(result["payload_json"])
        return result

    def undo_adjustment(self, adjustment_id: str, *, reason: str, source: str = "gui") -> dict[str, Any]:
        """Retire an adjustment by appending the record that supersedes it.

        Deliberately not a DELETE. What the trader corrected, and then
        un-corrected, is part of the audit trail (I3).

        The superseding record uses ``SUPERSEDE``, which assembly ignores. An
        empty payload is not enough to make a record inert: an ``EDIT_EXECUTION``
        that names no fields overlays nothing, but a ``FORCE_CLOSE`` with an
        empty payload closes the position again. The first version of this
        method did exactly that, and
        ``test_a_force_close_can_be_undone_and_the_position_reopens`` is why it
        does not now.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM trade_adjustments WHERE adjustment_id = ?", (str(adjustment_id),)
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown adjustment {adjustment_id!r}")
        existing = _row_to_dict(row)
        if existing["action"] == self.INERT_ADJUSTMENT_ACTION:
            with self.connection() as conn:
                predecessor = conn.execute(
                    "SELECT * FROM trade_adjustments WHERE superseded_by = ?",
                    (str(adjustment_id),),
                ).fetchone()
            if predecessor is None:
                raise ValueError("cannot re-apply an undo whose original adjustment is missing")
            original = _row_to_dict(predecessor)
            try:
                original_payload = json.loads(original.get("payload_json") or "{}")
            except json.JSONDecodeError:
                original_payload = {}
            return self.record_adjustment(
                action=original["action"], target_kind=original["target_kind"],
                target_uid=original["target_uid"], payload=original_payload,
                reason=reason, source=source, supersedes=str(adjustment_id),
            )
        return self.record_adjustment(
            action=self.INERT_ADJUSTMENT_ACTION,
            target_kind=existing["target_kind"],
            target_uid=existing["target_uid"],
            payload={},
            reason=reason,
            source=source,
            supersedes=str(adjustment_id),
        )

    def list_adjustments(
        self, *, target_uid: str = "", include_superseded: bool = True, limit: int = 500
    ) -> list[dict[str, Any]]:
        """The audit trail, newest first, for the Health tab and the trade pane."""
        clauses: list[str] = []
        params: list[Any] = []
        if str(target_uid or "").strip():
            clauses.append("target_uid = ?")
            params.append(str(target_uid).strip())
        if not include_superseded:
            clauses.append("COALESCE(superseded_by, '') = ''")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, min(10000, int(limit))))
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM trade_adjustments {where} "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                params,
            ).fetchall()
        result = []
        for raw in rows:
            row = _row_to_dict(raw)
            try:
                payload = json.loads(row.get("payload_json") or "{}")
            except json.JSONDecodeError:
                payload = {}
            row["payload"] = payload if isinstance(payload, dict) else {}
            result.append(row)
        return result

    def list_active_adjustments(self) -> list[dict[str, Any]]:
        """Adjustments that still apply, oldest first.

        A superseded record is history, not an instruction. That is how undo
        works here without deleting anything (I3).

        Ordered by ``rowid`` after ``created_at``, not by ``adjustment_id``.
        ``created_at`` has second precision, so two corrections made in the same
        second tie - and the tiebreaker used to be a random uuid, which made the
        order two same-second edits of the same field were applied in a coin
        flip. ``rowid`` is insertion order, which is what "later wins" has to
        mean. Found by ``test_the_trail_is_newest_first``.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trade_adjustments
                WHERE COALESCE(superseded_by, '') = ''
                ORDER BY created_at, rowid
                """
            ).fetchall()
        result = []
        for raw in rows:
            row = _row_to_dict(raw)
            try:
                payload = json.loads(row.get("payload_json") or "{}")
            except json.JSONDecodeError:
                payload = {}
            row["payload"] = payload if isinstance(payload, dict) else {}
            result.append(row)
        return result

    def _apply_adjustments(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rewrite the execution list per the active adjustments, in order."""
        adjustments = [
            item
            for item in self.list_active_adjustments()
            if str(item.get("target_kind") or "").upper() == "EXECUTION"
            and str(item.get("action") or "").upper() in self.EXECUTION_ADJUSTMENT_ACTIONS
        ]
        if not adjustments:
            return rows

        by_uid = {str(row.get("execution_uid") or ""): dict(row) for row in rows}
        for item in adjustments:
            action = str(item["action"]).upper()
            target = str(item.get("target_uid") or "")
            payload = item["payload"]
            if action == "VOID_EXECUTION":
                by_uid.pop(target, None)
                continue
            if action == "ADD_EXECUTION":
                injected = {
                    key: payload[key] for key in payload if key in self.ADJUSTABLE_EXECUTION_FIELDS
                }
                injected["execution_uid"] = target or f"ADJ:{item['adjustment_id']}"
                injected.setdefault("raw_json", _json_dumps({"adjustment_id": item["adjustment_id"]}))
                injected.setdefault("source", "MANUAL")
                injected["trade_date"] = _date_text(injected.get("trade_date") or injected.get("timestamp"))
                by_uid[injected["execution_uid"]] = injected
                continue
            existing = by_uid.get(target)
            if existing is None:
                continue
            for field, value in payload.items():
                if field in self.ADJUSTABLE_EXECUTION_FIELDS:
                    existing[field] = value
            if "timestamp" in payload and "trade_date" not in payload:
                existing["trade_date"] = _date_text(existing.get("timestamp"))

        # Re-sort. An EDIT can move a timestamp and an ADD injects a row, and
        # assembly nets in the order it is handed - the same ordering as
        # `_load_raw_executions`, for the same reason.
        return sorted(
            by_uid.values(),
            key=lambda row: (
                str(row.get("broker") or "").upper(),
                str(row.get("account_number") or ""),
                str(row.get("symbol") or "").upper(),
                str(row.get("currency") or "").upper(),
                str(row.get("timestamp") or ""),
                str(row.get("execution_uid") or ""),
            ),
        )

    def _group_adjustments(self) -> dict[str, dict[str, Any]]:
        """Active FORCE_CLOSE records, keyed by their target group text.

        Later records win: the list is oldest-first, so a second FORCE_CLOSE on
        the same position is the trader changing the price they want it closed
        at.
        """
        actions: dict[str, dict[str, Any]] = {}
        for item in self.list_active_adjustments():
            if str(item.get("target_kind") or "").upper() != "TRADE_GROUP":
                continue
            if str(item.get("action") or "").upper() != "FORCE_CLOSE":
                continue
            actions[str(item.get("target_uid") or "")] = item
        return actions

    def _force_close_trade_state(self, trade: dict[str, Any], adjustment: dict[str, Any]) -> None:
        """Close a stuck position with a synthetic fill, because a human said so.

        The price defaults to the average entry, which books **zero** P&L on the
        forced portion. That is deliberate: the system does not know what the
        position was really worth, and inventing a number would put a fabricated
        gain or loss into a tax record. ``FORCED_CLOSED`` records on the trade
        that a human closed it, not a broker.
        """
        payload = adjustment.get("payload") or {}
        remaining = float(trade["position_qty"])
        if abs(remaining) <= EPSILON:
            return
        price = _coerce_float(payload.get("price"), default=float(trade["average_entry_price"]))
        row = {
            "execution_uid": f"ADJ:{adjustment['adjustment_id']}",
            "price": price,
            "timestamp": str(payload.get("timestamp") or trade["last_at"] or ""),
        }
        self._add_close_quantity(trade, row, -remaining, 0.0, 0.0, role="SYNTHETIC_CLOSE")
        trade["forced_closed"] = True

    # ------------------------------------------------------------------
    # Annotation survival across a rebuild (I4, B6)
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_referenced_trades(conn: sqlite3.Connection) -> dict[str, set[str]]:
        """Every trade a human or an event refers to, with the executions it held.

        Taken before the DELETE, because afterwards there is nothing left to map
        from. Only referenced trades are snapshotted: a trade nobody has tagged,
        noted or recorded an event against loses nothing by being re-keyed.
        """
        referenced = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT trade_id FROM trade_annotations
                UNION
                SELECT trade_id FROM opportunity_events WHERE COALESCE(trade_id, '') != ''
                """
            )
        }
        if not referenced:
            return {}
        snapshot: dict[str, set[str]] = {}
        for trade_id, execution_uid in conn.execute("SELECT trade_id, execution_uid FROM trade_legs"):
            if str(trade_id) in referenced:
                snapshot.setdefault(str(trade_id), set()).add(str(execution_uid))
        return snapshot

    def _rekey_annotations(
        self,
        conn: sqlite3.Connection,
        carried: dict[str, set[str]],
        leg_payloads: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        """Move annotations from the old trade ids onto the rebuilt ones.

        The map is by **largest execution-uid overlap**: the rebuilt trade
        holding the most of the old trade's executions is the same trade. A tie
        is not resolved - it is reported, because guessing which of two trades a
        human's note belongs to is exactly the kind of quiet wrong answer this
        packet exists to remove.

        ``trade_aliases`` records every move, so ``opportunity_events`` - which
        are immutable and are never rewritten - can still be resolved to the
        trade they belong to.
        """
        summary: dict[str, list[Any]] = {"remapped": [], "ambiguous": [], "orphaned": []}
        if not carried:
            return summary

        new_by_trade: dict[str, set[str]] = {}
        for leg in leg_payloads:
            new_by_trade.setdefault(str(leg["trade_id"]), set()).add(str(leg["execution_uid"]))
        surviving = set(new_by_trade)
        annotated = {str(row[0]) for row in conn.execute("SELECT trade_id FROM trade_annotations")}
        claimed: set[str] = set()

        for old_id, old_uids in sorted(carried.items()):
            if old_id in surviving:
                continue  # the id did not move; there is nothing to carry.
            scored = sorted(
                ((len(old_uids & uids), new_id) for new_id, uids in new_by_trade.items() if old_uids & uids),
                reverse=True,
            )
            if not scored:
                summary["orphaned"].append(
                    {"old_trade_id": old_id, "reason": "no rebuilt trade shares any of its executions"}
                )
                continue
            best_score, best_id = scored[0]
            tied = [item[1] for item in scored if item[0] == best_score]
            if len(tied) > 1:
                summary["ambiguous"].append(
                    {
                        "old_trade_id": old_id,
                        "candidates": tied,
                        "reason": f"{len(tied)} rebuilt trades each share {best_score} execution(s)",
                    }
                )
                continue
            if best_id in claimed or (best_id in annotated and old_id in annotated):
                summary["ambiguous"].append(
                    {
                        "old_trade_id": old_id,
                        "candidates": [best_id],
                        "reason": "the rebuilt trade already carries another annotation",
                    }
                )
                continue
            if old_id in annotated:
                conn.execute(
                    "UPDATE trade_annotations SET trade_id = ? WHERE trade_id = ?", (best_id, old_id)
                )
                claimed.add(best_id)
            conn.execute(
                """
                INSERT OR IGNORE INTO trade_aliases(old_trade_id, new_trade_id, reason, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (old_id, best_id, f"rebuild re-key on {best_score} shared execution(s)", _now_iso()),
            )
            summary["remapped"].append({"old_trade_id": old_id, "new_trade_id": best_id, "shared": best_score})
        return summary

    def resolve_trade_id(self, trade_id: str) -> str:
        """Follow ``trade_aliases`` to the trade a historical id became.

        ``opportunity_events`` are immutable by design, so an event recorded
        against a trade that has since been re-keyed still names the old id.
        This is how a reader gets from that id to the live trade. Transitive,
        and it refuses to loop.
        """
        current = str(trade_id or "")
        if not current:
            return ""
        seen = {current}
        with self.connection() as conn:
            while True:
                row = conn.execute(
                    "SELECT new_trade_id FROM trade_aliases WHERE old_trade_id = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (current,),
                ).fetchone()
                if row is None:
                    return current
                nxt = str(row[0])
                if nxt in seen:
                    return current
                seen.add(nxt)
                current = nxt

    # ------------------------------------------------------------------
    # Assembly state
    # ------------------------------------------------------------------

    def _new_trade_state(
        self,
        key: tuple[str, str, str, str, str],
        row: dict[str, Any],
        signed_qty: float,
        anchor_execution_uid: str,
        occurrence: int,
    ) -> dict[str, Any]:
        direction = "LONG" if signed_qty > 0 else "SHORT"
        # Anchored to the execution that opened the position, not to a counter
        # within the group (B6). A backfill that inserted an earlier fill used to
        # renumber every later trade in the group and strand every annotation on
        # them; now only trades whose opening execution actually changed are
        # re-keyed, and the re-key pass carries their annotations across.
        # `occurrence` disambiguates the rare case where one execution opens two
        # trades - an oversell whose leftover starts a new position.
        trade_id = _hash_id(*key, anchor_execution_uid, direction, occurrence)
        return {
            "trade_id": trade_id,
            "anchor_execution_uid": anchor_execution_uid,
            "synthetic_open": False,
            "forced_closed": False,
            "broker": key[0],
            "account_number": key[1],
            "symbol": key[2],
            "security_type": key[3],
            "currency": key[4] or "USD",
            "direction": direction,
            "account_label": str(row.get("account_label") or key[1]),
            "opened_at": str(row.get("timestamp") or ""),
            "last_at": str(row.get("timestamp") or ""),
            "position_qty": 0.0,
            "average_entry_price": 0.0,
            "entry_notional": 0.0,
            "exit_notional": 0.0,
            "quantity_opened": 0.0,
            "quantity_closed": 0.0,
            "gross_pnl": 0.0,
            "commission": 0.0,
            "fees": 0.0,
            "multiplier": _contract_multiplier(row),
            "legs": [],
        }

    def _add_trade_leg(
        self,
        trade: dict[str, Any],
        row: dict[str, Any],
        signed_qty: float,
        commission: float,
        fees: float,
        *,
        role: str,
    ) -> None:
        side = "BUY" if signed_qty > 0 else "SELL"
        qty = abs(float(signed_qty))
        trade["legs"].append(
            {
                "trade_id": trade["trade_id"],
                "execution_uid": row.get("execution_uid"),
                "side": side,
                "role": role,
                "quantity": qty,
                "price": _coerce_float(row.get("price")),
                "timestamp": str(row.get("timestamp") or ""),
                "commission": abs(float(commission)),
                "fees": abs(float(fees)),
            }
        )
        trade["commission"] = float(trade["commission"]) + abs(float(commission))
        trade["fees"] = float(trade["fees"]) + abs(float(fees))
        trade["last_at"] = str(row.get("timestamp") or trade["last_at"])

    def _add_open_quantity(
        self,
        trade: dict[str, Any],
        row: dict[str, Any],
        signed_qty: float,
        commission: float,
        fees: float,
        *,
        synthetic: bool = False,
    ) -> None:
        qty = abs(float(signed_qty))
        old_abs = abs(float(trade["position_qty"]))
        price = _coerce_float(row.get("price"))
        new_abs = old_abs + qty
        if new_abs > EPSILON:
            trade["average_entry_price"] = ((float(trade["average_entry_price"]) * old_abs) + (price * qty)) / new_abs
        trade["position_qty"] = float(trade["position_qty"]) + float(signed_qty)
        trade["quantity_opened"] = float(trade["quantity_opened"]) + qty
        trade["entry_notional"] = float(trade["entry_notional"]) + (price * qty)
        if old_abs <= EPSILON:
            # SYNTHETIC_OPEN says the opening fill was never imported - the leg
            # below is the closing execution standing in for it, at its own
            # price, which is why the trade is also stamped NEEDS_REVIEW rather
            # than presented as a real entry.
            role = "SYNTHETIC_OPEN" if synthetic else "OPEN"
        else:
            role = "SCALE"
        self._add_trade_leg(trade, row, signed_qty, commission, fees, role=role)

    def _add_close_quantity(
        self,
        trade: dict[str, Any],
        row: dict[str, Any],
        signed_qty: float,
        commission: float,
        fees: float,
        *,
        role: str = "CLOSE",
    ) -> None:
        qty = abs(float(signed_qty))
        price = _coerce_float(row.get("price"))
        direction = 1.0 if str(trade["direction"]) == "LONG" else -1.0
        avg_entry = float(trade["average_entry_price"])
        multiplier = float(trade["multiplier"])
        trade["gross_pnl"] = float(trade["gross_pnl"]) + ((price - avg_entry) * qty * direction * multiplier)
        trade["position_qty"] = float(trade["position_qty"]) + float(signed_qty)
        trade["quantity_closed"] = float(trade["quantity_closed"]) + qty
        trade["exit_notional"] = float(trade["exit_notional"]) + (price * qty)
        self._add_trade_leg(trade, row, signed_qty, commission, fees, role=role)

    def _finalize_trade_state(self, trade: dict[str, Any]) -> dict[str, Any]:
        flat = abs(float(trade["position_qty"])) <= EPSILON
        if flat:
            status = "CLOSED"
        elif float(trade["quantity_closed"]) > EPSILON:
            # B1. A position that has been half exited is not the same thing as
            # one nobody has touched, and the trader cannot act on the second
            # reading of the first.
            status = "CLOSED_PARTIAL"
        else:
            status = "OPEN"
        closed_at = str(trade["last_at"] or "") if flat else ""
        if trade.get("forced_closed"):
            reconcile_status = "FORCED_CLOSED"
        elif trade.get("synthetic_open"):
            reconcile_status = "NEEDS_REVIEW"
        else:
            reconcile_status = ""
        commission = float(trade["commission"])
        fees = float(trade["fees"])
        net_pnl = float(trade["gross_pnl"]) - commission - fees
        currency = str(trade["currency"] or "USD").upper()
        quantity_closed = float(trade["quantity_closed"])
        average_exit = (float(trade["exit_notional"]) / quantity_closed) if quantity_closed > EPSILON else 0.0
        return {
            "trade_id": trade["trade_id"],
            "broker": trade["broker"],
            "account_number": trade["account_number"],
            "account_label": trade["account_label"],
            "symbol": trade["symbol"],
            "security_type": trade["security_type"],
            "currency": currency,
            "direction": trade["direction"],
            "status": status,
            "opened_at": trade["opened_at"],
            "closed_at": closed_at,
            "trade_date": _date_text(closed_at or trade["opened_at"]),
            "quantity_opened": float(trade["quantity_opened"]),
            "quantity_closed": quantity_closed,
            "average_entry_price": float(trade["average_entry_price"]),
            "average_exit_price": average_exit,
            "gross_pnl": float(trade["gross_pnl"]),
            "commission": commission,
            "fees": fees,
            "net_pnl": net_pnl,
            "pnl_usd": net_pnl if currency == "USD" else None,
            "reconcile_status": reconcile_status,
            "anchor_execution_uid": str(trade.get("anchor_execution_uid") or ""),
        }

    def book_currency_values(self) -> dict[str, int]:
        """Book every trade's P&L in CAD **and** in USD from the stored rates.

        I5, and the reason it is a separate pass rather than part of assembly:
        assembly is a pure function of executions and adjustments, and it must
        stay that way. Booking reads ``fx_rates`` and never fetches - a rate
        pulled at render time would make the same trade worth different amounts
        on different days, which is not a tax figure.

        A trade whose rate is not booked keeps ``net_pnl_cad = NULL``. That is
        the "unconverted" state the UI renders explicitly; it is never 0, and
        never the native number quietly relabelled.

        **The USD half is a DISPLAY value and CAD stays the tax-grade one**
        (2026-08-24; the 2026-08-18 deferral was reversed by the trader once the
        BoC chain was booking observations nightly). It is computed from this
        trade's own session rate - ``net_pnl_cad / rate_to_cad(USD, trade_date)``
        - so a trade taken on a 1.28 day is not valued at what a 1.42 day says.
        A USD-native trade books its own number and no rate at all, because
        dividing USD by USD would introduce rounding for nothing.

        Every rule the CAD half keeps, the USD half keeps: a missing observation
        clears the booking rather than leaving a stale number, and the effective
        date - which observation was actually used, after any weekend carry-back
        - is stored beside the value.
        """
        summary = {"converted": 0, "unconverted": 0, "usd_converted": 0, "usd_unconverted": 0}
        with self.connection() as conn:
            rates = {
                (str(row["rate_date"]), str(row["currency"]).upper()): (
                    float(row["rate_to_cad"]),
                    str(row["effective_date"] or row["rate_date"]),
                )
                for row in conn.execute("SELECT * FROM fx_rates")
            }
            for row in conn.execute(
                "SELECT trade_id, trade_date, currency, net_pnl FROM trades"
            ).fetchall():
                trade_id = str(row["trade_id"])
                currency = str(row["currency"] or "").upper()
                trade_date = _date_text(row["trade_date"])
                net_pnl = _coerce_float(row["net_pnl"])
                cad_value: float | None
                if currency == "CAD":
                    rate, effective = 1.0, trade_date
                else:
                    booked = rates.get((trade_date, currency))
                    if booked is None:
                        conn.execute(
                            "UPDATE trades SET net_pnl_cad = NULL, fx_rate = NULL, fx_rate_date = '' "
                            "WHERE trade_id = ?",
                            (trade_id,),
                        )
                        summary["unconverted"] += 1
                        cad_value = None
                        self._book_usd(conn, trade_id, currency, net_pnl, cad_value, rates, trade_date, summary)
                        continue
                    rate, effective = booked
                cad_value = net_pnl * rate
                conn.execute(
                    "UPDATE trades SET net_pnl_cad = ?, fx_rate = ?, fx_rate_date = ? WHERE trade_id = ?",
                    (cad_value, rate, effective, trade_id),
                )
                summary["converted"] += 1
                self._book_usd(conn, trade_id, currency, net_pnl, cad_value, rates, trade_date, summary)
        return summary

    @staticmethod
    def _book_usd(conn, trade_id, currency, net_pnl, cad_value, rates, trade_date, summary) -> None:
        """The USD display value for one trade, from the same stored table.

        Split out so the CAD path above reads as it always did. Clearing on a
        missing rate is deliberate: a stale USD number surviving a deleted
        observation would be a figure nothing on disk supports any more.
        """
        if currency == "USD":
            # Its own number. No rate, and none recorded - there is nothing to
            # be point-in-time about.
            conn.execute(
                "UPDATE trades SET net_pnl_usd = ?, fx_usd_rate = NULL, fx_usd_rate_date = '' "
                "WHERE trade_id = ?",
                (net_pnl, trade_id),
            )
            summary["usd_converted"] += 1
            return
        booked = rates.get((trade_date, "USD"))
        if booked is None or cad_value is None or not booked[0]:
            conn.execute(
                "UPDATE trades SET net_pnl_usd = NULL, fx_usd_rate = NULL, fx_usd_rate_date = '' "
                "WHERE trade_id = ?",
                (trade_id,),
            )
            summary["usd_unconverted"] += 1
            return
        usd_rate, usd_effective = booked
        conn.execute(
            "UPDATE trades SET net_pnl_usd = ?, fx_usd_rate = ?, fx_usd_rate_date = ? "
            "WHERE trade_id = ?",
            (cad_value / usd_rate, usd_rate, usd_effective, trade_id),
        )
        summary["usd_converted"] += 1

    def list_trades(
        self,
        *,
        trade_date: str | date | None = None,
        broker: str | None = None,
        account: str | None = None,
        symbol: str | None = None,
        status: str | None = None,
        direction: str | None = None,
        date_from: str | date | None = None,
        date_to: str | date | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if date_from:
            clauses.append("trade_date >= ?")
            params.append(_date_text(date_from))
        if date_to:
            clauses.append("trade_date <= ?")
            params.append(_date_text(date_to))
        if trade_date:
            date_value = _date_text(trade_date)
            clauses.append("(substr(opened_at, 1, 10) = ? OR substr(closed_at, 1, 10) = ? OR trade_date = ?)")
            params.extend([date_value, date_value, date_value])
        if broker and str(broker) != "All":
            clauses.append("broker = ?")
            params.append(str(broker).upper())
        if account and str(account) != "All":
            clauses.append("(account_number = ? OR account_label = ?)")
            params.extend([str(account), str(account)])
        if symbol:
            clauses.append("symbol = ?")
            params.append(str(symbol).upper())
        if status and str(status) != "All":
            clauses.append("status = ?")
            params.append(str(status).upper())
        if direction and str(direction) != "All":
            clauses.append("direction = ?")
            params.append(str(direction).upper())
        where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT t.*, COALESCE(a.setup_tags, '') AS setup_tags, COALESCE(a.notes, '') AS notes,
                       a.planned_entry AS planned_entry, a.planned_stop AS planned_stop,
                       a.planned_risk AS planned_risk, COALESCE(a.risk_source, '') AS risk_source
                FROM trades t
                LEFT JOIN trade_annotations a ON a.trade_id = t.trade_id
                {where_sql}
                ORDER BY t.trade_date DESC, t.opened_at DESC, t.symbol
                """,
                params,
            ).fetchall()
        trades = [_row_to_dict(row) for row in rows]
        for trade in trades:
            regime = self.get_regime_for_date(_date_text(trade.get("opened_at") or trade.get("trade_date")))
            trade.update(regime)
            trade["display_tags"] = trade.get("setup_tags") or trade.get("auto_tag_summary") or ""
        return trades

    def save_risk_fields(
        self,
        trade_id: str,
        *,
        planned_entry: float | None = None,
        planned_stop: float | None = None,
        planned_risk: float | None = None,
        risk_source: str = "manual",
    ) -> None:
        """Store the plan behind a trade. Trader-owned; no import path writes it.

        These live on ``trade_annotations`` rather than ``trades`` for one
        reason: that table survives a rebuild and carries its own re-key pass,
        so the risk a trader typed is not thrown away the next time a backfill
        re-assembles the trade (I4, I7).
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_annotations(
                    trade_id, setup_tags, notes, updated_at,
                    planned_entry, planned_stop, planned_risk, risk_source
                ) VALUES(?, '', '', ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    planned_entry = excluded.planned_entry,
                    planned_stop = excluded.planned_stop,
                    planned_risk = excluded.planned_risk,
                    risk_source = excluded.risk_source
                """,
                (
                    str(trade_id),
                    _now_iso(),
                    None if planned_entry is None else float(planned_entry),
                    None if planned_stop is None else float(planned_stop),
                    None if planned_risk is None else float(planned_risk),
                    str(risk_source or ""),
                ),
            )

    def list_trade_legs(self, trade_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT l.*, e.broker, e.account_number, e.symbol, e.security_type, e.currency
                FROM trade_legs l
                LEFT JOIN raw_executions e ON e.execution_uid = l.execution_uid
                WHERE l.trade_id = ?
                ORDER BY l.timestamp, l.leg_id
                """,
                (trade_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_trade(self, trade_id: str) -> dict[str, Any] | None:
        rows = self.list_trades()
        for row in rows:
            if row.get("trade_id") == trade_id:
                return row
        return None

    def save_trade_annotation(self, trade_id: str, *, setup_tags: str, notes: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_annotations(trade_id, setup_tags, notes, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    setup_tags = excluded.setup_tags,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                    -- planned_entry/stop/risk are absent on purpose: saving a
                    -- note must not erase the plan the trader typed earlier.
                """,
                (trade_id, str(setup_tags or "").strip(), str(notes or "").strip(), _now_iso()),
            )

    def record_opportunity_event(
        self,
        *,
        opportunity_id: str,
        event_type: str,
        lifecycle_id: str = "",
        symbol: str = "",
        side: str = "",
        occurred_at: str | datetime | None = None,
        trade_id: str = "",
        reason: str = "",
        payload: Mapping[str, Any] | None = None,
        source: str = "gui",
        event_id: str = "",
    ) -> dict[str, Any]:
        """Append one immutable opportunity lifecycle event.

        Scanner/UI callers may share ``lifecycle_id`` across evolving snapshot
        IDs. Imported trades use deterministic IDs; human/research events use a
        random event ID so repeated reviews remain an honest history.
        """

        opportunity = str(opportunity_id or "").strip()
        if not opportunity:
            raise ValueError("opportunity_id is required")
        normalized_type = str(event_type or "").strip().upper()
        if normalized_type not in OPPORTUNITY_EVENT_TYPES:
            raise ValueError(f"unsupported opportunity event type: {event_type}")
        normalized_side = str(side or "").strip().upper()
        if normalized_side and normalized_side not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG, SHORT, or blank")
        if isinstance(occurred_at, datetime):
            occurred_text = occurred_at.isoformat(timespec="seconds")
        else:
            occurred_text = str(occurred_at or _now_iso()).strip()
        row = {
            "event_id": str(event_id or uuid.uuid4().hex),
            "opportunity_id": opportunity,
            "lifecycle_id": str(lifecycle_id or opportunity).strip(),
            "symbol": str(symbol or "").strip().upper(),
            "side": normalized_side,
            "event_type": normalized_type,
            "occurred_at": occurred_text,
            "trade_id": str(trade_id or "").strip(),
            "reason": str(reason or "").strip(),
            "payload_json": _json_dumps(dict(payload or {})),
            "source": str(source or "").strip(),
            "created_at": _now_iso(),
        }
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO opportunity_events(
                    event_id, opportunity_id, lifecycle_id, symbol, side, event_type,
                    occurred_at, trade_id, reason, payload_json, source, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row[key] for key in (
                    "event_id", "opportunity_id", "lifecycle_id", "symbol", "side", "event_type",
                    "occurred_at", "trade_id", "reason", "payload_json", "source", "created_at",
                )),
            )
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def list_opportunity_events(
        self,
        *,
        opportunity_id: str = "",
        lifecycle_id: str = "",
        trade_id: str = "",
        trade_date: str | date | None = None,
        event_type: str = "",
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("opportunity_id", opportunity_id),
            ("lifecycle_id", lifecycle_id),
            ("trade_id", trade_id),
        ):
            if str(value or "").strip():
                clauses.append(f"{column} = ?")
                params.append(str(value).strip())
        if trade_date:
            clauses.append("substr(occurred_at, 1, 10) = ?")
            params.append(_date_text(trade_date))
        if str(event_type or "").strip():
            clauses.append("event_type = ?")
            params.append(str(event_type).strip().upper())
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, min(10000, int(limit))))
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM opportunity_events {where} ORDER BY occurred_at, created_at, event_id LIMIT ?",
                params,
            ).fetchall()
        result: list[dict[str, Any]] = []
        for raw in rows:
            row = _row_to_dict(raw)
            try:
                row["payload"] = json.loads(row.pop("payload_json") or "{}")
            except json.JSONDecodeError:
                row["payload"] = {}
            result.append(row)
        return result

    def latest_trade_review(self, trade_id: str) -> dict[str, Any] | None:
        rows = self.list_opportunity_events(trade_id=trade_id, event_type="REVIEWED", limit=10000)
        return rows[-1] if rows else None

    def record_tag_corrections(self, trade: dict[str, Any], tags: str) -> None:
        symbol = str(trade.get("symbol") or "").strip().upper()
        if not symbol:
            return
        parsed_tags = [part.strip() for part in str(tags or "").replace(",", ";").split(";") if part.strip()]
        if not parsed_tags:
            return
        with self.connection() as conn:
            for tag in parsed_tags:
                conn.execute(
                    """
                    INSERT INTO tag_corrections(symbol, setup_tag, source_trade_id, confidence_boost, created_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (symbol, tag, str(trade.get("trade_id") or ""), 0.12, _now_iso()),
                )

    def list_tag_corrections(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT symbol, setup_tag, source_trade_id, confidence_boost, created_at FROM tag_corrections"
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def refresh_auto_tags(self, tagger: AutoTagger | None = None) -> None:
        tagger = tagger or AutoTagger()
        corrections = self.list_tag_corrections()
        with self.connection() as conn:
            trade_rows = conn.execute("SELECT * FROM trades ORDER BY opened_at").fetchall()
            for row in trade_rows:
                trade = _row_to_dict(row)
                suggestions = tagger.suggest_for_trade(trade, corrections=corrections)
                top_summary = "; ".join(item["tag"] for item in suggestions[:3])
                top_confidence = suggestions[0]["confidence"] if suggestions else None
                conn.execute("DELETE FROM auto_tag_candidates WHERE trade_id = ?", (trade["trade_id"],))
                for item in suggestions:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO auto_tag_candidates(
                            trade_id, tag, confidence, source, rationale, created_at
                        ) VALUES(?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trade["trade_id"],
                            item["tag"],
                            float(item["confidence"]),
                            item.get("source", ""),
                            item.get("rationale", ""),
                            _now_iso(),
                        ),
                    )
                conn.execute(
                    "UPDATE trades SET auto_tag_summary = ?, tag_confidence = ?, updated_at = ? WHERE trade_id = ?",
                    (top_summary, top_confidence, _now_iso(), trade["trade_id"]),
                )

    def list_auto_tag_candidates(self, trade_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM auto_tag_candidates
                WHERE trade_id = ?
                ORDER BY confidence DESC, tag
                """,
                (trade_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def upsert_regime(
        self,
        trade_date: str | date,
        *,
        mid_term_regime: str = "",
        short_term_regime: str = "",
        intraday_regime: str = "",
        notes: str = "",
    ) -> None:
        date_value = _date_text(trade_date)
        with self.connection() as conn:
            existing = conn.execute("SELECT * FROM regimes WHERE trade_date = ?", (date_value,)).fetchone()
            row = _row_to_dict(existing) if existing else {}
            conn.execute(
                """
                INSERT INTO regimes(
                    trade_date, mid_term_regime, short_term_regime, intraday_regime, notes, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_date) DO UPDATE SET
                    mid_term_regime = excluded.mid_term_regime,
                    short_term_regime = excluded.short_term_regime,
                    intraday_regime = excluded.intraday_regime,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (
                    date_value,
                    str(mid_term_regime if mid_term_regime != "" else row.get("mid_term_regime", "")).strip(),
                    str(short_term_regime if short_term_regime != "" else row.get("short_term_regime", "")).strip(),
                    str(intraday_regime if intraday_regime != "" else row.get("intraday_regime", "")).strip(),
                    str(notes if notes != "" else row.get("notes", "")).strip(),
                    _now_iso(),
                ),
            )

    def get_regime_for_date(self, trade_date: str | date) -> dict[str, str]:
        date_value = _date_text(trade_date)
        with self.connection() as conn:
            exact = conn.execute("SELECT * FROM regimes WHERE trade_date = ?", (date_value,)).fetchone()
            carry = conn.execute(
                """
                SELECT * FROM regimes
                WHERE trade_date <= ?
                  AND (mid_term_regime != '' OR short_term_regime != '')
                ORDER BY trade_date DESC
                LIMIT 1
                """,
                (date_value,),
            ).fetchone()
        exact_row = _row_to_dict(exact) if exact else {}
        carry_row = _row_to_dict(carry) if carry else {}
        return {
            "mid_term_regime": str(exact_row.get("mid_term_regime") or carry_row.get("mid_term_regime") or ""),
            "short_term_regime": str(exact_row.get("short_term_regime") or carry_row.get("short_term_regime") or ""),
            "intraday_regime": str(exact_row.get("intraday_regime") or ""),
            "regime_notes": str(exact_row.get("notes") or ""),
        }

    def list_import_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM import_runs
                ORDER BY import_run_id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def distinct_values(self, column: str) -> list[str]:
        if column not in {"broker", "account_number", "account_label", "symbol"}:
            return []
        with self.connection() as conn:
            rows = conn.execute(f"SELECT DISTINCT {column} AS value FROM trades ORDER BY value").fetchall()
        return [str(row["value"]) for row in rows if str(row["value"] or "").strip()]

    def export_trades_csv(self, path: Path | None = None) -> Path:
        target = Path(path) if path else JOURNAL_EXPORT_DIR / f"journal_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.list_trades()
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return target
