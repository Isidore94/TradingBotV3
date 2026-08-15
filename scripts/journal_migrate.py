"""Journal schema v2 -> v3 migration, and the CLI that rehearses it.

R7 §9 step 2, contract in ``docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`` §4.

Two things happen here, and only one of them is reversible by re-running:

* **Additive DDL.** Five new tables and a dozen new columns. Idempotent, safe,
  boring - re-running finds everything present and does nothing.
* **The execution_uid collapse.** ``execution_uid`` stops embedding the symbol
  and the timestamp, so the same broker fill seen twice - once over the IBKR
  socket during the session, once in the Flex statement that night - stops
  being two executions that double the position (root cause B4). Rows that
  collapse are *deleted*, which is why :func:`backup_database` runs first and
  why the CLI defaults to ``--dry-run``.

The collapse keeps the richest row rather than the first one: for IBKR, a Flex
row beats a socket row, because Flex carries commissions, fees, netCash and the
option fields the socket never sends. Ties, and Questrade, keep the newest
``imported_at``. Every kept and dropped uid is named in the report - a silent
row count is not evidence.

WHAT "BROKER" MEANS IN A UID

Spec §4 writes the new identity as ``BROKER:account:exec_id``. The token used is
the one the importers already emit - ``QT``, ``IBKR``, or the manual broker
string - not the long ``QUESTRADE`` spelling from the ``broker`` column.
Rewriting ``QT`` to ``QUESTRADE`` would churn every Questrade uid for no gain
and break every uid a human has already seen in a report.

WHAT THIS STEP DOES NOT DO

It does not re-key ``trade_id`` or rescue annotations - that is the anchor pass
in §9 step 4. The report says how many annotations would be orphaned by a
rebuild today, so the number is on the record before step 4 claims to fix it.

Usage::

    python scripts/journal_migrate.py            # dry run on a copy (default)
    python scripts/journal_migrate.py --apply    # the real thing, after a backup
    python scripts/journal_migrate.py --db PATH --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

JOURNAL_SCHEMA_VERSION_V3 = 3

#: Which row wins when two collapse onto one uid. Flex beats socket because it
#: carries commission, fees, netCash and the option fields the socket omits.
SOURCE_RANK = {
    "IBKR_FLEX": 3,
    "QT_API": 2,
    "MANUAL": 2,
    "CSV": 1,
    "IBKR_SOCKET": 1,
    "": 0,
}

NEW_TABLES_V3: dict[str, str] = {
    "import_coverage": """
        CREATE TABLE IF NOT EXISTS import_coverage (
            broker TEXT NOT NULL,
            account_number TEXT NOT NULL,
            day TEXT NOT NULL,
            status TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            import_run_id INTEGER,
            attempts INTEGER NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (broker, account_number, day)
        )
    """,
    "fx_rates": """
        CREATE TABLE IF NOT EXISTS fx_rates (
            rate_date TEXT NOT NULL,
            currency TEXT NOT NULL,
            rate_to_cad REAL NOT NULL,
            source TEXT NOT NULL DEFAULT 'BOC_VALET',
            effective_date TEXT NOT NULL DEFAULT '',
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (rate_date, currency)
        )
    """,
    "trade_adjustments": """
        CREATE TABLE IF NOT EXISTS trade_adjustments (
            adjustment_id TEXT PRIMARY KEY,
            target_kind TEXT NOT NULL,
            target_uid TEXT NOT NULL,
            action TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            reason TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            superseded_by TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
    """,
    "trade_aliases": """
        CREATE TABLE IF NOT EXISTS trade_aliases (
            old_trade_id TEXT NOT NULL,
            new_trade_id TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            PRIMARY KEY (old_trade_id, new_trade_id)
        )
    """,
    "cash_transactions": """
        CREATE TABLE IF NOT EXISTS cash_transactions (
            txn_uid TEXT PRIMARY KEY,
            broker TEXT NOT NULL,
            account_number TEXT NOT NULL,
            txn_date TEXT NOT NULL,
            activity_type TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            symbol TEXT NOT NULL DEFAULT '',
            amount REAL NOT NULL DEFAULT 0,
            currency TEXT NOT NULL DEFAULT 'USD',
            raw_json TEXT NOT NULL DEFAULT '{}',
            imported_at TEXT NOT NULL
        )
    """,
}

NEW_INDEXES_V3 = (
    "CREATE INDEX IF NOT EXISTS idx_coverage_day ON import_coverage(day, status)",
    "CREATE INDEX IF NOT EXISTS idx_adjustments_target ON trade_adjustments(target_kind, target_uid)",
    "CREATE INDEX IF NOT EXISTS idx_aliases_new ON trade_aliases(new_trade_id)",
    "CREATE INDEX IF NOT EXISTS idx_cash_txn_date ON cash_transactions(txn_date, activity_type)",
    "CREATE INDEX IF NOT EXISTS idx_raw_exec_source ON raw_executions(source)",
)

#: (table, column, DDL type + default). All additive; SQLite rewrites nothing.
NEW_COLUMNS_V3: tuple[tuple[str, str, str], ...] = (
    ("import_runs", "coverage_start", "TEXT NOT NULL DEFAULT ''"),
    ("import_runs", "coverage_end", "TEXT NOT NULL DEFAULT ''"),
    ("import_runs", "account_number", "TEXT NOT NULL DEFAULT ''"),
    ("import_runs", "trigger", "TEXT NOT NULL DEFAULT ''"),
    ("accounts", "tax_status", "TEXT NOT NULL DEFAULT ''"),
    ("accounts", "tax_status_source", "TEXT NOT NULL DEFAULT ''"),
    ("raw_executions", "source", "TEXT NOT NULL DEFAULT ''"),
    ("raw_executions", "multiplier", "REAL NOT NULL DEFAULT 1"),
    ("trades", "net_pnl_cad", "REAL"),
    ("trades", "fx_rate", "REAL"),
    ("trades", "fx_rate_date", "TEXT NOT NULL DEFAULT ''"),
    ("trades", "reconcile_status", "TEXT NOT NULL DEFAULT ''"),
    ("trades", "anchor_execution_uid", "TEXT NOT NULL DEFAULT ''"),
    ("trade_annotations", "planned_entry", "REAL"),
    ("trade_annotations", "planned_stop", "REAL"),
    ("trade_annotations", "planned_risk", "REAL"),
    ("trade_annotations", "risk_source", "TEXT NOT NULL DEFAULT ''"),
)

#: Questrade account_type -> tax status. Spec §4; only ever seeds a blank value,
#: and never overwrites one the trader set (I7).
TAX_STATUS_BY_ACCOUNT_TYPE = {
    "TFSA": "TAX_FREE",
    "RRSP": "TAX_DEFERRED",
    "SRRSP": "TAX_DEFERRED",
    "RESP": "TAX_DEFERRED",
    "LIRA": "TAX_DEFERRED",
    "LIF": "TAX_DEFERRED",
    "RRIF": "TAX_DEFERRED",
    "FHSA": "TAX_FREE",
    "MARGIN": "TAXABLE",
    "CASH": "TAXABLE",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class MigrationReport:
    """Everything the migration did, in a form a human can audit line by line."""

    db_path: str = ""
    from_version: int | None = None
    to_version: int = JOURNAL_SCHEMA_VERSION_V3
    dry_run: bool = False
    backup_path: str = ""
    tables_created: list[str] = field(default_factory=list)
    columns_added: list[str] = field(default_factory=list)
    executions_before: int = 0
    executions_after: int = 0
    uids_rewritten: int = 0
    collapsed: list[dict[str, Any]] = field(default_factory=list)
    unmigrated: list[dict[str, Any]] = field(default_factory=list)
    sources_backfilled: dict[str, int] = field(default_factory=dict)
    tax_status_seeded: int = 0
    annotations_total: int = 0
    annotations_orphaned_now: int = 0
    rebuild_required: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "db_path": self.db_path,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "dry_run": self.dry_run,
            "backup_path": self.backup_path,
            "tables_created": list(self.tables_created),
            "columns_added": list(self.columns_added),
            "executions_before": self.executions_before,
            "executions_after": self.executions_after,
            "uids_rewritten": self.uids_rewritten,
            "collapsed": list(self.collapsed),
            "unmigrated": list(self.unmigrated),
            "sources_backfilled": dict(self.sources_backfilled),
            "tax_status_seeded": self.tax_status_seeded,
            "annotations_total": self.annotations_total,
            "annotations_orphaned_now": self.annotations_orphaned_now,
            "rebuild_required": self.rebuild_required,
        }

    def render(self) -> str:
        lines = [
            f"journal migration report ({'DRY RUN - nothing was written' if self.dry_run else 'APPLIED'})",
            f"  database          {self.db_path}",
            f"  schema            {self.from_version} -> {self.to_version}",
        ]
        if self.backup_path:
            lines.append(f"  backup            {self.backup_path}")
        lines.append(f"  tables created    {len(self.tables_created)}: {', '.join(self.tables_created) or 'none'}")
        lines.append(f"  columns added     {len(self.columns_added)}: {', '.join(self.columns_added) or 'none'}")
        lines.append(f"  executions        {self.executions_before} -> {self.executions_after}")
        lines.append(f"  uids rewritten    {self.uids_rewritten}")
        if self.sources_backfilled:
            detail = ", ".join(f"{key}={value}" for key, value in sorted(self.sources_backfilled.items()))
            lines.append(f"  source backfilled {detail}")
        if self.tax_status_seeded:
            lines.append(f"  tax status seeded {self.tax_status_seeded} account(s) from account_type")
        lines.append(f"  duplicates merged {len(self.collapsed)}")
        for item in self.collapsed:
            lines.append(f"    keep  {item['kept']}  ({item['kept_source'] or 'unknown source'})")
            lines.append(f"    drop  {item['dropped']}  ({item['dropped_source'] or 'unknown source'}) - {item['reason']}")
        if self.unmigrated:
            lines.append(f"  NOT migrated      {len(self.unmigrated)} uid(s) left exactly as they were:")
            for item in self.unmigrated:
                lines.append(f"    {item['execution_uid']} - {item['reason']}")
        lines.append(
            f"  annotations       {self.annotations_total} total, "
            f"{self.annotations_orphaned_now} would be orphaned by a rebuild today"
        )
        if self.annotations_orphaned_now:
            lines.append("    (the re-key pass that fixes this is §9 step 4; this line is the before-number)")
        if self.rebuild_required:
            lines.append("  a trade rebuild is required and is run automatically when the store opens")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def read_schema_version(conn: sqlite3.Connection) -> int | None:
    """The recorded schema version, or None when the database has no meta row."""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    try:
        return int(str(row[0]).strip())
    except (TypeError, ValueError):
        return None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def classify_execution_source(row: dict[str, Any]) -> str:
    """Which importer wrote this row, read from the shape of its raw payload.

    Nothing recorded the source before v3, so it has to be inferred - and the
    collapse rule depends on telling a Flex row from a socket row. The shapes
    are distinct enough to be safe: the socket importer writes a nested
    ``{"contract": ..., "execution": ...}`` object, Flex writes the statement's
    flat attributes, and Questrade writes the API's own JSON.
    """
    broker = str(row.get("broker") or "").upper()
    try:
        raw = json.loads(row.get("raw_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    if broker == "IBKR":
        if "contract" in raw and "execution" in raw:
            return "IBKR_SOCKET"
        if raw.keys() & {"ibExecID", "tradePrice", "assetCategory", "accountId", "tradeID"}:
            return "IBKR_FLEX"
        return ""
    if broker == "QUESTRADE":
        return "QT_API"
    if broker == "MANUAL":
        return "MANUAL"
    return ""


def contract_multiplier(row: dict[str, Any]) -> float:
    """The contract multiplier, from the raw payload or the security type.

    Deliberately the same rule as ``journal_store._contract_multiplier``; this
    copy exists so the migration does not have to import the store (which
    imports this module).
    """
    try:
        raw = json.loads(row.get("raw_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    candidates = [raw.get("multiplier")]
    contract = raw.get("contract")
    if isinstance(contract, dict):
        candidates.append(contract.get("multiplier"))
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    if str(row.get("security_type") or "").upper() in {"OPT", "OPTION", "OPTIONS", "FOP"}:
        return 100.0
    return 1.0


def stable_execution_uid(prefix: str, account_number: str, exec_id: Any, *fallback_parts: Any) -> str:
    """``PREFIX:account:exec_id``, with a deterministic surrogate when there is no exec id.

    Dropping the symbol and timestamp from the uid removes the accidental
    uniqueness they used to provide, so a broker row with no execution id can no
    longer be allowed a random uuid - it would re-import as a new execution
    every night and double the position by a different route than B4 did. The
    surrogate hashes the fields that identify the fill instead.
    """
    prefix = str(prefix or "").strip().upper()
    account = str(account_number or "").strip()
    cleaned_id = str(exec_id or "").strip()
    if cleaned_id:
        return f"{prefix}:{account}:{cleaned_id}"
    blob = "|".join(str(part or "") for part in fallback_parts)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{account}:auto-{digest}"


def canonical_execution_uid(row: dict[str, Any]) -> tuple[str | None, str]:
    """Rewrite a stored v2 uid to its v3 form.

    Returns ``(new_uid, reason)``; ``new_uid`` is None when the old uid cannot be
    read confidently, in which case it is left exactly as it was and named in
    the report. Guessing at an identity is the one thing this function must not
    do - a wrong guess silently merges two different fills.

    The v2 spellings were ``PREFIX:account:exec_id:SYMBOL:timestamp`` (Questrade,
    IBKR socket, IBKR Flex) and ``PREFIX:account:exec_id`` (manual). Both the
    symbol and the timestamp are columns on the row, so the old suffix can be
    reconstructed exactly and removed, rather than split off by counting colons -
    which would not survive either the colons in an ISO timestamp or the dots in
    an IBKR exec id.
    """
    uid = str(row.get("execution_uid") or "")
    if not uid:
        return None, "row has no execution_uid"
    parts = uid.split(":", 2)
    if len(parts) < 3:
        return None, f"uid has no account/exec-id section: {uid!r}"
    prefix, account, rest = parts[0], parts[1], parts[2]

    symbol = str(row.get("symbol") or "")
    timestamp = str(row.get("timestamp") or "")
    suffix = f":{symbol}:{timestamp}"
    if rest.endswith(suffix) and len(rest) > len(suffix):
        exec_id = rest[: -len(suffix)]
    elif symbol and f":{symbol}:" in rest:
        return None, f"uid carries a symbol but not this row's timestamp: {uid!r}"
    else:
        # Already canonical (manual rows, and anything migrated before).
        exec_id = rest

    if not exec_id:
        return None, f"uid has an empty execution id: {uid!r}"
    return f"{prefix}:{account}:{exec_id}", ""


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------


def backup_database(db_path: Path, from_version: int | None) -> Path:
    """Copy the database beside itself before anything destructive happens.

    Named for the version it holds, stamped to the second, and never reused: a
    backup that a second run can overwrite is not a backup.
    """
    db_path = Path(db_path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version_text = "unknown" if from_version is None else f"v{from_version}"
    target = db_path.with_name(f"{db_path.name}.bak-{version_text}-{stamp}")
    counter = 1
    while target.exists():
        target = db_path.with_name(f"{db_path.name}.bak-{version_text}-{stamp}-{counter}")
        counter += 1
    shutil.copy2(db_path, target)
    # Rollback-journal siblings, if the process that wrote last left any.
    for suffix in ("-wal", "-shm", "-journal"):
        sibling = db_path.with_name(db_path.name + suffix)
        if sibling.exists():
            shutil.copy2(sibling, target.with_name(target.name + suffix))
    return target


# ---------------------------------------------------------------------------
# The migration itself
# ---------------------------------------------------------------------------


def migrate_to_v3(conn: sqlite3.Connection, *, report: MigrationReport | None = None) -> MigrationReport:
    """Bring an open journal database to schema v3. Idempotent."""
    report = report or MigrationReport()
    report.from_version = read_schema_version(conn)

    for table, ddl in NEW_TABLES_V3.items():
        if not _table_exists(conn, table):
            report.tables_created.append(table)
        conn.execute(ddl)

    for table, column, decl in NEW_COLUMNS_V3:
        if not _table_exists(conn, table):
            continue
        if column in _columns(conn, table):
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        report.columns_added.append(f"{table}.{column}")

    # After the columns exist: one of these indexes is on a column this
    # migration just added.
    for statement in NEW_INDEXES_V3:
        conn.execute(statement)

    _collapse_execution_uids(conn, report)
    _backfill_sources_and_multipliers(conn, report)
    _seed_tax_status(conn, report)
    _count_annotation_orphans(conn, report)

    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
        ("schema_version", str(JOURNAL_SCHEMA_VERSION_V3)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
        ("last_migration_at", _now_iso()),
    )
    return report


def _collapse_execution_uids(conn: sqlite3.Connection, report: MigrationReport) -> None:
    rows = [dict(row) for row in conn.execute("SELECT * FROM raw_executions")]
    report.executions_before = len(rows)
    if not rows:
        report.executions_after = 0
        return

    keyed: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        new_uid, reason = canonical_execution_uid(row)
        if new_uid is None:
            report.unmigrated.append({"execution_uid": row.get("execution_uid"), "reason": reason})
            continue
        row["_new_uid"] = new_uid
        row["_source"] = classify_execution_source(row)
        keyed.setdefault(new_uid, []).append(row)

    def _rank(row: dict[str, Any]) -> tuple[int, str, str]:
        return (
            SOURCE_RANK.get(str(row.get("_source") or ""), 0),
            str(row.get("imported_at") or ""),
            str(row.get("execution_uid") or ""),
        )

    for new_uid, group in sorted(keyed.items()):
        winner = max(group, key=_rank)
        for loser in sorted(group, key=lambda item: str(item.get("execution_uid") or "")):
            if loser is winner:
                continue
            report.collapsed.append(
                {
                    "new_uid": new_uid,
                    "kept": winner.get("execution_uid"),
                    "kept_source": winner.get("_source") or "",
                    "dropped": loser.get("execution_uid"),
                    "dropped_source": loser.get("_source") or "",
                    "reason": (
                        "same broker execution id; kept the richer source"
                        if _rank(winner)[0] != _rank(loser)[0]
                        else "same broker execution id; kept the newer import"
                    ),
                }
            )
            conn.execute("DELETE FROM raw_executions WHERE execution_uid = ?", (loser["execution_uid"],))
        if str(winner.get("execution_uid")) != new_uid:
            try:
                conn.execute(
                    "UPDATE raw_executions SET execution_uid = ? WHERE execution_uid = ?",
                    (new_uid, winner["execution_uid"]),
                )
            except sqlite3.IntegrityError as exc:
                # A row that could not be rewritten already carries this uid.
                # Leave both alone and say so; merging two executions on a
                # coincidence is the one outcome worse than a duplicate.
                report.unmigrated.append(
                    {"execution_uid": winner["execution_uid"], "reason": f"target uid already exists ({exc})"}
                )
                continue
            report.uids_rewritten += 1

    report.executions_after = int(
        conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0]
    )
    report.rebuild_required = bool(report.collapsed or report.uids_rewritten)


def _backfill_sources_and_multipliers(conn: sqlite3.Connection, report: MigrationReport) -> None:
    for row in [dict(item) for item in conn.execute("SELECT * FROM raw_executions")]:
        updates: list[str] = []
        params: list[Any] = []
        if not str(row.get("source") or "").strip():
            source = classify_execution_source(row)
            if source:
                updates.append("source = ?")
                params.append(source)
                report.sources_backfilled[source] = report.sources_backfilled.get(source, 0) + 1
        multiplier = contract_multiplier(row)
        if float(row.get("multiplier") or 1.0) != multiplier:
            updates.append("multiplier = ?")
            params.append(multiplier)
        if not updates:
            continue
        params.append(row["execution_uid"])
        conn.execute(f"UPDATE raw_executions SET {', '.join(updates)} WHERE execution_uid = ?", params)


def _seed_tax_status(conn: sqlite3.Connection, report: MigrationReport) -> None:
    """Seed a blank tax status from the broker's account type. Never overwrite.

    I7: a value the trader set carries ``tax_status_source='trader'`` and is not
    touched here or by any import. Seeding is a starting point for the labeling
    step the trader does in §9 step 11, not an answer.
    """
    if not _table_exists(conn, "accounts"):
        return
    for row in [dict(item) for item in conn.execute("SELECT * FROM accounts")]:
        if str(row.get("tax_status") or "").strip():
            continue
        account_type = str(row.get("account_type") or "").strip().upper()
        seeded = TAX_STATUS_BY_ACCOUNT_TYPE.get(account_type, "")
        if not seeded:
            continue
        conn.execute(
            "UPDATE accounts SET tax_status = ?, tax_status_source = 'auto' "
            "WHERE broker = ? AND account_number = ?",
            (seeded, row.get("broker"), row.get("account_number")),
        )
        report.tax_status_seeded += 1


def _count_annotation_orphans(conn: sqlite3.Connection, report: MigrationReport) -> None:
    if not _table_exists(conn, "trade_annotations"):
        return
    report.annotations_total = int(
        conn.execute("SELECT COUNT(*) FROM trade_annotations").fetchone()[0]
    )
    report.annotations_orphaned_now = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM trade_annotations a
            LEFT JOIN trades t ON t.trade_id = a.trade_id
            WHERE t.trade_id IS NULL
            """
        ).fetchone()[0]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def run_migration(db_path: Path, *, dry_run: bool = True) -> MigrationReport:
    """Migrate ``db_path``, or a throwaway copy of it when ``dry_run``."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"no journal database at {db_path}")

    if dry_run:
        with tempfile.TemporaryDirectory(prefix="journal-migrate-dry-") as tmp:
            copy_path = Path(tmp) / db_path.name
            shutil.copy2(db_path, copy_path)
            conn = _connect(copy_path)
            try:
                report = MigrationReport(db_path=str(db_path), dry_run=True)
                migrate_to_v3(conn, report=report)
                conn.commit()
            finally:
                conn.close()
        return report

    conn = _connect(db_path)
    try:
        from_version = read_schema_version(conn)
    finally:
        conn.close()
    report = MigrationReport(db_path=str(db_path), dry_run=False)
    report.backup_path = str(backup_database(db_path, from_version))
    conn = _connect(db_path)
    try:
        migrate_to_v3(conn, report=report)
        conn.commit()
    finally:
        conn.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate the trade journal to schema v3.")
    parser.add_argument("--db", default="", help="database path (default: the configured journal DB)")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually migrate. Without this the run is a dry run against a temporary copy.",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    if args.db:
        db_path = Path(args.db)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from project_paths import JOURNAL_DB_FILE

        db_path = Path(JOURNAL_DB_FILE)

    try:
        report = run_migration(db_path, dry_run=not args.apply)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(report.as_dict(), indent=2) if args.json else report.render())
    if not args.apply:
        print("\nNothing was written. Re-run with --apply to migrate for real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
