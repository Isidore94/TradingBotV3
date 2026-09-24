"""Repair stored IBKR fills the 2026-09-23 P&L audit found wrong. Dry run by default.

Three repairs, then one trade rebuild:

* Flex fills were stored with their New York ``dateTime`` labelled as Pacific
  time, 3 hours late. The time is re-read from each row's own raw payload.
* A socket combo-leg fill carries one extra ``.01`` on its exec id, so it and
  its Flex row were stored twice. The socket copy is dropped (Flex is richer),
  or renamed to the Flex spelling when no Flex row exists yet.
* The socket's combo (BAG) parent row is dropped when its legs are stored at
  the same instant; it made a fake open position.

No trade is deleted by hand: trades are rebuilt from the repaired fills.
``--apply`` backs the journal up first and puts the backup back on any error.

Usage::

    python scripts/journal_pnl_repair.py                      # dry run on a temp copy
    python scripts/journal_pnl_repair.py --apply --i-am-the-trader
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from journal_analytics import counts_in_pnl, has_invented_entry
from journal_identity import canonical_ibkr_exec_id, classify_execution_source
from journal_importers import IBKR_FLEX_TZ_NAME, BrokerTimestampError, parse_broker_datetime

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED_TO_START = 2
EXIT_BUSY = 3

#: How far apart a BAG parent and its legs may be stamped and still be one order.
BAG_LEG_WINDOW_SECONDS = 60.0


def _raw(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("raw_json") or "{}"))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _instant(text: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(text or ""))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _flex_timestamp(row: dict[str, Any]) -> str | None:
    """The Flex Trade row's time re-read from its own payload, or None if it is not one."""
    raw = _raw(row)
    if "tradePrice" not in raw:
        return None
    text = str(raw.get("dateTime") or raw.get("tradeDate") or "").replace(";", " ").strip()
    if not text:
        return None
    try:
        return parse_broker_datetime(text, strict=True, default_tz=IBKR_FLEX_TZ_NAME).isoformat()
    except BrokerTimestampError:
        return None


def repair_executions(conn: sqlite3.Connection) -> dict[str, Any]:
    """Apply the three fill repairs on an open connection. Idempotent."""
    conn.row_factory = sqlite3.Row
    report: dict[str, Any] = {
        "flex_retimed": [],
        "socket_duplicates_dropped": [],
        "socket_renamed": [],
        "bag_parents_dropped": [],
        "bag_parents_kept": [],
    }
    rows = [dict(row) for row in conn.execute("SELECT * FROM raw_executions WHERE broker = 'IBKR'")]

    for row in rows:
        source = str(row.get("source") or "") or classify_execution_source(row)
        if source != "IBKR_FLEX":
            continue
        fixed = _flex_timestamp(row)
        if fixed is None or fixed == str(row.get("timestamp") or ""):
            continue
        conn.execute(
            "UPDATE raw_executions SET timestamp = ?, trade_date = ? WHERE execution_uid = ?",
            (fixed, fixed[:10], row["execution_uid"]),
        )
        report["flex_retimed"].append(
            {"execution_uid": row["execution_uid"], "old": row.get("timestamp"), "new": fixed}
        )
        row["timestamp"] = fixed

    existing = {str(row["execution_uid"]) for row in rows}
    for row in rows:
        source = str(row.get("source") or "") or classify_execution_source(row)
        if source != "IBKR_SOCKET" or str(row.get("security_type") or "").upper() == "BAG":
            continue
        exec_id = str(row.get("exchange_exec_id") or "")
        canonical = canonical_ibkr_exec_id(exec_id)
        if canonical == exec_id:
            continue
        old_uid = str(row["execution_uid"])
        new_uid = f"IBKR:{row.get('account_number') or ''}:{canonical}"
        if new_uid in existing:
            conn.execute("DELETE FROM raw_executions WHERE execution_uid = ?", (old_uid,))
            report["socket_duplicates_dropped"].append({"dropped": old_uid, "kept": new_uid})
        else:
            conn.execute(
                "UPDATE raw_executions SET execution_uid = ?, exchange_exec_id = ? WHERE execution_uid = ?",
                (new_uid, canonical, old_uid),
            )
            existing.add(new_uid)
            report["socket_renamed"].append({"old": old_uid, "new": new_uid})
        existing.discard(old_uid)
        # Keeps annotations attached across the rebuild (they are re-keyed by leg uid).
        conn.execute("UPDATE trade_legs SET execution_uid = ? WHERE execution_uid = ?", (new_uid, old_uid))

    legs = [
        row for row in rows
        if str(row.get("security_type") or "").upper() != "BAG" and str(row["execution_uid"]) in existing
    ]
    for row in rows:
        source = str(row.get("source") or "") or classify_execution_source(row)
        if source != "IBKR_SOCKET" or str(row.get("security_type") or "").upper() != "BAG":
            continue
        when = _instant(row.get("timestamp"))
        underlying = str(row.get("symbol") or "").strip().upper()
        matched = [
            leg["execution_uid"] for leg in legs
            if leg.get("account_number") == row.get("account_number")
            and underlying
            and str(leg.get("symbol") or "").upper().startswith(underlying)
            and when is not None
            and (leg_when := _instant(leg.get("timestamp"))) is not None
            and abs((leg_when - when).total_seconds()) <= BAG_LEG_WINDOW_SECONDS
        ]
        if matched:
            conn.execute("DELETE FROM raw_executions WHERE execution_uid = ?", (row["execution_uid"],))
            report["bag_parents_dropped"].append({"execution_uid": row["execution_uid"], "legs": matched})
        else:
            report["bag_parents_kept"].append(
                {"execution_uid": row["execution_uid"], "reason": "no stored legs at the same time"}
            )
    return report


def closed_totals(store: Any) -> dict[str, Any]:
    """CLOSED trade money by currency: everything, and only what totals may count."""
    totals: dict[str, Any] = {"all_closed": {}, "counted": {}, "not_counted": {}, "open_like": 0}
    for trade in store.list_trades():
        currency = str(trade.get("currency") or "?").upper()
        net = float(trade.get("net_pnl") or 0.0)
        if str(trade.get("status") or "").upper() != "CLOSED":
            totals["open_like"] += 1
            continue
        buckets = ["all_closed", "counted" if counts_in_pnl(trade) else "not_counted"]
        for bucket in buckets:
            cell = totals[bucket].setdefault(currency, {"trades": 0, "net_pnl": 0.0})
            cell["trades"] += 1
            cell["net_pnl"] += net
    for bucket in ("all_closed", "counted", "not_counted"):
        for cell in totals[bucket].values():
            cell["net_pnl"] = round(cell["net_pnl"], 2)
    totals["invented_entry_trades"] = sum(1 for t in store.list_trades() if has_invented_entry(t))
    return totals


def _repair_file(db_path: Path) -> dict[str, Any]:
    from journal_store import JournalStore

    store = JournalStore(db_path)
    store.initialize_schema()
    before = closed_totals(store)
    conn = sqlite3.connect(db_path)
    try:
        report = repair_executions(conn)
        conn.commit()
    finally:
        conn.close()
    store.rebuild_trades(refresh_tags=False)
    report["before"] = before
    report["after"] = closed_totals(store)
    return report


def _backup_path(db_path: Path) -> Path:
    # Microseconds plus a counter so a second run never overwrites the first backup.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = db_path.parent / f"{db_path.stem}.pre-pnl-repair-{stamp}{db_path.suffix}.bak"
    n = 1
    while path.exists():
        path = db_path.parent / f"{db_path.stem}.pre-pnl-repair-{stamp}-{n}{db_path.suffix}.bak"
        n += 1
    return path


def render(report: dict[str, Any], *, applied: bool) -> str:
    lines = ["Journal P&L repair - " + ("APPLIED" if applied else "DRY RUN (nothing written)")]
    lines.append(f"  Flex fills re-timed to New York time: {len(report['flex_retimed'])}")
    lines.append(f"  Socket duplicates dropped (Flex kept): {len(report['socket_duplicates_dropped'])}")
    for item in report["socket_duplicates_dropped"]:
        lines.append(f"    {item['dropped']} -> {item['kept']}")
    lines.append(f"  Socket fills renamed to the Flex id: {len(report['socket_renamed'])}")
    lines.append(f"  Combo (BAG) parent rows dropped: {len(report['bag_parents_dropped'])}")
    for item in report["bag_parents_kept"]:
        lines.append(f"    kept {item['execution_uid']}: {item['reason']}")
    for label in ("before", "after"):
        totals = report.get(label) or {}
        lines.append(f"  {label.upper()}:")
        for bucket in ("all_closed", "counted", "not_counted"):
            for currency, cell in sorted((totals.get(bucket) or {}).items()):
                lines.append(
                    f"    {bucket:<12} {currency} {cell['trades']:>4} trades  {cell['net_pnl']:>12,.2f}"
                )
        lines.append(f"    trades not yet closed: {totals.get('open_like', 0)}")
    if report.get("backup"):
        lines.append(f"  Backup: {report['backup']} (copy it back over the journal to undo)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="", help="journal database (default: the desk's journal)")
    parser.add_argument("--apply", action="store_true", help="back up, then repair for real")
    parser.add_argument(
        "--i-am-the-trader", action="store_true",
        help="required before --apply may touch a database under the live data folder",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from journal_reclassify import _is_live_store, busy_reasons

    if args.db:
        db_path = Path(args.db).expanduser()
    else:
        from project_paths import JOURNAL_DB_FILE

        db_path = Path(JOURNAL_DB_FILE)
    if not db_path.is_file():
        print(f"No journal database at {db_path}", file=sys.stderr)
        return EXIT_REFUSED_TO_START

    if not args.apply:
        with tempfile.TemporaryDirectory(prefix="journal-pnl-repair-") as scratch:
            copy = Path(scratch) / db_path.name
            shutil.copy2(db_path, copy)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(db_path) + suffix)
                if sidecar.is_file():
                    shutil.copy2(sidecar, Path(str(copy) + suffix))
            report = _repair_file(copy)
        print(json.dumps(report, indent=2, default=str) if args.json else render(report, applied=False))
        print("\nNothing was written. Re-run with --apply to repair for real.")
        return EXIT_OK

    if _is_live_store(db_path) and not args.i_am_the_trader:
        print(
            f"{db_path} is inside the live data folder. Applying there is the trader's own "
            "act: re-run with --i-am-the-trader if you are the trader.",
            file=sys.stderr,
        )
        return EXIT_REFUSED_TO_START
    reasons = busy_reasons(db_path)
    if reasons:
        print("Not now - something else is writing this journal:", file=sys.stderr)
        for reason in reasons:
            print(f"  {reason}", file=sys.stderr)
        return EXIT_BUSY

    backup = _backup_path(db_path)
    shutil.copy2(db_path, backup)
    try:
        report = _repair_file(db_path)
    except Exception as exc:  # noqa: BLE001 - the restore is the point
        shutil.copy2(backup, db_path)
        print(f"REFUSED - {type(exc).__name__}: {exc}. The journal was put back.", file=sys.stderr)
        print(f"  the backup is still at {backup}", file=sys.stderr)
        return EXIT_FAILED
    report["backup"] = str(backup)
    print(json.dumps(report, indent=2, default=str) if args.json else render(report, applied=True))
    return EXIT_OK


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
