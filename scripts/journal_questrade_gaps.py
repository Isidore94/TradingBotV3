"""List the Questrade days the journal could not import, and retry them on request.

Read-only by default: prints every FAILED ``import_coverage`` day (grouped by
reason) and every open-position mismatch from the last reconciliation. Nothing
contacts a broker.

``--reimport`` shows which days a retry would fetch. ``--reimport --apply``
retries them through the desk's own per-day Questrade import
(``journal_runner._fetch_one_day`` via ``journal_coverage.self_heal``), then
rebuilds trades. It backs the journal up first. Close the desk before running
it: the Questrade token chain is single-use, and two processes refreshing it
at once can break it.

Usage::

    python scripts/journal_questrade_gaps.py
    python scripts/journal_questrade_gaps.py --reimport
    python scripts/journal_questrade_gaps.py --reimport --apply --i-am-the-trader
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED_TO_START = 2
EXIT_BUSY = 3

#: The reason Questrade's executions endpoint gave nothing the activities
#: endpoint says was traded. An API retry usually cannot fix these days; a
#: broker statement import can.
EXECUTIONS_MISSING_REASON = "activities report trades the executions endpoint did not return"


def _read_only(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def gap_report(db_path: Path) -> dict[str, Any]:
    """FAILED Questrade days and reconciliation mismatches, read without writing."""
    conn = _read_only(db_path)
    try:
        failed = [
            dict(row)
            for row in conn.execute(
                "SELECT account_number, day, attempts, message, updated_at FROM import_coverage "
                "WHERE broker = 'QUESTRADE' AND status = 'FAILED' ORDER BY day, account_number"
            )
        ]
        meta = conn.execute("SELECT value FROM meta WHERE key = 'last_reconciliation'").fetchone()
    finally:
        conn.close()
    reconciliation: dict[str, Any] = {}
    if meta:
        try:
            reconciliation = json.loads(meta[0]) or {}
        except (TypeError, ValueError):
            reconciliation = {}
    by_reason: dict[str, int] = defaultdict(int)
    for row in failed:
        by_reason[_reason_group(row.get("message"))] += 1
    return {
        "failed_days": failed,
        "failed_by_reason": dict(sorted(by_reason.items(), key=lambda item: -item[1])),
        "mismatches": list(reconciliation.get("mismatched") or []),
        "reconciled_at": str(reconciliation.get("checked_at") or ""),
    }


def _reason_group(message: Any) -> str:
    text = str(message or "")
    if text.startswith(EXECUTIONS_MISSING_REASON):
        return "executions endpoint returned nothing the activities show (statement import fixes)"
    if "activities cross-check unavailable" in text:
        return "activities cross-check failed (API error)"
    if "Server Error" in text:
        return "Questrade server error"
    return text[:80] or "(no message)"


def render(report: dict[str, Any]) -> str:
    failed = report["failed_days"]
    lines = [f"Questrade FAILED import days: {len(failed)} (account-days)"]
    for reason, count in report["failed_by_reason"].items():
        lines.append(f"  {count:>4}  {reason}")
    by_account: dict[str, list[str]] = defaultdict(list)
    for row in failed:
        by_account[str(row["account_number"])].append(str(row["day"]))
    for account, days in sorted(by_account.items()):
        lines.append(f"  account {account}: {len(days)} day(s), {days[0]} .. {days[-1]}")
        lines.append("    " + ", ".join(days))
    mismatches = report["mismatches"]
    lines.append("")
    lines.append(
        f"Reconciliation mismatches: {len(mismatches)}"
        + (f" (checked {report['reconciled_at']})" if report["reconciled_at"] else "")
    )
    for item in mismatches:
        lines.append(
            f"  {item.get('account_number', '')} {item.get('symbol', ''):<22} journal "
            f"{float(item.get('journal_quantity') or 0):>8g}  broker {float(item.get('broker_quantity') or 0):>8g}  "
            f"{item.get('kind', '')}"
        )
    return "\n".join(lines)


def _default_fetch(db_path: Path) -> Callable[[str, str, date], int]:
    from journal_runner import _fetch_one_day
    from journal_store import JournalStore

    store = JournalStore(db_path)
    return lambda broker, account, day: _fetch_one_day(store, broker, account, day)


def reimport(
    db_path: Path,
    report: dict[str, Any],
    *,
    fetch: Callable[[str, str, date], int],
    max_days: int,
    today: date | None = None,
) -> dict[str, Any]:
    """Retry every FAILED Questrade day through the desk's own self-heal, then rebuild."""
    import journal_coverage
    from journal_store import JournalStore

    store = JournalStore(db_path)
    days = [date.fromisoformat(str(row["day"])) for row in report["failed_days"]]
    if not days:
        return {"attempted": [], "repaired": [], "failed": []}
    reference = today or date.today()
    accounts = sorted({("QUESTRADE", str(row["account_number"])) for row in report["failed_days"]})
    summary = journal_coverage.self_heal(
        store,
        fetch,
        accounts=accounts,
        today=reference,
        lookback_days=(reference - min(days)).days + 1,
        max_days_per_night=max_days,
        failed_only=True,
        include_exhausted=True,
    )
    store.rebuild_trades()
    return summary


def _backup_path(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return db_path.parent / f"{db_path.stem}.pre-qt-reimport-{stamp}{db_path.suffix}.bak"


def main(
    argv: Sequence[str] | None = None,
    *,
    fetch: Callable[[str, str, date], int] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="", help="journal database (default: the desk's journal)")
    parser.add_argument("--reimport", action="store_true", help="show (or with --apply, run) the retry")
    parser.add_argument("--apply", action="store_true", help="with --reimport: contact Questrade and import")
    parser.add_argument(
        "--i-am-the-trader", action="store_true",
        help="required before --apply may touch a database under the live data folder",
    )
    parser.add_argument("--max-days", type=int, default=300, help="most account-days to retry in one run")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.db:
        db_path = Path(args.db).expanduser()
    else:
        from project_paths import JOURNAL_DB_FILE

        db_path = Path(JOURNAL_DB_FILE)
    if not db_path.is_file():
        print(f"No journal database at {db_path}", file=sys.stderr)
        return EXIT_REFUSED_TO_START
    if args.apply and not args.reimport:
        print("--apply only means something with --reimport.", file=sys.stderr)
        return EXIT_REFUSED_TO_START

    report = gap_report(db_path)
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))
    if not args.reimport:
        return EXIT_OK

    if not args.apply:
        count = min(len(report["failed_days"]), max(0, args.max_days))
        print(
            f"\n--reimport would retry {count} account-day(s) through the Questrade API and then "
            "rebuild trades. Nothing was fetched or written. Close the desk, then add --apply."
        )
        return EXIT_OK

    from journal_reclassify import _is_live_store, busy_reasons

    if _is_live_store(db_path) and not args.i_am_the_trader:
        print(
            f"{db_path} is inside the live data folder. Re-importing there is the trader's own "
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
        summary = reimport(db_path, report, fetch=fetch or _default_fetch(db_path), max_days=args.max_days)
    except Exception as exc:  # noqa: BLE001 - the restore is the point
        shutil.copy2(backup, db_path)
        print(f"REFUSED - {type(exc).__name__}: {exc}. The journal was put back.", file=sys.stderr)
        print(f"  the backup is still at {backup}", file=sys.stderr)
        return EXIT_FAILED
    print(
        f"\nRetried {len(summary.get('attempted') or [])} account-day(s): "
        f"{len(summary.get('repaired') or [])} imported, {len(summary.get('failed') or [])} still failed. "
        f"Backup: {backup}"
    )
    for item in summary.get("failed") or []:
        print(f"  still failed {item['account']} {item['day']}: {item['message'][:120]}")
    return EXIT_OK


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
