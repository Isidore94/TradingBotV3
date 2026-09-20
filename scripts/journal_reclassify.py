"""Move stored Questrade fills onto the broker's own words. Dry run by default.

TJ-9Q. Measured 2026-09-19 on a READ-ONLY copy of the trader's journal:

* 226 of 226 Questrade executions carry ``security_type = 'UNKNOWN'`` - the
  ``v1/accounts/{id}/executions`` payload states no type at all - so every
  option fill is priced with a contract multiplier of ONE;
* ``STO``, ``BTC`` and ``COV`` were not in the side vocabulary, so the trader's
  three sold puts opened LONG and their buy-backs ADDED to them instead of
  closing them: three positions that have sat OPEN with ``quantity_closed = 0``
  since June, and one bought put closed for 1/100th of its real loss.

This tool is the ONLY way those stored rows move. It is a DRY RUN by default,
and the dry run reads copies - it does not open the journal it is reporting on,
so it cannot change one byte of it. ``--apply`` takes a byte-exact timestamped
backup first, moves ``security_type``, ``side`` and ``multiplier`` together
through one tested ``JournalStore`` method, rebuilds, verifies, and only then
writes the ``local_settings`` key that lets new fills arrive under the same
convention. If the verification fails it puts the backup back.

**Running ``--apply`` on the live journal is the TRADER's act.** No agent and no
nightly job runs it. It refuses a database under ``C:\\TradingBotData`` unless
``--i-am-the-trader`` is given, and it refuses while the desk or another writer
holds the journal.

WHAT IT NEVER TOUCHES
    ``net_amount`` and ``gross_amount`` (the tax number is the BROKER's and is
    summed from them), the commission SIGN (a broker credit stays a credit),
    quantity, price, symbol, the raw payload and ``execution_uid``. The
    Questrade payload states no ``netAmount`` at all, so this report prints the
    broker's own ``totalCost`` arithmetic beside our recomputed P&L and SAYS
    that ``net_amount`` is absent - it never prints 0.00 for a number nobody
    stated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:  # pragma: no cover - import convenience
    sys.path.insert(0, str(ROOT_DIR))

from journal_file_authority import cash_by_day  # noqa: E402
from journal_identity import (  # noqa: E402
    contract_multiplier,
    group_key,
    group_key_text,
    normalize_security_type,
)
from journal_importers import (  # noqa: E402
    QUESTRADE_INSTRUMENT_SETTING,
    classify_questrade_security_type,
    normalize_side,
)
from journal_store import JournalStore  # noqa: E402

#: Roots that hold the trader's live data. A database under one of these is the
#: real journal until proven otherwise, and this tool will not write it without
#: the trader saying so in as many words.
LIVE_DATA_ROOTS = (Path(r"C:\TradingBotData"), Path(r"\\MINI-PC\Trading Bot Data"))

#: Positions whose status means a new fill could still land in them.
OPEN_STATUSES = frozenset({"OPEN", "CLOSED_PARTIAL"})

#: Sides the broker spells for a sale, used only to read the broker's own cash.
_BROKER_SELL_WORDS = frozenset({"SELL", "SLD", "STO", "STC", "SHORT", "SSHORT", "SELLSHORT"})


# ---------------------------------------------------------------------------
# Reading what the broker said
# ---------------------------------------------------------------------------


def _payload(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        raw = json.loads(str(row.get("raw_json") or "{}"))
    except (json.JSONDecodeError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def broker_cash(payload: Mapping[str, Any]) -> float | None:
    """What the BROKER says this fill did to cash, from its own fields.

    ``totalCost`` is Questrade's own gross figure and ALREADY carries the
    contract multiplier - 145 for one contract at 1.45 - so this is an
    independent statement of the money rather than a restatement of our own
    arithmetic. ``None`` when the payload does not carry it: an absent number is
    reported as absent, never as zero.
    """
    if "totalCost" not in payload:
        return None
    try:
        total = float(payload.get("totalCost"))
    except (TypeError, ValueError):
        return None
    side = str(payload.get("side") or "").strip().upper()
    sign = 1.0 if side in _BROKER_SELL_WORDS else -1.0
    try:
        commission = abs(float(payload.get("commission") or 0.0))
    except (TypeError, ValueError):
        commission = 0.0
    fees = 0.0
    for key in ("secFee", "fees", "executionFee", "canadianExecutionFee", "orderPlacementCommission"):
        try:
            fees += abs(float(payload.get(key) or 0.0))
        except (TypeError, ValueError):
            continue
    return sign * total - commission - fees


# ---------------------------------------------------------------------------
# What would move
# ---------------------------------------------------------------------------


def pending_updates(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every stored Questrade row whose instrument, side or multiplier is wrong.

    The classifier is the SAME function the import seam uses - one rule, so a
    stored row and a new fill can never disagree about what an instrument is. A
    row that already carries a stated type is left alone: this pass completes
    what the endpoint never said, it does not overrule what a broker did say.
    """
    updates: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("broker") or "").strip().upper() != "QUESTRADE":
            continue
        payload = _payload(row)
        stored_type = normalize_security_type(row.get("security_type"))
        stored_side = str(row.get("side") or "").strip().upper()
        candidate = dict(payload)
        candidate.setdefault("symbol", row.get("symbol"))
        candidate.setdefault("side", stored_side)
        new_type = stored_type
        if stored_type == "UNKNOWN":
            new_type = classify_questrade_security_type(candidate)
        new_side = normalize_side(payload.get("side") or stored_side)
        probe = dict(row)
        probe["security_type"] = new_type
        new_multiplier = contract_multiplier(probe)
        try:
            stored_multiplier = float(row.get("multiplier") or 0.0)
        except (TypeError, ValueError):
            stored_multiplier = 0.0
        if (
            new_type == stored_type
            and new_side == stored_side
            and abs(new_multiplier - stored_multiplier) < 1e-9
        ):
            continue
        updates.append(
            {
                "execution_uid": str(row.get("execution_uid") or ""),
                "security_type": new_type,
                "side": new_side,
                "multiplier": new_multiplier,
                "symbol": str(row.get("symbol") or ""),
                "account_number": str(row.get("account_number") or ""),
                "before": {
                    "security_type": stored_type,
                    "side": stored_side,
                    "multiplier": stored_multiplier,
                },
            }
        )
    return updates


# ---------------------------------------------------------------------------
# Reading a journal without writing it
# ---------------------------------------------------------------------------


def _raw_rows(store: JournalStore) -> list[dict[str, Any]]:
    with store.connection() as conn:
        rows = conn.execute("SELECT * FROM raw_executions").fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def _trades(store: JournalStore) -> list[dict[str, Any]]:
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT trade_id, broker, account_number, symbol, security_type, currency, "
            "direction, status, quantity_opened, quantity_closed, gross_pnl, net_pnl "
            "FROM trades"
        ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def _annotation_ids(store: JournalStore) -> list[str]:
    with store.connection() as conn:
        return [str(row[0]) for row in conn.execute("SELECT trade_id FROM trade_annotations")]


def _legs_by_trade(store: JournalStore) -> dict[str, set[str]]:
    with store.connection() as conn:
        rows = conn.execute("SELECT trade_id, execution_uid FROM trade_legs").fetchall()
    legs: dict[str, set[str]] = {}
    for trade_id, uid in rows:
        legs.setdefault(str(trade_id), set()).add(str(uid))
    return legs


def _stranded(store: JournalStore) -> set[str]:
    """Annotations pointing at a trade id that is not in ``trades``."""
    live = {str(trade["trade_id"]) for trade in _trades(store)}
    return {trade_id for trade_id in _annotation_ids(store) if trade_id not in live}


class _Snapshot:
    """One reading of a journal: its rows, its trades and its loose ends."""

    def __init__(self, store: JournalStore) -> None:
        self.rows = _raw_rows(store)
        self.trades = _trades(store)
        self.legs = _legs_by_trade(store)
        self.annotations = _annotation_ids(store)
        self.stranded = _stranded(store)
        self.net_amounts = {
            str(row.get("execution_uid")): row.get("net_amount") for row in self.rows
        }

    def positions(self) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for trade in self.trades:
            key = (
                str(trade.get("broker") or ""),
                str(trade.get("account_number") or ""),
                str(trade.get("symbol") or ""),
            )
            grouped.setdefault(key, []).append(trade)
        for entries in grouped.values():
            entries.sort(key=lambda item: (str(item.get("status")), str(item.get("trade_id"))))
        return grouped

    def open_unknown(self) -> list[dict[str, Any]]:
        return sorted(
            (
                trade
                for trade in self.trades
                if str(trade.get("broker") or "").upper() == "QUESTRADE"
                and normalize_security_type(trade.get("security_type")) == "UNKNOWN"
                and str(trade.get("status") or "").upper() in OPEN_STATUSES
            ),
            key=lambda item: str(item.get("symbol") or ""),
        )


def _copy_database(source: Path, target: Path) -> Path:
    """A byte-exact copy, with the sidecars sqlite may have left beside it."""
    shutil.copy2(source, target)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(source) + suffix)
        if sidecar.is_file():
            shutil.copy2(sidecar, Path(str(target) + suffix))
    return target


def _simulate(
    source: Path, updates: Sequence[Mapping[str, Any]], workdir: Path, name: str
) -> tuple[_Snapshot, dict[str, set[str]], set[str]]:
    """Run the whole pass on a COPY and report what it did to the annotations.

    Returns the rebuilt snapshot, the leg map as it was BEFORE the rebuild, and
    the annotations the rebuild left pointing at nothing.
    """
    copy = _copy_database(source, workdir / name)
    store = JournalStore(copy)
    legs_before = _legs_by_trade(store)
    before_stranded = _stranded(store)
    store.reclassify_executions(updates, refresh_tags=False)
    after = _Snapshot(store)
    return after, legs_before, after.stranded - before_stranded


def plan_reclassify(db_path: Path, workdir: Path) -> dict[str, Any]:
    """What ``--apply`` would do, measured on copies and written to nothing.

    The refusal loop is the point: a contract sold, bought back, sold again and
    bought back again is ONE open position today and TWO closed round trips once
    the sides are right, each holding half of the old position's executions - so
    the re-key sees a tie and will not guess which of them the trader's note
    belongs to. Rather than reclassify it and let the note fall off, this leaves
    that position exactly as it found it and says so.
    """
    before_copy = _copy_database(db_path, workdir / "before.sqlite3")
    before_store = JournalStore(before_copy)
    before = _Snapshot(before_store)

    updates = pending_updates(before.rows)
    refused_uids: set[str] = set()
    refused_trades: list[dict[str, Any]] = []
    live_updates = list(updates)
    after, legs_before, stranded = _simulate(db_path, live_updates, workdir, "after-0.sqlite3")
    attempt = 0
    while stranded and attempt < 3:
        attempt += 1
        for trade_id in sorted(stranded):
            uids = legs_before.get(trade_id, set())
            refused_uids |= uids
            trade = next(
                (item for item in before.trades if str(item.get("trade_id")) == trade_id), {}
            )
            refused_trades.append(
                {
                    "trade_id": trade_id,
                    "symbol": str(trade.get("symbol") or ""),
                    "account_number": str(trade.get("account_number") or ""),
                    "executions": sorted(uids),
                    "reason": (
                        "the rebuilt trades share its executions evenly, so the trader's "
                        "annotation cannot be carried without guessing"
                    ),
                }
            )
        live_updates = [item for item in updates if item["execution_uid"] not in refused_uids]
        after, legs_before, stranded = _simulate(
            db_path, live_updates, workdir, f"after-{attempt}.sqlite3"
        )
    return {
        "db_path": db_path,
        "before": before,
        "after": after,
        "updates": live_updates,
        "refused_updates": [item for item in updates if item["execution_uid"] in refused_uids],
        "refused_trades": refused_trades,
        "cash_days": _cash_day_changes(before.rows, after.rows),
    }


def _cash_day_changes(
    before_rows: Sequence[Mapping[str, Any]], after_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Every (account, day) whose file-authority signed cash moves.

    That comparison is what decides whether a broker's own statement takes a day
    over from the live sync, and both halves of this change reach it: an option
    fill stops being priced at a multiplier of one, and a cover stops being
    counted as money coming in.
    """
    before = cash_by_day(before_rows)
    after = cash_by_day(after_rows)
    changes: list[dict[str, Any]] = []
    for key in sorted(set(before) | set(after)):
        old = before.get(key, (0.0, 0))[0]
        new = after.get(key, (0.0, 0))[0]
        if abs(new - old) > 1e-9:
            changes.append(
                {"account": key[0], "day": key[1].isoformat(), "before": old, "after": new}
            )
    return changes


# ---------------------------------------------------------------------------
# Saying it out loud
# ---------------------------------------------------------------------------


def _money(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "not stated"


def _trade_line(trade: Mapping[str, Any]) -> str:
    return (
        f"{str(trade.get('direction') or ''):<5} "
        f"{str(trade.get('status') or ''):<14} "
        f"qty {float(trade.get('quantity_opened') or 0.0):g}/"
        f"{float(trade.get('quantity_closed') or 0.0):g}  "
        f"recomputed P&L {_money(trade.get('net_pnl')):>14}"
    )


def _position_broker_cash(rows: Sequence[Mapping[str, Any]], account: str, symbol: str) -> float | None:
    total = 0.0
    seen = False
    for row in rows:
        if str(row.get("account_number") or "") != account:
            continue
        if str(row.get("symbol") or "") != symbol:
            continue
        cash = broker_cash(_payload(row))
        if cash is None:
            return None
        seen = True
        total += cash
    return total if seen else None


def print_report(plan: Mapping[str, Any], *, applied: bool, stream=None) -> None:
    out = stream or sys.stdout
    before: _Snapshot = plan["before"]
    after: _Snapshot = plan["after"]
    def write(text: str = "") -> None:
        print(text, file=out)

    write(f"Journal: {plan['db_path']}")
    write(
        "APPLIED - the stored rows have moved."
        if applied
        else "DRY RUN - read from copies; not one byte of this database was opened for writing."
    )
    write()
    write(
        f"Questrade fills to move: {len(plan['updates'])}"
        f"  (refused: {len(plan['refused_updates'])})"
    )
    write(
        "Questrade states no netAmount and no grossAmount on any execution, so the "
        "broker's cash below is its own totalCost arithmetic;"
    )
    write("net_amount is ABSENT for these fills - it is not zero, and nothing here writes it.")
    write()

    open_unknown = before.open_unknown()
    write(
        f"Open UNKNOWN Questrade positions a new fill could close: {len(open_unknown)}"
    )
    for trade in open_unknown:
        write(
            f"  {str(trade.get('symbol') or ''):<22} "
            f"{str(trade.get('direction') or ''):<5} {str(trade.get('status') or ''):<14} UNKNOWN"
        )
    if not open_unknown:
        write("  (none - a new fill cannot be split from a position it belongs to)")
    write()

    write("BEFORE and AFTER, by position")
    before_positions = before.positions()
    after_positions = after.positions()
    for key in sorted(set(before_positions) | set(after_positions)):
        broker, account, symbol = key
        old_entries = before_positions.get(key, [])
        new_entries = after_positions.get(key, [])
        old_text = [
            (group_key_text(_group_of(entry)), _trade_line(entry)) for entry in old_entries
        ]
        new_text = [
            (group_key_text(_group_of(entry)), _trade_line(entry)) for entry in new_entries
        ]
        if old_text == new_text:
            continue
        write(f"  {broker} {account} {symbol}")
        for group, line in old_text:
            write(f"    before  {group}")
            write(f"            {line}")
        if not old_text:
            write("    before  (no trade)")
        for group, line in new_text:
            write(f"    after   {group}")
            write(f"            {line}")
        if not new_text:
            write("    after   (no trade)")
        cash = _position_broker_cash(before.rows, account, symbol)
        write(
            "    broker  cash (payload totalCost) "
            + (f"{cash:>14.6f}" if cash is not None else "   not stated")
            + "   |  broker net_amount: not stated by Questrade"
        )
    write()

    if plan["refused_trades"]:
        write("REFUSED - left exactly as found, because an annotation cannot be carried:")
        for item in plan["refused_trades"]:
            write(f"  {item['symbol']} ({item['account_number']}) trade {item['trade_id']}")
            write(f"    {item['reason']}")
            write(f"    {len(item['executions'])} execution(s) left on the old convention")
        write()

    write("File-authority signed cash that moves (account, day):")
    for change in plan["cash_days"]:
        write(
            f"  {change['account']} {change['day']}  "
            f"{change['before']:>14.4f} -> {change['after']:>14.4f}"
        )
    if not plan["cash_days"]:
        write("  (none)")
    covers = _stored_cover_rows(before.rows)
    if covers:
        write(
            f"  ...and {len(covers)} stored fill(s) still spelled COV, whose cash this build "
            "ALREADY reads correctly:"
        )
        write(
            "     the buy set held COVER and not COV, so each of them used to count as money "
            "coming IN."
        )
        write(
            "     That correction is in the code, not in this run, so it is not in the "
            "before/after above."
        )
    write()

    total_before = sum(float(trade.get("net_pnl") or 0.0) for trade in before.trades)
    total_after = sum(float(trade.get("net_pnl") or 0.0) for trade in after.trades)
    write(
        f"Total recomputed P&L across every trade: {total_before:.6f} -> {total_after:.6f}"
    )
    write(
        f"Trades: {len(before.trades)} -> {len(after.trades)};  "
        f"annotations: {len(before.annotations)} (stranded before: {len(before.stranded)}, "
        f"after: {len(after.stranded)})"
    )
    if not applied:
        write()
        write("Nothing was written. Run again with --apply to move the stored rows.")


def _stored_cover_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Fills still holding Questrade's own ``COV`` spelling."""
    return [row for row in rows if str(row.get("side") or "").strip().upper() == "COV"]


def _group_of(trade: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return group_key(
        {
            "broker": trade.get("broker"),
            "account_number": trade.get("account_number"),
            "symbol": trade.get("symbol"),
            "security_type": trade.get("security_type"),
            "currency": trade.get("currency"),
        }
    )


# ---------------------------------------------------------------------------
# Refusing to run at a bad moment
# ---------------------------------------------------------------------------


def _is_live_store(db_path: Path) -> bool:
    resolved = Path(db_path).expanduser().resolve()
    for root in LIVE_DATA_ROOTS:
        try:
            if resolved.is_relative_to(root.resolve(strict=False)):
                return True
        except (OSError, ValueError):
            continue
    return False


def _is_the_desks_journal(db_path: Path) -> bool:
    try:
        from journal_store import JOURNAL_DB_FILE

        return Path(db_path).expanduser().resolve() == Path(JOURNAL_DB_FILE).resolve()
    except Exception:  # pragma: no cover - defensive
        return False


def _desk_is_running() -> bool:
    """Whether another process holds this machine's desk slot.

    Only asked about the desk's OWN journal: a copy in a scratch directory is
    nobody's live store, and refusing to reclassify one because a desk is open
    would make the tool untestable for the exact case it exists for.
    """
    try:
        from local_writer_lock import LocalLockUnavailable, local_writer_lock
        from single_instance import DESK_LOCK_KEY
    except Exception:  # pragma: no cover - defensive
        return False
    try:
        with local_writer_lock(DESK_LOCK_KEY, timeout_seconds=0.0):
            return False
    except LocalLockUnavailable as exc:
        if "no machine-local exclusion primitive" in str(exc):
            return False
        return True
    except Exception:  # pragma: no cover - defensive
        return False


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="journal_reclassify",
        description=(
            "Move stored Questrade fills onto the broker's own instrument and side words. "
            "Dry run by default."
        ),
    )
    parser.add_argument("--db", required=True, help="the journal database to read")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report only (the default); reads copies and writes nothing",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="back up, then move the stored rows and turn the import convention on",
    )
    parser.add_argument(
        "--i-am-the-trader",
        action="store_true",
        help="required before --apply may touch a database under the live data folder",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _backup_path(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return db_path.parent / f"{db_path.stem}.pre-tj9q-{stamp}{db_path.suffix}.bak"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = Path(args.db).expanduser()
    if args.apply and args.dry_run:
        print("--dry-run and --apply contradict each other; pick one.", file=sys.stderr)
        return 2
    if not db_path.is_file():
        print(f"No journal database at {db_path}", file=sys.stderr)
        return 2
    if args.apply and _is_live_store(db_path) and not args.i_am_the_trader:
        print(
            f"{db_path} is inside the live data folder. Applying there is the trader's own "
            "act: re-run with --i-am-the-trader if you are the trader.",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="tj9q-reclassify-") as scratch:
        workdir = Path(scratch)
        plan = plan_reclassify(db_path, workdir)
        if not args.apply:
            print_report(plan, applied=False)
            return 0
        return _apply(plan, db_path, workdir)


def _apply(plan: Mapping[str, Any], db_path: Path, workdir: Path) -> int:
    from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path

    if _is_the_desks_journal(db_path) and _desk_is_running():
        print(
            "The desk is running and it writes this journal. Close the Trading Desk "
            "window, then run this again.",
            file=sys.stderr,
        )
        return 3

    try:
        guard = local_writer_lock(lock_key_for_path(db_path), timeout_seconds=0.0)
        guard.__enter__()
    except LocalLockUnavailable as exc:
        if "no machine-local exclusion primitive" not in str(exc):
            print(
                "Something else is writing this journal right now (the desk, or a nightly "
                f"import): {exc}",
                file=sys.stderr,
            )
            return 3
        guard = None
        print(
            "Warning: this machine has no writer-exclusion primitive, so nothing can prove "
            "the journal is idle. Continuing.",
            file=sys.stderr,
        )
    try:
        return _apply_locked(plan, db_path, workdir)
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)


def _apply_locked(plan: Mapping[str, Any], db_path: Path, workdir: Path) -> int:
    before: _Snapshot = plan["before"]
    backup = _backup_path(db_path)
    _copy_database(db_path, backup)

    store = JournalStore(db_path)
    try:
        store.reclassify_executions(plan["updates"], refresh_tags=False)
        after = _Snapshot(store)
        problems = _verify(before, after)
    except Exception as exc:  # noqa: BLE001 - the restore is the point
        problems = [f"the reclassify raised {type(exc).__name__}: {exc}"]
        after = before

    if problems:
        shutil.copy2(backup, db_path)
        print("REFUSED - the journal was put back exactly as it was:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(f"  the backup is still at {backup}", file=sys.stderr)
        return 1

    applied_plan = dict(plan)
    applied_plan["after"] = after
    applied_plan["cash_days"] = _cash_day_changes(before.rows, after.rows)
    print_report(applied_plan, applied=True)
    print(f"Backup of the journal as it was: {backup}")

    # LAST, and only now: new fills may arrive under the same convention as the
    # rows this run just moved. Until this line the desk and the nightly import
    # keep storing exactly what they stored yesterday.
    from project_paths import save_local_setting

    save_local_setting(QUESTRADE_INSTRUMENT_SETTING, True)
    print(
        f"local_settings['{QUESTRADE_INSTRUMENT_SETTING}'] is now true: new Questrade fills "
        "will be stored the same way."
    )
    return 0


def _verify(before: _Snapshot, after: _Snapshot) -> list[str]:
    """Everything that must still be true, checked before the run is kept."""
    problems: list[str] = []
    if set(before.net_amounts) != set(after.net_amounts):
        problems.append("the set of executions changed")
    else:
        moved = [
            uid
            for uid, value in before.net_amounts.items()
            if _amount(value) != _amount(after.net_amounts.get(uid))
        ]
        if moved:
            problems.append(f"{len(moved)} broker-stated amount(s) changed")
    newly_stranded = after.stranded - before.stranded
    if newly_stranded:
        problems.append(
            f"{len(newly_stranded)} annotation(s) would be left pointing at a trade that "
            "no longer exists"
        )
    if len(after.annotations) != len(before.annotations):
        problems.append(
            f"{len(before.annotations)} annotation(s) went in and {len(after.annotations)} "
            "came out"
        )
    return problems


def _amount(value: Any) -> str:
    if value is None:
        return "absent"
    try:
        return f"{float(value):.10f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":  # pragma: no cover - console entry
    raise SystemExit(main())
