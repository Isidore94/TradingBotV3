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
import stat
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

#: Exit codes, because a script that says "0" for a half-done job teaches a
#: person to stop reading.
#:
#: 0  the run did what it said and the import convention is now on;
#: 1  the verification failed and the journal was put back, byte for byte;
#: 2  it would not start (no such database, contradictory flags, the live data
#:    folder without ``--i-am-the-trader``);
#: 3  something else is writing this journal right now - nothing was done;
#: 4  the rows that could move DID move and are correct, but the switch stayed
#:    OFF because something is still on the old convention. Re-runnable.
EXIT_OK = 0
EXIT_VERIFY_FAILED = 1
EXIT_REFUSED_TO_START = 2
EXIT_BUSY = 3
EXIT_SWITCH_STAYED_OFF = 4

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
            "direction, status, quantity_opened, quantity_closed, gross_pnl, net_pnl, "
            "opened_at, closed_at, trade_date "
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


#: The machine's own trade-keyed tables, and how a reader of each reaches the
#: trade: ``literal`` matches ``trade_id`` directly (``list_ai_enrichment``, the
#: note-lane join), ``alias`` follows ``trade_aliases`` because the rows
#: themselves are immutable and are never rewritten (``opportunity_events``).
MACHINE_TABLES = (
    ("ai_trade_enrichment", "literal"),
    ("note_lane_verdicts", "literal"),
    ("opportunity_events", "alias"),
)


def _machine_dead_counts(store: JournalStore) -> dict[str, int]:
    """How many rows in each machine table point at a trade that is not there.

    Measured the way each table's own reader reads it, so the number means "the
    Journal page cannot show this row", not "a column looks odd".
    """
    with store.connection() as conn:
        live = {str(row[0]) for row in conn.execute("SELECT trade_id FROM trades")}
        aliases: dict[str, str] = {}
        for old, new in conn.execute("SELECT old_trade_id, new_trade_id FROM trade_aliases"):
            aliases[str(old)] = str(new)
        counts: dict[str, int] = {}
        for table, how in MACHINE_TABLES:
            dead = 0
            for (trade_id,) in conn.execute(
                f"SELECT trade_id FROM {table} WHERE COALESCE(trade_id, '') != ''"
            ):
                current = str(trade_id)
                if how == "alias":
                    seen = {current}
                    while current not in live and current in aliases:
                        current = aliases[current]
                        if current in seen:
                            break
                        seen.add(current)
                if current not in live:
                    dead += 1
            counts[table] = dead
    return counts


class _Snapshot:
    """One reading of a journal: its rows, its trades and its loose ends."""

    def __init__(self, store: JournalStore) -> None:
        self.rows = _raw_rows(store)
        self.trades = _trades(store)
        self.legs = _legs_by_trade(store)
        self.annotations = _annotation_ids(store)
        self.stranded = _stranded(store)
        self.machine_dead = _machine_dead_counts(store)
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
            # Chronological, so a re-key cannot reshuffle a position's own list
            # and make an unchanged position look like a changed one.
            entries.sort(
                key=lambda item: (
                    str(item.get("opened_at") or ""),
                    str(item.get("closed_at") or ""),
                    str(item.get("trade_id") or ""),
                )
            )
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


def _copy_database(source: Path, target: Path, *, writable: bool = False) -> Path:
    """A byte-exact copy, with the sidecars sqlite may have left beside it.

    ``writable`` clears the read-only attribute on the COPY and never on the
    source. ``shutil.copy2`` preserves the attribute, and sqlite then raises
    "attempt to write a readonly database" the moment ``JournalStore`` opens the
    working copy to migrate it - which used to come out of a dry run as a raw
    traceback. A trader who has write-protected their journal deserves a report,
    not a stack trace, and the file they protected is still never touched.
    """
    shutil.copy2(source, target)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(source) + suffix)
        if sidecar.is_file():
            shutil.copy2(sidecar, Path(str(target) + suffix))
            if writable:
                _make_writable(Path(str(target) + suffix))
    if writable:
        _make_writable(target)
    return target


def _make_writable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IWRITE)
    except OSError:  # pragma: no cover - defensive
        pass


def _simulate(
    source: Path, updates: Sequence[Mapping[str, Any]], workdir: Path, name: str
) -> tuple[_Snapshot, dict[str, set[str]], set[str], dict[str, dict[str, int]]]:
    """Run the whole pass on a COPY and report what it did to the annotations.

    Returns the rebuilt snapshot, the leg map as it was BEFORE the rebuild, the
    annotations the rebuild left pointing at nothing, and the machine-row carry.
    """
    copy = _copy_database(source, workdir / name, writable=True)
    store = JournalStore(copy)
    legs_before = _legs_by_trade(store)
    before_stranded = _stranded(store)
    report = store.reclassify_executions(updates, refresh_tags=False)
    after = _Snapshot(store)
    return after, legs_before, after.stranded - before_stranded, report.get("machine_rows") or {}


def plan_reclassify(db_path: Path, workdir: Path) -> dict[str, Any]:
    """What ``--apply`` would do, measured on copies and written to nothing.

    The refusal loop is the point: a contract sold, bought back, sold again and
    bought back again is ONE open position today and TWO closed round trips once
    the sides are right, each holding half of the old position's executions - so
    the re-key sees a tie and will not guess which of them the trader's note
    belongs to. Rather than reclassify it and let the note fall off, this leaves
    that position exactly as it found it and says so.
    """
    before_copy = _copy_database(db_path, workdir / "before.sqlite3", writable=True)
    before_store = JournalStore(before_copy)
    before = _Snapshot(before_store)

    updates = pending_updates(before.rows)
    refused_uids: set[str] = set()
    refused_trades: list[dict[str, Any]] = []
    live_updates = list(updates)
    after, legs_before, stranded, machine_rows = _simulate(
        db_path, live_updates, workdir, "after-0.sqlite3"
    )
    stranded = stranded | _unreachable_narration(machine_rows)
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
                        "the rebuilt trades share its executions evenly, so what is written "
                        "about it (the trader's annotation, or narration nothing regenerates) "
                        "cannot be carried without guessing"
                    ),
                }
            )
        live_updates = [item for item in updates if item["execution_uid"] not in refused_uids]
        after, legs_before, stranded, machine_rows = _simulate(
            db_path, live_updates, workdir, f"after-{attempt}.sqlite3"
        )
        stranded = stranded | _unreachable_narration(machine_rows)
    plan = {
        "db_path": db_path,
        "before": before,
        "after": after,
        "updates": live_updates,
        "refused_updates": [item for item in updates if item["execution_uid"] in refused_uids],
        "refused_trades": refused_trades,
        "cash_days": _cash_day_changes(before.rows, after.rows),
        "machine_rows": machine_rows,
    }
    plan["switch"] = switch_decision(plan)
    return plan


#: The machine table whose rows nothing regenerates. A position whose narration
#: cannot be carried is REFUSED, exactly like one whose trader annotation cannot
#: be - the derived note verdicts are rewritten by the note lane and the
#: immutable events are reached through an alias, but this one would simply be
#: gone from the Journal page.
NON_REGENERABLE_TABLE = "ai_trade_enrichment"


def _unreachable_narration(machine_rows: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """Old trade ids whose AI narration the overlap rule could not carry."""
    counts = machine_rows.get(NON_REGENERABLE_TABLE) or {}
    return {str(item) for item in (counts.get("left_trade_ids") or [])}


def switch_decision(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Whether new fills may arrive under the corrected convention - and why not.

    TWO conditions, and both are about the same danger: a journal that is half
    in each convention. The switch may only go on when

    * nothing was REFUSED - a refused position keeps the old spelling, so a new
      fill classified ``OPT`` would open a SECOND position for that contract and
      the refused one could never close; and
    * no open ``UNKNOWN`` Questrade position is left at all - the same split by
      another route (a position this pass could not type, that a new fill would
      try to close).

    Otherwise the rows that did move stay moved - they are correct and
    consistent - and the switch stays OFF until the trader has dealt with what
    is named here.
    """
    after: _Snapshot = plan["after"]
    refused = list(plan.get("refused_trades") or [])
    open_unknown = after.open_unknown()
    blockers: list[str] = []
    if refused:
        for item in refused:
            blockers.append(
                f"{item['symbol']} was REFUSED and still holds the old spelling"
            )
    for trade in open_unknown:
        blockers.append(
            f"{trade.get('symbol')} is still an OPEN UNKNOWN position "
            f"({trade.get('direction')} {trade.get('status')})"
        )
    return {"on": not blockers, "blockers": blockers}


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


def _position_change(
    old_entries: Sequence[Mapping[str, Any]], new_entries: Sequence[Mapping[str, Any]]
) -> str:
    """``""`` (nothing moved), ``"type"`` (only the instrument) or ``"matters"``.

    A position whose only change is ``UNKNOWN -> STK`` inside its group key has
    the same direction, the same status and the same money to the cent. On the
    trader's journal there are about 200 of those and four that matter, and a
    report that lists them all in one flat list is a report nobody reads.

    Compared as a SET of trades, never as a list: a re-key renames every trade
    in the position, and a position whose trades merely come back in a different
    order has not changed - saying it did would put nine positions on page one
    where four belong.
    """
    def shape(entries: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
        return sorted(
            (
                str(item.get("direction") or ""),
                str(item.get("status") or ""),
                f"{float(item.get('net_pnl') or 0.0):.4f}",
            )
            for item in entries
        )

    def typed(entries: Sequence[Mapping[str, Any]]) -> list[str]:
        return sorted(group_key_text(_group_of(item)) for item in entries)

    if shape(old_entries) != shape(new_entries):
        return "matters"
    if typed(old_entries) != typed(new_entries):
        return "type"
    return ""


def print_report(
    plan: Mapping[str, Any], *, applied: bool, stream=None, verbose: bool = False
) -> None:
    """The report, decision first.

    Page one is what a person has to decide on: what moves that is money or
    direction, what was refused, whether the switch goes on, where the backup is
    and how to undo the whole thing. Everything else is below it, and the long
    per-position list is behind ``--verbose``.
    """
    out = stream or sys.stdout
    before: _Snapshot = plan["before"]
    after: _Snapshot = plan["after"]
    switch = plan.get("switch") or switch_decision(plan)

    def write(text: str = "") -> None:
        print(text, file=out)

    before_positions = before.positions()
    after_positions = after.positions()
    matters: list[tuple[str, str, str]] = []
    type_only: list[str] = []
    for key in sorted(set(before_positions) | set(after_positions)):
        broker, account, symbol = key
        kind = _position_change(before_positions.get(key, []), after_positions.get(key, []))
        if kind == "matters":
            matters.append(key)
        elif kind == "type":
            type_only.append(symbol)

    total_before = sum(float(trade.get("net_pnl") or 0.0) for trade in before.trades)
    total_after = sum(float(trade.get("net_pnl") or 0.0) for trade in after.trades)

    write(f"Journal: {plan['db_path']}")
    write(
        "APPLIED - the stored rows have moved."
        if applied
        else "DRY RUN - read from copies; this database was never opened for writing."
    )
    write()
    write("=" * 78)
    write("WHAT CHANGES")
    write("=" * 78)
    write(
        f"  Questrade fills moved: {len(plan['updates'])}"
        f"   refused: {len(plan['refused_updates'])}"
        f"   positions whose money or direction moves: {len(matters)}"
    )
    write(
        f"  Positions that only gain an instrument type (no money, no direction): "
        f"{len(type_only)}" + ("" if verbose else "   [--verbose lists them]")
    )
    write(f"  Total recomputed P&L across every trade: {total_before:.2f} -> {total_after:.2f}")
    write()
    for key in matters:
        broker, account, symbol = key
        write(f"  {symbol}   instrument {_types_text(before_positions.get(key, []))} -> "
              f"{_types_text(after_positions.get(key, []))}")
        for entry in before_positions.get(key, []) or [None]:
            write(f"      before   {_trade_line(entry) if entry else '(no trade)'}")
        for entry in after_positions.get(key, []) or [None]:
            write(f"      after    {_trade_line(entry) if entry else '(no trade)'}")
        cash = _position_broker_cash(before.rows, account, symbol)
        write(
            "      broker   cash (payload totalCost) "
            + (f"{cash:.6f}" if cash is not None else "not stated")
            + "  |  net_amount: not stated by Questrade"
        )
    if not matters:
        write("  (no position's money or direction moves)")
    write()

    write("=" * 78)
    write("REFUSED" if plan["refused_trades"] else "REFUSED: none")
    write("=" * 78)
    for item in plan["refused_trades"]:
        write(f"  {item['symbol']} ({item['account_number']}) - left exactly as it was found")
        write(f"    {item['reason']}")
        write(f"    {len(item['executions'])} execution(s) keep the old spelling")
    write()

    write("=" * 78)
    write("THE SWITCH (whether NEW fills arrive under the corrected convention)")
    write("=" * 78)
    if switch["on"]:
        write(
            "  ON."
            + (
                "  New Questrade fills are stored the same way from now on."
                if applied
                else "  --apply would turn it on."
            )
        )
    else:
        write("  OFF, and it stays off. New fills keep arriving the old way.")
        write("  Because a journal may never be half in each convention, and:")
        for blocker in switch["blockers"][:12]:
            write(f"    - {blocker}")
        if len(switch["blockers"]) > 12:
            write(f"    - ...and {len(switch['blockers']) - 12} more")
        write(
            "  Everything this run did reclassify IS reclassified and correct; only the "
            "switch is held back."
        )
    write()

    write("=" * 78)
    write("BACKUP AND UNDO")
    write("=" * 78)
    if applied:
        write(f"  A byte-exact copy of the journal as it was: {plan.get('backup', '(none)')}")
        write("  To undo: close the desk, copy that file back over the journal, done.")
    else:
        write("  Nothing was written. --apply takes a byte-exact backup before it writes.")
    write(
        "  If the machine loses power in the middle of --apply, run --apply again: it "
        "repairs a journal whose rows moved but whose trades were never rebuilt."
    )
    write()

    write("-" * 78)
    write("DETAIL")
    write("-" * 78)
    write(
        "Questrade states no netAmount and no grossAmount on any execution, so the broker's "
        "cash above is its own totalCost arithmetic; net_amount is ABSENT for these fills - "
        "it is not zero, and nothing here writes it."
    )
    open_unknown_before = before.open_unknown()
    open_unknown_after = after.open_unknown()
    write(
        f"Open UNKNOWN Questrade positions a new fill could close: "
        f"{len(open_unknown_before)} before, {len(open_unknown_after)} after."
    )
    for trade in open_unknown_before:
        write(
            f"  {str(trade.get('symbol') or ''):<22} "
            f"{str(trade.get('direction') or ''):<5} {str(trade.get('status') or ''):<14} UNKNOWN"
        )
    machine = plan.get("machine_rows") or {}
    if machine:
        write("Machine rows carried onto the rebuilt trades (the trader's own are separate):")
        for table in sorted(machine):
            counts = machine[table]
            write(
                f"  {table:<22} carried {counts.get('carried', 0)}, "
                f"left where they were {counts.get('left', 0)}, "
                f"dropped (derived, regenerated) {counts.get('dropped', 0)}, "
                f"already dead before this run {counts.get('already_dead', 0)}"
            )
            named = list(counts.get("symbols") or [])
            if named:
                write(f"      not carried: {', '.join(named[:10])}")
        write(
            "  A row is only ever moved onto the trade that holds MOST of its executions, "
            "and a tie is never guessed."
        )
    write(
        f"Trades: {len(before.trades)} -> {len(after.trades)};  "
        f"annotations: {len(before.annotations)} (stranded before: {len(before.stranded)}, "
        f"after: {len(after.stranded)})"
    )
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

    if verbose:
        write("Every position this run touches, before and after:")
        for key in sorted(set(before_positions) | set(after_positions)):
            broker, account, symbol = key
            old_entries = before_positions.get(key, [])
            new_entries = after_positions.get(key, [])
            if not _position_change(old_entries, new_entries):
                continue
            write(f"  {broker} {account} {symbol}")
            for entry in old_entries:
                write(f"    before  {group_key_text(_group_of(entry))}")
                write(f"            {_trade_line(entry)}")
            if not old_entries:
                write("    before  (no trade)")
            for entry in new_entries:
                write(f"    after   {group_key_text(_group_of(entry))}")
                write(f"            {_trade_line(entry)}")
            if not new_entries:
                write("    after   (no trade)")
        write()
    elif type_only:
        write(
            f"{len(type_only)} position(s) change instrument type only: "
            + ", ".join(sorted(type_only)[:12])
            + ("..." if len(type_only) > 12 else "")
        )
        write("Run again with --verbose for the full before/after list.")
        write()

    if not applied:
        write("Nothing was written. Run again with --apply to move the stored rows.")


def _types_text(entries: Sequence[Mapping[str, Any]]) -> str:
    """The instrument type(s) a position's trades carry, as one readable word."""
    types = sorted({normalize_security_type(item.get("security_type")) for item in entries})
    return "/".join(types) if types else "(none)"


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


def _lock_is_held(key: str) -> bool:
    """Whether some other process on this machine holds ``key`` right now.

    Fails OPEN when the box has no exclusion primitive: refusing the trader's
    own repair because nothing can prove the machine is idle is a worse failure
    than the overlap it would prevent, and the backup is the real safety net.
    """
    try:
        from local_writer_lock import LocalLockUnavailable, local_writer_lock
    except Exception:  # pragma: no cover - defensive
        return False
    try:
        with local_writer_lock(key, timeout_seconds=0.0):
            return False
    except LocalLockUnavailable as exc:
        if "no machine-local exclusion primitive" in str(exc):
            return False
        return True
    except Exception:  # pragma: no cover - defensive
        return False


def _desk_is_running() -> bool:
    """Whether another process holds this machine's desk slot."""
    try:
        from single_instance import DESK_LOCK_KEY
    except Exception:  # pragma: no cover - defensive
        return False
    return _lock_is_held(DESK_LOCK_KEY)


def _nightly_jobs_are_running() -> bool:
    """Whether the overnight runner holds its lock.

    The nightly journal import (``journal_runner.run_journal_import_for_date``)
    runs INSIDE that runner, and nothing in it takes a lock on the journal file
    itself - so the per-path lock this tool holds would not have noticed it.
    """
    try:
        from ai_jobs.runner import RUNNER_LOCK_KEY
    except Exception:  # pragma: no cover - defensive
        return False
    return _lock_is_held(RUNNER_LOCK_KEY)


def busy_reasons(db_path: Path) -> list[str]:
    """Why ``--apply`` must not run right now, in the trader's own words.

    The desk and the overnight runner are only asked about when the target IS
    the journal they write - the desk's own. A copy in a scratch directory is
    nobody's live store, and refusing to reclassify one because a desk happens
    to be open would make this tool untestable for the exact case it exists for.
    """
    reasons: list[str] = []
    if _is_the_desks_journal(db_path) or _is_live_store(db_path):
        if _desk_is_running():
            reasons.append(
                "The Trading Desk is running and it writes this journal. Close its window."
            )
        if _nightly_jobs_are_running():
            reasons.append(
                "The overnight AI runner is going, and the nightly journal import runs "
                "inside it. Wait for it to finish (it is done by about 06:00 Pacific)."
            )
    return reasons


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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="list every position that changes, including the type-only ones",
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
        return EXIT_REFUSED_TO_START
    if not db_path.is_file():
        print(f"No journal database at {db_path}", file=sys.stderr)
        return EXIT_REFUSED_TO_START
    if args.apply and _is_live_store(db_path) and not args.i_am_the_trader:
        print(
            f"{db_path} is inside the live data folder. Applying there is the trader's own "
            "act: re-run with --i-am-the-trader if you are the trader.",
            file=sys.stderr,
        )
        return EXIT_REFUSED_TO_START

    with tempfile.TemporaryDirectory(prefix="tj9q-reclassify-") as scratch:
        workdir = Path(scratch)
        plan = plan_reclassify(db_path, workdir)
        if not args.apply:
            print_report(plan, applied=False, verbose=args.verbose)
            return EXIT_OK
        return _apply(plan, db_path, workdir, verbose=args.verbose)


def _apply(plan: Mapping[str, Any], db_path: Path, workdir: Path, *, verbose: bool = False) -> int:
    from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path

    reasons = busy_reasons(db_path)
    if reasons:
        print("Not now - something else is writing this journal:", file=sys.stderr)
        for reason in reasons:
            print(f"  {reason}", file=sys.stderr)
        return EXIT_BUSY

    try:
        guard = local_writer_lock(lock_key_for_path(db_path), timeout_seconds=0.0)
        guard.__enter__()
    except LocalLockUnavailable as exc:
        if "no machine-local exclusion primitive" not in str(exc):
            print(
                "Something else is writing this journal right now (another copy of this "
                f"tool, or the desk): {exc}",
                file=sys.stderr,
            )
            return EXIT_BUSY
        guard = None
        print(
            "Warning: this machine has no writer-exclusion primitive, so nothing can prove "
            "the journal is idle. Continuing.",
            file=sys.stderr,
        )
    try:
        return _apply_locked(plan, db_path, workdir, verbose=verbose)
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)


def _apply_locked(
    plan: Mapping[str, Any], db_path: Path, workdir: Path, *, verbose: bool = False
) -> int:
    before: _Snapshot = plan["before"]
    backup = _backup_path(db_path)
    _copy_database(db_path, backup)

    store = JournalStore(db_path)
    machine_rows: dict[str, Any] = {}
    try:
        report = store.reclassify_executions(plan["updates"], refresh_tags=False)
        machine_rows = report.get("machine_rows") or {}
        after = _Snapshot(store)
        problems = _verify(before, after, machine_rows)
    except Exception as exc:  # noqa: BLE001 - the restore is the point
        problems = [f"the reclassify raised {type(exc).__name__}: {exc}"]
        after = before

    if problems:
        shutil.copy2(backup, db_path)
        print("REFUSED - the journal was put back exactly as it was:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(f"  the backup is still at {backup}", file=sys.stderr)
        return EXIT_VERIFY_FAILED

    applied_plan = dict(plan)
    applied_plan["after"] = after
    applied_plan["cash_days"] = _cash_day_changes(before.rows, after.rows)
    applied_plan["machine_rows"] = machine_rows
    applied_plan["backup"] = str(backup)
    # Decided on what the journal ACTUALLY says now, not on the simulation.
    switch = switch_decision(applied_plan)
    applied_plan["switch"] = switch
    print_report(applied_plan, applied=True, verbose=verbose)

    if not switch["on"]:
        # The rows that moved stay moved - they are correct and consistent. What
        # is held back is the convention NEW fills arrive under, because a
        # refused or still-UNKNOWN open position would be split in two by the
        # very next fill that tried to close it.
        print(
            "The switch was NOT turned on. Read the reasons above, then run this again "
            "once they are dealt with.",
            file=sys.stderr,
        )
        return EXIT_SWITCH_STAYED_OFF

    # LAST, and only now: new fills may arrive under the same convention as the
    # rows this run just moved. Until this line the desk and the nightly import
    # keep storing exactly what they stored yesterday.
    from project_paths import save_local_setting

    save_local_setting(QUESTRADE_INSTRUMENT_SETTING, True)
    print(
        f"local_settings['{QUESTRADE_INSTRUMENT_SETTING}'] is now true: new Questrade fills "
        "will be stored the same way."
    )
    return EXIT_OK


def _verify(
    before: _Snapshot, after: _Snapshot, machine_rows: Mapping[str, Mapping[str, Any]] | None = None
) -> list[str]:
    """Everything that must still be true, checked before the run is kept."""
    problems: list[str] = []
    machine_rows = machine_rows or {}
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
    for table, dead_after in after.machine_dead.items():
        dead_before = before.machine_dead.get(table, 0)
        # Strict for the two tables whose rows this pass can move. The immutable
        # one is never rewritten, so a reference the overlap rule could not
        # resolve stays unresolvable - allowed ONLY up to the number this run
        # counted and printed, because an event going dark unaccounted for is
        # exactly the failure this check exists to catch.
        allowance = (
            int((machine_rows.get(table) or {}).get("left", 0) or 0)
            if table == "opportunity_events"
            else 0
        )
        if dead_after > dead_before + allowance:
            problems.append(
                f"{table} would be left with {dead_after} row(s) pointing at a trade that "
                f"no longer exists, up from {dead_before} "
                f"({allowance} accounted for in the report)"
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
