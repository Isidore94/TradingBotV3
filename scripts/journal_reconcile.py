"""Does the journal agree with the broker about what is open?

R7 §9 step 9, root cause B1 ("no reconciliation against broker positions exists
anywhere"), and the place where step 4's deliberate narrowing gets its evidence.

WHAT THIS IS FOR

Assembly can only reason about the fills it was given. A sell with no matching
buy is genuinely ambiguous - a real short entry, or a sale of shares bought
before the import window - and nothing in the execution distinguishes them, so
step 4 assembles it as an ordinary short and says nothing. **This** is where that
gets resolved, because the broker reporting flat against a journal that says
short is exactly the evidence assembly could not have.

WHAT IT NEVER DOES

It never closes anything. A journal-open-but-broker-flat position produces a
*suggested* FORCE_CLOSE that the trader confirms in the UI; the suggestion is
stored outside ``trade_adjustments`` precisely so that it cannot apply itself.
An adjustment is a thing a human decided, and this module is not a human.

It also never edits raw broker rows (I3) and never writes a trader-owned field
(I7). Its whole output is a ``NEEDS_REVIEW`` flag, an append-only run row, and a
report the Health tab renders.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from journal_identity import group_key_text, normalize_security_type

#: Under this many shares/contracts, two positions are the same position.
#: Fractional-share brokers and option multipliers both produce dust.
QUANTITY_EPSILON = 0.0001

#: Where the latest report lives for the Health tab to read.
REPORT_META_KEY = "last_reconciliation"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _position_key(broker: str, account_number: str, symbol: str, security_type: Any, currency: str):
    return (
        str(broker or "").upper(),
        str(account_number or ""),
        str(symbol or "").upper(),
        normalize_security_type(security_type),
        str(currency or "").upper(),
    )


def journal_open_positions(store: Any) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    """What the journal believes is open, netted per instrument.

    Netted across trades rather than read per trade: two OPEN trades in the same
    instrument are one position as far as the broker is concerned, and comparing
    per trade would report a mismatch that is only an artifact of how assembly
    split them.
    """
    positions: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    with store.connection() as conn:
        rows = conn.execute(
            """
            SELECT trade_id, broker, account_number, symbol, security_type, currency,
                   direction, status, quantity_opened, quantity_closed
            FROM trades
            WHERE status IN ('OPEN', 'CLOSED_PARTIAL')
            """
        ).fetchall()
    for row in rows:
        remaining = float(row["quantity_opened"]) - float(row["quantity_closed"])
        if abs(remaining) <= QUANTITY_EPSILON:
            continue
        signed = remaining if str(row["direction"]).upper() == "LONG" else -remaining
        key = _position_key(
            row["broker"], row["account_number"], row["symbol"], row["security_type"], row["currency"]
        )
        entry = positions.setdefault(key, {"quantity": 0.0, "trade_ids": []})
        entry["quantity"] += signed
        entry["trade_ids"].append(str(row["trade_id"]))
    return positions


def normalize_broker_positions(rows: Iterable[Mapping]) -> dict[tuple[str, str, str, str, str], float]:
    """Broker-reported positions, keyed the same way the journal's are."""
    result: dict[tuple[str, str, str, str, str], float] = {}
    for row in rows:
        key = _position_key(
            row.get("broker"),
            row.get("account_number"),
            row.get("symbol"),
            row.get("security_type"),
            row.get("currency") or "USD",
        )
        try:
            quantity = float(row.get("quantity") or 0.0)
        except (TypeError, ValueError):
            continue
        result[key] = result.get(key, 0.0) + quantity
    return result


def compare(
    store: Any,
    broker_positions: Iterable[Mapping],
    *,
    brokers: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compare journal against broker. Pure - reads only, writes nothing.

    ``brokers`` scopes the comparison. Without it, an IBKR-only reconciliation
    would report every Questrade position as "broker says flat", which is not a
    mismatch - it is a question nobody asked.
    """
    scope = {str(item).upper() for item in brokers} if brokers else None
    broker_map = normalize_broker_positions(broker_positions)
    journal_map = journal_open_positions(store)

    keys = set(broker_map) | set(journal_map)
    if scope is not None:
        keys = {key for key in keys if key[0] in scope}

    agreed: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    for key in sorted(keys):
        journal_entry = journal_map.get(key, {"quantity": 0.0, "trade_ids": []})
        journal_quantity = float(journal_entry["quantity"])
        broker_quantity = float(broker_map.get(key, 0.0))
        record = {
            "broker": key[0],
            "account_number": key[1],
            "symbol": key[2],
            "security_type": key[3],
            "currency": key[4],
            "group_key": group_key_text(key),
            "journal_quantity": journal_quantity,
            "broker_quantity": broker_quantity,
            "delta": broker_quantity - journal_quantity,
            "trade_ids": list(journal_entry["trade_ids"]),
        }
        if abs(record["delta"]) <= QUANTITY_EPSILON:
            agreed.append(record)
            continue
        if abs(broker_quantity) <= QUANTITY_EPSILON:
            # The journal holds a position the broker does not. This is the case
            # step 4 deliberately could not judge, and the one worth suggesting a
            # close for.
            record["kind"] = "JOURNAL_OPEN_BROKER_FLAT"
        elif abs(journal_quantity) <= QUANTITY_EPSILON:
            record["kind"] = "BROKER_OPEN_JOURNAL_FLAT"
        else:
            record["kind"] = "QUANTITY_MISMATCH"
        mismatched.append(record)

    suggestions = [
        {
            "action": "FORCE_CLOSE",
            "target_kind": "TRADE_GROUP",
            "target_uid": item["group_key"],
            "reason": (
                f"reconciliation: journal holds {item['journal_quantity']:g} "
                f"{item['symbol']} but {item['broker']} reports flat"
            ),
            "trade_ids": item["trade_ids"],
        }
        for item in mismatched
        if item["kind"] == "JOURNAL_OPEN_BROKER_FLAT"
    ]

    return {
        "checked_at": _now_iso(),
        "brokers": sorted(scope) if scope else sorted({key[0] for key in keys}),
        "agreed": agreed,
        "mismatched": mismatched,
        "suggestions": suggestions,
        "positions_checked": len(keys),
    }


def apply_review_flags(store: Any, report: Mapping) -> int:
    """Stamp NEEDS_REVIEW on the trades behind each mismatch, and clear the rest.

    Clearing matters as much as stamping: a position that reconciled today must
    lose yesterday's flag, or the review queue only ever grows and stops meaning
    anything. ``FORCED_CLOSED`` is left alone - it records that a human closed
    the trade, and reconciliation does not get to overwrite that.
    """
    flagged: set[str] = set()
    for item in report.get("mismatched") or []:
        flagged.update(str(trade_id) for trade_id in item.get("trade_ids") or [])
    brokers = sorted(
        {str(broker).upper() for broker in report.get("brokers") or [] if str(broker).strip()}
    )
    with store.connection() as conn:
        if brokers:
            placeholders = ", ".join("?" for _ in brokers)
            conn.execute(
                "UPDATE trades SET reconcile_status = '' "
                f"WHERE reconcile_status = 'NEEDS_REVIEW' AND broker IN ({placeholders})",
                brokers,
            )
        for trade_id in sorted(flagged):
            conn.execute(
                "UPDATE trades SET reconcile_status = 'NEEDS_REVIEW' "
                "WHERE trade_id = ? AND reconcile_status != 'FORCED_CLOSED'",
                (trade_id,),
            )
    return len(flagged)


def store_report(store: Any, report: Mapping) -> None:
    """Keep the latest report where the Health tab can read it.

    Deliberately **not** in ``trade_adjustments``. A suggestion stored there
    would be applied by the next rebuild, and the whole point is that a human
    confirms it first.
    """
    with store.connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            (REPORT_META_KEY, json.dumps(dict(report), sort_keys=True, default=str)),
        )


def last_report(store: Any) -> dict[str, Any] | None:
    with store.connection() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (REPORT_META_KEY,)).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row[0])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def reconcile(
    store: Any,
    broker_positions: Iterable[Mapping],
    *,
    brokers: Iterable[str] | None = None,
    trigger: str = "manual",
) -> dict[str, Any]:
    """Compare, flag, record. The one entry point the runner and the UI call."""
    report = compare(store, broker_positions, brokers=brokers)
    report["trigger"] = trigger
    flagged = apply_review_flags(store, report)
    report["flagged_trades"] = flagged

    run_id = store.start_import_run("RECONCILE", trigger=trigger)
    store.finish_import_run(
        run_id,
        status="OK" if not report["mismatched"] else "MISMATCH",
        imported_executions=0,
        message=(
            f"{report['positions_checked']} position(s), "
            f"{len(report['mismatched'])} mismatch(es), {flagged} trade(s) flagged"
        ),
    )
    report["import_run_id"] = run_id
    store_report(store, report)
    return report


def confirm_suggestion(store: Any, suggestion: Mapping, *, reason: str = "", source: str = "gui") -> dict[str, Any]:
    """Turn a suggestion into a real adjustment, on the trader's say-so.

    The one path from "reconciliation thinks" to "the journal does". It exists
    here so the suggestion never has to be stored as an applicable record.
    """
    return store.record_adjustment(
        action=str(suggestion.get("action") or "FORCE_CLOSE"),
        target_kind=str(suggestion.get("target_kind") or "TRADE_GROUP"),
        target_uid=str(suggestion.get("target_uid") or ""),
        payload=dict(suggestion.get("payload") or {}),
        reason=str(reason or suggestion.get("reason") or "confirmed from reconciliation"),
        source=source,
    )
