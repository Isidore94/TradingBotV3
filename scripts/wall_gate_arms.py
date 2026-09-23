"""The wall gate's ledger of follow-up watches it armed (trader, 2026-09-23).

When the Alert Center hides a chart at an SMA or trendline wall it arms
follow-up watches in the existing stores (D1 event watches, the Pullback
alert). Those stores cannot say "the desk armed this" or "the trader turned
this off" for a D1 event watch, so this small file does:

    {symbol: {"side", "wall", "source_text", "armed_at",
              "kinds": {kind: {"state": "armed" | "declined", "watch_id"}}}}

A declined kind is never re-armed while its entry lives. An entry lives the
same ten trading days an armed watch does (`armed_alert_expiry`). Plain
Python, no Qt. A failed read is an empty ledger; a failed write loses the
ledger row, never the watch.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA = "wall_gate_arms_v1"
STATE_ARMED = "armed"
STATE_DECLINED = "declined"


def load(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    out: dict[str, dict[str, Any]] = {}
    for symbol, entry in (entries or {}).items():
        if not isinstance(entry, dict) or not isinstance(entry.get("kinds"), dict):
            continue
        name = str(symbol or "").strip().upper()
        if name:
            out[name] = dict(entry)
    return out


def save(ledger: dict[str, dict[str, Any]], path: Path | None) -> bool:
    if path is None:
        return True
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(
            prefix=target.name, suffix=".tmp", dir=str(target.parent)
        )
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump({"schema": SCHEMA, "entries": ledger}, stream, indent=1, sort_keys=True)
        os.replace(temp, target)
        return True
    except OSError:
        logging.debug("Wall-gate ledger write failed", exc_info=True)
        return False


def prune_expired(ledger: dict[str, dict[str, Any]], *, today: date) -> bool:
    """Drop entries past their trading-day life. True when anything went.

    An entry the calendar cannot date stays (uncertainty never deletes).
    """
    try:
        import armed_alert_expiry
    except Exception:  # pragma: no cover - module always ships
        return False
    gone = [
        symbol
        for symbol, entry in ledger.items()
        if armed_alert_expiry.is_expired(entry.get("armed_at"), "wall_gate", today=today)
        is True
    ]
    for symbol in gone:
        ledger.pop(symbol, None)
    return bool(gone)


def declined_kinds(entry: dict[str, Any] | None) -> set[str]:
    kinds = (entry or {}).get("kinds") or {}
    return {
        kind
        for kind, record in kinds.items()
        if isinstance(record, dict) and record.get("state") == STATE_DECLINED
    }


def armed_kinds(entry: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    kinds = (entry or {}).get("kinds") or {}
    return {
        kind: record
        for kind, record in kinds.items()
        if isinstance(record, dict) and record.get("state") == STATE_ARMED
    }
