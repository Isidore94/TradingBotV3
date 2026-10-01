"""The journal's open trades checked against the last broker reconciliation. Read-only.

``journal_reconcile`` stores its last report in the journal's ``meta`` table
(``last_reconciliation``). A journal-open trade that report calls
``JOURNAL_OPEN_BROKER_FLAT`` is stale (an exit the import lost), not a position; a
``BROKER_OPEN_JOURNAL_FLAT`` entry is a position the journal is missing. The packs use
:func:`split` so the Mentor never reads stale journal rows as the trader's book.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

REPORT_META_KEY = "last_reconciliation"  # journal_reconcile.REPORT_META_KEY
PT = ZoneInfo("America/Los_Angeles")
UNCHECKED = "journal (not checked against broker)"


def read_report(path: Path | str) -> dict[str, Any] | None:
    """The last reconciliation report from the journal (``mode=ro``); None when there is none."""
    db = Path(path)
    if not db.exists():
        return None
    conn = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=5)
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone() is None:
            return None
        found = conn.execute("SELECT value FROM meta WHERE key = ?", (REPORT_META_KEY,)).fetchone()
    finally:
        conn.close()
    if not found:
        return None
    try:
        payload = json.loads(found[0])
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) and payload else None


def live_report() -> dict[str, Any] | None:
    from project_paths import JOURNAL_DB_FILE

    return read_report(JOURNAL_DB_FILE)


def _broker(value: Any) -> str:
    return "IBKR" if str(value or "").strip().upper().startswith("IBKR") else "QUESTRADE"


def _moment(value: Any) -> datetime | None:
    """An aware datetime; a naive stamp is this machine's local time (how the reconciler writes it)."""
    try:
        moment = datetime.fromisoformat(str(value or "").strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.astimezone()


def when_text(value: Any) -> str:
    moment = _moment(value)
    return f"{moment.astimezone(PT):%a %m-%d %H:%M} PT" if moment else (str(value or "") or "an unknown time")


@dataclass
class Checked:
    """Journal open trades split by the last reconciliation."""

    open: list[dict[str, Any]] = field(default_factory=list)
    stale: list[dict[str, Any]] = field(default_factory=list)
    #: ``BROKER_OPEN_JOURNAL_FLAT`` records: broker, account_number, symbol, broker_quantity (signed).
    broker_only: list[dict[str, Any]] = field(default_factory=list)
    #: ``checked_at`` of the report; "" = no report (every open trade is unchecked).
    checked_at: str = ""
    labels: dict[int, str] = field(default_factory=dict)

    def label(self, trade: Mapping[str, Any]) -> str:
        return self.labels.get(id(trade), UNCHECKED)

    @property
    def when(self) -> str:
        return when_text(self.checked_at)

    def stale_text(self) -> str:
        syms = ", ".join(dict.fromkeys(str(t.get("symbol") or "").strip().upper() for t in self.stale))
        return (f"Journal shows {len(self.stale)} stale open trade(s) the broker reports flat as of "
                f"{self.when}: {syms} - not positions")


def _key(broker: Any, account: Any, symbol: Any) -> tuple[str, str, str]:
    return _broker(broker), str(account or "").strip(), str(symbol or "").strip().upper()


def _trade_key(trade: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """(broker, account, symbol); None when the trade has no account (too vague to match)."""
    if not str(trade.get("account_number") or "").strip():
        return None
    return _key(trade.get("broker"), trade.get("account_number"), trade.get("symbol"))


def _matches(trade: Mapping[str, Any], record: Mapping[str, Any], checked: datetime | None) -> bool:
    """The report entry covers this trade: its trade id is listed, or same broker/account/symbol and
    the trade was opened before the check (a newer trade was never checked)."""
    if str(trade.get("trade_id") or "") in {str(t) for t in record.get("trade_ids") or ()}:
        return True
    key = _trade_key(trade)
    if key is None:
        return False
    rec = _key(record.get("broker"), record.get("account_number"), record.get("symbol"))
    no_broker = not str(trade.get("broker") or "").strip()
    if not (key == rec or (no_broker and key[1:] == rec[1:])):
        return False
    opened = _moment(trade.get("opened_at"))
    return checked is not None and opened is not None and opened <= checked


def split(trades: Iterable[Mapping[str, Any]], report: Mapping[str, Any] | None) -> Checked:
    """Split journal open trades by ``report``; no report = every trade open and unchecked."""
    rows = [t for t in trades or ()]
    if not report:
        return Checked(open=list(rows))
    checked_at = str(report.get("checked_at") or "")
    checked = _moment(checked_at)
    mismatched = [m for m in report.get("mismatched") or () if isinstance(m, Mapping)]
    flat = [m for m in mismatched if m.get("kind") == "JOURNAL_OPEN_BROKER_FLAT"]
    other = [m for m in mismatched if m.get("kind") != "JOURNAL_OPEN_BROKER_FLAT"]
    agreed = [a for a in report.get("agreed") or () if isinstance(a, Mapping)]
    out = Checked(checked_at=checked_at)
    for trade in rows:
        if any(_matches(trade, m, checked) for m in flat):
            out.stale.append(trade)
            continue
        out.open.append(trade)
        if any(_matches(trade, a, checked) for a in agreed) and not any(_matches(trade, m, checked) for m in other):
            out.labels[id(trade)] = f"journal, broker agreed as of {when_text(checked_at)}"
    held = {_trade_key(t) for t in out.open}
    for record in mismatched:
        if record.get("kind") != "BROKER_OPEN_JOURNAL_FLAT":
            continue
        key = _key(record.get("broker"), record.get("account_number"), record.get("symbol"))
        try:
            qty = float(record.get("broker_quantity") or 0.0)
        except (TypeError, ValueError):
            continue
        if key in held or not key[2] or not qty:
            continue
        out.broker_only.append({"broker": key[0], "account_number": key[1], "symbol": key[2],
                                "broker_quantity": qty, "side": "LONG" if qty > 0 else "SHORT",
                                "security_type": str(record.get("security_type") or "")})
    return out


def broker_only_text(record: Mapping[str, Any], when: str) -> str:
    return (f"{record['side']} {record['symbol']} {abs(float(record['broker_quantity'])):g} in "
            f"{record['account_number'] or 'unknown account'} (broker only, not in journal; broker check {when})")

