"""Shared fixtures-in-code for the TJ-9 red tests. NOT a test module.

Everything here builds a REAL `JournalStore` in a temp directory through the
real import path (`manual_execution_from_fields` -> `upsert_executions` ->
`rebuild_trades`), so a test that passes has driven the same assembly the desk
drives. Nothing here writes to `C:\\TradingBotData`: pytest's own `conftest.py`
points `TRADINGBOTV3_DATA_DIR` at a temp directory before anything under
`scripts/` is imported, and every store below is built under `tmp_path`.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

#: A plain Monday. Its previous exchange session is Friday 2026-09-11.
SESSION_TODAY = date(2026, 9, 14)
REVIEWED = "2026-09-11"
#: The desk's own account label in these tests. Never a real account number.
ACCOUNT = "TJ9-TEST-ACCOUNT"


def new_store(tmp_path: Path):
    """A real journal database, empty, under `tmp_path`."""
    from journal_store import JournalStore

    return JournalStore(Path(tmp_path) / "journal.sqlite3")


def _execution(
    execution_id: str,
    *,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    timestamp: str,
    security_type: str = "STK",
) -> Any:
    from journal_importers import manual_execution_from_fields

    return manual_execution_from_fields(
        {
            "broker": "MANUAL",
            "account_number": ACCOUNT,
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "price": price,
            "timestamp": timestamp,
            "security_type": security_type,
            "currency": "USD",
            "commission": 0,
            "fees": 0,
            "execution_id": execution_id,
        }
    )


def add_round_trip(store, symbol: str, *, day: str = REVIEWED, entry_hour: int = 7) -> str:
    """One CLOSED long round trip on `day`, assembled by the real rebuild.

    Returns the assembled `trade_id`. The stamps are market-local with a clock
    time, so `journal_trade_shape.is_date_only` is False - a date-only fill is
    a different case and is modelled explicitly where it matters.
    """
    store.upsert_executions(
        [
            _execution(
                f"{symbol}-open",
                symbol=symbol,
                side="BUY",
                qty=100,
                price=10.0,
                timestamp=f"{day}T{entry_hour:02d}:31:00",
            ),
            _execution(
                f"{symbol}-close",
                symbol=symbol,
                side="SELL",
                qty=100,
                price=11.0,
                timestamp=f"{day}T{entry_hour + 2:02d}:05:00",
            ),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    for trade in store.list_trades(trade_date=day):
        if str(trade.get("symbol") or "") == symbol:
            return str(trade["trade_id"])
    raise AssertionError(f"{symbol} did not assemble into a trade on {day}")


def mark_covered(store, day: str, *, status: str = "COVERED") -> None:
    """The broker statement for `day` landed - what `_journal_ready` reads."""
    import journal_coverage

    journal_coverage.mark_range(
        store,
        broker="QUESTRADE",
        account_number=ACCOUNT,
        start=day,
        end=day,
        status=status,
        source="tj9-test",
    )


def slot_at(session: date, hour: int):
    """The real scheduled slot for `hour` on `session`, or an error."""
    from trade_mentor_schedule import slots_for_session

    for slot in slots_for_session(session):
        if slot.scheduled_at.hour == hour:
            return slot
    raise AssertionError(f"no {hour:02d}:00 slot on {session}")
