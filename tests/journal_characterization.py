"""The R7 characterization fixture: what ``rebuild_trades`` does TODAY.

Phase 0.5 packet R7, ``docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`` §9 step 0. This
module exists *before* any assembly change so that every later step has to state
what it changed rather than discover it.

WHAT IS FROZEN, AND WHY IT IS THE STORED ROWS AND NOT THE IMPORTERS

The input is a literal list of ``raw_executions`` rows, not a set of broker
payloads run through the importers. That is deliberate. §9 step 2 rewrites
``execution_uid`` (spec §4 "Identity fixes") and step 3 rewrites the group key,
so a fixture whose inputs were *computed* by the importers would silently change
its own inputs at exactly the moment it is most needed. Freezing the stored rows
makes this a pure assembly characterization: given these rows in the table,
``rebuild_trades`` produces this output.

The uid spellings below are copied from today's call sites -
``journal_importers.py:323`` (Questrade), ``:447`` (IBKR socket), ``:562`` (IBKR
Flex) and ``:637`` (manual) - so the fixture reproduces the real defects instead
of an idealized input. Several cases exist *because* they are wrong today; each
one names the root-cause register entry it demonstrates, and the golden records
the broken output honestly. When a later step fixes one, the golden changes and
the fixture's ``intentional_difference`` field has to say so.

WHAT IS EXCLUDED FROM THE SNAPSHOT

Wall-clock columns (``trades.updated_at``, ``raw_executions.imported_at``,
``opportunity_events.created_at``) and the autoincrement ``trade_legs.leg_id``.
None is assembly output; including them would make the golden unmatchable.

AUTO-TAGS ARE DELIBERATELY OUT OF SCOPE

The snapshot is taken with ``refresh_tags=False``. ``refresh_auto_tags`` reads
the machine's setup-tracker and focus files through ``AutoTagger``, so its output
depends on the desk's state rather than on assembly. The default
``refresh_tags=True`` path is still covered - see
``test_journal_characterization.py`` - but by an invariance assertion, not by
freezing environment-dependent bytes into a golden.

REGENERATING

    .venv\\Scripts\\python.exe tests/journal_characterization.py

Rewrites ``tests/fixtures/journal_rebuild_trades_v1.json``. Only ever run this
with an intended change in hand, and write that change into the fixture's
``intentional_difference`` field in the same commit - the trader's standing rule
for golden fixtures (``plan.md`` sec 5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_NAME = "journal_rebuild_trades_v1"

#: Columns whose value is the clock or an autoincrement counter, not assembly.
VOLATILE_TRADE_COLUMNS = ("updated_at",)
VOLATILE_LEG_COLUMNS = ("leg_id",)
VOLATILE_EVENT_COLUMNS = ("created_at",)


def _execution(
    *,
    execution_uid: str,
    broker: str,
    account_number: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    timestamp: str,
    security_type: str = "STK",
    currency: str = "USD",
    account_label: str = "",
    commission: float = 0.0,
    fees: float = 0.0,
    raw_json: str = "{}",
    order_id: str = "",
) -> dict[str, Any]:
    return {
        "execution_uid": execution_uid,
        "broker": broker,
        "account_number": account_number,
        "account_label": account_label or account_number,
        "account_type": "",
        "symbol": symbol,
        "security_type": security_type,
        "currency": currency,
        "side": side,
        "quantity": quantity,
        "price": price,
        "timestamp": timestamp,
        "trade_date": timestamp[:10],
        "commission": commission,
        "fees": fees,
        "gross_amount": None,
        "net_amount": None,
        "order_id": order_id,
        "exchange_exec_id": "",
        "raw_json": raw_json,
    }


# ---------------------------------------------------------------------------
# The cases. Each block names what it covers and, where it is a defect, the
# root-cause register entry from the spec's §3.
# ---------------------------------------------------------------------------

CHARACTERIZATION_EXECUTIONS: list[dict[str, Any]] = [
    # (a) Scale-in then a partial exit that leaves the trade open.
    #     Covers: scale-ins, partial exits, OPEN status, commission splitting.
    _execution(
        execution_uid="QT:51234567:e-aapl-1:AAPL:2026-08-03T09:31:00-07:00",
        broker="QUESTRADE",
        account_number="51234567",
        account_label="Margin",
        symbol="AAPL",
        security_type="STOCK",
        side="BUY",
        quantity=100,
        price=150.00,
        timestamp="2026-08-03T09:31:00-07:00",
        commission=4.95,
        fees=0.10,
    ),
    _execution(
        execution_uid="QT:51234567:e-aapl-2:AAPL:2026-08-03T10:05:00-07:00",
        broker="QUESTRADE",
        account_number="51234567",
        account_label="Margin",
        symbol="AAPL",
        security_type="STOCK",
        side="BUY",
        quantity=50,
        price=152.00,
        timestamp="2026-08-03T10:05:00-07:00",
        commission=4.95,
        fees=0.05,
    ),
    _execution(
        execution_uid="QT:51234567:e-aapl-3:AAPL:2026-08-03T11:20:00-07:00",
        broker="QUESTRADE",
        account_number="51234567",
        account_label="Margin",
        symbol="AAPL",
        security_type="STOCK",
        side="SELL",
        quantity=120,
        price=155.00,
        timestamp="2026-08-03T11:20:00-07:00",
        commission=4.95,
        fees=0.22,
    ),
    # (b) A clean round trip. The control case: whatever else changes, this one
    #     should keep its numbers.
    _execution(
        execution_uid="QT:51234567:e-msft-1:MSFT:2026-08-03T09:45:00-07:00",
        broker="QUESTRADE",
        account_number="51234567",
        account_label="Margin",
        symbol="MSFT",
        security_type="STOCK",
        side="BUY",
        quantity=200,
        price=300.00,
        timestamp="2026-08-03T09:45:00-07:00",
        commission=4.95,
    ),
    _execution(
        execution_uid="QT:51234567:e-msft-2:MSFT:2026-08-03T12:10:00-07:00",
        broker="QUESTRADE",
        account_number="51234567",
        account_label="Margin",
        symbol="MSFT",
        security_type="STOCK",
        side="SELL",
        quantity=200,
        price=305.50,
        timestamp="2026-08-03T12:10:00-07:00",
        commission=4.95,
    ),
    # (c) A short round trip, opened by a SELL. Covers direction inference and
    #     the sign of gross P&L on the short side.
    _execution(
        execution_uid="IBKR:U1234567:0000e0d5.68a1.01.01:TSLA:2026-08-04T06:35:00-07:00",
        broker="IBKR",
        account_number="U1234567",
        symbol="TSLA",
        side="SELL",
        quantity=50,
        price=250.00,
        timestamp="2026-08-04T06:35:00-07:00",
        commission=1.00,
    ),
    _execution(
        execution_uid="IBKR:U1234567:0000e0d5.68a1.01.02:TSLA:2026-08-04T07:15:00-07:00",
        broker="IBKR",
        account_number="U1234567",
        symbol="TSLA",
        side="BUY",
        quantity=50,
        price=240.00,
        timestamp="2026-08-04T07:15:00-07:00",
        commission=1.00,
    ),
    # (d) An option round trip carrying an explicit contract multiplier in
    #     raw_json. Covers `_contract_multiplier` and the x100 P&L scaling.
    _execution(
        execution_uid="IBKR:U1234567:0000e0d5.68a2.01.01:SPY260116C00500000:2026-08-04T06:40:00-07:00",
        broker="IBKR",
        account_number="U1234567",
        symbol="SPY260116C00500000",
        security_type="OPT",
        side="BUY",
        quantity=2,
        price=5.00,
        timestamp="2026-08-04T06:40:00-07:00",
        commission=1.30,
        raw_json=json.dumps({"multiplier": "100"}, sort_keys=True),
    ),
    _execution(
        execution_uid="IBKR:U1234567:0000e0d5.68a2.01.02:SPY260116C00500000:2026-08-04T09:55:00-07:00",
        broker="IBKR",
        account_number="U1234567",
        symbol="SPY260116C00500000",
        security_type="OPT",
        side="SELL",
        quantity=2,
        price=7.50,
        timestamp="2026-08-04T09:55:00-07:00",
        commission=1.30,
        raw_json=json.dumps({"multiplier": "100"}, sort_keys=True),
    ),
    # (e) DEFECT B4 - the same IBKR fill seen by the socket and by Flex. Same
    #     broker, account and exec id; the timestamp differs only in whether the
    #     offset survived, and the uid embeds the timestamp, so the two dedupe
    #     as two fills and the position doubles. Flex also spells the security
    #     type from `assetCategory` while the socket spells it from the contract
    #     - here both say STK, so this case isolates the uid defect alone.
    _execution(
        execution_uid="IBKR:U1234567:0000e0d5.68a3.01.01:NVDA:2026-08-04T06:31:00-07:00",
        broker="IBKR",
        account_number="U1234567",
        symbol="NVDA",
        side="BUY",
        quantity=10,
        price=100.00,
        timestamp="2026-08-04T06:31:00-07:00",
        commission=1.00,
    ),
    _execution(
        execution_uid="IBKR:U1234567:0000e0d5.68a3.01.01:NVDA:2026-08-04T06:31:00",
        broker="IBKR",
        account_number="U1234567",
        symbol="NVDA",
        side="BUY",
        quantity=10,
        price=100.00,
        timestamp="2026-08-04T06:31:00",
        commission=1.00,
    ),
    _execution(
        execution_uid="IBKR:U1234567:0000e0d5.68a3.01.09:NVDA:2026-08-04T10:02:00-07:00",
        broker="IBKR",
        account_number="U1234567",
        symbol="NVDA",
        side="SELL",
        quantity=10,
        price=110.00,
        timestamp="2026-08-04T10:02:00-07:00",
        commission=1.00,
    ),
    # (f) DEFECT B2 - a closing fill whose opening fill is missing (the day it
    #     was executed was never imported). Today this fabricates a phantom
    #     SHORT trade out of a long exit.
    _execution(
        execution_uid="QT:51234567:e-amd-9:AMD:2026-08-05T09:40:00-07:00",
        broker="QUESTRADE",
        account_number="51234567",
        account_label="Margin",
        symbol="AMD",
        security_type="STOCK",
        side="SELL",
        quantity=100,
        price=95.00,
        timestamp="2026-08-05T09:40:00-07:00",
        commission=4.95,
    ),
    # (g) DEFECT B3 - a manual fill for a position that is really part of the
    #     AAPL group in (a), but is keyed broker="MANUAL" and so can never
    #     attach to it.
    _execution(
        execution_uid="MANUAL:MANUAL:manual-aapl-1",
        broker="MANUAL",
        account_number="MANUAL",
        symbol="AAPL",
        security_type="STK",
        side="BUY",
        quantity=10,
        price=149.00,
        timestamp="2026-08-03T09:30:00-07:00",
    ),
    # (h) DEFECT B3 - the `listingExchange` fallback. Both rows are the same
    #     AMZN position in the same account; one carries a real securityType and
    #     the other does not, so the fallback spells its group key "NASDAQ" and
    #     the two never net against each other.
    _execution(
        execution_uid="QT:51234567:e-amzn-1:AMZN:2026-08-05T09:35:00-07:00",
        broker="QUESTRADE",
        account_number="51234567",
        account_label="Margin",
        symbol="AMZN",
        security_type="STOCK",
        side="BUY",
        quantity=25,
        price=180.00,
        timestamp="2026-08-05T09:35:00-07:00",
        commission=4.95,
    ),
    _execution(
        execution_uid="QT:51234567:e-amzn-2:AMZN:2026-08-05T11:05:00-07:00",
        broker="QUESTRADE",
        account_number="51234567",
        account_label="Margin",
        symbol="AMZN",
        security_type="NASDAQ",
        side="SELL",
        quantity=25,
        price=184.00,
        timestamp="2026-08-05T11:05:00-07:00",
        commission=4.95,
    ),
    # (i) Multi-currency. A CAD round trip in a second account; `pnl_usd` is
    #     None here and set on the USD trades, which is the seam B8's
    #     cross-currency summing sits behind.
    _execution(
        execution_uid="QT:52222222:e-shop-1:SHOP:2026-08-05T06:35:00-07:00",
        broker="QUESTRADE",
        account_number="52222222",
        account_label="TFSA",
        symbol="SHOP.TO",
        security_type="STOCK",
        currency="CAD",
        side="BUY",
        quantity=100,
        price=80.00,
        timestamp="2026-08-05T06:35:00-07:00",
        commission=4.95,
    ),
    _execution(
        execution_uid="QT:52222222:e-shop-2:SHOP:2026-08-05T12:45:00-07:00",
        broker="QUESTRADE",
        account_number="52222222",
        account_label="TFSA",
        symbol="SHOP.TO",
        security_type="STOCK",
        currency="CAD",
        side="SELL",
        quantity=100,
        price=85.00,
        timestamp="2026-08-05T12:45:00-07:00",
        commission=4.95,
    ),
]


def build_fixture_store(db_path: Path):
    """Create a journal DB at ``db_path`` holding exactly the frozen executions."""
    from journal_store import JournalStore

    store = JournalStore(db_path)
    store.upsert_executions([dict(row) for row in CHARACTERIZATION_EXECUTIONS])
    return store


def _scrub(row: dict[str, Any], volatile: tuple[str, ...]) -> dict[str, Any]:
    return {key: value for key, value in sorted(row.items()) if key not in volatile}


def capture_rebuild_output(store) -> dict[str, Any]:
    """Rebuild and return the assembly output, free of clock and counter columns.

    ``refresh_tags=False``: see this module's docstring.
    """
    trade_count = store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        trades = [
            _scrub(dict(row), VOLATILE_TRADE_COLUMNS)
            for row in conn.execute("SELECT * FROM trades ORDER BY trade_id").fetchall()
        ]
        legs = [
            _scrub(dict(row), VOLATILE_LEG_COLUMNS)
            for row in conn.execute(
                "SELECT * FROM trade_legs ORDER BY trade_id, timestamp, execution_uid, role"
            ).fetchall()
        ]
        events = [
            _scrub(dict(row), VOLATILE_EVENT_COLUMNS)
            for row in conn.execute(
                "SELECT * FROM opportunity_events ORDER BY event_id"
            ).fetchall()
        ]
    return {
        "trades": trades,
        "legs": legs,
        "opportunity_events": events,
        "summary": {
            "returned_trade_count": int(trade_count),
            "trade_rows": len(trades),
            "leg_rows": len(legs),
            "opportunity_event_rows": len(events),
            "open_trades": sum(1 for row in trades if row["status"] == "OPEN"),
            "closed_trades": sum(1 for row in trades if row["status"] == "CLOSED"),
        },
    }


def _generate(tmp_dir: Path) -> dict[str, Any]:
    store = build_fixture_store(tmp_dir / "trade_journal.sqlite3")
    return capture_rebuild_output(store)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the golden fixture. See this module's docstring first."""
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--note",
        default="",
        help="why the expected output changed. Required whenever it did.",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conftest import FIXTURES_DIR, _canonical_json, validate_fixture_contract

    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        captured = _generate(Path(first))
        # A golden that is not reproducible inside one process is not a golden.
        # Refuse to write rather than freeze a coin flip.
        if _generate(Path(second)) != captured:
            print("REFUSED: rebuild_trades output differed between two runs", file=sys.stderr)
            return 1

    path = FIXTURES_DIR / f"{FIXTURE_NAME}.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    # The trader's standing rule for golden fixtures, enforced here rather than
    # left to whoever regenerates: expected output may change, but not quietly.
    changed = [key for key in ("trades", "legs", "opportunity_events", "summary")
               if key in previous and previous[key] != captured[key]]
    if changed and not args.note.strip():
        print(
            "REFUSED: the expected output changed in section(s) "
            f"{', '.join(changed)} and no --note was given.\n"
            "Re-run with --note \"why this changed and who approved it\".",
            file=sys.stderr,
        )
        return 1
    payload: dict[str, Any] = {
        "schema": "journal_rebuild_trades/v1",
        "feature_version": previous.get("feature_version") or "r7-step0-pre-assembly",
        "universe_version": "n/a (journal fixture; no symbol universe is read)",
        "provider_assumptions": (
            "No broker or network access. Input is frozen raw_executions rows, not importer "
            "output; rebuild_trades is called with refresh_tags=False so AutoTagger's "
            "machine-local setup-tracker and focus files are never read."
        ),
        "acquired_at": previous.get("acquired_at") or "2026-08-15T14:00:00-07:00",
        "as_of": "2026-08-05T13:00:00-07:00",
        "numeric_tolerance": 0.0,
        "intentional_difference": args.note.strip() or previous.get("intentional_difference") or "",
        "raw_input_keys": ["executions"],
        "expected_keys": ["trades", "legs", "opportunity_events", "summary"],
        "executions": CHARACTERIZATION_EXECUTIONS,
        **captured,
    }
    payload["raw_input_sha256"] = __import__("hashlib").sha256(
        _canonical_json(payload["executions"])
    ).hexdigest()
    validate_fixture_contract(payload, FIXTURE_NAME)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path} ({payload['summary']['trade_rows']} trades, {payload['summary']['leg_rows']} legs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
