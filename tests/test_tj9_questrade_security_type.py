"""TJ-9 item 7 - a Questrade fill says what instrument it is.

Written BEFORE the fix and red on `claude/tj9-forced-trade-labels`'s base commit
(`57151d0f`). The builder makes them pass and may only ADD.

THE MEASUREMENT BEHIND THESE FIXTURES (read-only, a COPY of the live journal,
2026-09-19)
---------------------------------------------------------------------------
`raw_executions` holds **226 Questrade rows and every one of them carries
`security_type = 'UNKNOWN'`** - not 18, and not September only. The cause is in
the payload, not in the mapper: Questrade's `v1/accounts/{id}/executions`
response carries exactly these keys and NEITHER `securityType` NOR `symbolType`

    canadianExecutionFee commission exchangeExecId executionFee id legId notes
    orderChainId orderId orderPlacementCommission parentId price quantity
    secFee side symbol symbolId timestamp totalCost venue

so `normalize_security_type(raw.get("securityType") or raw.get("symbolType"))`
is handed `None` and answers `UNKNOWN` (`journal_importers.py:575`). The comment
at `journal_identity.py:92` - *"Questrade sends a real securityType for options
and futures"* - is wrong for this endpoint; it is true of `get_positions`, which
is where the fallback was written.

**The payload does carry the answer, in the broker's own `symbol` field.** The
four option fills in the live store are spelled in Questrade's own option
format and the equities are plain tickers:

    AAOI18Jun26P120.00   QBTS26Jun26P22.50   BE2Jul26P260.00   QQQ26Jun26P700.00
    MARA  DKS  SMPL  ... (75 more plain tickers)

`side` agrees independently - `BTO`/`STO`/`BTC`/`STC` on the option rows,
`Buy`/`Sell`/`Short`/`Cov` on the equity rows - so the classification is read
from the broker's own fields and never from the length of a ticker.

The two fixtures below are REAL recorded payloads with every identifier
replaced: `id`, `orderId`, `orderChainId`, `parentId`, `legId`, `symbolId`,
`exchangeExecId` and the account number are test values. No key, token or
account number is in this file.

WHAT IS PINNED, AND WHY IT IS ADDITIVE
--------------------------------------
Every field but `security_type` is pinned to the byte-identical value the
CURRENT code produces, computed on this branch's base commit before any fix
existed:

* `symbol` stays `AAOI18JUN26P120.00`. This is the trap. `canonical_option_symbol`
  rewrites a symbol into OCC form as soon as the type is `OPT`, and the option
  root/expiry/strike/right are NOT in this payload, so a builder who "completes"
  the classification by parsing them would change the stored symbol - and the
  stored symbol is half of `journal_identity.group_key`, which is what the trade
  assembly groups on. Classification is additive; renaming is not.
* `execution_uid`, `quantity`, `price`, `commission`, `fees`, `currency`,
  `timestamp` and `trade_date` are unchanged. The packet's words: never touch
  amounts, signs or the fill identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

#: The account the fixtures belong to. A test value, never a real account.
ACCOUNT = {"number": "TJ9-TEST-ACCOUNT", "type": "Margin", "name": "Test Margin"}

#: A real recorded Questrade OPTION execution: one put sold to open, 2026-06-10.
OPTION_PAYLOAD = {
    "canadianExecutionFee": 0,
    "commission": 0.99,
    "exchangeExecId": "EXEC-OPT-1",
    "executionFee": 0,
    "id": 5001,
    "legId": 0,
    "notes": "",
    "orderChainId": 9001,
    "orderId": 9002,
    "orderPlacementCommission": 0,
    "parentId": 0,
    "price": 1.45,
    "quantity": 1,
    "secFee": 0.002987,
    "side": "STO",
    "symbol": "AAOI18Jun26P120.00",
    "symbolId": 12345,
    "timestamp": "2026-06-10T12:53:34.905000-04:00",
    "totalCost": 145,
    "venue": "ROPT",
}

#: A real recorded Questrade EQUITY execution: 35 shares shorted, 2026-09-01.
STOCK_PAYLOAD = {
    "canadianExecutionFee": 0,
    "commission": 0,
    "exchangeExecId": "EXEC-STK-1",
    "executionFee": 0,
    "id": 5002,
    "legId": 0,
    "notes": "",
    "orderChainId": 9003,
    "orderId": 9004,
    "orderPlacementCommission": 0,
    "parentId": 0,
    "price": 10.2302,
    "quantity": 35,
    "secFee": 0.007376,
    "side": "Short",
    "symbol": "MARA",
    "symbolId": 23456,
    "timestamp": "2026-09-01T15:59:14.076000-04:00",
    "totalCost": 358.057,
    "venue": "JANE",
}


def _normalized(payload):
    from journal_importers import QuestradeImporter

    return QuestradeImporter().normalize_execution(dict(payload), dict(ACCOUNT))


def test_a_questrade_option_fill_is_classified_as_an_option_from_the_brokers_own_symbol():
    """`AAOI18Jun26P120.00` is Questrade's own spelling of a contract. The
    importer reads it rather than leaving the trade an UNKNOWN instrument that
    is later priced with a multiplier of one."""
    execution = _normalized(OPTION_PAYLOAD)

    assert execution.security_type == "OPT"


def test_a_questrade_equity_fill_is_classified_as_stock():
    """A plain ticker with an equity side. `UNKNOWN` was never true of it."""
    execution = _normalized(STOCK_PAYLOAD)

    assert execution.security_type == "STK"


def test_classifying_changes_nothing_else_about_the_option_fill():
    """Additive, and the symbol is the trap: the option root, expiry, strike and
    right are NOT in this payload, so completing the OCC spelling would silently
    re-key the position. Every value here is what the code produced BEFORE the
    classifier existed."""
    execution = _normalized(OPTION_PAYLOAD)

    assert execution.execution_uid == "QT:TJ9-TEST-ACCOUNT:5001"
    assert execution.symbol == "AAOI18JUN26P120.00"
    assert execution.quantity == 1.0
    assert execution.price == 1.45
    assert execution.commission == 0.99
    assert execution.fees == 0.002987
    assert execution.currency == "USD"
    assert execution.timestamp == "2026-06-10T12:53:34.905000-04:00"
    assert execution.trade_date == "2026-06-10"
    assert execution.gross_amount is None
    assert execution.net_amount is None
    assert execution.order_id == "9002"
    assert execution.exchange_exec_id == "EXEC-OPT-1"
    assert execution.broker == "QUESTRADE"
    assert execution.source == "QT_API"


def test_classifying_changes_nothing_else_about_the_equity_fill():
    execution = _normalized(STOCK_PAYLOAD)

    assert execution.execution_uid == "QT:TJ9-TEST-ACCOUNT:5002"
    assert execution.symbol == "MARA"
    assert execution.side == "SELL"
    assert execution.quantity == 35.0
    assert execution.price == 10.2302
    assert execution.commission == 0.0
    assert execution.fees == 0.007376
    assert execution.currency == "USD"
    assert execution.timestamp == "2026-09-01T15:59:14.076000-04:00"
    assert execution.trade_date == "2026-09-01"
    assert execution.order_id == "9004"
    assert execution.exchange_exec_id == "EXEC-STK-1"


def test_a_payload_that_still_says_what_it_is_is_believed_over_the_symbol():
    """`get_positions` DOES carry `securityType`, and a future endpoint might
    too. The broker's own statement of the type outranks anything read off the
    symbol - the classifier is a fallback, not a replacement."""
    payload = dict(STOCK_PAYLOAD)
    payload["securityType"] = "Option"
    payload["symbol"] = "MARA"

    assert _normalized(payload).security_type == "OPT"


def test_a_symbol_that_says_nothing_stays_unknown_rather_than_being_guessed():
    """Uncertainty is never confirmation. A payload with neither a type nor a
    readable symbol is still `UNKNOWN`; nothing here infers from a ticker's
    length."""
    payload = dict(STOCK_PAYLOAD)
    payload["symbol"] = ""

    assert _normalized(payload).security_type == "UNKNOWN"
