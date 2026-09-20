"""TJ-9Q fixtures: the trader's own Questrade fills, scrubbed. NOT a test module.

Every payload below is a REAL recorded Questrade `v1/accounts/{id}/executions`
row from the live journal, read on 2026-09-19 from a read-only COPY, with every
identifier replaced by a test value: `id`, `orderId`, `orderChainId`, `parentId`,
`legId`, `symbolId`, `exchangeExecId` and the account number. No key, token or
account number is in this file. The symbols, sides, quantities, prices,
commissions, fees, `totalCost` figures and timestamps are the trader's real
ones, because they are what the arithmetic has to come out right on.

WHAT THE LIVE STORE ACTUALLY HOLDS (the numbers the tests are built on)
----------------------------------------------------------------------
* 226 of 226 Questrade executions carry `security_type = 'UNKNOWN'`; the payload
  carries exactly 20 keys and NEITHER `securityType` NOR `symbolType`.
* stored `side` values: SELL 101, BUY 92, COV 25, STO 4, BTC 4. The payload's own
  words: Buy 91, Sell 72, Short 28, Cov 25, STO 4, BTC 4, BTO 1, STC 1.
* `totalCost` is the broker's own gross cash for the fill and it ALREADY carries
  the contract multiplier: every option row's `totalCost` is exactly 100 x
  quantity x price, every equity row's is exactly quantity x price. That is what
  makes it an independent check on our arithmetic rather than a restatement of
  it - it is the broker's number, not ours.
* `net_amount` and `gross_amount` are NULL on all 226 Questrade rows, so the
  Questrade half of the tax report is already excluded ("a fill carries no
  broker-stated amount"). The IBKR Flex rows are the ones that carry stated
  amounts (387 of 390), which is why the tax guard below is built on one.

THE FOUR OPTION POSITIONS
-------------------------
Three SOLD puts - AAOI (two STO fills), BE, QBTS - which today rebuild as
`direction = 'LONG'`, `status = 'OPEN'`, `quantity_closed = 0`, because
`_signed_quantity` reads `STO` as +qty and `BTC` as +qty, so the closing fills
ADD to the position instead of closing it. And one BOUGHT put - QQQ, `BTO` then
`STC` - which is correctly LONG and CLOSED because those two sides ARE mapped,
but whose P&L is 100x too small.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

#: A test account. Never a real account number.
ACCOUNT_NUMBER = "TJ9Q-TEST-ACCOUNT"
ACCOUNT = {"number": ACCOUNT_NUMBER, "type": "Margin", "name": "Test Margin"}


def _payload(
    exec_id: int,
    *,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    commission: float,
    sec_fee: float,
    total_cost: float,
    timestamp: str,
    venue: str,
) -> dict[str, Any]:
    """One recorded execution payload, in the endpoint's exact 20-key shape."""
    return {
        "canadianExecutionFee": 0,
        "commission": commission,
        "exchangeExecId": f"EXEC-{exec_id}",
        "executionFee": 0,
        "id": exec_id,
        "legId": 0,
        "notes": "",
        "orderChainId": 9000 + exec_id,
        "orderId": 8000 + exec_id,
        "orderPlacementCommission": 0,
        "parentId": 0,
        "price": price,
        "quantity": quantity,
        "secFee": sec_fee,
        "side": side,
        "symbol": symbol,
        "symbolId": 1000 + exec_id,
        "timestamp": timestamp,
        "totalCost": total_cost,
        "venue": venue,
    }


# --- the three SOLD puts -------------------------------------------------
AAOI_SELL_1 = _payload(
    7001, symbol="AAOI18Jun26P120.00", side="STO", quantity=1, price=1.45,
    commission=0.99, sec_fee=0.002987, total_cost=145,
    timestamp="2026-06-10T12:53:34.905000-04:00", venue="ROPT",
)
AAOI_SELL_2 = _payload(
    7002, symbol="AAOI18Jun26P120.00", side="STO", quantity=1, price=1.30,
    commission=0.99, sec_fee=0.002678, total_cost=130,
    timestamp="2026-06-10T15:39:06.648000-04:00", venue="ROPT",
)
AAOI_BUY_BACK_1 = _payload(
    7003, symbol="AAOI18Jun26P120.00", side="BTC", quantity=1, price=0.15,
    commission=0.99, sec_fee=0.0, total_cost=15,
    timestamp="2026-06-15T10:32:33.707000-04:00", venue="ROPT",
)
AAOI_BUY_BACK_2 = _payload(
    7004, symbol="AAOI18Jun26P120.00", side="BTC", quantity=1, price=0.15,
    commission=0.99, sec_fee=0.0, total_cost=15,
    timestamp="2026-06-15T10:32:33.708000-04:00", venue="ROPT",
)
AAOI_PUT = [AAOI_SELL_1, AAOI_SELL_2, AAOI_BUY_BACK_1, AAOI_BUY_BACK_2]

BE_SELL = _payload(
    7005, symbol="BE2Jul26P260.00", side="STO", quantity=1, price=3.45,
    commission=0.99, sec_fee=0.007107, total_cost=345,
    timestamp="2026-06-22T15:17:50.982000-04:00", venue="ROPT",
)
BE_BUY_BACK = _payload(
    7006, symbol="BE2Jul26P260.00", side="BTC", quantity=1, price=10.10,
    commission=0.99, sec_fee=0.0, total_cost=1010,
    timestamp="2026-06-26T10:10:52.722000-04:00", venue="ROPT",
)
BE_PUT = [BE_SELL, BE_BUY_BACK]

QBTS_SELL = _payload(
    7007, symbol="QBTS26Jun26P22.50", side="STO", quantity=2, price=0.43,
    commission=1.98, sec_fee=0.001772, total_cost=86,
    timestamp="2026-06-15T10:39:49.581000-04:00", venue="ROPT",
)
QBTS_BUY_BACK = _payload(
    7008, symbol="QBTS26Jun26P22.50", side="BTC", quantity=2, price=0.70,
    commission=1.98, sec_fee=0.0, total_cost=140,
    timestamp="2026-06-25T09:47:03.463000-04:00", venue="ROPT",
)
QBTS_PUT = [QBTS_SELL, QBTS_BUY_BACK]

SOLD_PUTS = AAOI_PUT + BE_PUT + QBTS_PUT

# --- the one BOUGHT put (BTO/STC - already mapped, still 100x wrong) ------
QQQ_BUY = _payload(
    7009, symbol="QQQ26Jun26P700.00", side="BTO", quantity=1, price=1.95,
    commission=0.99, sec_fee=0.0, total_cost=195,
    timestamp="2026-06-26T09:52:49.007000-04:00", venue="ROPT",
)
QQQ_SELL = _payload(
    7010, symbol="QQQ26Jun26P700.00", side="STC", quantity=1, price=1.15,
    commission=0.99, sec_fee=0.002369, total_cost=115,
    timestamp="2026-06-26T09:56:32.118000-04:00", venue="ROPT",
)
BOUGHT_PUT = [QQQ_BUY, QQQ_SELL]

# --- an equity short that a COVER closed ---------------------------------
# The "accidentally correct" case, recorded whole: two `Short` fills and one
# `Cov`, which rebuild today as SHORT/CLOSED with the right money.
SMPL_SHORT_1 = _payload(
    7011, symbol="SMPL", side="Short", quantity=25, price=10.14,
    commission=0.0, sec_fee=0.005222, total_cost=253.5,
    timestamp="2026-08-27T12:04:11.000000-04:00", venue="JANE",
)
SMPL_SHORT_2 = _payload(
    7012, symbol="SMPL", side="Short", quantity=12, price=11.2802,
    commission=0.0, sec_fee=0.002788, total_cost=135.3624,
    timestamp="2026-09-02T10:18:02.000000-04:00", venue="JANE",
)
SMPL_COVER = _payload(
    7013, symbol="SMPL", side="Cov", quantity=37, price=9.9185,
    commission=0.0, sec_fee=0.0, total_cost=366.9845,
    timestamp="2026-09-18T09:47:20.000000-04:00", venue="JANE",
)
COVERED_SHORT = [SMPL_SHORT_1, SMPL_SHORT_2, SMPL_COVER]

# --- a plain equity short that is still open -----------------------------
MARA_SHORT = _payload(
    7014, symbol="MARA", side="Short", quantity=35, price=10.2302,
    commission=0.0, sec_fee=0.007376, total_cost=358.057,
    timestamp="2026-09-01T15:59:14.076000-04:00", venue="JANE",
)


def broker_cash(payload: dict[str, Any]) -> float:
    """What the BROKER says this fill did to cash, from its own fields.

    `totalCost` is Questrade's own gross figure and already carries the contract
    multiplier, so this is an independent statement of the money - nothing here
    multiplies, and nothing here reads our own `price x quantity`. The sign comes
    from the broker's own word for the side.
    """
    sells = {"SELL", "STO", "STC", "SHORT"}
    side = str(payload["side"]).strip().upper()
    sign = 1.0 if side in sells else -1.0
    return (
        sign * float(payload["totalCost"])
        - abs(float(payload["commission"]))
        - abs(float(payload["secFee"]))
    )


def broker_pnl(payloads: list[dict[str, Any]]) -> float:
    """The realised P&L of a flat position, added up from the broker's own cash."""
    return sum(broker_cash(item) for item in payloads)


def new_store(tmp_path: Path, name: str = "journal.sqlite3"):
    """A real journal database, empty, under `tmp_path`."""
    from journal_store import JournalStore

    store = JournalStore(Path(tmp_path) / name)
    store.initialize_schema()
    return store


def import_payloads(store, payloads: list[dict[str, Any]], *, rebuild: bool = True) -> None:
    """Drive the REAL Questrade import seam: payload -> normalize -> store.

    `import_executions_for_date` is the function the desk's nightly import calls;
    only its two HTTP reads are replaced, so everything between the recorded
    payload and the stored row is the shipped code. No broker is contacted.
    """
    from journal_importers import QuestradeImporter

    importer = QuestradeImporter()
    importer.get_accounts = lambda: [dict(ACCOUNT)]  # type: ignore[method-assign]
    importer.get_executions = lambda account_number, start, end: [  # type: ignore[method-assign]
        dict(item) for item in payloads
    ]
    executions, accounts = importer.import_executions_for_date()
    assert not importer.quarantined, f"quarantined: {importer.quarantined}"
    store.upsert_accounts("QUESTRADE", accounts)
    store.upsert_executions(executions)
    if rebuild:
        store.rebuild_trades(refresh_tags=False)


def store_old_convention(store, payloads: list[dict[str, Any]], *, rebuild: bool = True) -> None:
    """Write rows exactly as the journal holds them TODAY, by hand.

    Not by calling the importer: once the importer classifies, rows it writes are
    no longer old rows, and a "does a new fill split an old position?" test whose
    old rows were written by the new code proves nothing. These five fields are
    the live values, read off the trader's own store: `security_type` UNKNOWN,
    the side left as the broker spelled it for STO/BTC/COV, the symbol
    uppercased and otherwise untouched, and the amounts positive.
    """
    import json

    from journal_importers import NormalizedExecution, parse_broker_datetime

    unmapped = {"STO", "BTC", "COV"}
    rows = []
    for payload in payloads:
        stamp = parse_broker_datetime(payload["timestamp"], strict=True)
        raw_side = str(payload["side"]).strip().upper()
        side = raw_side if raw_side in unmapped else {
            "BTO": "BUY", "BUY": "BUY", "SELL": "SELL", "SHORT": "SELL", "STC": "SELL",
        }[raw_side]
        rows.append(
            NormalizedExecution(
                execution_uid=f"QT:{ACCOUNT_NUMBER}:{payload['id']}",
                broker="QUESTRADE",
                account_number=ACCOUNT_NUMBER,
                account_label=str(ACCOUNT["name"]),
                account_type=str(ACCOUNT["type"]),
                symbol=str(payload["symbol"]).strip().upper(),
                security_type="UNKNOWN",
                currency="USD",
                side=side,
                quantity=abs(float(payload["quantity"])),
                price=float(payload["price"]),
                timestamp=stamp.isoformat(),
                trade_date=stamp.date().isoformat(),
                commission=abs(float(payload["commission"])),
                fees=abs(float(payload["secFee"])),
                gross_amount=None,
                net_amount=None,
                order_id=str(payload["orderId"]),
                exchange_exec_id=str(payload["exchangeExecId"]),
                raw_json=json.dumps(payload, sort_keys=True, default=str),
                source="QT_API",
            )
        )
    store.upsert_executions(rows)
    if rebuild:
        store.rebuild_trades(refresh_tags=False)


def trade_for(store, symbol: str) -> dict[str, Any]:
    """The one assembled trade for `symbol`, or a readable failure."""
    matches = [row for row in store.list_trades() if str(row.get("symbol")) == symbol.upper()]
    assert len(matches) == 1, f"{symbol}: expected one trade, got {len(matches)}: {matches}"
    return matches[0]


def ib_flex_execution(
    execution_id: str,
    *,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    commission: float,
    net_amount: float,
    timestamp: str,
    security_type: str = "STK",
):
    """One IBKR Flex row - the kind that DOES carry a broker-stated amount.

    387 of the 390 IBKR rows in the live store carry `net_amount`, and those are
    the rows the tax report is allowed to add up. A reclassify of the Questrade
    half must leave this number exactly where it is, including a commission
    CREDIT: the trader's own file carries 18 of them.
    """
    import json

    from journal_importers import NormalizedExecution

    return NormalizedExecution(
        execution_uid=f"IBKR:{ACCOUNT_NUMBER}:{execution_id}",
        broker="IBKR",
        account_number=ACCOUNT_NUMBER,
        account_label="Test IB",
        account_type="Margin",
        symbol=symbol,
        security_type=security_type,
        currency="USD",
        side=side,
        quantity=abs(float(quantity)),
        price=float(price),
        timestamp=timestamp,
        trade_date=timestamp[:10],
        commission=float(commission),
        fees=0.0,
        gross_amount=None,
        net_amount=float(net_amount),
        order_id="",
        exchange_exec_id=execution_id,
        raw_json=json.dumps({"tradeID": execution_id, "assetCategory": security_type}, sort_keys=True),
        source="IBKR_FLEX",
    )
