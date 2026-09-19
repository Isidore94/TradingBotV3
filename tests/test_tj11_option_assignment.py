"""TJ-11 item 6 - was the option assigned? Read it off the LEGS, or say nothing.

The packet's own instruction (`.claude/packets/TJ-11.md`, CORRECTED block,
2026-09-19): *"`journal_exposure` has no assignment field. Read it off the legs
if the legs say it; otherwise the row says `assignment: unmeasured`. Add your
own test."* This is that test, written by the builder beside the tester's files.

An assigned option reaches the journal as a REAL FILL whose broker payload says
so - IBKR writes `Buy 100 ROUNDHILL MEMORY ETF (Assignment)`
(`journal_ib_transactions`, `TRADE_TYPES` includes `ASSIGNMENT`). Nothing on the
`trades` row records it, and this packet may not invent an identifier. So the
answer comes from the legs when the legs carry them, and is `unmeasured` when
they do not - never "not assigned", which would be a claim about a fill nobody
read.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SESSION = "2026-09-18"


def _trade(**overrides) -> dict:
    trade = {
        "trade_id": "T1",
        "symbol": "AAA",
        "direction": "LONG",
        "status": "closed",
        "security_type": "OPT",
        "opened_at": "2026-09-18T09:40:00-04:00",
        "closed_at": "2026-09-18T10:00:00-04:00",
        "last_closing_leg_at": "2026-09-18T10:00:00-04:00",
        "net_pnl": 120.0,
    }
    trade.update(overrides)
    return trade


def _row(trade: dict):
    from walkaway_day import build

    day = build(
        SESSION,
        sources={"decisions": (), "preference": (), "outcomes": ()},
        bars={},
        trades=(trade,),
        now=datetime(2026, 9, 19, 8, 0),
        daily_bars={},
    )
    assert len(day.traded_left_early) == 1, day.traded_left_early
    return day.traded_left_early[0]


def test_an_option_trade_with_no_legs_says_unmeasured_and_never_not_assigned():
    row = _row(_trade())

    assert row.instrument == "OPT"
    assert row.assignment == "unmeasured"


def test_a_leg_whose_broker_payload_says_assignment_is_read_as_assigned():
    legs = [
        {"role": "open", "side": "SELL", "raw_json": '{"description": "Sell 1 AAA PUT"}'},
        {
            "role": "close",
            "side": "BUY",
            "raw_json": '{"type": "Assignment", "description": "Buy 100 AAA (Assignment)"}',
        },
    ]

    assert _row(_trade(legs=legs)).assignment == "assigned"


def test_legs_that_say_nothing_about_assignment_say_not_assigned():
    """Legs WERE read and none of them is an assignment. That is an answer."""
    legs = [
        {"role": "open", "side": "BUY", "raw_json": '{"description": "Buy 1 AAA CALL"}'},
        {"role": "close", "side": "SELL", "raw_json": '{"description": "Sell 1 AAA CALL"}'},
    ]

    assert _row(_trade(legs=legs)).assignment == "not assigned"


def test_a_stock_trade_is_not_asked_the_question_at_all():
    row = _row(_trade(security_type="STK"))

    assert row.instrument == "STK"
    assert row.assignment == ""


def test_an_option_row_keeps_its_money_and_loses_only_the_wrong_ruler():
    """Premium kept and days held are reported; "left on the table" is not."""
    row = _row(_trade(legs=[{"role": "close", "raw_json": '{"type": "Assignment"}'}]))

    assert row.you_made == 120.0
    assert row.sessions_held == 0
    assert row.left_on_table_pct is None
    assert "option" in row.not_judged_reason.lower()
