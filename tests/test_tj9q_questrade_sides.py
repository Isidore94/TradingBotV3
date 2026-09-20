"""TJ-9Q item 2 - a sold put is a SALE, and the option multiplier is applied once.

Red before the fix, on `claude/tj9q-questrade-instrument` at `3c9d837a`. The
builder makes them pass and may only ADD.

WHAT THE TRADER'S STORE SAYS TODAY (read-only copy, 2026-09-19)
---------------------------------------------------------------
`journal_importers.normalize_side` maps `BTO`->BUY and `STC`->SELL but does not
know `STO`, `BTC` or `Cov`, so those three land in `raw_executions.side` as the
broker spelled them. `journal_store._signed_quantity` then reads anything that
is not in its SELL set as a BUY:

* `STO` -> +qty. The trader's three sold puts open as LONG.
* `BTC` -> +qty. The fills that bought them back ADD to the position instead of
  closing it, so all three sit `OPEN` with `quantity_closed = 0`.
* `COV` -> +qty, which is the right answer by accident: a cover IS a buy. The
  25 covered shorts in the live store are correct today and must stay correct.

The trader sells puts. Their strategy is recorded backwards.

WHY `totalCost` IS THE YARDSTICK
--------------------------------
The packet asked for the P&L sign to be checked against the broker's
`net_amount` sum. `net_amount` is NULL on all 226 Questrade rows (the endpoint
does not send it), so these tests use the field the payload DOES carry:
`totalCost`, the broker's own gross cash for the fill, which already includes
the contract multiplier - 145 for one contract at 1.45, 358.057 for 35 shares at
10.2302. Adding those up with the broker's own side words gives the realised
P&L without any arithmetic of ours, which is exactly what makes it a check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tj9q_support import (  # noqa: E402
    AAOI_PUT,
    BE_PUT,
    BOUGHT_PUT,
    COVERED_SHORT,
    QBTS_PUT,
    SOLD_PUTS,
    broker_pnl,
    import_payloads,
    new_store,
    trade_for,
)


# --------------------------------------------------------------------------
# The side map itself
# --------------------------------------------------------------------------


def test_a_sold_to_open_put_is_a_sale():
    """`STO` is the whole defect in one word: the trader SOLD it."""
    from journal_importers import normalize_side

    assert normalize_side("STO") == "SELL"


def test_a_bought_to_close_fill_is_a_buy():
    """`BTC` is what closes a sold put. Today it opens more of one."""
    from journal_importers import normalize_side

    assert normalize_side("BTC") == "BUY"


def test_a_cover_is_a_buy_in_words_as_well_as_in_effect():
    """`Cov` is Questrade's spelling. `COVER` was already mapped; this is not."""
    from journal_importers import normalize_side

    assert normalize_side("Cov") == "BUY"
    assert normalize_side("COV") == "BUY"


def test_the_sides_that_already_worked_are_not_disturbed():
    """A guard, green today: the 193 Buy/Sell/Short/BTO/STC fills keep their
    mapping. This is the half of the vocabulary that was never wrong."""
    from journal_importers import normalize_side

    assert normalize_side("Buy") == "BUY"
    assert normalize_side("Sell") == "SELL"
    assert normalize_side("Short") == "SELL"
    assert normalize_side("BTO") == "BUY"
    assert normalize_side("STC") == "SELL"


# --------------------------------------------------------------------------
# Characterize the cover FIRST - pin what it does today, before anything moves
# --------------------------------------------------------------------------


def test_a_cover_still_closes_the_short_it_always_closed(tmp_path):
    """CHARACTERIZATION, green before the fix and green after.

    Two `Short` fills and one `Cov`, recorded whole from the trader's SMPL
    position. `COV` reaches `_signed_quantity` unmapped and comes out +qty,
    which happens to be the right sign for a cover - so this position is one of
    the 25 that are already right. The side map must not move it: same
    direction, same status, same quantity, same money.
    """
    store = new_store(tmp_path)
    import_payloads(store, COVERED_SHORT)

    trade = trade_for(store, "SMPL")

    assert trade["direction"] == "SHORT"
    assert trade["status"] == "CLOSED"
    assert trade["quantity_opened"] == pytest.approx(37.0)
    assert trade["quantity_closed"] == pytest.approx(37.0)
    # The broker's own cash for the three fills, and our recomputed P&L, to the
    # hundredth of a cent. Both are 21.8699 today.
    assert trade["net_pnl"] == pytest.approx(broker_pnl(COVERED_SHORT), abs=1e-4)
    assert trade["net_pnl"] == pytest.approx(21.86989, abs=1e-4)


# --------------------------------------------------------------------------
# The golden: the trader's own sold puts
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "payloads", "contracts"),
    [
        ("AAOI18JUN26P120.00", AAOI_PUT, 2.0),
        ("BE2JUL26P260.00", BE_PUT, 1.0),
        ("QBTS26JUN26P22.50", QBTS_PUT, 2.0),
    ],
)
def test_a_sold_put_rebuilds_short_closes_and_prices_like_the_broker(
    tmp_path, symbol, payloads, contracts
):
    """The golden, one position at a time, through the real import seam.

    Today every one of these is `LONG` / `OPEN` / `quantity_closed = 0`, with a
    `net_pnl` that is only the commissions.
    """
    store = new_store(tmp_path)
    import_payloads(store, payloads)

    trade = trade_for(store, symbol)

    assert trade["direction"] == "SHORT"
    assert trade["status"] == "CLOSED"
    assert trade["quantity_opened"] == pytest.approx(contracts)
    assert trade["quantity_closed"] == pytest.approx(contracts)
    assert trade["security_type"] == "OPT"
    # The broker's own money, from `totalCost` and its own side words. Nothing
    # here multiplies by 100 - the broker already did.
    assert trade["net_pnl"] == pytest.approx(broker_pnl(payloads), abs=1e-4)


def test_the_sold_put_that_made_money_made_241_dollars_not_four_cents(tmp_path):
    """AAOI, spelled out, because the sign is the point.

    Sold two contracts for 1.45 and 1.30, bought both back at 0.15: the trader
    made 245.00 gross and 241.0343 after four 0.99 commissions. Today the
    journal shows -3.9657 - the commissions of a position it thinks is still
    open and long.
    """
    store = new_store(tmp_path)
    import_payloads(store, AAOI_PUT)

    trade = trade_for(store, "AAOI18JUN26P120.00")

    assert trade["gross_pnl"] == pytest.approx(245.00, abs=1e-4)
    assert trade["net_pnl"] == pytest.approx(241.034335, abs=1e-6)
    assert trade["commission"] == pytest.approx(3.96, abs=1e-9)


def test_the_option_multiplier_is_applied_once_and_not_twice(tmp_path):
    """One contract of BE, sold at 3.45 and bought back at 10.10.

    The loss is 665.00, not 6.65 (no multiplier) and not 66,500 (applied at the
    fill AND at the trade). `totalCost` settles it: the broker booked 345 for
    the sale and 1010 for the buy-back.
    """
    store = new_store(tmp_path)
    import_payloads(store, BE_PUT)

    trade = trade_for(store, "BE2JUL26P260.00")

    assert trade["gross_pnl"] == pytest.approx(-665.00, abs=1e-4)
    assert trade["net_pnl"] == pytest.approx(-666.987107, abs=1e-6)


def test_a_bought_put_keeps_its_direction_and_gains_its_multiplier(tmp_path):
    """The one option round trip the trader BOUGHT (`BTO` then `STC`).

    Its sides were already mapped, so it is already LONG and CLOSED - and its
    P&L is 100x too small, because `security_type` is `UNKNOWN`. It lost 80.00,
    not 0.80. Direction and status must not move; only the money.
    """
    store = new_store(tmp_path)
    import_payloads(store, BOUGHT_PUT)

    trade = trade_for(store, "QQQ26JUN26P700.00")

    assert trade["direction"] == "LONG"
    assert trade["status"] == "CLOSED"
    assert trade["gross_pnl"] == pytest.approx(-80.00, abs=1e-4)
    assert trade["net_pnl"] == pytest.approx(broker_pnl(BOUGHT_PUT), abs=1e-4)
    assert trade["net_pnl"] == pytest.approx(-81.982369, abs=1e-6)


def test_all_four_option_positions_come_out_of_one_import_correctly(tmp_path):
    """Three sold and one bought, imported together the way a night imports them.

    One position per contract, none of them merged with another, and the total
    realised money equal to the broker's own.
    """
    store = new_store(tmp_path)
    import_payloads(store, SOLD_PUTS + BOUGHT_PUT)

    trades = {str(row["symbol"]): row for row in store.list_trades()}

    assert set(trades) == {
        "AAOI18JUN26P120.00",
        "BE2JUL26P260.00",
        "QBTS26JUN26P22.50",
        "QQQ26JUN26P700.00",
    }
    assert [trades[key]["status"] for key in sorted(trades)] == ["CLOSED"] * 4
    directions = {key: trades[key]["direction"] for key in sorted(trades)}
    assert directions == {
        "AAOI18JUN26P120.00": "SHORT",
        "BE2JUL26P260.00": "SHORT",
        "QBTS26JUN26P22.50": "SHORT",
        "QQQ26JUN26P700.00": "LONG",
    }
    total = sum(float(row["net_pnl"]) for row in trades.values())
    assert total == pytest.approx(broker_pnl(SOLD_PUTS + BOUGHT_PUT), abs=1e-4)
