"""TJ-9Q - the money a reclassify must not move, and the money it must correct.

Red before the fix, on `claude/tj9q-questrade-instrument` at `3c9d837a`, except
where a test says GUARD: those are green today and must stay green.

`journal_file_authority.signed_cash` is the comparison that decides whether a
broker file takes a day over from the live sync. It reads three of the fields
this packet touches:

    signed cash = (+1 sell / -1 buy) x quantity x price x multiplier
                  - commission - fees

* the **multiplier** comes from `raw_executions.multiplier` FIRST and only then
  from `security_type` (`journal_file_authority._multiplier_for`), and the live
  store holds `multiplier = 1.0` on all 226 Questrade rows including the option
  ones - so reclassifying `security_type` alone does NOT reach this function.
  The stored column has to move with it.
* the **sign** comes from `_BUY_SIDES = {BUY, BOT, BTO, BTC, COVER}`. `STO` is
  correctly outside it and `BTC` correctly inside it, so those two words are
  already right here. `COV` is NOT in it - Questrade's spelling is `Cov`, the
  set holds `COVER` - so today each of the trader's 25 covers is counted as
  money coming IN. Mapping `COV` -> `BUY` corrects that, and the correction is a
  CHANGE to this comparison that the packet did not anticipate. It is recorded
  here rather than hidden.

So "the file-authority comparison is unaffected" is true of the equity fills
that already said Buy/Sell/Short, and of nothing else. The three tests below
say exactly which.
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
    AAOI_SELL_1,
    ACCOUNT_NUMBER,
    COVERED_SHORT,
    MARA_SHORT,
    SMPL_COVER,
    SMPL_SHORT_1,
    SMPL_SHORT_2,
    broker_cash,
    import_payloads,
    new_store,
    store_old_convention,
)


def _stored(store, execution_uid: str) -> dict:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT * FROM raw_executions WHERE execution_uid = ?", (execution_uid,)
        ).fetchone()
    assert row is not None, f"{execution_uid} is not in the store"
    return {key: row[key] for key in row.keys()}


def _uid(payload: dict) -> str:
    return f"QT:{ACCOUNT_NUMBER}:{payload['id']}"


# --------------------------------------------------------------------------
# GUARD - the fills that were already right
# --------------------------------------------------------------------------


def test_a_plain_equity_fills_cash_is_the_same_before_and_after_the_reclassify(tmp_path):
    """GUARD, green today. 35 MARA shorted at 10.2302.

    `Short` was already mapped to SELL and a stock's multiplier is 1 whether the
    type is `UNKNOWN` or `STK`, so classification cannot move this number. It is
    the broker's own `totalCost` (358.057) less the fees, both ways.
    """
    from journal_file_authority import signed_cash

    old = new_store(tmp_path / "old")
    store_old_convention(old, [MARA_SHORT])
    before = signed_cash(_stored(old, _uid(MARA_SHORT)))

    reclassified = dict(_stored(old, _uid(MARA_SHORT)))
    reclassified["security_type"] = "STK"
    reclassified["multiplier"] = 1.0

    assert signed_cash(reclassified) == pytest.approx(before, abs=1e-9)
    assert before == pytest.approx(broker_cash(MARA_SHORT), abs=1e-9)


def test_the_commission_on_a_broker_credit_is_never_made_a_charge(tmp_path):
    """GUARD, green today, and the one thing no reader may ever do.

    The trader's IBKR file carries 18 commission CREDITS across 609 fills. They
    reach the store as negative numbers and must stay negative through every
    pass that touches a row. Modelled here as a stored row that a reclassify
    walks past.
    """
    from tj9q_support import ib_flex_execution

    store = new_store(tmp_path)
    store.upsert_executions(
        [
            ib_flex_execution(
                "CREDIT-1", symbol="AAOI", side="BUY", quantity=100, price=10.0,
                commission=-2.17, net_amount=-997.83,
                timestamp="2026-06-10T10:30:00-07:00",
            )
        ]
    )

    assert _stored(store, f"IBKR:{ACCOUNT_NUMBER}:CREDIT-1")["commission"] == pytest.approx(-2.17)


# --------------------------------------------------------------------------
# RED - the money the classification is supposed to correct
# --------------------------------------------------------------------------


def test_an_option_fills_cash_is_the_brokers_own_number(tmp_path):
    """One AAOI put sold at 1.45. The broker booked 145.

    Today the journal's file-authority comparison books 1.45 for it, because the
    fill is `UNKNOWN` and its stored multiplier is 1 - so a Questrade statement
    covering 2026-06-10 would look like a 143-dollar disagreement and take the
    whole day over for a difference that is ours, not the broker's.
    """
    from journal_file_authority import signed_cash

    store = new_store(tmp_path)
    import_payloads(store, [AAOI_SELL_1], rebuild=False)

    row = _stored(store, _uid(AAOI_SELL_1))

    assert row["security_type"] == "OPT"
    assert row["multiplier"] == pytest.approx(100.0)
    assert signed_cash(row) == pytest.approx(broker_cash(AAOI_SELL_1), abs=1e-9)
    assert signed_cash(row) == pytest.approx(144.007013, abs=1e-9)


def test_a_cover_costs_cash_rather_than_raising_it(tmp_path):
    """37 SMPL bought back at 9.9185 is 366.98 LEAVING the account.

    `COV` is outside `_BUY_SIDES`, so today this fill is counted as +366.98 - a
    733.97 error on that day's cash, in a comparison whose whole job is to
    decide whether the broker's file disagrees with us. This is a CORRECTION,
    not a preservation: the number moves, and it moves onto the broker's.
    """
    from journal_file_authority import signed_cash

    store = new_store(tmp_path)
    import_payloads(store, [SMPL_COVER], rebuild=False)

    assert signed_cash(_stored(store, _uid(SMPL_COVER))) == pytest.approx(
        broker_cash(SMPL_COVER), abs=1e-9
    )
    assert signed_cash(_stored(store, _uid(SMPL_COVER))) == pytest.approx(-366.9845, abs=1e-9)


def test_a_days_cash_adds_up_to_what_the_broker_says_it_did(tmp_path):
    """The whole SMPL short, day by day, against the broker's own `totalCost`.

    Three days, three numbers, none of them ours: +253.4948, +135.3596,
    -366.9845. Today the third is +366.9845 and the day is wrong by twice the
    cover.
    """
    from journal_file_authority import cash_by_day

    store = new_store(tmp_path)
    import_payloads(store, COVERED_SHORT, rebuild=False)

    with store.connection() as conn:
        rows = [
            {key: row[key] for key in row.keys()}
            for row in conn.execute("SELECT * FROM raw_executions")
        ]
    measured = {day.isoformat(): cash for (_, day), (cash, _) in cash_by_day(rows).items()}

    assert measured["2026-08-27"] == pytest.approx(broker_cash(SMPL_SHORT_1), abs=1e-9)
    assert measured["2026-09-02"] == pytest.approx(broker_cash(SMPL_SHORT_2), abs=1e-9)
    assert measured["2026-09-18"] == pytest.approx(broker_cash(SMPL_COVER), abs=1e-9)
