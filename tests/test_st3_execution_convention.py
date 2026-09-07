"""Packet ST3 - no impossible fills, no same-day knowledge.

The tracker replay books a hard stop at the STOP LEVEL even when the whole bar
traded below it (`_apply_scenario_exit_event(..., float(hard_stop_level), ...)`,
`master_avwap_lib/legacy.py`), and it tests a bar's own high/low against
`current_history[<that same day>]`, whose anchored-VWAP bands were computed WITH
that day's bar folded into the cumulative sums. Both are fills the simulated
decision time could not have produced.

These tests pin the ADDITIVE, OPT-IN repair:

* `master_avwap_lib.execution_convention` - `literal_level_v1` (today, default)
  and `gap_aware_v2`, plus `same_session_v1` (today, default) and
  `prior_session_v2` level knowledge;
* `_evaluate_tracker_scenario_bar(..., execution_convention=...)` routing every
  booked exit price through `resolve_fill`;
* `recompute_tracker_setup_record(..., execution_convention=..., level_knowledge=...)`
  carrying the two keys on the record only when a non-default is used;
* `scripts/tracker_execution_compare.py`, the evidence CLI.

**The default path is proved unchanged, not assumed.** Every v1 leg below calls
the function with NO new keyword, so it exercises the shipped signature, and
test 9 pins a whole `recompute_tracker_setup_record` record from `main` as a
byte-identical golden.

Every expected R is DERIVED from `TRACKER_COST_COMMISSION_PER_SHARE` and
`TRACKER_SLIPPAGE_FRACTION_PER_SIDE` by `_expected_r` below - there is one cost
model and no magic number restates it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402

from conftest import FIXTURES_DIR, load_fixture_contract  # noqa: E402

GOLDEN_FIXTURE_NAME = "st3_replay_golden"

#: Set to "1" to re-pin the golden from whatever code is checked out. It is a
#: door for the tester and the reviewer, never a way for a build to move the
#: pin: with the variable unset the golden test asserts, and CI never sets it.
REGEN_ENV = "ST3_REGEN_GOLDEN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _execution_convention():
    """The ST3.1 module. Absent on `main`, which is why these tests are red."""
    from master_avwap_lib import execution_convention

    return execution_convention


def _cost_per_share_per_side(entry_price: float) -> float:
    """The ONE cost model, read from the shipped constants."""
    return float(m.TRACKER_COST_COMMISSION_PER_SHARE) + float(
        m.TRACKER_SLIPPAGE_FRACTION_PER_SIDE
    ) * abs(float(entry_price))


def _expected_r(entry_price, exit_price, shares, direction, risk_usd):
    """Realized R for one exit, derived from the shipped cost constants."""
    gross = (float(exit_price) - float(entry_price)) * int(shares) * float(direction)
    cost = _cost_per_share_per_side(entry_price) * int(shares) * 2.0
    return (gross - cost) / float(risk_usd)


def _bar(*, high, low, close, open_=None, include_open=True):
    """A daily bar row.

    A real replay bar comes out of a frame that HAS an `open` column, so the
    realistic "missing open" is the key PRESENT and NaN; `include_open=False`
    covers the absent-key form too.
    """
    row = {"high": float(high), "low": float(low), "close": float(close)}
    if include_open:
        row["open"] = float("nan") if open_ is None else float(open_)
    return pd.Series(row)


def _scenario(**overrides):
    scenario = {
        "tradeable": True,
        "status": "OPEN",
        "entry_price": 100.0,
        "initial_risk_per_share": 5.0,
        "initial_risk_usd": 500.0,
        "direction": 1.0,
        "shares": 100,
        "remaining_shares": 100,
        "partial_taken": False,
        "partial_shares": 0,
        "realized_pnl": 0.0,
        "realized_r": 0.0,
        "events": [],
        "close_failure_count": 0,
        "close_failure_limit": 2,
        "stop_reference_label": "LOWER_1",
        "active_stop_label": "LOWER_1",
        "partial_target_label": None,
        "final_target_label": None,
        "trail_after_partial_label": None,
        "hard_stop_r_multiple": None,
    }
    scenario.update(overrides)
    return scenario


def _reasons(events):
    return [str(event.get("reason")) for event in events]


# ---------------------------------------------------------------------------
# 1-2. Opening gap through the stop, both sides
# ---------------------------------------------------------------------------
def test_a_long_gap_below_the_stop_books_the_open_not_the_untraded_level():
    """The review fixture: entry 100, risk 5, hard stop 95, next bar O80/H85/L79/C82.

    The bar never traded at 95. v1 books it anyway (characterized, unchanged);
    v2 books the open.
    """
    gap_bar = _bar(open_=80, high=85, low=79, close=82)

    # v1 = today, called through the SHIPPED signature (no new keyword).
    v1 = _scenario(hard_stop_r_multiple=1.0)
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", gap_bar, None, None, is_entry_day=False, bar_index=2
    )
    assert _reasons(v1_events) == ["HARD_STOP"]
    assert v1_events[0]["price"] == pytest.approx(95.0)
    assert v1["realized_r"] == pytest.approx(_expected_r(100.0, 95.0, 100, 1.0, 500.0))
    assert v1["realized_r"] == pytest.approx(-1.014)

    ec = _execution_convention()
    v2 = _scenario(hard_stop_r_multiple=1.0)
    v2_events = m._evaluate_tracker_scenario_bar(
        v2,
        "LONG",
        "2026-01-05",
        gap_bar,
        None,
        None,
        is_entry_day=False,
        bar_index=2,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert _reasons(v2_events) == ["HARD_STOP"]
    assert v2_events[0]["price"] == pytest.approx(80.0)
    assert v2_events[0]["fill_basis"] == "gap_open"
    assert v2_events[0]["execution_convention"] == ec.EXECUTION_GAP_AWARE_V2
    assert v2["realized_r"] == pytest.approx(_expected_r(100.0, 80.0, 100, 1.0, 500.0))
    assert v2["realized_r"] == pytest.approx(-4.014)
    assert 79.0 <= v2_events[0]["price"] <= 85.0

    assert ec.DEFAULT_EXECUTION_CONVENTION == ec.EXECUTION_LITERAL_LEVEL_V1


def test_a_short_gap_above_the_stop_books_the_open_not_the_untraded_level():
    """Mirror of the long case: short entry 100, stop 105, bar O120/H121/L115/C118."""
    gap_bar = _bar(open_=120, high=121, low=115, close=118)

    v1 = _scenario(direction=-1.0, hard_stop_r_multiple=1.0, stop_reference_label="UPPER_1", active_stop_label="UPPER_1")
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "SHORT", "2026-01-05", gap_bar, None, None, is_entry_day=False, bar_index=2
    )
    assert _reasons(v1_events) == ["HARD_STOP"]
    assert v1_events[0]["price"] == pytest.approx(105.0)
    assert v1["realized_r"] == pytest.approx(_expected_r(100.0, 105.0, 100, -1.0, 500.0))

    ec = _execution_convention()
    v2 = _scenario(direction=-1.0, hard_stop_r_multiple=1.0, stop_reference_label="UPPER_1", active_stop_label="UPPER_1")
    v2_events = m._evaluate_tracker_scenario_bar(
        v2,
        "SHORT",
        "2026-01-05",
        gap_bar,
        None,
        None,
        is_entry_day=False,
        bar_index=2,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert _reasons(v2_events) == ["HARD_STOP"]
    assert v2_events[0]["price"] == pytest.approx(120.0)
    assert v2_events[0]["fill_basis"] == "gap_open"
    assert v2["realized_r"] == pytest.approx(_expected_r(100.0, 120.0, 100, -1.0, 500.0))
    assert v2["realized_r"] == pytest.approx(-4.014)
    assert 115.0 <= v2_events[0]["price"] <= 121.0


# ---------------------------------------------------------------------------
# 3. Missing open -> clamped into the bar
# ---------------------------------------------------------------------------
def test_a_bar_with_no_open_clamps_the_fill_inside_the_bar_under_v2():
    """Long stop 95 against H85/L79/C82. Nothing traded at 95; the honest worst
    case inside the bar is its high."""
    v1 = _scenario(hard_stop_r_multiple=1.0)
    v1_events = m._evaluate_tracker_scenario_bar(
        v1,
        "LONG",
        "2026-01-05",
        _bar(high=85, low=79, close=82, include_open=False),
        None,
        None,
        is_entry_day=False,
        bar_index=2,
    )
    assert v1_events[0]["price"] == pytest.approx(95.0)

    ec = _execution_convention()
    # Both realistic shapes of "no open": the key absent, and the key present
    # and NaN (which is what a real frame column gives you).
    for bar in (
        _bar(high=85, low=79, close=82, include_open=False),
        _bar(high=85, low=79, close=82, open_=None),
    ):
        v2 = _scenario(hard_stop_r_multiple=1.0)
        events = m._evaluate_tracker_scenario_bar(
            v2,
            "LONG",
            "2026-01-05",
            bar,
            None,
            None,
            is_entry_day=False,
            bar_index=2,
            execution_convention=ec.EXECUTION_GAP_AWARE_V2,
        )
        assert _reasons(events) == ["HARD_STOP"]
        assert events[0]["price"] == pytest.approx(85.0)
        assert events[0]["fill_basis"] == "clamped_no_open"
        assert 79.0 <= events[0]["price"] <= 85.0
        assert v2["realized_r"] == pytest.approx(_expected_r(100.0, 85.0, 100, 1.0, 500.0))
        assert v2["realized_r"] == pytest.approx(-3.014)

    # The pure function says the same thing on its own.
    fill = ec.resolve_fill("LONG", "stop", 95.0, {"high": 85.0, "low": 79.0, "close": 82.0})
    assert fill.booked is True
    assert fill.basis == "clamped_no_open"
    assert 79.0 <= fill.price <= 85.0
    # A short stop below the low clamps the other way.
    short_fill = ec.resolve_fill("SHORT", "stop", 75.0, {"high": 85.0, "low": 79.0, "close": 82.0})
    assert short_fill.price == pytest.approx(79.0)
    assert short_fill.basis == "clamped_no_open"


# ---------------------------------------------------------------------------
# 4. Partials pay their own cost, and there is only one cost model
# ---------------------------------------------------------------------------
def test_a_partial_and_its_remainder_each_pay_their_own_cost_under_v2():
    """Half out at the target, the rest stopped on a gap. The v2 totals equal the
    v1 totals booked at the SAME prices, because the cost model is one."""
    levels = {"bands": {"UPPER_2": 110.0}}

    # --- v1 booked at the two prices v2 will reach: the hard stop level IS 90
    #     here (2R below entry), so no gap logic is needed to get there today.
    v1 = _scenario(hard_stop_r_multiple=2.0, partial_target_label="UPPER_2")
    m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", _bar(open_=105, high=111, low=99, close=105),
        levels, None, is_entry_day=False, bar_index=2,
    )
    m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-06", _bar(open_=90, high=92, low=88, close=89),
        levels, None, is_entry_day=False, bar_index=3,
    )
    assert [round(e["price"], 6) for e in v1["events"]] == [110.0, 90.0]
    assert [e["shares"] for e in v1["events"]] == [50, 50]

    ec = _execution_convention()
    # --- v2: partial fills at the level (open 105 is below the target), the
    #     remainder's stop fills at the gap open 90 rather than the level 95.
    v2 = _scenario(hard_stop_r_multiple=1.0, partial_target_label="UPPER_2")
    partial_events = m._evaluate_tracker_scenario_bar(
        v2,
        "LONG",
        "2026-01-05",
        _bar(open_=105, high=111, low=99, close=105),
        levels,
        None,
        is_entry_day=False,
        bar_index=2,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert _reasons(partial_events) == ["PARTIAL_TARGET"]
    assert partial_events[0]["price"] == pytest.approx(110.0)
    assert partial_events[0]["fill_basis"] == "level"
    assert partial_events[0]["shares"] == 50
    stop_events = m._evaluate_tracker_scenario_bar(
        v2,
        "LONG",
        "2026-01-06",
        _bar(open_=90, high=92, low=88, close=89),
        levels,
        None,
        is_entry_day=False,
        bar_index=3,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert _reasons(stop_events) == ["HARD_STOP"]
    assert stop_events[0]["price"] == pytest.approx(90.0)
    assert stop_events[0]["shares"] == 50

    # Same prices, same shares -> identical money and identical R.
    assert v2["realized_pnl"] == pytest.approx(v1["realized_pnl"])
    assert v2["realized_r"] == pytest.approx(v1["realized_r"])

    # Each leg pays cost on ITS OWN quantity, derived from the shipped model.
    per_side = _cost_per_share_per_side(100.0)
    for event in v2["events"]:
        assert event["cost"] == pytest.approx(per_side * event["shares"] * 2.0)
    assert sum(e["cost"] for e in v2["events"]) == pytest.approx(per_side * 100 * 2.0)
    assert v2["realized_r"] == pytest.approx(
        _expected_r(100.0, 110.0, 50, 1.0, 500.0) + _expected_r(100.0, 90.0, 50, 1.0, 500.0)
    )


# ---------------------------------------------------------------------------
# 5. Stop-first survives the gap
# ---------------------------------------------------------------------------
def test_a_gapped_stop_still_wins_the_same_bar_against_a_target():
    """The bar gaps below the stop AND its high reaches the target. Stop-first is
    kept: one event, the stop, at the open - never the target."""
    levels = {"bands": {"UPPER_3": 110.0}}
    bar = _bar(open_=90, high=111, low=88, close=92)

    v1 = _scenario(hard_stop_r_multiple=1.0, final_target_label="UPPER_3")
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", bar, levels, None, is_entry_day=False, bar_index=2
    )
    assert _reasons(v1_events) == ["HARD_STOP"]
    assert v1_events[0]["price"] == pytest.approx(95.0)
    assert v1["status"] == "STOPPED"

    ec = _execution_convention()
    v2 = _scenario(hard_stop_r_multiple=1.0, final_target_label="UPPER_3")
    v2_events = m._evaluate_tracker_scenario_bar(
        v2,
        "LONG",
        "2026-01-05",
        bar,
        levels,
        None,
        is_entry_day=False,
        bar_index=2,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert _reasons(v2_events) == ["HARD_STOP"]
    assert "FINAL_TARGET" not in _reasons(v2["events"])
    assert v2_events[0]["price"] == pytest.approx(90.0)
    assert v2_events[0]["fill_basis"] == "gap_open"
    assert v2["status"] == "STOPPED"
    assert v2["remaining_shares"] == 0


# ---------------------------------------------------------------------------
# 6. Target gap
# ---------------------------------------------------------------------------
def test_an_open_above_the_target_fills_at_the_open_under_v2():
    levels = {"bands": {"UPPER_3": 110.0}}
    bar = _bar(open_=114, high=116, low=112, close=115)

    v1 = _scenario(final_target_label="UPPER_3")
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", bar, levels, None, is_entry_day=False, bar_index=2
    )
    assert _reasons(v1_events) == ["FINAL_TARGET"]
    assert v1_events[0]["price"] == pytest.approx(110.0)
    assert v1["realized_r"] == pytest.approx(_expected_r(100.0, 110.0, 100, 1.0, 500.0))

    ec = _execution_convention()
    v2 = _scenario(final_target_label="UPPER_3")
    v2_events = m._evaluate_tracker_scenario_bar(
        v2,
        "LONG",
        "2026-01-05",
        bar,
        levels,
        None,
        is_entry_day=False,
        bar_index=2,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert _reasons(v2_events) == ["FINAL_TARGET"]
    assert v2_events[0]["price"] == pytest.approx(114.0)
    assert v2_events[0]["fill_basis"] == "gap_open"
    assert v2["realized_r"] == pytest.approx(_expected_r(100.0, 114.0, 100, 1.0, 500.0))
    assert 112.0 <= v2_events[0]["price"] <= 116.0


# ---------------------------------------------------------------------------
# 7. Which DAY's band a bar is tested against
# ---------------------------------------------------------------------------
ANCHOR_IDX = 5


def _rising_frame(periods: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2026-03-02", periods=periods)
    rows = []
    for i, dt in enumerate(dates):
        base = 100.0 + i * 0.3
        rows.append(
            {
                "datetime": dt,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.2,
                "volume": 1_000_000.0 + i * 1_000.0,
            }
        )
    return pd.DataFrame(rows)


def _band_moves_inside_its_own_bar_frame():
    """A frame whose LAST bar drags its own UPPER_3 down below its own high.

    The final bar's typical price is exactly the running VWAP and its volume is
    3x everything accumulated so far, so folding it in leaves VWAP where it was
    and halves the running deviation. Nothing here is a magic number: the shape
    is built from what the REAL `calc_anchored_vwap_band_history` reports for the
    previous day. The formula is frozen; this test is about which day is read.
    """
    df = _rising_frame()
    anchor = df.iloc[ANCHOR_IDX]["datetime"].date().isoformat()
    prior = df.iloc[:-1].reset_index(drop=True)
    prior_history = m.calc_anchored_vwap_band_history(prior, anchor)
    prev_key = prior.iloc[-1]["datetime"].date().isoformat()
    vwap_prev = float(prior_history[prev_key]["vwap"])
    stdev_prev = float(prior_history[prev_key]["stdev"])
    accumulated_volume = float(prior.iloc[ANCHOR_IDX:]["volume"].sum())

    half_width = 2.0 * stdev_prev
    last = len(df) - 1
    df.loc[last, "open"] = vwap_prev
    df.loc[last, "close"] = vwap_prev
    df.loc[last, "high"] = vwap_prev + half_width
    df.loc[last, "low"] = vwap_prev - half_width
    df.loc[last, "volume"] = 3.0 * accumulated_volume
    return df, anchor, prev_key


def _band_move_setup(df, anchor, entry_date):
    entry_price = float(df.iloc[-2]["close"])
    return {
        "symbol": "STBAND",
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "setup_family": "avwap_retest_followthrough",
        "entry_price": entry_price,
        "entry_trade_date": entry_date,
        "scan_date": entry_date,
        "anchor_date": anchor,
        "scenarios": {
            "s1": {
                "tradeable": True,
                "shares": 100,
                "direction": 1.0,
                "initial_risk_per_share": 5.0,
                "initial_risk_usd": 500.0,
                "stop_reference_label": "LOWER_1",
                "stop_reference_level": entry_price - 5.0,
                "final_target_label": "UPPER_3",
                "partial_target_label": None,
                "close_failure_limit": 2,
                "status": "OPEN",
            }
        },
    }


def test_a_band_known_only_at_the_close_does_not_book_that_days_target_under_v2():
    df, anchor, prev_key = _band_moves_inside_its_own_bar_frame()
    day_key = df.iloc[-1]["datetime"].date().isoformat()
    history = m.calc_anchored_vwap_band_history(df, anchor)
    target_today = float(history[day_key]["bands"]["UPPER_3"])
    target_yesterday = float(history[prev_key]["bands"]["UPPER_3"])
    day_high = float(df.iloc[-1]["high"])

    # The premise of the test, asserted rather than assumed: the day's own band
    # is inside its own range, yesterday's is not.
    assert target_today < day_high < target_yesterday

    setup = _band_move_setup(df, anchor, prev_key)

    same_session = m.recompute_tracker_setup_record(copy.deepcopy(setup), df)
    scenario = same_session["scenarios"]["s1"]
    assert scenario["status"] == "TARGET_HIT"
    assert _reasons(scenario["events"]) == ["FINAL_TARGET"]
    assert scenario["events"][0]["price"] == pytest.approx(target_today)

    ec = _execution_convention()
    prior_session = m.recompute_tracker_setup_record(
        copy.deepcopy(setup),
        df,
        level_knowledge=ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2,
    )
    repaired = prior_session["scenarios"]["s1"]
    assert _reasons(repaired["events"]) == []
    assert m._scenario_is_open(repaired["status"])
    assert prior_session["level_knowledge"] == ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2
    assert ec.DEFAULT_LEVEL_KNOWLEDGE == ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1
    # The default run never grew the key.
    assert "level_knowledge" not in same_session


# ---------------------------------------------------------------------------
# 8. An invalid bar books nothing, and the hold clock still runs
# ---------------------------------------------------------------------------
def test_an_invalid_ohlc_bar_books_no_fill_under_v2_and_the_hold_clock_advances():
    # low > high and low > open: the candle invariant `low <= open, close <= high`
    # is broken, so no sequence inside it can be believed.
    broken = _bar(open_=93, high=90, low=95, close=92)

    # v1 today: the broken bar's `low` clears the stop and books a fill at 95.
    v1 = _scenario(hard_stop_r_multiple=1.0)
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", broken, None, None, is_entry_day=False, bar_index=2
    )
    assert _reasons(v1_events) == ["HARD_STOP"]
    assert v1_events[0]["price"] == pytest.approx(95.0)

    ec = _execution_convention()
    v2 = _scenario(hard_stop_r_multiple=1.0)
    events = m._evaluate_tracker_scenario_bar(
        v2,
        "LONG",
        "2026-01-05",
        broken,
        None,
        None,
        is_entry_day=False,
        bar_index=m.TRACKER_MAX_HOLD_DAYS - 1,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert events == []
    assert v2["realized_pnl"] == pytest.approx(0.0)
    assert v2["remaining_shares"] == 100
    assert m._scenario_is_open(v2["status"])

    # The clock was not stalled by the unusable bar: the next bar is the max-hold
    # bar and the scenario force-closes at its close.
    later = m._evaluate_tracker_scenario_bar(
        v2,
        "LONG",
        "2026-01-06",
        _bar(open_=101, high=103, low=99, close=102),
        None,
        None,
        is_entry_day=False,
        bar_index=m.TRACKER_MAX_HOLD_DAYS,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert _reasons(later) == ["TIME_STOP"]
    assert v2["status"] == "TIME_STOP"

    fill = ec.resolve_fill("LONG", "stop", 95.0, {"open": 93.0, "high": 90.0, "low": 95.0, "close": 92.0})
    assert fill.booked is False
    assert fill.basis == "invalid_bar"

    nan_close = ec.resolve_fill("LONG", "stop", 95.0, {"open": 93.0, "high": 96.0, "low": 90.0, "close": float("nan")})
    assert nan_close.booked is False
    assert nan_close.basis == "invalid_bar"


# ---------------------------------------------------------------------------
# 9. The golden: the default replay is byte-identical
# ---------------------------------------------------------------------------
GOLDEN_ANCHOR_IDX = 8
GOLDEN_ENTRY_IDX = 25
GOLDEN_GAP_IDX = 32
GOLDEN_BARS = 45


def _build_golden_daily_bars() -> list[dict]:
    """JSON-native synthetic daily bars. No live data, no network.

    Bar 32 opens 8 points below the prior close and never trades back to the
    scenario's hard stop - that is the whole point of the fixture.
    """
    dates = pd.bdate_range("2026-03-02", periods=GOLDEN_BARS)
    rows = []
    for i, stamp in enumerate(dates):
        base = 100.0 + i * 0.25
        if i == GOLDEN_GAP_IDX:
            gap = base - 8.0
            row = {
                "datetime": stamp.date().isoformat(),
                "open": gap,
                "high": gap + 0.5,
                "low": gap - 1.5,
                "close": gap - 1.0,
                "volume": 4_000_000.0,
            }
        else:
            row = {
                "datetime": stamp.date().isoformat(),
                "open": base - 0.1,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.2,
                "volume": 1_000_000.0 + i * 1_000.0,
            }
        rows.append(row)
    return rows


def _build_golden_setup(bars: list[dict]) -> dict:
    entry_price = float(bars[GOLDEN_ENTRY_IDX]["close"])
    return {
        "symbol": "STTEST",
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "setup_family": "avwap_retest_followthrough",
        "entry_price": entry_price,
        "entry_trade_date": bars[GOLDEN_ENTRY_IDX]["datetime"],
        "scan_date": bars[GOLDEN_ENTRY_IDX]["datetime"],
        "anchor_date": bars[GOLDEN_ANCHOR_IDX]["datetime"],
        "scenarios": {
            "full_band3": {
                "tradeable": True,
                "shares": 100,
                "direction": 1.0,
                "initial_risk_per_share": 5.0,
                "initial_risk_usd": 500.0,
                "hard_stop_r_multiple": 1.0,
                "stop_reference_label": "LOWER_1",
                "stop_reference_level": entry_price - 5.0,
                "partial_target_label": "UPPER_2",
                "final_target_label": "UPPER_3",
                "trail_after_partial_label": "LOWER_1",
                "close_failure_limit": 2,
                "status": "OPEN",
            }
        },
    }


def _golden_frame(bars: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(bars))
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _json_plain(node):
    """Everything a record can hold, in JSON-native form.

    Non-finite floats become their tagged strings so equality is real equality;
    `NaN != NaN` would otherwise make a golden compare unequal to itself.
    """
    if isinstance(node, dict):
        return {str(key): _json_plain(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_json_plain(value) for value in node]
    if isinstance(node, bool) or node is None or isinstance(node, str):
        return node
    if isinstance(node, int):
        return int(node)
    if isinstance(node, float):
        if math.isnan(node):
            return "__nan__"
        if math.isinf(node):
            return "__inf__" if node > 0 else "__-inf__"
        return float(node)
    if hasattr(node, "item"):  # numpy scalar
        return _json_plain(node.item())
    return str(node)


def _canonical(node) -> str:
    return json.dumps(_json_plain(node), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_golden_fixture(path: Path) -> None:
    bars = _build_golden_daily_bars()
    setup = _build_golden_setup(bars)
    record = m.recompute_tracker_setup_record(copy.deepcopy(setup), _golden_frame(bars))
    payload = {
        "schema": "st3_replay_golden/1",
        "feature_version": "tracker_replay_default_path",
        "raw_input_keys": ["daily_bars", "setup"],
        "raw_input_sha256": "",
        "acquired_at": "2026-09-06T00:00:00-07:00",
        "universe_version": "synthetic-single-symbol-STTEST",
        "provider_assumptions": (
            "No provider. Synthetic daily bars built in the test module; the "
            "record is whatever recompute_tracker_setup_record produced on main "
            "before packet ST3 existed."
        ),
        "as_of": "2026-09-06T00:00:00-07:00",
        "expected_keys": ["record"],
        "numeric_tolerance": 0.0,
        "intentional_difference": (
            "None in the default path - this is a pure characterization pin. The "
            "gap_aware_v2 / prior_session_v2 differences are asserted by the test, "
            "not stored here."
        ),
        "daily_bars": bars,
        "setup": _json_plain(setup),
        "record": _json_plain(record),
    }
    digest_payload = {key: payload[key] for key in payload["raw_input_keys"]}
    payload["raw_input_sha256"] = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_the_default_replay_reproduces_the_pinned_golden_and_v2_only_adds_its_keys():
    """Pinned from `main` before the repair existed (`ST3_REGEN_GOLDEN=1` re-pins)."""
    fixture_path = FIXTURES_DIR / f"{GOLDEN_FIXTURE_NAME}.json"
    if os.environ.get(REGEN_ENV) == "1":  # pragma: no cover - tester/reviewer door
        _write_golden_fixture(fixture_path)

    contract = load_fixture_contract(GOLDEN_FIXTURE_NAME)
    bars = contract["daily_bars"]
    setup = contract["setup"]
    expected = contract["record"]

    # The fixture's inputs still are the ones this module knows how to build.
    assert _canonical(bars) == _canonical(_build_golden_daily_bars())
    assert _canonical(setup) == _canonical(_build_golden_setup(bars))

    frame = _golden_frame(bars)
    default_record = m.recompute_tracker_setup_record(copy.deepcopy(setup), frame)

    # (a) byte-identical default path - true today and after the repair.
    assert _canonical(default_record) == _canonical(expected)
    assert "execution_convention" not in default_record
    assert "level_knowledge" not in default_record
    # The fixture is worth pinning: the replay really books through the gap bar.
    booked = default_record["scenarios"]["full_band3"]["events"]
    assert [str(event["reason"]) for event in booked] == ["PARTIAL_TARGET", "HARD_STOP"]
    gap_open = float(bars[GOLDEN_GAP_IDX]["open"])
    assert booked[-1]["price"] > gap_open  # today's fill is above anything that traded

    # (b) the opt-in run names its policies and books the reachable price.
    ec = _execution_convention()
    v2_record = m.recompute_tracker_setup_record(
        copy.deepcopy(setup),
        frame,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
        level_knowledge=ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2,
    )
    assert v2_record["execution_convention"] == ec.EXECUTION_GAP_AWARE_V2
    assert v2_record["level_knowledge"] == ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2
    # Same bars walked, same scenarios - only the fills may move.
    assert [mark["trade_date"] for mark in v2_record["daily_marks"]] == [
        mark["trade_date"] for mark in expected["daily_marks"]
    ]
    assert set(v2_record["scenarios"]) == set(expected["scenarios"])
    v2_stop = [e for e in v2_record["scenarios"]["full_band3"]["events"] if e["reason"] == "HARD_STOP"]
    assert v2_stop and v2_stop[0]["price"] == pytest.approx(gap_open)
    assert v2_stop[0]["fill_basis"] == "gap_open"

    # (c) the opt-in run left no residue: the default still reproduces the golden.
    again = m.recompute_tracker_setup_record(copy.deepcopy(setup), frame)
    assert _canonical(again) == _canonical(expected)


# ---------------------------------------------------------------------------
# 10. The comparison CLI
# ---------------------------------------------------------------------------
def _write_compare_inputs(root: Path) -> tuple[Path, Path, Path]:
    bars = _build_golden_daily_bars()
    setup = _build_golden_setup(bars)
    tracker_path = root / "tracker_copy.json"
    tracker_path.write_text(
        json.dumps({"setups": {"STTEST|2026-03-12|LONG": setup}}, indent=2), encoding="utf-8"
    )
    bars_dir = root / "bars"
    bars_dir.mkdir()
    _golden_frame(bars).to_csv(bars_dir / "STTEST.csv", index=False)
    out_dir = root / "out"
    out_dir.mkdir()
    return tracker_path, bars_dir, out_dir


def test_the_compare_cli_refuses_a_protected_out_dir_and_never_overwrites_a_stamp(tmp_path, monkeypatch):
    import tracker_execution_compare as tec

    tracker_path, bars_dir, out_dir = _write_compare_inputs(tmp_path)

    # The scratch-script rule, exercised without going anywhere near the real
    # store: the guard root is monkeypatched onto a scratch tree.
    protected = tmp_path / "pretend_TradingBotData"
    forbidden_out = protected / "runtime" / "st3"
    forbidden_out.mkdir(parents=True)
    monkeypatch.setattr(tec, "PROTECTED_DATA_ROOT", protected)

    status = tec.main(
        [
            "--tracker", str(tracker_path),
            "--bars", str(bars_dir),
            "--out", str(forbidden_out),
            "--limit", "5",
        ]
    )
    assert status != 0
    assert list(forbidden_out.iterdir()) == []

    # A legal out dir: two runs, two stamped files, nothing overwritten.
    first = tec.main(
        ["--tracker", str(tracker_path), "--bars", str(bars_dir), "--out", str(out_dir), "--limit", "5"]
    )
    assert first == 0
    after_first = sorted(p.name for p in out_dir.glob("comparison_*.json"))
    assert len(after_first) == 1
    marker = uuid.uuid4().hex
    stamped = out_dir / after_first[0]
    original = stamped.read_text(encoding="utf-8")
    (out_dir / f"{marker}.sentinel").write_text(marker, encoding="utf-8")

    second = tec.main(
        ["--tracker", str(tracker_path), "--bars", str(bars_dir), "--out", str(out_dir), "--limit", "5"]
    )
    assert second == 0
    after_second = sorted(p.name for p in out_dir.glob("comparison_*.json"))
    assert len(after_second) == 2
    assert stamped.read_text(encoding="utf-8") == original
    assert sorted(p.name for p in out_dir.glob("comparison_*.csv")) == [
        name[:-5] + ".csv" for name in after_second
    ]
