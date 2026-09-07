"""Packet ST3 - no impossible fills, no same-day knowledge.

The tracker replay books a hard stop at the STOP LEVEL even when the whole bar
traded below it (`_apply_scenario_exit_event(..., float(hard_stop_level), ...)`,
`master_avwap_lib/legacy.py`), and it tests a bar's own high/low against
`current_history[<that same day>]`, whose anchored-VWAP bands were computed WITH
that day's bar folded into the cumulative sums. Both are fills the simulated
decision time could not have produced.

These tests pin the ADDITIVE, OPT-IN repair:

* `master_avwap_lib.execution_convention` - `literal_level_v1` and
  `gap_aware_v2`, plus `same_session_v1` and `prior_session_v2` level knowledge;
* `_evaluate_tracker_scenario_bar(..., execution_convention=...)` routing every
  booked exit price through `resolve_fill`;
* `recompute_tracker_setup_record(..., execution_convention=..., level_knowledge=...)`
  carrying the two keys on the record;
* `scripts/tracker_execution_compare.py`, the evidence CLI.

**Packet ST7 (2026-09-06, decision 0019) made `gap_aware_v2` / `prior_session_v2`
the DEFAULT.** Nothing here was weakened for it: every v1 leg below now NAMES
`literal_level_v1` / `same_session_v1` explicitly and asserts exactly the numbers
it asserted while v1 was the default, so v1 stays reproducible forever; every leg
that read "and the default does the same" now asserts that the DEFAULT produces
the v2 answer. Test 9 pins both whole records - `st3_replay_golden` for v1 by
name, `st7_v2_default_golden` for the default.

The two stamps are unconditional since ST7: a record that does not say which
convention produced it is unreadable after a default flip, so a v1 record says
`literal_level_v1` and a default record says `gap_aware_v2`.

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


def _prior_session(levels, trade_date):
    """The `prior_session_levels` block `prior_session_v2` reads (ST3), built
    from levels a PREVIOUS session established. Only the default arms below
    need it - `same_session_v1` never looks at it."""
    return {
        "trade_date": trade_date,
        "anchor_levels": levels,
        "indicator_row": None,
        "dynamic_level_overrides": None,
    }


# ---------------------------------------------------------------------------
# 1-2. Opening gap through the stop, both sides
# ---------------------------------------------------------------------------
def test_a_long_gap_below_the_stop_books_the_open_not_the_untraded_level():
    """The review fixture: entry 100, risk 5, hard stop 95, next bar O80/H85/L79/C82.

    The bar never traded at 95. v1 books it anyway (characterized, unchanged
    and now named explicitly); v2 books the open and is the default since ST7.
    """
    ec = _execution_convention()
    gap_bar = _bar(open_=80, high=85, low=79, close=82)

    # v1 = the convention that shipped until 2026-09-06, BY NAME.
    v1 = _scenario(hard_stop_r_multiple=1.0)
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", gap_bar, None, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
    )
    assert _reasons(v1_events) == ["HARD_STOP"]
    assert v1_events[0]["price"] == pytest.approx(95.0)
    assert v1["realized_r"] == pytest.approx(_expected_r(100.0, 95.0, 100, 1.0, 500.0))
    assert v1["realized_r"] == pytest.approx(-1.014)

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

    # ST7: the same call with NO keyword is now the v2 answer.
    assert ec.DEFAULT_EXECUTION_CONVENTION == ec.EXECUTION_GAP_AWARE_V2
    default = _scenario(hard_stop_r_multiple=1.0)
    default_events = m._evaluate_tracker_scenario_bar(
        default, "LONG", "2026-01-05", gap_bar, None, None, is_entry_day=False, bar_index=2
    )
    assert _reasons(default_events) == ["HARD_STOP"]
    assert default_events[0]["price"] == pytest.approx(80.0)
    assert default_events[0]["fill_basis"] == "gap_open"
    assert default["realized_r"] == pytest.approx(-4.014)


def test_a_short_gap_above_the_stop_books_the_open_not_the_untraded_level():
    """Mirror of the long case: short entry 100, stop 105, bar O120/H121/L115/C118."""
    ec = _execution_convention()
    gap_bar = _bar(open_=120, high=121, low=115, close=118)

    v1 = _scenario(direction=-1.0, hard_stop_r_multiple=1.0, stop_reference_label="UPPER_1", active_stop_label="UPPER_1")
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "SHORT", "2026-01-05", gap_bar, None, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
    )
    assert _reasons(v1_events) == ["HARD_STOP"]
    assert v1_events[0]["price"] == pytest.approx(105.0)
    assert v1["realized_r"] == pytest.approx(_expected_r(100.0, 105.0, 100, -1.0, 500.0))

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

    # ST7: the default books the same short gap.
    default = _scenario(
        direction=-1.0, hard_stop_r_multiple=1.0,
        stop_reference_label="UPPER_1", active_stop_label="UPPER_1",
    )
    default_events = m._evaluate_tracker_scenario_bar(
        default, "SHORT", "2026-01-05", gap_bar, None, None, is_entry_day=False, bar_index=2
    )
    assert default_events[0]["price"] == pytest.approx(120.0)
    assert default_events[0]["fill_basis"] == "gap_open"


# ---------------------------------------------------------------------------
# 3. Missing open -> clamped into the bar
# ---------------------------------------------------------------------------
def test_a_bar_with_no_open_clamps_the_fill_inside_the_bar_under_v2():
    """Long stop 95 against H85/L79/C82. Nothing traded at 95; the honest worst
    case inside the bar is its high."""
    ec = _execution_convention()
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
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
    )
    assert v1_events[0]["price"] == pytest.approx(95.0)

    # ST7: with no keyword at all the clamp is what happens.
    default = _scenario(hard_stop_r_multiple=1.0)
    default_events = m._evaluate_tracker_scenario_bar(
        default,
        "LONG",
        "2026-01-05",
        _bar(high=85, low=79, close=82, include_open=False),
        None,
        None,
        is_entry_day=False,
        bar_index=2,
    )
    assert default_events[0]["price"] == pytest.approx(85.0)
    assert default_events[0]["fill_basis"] == "clamped_no_open"

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
    ec = _execution_convention()
    # This test isolates the EXECUTION axis, so both arms name the SAME level
    # knowledge (`same_session_v1`); the level-knowledge axis is test 7's.
    v1 = _scenario(hard_stop_r_multiple=2.0, partial_target_label="UPPER_2")
    m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", _bar(open_=105, high=111, low=99, close=105),
        levels, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
        level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
    )
    m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-06", _bar(open_=90, high=92, low=88, close=89),
        levels, None, is_entry_day=False, bar_index=3,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
        level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
    )
    assert [round(e["price"], 6) for e in v1["events"]] == [110.0, 90.0]
    assert [e["shares"] for e in v1["events"]] == [50, 50]

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
        level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
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
        level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
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

    # ST7: no keyword at all books the same two legs. The default level
    # knowledge is `prior_session_v2`, so the caller hands the target level the
    # PREVIOUS session established - the same 110 - and the two legs are
    # unchanged. That is the point: the flip moved WHICH DAY's level is read,
    # not the arithmetic.
    default = _scenario(hard_stop_r_multiple=1.0, partial_target_label="UPPER_2")
    m._evaluate_tracker_scenario_bar(
        default, "LONG", "2026-01-05", _bar(open_=105, high=111, low=99, close=105),
        levels, None, is_entry_day=False, bar_index=2,
        prior_session_levels=_prior_session(levels, "2026-01-02"),
    )
    m._evaluate_tracker_scenario_bar(
        default, "LONG", "2026-01-06", _bar(open_=90, high=92, low=88, close=89),
        levels, None, is_entry_day=False, bar_index=3,
        prior_session_levels=_prior_session(levels, "2026-01-05"),
    )
    assert [round(e["price"], 6) for e in default["events"]] == [110.0, 90.0]
    assert default["realized_r"] == pytest.approx(v2["realized_r"])


# ---------------------------------------------------------------------------
# 5. Stop-first survives the gap
# ---------------------------------------------------------------------------
def test_a_gapped_stop_still_wins_the_same_bar_against_a_target():
    """The bar gaps below the stop AND its high reaches the target. Stop-first is
    kept: one event, the stop, at the open - never the target."""
    levels = {"bands": {"UPPER_3": 110.0}}
    ec = _execution_convention()
    bar = _bar(open_=90, high=111, low=88, close=92)

    v1 = _scenario(hard_stop_r_multiple=1.0, final_target_label="UPPER_3")
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", bar, levels, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
    )
    assert _reasons(v1_events) == ["HARD_STOP"]
    assert v1_events[0]["price"] == pytest.approx(95.0)
    assert v1["status"] == "STOPPED"

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

    # ST7: stop-first survives the flip on the default path too.
    default = _scenario(hard_stop_r_multiple=1.0, final_target_label="UPPER_3")
    default_events = m._evaluate_tracker_scenario_bar(
        default, "LONG", "2026-01-05", bar, levels, None, is_entry_day=False, bar_index=2
    )
    assert _reasons(default_events) == ["HARD_STOP"]
    assert "FINAL_TARGET" not in _reasons(default["events"])
    assert default_events[0]["price"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# 6. Target gap
# ---------------------------------------------------------------------------
def test_an_open_above_the_target_fills_at_the_open_under_v2():
    levels = {"bands": {"UPPER_3": 110.0}}
    ec = _execution_convention()
    bar = _bar(open_=114, high=116, low=112, close=115)

    v1 = _scenario(final_target_label="UPPER_3")
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", bar, levels, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
        level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
    )
    assert _reasons(v1_events) == ["FINAL_TARGET"]
    assert v1_events[0]["price"] == pytest.approx(110.0)
    assert v1["realized_r"] == pytest.approx(_expected_r(100.0, 110.0, 100, 1.0, 500.0))

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
        level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
    )
    assert _reasons(v2_events) == ["FINAL_TARGET"]
    assert v2_events[0]["price"] == pytest.approx(114.0)
    assert v2_events[0]["fill_basis"] == "gap_open"
    assert v2["realized_r"] == pytest.approx(_expected_r(100.0, 114.0, 100, 1.0, 500.0))
    assert 112.0 <= v2_events[0]["price"] <= 116.0

    # ST7: the symmetry (a target gap fills BETTER) is the default too, with
    # the target level read off the PREVIOUS session, as the default level
    # knowledge requires.
    default = _scenario(final_target_label="UPPER_3")
    default_events = m._evaluate_tracker_scenario_bar(
        default, "LONG", "2026-01-05", bar, levels, None, is_entry_day=False, bar_index=2,
        prior_session_levels=_prior_session(levels, "2026-01-02"),
    )
    assert default_events[0]["price"] == pytest.approx(114.0)
    assert default_events[0]["fill_basis"] == "gap_open"


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

    ec = _execution_convention()
    setup = _band_move_setup(df, anchor, prev_key)

    # v1 BY NAME - the look-ahead that shipped until 2026-09-06, unchanged.
    same_session = m.recompute_tracker_setup_record(
        copy.deepcopy(setup),
        df,
        level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
    )
    scenario = same_session["scenarios"]["s1"]
    assert scenario["status"] == "TARGET_HIT"
    assert _reasons(scenario["events"]) == ["FINAL_TARGET"]
    assert scenario["events"][0]["price"] == pytest.approx(target_today)

    prior_session = m.recompute_tracker_setup_record(
        copy.deepcopy(setup),
        df,
        level_knowledge=ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2,
    )
    repaired = prior_session["scenarios"]["s1"]
    assert _reasons(repaired["events"]) == []
    assert m._scenario_is_open(repaired["status"])
    assert prior_session["level_knowledge"] == ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2
    # ST7: v2 is the default, and BOTH runs now NAME the policy that produced
    # them - a record with no stamp could not be told apart from either.
    assert ec.DEFAULT_LEVEL_KNOWLEDGE == ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2
    assert same_session["level_knowledge"] == ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1

    default_record = m.recompute_tracker_setup_record(copy.deepcopy(setup), df)
    default_scenario = default_record["scenarios"]["s1"]
    assert _reasons(default_scenario["events"]) == []
    assert m._scenario_is_open(default_scenario["status"])
    assert default_record["level_knowledge"] == ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2


# ---------------------------------------------------------------------------
# 8. An invalid bar books nothing, and the hold clock still runs
# ---------------------------------------------------------------------------
def test_an_invalid_ohlc_bar_books_no_fill_under_v2_and_the_hold_clock_advances():
    # low > high and low > open: the candle invariant `low <= open, close <= high`
    # is broken, so no sequence inside it can be believed.
    ec = _execution_convention()
    broken = _bar(open_=93, high=90, low=95, close=92)

    # v1 BY NAME: the broken bar's `low` clears the stop and books a fill at 95.
    v1 = _scenario(hard_stop_r_multiple=1.0)
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", broken, None, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
    )
    assert _reasons(v1_events) == ["HARD_STOP"]
    assert v1_events[0]["price"] == pytest.approx(95.0)

    # ST7: the default books NOTHING off a candle that contradicts itself.
    default = _scenario(hard_stop_r_multiple=1.0)
    default_events = m._evaluate_tracker_scenario_bar(
        default, "LONG", "2026-01-05", broken, None, None, is_entry_day=False, bar_index=2
    )
    assert default_events == []
    assert m._scenario_is_open(default["status"])

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


#: The stamps ST7 made unconditional. They are excluded from the byte-identity
#: comparison against `st3_replay_golden` - which was pinned before they were
#: always written - and asserted separately by name, so no stamp escapes the
#: assertion and the v1 RECORD is otherwise byte-identical to the pin.
POLICY_STAMP_KEYS = ("execution_convention", "level_knowledge")


def _without_stamps(record: dict) -> dict:
    stripped = dict(record)
    for key in POLICY_STAMP_KEYS:
        stripped.pop(key, None)
    return stripped


def test_v1_by_name_reproduces_the_pinned_golden_and_the_default_is_the_v2_pin():
    """Two whole records, two pins.

    `st3_replay_golden` was pinned from `main` before the repair existed and is
    now reproduced by NAMING `literal_level_v1` / `same_session_v1`
    (`ST3_REGEN_GOLDEN=1` re-pins it). `st7_v2_default_golden` was pinned on
    `main` through the explicit v2 keywords, before the defaults flipped, and is
    what a DEFAULT replay must now produce - so neither pin is a self-portrait.
    """
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
    ec = _execution_convention()
    v1_record = m.recompute_tracker_setup_record(
        copy.deepcopy(setup),
        frame,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
        level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
    )

    # (a) byte-identical v1 path, by name - the characterization is permanent.
    assert _canonical(_without_stamps(v1_record)) == _canonical(_without_stamps(expected))
    assert v1_record["execution_convention"] == ec.EXECUTION_LITERAL_LEVEL_V1
    assert v1_record["level_knowledge"] == ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1
    # The fixture is worth pinning: the replay really books through the gap bar.
    booked = v1_record["scenarios"]["full_band3"]["events"]
    assert [str(event["reason"]) for event in booked] == ["PARTIAL_TARGET", "HARD_STOP"]
    gap_open = float(bars[GOLDEN_GAP_IDX]["open"])
    assert booked[-1]["price"] > gap_open  # v1's fill is above anything that traded

    # (b) the DEFAULT run names its policies and books the reachable price,
    #     and is the whole `st7_v2_default_golden` record.
    default_record = m.recompute_tracker_setup_record(copy.deepcopy(setup), frame)
    assert default_record["execution_convention"] == ec.EXECUTION_GAP_AWARE_V2
    assert default_record["level_knowledge"] == ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2
    v2_pin = load_fixture_contract("st7_v2_default_golden")
    assert _canonical(v2_pin["daily_bars"]) == _canonical(bars)
    assert _canonical(_without_stamps(default_record)) == _canonical(v2_pin["record"])
    assert _canonical(v2_pin["record"]) != _canonical(_without_stamps(expected))
    # Same bars walked, same scenarios - only the fills moved.
    assert [mark["trade_date"] for mark in default_record["daily_marks"]] == [
        mark["trade_date"] for mark in expected["daily_marks"]
    ]
    assert set(default_record["scenarios"]) == set(expected["scenarios"])
    v2_stop = [
        e for e in default_record["scenarios"]["full_band3"]["events"]
        if e["reason"] == "HARD_STOP"
    ]
    assert v2_stop and v2_stop[0]["price"] == pytest.approx(gap_open)
    assert v2_stop[0]["fill_basis"] == "gap_open"

    # The explicit v2 keywords are the same run as the default.
    explicit_v2 = m.recompute_tracker_setup_record(
        copy.deepcopy(setup),
        frame,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
        level_knowledge=ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2,
    )
    assert _canonical(explicit_v2) == _canonical(default_record)

    # (c) no residue in either direction: v1 by name is still v1 afterwards.
    again = m.recompute_tracker_setup_record(
        copy.deepcopy(setup),
        frame,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
        level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
    )
    assert _canonical(again) == _canonical(v1_record)


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
