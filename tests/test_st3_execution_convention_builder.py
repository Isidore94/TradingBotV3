"""Packet ST3, builder additions to the tester's suite.

Three things the tester's file left open, each named by the lead when the
packet's one ambiguity was decided:

1. **ST3.2's counter.** ``prior_session_v2`` skips an intrabar test it cannot
   answer, and the skip is COUNTED as ``no_prior_session_level`` on the
   scenario. A skip that is not counted reads as "the target was not hit",
   which is a made-up answer; these tests are what make the count real.
2. **An invalid bar ON the maximum-hold index** (lead decision, 2026-09-06):
   it books nothing, and the ``TIME_STOP`` fires on the next VALID bar with
   ``fill_basis`` ``deferred_invalid_bar``. Maximum hold is preserved, not
   cancelled, and the record says which bar could not answer.
3. **The default event dict is unchanged**, key for key - the additive keys
   appear only under ``gap_aware_v2``.

Nothing here weakens or restates a tester assertion; it only covers ground the
tester's file does not.

**Packet ST7 (2026-09-06, decision 0019) made `gap_aware_v2` / `prior_session_v2`
the DEFAULT.** No assertion below was removed for it. Every leg that used to
exercise v1 through the bare signature now NAMES `literal_level_v1` /
`same_session_v1` and asserts the same numbers, and each of those tests gained a
DEFAULT leg asserting the v2 answer. A test whose subject is ONE axis names the
other axis on both arms rather than letting it drift with the default.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402

from master_avwap_lib import execution_convention as ec  # noqa: E402


def _bar(*, high, low, close, open_=None):
    row = {"high": float(high), "low": float(low), "close": float(close)}
    if open_ is not None:
        row["open"] = float(open_)
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


# ---------------------------------------------------------------------------
# ST3.2 - the skip is counted, never silently read as "not hit"
# ---------------------------------------------------------------------------
def test_a_missing_prior_session_level_is_counted_and_the_target_is_not_booked():
    """No prior session at all: the target that WOULD have booked today is not
    booked, and the reason is on the scenario as a count."""
    levels = {"bands": {"UPPER_3": 110.0}}
    bar = _bar(open_=105, high=111, low=104, close=110)

    # v1 BY NAME books it - the look-ahead that shipped until 2026-09-06.
    same_session = _scenario(final_target_label="UPPER_3")
    events = m._evaluate_tracker_scenario_bar(
        same_session, "LONG", "2026-01-05", bar, levels, None, is_entry_day=False, bar_index=2,
        level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
    )
    assert [event["reason"] for event in events] == ["FINAL_TARGET"]
    assert "intrabar_skip_reasons" not in same_session

    # ST7: with NO keyword the skip is what happens, and it is counted.
    default = _scenario(final_target_label="UPPER_3")
    assert m._evaluate_tracker_scenario_bar(
        default, "LONG", "2026-01-05", bar, levels, None, is_entry_day=False, bar_index=2,
        prior_session_levels=None,
    ) == []
    assert default["intrabar_skip_reasons"] == {ec.NO_PRIOR_SESSION_LEVEL: 1}

    prior = _scenario(final_target_label="UPPER_3")
    skipped = m._evaluate_tracker_scenario_bar(
        prior,
        "LONG",
        "2026-01-05",
        bar,
        levels,
        None,
        is_entry_day=False,
        bar_index=2,
        level_knowledge=ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2,
        prior_session_levels=None,
    )
    assert skipped == []
    assert m._scenario_is_open(prior["status"])
    # Counted ONCE per bar, not once per resolution: the evaluator resolves the
    # final-target label twice and only the live one counts.
    assert prior["intrabar_skip_reasons"] == {ec.NO_PRIOR_SESSION_LEVEL: 1}

    # A second unanswerable bar accumulates rather than overwriting.
    m._evaluate_tracker_scenario_bar(
        prior,
        "LONG",
        "2026-01-06",
        bar,
        levels,
        None,
        is_entry_day=False,
        bar_index=3,
        level_knowledge=ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2,
        prior_session_levels=None,
    )
    assert prior["intrabar_skip_reasons"] == {ec.NO_PRIOR_SESSION_LEVEL: 2}


def test_a_prior_session_level_that_exists_answers_the_test_and_counts_nothing():
    levels = {"bands": {"UPPER_3": 110.0}}
    bar = _bar(open_=105, high=111, low=104, close=110)
    context = {
        "trade_date": "2026-01-02",
        "anchor_levels": {"bands": {"UPPER_3": 120.0}},
        "indicator_row": None,
        "dynamic_level_overrides": {},
    }

    scenario = _scenario(final_target_label="UPPER_3")
    events = m._evaluate_tracker_scenario_bar(
        scenario,
        "LONG",
        "2026-01-05",
        bar,
        levels,
        None,
        is_entry_day=False,
        bar_index=2,
        level_knowledge=ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2,
        prior_session_levels=context,
    )
    # Yesterday's band at 120 is above the bar's high of 111: answered, not hit.
    assert events == []
    assert "intrabar_skip_reasons" not in scenario

    # And a prior session whose band the bar DOES reach still books.
    reachable = dict(context, anchor_levels={"bands": {"UPPER_3": 108.0}})
    booked = _scenario(final_target_label="UPPER_3")
    booked_events = m._evaluate_tracker_scenario_bar(
        booked,
        "LONG",
        "2026-01-05",
        bar,
        levels,
        None,
        is_entry_day=False,
        bar_index=2,
        level_knowledge=ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2,
        prior_session_levels=reachable,
    )
    assert [event["reason"] for event in booked_events] == ["FINAL_TARGET"]
    # Level knowledge decides WHICH level; it never decides the fill price, so
    # with the default convention the fill is still the literal level.
    assert booked_events[0]["price"] == pytest.approx(108.0)
    assert "intrabar_skip_reasons" not in booked


def _late_anchor_frame(periods: int = 40):
    """A frame anchored on its LAST bar, so no prior session has a band."""
    dates = pd.bdate_range("2026-03-02", periods=periods)
    rows = []
    for i, stamp in enumerate(dates):
        base = 100.0 + i * 0.3
        rows.append(
            {
                "datetime": stamp,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.2,
                "volume": 1_000_000.0 + i * 1_000.0,
            }
        )
    return pd.DataFrame(rows)


def test_the_record_replay_counts_the_skip_when_the_anchor_starts_that_day():
    """Through the real ``recompute_tracker_setup_record``: the anchor is the
    LAST bar, so the band history has no key before it and the prior-session
    lookup has nothing to read."""
    df = _late_anchor_frame()
    anchor = df.iloc[-1]["datetime"].date().isoformat()
    entry_date = df.iloc[-2]["datetime"].date().isoformat()
    entry_price = float(df.iloc[-2]["close"])
    history = m.calc_anchored_vwap_band_history(df, anchor)
    assert set(history) == {anchor}, "premise: only the anchor day has a band"
    target_today = float(history[anchor]["bands"]["UPPER_3"])
    assert target_today <= float(df.iloc[-1]["high"]), "premise: today's band is reachable"

    setup = {
        "symbol": "STSKIP",
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

    same_session = m.recompute_tracker_setup_record(
        copy.deepcopy(setup), df, level_knowledge=ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1
    )
    booked = same_session["scenarios"]["s1"]
    assert [event["reason"] for event in booked["events"]] == ["FINAL_TARGET"]
    assert "intrabar_skip_reasons" not in booked
    assert same_session["level_knowledge"] == ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1

    prior = m.recompute_tracker_setup_record(
        copy.deepcopy(setup), df, level_knowledge=ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2
    )
    repaired = prior["scenarios"]["s1"]
    assert repaired["events"] == []
    assert repaired["intrabar_skip_reasons"] == {ec.NO_PRIOR_SESSION_LEVEL: 1}
    assert prior["level_knowledge"] == ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2

    # ST7: the DEFAULT record is the repaired one, and it says so.
    default = m.recompute_tracker_setup_record(copy.deepcopy(setup), df)
    default_scenario = default["scenarios"]["s1"]
    assert default_scenario["events"] == []
    assert default_scenario["intrabar_skip_reasons"] == {ec.NO_PRIOR_SESSION_LEVEL: 1}
    assert default["level_knowledge"] == ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2


# ---------------------------------------------------------------------------
# The lead's decision: an invalid bar ON the maximum-hold index
# ---------------------------------------------------------------------------
def test_an_invalid_bar_on_the_max_hold_index_defers_the_time_stop_to_the_next_valid_bar():
    """The bar that would have force-closed the trade is unusable.

    It books nothing - a candle whose low is above its high cannot price an
    exit - and the force close is DEFERRED, never cancelled: the next valid bar
    books the TIME_STOP and says why it is late.
    """
    broken = _bar(open_=93, high=90, low=95, close=92)

    # v1 BY NAME, characterized: the broken candle's `low` clears the hard stop
    # and books a fill at a level nothing traded at.
    v1 = _scenario(hard_stop_r_multiple=1.0)
    v1_events = m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", broken, None, None,
        is_entry_day=False, bar_index=m.TRACKER_MAX_HOLD_DAYS,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
    )
    assert [event["reason"] for event in v1_events] == ["HARD_STOP"]
    assert v1_events[0]["price"] == pytest.approx(95.0)

    # ST7: with NO keyword the same bar books nothing and defers the clock.
    default = _scenario(hard_stop_r_multiple=1.0)
    assert m._evaluate_tracker_scenario_bar(
        default, "LONG", "2026-01-05", broken, None, None,
        is_entry_day=False, bar_index=m.TRACKER_MAX_HOLD_DAYS,
    ) == []
    assert default["time_stop_deferred"] is True

    v2 = _scenario(hard_stop_r_multiple=1.0)
    nothing = m._evaluate_tracker_scenario_bar(
        v2,
        "LONG",
        "2026-01-05",
        broken,
        None,
        None,
        is_entry_day=False,
        bar_index=m.TRACKER_MAX_HOLD_DAYS,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert nothing == []
    assert m._scenario_is_open(v2["status"])
    assert v2["remaining_shares"] == 100
    assert v2["time_stop_deferred"] is True

    later = m._evaluate_tracker_scenario_bar(
        v2,
        "LONG",
        "2026-01-06",
        _bar(open_=101, high=103, low=99, close=102),
        None,
        None,
        is_entry_day=False,
        bar_index=m.TRACKER_MAX_HOLD_DAYS + 1,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert [event["reason"] for event in later] == ["TIME_STOP"]
    assert later[0]["fill_basis"] == ec.FILL_BASIS_DEFERRED_INVALID_BAR
    assert later[0]["price"] == pytest.approx(102.0)
    assert v2["status"] == "TIME_STOP"
    # The flag is consumed, not left on the record for a later reader to trip on.
    assert "time_stop_deferred" not in v2


def test_a_time_stop_on_a_valid_bar_is_a_plain_close_fill_under_v2():
    """The deferral basis has to MEAN something, so the ordinary case is named
    separately: a maximum-hold force close on a usable bar is `close`."""
    scenario = _scenario()
    events = m._evaluate_tracker_scenario_bar(
        scenario,
        "LONG",
        "2026-01-06",
        _bar(open_=101, high=103, low=99, close=102),
        None,
        None,
        is_entry_day=False,
        bar_index=m.TRACKER_MAX_HOLD_DAYS,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert [event["reason"] for event in events] == ["TIME_STOP"]
    assert events[0]["fill_basis"] == ec.FILL_BASIS_CLOSE
    assert events[0]["execution_convention"] == ec.EXECUTION_GAP_AWARE_V2


# ---------------------------------------------------------------------------
# Which keys each convention's event dict carries
# ---------------------------------------------------------------------------
SHIPPED_EVENT_KEYS = {
    "trade_date", "reason", "price", "shares", "pnl", "gross_pnl", "cost"
}


def test_the_v1_event_dict_keeps_its_keys_and_the_default_adds_exactly_two():
    """v1's event dict is the shipped one, key for key.

    Until ST7 that was also the DEFAULT event dict. It is not any more: the
    default is `gap_aware_v2`, whose events carry `fill_basis` and
    `execution_convention` - two ADDITIVE keys and no more, which is what this
    test now pins on both arms.
    """
    scenario = _scenario(hard_stop_r_multiple=1.0)
    events = m._evaluate_tracker_scenario_bar(
        scenario, "LONG", "2026-01-05", _bar(open_=80, high=85, low=79, close=82),
        None, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
    )
    assert set(events[0]) == SHIPPED_EVENT_KEYS

    default_scenario = _scenario(hard_stop_r_multiple=1.0)
    default_events = m._evaluate_tracker_scenario_bar(
        default_scenario, "LONG", "2026-01-05", _bar(open_=80, high=85, low=79, close=82),
        None, None, is_entry_day=False, bar_index=2,
    )
    assert set(default_events[0]) == SHIPPED_EVENT_KEYS | {
        "fill_basis", "execution_convention"
    }
    assert default_events[0]["execution_convention"] == ec.EXECUTION_GAP_AWARE_V2

    # The low-level writer is convention-blind and still writes the seven.
    direct = m._apply_scenario_exit_event(_scenario(), 100, 110.0, "2026-01-02", "FINAL_TARGET")
    assert set(direct) == SHIPPED_EVENT_KEYS


def test_a_close_based_stop_fail_is_named_close_under_v2_and_unnamed_under_v1():
    """The two-closes protective stop is a CLOSE decision, so no gap logic
    touches it under either policy - only its basis label appears."""
    levels = {"bands": {"LOWER_1": 99.0}}
    for convention, expected in (
        (ec.EXECUTION_LITERAL_LEVEL_V1, None),
        (ec.EXECUTION_GAP_AWARE_V2, ec.FILL_BASIS_CLOSE),
    ):
        scenario = _scenario(close_failure_limit=1)
        events = m._evaluate_tracker_scenario_bar(
            scenario,
            "LONG",
            "2026-01-05",
            _bar(open_=99.5, high=100.0, low=97.0, close=98.0),
            levels,
            None,
            is_entry_day=False,
            bar_index=2,
            execution_convention=convention,
        )
        assert [event["reason"] for event in events] == ["STOP_FAIL"]
        assert events[0]["price"] == pytest.approx(98.0)
        assert events[0].get("fill_basis") == expected


# ---------------------------------------------------------------------------
# The pure function's own guard rails
# ---------------------------------------------------------------------------
def test_resolve_fill_never_returns_a_price_outside_the_bar():
    bar = {"open": 96.0, "high": 101.0, "low": 96.0, "close": 99.0}
    # A level below the whole bar with a non-gapping open cannot be reached from
    # the replay (the hit test guarantees otherwise), and is clamped anyway.
    fill = ec.resolve_fill("LONG", "stop", 90.0, bar)
    assert fill.booked is True
    assert fill.price == pytest.approx(96.0)
    assert fill.basis == ec.FILL_BASIS_CLAMPED_LEVEL

    with pytest.raises(ValueError):
        ec.resolve_fill("LONG", "trail", 95.0, bar)

    absent = ec.resolve_fill("LONG", "stop", None, bar)
    assert absent.booked is False
    assert absent.basis == ec.FILL_BASIS_NO_LEVEL


def test_bar_is_valid_reads_a_missing_open_as_unknown_not_as_broken():
    assert ec.bar_is_valid({"high": 101.0, "low": 99.0, "close": 100.0}) is True
    assert ec.bar_is_valid({"open": float("nan"), "high": 101.0, "low": 99.0, "close": 100.0}) is True
    # An OPEN outside its own bar is broken, which is a different thing.
    assert ec.bar_is_valid({"open": 105.0, "high": 101.0, "low": 99.0, "close": 100.0}) is False
    assert ec.bar_is_valid({"high": 99.0, "low": 101.0, "close": 100.0}) is False
    assert ec.bar_is_valid({"high": 101.0, "low": 99.0, "close": float("nan")}) is False


# ---------------------------------------------------------------------------
# Review fix round (2026-09-06), advisories 2 and 3
# ---------------------------------------------------------------------------
def test_the_deferral_flag_never_labels_an_exit_that_is_not_a_time_stop():
    """A deferred maximum-hold close is not a licence to relabel other exits.

    The flag says "an unusable bar sat on the max-hold index". It must reach
    the `TIME_STOP` and NOTHING else: an exit that fires ahead of the time stop
    on the same bar is its own decision, and it must clear the flag rather than
    leave a stale `time_stop_deferred: True` on a closed record.
    """
    levels = {"bands": {"LOWER_1": 99.0}}
    broken = _bar(open_=93, high=90, low=95, close=92)

    scenario = _scenario(close_failure_limit=1)
    assert m._evaluate_tracker_scenario_bar(
        scenario, "LONG", "2026-01-05", broken, levels, None,
        is_entry_day=False, bar_index=m.TRACKER_MAX_HOLD_DAYS,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    ) == []
    assert scenario["time_stop_deferred"] is True

    # The next bar closes under the protective stop, so the two-closes stop
    # fires BEFORE the time stop is reached.
    events = m._evaluate_tracker_scenario_bar(
        scenario,
        "LONG",
        "2026-01-06",
        _bar(open_=99.5, high=100.0, low=97.0, close=98.0),
        levels,
        None,
        is_entry_day=False,
        bar_index=m.TRACKER_MAX_HOLD_DAYS + 1,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert [event["reason"] for event in events] == ["STOP_FAIL"]
    assert events[0]["fill_basis"] == ec.FILL_BASIS_CLOSE
    assert events[0]["fill_basis"] != ec.FILL_BASIS_DEFERRED_INVALID_BAR
    assert "time_stop_deferred" not in scenario


def test_the_deferral_flag_is_cleared_when_a_hard_stop_closes_the_scenario():
    broken = _bar(open_=93, high=90, low=95, close=92)
    scenario = _scenario(hard_stop_r_multiple=1.0)
    m._evaluate_tracker_scenario_bar(
        scenario, "LONG", "2026-01-05", broken, None, None,
        is_entry_day=False, bar_index=m.TRACKER_MAX_HOLD_DAYS,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert scenario["time_stop_deferred"] is True

    events = m._evaluate_tracker_scenario_bar(
        scenario,
        "LONG",
        "2026-01-06",
        _bar(open_=80.0, high=85.0, low=79.0, close=82.0),
        None,
        None,
        is_entry_day=False,
        bar_index=m.TRACKER_MAX_HOLD_DAYS + 1,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert [event["reason"] for event in events] == ["HARD_STOP"]
    assert events[0]["fill_basis"] == ec.FILL_BASIS_GAP_OPEN
    assert "time_stop_deferred" not in scenario


def test_the_deferral_flag_is_cleared_when_a_final_target_closes_the_scenario():
    levels = {"bands": {"UPPER_3": 110.0}}
    broken = _bar(open_=93, high=90, low=95, close=92)
    scenario = _scenario(final_target_label="UPPER_3")
    m._evaluate_tracker_scenario_bar(
        scenario, "LONG", "2026-01-05", broken, levels, None,
        is_entry_day=False, bar_index=m.TRACKER_MAX_HOLD_DAYS,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert scenario["time_stop_deferred"] is True

    # ST7: the default level knowledge is `prior_session_v2`, so the target the
    # replay may test against is the one the PREVIOUS session established. It
    # is the same 110 here - the subject of this test is the deferral flag, not
    # which day's level is read.
    events = m._evaluate_tracker_scenario_bar(
        scenario,
        "LONG",
        "2026-01-06",
        _bar(open_=114.0, high=116.0, low=112.0, close=115.0),
        levels,
        None,
        is_entry_day=False,
        bar_index=m.TRACKER_MAX_HOLD_DAYS + 1,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
        prior_session_levels={
            "trade_date": "2026-01-05",
            "anchor_levels": levels,
            "indicator_row": None,
            "dynamic_level_overrides": None,
        },
    )
    assert [event["reason"] for event in events] == ["FINAL_TARGET"]
    assert events[0]["fill_basis"] == ec.FILL_BASIS_GAP_OPEN
    assert "time_stop_deferred" not in scenario


def test_an_invalid_bar_is_counted_the_way_a_missing_prior_level_is():
    """A skipped bar that is not counted is indistinguishable from a quiet one.

    `no_prior_session_level` is counted, so `invalid_bar` is too - in the
    sibling `skipped_bar_reasons`, because an invalid candle skips the WHOLE
    bar (excursion and unrealized mark included), not just an intrabar test.
    """
    broken = _bar(open_=93, high=90, low=95, close=92)

    # v1 BY NAME books through it and counts nothing: the count is a v2 idea.
    v1 = _scenario(hard_stop_r_multiple=1.0)
    m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", broken, None, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
    )
    assert "skipped_bar_reasons" not in v1

    # ST7: the default counts it.
    default = _scenario(hard_stop_r_multiple=1.0)
    assert m._evaluate_tracker_scenario_bar(
        default, "LONG", "2026-01-05", broken, None, None, is_entry_day=False, bar_index=2
    ) == []
    assert default["skipped_bar_reasons"] == {ec.FILL_BASIS_INVALID_BAR: 1}

    v2 = _scenario(hard_stop_r_multiple=1.0)
    assert m._evaluate_tracker_scenario_bar(
        v2, "LONG", "2026-01-05", broken, None, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    ) == []
    assert v2["skipped_bar_reasons"] == {ec.FILL_BASIS_INVALID_BAR: 1}

    # A second unusable bar accumulates; a usable one adds nothing.
    m._evaluate_tracker_scenario_bar(
        v2, "LONG", "2026-01-06", _bar(high=99.0, low=101.0, close=100.0), None, None,
        is_entry_day=False, bar_index=3, execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert v2["skipped_bar_reasons"] == {ec.FILL_BASIS_INVALID_BAR: 2}
    m._evaluate_tracker_scenario_bar(
        v2, "LONG", "2026-01-07", _bar(open_=101, high=103, low=99, close=102), None, None,
        is_entry_day=False, bar_index=4, execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert v2["skipped_bar_reasons"] == {ec.FILL_BASIS_INVALID_BAR: 2}


def test_an_invalid_bar_under_v2_skips_the_excursion_and_the_unrealized_mark():
    """The consequence advisory 4 asks to be STATED, pinned so it stays true.

    v1 reads an excursion off a candle whose low is above its own high. v2
    reads nothing off it at all - which is the point, and is also why the
    skip has to be counted.
    """
    broken = _bar(open_=93, high=90, low=95, close=92)

    v1 = _scenario()
    m._evaluate_tracker_scenario_bar(
        v1, "LONG", "2026-01-05", broken, None, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_LITERAL_LEVEL_V1,
    )
    assert v1["max_adverse_r"] > 0.0
    assert v1["unrealized_pnl"] != 0.0

    v2 = _scenario()
    m._evaluate_tracker_scenario_bar(
        v2, "LONG", "2026-01-05", broken, None, None, is_entry_day=False, bar_index=2,
        execution_convention=ec.EXECUTION_GAP_AWARE_V2,
    )
    assert "max_adverse_r" not in v2
    assert "max_favorable_r" not in v2
    assert "unrealized_pnl" not in v2

    # ST7: the DEFAULT reads nothing off it either.
    default = _scenario()
    m._evaluate_tracker_scenario_bar(
        default, "LONG", "2026-01-05", broken, None, None, is_entry_day=False, bar_index=2
    )
    assert "max_adverse_r" not in default
    assert "max_favorable_r" not in default
    assert "unrealized_pnl" not in default
