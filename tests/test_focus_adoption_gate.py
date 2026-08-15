"""The combined M5 Focus adoption gate (plan.md Phase 0.5, packet R2 Part A).

Trader rule 2026-08-14: an auto M5 Focus pick must be **above the previous
day's high AND above session VWAP** on the M5 for longs, and below both for
shorts. The same test runs at candidate build, on every staging refresh (a
pick that falls back through either level is evicted), and again at adoption.

This file opens with the golden characterization fixture required by
plan.md sec 5 before any detector/routing change: it freezes what the
candidate filter selected BEFORE the VWAP half existed, so the gate's effect
on selection is a reviewable diff rather than an assertion nobody can check.
"""

import sys
from pathlib import Path

import pytest

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_NAME = "auto_pick_focus_gate_v1"


def _filtered(fixture):
    import autopilot_core as core

    result = core.filter_candidates_by_prev_day_extremes(
        fixture["candidates"], fixture["profiles"], fixture["daily_context"]
    )
    return {
        side: sorted(row["symbol"] for row in result[side])
        for side in ("longs", "shorts")
    }


def test_candidate_filter_golden_fixture():
    """Loading re-verifies raw_input_sha256 over the fixture's own inputs, so
    editing a profile without re-freezing the expectations fails here."""
    fixture = load_fixture_contract(FIXTURE_NAME)
    assert fixture.schema == "auto_pick_focus_gate_v1"
    fixture.assert_matches(_filtered(fixture), fixture["expected"], "candidate filter")


def test_the_gate_removed_exactly_the_documented_rows():
    """Name the difference the fixture's re-freeze recorded.

    The baseline kept 5 longs and 2 shorts. Three of those survivors were the
    cases the VWAP half exists for, and two more are the completed-bar rule
    biting. Spelling them out here means a future edit that quietly widens the
    gate cannot hide inside a re-frozen expectation.
    """
    fixture = load_fixture_contract(FIXTURE_NAME)
    kept = _filtered(fixture)
    assert kept["longs"] == ["LONG_PDH_ABOVE_VWAP"]
    assert kept["shorts"] == ["SHORT_PDL_BELOW_VWAP"]


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def test_a_long_needs_both_halves():
    from focus_adoption_gate import passes_focus_adoption_gate

    # Above yesterday's high AND above VWAP.
    passes, reason = passes_focus_adoption_gate("long", 103.0, 100.5, 98.0, 101.0)
    assert passes and "above" in reason

    # Above the high, under VWAP.
    passes, reason = passes_focus_adoption_gate("long", 103.0, 100.5, 98.0, 104.0)
    assert not passes and reason == "not above session VWAP"

    # Above VWAP, inside yesterday's range.
    passes, reason = passes_focus_adoption_gate("long", 100.0, 100.5, 98.0, 99.0)
    assert not passes and reason == "not above yesterday's high"

    # Neither.
    passes, reason = passes_focus_adoption_gate("long", 99.0, 100.5, 98.0, 99.5)
    assert not passes and reason == "not above yesterday's high and not above session VWAP"


def test_a_short_mirrors_it():
    from focus_adoption_gate import passes_focus_adoption_gate

    passes, _ = passes_focus_adoption_gate("short", 95.0, 100.5, 98.0, 97.0)
    assert passes

    passes, reason = passes_focus_adoption_gate("short", 95.0, 100.5, 98.0, 94.0)
    assert not passes and reason == "not below session VWAP"

    passes, reason = passes_focus_adoption_gate("short", 99.0, 100.5, 98.0, 100.0)
    assert not passes and reason == "not below yesterday's low"


def test_unknown_always_fails_and_says_which_half():
    """plan.md sec 5: missing data is uncertainty, never confirmation."""
    from focus_adoption_gate import UNKNOWN, focus_adoption_gate_state

    state, reason = focus_adoption_gate_state("long", 103.0, 100.5, 98.0, None)
    assert state == UNKNOWN and reason == "cannot verify session VWAP"

    state, reason = focus_adoption_gate_state("long", 103.0, None, None, 101.0)
    assert state == UNKNOWN and "break of yesterday's high" in reason

    state, reason = focus_adoption_gate_state("long", None, 100.5, 98.0, 101.0)
    assert state == UNKNOWN

    state, reason = focus_adoption_gate_state("long", None, None, None, None)
    assert state == UNKNOWN and "no completed price" in reason


def test_a_non_finite_price_or_vwap_is_unknown_not_a_pass():
    """NaN compares False against everything, so an unguarded gate would read
    a NaN VWAP as 'not above' - or worse, a NaN price as a pass on the other
    side. Both must be UNKNOWN."""
    from focus_adoption_gate import UNKNOWN, focus_adoption_gate_state

    for bad in (float("nan"), float("inf"), "not a number"):
        state, _ = focus_adoption_gate_state("long", 103.0, 100.5, 98.0, bad)
        assert state == UNKNOWN, f"vwap={bad!r}"
        state, _ = focus_adoption_gate_state("long", bad, 100.5, 98.0, 101.0)
        assert state == UNKNOWN, f"price={bad!r}"


def test_the_gate_reads_the_completed_bar_not_the_forming_one():
    """A break the forming bar made and then closed back inside is the noise
    this gate removes, so the candidate verdict reads `last_complete`."""
    import autopilot_core as core

    ctx = {"prev_high": 100.5, "prev_low": 98.0}
    forming_only = {"last": 103.0, "last_complete": 100.2, "completed_session_vwap": 99.0}
    passes, reason = core.candidate_focus_gate_verdict("long", forming_only, ctx)
    assert not passes and reason == "not above yesterday's high"

    completed = {"last": 100.2, "last_complete": 103.0, "completed_session_vwap": 99.0}
    passes, _ = core.candidate_focus_gate_verdict("long", completed, ctx)
    assert passes, "a break the completed bar holds still qualifies"


def test_the_filter_fails_open_only_when_there_is_no_daily_store_at_all():
    """One unknown symbol fails alone; a missing store never empties the list."""
    import autopilot_core as core

    candidates = {"longs": [{"symbol": "AAA"}, {"symbol": "BBB"}], "shorts": []}
    profiles = {
        "AAA": {"last_complete": 103.0, "completed_session_vwap": 101.0},
        "BBB": {"last_complete": 103.0, "completed_session_vwap": None},
    }
    context = {
        "AAA": {"prev_high": 100.5, "prev_low": 98.0},
        "BBB": {"prev_high": 100.5, "prev_low": 98.0},
    }
    kept = core.filter_candidates_by_prev_day_extremes(candidates, profiles, context)
    assert [row["symbol"] for row in kept["longs"]] == ["AAA"]

    for empty in (None, {}):
        untouched = core.filter_candidates_by_prev_day_extremes(candidates, profiles, empty)
        assert [row["symbol"] for row in untouched["longs"]] == ["AAA", "BBB"]


def test_the_refusals_are_logged_with_their_reasons():
    """An evicted or refused pick's disappearance has to be explainable."""
    import autopilot_core as core

    lines: list[str] = []
    core.filter_candidates_by_prev_day_extremes(
        {"longs": [{"symbol": "AAA"}], "shorts": []},
        {"AAA": {"last_complete": 103.0, "completed_session_vwap": 104.0}},
        {"AAA": {"prev_high": 100.5, "prev_low": 98.0}},
        log=lines.append,
    )
    assert any("AAA (not above session VWAP)" in line for line in lines)


def test_session_vwap_rides_on_the_completed_bars_of_this_session():
    """The profile's VWAP must come from the same completed bars as its close,
    and restart on the session boundary rather than blending yesterday."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import autopilot_core as core

    tz = ZoneInfo("America/New_York")
    yesterday = datetime(2026, 7, 1, 9, 30, tzinfo=tz)
    today = datetime(2026, 7, 2, 9, 30, tzinfo=tz)
    rows = []
    # Yesterday traded far away from today's prices; if it leaked into the
    # accumulator the VWAP below would be nowhere near 100.
    for index in range(6):
        rows.append({
            "dt": yesterday + timedelta(minutes=5 * index),
            "open": 50.0, "high": 50.5, "low": 49.5, "close": 50.0, "volume": 1000.0,
        })
    for index in range(6):
        rows.append({
            "dt": today + timedelta(minutes=5 * index),
            "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0,
        })

    metrics = core._intraday_extreme_metrics(rows, now=today + timedelta(minutes=30))
    assert metrics["completed_bar_count"] == 6
    assert metrics["completed_session_vwap"] == pytest.approx(100.0, abs=0.01)


def test_a_profile_without_bars_carries_no_vwap():
    import autopilot_core as core

    metrics = core._intraday_extreme_metrics([], now=None)
    assert metrics["completed_session_vwap"] is None
