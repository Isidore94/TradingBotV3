"""What the regime-pause sweep selects, frozen (plan.md sec 5).

The near-extreme-in-ATR gate is a change to what a champion detector emits, so
this fixture came first: four long cases and four short ones, each reaching the
sweep through a DIFFERENT branch of

    still_trending or made_new_extreme or window_excess >= 0.20

Two per side are genuinely at their extreme. Two are not - one drifting flat
while SPY falls (window_excess), one bouncing off the day's low
(still_trending) - and before 2026-08-21 both were flagged and captioned
"holding highs". That is the defect the trader photographed on MRK, reproduced
in a form a test can hold.

The gate's effect is therefore a reviewable diff and not an assertion nobody
can check: `test_the_gate_dropped_exactly_the_documented_rows` names the rows
that left and why they were there.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_NAME = "regime_pause_sweep_v1"
OPEN = datetime(2026, 8, 21, 6, 30)
#: Five minutes after the last fixture bar starts: every bar complete, and the
#: replay does not depend on the day it is run.
MEASURED_AT = datetime(2026, 8, 21, 8, 20)


def _to_ib(rows):
    from bounce_bot_lib.legacy import IbBar

    return [
        IbBar(
            dt=OPEN + timedelta(minutes=5 * row["index"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )
        for row in rows
    ]


def _run_side(fixture, side: str) -> dict:
    """Drive the real sweep with the fixture's bars and stubbed recorders.

    Only the bookkeeping is stubbed - observations, the tracker row and the
    per-hit candidate event. The selection logic under test is the shipped
    method, called on a real BounceBot instance.
    """
    from bounce_bot_lib.legacy import BounceBot

    bot = object.__new__(BounceBot)
    cases = fixture["cases"][side]
    symbols = sorted(cases)
    bot.longs = symbols if side == "long" else []
    bot.shorts = symbols if side == "short" else []
    series = {name: _to_ib(rows) for name, rows in cases.items()}
    bot.get_cached_5m_bars = lambda symbol: series.get(symbol, [])
    bot._record_regime_pause_observation = lambda *a, **k: None
    bot._save_regime_pause_observations = lambda: None
    bot._record_regime_pause_banger = lambda hit: None
    emitted: list[str] = []
    bot.gui_callback = lambda message, colour=None: emitted.append(message)

    spy = _to_ib(fixture["spy"][side])
    state = {
        "date": OPEN.date(),
        "side": side,
        "start_dt": spy[fixture["pause_start_index"]].dt,
        "alerted": set(),
        "observed": set(),
    }
    flagged = bot._sweep_regime_pause_bangers(state, spy, side, now=MEASURED_AT)
    return {
        "flagged": sorted(hit["symbol"] for hit in flagged),
        "measures": {
            hit["symbol"]: {
                "sym_day": round(float(hit["sym_day"]), 4),
                "sym_window": round(float(hit["sym_window"]), 4),
                "day_excess": round(float(hit["day_excess"]), 4),
            }
            for hit in flagged
        },
        "summary_lines": emitted,
    }


def _actual(fixture) -> dict:
    return {side: _run_side(fixture, side) for side in ("long", "short")}


def test_regime_pause_sweep_golden_fixture():
    """Loading re-verifies raw_input_sha256 over the fixture's own bars, so
    editing a case without re-freezing the expectations fails here."""
    fixture = load_fixture_contract(FIXTURE_NAME)
    assert fixture.schema == "regime_pause_sweep_v1"
    fixture.assert_matches(_actual(fixture), fixture["expected"], "regime pause sweep")


def test_the_gate_dropped_exactly_the_documented_rows():
    """Name the difference the re-freeze recorded.

    The baseline flagged all four cases per side. Two of them were the reason
    the gate exists, and each arrived through a different branch - so a future
    edit that quietly re-opens either one cannot hide inside a re-frozen
    expectation.
    """
    fixture = load_fixture_contract(FIXTURE_NAME)
    actual = _actual(fixture)
    assert actual["long"]["flagged"] == ["HOLDS_AT_HIGH", "HOLDS_JUST_UNDER"]
    assert actual["short"]["flagged"] == ["PRESSES_AT_LOW", "PRESSES_JUST_OVER"]
    for side in ("long", "short"):
        assert "FELL_LESS_THAN_SPY" not in actual[side]["flagged"]
    assert "BOUNCING_OFF_LOW" not in actual["long"]["flagged"]
    assert "BOUNCING_OFF_HIGH" not in actual["short"]["flagged"]


def test_the_dropped_rows_still_satisfy_the_old_predicate():
    """The gate ADDS a condition; it does not repeal the old ones.

    If this ever fails, the two cases stopped being evidence about the gate and
    started passing for some unrelated reason - the fixture would still be
    green while proving nothing.
    """
    from bounce_bot_lib.legacy import (
        REGIME_BANGER_WINDOW_EXCESS_PCT,
        BounceBot,
    )

    fixture = load_fixture_contract(FIXTURE_NAME)
    spy = _to_ib(fixture["spy"]["long"])
    start = spy[fixture["pause_start_index"]].dt
    spy_window, _ = BounceBot._window_change_pct(spy, start)

    for name, branch in (
        ("FELL_LESS_THAN_SPY", "window_excess"),
        ("BOUNCING_OFF_LOW", "still_trending"),
    ):
        bars = _to_ib(fixture["cases"]["long"][name])
        sym_window, _ = BounceBot._window_change_pct(bars, start)
        still_trending = sym_window > 0
        window_excess = sym_window - spy_window
        if branch == "still_trending":
            assert still_trending, f"{name} no longer trends through the window"
        else:
            assert not still_trending
            assert window_excess >= REGIME_BANGER_WINDOW_EXCESS_PCT, (
                f"{name} no longer clears the window-excess branch"
            )


def test_the_kept_rows_are_the_ones_actually_at_their_extreme():
    """Cross-check the survivors against the shared measurement rather than
    trusting the detector to agree with itself."""
    import regime_pause_hold as rph

    fixture = load_fixture_contract(FIXTURE_NAME)
    for side, keep, drop in (
        ("long", "HOLDS_AT_HIGH", "FELL_LESS_THAN_SPY"),
        ("short", "PRESSES_AT_LOW", "FELL_LESS_THAN_SPY"),
    ):
        label = "LONG" if side == "long" else "SHORT"
        for name, expected in ((keep, True), (drop, False)):
            bars = _to_ib(fixture["cases"][side][name])
            now = MEASURED_AT
            state = rph.hold_state(bars, label, now=now)
            assert state.holding is expected, f"{side}/{name}: {state.describe()}"
