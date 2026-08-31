"""What the regime-pause sweep selects, frozen (plan.md sec 5).

Two detector changes landed against this fixture, each frozen before and
re-frozen after, so both are a reviewable diff rather than an assertion nobody
can check:

1. **near the extreme, in ATR** - a flagged name must be within 1.0 ATR of its
   session extreme on the last completed bar (trader, 2026-08-21);
2. **beyond yesterday's extreme and the right side of session VWAP** - the M5
   Focus adoption gate, applied here too (trader, same day).

Six cases per side, each isolating ONE reason to be flagged or dropped, and all
six clear the day-excess gate and the defiance test first:

| case | why it is here |
|---|---|
| HOLDS_AT_HIGH | at its high; survives everything |
| HOLDS_JUST_UNDER | 0.7 ATR under; survives everything |
| FELL_LESS_THAN_SPY | flat while SPY fell - passes defiance, 7 ATR off |
| BOUNCING_OFF_LOW / _HIGH | still trending - passes defiance, 6 ATR off |
| INSIDE_PREV_RANGE | at its high, but never left yesterday's range |
| BELOW_VWAP / ABOVE_VWAP | 0.97 ATR off, but the wrong side of VWAP |

Each case is TWO sessions - a prior day that sets the previous high and low,
then today - because the second gate cannot be measured without one.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path


from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_NAME = "regime_pause_sweep_v1"
PRIOR_OPEN = datetime(2026, 8, 20, 6, 30)
OPEN = datetime(2026, 8, 21, 6, 30)
#: Five minutes after the last fixture bar starts: every bar complete, and the
#: replay does not depend on the day it is run.
MEASURED_AT = datetime(2026, 8, 21, 8, 20)


def _to_ib(rows):
    """Fixture rows to IB bars. ``day`` 0 is the prior session, 1 is today."""
    from bounce_bot_lib.legacy import IbBar

    base = {0: PRIOR_OPEN, 1: OPEN}
    return [
        IbBar(
            dt=base[int(row.get("day", 1))] + timedelta(minutes=5 * row["index"]),
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


def test_the_gates_dropped_exactly_the_documented_rows():
    """Name the difference each re-freeze recorded.

    Four of the six cases per side are dropped, and each for its own reason.
    A future edit that quietly re-opens any one of them cannot hide inside a
    re-frozen expectation.
    """
    fixture = load_fixture_contract(FIXTURE_NAME)
    actual = _actual(fixture)
    assert actual["long"]["flagged"] == ["HOLDS_AT_HIGH", "HOLDS_JUST_UNDER"]
    assert actual["short"]["flagged"] == ["PRESSES_AT_LOW", "PRESSES_JUST_OVER"]
    for side in ("long", "short"):
        dropped = set(fixture["cases"][side]) - set(actual[side]["flagged"])
        assert "FELL_LESS_THAN_SPY" in dropped
        assert "INSIDE_PREV_RANGE" in dropped
    assert "BOUNCING_OFF_LOW" not in actual["long"]["flagged"]
    assert "BELOW_VWAP" not in actual["long"]["flagged"]
    assert "BOUNCING_OFF_HIGH" not in actual["short"]["flagged"]
    assert "ABOVE_VWAP" not in actual["short"]["flagged"]


def test_every_case_still_passes_the_defiance_test_it_was_built_for():
    """The gates ADD conditions; they do not repeal the old ones.

    If a case ever stops clearing `still_trending or made_new_extreme or
    window_excess`, it has stopped being evidence about the new gates and the
    fixture would stay green while proving nothing.
    """
    from bounce_bot_lib.legacy import (
        REGIME_BANGER_WINDOW_EXCESS_PCT,
        BounceBot,
    )

    fixture = load_fixture_contract(FIXTURE_NAME)
    for side in ("long", "short"):
        sign = -1.0 if side == "short" else 1.0
        spy = _to_ib(fixture["spy"][side])
        start = spy[fixture["pause_start_index"]].dt
        spy_window, _ = BounceBot._window_change_pct(spy, start)
        for name, rows in fixture["cases"][side].items():
            bars = [bar for bar in _to_ib(rows) if bar.dt.date() == OPEN.date()]
            sym_window, window_bars = BounceBot._window_change_pct(bars, start)
            pre = [bar for bar in bars if bar.dt < start]
            if side == "short":
                new_extreme = bool(pre) and min(b.low for b in window_bars) < min(
                    b.low for b in pre
                )
            else:
                new_extreme = bool(pre) and max(b.high for b in window_bars) > max(
                    b.high for b in pre
                )
            defiant = (
                sign * sym_window > 0
                or new_extreme
                or sign * (sym_window - spy_window) >= REGIME_BANGER_WINDOW_EXCESS_PCT
            )
            assert defiant, f"{side}/{name} no longer defies the pause"


def test_each_dropped_case_fails_for_its_own_documented_reason():
    """Cross-check every drop against the shared measurements, so the fixture
    records WHICH gate did it rather than only that something did."""
    import regime_pause_hold as rph
    from focus_adoption_gate import CLOSED, OPEN as GATE_OPEN, focus_adoption_gate_state

    fixture = load_fixture_contract(FIXTURE_NAME)
    expectations = {
        "long": {
            "HOLDS_AT_HIGH": ("hold", GATE_OPEN),
            "HOLDS_JUST_UNDER": ("hold", GATE_OPEN),
            "FELL_LESS_THAN_SPY": ("too_far", None),
            "BOUNCING_OFF_LOW": ("too_far", None),
            "INSIDE_PREV_RANGE": ("hold", CLOSED),
            "BELOW_VWAP": ("hold", CLOSED),
        },
        "short": {
            "PRESSES_AT_LOW": ("hold", GATE_OPEN),
            "PRESSES_JUST_OVER": ("hold", GATE_OPEN),
            "FELL_LESS_THAN_SPY": ("too_far", None),
            "BOUNCING_OFF_HIGH": ("too_far", None),
            "INSIDE_PREV_RANGE": ("hold", CLOSED),
            "ABOVE_VWAP": ("hold", CLOSED),
        },
    }
    for side, cases in expectations.items():
        label = "LONG" if side == "long" else "SHORT"
        for name, (hold_expectation, gate_expectation) in cases.items():
            bars = _to_ib(fixture["cases"][side][name])
            hold = rph.hold_state(bars, label, now=MEASURED_AT)
            assert hold.holding is (hold_expectation == "hold"), (
                f"{side}/{name}: {hold.describe()}"
            )
            if gate_expectation is None:
                continue
            levels = rph.session_levels(bars, now=MEASURED_AT)
            gate, reason = focus_adoption_gate_state(
                label, levels.price, levels.prev_high, levels.prev_low, levels.vwap
            )
            assert gate == gate_expectation, f"{side}/{name}: {gate} [{reason}]"


def test_the_two_new_drops_fail_on_different_halves_of_the_gate():
    """INSIDE_PREV_RANGE fails the prior-extreme half and BELOW_VWAP the VWAP
    half - so a change that silently removed either half would show up."""
    import regime_pause_hold as rph
    from focus_adoption_gate import focus_adoption_gate_state

    fixture = load_fixture_contract(FIXTURE_NAME)
    levels = rph.session_levels(
        _to_ib(fixture["cases"]["long"]["INSIDE_PREV_RANGE"]), now=MEASURED_AT
    )
    _state, reason = focus_adoption_gate_state(
        "LONG", levels.price, levels.prev_high, levels.prev_low, levels.vwap
    )
    assert "yesterday's high" in reason

    levels = rph.session_levels(
        _to_ib(fixture["cases"]["long"]["BELOW_VWAP"]), now=MEASURED_AT
    )
    _state, reason = focus_adoption_gate_state(
        "LONG", levels.price, levels.prev_high, levels.prev_low, levels.vwap
    )
    assert "session VWAP" in reason
