import sys
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prev_day_gate import (  # noqa: E402
    CLOSED,
    OPEN,
    UNKNOWN,
    passes_prev_day_extreme_gate,
    prev_day_break_state,
    prev_session_extremes,
)


def test_break_state_is_directional():
    # Long: above yesterday's high opens, at or below it stays shut.
    assert prev_day_break_state("long", 101.0, 100.0, 95.0) == OPEN
    assert prev_day_break_state("long", 100.0, 100.0, 95.0) == CLOSED
    assert prev_day_break_state("long", 99.0, 100.0, 95.0) == CLOSED
    # Short: below yesterday's low opens; being above the HIGH does not.
    assert prev_day_break_state("short", 94.0, 100.0, 95.0) == OPEN
    assert prev_day_break_state("short", 95.0, 100.0, 95.0) == CLOSED
    assert prev_day_break_state("short", 101.0, 100.0, 95.0) == CLOSED
    # Side spellings the desk actually passes around.
    assert prev_day_break_state("SHORT", 94.0, 100.0, 95.0) == OPEN
    assert prev_day_break_state("shorts", 94.0, 100.0, 95.0) == OPEN
    assert prev_day_break_state("LONG", 101.0, 100.0, 95.0) == OPEN


def test_missing_data_is_unknown_and_never_passes():
    # plan.md sec 5: missing data is uncertainty, never confirmation.
    assert prev_day_break_state("long", None, 100.0, 95.0) == UNKNOWN
    assert prev_day_break_state("long", 101.0, None, 95.0) == UNKNOWN
    assert prev_day_break_state("short", 94.0, 100.0, None) == UNKNOWN
    assert prev_day_break_state("long", float("nan"), 100.0, 95.0) == UNKNOWN
    # A non-finite level is not a level: it reads UNKNOWN, never "cleared it".
    assert prev_day_break_state("long", 101.0, float("inf"), 95.0) == UNKNOWN
    for args in (
        ("long", None, 100.0, 95.0),
        ("long", 101.0, None, 95.0),
        ("short", 94.0, 100.0, None),
    ):
        assert not passes_prev_day_extreme_gate(*args)
    assert passes_prev_day_extreme_gate("long", 101.0, 100.0, 95.0)


def _bar(day, high, low):
    return {"dt": datetime(2026, 8, day, 0, 0), "high": high, "low": low, "close": high}


def test_prev_session_extremes_ignores_today_and_the_future():
    bars = [
        _bar(3, 90.0, 80.0),
        _bar(4, 100.0, 95.0),
        _bar(5, 120.0, 60.0),  # today's forming bar: must not measure itself
    ]
    assert prev_session_extremes(bars, session=date(2026, 8, 5)) == (100.0, 95.0)
    # Reading an earlier session walks back one more bar.
    assert prev_session_extremes(bars, session=date(2026, 8, 4)) == (90.0, 80.0)
    # Nothing before the first session.
    assert prev_session_extremes(bars, session=date(2026, 8, 3)) == (None, None)
    assert prev_session_extremes([], session=date(2026, 8, 5)) == (None, None)
    assert prev_session_extremes(None, session=date(2026, 8, 5)) == (None, None)


def test_prev_session_extremes_survives_junk_rows():
    bars = [
        {"dt": "not a datetime", "high": 999.0, "low": 1.0},
        {"high": 999.0, "low": 1.0},
        _bar(4, 100.0, 95.0),
        {"dt": datetime(2026, 8, 4, 0, 0), "high": "n/a", "low": None},
    ]
    high, low = prev_session_extremes(bars, session=date(2026, 8, 5))
    # The junk row shares the winning date; an unparseable level reads None
    # rather than poisoning the gate with a fabricated number.
    assert (high, low) in {(100.0, 95.0), (None, None)}


def test_autopilot_gate_delegates_to_the_shared_rule():
    import autopilot_core

    ctx = {"prev_high": 100.0, "prev_low": 95.0}
    assert autopilot_core.passes_prev_day_extreme_gate("long", 101.0, ctx)
    assert not autopilot_core.passes_prev_day_extreme_gate("long", 99.0, ctx)
    assert autopilot_core.passes_prev_day_extreme_gate("short", 94.0, ctx)
    assert not autopilot_core.passes_prev_day_extreme_gate("short", 96.0, ctx)
    assert not autopilot_core.passes_prev_day_extreme_gate("long", 101.0, None)
    assert not autopilot_core.passes_prev_day_extreme_gate("long", None, ctx)
    assert not autopilot_core.passes_prev_day_extreme_gate("long", 101.0, {})
