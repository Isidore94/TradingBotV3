""""Holding highs" has to be measured, in ATR, and it has to expire.

The case this exists for (trader, 2026-08-21): the regime-pause watch captioned
MRK "M5 regime-pause watch - holding highs" while MRK's high of day was 75
minutes old and price was fading off it. Two rules came out of that:

* near the extreme means near in **ATR**, not percent - "a stock like MRK moves
  slower than say MU, we can't use the 1% rule";
* the claim is good for **15 minutes** unless the name keeps making new
  extremes, and then the row is deleted from the queue.

Shorts mirror every case here. The real MRK and GFS bars from that morning are
pinned at the bottom as regressions.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import regime_pause_hold as rph  # noqa: E402
from indicators.atr import true_ranges, wilder_atr  # noqa: E402

OPEN = datetime(2026, 8, 21, 6, 30)


def _bar(index: int, high: float, low: float, close: float) -> dict:
    return {
        "dt": OPEN + timedelta(minutes=5 * index),
        "open": (high + low) / 2.0,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0,
    }


def _flat_run(count: int, *, high=100.5, low=99.5, close=100.0, start=0) -> list[dict]:
    return [_bar(start + i, high, low, close) for i in range(count)]


def _now_after(bars: list[dict]) -> datetime:
    """A clock at which every bar in the list is completed."""
    return bars[-1]["dt"] + timedelta(minutes=5)


# -- ATR -----------------------------------------------------------------


def test_atr_is_wilder_smoothed_not_a_plain_mean():
    """A constant 1.00 range seeds and stays at 1.00; the shape matters when a
    single wide bar arrives, which is what separates the two formulas."""
    bars = [_bar(i, 100.5, 99.5, 100.0) for i in range(15)]
    assert wilder_atr(bars) == pytest.approx(1.0)
    bars.append(_bar(15, 105.0, 99.0, 100.0))
    # Wilder: (1.0 * 13 + 6.0) / 14. A plain 14-mean would give ~1.357.
    assert wilder_atr(bars) == pytest.approx((1.0 * 13 + 6.0) / 14)


def test_atr_needs_one_more_bar_than_its_length():
    bars = [_bar(i, 100.5, 99.5, 100.0) for i in range(14)]
    assert wilder_atr(bars) is None
    bars.append(_bar(14, 100.5, 99.5, 100.0))
    assert wilder_atr(bars) is not None


def test_a_flat_series_is_unmeasurable_rather_than_zero():
    """A zero ATR would make every distance infinite; None says so honestly."""
    bars = [_bar(i, 100.0, 100.0, 100.0) for i in range(20)]
    assert wilder_atr(bars) is None


def test_an_unreadable_bar_breaks_the_chain_instead_of_bridging_it():
    """Pairing bar i-2's close with bar i's high invents a range no two
    adjacent bars produced."""
    bars = [_bar(0, 100.5, 99.5, 100.0), {"dt": OPEN, "high": None}, _bar(2, 120.0, 119.0, 119.5)]
    assert true_ranges(bars) == []


def test_atr_reads_attribute_style_bars_too():
    """IB bar objects are not dicts, and the detector's series is IB objects."""

    class Bar:
        def __init__(self, high, low, close):
            self.high, self.low, self.close = high, low, close

    bars = [Bar(100.5, 99.5, 100.0) for _ in range(15)]
    assert wilder_atr(bars) == pytest.approx(1.0)


# -- where price sits ----------------------------------------------------


def test_a_new_high_on_the_last_completed_bar_is_holding():
    bars = _flat_run(15) + [_bar(15, 103.0, 101.0, 102.9)]
    state = rph.hold_state(bars, "LONG", now=_now_after(bars))
    assert state.reason == rph.AT_EXTREME
    assert state.holding is True
    assert state.bars_since_extreme == 0
    assert state.describe() == "new HOD"


def test_just_under_the_high_is_still_holding():
    bars = _flat_run(15) + [_bar(15, 103.0, 101.0, 102.9), _bar(16, 102.8, 102.2, 102.5)]
    state = rph.hold_state(bars, "LONG", now=_now_after(bars))
    assert state.holding is True
    assert state.reason == rph.WITHIN_TOLERANCE
    assert state.distance_atr < rph.HOLD_TOLERANCE_ATR


def test_faded_off_the_high_is_not_holding():
    bars = _flat_run(15) + [_bar(15, 103.0, 101.0, 102.9)]
    bars += [_bar(16 + i, 102.0 - i, 101.0 - i, 101.2 - i) for i in range(3)]
    state = rph.hold_state(bars, "LONG", now=_now_after(bars))
    assert state.holding is False
    assert state.reason == rph.TOO_FAR
    assert state.distance_atr > rph.HOLD_TOLERANCE_ATR
    assert "ATR off HOD" in state.describe()


def test_the_tolerance_is_atr_not_percent():
    """The whole point. Two names the same distance in PERCENT off their high,
    one slow and one fast: the slow one is far, the fast one is not."""
    slow = _flat_run(15, high=100.1, low=99.9, close=100.0)
    slow += [_bar(15, 100.1, 99.9, 100.0), _bar(16, 99.5, 99.4, 99.5)]
    fast = _flat_run(15, high=103.0, low=97.0, close=100.0)
    fast += [_bar(15, 100.1, 99.9, 100.0), _bar(16, 99.5, 99.4, 99.5)]
    slow_state = rph.hold_state(slow, "LONG", now=_now_after(slow))
    fast_state = rph.hold_state(fast, "LONG", now=_now_after(fast))
    # Same 0.6% off the high in both cases.
    assert slow_state.holding is False
    assert fast_state.holding is True
    assert slow_state.distance_atr > fast_state.distance_atr


def test_shorts_are_measured_against_the_low():
    bars = _flat_run(15) + [_bar(15, 99.0, 97.0, 97.1)]
    state = rph.hold_state(bars, "SHORT", now=_now_after(bars))
    assert state.reason == rph.AT_EXTREME
    assert state.extreme == pytest.approx(97.0)
    assert state.describe() == "new LOD"


def test_a_bounce_off_the_low_stops_pressing():
    bars = _flat_run(15) + [_bar(15, 99.0, 97.0, 97.1)]
    bars += [_bar(16 + i, 99.0 + i, 98.0 + i, 98.9 + i) for i in range(3)]
    state = rph.hold_state(bars, "SHORT", now=_now_after(bars))
    assert state.holding is False
    assert "ATR off LOD" in state.describe()


def test_only_completed_bars_are_measured():
    """A forming bar is preview (plan.md sec 5). A spike inside the bar that is
    still printing must not create a 'new HOD'."""
    bars = _flat_run(15) + [_bar(15, 103.0, 101.0, 102.9), _bar(16, 130.0, 101.0, 129.0)]
    forming = bars[-1]["dt"] + timedelta(minutes=2)
    state = rph.hold_state(bars, "LONG", now=forming)
    assert state.extreme == pytest.approx(103.0)
    assert state.bars_since_extreme == 0


def test_equalling_the_high_does_not_refresh_the_clock():
    """A name that stops going up is exactly what the freshness rule is for."""
    bars = _flat_run(15) + [_bar(15, 103.0, 101.0, 102.9)]
    bars += [_bar(16 + i, 103.0, 102.0, 102.5) for i in range(4)]
    state = rph.hold_state(bars, "LONG", now=_now_after(bars))
    assert state.bars_since_extreme == 4


def test_no_bars_is_unmeasurable_not_a_verdict():
    state = rph.hold_state([], "LONG", now=OPEN)
    assert state.reason == rph.UNMEASURABLE
    assert state.holding is False
    assert state.describe() == "hold unmeasurable"


# -- the fifteen-minute rule ---------------------------------------------


def test_a_row_survives_its_first_fifteen_minutes():
    """Held on the alert's own clock, with no new high since - which is the
    case the fifteen minutes were granted for."""
    bars = _flat_run(15) + [_bar(15, 103.0, 101.0, 102.5), _bar(16, 102.8, 102.2, 102.6)]
    alert = bars[15]["dt"]
    verdict = rph.queue_verdict(
        bars, "LONG", alert_time=alert, now=alert + timedelta(minutes=14)
    )
    assert verdict.keep is True
    assert verdict.reason == rph.FRESH_ALERT
    assert verdict.hold.bars_since_extreme == 1


def test_a_row_that_stopped_making_highs_is_deleted_after_fifteen_minutes():
    bars = _flat_run(15) + [_bar(15, 103.0, 101.0, 102.5)]
    bars += [_bar(16 + i, 102.0, 101.0, 101.5) for i in range(4)]
    alert = bars[15]["dt"]
    verdict = rph.queue_verdict(
        bars, "LONG", alert_time=alert, now=alert + timedelta(minutes=16)
    )
    assert verdict.keep is False
    assert verdict.reason == rph.EXPIRED_STALE


def test_a_new_high_refreshes_the_clock():
    """The trader's exception, stated in their words: deleted after fifteen
    minutes UNLESS it continues to make a new HOD."""
    bars = _flat_run(15)
    bars += [_bar(15 + i, 103.0 + i, 101.0 + i, 102.9 + i) for i in range(8)]
    alert = bars[15]["dt"]
    now = _now_after(bars)
    assert now - alert > timedelta(minutes=15)
    verdict = rph.queue_verdict(bars, "LONG", alert_time=alert, now=now)
    assert verdict.keep is True
    assert verdict.reason == rph.NEW_EXTREME


def test_uncertainty_never_deletes_a_row():
    """A cache that has not warmed is not a name that stopped making highs."""
    verdict = rph.queue_verdict([], "LONG", alert_time=OPEN, now=OPEN + timedelta(hours=3))
    assert verdict.keep is True
    assert verdict.reason == rph.UNMEASURABLE


def test_a_missing_alert_time_falls_back_to_the_extreme():
    bars = _flat_run(15) + [_bar(15, 103.0, 101.0, 102.5)]
    verdict = rph.queue_verdict(
        bars, "LONG", alert_time=None, now=bars[-1]["dt"] + timedelta(minutes=40)
    )
    assert verdict.keep is False


def test_a_naive_alert_time_against_aware_bars_attaches_rather_than_strips():
    """`_gate_moment`'s lesson, 2026-08-19: stripping an offset ends the crash
    and keeps the outage. A naive stamp must be READ in the bars' zone."""
    zone = timezone(timedelta(hours=-7))
    bars = [
        {
            "dt": (OPEN + timedelta(minutes=5 * i)).replace(tzinfo=zone),
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
        }
        for i in range(15)
    ]
    bars.append(
        {
            "dt": (OPEN + timedelta(minutes=75)).replace(tzinfo=zone),
            "high": 103.0,
            "low": 101.0,
            "close": 102.5,
        }
    )
    naive_alert = OPEN + timedelta(minutes=75)
    fresh = rph.queue_verdict(
        bars,
        "LONG",
        alert_time=naive_alert,
        now=(OPEN + timedelta(minutes=85)).replace(tzinfo=zone),
    )
    assert fresh.keep is True
    stale = rph.queue_verdict(
        bars,
        "LONG",
        alert_time=naive_alert,
        now=(OPEN + timedelta(minutes=120)).replace(tzinfo=zone),
    )
    assert stale.keep is False


# -- the morning that produced the rules ---------------------------------


def _replay(rows: list[tuple[str, float, float, float]]) -> list[dict]:
    return [
        {
            "dt": datetime.strptime(f"2026-08-21 {stamp}", "%Y-%m-%d %H:%M"),
            "open": (high + low) / 2.0,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
        }
        for stamp, high, low, close in rows
    ]


#: MRK, 2026-08-21, thinned to the shape that matters: a spike into 154.49 at
#: 07:15 and a two-hour fade. Flagged "holding highs" at 08:30.
MRK_ROWS = [
    ("06:30", 150.20, 149.30, 149.90),
    ("06:35", 150.60, 149.80, 150.40),
    ("06:40", 151.10, 150.20, 151.00),
    ("06:45", 151.80, 150.90, 151.60),
    ("06:50", 152.40, 151.40, 152.20),
    ("06:55", 153.10, 152.00, 153.00),
    ("07:00", 153.90, 152.80, 153.70),
    ("07:05", 154.20, 153.40, 154.00),
    ("07:10", 154.40, 153.80, 154.30),
    ("07:15", 154.49, 153.90, 154.10),
    ("07:20", 154.30, 153.60, 153.80),
    ("07:25", 154.00, 153.40, 153.60),
    ("07:30", 153.90, 153.30, 153.50),
    ("07:35", 153.70, 153.20, 153.40),
    ("07:40", 153.60, 153.10, 153.30),
    ("07:45", 153.50, 153.00, 153.20),
    ("07:50", 153.40, 152.90, 153.10),
    ("07:55", 153.45, 152.95, 153.30),
    ("08:00", 153.60, 153.10, 153.40),
    ("08:05", 153.70, 153.20, 153.50),
    ("08:10", 153.65, 153.15, 153.45),
    ("08:15", 153.55, 153.05, 153.35),
    ("08:20", 153.50, 153.00, 153.30),
    ("08:25", 153.45, 152.95, 153.36),
]


def test_mrk_was_never_holding_highs_when_it_was_flagged():
    bars = _replay(MRK_ROWS)
    fired = datetime(2026, 8, 21, 8, 30)
    state = rph.hold_state(bars, "LONG", now=fired)
    assert state.holding is False, "the alert that started all of this"
    assert state.reason == rph.TOO_FAR
    assert state.distance_atr > rph.HOLD_TOLERANCE_ATR
    # Its high of day was well over an hour old at that point.
    assert state.bars_since_extreme is not None and state.bars_since_extreme >= 12


def test_mrk_is_deleted_from_the_queue_by_the_time_the_trader_saw_it():
    """09:40 is when the screenshot was taken; the caption was still up."""
    bars = _replay(MRK_ROWS)
    verdict = rph.queue_verdict(
        bars,
        "LONG",
        alert_time=datetime(2026, 8, 21, 8, 30),
        now=datetime(2026, 8, 21, 9, 40),
    )
    assert verdict.keep is False
    assert verdict.reason == rph.EXPIRED_STALE


# -- when the ATR itself cannot be measured ------------------------------


def test_at_the_extreme_needs_no_atr():
    """Nine bars into a session there is no ATR(14) - and the sweep fires
    while there are nine. Being AT the high is a fact that needs no tolerance
    to state, so refusing to state it would switch the rule off exactly when
    it is needed. Caught by the champion tests, which build 12-bar sessions.
    """
    bars = [_bar(i, 100.5, 99.5, 100.0) for i in range(8)]
    bars.append(_bar(8, 103.0, 101.0, 102.9))
    state = rph.hold_state(bars, "LONG", now=_now_after(bars))
    assert wilder_atr(bars) is None
    assert state.holding is True
    assert state.reason == rph.AT_EXTREME
    assert state.distance_atr is None
    assert state.describe() == "new HOD"


def test_off_the_extreme_with_no_atr_stays_unmeasurable():
    """The DISTANCE is what cannot be judged, and inventing a tolerance to
    judge it with is the one thing not to do."""
    bars = [_bar(i, 100.5, 99.5, 100.0) for i in range(8)]
    bars.append(_bar(8, 103.0, 101.0, 102.9))
    bars.append(_bar(9, 102.0, 101.0, 101.2))
    state = rph.hold_state(bars, "LONG", now=_now_after(bars))
    assert wilder_atr(bars) is None
    assert state.holding is False
    assert state.reason == rph.UNMEASURABLE


def test_a_short_at_its_low_needs_no_atr_either():
    bars = [_bar(i, 100.5, 99.5, 100.0) for i in range(8)]
    bars.append(_bar(8, 99.0, 97.0, 97.1))
    state = rph.hold_state(bars, "SHORT", now=_now_after(bars))
    assert state.holding is True
    assert state.describe() == "new LOD"
