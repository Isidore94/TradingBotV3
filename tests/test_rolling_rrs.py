"""Rolling Real Relative Strength (scripts/indicators/rolling_rrs.py)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from group_rrs import real_relative_strength  # noqa: E402
from indicators.rolling_rrs import (  # noqa: E402
    RollingRrsConfig,
    hourly_rrs_series,
    rolling_rrs,
    rrs_series,
)

START = datetime(2026, 9, 28, 6, 30)
SESSION_BARS = 78  # 06:30-13:00 PT in 5-minute bars
BAR = RollingRrsConfig(atr_mode="bar")  # the desk's existing RRS scale


def _bars(closes, *, half_range=0.10, wobble=True):
    """Bar dicts; with wobble off and small steps, every true range is 2*half_range."""
    out = []
    for index, close in enumerate(closes):
        width = half_range * (1.0 + (index % 3) * 0.25) if wobble else half_range
        out.append(
            {
                "dt": START + timedelta(minutes=5 * index),
                "open": close,
                "high": close + width,
                "low": close - width,
                "close": close,
            }
        )
    return out


def _sessions(history_sessions, today_closes, *, half_range=0.10, wobble=True):
    """Flat prior sessions at the first close, then today; one calendar day per session."""
    flat = [today_closes[0]] * SESSION_BARS
    out = []
    for day, closes in enumerate([flat] * history_sessions + [today_closes]):
        bars = _bars(closes, half_range=half_range, wobble=wobble)
        for bar in bars:
            bar["dt"] += timedelta(days=day - history_sessions)
        out.extend(bars)
    return out


def _path(start, steps):
    closes = [start]
    for step in steps:
        closes.append(closes[-1] + step)
    return closes


@pytest.mark.parametrize("length", [6, 12, 18])
def test_every_point_read_equals_the_desk_rrs(length):
    stock = _bars(_path(50.0, [0.03 * ((i % 7) - 2) for i in range(60)]))
    spy = _bars(_path(600.0, [0.05 * ((i % 5) - 2) for i in range(60)]), half_range=0.3)
    series = rrs_series(stock, spy, length)
    for end in range(len(stock)):
        want = real_relative_strength(stock[: end + 1], spy[: end + 1], length)
        got = series[end]
        if want[0] is None:
            assert got == (None, None)
        else:
            assert got[0] == pytest.approx(want[0], abs=1e-9)
            assert got[1] == pytest.approx(want[1], abs=1e-9)


def test_desk_mode_reads_the_worked_example_on_its_own_scale():
    # SPY ATR 0.50 falls 2.00 (power -4); stock ATR 0.20 falls only 0.20.
    spy = _bars(_path(370.0, [0.0] * 13 + [-2.0 / 12] * 12), half_range=0.25, wobble=False)
    stock = _bars(_path(100.0, [0.0] * 13 + [-0.2 / 12] * 12), half_range=0.10, wobble=False)
    rrs, power = rrs_series(stock, spy, 12)[-1]
    assert power == pytest.approx(-4.0, rel=0.05)
    assert rrs == pytest.approx(3.0, rel=0.05)


def test_rolling_is_the_mean_of_the_last_reads():
    stock = _bars(_path(50.0, [0.04 * ((i % 5) - 1) for i in range(50)]))
    spy = _bars(_path(600.0, [0.05 * ((i % 4) - 1) for i in range(50)]), half_range=0.3)
    config = RollingRrsConfig(length=12, roll=12, atr_mode="bar")
    result = rolling_rrs(stock, spy, config)
    tail = [rrs for rrs, _ in rrs_series(stock, spy, 12)[-12:]]
    assert result is not None
    assert result.reads == pytest.approx(tuple(tail))
    assert result.rolling == pytest.approx(sum(tail) / 12)
    assert result.point == pytest.approx(tail[-1])


def test_one_burst_then_flat_fades_slower_on_the_rolling_read():
    # SPY grinds up; the stock jumps once, then sits flat.
    spy = _bars(_path(370.0, [0.08] * 45), half_range=0.25)
    stock = _bars(_path(100.0, [0.0] * 25 + [1.0] + [0.0] * 19))
    result = rolling_rrs(stock, spy, BAR)
    assert result is not None
    assert result.point < 0  # the jump left the 12-bar window
    assert result.rolling > result.point  # the rolling read still remembers it


def test_a_long_leader_holds_on_dips_and_drives_on_rips():
    # SPY swings down then up; the leader barely dips and runs harder up.
    spy_steps = [0.0] * 14 + [-0.15] * 12 + [0.15] * 12
    stock_steps = [0.0] * 14 + [-0.01] * 12 + [0.12] * 12
    spy = _bars(_path(600.0, spy_steps), half_range=0.3)
    stock = _bars(_path(50.0, stock_steps), half_range=0.10)
    result = rolling_rrs(stock, spy, RollingRrsConfig(roll=24, atr_mode="bar"))
    assert result is not None
    assert result.hold_on_dips is not None and result.hold_on_dips > 0
    assert result.drive_on_rips is not None and result.drive_on_rips > 0


def test_weakness_is_the_mirror_of_strength():
    spy = _bars(_path(600.0, [0.1 * ((i % 3) - 1) for i in range(40)]), half_range=0.3)
    up = _bars(_path(50.0, [0.05] * 40))
    down = _bars(_path(50.0, [-0.05] * 40))
    strong = rolling_rrs(up, spy, BAR)
    assert strong is not None and strong.rolling > 0
    weak = rolling_rrs(down, spy, BAR)
    assert weak is not None and weak.rolling < 0


def test_too_few_bars_is_unknown_not_zero():
    stock = _bars(_path(50.0, [0.05] * 20))  # 21 bars
    spy = _bars(_path(600.0, [0.05] * 20), half_range=0.3)
    # The first point read needs 14 bars, so 21 bars give 8 reads of the 12 wanted.
    assert rolling_rrs(stock, spy, BAR) is None
    partial = rolling_rrs(stock, spy, RollingRrsConfig(min_reads=6, atr_mode="bar"))
    assert partial is not None and len(partial.reads) == 8


def test_unaligned_series_are_unknown():
    stock = _bars(_path(50.0, [0.05] * 40))
    spy = _bars(_path(600.0, [0.05] * 39), half_range=0.3)
    assert rolling_rrs(stock, spy) is None
    assert rolling_rrs(stock, spy, BAR) is None
    assert all(pair == (None, None) for pair in rrs_series(stock, spy, 12))


def test_a_bad_bar_leaves_later_reads_unknown():
    stock = _bars(_path(50.0, [0.05] * 40))
    spy = _bars(_path(600.0, [0.05] * 40), half_range=0.3)
    stock[30]["high"] = None
    series = rrs_series(stock, spy, 12)
    assert series[29][0] is not None
    assert all(pair == (None, None) for pair in series[31:])
    assert rolling_rrs(stock, spy, BAR) is None


# ------------------------------------------------------------- H.S. hourly ATR


def test_hourly_worked_example_reads_exactly_three():
    # Hourly ATR: SPY 0.50, stock 0.20. SPY falls 2.00 in the hour (power -4);
    # the stock falls only 0.20, so it beat the expected -0.80 by 3 ATRs.
    spy = _sessions(4, _path(370.0, [0.0] * 11 + [-2.0 / 12] * 12), half_range=0.25, wobble=False)
    stock = _sessions(4, _path(100.0, [0.0] * 11 + [-0.2 / 12] * 12), half_range=0.10, wobble=False)
    rrs, power = hourly_rrs_series(stock, spy)[-1]
    assert power == pytest.approx(-4.0)
    assert rrs == pytest.approx(3.0)


def test_hourly_is_the_default_mode():
    spy = _sessions(4, _path(370.0, [0.0] * 11 + [-2.0 / 12] * 12), half_range=0.25, wobble=False)
    stock = _sessions(4, _path(100.0, [0.0] * 11 + [-0.2 / 12] * 12), half_range=0.10, wobble=False)
    result = rolling_rrs(stock, spy, RollingRrsConfig(min_reads=1))
    assert result is not None and result.atr_mode == "hourly"
    assert result.point == pytest.approx(3.0)


def test_a_move_never_spans_the_overnight_gap():
    spy = _sessions(4, _path(370.0, [0.05] * 30), half_range=0.25)
    stock = _sessions(4, _path(100.0, [0.05] * 30))
    series = hourly_rrs_series(stock, spy)
    today = len(series) - 31
    # Today's first 12 bars cannot read: their hour would reach into yesterday.
    assert all(pair == (None, None) for pair in series[today : today + 12])
    assert series[today + 12][0] is not None


def test_too_few_finished_hours_is_unknown():
    spy = _sessions(2, _path(370.0, [0.05] * 30), half_range=0.25)  # 14 hour blocks
    stock = _sessions(2, _path(100.0, [0.05] * 30))
    assert all(pair == (None, None) for pair in hourly_rrs_series(stock, spy))
    loose = hourly_rrs_series(stock, spy, min_atr_hours=10)
    assert loose[-1][0] is not None


def test_hourly_leader_holds_on_dips_and_drives_on_rips():
    spy = _sessions(4, _path(600.0, [0.0] * 12 + [-0.15] * 12 + [0.15] * 12), half_range=0.3)
    stock = _sessions(4, _path(50.0, [0.0] * 12 + [-0.01] * 12 + [0.12] * 12))
    result = rolling_rrs(stock, spy, RollingRrsConfig(roll=24))
    assert result is not None
    assert result.hold_on_dips is not None and result.hold_on_dips > 0
    assert result.drive_on_rips is not None and result.drive_on_rips > 0


def test_hourly_burst_then_flat_fades_slower_on_the_rolling_read():
    spy = _sessions(4, _path(370.0, [0.08] * 45), half_range=0.25)
    stock = _sessions(4, _path(100.0, [0.0] * 25 + [1.0] + [0.0] * 19))
    result = rolling_rrs(stock, spy)
    assert result is not None
    assert result.point < 0
    assert result.rolling > result.point


def test_unknown_atr_mode_is_refused():
    stock = _bars(_path(50.0, [0.05] * 40))
    with pytest.raises(ValueError):
        rolling_rrs(stock, stock, RollingRrsConfig(atr_mode="daily"))
