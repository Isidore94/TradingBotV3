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
from indicators.rolling_rrs import RollingRrsConfig, rolling_rrs, rrs_series  # noqa: E402

START = datetime(2026, 9, 28, 6, 30)


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


def test_the_worked_example_from_the_post_reads_three():
    # SPY ATR 0.50 falls 2.00 (power -4); stock ATR 0.20 falls only 0.20.
    spy = _bars(_path(370.0, [0.0] * 13 + [-2.0 / 12] * 12), half_range=0.25, wobble=False)
    stock = _bars(_path(100.0, [0.0] * 13 + [-0.2 / 12] * 12), half_range=0.10, wobble=False)
    rrs, power = rrs_series(stock, spy, 12)[-1]
    assert power == pytest.approx(-4.0, rel=0.05)
    assert rrs == pytest.approx(3.0, rel=0.05)


def test_rolling_is_the_mean_of_the_last_reads():
    stock = _bars(_path(50.0, [0.04 * ((i % 5) - 1) for i in range(50)]))
    spy = _bars(_path(600.0, [0.05 * ((i % 4) - 1) for i in range(50)]), half_range=0.3)
    config = RollingRrsConfig(length=12, roll=12)
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
    result = rolling_rrs(stock, spy)
    assert result is not None
    assert result.point < 0  # the jump left the 12-bar window
    assert result.rolling > result.point  # the rolling read still remembers it


def test_a_long_leader_holds_on_dips_and_drives_on_rips():
    # SPY swings down then up; the leader barely dips and runs harder up.
    spy_steps = [0.0] * 14 + [-0.15] * 12 + [0.15] * 12
    stock_steps = [0.0] * 14 + [-0.01] * 12 + [0.12] * 12
    spy = _bars(_path(600.0, spy_steps), half_range=0.3)
    stock = _bars(_path(50.0, stock_steps), half_range=0.10)
    result = rolling_rrs(stock, spy, RollingRrsConfig(roll=24))
    assert result is not None
    assert result.hold_on_dips is not None and result.hold_on_dips > 0
    assert result.drive_on_rips is not None and result.drive_on_rips > 0


def test_weakness_is_the_mirror_of_strength():
    spy = _bars(_path(600.0, [0.1 * ((i % 3) - 1) for i in range(40)]), half_range=0.3)
    up = _bars(_path(50.0, [0.05] * 40))
    down = _bars(_path(50.0, [-0.05] * 40))
    strong = rolling_rrs(up, spy)
    assert strong is not None and strong.rolling > 0
    weak = rolling_rrs(down, spy)
    assert weak is not None and weak.rolling < 0


def test_too_few_bars_is_unknown_not_zero():
    stock = _bars(_path(50.0, [0.05] * 20))  # 21 bars
    spy = _bars(_path(600.0, [0.05] * 20), half_range=0.3)
    # The first point read needs 14 bars, so 21 bars give 8 reads of the 12 wanted.
    assert rolling_rrs(stock, spy) is None
    partial = rolling_rrs(stock, spy, RollingRrsConfig(min_reads=6))
    assert partial is not None and len(partial.reads) == 8


def test_unaligned_series_are_unknown():
    stock = _bars(_path(50.0, [0.05] * 40))
    spy = _bars(_path(600.0, [0.05] * 39), half_range=0.3)
    assert rolling_rrs(stock, spy) is None
    assert all(pair == (None, None) for pair in rrs_series(stock, spy, 12))


def test_a_bad_bar_leaves_later_reads_unknown():
    stock = _bars(_path(50.0, [0.05] * 40))
    spy = _bars(_path(600.0, [0.05] * 40), half_range=0.3)
    stock[30]["high"] = None
    series = rrs_series(stock, spy, 12)
    assert series[29][0] is not None
    assert all(pair == (None, None) for pair in series[31:])
    assert rolling_rrs(stock, spy) is None
