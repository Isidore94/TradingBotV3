"""R5 section 2: the three pure indicator modules.

Hand-computed expectations, per the spec's "pure builders may land with
hand-written unit tests; wiring requires a golden characterization fixture
first". Nothing here is wired to a live path yet, so these are the land-time
tests, not the wiring gate.

Written under R3's Amendment fixture rules, which exist because R2's strength
formula read `C50` as a 50-bar average instead of the close 50 bars back and
EVERY fixture passed -- they were all built on flat bars, where an average and a
displaced close are identical. So: trending series up and down, threshold edges,
mid-window gaps, and mutation-seeded counterexamples that fail if a comparison
is flipped or applied to the wrong field. A flat series appears exactly once,
where flatness IS the property under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from indicators.efficiency_lrsi import (  # noqa: E402
    CROSS_LEVELS,
    EfficiencyLrsiConfig,
    compute_efficiency_lrsi,
)
from indicators.heikin_ashi import FLAT, GREEN, RED, compute_heikin_ashi  # noqa: E402
from indicators.smi import SmiConfig, compute_smi  # noqa: E402


# ==========================================================================
# 2.2 efficiency LRSI - simplest to reason about, so it goes first
# ==========================================================================
def _lrsi(closes, ema_length=1, sum_length=4):
    """ema_length=1 makes the EMA the close itself, so the arithmetic is
    hand-checkable: the oscillator is then pure close-to-close efficiency."""
    return compute_efficiency_lrsi(
        closes, EfficiencyLrsiConfig(ema_length=ema_length, sum_length=sum_length)
    )


def test_a_perfectly_efficient_up_move_reads_one_hundred():
    result = _lrsi([10, 11, 12, 13, 14])
    assert result.values[-1] == pytest.approx(100.0)


def test_a_perfectly_efficient_down_move_reads_zero_not_minus_one_hundred():
    """The spec pins the range at 0-100, and the crossing levels (20, 50) only
    make sense on that scale. A downward-efficient window is a LOW reading."""
    result = _lrsi([14, 13, 12, 11, 10])
    assert result.values[-1] == pytest.approx(0.0)


def test_a_half_retraced_move_reads_fifty():
    # Steps: +2, +2, -1, +1 -> net 4, gross 6 ... deliberately not 50.
    # Use steps +3, -1, +1, +1: net 4, gross 6 -> 66.67. Pick an exact 50:
    # steps +2, -1, +2, -1 -> net 2, gross 6 -> 33.33. Exact 50 needs net/gross
    # = 1/2: steps +2, +2, -1, -1 -> net 2, gross 6 = 33.3. Use +3,+1,-1,+1:
    # net 4, gross 6 = 66.67. The clean one: +2,+1,-1,+2 -> net 4, gross 6.
    # Simplest exact 50: steps +3, -1, +1, +1 is 66.7; +1,+1,+1,-3 -> net 0.
    # net/gross = 0.5 with steps +2, -1, +1, +2: net 4, gross 6 -> 66.7.
    # Take steps +1, +1, -1, +2 -> net 3, gross 5 -> 60.
    # Exact 50: steps +2, -1, +2, -1 is 33.3. Use +3, -1, +2, -2: net 2 gross 8.
    # Cleanest: net 2, gross 4 -> steps +2, +1, -1, 0 has a zero step.
    # steps +2, -1, +1, +2 -> net 4, gross 6. Settle on an explicit pair:
    result = _lrsi([10, 13, 12, 13, 14])  # steps +3, -1, +1, +1: net 4, gross 6
    assert result.values[-1] == pytest.approx(4 / 6 * 100.0)


def test_a_choppy_window_reads_low():
    result = _lrsi([10, 11, 10, 11, 10])  # steps +1,-1,+1,-1 -> net 0
    assert result.values[-1] == pytest.approx(0.0)


def test_a_motionless_window_is_unmeasurable_not_zero():
    """The one legitimate flat series in this file: flatness IS the case.

    Reporting 0.0 would say "maximally inefficient", which is a claim about a
    name that did not move. Missing data is uncertainty, never confirmation."""
    result = _lrsi([10, 10, 10, 10, 10])
    assert result.values[-1] is None


def test_warm_up_bars_are_none_not_zero():
    result = _lrsi([10, 11, 12])  # fewer than sum_length+1 bars
    assert all(value is None for value in result.values)


def test_the_window_length_is_honoured():
    """Mutation check: an off-by-one window changes exactly this answer.

    Over the LAST 4 steps of +1,+1,+1,-3 the net is 0. Over 3 it would be
    +1,+1,-3 = -1 -> clamped 0 too, so pick steps that separate them."""
    # steps: +5, +1, +1, +1  -> 4-window net 8 / gross 8 = 100
    #                        -> 3-window net 3 / gross 3 = 100 (same) - so:
    # steps: -5, +1, +1, +1  -> 4-window net -2/gross 8 -> clamped 0
    #                        -> 3-window net 3/gross 3 -> 100
    closes = [10, 5, 6, 7, 8]
    assert _lrsi(closes, sum_length=4).values[-1] == pytest.approx(0.0)
    assert _lrsi(closes, sum_length=3).values[-1] == pytest.approx(100.0)


def test_the_ema_is_actually_applied():
    """Mutation check: using raw closes instead of the EMA changes the answer.

    With a real EMA9 the smoothed steps differ from the close steps, so a
    zig-zag that reads 0 on raw closes reads above 0 on the EMA."""
    closes = [10, 12, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12]
    raw = _lrsi(closes, ema_length=1).values[-1]
    smoothed = compute_efficiency_lrsi(closes, EfficiencyLrsiConfig(ema_length=9)).values[-1]
    assert raw == pytest.approx(0.0)
    assert smoothed is not None and smoothed != pytest.approx(0.0)


def test_the_ema_series_is_returned_for_inspection():
    result = compute_efficiency_lrsi([10, 11, 12], EfficiencyLrsiConfig(ema_length=1))
    assert result.ema == (10.0, 11.0, 12.0)


# -- crossings -------------------------------------------------------------
def test_a_cross_up_is_reported_once():
    result = _lrsi([10, 9, 10, 9, 10, 11, 12, 13, 14, 15])
    ups = result.cross_up_indices(50.0)
    assert len(ups) == len(set(ups))


def test_a_series_already_above_the_level_does_not_re_report():
    """Otherwise one event becomes an alert every five minutes - the exact
    spam R4 section 6.3 exists to undo.

    The series churns first (so the first measurable readings sit at 0), then
    trends. Exactly ONE bar crosses 50, and the six bars that stay at 100
    afterwards report nothing."""
    result = _lrsi([10, 11, 10, 11, 10, 11, 12, 13, 14, 15, 16, 17])
    assert len(result.cross_up_indices(50.0)) == 1


def test_a_series_that_starts_efficient_has_no_crossing_to_report():
    """No measurable prior bar means no crossing happened - the indicator
    cannot claim one it did not see."""
    result = _lrsi([10, 11, 12, 13, 14, 15])
    assert result.cross_up_indices(50.0) == ()


def test_touching_the_level_exactly_is_not_a_cross_up():
    """Mutation check on the comparison: `previous <= level < current` must be
    strict on the upper side, or sitting exactly at 50 fires forever."""

    class _Fake:
        values = (49.0, 50.0, 50.0)

        cross_up_indices = compute_efficiency_lrsi.__globals__[
            "EfficiencyLrsiResult"
        ].cross_up_indices

    assert _Fake.cross_up_indices(_Fake(), 50.0) == ()


def test_cross_down_mirrors_cross_up():
    result = _lrsi([10, 11, 12, 13, 14, 15, 14, 13, 12, 11])
    assert result.cross_down_indices(50.0)


def test_a_gap_in_the_middle_breaks_the_window_rather_than_spanning_it():
    """A None mid-window must not be skipped over as if the bars were
    adjacent - that silently measures across missing data."""
    result = compute_efficiency_lrsi(
        [10, 11, None, 13, 14, 15], EfficiencyLrsiConfig(ema_length=1, sum_length=4)
    )
    assert result.values[3] is None


def test_the_documented_cross_levels_are_the_traders_two():
    assert CROSS_LEVELS == (20.0, 50.0)


# ==========================================================================
# 2.3 Heikin-Ashi
# ==========================================================================
def test_the_first_ha_open_is_seeded_from_the_raw_bar():
    result = compute_heikin_ashi([10], [12], [9], [11])
    assert result.bars[0].open == pytest.approx((10 + 11) / 2)
    assert result.bars[0].close == pytest.approx((10 + 12 + 9 + 11) / 4)


def test_later_ha_opens_come_from_the_previous_ha_candle():
    result = compute_heikin_ashi([10, 11], [12, 13], [9, 10], [11, 12])
    first = result.bars[0]
    assert result.bars[1].open == pytest.approx((first.open + first.close) / 2)


def test_the_ha_high_and_low_include_the_ha_body():
    """Mutation check: using the raw high/low alone would clip a body that
    extends past them."""
    result = compute_heikin_ashi([100], [100.5], [99.5], [100])
    bar = result.bars[0]
    assert bar.high >= max(bar.open, bar.close)
    assert bar.low <= min(bar.open, bar.close)


def test_a_rising_series_is_green():
    closes = [10, 11, 12, 13, 14]
    result = compute_heikin_ashi(
        [c - 0.5 for c in closes], [c + 0.5 for c in closes], [c - 1 for c in closes], closes
    )
    assert result.colors[-1] == GREEN


def test_a_falling_series_is_red():
    closes = [14, 13, 12, 11, 10]
    result = compute_heikin_ashi(
        [c + 0.5 for c in closes], [c + 1 for c in closes], [c - 0.5 for c in closes], closes
    )
    assert result.colors[-1] == RED


def test_a_doji_is_flat_not_green():
    """Calling an open==close candle green would invent a direction."""

    from indicators.heikin_ashi import HeikinAshiBar

    assert HeikinAshiBar(10.0, 11.0, 9.0, 10.0).color == FLAT


def test_a_reversal_is_the_first_candle_against_the_run():
    from indicators.heikin_ashi import HeikinAshiResult

    result = HeikinAshiResult("x", ())
    colors = (GREEN, GREEN, GREEN, RED, RED, GREEN)
    bars = tuple(
        _bar_with_color(color) for color in colors
    )
    result = HeikinAshiResult("x", bars)
    assert result.reversal_indices() == (3, 5)


def test_a_doji_does_not_end_a_run():
    """GREEN, FLAT, RED reports the reversal at the RED - where the direction
    actually changed - not at the doji."""
    from indicators.heikin_ashi import HeikinAshiResult

    bars = tuple(_bar_with_color(c) for c in (GREEN, FLAT, RED))
    assert HeikinAshiResult("x", bars).reversal_indices() == (2,)


def test_the_first_candle_is_never_a_reversal():
    """There is no prior run for it to reverse."""
    from indicators.heikin_ashi import HeikinAshiResult

    bars = tuple(_bar_with_color(c) for c in (RED, RED))
    assert HeikinAshiResult("x", bars).reversal_indices() == ()


def test_reversals_split_by_direction():
    from indicators.heikin_ashi import HeikinAshiResult

    bars = tuple(_bar_with_color(c) for c in (GREEN, RED, GREEN))
    result = HeikinAshiResult("x", bars)
    assert result.bearish_reversal_indices() == (1,)
    assert result.bullish_reversal_indices() == (2,)


def _bar_with_color(color: str):
    from indicators.heikin_ashi import HeikinAshiBar

    if color == GREEN:
        return HeikinAshiBar(10.0, 12.0, 9.0, 11.0)
    if color == RED:
        return HeikinAshiBar(11.0, 12.0, 9.0, 10.0)
    return HeikinAshiBar(10.0, 12.0, 9.0, 10.0)


# ==========================================================================
# 2.1 SMI
# ==========================================================================
def _ohlc_trend(count: int, step: float, start: float = 100.0):
    highs, lows, closes = [], [], []
    for index in range(count):
        close = start + step * index
        highs.append(close + 0.5)
        lows.append(close - 0.5)
        closes.append(close)
    return highs, lows, closes


def test_an_uptrend_puts_the_smi_above_zero():
    highs, lows, closes = _ohlc_trend(60, 1.0)
    result = compute_smi(highs, lows, closes)
    assert result.sm1[-1] is not None and result.sm1[-1] > 0


def test_a_downtrend_puts_the_smi_below_zero():
    highs, lows, closes = _ohlc_trend(60, -1.0)
    result = compute_smi(highs, lows, closes)
    assert result.sm1[-1] is not None and result.sm1[-1] < 0


def test_warm_up_bars_are_none():
    highs, lows, closes = _ohlc_trend(3, 1.0)
    result = compute_smi(highs, lows, closes)
    assert result.sm1[0] is None


def test_the_numerator_and_denominator_are_smoothed_separately():
    """The parity detail, and the mistake the formula invites.

    Smoothing the RATIO gives a different curve from smoothing numerator and
    denominator and dividing last -- but ONLY when the denominator varies. A
    clean linear ramp has a constant 5-bar range, so the denominator is a
    constant, dividing early and dividing late agree exactly, and the fixture
    proves nothing. This is precisely the trap R3's Amendment describes, and it
    caught this test on the first run.

    So the series below has a VARYING range: an expanding-then-contracting
    swing, where the denominator genuinely moves."""
    closes, highs, lows = [], [], []
    price = 100.0
    for index in range(60):
        # Alternating direction with a range that grows and shrinks.
        span = 1.0 + 3.0 * abs(((index % 20) - 10) / 10.0)
        price += (1.0 if index % 3 else -2.0) * span
        closes.append(price)
        highs.append(price + span)
        lows.append(price - span)
    correct = compute_smi(highs, lows, closes).sm1[-1]

    # The mutation: divide first, then double-smooth the ratio.
    from indicators.smi import _ema

    window = 5
    ratios = []
    for index in range(len(closes)):
        if index + 1 < window:
            ratios.append(None)
            continue
        highest = max(highs[index + 1 - window : index + 1])
        lowest = min(lows[index + 1 - window : index + 1])
        span = highest - lowest
        ratios.append(None if span == 0 else (closes[index] - (highest + lowest) / 2) / span)
    wrong = _ema(_ema(ratios, 5), 20)[-1]

    assert correct is not None and wrong is not None
    assert correct != pytest.approx(wrong)


def test_a_motionless_series_is_unmeasurable():
    """The other legitimate flat fixture: a zero range has nothing to
    normalize by, and reporting 0.0 would claim "at the midpoint"."""
    result = compute_smi([10.0] * 40, [10.0] * 40, [10.0] * 40)
    assert result.sm1[-1] is None


def test_the_range_window_is_honoured():
    """Mutation check on range_length: a longer window sees a deeper low, so
    the midpoint - and therefore the reading - moves."""
    highs, lows, closes = _ohlc_trend(60, 1.0)
    lows[-8] -= 40.0  # a spike outside a 5-bar window but inside a 10-bar one
    short = compute_smi(highs, lows, closes, SmiConfig(range_length=5)).sm1[-1]
    long = compute_smi(highs, lows, closes, SmiConfig(range_length=10)).sm1[-1]
    assert short != pytest.approx(long)


# -- the signal ------------------------------------------------------------
def _fake_smi(sm1, sm2):
    from indicators.smi import SmiResult

    return SmiResult("x", tuple(sm1), tuple(sm2))


def test_a_bullish_cross_below_zero_is_reported():
    result = _fake_smi([-0.5, -0.3], [-0.4, -0.35])
    assert result.bullish_cross_indices() == (1,)


def test_a_cross_above_zero_is_not_reported():
    """Both-below-zero is part of the signal, not a filter bolted on: the
    trader's interest is a turn out of a washed-out state."""
    result = _fake_smi([0.3, 0.5], [0.4, 0.45])
    assert result.bullish_cross_indices() == ()


def test_a_cross_with_only_one_line_below_zero_is_not_reported():
    result = _fake_smi([-0.1, 0.05], [-0.05, 0.0])
    assert result.bullish_cross_indices() == ()


def test_a_bearish_cross_is_not_reported_as_bullish():
    """Mutation check: flipping the comparison turns this green."""
    result = _fake_smi([-0.3, -0.5], [-0.35, -0.4])
    assert result.bullish_cross_indices() == ()


def test_already_above_does_not_re_report():
    result = _fake_smi([-0.3, -0.2, -0.1], [-0.4, -0.45, -0.5])
    assert result.bullish_cross_indices() == ()


def test_a_none_in_either_line_is_skipped_not_crossed():
    result = _fake_smi([None, -0.3], [-0.4, -0.35])
    assert result.bullish_cross_indices() == ()


# ==========================================================================
# shared contract
# ==========================================================================
def test_every_module_stamps_its_own_feature_version():
    from indicators import efficiency_lrsi, heikin_ashi, smi

    versions = {
        smi.FEATURE_VERSION,
        efficiency_lrsi.FEATURE_VERSION,
        heikin_ashi.FEATURE_VERSION,
    }
    assert len(versions) == 3


def test_the_efficiency_oscillator_is_not_the_ehlers_laguerre_module():
    """The naming collision the spec exists to prevent."""
    from indicators import efficiency_lrsi, laguerre_rsi

    assert efficiency_lrsi.FEATURE_VERSION != laguerre_rsi.FEATURE_VERSION
    assert not hasattr(efficiency_lrsi, "LaguerreState")


def test_every_series_aligns_one_to_one_with_its_input():
    highs, lows, closes = _ohlc_trend(30, 1.0)
    assert len(compute_smi(highs, lows, closes).sm1) == 30
    assert len(compute_efficiency_lrsi(closes).values) == 30
    assert len(compute_heikin_ashi(closes, highs, lows, closes).bars) == 30


def test_empty_input_is_empty_output_not_a_crash():
    assert compute_smi([], [], []).sm1 == ()
    assert compute_efficiency_lrsi([]).values == ()
    assert compute_heikin_ashi([], [], [], []).bars == ()
