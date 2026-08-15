"""The M5 strength formula, against hand-computed values (packet R2 Part B).

The spec asks for fixture tests over hand-computed OHLCV series, so the
arithmetic here is worked out in the comments rather than recomputed by the
same code it is checking. A test that calls `strength_score` twice and compares
the answers to itself would pass no matter what the formula did.
"""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from strength_scan import (  # noqa: E402
    STRENGTH_ATR_PERIOD,
    atr,
    percentile_cut,
    sma,
    strength_score,
    true_ranges,
)


def _flat_bars(count, *, close=100.0, spread=1.0):
    """`count` identical bars: open == close, so every body move is zero."""
    return [
        {"open": close, "high": close + spread / 2, "low": close - spread / 2, "close": close}
        for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------


def test_sma_refuses_a_window_it_cannot_fill():
    """A 12-bar average called SMA50 would rank a fresh listing against a name
    with real history."""
    assert sma([1.0, 2.0, 3.0], 3) == 2.0
    assert sma([1.0, 2.0, 3.0], 4) is None
    assert sma([], 1) is None
    assert sma([1.0, float("nan")], 2) is None


def test_true_range_is_wilders_and_starts_at_the_second_bar():
    # Bar 1 close 100. Bar 2: high 103, low 99 -> max(4, |103-100|, |99-100|) = 4.
    # Bar 3 (prev close 101): high 102, low 95 -> max(7, |102-101|, |95-101|) = 7.
    bars = [
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        {"open": 100.0, "high": 103.0, "low": 99.0, "close": 101.0},
        {"open": 101.0, "high": 102.0, "low": 95.0, "close": 96.0},
    ]
    assert true_ranges(bars) == [4.0, 7.0]
    assert true_ranges(bars[:1]) is None


def test_atr_needs_one_more_bar_than_its_period():
    """The first bar contributes no true range, so ATR50 needs 51 bars."""
    assert atr(_flat_bars(50, spread=2.0), 50) is None
    assert atr(_flat_bars(51, spread=2.0), 50) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------


def test_a_flat_tape_scores_zero():
    """Every bar closes where it opened: the body-move sum is zero, so the
    price and volatility factors have nothing to scale."""
    assert strength_score(_flat_bars(60)) == pytest.approx(0.0)


def test_the_score_matches_a_hand_computed_series():
    """A deliberately simple series, worked out by hand.

    51 flat bars at 100.00 with a 2.00 range (open == close == 100), then 12
    bars that each open at 100.00 and close at 101.00 with the same 2.00 range
    (high 102, low 100).

    Body factor: each of the 12 bars gives (101/100 - 1) * 100 = 1.0, so the
        sum is 12.0 and the average is 1.0.
    SMA50 of closes: the last 50 closes are the 12 at 101.00 plus the 38
        remaining flat ones at 100.00 -> (12*101 + 38*100) / 50 = 100.24.
    Price factor: (101.00 + 100.24) / 2 = 100.62.
    ATR50: every true range is 2.00 (each bar's high-low, which dominates the
        gap terms), so ATR50 = 2.00.
    Score = 1.0 * 100.62 / 2.00 = 50.31.
    """
    bars = _flat_bars(51, close=100.0, spread=2.0)
    bars += [
        {"open": 100.0, "high": 102.0, "low": 100.0, "close": 101.0} for _ in range(12)
    ]
    assert strength_score(bars) == pytest.approx(50.31, abs=1e-9)


def test_a_falling_tape_scores_negative_and_mirrors():
    """The short side is the same formula, not a second one."""
    bars = _flat_bars(51, close=100.0, spread=2.0)
    bars += [
        {"open": 100.0, "high": 100.0, "low": 98.0, "close": 99.0} for _ in range(12)
    ]
    # Body: (99/100 - 1) * 100 = -1.0 per bar -> average -1.0.
    # SMA50 = (12*99 + 38*100) / 50 = 99.76; price factor = (99 + 99.76)/2 = 99.38.
    # Every true range is 2.00 -> ATR50 = 2.00. Score = -1.0 * 99.38 / 2.0.
    assert strength_score(bars) == pytest.approx(-49.69, abs=1e-9)


def test_volatility_normalisation_ranks_the_quiet_mover_higher():
    """The point of dividing by ATR: the same 1% push counts for more when the
    name is not already flailing."""

    def mover(spread):
        bars = [
            {"open": 100.0, "high": 100.0 + spread / 2, "low": 100.0 - spread / 2, "close": 100.0}
            for _ in range(51)
        ]
        bars += [
            {"open": 100.0, "high": 101.0 + spread / 2, "low": 100.0 - spread / 2, "close": 101.0}
            for _ in range(12)
        ]
        return strength_score(bars)

    assert mover(1.0) > mover(4.0)


def test_price_level_keeps_a_cheap_and_an_expensive_name_comparable():
    """Two names making the same percentage move rank in price order, which is
    what the (C + SMA50)/2 factor is for - and neither is excluded."""

    def name(level):
        spread = level * 0.02
        bars = [
            {"open": level, "high": level + spread / 2, "low": level - spread / 2, "close": level}
            for _ in range(51)
        ]
        bars += [
            {
                "open": level,
                "high": level * 1.01 + spread / 2,
                "low": level - spread / 2,
                "close": level * 1.01,
            }
            for _ in range(12)
        ]
        return strength_score(bars)

    cheap, dear = name(20.0), name(400.0)
    assert cheap is not None and dear is not None
    # Same percentage move and same relative volatility -> the same score, so
    # the ranking is about the move rather than the share price.
    assert cheap == pytest.approx(dear, rel=1e-9)


# ---------------------------------------------------------------------------
# Refusals: an unmeasurable row is not a weak row
# ---------------------------------------------------------------------------


def test_short_history_scores_nothing_rather_than_something():
    assert strength_score(_flat_bars(11)) is None       # fewer than 12 body bars
    assert strength_score(_flat_bars(50)) is None       # cannot fill SMA50/ATR50
    assert strength_score(_flat_bars(51)) is not None


def test_bad_values_score_nothing():
    bars = _flat_bars(60)
    for bad in (float("nan"), float("inf"), None, "x"):
        broken = [dict(bar) for bar in bars]
        broken[-1]["close"] = bad
        assert strength_score(broken) is None, f"close={bad!r}"
    zero_open = [dict(bar) for bar in bars]
    zero_open[-1]["open"] = 0.0
    assert strength_score(zero_open) is None


def test_a_zero_atr_scores_nothing_rather_than_infinity():
    """A perfectly flat name would divide by zero; the row is unmeasurable."""
    bars = [{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0} for _ in range(51)]
    bars += [{"open": 100.0, "high": 101.0, "low": 100.0, "close": 101.0} for _ in range(12)]
    # The 12 moving bars give the series a nonzero ATR, so this one scores...
    assert strength_score(bars) is not None
    # ... but a completely motionless series does not.
    assert strength_score(_flat_bars(63, spread=0.0)) is None


def test_the_atr_period_is_the_documented_fifty():
    assert STRENGTH_ATR_PERIOD == 50


# ---------------------------------------------------------------------------
# The percentile cut
# ---------------------------------------------------------------------------


def test_the_cut_keeps_the_top_quarter_for_longs_and_the_bottom_for_shorts():
    scored = [(f"S{index}", float(index)) for index in range(20)]
    longs = percentile_cut(scored, fraction=0.25, side="long")
    shorts = percentile_cut(scored, fraction=0.25, side="short")

    # 25% of 20 measurable names is 5 rows a side.
    assert [symbol for symbol, _ in longs] == ["S19", "S18", "S17", "S16", "S15"]
    # Shorts read weakest-first, so both sides put the best row at the top.
    assert [symbol for symbol, _ in shorts] == ["S0", "S1", "S2", "S3", "S4"]


def test_the_cut_is_a_proportion_of_what_was_measurable():
    """A session where half the universe is unmeasurable narrows the board
    rather than promoting noise into it."""
    assert len(percentile_cut([(f"S{i}", float(i)) for i in range(100)], fraction=0.25)) == 25
    assert len(percentile_cut([(f"S{i}", float(i)) for i in range(40)], fraction=0.25)) == 10


def test_the_cut_never_returns_an_empty_board_from_a_real_population():
    """25% of three names is not zero names."""
    assert len(percentile_cut([("A", 1.0), ("B", 2.0), ("C", 3.0)], fraction=0.25)) == 1
    assert percentile_cut([], fraction=0.25) == []


def test_unmeasurable_rows_never_reach_the_cut():
    scored = [("GOOD", 5.0), ("NAN", float("nan")), ("NONE", None), ("ALSO", 1.0)]
    kept = percentile_cut(scored, fraction=1.0, side="long")
    assert [symbol for symbol, _ in kept] == ["GOOD", "ALSO"]


# ---------------------------------------------------------------------------
# The board: score everything, cut, then filter
# ---------------------------------------------------------------------------


def _series(symbol_close, *, prev_high, prev_low, vwap_pull=1.0, sessions=2):
    """Two sessions of M5 bars ending at `symbol_close`.

    The prior session is shaped to the requested high/low; today's bars carry
    volume so a session VWAP exists, and `vwap_pull` scales today's opening
    prices so the VWAP can be put above or below the close on demand.
    """
    from datetime import datetime, timedelta

    bars = []
    day_one = datetime(2026, 7, 1, 6, 30)
    # 60 prior-session bars spanning prev_low..prev_high.
    for index in range(60):
        mid = (prev_high + prev_low) / 2
        bars.append({
            "dt": day_one + timedelta(minutes=5 * index),
            "open": mid, "high": prev_high, "low": prev_low, "close": mid,
            "volume": 1000.0,
        })
    day_two = datetime(2026, 7, 2, 6, 30)
    for index in range(20):
        opening = symbol_close * vwap_pull
        bars.append({
            "dt": day_two + timedelta(minutes=5 * index),
            "open": opening,
            "high": max(opening, symbol_close) + 0.1,
            "low": min(opening, symbol_close) - 0.1,
            "close": symbol_close,
            "volume": 1000.0,
        })
    return bars


def test_a_row_carries_the_numbers_the_board_shows():
    bars = _series(105.0, prev_high=100.0, prev_low=98.0)
    from strength_scan import score_symbol

    row = score_symbol("NVDA", bars)
    assert row["symbol"] == "NVDA"
    assert row["prev_high"] == pytest.approx(100.0)
    assert row["prev_low"] == pytest.approx(98.0)
    assert row["last"] == pytest.approx(105.0)
    assert row["session_vwap"] is not None
    assert row["ema15"] is not None


def test_a_symbol_without_a_prior_session_is_unmeasurable():
    """No yesterday means no yesterday's high to clear."""
    from datetime import datetime, timedelta

    from strength_scan import score_symbol

    day = datetime(2026, 7, 2, 6, 30)
    bars = [
        {"dt": day + timedelta(minutes=5 * i), "open": 100.0, "high": 101.0,
         "low": 99.0, "close": 100.5, "volume": 1000.0}
        for i in range(80)
    ]
    assert score_symbol("NVDA", bars) is None


def test_the_board_cuts_before_it_filters():
    """Order matters: "top 25%" must mean 25% of the measured population, not
    25% of an already-filtered one."""
    from strength_scan import build_strength_board

    # Eight names, all measurable. Only some clear their filters.
    bars_by_symbol = {
        f"S{index}": _series(105.0 + index, prev_high=100.0, prev_low=98.0)
        for index in range(8)
    }
    board = build_strength_board(bars_by_symbol, fraction=0.25)
    assert board["offered"] == 8
    assert board["measured"] == 8
    # 25% of 8 = 2 rows a side entered the filters.
    assert len(board["long"]) + board["long_filtered_out"] == 2
    assert len(board["short"]) + board["short_filtered_out"] == 2


def test_a_long_row_must_clear_vwap_the_ema_and_yesterdays_high():
    from strength_scan import build_strength_board

    # Above yesterday's high and pulled up through VWAP: qualifies.
    good = _series(105.0, prev_high=100.0, prev_low=98.0, vwap_pull=0.97)
    # Below yesterday's high: cut regardless of strength.
    inside = _series(99.0, prev_high=100.0, prev_low=98.0, vwap_pull=0.97)

    board = build_strength_board({"GOOD": good}, fraction=1.0)
    assert [row["symbol"] for row in board["long"]] == ["GOOD"]

    board = build_strength_board({"INSIDE": inside}, fraction=1.0)
    assert board["long"] == []
    assert board["long_filtered_out"] == 1


def test_the_board_reports_what_it_could_not_measure():
    """A short board has to be explainable, not just short."""
    from strength_scan import build_strength_board

    board = build_strength_board(
        {
            "GOOD": _series(105.0, prev_high=100.0, prev_low=98.0, vwap_pull=0.97),
            "TOOSHORT": _series(105.0, prev_high=100.0, prev_low=98.0)[:20],
        },
        fraction=1.0,
    )
    assert board["offered"] == 2
    assert board["measured"] == 1


def test_an_empty_universe_produces_an_honest_empty_board():
    from strength_scan import build_strength_board

    board = build_strength_board({})
    assert board["long"] == [] and board["short"] == []
    assert board["offered"] == 0 and board["measured"] == 0


def test_the_fetch_period_is_five_days_not_one():
    """The formula needs 50 completed bars; a 1d window holds six at 07:00."""
    from strength_scan import STRENGTH_FETCH_PERIOD

    assert STRENGTH_FETCH_PERIOD == "5d"
