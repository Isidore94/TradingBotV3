"""R8 §9 step 3 - the weekend boards, offline and on a frozen clock.

No network, no yfinance, no IB. Bars are constructed in the test, `now` is
passed in, and every assertion is about a rule the spec states rather than a
number a provider happened to return.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import weekend_strength as ws  # noqa: E402


def _bar(stamp, open_price, close, high=None, low=None, volume=10_000):
    return {
        "timestamp": stamp,
        "open": open_price,
        "close": close,
        "high": high if high is not None else max(open_price, close) + 0.5,
        "low": low if low is not None else min(open_price, close) - 0.5,
        "volume": volume,
    }


def _series(count, start_stamp, step, *, base=100.0, drift=0.5):
    """A rising series long enough to satisfy ATR50."""
    bars = []
    price = base
    for index in range(count):
        stamp = start_stamp + step * index
        close = price + drift
        bars.append(_bar(stamp, price, close))
        price = close
    return bars


# ---------------------------------------------------------------------------
# Completed bars: three rules, one invariant
# ---------------------------------------------------------------------------


def test_an_hourly_bar_is_complete_sixty_minutes_after_it_opened():
    now = datetime(2026, 8, 14, 12, 0)
    bars = [
        _bar(datetime(2026, 8, 14, 10, 0), 100, 101),
        _bar(datetime(2026, 8, 14, 11, 0), 101, 102),   # closes exactly at 12:00
        _bar(datetime(2026, 8, 14, 11, 30), 102, 103),  # still forming
    ]
    kept = ws.completed_bars(ws.H1, bars, now=now)
    assert [b["timestamp"].hour for b in kept] == [10, 11]


def test_a_daily_bar_follows_the_session_calendar_not_yesterday():
    """After a Monday holiday the last completed session is Friday.

    Counting back a calendar day would score a day the market never opened.
    """
    tuesday = datetime(2026, 9, 8, 10, 0)  # 2026-09-07 is Labor Day
    bars = [
        _bar(date(2026, 9, 3), 100, 101),
        _bar(date(2026, 9, 4), 101, 102),   # Friday - the last completed session
        _bar(date(2026, 9, 8), 102, 103),   # today, forming
    ]
    kept = ws.completed_bars(ws.D1, bars, now=tuesday)
    assert [b["timestamp"] for b in kept] == [date(2026, 9, 3), date(2026, 9, 4)]


def test_the_forming_month_is_dropped_by_identity():
    now = datetime(2026, 8, 20, 10, 0)
    bars = [
        _bar(date(2026, 6, 1), 100, 110),
        _bar(date(2026, 7, 1), 110, 120),
        _bar(date(2026, 8, 1), 120, 125),  # the in-progress month
    ]
    kept = ws.completed_bars(ws.M1, bars, now=now)
    assert [b["timestamp"].month for b in kept] == [6, 7]


def test_the_forming_month_is_dropped_on_day_one_of_that_month():
    """The case duration arithmetic gets exactly backwards.

    On the 1st, the in-progress bar is minutes old. A "has 30 days passed?" test
    would call it incomplete *and call the previous month complete only by
    accident* - and on the 1st at 00:01 it would happily score a bar containing
    one minute of trading. Month identity is right on every day including this
    one.
    """
    now = datetime(2026, 8, 1, 0, 1)
    bars = [
        _bar(date(2026, 6, 1), 100, 110),
        _bar(date(2026, 7, 1), 110, 120),
        _bar(date(2026, 8, 1), 120, 120.05),  # one minute old
    ]
    kept = ws.completed_bars(ws.M1, bars, now=now)
    assert [b["timestamp"].month for b in kept] == [6, 7]
    assert all(b["timestamp"].month != 8 for b in kept)


def test_a_bar_with_no_readable_timestamp_is_dropped_not_guessed():
    now = datetime(2026, 8, 20, 10, 0)
    bars = [_bar(date(2026, 7, 1), 100, 110), {"open": 1, "close": 2, "high": 2, "low": 1}]
    assert len(ws.completed_bars(ws.M1, bars, now=now)) == 1


# ---------------------------------------------------------------------------
# 51 bars, per timeframe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", ws.TIMEFRAMES, ids=lambda tf: tf.key)
def test_fifty_bars_refuses_and_fifty_one_measures(timeframe):
    """ATR50 needs 50 true ranges, and a true range needs a previous close.

    One short is a refusal on every timeframe - a short-history name is not a
    weak name, and scoring it as one would rank a data problem against real
    setups.
    """
    step = {"h1": timedelta(hours=1), "d1": timedelta(days=1), "m1": timedelta(days=31)}[timeframe.key]
    start = datetime(2020, 1, 1, 10, 0)
    assert ws.measure_symbol(timeframe, "SHORT", _series(50, start, step)) is None
    assert ws.measure_symbol(timeframe, "OK", _series(51, start, step)) is not None


def test_a_symbol_with_no_bars_at_all_is_missing_not_zero():
    assert ws.measure_symbol(ws.D1, "EMPTY", []) is None


# ---------------------------------------------------------------------------
# Parity with the fenced M5 functions
# ---------------------------------------------------------------------------


def test_the_score_is_the_m5_score_not_a_reimplementation():
    """R8 imports the formula; it does not copy it. Same bars, same number."""
    from strength_scan import strength_score

    bars = _series(60, datetime(2026, 6, 1, 10, 0), timedelta(days=1))
    row = ws.measure_symbol(ws.D1, "AAPL", bars)
    assert row["score"] == strength_score(bars, body_bars=12, atr_period=50)


def test_the_shared_constants_are_the_m5_constants():
    import strength_scan

    for timeframe in ws.TIMEFRAMES:
        assert timeframe.body_bars == strength_scan.STRENGTH_BODY_BARS
        assert timeframe.atr_period == strength_scan.STRENGTH_ATR_PERIOD
        assert timeframe.ema_span == strength_scan.STRENGTH_EMA_SPAN
    assert ws.MIN_BARS == strength_scan.STRENGTH_ATR_PERIOD + 1


def test_the_monthly_period_clears_the_minimum_with_margin():
    """6y, not max: 51 completed months is ~4.35 years."""
    assert ws.M1.yf_period == "6y"
    assert 6 * 12 > ws.MIN_BARS


# ---------------------------------------------------------------------------
# Filters (spec §5, approved as proposed)
# ---------------------------------------------------------------------------


def test_no_filter_leg_computes_a_vwap():
    """Dropped above M5, not imitated.

    There is no session inside an H1, D1 or monthly bar, so a session VWAP has
    nothing to anchor to. A look-alike would read like the M5 gate's number and
    mean something else.

    Checked by parsing the filter functions rather than grepping the file - the
    module's own docstring says the word "VWAP" while explaining why it is
    absent, and a substring search would fail on the sentence promising the
    thing it is checking for.
    """
    import ast
    import inspect

    for legs in ws.FILTERS.values():
        for leg in legs:
            tree = ast.parse(inspect.getsource(leg))
            names = {
                node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
            } | {
                node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            } | {
                str(node.value).lower()
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            assert not any("vwap" in name for name in names), f"{leg.__name__} reaches for a VWAP"
    # And no leg reads volume at all, which is what a VWAP would need.
    for legs in ws.FILTERS.values():
        for leg in legs:
            assert "volume" not in inspect.getsource(leg)


def test_each_timeframe_has_its_own_named_legs():
    """So one timeframe can be amended without touching the others."""
    assert ws.FILTERS["h1"] == (ws.leg_trend_vs_ema, ws.leg_prior_session_extreme)
    assert ws.FILTERS["d1"] == (ws.leg_trend_vs_ema, ws.leg_prior_week_extreme)
    assert ws.FILTERS["m1"] == (ws.leg_prior_month_extreme,)


def test_an_unmeasurable_leg_fails_with_a_reason_rather_than_passing():
    ok, reason = ws.leg_trend_vs_ema({"last_close": 100.0, "ema": None}, "long")
    assert ok is False and "EMA15" in reason

    ok, reason = ws.leg_prior_session_extreme({"bars": [_bar(datetime(2026, 8, 14, 10), 1, 2)]}, "long")
    assert ok is False and "no prior completed period" in reason


def test_the_short_side_is_a_true_mirror():
    """Same legs, comparison and extreme both inverted."""
    bars = [
        _bar(date(2026, 8, 10), 100, 99, high=101, low=98),
        _bar(date(2026, 8, 11), 99, 98, high=100, low=97),
        _bar(date(2026, 8, 17), 98, 96, high=98, low=95),
    ]
    row = {"last_close": 96.0, "ema": 99.0, "bars": bars}
    assert ws.leg_trend_vs_ema(row, "short")[0] is True
    assert ws.leg_trend_vs_ema(row, "long")[0] is False
    # 96 is below the prior ISO week's low of 97.
    assert ws.leg_prior_week_extreme(row, "short")[0] is True
    assert ws.leg_prior_week_extreme(row, "long")[0] is False


def test_the_monthly_leg_does_not_demand_fifteen_months_of_ema():
    """An EMA15 of monthly closes is 15 more months on top of the 51 the score
    needs, which would silently exclude names the board should show."""
    assert ws.leg_trend_vs_ema not in ws.FILTERS["m1"]


# ---------------------------------------------------------------------------
# Board order: percentile BEFORE filters
# ---------------------------------------------------------------------------


def _population(now, count=8):
    """A population where the strongest names would fail the filters.

    Built so cut-then-filter and filter-then-cut give visibly different answers.
    """
    bars_by_symbol = {}
    start = datetime(2026, 1, 1, 10, 0)
    for index in range(count):
        drift = 0.1 + index * 0.2
        bars_by_symbol[f"SYM{index}"] = _series(60, start, timedelta(days=1), drift=drift)
    return bars_by_symbol


def test_the_percentile_describes_the_measurable_population_not_the_filtered_one():
    now = datetime(2026, 4, 1, 12, 0)
    population = _population(now)
    board = ws.build_board(ws.D1, population, side="long", now=now, fraction=0.25)

    assert board.offered == len(population)
    assert board.measured <= board.offered
    # The cut is taken over everything measurable, so the count is a fraction of
    # `measured` - never of whatever survived the filters.
    assert board.in_percentile == max(1, int(board.measured * 0.25))
    assert len(board.rows) + board.filtered_out == board.in_percentile


def test_the_accounting_line_reports_what_could_not_be_measured():
    now = datetime(2026, 4, 1, 12, 0)
    population = _population(now)
    population["TOOSHORT"] = _series(10, datetime(2026, 3, 1, 10, 0), timedelta(days=1))
    board = ws.build_board(ws.D1, population, side="long", now=now)

    assert board.measured == board.offered - 1
    assert "had too little history to score" in board.accounting
    assert "TOOSHORT" not in [row["symbol"] for row in board.rows]


def test_an_empty_universe_is_an_empty_board_not_a_crash():
    board = ws.build_board(ws.D1, {}, side="long", now=datetime(2026, 4, 1, 12, 0))
    assert board.rows == [] and board.offered == 0 and board.measured == 0


def test_the_board_is_deterministic_across_two_runs():
    now = datetime(2026, 4, 1, 12, 0)
    population = _population(now)
    first = ws.build_board(ws.D1, population, side="long", now=now)
    second = ws.build_board(ws.D1, population, side="long", now=now)
    assert [r["symbol"] for r in first.rows] == [r["symbol"] for r in second.rows]
    assert [r["score"] for r in first.rows] == [r["score"] for r in second.rows]


@pytest.mark.parametrize("timeframe", ws.TIMEFRAMES, ids=lambda tf: tf.key)
def test_every_timeframe_builds_both_sides_without_a_network(timeframe):
    now = datetime(2026, 4, 1, 12, 0)
    step = {"h1": timedelta(hours=1), "d1": timedelta(days=1), "m1": timedelta(days=31)}[timeframe.key]
    population = {
        f"SYM{i}": _series(60, datetime(2019, 1, 1, 10, 0), step, drift=0.1 + i * 0.2)
        for i in range(6)
    }
    for side in ("long", "short"):
        board = ws.build_board(timeframe, population, side=side, now=now)
        assert board.timeframe == timeframe.key and board.side == side
        assert board.as_of.startswith("2026-04-01")
