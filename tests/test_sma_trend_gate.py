"""Longs above the SMA200, shorts below the SMA50 - the rule and nothing tighter."""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sma_trend_gate as gate  # noqa: E402
from prev_day_gate import CLOSED, OPEN, UNKNOWN  # noqa: E402


def _bars(closes, *, end: date, preview_last: bool = False):
    start = datetime.combine(end, datetime.min.time()) - timedelta(days=len(closes) - 1)
    bars = [
        {"dt": start + timedelta(days=i), "open": c, "high": c + 1, "low": c - 1, "close": c}
        for i, c in enumerate(closes)
    ]
    if preview_last:
        bars[-1]["preview"] = True
    return bars


class TestTheRule:
    def test_a_long_above_the_sma200_is_open(self):
        assert gate.sma_trend_state("long", 10.0, 12.0, 9.0) == (OPEN, "above the SMA200")

    def test_a_long_under_the_sma200_is_closed_whatever_the_sma50_says(self):
        state, reason = gate.sma_trend_state("long", 10.0, 9.0, 11.0)
        assert state == CLOSED and "SMA200" in reason

    def test_a_short_below_the_sma50_is_open(self):
        assert gate.sma_trend_state("short", 10.0, 11.0, 8.0) == (OPEN, "below the SMA50")

    def test_a_short_over_the_sma50_is_closed_even_under_the_sma200(self):
        """MUFG's mirror: the short needs the 50, the 200 is not consulted."""
        state, reason = gate.sma_trend_state("short", 10.0, 9.0, 12.0)
        assert state == CLOSED and "SMA50" in reason

    def test_a_long_does_not_need_the_sma50(self):
        """Trader: "at least". Above the 200 but under the 50 still shows."""
        assert gate.sma_trend_state("long", 10.0, 11.0, 9.0)[0] == OPEN

    def test_at_the_line_is_not_beyond_it(self):
        assert gate.sma_trend_state("long", 10.0, None, 10.0)[0] == CLOSED
        assert gate.sma_trend_state("short", 10.0, 10.0, None)[0] == CLOSED

    def test_missing_price_or_the_needed_average_is_unknown_never_closed(self):
        assert gate.sma_trend_state("long", None, 9.0, 8.0)[0] == UNKNOWN
        assert gate.sma_trend_state("long", 10.0, 9.0, None)[0] == UNKNOWN
        assert gate.sma_trend_state("short", 10.0, None, 8.0)[0] == UNKNOWN
        assert gate.sma_trend_state("long", float("nan"), 9.0, 8.0)[0] == UNKNOWN


class TestTheLevels:
    def test_short_history_has_no_average(self):
        bars = _bars([10.0] * 60, end=date(2026, 8, 26))
        sma50, sma200 = gate.trend_levels(bars, today=date(2026, 8, 27))
        assert sma50 == 10.0
        assert sma200 is None, "60 closes is not a 200-day average"

    def test_the_forming_candle_is_left_out(self):
        """A preview bar, or today's bar while today trades, moves every tick."""
        closes = [10.0] * 200 + [50.0]
        today = date(2026, 8, 27)
        by_flag = _bars(closes, end=today, preview_last=True)
        by_date = _bars(closes, end=today)
        for bars in (by_flag, by_date):
            sma50, sma200 = gate.trend_levels(bars, today=today)
            assert sma50 == 10.0 and sma200 == 10.0

    def test_yesterdays_bar_counts_once_the_day_has_rolled(self):
        closes = [10.0] * 200 + [50.0]
        bars = _bars(closes, end=date(2026, 8, 26))
        sma50, _sma200 = gate.trend_levels(bars, today=date(2026, 8, 27))
        assert sma50 > 10.0

    def test_junk_rows_are_skipped_not_fatal(self):
        bars = _bars([10.0] * 50, end=date(2026, 8, 26))
        bars.insert(0, {"dt": None, "close": "x"})
        bars.insert(0, "not a bar")
        assert gate.trend_levels(bars, today=date(2026, 8, 27))[0] == 10.0
