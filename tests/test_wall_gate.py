"""wall_gate: hide a chart sitting right under an SMA or a D1 trendline (trader, 2026-09-23)."""

from __future__ import annotations

import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import wall_gate  # noqa: E402
from wall_gate import CLOSED, OPEN, UNKNOWN, wall_state  # noqa: E402

TODAY = date(2026, 9, 23)  # a Wednesday


def daily(segments, *, end: date = TODAY, spread: float = 0.5) -> list[dict]:
    """Daily bars ending the calendar day before ``end``; high/low = close +/- spread."""
    closes: list[float] = []
    for count, first, last in segments:
        for i in range(count):
            step = 0.0 if count == 1 else (last - first) * i / (count - 1)
            closes.append(first + step)
    start = datetime.combine(end, datetime.min.time()) - timedelta(days=len(closes))
    return [
        {
            "dt": start + timedelta(days=i),
            "open": c,
            "high": c + spread,
            "low": c - spread,
            "close": c,
        }
        for i, c in enumerate(closes)
    ]


# 150 closes at 22 then 100 at 20: SMA50 = SMA100 = 20, SMA200 = 21, ATR20 ~ 1.0.
STEP_DOWN = [(150, 22.0, 22.0), (100, 20.0, 20.0)]
# 150 at 18 then 100 at 20: SMA50 = SMA100 = 20, SMA200 = 19.
STEP_UP = [(150, 18.0, 18.0), (100, 20.0, 20.0)]


def line(value: float, *, end: date, slope: float = 0.0, **extra) -> dict:
    record = {
        "current_line_price": value,
        "slope_log_per_bar": slope,
        "lookback_end": end.isoformat(),
    }
    record.update(extra)
    return record


class TestSmaWallIsInThePathOnly:
    def test_a_long_just_under_its_sma200_is_at_a_wall(self):
        verdict = wall_state("LONG", 20.8, daily(STEP_DOWN), today=TODAY)
        assert verdict.state == CLOSED
        assert verdict.wall == "SMA200"
        assert verdict.follow_up == "sma"
        assert math.isclose(verdict.level, 21.0)
        assert 0.1 < verdict.distance_atr < 0.3
        assert verdict.source_text() == f"auto: SMA200 wall {verdict.distance_atr:.1f} ATR"

    def test_a_long_just_above_its_smas_is_support_and_shows(self):
        verdict = wall_state("LONG", 20.8, daily(STEP_UP), today=TODAY)
        assert verdict.state == OPEN

    def test_a_short_just_above_its_smas_is_at_a_wall(self):
        verdict = wall_state("SHORT", 20.4, daily(STEP_UP), today=TODAY)
        assert verdict.state == CLOSED
        assert verdict.wall in ("SMA50", "SMA100")

    def test_a_short_with_the_smas_above_it_shows(self):
        verdict = wall_state("SHORT", 19.2, daily(STEP_DOWN), today=TODAY)
        assert verdict.state == OPEN

    def test_more_than_one_atr_away_is_not_a_wall(self):
        verdict = wall_state("LONG", 19.5, daily(STEP_DOWN), today=TODAY)
        # SMA50/100 = 20 is 0.5 ATR above a long at 19.5: a wall.
        assert verdict.state == CLOSED and verdict.wall in ("SMA50", "SMA100")
        verdict = wall_state("LONG", 18.7, daily(STEP_DOWN), today=TODAY)
        assert verdict.state == OPEN

    def test_once_through_the_wall_the_gate_opens(self):
        """The SMA break fires and price is above the SMA200: nothing in the path."""
        assert wall_state("LONG", 20.9, daily(STEP_DOWN), today=TODAY).state == CLOSED
        assert wall_state("LONG", 21.1, daily(STEP_DOWN), today=TODAY).state == OPEN

    def test_the_multiple_is_the_kill_switch_constant(self):
        assert wall_gate.WALL_ATR_MULTIPLE == 1.0
        assert wall_gate.WALL_GATE_ENABLED is True
        verdict = wall_state("LONG", 20.8, daily(STEP_DOWN), today=TODAY, enabled=False)
        assert verdict.state == OPEN

    def test_a_short_history_just_skips_the_sma200(self):
        # 120 closes: SMA50 and SMA100 exist, SMA200 does not.
        bars = daily([(70, 22.0, 22.0), (50, 20.0, 20.0)])
        levels = wall_gate.sma_levels(wall_gate.completed_daily_bars(bars, today=TODAY))
        assert set(levels) == {"SMA50", "SMA100"}
        assert wall_state("LONG", 21.8, bars, today=TODAY).state == OPEN


class TestAtrAndLevelsUseCompletedBarsOnly:
    def test_the_forming_bar_is_left_out(self):
        bars = daily(STEP_DOWN)
        clean = wall_state("LONG", 20.8, bars, today=TODAY)
        forming = dict(bars[-1])
        forming["dt"] = datetime.combine(TODAY, datetime.min.time())
        forming.update({"high": 60.0, "low": 1.0, "close": 50.0})
        preview = dict(forming, preview=True)
        with_today = wall_state("LONG", 20.8, bars + [forming], today=TODAY)
        with_preview = wall_state("LONG", 20.8, bars + [preview], today=TODAY)
        assert with_today == clean
        assert with_preview == clean

    def test_atr_is_twenty_day(self):
        bars = wall_gate.completed_daily_bars(daily([(60, 20.0, 20.0)], spread=1.0), today=TODAY)
        assert math.isclose(wall_gate.atr20(bars), 2.0, rel_tol=1e-9)
        # 20 true ranges need 21 bars; 20 bars are unmeasurable, never zero.
        assert wall_gate.atr20(bars[:21]) is not None
        assert wall_gate.atr20(bars[:20]) is None


class TestUnknownIsNeverClosed:
    def test_no_price(self):
        assert wall_state("LONG", None, daily(STEP_DOWN), today=TODAY).state == UNKNOWN

    def test_no_bars_no_atr(self):
        assert wall_state("LONG", 20.8, [], today=TODAY).state == UNKNOWN
        assert wall_state("LONG", 20.8, daily([(15, 20.0, 20.0)]), today=TODAY).state == UNKNOWN

    def test_atr_but_no_level(self):
        verdict = wall_state("LONG", 20.1, daily([(30, 20.0, 20.0)]), today=TODAY)
        assert verdict.state == UNKNOWN

    def test_no_side(self):
        assert wall_state("WATCH", 20.8, daily(STEP_DOWN), today=TODAY).state == UNKNOWN


class TestTrendline:
    def _bars(self):
        # Flat far-away SMAs (all 10): only the trendline can be a wall.
        return daily([(250, 10.0, 10.0)])

    def test_within_one_atr_on_either_side_is_a_wall(self):
        completed = wall_gate.completed_daily_bars(daily([(250, 20.0, 20.0)]), today=TODAY)
        end = TODAY - timedelta(days=1)

        def at(price, level):
            return wall_gate.trendline_wall(
                price, [line(level, end=end)], completed, today=TODAY, atr=1.0, multiple=1.0
            )

        above = at(20.3, 20.9)  # line 0.6 ATR above price
        below = at(20.3, 19.7)  # line 0.6 ATR below price
        assert above.state == CLOSED and above.wall == "trendline"
        assert below.state == CLOSED and below.wall == "trendline"
        assert above.follow_up == "trendline"
        assert math.isclose(above.distance_atr, 0.6)
        assert at(20.3, 21.4) is None  # 1.1 ATR away

    def test_a_trendline_wall_hides_through_wall_state(self):
        bars = daily([(250, 30.0, 30.0)])
        end = TODAY - timedelta(days=1)
        verdict = wall_state("SHORT", 29.5, bars, today=TODAY, trendlines=[line(29.0, end=end)])
        assert verdict.state == CLOSED and verdict.wall == "trendline"

    def test_the_line_is_projected_to_today(self):
        bars = daily([(250, 30.0, 30.0)])
        end = TODAY - timedelta(days=1)
        # 30 -> one bar later at 30 * exp(0.05) = 31.54; price 33 is 3 ATR away from 30.
        record = line(30.0, end=end, slope=0.05)
        value = wall_gate.trendline_value(
            record, wall_gate.completed_daily_bars(bars, today=TODAY), today=TODAY
        )
        assert math.isclose(value, 30.0 * math.exp(0.05))

    def test_a_stale_record_is_ignored(self):
        bars = daily([(250, 30.0, 30.0)])
        old = TODAY - timedelta(days=9)
        record = line(31.0, end=old)
        verdict = wall_state("SHORT", 31.2, bars, today=TODAY, trendlines=[record])
        assert verdict.wall != "trendline"

    def test_break_day_hides_even_far_from_the_line(self):
        bars = daily([(250, 30.0, 30.0)])
        end = TODAY - timedelta(days=1)
        record = line(28.0, end=end, break_date=TODAY.isoformat())
        verdict = wall_state("LONG", 33.0, bars, today=TODAY, trendlines=[record])
        assert verdict.state == CLOSED
        assert verdict.wall == "trendline_break_day"
        assert verdict.follow_up == "trendline"

    def test_the_evening_of_the_break_is_still_the_break_day(self):
        """The scan saw the break on the latest completed bar, and no new
        session has started (a Friday break read on Saturday)."""
        saturday = date(2026, 9, 26)
        bars = daily([(250, 30.0, 30.0)], end=saturday)
        friday = saturday - timedelta(days=1)
        record = line(28.0, end=friday, break_date=friday.isoformat())
        verdict = wall_state("LONG", 33.0, bars, today=saturday, trendlines=[record])
        assert verdict.wall == "trendline_break_day"

    def test_the_next_session_shows_if_the_break_held(self):
        bars = daily([(250, 30.0, 30.0)])
        yesterday = TODAY - timedelta(days=1)
        record = line(28.0, end=yesterday, break_date=yesterday.isoformat())
        verdict = wall_state("LONG", 33.0, bars, today=TODAY, trendlines=[record])
        assert verdict.state == OPEN
        near = wall_state("LONG", 28.5, bars, today=TODAY, trendlines=[record])
        assert near.state == CLOSED and near.wall == "trendline"

    def test_crossing_the_line_today_is_the_break_day(self):
        bars = daily([(250, 30.0, 30.0)])
        end = TODAY - timedelta(days=1)
        record = line(31.0, end=end)  # yesterday's close 30 was under it
        verdict = wall_state("LONG", 33.5, bars, today=TODAY, trendlines=[record])
        assert verdict.wall == "trendline_break_day"

    def test_a_record_without_a_slope_is_ignored(self):
        bars = daily([(250, 30.0, 30.0)])
        record = {"current_line_price": 30.2, "lookback_end": (TODAY - timedelta(days=1)).isoformat()}
        verdict = wall_state("SHORT", 30.4, bars, today=TODAY, trendlines=[record])
        assert verdict.wall != "trendline"
