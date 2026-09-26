"""S7 shadow engines: fires / does not fire, completed bars only, missing data is no event.

Bars are naive Pacific market-local (06:30 = 09:30 ET), as the bot caches them.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import m5_signal_engines as eng  # noqa: E402

LOCAL = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")
DAYS = [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19)]
TODAY = date(2026, 8, 20)


def _start(day: date, index: int) -> datetime:
    return datetime(day.year, day.month, day.day, 6, 30) + timedelta(minutes=5 * index)


def _bar(day, index, close, *, open_=None, high=None, low=None, volume=1000.0):
    open_ = close if open_ is None else open_
    return {
        "dt": _start(day, index),
        "open": open_,
        "high": max(open_, close) + 0.01 if high is None else high,
        "low": min(open_, close) - 0.01 if low is None else low,
        "close": close,
        "volume": volume,
    }


def _session(day, closes, *, volume=1000.0):
    out, previous = [], None
    for index, close in enumerate(closes):
        out.append(_bar(day, index, close, open_=previous, volume=volume))
        previous = close
    return out


def _after(bars) -> datetime:
    """A clock at which every bar has completed."""
    return bars[-1]["dt"] + timedelta(minutes=5)


def _forming(bars) -> datetime:
    """A clock at which the last bar is still forming."""
    return bars[-1]["dt"] + timedelta(minutes=4)


# ----------------------------------------------------------------------- (a) PDH / PDL break-and-hold
def _pd_history(*, prior_days=3, short_previous=False):
    """Flat full sessions at 10.0; the last one prints a 10.2 high and a 9.8 low."""
    bars = []
    for day in DAYS[-prior_days:]:
        session = _session(day, [10.0] * 78)
        if day == DAYS[-1]:
            session[40] = _bar(day, 40, 10.0, high=10.2)
            session[50] = _bar(day, 50, 10.0, low=9.8)
            if short_previous:
                session = session[:-1]
        bars.extend(session)
    return bars


def _pd_today(closes, volume=2000.0):
    return _session(TODAY, closes, volume=volume)


LONG_BREAK = [10.0] * 7 + [10.3, 10.35]  # break 10:05 ET, hold 10:10 ET
SHORT_BREAK = [10.0] * 7 + [9.7, 9.65]


def _pd(bars, side="long", now=None):
    return eng.pd_level_break_hold_events(bars, symbol="abc", side=side, now=now or _after(bars), tz=LOCAL)


class TestPdLevelBreakHold:
    def test_fires_on_the_hold_bar_with_level_entry_stop_and_zoned_time(self):
        bars = _pd_history() + _pd_today(LONG_BREAK)
        (event,) = _pd(bars)
        assert event.engine == eng.SHADOW_PD_BREAK_HOLD
        assert (event.symbol, event.side) == ("ABC", "long")
        assert event.bar_time == datetime(2026, 8, 20, 10, 10, tzinfo=ET)
        assert event.bar_time.utcoffset() == timedelta(hours=-4)
        assert event.bar_close == datetime(2026, 8, 20, 10, 15, tzinfo=ET)
        assert (event.level, event.entry, event.stop) == (10.2, 10.35, 9.99)
        assert event.risk_per_share == pytest.approx(0.36)
        assert event.event_id == "s7:pd_level_break_hold:ABC:long:2026-08-20T10:10:00-04:00"
        details = dict(event.details)
        assert details["rvol"] == pytest.approx(2.0)
        assert details["rvol_sessions"] == 3

    def test_short_mirrors_on_the_previous_day_low(self):
        bars = _pd_history() + _pd_today(SHORT_BREAK)
        (event,) = _pd(bars, side="short")
        assert (event.side, event.level, event.entry) == ("short", 9.8, 9.65)
        assert event.stop > event.entry
        assert _pd(bars, side="long") == ()

    def test_low_rvol_is_no_event(self):
        bars = _pd_history() + _pd_today(LONG_BREAK, volume=1000.0)
        assert _pd(bars) == ()

    def test_a_hold_before_ten_is_no_event(self):
        bars = _pd_history() + _pd_today([10.0] * 3 + [10.3, 10.35, 10.4, 10.4, 10.4, 10.4])
        assert _pd(bars) == ()

    def test_the_hold_bar_must_be_completed(self):
        bars = _pd_history() + _pd_today(LONG_BREAK)
        assert _pd(bars, now=_forming(bars)) == ()

    def test_a_short_previous_session_is_unknown(self):
        bars = _pd_history(short_previous=True) + _pd_today(LONG_BREAK)
        assert _pd(bars) == ()

    def test_too_few_sessions_for_rvol_is_unknown(self):
        bars = _pd_history(prior_days=1) + _pd_today(LONG_BREAK)
        assert _pd(bars) == ()

    def test_missing_volume_is_unknown(self):
        today = _pd_today(LONG_BREAK)
        today[3]["volume"] = None
        assert _pd(_pd_history() + today) == ()

    def test_an_unreadable_bar_is_no_event(self):
        today = _pd_today(LONG_BREAK)
        today[2]["close"] = None
        assert _pd(_pd_history() + today) == ()


# ----------------------------------------------------------------------- (b) VWAP reclaim after a flush
FLUSH = [9.8, 9.7, 9.6, 9.5, 9.5, 9.5, 10.2]  # 09:30-09:55 flush, 10:00 ET reclaim


def _flush_bars(closes=FLUSH):
    bars = _session(TODAY, closes)
    bars[0] = _bar(TODAY, 0, closes[0], open_=10.0)
    return bars


def _vwap(bars, *, side="long", env="bullish_strong", now=None, reader=True):
    environment_at = (lambda _start: env) if reader else None
    return eng.vwap_reclaim_after_flush_events(
        bars, symbol="abc", side=side, now=now or _after(bars), tz=LOCAL, environment_at=environment_at
    )


class TestVwapReclaimAfterFlush:
    def test_fires_on_the_first_close_back_over_vwap(self):
        (event,) = _vwap(_flush_bars())
        assert event.engine == eng.SHADOW_VWAP_RECLAIM
        assert event.bar_time == datetime(2026, 8, 20, 10, 0, tzinfo=ET)
        assert event.entry == 10.2
        assert event.stop == pytest.approx(9.49)
        assert event.stop < event.level < event.entry
        assert dict(event.details)["environment"] == "bullish_strong"

    def test_the_environment_is_read_at_the_reclaim_bar(self):
        seen = []
        eng.vwap_reclaim_after_flush_events(
            _flush_bars(), symbol="abc", side="long", now=_after(_flush_bars()), tz=LOCAL,
            environment_at=lambda start: seen.append(start) or "bullish_strong",
        )
        assert seen == [datetime(2026, 8, 20, 10, 0, tzinfo=ET)]

    @pytest.mark.parametrize("env", ["bullish_weak", "neutral_chop", "", None, "unknown"])
    def test_any_other_environment_is_no_event(self, env):
        assert _vwap(_flush_bars(), env=env) == ()

    def test_no_environment_reader_is_no_event(self):
        assert _vwap(_flush_bars(), reader=False) == ()

    def test_longs_only(self):
        assert _vwap(_flush_bars(), side="short") == ()

    def test_no_flush_is_no_event(self):
        assert _vwap(_flush_bars([10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7])) == ()

    def test_a_missing_first_thirty_bar_is_unknown(self):
        bars = _flush_bars()
        del bars[2]
        assert _vwap(bars) == ()

    def test_the_reclaim_bar_must_be_completed(self):
        bars = _flush_bars()
        assert _vwap(bars, now=_forming(bars)) == ()

    def test_missing_volume_is_unknown(self):
        bars = _flush_bars()
        bars[1]["volume"] = None
        assert _vwap(bars) == ()


# ----------------------------------------------------------------------- (c) compression break
def _squeeze_bars(break_close=10.6, *, trend_box=False, extra=()):
    """20 wide trending bars (ATR ~0.6), a 12-bar box, then the break bar and ``extra``."""
    bars = [_bar(TODAY, i, 8.0 + 0.2 * i, high=8.3 + 0.2 * i, low=7.7 + 0.2 * i) for i in range(20)]
    box = [10.0 + 0.5 * i for i in range(12)] if trend_box else [10.5] * 12
    for offset, close in enumerate([*box, break_close, *extra]):
        bars.append(_bar(TODAY, 20 + offset, close))
    return bars


def _squeeze(bars, side="long", now=None):
    return eng.compression_break_events(bars, symbol="abc", side=side, now=now or _after(bars), tz=LOCAL)


class TestCompressionBreak:
    def test_fires_on_the_close_over_the_box(self):
        (event,) = _squeeze(_squeeze_bars())
        assert event.engine == eng.SHADOW_COMPRESSION_BREAK
        assert event.bar_time == datetime(2026, 8, 20, 12, 10, tzinfo=ET)
        assert (event.level, event.entry, event.stop) == (10.51, 10.6, 10.49)
        assert dict(event.details)["range_atr"] <= eng.SQUEEZE_RANGE_ATR

    def test_short_mirrors_through_the_box_low(self):
        (event,) = _squeeze(_squeeze_bars(10.4), side="short")
        assert (event.side, event.level, event.entry, event.stop) == ("short", 10.49, 10.4, 10.51)

    def test_no_break_is_no_event(self):
        assert _squeeze(_squeeze_bars(10.5)) == ()

    def test_a_wide_box_is_not_a_squeeze(self):
        assert _squeeze(_squeeze_bars(20.0, trend_box=True)) == ()

    def test_one_event_per_box(self):
        assert len(_squeeze(_squeeze_bars(10.6, extra=(10.7, 10.8)))) == 1

    def test_the_break_bar_must_be_completed(self):
        bars = _squeeze_bars()
        assert _squeeze(bars, now=_forming(bars)) == ()

    def test_the_threshold_is_the_s6_squeeze(self):
        import m5_setup_key_stamp
        import setup_permutations

        assert eng.SQUEEZE_RANGE_ATR == setup_permutations.M5_SQUEEZE_RANGE_ATR
        assert eng.SQUEEZE_BOX_BARS == m5_setup_key_stamp.M5_COMPRESSION_BARS
        assert eng.SQUEEZE_ATR_BARS == m5_setup_key_stamp.M5_COMPRESSION_ATR_BARS


# ----------------------------------------------------------------------- (d) trendline break
TL_HIGHS = [10.0, 10.5, 11.0, 10.5, 10.0, 10.2, 10.5, 10.2, 9.9, 9.8, 9.8, 10.5, 10.6]
TL_CLOSES = [high - 0.1 for high in TL_HIGHS]
TL_CLOSES[9] = TL_CLOSES[10] = 9.7
TL_CLOSES[11] = 10.4
TL_CLOSES[12] = 10.5


def _trend_bars(*, mirror=False):
    bars = []
    for index, (high, close) in enumerate(zip(TL_HIGHS, TL_CLOSES, strict=True)):
        low = close - 0.2
        if mirror:
            high, low, close = 20.0 - low, 20.0 - high, 20.0 - close
        bars.append(_bar(TODAY, index, close, open_=close, high=high, low=low))
    return bars


def _trend(bars, side="long", now=None):
    return eng.trendline_break_events(bars, symbol="abc", side=side, now=now or _after(bars), tz=LOCAL)


class TestTrendlineBreak:
    def test_fires_once_on_the_close_over_the_falling_line(self):
        (event,) = _trend(_trend_bars())
        assert event.engine == eng.SHADOW_TRENDLINE_BREAK
        assert event.bar_time == datetime(2026, 8, 20, 10, 25, tzinfo=ET)
        assert event.level == pytest.approx(9.875)
        assert event.entry == pytest.approx(10.4)
        assert event.stop == pytest.approx(9.5)
        details = dict(event.details)
        assert (details["pivot_1_price"], details["pivot_2_price"]) == (11.0, 10.5)
        assert details["pivot_2"] == "2026-08-20T10:00:00-04:00"

    def test_short_mirrors_on_a_rising_line_through_pivot_lows(self):
        (event,) = _trend(_trend_bars(mirror=True), side="short")
        assert event.side == "short"
        assert event.level == pytest.approx(10.125)
        assert event.entry == pytest.approx(9.6)
        assert event.stop > event.entry

    def test_a_pivot_is_used_only_after_its_confirming_bars_complete(self):
        # Cut the tape at bar 10: pivot 6 is known from bar 9 on, and nothing has broken yet.
        bars = _trend_bars()[:11]
        assert _trend(bars) == ()

    def test_the_break_bar_must_be_completed(self):
        bars = _trend_bars()[:12]
        assert _trend(bars, now=_forming(bars)) == ()
        assert len(_trend(bars)) == 1

    def test_an_unreadable_bar_is_no_event(self):
        bars = _trend_bars()
        bars[4]["high"] = "n/a"
        assert _trend(bars) == ()


# ----------------------------------------------------------------------- all four together
class TestShadowSetupEvents:
    def test_runs_all_four_both_sides_in_bar_order(self):
        events = eng.shadow_setup_events(
            _pd_history() + _pd_today(LONG_BREAK), symbol="abc", now=_after(_pd_today(LONG_BREAK)),
            tz=LOCAL, environment_at=lambda _s: "bullish_strong",
        )
        assert [event.engine for event in events] == [eng.SHADOW_PD_BREAK_HOLD]
        assert eng.SHADOW_ENGINES == (
            "pd_level_break_hold", "vwap_reclaim_after_flush", "m5_compression_break", "trendline_break",
        )

    def test_bars_outside_the_regular_session_are_ignored(self):
        premarket = [_bar(TODAY, -12 + i, 10.0) for i in range(12)]  # 05:30-06:25 local

        def strip(events):  # bar_index counts completed bars, pre-market included
            return [(e.event_id, e.level, e.entry, e.stop, e.details) for e in events]

        assert strip(_trend(premarket + _trend_bars())) == strip(_trend(_trend_bars()))
