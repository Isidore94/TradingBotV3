"""The close scan must not publish on yesterday's daily bar.

WS-FC1 keeps today's forming bar out of the daily-bar cache, so the 12:45 PT
pre-close preview writes caches that end on YESTERDAY. The 13:00/13:05 PT close
scan then found a cache touched under 30 minutes ago and took the cache-hit
path: on 2026-09-17 the 13:00 run published on 09-16 bars
(``{'cache': 766}``), on 09-18 the 13:05 run on 09-17 (``{'cache': 1090}``).
The file mtime was standing in for a content check.

A cache hit now also requires the cache to hold the LATEST COMPLETED session
(``daily_bar_cache.cache_holds_latest_completed_session``). Before the close
that is the previous session, so intraday behaviour is unchanged.

Clock, cache loader and live fetch are all monkeypatched: no network, no live
store (conftest points LOCALAPPDATA and the data dir at scratch folders, and the
cache writer is stubbed here as well).
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import daily_bar_cache, legacy  # noqa: E402

ET = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")
SYMBOL = "TEST"


def _bars(last_day: date, sessions: int = 420) -> pd.DataFrame:
    days = pd.bdate_range(end=pd.Timestamp(last_day), periods=sessions)
    n = len(days)
    return pd.DataFrame(
        {
            "datetime": days,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1_000_000] * n,
        }
    )


@pytest.fixture
def desk(monkeypatch):
    """Freeze the desk (Pacific wall clock) and stub every I/O seam."""

    state: dict = {"live_calls": [], "writes": []}
    legacy._DAILY_BAR_CACHE_TOUCHED_AT.clear()
    legacy._DAILY_BAR_LIVE_FAILURE_AT.clear()

    def _setup(now_pacific: datetime, cache_last: date, live_last: date | None):
        assert now_pacific.tzinfo is not None
        naive_local = now_pacific.astimezone(PACIFIC).replace(tzinfo=None)

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):  # noqa: D401 - clock stub
                if tz is None:
                    return naive_local
                return now_pacific.astimezone(tz)

        monkeypatch.setattr(legacy, "datetime", _FrozenDatetime)
        monkeypatch.setattr(daily_bar_cache, "market_now", lambda: now_pacific)

        cached = _bars(cache_last)
        monkeypatch.setattr(legacy, "_load_cached_daily_bar_frame", lambda symbol: cached.copy())
        # Touched five minutes ago - the pre-close preview just wrote it.
        legacy._DAILY_BAR_CACHE_TOUCHED_AT[SYMBOL] = naive_local - timedelta(minutes=5)

        def _live(ib, symbol, days):
            state["live_calls"].append((symbol, days))
            if live_last is None:
                return legacy._empty_daily_bar_frame(source=legacy.DAILY_BAR_SOURCE_YAHOO)
            return legacy._set_daily_bar_source(_bars(live_last, sessions=10), legacy.DAILY_BAR_SOURCE_YAHOO)

        monkeypatch.setattr(legacy, "_fetch_live_daily_bars", _live)
        monkeypatch.setattr(
            legacy, "_write_cached_daily_bar_frame", lambda symbol, df: state["writes"].append(symbol)
        )
        monkeypatch.setattr(legacy, "_persist_durable_daily_bars", lambda *a, **k: None)
        return state

    yield _setup
    legacy._DAILY_BAR_CACHE_TOUCHED_AT.clear()
    legacy._DAILY_BAR_LIVE_FAILURE_AT.clear()


def _last(frame: pd.DataFrame) -> date:
    return pd.Timestamp(frame["datetime"].iloc[-1]).date()


# ---------------------------------------------------------------------------
# fetch_daily_bars
# ---------------------------------------------------------------------------
def test_after_close_a_fresh_touched_cache_ending_yesterday_refreshes_live(desk):
    # Thu 2026-09-17, 13:05 Pacific = 16:05 ET: today's session is complete.
    state = desk(datetime(2026, 9, 17, 13, 5, tzinfo=PACIFIC), date(2026, 9, 16), date(2026, 9, 17))
    frame = legacy.fetch_daily_bars(None, SYMBOL, 300)
    assert state["live_calls"], "the close scan took the cache hit on yesterday's bar"
    assert _last(frame) == date(2026, 9, 17)
    assert legacy._get_daily_bar_source(frame) == legacy.DAILY_BAR_SOURCE_YAHOO


def test_after_close_a_cache_holding_today_is_a_hit(desk):
    state = desk(datetime(2026, 9, 17, 13, 5, tzinfo=PACIFIC), date(2026, 9, 17), date(2026, 9, 17))
    frame = legacy.fetch_daily_bars(None, SYMBOL, 300)
    assert state["live_calls"] == []
    assert _last(frame) == date(2026, 9, 17)
    assert legacy._get_daily_bar_source(frame) == legacy.DAILY_BAR_SOURCE_CACHE


def test_intraday_a_cache_ending_yesterday_is_still_a_hit(desk):
    # 10:00 ET = 07:00 Pacific; the latest completed session is yesterday.
    state = desk(datetime(2026, 9, 17, 10, 0, tzinfo=ET).astimezone(PACIFIC), date(2026, 9, 16), None)
    frame = legacy.fetch_daily_bars(None, SYMBOL, 300)
    assert state["live_calls"] == []
    assert _last(frame) == date(2026, 9, 16)
    assert legacy._get_daily_bar_source(frame) == legacy.DAILY_BAR_SOURCE_CACHE


def test_monday_premarket_a_cache_ending_friday_is_a_hit(desk):
    state = desk(datetime(2026, 9, 21, 5, 30, tzinfo=PACIFIC), date(2026, 9, 18), None)
    frame = legacy.fetch_daily_bars(None, SYMBOL, 300)
    assert state["live_calls"] == []
    assert _last(frame) == date(2026, 9, 18)


def test_on_a_holiday_after_the_close_a_cache_ending_the_prior_session_is_a_hit(desk):
    # Thanksgiving 2026-11-26: no session, so Wednesday is still the latest.
    state = desk(datetime(2026, 11, 26, 13, 5, tzinfo=PACIFIC), date(2026, 11, 25), None)
    frame = legacy.fetch_daily_bars(None, SYMBOL, 300)
    assert state["live_calls"] == []
    assert _last(frame) == date(2026, 11, 25)


def test_early_close_day_after_the_scan_close_refreshes_live(desk):
    # Fri 2026-11-27 closes at 13:00 ET. The desk's close scan (13:05 PT =
    # 16:05 ET) must not reuse Wednesday's bar.
    state = desk(datetime(2026, 11, 27, 13, 5, tzinfo=PACIFIC), date(2026, 11, 25), date(2026, 11, 27))
    frame = legacy.fetch_daily_bars(None, SYMBOL, 300)
    assert state["live_calls"]
    assert _last(frame) == date(2026, 11, 27)


def test_early_close_day_between_1300_and_1600_et_follows_the_writer_rule(desk):
    # market_calendar judges every session against 16:00 ET, and the WS-FC1
    # writer refuses today's bar until then. Demanding it here would refetch
    # on every scan without ever caching it, so freshness follows the same rule.
    state = desk(datetime(2026, 11, 27, 14, 0, tzinfo=ET).astimezone(PACIFIC), date(2026, 11, 25), None)
    legacy.fetch_daily_bars(None, SYMBOL, 300)
    assert state["live_calls"] == []


def test_after_close_a_failed_live_refresh_falls_back_to_the_cache(desk):
    state = desk(datetime(2026, 9, 17, 13, 5, tzinfo=PACIFIC), date(2026, 9, 16), None)
    frame = legacy.fetch_daily_bars(None, SYMBOL, 300)
    assert state["live_calls"]
    assert _last(frame) == date(2026, 9, 16)
    assert legacy._get_daily_bar_source(frame) == legacy.DAILY_BAR_SOURCE_CACHE
    assert state["writes"] == []


# ---------------------------------------------------------------------------
# the pure helper, and its timezone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("moment", "last_bar", "expected"),
    [
        # 12:55 Pacific = 15:55 ET: today not complete, yesterday is enough.
        (datetime(2026, 9, 17, 12, 55, tzinfo=PACIFIC), date(2026, 9, 16), True),
        # 13:00 Pacific = 16:00 ET: today is complete.
        (datetime(2026, 9, 17, 13, 0, tzinfo=PACIFIC), date(2026, 9, 16), False),
        (datetime(2026, 9, 17, 13, 0, tzinfo=PACIFIC), date(2026, 9, 17), True),
        # 21:04 Pacific Friday is 00:04 ET Saturday: Friday is the latest.
        (datetime(2026, 9, 18, 21, 4, tzinfo=PACIFIC), date(2026, 9, 18), True),
        (datetime(2026, 9, 18, 21, 4, tzinfo=PACIFIC), date(2026, 9, 17), False),
        # Unknown last bar is never "fresh".
        (datetime(2026, 9, 17, 10, 0, tzinfo=PACIFIC), None, False),
    ],
)
def test_helper_judges_the_session_in_exchange_time(moment, last_bar, expected):
    assert daily_bar_cache.cache_holds_latest_completed_session(last_bar, now=moment) is expected


def test_helper_reads_the_desk_clock_when_now_is_omitted(monkeypatch):
    monkeypatch.setattr(daily_bar_cache, "market_now", lambda: datetime(2026, 9, 17, 13, 5, tzinfo=PACIFIC))
    assert daily_bar_cache.cache_holds_latest_completed_session(date(2026, 9, 16)) is False
    assert daily_bar_cache.cache_holds_latest_completed_session(date(2026, 9, 17)) is True
