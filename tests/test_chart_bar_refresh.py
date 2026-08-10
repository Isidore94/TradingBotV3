"""Stale queued alerts get current bars, without spending the scanner's budget.

The defect these lock in: ``bot.latest_bars`` is only rewritten when the scan
loop reaches a symbol (~28 min), so an alert opened twenty minutes after it
fired charted its scan-time bars - and because the rebuilt series was
identical, the chart's repaint guard correctly drew nothing new.

The two boundaries that matter more than the feature: a chart refresh must
never write the detector-facing cache, and it must never outgrow the IB
historical budget the champion scan depends on.
"""

import os
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.services.chart_bar_refresh import (  # noqa: E402
    DEFAULT_LOOKAHEAD,
    REFRESH_COOLDOWN,
    STALE_AFTER,
    ChartBarRefreshService,
    bars_age,
    bars_are_stale,
)


def _bars(last_dt, count=5):
    return [
        {
            "dt": last_dt - timedelta(minutes=5 * offset),
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.2,
            "volume": 1000.0,
        }
        for offset in reversed(range(count))
    ]


class _Bot:
    """Just enough bot: a frozen cache and a counted fetch."""

    def __init__(self, cached=None, fetched=None):
        self.latest_bars = {"AAA": "detector-facing sentinel"}
        self._cached = cached or {}
        self._fetched = fetched or {}
        self.fetch_calls = []

    def m5_chart_bars(self, symbol, max_sessions=2):
        return list(self._cached.get(symbol) or [])

    def fetch_m5_chart_bars(self, symbol, max_sessions=2):
        self.fetch_calls.append(symbol)
        return list(self._fetched.get(symbol) or [])


def _drain(service, timeout=5.0):
    thread = service._thread
    if thread is not None:
        thread.join(timeout=timeout)


def test_bar_age_discounts_the_bar_length_itself():
    """The stamp is the bar's START, so a just-closed bar is not 5 min stale."""
    now = datetime(2026, 8, 10, 10, 30)
    just_closed = bars_age(_bars(now - timedelta(minutes=5)), now=now)
    assert just_closed == timedelta(0)
    assert not bars_are_stale(_bars(now - timedelta(minutes=5)), now=now)


def test_three_bars_behind_is_stale_and_two_is_not():
    now = datetime(2026, 8, 10, 10, 30)
    two_behind = _bars(now - timedelta(minutes=5 * 3))
    three_behind = _bars(now - timedelta(minutes=5 * 4))
    assert not bars_are_stale(two_behind, now=now)
    assert bars_are_stale(three_behind, now=now)
    assert STALE_AFTER == timedelta(minutes=15)


def test_empty_bars_are_not_stale():
    """Nothing to refresh from a cache; the chart says so in its own words."""
    assert bars_age([]) is None
    assert not bars_are_stale([])


def test_a_stale_symbol_is_refetched_and_the_fresh_bars_win():
    now = datetime(2026, 8, 10, 10, 30)
    stale = _bars(now - timedelta(minutes=40))
    fresh = _bars(now - timedelta(minutes=5))
    bot = _Bot(cached={"NVDA": stale}, fetched={"NVDA": fresh})
    service = ChartBarRefreshService()

    queued = service.refresh_if_stale(["NVDA"], bot.m5_chart_bars, bot, now=now)
    _drain(service)

    assert queued == ["NVDA"]
    assert bot.fetch_calls == ["NVDA"]
    assert service.best_bars("NVDA", stale) == fresh


def test_a_current_symbol_is_never_refetched():
    """The IB budget is the champion scan's; fresh charts must not spend it."""
    now = datetime(2026, 8, 10, 10, 30)
    fresh = _bars(now - timedelta(minutes=5))
    bot = _Bot(cached={"NVDA": fresh}, fetched={"NVDA": fresh})
    service = ChartBarRefreshService()

    queued = service.refresh_if_stale(["NVDA"], bot.m5_chart_bars, bot, now=now)
    _drain(service)

    assert queued == []
    assert bot.fetch_calls == []


def test_the_refresh_never_writes_the_detector_facing_cache():
    """plan.md sec 5: a chart view must not change what the champions see."""
    now = datetime(2026, 8, 10, 10, 30)
    stale = _bars(now - timedelta(minutes=40))
    bot = _Bot(cached={"NVDA": stale}, fetched={"NVDA": _bars(now)})
    service = ChartBarRefreshService()

    service.refresh_if_stale(["NVDA"], bot.m5_chart_bars, bot, now=now)
    _drain(service)

    assert bot.latest_bars == {"AAA": "detector-facing sentinel"}


def test_a_symbol_in_cooldown_is_not_refetched_again():
    """A halted name must not be refetched every tick forever."""
    now = datetime(2026, 8, 10, 10, 30)
    stale = _bars(now - timedelta(minutes=40))
    bot = _Bot(cached={"NVDA": stale}, fetched={"NVDA": stale})
    service = ChartBarRefreshService()

    service.refresh_if_stale(["NVDA"], bot.m5_chart_bars, bot, now=now)
    _drain(service)
    again = service.refresh_if_stale(
        ["NVDA"], bot.m5_chart_bars, bot, now=now + timedelta(minutes=1)
    )
    _drain(service)
    later = service.refresh_if_stale(
        ["NVDA"], bot.m5_chart_bars, bot, now=now + REFRESH_COOLDOWN
    )
    _drain(service)

    assert again == []
    assert later == ["NVDA"]


def test_a_shorter_refetch_never_replaces_a_longer_cached_series():
    """A partial provider answer must not truncate the chart."""
    now = datetime(2026, 8, 10, 10, 30)
    long_cached = _bars(now - timedelta(minutes=5), count=60)
    service = ChartBarRefreshService()
    with service._lock:
        service._bars["NVDA"] = _bars(now - timedelta(minutes=90), count=2)
    assert service.best_bars("NVDA", long_cached) == long_cached


def test_a_newer_but_truncated_refetch_loses_to_the_longer_series():
    """Three current candles are not worth the history the trader was reading."""
    now = datetime(2026, 8, 10, 10, 30)
    long_cached = _bars(now - timedelta(minutes=30), count=60)
    service = ChartBarRefreshService()
    with service._lock:
        service._bars["NVDA"] = _bars(now, count=2)
    assert service.best_bars("NVDA", long_cached) == long_cached


def test_a_symbol_outside_the_scan_set_is_refetched():
    """An empty cache is the one case where refetching is the ONLY way to chart."""
    now = datetime(2026, 8, 10, 10, 30)
    bot = _Bot(cached={"NVDA": []}, fetched={"NVDA": _bars(now)})
    service = ChartBarRefreshService()

    queued = service.refresh_if_stale(["NVDA"], bot.m5_chart_bars, bot, now=now)
    _drain(service)

    assert queued == ["NVDA"]


def test_only_one_worker_runs_at_a_time():
    """The fetch blocks and IB serialises anyway; a second thread just queues."""
    now = datetime(2026, 8, 10, 10, 30)
    stale = _bars(now - timedelta(minutes=40))
    release = threading.Event()
    bot = _Bot(cached={"AAA": stale, "BBB": stale}, fetched={})

    def slow_fetch(symbol, max_sessions=2):
        bot.fetch_calls.append(symbol)
        release.wait(timeout=5.0)
        return []

    bot.fetch_m5_chart_bars = slow_fetch
    service = ChartBarRefreshService()
    try:
        first = service.refresh_if_stale(["AAA"], bot.m5_chart_bars, bot, now=now)
        second = service.refresh_if_stale(["BBB"], bot.m5_chart_bars, bot, now=now)
        assert first == ["AAA"]
        assert second == []
    finally:
        release.set()
        _drain(service)


def test_the_lookahead_stays_small_enough_for_the_ib_budget():
    """IB allows ~60 historical requests per 10 min and the scan needs them.

    Displayed + lookahead, each behind a 5-minute cooldown, is the ceiling.
    """
    per_ten_minutes = (1 + DEFAULT_LOOKAHEAD) * (
        timedelta(minutes=10) / REFRESH_COOLDOWN
    )
    assert per_ten_minutes <= 12, (
        f"chart refresh would take {per_ten_minutes:.0f} of the ~60 request budget"
    )


def test_a_fetch_failure_costs_nothing_but_the_refresh():
    now = datetime(2026, 8, 10, 10, 30)
    stale = _bars(now - timedelta(minutes=40))
    bot = _Bot(cached={"NVDA": stale})

    def boom(symbol, max_sessions=2):
        raise RuntimeError("provider down")

    bot.fetch_m5_chart_bars = boom
    service = ChartBarRefreshService()
    service.refresh_if_stale(["NVDA"], bot.m5_chart_bars, bot, now=now)
    _drain(service)
    assert service.bars_for("NVDA") == []
    assert service.best_bars("NVDA", stale) == stale


def test_no_bot_means_no_work():
    service = ChartBarRefreshService()
    assert service.refresh_if_stale(["NVDA"], lambda _s: [], None) == []
