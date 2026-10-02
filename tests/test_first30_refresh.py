"""First-30 hold: the 10:00 check asks the chart refresh service for fresh bars.

`latest_bars` is rewritten only when the ~28-minute scan reaches a symbol, and
Focus D1 names outside the scan set have none, so at 10:00 most held charts
would read as "no data". The hold asks `ChartBarRefreshService.refresh_now`
(display-only, never `latest_bars`) for its few symbols and re-judges when
`barsRefreshed` lands.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for _path in (SCRIPTS_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_first30_chart_hold import (  # noqa: E402,F401
    ET,
    _at,
    _bar,
    _focus_d1,
    _release,
    _review_symbols,
    env,
    panel,
)


class _FakeRefresh:
    def __init__(self):
        self.bars: dict[str, list] = {}
        self.requests: list[list[str]] = []

    def bars_for(self, symbol):
        return list(self.bars.get(symbol, []))

    def refresh_now(self, symbols, bot, *, now=None):
        self.requests.append(list(symbols))
        return list(symbols)


@pytest.fixture
def refresh(panel, monkeypatch):  # noqa: F811
    fake = _FakeRefresh()
    monkeypatch.setattr(panel, "_first30_refresh_service", lambda: fake, raising=False)
    monkeypatch.setattr(panel, "_current_bot", lambda: object())
    return fake


def test_a_stale_cache_is_refetched_and_judged_on_the_fresh_0955_bar(panel, refresh):  # noqa: F811
    panel.add_alert(_focus_d1("HBM", _at(9, 40), level=10.0))
    # The bot cache stops at the 09:50 bar (last scan).
    panel.test_bars["HBM"] = [_bar(9, 45, 9.0), _bar(9, 50, 9.5)]
    _release(panel)
    assert refresh.requests == [["HBM"]]
    assert ("HBM", "LONG") in panel._first30_held
    # The display refetch lands with the finished 09:55 bar and the forming 10:00 one.
    refresh.bars["HBM"] = [_bar(9, 50, 9.5), _bar(9, 55, 10.4), _bar(10, 0, 9.8)]
    panel.test_clock["now"] = _at(10, 0, 3)
    panel._on_bars_refreshed("HBM")
    assert "HBM" in _review_symbols(panel)
    assert not panel._first30_held


def test_a_fresh_refetch_below_the_level_fails(panel, refresh):  # noqa: F811
    panel.add_alert(_focus_d1("LOW", _at(9, 40), level=10.0))
    _release(panel)
    refresh.bars["LOW"] = [_bar(9, 55, 9.9), _bar(10, 0, 9.8)]
    panel._on_bars_refreshed("LOW")
    assert panel._first30_failed[("LOW", "LONG")][1] == "failed"


def test_no_fresh_bars_after_five_minutes_is_no_data(panel, refresh):  # noqa: F811
    panel.add_alert(_focus_d1("NOD", _at(9, 40)))
    _release(panel, 10, 4, 30)
    assert ("NOD", "LONG") in panel._first30_held  # still inside the 5-minute grace
    _release(panel, 10, 5, 1)
    assert panel._first30_failed[("NOD", "LONG")][1] == "no_data"


# --------------------------------------------------------------------- service
class _Bot:
    def __init__(self, fetched):
        self._fetched = fetched
        self.fetch_calls: list[str] = []
        self.latest_bars = {"X": "detector-facing sentinel"}

    def fetch_m5_chart_bars(self, symbol, max_sessions=2):
        self.fetch_calls.append(symbol)
        return list(self._fetched.get(symbol) or [])


def _drain(service):
    if service._thread is not None:
        service._thread.join(timeout=5.0)


def test_refresh_now_ignores_staleness_and_the_queue_cooldown():
    from ui.services.chart_bar_refresh import ChartBarRefreshService

    now = datetime(2026, 10, 2, 10, 0, 5)
    fresh = [{"dt": datetime(2026, 10, 2, 9, 55), "close": 1.0}]
    bot = _Bot({"AAA": fresh})
    service = ChartBarRefreshService()
    # A queue refresh at 09:57 put AAA in the 5-minute cooldown.
    service._attempted_at["AAA"] = now - timedelta(minutes=3)
    assert service.refresh_now(["AAA", "AAA"], bot, now=now) == ["AAA"]
    _drain(service)
    assert bot.fetch_calls == ["AAA"]
    assert service.bars_for("AAA") == fresh
    assert bot.latest_bars == {"X": "detector-facing sentinel"}
    # The same symbol is not asked again within the forced gap.
    assert service.refresh_now(["AAA"], bot, now=now + timedelta(seconds=30)) == []
    assert service.refresh_now(["AAA"], bot, now=now + timedelta(seconds=95)) == ["AAA"]
    _drain(service)


def test_refresh_now_with_a_busy_worker_queues_and_marks_nothing():
    from ui.services.chart_bar_refresh import ChartBarRefreshService

    now = datetime(2026, 10, 2, 10, 0, 5)
    gate = threading.Event()
    bot = _Bot({})

    def slow(symbol, max_sessions=2):
        gate.wait(timeout=5.0)
        return []

    bot.fetch_m5_chart_bars = slow
    service = ChartBarRefreshService()
    try:
        assert service.refresh_now(["AAA"], bot, now=now) == ["AAA"]
        assert service.refresh_now(["BBB"], bot, now=now) == []
    finally:
        gate.set()
        _drain(service)
    # BBB was not marked, so the next ask goes through at once.
    assert service.refresh_now(["BBB"], bot, now=now) == ["BBB"]
    _drain(service)
