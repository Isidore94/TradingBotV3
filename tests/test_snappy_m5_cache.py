"""The Qt thread never waits on the scanner child for M5 bars.

2026-09-23 stall log: ~30 min of GUI blocking, most of it `_rpc` to the
BounceBot child from Alert Center polls, the armed list, the snapshot chart
and the Auto Pilot tick. These pin:

* a poll / armed-list redraw / snapshot read on a PROXY bot makes no RPC on
  the Qt thread and returns fast even when an RPC takes 2 s;
* bars not fetched yet are UNKNOWN: a watch neither triggers nor clears;
* given the same bars a poll decides exactly what it decided before (parity
  between an in-process bot and the proxy path);
* one RPC per (symbol, sessions), not two.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

NOON = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
GUI_BUDGET_S = 0.25


def _bar(minute, high, low, hour=11):
    mid = (high + low) / 2
    return {
        "dt": NOON.replace(hour=hour, minute=minute),
        "open": mid,
        "high": high,
        "low": low,
        "close": mid,
        "volume": 1000.0,
    }


BASE_BARS = [_bar(20, 110.0, 99.0), _bar(25, 108.0, 100.0)]
BREAK_BARS = BASE_BARS + [_bar(45, 111.0, 104.0)]


class _InProcessBot:
    def __init__(self, bars):
        self.bars = list(bars)

    def m5_chart_bars(self, symbol, max_sessions=2):
        return list(self.bars)


class _SlowProxy:
    """A process proxy: every read is an RPC that takes `delay` seconds."""

    is_process_proxy = True

    def __init__(self, bars, delay=2.0):
        self.bars = list(bars)
        self.delay = delay
        self.calls: list[tuple[str, int, threading.Thread]] = []
        self.d1_zone_arms = {}

    def m5_chart_bars(self, symbol, max_sessions=2):
        self.calls.append((symbol, max_sessions, threading.current_thread()))
        time.sleep(self.delay)
        return list(self.bars)

    def get_market_environment(self):
        time.sleep(self.delay)
        return "bullish"

    def rpcs_on(self, thread):
        return [call for call in self.calls if call[2] is thread]


class _Service:
    def __init__(self, bot):
        self._bot = bot

    def current_bot(self):
        return self._bot


@pytest.fixture(autouse=True)
def _fresh_cache():
    from ui.services import m5_bar_cache

    m5_bar_cache.reset_shared_m5_cache()
    yield
    m5_bar_cache.reset_shared_m5_cache()


def _panel(tmp_path, bot):
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel(
        parked_symbols_path=tmp_path / "parked.json",
        focus_d1_flags_path=tmp_path / "focus_flags.json",
        chart_watches_path=tmp_path / "chart_watches.json",
        d1_level_watches_path=tmp_path / "d1_level_watches.json",
    )
    panel._review_movers_only = False
    panel._bounce_service = _Service(bot)
    panel._d1_bars_for = lambda _symbol: []
    return panel


def _hod_watch():
    from chart_watch import arm_chart_watch

    watch = arm_chart_watch("new_hod", "NVDA", "LONG", BASE_BARS, now=NOON.replace(hour=11, minute=30))
    return dataclasses.replace(watch, armed_at=NOON.replace(hour=11, minute=40))


def _level_watch():
    from chart_watch import D1LevelWatch

    return D1LevelWatch(
        symbol="AMD", direction="above", level=110.5, armed_at=NOON.replace(hour=11, minute=40)
    )


def _warm(bot, *keys):
    from ui.services.m5_bar_cache import shared_m5_cache

    cache = shared_m5_cache()
    for symbol, sessions in keys:
        assert cache.wait_known(bot, symbol, sessions, timeout=10.0), symbol


def test_polls_and_the_armed_list_make_no_rpc_on_the_gui_thread(tmp_path):
    bot = _SlowProxy(BREAK_BARS, delay=2.0)
    panel = _panel(tmp_path, bot)
    try:
        panel._chart_watches = [_hod_watch()]
        panel._d1_level_watches = [_level_watch()]
        main = threading.current_thread()

        started = time.perf_counter()
        panel._poll_chart_watches(now=NOON)
        panel._poll_d1_level_watches(now=NOON)
        panel._refresh_armed_list()
        panel.vwap_state("NVDA", "long")
        elapsed = time.perf_counter() - started

        assert bot.rpcs_on(main) == [], "no child RPC may run on the Qt thread"
        assert elapsed < GUI_BUDGET_S, f"GUI thread blocked {elapsed:.2f}s"
    finally:
        panel.close()
        panel.deleteLater()


def test_bars_not_fetched_yet_neither_trigger_nor_clear(tmp_path):
    bot = _SlowProxy(BREAK_BARS, delay=0.2)
    panel = _panel(tmp_path, bot)
    try:
        hod = _hod_watch()
        level = _level_watch()
        panel._chart_watches = [hod]
        panel._d1_level_watches = [level]

        panel._poll_chart_watches(now=NOON)
        panel._poll_d1_level_watches(now=NOON)
        assert panel._chart_watches == [hod], "unknown bars must not fire or clear a watch"
        assert panel._d1_level_watches == [level]
        assert not [alert for alert in panel._alerts if alert.tag == "chart_watch"]

        _warm(bot, ("NVDA", 1), ("AMD", 1))
        panel._poll_chart_watches(now=NOON)
        panel._poll_d1_level_watches(now=NOON)
        assert panel._chart_watches == [], "fetched bars that break the high fire"
        assert panel._d1_level_watches == []
    finally:
        panel.close()
        panel.deleteLater()


def _fired_triggers(panel):
    return sorted(
        (alert.symbol, alert.trigger) for alert in panel._alerts if alert.tag == "chart_watch"
    )


@pytest.mark.parametrize("bars", [BASE_BARS, BREAK_BARS], ids=["quiet", "break"])
def test_a_poll_decides_the_same_on_the_same_bars(tmp_path, bars):
    """Trigger parity: in-process bot (old path, unchanged) vs the proxy cache."""
    local = _panel(tmp_path / "local", _InProcessBot(bars))
    proxy_bot = _SlowProxy(bars, delay=0.0)
    proxy = _panel(tmp_path / "proxy", proxy_bot)
    try:
        for panel in (local, proxy):
            panel._chart_watches = [_hod_watch()]
            panel._d1_level_watches = [_level_watch()]
        _warm(proxy_bot, ("NVDA", 1), ("AMD", 1))
        for panel in (local, proxy):
            panel._poll_chart_watches(now=NOON)
            panel._poll_d1_level_watches(now=NOON)

        assert [w.symbol for w in proxy._chart_watches] == [w.symbol for w in local._chart_watches]
        assert [w.symbol for w in proxy._d1_level_watches] == [
            w.symbol for w in local._d1_level_watches
        ]
        assert _fired_triggers(proxy) == _fired_triggers(local)
    finally:
        for panel in (local, proxy):
            panel.close()
            panel.deleteLater()


def test_one_rpc_per_symbol_and_session_count(tmp_path):
    bot = _SlowProxy(BASE_BARS, delay=0.0)
    panel = _panel(tmp_path, bot)
    try:
        panel._m5_bars_for("NVDA")
        _warm(bot, ("NVDA", 1))
        for _ in range(5):
            assert panel._m5_bars_for("NVDA") == BASE_BARS
        assert [(symbol, sessions) for symbol, sessions, _t in bot.calls] == [("NVDA", 1)]
    finally:
        panel.close()
        panel.deleteLater()


def test_the_snapshot_chart_reads_proxy_bars_without_an_rpc(tmp_path):
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    bot = _SlowProxy(BASE_BARS, delay=2.0)
    widget = SymbolSnapshotWidget()
    try:
        widget._symbol = "NVDA"
        widget._bot = bot
        main = threading.current_thread()
        started = time.perf_counter()
        widget._read_m5_bars()
        elapsed = time.perf_counter() - started
        assert bot.rpcs_on(main) == []
        assert elapsed < GUI_BUDGET_S, f"GUI thread blocked {elapsed:.2f}s"
    finally:
        widget.close()
        widget.deleteLater()


def test_the_auto_pilot_near_extreme_check_does_not_rpc_on_the_tick(monkeypatch):
    from ui.services import autopilot_service

    monkeypatch.setattr(autopilot_service, "is_within_regular_market_session", lambda: True)
    bot = _SlowProxy(BASE_BARS, delay=2.0)
    bot._spy_session_bars = lambda: (time.sleep(2.0), ([], None))[1]

    class _Host:
        _hod_check_running = False
        _enabled = True
        _state = {"date": NOON.date().isoformat()}

        def _shadow_research_allowed(self):
            return False

        def _current_bot(self):
            return bot

        def _save_state(self):
            pass

        def _log(self, _line):
            pass

    host = _Host()
    started = time.perf_counter()
    autopilot_service.AutopilotService._maybe_add_near_extreme_names(host, NOON)
    elapsed = time.perf_counter() - started
    assert elapsed < GUI_BUDGET_S, f"Auto Pilot tick blocked {elapsed:.2f}s"
    deadline = time.monotonic() + 10
    while host._hod_check_running and time.monotonic() < deadline:
        time.sleep(0.05)
    assert host._hod_check_running is False
