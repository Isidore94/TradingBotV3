"""Review advisories on the snappy-arm branch (2026-09-23).

1. A cancel that lands between the worker's CANCELLED check and its RUNNING
   write must still win: a cancelled arm never commits.
2. App close shuts the shared M5 cache down before the bot stops.
3. The M5 cache's child load is bounded: one RPC per key per 5-minute bar,
   and the H1 leg asks for bars only for a watch that is due.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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


def _pump(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


def test_a_cancel_racing_the_worker_start_still_wins():
    from ui.services.arm_queue import ArmJob, ArmQueue

    queue = ArmQueue()
    committed: list[str] = []

    class _RacyJob(ArmJob):
        """The first time the WORKER reads `state`, a cancel arrives from elsewhere."""

        raced = False

        def __getattribute__(self, name):
            value = object.__getattribute__(self, name)
            if (
                name == "state"
                and not object.__getattribute__(self, "raced")
                and threading.current_thread().name == "alert-arm-queue"
            ):
                # The worker has READ "queued"; the cancel lands before it writes.
                object.__setattr__(self, "raced", True)
                canceller = threading.Thread(target=queue.cancel, args=(self.key,))
                canceller.start()
                canceller.join(0.2)  # returns at once without a lock; blocks with one
            return value

    job = _RacyJob(
        ("watch", "AAA", "new_hod"),
        "AAA",
        "New HOD",
        lambda: time.sleep(0.3),
        lambda _result: (committed.append("AAA") or True, ""),
    )
    queue._jobs[job.key] = job
    queue._put(("arm", job))
    assert queue.wait_idle(5.0)
    _pump(0.3)
    assert committed == [], "a cancelled arm must never commit"
    queue.shutdown(timeout=1.0)


def test_app_close_stops_the_m5_cache_before_the_bot():
    from ui.panels.trading_desk import TradingDeskPanel
    from ui.services import m5_bar_cache

    order: list[str] = []
    m5_bar_cache.reset_shared_m5_cache()
    cache = m5_bar_cache.shared_m5_cache()
    real_shutdown = cache.shutdown

    def record_cache(timeout=1.0):
        order.append("m5 cache")
        real_shutdown(timeout)

    cache.shutdown = record_cache
    host = SimpleNamespace(
        alert_center=SimpleNamespace(shutdown=lambda: order.append("arms")),
        bounce_panel=SimpleNamespace(on_close=lambda: order.append("bot")),
        industry_panel=SimpleNamespace(shutdown=lambda: None),
        master_panel=SimpleNamespace(scan_service=SimpleNamespace(shutdown=lambda: None)),
    )
    try:
        TradingDeskPanel.shutdown(host)
    finally:
        m5_bar_cache.reset_shared_m5_cache()
    assert order[:3] == ["arms", "m5 cache", "bot"]


class _CountingProxy:
    is_process_proxy = True

    def __init__(self):
        self.calls = 0
        self.d1_zone_arms = {}

    def m5_chart_bars(self, symbol, max_sessions=2):
        self.calls += 1
        return [{"dt": None, "close": 1.0}]


def test_a_refresh_sweep_is_one_rpc_per_key_per_five_minute_bar():
    from ui.services.m5_bar_cache import M5BarCache

    clock = [1_000_000.0 * 300 + 60.0]  # a minute into a 5-minute bar
    cache = M5BarCache(refresh_seconds=0.2)
    cache._clock = lambda: clock[0]
    bot = _CountingProxy()
    symbols = [f"S{index:02d}" for index in range(20)]
    try:
        for symbol in symbols:
            cache.peek(bot, symbol, 1)
        for symbol in symbols:
            assert cache.wait_known(bot, symbol, 1, timeout=10.0)
        assert bot.calls == 20, "new keys fetch at once, once each"

        # Polls keep asking inside the same bar: no refetch.
        for _ in range(3):
            for symbol in symbols:
                cache.peek(bot, symbol, 1)
            time.sleep(0.5)
        assert bot.calls == 20, f"{bot.calls - 20} extra RPCs inside one 5-minute bar"

        # A new bar boundary (plus its grace) has passed: one more sweep.
        clock[0] += 300.0
        for symbol in symbols:
            cache.peek(bot, symbol, 1)
        deadline = time.monotonic() + 10.0
        while bot.calls < 40 and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(1.5)
        assert bot.calls == 40
    finally:
        cache.shutdown()


def test_the_h1_leg_asks_for_bars_only_when_a_watch_is_due(tmp_path):
    from chart_watch import TRIGGER_H1_EMA15_BOUNCE
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel(
        parked_symbols_path=tmp_path / "parked.json",
        focus_d1_flags_path=tmp_path / "focus_flags.json",
    )
    asked: list[str] = []
    try:
        panel._pullback_triggers = lambda watch: {TRIGGER_H1_EMA15_BOUNCE}
        panel._pullback_uses_h1 = lambda watch: True
        panel._h1_history_cache = lambda: None
        panel._m5_unknown = lambda symbol, sessions=1: asked.append(symbol) or True
        panel._pullback_due = lambda *_args: (False, None)
        watches = [SimpleNamespace(symbol=f"S{index}") for index in range(5)]

        assert panel._h1_watches_due(watches, None) == []
        assert asked == [], "a watch with no new H1 bucket must not touch the bar cache"

        marked: list[str] = []
        panel._pullback_due = lambda *_args: (True, "end")
        panel._mark_pullback_judged = lambda watch, *_args: marked.append(watch.symbol)
        assert panel._h1_watches_due(watches, None) == []
        assert asked == [w.symbol for w in watches]
        assert marked == [], "a due watch with unknown bars is not marked judged"
    finally:
        panel.close()
        panel.deleteLater()
