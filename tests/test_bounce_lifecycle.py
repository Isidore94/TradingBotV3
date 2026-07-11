"""Packet A (plan.md): BounceBot/BounceService start-stop lifecycle."""

import os
import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_wait_for_candle_close_aborts_immediately_on_stop_event():
    from bounce_bot_lib.legacy import wait_for_candle_close

    stop = threading.Event()
    stop.set()
    started = time.monotonic()
    wait_for_candle_close(stop)
    assert time.monotonic() - started < 2.0, "stop event must abort the candle wait"


def test_bouncebot_stop_ends_strategy_loop():
    from bounce_bot_lib.legacy import BounceBot

    bot = BounceBot(gui_callback=None, start_scanning_enabled=False)
    # Keep the paused-loop bookkeeping inert so the test never touches disk/network.
    bot._maybe_refresh_learning_after_close = lambda: None
    bot._maybe_refresh_auto_regime_while_paused = lambda: None

    thread = threading.Thread(target=bot.run_strategy, daemon=True)
    bot.strategy_thread = thread
    thread.start()
    time.sleep(0.2)
    assert thread.is_alive()

    bot.stop(timeout=5.0)
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "run_strategy must exit after stop()"
    assert bot.is_stopping()
    bot.stop(timeout=1.0)  # idempotent


def test_bounce_service_stop_stops_bot_and_all_timers():
    _qapp()
    from ui.services.bounce_service import BounceService

    service = BounceService()

    calls = {}

    class FakeBot:
        connection_status = True

        def stop(self, timeout=None):
            calls["stop"] = timeout

        def disconnect(self):
            calls["disconnect"] = True

    with service._lock:
        service._bot = FakeBot()
    service._health_timer.start()
    service._regime_timer.start()
    service._board_timer.start()

    service.stop()

    assert calls.get("stop") == 5.0, "service.stop() must call bot.stop(), not bare disconnect"
    assert not service._health_timer.isActive()
    assert not service._regime_timer.isActive()
    assert not service._board_timer.isActive(), "board timer must stop with the service"
    assert service._current_bot() is None


def test_stop_during_startup_cannot_install_a_late_bot(monkeypatch):
    _qapp()
    import bounce_bot

    from ui.services.bounce_service import BounceService

    release = threading.Event()
    calls = {}

    class FakeBot:
        connection_status = False

        def stop(self, timeout=None):
            calls["late_stop"] = True

        def disconnect(self):
            calls["late_disconnect"] = True

        # state-sync surface used by _apply_saved_state/_sync_state_from_bot
        def set_rrs_threshold(self, *_):
            pass

        def set_rrs_timeframe(self, *_):
            pass

        def set_market_environment(self, *_):
            pass

        def set_scanning_enabled(self, *_):
            pass

        def set_bounce_type_enabled(self, *_):
            pass

        def get_market_environment(self):
            return "bullish_strong"

        def is_scanning_enabled(self):
            return False

        def is_bounce_type_enabled(self, *_):
            return True

    def fake_run_bot_with_gui(callback, start_scanning_enabled=False):
        release.wait(timeout=10)
        return FakeBot()

    monkeypatch.setattr(bounce_bot, "run_bot_with_gui", fake_run_bot_with_gui)

    service = BounceService()
    service.start()
    time.sleep(0.2)
    assert service.running  # _starting is True while the worker is blocked

    service.stop()  # user stops while startup is still in flight
    release.set()  # startup then completes late
    time.sleep(0.5)

    assert calls.get("late_stop"), "the late bot must be stopped, not installed"
    assert service._current_bot() is None
