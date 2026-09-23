"""The "Working now" strip (offscreen Qt). Display only.

Seen to fail (ImportError, no `ui.widgets.live_results_strip`) before the
widget was written.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

NY = ZoneInfo("America/New_York")


@pytest.fixture(scope="module", autouse=True)
def _app():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _alert(symbol, side="LONG", entry=100.0, stop=99.0, types="vwap"):
    feedback = {
        "symbol": symbol,
        "direction": side.lower(),
        "bounce_types": types,
        "entry_price": entry,
        "stop_price": stop,
        "risk_per_share": "",
    }
    return SimpleNamespace(symbol=symbol, side=side, trigger=types, payload={"feedback": feedback})


def _bar(hh, mm, o, h, low, c):
    return {"dt": datetime(2026, 9, 22, hh, mm), "open": o, "high": h, "low": low, "close": c, "volume": 1.0}


class _Clock:
    def __init__(self, moment):
        self.moment = moment

    def __call__(self):
        return self.moment


BARS = {
    "NVDA": [_bar(10, 35, 100.0, 102.1, 99.7, 101.0), _bar(10, 40, 101.0, 101.5, 100.8, 101.4)],
    "AMD": [_bar(10, 35, 100.0, 100.2, 98.5, 99.0)],  # stopped
}


def _strip(clock=None, **kwargs):
    from ui.widgets.live_results_strip import LiveResultsStrip

    clock = clock or _Clock(datetime(2026, 9, 22, 10, 35, 20, tzinfo=NY))
    strip = LiveResultsStrip(threaded=False, clock=clock, **kwargs)
    strip.set_bars_provider(lambda symbol: BARS.get(symbol, []))
    return strip, clock


def _grades():
    return {
        "daytrade": [
            {"key": "vwap|LONG", "bounce_type": "vwap", "side": "LONG", "grade": "A", "n": 40},
            {"key": "ema|LONG", "bounce_type": "ema", "side": "LONG", "grade": "C", "n": 35},
        ]
    }


def _bounce_alert(symbol):
    """A real BounceAlert, as the Alert Center posts it."""
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="10:35:20",
        symbol=symbol,
        side="LONG",
        trigger="vwap",
        timeframe="M5",
        payload=_alert(symbol).payload,
    )


def test_record_then_refresh_shows_r_by_grade(monkeypatch):
    import live_alert_results

    monkeypatch.setattr(live_alert_results, "desk_zone", lambda: NY)
    strip, clock = _strip()
    assert strip.line_text() == "Working now · no M5 alerts yet today"
    strip.record(_alert("NVDA"))
    strip.record(_alert("NVDA", entry=110.0, stop=109.0))  # repeat: no new row
    strip.record(_alert("AMD", types="ema"))
    strip.record(_alert("TSLA", entry="", stop=""))
    assert strip.line_text() == "Working now · 3 no data"  # before any bars are read
    clock.moment = datetime(2026, 9, 22, 10, 45, 5, tzinfo=NY)
    strip.refresh()
    # no grades yet: every alert is New
    assert strip.line_text() == "Working now · NEW 2 (+0.2R) · 1 no data"
    strip.set_setup_grades(_grades())
    assert strip.line_text() == "Working now · A 1 (+1.4R) · C 1 (−1.0R) · 1 no data"
    tooltip = strip.toolTip()
    assert "NVDA LONG 10:35 [A] +1.4R now · best +2.1R · worst −0.3R" in tooltip
    assert "AMD LONG 10:35 [C] STOPPED −1.0R" in tooltip
    assert "TSLA LONG 10:35 [A] no data (no entry/stop)" in tooltip


def test_day_roll_clears_the_strip(monkeypatch):
    import live_alert_results

    monkeypatch.setattr(live_alert_results, "desk_zone", lambda: NY)
    strip, clock = _strip()
    strip.record(_alert("NVDA"))
    clock.moment = datetime(2026, 9, 22, 10, 45, 5, tzinfo=NY)
    strip.refresh()
    assert "NEW 1" in strip.line_text()
    strip.clear_day()
    assert strip.line_text() == "Working now · no M5 alerts yet today"
    assert strip.results() == []
    strip.record(_alert("NVDA"))  # the same name counts again after the roll
    assert len(strip.results()) == 1


def test_a_provider_that_raises_shows_no_data_not_an_error():
    strip, clock = _strip()

    def broken(_symbol):
        raise RuntimeError("BounceBot child is not running")

    strip.set_bars_provider(broken)
    strip.record(_alert("NVDA"))
    clock.moment = datetime(2026, 9, 22, 11, 0, tzinfo=NY)
    strip.refresh()
    assert strip.line_text() == "Working now · 1 no data"


def test_regime_pause_rows_are_not_trades():
    from ui.models.bounce import REGIME_PAUSE_TRIGGER_PREFIX

    strip, _clock = _strip()
    alert = _alert("SPY")
    alert.trigger = f"{REGIME_PAUSE_TRIGGER_PREFIX} SPY"
    strip.record(alert)
    assert strip.results() == []


def test_threaded_refresh_reads_bars_off_the_qt_thread(monkeypatch):
    import threading

    import live_alert_results
    from PySide6.QtWidgets import QApplication
    from ui.widgets.live_results_strip import LiveResultsStrip

    monkeypatch.setattr(live_alert_results, "desk_zone", lambda: NY)
    clock = _Clock(datetime(2026, 9, 22, 10, 35, 20, tzinfo=NY))
    strip = LiveResultsStrip(threaded=True, clock=clock)
    threads = []

    def provider(symbol):
        threads.append(threading.current_thread())
        return BARS.get(symbol, [])

    strip.set_bars_provider(provider)
    strip.record(_alert("NVDA"))  # hidden: no read yet
    clock.moment = datetime(2026, 9, 22, 10, 45, 5, tzinfo=NY)
    strip.refresh()
    deadline = time.monotonic() + 5
    while "NEW 1" not in strip.line_text() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert strip.line_text() == "Working now · NEW 1 (+1.4R)"
    assert threads and threads[0] is not threading.main_thread()


def test_timer_runs_only_while_shown():
    strip, _clock = _strip()
    assert not strip.timer_active()
    strip.show()
    assert strip.timer_active()
    strip.hide()
    assert not strip.timer_active()


def test_the_desk_wires_the_strip_under_working_lately():
    """Mounted under the Working-lately line; fed by the Alert Center's M5
    post; cleared by the M5 DAY ROLL and never by the bar's own "Clear all";
    graded from the same snapshot the bar is; bars from the live bot only."""
    from ui.panels.trading_desk import TradingDeskPanel
    from ui.widgets.live_results_strip import LiveResultsStrip

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        strip = desk.live_results_strip
        assert isinstance(strip, LiveResultsStrip)
        layout = desk.m5_alert_bar.layout()
        assert layout.itemAt(0).widget() is desk.working_lately_strip
        assert layout.itemAt(1).widget() is strip

        desk.alert_center.m5AlertPosted.emit(_bounce_alert("NVDA"))
        assert [row["symbol"] for row in strip.results()] == ["NVDA"]
        desk.m5_alert_bar.clear_all()
        assert len(strip.results()) == 1  # the bar's Clear all is the bar's only
        desk.alert_center.m5AlertsDayRolled.emit()
        assert strip.results() == []

        desk.set_working_lately_snapshot({"setup_grades": _grades()})
        desk.alert_center.m5AlertPosted.emit(_bounce_alert("NVDA"))
        assert strip.results()[0]["grade"] == "A"

        # No live bot: the provider answers "no bars" and never raises.
        desk.bounce_panel.service.current_bot = lambda: None
        assert desk._live_results_bars("NVDA") == []
        calls = []

        def m5_chart_bars(symbol, max_sessions=2):
            calls.append((symbol, max_sessions))
            return [1]

        bot = SimpleNamespace(m5_chart_bars=m5_chart_bars)
        desk.bounce_panel.service.current_bot = lambda: bot
        assert desk._live_results_bars("NVDA") == [1]
        assert calls == [("NVDA", 1)]
    finally:
        desk.shutdown()
        desk.close()
