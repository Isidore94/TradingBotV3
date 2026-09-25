"""P1-6 6a: the entry chip on the Working-now strip and on each M5 alert row (offscreen Qt)."""

from __future__ import annotations

import os
import sys
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


@pytest.fixture(autouse=True)
def _ny(monkeypatch):
    import live_alert_results

    monkeypatch.setattr(live_alert_results, "desk_zone", lambda: NY)


def _alert(symbol, entry, stop, side="LONG"):
    feedback = {"symbol": symbol, "bounce_types": "vwap", "entry_price": entry, "stop_price": stop}
    return SimpleNamespace(
        symbol=symbol, side=side, trigger="vwap", time_text="10:35:20", timeframe="M5", raw_text="",
        payload={"feedback": feedback},
    )


def _bar(hh, mm, o, h, low, c):
    return {"dt": datetime(2026, 9, 22, hh, mm), "open": o, "high": h, "low": low, "close": c}


BARS = {
    "NVDA": [_bar(10, 35, 100.0, 100.6, 100.1, 100.5), _bar(10, 40, 100.5, 100.8, 100.3, 100.6)],
    "AMD": [_bar(10, 35, 50.0, 50.2, 49.4, 49.6)],
}


class _Clock:
    def __init__(self, moment):
        self.moment = moment

    def __call__(self):
        return self.moment


def _strip():
    from ui.widgets.live_results_strip import LiveResultsStrip

    clock = _Clock(datetime(2026, 9, 22, 10, 36, 0, tzinfo=NY))
    strip = LiveResultsStrip(threaded=False, clock=clock)
    strip.set_bars_provider(lambda symbol: BARS.get(symbol, []))
    return strip, clock


def test_the_strip_names_each_alerts_entry_state():
    strip, clock = _strip()
    strip.record(_alert("NVDA", 100.0, 99.0))
    strip.record(_alert("AMD", 50.0, 49.5))
    clock.moment = datetime(2026, 9, 22, 10, 45, 5, tzinfo=NY)
    strip.refresh()
    assert strip.line_text().endswith(" · 1 entry valid")
    tooltip = strip.toolTip()
    assert "NVDA LONG 10:36" in tooltip and "entry: valid" in tooltip
    assert "entry: gone (stop hit 10:35)" in tooltip


def test_the_newest_alert_on_a_row_sets_its_state():
    """A repeat folds into the row; the row's chip reads the NEWEST alert's entry/stop."""
    strip, clock = _strip()
    seen = []
    strip.entryStatesChanged.connect(seen.append)
    strip.record(_alert("NVDA", 100.0, 99.0))
    strip.record(_alert("NVDA", 100.6, 100.0))  # the 10:40 bar comes back to 100.6
    clock.moment = datetime(2026, 9, 22, 10, 45, 5, tzinfo=NY)
    strip.refresh()
    states = strip.entry_states()
    assert states[("NVDA", "LONG")]["state"] == "improved"
    assert "entry: valid" in strip.toolTip()  # the strip's own row is the FIRST alert
    assert seen and seen[-1] == states
    strip.clear_day()
    assert strip.entry_states() == {}
    assert seen[-1] == {}


def test_the_m5_row_carries_the_chip_and_rewrites_only_changed_rows():
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar()
    bar.post(_alert("NVDA", 100.0, 99.0))
    bar.post(_alert("AMD", 50.0, 49.5))
    assert "valid" not in bar.list.item(0).text()
    bar.set_entry_states({
        ("NVDA", "LONG"): {"state": "valid", "reason": "x", "at": None},
        ("AMD", "LONG"): {"state": "gone", "reason": "stop hit", "at": datetime(2026, 9, 22, 10, 35)},
    })
    texts = {bar.list.item(i).data(0x0100).symbol: bar.list.item(i).text() for i in range(bar.count())}
    assert texts["NVDA"].endswith("· valid")
    assert "· gone" in texts["AMD"]
    amd = next(bar.list.item(i) for i in range(bar.count()) if "AMD" in bar.list.item(i).text())
    assert amd.toolTip().startswith("entry: gone (stop hit 10:35)")
    bar.set_entry_states({})
    assert all("· valid" not in bar.list.item(i).text() for i in range(bar.count()))


def test_the_desk_hands_the_states_to_the_bar():
    source = (SCRIPTS_DIR / "ui" / "panels" / "trading_desk.py").read_text(encoding="utf-8")
    assert "self.live_results_strip.entryStatesChanged.connect(self.m5_alert_bar.set_entry_states)" in source
