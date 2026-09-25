"""P1-6 6a: the entry state machine (valid / improved / gone / unknown). Pure fixtures."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import entry_state as es  # noqa: E402
import live_alert_results as lar  # noqa: E402

NY = ZoneInfo("America/New_York")
RECEIVED = datetime(2026, 9, 22, 10, 36, 10, tzinfo=NY)  # bucket 10:35


def _entry(side="LONG", entry=100.0, stop=99.0):
    alert = SimpleNamespace(
        symbol="NVDA",
        side=side,
        trigger="vwap",
        payload={"feedback": {"entry_price": entry, "stop_price": stop, "bounce_types": "vwap"}},
    )
    return lar.entry_from_alert(alert, RECEIVED)


def _bar(hh, mm, o, h, low, c):
    return {"dt": datetime(2026, 9, 22, hh, mm), "open": o, "high": h, "low": low, "close": c}


def _state(entry, bars, now=datetime(2026, 9, 22, 11, 0, tzinfo=NY)):
    return es.entry_state(entry, bars, now, naive_zone=NY)


def test_valid_when_price_sits_between_entry_and_one_r_and_the_level_held():
    bars = [_bar(10, 35, 100.0, 100.6, 100.1, 100.5), _bar(10, 40, 100.5, 100.9, 100.3, 100.7)]
    state = _state(_entry(), bars)
    assert state["state"] == es.VALID
    assert state["r"] == 0.7


def test_improved_when_a_later_bar_touches_the_entry_and_the_stop_holds():
    bars = [
        _bar(10, 35, 100.0, 100.6, 100.1, 100.5),
        _bar(10, 40, 100.5, 100.6, 99.8, 100.2),  # back to the level, stop 99 holds
        _bar(10, 45, 100.2, 100.5, 100.0, 100.4),
    ]
    state = _state(_entry(), bars)
    assert state["state"] == es.IMPROVED
    assert state["at"].strftime("%H:%M") == "10:40"


def test_a_touch_in_the_alerts_own_bar_is_not_a_later_touch():
    bars = [_bar(10, 35, 100.0, 100.6, 99.9, 100.5)]
    assert _state(_entry(), bars)["state"] == es.VALID


def test_gone_when_the_stop_is_hit_even_if_price_came_back():
    bars = [
        _bar(10, 35, 100.0, 100.2, 98.9, 99.5),
        _bar(10, 40, 99.5, 100.6, 99.4, 100.5),
    ]
    state = _state(_entry(), bars)
    assert state["state"] == es.GONE
    assert state["reason"] == "stop hit"


def test_gone_when_the_last_close_is_beyond_one_r():
    bars = [_bar(10, 35, 100.0, 101.3, 100.1, 101.2)]
    state = _state(_entry(), bars)
    assert state["state"] == es.GONE
    assert state["reason"] == "beyond +1R"


def test_short_side_mirrors():
    valid = [_bar(10, 35, 100.0, 99.9, 99.4, 99.5)]
    assert _state(_entry("SHORT", 100.0, 101.0), valid)["state"] == es.VALID
    improved = valid + [_bar(10, 40, 99.5, 100.1, 99.4, 99.6)]
    assert _state(_entry("SHORT", 100.0, 101.0), improved)["state"] == es.IMPROVED
    stopped = valid + [_bar(10, 40, 99.5, 101.0, 99.4, 100.5)]
    assert _state(_entry("SHORT", 100.0, 101.0), stopped)["state"] == es.GONE
    chased = [_bar(10, 35, 100.0, 99.9, 98.8, 98.9)]
    assert _state(_entry("SHORT", 100.0, 101.0), chased)["state"] == es.GONE


def test_a_forming_bar_is_never_read():
    """The 10:55 bar is still open at 10:57, so only 10:35 counts: valid, not gone."""
    bars = [_bar(10, 35, 100.0, 100.4, 100.1, 100.3), _bar(10, 55, 100.3, 101.5, 100.2, 101.4)]
    state = _state(_entry(), bars, now=datetime(2026, 9, 22, 10, 57, tzinfo=NY))
    assert state["state"] == es.VALID


def test_missing_data_is_unknown_never_valid():
    assert _state(_entry(), [])["state"] == es.UNKNOWN
    assert _state(_entry(stop=None), [_bar(10, 35, 100, 100.5, 100, 100.2)])["state"] == es.UNKNOWN
    assert _state(None, [])["state"] == es.UNKNOWN
    # Bars before the alert's bucket say nothing about it.
    assert _state(_entry(), [_bar(10, 30, 100, 101, 99.5, 100.5)])["state"] == es.UNKNOWN


def test_chip_text_and_detail():
    assert es.chip_text({"state": es.IMPROVED}) == "improved"
    assert es.chip_text(None) == "unknown"
    assert es.chip_text({"state": "odd"}) == "unknown"
    line = es.chip_detail({"state": es.GONE, "reason": "stop hit", "at": datetime(2026, 9, 22, 10, 40)})
    assert line == "entry: gone (stop hit 10:40)"
