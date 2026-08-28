"""The Alert Center re-derived mover state from the same bars, over and over.

G-P1.2b. The Focus board's own memo (shipped 2026-08-26) fixed that board; this
fixes the SOURCE, so the review queue - which asks the same question once per
alert - gets it too. The trader extended the file-scoped ask-first
authorization on `alert_center_panel.py` to cover this change on 2026-08-26.

Measured cost per (symbol, side), synthetic series at realistic sizes
(5 sessions of M5, 490 D1 rows):

    m5 materialize          0.049 ms   <- paid before any memo can help
    prev_session_extremes   0.177 ms
    completed_session_bars  0.008 ms
    gate mover_state        0.001 ms
    ------------------------------------
    total                   0.234 ms, of which 79% is memo-able

That distribution is why this memo is keyed the way it is. **It is a pure memo,
not a cache with an expiry.** The key carries the identity of the bars the
answer was computed from - session date, and the length and last timestamp of
both series - so a reused answer is one that provably could not have changed.

That distinction matters more here than anywhere else in this pass, because
`mover_state` feeds the movers-only review filter, which DECIDES WHAT THE
TRADER SEES. A time-based cache would let a name that has just broken
yesterday's high stay hidden for up to a minute. This one cannot: a new bar is
a new key.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Alert Center is a Qt panel")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _m5(count: int, *, day: int = 26):
    start = datetime(2026, 8, day, 6, 30)
    return [
        {
            "dt": start + timedelta(minutes=5 * i),
            "open": 10.0,
            "high": 10.4,
            "low": 9.9,
            "close": 10.3,
            "volume": 1000,
        }
        for i in range(count)
    ]


def _d1(count: int):
    start = datetime(2026, 1, 1)
    return [
        {
            "dt": start + timedelta(days=i),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1_000_000,
        }
        for i in range(count)
    ]


def _panel(tmp_path, m5, d1):
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    panel._m5_bars_for = lambda symbol, **kwargs: list(m5)
    panel._d1_bars_for = lambda symbol: list(d1)
    return panel


def _count_gate_calls(monkeypatch):
    """Count the measurement itself, not the accessors around it."""
    import focus_adoption_gate

    calls: list[tuple] = []
    original = focus_adoption_gate.mover_state

    def counting(side, price, prev_high, prev_low):
        calls.append((side, price, prev_high, prev_low))
        return original(side, price, prev_high, prev_low)

    monkeypatch.setattr(focus_adoption_gate, "mover_state", counting)
    return calls


def test_the_same_bars_are_measured_once(tmp_path, monkeypatch):
    panel = _panel(tmp_path, _m5(40), _d1(200))
    calls = _count_gate_calls(monkeypatch)

    first = panel._measure_mover_state("AAA", "long")
    second = panel._measure_mover_state("AAA", "long")
    third = panel._measure_mover_state("AAA", "long")

    assert first == second == third
    assert len(calls) == 1, f"measured {len(calls)} times over identical bars"


def test_a_new_bar_is_a_new_measurement(tmp_path, monkeypatch):
    """The whole point. A memo that outlived its bars would hide a live break."""
    from ui.panels.alert_center_panel import AlertCenterPanel

    bars = _m5(40)
    daily = _d1(200)
    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    panel._m5_bars_for = lambda symbol, **kwargs: list(bars)
    panel._d1_bars_for = lambda symbol: list(daily)

    calls = _count_gate_calls(monkeypatch)
    panel._measure_mover_state("AAA", "long")
    assert len(calls) == 1

    # A completed bar arrives.
    bars.append(
        {
            "dt": bars[-1]["dt"] + timedelta(minutes=5),
            "open": 10.3,
            "high": 12.0,
            "low": 10.2,
            "close": 11.9,
            "volume": 5000,
        }
    )
    panel._measure_mover_state("AAA", "long")
    assert len(calls) == 2, "a new bar reused the old answer"


def test_the_two_sides_do_not_share_an_answer(tmp_path, monkeypatch):
    panel = _panel(tmp_path, _m5(40), _d1(200))
    calls = _count_gate_calls(monkeypatch)

    panel._measure_mover_state("AAA", "long")
    panel._measure_mover_state("AAA", "short")

    assert {side for side, *_ in calls} == {"long", "short"}, calls


def test_the_two_symbols_do_not_share_an_answer(tmp_path, monkeypatch):
    panel = _panel(tmp_path, _m5(40), _d1(200))
    calls = _count_gate_calls(monkeypatch)

    panel._measure_mover_state("AAA", "long")
    panel._measure_mover_state("BBB", "long")

    assert len(calls) == 2, calls


def test_a_failed_measurement_is_never_remembered(tmp_path, monkeypatch):
    """UNKNOWN from a broken read is not an answer; it is the absence of one.

    Caching it would let one hiccup pin a symbol to UNKNOWN until its next bar,
    and UNKNOWN is the state that SHOWS in the review queue - so the cost of
    getting this wrong is a queue that disagrees with the bars behind it.
    """
    from ui.panels.alert_center_panel import AlertCenterPanel, PREV_DAY_UNKNOWN

    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    broken = {"yes": True}

    def flaky_d1(symbol):
        if broken["yes"]:
            raise RuntimeError("the daily store is unreadable")
        return _d1(200)

    panel._d1_bars_for = flaky_d1
    panel._m5_bars_for = lambda symbol, **kwargs: _m5(40)

    assert panel._measure_mover_state("AAA", "long") == PREV_DAY_UNKNOWN

    broken["yes"] = False
    calls = _count_gate_calls(monkeypatch)
    panel._measure_mover_state("AAA", "long")
    assert len(calls) == 1, "the failure was cached as though it were an answer"


def test_the_memo_keeps_only_the_newest_entry_per_symbol_side(tmp_path):
    """It must not grow a row per five-minute bucket for the whole session."""
    from ui.panels.alert_center_panel import AlertCenterPanel

    bars = _m5(40)
    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    panel._m5_bars_for = lambda symbol, **kwargs: list(bars)
    panel._d1_bars_for = lambda symbol: _d1(200)

    for _ in range(50):
        panel._measure_mover_state("AAA", "long")
        bars.append(
            {
                "dt": bars[-1]["dt"] + timedelta(minutes=5),
                "open": 10.0,
                "high": 10.4,
                "low": 9.9,
                "close": 10.3,
                "volume": 1000,
            }
        )

    assert len(panel._mover_measure_cache) == 1, panel._mover_measure_cache


def test_the_answer_is_unchanged_by_the_memo(tmp_path):
    """A memo that changed a verdict would be a detector change, not a repair."""
    import focus_adoption_gate
    from prev_day_gate import prev_session_extremes
    from chart_watch import completed_session_bars

    m5, d1 = _m5(40), _d1(200)
    panel = _panel(tmp_path, m5, d1)

    moment = datetime.now()
    prev_high, prev_low = prev_session_extremes(d1, session=moment.date())
    completed = completed_session_bars(m5, now=moment)
    price = completed[-1]["close"] if completed else None
    expected, _reason = focus_adoption_gate.mover_state("long", price, prev_high, prev_low)

    assert panel._measure_mover_state("AAA", "long") == expected
