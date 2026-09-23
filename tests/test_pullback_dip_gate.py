"""The Pullback dip gate (trader, 2026-09-23; `pullback_sma_reclaim_v2`).

A trigger may fire only if, before its LRSI 80 up-cross: price came to the SMA
(a bar's low within 0.2 x D1 ATR(20) of it or through it, or a close below it
and a later close back above), the LRSI then crossed DOWN through 20 at or
after that dip began, and the dip is no older than 15 M30 / 30 M15 bars.

The oscillator is stubbed so each case places its crosses and flips exactly;
the price tape is a hand-built rise over a 5-bar SMA, so every SMA value is
checkable by eye. The real ARM/QDEL tapes are in the golden file beside this.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_pct1_pullback_alert import bar_dt, bar_end, mirror  # noqa: E402

SMA = 5
LAST = 44  # 45 M30 bars: four sessions, 2026-08-17 .. 2026-08-20
TOUCH_INDEX = 33


def closes_of(overrides=None):
    """Flat at 100 (bars 0-9: the episode's closes on the line), then +1 a bar."""
    closes = [100.0] * 10 + [92.0 + index for index in range(10, LAST + 1)]
    for index, value in (overrides or {}).items():
        closes[index] = value
    return closes


def sma_at(closes, index):
    return sum(closes[index + 1 - SMA: index + 1]) / SMA


def make_tape(*, closes=None, lows=None, minutes=30):
    closes = closes or closes_of()
    bars = []
    for index, close in enumerate(closes):
        bars.append({
            "dt": bar_dt(index, minutes),
            "open": close,
            "high": close + 0.3,
            "low": (lows or {}).get(index, close - 0.3),
            "close": close,
            "volume": 1_000,
        })
    return bars


def touch_lows(closes, index=TOUCH_INDEX):
    """Bar ``index``'s low 0.1 above the SMA: a touch at 0.2 x ATR 1.0."""
    return {index: sma_at(closes, index) + 0.1}


class _Stub:
    def __init__(self, count, ups, downs):
        self.values = [50.0] * count
        self._ups = tuple(ups)
        self._downs = tuple(downs)

    def cross_up_indices(self, _level):
        return self._ups

    def cross_down_indices(self, level):
        assert level == 20.0
        return self._downs


def stub_lrsi(monkeypatch, *, ups=(), downs=(), companion_ups=()):
    from indicators import pullback_sma_reclaim as rule

    def fake(closes, _side):
        if len(closes) > LAST + 1:  # the M15 companion
            return _Stub(len(closes), companion_ups, ())
        return _Stub(len(closes), ups, downs)

    monkeypatch.setattr(rule, "_lrsi_series", fake)


def run(bars, *, side="long", companion=None, **kwargs):
    from indicators.pullback_sma_reclaim import evaluate

    return evaluate(
        bars,
        side=side,
        sma_length=SMA,
        bar_minutes=30,
        armed_at=bar_dt(0, 30),
        now=bar_end(LAST, 30),
        companion_bars=companion,
        companion_minutes=15 if companion is not None else None,
        **kwargs,
    )


def fired(result, trigger):
    return [fire for fire in result.fired if fire.trigger == trigger]


# ---------------------------------------------------------------------------
def test_the_constants_are_the_trader_s_numbers():
    from indicators import pullback_sma_reclaim as rule

    assert rule.RULE_VERSION == "pullback_sma_reclaim_v2"
    assert rule.DIP_TOLERANCE_D1_ATR == 0.2
    assert rule.D1_ATR_LENGTH == 20
    assert rule.BEAR_FLIP_LEVEL == 20.0
    assert rule.DIP_WINDOW_BARS == {30: 15, 15: 30}


def test_the_tape_is_what_it_says():
    closes = closes_of()
    # Reclaim at bar 10; afterwards every low sits 1.7 above the SMA.
    assert closes[9] == sma_at(closes, 9) and closes[10] > sma_at(closes, 10)
    assert closes[30] - 0.3 - sma_at(closes, 30) == pytest.approx(1.7)


def test_a_dip_by_touch_then_a_bear_flip_lets_the_cross_fire(monkeypatch):
    closes = closes_of()
    stub_lrsi(monkeypatch, ups=(38,), downs=(34,))
    result = run(make_tape(closes=closes, lows=touch_lows(closes)), d1_atr=1.0)

    [fire] = fired(result, "reclaim_then_lrsi")
    assert fire.cross_bar_dt == bar_dt(38, 30)
    assert fire.dip_path == "touch"
    assert fire.dip_bar_dt == bar_dt(TOUCH_INDEX, 30)
    assert fire.bear_flip_bar_dt == bar_dt(34, 30)
    assert fire.d1_atr == 1.0
    assert "gate_blocked" not in result.details


def test_a_dip_by_close_break_and_reclaim_fires_without_any_daily_bars(monkeypatch):
    closes = closes_of({TOUCH_INDEX: 110.0})  # one close under the SMA, then back
    assert closes[TOUCH_INDEX] < sma_at(closes, TOUCH_INDEX)
    assert closes[TOUCH_INDEX + 1] > sma_at(closes, TOUCH_INDEX + 1)
    stub_lrsi(monkeypatch, ups=(38,), downs=(35,))
    result = run(make_tape(closes=closes))

    [fire] = fired(result, "reclaim_then_lrsi")
    assert fire.dip_path == "break_reclaim"
    assert fire.dip_bar_dt == bar_dt(TOUCH_INDEX, 30)
    assert fire.d1_atr is None


def test_sma_reclaim_lrsi_needs_the_flip_inside_the_break(monkeypatch):
    # The episode: closes on the line from bar 4, reclaim at bar 10.
    stub_lrsi(monkeypatch, ups=(9,), downs=(6,))
    [fire] = fired(run(make_tape()), "sma_reclaim_lrsi")
    assert fire.dip_path == "break_reclaim"
    assert fire.dip_bar_dt == bar_dt(4, 30)
    assert fire.bear_flip_bar_dt == bar_dt(6, 30)

    stub_lrsi(monkeypatch, ups=(9,), downs=())
    result = run(make_tape())
    assert fired(result, "sma_reclaim_lrsi") == []
    assert result.details["gate_blocked"]["sma_reclaim_lrsi"] == "no_bear_flip"


def test_a_tape_that_never_came_back_to_the_sma_is_blocked(monkeypatch):
    # ARM's shape: the only dip is the old episode, 29 bars before the cross.
    stub_lrsi(monkeypatch, ups=(38,), downs=(35,))
    result = run(make_tape(), d1_atr=1.0)

    assert fired(result, "reclaim_then_lrsi") == []
    assert result.details["gate_blocked"] == {"reclaim_then_lrsi": "dip_stale"}


def test_no_dip_anywhere_is_no_dip():
    from indicators.pullback_sma_reclaim import _dip_gate

    bars = [{"close": 110.0, "low": 109.0, "high": 111.0} for _ in range(20)]
    gate = _dip_gate(
        bars, [100.0] * 20, long_side=True, cross_index=19, limit_index=19,
        window=15, tolerance=0.2, bear_flips=(10,), flip_ok=lambda index: True,
    )
    assert gate.blocked == "no_dip"


def test_a_dip_without_a_bear_flip_is_blocked(monkeypatch):
    closes = closes_of()
    tape = make_tape(closes=closes, lows=touch_lows(closes))
    stub_lrsi(monkeypatch, ups=(38,), downs=())
    result = run(tape, d1_atr=1.0)
    assert fired(result, "reclaim_then_lrsi") == []
    assert result.details["gate_blocked"] == {"reclaim_then_lrsi": "no_bear_flip"}

    # A flip BEFORE the dip began is not the flip the trader asked for.
    stub_lrsi(monkeypatch, ups=(38,), downs=(30,))
    result = run(tape, d1_atr=1.0)
    assert result.details["gate_blocked"] == {"reclaim_then_lrsi": "no_bear_flip"}


def test_a_dip_outside_the_window_is_stale_and_the_edge_is_inside(monkeypatch):
    closes = closes_of()
    stub_lrsi(monkeypatch, ups=(38,), downs=(21,))
    stale = run(make_tape(closes=closes, lows=touch_lows(closes, 20)), d1_atr=1.0)
    assert fired(stale, "reclaim_then_lrsi") == []
    assert stale.details["gate_blocked"] == {"reclaim_then_lrsi": "dip_stale"}

    stub_lrsi(monkeypatch, ups=(38,), downs=(24,))
    edge = run(make_tape(closes=closes, lows=touch_lows(closes, 23)), d1_atr=1.0)
    assert len(fired(edge, "reclaim_then_lrsi")) == 1  # 38 - 23 == 15 bars


def test_missing_daily_atr_blocks_the_touch_path_only(monkeypatch):
    closes = closes_of()
    stub_lrsi(monkeypatch, ups=(38,), downs=(34,))
    result = run(make_tape(closes=closes, lows=touch_lows(closes)))  # no D1 at all
    assert fired(result, "reclaim_then_lrsi") == []
    assert result.details["gate_blocked"] == {"reclaim_then_lrsi": "dip_stale"}
    # (The break-and-reclaim path without D1 is the test above that fires.)


def test_daily_bars_give_the_atr_from_sessions_before_the_cross_day(monkeypatch):
    closes = closes_of()
    daily = [
        {"dt": datetime(2026, 7, 1 + offset), "open": 100.0, "high": 101.0,
         "low": 100.0, "close": 100.5}
        for offset in range(25)
    ]
    # The cross (bar 38) is on 2026-08-19; a wild bar that day or later is
    # not what the trader could see that morning.
    daily += [
        {"dt": datetime(2026, 8, 19), "open": 100.0, "high": 900.0, "low": 1.0, "close": 100.0},
        {"dt": datetime(2026, 8, 20), "open": 100.0, "high": 900.0, "low": 1.0, "close": 100.0},
    ]
    assert bar_dt(38, 30).date() == datetime(2026, 8, 19).date()
    stub_lrsi(monkeypatch, ups=(38,), downs=(34,))
    result = run(make_tape(closes=closes, lows=touch_lows(closes)), daily_bars=daily)
    [fire] = fired(result, "reclaim_then_lrsi")
    assert fire.d1_atr == pytest.approx(1.0)


def test_the_short_side_is_the_mirror(monkeypatch):
    closes = closes_of()
    tape = mirror(make_tape(closes=closes, lows=touch_lows(closes)))
    stub_lrsi(monkeypatch, ups=(38,), downs=(34,))
    [fire] = fired(run(tape, side="short", d1_atr=1.0), "reclaim_then_lrsi")
    assert fire.dip_path == "touch"
    assert fire.dip_bar_dt == bar_dt(TOUCH_INDEX, 30)

    stub_lrsi(monkeypatch, ups=(38,), downs=())
    blocked = run(tape, side="short", d1_atr=1.0)
    assert blocked.details["gate_blocked"] == {"reclaim_then_lrsi": "no_bear_flip"}


def _companion():
    return make_tape(closes=[100.0] * 90, minutes=15)


def test_an_m15_companion_cross_before_the_m30_bear_flip_is_blocked(monkeypatch):
    closes = closes_of()
    tape = make_tape(closes=closes, lows=touch_lows(closes))
    # The M30 flip is bar 36 (08-19 11:30-12:00). M15 bar 72 starts 11:30.
    assert bar_dt(36, 30) == bar_dt(72, 15) == datetime(2026, 8, 19, 11, 30)
    stub_lrsi(monkeypatch, downs=(36,), companion_ups=(72,))
    early = run(tape, d1_atr=1.0, companion=_companion())
    assert fired(early, "reclaim_then_lrsi") == []
    assert early.details["gate_blocked"] == {"reclaim_then_lrsi": "no_bear_flip"}

    # The first M15 cross that starts after the flip bar closed does fire.
    stub_lrsi(monkeypatch, downs=(36,), companion_ups=(72, 74))
    [fire] = fired(run(tape, d1_atr=1.0, companion=_companion()), "reclaim_then_lrsi")
    assert fire.cross_timeframe == "M15"
    assert fire.cross_bar_dt == bar_dt(74, 15)
    assert fire.bear_flip_bar_dt == bar_dt(36, 30)


def test_the_retest_trigger_is_gated_too(monkeypatch):
    closes = closes_of()
    tape = make_tape(closes=closes, lows=touch_lows(closes))
    # D1 tolerance 2.0 makes every bar a touch; the retest itself is bar 33.
    stub_lrsi(monkeypatch, ups=(32,), downs=(30,))
    [fire] = fired(run(tape, d1_atr=10.0), "sma_retest")
    assert fire.bar_dt == bar_dt(TOUCH_INDEX, 30)
    assert fire.bear_flip_bar_dt == bar_dt(30, 30)

    stub_lrsi(monkeypatch, ups=(32,), downs=())
    result = run(tape, d1_atr=10.0)
    assert fired(result, "sma_retest") == []
    assert result.details["gate_blocked"]["sma_retest"] == "no_bear_flip"


def test_the_desk_hands_its_cached_daily_bars_to_the_rule(monkeypatch, tmp_path):
    """The worker evaluates with the D1 dicts the chart service already holds."""
    from test_pct1_pullback_alert import M15_LONG_CLOSES, M15_RECLAIM_INDEX, make_bars
    from test_pct1_pullback_desk import (
        _arm_before,
        _events,
        _install_stub_caches,
        _panel,
        settle_pullback,
    )

    from indicators import pullback_sma_reclaim as rule

    m15 = make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    _install_stub_caches(monkeypatch, bars={("NVDA", 15): m15})
    panel = _panel(monkeypatch, tmp_path)
    # The real `_pullback_d1_bars`, over a stubbed `_d1_bars_for`.
    monkeypatch.delattr(panel, "_pullback_d1_bars")
    daily = [
        {"dt": datetime(2026, 7, 1 + offset), "open": 1.0, "high": 1e9, "low": 0.0, "close": 1.0}
        for offset in range(25)
    ]
    asked: list[str] = []

    def d1_bars_for(symbol, **_kwargs):
        asked.append(symbol)
        return daily

    monkeypatch.setattr(panel, "_d1_bars_for", d1_bars_for)
    seen: list[object] = []
    real = rule.evaluate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("daily_bars"))
        return real(*args, **kwargs)

    monkeypatch.setattr(rule, "evaluate", spy)
    _arm_before(panel, "NVDA", "LONG", armed_at=bar_dt(M15_RECLAIM_INDEX - 9, 15))

    panel._poll_pullback_watches(now=bar_end(M15_RECLAIM_INDEX, 15))
    settle_pullback(panel)

    assert "NVDA" in asked
    assert seen and all(value == daily for value in seen)
    detail = _events(tmp_path, "watch_fired")[0]["detail"]
    assert detail["dip_path"] == "touch"
    assert detail["bear_flip_bar_dt"]
    assert detail["d1_atr"] == pytest.approx(1e9, rel=1e-3)
