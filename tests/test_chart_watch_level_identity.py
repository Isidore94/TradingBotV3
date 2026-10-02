"""Chart-watch triggers expose the reference level they crossed (2026-10-02).

`ChartWatchTrigger.level` was added for the first-30 chart hold. Firing must be
byte-identical: the same hits on the same fixture bars, with the same price,
bar time, message, side and details. `FIRED_SHA256` was taken from the code
BEFORE the field existed; it must never be regenerated to make a change pass.
"""

from __future__ import annotations

import hashlib
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import chart_watch as cw  # noqa: E402

TODAY = date(2026, 10, 2)
#: sha256 of `_fired_lines()` on the pre-change code, and how many lines it had.
FIRED_SHA256 = "f7e3434343ee349ae3e4fb842fb7b35682a086d43ee11a28a1941dc3c20b08ad"
FIRED_COUNT = 19712

_LEVELS = (
    {
        "high_5d": 100.0, "low_5d": 90.0, "high_20d": 105.0, "low_20d": 85.0,
        "sma50": 95.0, "sma100": 97.0, "sma200": 93.0, "ema15": 96.0,
        "atr14": 4.0, "base_range_20d": 6.0,
        "avwape_levels": [("", 94.0), ("+1σ", 98.0), ("-1σ", 90.0), ("+2σ", 102.0)],
    },
    {
        "high_5d": 99.5, "low_5d": 92.5, "high_20d": 101.0, "low_20d": 88.0,
        "sma50": 92.0, "ema15": 99.0, "atr14": 1.0, "base_range_20d": 13.0,
        "avwape_levels": [("", 97.5), ("+1σ", 100.5), ("-1σ", 91.5)],
    },
)
_GRID = (84.0, 87.5, 90.0, 92.5, 94.0, 96.0, 98.0, 100.5, 103.0, 106.0)
_D1_KINDS = (
    "ema15_reject", "new_5d_high", "new_5d_low", "new_20d_high", "new_20d_low",
    "sma_break", "avwape_bounce", "avwape_break", "avwape_dev1_bounce",
    "avwape_dev1_break", "d1_line_pullback", "range_breakout", "line_break",
)


def _daily(sessions: int = 70) -> list[dict]:
    bars = []
    day = TODAY - timedelta(days=sessions * 7 // 5 + 3)
    i = 0
    while len(bars) < sessions:
        day += timedelta(days=1)
        if day.weekday() >= 5 or day >= TODAY:
            continue
        base = 100.0 + 6.0 * math.sin(i / 4.0) + 0.05 * i
        bars.append({
            "dt": datetime(day.year, day.month, day.day),
            "open": round(base - 0.4, 2),
            "high": round(base + 1.3 + (i % 3) * 0.2, 2),
            "low": round(base - 1.5 - (i % 4) * 0.2, 2),
            "close": round(base + (0.6 if i % 2 else -0.5), 2),
            "volume": 1_000_000 + i * 1000,
        })
        i += 1
    return bars


def _m5(start_price: float, drift: float) -> list[dict]:
    bars = []
    stamp = datetime(TODAY.year, TODAY.month, TODAY.day, 9, 30)
    price = start_price
    for i in range(60):
        swing = 1.2 * math.sin(i / 3.0)
        close = round(price + swing + drift * i, 2)
        bars.append({
            "dt": stamp,
            "open": round(close - 0.15, 2),
            "high": round(close + 0.35 + (i % 3) * 0.1, 2),
            "low": round(close - 0.4 - (i % 2) * 0.1, 2),
            "close": close,
            "volume": 10_000 + 37 * i,
        })
        stamp += timedelta(minutes=5)
    return bars


def _line(case: str, hit, hits: list | None = None) -> str:
    if hits is not None and hit is not None:
        hits.append((case, hit))
    if hit is None:
        return f"{case}|-"
    details = sorted((str(k), repr(v)) for k, v in dict(hit.details or {}).items())
    return (
        f"{case}|{hit.price!r}|{hit.bar_dt.isoformat()}|{hit.message}|"
        f"{hit.resolved_side}|{details}"
    )


def _fired_lines(hits: list | None = None) -> list[str]:
    lines: list[str] = []
    for li, levels in enumerate(_LEVELS):
        for kind in _D1_KINDS:
            for prev in (None, *_GRID):
                for high in _GRID:
                    for low in _GRID:
                        if low > high:
                            continue
                        for close in (low, (low + high) / 2, high):
                            hit = cw._d1_event_hit(kind, levels, prev, high, low, close)
                            if hit is not None:
                                lines.append(f"d1hit|{li}|{kind}|{prev}|{high}|{low}|{close}|{hit!r}")
    now = datetime(TODAY.year, TODAY.month, TODAY.day, 14, 30)
    daily = _daily()
    for mi, m5 in enumerate((_m5(99.0, 0.05), _m5(101.0, -0.06), _m5(97.0, 0.12))):
        for kind in ("new_hod", "new_lod", "hod_avwap", "lod_avwap", "vwap_bounce", "band_bounce"):
            for side in ("LONG", "SHORT", "WATCH"):
                for arm_minute in (0, 45, 120):
                    armed = datetime(TODAY.year, TODAY.month, TODAY.day, 9, 30) + timedelta(minutes=arm_minute)
                    for baseline in (None, 100.0):
                        watch = cw.ChartWatch(
                            symbol="FIX", kind=kind, armed_at=armed, side=side, baseline=baseline
                        )
                        lines.append(_line(f"m5|{mi}|{kind}|{side}|{arm_minute}|{baseline}",
                                           cw.evaluate_chart_watch(watch, m5, now=now), hits))
        for kind in (*_D1_KINDS, "sma_break_retest"):
            for side in ("", "LONG", "SHORT"):
                for armed in (datetime(2026, 7, 1), datetime(2026, 9, 1), datetime(2026, 10, 2, 9, 0)):
                    watch = cw.D1EventWatch(symbol="FIX", kind=kind, armed_at=armed, side=side)
                    for anchor in (None, daily[20]["dt"].date()):
                        hit = cw.evaluate_d1_event_watch(
                            watch, m5, daily, now=now, avwape_anchor=anchor
                        )
                        lines.append(_line(f"d1|{mi}|{kind}|{side}|{armed.isoformat()}|{anchor}", hit, hits))
        for direction in ("above", "below"):
            for level in (95.0, 100.0, 103.5):
                for armed in (datetime(2026, 9, 1), datetime(2026, 10, 2, 9, 0)):
                    watch = cw.D1LevelWatch(symbol="FIX", direction=direction, level=level, armed_at=armed)
                    lines.append(_line(
                        f"lvl|{mi}|{direction}|{level}|{armed.isoformat()}",
                        cw.evaluate_d1_level_watch(watch, m5, daily, now=now),
                        hits,
                    ))
    return lines


def test_the_same_hits_fire_with_the_same_price_and_message():
    lines = _fired_lines()
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    fired = sum(1 for line in lines if not line.endswith("|-"))
    assert fired > 200  # the fixture really exercises the evaluators
    assert (len(lines), digest) == (FIRED_COUNT, FIRED_SHA256)


def test_every_fired_trigger_carries_the_level_it_crossed():
    hits: list = []
    _fired_lines(hits)
    assert len(hits) > 200
    for case, hit in hits:
        level = getattr(hit, "level", None)
        assert level is not None and math.isfinite(level), case
        if "|sma_break_retest|" in case:
            continue  # its message names the close, not the 15EMA it retested
        assert f"{level:.2f}" in hit.message, (case, level, hit.message)


def test_the_d1_event_level_is_the_line_not_the_trigger_price():
    levels = _LEVELS[0]
    assert cw._d1_event_hit_level("new_20d_high", levels, 100.0, 107.0, 101.0, 106.0)[3] == 105.0
    assert cw._d1_event_hit_level("range_breakout", levels, 100.0, 107.0, 101.0, 106.0)[3] == 105.0
    assert cw._d1_event_hit_level("avwape_bounce", levels, 96.0, 97.0, 93.5, 95.0)[3] == 94.0
    assert cw._d1_event_hit_level("ema15_reject", levels, 98.0, 98.0, 95.5, 96.5)[3] == 96.0
    assert cw._d1_event_hit_level("sma_break", levels, 94.0, 96.0, 94.0, 95.5)[3] == 95.0
    # The 3-tuple callers see is unchanged.
    assert cw._d1_event_hit("new_20d_high", levels, 100.0, 107.0, 101.0, 106.0) == (
        "New 20-day high: 107.00 > 105.00 (prior 20-session high)", "long", 107.0
    )
