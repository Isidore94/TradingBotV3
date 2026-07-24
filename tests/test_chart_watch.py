import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chart_watch import (  # noqa: E402
    D1LevelWatch,
    D1_LEVEL_KINDS,
    WATCH_KINDS,
    arm_chart_watch,
    evaluate_chart_watch,
    evaluate_d1_level_watch,
    load_chart_watches,
    load_d1_level_watches,
    save_chart_watches,
    save_d1_level_watches,
    watch_is_stale,
)

DAY = datetime(2026, 7, 24)


def _bar(hour, minute, *, o=100.0, h=100.0, low=100.0, c=100.0, v=1000.0, day=DAY):
    return {
        "dt": day.replace(hour=hour, minute=minute),
        "open": float(o),
        "high": float(h),
        "low": float(low),
        "close": float(c),
        "volume": float(v),
    }


def test_arm_baseline_uses_todays_bars_only_and_includes_forming_bar():
    yesterday = DAY - timedelta(days=1)
    bars = [
        _bar(15, 55, h=200.0, low=90.0, day=yesterday),  # prior session ignored
        _bar(9, 30, h=110.0, low=99.0),
        _bar(9, 35, h=108.0, low=101.0),  # forming at arm time: still counts
    ]
    now = DAY.replace(hour=9, minute=37)
    hod = arm_chart_watch("new_hod", "nvda", "LONG", bars, now=now)
    lod = arm_chart_watch("new_lod", "NVDA", "SHORT", bars, now=now)
    vwap = arm_chart_watch("vwap_bounce", "NVDA", "banana", bars, now=now)

    assert hod.symbol == "NVDA" and hod.baseline == 110.0 and hod.side == "LONG"
    assert lod.baseline == 99.0
    # VWAP bounce has no fixed level; unknown side falls back to WATCH.
    assert vwap.baseline is None and vwap.side == "WATCH"

    with pytest.raises(ValueError):
        arm_chart_watch("teleport", "NVDA", "LONG", bars, now=now)


def test_new_hod_triggers_only_on_completed_post_arm_break():
    bars = [
        _bar(9, 30, h=110.0, low=99.0),
        _bar(9, 35, h=108.0, low=101.0),
    ]
    armed = arm_chart_watch("new_hod", "NVDA", "LONG", bars, now=DAY.replace(hour=9, minute=41))

    # A forming bar above the armed high is preview only - no trigger.
    bars.append(_bar(9, 45, h=111.0, low=104.0, c=110.8))
    assert evaluate_chart_watch(armed, bars, now=DAY.replace(hour=9, minute=49)) is None

    # The same bar completed (09:45 + 5min <= 09:50) fires exactly once.
    hit = evaluate_chart_watch(armed, bars, now=DAY.replace(hour=9, minute=50))
    assert hit is not None
    assert hit.price == 111.0
    assert "New HOD 111.00 > armed day high 110.00" in hit.message

    # A bar that only matches the armed high is not a NEW high.
    equal_only = bars[:2] + [_bar(9, 45, h=110.0, low=104.0)]
    assert evaluate_chart_watch(armed, equal_only, now=DAY.replace(hour=9, minute=50)) is None


def test_extreme_watch_without_cached_bars_builds_baseline_before_firing():
    # Armed before the bot cached any bars for the symbol: pre-arm completed
    # bars tighten the reference, and the first post-arm bar cannot trivially
    # "break" a missing baseline.
    armed = arm_chart_watch("new_lod", "NVDA", "SHORT", [], now=DAY.replace(hour=9, minute=40))
    bars = [
        _bar(9, 30, h=110.0, low=99.0),  # completed pre-arm: becomes baseline
        _bar(9, 45, h=105.0, low=98.5),  # post-arm break of 99.0
    ]
    hit = evaluate_chart_watch(armed, bars, now=DAY.replace(hour=9, minute=55))
    assert hit is not None
    assert hit.price == 98.5
    assert "New LOD 98.50 < armed day low 99.00" in hit.message

    fresh_only = [_bar(9, 45, h=105.0, low=98.5)]
    assert evaluate_chart_watch(armed, fresh_only, now=DAY.replace(hour=9, minute=55)) is None


def test_vwap_bounce_touch_and_reclaim_by_side():
    flat = [
        _bar(9, 30),
        _bar(9, 35),
        _bar(9, 40),
    ]
    # VWAP sits at ~100 after three flat bars; the 09:45 bar tags it from
    # above and closes back over it - the long touch-and-reclaim.
    bounce = _bar(9, 45, o=100.2, h=100.5, low=99.8, c=100.3)
    bars = flat + [bounce]
    now = DAY.replace(hour=9, minute=55)

    long_watch = arm_chart_watch("vwap_bounce", "NVDA", "LONG", flat, now=DAY.replace(hour=9, minute=42))
    hit = evaluate_chart_watch(long_watch, bars, now=now)
    assert hit is not None
    assert hit.price == 100.3
    assert "VWAP bounce (long)" in hit.message

    # The same tape is NOT a short bounce (it closed above VWAP).
    short_watch = arm_chart_watch("vwap_bounce", "NVDA", "SHORT", flat, now=DAY.replace(hour=9, minute=42))
    assert evaluate_chart_watch(short_watch, bars, now=now) is None

    # WATCH side accepts either direction.
    watch_watch = arm_chart_watch("vwap_bounce", "NVDA", "WATCH", flat, now=DAY.replace(hour=9, minute=42))
    assert evaluate_chart_watch(watch_watch, bars, now=now) is not None

    # A bar that never touches VWAP does not fire.
    above = flat + [_bar(9, 45, o=100.6, h=100.9, low=100.5, c=100.8)]
    assert evaluate_chart_watch(long_watch, above, now=now) is None

    # Short flavor: tag VWAP from below, close back under it.
    short_bounce = flat + [_bar(9, 45, o=99.8, h=100.2, low=99.6, c=99.7)]
    hit = evaluate_chart_watch(short_watch, short_bounce, now=now)
    assert hit is not None
    assert "VWAP bounce (short)" in hit.message


def test_pre_arm_bars_never_trigger():
    bars = [
        _bar(9, 30, h=110.0, low=99.0),
        _bar(9, 35, h=112.0, low=101.0),  # day high already 112 before arming
    ]
    armed = arm_chart_watch("new_hod", "NVDA", "LONG", bars, now=DAY.replace(hour=9, minute=41))
    # Nothing new after arming: the pre-arm 112 print must not fire.
    assert evaluate_chart_watch(armed, bars, now=DAY.replace(hour=9, minute=55)) is None


def test_watch_is_stale_next_session():
    armed = arm_chart_watch("new_hod", "NVDA", "LONG", [], now=DAY.replace(hour=15, minute=55))
    assert not watch_is_stale(armed, now=DAY.replace(hour=16, minute=5))
    assert watch_is_stale(armed, now=DAY + timedelta(days=1, hours=9))


def test_watch_kind_labels_cover_all_buttons():
    assert list(WATCH_KINDS) == ["new_hod", "new_lod", "vwap_bounce", "band_bounce"]
    assert WATCH_KINDS["new_hod"] == "New HOD"
    assert WATCH_KINDS["new_lod"] == "New LOD"
    assert WATCH_KINDS["vwap_bounce"] == "VWAP bounce"
    assert WATCH_KINDS["band_bounce"] == "σ-band bounce"
    assert set(D1_LEVEL_KINDS) == {"d1_level_above", "d1_level_below"}


def test_band_bounce_touch_and_reclaim_by_side():
    # Two volume bars build VWAP 102 with ±1σ ≈ 102 ± 1.414; the trigger bar
    # carries zero volume so the band at its index is exactly that value.
    base = [
        _bar(9, 30),  # tp 100
        _bar(9, 35, o=100.0, h=108.0, low=100.0, c=108.0),  # tp 104
    ]
    upper = 102.0 + (4000.0 / 2000.0) ** 0.5  # ≈ 103.414
    lower = 102.0 - (4000.0 / 2000.0) ** 0.5  # ≈ 100.586
    now = DAY.replace(hour=9, minute=50)
    armed = DAY.replace(hour=9, minute=41)

    long_hit = base + [
        _bar(9, 45, o=upper + 0.6, h=upper + 0.9, low=upper - 0.2, c=upper + 0.7, v=0.0)
    ]
    long_watch = arm_chart_watch("band_bounce", "NVDA", "LONG", base, now=armed)
    hit = evaluate_chart_watch(long_watch, long_hit, now=now)
    assert hit is not None
    assert "σ-band bounce (long)" in hit.message
    assert hit.resolved_side == "long"

    # The same tape is not a short bounce (never tagged the LOWER band).
    short_watch = arm_chart_watch("band_bounce", "NVDA", "SHORT", base, now=armed)
    assert evaluate_chart_watch(short_watch, long_hit, now=now) is None

    short_hit = base + [
        _bar(9, 45, o=lower - 0.3, h=lower + 0.2, low=lower - 0.6, c=lower - 0.4, v=0.0)
    ]
    hit = evaluate_chart_watch(short_watch, short_hit, now=now)
    assert hit is not None
    assert "σ-band bounce (short)" in hit.message
    assert hit.resolved_side == "short"

    # A bar that stays inside the bands fires neither side.
    inside = base + [_bar(9, 45, o=102.0, h=102.5, low=101.5, c=102.2, v=0.0)]
    watch_watch = arm_chart_watch("band_bounce", "NVDA", "WATCH", base, now=armed)
    assert evaluate_chart_watch(watch_watch, inside, now=now) is None


def test_chart_watches_persist_for_the_same_day_only(tmp_path):
    path = tmp_path / "alert_chart_watches.json"
    bars = [_bar(9, 30, h=110.0, low=99.0)]
    armed = [
        arm_chart_watch("new_hod", "NVDA", "LONG", bars, now=DAY.replace(hour=9, minute=40)),
        arm_chart_watch("band_bounce", "TSLA", "SHORT", [], now=DAY.replace(hour=9, minute=41), source_text="ctx"),
    ]
    save_chart_watches(armed, path, market_date=DAY.date())

    restored = load_chart_watches(path, market_date=DAY.date())
    assert restored == armed  # frozen dataclasses compare by value

    # A new session starts clean.
    assert load_chart_watches(path, market_date=(DAY + timedelta(days=1)).date()) == []
    # Corrupt file degrades to empty.
    path.write_text("{not json", encoding="utf-8")
    assert load_chart_watches(path, market_date=DAY.date()) == []


def test_d1_level_watch_persistence_and_evaluation(tmp_path):
    path = tmp_path / "d1_level_watches.json"
    armed_at = DAY.replace(hour=14, minute=0)  # 2026-07-24 14:00
    above = D1LevelWatch(
        symbol="NVDA", direction="above", level=50.0, armed_at=armed_at, candle_date="2026-07-20"
    )
    below = D1LevelWatch(symbol="TSLA", direction="below", level=20.0, armed_at=armed_at)
    save_d1_level_watches([above, below], path)
    assert load_d1_level_watches(path) == [above, below]
    assert above.kind == "d1_level_above" and below.kind == "d1_level_below"

    def d1_bar(day_offset, high, low):
        return {
            "dt": DAY + timedelta(days=day_offset),
            "open": (high + low) / 2,
            "high": high,
            "low": low,
            "close": (high + low) / 2,
            "volume": 1000.0,
        }

    # The armed day's own D1 bar never triggers (it contains pre-arm prices);
    # a later completed session crossing the level does.
    later = DAY + timedelta(days=3)
    assert (
        evaluate_d1_level_watch(above, [], [d1_bar(0, 51.0, 45.0)], now=later) is None
    )
    hit = evaluate_d1_level_watch(
        above, [], [d1_bar(0, 51.0, 45.0), d1_bar(1, 50.4, 46.0)], now=later
    )
    assert hit is not None
    assert hit.price == 50.4
    assert "D1 level break above 50.00" in hit.message
    assert hit.resolved_side == "long"
    # Today's forming daily bar is preview only.
    assert (
        evaluate_d1_level_watch(above, [], [d1_bar(3, 55.0, 46.0)], now=later) is None
    )

    # Completed M5 evidence covers the armed day itself while scanned.
    m5 = [
        {
            "dt": armed_at.replace(hour=14, minute=30),
            "open": 49.5,
            "high": 50.2,
            "low": 49.4,
            "close": 50.1,
            "volume": 500.0,
        }
    ]
    hit = evaluate_d1_level_watch(above, m5, [], now=armed_at.replace(hour=14, minute=40))
    assert hit is not None and hit.price == 50.2
    # ...but not while the bar is still forming.
    assert (
        evaluate_d1_level_watch(above, m5, [], now=armed_at.replace(hour=14, minute=33))
        is None
    )

    # Short side: a later session probing under the level flags.
    hit = evaluate_d1_level_watch(
        below, [], [d1_bar(1, 22.0, 19.8)], now=later
    )
    assert hit is not None
    assert hit.resolved_side == "short"
    assert "D1 level break below 20.00" in hit.message
