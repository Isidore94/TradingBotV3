"""A4 paint-lines: level building, stable ids, and the off-thread guarantee.

The widget-side tests (rendering, the toggle, hit-testing, y-range) live in
``test_chart_paint_lines.py``; this file is the pure-data half.
"""

from __future__ import annotations

import json
import math
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chart_levels  # noqa: E402


def _bars(count: int = 30, *, start: str = "2026-06-01", base: float = 100.0) -> list[dict]:
    """Consecutive daily bars. Dates are calendar days; the projection math
    counts BARS, so a synthetic calendar is fine and keeps the fixtures short."""
    first = datetime.fromisoformat(start)
    out = []
    for index in range(count):
        close = base + index
        out.append(
            {
                "dt": first + timedelta(days=index),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000.0,
            }
        )
    return out


# --------------------------------------------------------------------------
# prev-day high/low
# --------------------------------------------------------------------------
def test_prev_day_levels_come_from_the_previous_session():
    bars = _bars(5)
    levels = chart_levels.prev_day_levels(bars)
    families = {level["family"]: level for level in levels}
    assert set(families) == {"prev_day_high", "prev_day_low"}
    assert families["prev_day_high"]["price"] == pytest.approx(bars[-2]["high"])
    assert families["prev_day_low"]["price"] == pytest.approx(bars[-2]["low"])
    assert all(level["group"] == chart_levels.GROUP_PREV_DAY for level in levels)
    assert all(level["values"] is None for level in levels)


def test_prev_day_is_relative_to_a_forming_last_bar():
    """A preview candle is still "today", so prev-day is the bar before it."""
    bars = _bars(4)
    bars[-1]["preview"] = True
    levels = chart_levels.prev_day_levels(bars)
    assert levels[0]["price"] == pytest.approx(bars[-2]["high"])


def test_prev_day_needs_two_distinct_sessions():
    assert chart_levels.prev_day_levels([]) == []
    assert chart_levels.prev_day_levels(_bars(1)) == []
    same_day = _bars(2)
    same_day[1]["dt"] = same_day[0]["dt"]
    assert chart_levels.prev_day_levels(same_day) == []


# --------------------------------------------------------------------------
# horizontal S/R
# --------------------------------------------------------------------------
def _hv(price: float, bucket: str = "green", **extra) -> dict:
    record = {
        "kind": "hv_horizontal",
        "price": price,
        "bucket": bucket,
        "first_seen": "2026-05-01",
        "last_seen": "2026-05-20",
        "touch_count": 2,
        "respect_count": 2,
        "break_count": 0,
        "strength": 1.4,
    }
    record.update(extra)
    return record


def test_horizontal_levels_style_by_bucket():
    levels = chart_levels.horizontal_levels([_hv(105.0), _hv(107.0, "red")])
    by_family = {level["price"]: level for level in levels}
    green = by_family[105.0]
    red = by_family[107.0]
    assert green["color"] == "chart_green" and green["dash"] is False
    assert red["color"] == "chart_grey" and red["dash"] is True
    # Conviction earns weight, and a green level always outweighs a red one.
    assert green["width"] > red["width"]
    assert all(level["group"] == chart_levels.GROUP_HORIZONTAL for level in levels)


def test_horizontal_levels_drop_weak_and_out_of_range_records():
    records = [
        _hv(105.0),
        _hv(106.0, strength=0.2),          # below MIN_HORIZONTAL_STRENGTH
        _hv(400.0),                        # outside the chart's price range
        {"kind": "hv_horizontal", "price": 0.0},   # unusable
    ]
    levels = chart_levels.horizontal_levels(records, price_range=(90.0, 130.0))
    assert [level["price"] for level in levels] == [105.0]


def test_cloud_flat_respects_its_effective_range():
    from datetime import date

    flat = {
        "kind": "cloud_flat",
        "price": 104.0,
        "bucket": "cloud",
        "effective_range": ["2026-06-01", "2026-06-10"],
        "first_seen": "2026-06-01",
        "touch_count": 0,
        "respect_count": 0,
        "break_count": 0,
        "strength": 1.0,
    }
    inside = chart_levels.horizontal_levels([flat], as_of=date(2026, 6, 5))
    outside = chart_levels.horizontal_levels([flat], as_of=date(2026, 7, 5))
    assert [level["family"] for level in inside] == ["d1_cloud_flat"]
    assert outside == []


def test_horizontal_levels_stay_inside_the_clutter_budget():
    records = [_hv(100.0 + index * 0.1) for index in range(40)]
    records += [_hv(120.0 + index * 0.1, "red") for index in range(40)]
    levels = chart_levels.horizontal_levels(records)
    greens = [level for level in levels if level["color"] == "chart_green"]
    reds = [level for level in levels if level["color"] == "chart_grey"]
    assert len(greens) == chart_levels.MAX_GREEN_HORIZONTALS
    assert len(reds) == chart_levels.MAX_RED_HORIZONTALS


# --------------------------------------------------------------------------
# stable ids
# --------------------------------------------------------------------------
def test_level_ids_are_stable_across_two_builds_of_the_same_store():
    records = [_hv(105.0), _hv(107.5, "red")]
    first = chart_levels.horizontal_levels(list(records))
    # A later scan rewrites the store with the same levels, plus touches: the
    # geometry the trader is looking at has not moved, so the ids must not.
    later = [dict(record, touch_count=5, last_seen="2026-06-30") for record in records]
    second = chart_levels.horizontal_levels(later)
    assert [level["id"] for level in first] == [level["id"] for level in second]


def test_level_id_prefers_an_id_the_store_supplies():
    record = _hv(105.0, id="store-native-42")
    (level,) = chart_levels.horizontal_levels([record])
    assert level["id"] == "store-native-42"


def test_trendline_id_ignores_the_projected_price():
    """The line moves every session; its identity is its two pivots."""
    candidate = {
        "type": "H-",
        "start_date": "2026-01-05",
        "end_date": "2026-03-02",
        "current_line_price": 118.0,
    }
    moved = dict(candidate, current_line_price=121.0)
    assert chart_levels.trendline_id(candidate) == chart_levels.trendline_id(moved)


def test_level_id_shape_is_family_anchor_price():
    assert chart_levels.level_id("d1_horizontal", "2026-05-01", 105.004) == (
        "d1_horizontal:2026-05-01:105.00"
    )


# --------------------------------------------------------------------------
# the trendline projection
# --------------------------------------------------------------------------
def _candidate(bars: list[dict], slope: float = 0.004, price: float = 120.0) -> dict:
    return {
        "type": "H-",
        "start_date": bars[5]["dt"].date().isoformat(),
        "end_date": bars[15]["dt"].date().isoformat(),
        "lookback_start": bars[0]["dt"].date().isoformat(),
        "lookback_end": bars[-1]["dt"].date().isoformat(),
        "current_line_price": price,
        "slope_log_per_bar": slope,
        "touch_count": 3,
    }


def test_trendline_projects_along_its_slope_in_log_space():
    bars = _bars(30)
    line = chart_levels.trendline_level(_candidate(bars), bars)
    assert line is not None
    values = line["values"]
    assert len(values) == len(bars)
    # Anchored at lookback_end (the last bar) at exactly current_line_price.
    assert values[-1] == pytest.approx(120.0)
    # One bar back is one slope step back, in LOG space.
    assert values[-2] == pytest.approx(120.0 * math.exp(-0.004))
    # Ten bars back is ten steps, not ten times one step.
    assert values[-11] == pytest.approx(120.0 * math.exp(-0.04))
    assert line["price"] == pytest.approx(120.0)
    assert line["group"] == chart_levels.GROUP_TRENDLINE


def test_trendline_starts_at_its_first_pivot():
    bars = _bars(30)
    line = chart_levels.trendline_level(_candidate(bars), bars)
    assert line["values"][4] is None       # before the start pivot
    assert line["values"][5] is not None   # the start pivot itself


def test_trendline_is_omitted_when_the_slope_is_missing():
    """No slope means no honest projection - and no line, not a flat guess."""
    bars = _bars(30)
    candidate = _candidate(bars)
    candidate.pop("slope_log_per_bar")
    assert chart_levels.trendline_level(candidate, bars) is None


def test_trendline_is_omitted_when_its_anchor_bar_is_not_on_the_chart():
    bars = _bars(30)
    candidate = _candidate(bars)
    candidate["lookback_end"] = "2019-01-02"
    assert chart_levels.trendline_level(candidate, bars) is None


def test_trendline_is_omitted_once_the_scan_behind_it_is_stale():
    bars = _bars(30)
    candidate = _candidate(bars)
    last_session = bars[-1]["dt"].date()
    fresh = last_session - timedelta(days=1)
    stale = last_session - timedelta(days=chart_levels.TRENDLINE_MAX_AGE_DAYS + 1)
    assert chart_levels.trendline_level(candidate, bars, scan_date=fresh) is not None
    assert chart_levels.trendline_level(candidate, bars, scan_date=stale) is None


def test_trendline_does_not_overflow_on_an_absurd_slope():
    bars = _bars(30)
    line = chart_levels.trendline_level(_candidate(bars, slope=9.0), bars)
    # Whatever survives must be finite; the far end is simply not drawn.
    assert line is None or all(
        value is None or math.isfinite(value) for value in line["values"]
    )


# --------------------------------------------------------------------------
# grouping / filtering (the toggle's data half)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,group",
    [
        ("SMA50", chart_levels.GROUP_SMA),
        ("SMA200", chart_levels.GROUP_SMA),
        ("EMA21", chart_levels.GROUP_EMA),
        ("AVWAPE", chart_levels.GROUP_AVWAP),
        ("AVWAPE prev", chart_levels.GROUP_AVWAP),
        ("±2σ", chart_levels.GROUP_AVWAP),
        ("VWAP", ""),
    ],
)
def test_overlay_group_names_the_existing_overlays(label, group):
    assert chart_levels.overlay_group(label) == group


def test_hiding_a_group_removes_exactly_its_lines():
    overlays = [
        {"label": "SMA50", "values": [1.0]},
        {"label": "EMA21", "values": [1.0]},
        {"label": "AVWAPE", "values": [1.0]},
    ]
    levels = [
        {"id": "a", "group": chart_levels.GROUP_HORIZONTAL, "price": 1.0},
        {"id": "b", "group": chart_levels.GROUP_PREV_DAY, "price": 2.0},
    ]
    hidden = [chart_levels.GROUP_SMA, chart_levels.GROUP_PREV_DAY]
    assert [o["label"] for o in chart_levels.visible_overlays(overlays, hidden)] == [
        "EMA21",
        "AVWAPE",
    ]
    assert [level["id"] for level in chart_levels.visible_levels(levels, hidden)] == ["a"]


def test_an_overlay_no_group_can_name_is_never_hidden():
    overlays = [{"label": "VWAP", "values": [1.0]}]
    kept = chart_levels.visible_overlays(overlays, list(chart_levels.GROUP_NAMES))
    assert [o["label"] for o in kept] == ["VWAP"]


def test_nothing_hidden_means_nothing_removed():
    overlays = [{"label": "SMA50", "values": [1.0]}]
    assert chart_levels.visible_overlays(overlays, []) == overlays


# --------------------------------------------------------------------------
# build_d1_levels end to end, off real files
# --------------------------------------------------------------------------
def _write_store(tmp_path: Path, symbol: str, levels: list[dict]) -> Path:
    levels_dir = tmp_path / "levels"
    levels_dir.mkdir(exist_ok=True)
    (levels_dir / f"{symbol}.json").write_text(
        json.dumps({"symbol": symbol, "levels": levels}), encoding="utf-8"
    )
    return levels_dir


def _write_ai_state(tmp_path: Path, symbol: str, entry: dict) -> Path:
    path = tmp_path / "ai_state.json"
    path.write_text(json.dumps({"symbols": {symbol: entry}}), encoding="utf-8")
    return path


def test_build_d1_levels_reads_both_stores(tmp_path):
    chart_levels.reset_caches()
    bars = _bars(30)
    levels_dir = _write_store(tmp_path, "AAA", [_hv(105.0), _hv(112.0, "red")])
    ai_state = _write_ai_state(
        tmp_path,
        "AAA",
        {
            "last_trade_date": bars[-1]["dt"].date().isoformat(),
            "priority_trendline_candidate": _candidate(bars),
        },
    )
    levels = chart_levels.build_d1_levels(
        "AAA",
        bars,
        levels_dir=levels_dir,
        ai_state_path=ai_state,
        # Explicit empty alert stores: without them this reads the DESK's live
        # price_alerts.json, and a test whose result depends on what the trader
        # armed this morning is not a test.
        price_alerts_path=tmp_path / "no_alerts.json",
        d1_level_watches_path=tmp_path / "no_watches.json",
    )
    groups = {level["group"] for level in levels}
    assert groups == {
        chart_levels.GROUP_HORIZONTAL,
        chart_levels.GROUP_PREV_DAY,
        chart_levels.GROUP_TRENDLINE,
    }
    assert all(level["id"] for level in levels)
    assert len({level["id"] for level in levels}) == len(levels)


def test_build_d1_levels_survives_a_missing_store(tmp_path):
    """No level store and no ai_state still yields the bars-only families."""
    chart_levels.reset_caches()
    bars = _bars(10)
    levels = chart_levels.build_d1_levels(
        "AAA",
        bars,
        levels_dir=tmp_path / "nope",
        ai_state_path=tmp_path / "nothing.json",
        price_alerts_path=tmp_path / "no_alerts.json",
        d1_level_watches_path=tmp_path / "no_watches.json",
    )
    assert {level["group"] for level in levels} == {chart_levels.GROUP_PREV_DAY}


def test_build_d1_levels_survives_a_corrupt_store(tmp_path):
    chart_levels.reset_caches()
    bars = _bars(10)
    levels_dir = tmp_path / "levels"
    levels_dir.mkdir()
    (levels_dir / "AAA.json").write_text("{not json", encoding="utf-8")
    levels = chart_levels.build_d1_levels(
        "AAA",
        bars,
        levels_dir=levels_dir,
        ai_state_path=tmp_path / "nothing.json",
        price_alerts_path=tmp_path / "no_alerts.json",
        d1_level_watches_path=tmp_path / "no_watches.json",
    )
    assert {level["group"] for level in levels} == {chart_levels.GROUP_PREV_DAY}


def test_build_d1_levels_needs_a_symbol_and_bars(tmp_path):
    assert chart_levels.build_d1_levels("", _bars(3)) == []
    assert chart_levels.build_d1_levels("AAA", []) == []


def test_store_reads_are_mtime_cached(tmp_path, monkeypatch):
    """The level store lives on Drive: the same unchanged file is read once."""
    chart_levels.reset_caches()
    levels_dir = _write_store(tmp_path, "AAA", [_hv(105.0)])
    reads = {"count": 0}
    real_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        if self.name == "AAA.json":
            reads["count"] += 1
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    for _ in range(4):
        chart_levels._store_levels("AAA", levels_dir)
    assert reads["count"] == 1


# --------------------------------------------------------------------------
# the constraint the whole packet is built around
# --------------------------------------------------------------------------
@pytest.mark.qt
def test_levels_are_built_on_the_worker_never_the_calling_thread():
    """The A4 non-negotiable: zero I/O on the paint path.

    ``ChartDataService`` is the only production caller of ``build_d1_levels``,
    and this proves the call actually lands on a pool thread rather than the
    thread that asked for the snapshot.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return
    QApplication.instance() or QApplication([])

    from ui.services.chart_data_service import ChartDataService

    calling_thread = threading.get_ident()
    seen: list[int] = []

    real_build = chart_levels.build_d1_levels

    def recording_build(symbol, bars, **kwargs):
        seen.append(threading.get_ident())
        return real_build(symbol, bars, **kwargs)

    chart_levels.build_d1_levels = recording_build
    service = ChartDataService(max_threads=1)
    try:
        service.request("AAA", [])
        service.wait_for_idle(10_000)
    finally:
        chart_levels.build_d1_levels = real_build
        service.shutdown()

    assert seen, "the level build never ran"
    assert calling_thread not in seen


# --------------------------------------------------------------------------
# R4 section 4: armed alerts painted as a levels family
#
# Read-only display. These tests exist to prove three things the spec is
# explicit about: the lines appear and disappear with the STORES (never with
# a cached copy), nothing here writes either store, and an event watch with
# no price contributes no line rather than an invented one.
# --------------------------------------------------------------------------
def _write_price_alerts(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "price_alerts.json"
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path


def _write_level_watches(tmp_path: Path, watches: list[dict]) -> Path:
    path = tmp_path / "d1_level_watches.json"
    path.write_text(json.dumps({"watches": watches}), encoding="utf-8")
    return path


def _level_watch(symbol: str, direction: str, level: float, armed: str = "2026-06-10T09:30:00") -> dict:
    return {
        "symbol": symbol,
        "direction": direction,
        "level": level,
        "armed_at": armed,
        "candle_date": "2026-06-10",
    }


def test_armed_price_alerts_paint_both_sides():
    levels = chart_levels.armed_alert_levels(
        "AAA",
        price_alerts=[
            {"symbol": "AAA", "above": 120.0, "below": 95.0},
        ],
        level_watches=[],
    )
    by_family = {level["family"]: level for level in levels}
    assert set(by_family) == {"price_alert_above", "price_alert_below"}
    assert by_family["price_alert_above"]["price"] == pytest.approx(120.0)
    assert by_family["price_alert_below"]["price"] == pytest.approx(95.0)
    assert all(level["group"] == chart_levels.GROUP_ALERTS for level in levels)


def test_a_disarmed_side_paints_no_line():
    """The store keeps the price after a disarm; the chart must not."""
    levels = chart_levels.armed_alert_levels(
        "AAA",
        price_alerts=[
            {"symbol": "AAA", "above": 120.0, "below": 95.0, "armed_above": False},
        ],
        level_watches=[],
    )
    assert [level["family"] for level in levels] == ["price_alert_below"]


def test_alerts_for_another_symbol_are_not_painted():
    levels = chart_levels.armed_alert_levels(
        "AAA",
        price_alerts=[{"symbol": "BBB", "above": 120.0}],
        level_watches=[_level_watch("BBB", "above", 130.0)],
    )
    assert levels == []


def test_armed_d1_level_watches_paint_on_their_side():
    levels = chart_levels.armed_alert_levels(
        "AAA",
        price_alerts=[],
        level_watches=[
            _level_watch("AAA", "above", 130.0),
            _level_watch("AAA", "below", 88.0),
        ],
    )
    families = [level["family"] for level in levels]
    assert families == ["d1_level_watch_above", "d1_level_watch_below"]
    assert [level["price"] for level in levels] == [pytest.approx(130.0), pytest.approx(88.0)]


def test_an_event_watch_with_no_price_paints_nothing():
    """A D1 EVENT watch is a condition, not a level. Inventing a price for it
    would draw a line the trader never armed."""
    levels = chart_levels.armed_alert_levels(
        "AAA",
        price_alerts=[],
        level_watches=[],
        event_watches=[{"symbol": "AAA", "kind": "new_high_20", "armed_at": "2026-06-10T09:30:00"}],
    )
    assert levels == []


def test_armed_alert_ids_are_stable_across_two_builds():
    store = [{"symbol": "AAA", "above": 120.0}]
    watches = [_level_watch("AAA", "above", 130.0)]
    first = chart_levels.armed_alert_levels("AAA", price_alerts=store, level_watches=watches)
    second = chart_levels.armed_alert_levels("AAA", price_alerts=store, level_watches=watches)
    assert [level["id"] for level in first] == [level["id"] for level in second]
    assert len({level["id"] for level in first}) == len(first)


def test_moving_an_alert_changes_its_id():
    """Ids carry the price, so a re-armed alert is a different line - which is
    what keeps a stale selection from following the trader to a new level."""
    at_120 = chart_levels.armed_alert_levels(
        "AAA", price_alerts=[{"symbol": "AAA", "above": 120.0}], level_watches=[]
    )
    at_121 = chart_levels.armed_alert_levels(
        "AAA", price_alerts=[{"symbol": "AAA", "above": 121.0}], level_watches=[]
    )
    assert at_120[0]["id"] != at_121[0]["id"]


def test_a_duplicate_watch_paints_one_line():
    watch = _level_watch("AAA", "above", 130.0)
    levels = chart_levels.armed_alert_levels(
        "AAA", price_alerts=[], level_watches=[watch, dict(watch)]
    )
    assert len(levels) == 1


def test_a_price_alert_far_off_the_chart_is_still_painted():
    """Armed alerts are the trader's own decisions. Painted levels are excluded
    from autoscale, so an off-range alert costs nothing and dropping it would
    hide a live alarm."""
    levels = chart_levels.armed_alert_levels(
        "AAA", price_alerts=[{"symbol": "AAA", "above": 9_999.0}], level_watches=[]
    )
    assert len(levels) == 1
    assert levels[0]["price"] == pytest.approx(9_999.0)


def test_the_armed_alert_group_is_in_the_toggle_order():
    assert chart_levels.GROUP_ALERTS in dict(chart_levels.LEVEL_GROUPS)
    assert chart_levels.GROUP_NAMES[chart_levels.GROUP_ALERTS]


def test_build_d1_levels_reads_the_alert_stores(tmp_path):
    chart_levels.reset_caches()
    bars = _bars(30)
    alerts = _write_price_alerts(tmp_path, [{"symbol": "AAA", "above": 140.0}])
    watches = _write_level_watches(tmp_path, [_level_watch("AAA", "below", 90.0)])
    levels = chart_levels.build_d1_levels(
        "AAA",
        bars,
        levels_dir=tmp_path / "missing",
        ai_state_path=tmp_path / "missing.json",
        price_alerts_path=alerts,
        d1_level_watches_path=watches,
    )
    alert_levels = [level for level in levels if level["group"] == chart_levels.GROUP_ALERTS]
    assert {level["family"] for level in alert_levels} == {
        "price_alert_above",
        "d1_level_watch_below",
    }


def test_build_d1_levels_survives_missing_alert_stores(tmp_path):
    chart_levels.reset_caches()
    bars = _bars(10)
    levels = chart_levels.build_d1_levels(
        "AAA",
        bars,
        levels_dir=tmp_path / "missing",
        ai_state_path=tmp_path / "missing.json",
        price_alerts_path=tmp_path / "no_alerts.json",
        d1_level_watches_path=tmp_path / "no_watches.json",
    )
    assert not [level for level in levels if level["group"] == chart_levels.GROUP_ALERTS]
    assert [level for level in levels if level["group"] == chart_levels.GROUP_PREV_DAY]


def test_building_alert_levels_never_writes_either_store(tmp_path):
    chart_levels.reset_caches()
    alerts = _write_price_alerts(tmp_path, [{"symbol": "AAA", "above": 140.0}])
    watches = _write_level_watches(tmp_path, [_level_watch("AAA", "below", 90.0)])
    before = (alerts.read_bytes(), watches.read_bytes())
    chart_levels.build_d1_levels(
        "AAA",
        _bars(30),
        levels_dir=tmp_path / "missing",
        ai_state_path=tmp_path / "missing.json",
        price_alerts_path=alerts,
        d1_level_watches_path=watches,
    )
    assert (alerts.read_bytes(), watches.read_bytes()) == before
