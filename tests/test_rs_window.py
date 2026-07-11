"""RS Window tab: feed joins/sorting (pure) + chart/panel behavior (Qt)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _mover(symbol, side, excess, **extra):
    return {
        "symbol": symbol,
        "side": side,
        "window_pct": excess,
        "spy_pct": 0.0,
        "excess": excess,
        **extra,
    }


def test_decorate_rows_joins_pick_industry_and_strength_maps():
    from ui.services.rs_window_feed import decorate_mover_rows

    rows = decorate_mover_rows(
        [_mover("AAA", "LONG", 1.0), _mover("CCC", "SHORT", 0.5)],
        pick_map={
            ("AAA", "LONG"): {
                "tier": "S",
                "priority_bucket": "favorite_setup",
                "setup_family": "avwap_breakout",
                "favorite_zone": "AVWAPE to UPPER_1",
                "priority_score": 250.0,
            }
        },
        industry_map={
            "AAA": {"sector": "Technology", "sector_rs": 1.2, "sector_rank": 1.0,
                    "industry": "Semiconductors", "industry_rs": 2.0, "industry_rank": 3.0},
        },
        strength_map={"AAA": {"d1_rs_5d": 2.5, "d1_rs_20d": 4.0, "weekly_streak": 7}},
    )
    by_symbol = {row["symbol"]: row for row in rows}
    aaa = by_symbol["AAA"]
    assert aaa["favorite_setup"] and aaa["tier"] == "S" and aaa["setup_family"] == "avwap_breakout"
    assert aaa["industry"] == "Semiconductors" and aaa["industry_rs"] == 2.0
    assert aaa["sector_rs"] == 1.2 and aaa["d1_rs_20d"] == 4.0 and aaa["weekly_streak"] == 7
    ccc = by_symbol["CCC"]
    assert not ccc["favorite_setup"] and ccc["tier"] == ""


def test_filter_rows_by_side_and_favorites():
    from ui.services.rs_window_feed import filter_mover_rows

    rows = [
        _mover("AAA", "LONG", 1.0, favorite_setup=True),
        _mover("BBB", "LONG", 0.5, favorite_setup=False),
        _mover("CCC", "SHORT", 0.8, favorite_setup=True),
    ]
    assert [r["symbol"] for r in filter_mover_rows(rows, side="LONG")] == ["AAA", "BBB"]
    assert [r["symbol"] for r in filter_mover_rows(rows, favorites_only=True)] == ["AAA", "CCC"]
    assert [r["symbol"] for r in filter_mover_rows(rows, side="SHORT", favorites_only=True)] == ["CCC"]


def test_sort_rows_side_aligned_strength():
    from ui.services.rs_window_feed import sort_mover_rows

    rows = [
        _mover("LONG_WEAK", "LONG", 0.2, industry_rs=-3.0, weekly_streak=-6),
        _mover("LONG_STRONG", "LONG", 0.1, industry_rs=5.0, weekly_streak=8),
        _mover("SHORT_IN_WEAK", "SHORT", 0.3, industry_rs=-4.0, weekly_streak=-7),
        _mover("SHORT_IN_STRONG", "SHORT", 0.4, industry_rs=6.0, weekly_streak=9),
        _mover("NO_DATA", "LONG", 0.0),
    ]
    # Excess sort: plain descending.
    assert [r["symbol"] for r in sort_mover_rows(rows, "excess")][0] == "SHORT_IN_STRONG"
    # Industry sort is side-aligned: strong-industry long and weak-industry
    # short lead; misaligned rows sink; missing data last.
    ordered = [r["symbol"] for r in sort_mover_rows(rows, "industry_rs")]
    assert ordered[:2] == ["LONG_STRONG", "SHORT_IN_WEAK"]
    assert ordered[-1] == "NO_DATA"
    # Weekly sort likewise.
    ordered = [r["symbol"] for r in sort_mover_rows(rows, "weekly_streak")]
    assert ordered[:2] == ["LONG_STRONG", "SHORT_IN_WEAK"]


def test_load_bot_pick_map_reads_tier_list(tmp_path):
    from ui.services.rs_window_feed import load_bot_pick_map

    csv_path = tmp_path / "tier_list.csv"
    csv_path.write_text(
        "tier,symbol,side,priority_bucket,priority_score,setup_family,favorite_zone\n"
        "S,CRH,SHORT,favorite_setup,322.1,avwap_retest_followthrough,LOWER_1 to AVWAPE\n"
        "B,NVDA,LONG,near_favorite_zone,120.0,avwap_breakout,\n",
        encoding="utf-8",
    )
    picks = load_bot_pick_map(csv_path)
    assert picks[("CRH", "SHORT")]["tier"] == "S"
    assert picks[("CRH", "SHORT")]["priority_bucket"] == "favorite_setup"
    assert picks[("NVDA", "LONG")]["setup_family"] == "avwap_breakout"
    assert load_bot_pick_map(tmp_path / "missing.csv") == {}


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _chart_bars(count=30, start=None):
    start = start or datetime(2026, 7, 8, 9, 30)
    bars = []
    price = 100.0
    for index in range(count):
        close = price + 0.1 * index
        bars.append(
            {
                "dt": start + timedelta(minutes=5 * index),
                "open": close - 0.05,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": 1000.0,
            }
        )
    return bars


def test_chart_selection_maps_region_to_datetimes():
    if _qt_app() is None:
        return
    from ui.widgets.spy_m5_chart import SpyM5Chart

    chart = SpyM5Chart()
    assert chart.selected_range() is None
    bars = _chart_bars(30)
    chart.set_bars(bars)
    assert chart.bar_count() == 30
    # Default selection: trailing hour ending on the last bar.
    start_dt, end_dt = chart.selected_range()
    assert end_dt == bars[-1]["dt"]
    assert start_dt == bars[-13]["dt"]
    # Moving the region maps back to bar datetimes, clamped to the data.
    chart._region.setRegion((5, 999))
    start_dt, end_dt = chart.selected_range()
    assert start_dt == bars[5]["dt"] and end_dt == bars[-1]["dt"]


def test_panel_ranks_through_stub_service():
    if _qt_app() is None:
        return
    from ui.panels.rs_window_panel import RsWindowPanel

    class _StubBot:
        def spy_m5_chart_bars(self, max_sessions=2):
            return _chart_bars(30)

        def rank_window_movers(self, start_dt, end_dt, sides=("long", "short")):
            self.window = (start_dt, end_dt)
            return {
                "ok": True,
                "spy_pct": -0.5,
                "rows": [
                    {"symbol": "AAA", "side": "LONG", "window_pct": 0.2, "spy_pct": -0.5, "excess": 0.7},
                    {"symbol": "CCC", "side": "SHORT", "window_pct": -1.2, "spy_pct": -0.5, "excess": 0.7},
                ],
            }

    class _StubService:
        def __init__(self):
            self.bot = _StubBot()

        def current_bot(self):
            return self.bot

    service = _StubService()
    panel = RsWindowPanel(service)
    panel.refresh_chart()
    assert panel.chart.bar_count() == 30

    # Deterministic context joins: patch the feed loaders via the kwargs path.
    import ui.services.rs_window_feed as feed

    original = feed.decorate_mover_rows
    feed_calls = {}

    def _fake_decorate(rows, **_kwargs):
        feed_calls["rows"] = rows
        return original(
            rows,
            pick_map={("AAA", "LONG"): {"tier": "A", "priority_bucket": "favorite_setup",
                                          "setup_family": "avwap_breakout", "favorite_zone": "",
                                          "priority_score": 200.0}},
            industry_map={},
            strength_map={},
        )

    feed.decorate_mover_rows = _fake_decorate
    try:
        panel.rank_selected_window()
    finally:
        feed.decorate_mover_rows = original

    assert service.bot.window[1] == _chart_bars(30)[-1]["dt"]
    assert panel.model.rowCount() == 2
    # Favorite filter narrows to the decorated favorite pick.
    panel.favorites_input.setChecked(True)
    assert panel.model.rowCount() == 1
    row = panel.model.index(0, 0).data()
    assert "AAA" in str(row)


def test_panel_auto_tick_fills_without_clicks():
    if _qt_app() is None:
        return
    import ui.services.rs_window_feed as feed
    from ui.panels.rs_window_panel import RsWindowPanel

    class _StubBot:
        def spy_m5_chart_bars(self, max_sessions=2):
            return _chart_bars(30)

        def rank_window_movers(self, start_dt, end_dt, sides=("long", "short")):
            self.window = (start_dt, end_dt)
            return {
                "ok": True,
                "spy_pct": 0.1,
                "rows": [{"symbol": "AAA", "side": "LONG", "window_pct": 0.4, "spy_pct": 0.1, "excess": 0.3}],
            }

    class _StubService:
        def __init__(self):
            self.bot = _StubBot()

        def current_bot(self):
            return self.bot

    original = feed.decorate_mover_rows
    feed.decorate_mover_rows = lambda rows, **_kw: original(
        rows, pick_map={}, industry_map={}, strength_map={}
    )
    try:
        panel = RsWindowPanel(_StubService())
        assert panel._auto_timer.isActive()
        panel._auto_tick()  # what the timer does: refresh + rank, zero clicks
        assert panel.chart.bar_count() == 30
        assert panel.model.rowCount() == 1
        # Untouched region keeps tracking the trailing hour...
        assert not panel._region_customized
        start_dt, end_dt = panel.chart.selected_range()
        assert end_dt == _chart_bars(30)[-1]["dt"] and start_dt == _chart_bars(30)[-13]["dt"]
        # ...until the trader drags it, after which refreshes preserve it.
        panel.chart._region.setRegion((2, 9))
        panel._on_region_changed()
        assert panel._region_customized
        pinned = panel.chart.selected_range()
        panel._auto_tick()
        assert panel.chart.selected_range() == pinned
    finally:
        feed.decorate_mover_rows = original

    # Disconnected: the auto tick stays quiet instead of nagging the status bar.
    class _DeadService:
        def current_bot(self):
            return None

    dead_panel = RsWindowPanel(_DeadService())
    before = dead_panel.status_label.text()
    dead_panel._auto_tick()
    assert dead_panel.status_label.text() == before
