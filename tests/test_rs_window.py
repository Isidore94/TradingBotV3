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


def test_primary_industry_uses_classification_not_best_overlapping_theme():
    from ui.services.rs_window_feed import build_primary_industry_context

    classifications = {
        "AAA": {"industry": "Semiconductor Equipment"},
        "BBB": {"industry": "Semiconductor Equipment"},
        "CCC": {"industry": "Semiconductor Equipment"},
    }
    industry_rows = [
        {"industry": "Semiconductors", "rs_score": "-2.0", "rs_rank": "10"},
        {"industry": "AI Winners", "rs_score": "9.0", "rs_rank": "1"},
    ]
    members = {
        "Semiconductors": ["AAA", "BBB", "CCC"],
        "AI Winners": ["AAA"],
    }
    definitions = {
        "Semiconductors": {"industries": ["Semiconductor Equipment"]},
        # AAA is explicitly listed in this stronger cross-theme group. That
        # must remain context rather than becoming the primary comparison.
        "AI Winners": {"industries": [], "tickers": ["AAA"]},
    }

    context = build_primary_industry_context(
        classifications, industry_rows, members, definitions
    )
    assert context["AAA"]["industry"] == "Semiconductors"
    assert context["AAA"]["industry_primary_source"] == "classification_definition"
    assert context["AAA"]["industry_rs"] == -2.0
    assert context["AAA"]["additional_industries"] == ["AI Winners"]


def test_primary_industry_exact_board_row_wins_and_raw_classification_is_safe_fallback():
    from ui.services.rs_window_feed import build_primary_industry_context

    classifications = {
        "AAA": {"industry": "Banks - Regional"},
        "BBB": {"industry": "Unmapped Widgets"},
        "CCC": {"industry": "Unmapped Widgets"},
    }
    rows = [{"industry": "Banks - Regional", "rs_score": "1.5", "rs_rank": "4"}]
    members = {
        "Banks - Regional": ["AAA"],
        "Hot Theme": ["BBB"],
    }
    context = build_primary_industry_context(
        classifications,
        rows,
        members,
        {"Hot Theme": {"industries": [], "tickers": ["BBB"]}},
    )
    assert context["AAA"]["industry"] == "Banks - Regional"
    assert context["AAA"]["industry_primary_source"] == "exact_classification"
    assert context["AAA"]["industry_rs"] == 1.5
    assert context["BBB"]["industry"] == "Unmapped Widgets"
    assert context["BBB"]["industry_primary_source"] == "raw_classification"
    assert context["BBB"]["industry_member_symbols"] == ["BBB", "CCC"]
    assert context["BBB"]["additional_industries"] == ["Hot Theme"]


def test_industry_m5_composite_is_endpoint_aligned_covered_and_side_symmetric():
    from ui.services.rs_window_feed import (
        add_intraday_industry_context,
        compute_industry_m5_composites,
    )

    start = datetime(2026, 7, 8, 9, 30)
    end = start + timedelta(minutes=10)

    def bars(open_price, closes, *, missing_middle=False):
        values = []
        for index, close in enumerate(closes):
            if missing_middle and index == 1:
                continue
            values.append(
                {
                    "dt": start + timedelta(minutes=5 * index),
                    "open": open_price if index == 0 else closes[index - 1],
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1000,
                }
            )
        return values

    industry_map = {
        symbol: {
            "industry": "Widgets",
            "industry_member_symbols": ["AAA", "BBB", "CCC", "DDD"],
            "industry_expected_members": 4,
        }
        for symbol in ("AAA", "BBB", "CCC", "DDD")
    }
    bars_by_symbol = {
        "SPY": bars(100.0, [100.0, 100.5, 101.0]),  # +1%
        "AAA": bars(100.0, [100.0, 101.0, 104.0]),  # +4%
        "BBB": bars(100.0, [100.0, 100.5, 102.0]),  # +2%
        "CCC": bars(100.0, [100.0, 99.5, 100.0]),   # flat
        # Exact endpoints but only 2/3 aligned timestamps: excluded at 80%.
        "DDD": bars(100.0, [100.0, 100.5, 110.0], missing_middle=True),
    }
    composite = compute_industry_m5_composites(
        industry_map,
        bars_by_symbol,
        start_dt=start,
        end_dt=end,
    )["Widgets"]
    assert composite["industry_m5_status"] == "QUALIFIED_ADVISORY"
    assert composite["industry_m5_members_used"] == 3
    assert composite["industry_m5_member_coverage"] == 0.75
    assert composite["industry_m5_window_pct"] == 2.0
    assert composite["industry_m5_vs_spy"] == 1.0
    changed_bars = {**bars_by_symbol, "AAA": bars(100.0, [100.0, 101.0, 105.0])}
    changed = compute_industry_m5_composites(
        industry_map,
        changed_bars,
        start_dt=start,
        end_dt=end,
    )["Widgets"]
    assert changed["industry_m5_snapshot_id"] != composite["industry_m5_snapshot_id"]

    rows = add_intraday_industry_context(
        [
            _mover("AAA", "LONG", 4.0, window_pct=4.0),
            _mover("AAA", "SHORT", -4.0, window_pct=4.0),
        ],
        industry_map=industry_map,
        bars_by_symbol=bars_by_symbol,
        start_dt=start,
        end_dt=end,
    )
    # AAA is removed from its own comparison: median(BBB +2%, CCC 0%) = +1%.
    assert rows[0]["industry_m5_comparison_pct"] == 1.0
    assert rows[0]["stock_vs_industry_m5"] == 3.0
    assert rows[0]["industry_comparison_includes_symbol"] is False
    assert rows[1]["stock_vs_industry_m5"] == -3.0


def test_industry_m5_insufficient_coverage_is_not_a_confident_value():
    from ui.services.rs_window_feed import (
        add_intraday_industry_context,
        compute_industry_m5_composites,
    )

    start = datetime(2026, 7, 8, 9, 30)
    end = start + timedelta(minutes=5)
    bars = lambda first, last: [
        {"dt": start, "open": first, "close": first},
        {"dt": end, "open": first, "close": last},
    ]
    context = {
        "AAA": {
            "industry": "Sparse",
            "industry_member_symbols": ["AAA", "BBB", "CCC", "DDD"],
        }
    }
    value = compute_industry_m5_composites(
        context,
        {"SPY": bars(100, 101), "AAA": bars(10, 11)},
        start_dt=start,
        end_dt=end,
    )["Sparse"]
    assert value["industry_m5_status"] == "UNAVAILABLE"
    assert value["industry_m5_window_pct"] is None
    assert value["industry_m5_vs_spy"] is None
    decorated = add_intraday_industry_context(
        [_mover("AAA", "LONG", 10.0, window_pct=10.0)],
        industry_map=context,
        bars_by_symbol={"SPY": bars(100, 101), "AAA": bars(10, 11)},
        start_dt=start,
        end_dt=end,
    )[0]
    assert decorated["industry_m5_comparison_pct"] is None
    assert decorated["stock_vs_industry_m5"] is None


def test_industry_m5_snapshot_is_atomic_advisory_and_links_daily_board(tmp_path):
    import json
    from datetime import timezone

    from ui.services.rs_window_feed import save_industry_intraday_snapshot

    board_state = tmp_path / "industry_board_snapshot.json"
    board_state.write_text(json.dumps({"snapshot_id": "daily-board-123"}), encoding="utf-8")
    target = tmp_path / "industry_intraday_rs_snapshot.json"
    start = datetime(2026, 7, 8, 9, 30)
    end = datetime(2026, 7, 8, 10, 0)
    rows = [
        _mover(
            "AAA",
            "LONG",
            2.0,
            industry="Widgets",
            additional_industries=["AI Theme"],
            industry_m5_window_pct=1.0,
            industry_m5_vs_spy=0.5,
            industry_m5_members_used=4,
            industry_m5_members_expected=5,
            industry_m5_member_coverage=0.8,
            industry_m5_timestamp_coverage=1.0,
            industry_m5_status="QUALIFIED_ADVISORY",
            industry_m5_first_ts="2026-07-08T09:30",
            industry_m5_last_ts="2026-07-08T10:00",
            industry_m5_snapshot_id="industry-1",
            industry_m5_stock_window_pct=2.0,
            stock_vs_industry_m5=1.0,
            spy_window_pct=0.5,
            advisory_only=True,
        )
    ]
    payload = save_industry_intraday_snapshot(
        rows,
        start_dt=start,
        end_dt=end,
        output_path=target,
        board_state_path=board_state,
        now=datetime(2026, 7, 8, 17, 0, tzinfo=timezone.utc),
    )
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk == payload
    assert payload["schema"] == "industry_intraday_rs_snapshot_v1"
    assert payload["advisory_only"] is True
    assert payload["production_score_effect"] == "none"
    assert payload["source_board_snapshot_id"] == "daily-board-123"
    assert payload["qualified_industry_count"] == 1
    assert payload["candidates"][0]["side_aligned_stock_vs_primary_industry"] == 1.0
    assert not list(tmp_path.glob("*.tmp"))


def test_empty_regeneration_never_clobbers_a_snapshot_with_signal(tmp_path):
    """2026-07-17: an after-close recalculation with an empty bar cache
    overwrote the session's last useful advisory (5/67 qualified) with an
    all-UNAVAILABLE husk. No-signal payloads keep the stored snapshot."""
    import json
    from datetime import timezone

    from ui.services.rs_window_feed import save_industry_intraday_snapshot

    board_state = tmp_path / "industry_board_snapshot.json"
    board_state.write_text(json.dumps({"snapshot_id": "daily-board-123"}), encoding="utf-8")
    target = tmp_path / "industry_intraday_rs_snapshot.json"
    start = datetime(2026, 7, 8, 9, 30)
    end = datetime(2026, 7, 8, 10, 0)
    good_rows = [
        _mover(
            "AAA",
            "LONG",
            2.0,
            industry="Widgets",
            industry_m5_window_pct=1.0,
            industry_m5_vs_spy=0.5,
            industry_m5_members_used=4,
            industry_m5_members_expected=5,
            industry_m5_member_coverage=0.8,
            industry_m5_timestamp_coverage=1.0,
            industry_m5_status="QUALIFIED_ADVISORY",
            industry_m5_first_ts="2026-07-08T09:30",
            industry_m5_last_ts="2026-07-08T10:00",
            industry_m5_snapshot_id="industry-1",
            industry_m5_stock_window_pct=2.0,
            stock_vs_industry_m5=1.0,
            spy_window_pct=0.5,
            advisory_only=True,
        )
    ]
    good = save_industry_intraday_snapshot(
        good_rows,
        start_dt=start,
        end_dt=end,
        output_path=target,
        board_state_path=board_state,
        now=datetime(2026, 7, 8, 17, 0, tzinfo=timezone.utc),
    )
    assert good["qualified_industry_count"] == 1

    # Empty after-close regeneration: the stored snapshot survives.
    kept = save_industry_intraday_snapshot(
        [],
        start_dt=start,
        end_dt=end,
        output_path=target,
        board_state_path=board_state,
        now=datetime(2026, 7, 9, 0, 55, tzinfo=timezone.utc),
    )
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk == good == kept

    # A later regeneration WITH signal still replaces it normally.
    fresh = save_industry_intraday_snapshot(
        good_rows,
        start_dt=start,
        end_dt=end.replace(minute=30),
        output_path=target,
        board_state_path=board_state,
        now=datetime(2026, 7, 9, 1, 0, tzinfo=timezone.utc),
    )
    assert json.loads(target.read_text(encoding="utf-8")) == fresh
    assert fresh["requested_window"]["end"] != good["requested_window"]["end"]

    # And an empty payload with NO stored signal still writes (first run).
    empty_target = tmp_path / "fresh_snapshot.json"
    written = save_industry_intraday_snapshot(
        [],
        start_dt=start,
        end_dt=end,
        output_path=empty_target,
        board_state_path=board_state,
        now=datetime(2026, 7, 9, 1, 5, tzinfo=timezone.utc),
    )
    assert json.loads(empty_target.read_text(encoding="utf-8")) == written


def test_sort_rows_supports_intraday_industry_advisory_fields():
    from ui.services.rs_window_feed import sort_mover_rows

    rows = [
        _mover("LONG_STRONG", "LONG", 0.1, industry_m5_vs_spy=2.0, stock_vs_industry_m5=1.0),
        _mover("SHORT_WEAK", "SHORT", 0.1, industry_m5_vs_spy=-3.0, stock_vs_industry_m5=2.0),
        _mover("SHORT_STRONG", "SHORT", 0.1, industry_m5_vs_spy=4.0, stock_vs_industry_m5=-1.0),
    ]
    assert [row["symbol"] for row in sort_mover_rows(rows, "industry_m5_vs_spy")][:2] == [
        "SHORT_WEAK",
        "LONG_STRONG",
    ]
    assert [row["symbol"] for row in sort_mover_rows(rows, "stock_vs_industry_m5")][:2] == [
        "SHORT_WEAK",
        "LONG_STRONG",
    ]


def test_tracker_numeric_header_sort_keeps_missing_values_last_both_directions():
    if _qt_app() is None:
        return
    from PySide6.QtCore import Qt

    from ui.models.tracker_table_model import (
        ROW_ROLE,
        TrackerSortProxyModel,
        TrackerTableModel,
    )

    model = TrackerTableModel(
        (("name", "Name"), ("score", "Score")),
        [
            {"name": "missing", "score": None},
            {"name": "weak", "score": -2.0},
            {"name": "strong", "score": 3.0},
        ],
        numeric_keys={"score"},
    )
    proxy = TrackerSortProxyModel()
    proxy.setSourceModel(model)

    def names():
        return [proxy.index(row, 0).data(ROW_ROLE)["name"] for row in range(proxy.rowCount())]

    proxy.sort(1, Qt.SortOrder.AscendingOrder)
    assert names() == ["weak", "strong", "missing"]
    proxy.sort(1, Qt.SortOrder.DescendingOrder)
    assert names() == ["strong", "weak", "missing"]


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


def test_chart_auto_window_never_reaches_into_previous_session():
    if _qt_app() is None:
        return
    from ui.widgets.spy_m5_chart import SpyM5Chart

    prior = _chart_bars(20, datetime(2026, 7, 7, 14, 20))
    current = _chart_bars(3, datetime(2026, 7, 8, 9, 30))
    chart = SpyM5Chart()
    chart.set_bars(prior + current, preserve_selection=False)
    start_dt, end_dt = chart.selected_range()
    assert start_dt == current[0]["dt"]
    assert end_dt == current[-1]["dt"]

    # Cross-session replay remains available when the trader deliberately
    # drags the selection rather than relying on Auto's trailing window.
    chart._region.setRegion((18, 22))
    start_dt, end_dt = chart.selected_range()
    assert start_dt.date() < end_dt.date()


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


def test_rs_window_symbol_click_opens_snapshot_once(monkeypatch):
    if _qt_app() is None:
        return
    from ui.panels.rs_window_panel import RsWindowPanel
    import ui.widgets.symbol_snapshot_dialog as snapshot_dialog

    bot = object()

    class _StubService:
        def current_bot(self):
            return bot

    panel = RsWindowPanel(_StubService())
    panel.model.set_rows([{"symbol": "AAA", "side": "LONG", "excess": 1.2}])
    calls = []
    monkeypatch.setattr(
        snapshot_dialog,
        "show_symbol_snapshot",
        lambda owner, symbol, **kwargs: calls.append(
            (owner, symbol, kwargs.get("bot"), kwargs.get("side"))
        ),
    )

    symbol_index = panel.table.model().index(0, 0)
    panel.table.clicked.emit(symbol_index)
    assert calls == [(panel, "AAA", bot, "LONG")]
    panel._open_symbol_snapshot_from_double_click(symbol_index)
    assert len(calls) == 1
    panel.table.clicked.emit(panel.table.model().index(0, 1))
    assert len(calls) == 1


def test_panel_builds_sortable_completed_m5_industry_board(monkeypatch):
    if _qt_app() is None:
        return
    import ui.services.rs_window_feed as feed
    from ui.panels.rs_window_panel import RsWindowPanel

    spy = _chart_bars(30)

    def scaled_bars(multiplier):
        return [
            {
                **bar,
                "open": bar["open"] * multiplier,
                "high": bar["high"] * multiplier,
                "low": bar["low"] * multiplier,
                "close": bar["close"] * multiplier,
            }
            for bar in spy
        ]

    cached = {
        "SPY": spy,
        "AAA": scaled_bars(0.5),
        "BBB": scaled_bars(0.8),
        "CCC": scaled_bars(1.2),
    }
    industry_map = {
        symbol: {
            "industry": "Widgets",
            "industry_primary_source": "classification_definition",
            "industry_member_symbols": ["AAA", "BBB", "CCC"],
            "industry_expected_members": 3,
            "additional_industries": [],
        }
        for symbol in ("AAA", "BBB", "CCC")
    }
    saved = {}

    class _StubBot:
        def spy_m5_chart_bars(self, max_sessions=2):
            return spy

        def rank_window_movers(self, start_dt, end_dt, *, completed_only=False):
            self.completed_only = completed_only
            return {
                "ok": True,
                "spy_pct": 1.0,
                "data_complete_through": end_dt.isoformat(timespec="minutes"),
                "rows": [_mover("AAA", "LONG", 0.5, window_pct=1.5)],
            }

        def cached_m5_window_bars(self, start_dt, end_dt, *, completed_only=True):
            self.cache_completed_only = completed_only
            return cached

    class _StubService:
        def __init__(self):
            self.bot = _StubBot()

        def current_bot(self):
            return self.bot

    monkeypatch.setattr(feed, "load_industry_context_map", lambda: industry_map)
    monkeypatch.setattr(feed, "load_bot_pick_map", lambda: {})
    monkeypatch.setattr(feed, "daily_strength_map", lambda _symbols: {})

    def fake_save(rows, **kwargs):
        saved["rows"] = rows
        saved.update(kwargs)
        return {"snapshot_id": "intraday-test"}

    monkeypatch.setattr(feed, "save_industry_intraday_snapshot", fake_save)

    service = _StubService()
    panel = RsWindowPanel(service)
    panel.refresh_chart()
    panel.rank_selected_window()
    assert service.bot.completed_only is True
    assert service.bot.cache_completed_only is True
    assert panel.industry_model.rowCount() == 1
    industry_row = panel.industry_model.rows()[0]
    assert industry_row["industry"] == "Widgets"
    assert industry_row["industry_m5_status"] == "QUALIFIED_ADVISORY"
    assert saved["industry_rows"][0]["industry"] == "Widgets"
    assert "Advisory fields do not affect live scoring" in panel.status_label.text()


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
