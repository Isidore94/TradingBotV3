"""Arming from the chart: type-a-symbol, arbitrary levels, quick-fill, inventory.

Before the arm dock there were exactly two ways to arm anything: four session
toggles that only lit up when the review queue handed you a symbol, and a D1
candle's literal high or low. Nothing listed what was already armed, and
persistent level alerts had no cancel path at all.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _m5_bars(count=12, start=100.0):
    base = datetime(2026, 7, 24, 9, 30)
    return [
        {
            "dt": base + timedelta(minutes=5 * index),
            "open": start + index,
            "high": start + index + 1.0,
            "low": start + index - 1.0,
            "close": start + index + 0.5,
            "volume": 1000.0,
        }
        for index in range(count)
    ]


# ---------------------------------------------------------------- quick fill
def test_quick_fill_reads_the_lines_actually_drawn():
    from ui.widgets.arm_bar import quick_fill_value

    bars = _m5_bars(5)
    overlays = [
        {"label": "VWAP", "values": [100.0, 101.0, None, 103.0, 104.5]},
        {"label": "+1σ", "values": [None, None, None, None, 107.25]},
        {"label": "-1σ", "values": [95.0, 95.5, 96.0, 96.5, 97.0]},
    ]
    assert quick_fill_value("last", bars, overlays) == bars[-1]["close"]
    assert quick_fill_value("hod", bars, overlays) == max(b["high"] for b in bars)
    assert quick_fill_value("lod", bars, overlays) == min(b["low"] for b in bars)
    # The last NON-None entry is the line's current level.
    assert quick_fill_value("vwap", bars, overlays) == 104.5
    assert quick_fill_value("upper_1", bars, overlays) == 107.25


def test_quick_fill_is_none_rather_than_wrong_when_a_line_is_absent():
    from ui.widgets.arm_bar import quick_fill_value

    bars = _m5_bars(3)
    assert quick_fill_value("vwap", bars, []) is None
    assert quick_fill_value("vwap", bars, [{"label": "VWAP", "values": [None, None, None]}]) is None
    assert quick_fill_value("last", [], []) is None
    assert quick_fill_value("nonsense", bars, []) is None


# ------------------------------------------------------------ chart on demand
def test_typing_a_symbol_charts_it_without_an_alert():
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    assert panel._current_review_alert is None

    assert panel.chart_symbol("nvda") is True
    assert panel._current_review_alert.symbol == "NVDA"
    # It is not scanner output and must not pollute the alert feed.
    assert panel._alerts == []
    panel.close()


def test_typing_a_symbol_unignores_it():
    """"Remove for today" must not make a symbol permanently un-chartable."""
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    panel._ignore_alert_symbol("NVDA")
    assert "NVDA" in panel._ignored_symbols

    assert panel.chart_symbol("NVDA") is True
    assert "NVDA" not in panel._ignored_symbols
    assert panel._current_review_alert.symbol == "NVDA"
    panel.close()


def test_a_junk_symbol_is_refused():
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    assert panel.chart_symbol("(BULLISH_STRONG)") is False
    assert panel.chart_symbol("") is False
    assert panel._current_review_alert is None
    panel.close()


# ------------------------------------------------------------- arbitrary level
def test_arm_bar_arms_and_disarms_an_arbitrary_price(tmp_path):
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel(d1_level_watches_path=tmp_path / "levels.json")
    panel.chart_symbol("NVDA")

    panel.chart_review.arm_bar.level_input.setValue(187.40)
    panel.chart_review.arm_bar.direction_input.setCurrentIndex(0)  # Above
    panel.chart_review.arm_bar.arm_level_button.click()

    armed = panel.armed_d1_levels()
    assert [(w.symbol, w.direction, w.level) for w in armed] == [("NVDA", "above", 187.40)]

    assert panel.disarm_d1_level_watch("NVDA", "above", 187.40) is True
    assert panel.armed_d1_levels() == []
    # Disarming something that is not armed is a no-op, not an error.
    assert panel.disarm_d1_level_watch("NVDA", "above", 187.40) is False
    panel.close()


def test_click_to_price_fills_the_level_box():
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    bar.set_level(123.45)
    assert bar.level_input.value() == 123.45
    # A nonsensical price is ignored rather than clamped to something wrong.
    bar.set_level(0.0)
    assert bar.level_input.value() == 123.45
    bar.close()


# ------------------------------------------------------------------ inventory
def test_armed_inventory_lists_both_kinds_with_health_and_cancels():
    from chart_watch import ChartWatch, D1LevelWatch
    from ui.widgets.armed_watch_list import ArmedWatchList

    now = datetime(2026, 7, 24, 11, 45)
    widget = ArmedWatchList()
    watches = [
        ChartWatch(symbol="NVDA", kind="new_hod", armed_at=now - timedelta(hours=1), baseline=187.4),
        ChartWatch(symbol="AMD", kind="vwap_bounce", armed_at=now - timedelta(minutes=30)),
    ]
    levels = [
        D1LevelWatch(
            symbol="MU",
            direction="above",
            level=118.42,
            armed_at=now - timedelta(days=3),
            candle_date="2026-07-21",
        )
    ]
    widget.set_watches(
        watches, levels, has_m5_bars=lambda symbol: symbol != "AMD", now=now
    )
    assert widget.table.rowCount() == 3

    captured = []
    widget.disarmWatchRequested.connect(lambda s, k: captured.append(("watch", s, k)))
    widget.disarmLevelRequested.connect(lambda s, d, v: captured.append(("level", s, d, v)))
    last_column = widget.table.columnCount() - 1
    widget._on_cell_clicked(0, last_column)
    widget._on_cell_clicked(2, last_column)
    assert captured == [("watch", "NVDA", "new_hod"), ("level", "MU", "above", 118.42)]
    widget.close()


def test_watch_health_explains_why_a_watch_cannot_fire():
    from ui.widgets.armed_watch_list import (
        HEALTH_NO_BARS,
        HEALTH_OK,
        HEALTH_STALE,
        watch_health,
    )

    now = datetime(2026, 7, 24, 11, 0)
    today = datetime(2026, 7, 24, 9, 40)
    yesterday = datetime(2026, 7, 23, 9, 40)

    assert watch_health("new_hod", True, today, now) == HEALTH_OK
    assert watch_health("new_hod", False, today, now) == HEALTH_NO_BARS
    # Session watches never survive into the next session.
    assert watch_health("new_hod", True, yesterday, now) == HEALTH_STALE
    # Persistent level alerts are exempt from both constraints.
    assert watch_health("d1_level_above", False, yesterday, now) == HEALTH_OK


def test_arming_stays_permissive_when_bars_are_missing():
    """A missing-bars warning must not block the arm.

    chart_watch adopts the first tracked bar as its baseline, so a watch armed
    before the bot has cached the symbol still works.
    """
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    bar.set_enabled_for_symbol(True)
    bar.set_watch_availability(False, "no cached M5 bars")
    button = bar.watch_buttons["new_hod"]
    assert button.isEnabled()
    assert "no cached M5 bars" in button.toolTip()

    bar.set_watch_availability(True)
    assert "no cached M5 bars" not in bar.watch_buttons["new_hod"].toolTip()
    bar.close()


# -------------------------------------------------------------- queue priority
def test_a_fired_watch_jumps_the_review_queue():
    """The chart the trader armed and is waiting on must not queue behind 40."""
    from ui.models.bounce import CHART_WATCH_TAG, BounceAlert
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    for index in range(4):
        panel.add_alert(
            BounceAlert(
                time_text="09:3%d:00" % index,
                symbol=f"SYM{index}",
                side="LONG",
                trigger="bounce",
                raw_text=f"[B-TIER] SYM{index}: bounce",
            )
        )
    assert len(panel._review_queue) == 3  # one is on screen

    panel.add_alert(
        BounceAlert(
            time_text="09:40:00",
            symbol="NVDA",
            side="LONG",
            trigger="New HOD 187.40",
            tag=CHART_WATCH_TAG,
            raw_text="CHART WATCH NVDA (LONG): New HOD 187.40",
        )
    )
    assert panel._review_queue[0].symbol == "NVDA"
    panel.close()


def test_m5_candle_clicks_can_arm_a_level():
    """M5 barClicked was connected to nothing, so intraday levels were unarmable."""
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    widget = SymbolSnapshotWidget()
    widget._symbol = "NVDA"
    bars = _m5_bars(6)
    widget.m5_chart.set_data(bars, [], timeframe="m5")

    captured = []
    widget.d1LevelAlertRequested.connect(
        lambda s, d, level, date: captured.append((s, d, level))
    )
    widget.request_m5_level_alert("above", 3)
    assert captured == [("NVDA", "above", bars[3]["high"])]
    widget.close()


# ------------------------------------------------- phone alerts off a D1 line
# A5: clicking a painted D1 level and arming a PHONE price alert at exactly
# that line. The chart never writes price_alerts.json - it asks, and the
# Alert Center panel (which borrows the desk's single PriceAlertService) does
# the caller-only merge. Arming only: nothing here mutes, suppresses, scores,
# gates or reorders anything.


def _d1_bars(count=20, base=100.0):
    first = datetime(2026, 6, 1)
    return [
        {
            "dt": first + timedelta(days=index),
            "open": base + index * 0.1 - 0.2,
            "high": base + index * 0.1 + 0.4,
            "low": base + index * 0.1 - 0.4,
            "close": base + index * 0.1,
            "volume": 1000.0,
        }
        for index in range(count)
    ]


def _painted_level(level_id, price):
    import chart_levels

    return {
        "id": level_id,
        "family": "d1_horizontal",
        "group": chart_levels.GROUP_HORIZONTAL,
        "price": price,
        "values": None,
        "label": f"L {price:.2f}",
        "color": "chart_green",
        "width": 1.2,
        "dash": False,
        "conviction": 1.0,
    }


def _price_alert_service(monkeypatch, tmp_path, *, engine_enabled=True):
    import price_alerts
    from ui.services.price_alert_service import PriceAlertService

    path = tmp_path / "price_alerts.json"
    original_load = price_alerts.load_price_alerts
    original_save = price_alerts.save_price_alerts
    monkeypatch.setattr(price_alerts, "load_price_alerts", lambda: original_load(path))
    monkeypatch.setattr(
        price_alerts, "save_price_alerts", lambda entries: original_save(entries, path)
    )
    return PriceAlertService(engine_enabled=engine_enabled), path


def _charted_panel_with_a_picked_level(monkeypatch, tmp_path, symbol="NVDA"):
    """Chart a symbol, paint one D1 level, and click it. Returns everything."""
    from ui.panels.alert_center_panel import AlertCenterPanel

    service, path = _price_alert_service(monkeypatch, tmp_path)
    panel = AlertCenterPanel(d1_level_watches_path=tmp_path / "levels.json")
    # Exactly how TradingDeskPanel wires it: the desk owns the one service,
    # the panel borrows it.
    panel.price_alert_service = service
    panel.chart_symbol(symbol)

    review = panel.chart_review
    bars = _d1_bars()
    price = bars[5]["close"]
    review.snapshot._render_snapshots(
        {
            "symbol": symbol,
            "timeframe": "D1",
            "bars": bars,
            "overlays": [],
            "levels": [_painted_level("sr", price)],
            "note": "",
        },
        {"bars": [], "overlays": [], "note": ""},
    )
    chart = review.snapshot.d1_chart
    chart.resize(600, 400)
    chart._select_level_at(5, chart._y(price))
    return panel, service, path, price


def test_clicking_a_painted_level_arms_a_phone_alert_through_the_panel(
    monkeypatch, tmp_path
):
    panel, service, _path, price = _charted_panel_with_a_picked_level(
        monkeypatch, tmp_path
    )
    review = panel.chart_review
    try:
        # The click was recorded, and the affordance only lights up once a
        # line is actually picked.
        assert review.selected_level()[0] == "NVDA"
        assert review.selected_level()[1] == "sr"
        assert review.arm_bar.phone_alert_button.isEnabled()

        review.arm_bar.direction_input.setCurrentIndex(0)  # Above
        review.arm_bar.phone_alert_button.click()

        entries = service.entries()
        assert len(entries) == 1
        assert entries[0]["symbol"] == "NVDA"
        assert entries[0]["above"] == price
        assert entries[0]["armed_above"] is True
        # The other side is left alone, not zeroed.
        assert entries[0]["below"] is None
        assert entries[0]["armed_below"] is False
    finally:
        service.shutdown()
        panel.close()


def test_the_direction_box_picks_which_side_the_line_arms(monkeypatch, tmp_path):
    panel, service, _path, price = _charted_panel_with_a_picked_level(
        monkeypatch, tmp_path
    )
    try:
        panel.chart_review.arm_bar.direction_input.setCurrentIndex(1)  # Below
        panel.chart_review.arm_bar.phone_alert_button.click()

        entries = service.entries()
        assert entries[0]["below"] == price and entries[0]["armed_below"] is True
        assert entries[0]["above"] is None and entries[0]["armed_above"] is False
    finally:
        service.shutdown()
        panel.close()


def test_arming_from_a_line_preserves_the_other_side_and_the_history(
    monkeypatch, tmp_path
):
    """The merge is the Focus board's, key for key - it never rewrites a row."""
    panel, service, _path, price = _charted_panel_with_a_picked_level(
        monkeypatch, tmp_path
    )
    history = [{"date": "2026-08-01", "side": "below", "level": 90.0, "last": 89.5}]
    service.save_entries(
        [
            {
                "symbol": "NVDA",
                "above": None,
                "below": 90.0,
                "armed_above": False,
                "armed_below": False,
                "note": "keep me",
                "history": history,
            }
        ]
    )
    try:
        panel.chart_review.arm_bar.direction_input.setCurrentIndex(0)  # Above
        panel.chart_review.arm_bar.phone_alert_button.click()

        entry = service.entries()[0]
        assert entry["above"] == price and entry["armed_above"] is True
        # Untouched: the fired-and-disarmed cross-down, the note, the log.
        assert entry["below"] == 90.0 and entry["armed_below"] is False
        assert entry["note"] == "keep me"
        assert entry["history"] == history
    finally:
        service.shutdown()
        panel.close()


def test_no_line_picked_means_nothing_is_written(monkeypatch, tmp_path):
    """Clicking away clears the highlight, so there is nothing to arm at."""
    panel, service, _path, price = _charted_panel_with_a_picked_level(
        monkeypatch, tmp_path
    )
    chart = panel.chart_review.snapshot.d1_chart
    try:
        pixel = float(chart.getPlotItem().vb.viewPixelSize()[1])
        chart._select_level_at(5, chart._y(price) + pixel * 60)
        assert chart.selected_level_id() == ""

        panel.chart_review.arm_bar.phone_alert_button.click()
        assert service.entries() == []
    finally:
        service.shutdown()
        panel.close()


def test_the_chart_widgets_never_hold_a_price_alert_service(monkeypatch, tmp_path):
    """Single-writer invariant (plan.md sec 5): only the panel writes the store."""
    from ui.services.price_alert_service import PriceAlertService
    from ui.widgets.alert_chart_review import AlertChartReview
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    review = AlertChartReview()
    try:
        for widget in (review, review.snapshot, review.arm_bar):
            assert not any(
                isinstance(value, PriceAlertService)
                for value in vars(widget).values()
            ), f"{type(widget).__name__} holds a PriceAlertService"
            assert not any(
                isinstance(child, PriceAlertService)
                for child in widget.findChildren(PriceAlertService)
            )
        # And neither module even imports it - a handle cannot appear later
        # without this test's file changing too.
        import ui.widgets.alert_chart_review as review_module
        import ui.widgets.symbol_snapshot_dialog as snapshot_module

        for module in (review_module, snapshot_module):
            assert not any(
                isinstance(value, type) and issubclass(value, PriceAlertService)
                for value in vars(module).values()
            )
        assert SymbolSnapshotWidget is snapshot_module.SymbolSnapshotWidget
    finally:
        review.close()
