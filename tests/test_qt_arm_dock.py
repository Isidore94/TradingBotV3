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
