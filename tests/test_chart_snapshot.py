"""Symbol snapshot popup: pure chart data (SMA/EMA/VWAP-sigma) + widgets."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chart_snapshot


def _m5_bars(count=30, start=None, *, base=100.0, volume=1000.0):
    start = start or datetime(2026, 7, 8, 9, 30)
    bars = []
    for index in range(count):
        close = base + 0.1 * index
        bars.append(
            {
                "dt": start + timedelta(minutes=5 * index),
                "open": close - 0.05,
                "high": close + 0.12,
                "low": close - 0.12,
                "close": close,
                "volume": volume + 10.0 * (index % 3),
            }
        )
    return bars


def test_sma_series_windows():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert chart_snapshot.sma_series(values, 3) == [None, None, 2.0, 3.0, 4.0]
    assert chart_snapshot.sma_series(values, 1) == values
    assert chart_snapshot.sma_series(values, 10) == [None] * 5


def test_ema_series_matches_pandas_ewm():
    import pandas as pd

    values = [100.0, 101.5, 99.8, 102.2, 103.0, 101.1, 104.4, 105.0]
    expected = pd.Series(values).ewm(span=5, adjust=False).mean().tolist()
    result = chart_snapshot.ema_series(values, 5)
    assert len(result) == len(expected)
    for mine, pandas_value in zip(result, expected):
        assert abs(mine - pandas_value) < 1e-9


def test_session_vwap_final_point_matches_calc_anchored_vwap_bands():
    """The plotted series is the running-deviation sigma variant: its final
    point must equal calc_anchored_vwap_bands over the same bars (invariant:
    never a second sigma formula in the codebase)."""
    import pandas as pd
    from master_avwap_lib.legacy import calc_anchored_vwap_bands

    bars = _m5_bars(24)
    series = chart_snapshot.session_vwap_series(bars)
    frame = pd.DataFrame(
        [
            {
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            }
            for bar in bars
        ]
    )
    vwap, stdev, bands = calc_anchored_vwap_bands(frame, 0)
    assert abs(series["vwap"][-1] - vwap) < 1e-9
    assert abs(series["upper_1"][-1] - bands["UPPER_1"]) < 1e-9
    assert abs(series["lower_1"][-1] - bands["LOWER_1"]) < 1e-9


def test_session_vwap_resets_on_new_session():
    day_one = _m5_bars(12, datetime(2026, 7, 7, 9, 30), base=100.0)
    day_two = _m5_bars(12, datetime(2026, 7, 8, 9, 30), base=140.0)
    series = chart_snapshot.session_vwap_series(day_one + day_two)
    first_new = series["vwap"][12]
    bar = day_two[0]
    tp = (bar["open"] + bar["high"] + bar["low"] + bar["close"]) / 4.0
    assert abs(first_new - tp) < 1e-9  # accumulation restarted
    assert abs(series["upper_1"][12] - series["lower_1"][12]) < 1e-9  # sigma back to ~0


def test_session_vwap_zero_volume_carries_forward():
    bars = _m5_bars(5)
    bars[3]["volume"] = 0.0
    series = chart_snapshot.session_vwap_series(bars)
    assert series["vwap"][3] == series["vwap"][2]
    assert all(value is not None for value in series["vwap"])


def test_build_d1_snapshot_overlays_and_tail():
    start = datetime(2026, 1, 1)
    bars = [
        {
            "dt": start + timedelta(days=index),
            "open": 100.0 + index * 0.2,
            "high": 100.4 + index * 0.2,
            "low": 99.6 + index * 0.2,
            "close": 100.1 + index * 0.2,
            "volume": 1_000.0,
        }
        for index in range(260)
    ]
    snapshot = chart_snapshot.build_d1_snapshot("TEST", sessions=90, loader=lambda _s: bars)
    assert snapshot["timeframe"] == "D1"
    assert len(snapshot["bars"]) == 90
    labels = [overlay["label"] for overlay in snapshot["overlays"]]
    assert labels == ["SMA50", "SMA100", "SMA200", "EMA8", "EMA15", "EMA21"]
    for overlay in snapshot["overlays"]:
        assert len(overlay["values"]) == 90
    # SMA200 computes on the FULL history, so the displayed tail ends with a
    # correct long-lookback value (bars 199+ have one; earlier tail bars not).
    closes = [bar["close"] for bar in bars]
    expected_sma200 = sum(closes[-200:]) / 200.0
    assert abs(snapshot["overlays"][2]["values"][-1] - expected_sma200) < 1e-9
    # SMA50 is defined across the whole displayed tail.
    assert all(value is not None for value in snapshot["overlays"][0]["values"])


def test_build_d1_snapshot_missing_store():
    snapshot = chart_snapshot.build_d1_snapshot("NOPE", loader=lambda _s: [])
    assert snapshot["bars"] == [] and snapshot["note"] == "no daily store"


def test_build_m5_snapshot_overlays():
    bars = _m5_bars(30)
    snapshot = chart_snapshot.build_m5_snapshot("TEST", bars)
    labels = [overlay["label"] for overlay in snapshot["overlays"]]
    assert labels == ["VWAP", "+1σ", "-1σ", "EMA15", "EMA21"]
    for overlay in snapshot["overlays"]:
        assert len(overlay["values"]) == len(bars)
    empty = chart_snapshot.build_m5_snapshot("TEST", [])
    assert empty["bars"] == [] and empty["note"] == "no cached M5 bars"


# ---------------------------------------------------------------------------
# Qt widgets (offscreen; skipped when PySide6 is unavailable)
# ---------------------------------------------------------------------------
def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_candle_chart_renders_bars_and_overlays():
    if _qt_app() is None:
        return
    from ui.widgets.candle_chart import CandleChart

    chart = CandleChart()
    bars = _m5_bars(20)
    snapshot = chart_snapshot.build_m5_snapshot("TEST", bars)
    chart.set_data(snapshot["bars"], snapshot["overlays"], timeframe="m5")
    assert chart.bar_count() == 20
    chart.set_data([], [])
    assert chart.bar_count() == 0


def test_snapshot_dialog_populates_both_charts(monkeypatch):
    if _qt_app() is None:
        return
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotDialog

    daily = [
        {
            "dt": datetime(2026, 1, 1) + timedelta(days=index),
            "open": 50.0,
            "high": 51.0,
            "low": 49.0,
            "close": 50.5,
            "volume": 0.0,
        }
        for index in range(40)
    ]
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _s: daily)

    class StubBot:
        def m5_chart_bars(self, symbol, max_sessions=2):
            return _m5_bars(15)

    dialog = SymbolSnapshotDialog()
    dialog.show_symbol("NVDA", bot=StubBot(), side="LONG")
    assert dialog.d1_chart.bar_count() == 40
    assert dialog.m5_chart.bar_count() == 15
    assert "NVDA" in dialog.windowTitle()
    dialog.close()

    # No bot and no daily store: both notes, no crash.
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _s: [])
    dialog.show_symbol("XXXX", bot=None)
    assert dialog.d1_chart.bar_count() == 0
    assert dialog.m5_chart.bar_count() == 0
    assert dialog.d1_note.isVisibleTo(dialog) and dialog.m5_note.isVisibleTo(dialog)
    dialog.close()


def test_master_setups_double_click_opens_snapshot(monkeypatch):
    if _qt_app() is None:
        return
    from ui.models.setup import SetupRow
    from ui.panels.master_avwap_panel import MasterAvwapPanel
    import ui.widgets.symbol_snapshot_dialog as snapshot_dialog

    panel = MasterAvwapPanel(None)
    panel.set_rows([SetupRow(symbol="NVDA", side="LONG", score=90.0)])
    calls = []
    monkeypatch.setattr(
        snapshot_dialog,
        "show_symbol_snapshot",
        lambda owner, symbol, **kwargs: calls.append((symbol, kwargs.get("side"))),
    )
    panel._open_symbol_snapshot(panel.proxy.index(0, 2))  # symbol column
    assert calls == [("NVDA", "LONG")]
    # The ★/✕ cells are their own click targets: no popup from there.
    panel._open_symbol_snapshot(panel.proxy.index(0, 0))
    panel._open_symbol_snapshot(panel.proxy.index(0, 1))
    assert calls == [("NVDA", "LONG")]


def test_alert_feed_item_symbol_click_signal():
    if _qt_app() is None:
        return
    from ui.models.bounce import BounceAlert
    from ui.widgets.alert_feed_item import AlertFeedItem, _SymbolLabel

    alert = BounceAlert.from_callback("NVDA LONG [A-TIER] test bounce", "bounce")
    item = AlertFeedItem(alert)
    fired = []
    item.symbolClicked.connect(lambda: fired.append(True))
    label = item.findChild(_SymbolLabel)
    assert label is not None
    label.clicked.emit()
    assert fired == [True]
