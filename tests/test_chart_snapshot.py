"""Symbol snapshot popup: pure chart data (SMA/EMA/VWAP-sigma) + widgets."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

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


def test_session_vwap_handles_single_bar_and_zero_volume_sessions():
    single = _m5_bars(1)
    series = chart_snapshot.session_vwap_series(single)
    expected = sum(single[0][key] for key in ("open", "high", "low", "close")) / 4.0
    assert series["vwap"] == [expected]
    assert series["upper_1"] == series["lower_1"] == [expected]

    zero_bars = _m5_bars(4)
    for bar in zero_bars:
        bar["volume"] = 0.0
    zero_volume = chart_snapshot.session_vwap_series(zero_bars)
    assert zero_volume == {
        "vwap": [None] * 4,
        "upper_1": [None] * 4,
        "lower_1": [None] * 4,
    }


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


def _daily_bars(count, start=datetime(2026, 7, 1)):
    return [
        {
            "dt": start + timedelta(days=index),
            "open": 100.0 + index * 0.2,
            "high": 100.4 + index * 0.2,
            "low": 99.6 + index * 0.2,
            "close": 100.1 + index * 0.2,
            "volume": 1_000.0,
        }
        for index in range(count)
    ]


def test_forming_d1_preview_appends_todays_candle_from_m5():
    """The durable store only gains a session's bar after the close; the M5
    cache fills the gap with a preview candle so the D1 chart shows today."""
    daily = _daily_bars(2)  # store ends 07/02
    m5 = _m5_bars(6, datetime(2026, 7, 2, 9, 30)) + _m5_bars(
        6, datetime(2026, 7, 3, 9, 30), base=105.0
    )
    snapshot = chart_snapshot.build_d1_snapshot(
        "TEST", sessions=90, loader=lambda _s: daily, intraday_bars=m5
    )
    bars = snapshot["bars"]
    assert len(bars) == 3
    preview = bars[-1]
    assert preview["preview"] is True
    assert preview["dt"] == datetime(2026, 7, 3)
    session = m5[6:]  # only the NEWEST intraday session aggregates
    assert preview["open"] == session[0]["open"]
    assert preview["high"] == max(bar["high"] for bar in session)
    assert preview["low"] == min(bar["low"] for bar in session)
    assert preview["close"] == session[-1]["close"]
    assert preview["volume"] == pytest.approx(sum(bar["volume"] for bar in session))
    # Indicators stay computed on completed sessions only: every overlay
    # carries a trailing None at the preview candle (the line breaks there
    # instead of previewing a moving average off a partial day).
    for overlay in snapshot["overlays"]:
        assert len(overlay["values"]) == len(bars)
        assert overlay["values"][-1] is None


def test_anchored_vwap_band_series_final_point_matches_calc_anchored_vwap_bands():
    """The drawn AVWAPE lines are the running-deviation σ variant: their final
    point must equal calc_anchored_vwap_bands over the same bars (invariant:
    never a second σ formula in the codebase)."""
    import pandas as pd
    from master_avwap_lib.legacy import calc_anchored_vwap_bands

    bars = _daily_bars(30)
    anchor = 7
    series = chart_snapshot.anchored_vwap_band_series(bars, anchor)
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
    vwap, stdev, bands = calc_anchored_vwap_bands(frame, anchor)
    assert series["avwap"][-1] == pytest.approx(vwap)
    for k in (1, 2, 3):
        assert series[f"upper_{k}"][-1] == pytest.approx(bands[f"UPPER_{k}"])
        assert series[f"lower_{k}"][-1] == pytest.approx(bands[f"LOWER_{k}"])
    # Pre-anchor bars carry no value (the chart breaks the line there).
    assert series["avwap"][: anchor] == [None] * anchor
    assert series["avwap"][anchor] is not None
    # An out-of-range anchor yields an all-None series, never an exception.
    assert chart_snapshot.anchored_vwap_band_series(bars, len(bars))["avwap"] == [None] * 30


def test_build_d1_snapshot_draws_avwape_bands_from_the_anchor():
    daily = _daily_bars(20)  # sessions 07/01..07/20
    anchor_date = daily[4]["dt"].date()  # 07/05
    snapshot = chart_snapshot.build_d1_snapshot(
        "TEST",
        sessions=90,
        loader=lambda _s: daily,
        anchor_resolver=lambda _s: anchor_date,
    )
    labels = [overlay["label"] for overlay in snapshot["overlays"]]
    # One legend entry for the line, ONE shared entry for all six bands.
    assert labels.count("AVWAPE") == 1
    assert labels.count("AVWAPE ±1-3σ") == 6
    assert snapshot["avwape_anchor"] == anchor_date.isoformat()
    avwape = next(o for o in snapshot["overlays"] if o["label"] == "AVWAPE")
    assert avwape["values"][3] is None  # before the anchor
    assert avwape["values"][4] is not None
    assert len(avwape["values"]) == len(snapshot["bars"])

    # No anchor (not in the earnings cache): no AVWAPE overlays, empty stamp.
    bare = chart_snapshot.build_d1_snapshot(
        "TEST", loader=lambda _s: daily, anchor_resolver=lambda _s: None
    )
    assert all("AVWAPE" not in o["label"] for o in bare["overlays"])
    assert bare["avwape_anchor"] == ""

    # An anchor date with no stored candle mirrors the runner: no lines.
    from datetime import date as _date

    missing = chart_snapshot.build_d1_snapshot(
        "TEST", loader=lambda _s: daily, anchor_resolver=lambda _s: _date(2020, 1, 1)
    )
    assert all("AVWAPE" not in o["label"] for o in missing["overlays"])


def test_forming_d1_preview_skipped_when_store_is_current():
    """After the close the store holds the session itself - the real bar wins
    and no preview is appended (nor with an empty intraday cache)."""
    daily = _daily_bars(3)  # store ends 07/03
    m5 = _m5_bars(6, datetime(2026, 7, 3, 9, 30))
    snapshot = chart_snapshot.build_d1_snapshot(
        "TEST", loader=lambda _s: daily, intraday_bars=m5
    )
    assert len(snapshot["bars"]) == 3
    assert all(not bar.get("preview") for bar in snapshot["bars"])
    assert chart_snapshot.forming_d1_bar(daily, []) is None
    assert chart_snapshot.forming_d1_bar([], m5) is not None  # no store: still a candle


def test_load_d1_bars_resolves_dotted_alias_and_caches_by_mtime(tmp_path, monkeypatch):
    import os

    import pandas as pd
    import setup_playbook_study
    from master_avwap_lib import legacy as master_legacy

    stored = tmp_path / "BF-B.parquet"
    stored.write_bytes(b"first")
    monkeypatch.setattr(master_legacy, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=80, freq="B"),
            "open": [100.0] * 80,
            "high": [101.0] * 80,
            "low": [99.0] * 80,
            "close": [100.5] * 80,
            "volume": [1000.0] * 80,
        }
    )
    loaded_stems = []

    def fake_load(stem):
        loaded_stems.append(stem)
        return frame

    monkeypatch.setattr(setup_playbook_study, "_load_daily_frame", fake_load)
    chart_snapshot._daily_bars_cache.clear()
    try:
        assert len(chart_snapshot.load_d1_bars("BF.B")) == 80
        assert loaded_stems == ["BF-B"]
        assert len(chart_snapshot.load_d1_bars("BF.B")) == 80
        assert loaded_stems == ["BF-B"]  # unchanged mtime: no parquet read

        old_mtime = stored.stat().st_mtime_ns
        stored.write_bytes(b"changed")
        os.utime(stored, ns=(old_mtime + 1_000_000_000, old_mtime + 1_000_000_000))
        assert len(chart_snapshot.load_d1_bars("BF.B")) == 80
        assert loaded_stems == ["BF-B", "BF-B"]
    finally:
        chart_snapshot._daily_bars_cache.clear()


def test_build_m5_snapshot_overlays():
    bars = _m5_bars(30)
    snapshot = chart_snapshot.build_m5_snapshot("TEST", bars)
    labels = [overlay["label"] for overlay in snapshot["overlays"]]
    assert labels == ["VWAP", "+1σ", "-1σ", "EMA15", "EMA21"]
    for overlay in snapshot["overlays"]:
        assert len(overlay["values"]) == len(bars)
    empty = chart_snapshot.build_m5_snapshot("TEST", [])
    assert empty["bars"] == [] and empty["note"] == "no cached M5 bars"


def test_zero_volume_m5_legend_explains_missing_vwap():
    if _qt_app() is None:
        return
    from ui.widgets.symbol_snapshot_dialog import _legend_html

    bars = _m5_bars(4)
    for bar in bars:
        bar["volume"] = 0.0
    snapshot = chart_snapshot.build_m5_snapshot("TEST", bars)
    legend = _legend_html(
        "TEST · M5",
        snapshot["overlays"],
        missing_reason="needs positive cached volume",
    )
    assert "VWAP, +1σ, -1σ: needs positive cached volume" in legend


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
    from ui.widgets.candle_chart import CandleChart, _time_ticks

    chart = CandleChart()
    bars = _m5_bars(20)
    snapshot = chart_snapshot.build_m5_snapshot("TEST", bars)
    chart.set_data(snapshot["bars"], snapshot["overlays"], timeframe="m5")
    assert chart.bar_count() == 20
    two_sessions = _m5_bars(78) + _m5_bars(
        78,
        start=datetime(2026, 7, 9, 9, 30),
    )
    ticks = _time_ticks(two_sessions, "m5")
    assert len(ticks) <= 7
    assert ticks[0][0] == 0 and ticks[-1][0] == 155
    assert ticks[3] == (78, "07/09 09:30")
    # A forming preview candle draws (hollow) like any other bar.
    chart.set_data(bars + [dict(bars[-1], preview=True)], [], timeframe="d1")
    assert chart.bar_count() == 21
    assert chart.bar_at(20)["preview"] is True
    chart.set_data([], [])
    assert chart.bar_count() == 0


def test_price_axis_labels_log_positions_with_round_prices():
    from ui.widgets.candle_chart import _nice_price_ticks, _to_log_price

    # A 40 -> 90 daily range: ticks must be round prices, not the 39.8/44.7
    # levels evenly spaced log coordinates would land on.
    prices, step = _nice_price_ticks(40.0, 90.0)
    assert step == 10.0
    assert prices == [40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    # And they sit at their log positions, so the grid lines match the labels.
    assert _to_log_price(100.0) == pytest.approx(2.0)


def test_candle_chart_log_scaling_round_trips_clicked_prices():
    if _qt_app() is None:
        return
    from ui.widgets.candle_chart import CandleChart

    chart = CandleChart()
    bars = _m5_bars(20)
    chart.set_data(bars, [], timeframe="m5")
    assert chart.is_log_scaled()
    # Equal percentage moves must occupy equal vertical distance.
    assert chart._y(80.0) - chart._y(40.0) == pytest.approx(chart._y(40.0) - chart._y(20.0))
    # A clicked level is armed as a real price, never as its log coordinate.
    assert chart.price_at(chart._y(70.75)) == pytest.approx(70.75)

    chart.set_log_y(False)
    assert not chart.is_log_scaled()
    assert chart.price_at(70.75) == pytest.approx(70.75)
    assert chart.bar_count() == 20


def test_candle_chart_falls_back_to_linear_on_non_positive_prices():
    if _qt_app() is None:
        return
    from ui.widgets.candle_chart import CandleChart

    chart = CandleChart()
    bars = _m5_bars(5)
    bars[2]["low"] = 0.0  # a bad cache row: log10 is undefined here
    chart.set_data(bars, [], timeframe="m5")
    assert not chart.is_log_scaled()
    assert chart.price_at(12.5) == pytest.approx(12.5)


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
    assert dialog.width() >= 1180
    assert dialog.d1_legend.wordWrap() and dialog.m5_legend.wordWrap()
    dialog.show_symbol("NVDA", bot=StubBot(), side="LONG")
    # 40 stored sessions + today's forming candle synthesized from the M5
    # cache (the store itself only catches up after the close).
    assert dialog.d1_chart.bar_count() == 41
    assert dialog.d1_chart.bar_at(40)["preview"] is True
    assert "forming" in dialog.d1_legend.text()
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


def test_snapshot_widget_refresh_renders_only_on_change(monkeypatch):
    """The 30s refresh path: unchanged caches leave the widgets (and any
    pan/zoom) alone; a new M5 bar redraws both panes, and the forming D1
    candle tracks the newest session close."""
    if _qt_app() is None:
        return
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

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
        def __init__(self):
            self.bars = _m5_bars(10)

        def m5_chart_bars(self, symbol, max_sessions=2):
            return list(self.bars)

    bot = StubBot()
    widget = SymbolSnapshotWidget()
    widget.set_symbol("NVDA", bot=bot)
    assert widget.m5_chart.bar_count() == 10
    assert widget.d1_chart.bar_count() == 41  # 40 stored + forming preview

    assert widget.refresh() is False  # nothing changed: no re-render

    bot.bars = _m5_bars(11)
    assert widget.refresh() is True
    assert widget.m5_chart.bar_count() == 11
    assert widget.d1_chart.bar_count() == 41
    preview = widget.d1_chart.bar_at(40)
    assert preview["preview"] is True
    assert preview["close"] == pytest.approx(bot.bars[-1]["close"])


def test_snapshot_dialog_reuses_owner_child_without_stealing_editor_focus(monkeypatch):
    app = _qt_app()
    if app is None:
        return
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget
    from ui.widgets.symbol_snapshot_dialog import show_symbol_snapshot

    monkeypatch.setattr(
        chart_snapshot,
        "build_d1_snapshot",
        lambda symbol, **_kwargs: {
            "symbol": symbol,
            "timeframe": "D1",
            "bars": [],
            "overlays": [],
            "note": "no daily store",
        },
    )
    owner = QWidget()
    layout = QVBoxLayout(owner)
    editor = QPlainTextEdit("AAPL")
    layout.addWidget(editor)
    owner.show()
    editor.setFocus()
    app.processEvents()

    first = show_symbol_snapshot(owner, "AAPL")
    app.processEvents()
    assert first.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert first.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert show_symbol_snapshot(owner, "MSFT") is first
    assert first.parent() is owner
    first.close()
    owner.close()


def test_master_setups_symbol_click_opens_snapshot(monkeypatch):
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
    symbol_index = panel.proxy.index(0, 2)
    panel.table.clicked.emit(symbol_index)
    assert calls == [("NVDA", "LONG")]
    # A double-click emits after the first single click; it must not reopen.
    panel._open_symbol_snapshot_from_double_click(symbol_index)
    assert calls == [("NVDA", "LONG")]
    # Existing double-click behavior remains on the rest of the setup row.
    panel._open_symbol_snapshot_from_double_click(panel.proxy.index(0, 4))
    assert calls == [("NVDA", "LONG"), ("NVDA", "LONG")]
    # The ★/✕ cells are their own click targets: no popup from there.
    panel._open_symbol_snapshot(panel.proxy.index(0, 0))
    panel._open_symbol_snapshot(panel.proxy.index(0, 1))
    assert calls == [("NVDA", "LONG"), ("NVDA", "LONG")]


def test_master_setups_popup_carries_chart_watch_host(monkeypatch):
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
        lambda owner, symbol, **kwargs: calls.append((symbol, kwargs.get("watch_host"))),
    )
    symbol_index = panel.proxy.index(0, 2)

    # Standalone (no desk wiring): the popup opens without the action row.
    panel._open_symbol_snapshot(symbol_index)
    assert calls == [("NVDA", None)]

    # Wired by the desk: the Alert Center rides along as the watch host.
    host = object()
    panel.set_chart_watch_host(host)
    panel._open_symbol_snapshot(symbol_index)
    assert calls == [("NVDA", None), ("NVDA", host)]


def _focus_service(tmp_path):
    from focus_picks import FocusPickStore
    from ui.services.focus_service import FocusService

    return FocusService(
        FocusPickStore(
            focus_longs_path=tmp_path / "focus_longs.txt",
            focus_shorts_path=tmp_path / "focus_shorts.txt",
            longs_path=tmp_path / "longs.txt",
            shorts_path=tmp_path / "shorts.txt",
            membership_path=tmp_path / "focus_pick_membership.json",
        )
    )


def test_snapshot_popup_dislike_advances_to_next_chart(monkeypatch, tmp_path):
    if _qt_app() is None:
        return
    from review_events import load_review_events
    from ui.models.setup import SetupRow
    import ui.panels.master_avwap_panel as panel_module

    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _s: [])
    events_path = tmp_path / "events.jsonl"
    panel = panel_module.MasterAvwapPanel(
        _focus_service(tmp_path), review_events_path=events_path
    )
    panel.set_rows(
        [
            SetupRow(symbol="NVDA", side="LONG", score=90.0, bucket="favorite_setup"),
            SetupRow(symbol="TSLA", side="SHORT", score=80.0, bucket="high_conviction"),
        ]
    )
    symbol_index = panel.proxy.index(0, 2)
    panel.table.setCurrentIndex(symbol_index)
    panel._open_symbol_snapshot(symbol_index)
    dialog = panel._symbol_snapshot_dialog
    assert dialog._symbol == "NVDA"
    # Review host present: the ✕ shows; no watch host, so the focus/watch
    # toggles stay hidden instead of sitting there dead.
    assert dialog.dislike_button.isVisibleTo(dialog)
    assert not dialog.d1_focus_button.isVisibleTo(dialog)

    class _Prompt:
        @staticmethod
        def getMultiLineText(*_args, **_kwargs):
            return ("too extended from the level", True)

    monkeypatch.setattr(panel_module, "QInputDialog", _Prompt)
    dialog._review_dislike()
    # The dislike logged with the row's swing context and the popup advanced.
    rows = load_review_events(events_path)
    assert [row["action"] for row in rows] == ["dislike"]
    assert rows[0]["symbol"] == "NVDA"
    assert rows[0]["bucket"] == "favorite_setup"
    assert rows[0]["detail"]["reason"] == "too extended from the level"
    assert dialog._symbol == "TSLA"

    # A cancelled reason prompt = no dislike, no advance.
    class _Cancel:
        @staticmethod
        def getMultiLineText(*_args, **_kwargs):
            return ("", False)

    monkeypatch.setattr(panel_module, "QInputDialog", _Cancel)
    dialog._review_dislike()
    assert dialog._symbol == "TSLA"
    assert len(load_review_events(events_path)) == 1


def test_snapshot_popup_d1_focus_add_advances_to_next_chart(monkeypatch, tmp_path):
    if _qt_app() is None:
        return
    from ui.models.setup import SetupRow
    import ui.panels.master_avwap_panel as panel_module

    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _s: [])

    class _WatchHost:
        def __init__(self):
            self.toggles = []
            self.next_state = True

        def toggle_d1_focus(self, symbol, side="", *, origin="", context=""):
            self.toggles.append((symbol, side))
            return self.next_state

        def armed_watch_kinds(self, _symbol):
            return set()

        def is_d1_focus_active(self, _symbol, _side=""):
            return False

        def is_m5_focus(self, _symbol, _side=""):
            return False

    host = _WatchHost()
    panel = panel_module.MasterAvwapPanel(
        _focus_service(tmp_path), review_events_path=tmp_path / "events.jsonl"
    )
    panel.set_chart_watch_host(host)
    panel.set_rows(
        [
            SetupRow(symbol="NVDA", side="LONG", score=90.0),
            SetupRow(symbol="TSLA", side="SHORT", score=80.0),
        ]
    )
    symbol_index = panel.proxy.index(0, 2)
    panel.table.setCurrentIndex(symbol_index)
    panel._open_symbol_snapshot(symbol_index)
    dialog = panel._symbol_snapshot_dialog
    # Both hosts wired: ✕ and the focus/watch toggles all show.
    assert dialog.dislike_button.isVisibleTo(dialog)
    assert dialog.d1_focus_button.isVisibleTo(dialog)

    dialog._toggle_d1_focus()
    assert host.toggles == [("NVDA", "LONG")]
    assert dialog._symbol == "TSLA"  # decision made -> next chart

    # Toggling OFF is a correction, not a decision: stay on this chart.
    host.next_state = False
    dialog._toggle_d1_focus()
    assert host.toggles[-1] == ("TSLA", "SHORT")
    assert dialog._symbol == "TSLA"


def test_master_setups_space_advances_visible_rows_and_opens_snapshot(monkeypatch):
    app = _qt_app()
    if app is None:
        return
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from ui.models.setup import SetupRow
    from ui.panels.master_avwap_panel import MasterAvwapPanel
    import ui.widgets.symbol_snapshot_dialog as snapshot_dialog

    panel = MasterAvwapPanel(None)
    panel.set_rows(
        [
            SetupRow(symbol="NVDA", side="LONG", score=90.0),
            SetupRow(symbol="TSLA", side="SHORT", score=80.0),
        ]
    )
    calls = []
    monkeypatch.setattr(
        snapshot_dialog,
        "show_symbol_snapshot",
        lambda owner, symbol, **kwargs: calls.append((symbol, kwargs.get("side"))),
    )
    panel.show()
    first_symbol = panel.proxy.index(0, 2)
    panel.table.setCurrentIndex(first_symbol)
    panel.table.setFocus()
    app.processEvents()

    QTest.keyClick(panel.table, Qt.Key.Key_Space)
    assert panel.table.currentIndex().row() == 1
    assert calls == [("TSLA", "SHORT")]

    QTest.keyClick(panel.table, Qt.Key.Key_Space)
    assert panel.table.currentIndex().row() == 0
    assert calls == [("TSLA", "SHORT"), ("NVDA", "LONG")]
    panel.close()


def test_alert_feed_symbol_click_does_not_propagate_to_row():
    if _qt_app() is None:
        return
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from ui.models.bounce import BounceAlert
    from ui.panels.alert_center_panel import _ClickableItem
    from ui.widgets.alert_feed_item import _SymbolLabel

    alert = BounceAlert.from_callback("NVDA LONG [A-TIER] test bounce", "bounce")
    item = _ClickableItem(alert)
    fired = []
    item.symbolClicked.connect(lambda _alert: fired.append("symbol"))
    item.clicked.connect(lambda _alert: fired.append("row"))
    item.show()
    _qt_app().processEvents()
    label = item.findChild(_SymbolLabel)
    assert label is not None
    QTest.mouseClick(label, Qt.MouseButton.LeftButton)
    assert fired == ["symbol"]
    item.close()


def test_watchlist_symbol_double_click_preserves_text_selection():
    if _qt_app() is None:
        return
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from ui.panels.watchlists_panel import _SymbolTextEdit

    editor = _SymbolTextEdit()
    editor.setPlainText("AAPL\nMSFT")
    editor.resize(300, 100)
    activated = []
    editor.symbolActivated.connect(activated.append)
    editor.show()
    _qt_app().processEvents()
    QTest.mouseDClick(
        editor.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(20, 10),
    )
    assert activated == ["AAPL"]
    assert editor.textCursor().selectedText() == "AAPL"
    editor.close()
