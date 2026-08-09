"""Symbol snapshot popup: pure chart data (SMA/EMA/VWAP-sigma) + widgets."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chart_snapshot


def _seed_d1_store(symbol, bars):
    """Preload the chart bar cache so no test touches the real daily store.

    The chart builds snapshots through ``ui.services.bar_cache`` now, so
    patching ``chart_snapshot.load_d1_bars`` no longer intercepts anything -
    the loader seam the service passes in wins. Seeding the store is the
    equivalent hook, and it exercises the real path end to end.
    """
    import numpy as np
    from ui.services.bar_cache import BarSeries, shared_store

    def column(key):
        return np.array([float(bar[key]) for bar in bars], dtype="float64")

    shared_store().put(
        BarSeries(
            symbol=str(symbol).strip().upper(),
            dt=np.array([bar["dt"] for bar in bars], dtype="datetime64[ns]"),
            open=column("open"),
            high=column("high"),
            low=column("low"),
            close=column("close"),
            volume=np.array(
                [float(bar.get("volume") or 0.0) for bar in bars], dtype="float64"
            ),
            source="memory",
        )
    )


def _pump_until(predicate, timeout=10.0):
    """Spin the event loop until ``predicate`` holds; chart builds are async."""
    import time

    app = _qt_app()
    if app is None:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


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
    # `now` is pinned past the whole series so this stays a pure tail/overlay
    # test: with a live clock, whichever bar happened to be dated today would
    # be split off as the forming candle and the tail would read 90 + preview.
    snapshot = chart_snapshot.build_d1_snapshot(
        "TEST", sessions=90, loader=lambda _s: bars, now=datetime(2026, 12, 31, 15, 0)
    )
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
    prev_date = daily[1]["dt"].date()  # 07/02
    snapshot = chart_snapshot.build_d1_snapshot(
        "TEST",
        sessions=90,
        loader=lambda _s: daily,
        anchor_resolver=lambda _s: (anchor_date, prev_date),
    )
    labels = [overlay["label"] for overlay in snapshot["overlays"]]
    # One legend entry for the line, one per σ band pair, one for prev.
    assert labels.count("AVWAPE") == 1
    for k in (1, 2, 3):
        assert labels.count(f"±{k}σ") == 2
    assert labels.count("AVWAPE prev") == 1
    assert snapshot["avwape_anchor"] == anchor_date.isoformat()
    assert snapshot["avwape_prev_anchor"] == prev_date.isoformat()
    by_label = {o["label"]: o for o in snapshot["overlays"]}
    # Fixed color assignments (user-specified): white line, yellow prev,
    # blue/green/light-blue bands, dotted SMAs / solid EMAs.
    assert by_label["AVWAPE"]["color"] == "chart_white"
    assert by_label["AVWAPE prev"]["color"] == "chart_yellow"
    assert by_label["±1σ"]["color"] == "chart_blue"
    assert by_label["±2σ"]["color"] == "chart_green"
    assert by_label["±3σ"]["color"] == "chart_light_blue"
    assert by_label["SMA200"]["color"] == "chart_purple" and by_label["SMA200"]["dash"] == "dot"
    assert by_label["SMA100"]["color"] == "chart_pink" and by_label["SMA100"]["dash"] == "dot"
    assert by_label["SMA50"]["color"] == "chart_light_blue"
    assert by_label["EMA8"]["color"] == "chart_grey" and by_label["EMA8"]["dash"] is False
    assert by_label["EMA15"]["color"] == "chart_pink"
    assert by_label["EMA21"]["color"] == "chart_yellow"
    avwape = by_label["AVWAPE"]
    assert avwape["values"][3] is None  # before the anchor
    assert avwape["values"][4] is not None
    assert len(avwape["values"]) == len(snapshot["bars"])
    prev_line = by_label["AVWAPE prev"]
    assert prev_line["values"][0] is None
    assert prev_line["values"][1] is not None

    # No anchor (not in the earnings cache): no AVWAPE overlays, empty stamps.
    bare = chart_snapshot.build_d1_snapshot(
        "TEST", loader=lambda _s: daily, anchor_resolver=lambda _s: (None, None)
    )
    assert all("AVWAPE" not in o["label"] for o in bare["overlays"])
    assert bare["avwape_anchor"] == "" and bare["avwape_prev_anchor"] == ""

    # A bare current-anchor resolver (no tuple) still works; an anchor date
    # with no stored candle mirrors the runner: no lines.
    from datetime import date as _date

    single = chart_snapshot.build_d1_snapshot(
        "TEST", loader=lambda _s: daily, anchor_resolver=lambda _s: anchor_date
    )
    assert [o["label"] for o in single["overlays"]].count("AVWAPE") == 1
    missing = chart_snapshot.build_d1_snapshot(
        "TEST", loader=lambda _s: daily, anchor_resolver=lambda _s: (_date(2020, 1, 1), None)
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


def test_chart_reuses_its_items_across_symbol_switches():
    """Part C rule C5: update items in place, never rebuild them per switch.

    Rebuilding 14 curve items was measured at 10.6ms of a 13ms set_data - the
    largest single cost on the chart path - so item identity across switches
    is the thing worth pinning down, not just the pixels.
    """
    if _qt_app() is None:
        return
    from ui.widgets.candle_chart import CandleChart

    chart = CandleChart()
    first = chart_snapshot.build_m5_snapshot("A", _m5_bars(20))
    chart.set_data(first["bars"], first["overlays"], timeframe="m5")
    plot = chart.getPlotItem()
    items = plot.listDataItems()
    assert len(items) == len(first["overlays"])
    identities = [id(item) for item in items]
    candles = chart._candles

    second = chart_snapshot.build_m5_snapshot("B", _m5_bars(35, base=250.0))
    for _ in range(5):
        chart.set_data(second["bars"], second["overlays"], timeframe="m5")
        chart.set_data(first["bars"], first["overlays"], timeframe="m5")
    assert [id(item) for item in plot.listDataItems()] == identities
    assert chart._candles is candles, "the candle item must survive a switch"

    # A snapshot with fewer overlays hides the spares rather than destroying
    # them, so the next symbol needing them pays nothing.
    chart.set_data(first["bars"], first["overlays"][:2], timeframe="m5")
    assert [id(item) for item in plot.listDataItems()] == identities
    assert sum(1 for item in plot.listDataItems() if item.isVisible()) == 2

    # Emptying the chart must not leave a stale curve painted over nothing.
    chart.set_data([], [])
    assert chart.bar_count() == 0
    assert not any(item.isVisible() for item in plot.listDataItems())


def test_chart_drops_antialiasing_while_the_view_is_dragged():
    """C5: antialiasing off during interaction, restored once it settles."""
    app = _qt_app()
    if app is None:
        return
    from ui.widgets.candle_chart import CandleChart

    chart = CandleChart()
    snapshot = chart_snapshot.build_m5_snapshot("A", _m5_bars(20))
    chart.set_data(snapshot["bars"], snapshot["overlays"], timeframe="m5")
    curves = [item.curve for item in chart.getPlotItem().listDataItems()]
    assert all(curve.opts["antialias"] for curve in curves)

    chart._on_manual_range_change()
    assert not any(curve.opts["antialias"] for curve in curves)
    # set_data's own range calls must NOT count as interaction.
    chart._set_overlay_antialias(True)
    chart.set_data(snapshot["bars"], snapshot["overlays"], timeframe="m5")
    assert all(item.curve.opts["antialias"] for item in chart.getPlotItem().listDataItems())


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
    _seed_d1_store("NVDA", daily)

    class StubBot:
        def m5_chart_bars(self, symbol, max_sessions=2):
            return _m5_bars(15)

    dialog = SymbolSnapshotDialog()
    assert dialog.width() >= 1180
    assert dialog.d1_legend.wordWrap() and dialog.m5_legend.wordWrap()
    dialog.show_symbol("NVDA", bot=StubBot(), side="LONG")
    # The build is off-thread: the charts fill when the worker delivers, and
    # the GUI thread never blocked waiting for it.
    assert _pump_until(lambda: dialog.d1_chart.bar_count() == 41)
    # 40 stored sessions + today's forming candle synthesized from the M5
    # cache (the store itself only catches up after the close).
    assert dialog.d1_chart.bar_at(40)["preview"] is True
    assert "forming" in dialog.d1_legend.text()
    assert dialog.m5_chart.bar_count() == 15
    assert "NVDA" in dialog.windowTitle()
    dialog.close()

    # No bot and no daily store: both notes, no crash. Wait for the empty
    # payload itself - the loading skeleton also shows the notes, so keying
    # on note visibility would pass before the real answer arrived.
    dialog.show_symbol("XXXX", bot=None)
    assert _pump_until(lambda: dialog.d1_chart.bar_count() == 0)
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
    _seed_d1_store("NVDA", daily)

    class StubBot:
        def __init__(self):
            self.bars = _m5_bars(10)

        def m5_chart_bars(self, symbol, max_sessions=2):
            return list(self.bars)

    bot = StubBot()
    widget = SymbolSnapshotWidget()
    renders: list[str] = []
    widget.snapshotRendered.connect(renders.append)
    widget.set_symbol("NVDA", bot=bot)
    assert _pump_until(lambda: widget.m5_chart.bar_count() == 10)
    assert widget.d1_chart.bar_count() == 41  # 40 stored + forming preview

    # Unchanged caches must not repaint - a repaint would throw away the
    # trader's pan/zoom on every 30s tick. The rebuild still runs; what must
    # not happen is a second render.
    before = len(renders)
    widget.refresh()
    _pump_until(lambda: False, timeout=0.4)
    assert len(renders) == before

    bot.bars = _m5_bars(11)
    widget.refresh()
    assert _pump_until(lambda: widget.m5_chart.bar_count() == 11)
    assert len(renders) == before + 1
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


# ---------------------------------------------------------------------------
# "Sometimes the latest D1 bar isn't loaded in" (live complaint, 2026-07-30):
# a symbol outside the current scan set has nothing refreshing its durable
# store and no cached M5 bars to preview from, so the chart tail silently
# went stale. The predicate below is the noticing; the widget backfill test
# proves the off-paint refresh and its cooldown.
# ---------------------------------------------------------------------------
def _d1_row(day, **extra):
    row = {
        "dt": datetime(day.year, day.month, day.day),
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 1000.0,
    }
    row.update(extra)
    return row


def test_latest_completed_session_date_covers_the_clock(monkeypatch):
    from datetime import date, timezone

    monkeypatch.setenv("TRADINGBOT_MARKET_TIMEZONE", "America/Los_Angeles")
    tz = timezone(timedelta(hours=-7))
    f = chart_snapshot.latest_completed_session_date

    # Wednesday during RTH -> Tuesday is the last completed session.
    assert f(datetime(2026, 7, 29, 10, 0, tzinfo=tz)) == date(2026, 7, 28)
    # Wednesday after the 13:00 close -> Wednesday itself.
    assert f(datetime(2026, 7, 29, 14, 0, tzinfo=tz)) == date(2026, 7, 29)
    # Saturday -> Friday; Monday pre-open -> Friday.
    assert f(datetime(2026, 8, 1, 11, 0, tzinfo=tz)) == date(2026, 7, 31)
    assert f(datetime(2026, 8, 3, 5, 0, tzinfo=tz)) == date(2026, 7, 31)


def test_d1_store_is_stale_notices_a_missing_latest_bar(monkeypatch):
    from datetime import date, timezone

    monkeypatch.setenv("TRADINGBOT_MARKET_TIMEZONE", "America/Los_Angeles")
    tz = timezone(timedelta(hours=-7))
    during = datetime(2026, 7, 29, 10, 0, tzinfo=tz)   # Wed RTH
    after = datetime(2026, 7, 29, 14, 0, tzinfo=tz)    # Wed post-close

    tue = _d1_row(date(2026, 7, 28))
    mon = _d1_row(date(2026, 7, 27))
    wed = _d1_row(date(2026, 7, 29))

    # Through Tuesday during Wednesday RTH: current (today is preview's job).
    assert chart_snapshot.d1_store_is_stale([mon, tue], now=during) is False
    # Ends Monday during Wednesday RTH: Tuesday's bar is MISSING -> stale.
    assert chart_snapshot.d1_store_is_stale([mon], now=during) is True
    # After Wednesday's close the store must hold Wednesday itself.
    assert chart_snapshot.d1_store_is_stale([mon, tue], now=after) is True
    assert chart_snapshot.d1_store_is_stale([mon, tue, wed], now=after) is False
    # A preview candle is display-only, never stored evidence.
    preview = _d1_row(date(2026, 7, 29), preview=True)
    assert chart_snapshot.d1_store_is_stale([mon, tue, preview], now=after) is True
    # An empty store is the out-of-universe case, not staleness.
    assert chart_snapshot.d1_store_is_stale([], now=after) is False


# ---------------------------------------------------------------------------
# "It doesn't ALWAYS show me the latest D1 candle" (live complaint, 2026-08-05,
# RY: a big green day rendered as a tail of little red ones). Two holes, both
# invisible to the staleness probe because the store WAS current through the
# last close: an unscanned symbol has no M5 cache to preview from, and a
# symbol whose store picked up today's PARTIAL bar from a mid-session scan
# froze that bar as if it were final.
# ---------------------------------------------------------------------------
def test_session_has_opened_tracks_the_market_clock(monkeypatch):
    from datetime import timezone

    monkeypatch.setenv("TRADINGBOT_MARKET_TIMEZONE", "America/Los_Angeles")
    tz = timezone(timedelta(hours=-7))
    f = chart_snapshot.session_has_opened

    assert f(datetime(2026, 7, 29, 5, 0, tzinfo=tz)) is False   # Wed pre-open
    assert f(datetime(2026, 7, 29, 7, 0, tzinfo=tz)) is True    # Wed RTH
    assert f(datetime(2026, 7, 29, 15, 0, tzinfo=tz)) is True   # Wed post-close
    assert f(datetime(2026, 8, 1, 11, 0, tzinfo=tz)) is False   # Saturday


def test_live_aggregate_beats_a_stored_partial_bar_during_the_session():
    """A scan at 11:00 writes today's partial daily bar. Until the close, the
    live intraday aggregate is the better picture of the same session."""
    from datetime import date as _date

    stored_partial = _daily_bars(4)  # 07/01..07/04, so 07/04 is "today"
    m5 = _m5_bars(6, datetime(2026, 7, 4, 9, 30))
    m5[-1]["close"] = 999.0  # the live tape moved well past the frozen bar

    # Session still running: the aggregate wins.
    live = chart_snapshot.forming_d1_bar(
        stored_partial, m5, session_complete_through=_date(2026, 7, 3)
    )
    assert live is not None and live["close"] == 999.0 and live["preview"] is True
    # Session over: the stored bar is final and nothing overrides it.
    assert (
        chart_snapshot.forming_d1_bar(
            stored_partial, m5, session_complete_through=_date(2026, 7, 4)
        )
        is None
    )
    # Omitted argument keeps the historical behavior (the store always wins).
    assert chart_snapshot.forming_d1_bar(stored_partial, m5) is None


def test_todays_stored_partial_bar_draws_as_forming_and_leaves_the_mas_alone():
    """With no live source at all, today's stored bar is still shown - as a
    preview - and the moving averages stop at the last COMPLETED session."""
    bars = _daily_bars(60, start=datetime(2026, 6, 1))  # 06/01 .. 07/30
    during = datetime(2026, 7, 30, 10, 0)  # last bar is today, mid-session

    snapshot = chart_snapshot.build_d1_snapshot(
        "TEST", sessions=90, loader=lambda _s: bars, now=during
    )
    last = snapshot["bars"][-1]
    assert last["dt"].date() == during.date()
    assert last["preview"] is True, "a half-finished session must not draw as final"
    # The overlay tail ends on the last completed session, not on half a day.
    sma50 = next(o for o in snapshot["overlays"] if o["label"] == "SMA50")
    assert len(sma50["values"]) == len(snapshot["bars"])
    assert sma50["values"][-1] is None
    completed_closes = [bar["close"] for bar in bars[:-1]]
    assert abs(sma50["values"][-2] - sum(completed_closes[-50:]) / 50.0) < 1e-9

    # After the close the very same store draws that bar as a normal candle.
    after = chart_snapshot.build_d1_snapshot(
        "TEST", sessions=90, loader=lambda _s: bars, now=datetime(2026, 7, 30, 15, 0)
    )
    assert not after["bars"][-1].get("preview")


def test_unscanned_symbol_fetches_todays_candle_without_persisting_it(monkeypatch):
    """RY's case: store current through yesterday's close, no M5 cache. One
    off-paint fetch fills the forming candle, a second render inside the
    refresh window reuses it, and the PERSISTING wrapper is never called."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    import pandas as pd
    import master_avwap_lib.legacy as legacy
    import ui.widgets.symbol_snapshot_dialog as snap_mod

    yahoo_calls: list[tuple] = []
    persisted: list[tuple] = []

    def fake_yahoo(symbol, days):
        yahoo_calls.append((symbol, days))
        return pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp(datetime.now().date()),
                    "open": 208.49,
                    "high": 212.23,
                    "low": 208.41,
                    "close": 211.78,
                    "volume": 231_925.0,
                }
            ]
        )

    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", fake_yahoo)
    monkeypatch.setattr(
        legacy, "fetch_daily_bars", lambda *a, **k: persisted.append(a) or pd.DataFrame()
    )
    monkeypatch.setattr(chart_snapshot, "session_has_opened", lambda now=None: True)
    monkeypatch.setattr(snap_mod, "_FORMING_BARS", {})
    monkeypatch.setattr(snap_mod, "_FORMING_ATTEMPTS", {})

    # A store that is honestly current: it ends exactly at the last completed
    # session, which is what makes the staleness probe (correctly) say healthy
    # while today's candle is still missing.
    #
    # The session clock is PINNED to "the last session before today" rather
    # than read live: this scenario only exists while a session is running.
    # Run after the close (or on a weekend) the real helper names today as
    # complete, so the store would already hold today's bar and there would be
    # no forming candle to fetch - the assertions below would then be checking
    # a completed candle for a preview flag it is right not to carry.
    today = datetime.now().date()
    last_complete = chart_snapshot.latest_completed_session_date(
        datetime(today.year, today.month, today.day, 10, 0)
    )
    monkeypatch.setattr(
        chart_snapshot, "latest_completed_session_date", lambda now=None: last_complete
    )
    stored = _daily_bars(
        40, start=datetime(last_complete.year, last_complete.month, last_complete.day)
        - timedelta(days=39)
    )
    _seed_d1_store("RY", stored)

    widget = snap_mod.SymbolSnapshotWidget()
    try:
        widget.set_symbol("RY")
        # The freshness probe runs on the worker, so wait for the first
        # delivery before asking what it decided.
        assert _pump_until(lambda: bool(widget._d1.get("bars")))
        assert not chart_snapshot.d1_store_is_stale(
            widget._d1.get("bars") or []
        ), "the staleness probe must still call this store healthy"
        assert _pump_until(
            lambda: widget._forming_thread is not None
        ), "a missing forming candle must start a fetch"
        widget._forming_thread.join(5.0)
        assert _pump_until(lambda: yahoo_calls == [("RY", 5)])
        assert persisted == [], "the partial bar must never reach the durable store"

        # The fetched candle is now the chart's last bar, drawn as forming.
        assert _pump_until(
            lambda: bool(widget._d1.get("bars"))
            and widget._d1["bars"][-1].get("preview") is True
        )
        last = widget._d1["bars"][-1]
        assert last["dt"].date() == datetime.now().date()
        assert last["preview"] is True and last["close"] == 211.78
        assert "forming" in widget.d1_legend.text() or widget.d1_chart.bar_count() > 0

        # Inside the refresh window a re-render reuses the cached candle.
        widget.refresh()
        _pump_until(lambda: False, timeout=0.4)
        thread = widget._forming_thread
        if thread is not None:
            thread.join(5.0)
        app.processEvents()
        assert len(yahoo_calls) == 1, "the cooldown must prevent a fetch loop"
    finally:
        widget.deleteLater()
        app.processEvents()


def test_stale_d1_tail_triggers_one_backfill_with_cooldown(monkeypatch):
    """The widget kicks exactly one off-paint backfill per symbol per window,
    and re-renders when it lands. Paint itself never fetches."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    import master_avwap_lib.legacy as legacy
    import ui.widgets.symbol_snapshot_dialog as snap_mod

    calls: list[tuple] = []

    def fake_fetch(ib, symbol, days):
        calls.append((ib, symbol, days))
        import pandas as pd

        return pd.DataFrame()

    monkeypatch.setattr(legacy, "fetch_daily_bars", fake_fetch)
    monkeypatch.setattr(
        chart_snapshot, "d1_store_is_stale", lambda bars, now=None: True
    )
    stale_d1 = {"symbol": "STAL", "timeframe": "D1", "bars": [_d1_row(datetime(2026, 7, 27).date())], "overlays": []}
    monkeypatch.setattr(
        chart_snapshot, "build_d1_snapshot", lambda *a, **k: dict(stale_d1)
    )
    monkeypatch.setattr(
        chart_snapshot,
        "build_m5_snapshot",
        lambda *a, **k: {"symbol": "STAL", "timeframe": "M5", "bars": [], "overlays": []},
    )
    monkeypatch.setattr(snap_mod, "_D1_BACKFILL_ATTEMPTS", {})

    widget = snap_mod.SymbolSnapshotWidget()
    try:
        widget.set_symbol("STAL")
        # The staleness verdict rides back from the worker with the snapshot.
        assert _pump_until(
            lambda: widget._d1_backfill_thread is not None
        ), "a stale tail must start a backfill worker"
        widget._d1_backfill_thread.join(5.0)
        assert _pump_until(lambda: calls == [(None, "STAL", 260)])

        # Same symbol inside the cooldown window: no second fetch.
        widget.refresh()
        _pump_until(lambda: False, timeout=0.4)
        thread = widget._d1_backfill_thread
        if thread is not None:
            thread.join(5.0)
        app.processEvents()
        assert len(calls) == 1, "cooldown must prevent a backfill loop"
    finally:
        widget.deleteLater()
        app.processEvents()
