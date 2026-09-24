"""Relative volume on the chart review (trader, 2026-09-23).

"We need a rvol calculator at the top of visual chart review. Additionally
volume bars should be present on the D1. White = sub 1.0 rvol ... 2-3 green.
3+ should be blue."

Bands: under 1.0 white, 1.0-1.5 yellow, 1.5-2.0 orange, 2.0-3.0 green, 3.0+
blue. The trader wrote "orange" for both 1-1.5 and 1.5-2; the lead read the
first as yellow (the trader may overrule).
"""

from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import rvol  # noqa: E402


@pytest.mark.parametrize(
    "value, band",
    [
        (0.0, "quiet"),
        (0.99, "quiet"),
        (1.0, "warm"),
        (1.49, "warm"),
        (1.5, "hot"),
        (1.99, "hot"),
        (2.0, "strong"),
        (2.99, "strong"),
        (3.0, "extreme"),
        (12.0, "extreme"),
        (None, None),
        (float("nan"), None),
        (-1.0, None),
    ],
)
def test_bands_follow_the_traders_thresholds(value, band):
    assert rvol.rvol_band(value) == band


def test_band_colours_are_white_yellow_orange_green_blue():
    from ui import theme

    assert theme.RVOL_BAND_TOKENS == {
        "quiet": "chart_white",
        "warm": "chart_yellow",
        "hot": "chart_orange",
        "strong": "chart_green",
        "extreme": "chart_blue",
    }
    assert theme.rvol_color(3.2, "dark") == theme.color("chart_blue", "dark")
    assert theme.rvol_color(0.5, "dark") == theme.color("chart_white", "dark")
    assert theme.rvol_color(None) is None
    for name in ("dark", "light"):
        assert "chart_orange" in theme.THEMES[name]


def test_daily_rvol_matches_the_high_rvol_level_rule():
    """Same rule as the D1 high-rvol lines: volume / 50-day SMA including it."""
    import pandas as pd
    from master_avwap_lib.levels import HV_VOL_SMA, compute_relvol

    assert rvol.D1_RVOL_LOOKBACK == HV_VOL_SMA
    rng = random.Random(7)
    volumes = [rng.uniform(1e5, 5e6) for _ in range(300)]
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=300, freq="D"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": volumes,
        }
    )
    expected = compute_relvol(frame).tolist()
    got = rvol.daily_rvol_series(volumes)
    for index, (want, have) in enumerate(zip(expected, got)):
        if index < HV_VOL_SMA - 1:
            assert have is None
        else:
            assert have == pytest.approx(want, rel=1e-9)


def test_missing_daily_volume_is_unknown_not_quiet():
    volumes = [1_000.0] * 60
    volumes[30] = 0.0
    got = rvol.daily_rvol_series(volumes, lookback=10)
    assert got[8] is None  # short window
    assert got[9] == pytest.approx(1.0)
    assert all(value is None for value in got[30:40])  # window holds the hole
    assert got[40] == pytest.approx(1.0)


# -- intraday reading ------------------------------------------------------


def _intraday_frame(sessions=16, *, today_factor=2.0, today_bars=24, base=1000.0):
    """Regular-hours 5m frame; every day flat, today ``today_factor`` heavier."""
    import pandas as pd

    stamps, volumes = [], []
    day = datetime(2026, 8, 31)
    made = 0
    while made < sessions:
        if day.weekday() < 5:
            made += 1
            count = today_bars if made == sessions else 78
            factor = today_factor if made == sessions else 1.0
            opened = day.replace(hour=9, minute=30)
            for index in range(count):
                stamps.append(opened + timedelta(minutes=5 * index))
                volumes.append(base * factor)
        day += timedelta(days=1)
    index = pd.DatetimeIndex(stamps).tz_localize("America/New_York")
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": volumes},
        index=index,
    )


def _service(frame, *, clock=None):
    from ui.services.intraday_rvol_service import IntradayRvolService

    return IntradayRvolService(
        downloader=lambda _symbol: frame,
        clock=clock or (lambda: datetime(2026, 12, 31)),
    )


@pytest.fixture(scope="module")
def _qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_session_rvol_reads_today_against_the_same_time_of_day(_qapp):
    service = _service(_intraday_frame(today_factor=2.0))
    assert service.request("abc")
    service.wait_idle()
    reading = service.reading("ABC")
    assert reading is not None
    assert reading.session_rvol == pytest.approx(2.0)
    assert reading.last_bar_rvol == pytest.approx(2.0)
    assert reading.prior_sessions == 15


def test_the_forming_bar_is_dropped(_qapp):
    from ui.services.intraday_rvol_service import frame_to_volume_bars, _now

    frame = _intraday_frame()
    bars = frame_to_volume_bars(frame, now=datetime(2026, 12, 31))
    last = bars[-1]["dt"]
    # "now" two minutes into the last bar: that bar is still printing.
    cut = frame_to_volume_bars(frame, now=last + timedelta(minutes=2))
    assert len(cut) == len(bars) - 1
    assert callable(_now)


def test_too_little_history_reads_as_unmeasured(_qapp):
    service = _service(_intraday_frame(sessions=4))
    service.request("ABC")
    service.wait_idle()
    reading = service.reading("ABC")
    assert reading is not None and reading.session_rvol is None


def test_one_fetch_per_refresh_window(_qapp):
    calls = []
    frame = _intraday_frame()

    def downloader(symbol):
        calls.append(symbol)
        return frame

    from ui.services.intraday_rvol_service import IntradayRvolService

    service = IntradayRvolService(downloader=downloader, clock=lambda: datetime(2026, 12, 31))
    assert service.request("ABC")
    service.wait_idle()
    assert not service.request("ABC")
    assert calls == ["ABC"]


def test_a_failed_fetch_keeps_no_reading_and_does_not_raise(_qapp):
    from ui.services.intraday_rvol_service import IntradayRvolService

    def boom(_symbol):
        raise ConnectionError("down")

    service = IntradayRvolService(downloader=boom, clock=lambda: datetime(2026, 12, 31))
    service.request("ABC")
    service.wait_idle()
    assert service.reading("ABC") is None


# -- the chart pieces -------------------------------------------------------


def _d1_bars(count=80, *, spike=None):
    base = datetime(2026, 1, 1)
    bars = []
    for index in range(count):
        bars.append(
            {
                "dt": base + timedelta(days=index),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1_000_000.0,
            }
        )
    if spike is not None:
        bars[spike]["volume"] = 4_000_000.0
    return bars


def test_d1_volume_columns_carry_each_days_rvol(_qapp):
    from ui.widgets.candle_chart import CandleChart

    bars = _d1_bars(spike=70)
    bars.append(dict(bars[-1], dt=bars[-1]["dt"] + timedelta(days=1), preview=True))
    chart = CandleChart()
    try:
        chart.set_volume_visible(True)
        chart.set_data(bars, [], timeframe="d1")
        readings = chart._volume.rvol_readings()
        assert readings[10] is None  # not 50 days of history yet
        assert readings[60] == pytest.approx(1.0)
        assert rvol.rvol_band(readings[70]) == "extreme"
        assert readings[-1] is None  # today's unfinished bar stays grey
    finally:
        chart.close()


def test_header_shows_the_reading_in_its_band_colour(_qapp, monkeypatch):
    from ui import theme
    from ui.services import intraday_rvol_service
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    service = _service(_intraday_frame(today_factor=3.5))
    monkeypatch.setattr(intraday_rvol_service, "_SHARED", service)
    widget = SymbolSnapshotWidget(compact=True)
    try:
        widget.set_symbol("ABC")
        service.wait_idle()
        _qapp.processEvents()
        text = widget.rvol_label.text()
        assert "RVOL 3.50×" in text
        assert theme.rvol_color(3.5) in text
        assert "same time of day" in widget.rvol_label.toolTip()
    finally:
        widget.close()
        widget.deleteLater()


def test_header_says_unmeasured_rather_than_zero(_qapp):
    from ui.widgets.symbol_snapshot_dialog import rvol_label_html, rvol_tooltip

    assert "RVOL –" in rvol_label_html(None)
    assert "not measured" in rvol_tooltip(None)
