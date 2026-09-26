"""S17 desk regime strip: six auto-regime cells per index, fed off the auto-regime timer."""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research_warehouse import exchange_calendar as xcal  # noqa: E402

ET = xcal.EXCHANGE_TZ


def _qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ------------------------------------------------------------ formatting --


def test_cells_say_the_champion_word_and_tone():
    from ui.widgets.regime_strip import format_cell

    assert format_cell("M5", "bullish_strong") == ("M5 Up+", "bull")
    assert format_cell("H4", "bearish_weak") == ("H4 Dn", "bear")
    assert format_cell("D1", "neutral_chop") == ("D1 Chop", "chop")
    assert format_cell("W", "unknown") == ("W ?", "unknown")
    assert format_cell("W", None) == ("W ?", "unknown")


def test_the_strip_has_six_cells_for_each_index_and_unknown_when_empty():
    from ui.widgets.regime_strip import format_strip

    cells = format_strip({"as_of": "2026-09-24 12:00 ET", "symbols": {"SPY": {"M5": "bullish_weak", "W": "bearish_strong"}}})
    assert list(cells) == ["SPY", "QQQ", "IWM"]
    assert [text for text, _tone, _tip in cells["SPY"]] == ["M5 Up", "M30 ?", "H1 ?", "H4 ?", "D1 ?", "W Dn+"]
    assert "2026-09-24 12:00 ET" in cells["SPY"][0][2]
    assert all(tone == "unknown" for _text, tone, _tip in format_strip({})["IWM"])


def test_the_widget_sets_text_and_tone_without_a_stylesheet():
    _qapp()
    from ui.widgets.regime_strip import RegimeStrip

    strip = RegimeStrip()
    strip.set_readings({"symbols": {"QQQ": {"D1": "bearish_strong"}}})
    cell = strip.cells["QQQ"][4]
    assert cell.text() == "D1 Dn+" and cell.property("tone") == "bear"
    assert cell.styleSheet() == "" and strip.styleSheet() == ""
    assert sum(len(cells) for cells in strip.cells.values()) == 18


def test_the_theme_styles_the_cells_in_px():
    text = (ROOT / "scripts" / "ui" / "theme.qss").read_text(encoding="utf-8")
    assert 'QLabel#RegimeCell[tone="bull"]' in text and 'QLabel#RegimeCell[tone="bear"]' in text
    block = text[text.index("QLabel#RegimeCell {"):]
    block = block[: block.index("}")]
    assert "font-size: @font_body@" in block  # the px token, scaled with the desk


# ---------------------------------------------------------- live reads --


def test_live_bars_drop_the_forming_bar_and_carry_the_market_zone():
    import market_regimes as mr

    bars = [{"dt": datetime(2026, 9, 24, 10, 0) + timedelta(minutes=5 * i), "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10} for i in range(4)]
    now = datetime(2026, 9, 24, 10, 17, tzinfo=ET)
    rows = mr.live_m5_rows("SPY", bars, now=now, tz=ET)
    assert [row["interval_start"].strftime("%H:%M") for row in rows] == ["10:00", "10:05", "10:10"]
    assert rows[0]["interval_start"].utcoffset() == timedelta(hours=-4)


def test_the_strip_agrees_with_the_table_at_the_same_moment():
    """The desk and the night table are one read: SPY at 12:00 ET on 09-24."""
    import json

    import market_regimes as mr
    from tests.test_market_regime_golden_week import GOLDEN, load_fixture

    d1, m5 = load_fixture()
    payload = mr.strip_readings(datetime(2026, 9, 24, 12, 0, tzinfo=ET), d1, m5, symbols=("SPY",))
    row = next(
        json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if '"session_date": "2026-09-24"' in line and '"symbol": "SPY"' in line
    )
    noon = row["snapshots"]["12:00"]
    assert payload["symbols"]["SPY"] == {**{tf: noon[tf] for tf in ("M5", "M30", "H1", "H4")}, "D1": row["timeframes"]["D1"], "W": row["timeframes"]["W"]}
    assert payload["as_of"] == "2026-09-24 12:00 ET"


def test_off_hours_the_strip_reads_the_last_close():
    import market_regimes as mr

    saturday = datetime(2026, 9, 26, 11, 0, tzinfo=ET)
    assert mr.strip_moment(saturday) == xcal.trading_session(saturday.date() - timedelta(days=1)).rth_close_at
    before_open = datetime(2026, 9, 24, 8, 0, tzinfo=ET)
    assert mr.strip_moment(before_open) == xcal.trading_session(before_open.date() - timedelta(days=1)).rth_close_at


# --------------------------------------------------------------- service --


def _service():
    _qapp()
    from ui.services.bounce_service import BounceService

    return BounceService()


def test_the_auto_regime_tick_feeds_the_strip_off_the_gui_thread():
    service = _service()
    seen: dict = {}
    done = threading.Event()

    class _Bot:
        def get_auto_regime_reading(self):
            return {}

        def entry_assist_state(self):
            return {}

        def m5_chart_bars(self, symbol, max_sessions=2):
            seen.setdefault("symbols", []).append(symbol)
            seen["thread"] = threading.get_ident()
            return []

    def loader(d1_symbols, m5_symbols, **_kwargs):
        from tests.test_market_regime_golden_week import load_fixture

        d1, m5 = load_fixture()
        return d1, m5, "fixture"

    service._current_bot = lambda: _Bot()
    service._is_live = lambda: True
    service._regime_strip_loader = loader
    service._emit = lambda signal, *args: (seen.setdefault("payload", args[0]), done.set()) if signal is service._regimeStripReady else None
    service.refresh_auto_regime()

    assert done.wait(20), "the strip worker never ran"
    assert seen["thread"] != threading.get_ident()
    assert seen["symbols"] == ["SPY", "QQQ", "IWM"]
    assert set(seen["payload"]["symbols"]) == {"SPY", "QQQ", "IWM"}
    assert set(seen["payload"]["symbols"]["SPY"]) == {"M5", "M30", "H1", "H4", "D1", "W"}


def test_the_strip_worker_runs_once_per_five_minute_bar_and_single_flight():
    service = _service()
    started = []
    service._load_regime_strip_worker = lambda now: started.append(now)
    at = datetime(2026, 9, 24, 10, 1, tzinfo=ET)
    assert service._maybe_refresh_regime_strip(at) is True
    service._regime_strip_refreshing = False
    assert service._maybe_refresh_regime_strip(at + timedelta(minutes=2)) is False  # same bar
    assert service._maybe_refresh_regime_strip(at + timedelta(minutes=4)) is True  # 10:05 bar
    assert service._maybe_refresh_regime_strip(at + timedelta(minutes=10)) is False  # still in flight
    assert len(started) == 2


def test_the_strip_adds_no_timer():
    from PySide6.QtCore import QTimer

    service = _service()
    names = {name for name in vars(service) if "regime_strip" in name}
    assert not [name for name in names if isinstance(getattr(service, name), QTimer)]
    assert "_regime_timer" in vars(service)  # the existing auto-regime timer is the only clock
