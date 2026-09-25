"""P1-6 6c: fixed-dollar risk -> shares on both plans, the journal entry grade, MFE/MAE.

Decided 2026-09-24 (trader): risk is a fixed dollar amount per trade, one local
setting `risk_per_trade_dollars`; shares = floor(risk / |entry - stop|), blank
when the stop is unknown. Never an order. Seen to fail before the change
(no `entry_plan` / `journal_excursion`, no Settings box, no row shares).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import entry_plan  # noqa: E402
import journal_excursion as jx  # noqa: E402

NY = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture
def settings(monkeypatch):
    """An in-memory `local_settings.json` - the real file is never touched."""
    import project_paths

    store: dict = {}
    monkeypatch.setattr(project_paths, "get_local_setting", lambda key, default=None: store.get(key, default))
    monkeypatch.setattr(project_paths, "save_local_setting", lambda key, value: store.__setitem__(key, value))
    return store


# ------------------------------------------------------------------ shares
def test_shares_are_risk_over_the_stop_distance_rounded_down():
    assert entry_plan.shares_for(500, 100.0, 95.0) == 100
    assert entry_plan.shares_for(500, 100.0, 97.0) == 166  # 166.67 -> 166
    assert entry_plan.shares_for(500, 95.0, 100.0) == 100  # a short: |entry - stop|
    assert entry_plan.shares_for(300, 10.0, 9.9) == 3000  # float noise never loses a share


def test_shares_are_blank_when_anything_is_unknown():
    assert entry_plan.shares_for(None, 100.0, 95.0) is None
    assert entry_plan.shares_for(500, 100.0, None) is None
    assert entry_plan.shares_for(500, 100.0, 100.0) is None
    assert entry_plan.shares_for(0, 100.0, 95.0) is None
    assert entry_plan.shares_for(-50, 100.0, 95.0) is None


def test_the_setting_validates_a_positive_number_and_blank_is_off():
    assert entry_plan.validate_risk_text("") == (True, None)
    assert entry_plan.validate_risk_text("  ") == (True, None)
    assert entry_plan.validate_risk_text("250") == (True, 250.0)
    assert entry_plan.validate_risk_text("$1,000") == (True, 1000.0)
    assert entry_plan.validate_risk_text("0") == (False, None)
    assert entry_plan.validate_risk_text("-5") == (False, None)
    assert entry_plan.validate_risk_text("lots") == (False, None)
    assert entry_plan.validate_risk_text("nan") == (False, None)


def test_the_settings_page_saves_a_good_value_and_refuses_a_bad_one(app, settings):
    from ui.panels.settings_panel import SettingsPanel
    from ui.state import UiState

    panel = SettingsPanel(UiState())
    try:
        seen = []
        panel.riskPerTradeChanged.connect(seen.append)
        panel.risk_input.setText("250")
        assert panel._save_risk_per_trade() is True
        assert settings[entry_plan.RISK_SETTING] == 250.0
        assert seen == [250.0]
        panel.risk_input.setText("lots")
        assert panel._save_risk_per_trade() is False
        assert settings[entry_plan.RISK_SETTING] == 250.0
        assert "Not saved" in panel.risk_hint.text()
        panel.risk_input.setText("")
        assert panel._save_risk_per_trade() is True
        assert settings[entry_plan.RISK_SETTING] is None
        assert seen == [250.0, None]
        assert entry_plan.risk_per_trade_dollars() is None
    finally:
        panel.deleteLater()


def _m5(symbol, entry, stop):
    feedback = {"symbol": symbol, "entry_price": entry, "stop_price": stop, "bounce_types": "vwap"}
    return SimpleNamespace(symbol=symbol, side="LONG", trigger="vwap", time_text="10:35:00",
                           timeframe="M5", raw_text="", payload={"feedback": feedback})


def test_each_m5_row_shows_shares_from_its_own_entry_and_stop(app):
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar()
    bar.post(_m5("NVDA", 100.0, 99.5))
    bar.post(_m5("AMD", 50.0, None))
    assert all(" sh" not in bar.list.item(i).text() for i in range(bar.count()))
    bar.set_risk_per_trade(200)
    texts = {bar.list.item(i).data(0x0100).symbol: bar.list.item(i).text() for i in range(bar.count())}
    assert texts["NVDA"].endswith("· 400 sh")
    assert " sh" not in texts["AMD"]
    bar.set_risk_per_trade(None)
    assert all(" sh" not in bar.list.item(i).text() for i in range(bar.count()))


def test_the_d1_detail_plan_shows_shares(app, settings):
    from ui.widgets.setup_detail_view import SetupDetailView

    settings[entry_plan.RISK_SETTING] = 500
    view = SetupDetailView()
    view._symbol_levels = {
        "AAA": {"vwap": 97.0, "bands": {"LOWER_1": 95.0, "UPPER_2": 110.0, "UPPER_3": 115.0},
                "anchor_date": "2026-08-01", "atr20": 2.0, "last_close": 100.0, "side": "LONG"}
    }
    html = view._plan_html({"symbol": "AAA", "side": "LONG", "setup_family": "general",
                            "favorite_signals": [], "last_close": 100.0})
    assert "Shares at $500.00 risk:</b> 100" in html
    settings[entry_plan.RISK_SETTING] = None
    html = view._plan_html({"symbol": "AAA", "side": "LONG", "setup_family": "general",
                            "favorite_signals": [], "last_close": 100.0})
    assert "Shares at" not in html


def test_the_desk_and_app_carry_the_setting():
    app_source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
    assert "self.settings_panel.riskPerTradeChanged.connect(self.trading_panel.set_risk_per_trade)" in app_source
    desk_source = (SCRIPTS_DIR / "ui" / "panels" / "trading_desk.py").read_text(encoding="utf-8")
    assert "self.m5_alert_bar.set_risk_per_trade(value)" in desk_source
    assert "self.master_panel.set_risk_per_trade(value)" in desk_source


# ------------------------------------------------------------------ journal grade
def test_entry_grade_on_a_fixture_trade():
    # Long: planned 100, stop 99 (1R = 1.00), filled 100.20 -> paid up 0.20R.
    assert entry_plan.entry_grade(100.20, 100.0, 99.0) == pytest.approx(0.20)
    # Short: planned 100, stop 101, filled 99.80 -> also paid 0.20R worse.
    assert entry_plan.entry_grade(99.80, 100.0, 101.0) == pytest.approx(0.20)
    # A better fill is negative.
    assert entry_plan.entry_grade(99.90, 100.0, 99.0) == pytest.approx(-0.10)
    assert entry_plan.entry_grade(100.2, None, 99.0) is None
    assert entry_plan.entry_grade(100.2, 100.0, 100.0) is None
    assert entry_plan.entry_grade(0, 100.0, 99.0) is None  # the journal's 0 means no fill price
    assert entry_plan.entry_grade_text(0.2) == "Entry grade: +0.20R (paid up)"
    assert entry_plan.entry_grade_text(-0.1) == "Entry grade: -0.10R (better than plan)"
    assert entry_plan.entry_grade_text(None).startswith("Entry grade: -")


DAY_TRADE = {
    "trade_id": "t1", "symbol": "NVDA", "direction": "LONG", "security_type": "STK",
    "average_entry_price": 100.0, "opened_at": "2026-09-22T10:37:00", "closed_at": "2026-09-22T10:52:00",
    "planned_entry": 100.0, "planned_stop": 99.0,
}


def _m5bar(hh, mm, high, low):
    return {"dt": datetime(2026, 9, 22, hh, mm), "high": high, "low": low}


def test_day_trade_mfe_mae_from_cached_m5_bars_while_held():
    bars = [
        _m5bar(10, 30, 105.0, 90.0),  # before the entry bar: ignored
        _m5bar(10, 35, 100.8, 99.6),
        _m5bar(10, 40, 101.5, 99.9),
        _m5bar(10, 45, 101.2, 100.4),
        _m5bar(10, 50, 100.9, 100.2),
        _m5bar(10, 55, 110.0, 80.0),  # after the exit: ignored
    ]
    result = jx.excursion(DAY_TRADE, bars, now=datetime(2026, 9, 22, 16, 0, tzinfo=NY), zone=NY)
    assert result["state"] == jx.MEASURED and result["kind"] == "day"
    assert (result["mfe"], result["mae"]) == (1.5, 0.4)
    assert (result["mfe_r"], result["mae_r"]) == (1.5, 0.4)
    assert result["bars"] == 4
    assert jx.excursion_text(result) == "MFE +1.50 (+1.5R) · MAE -0.40 (-0.4R) · 4 M5 bars"


def test_swing_mfe_mae_from_daily_bars_after_the_entry_day():
    trade = {**DAY_TRADE, "direction": "SHORT", "opened_at": "2026-09-14T11:00:00",
             "closed_at": "2026-09-17T15:00:00", "planned_stop": 102.0}
    bars = [
        {"date": "2026-09-14", "high": 130.0, "low": 70.0},  # entry day: ignored
        {"date": "2026-09-15", "high": 101.0, "low": 97.0},
        {"date": "2026-09-16", "high": 100.5, "low": 95.0},
        {"date": "2026-09-17", "high": 99.0, "low": 96.0},
        {"date": "2026-09-18", "high": 140.0, "low": 60.0},  # after the exit: ignored
    ]
    result = jx.excursion(trade, bars, now=datetime(2026, 9, 24, 12, 0, tzinfo=NY), zone=NY)
    assert result["kind"] == "swing"
    assert (result["mfe"], result["mae"]) == (5.0, 1.0)
    assert (result["mfe_r"], result["mae_r"]) == (2.5, 0.5)
    assert "daily bars" in jx.excursion_text(result)


def test_missing_data_is_unknown_never_zero():
    now = datetime(2026, 9, 22, 16, 0, tzinfo=NY)
    assert jx.excursion(DAY_TRADE, [], now=now, zone=NY)["state"] == jx.UNKNOWN
    option = {**DAY_TRADE, "security_type": "OPT"}
    assert jx.excursion(option, [_m5bar(10, 40, 101, 99)], now=now, zone=NY)["state"] == jx.UNKNOWN
    midnight = {**DAY_TRADE, "opened_at": "2026-09-22T00:00:00", "closed_at": "2026-09-22T00:00:00"}
    unknown = jx.excursion(midnight, [_m5bar(10, 40, 101, 99)], now=now, zone=NY)
    assert unknown["state"] == jx.UNKNOWN and unknown["mfe"] is None
    assert jx.excursion_text(unknown).startswith("MFE / MAE: unknown")
    # A forming M5 bar is never read.
    early = jx.excursion(DAY_TRADE, [_m5bar(10, 40, 101, 99.9)], now=datetime(2026, 9, 22, 10, 42, tzinfo=NY), zone=NY)
    assert early["state"] == jx.UNKNOWN


def test_measure_trade_picks_the_loader_by_holding_period():
    calls = []
    jx.measure_trade(DAY_TRADE, now=datetime(2026, 9, 22, 16, 0, tzinfo=NY), zone=NY,
                     m5_loader=lambda s, d: calls.append(("m5", s, d)) or [],
                     daily_loader=lambda s: calls.append(("d1", s)) or [])
    swing = {**DAY_TRADE, "closed_at": "2026-09-23T10:00:00"}
    jx.measure_trade(swing, now=datetime(2026, 9, 24, 16, 0, tzinfo=NY), zone=NY,
                     m5_loader=lambda s, d: calls.append(("m5", s, d)) or [],
                     daily_loader=lambda s: calls.append(("d1", s)) or [])
    assert calls == [("m5", "NVDA", "2026-09-22"), ("d1", "NVDA")]


def test_the_trades_tab_shows_the_grade_and_the_excursion(app, monkeypatch):
    from ui.models.journal import JournalTrade
    from ui.panels.journal.trades_tab import TradesTab
    from ui.services import journal_feed

    monkeypatch.setattr(journal_feed, "trade_legs", lambda _trade_id: [])
    monkeypatch.setattr(journal_feed, "unaccepted_auto_tag_candidates", lambda *_a: [])
    monkeypatch.setattr(journal_feed, "latest_trade_review", lambda _trade_id: {})
    monkeypatch.setattr(journal_feed, "list_adjustments", lambda **_k: [])
    monkeypatch.setattr(
        jx, "measure_trade",
        lambda raw: {"state": jx.MEASURED, "kind": "day", "mfe": 1.5, "mae": 0.4,
                     "mfe_r": 1.5, "mae_r": 0.4, "bars": 4},
    )
    tab = TradesTab(header=None, threaded=False)
    try:
        trade = JournalTrade.from_mapping({**DAY_TRADE, "average_entry_price": 100.2})
        tab._current = trade
        tab._show_trade(trade)
        assert tab.entry_grade_label.text() == "Entry grade: +0.20R (paid up)"
        assert tab.excursion_label.text() == "MFE +1.50 (+1.5R) · MAE -0.40 (-0.4R) · 4 M5 bars"
        # A late result for another trade never paints over this one.
        tab._on_excursion("someone-else", {"state": jx.UNKNOWN, "reason": "x"})
        assert tab.excursion_label.text().startswith("MFE +1.50")
    finally:
        tab.deleteLater()
