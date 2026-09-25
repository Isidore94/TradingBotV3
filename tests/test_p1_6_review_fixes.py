"""P1-6 review fixes: wrong-side stops size nothing, sizes show their notional,
and entry-state times read in the desk's market-local zone.

Seen to fail on the pre-fix branch: the M5 row sized a stop on the wrong side
(abs()), no plan showed a notional, and `chip_detail` printed New York time.
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
import entry_state  # noqa: E402

NY = ZoneInfo("America/New_York")
LA = ZoneInfo("America/Los_Angeles")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _m5(symbol, side, entry, stop):
    feedback = {"symbol": symbol, "entry_price": entry, "stop_price": stop, "bounce_types": "vwap"}
    return SimpleNamespace(symbol=symbol, side=side, trigger="vwap", time_text="10:35:00",
                           timeframe="M5", raw_text="", payload={"feedback": feedback})


def test_a_stop_on_the_wrong_side_sizes_nothing():
    assert entry_plan.shares_for(500, 100.0, 95.0, side="LONG") == 100
    assert entry_plan.shares_for(500, 100.0, 105.0, side="LONG") is None
    assert entry_plan.shares_for(500, 100.0, 105.0, side="SHORT") == 100
    assert entry_plan.shares_for(500, 100.0, 95.0, side="SHORT") is None
    assert entry_plan.shares_for(500, 100.0, 95.0, side="BUY") == 100
    assert entry_plan.shares_for(500, 100.0, 105.0, side="SELL") == 100


def test_the_m5_row_shows_blank_shares_for_a_wrong_side_stop(app):
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar()
    bar.post(_m5("NVDA", "LONG", 100.0, 101.0))  # a long with its stop above: wrong side
    bar.post(_m5("AMD", "SHORT", 50.0, 50.5))
    bar.set_risk_per_trade(200)
    texts = {bar.list.item(i).data(0x0100).symbol: bar.list.item(i).text() for i in range(bar.count())}
    assert " sh" not in texts["NVDA"]
    assert texts["AMD"].endswith("· 400 sh · $20.0k")


def test_sizes_carry_their_notional():
    assert entry_plan.size_text(1000, 10.0) == "1,000 sh · $10.0k"
    assert entry_plan.size_text(40, 20.0) == "40 sh · $800"
    assert entry_plan.size_text(5000, 400.0) == "5,000 sh · $2.0M"
    assert entry_plan.size_text(None, 10.0) == ""
    assert entry_plan.size_text(10, None) == "10 sh"


def test_the_d1_plan_shows_the_notional_and_checks_the_stop_side():
    levels = {"AAA": {"vwap": 97.0, "bands": {"LOWER_1": 95.0, "UPPER_2": 110.0, "UPPER_3": 115.0},
                      "atr20": 2.0, "last_close": 100.0}}
    plan = entry_plan.plan_for_row(symbol="AAA", side="LONG", setup_family="", last_close=100.0,
                                   levels_by_symbol=levels)
    assert entry_plan.plan_cells(plan, 500)["plan_shares"] == "100 sh · $10.0k"
    assert entry_plan.plan_line(plan, 500).endswith("· 100 sh · $10.0k")
    wrong = {**plan, "stop": 105.0}
    assert entry_plan.plan_cells(wrong, 500)["plan_shares"] == ""


def test_the_d1_detail_shows_the_notional(app, monkeypatch):
    from ui.widgets.setup_detail_view import SetupDetailView

    monkeypatch.setattr(entry_plan, "risk_per_trade_dollars", lambda: 500.0)
    view = SetupDetailView()
    view._symbol_levels = {"AAA": {"vwap": 97.0, "bands": {"LOWER_1": 95.0, "UPPER_2": 110.0},
                                   "anchor_date": "", "atr20": 2.0, "last_close": 100.0}}
    html = view._plan_html({"symbol": "AAA", "side": "LONG", "setup_family": "general",
                            "favorite_signals": [], "last_close": 100.0})
    assert "Shares at $500.00 risk:</b> 100 sh · $10.0k" in html


def test_chip_detail_reads_the_market_local_clock(monkeypatch):
    import live_alert_results

    monkeypatch.setattr(live_alert_results, "desk_zone", lambda: LA)
    state = {"state": "gone", "reason": "stop hit", "at": datetime(2026, 9, 22, 10, 40, tzinfo=NY)}
    assert entry_state.chip_detail(state) == "entry: gone (stop hit 07:40)"
    # A naive stamp is already market-local and is shown as it is.
    naive = {**state, "at": datetime(2026, 9, 22, 7, 40)}
    assert entry_state.chip_detail(naive) == "entry: gone (stop hit 07:40)"
