"""p9 runner dip watch on the Alert Center: the 60 s tick fires once per name per session,
records every field the grading needs, posts a chart-watch row and pushes only in AWAY."""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import long_setups_store  # noqa: E402
import runner_dip_watch  # noqa: E402

ET = ZoneInfo("America/New_York")
SESSION = date(2026, 9, 28)
PAYLOAD = {"as_of": "2026-09-25", "market_working": "yes", "members": [
    {"symbol": "GTLB", "armed": True, "avwape": 100.0, "as_of": "2026-09-25", "rs_percentile": 0.95,
     "strength_sma50_atr": 2.4},
    {"symbol": "NVDA", "armed": False, "avwape": 100.0}]}


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _local(stamp_et: datetime) -> datetime:
    from market_session import get_market_local_timezone

    return stamp_et.replace(tzinfo=ET).astimezone(get_market_local_timezone()[0]).replace(tzinfo=None)


def _bars(count=24, close=99.0):
    """Naive market-local M5 bars (the bot's shape), a tight squeeze from 09:30 ET."""
    out = []
    for index in range(count):
        start = _local(datetime(2026, 9, 28, 9, 30) + timedelta(minutes=5 * index))
        out.append({"dt": start, "open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
                    "volume": 1000.0})
    return out


class _Pushes:
    def __init__(self):
        self.sent = []

    def notify_armed_watch(self, **kwargs):
        self.sent.append(kwargs)
        return {"ok": True}


@pytest.fixture
def panel(tmp_path, monkeypatch):
    from ui.panels.alert_center_panel import AlertCenterPanel

    widget = AlertCenterPanel(parked_symbols_path=tmp_path / "parked.json",
                              focus_d1_flags_path=tmp_path / "focus_flags.json")
    monkeypatch.setattr(long_setups_store, "refresh_runner_dip_async", lambda path=None: False)
    long_setups_store.set_runner_dip_snapshot(PAYLOAD)
    bars = {"GTLB": _bars(), "NVDA": _bars(), "SPY": _bars(close=600.0)}
    widget._m5_bars_for = lambda symbol, sessions=1: bars.get(symbol, [])
    widget._events = []
    widget._record_review_event = lambda action, **kw: widget._events.append((action, kw))
    widget._alerts_added = []
    widget.add_alert = widget._alerts_added.append
    widget.price_alert_service = _Pushes()
    widget._mode = "DESK"
    widget._auto_mode_now = lambda: widget._mode
    yield widget
    long_setups_store.set_runner_dip_snapshot(None)
    widget.deleteLater()


NOW = _local(datetime(2026, 9, 28, 11, 31))


def test_an_armed_runner_fires_once_per_session_with_the_grading_fields(panel):
    panel._poll_runner_dips(now=NOW)
    panel._poll_runner_dips(now=NOW + timedelta(minutes=1))
    assert [action for action, _kw in panel._events] == [runner_dip_watch.FIRED_ACTION]
    _action, kwargs = panel._events[0]
    assert kwargs["symbol"] == "GTLB" and kwargs["side"] == "LONG"
    detail = kwargs["detail"]
    for key in ("session", "ts", "price", "avwape", "box_high", "box_low", "as_of", "rs_percentile",
                "strength_sma50_atr", "spy_price"):
        assert detail[key] is not None, key
    assert detail["ts"].endswith("-04:00") and detail["spy_price"] == 600.0
    assert len(panel._alerts_added) == 1
    alert = panel._alerts_added[0]
    assert alert.tag == "chart_watch" and alert.payload["chart_watch_kind"] == "runner_dip"
    assert alert.trigger.startswith("GTLB runner dip: strong name under the earnings VWAP (100.00)")
    assert panel.runner_dip_status_text() == "Runner dips: 1 armed (GTLB)."
    # The next session fires again.
    panel._poll_runner_dips(now=NOW + timedelta(days=1))
    assert len(panel._events) == 1  # yesterday's bars never fire today


def test_bars_landing_in_a_batch_still_fire(panel):
    """Three bars arrive between two polls; the dip was on the first of them."""
    bars = _bars(21, close=100.2)
    by_symbol = {"GTLB": bars, "SPY": _bars(close=600.0)}
    panel._m5_bars_for = lambda symbol, sessions=1: by_symbol.get(symbol, [])
    first_now = bars[-1]["dt"] + timedelta(minutes=5)
    panel._poll_runner_dips(now=first_now)
    assert panel._events == []
    for index, close in enumerate((99.95, 100.5, 100.5)):
        start = bars[-1]["dt"] + timedelta(minutes=5)
        bars.append({"dt": start, "open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
                     "volume": 1000.0})
    panel._poll_runner_dips(now=bars[-1]["dt"] + timedelta(minutes=5))
    assert [kw["detail"]["price"] for _a, kw in panel._events] == [99.95]


def test_the_phone_hears_a_fire_only_in_away(panel):
    panel._poll_runner_dips(now=NOW)
    assert panel.price_alert_service.sent == []
    panel._runner_dips_fired.clear()
    panel._runner_dip_checked.clear()
    panel._mode = "AWAY"
    panel._poll_runner_dips(now=NOW)
    assert panel.price_alert_service.sent == [{
        "watch_id": "runner_dip:GTLB:2026-09-28", "title": "Runner dip: GTLB",
        "message": "GTLB runner dip: strong name under the earnings VWAP (100.00), squeezing on M5 - box 98.90-99.10"}]
    panel._runner_dips_fired.clear()
    panel._runner_dip_checked.clear()
    panel._mode = "EVENING"
    panel._poll_runner_dips(now=NOW)
    assert len(panel.price_alert_service.sent) == 1


def test_no_bars_or_a_stale_list_is_no_fire(panel):
    panel._m5_bars_for = lambda symbol, sessions=1: []
    panel._poll_runner_dips(now=NOW)
    assert panel._events == []
    long_setups_store.set_runner_dip_snapshot({**PAYLOAD, "as_of": "2026-09-28"})
    panel._m5_bars_for = lambda symbol, sessions=1: _bars()
    panel._poll_runner_dips(now=NOW)
    assert panel._events == [] and panel.runner_dip_status_text() == ""


def test_the_tick_reads_no_file_on_the_qt_thread(panel, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the Qt thread read the runner file")

    monkeypatch.setattr(long_setups_store, "read_runner_dip_watch", refuse)
    panel._poll_runner_dips(now=NOW)
    assert panel._events


def test_the_worker_cache_reads_the_file_once_per_change(tmp_path, monkeypatch):
    import json

    path = tmp_path / "runner_dip_watch.json"
    path.write_text(json.dumps(PAYLOAD), encoding="utf-8")
    reads = []
    real = long_setups_store.read_runner_dip_watch
    monkeypatch.setattr(long_setups_store, "read_runner_dip_watch", lambda p=None: reads.append(p) or real(p))
    long_setups_store.set_runner_dip_snapshot(None)
    try:
        long_setups_store._refresh_runner_dip(path)
        long_setups_store._refresh_runner_dip(path)
        assert len(reads) == 1 and long_setups_store.runner_dip_snapshot()["as_of"] == "2026-09-25"
        assert long_setups_store.refresh_runner_dip_async(path) is True
        assert long_setups_store.refresh_runner_dip_async(path) is False  # checked moments ago
        import threading

        for thread in threading.enumerate():
            if thread.name == "runner-dip-watch-read":
                thread.join(5)
    finally:
        long_setups_store.set_runner_dip_snapshot(None)
