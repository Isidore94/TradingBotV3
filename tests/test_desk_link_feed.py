"""Satellite desk feed: relayed popups become real Alert Center alerts."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import chart_snapshot
from desk_link.popup_payload import build_alert_popup_payload
from desk_link.server import DeskLinkServer

WAIT = 5.0
EASTERN = ZoneInfo("America/New_York")


def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _pump_until(qapp, condition, timeout: float = WAIT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def _payload():
    start = datetime(2026, 7, 30, 9, 30, tzinfo=EASTERN)
    bars = [
        {
            "dt": start + timedelta(minutes=5 * i),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
        for i in range(12)
    ]
    return build_alert_popup_payload(
        {
            "time_text": "07:00:00",
            "symbol": "NVDA",
            "side": "LONG",
            "trigger": "VWAP bounce",
            "timeframe": "M5",
            "payload": {"tier": "A"},
            "future_field_from_newer_main": "ignored",
        },
        d1_snapshot={"symbol": "NVDA", "timeframe": "D1", "bars": [], "overlays": [], "note": ""},
        m5_snapshot=chart_snapshot.build_m5_snapshot("NVDA", bars),
    )


def test_feed_rebuilds_alerts_and_backs_m5_charts_end_to_end():
    qapp = _qapp()
    from ui.services.desk_link_feed import DeskLinkFeedService

    server = DeskLinkServer(token="tkn", machine_name="main", host="127.0.0.1", port=0)
    server.start()
    feed = DeskLinkFeedService()
    alerts: list = []
    feed.alertReceived.connect(alerts.append)
    try:
        feed.start(host="127.0.0.1", port=server.address[1], token="tkn", machine_name="sat-desk")
        assert _pump_until(qapp, lambda: server.client_count == 1)
        server.send_alert_popup(_payload())
        assert _pump_until(qapp, lambda: len(alerts) == 1)

        alert = alerts[0]
        # A real BounceAlert, unknown wire fields dropped, payload intact.
        assert alert.symbol == "NVDA" and alert.side == "LONG"
        assert alert.payload == {"tier": "A"}
        assert not hasattr(alert, "future_field_from_newer_main")
        # The relayed M5 bars now back the chart surfaces like a live bot.
        bars = feed.payload_bot().m5_chart_bars("NVDA")
        assert len(bars) == 12 and isinstance(bars[0]["dt"], datetime)
    finally:
        feed.stop()
        server.stop()


def test_panel_remote_feed_lands_alerts_in_the_real_feed(tmp_path):
    _qapp()
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.services.desk_link_feed import DeskLinkFeedService, _rebuild_alert

    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    feed = DeskLinkFeedService()
    panel.attach_remote_feed(feed)

    alert = _rebuild_alert({"time_text": "07:01:00", "symbol": "AMD", "side": "LONG", "trigger": "RS leader"})
    feed.alertReceived.emit(alert)
    assert any(entry.symbol == "AMD" for entry in panel._alerts)
    # With no bounce service, chart data falls back to the payload bot.
    assert panel._current_bot() is feed.payload_bot()
