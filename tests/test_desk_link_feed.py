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
import pytest

# Real client/server sockets on loopback ARE the point of this file: Desk Link
# is a wire protocol, and a test that mocked the transport would be testing a
# mock. Marked `network` so the offline tripwire lets it through - the marker
# permits, it does not deselect, so these still run in every suite.
#
# This is a re-marking, not a weakening: nothing here reaches the internet or a
# broker, and every assertion is unchanged.
pytestmark = pytest.mark.network


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
        # Wait for the hello, not just the accept: the server only broadcasts
        # to connections whose machine name is known, so a popup sent in the
        # accept window would be silently addressed to nobody.
        assert _pump_until(qapp, lambda: server.connected_machines() == ["sat-desk"])
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


def test_desk_streams_relay_live_surfaces_end_to_end(monkeypatch):
    qapp = _qapp()
    import ui.services.desk_link_service as service_module
    from ui.services.desk_link_feed import DeskLinkFeedService

    settings: dict = {"desk_link_port": 0, "desk_link_token": "tkn"}
    monkeypatch.setattr(service_module, "get_local_setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(service_module, "save_local_setting", lambda key, value: settings.__setitem__(key, value))
    main = service_module.DeskLinkService(machine_name="main")
    assert main.start()

    feed = DeskLinkFeedService()
    received: dict[str, list] = {
        "rrs": [],
        "status": [],
        "board": [],
        "regime": [],
        "price": [],
    }
    feed.rrsSnapshotChanged.connect(received["rrs"].append)
    feed.statusChanged.connect(received["status"].append)
    feed.entryBoardChanged.connect(received["board"].append)
    feed.autoRegimeChanged.connect(received["regime"].append)
    feed.priceAlertReceived.connect(received["price"].append)
    try:
        feed.start(host="127.0.0.1", port=main._server.address[1], token="tkn", machine_name="sat-desk")
        assert _pump_until(qapp, lambda: main.connected_machines() == ["sat-desk"])

        main.publish_stream("rrs", {"leaders": ["NVDA"]})
        main.publish_stream("status", "connected")
        main.publish_stream("entry_board", {"rows": []})
        main.publish_stream("auto_regime", {"env_key": "bullish_strong"})
        main.publish_stream(
            "price_alert",
            {
                "date": "2026-08-03",
                "at": "10:15:00",
                "symbol": "NVDA",
                "side": "above",
                "level": 190.0,
                "last": 190.2,
                "message": "NVDA crossed above 190",
                "priority": "urgent",
            },
        )
        main.publish_stream("stream_from_the_future", {"x": 1})  # skipped, not fatal

        assert _pump_until(qapp, lambda: all(received[key] for key in received))
        assert received["rrs"][0] == {"leaders": ["NVDA"]}
        assert received["status"][0] == "connected"
        assert received["regime"][0]["env_key"] == "bullish_strong"
        assert received["price"][0]["symbol"] == "NVDA"
        assert received["price"][0]["replayed"] is False
    finally:
        feed.stop()
        main.stop()


def test_sticky_snapshot_recovers_missed_price_alert_once(monkeypatch):
    qapp = _qapp()
    import price_alerts
    import ui.services.desk_link_service as service_module
    from ui.services.desk_link_feed import DeskLinkFeedService

    trigger = {
        "date": "2026-08-03",
        "at": "10:20:00",
        "symbol": "AMD",
        "side": "below",
        "level": "170.0",
        "last": "169.8",
        "note": "support",
    }
    monkeypatch.setattr(price_alerts, "todays_triggers", lambda: [trigger])
    settings: dict = {"desk_link_port": 0, "desk_link_token": "tkn"}
    monkeypatch.setattr(service_module, "get_local_setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(service_module, "save_local_setting", lambda key, value: settings.__setitem__(key, value))
    main = service_module.DeskLinkService(machine_name="main")
    assert main.start()

    feed = DeskLinkFeedService()
    received: list[dict] = []
    feed.priceAlertReceived.connect(received.append)
    try:
        feed.start(host="127.0.0.1", port=main._server.address[1], token="tkn", machine_name="sat-desk")
        assert _pump_until(qapp, lambda: len(received) == 1)
        assert received[0]["symbol"] == "AMD"
        assert received[0]["replayed"] is True

        # A delayed live copy of the same event cannot duplicate the toast/feed row.
        main.publish_stream("price_alert", {**trigger, "message": "same event", "priority": "urgent"})
        qapp.processEvents()
        time.sleep(0.05)
        qapp.processEvents()
        assert len(received) == 1
    finally:
        feed.stop()
        main.stop()


def test_live_m5_stream_updates_the_satellite_chart_cache(monkeypatch):
    qapp = _qapp()
    import ui.services.desk_link_service as service_module
    from ui.services.desk_link_feed import DeskLinkFeedService

    settings: dict = {"desk_link_port": 0, "desk_link_token": "tkn"}
    monkeypatch.setattr(service_module, "get_local_setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(service_module, "save_local_setting", lambda key, value: settings.__setitem__(key, value))
    main = service_module.DeskLinkService(machine_name="main")
    assert main.start()

    start = datetime(2026, 7, 30, 9, 30, tzinfo=EASTERN)
    live_bars = [
        {"dt": start + timedelta(minutes=5 * i), "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10}
        for i in range(6)
    ]

    class _Bot:
        def m5_chart_bars(self, symbol, max_sessions=2):
            return live_bars if symbol == "NVDA" else []

    main.set_live_chart_source(lambda: _Bot(), lambda: ["NVDA", "EMPTY"])

    feed = DeskLinkFeedService()
    try:
        feed.start(host="127.0.0.1", port=main._server.address[1], token="tkn", machine_name="sat-desk")
        assert _pump_until(qapp, lambda: main.connected_machines() == ["sat-desk"])

        main._publish_live_charts()  # what the 30s timer fires
        assert _pump_until(qapp, lambda: bool(feed.payload_bot().m5_chart_bars("NVDA")))
        bars = feed.payload_bot().m5_chart_bars("NVDA")
        assert len(bars) == 6
        assert isinstance(bars[0]["dt"], datetime)
        assert bars[0]["dt"] == live_bars[0]["dt"]
        assert feed.payload_bot().m5_chart_bars("EMPTY") == []
    finally:
        feed.stop()
        main.stop()


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
