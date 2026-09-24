"""EVENING price alerts ring the phone every 10 seconds (trader, 2026-09-23).

A price alert that fires while Auto mode is EVENING pushes at once and then
again every 10 seconds, one combined push, until the mode changes. No taper
and no cap (the trader's explicit call). A rejected or rate-limited send never
stops the loop. Armed chart watches do not ring.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _trigger(symbol="AAPL"):
    return {
        "date": "2026-09-23",
        "at": "06:40:00",
        "symbol": symbol,
        "side": "above",
        "level": 200.0,
        "last": 201.0,
        "note": "",
    }


class _Sender:
    def __init__(self, result=None, *, raises=False):
        self.calls: list[tuple] = []
        self.result = result or {"ok": True, "error": ""}
        self.raises = raises
        self.done = threading.Event()

    def __call__(self, title, message, **kwargs):
        self.calls.append((title, message, kwargs))
        self.done.set()
        if self.raises:
            raise RuntimeError("boom")
        return dict(self.result)


def _service(monkeypatch, mode, sender, *, ring_file=None):
    import tempfile

    import autopilot_core
    import push_notify
    from ui.services import price_alert_service
    from ui.services.price_alert_service import PriceAlertService

    if ring_file is None:
        ring_file = Path(tempfile.mkdtemp()) / "ring.json"
    monkeypatch.setattr(price_alert_service, "PRICE_ALERT_RING_FILE", ring_file, raising=False)
    monkeypatch.setattr(push_notify, "send_push", sender)
    monkeypatch.setattr(autopilot_core, "read_auto_pilot_mode", lambda *a, **kw: mode)
    service = PriceAlertService()
    service._timer.stop()
    return service


def _wait_for_ring(service, timeout=2.0):
    deadline = time.monotonic() + timeout
    while service._ring_sending and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not service._ring_sending


def test_the_ring_timer_runs_every_ten_seconds():
    from ui.services import price_alert_service

    assert price_alert_service.RING_INTERVAL_MS == 10_000


def test_an_evening_alert_rings_until_the_mode_changes(monkeypatch):
    sender = _Sender()
    service = _service(monkeypatch, "EVENING", sender)
    try:
        service._notify([_trigger("AAPL"), _trigger("MSFT")])
        _app.processEvents()
        assert len(sender.calls) == 2  # the first pushes, as before
        assert service._ring_timer.isActive()
        assert service._ring_timer.interval() == 10_000

        service._ring_tick()
        _wait_for_ring(service)
        service._ring_tick()
        _wait_for_ring(service)
        assert len(sender.calls) == 4
        title, message, kwargs = sender.calls[-1]
        assert "AAPL" in message and "MSFT" in message
        assert kwargs["priority"] == "urgent"
        assert "every 10 seconds" in message

        service.set_auto_mode("DESK")
        assert not service._ring_timer.isActive()
        service._ring_tick()
        _wait_for_ring(service)
        assert len(sender.calls) == 4
    finally:
        service.shutdown()


def test_a_rejected_or_raising_send_never_stops_the_ringing(monkeypatch):
    sender = _Sender({"ok": False, "error": "HTTP 429", "kind": "rejected"})
    service = _service(monkeypatch, "EVENING", sender)
    try:
        service._notify([_trigger()])
        _app.processEvents()
        service._ring_tick()
        _wait_for_ring(service)
        sender.raises = True
        service._ring_tick()
        _wait_for_ring(service)
        sender.raises = False
        service._ring_tick()
        _wait_for_ring(service)
        assert len(sender.calls) == 4
        assert service._ring_timer.isActive()
    finally:
        service.shutdown()


def test_only_one_ring_send_is_in_flight(monkeypatch):
    release = threading.Event()
    calls: list[str] = []

    def slow(title, message, **kwargs):
        calls.append(message)
        release.wait(2.0)
        return {"ok": True}

    service = _service(monkeypatch, "EVENING", lambda *a, **k: {"ok": True})
    try:
        service._notify([_trigger()])
        _app.processEvents()
        import push_notify

        monkeypatch.setattr(push_notify, "send_push", slow)
        service._ring_tick()
        service._ring_tick()
        release.set()
        _wait_for_ring(service)
        assert len(calls) == 1
    finally:
        release.set()
        service.shutdown()


def test_the_ring_notices_a_missed_mode_flip_on_its_own(monkeypatch):
    import autopilot_core

    sender = _Sender()
    service = _service(monkeypatch, "EVENING", sender)
    try:
        service._notify([_trigger()])
        _app.processEvents()
        monkeypatch.setattr(autopilot_core, "read_auto_pilot_mode", lambda *a, **kw: "DESK")
        service._auto_mode_cache = None
        service._ring_tick()
        _wait_for_ring(service)
        assert len(sender.calls) == 1
        assert not service._ring_timer.isActive()
    finally:
        service.shutdown()


def test_away_and_off_alerts_push_once_and_do_not_ring(monkeypatch):
    for mode in ("AWAY", "OFF"):
        sender = _Sender()
        service = _service(monkeypatch, mode, sender)
        try:
            service._notify([_trigger()])
            _app.processEvents()
            assert not service._ring_timer.isActive()
            service._ring_tick()
            _wait_for_ring(service)
            assert len(sender.calls) == 1
        finally:
            service.shutdown()


def test_an_armed_chart_watch_does_not_ring_in_evening(monkeypatch):
    sender = _Sender()
    service = _service(monkeypatch, "EVENING", sender)
    try:
        service.notify_armed_watch(watch_id="w", title="Pullback: AAPL", message="AAPL")
        _app.processEvents()
        assert not service._ring_timer.isActive()
    finally:
        service.shutdown()


# ---------------------------------------------------------------------------
# The ring list survives a desk restart (per machine, day-scoped)
# ---------------------------------------------------------------------------
def test_a_restart_in_evening_keeps_ringing(monkeypatch, tmp_path):
    path = tmp_path / "ring.json"
    sender = _Sender()
    first = _service(monkeypatch, "EVENING", sender, ring_file=path)
    try:
        first._notify([_trigger("AAPL")])
        _app.processEvents()
    finally:
        first.shutdown()

    second = _service(monkeypatch, "EVENING", sender, ring_file=path)
    try:
        assert second._ring_timer.isActive()
        second._ring_tick()
        _wait_for_ring(second)
        assert len(sender.calls) == 2
        assert "AAPL" in sender.calls[-1][1]
    finally:
        second.shutdown()


def test_a_restart_outside_evening_clears_the_saved_ring(monkeypatch, tmp_path):
    path = tmp_path / "ring.json"
    sender = _Sender()
    first = _service(monkeypatch, "EVENING", sender, ring_file=path)
    try:
        first._notify([_trigger("AAPL")])
        _app.processEvents()
    finally:
        first.shutdown()

    second = _service(monkeypatch, "DESK", sender, ring_file=path)
    try:
        assert not second._ring_timer.isActive()
    finally:
        second.shutdown()
    third = _service(monkeypatch, "EVENING", sender, ring_file=path)
    try:
        assert not third._ring_timer.isActive()
    finally:
        third.shutdown()


def test_leaving_evening_clears_the_saved_ring(monkeypatch, tmp_path):
    path = tmp_path / "ring.json"
    service = _service(monkeypatch, "EVENING", _Sender(), ring_file=path)
    try:
        service._notify([_trigger("AAPL")])
        _app.processEvents()
        assert "AAPL" in path.read_text(encoding="utf-8")
        service.set_auto_mode("DESK")
    finally:
        service.shutdown()
    assert not path.exists() or "AAPL" not in path.read_text(encoding="utf-8")


def test_yesterdays_saved_ring_is_dropped(monkeypatch, tmp_path):
    import json

    path = tmp_path / "ring.json"
    path.write_text(
        json.dumps({"date": "2000-01-03", "messages": ["OLD crossed"]}), encoding="utf-8"
    )
    service = _service(monkeypatch, "EVENING", _Sender(), ring_file=path)
    try:
        assert not service._ring_timer.isActive()
    finally:
        service.shutdown()
