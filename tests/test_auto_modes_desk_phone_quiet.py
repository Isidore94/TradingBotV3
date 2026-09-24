"""DESK sends nothing to the phone (trader, 2026-09-23).

"Only phone goes quiet": the price alert still fires on the desk (signals,
trigger log, toast), and the armed chart watch still reaches the feed. Only the
push is skipped. The manual test push stays allowed in every mode.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

_TRIGGER = {
    "date": "2026-09-23",
    "at": "10:00:00",
    "symbol": "SPY",
    "side": "above",
    "level": 600.0,
    "last": 601.0,
    "note": "",
}


def _service(monkeypatch, mode: str, sent: list):
    import tempfile

    import autopilot_core
    import push_notify
    from ui.services import price_alert_service
    from ui.services.price_alert_service import PriceAlertService

    # An EVENING fire persists its ring list; keep it out of later tests.
    monkeypatch.setattr(
        price_alert_service,
        "PRICE_ALERT_RING_FILE",
        Path(tempfile.mkdtemp()) / "ring.json",
    )

    monkeypatch.setattr(
        push_notify,
        "send_push",
        lambda *a, **kw: sent.append((a, kw)) or {"ok": True, "error": ""},
    )
    monkeypatch.setattr(autopilot_core, "read_auto_pilot_mode", lambda *a, **kw: mode)
    return PriceAlertService()


def test_desk_price_alert_fires_on_the_desk_but_not_the_phone(monkeypatch):
    sent: list = []
    service = _service(monkeypatch, "DESK", sent)
    payloads: list[dict] = []
    messages: list[str] = []
    service.alertTriggered.connect(payloads.append)
    service.triggered.connect(messages.append)
    try:
        service._notify([dict(_TRIGGER)])
    finally:
        service.shutdown()
    assert sent == []
    assert len(payloads) == 1 and len(messages) == 1
    assert payloads[0]["push_skipped"] == "DESK"
    assert payloads[0]["push_ok"] is None


@pytest.mark.parametrize("mode", ["AWAY", "OFF", "EVENING"])
def test_other_modes_still_push_the_price_alert(monkeypatch, mode):
    sent: list = []
    service = _service(monkeypatch, mode, sent)
    try:
        service._notify([dict(_TRIGGER)])
    finally:
        service.shutdown()
    assert len(sent) == 1


def test_desk_armed_watch_is_not_pushed(monkeypatch):
    sent: list = []
    service = _service(monkeypatch, "DESK", sent)
    try:
        result = service.notify_armed_watch(
            watch_id="desk-quiet", title="Pullback: AAPL", message="AAPL LONG"
        )
    finally:
        service.shutdown()
    assert sent == []
    assert result.get("ok") is False
    assert result.get("skipped") == "DESK"


def test_the_manual_test_push_still_works_in_desk(monkeypatch):
    sent: list = []
    service = _service(monkeypatch, "DESK", sent)
    try:
        assert service.test_push()["ok"] is True
    finally:
        service.shutdown()
    assert len(sent) == 1


def test_a_mode_flip_reaches_the_service_without_waiting_for_the_cache(monkeypatch):
    sent: list = []
    service = _service(monkeypatch, "AWAY", sent)
    try:
        service.notify_armed_watch(watch_id="w1", title="t", message="m1")
        service.set_auto_mode("DESK")
        service.notify_armed_watch(watch_id="w2", title="t", message="m2")
    finally:
        service.shutdown()
    assert [call[0][1] for call in sent] == ["m1"]


def test_the_toast_says_the_phone_stayed_quiet_in_desk():
    from ui.widgets.price_alert_toast import PriceAlertToast

    toast = PriceAlertToast(
        {"message": "SPY crossed", "push_ok": None, "push_skipped": "DESK"}
    )
    try:
        from PySide6.QtWidgets import QLabel

        texts = [label.text() for label in toast.findChildren(QLabel)]
    finally:
        toast.close()
    assert any("phone quiet" in text.lower() for text in texts)
