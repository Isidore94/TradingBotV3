import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def test_non_engine_never_polls_or_pushes(monkeypatch):
    import price_alerts
    import push_notify
    from ui.services.price_alert_service import PriceAlertService

    monkeypatch.setattr(
        price_alerts,
        "fetch_last_quotes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("quote fetch attempted")),
    )
    monkeypatch.setattr(
        push_notify,
        "send_push",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("push attempted")),
    )
    service = PriceAlertService(engine_enabled=False)
    try:
        assert not service._timer.isActive()
        service.check_now()
        snapshot = service.status_snapshot()
        assert snapshot["engine_enabled"] is False
        assert "not the engine machine" in snapshot["note"]
        assert service.test_push()["ok"] is False
    finally:
        service.shutdown()


def test_phone_push_is_urgent_and_precedes_desktop_signal(monkeypatch):
    import push_notify
    from ui.services.price_alert_service import PriceAlertService

    order: list[str] = []
    sent: list[dict] = []

    def send_push(*_args, **kwargs):
        order.append("push")
        sent.append(kwargs)
        return {"ok": True, "error": ""}

    monkeypatch.setattr(push_notify, "send_push", send_push)
    service = PriceAlertService()
    payloads: list[dict] = []
    service.alertTriggered.connect(lambda payload: (order.append("desktop"), payloads.append(payload)))
    try:
        service._notify(
            [
                {
                    "date": "2026-08-03",
                    "at": "10:00:00",
                    "symbol": "SPY",
                    "side": "above",
                    "level": 600.0,
                    "last": 601.0,
                    "note": "",
                }
            ]
        )
        assert order == ["push", "desktop"]
        assert sent[0]["priority"] == "urgent"
        assert payloads[0]["push_ok"] is True
        assert payloads[0]["priority"] == "urgent"
    finally:
        service.shutdown()
