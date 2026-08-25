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


# --------------------------------------------------------------------------
# 2026-08-20: the wake test.
#
# Both EVENING-permitted senders already push at ntfy's maximum - the price
# alerts here and the SPY +/-1% alarm in AutopilotService - but the only test
# push the desk could produce went out at "high", so "will an urgent push
# actually break through Sleep Focus" had never been answerable. This adds a
# way to ASK, not a new sender.
# --------------------------------------------------------------------------
def test_the_ordinary_test_push_is_unchanged(monkeypatch):
    import push_notify
    from ui.services.price_alert_service import PriceAlertService

    sent: list[dict] = []
    monkeypatch.setattr(
        push_notify,
        "send_push",
        lambda *args, **kwargs: (sent.append({"args": args, **kwargs}), {"ok": True, "error": ""})[1],
    )
    service = PriceAlertService()
    try:
        assert service.test_push()["ok"] is True
        assert sent[0]["priority"] == "high"
    finally:
        service.shutdown()


def test_the_wake_test_pushes_at_the_priority_the_real_alerts_use(monkeypatch):
    import push_notify
    from ui.services.price_alert_service import PriceAlertService

    sent: list[dict] = []
    monkeypatch.setattr(
        push_notify,
        "send_push",
        lambda *args, **kwargs: (sent.append({"args": args, **kwargs}), {"ok": True, "error": ""})[1],
    )
    service = PriceAlertService()
    try:
        result = service.test_push(urgent=True)
        assert result["ok"] is True
        assert len(sent) == 1
        assert sent[0]["priority"] == "urgent", "a 'high' wake test proves nothing"
        title, message = sent[0]["args"]
        # It has to be self-describing: the trader reads it half asleep.
        assert "WAKE TEST" in title
        assert "Sleep Focus" in message
    finally:
        service.shutdown()


def test_the_wake_test_fails_quiet_exactly_like_the_ordinary_one(monkeypatch):
    import push_notify
    from ui.services.price_alert_service import PriceAlertService

    monkeypatch.setattr(
        push_notify, "send_push", lambda *_a, **_k: {"ok": False, "error": ""}
    )
    service = PriceAlertService()
    try:
        result = service.test_push(urgent=True)
        assert result["ok"] is False
        # An unconfigured topic is REPORTED, never logged as a delivery.
        assert result["error"] == "No ntfy topic configured yet."
        assert service.status_snapshot()["push_error"] == "No ntfy topic configured yet."
    finally:
        service.shutdown()


def test_the_panel_offers_the_wake_test_beside_the_ordinary_one(monkeypatch):
    import push_notify
    from ui.panels import price_alerts_panel as panel_module
    from ui.panels.price_alerts_panel import PriceAlertsPanel

    sent: list[dict] = []
    monkeypatch.setattr(
        push_notify,
        "send_push",
        lambda *args, **kwargs: (sent.append(kwargs), {"ok": True, "error": ""})[1],
    )
    # The button saves the push settings first, exactly as the ordinary test
    # does. Keep that off this machine's real local_settings.json.
    monkeypatch.setattr(panel_module, "save_local_setting", lambda *_a, **_k: None)
    panel = PriceAlertsPanel()
    try:
        panel.wake_button.click()
        assert [entry["priority"] for entry in sent] == ["urgent"]
        assert "Sleep Focus" in panel.status_label.text()
    finally:
        panel.close()
        panel.deleteLater()
