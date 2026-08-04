import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _payload(index: int) -> dict:
    return {
        "date": "2026-08-03",
        "at": f"10:00:0{index}",
        "symbol": "SPY",
        "side": "above",
        "level": 600 + index,
        "last": 601 + index,
        "message": f"SPY alert {index}",
        "priority": "urgent",
    }


def test_toast_is_nonactivating_persistent_and_dismisses_cleanly():
    from ui.widgets.price_alert_toast import PriceAlertToast

    toast = PriceAlertToast(_payload(1))
    destroyed = []
    toast.destroyed.connect(lambda *_args: destroyed.append(True))
    toast.show()
    _app.processEvents()
    try:
        assert toast.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        assert toast.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
        assert toast.isVisible()
        toast.close()
        _app.processEvents()
        assert destroyed
    finally:
        if not destroyed:
            toast.close()


def test_toast_manager_caps_bursts_and_deduplicates(monkeypatch):
    from ui.widgets.price_alert_toast import PriceAlertToastManager

    parent = QWidget()
    manager = PriceAlertToastManager(parent, cap=2)
    beeps = []
    monkeypatch.setattr(QApplication, "beep", lambda: beeps.append(True))
    assert manager.show_alert(_payload(1)) is True
    assert manager.show_alert(_payload(1), replayed=True) is False
    assert manager.show_alert(_payload(2)) is True
    assert manager.show_alert(_payload(3)) is True
    _app.processEvents()
    try:
        assert len(manager.toasts) == 2
        assert len(beeps) == 3
    finally:
        for toast in list(manager.toasts):
            toast.close()
        parent.close()
