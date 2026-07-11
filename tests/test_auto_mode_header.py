"""Plan.md sec 15.2: persistent Auto Mode control in the global shell."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _FakeService:
    """auto_mode surface without the real AutopilotService side effects."""

    def __init__(self):
        self.enabled = False
        self.profile = "DESK"
        self.calls = []

    @property
    def auto_mode(self):
        return self.profile if self.enabled else "OFF"

    def set_profile(self, profile):
        self.calls.append(("profile", profile))
        self.profile = profile

    def set_enabled(self, enabled):
        self.calls.append(("enabled", enabled))
        self.enabled = enabled


def _shell_stub():
    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication, QPushButton

    QApplication.instance() or QApplication([])
    from ui.app import MainWindow

    stub = MainWindow.__new__(MainWindow)  # no real panels/services
    service = _FakeService()
    stub.autopilot_panel = SimpleNamespace(service=service)
    stub.auto_mode_button = QPushButton()
    return stub, service


def test_cycle_walks_off_desk_away_off():
    stub, service = _shell_stub()
    from ui.app import MainWindow

    assert service.auto_mode == "OFF"
    MainWindow._cycle_auto_mode(stub)
    assert service.auto_mode == "DESK" and ("enabled", True) in service.calls
    MainWindow._cycle_auto_mode(stub)
    assert service.auto_mode == "AWAY"
    MainWindow._cycle_auto_mode(stub)
    assert service.auto_mode == "OFF" and ("enabled", False) in service.calls


def test_button_text_reflects_mode():
    stub, service = _shell_stub()
    from ui.app import MainWindow

    MainWindow._sync_auto_mode_button(stub)
    assert stub.auto_mode_button.text() == "Auto: OFF"
    service.enabled = True
    service.profile = "AWAY"
    MainWindow._sync_auto_mode_button(stub)
    assert stub.auto_mode_button.text() == "Auto: AWAY"
