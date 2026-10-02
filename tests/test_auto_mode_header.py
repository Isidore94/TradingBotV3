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


def _menu_stub():
    from ui.app import MainWindow

    stub, service = _shell_stub()
    menu = MainWindow._build_auto_mode_menu(stub)
    stub.auto_mode_button.setMenu(menu)
    return stub, service, menu


def test_button_text_reflects_mode():
    stub, service = _shell_stub()
    from ui.app import MainWindow

    MainWindow._sync_auto_mode_button(stub)
    assert stub.auto_mode_button.text() == "Auto: OFF"
    service.enabled = True
    service.profile = "AWAY"
    MainWindow._sync_auto_mode_button(stub)
    assert stub.auto_mode_button.text() == "Auto: AWAY"


def test_menu_lists_every_mode_in_order():
    stub, service, menu = _menu_stub()
    from ui.app import AUTO_MODE_CHOICES

    assert AUTO_MODE_CHOICES == ("OFF", "DESK", "AWAY", "EVENING")
    assert stub.auto_mode_button.menu() is menu
    assert stub.auto_mode_menu is menu
    assert menu.objectName() == "AutoModeMenu"
    assert [a.text() for a in menu.actions()] == [
        "Auto: OFF",
        "Auto: DESK",
        "Auto: AWAY",
        "Auto: EVENING",
    ]
    assert all(a.isCheckable() for a in menu.actions())


def test_choosing_a_mode_sets_it_directly():
    stub, service, menu = _menu_stub()
    actions = stub._auto_mode_actions

    actions["EVENING"].trigger()  # straight from OFF, no cycling through DESK/AWAY
    assert service.auto_mode == "EVENING"
    assert service.calls == [("profile", "EVENING"), ("enabled", True)]
    assert stub.auto_mode_button.text() == "Auto: EVENING"

    actions["DESK"].trigger()
    assert service.auto_mode == "DESK"
    assert stub.auto_mode_button.text() == "Auto: DESK"

    service.calls.clear()
    actions["OFF"].trigger()
    assert service.auto_mode == "OFF"
    assert service.calls == [("enabled", False)]
    assert stub.auto_mode_button.text() == "Auto: OFF"


def test_current_mode_is_checked():
    stub, service, menu = _menu_stub()
    from ui.app import MainWindow

    def checked():
        return [m for m, a in stub._auto_mode_actions.items() if a.isChecked()]

    assert checked() == ["OFF"]
    service.enabled = True
    service.profile = "AWAY"
    MainWindow._sync_auto_mode_button(stub)
    assert checked() == ["AWAY"]
    # An outside flip (Auto Pilot panel) is picked up when the menu is about to show.
    service.profile = "EVENING"
    menu.aboutToShow.emit()
    assert checked() == ["EVENING"]


def test_cycle_method_is_gone():
    from ui.app import MainWindow

    assert not hasattr(MainWindow, "_cycle_auto_mode")

