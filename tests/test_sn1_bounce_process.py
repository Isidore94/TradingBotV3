from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
FIXTURE = Path(__file__).with_name("fixtures") / "sn1_fake_bot.py"


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_child_liveness_survives_a_process_wide_pid_diagnostic():
    from ui.services.bounce_process import BounceProcessProxy

    class ForeignParentView:
        exitcode = None

        @staticmethod
        def is_alive():
            raise AssertionError("can only test a child process")

    proxy = object.__new__(BounceProcessProxy)
    proxy._process = ForeignParentView()
    assert proxy._is_alive() is True
    proxy._process.exitcode = 0
    assert proxy._is_alive() is False


def test_scanner_child_is_separate_below_normal_and_preserves_callback_order():
    from ui.services.bounce_process import BounceProcessProxy

    expected = []
    spec = importlib.util.spec_from_file_location("sn1_direct", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    direct = module.run_bot_with_gui(lambda message, tag: expected.append((message, tag)), True)

    actual = []
    proxy = BounceProcessProxy(
        lambda message, tag: actual.append((message, tag)),
        start_scanning_enabled=True,
        launcher_spec=f"{FIXTURE}:run_bot_with_gui",
    )
    try:
        assert proxy.pid != os.getpid()
        assert proxy.process.is_alive()
        assert proxy.priority_below_normal is True
        assert proxy.is_scanning_enabled() is True
        assert proxy.m5_chart_bars("NVDA") == direct.m5_chart_bars("NVDA")
        assert _wait_for(lambda: len(actual) == len(expected))
        assert actual == expected
    finally:
        proxy.stop(timeout=2.0)
    assert not proxy.process.is_alive()


def test_bounce_service_replaces_a_dead_child_and_reconnects_its_callbacks():
    from PySide6.QtWidgets import QApplication
    from ui.services.bounce_service import BounceService

    app = QApplication.instance() or QApplication([])
    service = BounceService(process_launcher_spec=f"{FIXTURE}:run_bot_with_gui")
    statuses = []
    service.rrsStatusChanged.connect(statuses.append)
    assert service.start() is True
    assert _wait_for(lambda: (app.processEvents() or True) and service.current_bot() is not None)
    first = service.current_bot()
    first_pid = first.pid
    first.process.terminate()
    first.process.join(3.0)
    assert not first.process.is_alive()

    service.refresh_health()
    assert _wait_for(
        lambda: (app.processEvents() or True)
        and service.current_bot() is not None
        and service.current_bot().pid != first_pid
    )
    assert _wait_for(lambda: (app.processEvents() or True) and len(statuses) >= 2)
    service.shutdown()
    assert service.current_bot() is None
