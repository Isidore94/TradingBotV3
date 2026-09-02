"""A test run must never be able to launch a real market scan.

2026-09-02. The suite printed **6,145 passed** and the process then died with
`QThread: Destroyed while thread '' is still running` and a Windows fast-fail
exit code. Nothing had failed. The cause was not a test and not a merge:

* five test files build a real `MainWindow`, which builds a real
  `AutopilotService` with live timers, and nothing shuts them down;
* a later test called `QApplication.processEvents()`;
* the surviving timer ticked, `_maybe_auto_arm` saw it was after 07:00 on a
  weekday and flipped Auto Pilot **ON**;
* `_maybe_run_swing_slot` then **started a real master scan** - a child process,
  against the live tape, on the same machine as the running desk;
* the scan outlives the session, so its `QThread` is still running when the
  interpreter tears down, and Qt aborts the process.

**It depends on the wall clock, which is why it looked like something else.**
Every clean run that week happened between 04:00 and 05:00, before the arm hour.
The first run after lunch crashed, and every run after it crashed identically.

The guard is one line of machine-local settings, written by `conftest` into the
temp LOCALAPPDATA it already isolates: `qt_autopilot_auto_arm` is False for the
suite. It is correct for that key to default True in production and indefensible
here.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _FakeBounceService:
    """Enough of the bounce service for `AutopilotService.__init__`."""

    class _Signal:
        def connect(self, *_args, **_kwargs):
            return None

        def emit(self, *_args, **_kwargs):
            return None

    def __init__(self):
        self.running = False
        self.scanning_enabled = False
        self.alertReceived = self._Signal()
        self.connectionChanged = self._Signal()


def _service(monkeypatch):
    from ui.services.autopilot_service import AutopilotService

    monkeypatch.setattr(
        AutopilotService, "_load_state", lambda self: {"enabled": False}
    )
    monkeypatch.setattr(AutopilotService, "_save_state", lambda self: None)
    monkeypatch.setattr("job_ledger.get_default_ledger", lambda: None)
    monkeypatch.setattr(AutopilotService, "_log", lambda self, message: None)
    return AutopilotService(_FakeBounceService())


def test_the_suite_can_never_auto_arm_the_autopilot(monkeypatch):
    """The tick that armed it: a Wednesday, 09:30, Auto Pilot off, never armed.

    Without the guard this arms and goes on to reach for the scanner. The
    assertion is on `set_enabled`, not on the scan, because arming is the step
    that has to be impossible - everything the autopilot does downstream of ON
    is correct behaviour for a service that believes it is on a desk.
    """
    from ui.services.autopilot_service import AutopilotService

    service = _service(monkeypatch)
    armed: list[bool] = []
    monkeypatch.setattr(
        AutopilotService, "set_enabled", lambda self, on, *a, **k: armed.append(on)
    )

    service._maybe_auto_arm(datetime(2026, 9, 2, 9, 30))

    assert armed == [], (
        "a test process armed Auto Pilot; the next tick starts a real scan"
    )


def test_the_guard_is_the_setting_and_a_test_can_still_opt_in(monkeypatch):
    """It must be a SETTING, not a patched method.

    A guard that stubbed `_maybe_auto_arm` would also delete the behaviour from
    the tests that exist to check it. This one only changes the default, so a
    test that wants arming turns it back on.
    """
    import project_paths
    from ui.services.autopilot_service import AutopilotService

    assert project_paths.get_local_setting("qt_autopilot_auto_arm", True) is False

    service = _service(monkeypatch)
    armed: list[bool] = []
    monkeypatch.setattr(
        AutopilotService, "set_enabled", lambda self, on, *a, **k: armed.append(on)
    )
    # It is imported inside the method, so the patch belongs on the source
    # module - patching the consumer's namespace would find nothing there.
    monkeypatch.setattr(
        project_paths,
        "get_local_setting",
        lambda key, default=None: True if key == "qt_autopilot_auto_arm" else default,
    )

    service._maybe_auto_arm(datetime(2026, 9, 2, 9, 30))

    assert armed == [True], "opting back in must still work"
