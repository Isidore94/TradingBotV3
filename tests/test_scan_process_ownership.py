"""Plan.md P0 #5 / Phase 2.6: the GUI owns and reaps its scan children."""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Other tests legitimately register scan children (e.g. the marker
    tests); park them so counts here are hermetic, then restore."""
    from ui.services import scan_service as ss

    with ss._owned_processes_lock:
        parked = list(ss._owned_processes)
        ss._owned_processes.clear()
    try:
        yield
    finally:
        with ss._owned_processes_lock:
            ss._owned_processes.extend(parked)


def _spawn(code: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_lingering_child_is_terminated_within_bounds():
    from ui.services import scan_service as ss

    proc = _spawn("import time; time.sleep(60)")
    ss._register_owned_process(proc)
    assert ss.owned_scan_process_count() == 1

    started = time.monotonic()
    summary = ss.terminate_owned_scan_processes(grace_seconds=0.5)
    elapsed = time.monotonic() - started

    assert summary["terminated"] == 1
    assert proc.poll() is not None, "child must be dead after reaping"
    assert elapsed < 10, "shutdown reaping must stay bounded"
    assert ss.owned_scan_process_count() == 0


def test_naturally_finished_child_counts_as_finished():
    from ui.services import scan_service as ss

    proc = _spawn("pass")
    proc.wait(timeout=15)
    ss._register_owned_process(proc)
    summary = ss.terminate_owned_scan_processes(grace_seconds=0.5)
    # already exited before the call -> nothing left to reap
    assert summary == {"finished": 0, "terminated": 0}
    assert ss.owned_scan_process_count() == 0


def test_service_shutdown_reaps_children():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.services import scan_service as ss
    from ui.services.scan_service import ScanService

    proc = _spawn("import time; time.sleep(60)")
    ss._register_owned_process(proc)

    service = ScanService()
    service.shutdown()

    assert proc.poll() is not None, "ScanService.shutdown() must reap owned children"
