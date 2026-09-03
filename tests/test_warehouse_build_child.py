"""The post-scan warehouse build is a CHILD PROCESS, never a desk thread.

Packet F1 item 2, 2026-09-03. The build ran on a ``qt-warehouse-build`` thread
inside the desk process. py-spy on pid 11612 (08:45-08:55 PT) measured that
thread holding the GIL in **82.7%** of samples while ``MainThread`` got
**2.3%**; WM_NULL pings to the desk window hung 100-606 ms every few seconds.
The build's outcomes stage runs **27-57 minutes** per scan (``manifest_log``,
09-01 to 09-03) and there are four scans a session, all inside RTH - so the
freeze was the whole morning, and no priority or timer trick can fix it: a
CPU-bound Python thread holds the GIL by construction.

What this file pins:

* the child's argv, in both build shapes - a frozen ``sys.executable`` is
  ``TradingBotV3.exe`` and cannot be handed ``-m``;
* that starting a build spawns exactly ONE process, at BELOW_NORMAL priority,
  carrying the run id, and that a second start while it lives is refused
  rather than stacked;
* that an unconfigured warehouse spawns nothing at all;
* that ``launch_gui --warehouse-build <run_id>`` answers before the desk's
  own startup work, the way ``--run-scan`` does;
* that no thread in ``scan_service`` runs a build any more.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from ui.services import scan_service as scan_service_mod  # noqa: E402
from ui.services.scan_service import ScanService, warehouse_build_command  # noqa: E402


class _FakePopen:
    """A child that stays alive until the test releases it.

    It has to actually stay alive: the service starts a waiter thread that
    calls ``communicate()`` at once, and a fake that returns immediately would
    make every "one build at a time" assertion below a coin flip.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pid = 4242
        self.returncode = None
        self._exit = threading.Event()

    def _finish(self) -> None:
        self.returncode = 0
        self._exit.set()

    def poll(self):
        return 0 if self._exit.is_set() else None

    def wait(self, timeout=None):
        self._finish()
        return 0

    def communicate(self, *a, **k):
        self._exit.wait(30.0)
        return ("", "")

    def terminate(self):
        self._finish()


@pytest.fixture
def spawned(monkeypatch):
    """Capture every Popen the service makes, and enable the warehouse."""
    import research_warehouse.config as config

    monkeypatch.setattr(config, "warehouse_enabled", lambda: True)
    calls: list[_FakePopen] = []

    def fake_popen(*args, **kwargs):
        proc = _FakePopen(*args, **kwargs)
        calls.append(proc)
        return proc

    monkeypatch.setattr(scan_service_mod.subprocess, "Popen", fake_popen)
    return calls


# --- (a) the command shape -------------------------------------------------
def test_the_source_build_command_runs_the_warehouse_cli_module(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert warehouse_build_command("run-42") == [
        sys.executable,
        "-m",
        "research_warehouse.cli",
        "build",
        "--run-id",
        "run-42",
    ]


def test_the_frozen_build_command_uses_the_apps_own_flag(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    command = warehouse_build_command("run-42")
    assert command == [sys.executable, scan_service_mod.WAREHOUSE_BUILD_FLAG, "run-42"]
    # The frozen exe parses `-m` as its own CLI and would exit without
    # building anything, which is the bug that killed every scheduled scan in
    # 2026-08. Never let it back in.
    assert "-m" not in command


# --- (b) one child, below normal, carrying the run id ----------------------
def test_a_build_spawns_one_below_normal_child_with_the_run_id(spawned):
    service = ScanService()
    try:
        assert service.start_warehouse_build("r1") is True
        assert len(spawned) == 1
        proc = spawned[0]
        assert proc.args[0] == warehouse_build_command("r1")
        assert proc.kwargs["env"]["TRADINGBOT_RUN_ID"] == "r1"
        assert str(SCRIPTS_DIR) in proc.kwargs["env"]["PYTHONPATH"]
        assert proc.kwargs["cwd"] == str(scan_service_mod.ROOT_DIR)
        below_normal = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        if below_normal:  # Windows; elsewhere the flags are 0 by design
            assert proc.kwargs["creationflags"] & below_normal
        # stdout is discarded (the CLI prints a JSON report nobody reads here)
        # but stderr is kept, because a failed build must be able to say why.
        assert proc.kwargs["stdout"] == subprocess.DEVNULL
        assert proc.kwargs["stderr"] == subprocess.PIPE
    finally:
        service.wait_for_warehouse_build(5.0)


def test_a_second_build_while_one_lives_is_refused_not_stacked(spawned):
    service = ScanService()
    try:
        assert service.start_warehouse_build("r1") is True
        assert service.start_warehouse_build("r2") is False
        assert len(spawned) == 1, "a second child was spawned over a live build"
    finally:
        service.wait_for_warehouse_build(5.0)


def test_a_build_is_owned_so_shutdown_can_reap_it(spawned):
    before = scan_service_mod.owned_scan_process_count()
    service = ScanService()
    try:
        assert service.start_warehouse_build("r1") is True
        assert scan_service_mod.owned_scan_process_count() == before + 1
    finally:
        service.wait_for_warehouse_build(5.0)
        scan_service_mod.terminate_owned_scan_processes(grace_seconds=0.0)


# --- (c) no store, no child ------------------------------------------------
def test_a_disabled_warehouse_spawns_nothing(monkeypatch):
    import research_warehouse.config as config

    monkeypatch.setattr(config, "warehouse_enabled", lambda: False)
    calls = []
    monkeypatch.setattr(
        scan_service_mod.subprocess, "Popen", lambda *a, **k: calls.append(1)
    )
    service = ScanService()
    assert service.start_warehouse_build("r1") is False
    assert calls == []


# --- (d) the frozen entry point -------------------------------------------
def test_launch_gui_answers_the_build_flag_before_anything_else(monkeypatch):
    import launch_gui
    import research_warehouse.cli as warehouse_cli

    seen: list[list[str]] = []
    monkeypatch.setattr(warehouse_cli, "main", lambda argv: seen.append(argv) or 7)

    def explode() -> None:  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("the build child must never touch the desk's crash log")

    monkeypatch.setattr(launch_gui, "_enable_crash_log", explode)
    monkeypatch.setattr(sys, "argv", ["launch_gui.py", "--warehouse-build", "r1"])

    assert launch_gui.main() == 7
    assert seen == [["build", "--run-id", "r1"]]


# --- (e) nothing builds on a thread any more ------------------------------
def test_no_thread_in_scan_service_runs_a_warehouse_build():
    source = Path(scan_service_mod.__file__).read_text(encoding="utf-8")
    assert "qt-warehouse-build\"" not in source, (
        "the in-process build thread is back; a CPU-bound Python thread holds "
        "the GIL and the desk freezes (2026-09-03)"
    )
    assert "_run_warehouse_build" not in source
    # The only thread left is the one that BLOCKS on the child's pipe, which
    # holds no GIL while it waits.
    assert "qt-warehouse-build-wait" in source
    assert "run_build" not in source, "the build must not be importable inline again"
