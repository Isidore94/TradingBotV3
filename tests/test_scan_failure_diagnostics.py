"""P0-2 2a: a failed swing scan keeps the child's stderr tail, and run_master logs memory.

Two of the last three desk scan failures had no recorded cause because the
autopilot log kept only the first line. These pin the 40-line tail in
autopilot.log, one scan_failures.jsonl row, and the ``[run_master memory]`` line.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import project_paths  # noqa: E402


def _child_program(lines: list[str]) -> str:
    return chr(10).join(["import sys"] + lines)


def test_failed_child_carries_exit_code_and_last_40_stderr_lines():
    from ui.services.scan_service import ScanChildFailed, _wait_for_scan_marker

    code = _child_program([
        "for i in range(60): print('stderr line %d' % i, file=sys.stderr)",
        "print('MemoryError', file=sys.stderr)",
        "sys.exit(3)",
    ])
    with pytest.raises(ScanChildFailed) as excinfo:
        _wait_for_scan_marker([sys.executable, "-c", code], cwd=os.getcwd(), env=os.environ.copy())
    exc = excinfo.value
    assert exc.returncode == 3
    assert len(exc.stderr_tail) == 40
    assert exc.stderr_tail[-1] == "MemoryError"
    assert exc.stderr_tail[0] == "stderr line 21"
    # The first line keeps its old shape (feed + phone report read it).
    assert str(exc).splitlines()[0].startswith("Master AVWAP scan process exited with code 3.")


def test_record_scan_failure_writes_a_tz_aware_row(tmp_path):
    from ui.services.scan_service import record_scan_failure

    path = tmp_path / "scan_failures.jsonl"
    assert record_scan_failure(slot="13:00", exit_code=1, stderr_tail=["a", "b"], path=path)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["slot"] == "13:00"
    assert row["exit_code"] == 1
    assert row["stderr_tail"] == ["a", "b"]
    assert datetime.fromisoformat(row["ts"]).tzinfo is not None


def test_record_scan_failure_never_raises(tmp_path):
    from ui.services.scan_service import record_scan_failure

    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert record_scan_failure(slot="x", exit_code=1, stderr_tail=[], path=blocker / "sub" / "f.jsonl") is False


def test_autopilot_failed_scan_logs_tail_and_writes_failure_row(tmp_path, monkeypatch):
    import ui.services.autopilot_service as aps

    log_file = tmp_path / "autopilot.log"
    failures = tmp_path / "scan_failures.jsonl"
    monkeypatch.setattr(aps, "AUTOPILOT_LOG_FILE", log_file)
    monkeypatch.setattr(project_paths, "SCAN_FAILURES_FILE", failures)

    service = aps.AutopilotService.__new__(aps.AutopilotService)
    service._active_scan_slot = "12:45"
    service._waiting_scan_slot = None
    tail = [f"line {i}" for i in range(40)]
    service._scan_service = SimpleNamespace(last_failure={"exit_code": 1, "stderr_tail": tail})
    logged: list[str] = []
    service._log = logged.append
    service._mark_slots_done = lambda: None
    service._request_report_write = lambda: None
    service._maybe_run_wrapup = lambda now: None

    aps.AutopilotService._on_scan_failed(service, "Master AVWAP scan process exited with code 1.\n\nline 0")

    assert logged == ["Swing scan for slot 12:45 FAILED: Master AVWAP scan process exited with code 1."]
    text = log_file.read_text(encoding="utf-8")
    assert all(f"    | line {i}\n" in text for i in range(40))
    row = json.loads(failures.read_text(encoding="utf-8").splitlines()[0])
    assert row["slot"] == "12:45" and row["exit_code"] == 1 and row["stderr_tail"] == tail


@pytest.mark.skipif(sys.platform != "win32", reason="GetProcessMemoryInfo is Windows-only")
def test_process_memory_mb_reads_this_process():
    from master_avwap_lib.runner import process_memory_mb

    memory = process_memory_mb()
    assert memory is not None
    peak, current = memory
    assert peak >= current > 1.0


def test_process_memory_mb_is_a_no_op_off_windows(monkeypatch):
    from master_avwap_lib import runner

    monkeypatch.setattr(runner.sys, "platform", "linux")
    assert runner.process_memory_mb() is None


def test_phase_log_carries_a_memory_line(monkeypatch, caplog):
    from master_avwap_lib import runner

    monkeypatch.setattr(runner, "process_memory_mb", lambda: (512.4, 300.0))
    with caplog.at_level(logging.INFO):
        runner._log_phase_duration("studies", 0.0)
    assert any("[run_master memory] studies peak_ws=512 ws=300" in r.getMessage() for r in caplog.records)
