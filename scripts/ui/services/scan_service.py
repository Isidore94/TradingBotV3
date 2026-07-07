from __future__ import annotations

import os
import subprocess
import sys
import threading
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ui.models.setup import SetupRow
from ui.services.data_feed import load_latest_setup_rows, rows_from_run_result


SCRIPTS_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = SCRIPTS_DIR.parent


class ScanWorker(QObject):
    finished = Signal(dict, list, str)
    failed = Signal(str)

    def __init__(self, target: Callable[[], Any]) -> None:
        super().__init__()
        self._target = target

    @Slot()
    def run(self) -> None:
        try:
            result = self._target()
            run_result = result if isinstance(result, dict) else {}
            rows = rows_from_run_result(run_result)
            if not rows:
                rows = load_latest_setup_rows()
            stamp = datetime.now().strftime("%H:%M:%S")
            self.finished.emit(run_result, rows, stamp)
        except Exception as exc:
            details = traceback.format_exc()
            self.failed.emit(f"{exc}\n\n{details}")


class ScanService(QObject):
    started = Signal(str)
    finished = Signal(dict, list, str)
    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: ScanWorker | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def run_shared_watchlist_scan(self, label: str = "Running shared-watchlist Master AVWAP scan...") -> bool:
        return self._start(lambda: _run_master_scan_subprocess(use_shared_watchlists=True), label)

    def run_local_watchlist_scan(self) -> bool:
        return self._start(lambda: _run_master_scan_subprocess(use_shared_watchlists=False), "Running local-watchlist Master AVWAP scan...")

    def run_autopilot_scan(self, *, update_setup_tracker: bool, label: str) -> bool:
        """Shared-watchlist scan with an explicit tracker-write decision (Auto Pilot slots)."""
        return self._start(
            lambda: _run_master_scan_subprocess(
                use_shared_watchlists=True,
                update_setup_tracker=update_setup_tracker,
            ),
            label,
        )

    def _start(self, target: Callable[[], Any], label: str) -> bool:
        if self.running:
            return False

        thread = QThread(self)
        worker = ScanWorker(target)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_finished)
        worker.failed.connect(self._handle_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_thread)

        self._thread = thread
        self._worker = worker
        self.started.emit(label)
        thread.start()
        return True

    def shutdown(self) -> None:
        """Stop the worker thread on app close (best effort; waits briefly)."""
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(3000)

    @Slot(dict, list, str)
    def _handle_finished(self, run_result: dict, rows: list[SetupRow], stamp: str) -> None:
        self.finished.emit(run_result, rows, stamp)

    @Slot(str)
    def _handle_failed(self, message: str) -> None:
        self.failed.emit(message)

    @Slot()
    def _clear_thread(self) -> None:
        self._thread = None
        self._worker = None


_SCAN_OK_MARKER = "SCAN_SUBPROCESS_OK"


def _run_master_scan_subprocess(
    *,
    use_shared_watchlists: bool,
    update_setup_tracker: bool | None = None,
) -> dict[str, Any]:
    """Run scanner work outside the Qt process so native faults do not close the GUI."""
    if update_setup_tracker is None:
        run_call = f"run_master_with_shared_watchlists() if {use_shared_watchlists!r} else run_master()"
    else:
        run_call = (
            "run_master(use_shared_watchlists=True, "
            f"update_setup_tracker={bool(update_setup_tracker)!r}, "
            "require_ib_for_setup_tracker=True, include_theta=True)"
        )
    code = (
        "import faulthandler; "
        "faulthandler.enable(); "
        "from master_avwap_lib.runner import run_master, run_master_with_shared_watchlists; "
        f"{run_call}; "
        f"print('{_SCAN_OK_MARKER}', flush=True)"
    )
    env = os.environ.copy()
    pythonpath = str(SCRIPTS_DIR)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    stdout_text = _wait_for_scan_marker(
        [sys.executable, "-c", code],
        cwd=str(ROOT_DIR),
        env=env,
    )
    return {
        "watchlist_label": "home folder watchlists + swing watchlists" if use_shared_watchlists else "local project watchlists",
        "subprocess_stdout": stdout_text,
    }


def _wait_for_scan_marker(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    marker: str = _SCAN_OK_MARKER,
    tail_lines: int = 200,
) -> str:
    """Start the scan process and return once it prints the completion marker.

    run_master prints the marker only after every report file is written; the
    process then stays alive for minutes while the deferred theta option
    enrichment thread finishes. Waiting for process exit would hold the GUI's
    "scan running" state (and the next scheduler slot) hostage to that tail, so
    the marker is the success signal and the process is left to exit on its
    own. Pipes are drained by daemon threads for the process's whole life so
    the child never blocks on a full pipe. Raises RuntimeError when the
    process exits without printing the marker.
    """
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_tail: deque[str] = deque(maxlen=tail_lines)
    stderr_tail: deque[str] = deque(maxlen=tail_lines)
    marker_seen = threading.Event()

    def _drain(stream, sink: deque[str], watch_marker: bool) -> None:
        try:
            for line in stream:
                sink.append(line)
                if watch_marker and marker in line:
                    marker_seen.set()
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    drains = [
        threading.Thread(target=_drain, args=(proc.stdout, stdout_tail, True), name="scan-stdout-drain", daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, stderr_tail, False), name="scan-stderr-drain", daemon=True),
    ]
    for thread in drains:
        thread.start()

    while not marker_seen.is_set() and proc.poll() is None:
        marker_seen.wait(0.25)
    if not marker_seen.is_set():
        # The process exited; let the drains catch the final buffered lines
        # (the marker may arrive with the interpreter's exit flush).
        for thread in drains:
            thread.join(timeout=5)
    if marker_seen.is_set():
        return "".join(stdout_tail).strip()

    returncode = proc.wait()
    stderr_text = "".join(stderr_tail).strip()
    stdout_text = "".join(stdout_tail).strip()
    details = "\n\n".join(part for part in (stderr_text, stdout_text) if part)
    raise RuntimeError(
        f"Master AVWAP scan process exited with code {returncode}."
        + (f"\n\n{details}" if details else "")
    )
