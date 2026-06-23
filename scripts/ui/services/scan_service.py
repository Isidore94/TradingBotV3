from __future__ import annotations

import os
import subprocess
import sys
import traceback
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


def _run_master_scan_subprocess(*, use_shared_watchlists: bool) -> dict[str, Any]:
    """Run scanner work outside the Qt process so native faults do not close the GUI."""
    code = (
        "import faulthandler; "
        "faulthandler.enable(); "
        "from master_avwap_lib.runner import run_master, run_master_with_shared_watchlists; "
        f"run_master_with_shared_watchlists() if {use_shared_watchlists!r} else run_master(); "
        "print('SCAN_SUBPROCESS_OK')"
    )
    env = os.environ.copy()
    pythonpath = str(SCRIPTS_DIR)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT_DIR),
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = "\n\n".join(part for part in (stderr, stdout) if part)
        raise RuntimeError(
            f"Master AVWAP scan process exited with code {completed.returncode}."
            + (f"\n\n{details}" if details else "")
        )
    return {
        "watchlist_label": "home folder watchlists + swing watchlists" if use_shared_watchlists else "local project watchlists",
        "subprocess_stdout": (completed.stdout or "").strip(),
    }
