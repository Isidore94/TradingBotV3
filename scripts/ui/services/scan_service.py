from __future__ import annotations

import os
import subprocess
import sys
import threading
import traceback
import uuid
import weakref
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ui.models.setup import SetupRow
from ui.services.data_feed import enrich_setup_rows_for_display, load_latest_setup_rows, rows_from_run_result


SCRIPTS_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = SCRIPTS_DIR.parent


# MasterAvwapPanel and AutopilotService each own a ScanService.  Without a
# shared claim they can both start the same heavyweight scanner, racing report
# files and competing for the same IB client IDs.  The owner is weakly held so
# a discarded Qt service cannot leave the process permanently "busy".
_active_scan_lock = threading.Lock()
_active_scan_owner: weakref.ReferenceType["ScanService"] | None = None


def _claim_active_scan(service: "ScanService") -> bool:
    global _active_scan_owner
    with _active_scan_lock:
        owner = _active_scan_owner() if _active_scan_owner is not None else None
        if owner is not None and owner is not service:
            return False
        _active_scan_owner = weakref.ref(service)
        return True


def _release_active_scan(service: "ScanService") -> None:
    global _active_scan_owner
    with _active_scan_lock:
        owner = _active_scan_owner() if _active_scan_owner is not None else None
        if owner is service or owner is None:
            _active_scan_owner = None


def active_scan_label() -> str:
    """Process-wide active scan description for heartbeat/status surfaces."""
    with _active_scan_lock:
        owner = _active_scan_owner() if _active_scan_owner is not None else None
        return str(getattr(owner, "_active_label", "") or "") if owner is not None else ""


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
            else:
                enrich_setup_rows_for_display(rows, supplemental_rows=rows)
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
        self._active_label = ""
        self._active_job_key = ""
        self._active_run_id = ""
        self._active_worker_pid: int | None = None
        self._active_job_started = False
        self._last_rejection_reason = ""
        try:
            from job_ledger import get_default_ledger

            self._job_ledger = get_default_ledger()
        except Exception:
            self._job_ledger = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @property
    def last_rejection_reason(self) -> str:
        return self._last_rejection_reason

    def run_shared_watchlist_scan(
        self,
        label: str = "Running shared-watchlist Master AVWAP scan...",
        *,
        scheduled_slot: str = "",
    ) -> bool:
        return self._start(
            lambda: _run_master_scan_subprocess(
                use_shared_watchlists=True,
                run_id=self._active_run_id,
                trigger=self._active_label,
                on_process_started=self._record_worker_pid,
            ),
            label,
            job_type="swing_scan" if scheduled_slot else "manual_master_scan",
            job_slot=scheduled_slot,
            dedupe=bool(scheduled_slot),
            config_hash="shared-v1",
        )

    def run_local_watchlist_scan(self) -> bool:
        return self._start(
            lambda: _run_master_scan_subprocess(
                use_shared_watchlists=False,
                run_id=self._active_run_id,
                trigger=self._active_label,
                on_process_started=self._record_worker_pid,
            ),
            "Running local-watchlist Master AVWAP scan...",
            job_type="manual_master_scan",
            config_hash="local-v1",
        )

    def run_autopilot_scan(self, *, update_setup_tracker: bool, label: str, slot_label: str) -> bool:
        """Shared-watchlist scan with an explicit tracker-write decision (Auto Pilot slots)."""
        return self._start(
            lambda: _run_master_scan_subprocess(
                use_shared_watchlists=True,
                update_setup_tracker=update_setup_tracker,
                run_id=self._active_run_id,
                trigger=self._active_label,
                on_process_started=self._record_worker_pid,
            ),
            label,
            job_type="swing_scan" if not str(slot_label).startswith("manual ") else "manual_master_scan",
            job_slot=str(slot_label),
            dedupe=not str(slot_label).startswith("manual "),
            config_hash="shared-v1",
        )

    def _start(
        self,
        target: Callable[[], Any],
        label: str,
        *,
        job_type: str = "manual_master_scan",
        job_slot: str = "",
        dedupe: bool = False,
        config_hash: str = "",
    ) -> bool:
        self._last_rejection_reason = ""
        if self.running:
            self._last_rejection_reason = "service busy"
            return False
        # The completion marker means reports are ready, but the child can
        # remain alive during deferred theta enrichment.  Do not let another
        # service start a new IB-heavy scanner until every owned child exits.
        if owned_scan_process_count() > 0:
            self._last_rejection_reason = "previous scan child still running"
            return False
        if not _claim_active_scan(self):
            self._last_rejection_reason = "another scan is active"
            return False

        try:
            self._active_label = str(label or "Master AVWAP scan")
            if not self._prepare_ledger_job(
                job_type=job_type,
                job_slot=job_slot,
                dedupe=dedupe,
                config_hash=config_hash,
            ):
                self._active_label = ""
                _release_active_scan(self)
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
        except Exception:
            self._fail_ledger_job("unexpected", "scan service failed before worker start")
            self._active_label = ""
            _release_active_scan(self)
            raise

    def _prepare_ledger_job(
        self,
        *,
        job_type: str,
        job_slot: str,
        dedupe: bool,
        config_hash: str,
    ) -> bool:
        from job_ledger import job_key

        now = datetime.now()
        slot = str(job_slot or now.strftime("manual-%H%M%S-%f"))
        key = job_key(now.date().isoformat(), job_type, slot, config_hash)
        ledger = self._job_ledger
        if ledger is not None and dedupe:
            if ledger.is_done(key):
                self._last_rejection_reason = "scheduled slot already completed"
                return False
            if ledger.is_active(key):
                self._last_rejection_reason = "scheduled slot already active"
                return False
        self._active_job_key = key
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._active_run_id = f"master_scan-{stamp}-{uuid.uuid4().hex[:8]}"
        self._active_worker_pid = None
        self._active_job_started = False
        if ledger is not None:
            ledger.schedule(
                now.date().isoformat(),
                job_type,
                slot,
                config_hash=config_hash,
                now=now,
            )
        return True

    def _record_worker_pid(self, worker_pid: int) -> None:
        self._active_worker_pid = int(worker_pid)
        ledger = self._job_ledger
        if ledger is not None and self._active_job_key and not self._active_job_started:
            ledger.start(
                self._active_job_key,
                run_id=self._active_run_id,
                worker_pid=self._active_worker_pid,
            )
            self._active_job_started = True

    def _complete_ledger_job(self) -> None:
        ledger = self._job_ledger
        if ledger is not None and self._active_job_key:
            ledger.complete(self._active_job_key, run_id=self._active_run_id)

    def _fail_ledger_job(self, error_class: str, error: str) -> None:
        ledger = self._job_ledger
        if ledger is not None and self._active_job_key:
            ledger.fail(self._active_job_key, error_class=error_class, error=error)

    def shutdown(self) -> None:
        """Stop the worker thread on app close (best effort; waits briefly),
        then reap every scan child this process spawned - a closed desk must
        not leave a multi-GB scanner running invisibly (plan.md P0 #5)."""
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(3000)
        summary = terminate_owned_scan_processes()
        if summary["finished"] or summary["terminated"]:
            import logging

            logging.info(
                "Scan children reaped at shutdown: %s finished, %s terminated.",
                summary["finished"],
                summary["terminated"],
            )

    @Slot(dict, list, str)
    def _handle_finished(self, run_result: dict, rows: list[SetupRow], stamp: str) -> None:
        self._complete_ledger_job()
        payload = dict(run_result or {})
        payload.setdefault("run_id", self._active_run_id)
        payload.setdefault("worker_pid", self._active_worker_pid)
        self.finished.emit(payload, rows, stamp)

    @Slot(str)
    def _handle_failed(self, message: str) -> None:
        error_class = "ib_disconnected" if "IB" in str(message or "") else "unexpected"
        self._fail_ledger_job(error_class, str(message or ""))
        self.failed.emit(message)

    @Slot()
    def _clear_thread(self) -> None:
        self._thread = None
        self._worker = None
        self._active_label = ""
        self._active_job_key = ""
        self._active_run_id = ""
        self._active_worker_pid = None
        self._active_job_started = False
        _release_active_scan(self)


_SCAN_OK_MARKER = "SCAN_SUBPROCESS_OK"

# Every scan subprocess this GUI spawns is registered here so shutdown can
# reap it (plan.md P0 #5 / Phase 2.6): the marker-based early return means a
# child (theta tail included) can outlive its scan - it must never outlive
# the application unnoticed.
_owned_processes_lock = threading.Lock()
_owned_processes: list[subprocess.Popen] = []


def _register_owned_process(proc: subprocess.Popen) -> None:
    with _owned_processes_lock:
        _owned_processes[:] = [p for p in _owned_processes if p.poll() is None]
        _owned_processes.append(proc)


def owned_scan_process_count() -> int:
    """Live scan children owned by this GUI (health/status surface)."""
    with _owned_processes_lock:
        _owned_processes[:] = [p for p in _owned_processes if p.poll() is None]
        return len(_owned_processes)


def terminate_owned_scan_processes(grace_seconds: float = 3.0) -> dict[str, int]:
    """Bounded-graceful reap of every owned child: wait briefly for a natural
    exit, then terminate. Only processes this GUI spawned are touched."""
    with _owned_processes_lock:
        procs = [p for p in _owned_processes if p.poll() is None]
        _owned_processes.clear()
    summary = {"finished": 0, "terminated": 0}
    for proc in procs:
        try:
            proc.wait(timeout=max(0.0, grace_seconds))
            summary["finished"] += 1
            continue
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
            summary["terminated"] += 1
        except Exception:
            pass
    return summary


def _run_master_scan_subprocess(
    *,
    use_shared_watchlists: bool,
    update_setup_tracker: bool | None = None,
    run_id: str = "",
    trigger: str = "",
    on_process_started: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run scanner work outside the Qt process so native faults do not close the GUI."""
    if update_setup_tracker is None:
        run_call = f"run_master_with_shared_watchlists() if {use_shared_watchlists!r} else run_master()"
    else:
        run_call = (
            "run_master(use_shared_watchlists=True, "
            f"update_setup_tracker={bool(update_setup_tracker)!r}, "
            "require_ib_for_setup_tracker=True)"
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
    if run_id:
        env["TRADINGBOT_RUN_ID"] = str(run_id)
    if trigger:
        env["TRADINGBOT_RUN_TRIGGER"] = str(trigger)
    stdout_text = _wait_for_scan_marker(
        [sys.executable, "-c", code],
        cwd=str(ROOT_DIR),
        env=env,
        on_process_started=on_process_started,
    )
    return {
        "watchlist_label": "home folder watchlists + swing watchlists" if use_shared_watchlists else "local project watchlists",
        "subprocess_stdout": stdout_text,
        "run_id": str(run_id or ""),
    }


def _wait_for_scan_marker(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    marker: str = _SCAN_OK_MARKER,
    tail_lines: int = 200,
    on_process_started: Callable[[int], None] | None = None,
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
    _register_owned_process(proc)
    if on_process_started is not None:
        on_process_started(proc.pid)
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
