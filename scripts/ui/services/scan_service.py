from __future__ import annotations

import json
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
        # The research warehouse's post-scan build (plan sec 8.4, LD-01: one
        # post-scan/EOD CLI build job, no daemon). Owned here because this is
        # where a scan finishes; it runs on its own thread and can never affect
        # the scan that triggered it.
        self._warehouse_proc: subprocess.Popen | None = None
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

    #: Job-ledger idempotency token. Deliberately unchanged when the shared/
    #: local vocabulary was retired in packet R1: this is an opaque key, not
    #: user-facing text, and renaming it would orphan every in-flight ledger
    #: row on the changeover day for no gain.
    _SCAN_CONFIG_HASH = "shared-v1"

    def run_watchlist_scan(
        self,
        label: str = "Running Master AVWAP scan...",
        *,
        scheduled_slot: str = "",
    ) -> bool:
        """The one Master AVWAP scan.

        This used to be a Shared/Local pair. Both ran the identical scan over
        the identical two files - `resolve_scan_watchlist_paths` returned
        `(LONGS_FILE, SHORTS_FILE)` either way - so the choice the menu offered
        the trader was never a choice at all.
        """
        return self._start(
            lambda: _run_master_scan_subprocess(
                run_id=self._active_run_id,
                trigger=self._active_label,
                on_process_started=self._record_worker_pid,
                # The tree's own distinction, reused rather than re-invented:
                # a run with no scheduled slot is already `manual_master_scan`
                # to the job ledger, so it is a manual tracker write too.
                saved_by="close_slot" if scheduled_slot else "manual",
            ),
            label,
            job_type="swing_scan" if scheduled_slot else "manual_master_scan",
            job_slot=scheduled_slot,
            dedupe=bool(scheduled_slot),
            config_hash=self._SCAN_CONFIG_HASH,
        )

    def run_autopilot_scan(self, *, update_setup_tracker: bool, label: str, slot_label: str) -> bool:
        """The same scan with an explicit tracker-write decision (Auto Pilot slots)."""
        return self._start(
            lambda: _run_master_scan_subprocess(
                update_setup_tracker=update_setup_tracker,
                run_id=self._active_run_id,
                trigger=self._active_label,
                on_process_started=self._record_worker_pid,
                saved_by=(
                    "manual" if str(slot_label).startswith("manual ") else "close_slot"
                ),
            ),
            label,
            job_type="swing_scan" if not str(slot_label).startswith("manual ") else "manual_master_scan",
            job_slot=str(slot_label),
            dedupe=not str(slot_label).startswith("manual "),
            config_hash=self._SCAN_CONFIG_HASH,
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
        # A build mid-seal must finish its manifest line rather than be cut off.
        self.wait_for_warehouse_build(timeout=5.0)
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
        self.start_warehouse_build(str(payload.get("run_id") or ""))

    def start_warehouse_build(self, run_id: str = "") -> bool:
        """Seal the spool and run the EOD build after a scan, IN A CHILD.

        This is the "post-scan" half of LD-01's *post-scan/EOD CLI build job*,
        and LD-01 said CLI for a reason. Without it the GUI tee spools M5 bars
        every minute and nothing ever seals them - and because M5 segments are
        PROTECTED and never shed (LD-12/BD-18), the backlog would simply grow
        until Health went red.

        It used to run on a ``qt-warehouse-build`` THREAD inside the desk, and
        on 2026-09-03 that made the desk unusable for a morning: py-spy on pid
        11612 measured that thread holding the GIL in **82.7%** of samples
        while ``MainThread`` got **2.3%**, with WM_NULL pings to the desk
        window hanging 100-606 ms every few seconds. A CPU-bound Python thread
        holds the GIL; no priority, timer or chunking trick gives it back. The
        build is 27-57 minutes of pure Python per scan (measured in
        ``manifest_log.jsonl``, 09-01 to 09-03), four scans a day, all inside
        RTH. So it leaves the process entirely, at BELOW_NORMAL priority,
        where the OS scheduler rather than the GIL decides who runs.

        Never blocks, never raises, and never touches the scan: one build at a
        time (a second is skipped, and the build's own single-flight lock
        refuses a concurrent one from any other process anyway, reclaiming a
        dead holder's lock so a reaped child cannot wedge the next build).
        """
        import logging

        proc = self._warehouse_proc
        if proc is not None and proc.poll() is None:
            return False  # the next scan picks up whatever this one misses
        try:
            if str(SCRIPTS_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPTS_DIR))
            from research_warehouse import config as warehouse_config

            # Asked in the PARENT: with no research store there is nothing to
            # build, and spawning an interpreter four times a day to discover
            # that is a cost with no answer behind it.
            if not warehouse_config.warehouse_enabled():
                return False
        except Exception:
            logging.exception("Research warehouse build not started; the scan is unaffected.")
            return False
        try:
            child = subprocess.Popen(
                warehouse_build_command(str(run_id or "")),
                cwd=str(ROOT_DIR),
                env=_scan_child_env(run_id=str(run_id or "")),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                # Windows-only flags, read BY NAME so macOS still launches.
                creationflags=(
                    getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
                    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                ),
            )
        except Exception:
            # Research evidence must never be able to break a scan.
            logging.exception(
                "Research warehouse post-scan build failed to start; the scan is unaffected."
            )
            return False
        self._warehouse_proc = child
        # Owned, so shutdown reaps it: a closed desk must not leave a multi-GB
        # build running invisibly (plan.md P0 #5). Flagged as a BUILD, so it is
        # not counted as a scan child - `ScanService.start` refuses a new scan
        # while one of those is alive, and a research build must never be the
        # reason a scheduled scan does not run.
        _register_owned_process(child, is_build=True)
        threading.Thread(
            target=self._await_warehouse_build,
            args=(child,),
            name="qt-warehouse-build-wait",
            daemon=True,
        ).start()
        return True

    def _await_warehouse_build(self, proc: subprocess.Popen) -> None:
        """Reap the build child and say how it went.

        This thread blocks on the child's pipe, so it holds no GIL while it
        waits - which is the entire point of the change above. It also drains
        stderr, so a chatty failure cannot fill the pipe and wedge the build.
        """
        import logging

        try:
            _, stderr_text = proc.communicate()
        except Exception:  # pragma: no cover - a reaped child races here
            return
        code = proc.returncode
        if code:
            tail = "\n".join(str(stderr_text or "").strip().splitlines()[-20:])
            logging.warning(
                "Research warehouse build child exited %s%s",
                code,
                f"\n{tail}" if tail else "",
            )
        else:
            logging.info("Research warehouse build child exited %s", code)

    def wait_for_warehouse_build(self, timeout: float = 30.0) -> None:
        """Test/shutdown helper: wait on the in-flight build CHILD, if any.

        A build mid-seal is given its moment to finish its manifest line; past
        the timeout it is left to the shutdown reap, which is safe because the
        build's single-flight lock reclaims a dead holder rather than obeying
        it.
        """
        proc = self._warehouse_proc
        if proc is None:
            return
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

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


#: One definition, imported rather than restated: the worker prints it and the
#: parent waits for it, so a drifting copy would hang every scan.
from scan_worker import SCAN_OK_MARKER as _SCAN_OK_MARKER  # noqa: E402

# Every scan subprocess this GUI spawns is registered here so shutdown can
# reap it (plan.md P0 #5 / Phase 2.6): the marker-based early return means a
# child (theta tail included) can outlive its scan - it must never outlive
# the application unnoticed.
_owned_processes_lock = threading.Lock()
_owned_processes: list[subprocess.Popen] = []
#: The subset of the above that are research-warehouse BUILD children. They are
#: owned - shutdown reaps them and the Health accounting sees them - but they
#: are not SCAN children, and the distinction is load-bearing: `ScanService.start`
#: refuses a new scan while a scan child is alive, and a research build that can
#: run for tens of minutes must never be the reason a scheduled scan does not
#: run. Held as a second list rather than a pid set because the OS recycles pids
#: and identity here must not depend on it.
_owned_build_processes: list[subprocess.Popen] = []


def _register_owned_process(proc: subprocess.Popen, *, is_build: bool = False) -> None:
    with _owned_processes_lock:
        _owned_processes[:] = [p for p in _owned_processes if p.poll() is None]
        _owned_processes.append(proc)
        if is_build:
            _owned_build_processes[:] = [
                p for p in _owned_build_processes if p.poll() is None
            ]
            _owned_build_processes.append(proc)


def owned_scan_process_count() -> int:
    """Live SCAN children owned by this GUI (health/status surface).

    Build children are deliberately excluded - see `_owned_build_processes`.
    `owned_build_process_count` answers for those, and
    `owned_scan_process_snapshot` still accounts for every owned child, because
    that one is about reaping rather than about whether a scan may start.
    """
    with _owned_processes_lock:
        _owned_processes[:] = [p for p in _owned_processes if p.poll() is None]
        _owned_build_processes[:] = [
            p for p in _owned_build_processes if p.poll() is None
        ]
        builds = {id(p) for p in _owned_build_processes}
        return len([p for p in _owned_processes if id(p) not in builds])


def owned_build_process_count() -> int:
    """Live research-warehouse build children owned by this GUI."""
    with _owned_processes_lock:
        _owned_build_processes[:] = [
            p for p in _owned_build_processes if p.poll() is None
        ]
        return len(_owned_build_processes)


def owned_scan_process_snapshot() -> dict[str, Any]:
    """Read-only accounting of this process's scan children and worker threads.

    plan.md sec 6.3 requires the Health page to show owned process/thread
    counts, and sec 6.1's after-session checklist requires "owned child-process
    count returns to zero" and "no scanner or worker remains orphaned". Both are
    answerable only from *inside* the process that owns the children, so this is
    the accounting hook the audit reads.

    Strictly observational: unlike :func:`owned_scan_process_count` it does not
    prune the registry, it never registers, reaps, claims or releases anything,
    and it holds each lock only long enough to copy the state out. ``poll()`` is
    the only way to learn whether a child is alive; it changes no ownership.
    """
    with _owned_processes_lock:
        tracked = list(_owned_processes)
    children: list[dict[str, Any]] = []
    for proc in tracked:
        try:
            returncode = proc.poll()
        except Exception:  # pragma: no cover - a handle that can no longer be polled
            returncode = None
        children.append(
            {
                "pid": getattr(proc, "pid", None),
                "returncode": returncode,
                "running": returncode is None,
            }
        )
    live = [child for child in children if child["running"]]

    # One lock acquisition for both the owner and its label: active_scan_label()
    # takes the same non-reentrant lock, so calling it from inside would deadlock.
    with _active_scan_lock:
        owner = _active_scan_owner() if _active_scan_owner is not None else None
        active_label = str(getattr(owner, "_active_label", "") or "") if owner is not None else ""
    try:
        scan_worker_threads = 1 if (owner is not None and owner.running) else 0
    except Exception:  # pragma: no cover - deleted Qt wrapper
        scan_worker_threads = 0

    threads = list(threading.enumerate())
    return {
        "process_pid": os.getpid(),
        "owned_child_count": len(live),
        "registered_child_count": len(children),
        "exited_children_pending_cleanup": len(children) - len(live),
        "lingering_child_pids": [child["pid"] for child in live],
        "children": children,
        "active_scan_label": active_label,
        "scan_owner_claimed": owner is not None,
        "scan_worker_threads": scan_worker_threads,
        "python_thread_count": len(threads),
        "non_daemon_thread_count": sum(1 for thread in threads if not thread.daemon),
        "thread_names": sorted(str(thread.name) for thread in threads),
    }


def terminate_owned_scan_processes(grace_seconds: float = 3.0) -> dict[str, int]:
    """Bounded-graceful reap of every owned child: wait briefly for a natural
    exit, then terminate. Only processes this GUI spawned are touched."""
    with _owned_processes_lock:
        procs = [p for p in _owned_processes if p.poll() is None]
        _owned_processes.clear()
        _owned_build_processes.clear()
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


#: CLI flag the frozen application answers before its own argument parser runs,
#: mirroring ``--selftest``. See ``launch_gui.main``.
SCAN_WORKER_FLAG = "--run-scan"


def scan_worker_command(payload: str) -> list[str]:
    """Argv that runs one scan in a child process, correct for this build.

    A frozen build **cannot** use ``sys.executable -c``: ``sys.executable`` is
    ``TradingBotV3.exe``, which parses ``-c`` as its own CLI and exits 2 before
    fetching a bar. That killed every scheduled swing scan from 2026-08-12 07:30
    onward on the desk, silently, while everything running in-process kept
    working (see ``scripts/scan_worker.py``).

    Both forms call :func:`scan_worker.run` with the same payload, so the work
    is identical and only the transport differs. The source form keeps ``-c``
    rather than a script path because the child resolves ``scan_worker`` through
    the ``PYTHONPATH`` this module already sets, which a frozen bundle has no
    equivalent of.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, SCAN_WORKER_FLAG, payload]
    return [
        sys.executable,
        "-c",
        f"from scan_worker import run; run({payload!r})",
    ]


#: CLI flag the frozen application answers for the post-scan research build,
#: exactly as it answers ``--run-scan``. See ``launch_gui.main``.
WAREHOUSE_BUILD_FLAG = "--warehouse-build"


def warehouse_build_command(run_id: str) -> list[str]:
    """Argv that runs one post-scan warehouse build in a child, per build.

    Same shape rule as :func:`scan_worker_command`, and for the same reason: a
    frozen build's ``sys.executable`` is ``TradingBotV3.exe``, which parses
    ``-m`` as its own CLI and exits before building anything, so the frozen
    form goes through the flag the app answers itself. The source form invokes
    the module the warehouse already exposes - ``research_warehouse.cli`` has
    parsed ``build --run-id`` since Phase 8 - rather than a second entry point
    that could drift from it.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, WAREHOUSE_BUILD_FLAG, str(run_id or "")]
    return [
        sys.executable,
        "-m",
        "research_warehouse.cli",
        "build",
        "--run-id",
        str(run_id or ""),
    ]


def _scan_child_env(*, run_id: str = "", trigger: str = "") -> dict[str, str]:
    """The environment every child this module spawns inherits.

    One definition, because the scan child and the warehouse build child must
    resolve the same first-party packages and stamp the same run id: a second
    copy of these four lines is a place for the two to silently diverge.
    """
    env = os.environ.copy()
    pythonpath = str(SCRIPTS_DIR)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    if run_id:
        env["TRADINGBOT_RUN_ID"] = str(run_id)
    if trigger:
        env["TRADINGBOT_RUN_TRIGGER"] = str(trigger)
    return env


def _run_master_scan_subprocess(
    *,
    update_setup_tracker: bool | None = None,
    saved_by: str = "close_slot",
    run_id: str = "",
    trigger: str = "",
    on_process_started: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run scanner work outside the Qt process so native faults do not close the GUI."""
    payload = json.dumps(
        {
            "saved_by": str(saved_by or "manual"),
            "update_setup_tracker": (
                None if update_setup_tracker is None else bool(update_setup_tracker)
            ),
        },
        sort_keys=True,
    )
    env = _scan_child_env(run_id=run_id, trigger=trigger)
    stdout_text = _wait_for_scan_marker(
        scan_worker_command(payload),
        cwd=str(ROOT_DIR),
        env=env,
        on_process_started=on_process_started,
    )
    return {
        # There is one set of watchlists. The old "local project watchlists"
        # label was attached to a branch that read the identical two files, so
        # it named a distinction that never existed (packet R1).
        "watchlist_label": "home folder watchlists + swing watchlists",
        "subprocess_stdout": stdout_text,
        "run_id": str(run_id or ""),
    }


#: Cap on the cause appended to the failure's first line. Auto Pilot's activity
#: feed and the phone report both render that line; a 5,000-character exception
#: would swamp both.
_FAILURE_SUMMARY_MAX_CHARS = 240


def child_failure_summary(stderr_text: str) -> str:
    """The child's own final exception line, for the first line of the error.

    ``AutopilotService._on_scan_failed`` writes ``detail.splitlines()[0]`` to
    ``autopilot.log``. With the cause only in the *later* lines, three real desk
    failures (2026-08-17 07:30 and 10:00, 2026-08-18 12:00) each read

        Swing scan for slot 12:00 FAILED: Master AVWAP scan process exited with
        code 1.

    and named nothing; identifying them needed the run manifest and a log that
    had since rotated. A Python traceback ends with an *unindented* exception
    line, so that is what is lifted. A child that dies without one (a native
    fault, a kill) leaves this empty and the message keeps its old shape rather
    than quoting a random stack frame.
    """
    for line in reversed(str(stderr_text or "").splitlines()):
        if not line.strip() or line[:1].isspace():
            continue
        summary = line.strip()
        if len(summary) > _FAILURE_SUMMARY_MAX_CHARS:
            summary = summary[: _FAILURE_SUMMARY_MAX_CHARS - 3] + "..."
        return summary
    return ""


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
    # The cause goes on the FIRST line: that is the only line Auto Pilot's
    # activity feed keeps, and a bare "exited with code 1" sent the last
    # three desk failures to the run manifests to be identified at all.
    summary = child_failure_summary(stderr_text)
    raise RuntimeError(
        f"Master AVWAP scan process exited with code {returncode}."
        + (f" {summary}" if summary else "")
        + (f"\n\n{details}" if details else "")
    )

