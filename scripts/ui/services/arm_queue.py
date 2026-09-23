"""One ordered worker for the slow half of an arm click.

A click marks the arm QUEUED at once and returns. The worker runs each job's
``prepare`` (the child RPC for the M5 baseline, a report read) strictly in
submit order, one at a time, then hands the answer back to the Qt thread,
where ``commit`` saves the store and flips the arm to ARMED or FAILED.
``post`` jobs (review-event appends: a cross-process lock plus fsync) ride
the same FIFO so writes stay ordered.

The job keeps the symbol it was clicked on, so it completes even after the
chart has moved on. A job cancelled before its commit never commits.
Shutdown drains with a bounded join and logs every arm it could not finish.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal, Slot

QUEUED = "queued"
RUNNING = "arming"
ARMED = "armed"
FAILED = "failed"
CANCELLED = "cancelled"

#: How long shutdown waits for the queue to drain.
SHUTDOWN_TIMEOUT_S = 5.0


@dataclass(eq=False)
class ArmJob:
    key: tuple
    symbol: str
    label: str
    prepare: Callable[[], Any] | None
    commit: Callable[[Any], tuple[bool, str]]
    state: str = QUEUED
    reason: str = ""
    result: Any = None
    error: str = ""
    prepared: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def pending(self) -> bool:
        return self.state in (QUEUED, RUNNING)


class ArmQueue(QObject):
    """Single-threaded, FIFO. Owned by the Alert Center panel."""

    #: (ArmJob) - a job changed state; always delivered on the Qt thread.
    jobChanged = Signal(object)
    _prepared = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._queue: queue.Queue = queue.Queue()
        self._jobs: dict[tuple, ArmJob] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._prepared.connect(self._on_prepared)

    # -- Qt thread -----------------------------------------------------------
    def submit(self, key, symbol, label, prepare, commit, **extra) -> ArmJob:
        job = ArmJob(tuple(key), str(symbol), str(label), prepare, commit, extra=dict(extra))
        self._jobs[job.key] = job
        self._put(("arm", job))
        self.jobChanged.emit(job)
        return job

    def post(self, work: Callable[[], Any], label: str = "") -> None:
        """An ordered write with no commit step (worker only)."""
        self._put(("post", work, str(label)))

    def job(self, key) -> ArmJob | None:
        return self._jobs.get(tuple(key))

    def pending(self, key) -> ArmJob | None:
        job = self._jobs.get(tuple(key))
        return job if job is not None and job.pending else None

    def jobs(self) -> list[ArmJob]:
        """Queued, arming and failed jobs - what the UI shows."""
        return list(self._jobs.values())

    def cancel(self, key) -> bool:
        job = self.pending(key)
        if job is None:
            return False
        job.state = CANCELLED
        self._jobs.pop(job.key, None)
        self.jobChanged.emit(job)
        return True

    def dismiss(self, key) -> None:
        job = self._jobs.get(tuple(key))
        if job is not None and not job.pending:
            self._jobs.pop(job.key, None)
            self.jobChanged.emit(job)

    @Slot(object)
    def _on_prepared(self, job: ArmJob) -> None:
        if job.state == CANCELLED or job.state in (ARMED, FAILED):
            return
        if job.error:
            ok, reason = False, job.error
        else:
            try:
                ok, reason = job.commit(job.result)
            except Exception as exc:  # noqa: BLE001 - a commit failure is shown, never raised
                logging.exception("Arm commit failed for %s %s.", job.symbol, job.label)
                ok, reason = False, str(exc) or exc.__class__.__name__
        job.state = ARMED if ok else FAILED
        job.reason = str(reason or "")
        if ok:
            self._jobs.pop(job.key, None)
        self.jobChanged.emit(job)

    # -- the worker ----------------------------------------------------------
    def _put(self, item) -> None:
        if self._closed:
            if item[0] == "arm":
                item[1].error = "the desk is shutting down"
                self._on_prepared(item[1])
            else:
                logging.warning("Review-event write dropped at shutdown: %s", item[2])
            return
        self._queue.put(item)
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="alert-arm-queue", daemon=True
                )
                self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if item[0] == "post":
                try:
                    item[1]()
                except Exception:  # noqa: BLE001 - evidence loses the event, never the arm
                    logging.exception("Queued write failed: %s", item[2])
                continue
            job: ArmJob = item[1]
            if job.state == CANCELLED:
                continue
            job.state = RUNNING
            try:
                job.result = job.prepare() if job.prepare is not None else None
            except Exception as exc:  # noqa: BLE001
                job.error = str(exc) or exc.__class__.__name__
            job.prepared = True
            try:
                self._prepared.emit(job)
            except RuntimeError:
                return  # the owner is gone

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Test helper: True once the worker has drained what was queued."""
        done = threading.Event()
        self._put(("post", done.set, "idle marker"))
        return done.wait(timeout)

    def shutdown(self, timeout: float = SHUTDOWN_TIMEOUT_S) -> list[ArmJob]:
        """Drain with a bounded join; commit what finished; log what did not."""
        if self._closed:
            return []
        self._closed = True
        thread = self._thread
        self._queue.put(None)
        if thread is not None and thread.is_alive():
            thread.join(max(0.0, float(timeout)))
        dropped: list[ArmJob] = []
        for job in list(self._jobs.values()):
            if not job.pending:
                continue
            if job.prepared:
                self._on_prepared(job)  # its queued delivery may never run now
            else:
                dropped.append(job)
        for job in dropped:
            logging.error(
                "ARM DROPPED at shutdown: %s %s was still %s and was NOT armed.",
                job.symbol, job.label, job.state,
            )
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None and item[0] == "post":
                logging.error("Queued write dropped at shutdown: %s", item[2])
        return dropped
