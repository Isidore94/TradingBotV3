"""The background work queue: keeps the GPU busy between questions, never in the way of one.

One consumer thread, one job at a time, highest priority first (interactive > refresh >
embed). A job that needs the model waits while an interactive turn is in flight, while
the brain is down, and through the night window; a deterministic job (a pack rebuild)
still runs then. A generating job may ask for at most MAX_JOB_OUTPUT_TOKENS. Qt-free:
results come back through ``on_done`` on the consumer thread; the window re-emits them.
"""

from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

PRIORITY_INTERACTIVE = 0
PRIORITY_REFRESH = 1
PRIORITY_EMBED = 2
#: Runs only when nothing else waits (the rest of Focus after the liked picks).
PRIORITY_IDLE = 3
MAX_JOB_OUTPUT_TOKENS = 600
IDLE_WAIT_SECONDS = 1.0


@dataclass(order=True)
class Job:
    priority: int
    seq: int
    name: str = field(compare=False)
    run: Callable[[], Any] = field(compare=False)
    needs_model: bool = field(compare=False, default=False)
    max_tokens: int = field(compare=False, default=0)
    key: str = field(compare=False, default="")
    on_done: Callable[[Any], None] | None = field(compare=False, default=None)
    on_error: Callable[[BaseException], None] | None = field(compare=False, default=None)


class PrefetchQueue:
    def __init__(
        self,
        *,
        blocked: Callable[[], str] = lambda: "",
        model_ready: Callable[[], bool] = lambda: True,
        thread_name: str = "mentor-prefetch",
    ) -> None:
        self.thread_name = str(thread_name)
        self._blocked = blocked
        self._model_ready = model_ready
        self._jobs: list[Job] = []
        self._seq = itertools.count()
        self._lock = threading.Condition()
        self._interactive = 0
        #: True when the host's serve has one slot (found running, night-started).
        self.single_slot = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ran: list[str] = []

    # ------------------------------------------------------------------ producers
    def submit(
        self,
        name: str,
        run: Callable[[], Any],
        *,
        priority: int = PRIORITY_REFRESH,
        needs_model: bool = False,
        max_tokens: int = 0,
        key: str = "",
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> bool:
        """Queue a job; False when a job with the same key is already waiting."""
        if int(max_tokens) > MAX_JOB_OUTPUT_TOKENS:
            raise ValueError(f"a background job may ask for at most {MAX_JOB_OUTPUT_TOKENS} output tokens")
        with self._lock:
            if key and any(job.key == key for job in self._jobs):
                return False
            self._jobs.append(
                Job(int(priority), next(self._seq), str(name), run, bool(needs_model), int(max_tokens), key, on_done, on_error)
            )
            self._jobs.sort()
            self._lock.notify_all()
        return True

    def begin_interactive(self) -> None:
        with self._lock:
            self._interactive += 1

    def end_interactive(self) -> None:
        with self._lock:
            self._interactive = max(0, self._interactive - 1)
            self._lock.notify_all()

    def set_single_slot(self, single: bool) -> None:
        with self._lock:
            self.single_slot = bool(single)

    def should_yield(self) -> bool:
        """True when a running model job must stop at its next step: one slot and a chat turn waits."""
        with self._lock:
            return self.single_slot and self._interactive > 0

    def cancel(self, key: str) -> bool:
        """Drop a job that has not started; False when none with ``key`` is waiting."""
        with self._lock:
            for index, job in enumerate(self._jobs):
                if key and job.key == key:
                    del self._jobs[index]
                    return True
        return False

    def running(self) -> bool:
        """True while the consumer thread is alive."""
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def pending_keys(self) -> list[str]:
        with self._lock:
            return [job.key for job in self._jobs]

    def pending(self) -> list[str]:
        with self._lock:
            return [job.name for job in self._jobs]

    # ------------------------------------------------------------------ consumer
    def _may_run(self, job: Job) -> bool:
        if not job.needs_model:
            return True
        if self._blocked() or not self._model_ready():
            return False
        # A background model job never starts while a chat turn is in flight.
        return job.priority == PRIORITY_INTERACTIVE or self._interactive == 0

    def next_job(self) -> Job | None:
        """Pop the best job that may run now, or None."""
        with self._lock:
            for index, job in enumerate(self._jobs):
                if self._may_run(job):
                    return self._jobs.pop(index)
        return None

    def run_one(self) -> bool:
        job = self.next_job()
        if job is None:
            return False
        try:
            value = job.run()
        except Exception as exc:  # noqa: BLE001 - one failed job never stops the queue
            logging.warning("Trade Mentor prefetch job %s failed: %s", job.name, exc)
            if job.on_error:
                job.on_error(exc)
        else:
            if job.on_done:
                job.on_done(value)
        self.ran.append(job.name)
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.run_one():
                continue
            with self._lock:
                self._lock.wait(IDLE_WAIT_SECONDS)

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name=self.thread_name, daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
