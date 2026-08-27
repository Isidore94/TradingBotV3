"""One owner for "read this off the Qt thread and hand back what it returned".

Extracted from `weekend_prep_panel` (G-P1.1) when a second page needed the same
shape (G-P1.5). Deliberately the smallest thing that works: it runs one
callable, emits its result or the string of its failure, and does nothing else.

It measures nothing, decides nothing and schedules nothing. What it also does
NOT do is touch the widget it will update, or blank anything - the caller
renders, and the rule the callers share is that a page which is refreshing goes
on showing what it already had. Clearing a populated page to announce a refresh
destroys the only copy of what it knew, and does so most damagingly in exactly
the case you would want it least: a refresh that then fails.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

#: How long a shutdown may wait for one reader. Bounded on purpose.
#: `_GuiGcController` learned this the hard way on 2026-08-21: a wait with no
#: upper bound is a hang waiting for a slow day. The warehouse readout waits on
#: the DAS, which is exactly the read that can take minutes when the share is
#: unwell, so an unbounded join there would hold the whole process open at exit.
SHUTDOWN_JOIN_MS = 5_000

#: Readers that outlived their deadline. Kept referenced ON PURPOSE: dropping
#: the last Python reference to a running QThread destroys its C++ half while
#: it runs, which is a crash rather than a leak. These are READS - no writes,
#: no side effects - so letting one finish into a void costs nothing, and the
#: process is on its way out anyway.
_abandoned: list["ReadWorker"] = []


class ReadWorker(QThread):
    """Runs one read function off the Qt thread. Never raises into the caller."""

    finished_with = Signal(object)
    failed = Signal(str)

    def __init__(self, work, parent=None) -> None:
        super().__init__(parent)
        self._work = work

    def run(self) -> None:  # pragma: no cover - exercised through its signals
        try:
            self.finished_with.emit(self._work())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


def join_worker(worker, *, timeout_ms: int = SHUTDOWN_JOIN_MS) -> bool:
    """Wait for one reader, but never forever. True if it finished in time.

    On timeout the worker is disowned and parked in `_abandoned` rather than
    left attached to a widget that is being destroyed. Shutdown continues: a
    desk that will not close is worse than a read that nobody collects.
    """
    if worker is None:
        return True
    try:
        if not worker.isRunning():
            return True
        if worker.wait(int(timeout_ms)):
            return True
        worker.setParent(None)
        _abandoned.append(worker)
        logging.warning(
            "A background read did not finish within %.1fs of shutdown; "
            "closing anyway and leaving it to end on its own.",
            timeout_ms / 1000.0,
        )
        return False
    except Exception:  # noqa: BLE001 - shutdown must not raise
        logging.debug("Joining a background read failed.", exc_info=True)
        return False
