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

from PySide6.QtCore import QThread, Signal


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
