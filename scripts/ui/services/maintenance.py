from __future__ import annotations

import traceback
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot


class _WarmWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        try:
            from master_avwap import warm_durable_stores_for_watchlists

            summary = warm_durable_stores_for_watchlists()
            self.finished.emit(summary if isinstance(summary, dict) else {})
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class WarmingService(QObject):
    """Runs the one-shot durable-store warm (daily + H1 bars) off the GUI thread."""

    started = Signal()
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _WarmWorker | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def warm(self) -> None:
        if self.running:
            return
        thread = QThread(self)
        worker = _WarmWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.finished)
        worker.failed.connect(self.failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_thread)
        self._thread = thread
        self._worker = worker
        self.started.emit()
        thread.start()

    def shutdown(self) -> None:
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(3000)

    @Slot()
    def _clear_thread(self) -> None:
        self._thread = None
        self._worker = None


def format_warm_summary(summary: dict[str, Any]) -> str:
    requested = summary.get("requested", 0)
    daily = summary.get("daily", 0)
    intraday = summary.get("intraday", 0)
    failed = summary.get("failed") or []
    failed_count = len(failed) if isinstance(failed, list) else failed
    return f"Warmed {requested} symbol(s): daily={daily}, intraday={intraday}, failed={failed_count}."
