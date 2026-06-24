from __future__ import annotations

import traceback
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot

from journal_importers import pacific_now
from journal_runner import run_journal_import_for_date
from ui.services.journal_import_helpers import recent_import_dates, summarize_import_results


class _JournalImportWorker(QObject):
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, days: int) -> None:
        super().__init__()
        self._days = days

    @Slot()
    def run(self) -> None:
        try:
            summaries: list[dict[str, Any]] = []
            for target_date in recent_import_dates(self._days, today=pacific_now().date()):
                summaries.append(
                    run_journal_import_for_date(
                        target_date,
                        trigger="qt_broker_sync",
                        include_questrade=True,
                        include_ibkr=False,
                    )
                )
            self.finished.emit(summaries)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class JournalImportService(QObject):
    started = Signal(int)
    finished = Signal(list, str)
    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _JournalImportWorker | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def pull_recent_questrade(self, days: int) -> bool:
        if self.running:
            return False

        thread = QThread(self)
        worker = _JournalImportWorker(days)
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
        self.started.emit(days)
        thread.start()
        return True

    def shutdown(self) -> None:
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(3000)

    @Slot(list)
    def _handle_finished(self, summaries: list[dict[str, Any]]) -> None:
        self.finished.emit(summaries, summarize_import_results(summaries))

    @Slot(str)
    def _handle_failed(self, message: str) -> None:
        self.failed.emit(message)

    @Slot()
    def _clear_thread(self) -> None:
        self._thread = None
        self._worker = None
