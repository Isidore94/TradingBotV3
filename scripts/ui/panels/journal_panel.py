"""The Journal tab: a shell over five sub-tabs (R7 §9 steps 11-13, spec §7).

This file used to be the whole Journal - one table, one detail pane, a broker
sync drawer that could not reach IBKR, and no way anywhere to tell the journal
it was wrong. The five tabs it now hosts live in ``ui/panels/journal/``; this
shell owns the shared header, routes to whichever tab is visible, and keeps the
public surface ``ui/app.py`` already depends on (``statusChanged``,
``rebuild_trades``, ``export_csv``, ``shutdown``).

Everything reads through ``ui.services.journal_feed``. No tab holds a
``JournalStore``, which is what makes the behaviour testable in
``tests/test_journal_feed.py`` without a Qt event loop anywhere near it.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QTabWidget, QVBoxLayout

from ui.panels.journal.analytics_tab import AnalyticsTab
from ui.panels.journal.calendar_tab import CalendarTab
from ui.panels.journal.fees_tab import FeesTab
from ui.panels.journal.header import JournalHeader
from ui.panels.journal.health_tab import HealthTab
from ui.panels.journal.trades_tab import TradesTab
from ui.services import journal_feed


class _JournalInitWorker(QThread):
    ready = Signal(dict)
    failed = Signal(str)

    def run(self) -> None:  # pragma: no cover - exercised through the real Qt seam
        try:
            self.ready.emit(journal_feed.initialize_store())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class JournalPanel(QFrame):
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        self._migration_worker: _JournalInitWorker | None = None
        self.migration_status = QLabel("")
        self.migration_status.setObjectName("JournalMigrationStatus")
        self.migration_status.setWordWrap(True)
        self.migration_status.setVisible(False)
        self.prepare_button = QPushButton("Prepare Journal database")
        self.prepare_button.clicked.connect(self._start_initialization)
        self.prepare_button.setVisible(False)

        self.header = JournalHeader(autoload=False)
        self.header.selectionChanged.connect(self._reload_current)

        self.trades_tab = TradesTab(self.header)
        self.calendar_tab = CalendarTab(self.header)
        self.analytics_tab = AnalyticsTab(self.header)
        self.health_tab = HealthTab(self.header)
        self.fees_tab = FeesTab(self.header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.trades_tab, "Trades")
        self.tabs.addTab(self.calendar_tab, "Calendar")
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.tabs.addTab(self.health_tab, "Health")
        self.tabs.addTab(self.fees_tab, "Fees")
        # Only the visible tab reloads. Analytics and Health are the expensive
        # ones, and rebuilding all five on every click of the account tree is
        # work nobody is looking at.
        self.tabs.currentChanged.connect(lambda _index: self._reload_current())

        for tab in self._tabs():
            tab.statusChanged.connect(self._set_status)
        self.calendar_tab.daySelected.connect(self._on_day_selected)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.migration_status)
        layout.addWidget(self.prepare_button)
        layout.addWidget(self.header)
        layout.addWidget(self.tabs)

        if not journal_feed.store_needs_preparation():
            self._finish_initialization({"migrated": False, "report": None})
        else:
            self.migration_status.setText(
                "Journal preparation is required. Review the migration dry-run first; then click "
                "Prepare Journal database to run backup, migration, and rebuild in the background."
            )
            self.migration_status.setVisible(True)
            self.prepare_button.setVisible(True)
            self.header.setEnabled(False)
            self.tabs.setEnabled(False)

    def _tabs(self):
        return (
            self.trades_tab,
            self.calendar_tab,
            self.analytics_tab,
            self.health_tab,
            self.fees_tab,
        )

    # -- routing -----------------------------------------------------------

    def _reload_current(self) -> None:
        current = self.tabs.currentWidget()
        if current is not None and hasattr(current, "reload"):
            current.reload()

    def _on_day_selected(self, day: str) -> None:
        """A day on the Calendar filters the Trades tab. That is why it is clickable."""
        from PySide6.QtCore import QDate

        self.header.range_input.setCurrentText("Custom")
        picked = QDate.fromString(day, "yyyy-MM-dd")
        if picked.isValid():
            self.header.date_from.setDate(picked)
            self.header.date_to.setDate(picked)
        self.tabs.setCurrentWidget(self.trades_tab)
        self.trades_tab.reload()

    def _set_status(self, message: str) -> None:
        self.statusChanged.emit(f"Journal: {message}")

    def _start_initialization(self) -> None:
        if self._migration_worker is not None and self._migration_worker.isRunning():
            return
        self.prepare_button.setEnabled(False)
        self.migration_status.setText(
            "Preparing Journal database… backup, migration, and rebuild are running in the background."
        )
        self._migration_worker = _JournalInitWorker(self)
        self._migration_worker.ready.connect(self._finish_initialization)
        self._migration_worker.failed.connect(self._initialization_failed)
        self._migration_worker.start()

    def _finish_initialization(self, result: dict) -> None:
        self.header.setEnabled(True)
        self.tabs.setEnabled(True)
        self.prepare_button.setVisible(False)
        self.header.refresh_accounts()
        self._reload_current()
        if result.get("migrated"):
            self.migration_status.setText("Journal migration completed. The audit report is available in Health.")
            self.migration_status.setVisible(True)
        else:
            self.migration_status.setVisible(False)

    def _initialization_failed(self, message: str) -> None:
        self.header.setEnabled(False)
        self.tabs.setEnabled(False)
        self.prepare_button.setEnabled(True)
        self.prepare_button.setVisible(True)
        self.migration_status.setText(f"Journal unavailable: migration failed — {message}")
        self.migration_status.setVisible(True)
        self._set_status(f"migration failed: {message}")

    def shutdown(self) -> None:
        worker = self._migration_worker
        if worker is not None and worker.isRunning():
            worker.wait()
        for tab in self._tabs():
            if hasattr(tab, "shutdown"):
                tab.shutdown()
