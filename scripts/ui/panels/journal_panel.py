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

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QTabWidget, QVBoxLayout

from ui.panels.journal.analytics_tab import AnalyticsTab
from ui.panels.journal.calendar_tab import CalendarTab
from ui.panels.journal.fees_tab import FeesTab
from ui.panels.journal.header import JournalHeader
from ui.panels.journal.health_tab import HealthTab
from ui.panels.journal.trades_tab import TradesTab
from ui.services import journal_feed


class JournalPanel(QFrame):
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        self.header = JournalHeader()
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
        layout.addWidget(self.header)
        layout.addWidget(self.tabs)

        self._reload_current()

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

    # -- the surface ui/app.py depends on ----------------------------------

    def refresh(self) -> None:
        self.header.refresh_accounts()
        self._reload_current()

    def rebuild_trades(self) -> None:
        try:
            count = journal_feed.rebuild_trades()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"rebuild failed: {exc}")
            return
        self._set_status(f"rebuilt {count} trades")
        self.refresh()

    def export_csv(self) -> None:
        try:
            path = journal_feed.export_trades_csv()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"export failed: {exc}")
            return
        self._set_status(f"exported {path}")

    def shutdown(self) -> None:
        for tab in self._tabs():
            if hasattr(tab, "shutdown"):
                tab.shutdown()
