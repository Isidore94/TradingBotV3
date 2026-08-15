"""The Journal tab's sub-tabs (R7 §9 steps 11-13, spec §7).

``JournalPanel`` is a shell over a ``QTabWidget``; each tab lives here. Every
one of them reads through ``ui.services.journal_feed`` and holds no
``JournalStore`` and no SQL, which is why the behaviour that matters is tested
in ``tests/test_journal_feed.py`` without a Qt event loop anywhere near it.
"""

from __future__ import annotations

__all__ = [
    "AnalyticsTab",
    "CalendarTab",
    "FeesTab",
    "HealthTab",
    "JournalHeader",
    "TradesTab",
]


def __getattr__(name: str):
    # Imported lazily so that a headless test importing this package does not
    # drag in PySide6, and so a missing optional charting dependency cannot stop
    # the whole Journal tab from loading.
    if name == "JournalHeader":
        from ui.panels.journal.header import JournalHeader

        return JournalHeader
    if name == "TradesTab":
        from ui.panels.journal.trades_tab import TradesTab

        return TradesTab
    if name == "CalendarTab":
        from ui.panels.journal.calendar_tab import CalendarTab

        return CalendarTab
    if name == "AnalyticsTab":
        from ui.panels.journal.analytics_tab import AnalyticsTab

        return AnalyticsTab
    if name == "HealthTab":
        from ui.panels.journal.health_tab import HealthTab

        return HealthTab
    if name == "FeesTab":
        from ui.panels.journal.fees_tab import FeesTab

        return FeesTab
    raise AttributeError(name)
