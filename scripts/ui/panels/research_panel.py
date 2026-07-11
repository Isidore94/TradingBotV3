from __future__ import annotations

from PySide6.QtWidgets import QFrame, QTabWidget, QVBoxLayout

from ui.panels.daytrade_tracker_panel import DaytradeTrackerPanel
from ui.panels.master_market_prep_panel import MasterMarketPrepPanel
from ui.panels.move_forensics_panel import MoveForensicsPanel
from ui.panels.setup_docs_panel import SetupDocsPanel
from ui.panels.setup_tracker_panel import SetupTrackerPanel
from ui.panels.ticker_lookup_panel import TickerLookupPanel
from ui.widgets.section_header import SectionHeader


class ResearchPanel(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        tabs = QTabWidget()
        self.market_prep_panel = MasterMarketPrepPanel()
        self.setup_tracker_panel = SetupTrackerPanel()
        self.setup_docs_panel = SetupDocsPanel()
        self.move_forensics_panel = MoveForensicsPanel()
        self.daytrade_tracker_panel = DaytradeTrackerPanel()
        self.ticker_lookup_panel = TickerLookupPanel()
        tabs.addTab(self.market_prep_panel, "Master AVWAP Market Prep")
        tabs.addTab(self.setup_tracker_panel, "Setup Tracker")
        tabs.addTab(self.setup_docs_panel, "Setup Playbook")
        tabs.addTab(self.move_forensics_panel, "Move Forensics")
        tabs.addTab(self.daytrade_tracker_panel, "Day Trade Tracker")
        tabs.addTab(self.ticker_lookup_panel, "Ticker Lookup")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(SectionHeader("Research", "Market prep, setup performance, and ticker lookup."))
        layout.addWidget(tabs, 1)

    def shutdown(self) -> None:
        self.ticker_lookup_panel.shutdown()
