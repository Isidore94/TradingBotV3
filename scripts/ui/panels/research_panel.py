from __future__ import annotations

from PySide6.QtWidgets import QFrame, QTabWidget, QVBoxLayout

from ui.panels.master_market_prep_panel import MasterMarketPrepPanel
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
        self.ticker_lookup_panel = TickerLookupPanel()
        tabs.addTab(self.market_prep_panel, "Master AVWAP Market Prep")
        tabs.addTab(self.setup_tracker_panel, "Setup Tracker")
        tabs.addTab(self.ticker_lookup_panel, "Ticker Lookup")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(SectionHeader("Research", "Market prep, setup performance, and ticker lookup."))
        layout.addWidget(tabs, 1)

    def shutdown(self) -> None:
        self.ticker_lookup_panel.shutdown()
