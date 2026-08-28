from __future__ import annotations

from PySide6.QtWidgets import QFrame, QTabWidget, QVBoxLayout

from ui.panels.daytrade_tracker_panel import DaytradeTrackerPanel
from ui.panels.master_market_prep_panel import MasterMarketPrepPanel
from ui.panels.move_forensics_panel import MoveForensicsPanel
from ui.panels.price_alerts_panel import PriceAlertsPanel
from ui.panels.setup_docs_panel import SetupDocsPanel
from ui.panels.setup_tracker_panel import SetupTrackerPanel
from ui.panels.ticker_lookup_panel import TickerLookupPanel
from ui.panels.warehouse_readout_panel import WarehouseReadoutPanel
from ui.services.price_alert_service import PriceAlertService
from ui.widgets.section_header import SectionHeader


class ResearchPanel(QFrame):
    def __init__(
        self,
        price_alert_service: PriceAlertService | None = None,
        *,
        price_alert_read_only: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        tabs = QTabWidget()
        self.market_prep_panel = MasterMarketPrepPanel()
        self.setup_tracker_panel = SetupTrackerPanel()
        self.setup_docs_panel = SetupDocsPanel()
        self.move_forensics_panel = MoveForensicsPanel()
        self.daytrade_tracker_panel = DaytradeTrackerPanel()
        self.ticker_lookup_panel = TickerLookupPanel()
        self.warehouse_readout_panel = WarehouseReadoutPanel()
        self.price_alerts_panel = PriceAlertsPanel(
            price_alert_service,
            read_only=price_alert_read_only,
        )
        tabs.addTab(self.market_prep_panel, "Master AVWAP Market Prep")
        tabs.addTab(self.setup_tracker_panel, "Setup Tracker")
        tabs.addTab(self.setup_docs_panel, "Setup Playbook")
        tabs.addTab(self.move_forensics_panel, "Move Forensics")
        tabs.addTab(self.daytrade_tracker_panel, "Day Trade Tracker")
        tabs.addTab(self.ticker_lookup_panel, "Ticker Lookup")
        tabs.addTab(self.price_alerts_panel, "Price Alerts")
        tabs.addTab(self.warehouse_readout_panel, "Research Warehouse")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(SectionHeader("Research", "Market prep, setup performance, and ticker lookup."))
        layout.addWidget(tabs, 1)

    def shutdown(self) -> None:
        # Named children, so this list has to be kept complete by hand - and it
        # already fell behind once: the warehouse readout grew a worker in
        # G-P1.5 and was not here. Anything below that owns a thread belongs in
        # this list the same day it grows one.
        self.ticker_lookup_panel.shutdown()
        self.price_alerts_panel.shutdown()
        self.warehouse_readout_panel.shutdown()
