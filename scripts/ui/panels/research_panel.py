from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.panels.daytrade_tracker_panel import DaytradeTrackerPanel
from ui.panels.long_lab_panel import LongLabPanel
from ui.panels.master_market_prep_panel import MasterMarketPrepPanel
from ui.panels.move_forensics_panel import MoveForensicsPanel
from ui.panels.price_alerts_panel import PriceAlertsPanel
from ui.panels.research_results_panel import ResearchResultsPanel
from ui.panels.retest_study_panel import RetestStudyPanel
from ui.panels.setup_docs_panel import SetupDocsPanel
from ui.panels.setup_keys_panel import SetupKeysPanel
from ui.panels.setup_tracker_panel import SetupTrackerPanel
from ui.panels.ticker_lookup_panel import TickerLookupPanel
from ui.panels.warehouse_readout_panel import WarehouseReadoutPanel
from ui.services.price_alert_service import PriceAlertService
from ui import theme
from ui.widgets.section_header import SectionHeader


LOCAL_NAV_DESTINATIONS = (
    ("Results", (("Results", "results_panel"),)),
    (
        "Setups",
        (
            ("Setup Tracker", "setup_tracker_panel"),
            ("Setup Playbook", "setup_docs_panel"),
            ("Setup keys", "setup_keys_panel"),
        ),
    ),
    (
        "Studies",
        (
            ("Move Forensics", "move_forensics_panel"),
            ("Day Trade Tracker", "daytrade_tracker_panel"),
            ("Long lab", "long_lab_panel"),
            ("Retest entry", "retest_study_panel"),
        ),
    ),
    (
        "Tools",
        (
            ("Master AVWAP Market Prep", "market_prep_panel"),
            ("Ticker Lookup", "ticker_lookup_panel"),
            ("Price Alerts", "price_alerts_panel"),
        ),
    ),
    ("Data", (("Research Warehouse", "warehouse_readout_panel"),)),
)


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
        # Packet G5: the Results page is FIRST and is the one the tab opens
        # on (decision 0016 answer 7 as amended 2026-09-06, and decision 3
        # of that day). The other eight keep the order they had.
        self.results_panel = ResearchResultsPanel()
        self.market_prep_panel = MasterMarketPrepPanel()
        self.setup_tracker_panel = SetupTrackerPanel()
        self.setup_docs_panel = SetupDocsPanel()
        self.move_forensics_panel = MoveForensicsPanel()
        self.daytrade_tracker_panel = DaytradeTrackerPanel()
        self.ticker_lookup_panel = TickerLookupPanel()
        self.warehouse_readout_panel = WarehouseReadoutPanel()
        self.setup_keys_panel = SetupKeysPanel()
        self.retest_study_panel = RetestStudyPanel()
        self.long_lab_panel = LongLabPanel()
        self.price_alerts_panel = PriceAlertsPanel(
            price_alert_service,
            read_only=price_alert_read_only,
        )
        tabs.addTab(self.results_panel, "Results")
        tabs.addTab(self.market_prep_panel, "Master AVWAP Market Prep")
        tabs.addTab(self.setup_tracker_panel, "Setup Tracker")
        tabs.addTab(self.setup_docs_panel, "Setup Playbook")
        tabs.addTab(self.move_forensics_panel, "Move Forensics")
        tabs.addTab(self.daytrade_tracker_panel, "Day Trade Tracker")
        tabs.addTab(self.ticker_lookup_panel, "Ticker Lookup")
        tabs.addTab(self.price_alerts_panel, "Price Alerts")
        tabs.addTab(self.warehouse_readout_panel, "Research Warehouse")
        tabs.addTab(self.setup_keys_panel, "Setup keys")
        # Shadow long-rule replay; placed before Retest entry, which stays last.
        tabs.addTab(self.long_lab_panel, "Long lab")
        # S8 (2026-09-26): the shadow retest-entry study, appended last.
        tabs.addTab(self.retest_study_panel, "Retest entry")
        # Held so the Working-lately strip's click-through can raise the one
        # tab it points at (ST6.4). A named reference rather than a walk over
        # `findChildren`, which would find the first QTabWidget on the page and
        # break the day a second one appears.
        self.tabs = tabs
        tabs.setCurrentIndex(0)
        tabs.tabBar().hide()

        self.local_nav_buttons: dict[str, QPushButton] = {}
        self.local_nav_groups: dict[str, tuple[QPushButton, ...]] = {}
        self.local_nav_destinations: dict[str, QWidget] = {}
        nav_group = QButtonGroup(self)
        nav_group.setExclusive(True)
        navigation = QFrame()
        navigation.setObjectName("NavRail")
        navigation.setProperty("researchNavigation", True)
        navigation.setMinimumWidth(theme.px(205))
        navigation.setMaximumWidth(theme.px(250))
        navigation_layout = QVBoxLayout(navigation)
        navigation_layout.setContentsMargins(
            theme.px(10), theme.px(12), theme.px(10), theme.px(12)
        )
        navigation_layout.setSpacing(theme.px(8))
        for group_title, destinations in LOCAL_NAV_DESTINATIONS:
            heading = QLabel(group_title)
            heading.setObjectName("MutedLabel")
            navigation_layout.addWidget(heading)
            buttons: list[QPushButton] = []
            for label, panel_name in destinations:
                destination = getattr(self, panel_name)
                button = QPushButton(label)
                button.setObjectName("NavButton")
                button.setCheckable(True)
                button.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
                )
                button.setMinimumHeight(theme.px(34))
                nav_group.addButton(button)
                button.clicked.connect(
                    lambda _checked=False, page=destination: tabs.setCurrentWidget(page)
                )
                buttons.append(button)
                self.local_nav_buttons[label] = button
                self.local_nav_destinations[label] = destination
                navigation_layout.addWidget(button)
            self.local_nav_groups[group_title] = tuple(buttons)
        navigation_layout.addStretch(1)
        tabs.currentChanged.connect(self._sync_local_navigation)
        self._sync_local_navigation(tabs.currentIndex())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        header = SectionHeader(
            "Research", "Results for the trader; studies are research."
        )
        header.title_label.setObjectName("TitleLabel")
        layout.addWidget(header)

        workspace = QHBoxLayout()
        workspace.setContentsMargins(0, 0, 0, 0)
        workspace.setSpacing(theme.px(12))
        workspace.addWidget(navigation, 0)
        workspace.addWidget(tabs, 1)
        layout.addLayout(workspace, 1)

    def _sync_local_navigation(self, index: int) -> None:
        current = self.tabs.widget(index)
        for label, button in self.local_nav_buttons.items():
            button.setChecked(self.local_nav_destinations[label] is current)

    def set_working_lately_snapshot(self, payload) -> None:
        """Forward the desk's ONE snapshot to the Results page (packet G5.2).

        `app.py` connects `WorkingLatelyService.snapshotChanged` here as well as
        to the Setup Tracker, so the Results page renders the reading the
        service just published instead of the one it last read off disk.
        """
        self.results_panel.set_working_lately_snapshot(payload)

    def show_setup_tracker(self) -> None:
        """Raise the Setup Tracker tab. The strip on the desk clicks through here."""
        index = self.tabs.indexOf(self.setup_tracker_panel)
        if index >= 0:
            self.tabs.setCurrentIndex(index)

    def shutdown(self) -> None:
        # Named children, so this list has to be kept complete by hand - and it
        # already fell behind once: the warehouse readout grew a worker in
        # G-P1.5 and was not here. Anything below that owns a thread belongs in
        # this list the same day it grows one.
        self.results_panel.shutdown()
        self.ticker_lookup_panel.shutdown()
        self.price_alerts_panel.shutdown()
        self.warehouse_readout_panel.shutdown()
        self.setup_keys_panel.shutdown()
        # R1: it grew a reader thread in P4 B1 and was not on this list - the
        # exact failure the comment above already describes, one panel later.
        self.setup_tracker_panel.shutdown()
        self.retest_study_panel.shutdown()
        self.long_lab_panel.shutdown()
