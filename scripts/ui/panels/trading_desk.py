from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.panels.alert_center_panel import AlertCenterPanel
from ui.panels.bounce_panel import BouncePanel
from ui.panels.focus_picks_panel import FocusPicksPanel
from ui.panels.industry_panel import IndustryPanel
from ui.panels.master_avwap_panel import MasterAvwapPanel
from ui.panels.rs_window_panel import RsWindowPanel
from ui.panels.theta_panel import ThetaPanel
from ui.panels.watchlists_panel import WatchlistsPanel
from ui.services.focus_service import FocusService


class TradingDeskPanel(QWidget):
    statusChanged = Signal(str)
    rowsChanged = Signal(int, int, int)
    connectionChanged = Signal(str)

    def __init__(self, workspace_mode: str = "workspace", parent=None) -> None:
        super().__init__(parent)
        self.workspace_mode = workspace_mode
        self.focus_service = FocusService()
        self.master_panel = MasterAvwapPanel(self.focus_service)
        self.theta_panel = ThetaPanel()
        self.watchlists_panel = WatchlistsPanel()
        self.industry_panel = IndustryPanel()
        self.focus_picks_panel = FocusPicksPanel(self.focus_service)
        self.bounce_panel = BouncePanel(self.focus_service)
        self.rs_window_panel = RsWindowPanel(self.bounce_panel.service)
        self.master_workspace = MasterAvwapWorkspace(
            self.master_panel,
            self.theta_panel,
            self.watchlists_panel,
            self.industry_panel,
            rs_window_panel=self.rs_window_panel,
        )
        self.alert_center = AlertCenterPanel(self.focus_service)
        self.alert_center.attach_service(self.bounce_panel.service)
        # In workspace mode the Alert Center's embedded plan pane is off; a
        # clicked alert opens in the setups workspace's detail pane instead,
        # so the setup is described in exactly one place.
        self.alert_center.setupRequested.connect(self._show_setup_in_workspace)
        self._mode_widget: QWidget | None = None

        self.master_panel.statusChanged.connect(self.statusChanged)
        self.master_panel.rowsChanged.connect(self.rowsChanged)
        self.theta_panel.statusChanged.connect(self.statusChanged)
        self.watchlists_panel.statusChanged.connect(self.statusChanged)
        self.industry_panel.statusChanged.connect(self.statusChanged)
        self.rs_window_panel.statusChanged.connect(self.statusChanged)
        self.focus_picks_panel.statusChanged.connect(self.statusChanged)
        self.bounce_panel.statusChanged.connect(self.statusChanged)
        self.alert_center.statusChanged.connect(self.statusChanged)
        self.bounce_panel.service.connectionChanged.connect(self.connectionChanged)
        self.bounce_panel.service.alertReceived.connect(self.focus_picks_panel.record_bounce_alert)
        self.bounce_panel.service.rrsSnapshotChanged.connect(self.focus_picks_panel.record_rrs_snapshot)

        self.center_container = QWidget()
        self.center_layout = QVBoxLayout(self.center_container)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_layout.setSpacing(0)

        self._build_layout()
        self.set_mode(workspace_mode)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.center_container)

    def set_mode(self, workspace_mode: str) -> None:
        self.workspace_mode = workspace_mode if workspace_mode in {"workspace", "tabs"} else "workspace"
        self._detach_mode_panels()
        _clear_layout(self.center_layout)
        if self.workspace_mode == "tabs":
            self.alert_center.set_embedded_detail_enabled(True)
            tabs = QTabWidget()
            tabs.addTab(self.master_workspace, "Master AVWAP")
            tabs.addTab(self.alert_center, "Alert Center")
            tabs.addTab(self.bounce_panel, "BounceBot")
            self._mode_widget = tabs
            self.center_layout.addWidget(tabs)
            return

        # Workspace mode: the Alert Center owns the left column at full
        # height (the sit-back-and-wait surface); the right is the setups
        # workspace over the one-line BounceBot strip (fixed height - no
        # splitter, so it can't sprawl). Alert clicks show their plan in the
        # workspace's detail pane, not in a second embedded pane.
        self.alert_center.set_embedded_detail_enabled(False)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.master_workspace, 1)
        right_layout.addWidget(self.bounce_panel)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.alert_center)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([950, 1900])
        self._mode_widget = splitter
        self.center_layout.addWidget(splitter)

    def _show_setup_in_workspace(self, payload: dict) -> None:
        self.master_workspace.show_setups()
        self.master_panel.detail_view.show_setup(**payload)

    def shutdown(self) -> None:
        """Release live resources (IB connection, worker threads) on app close."""
        self.bounce_panel.on_close()
        self.master_panel.scan_service.shutdown()

    def _detach_mode_panels(self) -> None:
        if isinstance(self._mode_widget, QTabWidget):
            for panel in (self.master_workspace, self.alert_center, self.bounce_panel):
                index = self._mode_widget.indexOf(panel)
                if index >= 0:
                    self._mode_widget.removeTab(index)

        for panel in (self.master_workspace, self.alert_center, self.bounce_panel):
            try:
                panel.setParent(None)
            except RuntimeError:
                pass


class MasterAvwapWorkspace(QFrame):
    def __init__(
        self,
        master_panel: MasterAvwapPanel,
        theta_panel: ThetaPanel,
        watchlists_panel: WatchlistsPanel,
        industry_panel: IndustryPanel | None = None,
        rs_window_panel: RsWindowPanel | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.master_panel = master_panel
        self.theta_panel = theta_panel
        self.watchlists_panel = watchlists_panel
        self.industry_panel = industry_panel
        self.rs_window_panel = rs_window_panel
        self.tabs = QTabWidget()
        self.tabs.addTab(self.master_panel, "Setups")
        self.tabs.addTab(self.theta_panel, "Theta Plays")
        self.tabs.addTab(self.watchlists_panel, "Watchlists")
        if self.industry_panel is not None:
            self.tabs.addTab(self.industry_panel, "Industry Board")
        if self.rs_window_panel is not None:
            self.tabs.addTab(self.rs_window_panel, "RS Window")
        self.master_panel.scan_service.finished.connect(lambda *_args: self.theta_panel.refresh())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    def show_setups(self) -> None:
        self.tabs.setCurrentWidget(self.master_panel)

    def show_theta(self) -> None:
        self.theta_panel.refresh()
        self.tabs.setCurrentWidget(self.theta_panel)

    def show_watchlists(self) -> None:
        self.tabs.setCurrentWidget(self.watchlists_panel)

    def show_industry_board(self) -> None:
        if self.industry_panel is not None:
            self.industry_panel.reload_from_disk()
            self.tabs.setCurrentWidget(self.industry_panel)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.setParent(None)
        if child_layout is not None:
            _clear_layout(child_layout)


