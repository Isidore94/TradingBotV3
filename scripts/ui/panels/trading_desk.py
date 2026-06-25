from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.models.setup import SetupRow
from ui.panels.bounce_panel import BouncePanel
from ui.panels.focus_picks_panel import FocusPicksPanel
from ui.panels.master_avwap_panel import MasterAvwapPanel
from ui.panels.theta_panel import ThetaPanel
from ui.panels.watchlists_panel import WatchlistsPanel
from ui.services.focus_service import FocusService
from ui.widgets.badge import Badge
from ui.widgets.empty_state import EmptyState
from ui.widgets.section_header import SectionHeader


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
        self.focus_picks_panel = FocusPicksPanel(self.focus_service)
        self.master_workspace = MasterAvwapWorkspace(
            self.master_panel,
            self.theta_panel,
            self.watchlists_panel,
        )
        self.bounce_panel = BouncePanel(self.focus_service)
        self._mode_widget: QWidget | None = None

        self.master_panel.statusChanged.connect(self.statusChanged)
        self.master_panel.rowsChanged.connect(self.rowsChanged)
        self.theta_panel.statusChanged.connect(self.statusChanged)
        self.watchlists_panel.statusChanged.connect(self.statusChanged)
        self.focus_picks_panel.statusChanged.connect(self.statusChanged)
        self.bounce_panel.statusChanged.connect(self.statusChanged)
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
            tabs = QTabWidget()
            tabs.addTab(self.master_workspace, "Master AVWAP")
            tabs.addTab(self.bounce_panel, "BounceBot")
            self._mode_widget = tabs
            self.center_layout.addWidget(tabs)
            return

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.master_workspace)
        splitter.addWidget(self.bounce_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([620, 340])
        self._mode_widget = splitter
        self.center_layout.addWidget(splitter)

    def shutdown(self) -> None:
        """Release live resources (IB connection, worker threads) on app close."""
        self.bounce_panel.on_close()
        self.master_panel.scan_service.shutdown()

    def _detach_mode_panels(self) -> None:
        if isinstance(self._mode_widget, QTabWidget):
            for panel in (self.master_workspace, self.bounce_panel):
                index = self._mode_widget.indexOf(panel)
                if index >= 0:
                    self._mode_widget.removeTab(index)

        for panel in (self.master_workspace, self.bounce_panel):
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
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.master_panel = master_panel
        self.theta_panel = theta_panel
        self.watchlists_panel = watchlists_panel
        self.tabs = QTabWidget()
        self.tabs.addTab(self.master_panel, "Setups")
        self.tabs.addTab(self.theta_panel, "Theta Plays")
        self.tabs.addTab(self.watchlists_panel, "Watchlists")
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


class TradingControlRail(QFrame):
    statusChanged = Signal(str)

    def __init__(
        self,
        master_panel: MasterAvwapPanel,
        master_workspace: MasterAvwapWorkspace,
        bounce_panel: BouncePanel,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setMinimumWidth(230)
        self.setMaximumWidth(320)
        self.master_panel = master_panel
        self.master_workspace = master_workspace
        self.bounce_panel = bounce_panel
        self._build_layout()

    def _build_layout(self) -> None:
        header = SectionHeader("Desk Controls", "Active trading actions and copy lists.")

        run_shared = QPushButton("Run Shared Scan")
        run_shared.setObjectName("PrimaryButton")
        run_shared.clicked.connect(self.master_panel.run_shared_scan)

        run_local = QPushButton("Run Local Scan")
        run_local.clicked.connect(self.master_panel.run_local_scan)

        refresh = QPushButton("Refresh Reports")
        refresh.clicked.connect(self.master_panel.refresh_from_reports)

        copy_label = QLabel("Copy Lists")
        copy_label.setObjectName("MutedLabel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(header)
        master_label = QLabel("Master AVWAP")
        master_label.setObjectName("MutedLabel")
        layout.addWidget(master_label)
        layout.addWidget(run_shared)
        layout.addWidget(run_local)
        layout.addWidget(refresh)
        layout.addSpacing(10)
        layout.addWidget(copy_label)
        for label, kind in (
            ("Longs", "longs"),
            ("Shorts", "shorts"),
            ("Favorites", "favorites"),
            ("Active", "active"),
            ("Ranked", "ranked"),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, copy_kind=kind: self.master_panel.copy_list(copy_kind))
            layout.addWidget(button)
        layout.addSpacing(10)

        view_label = QLabel("Views")
        view_label.setObjectName("MutedLabel")
        layout.addWidget(view_label)
        setup_view = QPushButton("Setups Table")
        setup_view.clicked.connect(self.master_workspace.show_setups)
        theta_view = QPushButton("Theta Plays")
        theta_view.clicked.connect(self.master_workspace.show_theta)
        watchlist_view = QPushButton("Watchlists")
        watchlist_view.clicked.connect(self.master_workspace.show_watchlists)
        for button in (setup_view, theta_view, watchlist_view):
            layout.addWidget(button)
        layout.addSpacing(10)

        bounce_label = QLabel("BounceBot")
        bounce_label.setObjectName("MutedLabel")
        layout.addWidget(bounce_label)
        bounce_connect = QPushButton("Connect BounceBot")
        bounce_connect.clicked.connect(self.bounce_panel.start)
        bounce_disconnect = QPushButton("Disconnect")
        bounce_disconnect.clicked.connect(self.bounce_panel.stop)
        bounce_start = QPushButton("Start Scanning")
        bounce_start.clicked.connect(self.bounce_panel.start_scanning)
        bounce_stop = QPushButton("Stop Scanning")
        bounce_stop.clicked.connect(self.bounce_panel.stop_scanning)
        for button in (bounce_connect, bounce_disconnect, bounce_start, bounce_stop):
            layout.addWidget(button)
        layout.addStretch(1)


class SetupDetailPanel(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("InfoDrawer")
        self.setMinimumWidth(320)
        self.setMaximumWidth(480)

        self.header = SectionHeader("Setup Detail", "Select a row to inspect the playbook context.")
        self.empty = EmptyState("No setup selected", "Click a setup row to see score notes, levels, theta, and earnings context.")
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self.header)
        layout.addWidget(scroll, 1)

        self.set_setup(None)

    def set_setup(self, setup: SetupRow | None) -> None:
        _clear_layout(self.content_layout)
        if setup is None:
            self.content_layout.addWidget(self.empty)
            return

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        symbol = QLabel(setup.symbol)
        symbol.setObjectName("TitleLabel")
        title_row.addWidget(symbol)
        if setup.side:
            title_row.addWidget(Badge(setup.side, "long" if setup.side == "LONG" else "short"))
        title_row.addStretch(1)
        self.content_layout.addLayout(title_row)
        self.content_layout.addWidget(Badge(setup.bucket_label, _bucket_tone(setup.bucket)))

        summary_fields = [
            ("Score", "" if setup.score is None else f"{setup.score:.1f}"),
            ("Expected R", setup.expected_r_text),
            ("Key level", setup.key_level),
            ("Supports", setup.supports_text),
            ("Theta", setup.theta),
            ("Days to earnings", "" if setup.days_to_earnings is None else str(setup.days_to_earnings)),
            ("Last daily bar", setup.last_trade_date),
        ]
        for label, value in summary_fields:
            if value:
                self.content_layout.addWidget(_detail_line(label, value))

        notes = _important_notes(setup.raw)
        if notes:
            notes_box = QTextEdit()
            notes_box.setReadOnly(True)
            notes_box.setPlainText("\n\n".join(notes))
            self.content_layout.addWidget(notes_box)
        self.content_layout.addStretch(1)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.setParent(None)
        if child_layout is not None:
            _clear_layout(child_layout)


def _detail_line(label: str, value: str) -> QWidget:
    frame = QFrame()
    frame.setObjectName("Panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(2)
    label_widget = QLabel(label)
    label_widget.setObjectName("MutedLabel")
    value_widget = QLabel(value)
    value_widget.setWordWrap(True)
    layout.addWidget(label_widget)
    layout.addWidget(value_widget)
    return frame


def _bucket_tone(bucket: str) -> str:
    normalized = bucket.strip().lower()
    if normalized in {"favorite_setup", "high_conviction"}:
        return "favorite"
    if normalized == "near_favorite_zone":
        return "near"
    if "study" in normalized:
        return "study"
    return "neutral"


def _important_notes(raw: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for label, key in (
        ("Expected R", "expected_r_note"),
        ("Ranking", "ranking_note"),
        ("HV Level", "hv_level_note"),
        ("HTF Trend", "htf_trend_note"),
        ("Retest", "retest_note"),
        ("Post earnings", "post_earnings_note"),
        ("Mid earnings", "mid_earnings_note"),
        ("SMA breakout", "sma_breakout_note"),
        ("Compression", "compression_note"),
        ("Pre-earnings block", "pre_earnings_setup_block_reason"),
        ("Score cap", "rejection_score_cap_note"),
    ):
        value = raw.get(key)
        if value:
            notes.append(f"{label}: {value}")

    rejection_reasons = raw.get("candidate_rejection_reasons")
    if isinstance(rejection_reasons, list) and rejection_reasons:
        notes.append("Warnings: " + "; ".join(str(item) for item in rejection_reasons[:6]))
    return notes
