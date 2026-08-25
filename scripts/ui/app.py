#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import argparse
import gc
import logging
import re
import sys
import time
from pathlib import Path

import threading
from datetime import datetime

from PySide6.QtCore import QEvent, QObject, QProcess, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from project_paths import get_shared_watchlist_details, get_tracker_storage_details
from technical_integrity import (
    format_technical_integrity_snapshot,
    load_technical_integrity_snapshot,
)
from ui.panels.ai_summary_panel import AiSummaryPanel
from ui.panels.autopilot_panel import AutopilotPanel
from ui.panels.bounce_panel import format_auto_regime_reading
from ui.panels.health_panel import HealthPanel
from ui.panels.journal_panel import JournalPanel
from ui.panels.away_recap_panel import AwayRecapPanel
from ui.panels.market_journal_panel import MarketJournalPanel
from ui.panels.weekend_prep_panel import WeekendPrepPanel
from ui.panels.research_panel import ResearchPanel
from ui.panels.settings_panel import SettingsPanel
from ui.panels.strength_board_panel import StrengthBoardPanel
from ui.panels.trading_desk import TradingDeskPanel
from ui.panels.universe_panel import UniversePanel
from ui import theme
from ui.services.strength_board_service import StrengthBoardService
from ui.state import VALID_UI_SCALES, UiState
from ui.theme import apply_theme
from ui.widgets.price_alert_toast import PriceAlertToastManager
from ui.widgets.technical_integrity_dialog import TechnicalIntegrityDialog


@dataclass(frozen=True)
class PageSpec:
    """One nav entry: its title, its icon, and the widget it shows.

    The desk used to keep these as three parallel structures - the order pages
    were added, a ``nav_items`` tuple, and a ``titles`` tuple inside
    ``_select_page`` - all addressed by the same integer index and none of them
    aware of the others. Adding the Strength Board updated two of the three, and
    the third kept ten entries against eleven pages. Every title from index 3
    onward named the wrong page, and clicking **Settings** - the last one -
    raised ``IndexError`` on a list that had run out.

    One list now. A page cannot be half-added.
    """

    title: str
    icon: str
    #: Dotted attribute path on the window, so a page owned by another panel
    #: (Focus Picks lives on the trading panel) needs no special case.
    attribute: str


PAGE_SPECS: tuple[PageSpec, ...] = (
    PageSpec("Trading Desk", "mdi.chart-timeline-variant", "trading_panel"),
    PageSpec("Chart Review", "mdi.chart-line", "chart_review_panel"),
    PageSpec("Focus Picks", "mdi.star-outline", "trading_panel.focus_picks_panel"),
    PageSpec("Strength Board", "mdi.trending-up", "strength_board_panel"),
    PageSpec("Journal", "mdi.notebook-outline", "journal_panel"),
    # R10.H. The label difference from "Journal" above is deliberate and
    # recorded: that one is the trade and tax record, this one is what the
    # trader thought. Merging them would turn the tax journal into a diary.
    PageSpec("Market Journal", "mdi.book-open-variant", "market_journal_panel"),
    # R1 amendment 2026-08-24: an AWAY day ends in a recap, not a queue. This
    # is the return surface that replaced 317 pending review items.
    PageSpec("AWAY Recap", "mdi.calendar-check-outline", "away_recap_panel"),
    PageSpec("Weekend Prep", "mdi.calendar-weekend", "weekend_prep_panel"),
    PageSpec("Universe", "mdi.earth", "universe_panel"),
    PageSpec("Research", "mdi.flask-outline", "research_panel"),
    PageSpec("Auto Pilot", "mdi.robot-outline", "autopilot_panel"),
    PageSpec("A.I. Summary", "mdi.brain", "ai_summary_panel"),
    PageSpec("System Health", "mdi.heart-pulse", "health_panel"),
    PageSpec("Settings", "mdi.cog-outline", "settings_panel"),
)


class MainWindow(QMainWindow):
    def __init__(self, state: UiState, *, satellite_desk: bool = False) -> None:
        super().__init__()
        self.state = state
        self._satellite_desk = satellite_desk
        self.price_alert_toasts = PriceAlertToastManager(self)
        self.setWindowTitle("TradingBotV3 Trading Desk")
        # Open at the desk's preferred size, but never larger than the screen
        # actually offers: a 1640x980 default on a 1680x954 laptop opened the
        # window taller than the workspace, so the status strip sat under the
        # Dock. The floor scales too, or the minimum alone would force the
        # same overflow back.
        available = _available_screen_size()
        self.resize(min(1640, available[0]), min(980, available[1]))
        self.setMinimumSize(
            min(theme.px(1180), available[0]), min(theme.px(760), available[1])
        )

        self.trading_panel = TradingDeskPanel(
            workspace_mode=self.state.workspace_mode,
            price_alert_engine_enabled=not satellite_desk,
            price_alert_read_only=satellite_desk,
        )
        self.journal_panel = JournalPanel()
        self.market_journal_panel = MarketJournalPanel()
        self.away_recap_panel = AwayRecapPanel(
            focus_service=self.trading_panel.focus_service,
            journal_service=self.market_journal_panel.service,
        )
        self.weekend_prep_panel = WeekendPrepPanel(
            focus_service=self.trading_panel.focus_service
        )
        self.universe_panel = UniversePanel()
        self.research_panel = ResearchPanel(
            self.trading_panel.price_alert_service,
            price_alert_read_only=satellite_desk,
        )
        self.autopilot_panel = AutopilotPanel(bounce_service=self.trading_panel.bounce_panel.service)
        # D1 level/event alerts -> the hourly Away phone push. The Alert Center
        # classifies (it owns the D1 routing rules); Auto Pilot aggregates and
        # decides whether the current mode may push at all.
        self.trading_panel.alert_center.d1EventRecorded.connect(
            self.autopilot_panel.service.record_d1_event
        )
        self.autopilot_panel.service.enabledChanged.connect(self._sync_scan_scheduler_owner)
        self._sync_scan_scheduler_owner(self.autopilot_panel.service.enabled)
        # Desk Link relay (docs/MULTI_MACHINE_DESK_PROPOSAL.md Tier 1). The
        # service always exists so the Settings page can toggle it live; it
        # only serves after Settings (or the saved setting) enables it, and a
        # failed start (port in use) degrades to a normal single-machine desk.
        from ui.services.desk_link_service import DeskLinkService, desk_link_enabled

        self.desk_link_service = DeskLinkService(self)
        self.trading_panel.alert_center.attach_desk_link(self.desk_link_service)
        self.desk_link_service.controlChanged.connect(self._on_desk_link_control_changed)
        self.desk_link_service.intentReceived.connect(self._on_desk_link_intent)
        # Tier 3 full relay: every live surface the bot feeds locally also
        # publishes to satellites. publish_stream no-ops with no satellite
        # connected, so a lone desk pays nothing.
        _bounce = self.trading_panel.bounce_panel.service
        _publish = self.desk_link_service.publish_stream
        _bounce.rrsSnapshotChanged.connect(lambda snap: _publish("rrs", snap))
        _bounce.statusChanged.connect(lambda status: _publish("status", status))
        _bounce.autoRegimeChanged.connect(lambda reading: _publish("auto_regime", reading))
        _board_signal = getattr(_bounce, "entryBoardChanged", None)
        if _board_signal is not None:
            _board_signal.connect(lambda board: _publish("entry_board", board))
        self.desk_link_service.set_live_chart_source(
            self.trading_panel.alert_center._current_bot,
            self.trading_panel.alert_center.desk_link_stream_symbols,
        )
        self.desk_link_service.set_auto_mode_source(
            lambda: self.autopilot_panel.service.auto_mode
        )
        self.desk_link_feed = None
        if satellite_desk:
            # Satellite desk (--satellite-desk): the FULL desk UI, fed by the
            # main's relay instead of TWS. Never serves, never scans on its
            # own, never re-relays - and deliberately does not auto-start the
            # local Desk Link server even if this machine has it enabled.
            import socket as _socket

            from ui.satellite import load_saved_connection
            from ui.services.desk_link_feed import DeskLinkFeedService

            host, port, link_token = load_saved_connection()
            self.desk_link_feed = DeskLinkFeedService(self)
            self.desk_link_feed.priceAlertReceived.connect(self._on_remote_price_alert)
            self.trading_panel.alert_center.attach_remote_feed(self.desk_link_feed)
            self.desk_link_feed.linkStatusChanged.connect(self._on_satellite_link_status)
            self.desk_link_feed.autoRegimeChanged.connect(self._set_auto_regime)
            # Unpaired is a normal state, not a launch blocker: the desk opens
            # and the trader pairs it in Settings -> Desk Link.
            self._satellite_paired = bool(host and link_token)
            if host and link_token:
                self.desk_link_feed.start(
                    host=host,
                    port=port,
                    token=link_token,
                    machine_name=_socket.gethostname() or "satellite-desk",
                )
        elif desk_link_enabled():
            self.desk_link_service.start()

        self.settings_panel = SettingsPanel(
            self.state,
            bounce_service=self.trading_panel.bounce_panel.service,
            desk_link_service=self.desk_link_service,
            desk_link_feed=self.desk_link_feed,
            desk_role="satellite" if self._satellite_desk else "main",
        )
        self.settings_panel.stateChanged.connect(self._apply_state_changes)
        self.settings_panel.deskRoleRestartRequested.connect(self._restart_for_desk_role)
        self.health_panel = HealthPanel()
        self.ai_summary_panel = AiSummaryPanel(bounce_service=self.trading_panel.bounce_panel.service)

        # Chart Review (plan.md 13d). Its annotation rail is analysis-only;
        # it receives the live bot solely for the shared in-memory M5 chart.
        from ui.panels.chart_review_panel import ChartReviewPanel

        # No focus_service here, deliberately: Chart Review is analysis-only.
        # Its captures must never add a symbol to Focus or any watchlist.
        # M5 strength board (packet R2 Part B). The service owns the data and
        # its single-flight refresh; the panel only shows it and routes adds
        # through the Part A adoption gate.
        self.strength_board_service = StrengthBoardService(self)
        self.strength_board_panel = StrengthBoardPanel(
            service=self.strength_board_service,
            focus_service=self.trading_panel.focus_service,
        )
        # Selecting a row on the strength board charts it in the desk's
        # existing snapshot popup - the same one the RS/RW and Industry
        # boards open, owned by the Alert Center, so the chart carries the
        # bot-backed series, the painted levels and the capture rail
        # without a second chart widget existing anywhere (R4 pattern).
        self.strength_board_panel.symbolActivated.connect(
            self.trading_panel.alert_center.show_board_symbol
        )
        # The page's RS/RW half reads the SAME rrsSnapshotChanged payload the
        # Alert Center's RS/RW tab reads (trader, 2026-08-21). A second
        # listener on one signal, not a second source: the bounce service
        # still owns and produces that data, and nothing on this page fetches.
        _bounce_service = getattr(
            getattr(self.trading_panel, "bounce_panel", None), "service", None
        )
        _rrs_signal = getattr(_bounce_service, "rrsSnapshotChanged", None)
        if _rrs_signal is not None:
            _rrs_signal.connect(self.strength_board_panel.update_rrs_snapshot)
        self.chart_review_panel = ChartReviewPanel(
            bot_provider=self.trading_panel.bounce_panel.service.current_bot
        )

        self.pages = QStackedWidget()
        for spec in PAGE_SPECS:
            self.pages.addWidget(self._page_widget(spec))

        self.title_label = QLabel("Trading Desk")
        self.title_label.setObjectName("TitleLabel")

        self.mode_group = QButtonGroup(self)
        self.workspace_button = _mode_button("Workspace")
        self.tabs_button = _mode_button("Tabs")
        self.mode_group.addButton(self.workspace_button)
        self.mode_group.addButton(self.tabs_button)
        self.workspace_button.clicked.connect(lambda: self._set_workspace_mode("workspace"))
        self.tabs_button.clicked.connect(lambda: self._set_workspace_mode("tabs"))

        self.nav_buttons: list[QPushButton] = []
        self._build_shell()
        self._build_status_bar()
        self._bind_shortcuts()
        self._sync_mode_buttons()

        self.trading_panel.statusChanged.connect(self._set_scan_status)
        self.trading_panel.rowsChanged.connect(self._set_setup_counts)
        self.trading_panel.connectionChanged.connect(self._set_ib_status)
        self.health_panel.statusChanged.connect(self._set_health_status)
        self.trading_panel.bounce_panel.service.technicalIntegrityChanged.connect(
            self._set_technical_integrity
        )
        self.trading_panel.bounce_panel.service.autoRegimeChanged.connect(self._set_auto_regime)
        # Price-level alert crossings land in the normal alert stream too, so
        # the Alert Center is the on-desk record of what buzzed the phone.
        self.trading_panel.price_alert_service.alertTriggered.connect(self._on_price_alert)
        self._set_auto_regime({})
        self._set_technical_integrity(load_technical_integrity_snapshot())

        # Self-heal a stale universe on launch AND on a recurring check (the
        # app often stays open across sessions, so launch-only healing left
        # the universe stale all day whenever the one launch attempt failed
        # or the close simply passed while running). The swing scans fold
        # universe_longs/shorts into every run, so a stale pool quietly
        # degrades manual scans too. yfinance-only, in a background thread -
        # IB and the UI are untouched; the rebuild lock dedupes callers.
        QTimer.singleShot(2500, self._self_heal_universe)
        self._universe_heal_timer = QTimer(self)
        self._universe_heal_timer.setInterval(30 * 60_000)
        self._universe_heal_timer.timeout.connect(self._self_heal_universe)
        self._universe_heal_timer.start()

        # Rebuild the review-preference scoreboard when it has gone stale
        # (review_learning.py: P(take|shown) per segment + taken-vs-passed
        # outcomes from the decision log). Background daemon thread; pure
        # local file reads, never touches IB or the UI.
        QTimer.singleShot(5000, self._refresh_review_learning)

    def _refresh_review_learning(self) -> None:
        def worker() -> None:
            try:
                from review_learning import refresh_review_learning_if_stale

                refresh_review_learning_if_stale()
            except Exception:
                pass  # the scoreboard is advisory; startup must never notice

        threading.Thread(target=worker, name="review-learning-refresh", daemon=True).start()

    def _build_shell(self) -> None:
        nav = QFrame()
        nav.setObjectName("NavRail")
        self.nav_rail = nav
        nav.setMinimumWidth(theme.px(178))
        nav.setMaximumWidth(theme.px(220))
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(*(theme.px(10),) * 4)
        nav_layout.setSpacing(theme.px(8))

        brand = QLabel("TradingBotV3")
        brand.setObjectName("SectionTitle")
        nav_layout.addWidget(brand)
        nav_layout.addSpacing(8)

        for index, spec in enumerate(PAGE_SPECS):
            label, icon_name = spec.title, spec.icon
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            icon = _nav_icon(icon_name)
            if icon is not None:
                button.setIcon(icon)
                button.setIconSize(QSize(theme.px(18), theme.px(18)))
            button.clicked.connect(lambda _checked=False, page=index: self._select_page(page))
            self.nav_buttons.append(button)
            nav_layout.addWidget(button)
        nav_layout.addStretch(1)
        self.nav_buttons[0].setChecked(True)

        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(
            theme.px(12), theme.px(10), theme.px(12), theme.px(10)
        )
        top_layout.addWidget(self.title_label)
        top_layout.addStretch(1)
        top_layout.addWidget(self.workspace_button)
        top_layout.addWidget(self.tabs_button)

        # Desk Link control banner (Tier 2): visible only while a satellite
        # holds the lease. It lives OUTSIDE the locked page stack so "Take
        # back control" stays clickable at all times (trader decision).
        self.desk_link_banner = QFrame()
        self.desk_link_banner.setObjectName("DeskLinkBanner")
        banner_layout = QHBoxLayout(self.desk_link_banner)
        banner_layout.setContentsMargins(12, 8, 12, 8)
        self.desk_link_banner_label = QLabel()
        self.desk_link_banner_label.setWordWrap(True)
        take_back_button = QPushButton("Take back control")
        take_back_button.clicked.connect(lambda: self.desk_link_service.take_back_control())
        banner_layout.addWidget(self.desk_link_banner_label, 1)
        banner_layout.addWidget(take_back_button)
        self.desk_link_banner.setVisible(False)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.desk_link_banner)
        right_layout.addWidget(top_bar)
        right_layout.addWidget(self.pages, 1)

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(theme.px(8), theme.px(8), theme.px(8), 0)
        central_layout.setSpacing(theme.px(8))
        central_layout.addWidget(nav)
        central_layout.addWidget(right, 1)
        self.setCentralWidget(central)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        # Persistent Auto Mode control (plan.md sec 15.2): visible and
        # clickable from every page - OFF -> AUTO-DESK -> AUTO-AWAY ->
        # AUTO-EVENING -> OFF.
        self.auto_mode_button = QPushButton()
        self.auto_mode_button.setObjectName("AutoModeButton")
        self.auto_mode_button.setToolTip(
            "Click to cycle Auto Mode: OFF -> AUTO-DESK -> AUTO-AWAY -> AUTO-EVENING -> OFF. "
            "Profiles change presentation only - never trading decisions. "
            "EVENING = arm the night before a sleep-in morning: picks stage silently, "
            "the morning briefing builds itself, and price alerts push to your phone."
        )
        self.auto_mode_button.clicked.connect(self._cycle_auto_mode)
        self.autopilot_panel.service.enabledChanged.connect(lambda *_: self._sync_auto_mode_button())
        self._sync_auto_mode_button()
        status.addWidget(self.auto_mode_button)
        self.ib_status = QLabel("IB/TWS: unknown")
        self.scan_status = QLabel("Scan: idle")
        self.setup_status = QLabel("Setups: 0")
        self.watchlist_status = QLabel(_watchlist_status_text())
        self.universe_status = QLabel(_universe_status_text())
        self.data_status = QLabel(_data_status_text())
        self.health_status = QLabel("Health: checking...")
        status.addWidget(self.ib_status)
        status.addWidget(self.scan_status, 1)
        status.addPermanentWidget(self.setup_status)
        self.market_regime_status = QLabel("Auto regime: n/a")
        status.addPermanentWidget(self.market_regime_status)
        self.technical_integrity_status = QPushButton("Technicals: building")
        self.technical_integrity_status.setObjectName("TechnicalIntegrityButton")
        self.technical_integrity_status.setFlat(True)
        self.technical_integrity_status.clicked.connect(self._show_technical_integrity_details)
        status.addPermanentWidget(self.technical_integrity_status)
        status.addPermanentWidget(self.watchlist_status)
        status.addPermanentWidget(self.universe_status)
        status.addPermanentWidget(self.data_status)
        status.addPermanentWidget(self.health_status)
        self.satellite_link_label = None
        if self._satellite_desk:
            self.setWindowTitle("TradingBotV3 Trading Desk — SATELLITE (fed by main)")
            self.satellite_link_label = QLabel(
                "LINK … connecting"
                if getattr(self, "_satellite_paired", False)
                else "LINK ✕ not paired — Settings ▸ Desk Link"
            )
            status.addPermanentWidget(self.satellite_link_label)

    def _set_auto_regime(self, reading) -> None:
        chip, tooltip = format_auto_regime_reading(reading)
        env_key = str((reading or {}).get("env_key") or "") if isinstance(reading, dict) else ""
        if env_key.startswith("bearish"):
            color = "#f85149"
        elif env_key.startswith("bullish"):
            color = "#3fb950"
        else:
            color = "#8b8fa3"
        self.market_regime_status.setText(chip)
        self.market_regime_status.setToolTip(tooltip)
        self.market_regime_status.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _set_technical_integrity(self, snapshot) -> None:
        self._technical_integrity_snapshot = snapshot if isinstance(snapshot, dict) else {}
        chip, tooltip, color = format_technical_integrity_snapshot(snapshot)
        self.technical_integrity_status.setText(chip)
        self.technical_integrity_status.setToolTip(f"{tooltip}\n\nClick to search the full hierarchy.")
        self.technical_integrity_status.setStyleSheet(
            f"QPushButton#TechnicalIntegrityButton {{ color: {color}; font-weight: 600; padding: 1px 5px; }}"
        )

    def _on_price_alert(self, payload: dict) -> None:
        self._present_price_alert(payload)
        if not self._satellite_desk:
            self.desk_link_service.publish_stream("price_alert", payload)
            # Also update the sticky snapshot after the trigger log is durable,
            # so reconnecting satellites see alerts fired during a Wi-Fi gap.
            self.desk_link_service.publish_state_snapshot()

    def _on_remote_price_alert(self, payload: dict) -> None:
        self._present_price_alert(payload, replayed=bool(payload.get("replayed")))

    def _present_price_alert(self, payload: dict, *, replayed: bool = False) -> None:
        message = str(payload.get("message") or "Price alert fired")
        try:
            from ui.models.bounce import BounceAlert

            self.trading_panel.bounce_panel.service.alertReceived.emit(
                BounceAlert.from_callback(f"PRICE ALERT: {message}", "red")
            )
        except Exception:
            pass  # the push already went out; the desk echo is best-effort
        self.price_alert_toasts.show_alert(payload, replayed=replayed)

    def _show_technical_integrity_details(self) -> None:
        TechnicalIntegrityDialog(
            getattr(self, "_technical_integrity_snapshot", {}),
            self,
        ).exec()

    def _cycle_auto_mode(self) -> None:
        service = self.autopilot_panel.service
        mode = service.auto_mode
        if mode == "OFF":
            self._set_auto_mode("DESK")
        elif mode == "DESK":
            self._set_auto_mode("AWAY")
        elif mode == "AWAY":
            self._set_auto_mode("EVENING")
        else:
            self._set_auto_mode("OFF")

    def _set_auto_mode(self, mode: str) -> None:
        """One entry point for every Auto mode change (button and Desk Link)."""
        service = self.autopilot_panel.service
        if mode == "OFF":
            service.set_enabled(False)
        else:
            service.set_profile(mode)
            service.set_enabled(True)
        self._sync_auto_mode_button()
        # Satellites mirror the mode from the state snapshot; push it now
        # instead of leaving them a snapshot interval behind.
        desk_link = getattr(self, "desk_link_service", None)
        if desk_link is not None:
            desk_link.publish_state_snapshot()

    def _sync_auto_mode_button(self) -> None:
        mode = self.autopilot_panel.service.auto_mode
        text = "Auto: OFF" if mode == "OFF" else f"Auto: {mode}"
        self.auto_mode_button.setText(text)
        color = {"OFF": "#8b8fa3", "DESK": "#3fb950", "AWAY": "#d29922", "EVENING": "#58a6ff"}.get(mode, "#8b8fa3")
        self.auto_mode_button.setStyleSheet(
            f"QPushButton#AutoModeButton {{ color: {color}; font-weight: 600; padding: 1px 10px; }}"
        )

    def _sync_scan_scheduler_owner(self, enabled: bool) -> None:
        owner = "Auto Pilot" if bool(enabled) else ""
        self.trading_panel.master_panel.set_external_scheduler_owner(owner)

    def _bind_shortcuts(self) -> None:
        run_action = QAction("Run Scan", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(self.trading_panel.master_panel.run_scan)
        self.addAction(run_action)

        focus_action = QAction("Focus Setup Filter", self)
        focus_action.setShortcut(QKeySequence("Ctrl+F"))
        focus_action.triggered.connect(lambda _checked=False: self.trading_panel.master_panel.search_input.setFocus())
        self.addAction(focus_action)

        expand_action = QAction("Expand Setups", self)
        expand_action.setShortcut(QKeySequence("F9"))
        expand_action.triggered.connect(self._toggle_setups_expanded)
        self.addAction(expand_action)

    def _toggle_setups_expanded(self) -> None:
        """F9: hand the setups table the whole desk, and give it back.

        Only meaningful on the Trading Desk page, so it selects that page
        first rather than silently doing nothing from elsewhere.
        """
        if self.pages.currentIndex() != 0:
            self._select_page(0)
        expanded = self.trading_panel.toggle_setups_expanded()
        self._set_scan_status(
            "setups expanded to full width (F9 to restore the chart column)"
            if expanded
            else "chart column restored"
        )

    def _page_widget(self, spec: "PageSpec"):
        """Resolve a spec's dotted attribute path to the widget it names."""
        target = self
        for part in spec.attribute.split("."):
            target = getattr(target, part)
        return target

    def _select_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.title_label.setText(PAGE_SPECS[index].title)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        mode_visible = index == 0
        self.workspace_button.setVisible(mode_visible)
        self.tabs_button.setVisible(mode_visible)

    def _set_workspace_mode(self, mode: str) -> None:
        self.state.workspace_mode = mode
        self.state.save()
        self.trading_panel.set_mode(mode)
        self.settings_panel.mode_input.blockSignals(True)
        self.settings_panel.mode_input.setCurrentText(mode)
        self.settings_panel.mode_input.blockSignals(False)
        self._sync_mode_buttons()

    def _sync_mode_buttons(self) -> None:
        self.workspace_button.setChecked(self.state.workspace_mode == "workspace")
        self.tabs_button.setChecked(self.state.workspace_mode == "tabs")

    def _apply_state_changes(self) -> None:
        app = QApplication.instance()
        if app is not None:
            apply_theme(
                app,
                self.state.theme_name,
                self.state.compact_density,
                theme.resolve_scale(self.state.ui_scale, _available_screen_size()),
            )
        self._apply_scaled_metrics()
        self.trading_panel.set_mode(self.state.workspace_mode)
        self._sync_mode_buttons()

    def _apply_scaled_metrics(self) -> None:
        """Re-apply the pixel budgets that live in Python, not the stylesheet.

        The stylesheet restyles itself on every apply_theme, but explicit
        minimum widths do not - and those are exactly what decides whether a
        column can shrink. Without this pass, moving the scale down restyled
        the text and left the layout jammed against its old floors.
        """
        available = _available_screen_size()
        self.setMinimumSize(
            min(theme.px(1180), available[0]), min(theme.px(760), available[1])
        )
        nav = getattr(self, "nav_rail", None)
        if nav is not None:
            nav.setMinimumWidth(theme.px(178))
            nav.setMaximumWidth(theme.px(220))
        for button in self.nav_buttons:
            if not button.icon().isNull():
                button.setIconSize(QSize(theme.px(18), theme.px(18)))
        self.trading_panel.apply_scaled_metrics()

    def _set_scan_status(self, message: str) -> None:
        self.scan_status.setText(f"Scan: {message}")

    def _set_ib_status(self, message: str) -> None:
        self.ib_status.setText(message if message.lower().startswith("ib") else f"IB/TWS: {message}")

    def _set_health_status(self, status: str) -> None:
        # UNKNOWN is a first-class status (plan.md sec 6.3): absent evidence
        # gets its own purple chip, distinct from measured-and-bad, and never
        # renders as green.
        normalized = str(status or "unknown").strip().lower()
        if normalized not in {"healthy", "degraded", "unhealthy", "unknown"}:
            normalized = "unknown"
        self.health_status.setText(f"Health: {normalized.upper()}")
        color = {
            "healthy": "#3fb950",
            "degraded": "#d29922",
            "unhealthy": "#f85149",
            "unknown": "#9b7cff",
        }.get(normalized, "#8b8fa3")
        self.health_status.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _set_setup_counts(self, total: int, favorites: int, near: int) -> None:
        self.setup_status.setText(f"Setups: {total} | Favorites: {favorites} | Near: {near}")

    def _self_heal_universe(self) -> None:
        import autopilot_core as core

        poll = getattr(self, "_universe_poll", None)
        if poll is not None and poll.isActive():
            return  # a heal attempt is already being tracked
        if not core.universe_is_stale(datetime.now()):
            self.universe_status.setText(_universe_status_text())
            return
        # Quiet hours (packet R1): this self-heal fired 2.5 s after launch and
        # every 30 minutes thereafter with no clock check at all, so booting the
        # desk at 21:00 sent a yfinance sweep of the whole universe down the
        # wire. The timer keeps ticking - the check is cheap - and the first
        # heal lands when the window opens. The Universe builder button is
        # manual and is deliberately not gated.
        allowed, _reason = core.auto_scanning_due(datetime.now())
        if not allowed:
            self.universe_status.setText(_universe_status_text())
            return
        self.universe_status.setText("Universe: stale - rebuilding...")
        self.universe_status.setStyleSheet("color: #E5C07B;")
        threading.Thread(
            target=core.rebuild_universe_if_stale,
            kwargs={"force": False},
            name="universe-self-heal",
            daemon=True,
        ).start()
        self._universe_poll_ticks = 0
        self._universe_poll = QTimer(self)
        self._universe_poll.setInterval(10_000)
        self._universe_poll.timeout.connect(self._poll_universe_heal)
        self._universe_poll.start()

    def _poll_universe_heal(self) -> None:
        import autopilot_core as core

        self._universe_poll_ticks += 1
        done = not core.universe_is_stale(datetime.now())
        if done or self._universe_poll_ticks > 120:  # give up after ~20 min
            self.universe_status.setText(_universe_status_text())
            self.universe_status.setStyleSheet("" if done else "color: #E06C75;")
            self._universe_poll.stop()

    def _on_satellite_link_status(self, link_state: str, detail: str) -> None:
        if getattr(self, "satellite_link_label", None) is None:
            return
        texts = {
            "connected": f"LINK ● {detail}",
            "connecting": "LINK … connecting",
            "disconnected": f"LINK ✕ {detail}",
            "rejected": "LINK ✕ token rejected - re-pair in Settings ▸ Desk Link",
            "stopped": "LINK ✕ not paired - Settings ▸ Desk Link",
        }
        self.satellite_link_label.setText(texts.get(link_state, f"LINK {link_state}"))

    def _restart_for_desk_role(self, _role: str) -> None:
        """Shut down current owners, then relaunch through the one entrypoint."""
        if getattr(self, "_desk_role_restart_pending", False):
            return
        self._desk_role_restart_pending = True
        app = QApplication.instance()
        if app is None:
            return
        app.aboutToQuit.connect(self._launch_replacement_desk)
        if self.close():
            QTimer.singleShot(0, app.quit)

    def _launch_replacement_desk(self) -> None:
        root = Path(__file__).resolve().parents[2]
        launcher = root / "launch_gui.py"
        ok, _pid = QProcess.startDetached(sys.executable, [str(launcher)], str(root))
        if not ok:
            logging.error(
                "Could not restart the Trading Desk automatically. Run %s manually; the saved "
                "desk role will apply.",
                launcher,
            )

    def _on_desk_link_control_changed(self, machine: str) -> None:
        """Satellite in control -> this desk is a relay: decision surfaces
        lock, engines keep running, and only 'Take back control' stays live."""
        controlled = bool(machine)
        self.desk_link_banner.setVisible(controlled)
        if controlled:
            self.desk_link_banner_label.setText(
                f"CONTROLLED BY {machine.upper()} — this desk is relaying. "
                "Alerts, scans, and TWS keep running here; decisions happen on the satellite."
            )
        self.pages.setEnabled(not controlled)
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.setEnabled(not controlled)

    def _on_desk_link_intent(self, machine: str, intent: dict) -> None:
        try:
            if str(intent.get("action") or "") == "set_auto_mode":
                ok, detail = self._apply_auto_mode_intent(machine, intent)
            else:
                ok, detail = self.trading_panel.alert_center.apply_desk_link_intent(machine, intent)
        except Exception:
            logging.exception("Desk Link intent application failed.")
            ok, detail = False, "intent application raised; see the main desk log"
        self.desk_link_service.send_intent_result(machine, intent.get("seq"), ok, detail)

    def _apply_auto_mode_intent(self, machine: str, intent: dict) -> tuple[bool, str]:
        """Apply a satellite's Auto mode change through the same path as the
        shell button. Idempotent, so at-least-once intent delivery is safe."""
        mode = str(intent.get("mode") or "").strip().upper()
        if mode not in ("OFF", "DESK", "AWAY", "EVENING"):
            return False, f"set_auto_mode needs mode OFF|DESK|AWAY|EVENING, got {mode!r}"
        self._set_auto_mode(mode)
        logging.info("Auto mode set to %s by Desk Link satellite %s.", mode, machine)
        return True, f"Auto mode -> {mode}"

    def closeEvent(self, event) -> None:
        for panel in (
            self.trading_panel,
            self.journal_panel,
            self.weekend_prep_panel,
            self.universe_panel,
            self.research_panel,
            self.autopilot_panel,
            self.ai_summary_panel,
            self.health_panel,
            self.settings_panel,
        ):
            try:
                panel.shutdown()
            except Exception:
                pass
        try:
            self.desk_link_service.stop()
        except Exception:
            pass
        if self.desk_link_feed is not None:
            try:
                self.desk_link_feed.stop()
            except Exception:
                pass
        # Backstop for the shared writer lease: AutopilotService.shutdown
        # normally releases it, but a panel that failed to shut down must not
        # leave the lease held. Releasing twice is a no-op, and a lease this
        # process instance does not own is never touched.
        try:
            import autopilot_core as core

            core.release_away_report_lease()
        except Exception:
            pass
        super().closeEvent(event)


def _mode_button(label: str) -> QPushButton:
    button = QPushButton(label)
    button.setCheckable(True)
    return button


def _available_screen_size() -> tuple[int, int]:
    """Usable logical size of the screen the desk will open on.

    availableGeometry, not geometry: it already excludes the macOS menu bar and
    Dock (and the Windows taskbar), which is the space the window actually has.
    Falls back to the 4K desk's workspace when no screen is reachable, e.g. an
    offscreen test run.
    """
    app = QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return (2560, 1440)
    size = screen.availableGeometry()
    return (max(640, size.width()), max(480, size.height()))


def _nav_icon(name: str) -> QIcon | None:
    """Material Design icon for the nav rail; degrade gracefully if qtawesome
    (or its font cache) is unavailable so the shell still launches."""
    try:
        import qtawesome as qta

        return qta.icon(name, color="#8A93A1")
    except Exception:
        return None


def _watchlist_status_text() -> str:
    details = get_shared_watchlist_details()
    longs = "yes" if details.get("longs_exists") == "yes" else "missing"
    shorts = "yes" if details.get("shorts_exists") == "yes" else "missing"
    return f"Watchlists: longs {longs} | shorts {shorts}"


def _data_status_text() -> str:
    details = get_tracker_storage_details()
    return f"Data: {details.get('source_label', details.get('source', 'unknown'))}"


def _universe_status_text() -> str:
    import autopilot_core as core

    built_at = core.universe_built_at()
    if built_at is None:
        return "Universe: missing"
    state = "stale" if core.universe_is_stale(datetime.now(), built_at) else "fresh"
    return f"Universe: {state} ({built_at:%b %d %H:%M})"


class UiActivityMonitor(QObject):
    """Monotonic timestamp of the latest user input delivered by Qt."""

    _INPUT_EVENTS = {
        QEvent.Type.KeyPress,
        QEvent.Type.MouseButtonPress,
        QEvent.Type.MouseButtonDblClick,
        QEvent.Type.Wheel,
        QEvent.Type.TouchBegin,
    }

    def __init__(self, parent=None, *, clock=time.perf_counter) -> None:
        super().__init__(parent)
        self._clock = clock
        self._last_input = float(clock())

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        if event.type() in self._INPUT_EVENTS:
            self._last_input = float(self._clock())
        return False

    def mark_input(self) -> None:
        """Testable/manual equivalent of receiving an input event."""

        self._last_input = float(self._clock())

    def idle_ms(self) -> float:
        return max(0.0, (float(self._clock()) - self._last_input) * 1000.0)


class _GuiGcController(QObject):
    """Keep Qt-wrapper collection on the GUI thread, but never over a click.

    Activity may DELAY a sweep. It may never CANCEL one.

    That distinction is the whole design, and getting it wrong cost a session:
    the first version of this controller returned early whenever input was
    more recent than ``young_idle_ms``, with no upper bound on the wait.
    Automatic collection is DISABLED process-wide (see install_gui_thread_gc),
    so this timer is the only collector there is - and a trader working the
    desk continuously produces input every few hundred milliseconds, which
    meant nothing was collected at all for as long as they kept working. On
    2026-08-21 the desk reached 8 GB in ninety minutes and then froze for
    **298 seconds** in the sweep that finally ran.

    So each preference now carries a deadline in ticks. Below it, idleness
    wins and the pause stays off the trader's clicks; at it, the sweep runs
    regardless, because a bounded pause now is strictly better than an
    unbounded heap and a five-minute pause later.
    """

    def __init__(
        self,
        activity: UiActivityMonitor,
        *,
        collector=gc.collect,
        full_every_ticks: int = 30,
        young_idle_ms: float = 250.0,
        full_idle_ms: float = 2_000.0,
        young_deadline_ticks: int = 5,
        full_deadline_ticks: int = 90,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.activity = activity
        self.collector = collector
        self.full_every_ticks = max(1, int(full_every_ticks))
        self.young_idle_ms = max(0.0, float(young_idle_ms))
        self.full_idle_ms = max(self.young_idle_ms, float(full_idle_ms))
        # At the production 2s tick: a young sweep waits at most 10 seconds for
        # quiet, and a due full sweep at most 3 minutes. The pre-repair code
        # ran them unconditionally every 2s and 60s, so the worst case here is
        # a small multiple of what shipped for months - not a new regime.
        self.young_deadline_ticks = max(0, int(young_deadline_ticks))
        self.full_deadline_ticks = max(0, int(full_deadline_ticks))
        self.tick = 0
        self.full_due = False
        self.full_due_at_tick = 0
        self.young_skipped = 0

    def sweep(self) -> None:
        self.tick += 1
        if self.tick % self.full_every_ticks == 0 and not self.full_due:
            self.full_due = True
            self.full_due_at_tick = self.tick
        idle_ms = self.activity.idle_ms()
        if self.full_due and (
            idle_ms >= self.full_idle_ms
            or self.tick - self.full_due_at_tick >= self.full_deadline_ticks
        ):
            self.collector(2)
            self.full_due = False
            self.young_skipped = 0
            return
        if idle_ms < self.young_idle_ms and self.young_skipped < self.young_deadline_ticks:
            self.young_skipped += 1
            return
        self.young_skipped = 0
        self.collector(0)


def install_gui_thread_gc(
    app: QApplication,
    interval_ms: int = 2_000,
    *,
    activity_monitor: UiActivityMonitor | None = None,
    collector=None,
    **controller_options,
) -> QTimer:
    """Run all cyclic garbage collection on the GUI thread.

    Every GUI session on 2026-07-29 died with an access violation inside
    python314.dll while ``Garbage-collecting`` on a worker thread
    (gui_crash.log, fault offset 0xc06b7). Automatic GC runs on whichever
    thread happens to allocate; when a collection on a scanner or wrap-up
    thread frees cycles that hold PySide6 wrappers, the QObject destructors
    run off the GUI thread — undefined behavior in Qt that corrupts the
    heap. Disabling automatic collection and sweeping from a main-thread
    timer keeps every Qt destructor on the owning thread. Reference
    counting still frees non-cyclic garbage immediately on any thread.

    Young objects are swept after 250 ms without input; a full heap sweep
    becomes due every 30th tick and waits for two seconds of user idleness.
    This retains GUI-thread ownership of Qt wrapper destruction without
    scheduling the largest pause directly on top of a click or wheel event.

    Both waits are BOUNDED (see _GuiGcController). Automatic collection is
    disabled here, so this timer is the process's only collector; an unbounded
    "wait for quiet" is indistinguishable from "never collect" while the desk
    is being used, which is exactly how it failed on 2026-08-21.
    """
    gc.disable()
    activity = activity_monitor or UiActivityMonitor(app)
    if activity_monitor is None:
        app.installEventFilter(activity)
    timer = QTimer(app)
    timer.setInterval(interval_ms)
    controller = _GuiGcController(
        activity,
        collector=collector if collector is not None else gc.collect,
        parent=timer,
        # Cadence/deadline knobs exist so a test can drive them deterministically;
        # production passes none of them and takes the documented defaults.
        **controller_options,
    )
    timer._gc_controller = controller  # type: ignore[attr-defined]
    timer.timeout.connect(controller.sweep)
    timer.start()
    return timer


#: Every distinct Qt message is printed ONCE and then counted. Keyed on the
#: message text with digits and hex stripped, so "row 41" and "row 42" of the
#: same complaint collapse together.
_qt_message_counts: dict[str, int] = {}
_qt_message_lock = threading.Lock()


def _qt_message_key(text: str) -> str:
    return re.sub(r"0x[0-9a-fA-F]+|\d+", "#", str(text or ""))[:200]


def install_qt_message_rate_limit() -> None:
    """Print each distinct Qt message once; count the repeats.

    Qt writes warnings straight to stderr from wherever they occur - including
    inside ``paint()``, on the GUI thread. On 2026-08-21 the desk flooded the
    console with ``QFont::setPointSizeF: Point size <= 0`` (one per visible row
    per repaint, see ui/widgets/setup_delegate.py), and every one of those lines
    was a synchronous console write competing with the frame it was drawing.

    The cause is fixed. This exists so the NEXT storm costs one line instead of
    thousands, and it deliberately does not silence anything: a message never
    seen before is always printed. Repeats are tallied and reported by
    :func:`report_qt_messages` at shutdown, so a flood is still visible - as a
    number rather than as noise.

    Installed before the QApplication, because Qt can complain during its own
    construction. Never raises: a diagnostic that can break a launch is worse
    than no diagnostic.
    """
    try:
        from PySide6.QtCore import qInstallMessageHandler
    except Exception:  # pragma: no cover - PySide6 is a hard dependency
        return

    def handler(mode, context, message) -> None:
        try:
            key = _qt_message_key(message)
            with _qt_message_lock:
                seen = _qt_message_counts.get(key, 0)
                _qt_message_counts[key] = seen + 1
            if seen == 0:
                sys.stderr.write(f"{message}\n")
        except Exception:
            return

    try:
        qInstallMessageHandler(handler)
    except Exception:
        logging.debug("Qt message handler not installed.", exc_info=True)


def report_qt_messages() -> list[tuple[int, str]]:
    """(count, message) for everything Qt said, busiest first."""
    with _qt_message_lock:
        items = sorted(
            ((count, text) for text, count in _qt_message_counts.items()), reverse=True
        )
    return items


def _print_qt_message_tally() -> None:
    """The one place a suppressed flood becomes visible again."""
    repeated = [(count, text) for count, text in report_qt_messages() if count > 1]
    if not repeated:
        return
    try:
        sys.stderr.write("Qt messages this session (first shown above, rest counted):\n")
        for count, text in repeated[:10]:
            sys.stderr.write(f"  {count:>8}x  {text}\n")
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the PySide6 TradingBotV3 UI.")
    parser.add_argument(
        "--mode",
        choices=("workspace", "tabs", "full", "simple"),
        default=None,
        help="Trading Desk layout mode. full maps to workspace; simple maps to tabs.",
    )
    parser.add_argument(
        "--theme",
        choices=("dark", "light"),
        default=None,
        help="GUI color theme. Saved as the default for future launches.",
    )
    parser.add_argument(
        "--satellite",
        nargs="?",
        const="",
        metavar="HOST[:PORT]",
        default=None,
        help=(
            "Launch as a view-only Desk Link satellite mirroring the main desk "
            "(docs/MULTI_MACHINE_DESK_PROPOSAL.md). HOST is optional: without it the "
            "window uses the saved connection or opens the connect dialog. No TWS, no scanners."
        ),
    )
    parser.add_argument(
        "--link-token",
        default=None,
        help="Desk Link token from the main machine. Saved locally after first use.",
    )
    parser.add_argument(
        "--satellite-desk",
        action="store_true",
        help=(
            "Compatibility alias for --desk-role satellite. Settings normally owns this choice."
        ),
    )
    parser.add_argument(
        "--desk-role",
        choices=("main", "satellite"),
        default=None,
        help=(
            "Compatibility override for the full desk role. The Settings page normally owns "
            "this choice and launch_gui.py remembers it."
        ),
    )
    parser.add_argument(
        "--ui-scale",
        choices=tuple(sorted(VALID_UI_SCALES)),
        default=None,
        help=(
            "Shell scale: auto sizes it from the screen (the 4K desk gets 1.00, "
            "a 1680px laptop 0.85). Saved as the default for future launches."
        ),
    )
    args = parser.parse_args(argv)

    state = UiState.load()
    if args.mode:
        state.workspace_mode = {"full": "workspace", "simple": "tabs"}.get(args.mode, args.mode)
        state.save()
    if args.theme:
        state.theme_name = args.theme
        state.save()
    if args.ui_scale:
        state.ui_scale = args.ui_scale
        state.save()

    install_qt_message_rate_limit()
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("TradingBotV3")
    app.setOrganizationName("TradingBotV3")
    app.ui_activity_monitor = UiActivityMonitor(app)
    app.installEventFilter(app.ui_activity_monitor)
    install_gui_thread_gc(app, activity_monitor=app.ui_activity_monitor)
    # Scale first: every widget built below reads theme.px() at construction.
    apply_theme(
        app,
        state.theme_name,
        state.compact_density,
        theme.resolve_scale(state.ui_scale, _available_screen_size()),
    )

    if args.satellite is not None:
        return _run_satellite(app, args.satellite, args.link_token)

    from ui.desk_role import ROLE_SATELLITE, startup_desk_role

    desk_role = startup_desk_role(
        explicit=args.desk_role,
        legacy_satellite=bool(args.satellite_desk),
    )
    satellite_desk = desk_role == ROLE_SATELLITE

    if satellite_desk and args.link_token:
        # Optional CLI convenience; pairing normally happens in the desk's own
        # Settings -> Desk Link -> "Connect to a main desk", so an unpaired
        # satellite desk still launches instead of blocking on a dialog.
        from ui.satellite import load_saved_connection, save_connection

        host, port, _ = load_saved_connection()
        if host:
            save_connection(host, port, args.link_token)

    window = MainWindow(state, satellite_desk=satellite_desk)
    window.show()
    # Off unless this machine asked for it. When on, every GUI-thread block
    # over the threshold is logged with the stack that caused it, which is
    # the only honest way to pick what to optimize next (Part C rule C1).
    from ui.stall_watchdog import install as install_stall_watchdog

    window.stall_watchdog = install_stall_watchdog(window)
    try:
        return app.exec()
    finally:
        _print_qt_message_tally()


def _run_satellite(app: QApplication, target: str, cli_token: str | None) -> int:
    """View-only satellite. CLI host/token are optional overrides — the
    window's connect dialog handles pairing and remembers everything."""
    import socket as _socket

    from desk_link.server import DEFAULT_PORT
    from ui.satellite import SatelliteWindow

    host, _, port_text = str(target or "").partition(":")
    host = host.strip()
    try:
        port = int(port_text) if port_text.strip() else DEFAULT_PORT
    except ValueError:
        port = DEFAULT_PORT

    window = SatelliteWindow(
        machine_name=_socket.gethostname() or "satellite",
        host=host,
        port=port,
        token=str(cli_token or "").strip(),
    )
    window.show()
    try:
        return app.exec()
    finally:
        _print_qt_message_tally()


if __name__ == "__main__":
    raise SystemExit(main())
