#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import threading
from datetime import datetime

from PySide6.QtCore import QProcess, QSize, Qt, QTimer
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
from ui.panels.research_panel import ResearchPanel
from ui.panels.settings_panel import SettingsPanel
from ui.panels.trading_desk import TradingDeskPanel
from ui.panels.universe_panel import UniversePanel
from ui import theme
from ui.state import VALID_UI_SCALES, UiState
from ui.theme import apply_theme
from ui.widgets.technical_integrity_dialog import TechnicalIntegrityDialog


class MainWindow(QMainWindow):
    def __init__(self, state: UiState, *, satellite_desk: bool = False) -> None:
        super().__init__()
        self.state = state
        self._satellite_desk = satellite_desk
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

        self.trading_panel = TradingDeskPanel(workspace_mode=self.state.workspace_mode)
        self.journal_panel = JournalPanel()
        self.universe_panel = UniversePanel()
        self.research_panel = ResearchPanel()
        self.autopilot_panel = AutopilotPanel(bounce_service=self.trading_panel.bounce_panel.service)
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

        self.pages = QStackedWidget()
        self.pages.addWidget(self.trading_panel)
        self.pages.addWidget(self.trading_panel.focus_picks_panel)
        self.pages.addWidget(self.journal_panel)
        self.pages.addWidget(self.universe_panel)
        self.pages.addWidget(self.research_panel)
        self.pages.addWidget(self.autopilot_panel)
        self.pages.addWidget(self.ai_summary_panel)
        self.pages.addWidget(self.health_panel)
        self.pages.addWidget(self.settings_panel)

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
        self.research_panel.price_alerts_panel.service.triggered.connect(self._on_price_alert)
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

        nav_items = (
            ("Trading Desk", "mdi.chart-timeline-variant"),
            ("Focus Picks", "mdi.star-outline"),
            ("Journal", "mdi.notebook-outline"),
            ("Universe", "mdi.earth"),
            ("Research", "mdi.flask-outline"),
            ("Auto Pilot", "mdi.robot-outline"),
            ("A.I. Summary", "mdi.brain"),
            ("System Health", "mdi.heart-pulse"),
            ("Settings", "mdi.cog-outline"),
        )
        for index, (label, icon_name) in enumerate(nav_items):
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

    def _on_price_alert(self, message: str) -> None:
        try:
            from ui.models.bounce import BounceAlert

            self.trading_panel.bounce_panel.service.alertReceived.emit(
                BounceAlert.from_callback(f"PRICE ALERT: {message}", "red")
            )
        except Exception:
            pass  # the push already went out; the desk echo is best-effort

    def _show_technical_integrity_details(self) -> None:
        TechnicalIntegrityDialog(
            getattr(self, "_technical_integrity_snapshot", {}),
            self,
        ).exec()

    def _cycle_auto_mode(self) -> None:
        service = self.autopilot_panel.service
        mode = service.auto_mode
        if mode == "OFF":
            service.set_profile("DESK")
            service.set_enabled(True)
        elif mode == "DESK":
            service.set_profile("AWAY")
        elif mode == "AWAY":
            service.set_profile("EVENING")
        else:
            service.set_enabled(False)
        self._sync_auto_mode_button()

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
        run_action = QAction("Run Shared Scan", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(self.trading_panel.master_panel.run_shared_scan)
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

    def _select_page(self, index: int) -> None:
        titles = (
            "Trading Desk",
            "Focus Picks",
            "Journal",
            "Universe",
            "Research",
            "Auto Pilot",
            "A.I. Summary",
            "System Health",
            "Settings",
        )
        self.pages.setCurrentIndex(index)
        self.title_label.setText(titles[index])
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
            ok, detail = self.trading_panel.alert_center.apply_desk_link_intent(machine, intent)
        except Exception:
            logging.exception("Desk Link intent application failed.")
            ok, detail = False, "intent application raised; see the main desk log"
        self.desk_link_service.send_intent_result(machine, intent.get("seq"), ok, detail)

    def closeEvent(self, event) -> None:
        for panel in (
            self.trading_panel,
            self.journal_panel,
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


def install_gui_thread_gc(app: QApplication, interval_ms: int = 2_000) -> QTimer:
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

    Young objects are swept every tick; the full heap every 30th tick —
    a full collection of this app's pandas-heavy heap is too slow to run
    every 2 seconds without stuttering the UI.
    """
    gc.disable()
    timer = QTimer(app)
    timer.setInterval(interval_ms)
    tick = 0

    def _sweep() -> None:
        nonlocal tick
        tick += 1
        gc.collect(2 if tick % 30 == 0 else 0)

    timer.timeout.connect(_sweep)
    timer.start()
    return timer


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

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("TradingBotV3")
    app.setOrganizationName("TradingBotV3")
    install_gui_thread_gc(app)
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
    return app.exec()


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
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
