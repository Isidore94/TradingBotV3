#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import argparse
import gc
import logging
import re
import sys
import time

import threading
from datetime import datetime

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer
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

#: The one page that must be HANDED its input before it can say anything.
#: Matched by title rather than index so a reorder cannot silently unwire it -
#: which is the class of bug `test_qt_page_specs` exists for.
AWAY_RECAP_PAGE_TITLE = "AWAY Recap"


class MainWindow(QMainWindow):
    def __init__(self, state: UiState) -> None:
        super().__init__()
        self.state = state
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

        self.trading_panel = TradingDeskPanel(workspace_mode=self.state.workspace_mode)
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
        self.research_panel = ResearchPanel(self.trading_panel.price_alert_service)
        self.autopilot_panel = AutopilotPanel(bounce_service=self.trading_panel.bounce_panel.service)
        # D1 level/event alerts -> the hourly Away phone push. The Alert Center
        # classifies (it owns the D1 routing rules); Auto Pilot aggregates and
        # decides whether the current mode may push at all.
        self.trading_panel.alert_center.d1EventRecorded.connect(
            self.autopilot_panel.service.record_d1_event
        )
        self.autopilot_panel.service.enabledChanged.connect(self._sync_scan_scheduler_owner)
        self._sync_scan_scheduler_owner(self.autopilot_panel.service.enabled)
        # Every auto-mode flip becomes a Market Journal row with SPY's tape
        # attached, so the journal reads as ONE timeline - what the trader
        # thought and what the desk did, in order. Evidence only: nothing here
        # reaches a detector, score, alert, watchlist, Focus or the queue.
        self.autopilot_panel.service.autoModeChanged.connect(self._record_auto_mode_flip)
        self.settings_panel = SettingsPanel(
            self.state,
            bounce_service=self.trading_panel.bounce_panel.service,
        )
        self.settings_panel.stateChanged.connect(self._apply_state_changes)
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
        #
        # Since 2026-08-31 the board is not a page. The trader asked for it in
        # the Desk's Strength window ("either integrated directly or be
        # positioned below it"), so the Alert Center hosts it in a collapsible
        # section under `FocusStrengthBoard` and the nav entry is gone. The
        # SERVICE still lives here: one instance, one timer, one fetch, owned
        # by the window that shuts it down. Only the wiring moved.
        self.strength_board_service = StrengthBoardService(self)
        self.trading_panel.alert_center.attach_strength_board(
            self.strength_board_service,
            focus_service=self.trading_panel.focus_service,
        )
        # The AWAY Recap charts through the SAME popup, for the same reason: a
        # trader reading the day back needs the chart beside the alert, and a
        # second chart widget on that page would be a second definition of what
        # a symbol looks like.
        self.away_recap_panel.symbolActivated.connect(
            self.trading_panel.alert_center.show_board_symbol
        )
        # The page used to carry a second RS/RW view, so that the two reads
        # could be compared without flipping pages (trader, 2026-08-21). With
        # the board inside the Alert Center, the Alert Center's own RS/RW
        # Board tab is one tab-click away in the SAME column, so that second
        # listener retired with the page. The tape, its owner and the RS/RW
        # tab are untouched.
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

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
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
        """One entry point for every Auto mode change."""
        service = self.autopilot_panel.service
        if mode == "OFF":
            service.set_enabled(False)
        else:
            service.set_profile(mode)
            service.set_enabled(True)
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
        # Diagnostics only (P1 item 3): a stall sampled inside Qt's own event
        # dispatch names no application code, so the watchdog's record needs
        # the click to be legible at all. This decides nothing and defers
        # nothing - see `ui/interaction_trace.py`.
        from ui import interaction_trace

        interaction_trace.begin("page_select", PAGE_SPECS[index].title)
        try:
            self.pages.setCurrentIndex(index)
            interaction_trace.mark("model_apply")
            self.title_label.setText(PAGE_SPECS[index].title)
            for button_index, button in enumerate(self.nav_buttons):
                button.setChecked(button_index == index)
            mode_visible = index == 0
            self.workspace_button.setVisible(mode_visible)
            self.tabs_button.setVisible(mode_visible)
            interaction_trace.mark("layout")
            if PAGE_SPECS[index].title == AWAY_RECAP_PAGE_TITLE:
                self._feed_away_recap()
        finally:
            # Closed here rather than left open: a span that outlived its click
            # would attribute every later idle stall to the last page visited.
            interaction_trace.end()

    def _record_auto_mode_flip(self, previous: str, current: str) -> None:
        """Write the flip into the Market Journal, with SPY as it stood.

        The trader asked for "what the charts looked like when the auto mode
        flipped" (2026-08-27). The row is marked machine-written through its
        ORIGIN, so a reader counting "what did you think?" never counts a
        sentence nobody thought.

        Quiet on every failure path: an evidence store must never cost the
        thing it records, and the mode has already changed by the time this
        runs.
        """
        try:
            import market_journal
            import market_journal_capture
            from datetime import date

            service = self.market_journal_panel.service
            benchmark = market_journal_capture.BENCHMARK_SYMBOL
            text = (
                f"Auto mode {previous or 'UNKNOWN'} -> {current or 'UNKNOWN'}. "
                "Written by the desk, not the trader."
            )
            result = service.write_entry(
                text=text,
                session_date=date.today().isoformat(),
                timeframe=market_journal.TIMEFRAME_M5,
                symbols=[benchmark],
                origin=market_journal.ORIGIN_AUTO_MODE_FLIP,
            )
            entry_id = str((result.get("entry") or {}).get("entry_id") or "")
            if not result.get("ok") or not entry_id:
                return
            m5_bars, d1_bars = self.trading_panel.alert_center.journal_chart_bars(benchmark)
            service.capture_charts(
                entry_id=entry_id,
                symbol=benchmark,
                reason=market_journal_capture.REASON_MODE_FLIP,
                note=f"{previous} -> {current}",
                m5_bars=m5_bars,
                d1_bars=d1_bars,
            )
        except Exception:
            logging.exception("The auto-mode flip could not be journalled.")

    def _feed_away_recap(self) -> None:
        """Hand the recap the Alert Center's own backing list, then reload.

        Sol C1: the panel was constructed and `set_alerts` had no caller, so a
        full AWAY day ended in an empty recap while the backing list, History
        and every evidence stream were full. The alerts are handed in rather
        than read by the panel, because the Alert Center's list IS the record
        and a second reader would be a second definition of what the day
        produced (ground rule 8).

        The two backing lists are exported as ONE ordered stream here - the
        Alert Center keeps them apart because the D1 feed is untiered and the
        min-tier gate would silently swallow it, but a recap of the day is the
        day, so a D1 row travels flagged rather than merged away. Both lists are
        newest-first, so both are reversed: the order is the order the day
        happened, which is the only ordering nobody has to defend.

        **Known limitation, deliberately not papered over:** the Alert Center's
        backing list is process-scoped and capped, not session-scoped. A desk
        left running across midnight, or one restarted mid-session, hands the
        recap what the PROCESS saw rather than what the session produced.

        Failure is quiet on purpose - a recap that cannot be filled must never
        cost the page switch that asked for it.
        """
        try:
            from ui.panels.alert_center_panel import extract_alert_tier

            center = self.trading_panel.alert_center
            ordered = sorted(
                list(reversed(center._alerts)) + list(reversed(center._d1_alerts)),
                key=lambda alert: str(getattr(alert, "time_text", "") or ""),
            )
            self.away_recap_panel.set_alerts(
                [
                    {
                        "symbol": str(getattr(alert, "symbol", "") or ""),
                        "side": str(getattr(alert, "side", "") or ""),
                        # The tier the desk assigned, read through the Alert
                        # Center's own function. This page computes none.
                        "tier": extract_alert_tier(alert),
                        "trigger": str(getattr(alert, "trigger", "") or ""),
                        "time_text": str(getattr(alert, "time_text", "") or ""),
                        "is_d1": bool(getattr(alert, "is_d1", False)),
                    }
                    for alert in ordered
                ]
            )
        except Exception:
            logging.exception("The AWAY Recap could not be handed the day's alerts.")
            return
        try:
            self.away_recap_panel.reload()
        except Exception:
            logging.exception("The AWAY Recap could not be reloaded.")

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

    def closeEvent(self, event) -> None:
        for panel in (
            self.trading_panel,
            self.journal_panel,
            # Added 2026-08-27: this page now owns two workers and the shared
            # journal service's capture threads. It was absent from this list
            # while it owned one, which cost nothing then and would cost a
            # half-written capture now.
            self.market_journal_panel,
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
        # The strength board's service is owned by the window rather than by a
        # panel (its surface is a section inside the Alert Center), so it is
        # not in the loop above and needs stopping here. Its timer is the only
        # thing it holds.
        try:
            self.strength_board_service.shutdown()
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

    window = MainWindow(state)
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


if __name__ == "__main__":
    raise SystemExit(main())
