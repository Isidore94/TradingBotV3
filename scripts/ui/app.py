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
from datetime import date as _date, datetime

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
# TJ-1 item 3: `DayReviewPanel` takes the place of BOTH `MarketJournalPanel` and
# `DailyRecapPanel`. Those two modules stay on disk, unregistered and no longer
# constructed, until TJ-8 deletes them - the reviewer reproduces against them and
# `SetupTrackerPanel`'s next-test route is still tested through the recap class.
from ui.panels.day_review_panel import DayReviewPanel
from ui.panels.weekend_prep_panel import WeekendPrepPanel
from ui.panels.research_panel import ResearchPanel
from ui.panels.settings_panel import SettingsPanel
from ui.panels.trading_desk import TradingDeskPanel
from ui.panels.universe_panel import UniversePanel
from ui import theme
from ui.services.strength_board_service import StrengthBoardService
from ui.services.working_lately_service import WorkingLatelyService
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
    # WS-WL (WISHLIST 10G), 2026-09-13: **Chart Review and Focus Picks are no
    # longer nav pages.** Every action they had lives on the Trading Desk's
    # Watchlist tab (`ui/panels/watchlist_tab.py`) - add, paste, copy, clear,
    # remove, like, "Not today", refresh, Snapshot Today, the price-alert
    # save/remove/re-arm, the chart, `Ctrl+L` and a faded pick's Restore. Both
    # PANEL CLASSES stay: `ChartReviewPanel` is still constructed below (other
    # code imports it and its capture rail is the annotation surface's
    # reference implementation) and `FocusPicksPanel` is still built by the
    # desk, where BounceBot alerts and the RRS snapshot still reach it.
    PageSpec("Journal", "mdi.notebook-outline", "journal_panel"),
    # TJ-1 item 3 (trader, 2026-09-17; decision 0021 answer 1). ONE page where
    # Market Journal and Daily Recap used to be two: "daily recap and market
    # journal feel less than ideal … I feel like currently it's overcomplicated
    # … if you think it's better to combine daily recap and market journal
    # that's fine with me." It takes the Market Journal's SLOT rather than
    # joining the end, because a page appended after Settings is a different
    # page to reach for. The label difference from "Journal" above is still
    # deliberate and still recorded: that one is the trade and tax record.
    PageSpec("Day Review", "mdi.calendar-text", "day_review_panel"),
    PageSpec("Weekend Prep", "mdi.calendar-weekend", "weekend_prep_panel"),
    PageSpec("Universe", "mdi.earth", "universe_panel"),
    PageSpec("Research", "mdi.flask-outline", "research_panel"),
    PageSpec("Auto Pilot", "mdi.robot-outline", "autopilot_panel"),
    PageSpec("A.I. Summary", "mdi.brain", "ai_summary_panel"),
    PageSpec("System Health", "mdi.heart-pulse", "health_panel"),
    PageSpec("Settings", "mdi.cog-outline", "settings_panel"),
)

#: The day page, matched by TITLE rather than index so a reorder cannot silently
#: unwire it - the class of bug `test_qt_page_specs` exists for. Selecting it
#: refreshes two things: the page's own store read, and the AWAY digest panel
#: that is still handed the Alert Center's backing list.
#:
#: ONE constant, renamed from `DAILY_RECAP_PAGE_TITLE` by TJ-1 item 3 rather than
#: aliased: two constants for one live page is the drift `test_qt_page_specs`
#: exists for, and the old `AWAY_RECAP_PAGE_TITLE` alias had no caller left.
DAY_REVIEW_PAGE_TITLE = "Day Review"


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
        from ui.services.market_journal_service import shared_journal_service

        self.away_recap_panel = AwayRecapPanel(
            focus_service=self.trading_panel.focus_service,
            journal_service=shared_journal_service(),
        )
        # TJ-1 item 3. The Day Review page reads the durable stores on ONE
        # worker and is handed no feed, which is the whole point of it. The one
        # thing the desk hands it is the Alert Center's memory-only bar accessor,
        # so the SPY section can draw today's tape without fetching anything.
        # It goes to the PAGE, not to its service: that accessor mutates the
        # Alert Center's cache and arms a `QTimer.singleShot`, so it may only be
        # called on the Qt thread, and the page calls it in the slot that starts
        # each read (reviewer, 2026-09-17).
        self.day_review_panel = DayReviewPanel()
        try:
            self.day_review_panel.set_bars_reader(
                self.trading_panel.alert_center.journal_chart_bars
            )
        except Exception:  # noqa: BLE001 - no bars is a note on the page, never a failure
            logging.debug("The Day Review bars reader could not be wired.", exc_info=True)
        self.weekend_prep_panel = WeekendPrepPanel(
            focus_service=self.trading_panel.focus_service
        )
        # ST5.5: "this closed trade has no planned risk" -> the Journal's Trades
        # tab, on that trade. Weekend Prep never writes `planned_risk`; it
        # refers, and the trader types the plan where `save_risk_fields` lives.
        self.weekend_prep_panel.openTradeRequested.connect(self._open_journal_trade)
        self.weekend_prep_panel.openSessionRequested.connect(self._open_day_review_session)
        self.day_review_panel.openTradeRequested.connect(self._open_journal_trade)
        # WS-WL item 4: the Journal LINKS to the one Watchlist's Positions view.
        self.journal_panel.positionsOnWatchlistRequested.connect(
            self.show_watchlist_positions
        )
        # WS-WL item 2: the desk owns the Watchlist service (it owns the Focus
        # and price-alert stores it reads through); the window aliases it so
        # every shutdown path can find it by the name the packet gave it.
        self.watchlist_tab_service = self.trading_panel.watchlist_tab_service
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
        # Every auto-mode flip says so in the Auto Pilot log, and nowhere else
        # (TJ-1 item 1). It used to write a Market Journal row with SPY's tape
        # attached; the trader asked for that to stop. Nothing here reaches a
        # detector, score, alert, watchlist, Focus or the queue.
        self.autopilot_panel.service.autoModeChanged.connect(self._record_auto_mode_flip)
        # TJ-1 item 6(b): the staged-pick table lives on the Auto Pilot page now.
        # The ADD is still performed here, by the store's own owner.
        self.autopilot_panel.focusAddRequested.connect(self._add_staged_pick_to_focus)
        self.settings_panel = SettingsPanel(
            self.state,
            bounce_service=self.trading_panel.bounce_panel.service,
        )
        self.settings_panel.stateChanged.connect(self._apply_state_changes)
        self.health_panel = HealthPanel()
        self.ai_summary_panel = AiSummaryPanel(bounce_service=self.trading_panel.bounce_panel.service)
        self._opening_latest_day_review = False
        self.ai_summary_panel.dailyReviewRequested.connect(self.show_latest_completed_day_review)

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
        # positioned below it"), so the Alert Center hosts it at the foot of
        # its Strength page (one flat scrolling page since 2026-09-07) under
        # `FocusStrengthBoard`, and the nav entry is gone. The SERVICE still
        # lives here: one instance, one timer, one fetch, owned by the window
        # that shuts it down. Only the wiring moved.
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
        # WS-DR, TJ-1. The Day Review page uses the SAME named door, and it is
        # routed through a method rather than the bound slot so the call is
        # resolved when the row is clicked: `show_board_symbol` is a board's
        # door, and a board chart holds no place in the waiting list - nothing
        # here reaches `_enqueue_review_alert`.
        self.day_review_panel.chartRequested.connect(self._chart_recap_row)
        # Packet 3's compact Tracker route receives the exact display object the
        # worker already read. It comes from Research > Results since TJ-1 item
        # 6(a), which is where the measured report is printed. No Tracker read,
        # ranking or model work is added to this connection.
        self.research_panel.results_panel.entryQualityProposalChanged.connect(
            lambda payload: self.research_panel.setup_tracker_panel.set_entry_quality_proposal(
                payload, daily_recap=self.research_panel.results_panel
            )
        )
        # ST6.3. ONE Working-lately snapshot for the whole desk, owned by the
        # window because four surfaces read it and no one panel is their parent.
        # Everything expensive is on its worker; the slots below only format.
        # Four triggers, one coalesced reaction: the first show, the day roll,
        # a finished scan (which is what rewrites the tracker exports) and its
        # own thirty-minute timer.
        # WISHLIST 10J. The Trade Mentor's scheduler is owned by the WINDOW for
        # the same reason the Working-lately service is: it holds one timer and
        # one state file, and the surface it drives (the reusable Mentor popup)
        # is built more than once in this process's lifetime. The card is the
        # Alert Center's; the decision about when to show it is this one's.
        from ui.services.trade_mentor_context_service import TradeMentorContextService
        from ui.services.trade_mentor_service import TradeMentorService

        self.trade_mentor_context_service = TradeMentorContextService(
            self, cache_loader=self._trade_mentor_cached_bars
        )
        self.trading_panel.alert_center.chart_review.mentor_card.set_context_service(
            self.trade_mentor_context_service
        )
        self.trade_mentor_service = TradeMentorService(self)
        # TJ-9 item 6. The journal importer is built on first need, and the
        # date of the last morning retry lives here so the once-a-morning rule
        # holds across every card of one session.
        self._journal_importer = None
        self._journal_retry_date = ""
        self.trade_mentor_service.promptDue.connect(self._show_trade_mentor_prompt)
        self.trade_mentor_service.promptExpired.connect(
            lambda _slot_id: self.trading_panel.alert_center.chart_review.hide_mentor_card()
        )
        mentor_card = self.trading_panel.alert_center.chart_review.mentor_card
        mentor_card.answered.connect(self.trade_mentor_service.mark_answered)
        mentor_card.skipped.connect(
            lambda record: self.trade_mentor_service.mark_skipped(
                str(record.get("slot_id") or ""), str(record.get("skipped_reason") or "")
            )
        )
        self.settings_panel.mentorPauseRequested.connect(self._pause_trade_mentor)

        self.working_lately_service = WorkingLatelyService(self)
        self.working_lately_service.snapshotChanged.connect(
            self.trading_panel.set_working_lately_snapshot
        )
        self.working_lately_service.snapshotChanged.connect(
            self.research_panel.setup_tracker_panel.set_working_lately_snapshot
        )
        # G5: Research > Results renders the SAME reading (decision 0016 answer
        # 7 as amended - "both read ONE evidence snapshot").
        self.working_lately_service.snapshotChanged.connect(
            self.research_panel.set_working_lately_snapshot
        )
        self.working_lately_service.snapshotChanged.connect(
            self.weekend_prep_panel.set_working_lately_snapshot
        )
        self.working_lately_service.statusChanged.connect(self._set_scan_status)
        self.trading_panel.workingLatelyOpenRequested.connect(
            self._show_setup_tracker_page
        )
        self.trading_panel.alert_center.m5AlertsDayRolled.connect(
            self.working_lately_service.on_day_roll
        )
        # Trigger (c), BOTH halves. The manual scan service fires only for a
        # scan the trader started; the CLOSE-SLOT write - which is the one that
        # produces the exports on a normal day, on a desk nobody is touching -
        # comes from Auto Pilot's own scan service (re-review advisory 5). Both
        # route through the same coalescer, so a manual scan that happens to
        # land on the slot is still one build.
        self.trading_panel.master_panel.scan_service.finished.connect(
            lambda *_args: self.working_lately_service.on_tracker_export()
        )
        self.autopilot_panel.service.setupTrackerWritten.connect(
            lambda *_args: self.working_lately_service.on_tracker_export()
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
        self.apply_unused_surface_visibility()
        #: V2 item 1's badge is started from `showEvent`, not here. See it.
        self._tag_badge_started = False

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

    #: The left-nav page the trader never opens (decision 0016 answer 7). The
    #: universe BUILDER is load-bearing - the scanner reads what it writes - so
    #: the page is hidden and the builder is untouched.
    UNUSED_PAGE_TITLES = ("Universe",)

    #: One machine-local setting, default OFF. Not a per-surface list: the
    #: trader asked for these four to go away together, and four switches would
    #: be four things to find.
    SHOW_UNUSED_SETTING = "qt_show_unused_tabs"

    def show_unused_surfaces(self) -> bool:
        """Whether the unused tabs and pages are shown. Default OFF.

        Fails to SHOWING on an unreadable settings file: a surface the trader
        cannot reach is worse than one they have to skip past, and this is the
        direction that cannot lose them anything.
        """
        try:
            from project_paths import get_local_setting

            return bool(get_local_setting(self.SHOW_UNUSED_SETTING, False))
        except Exception:  # noqa: BLE001
            return True

    def apply_unused_surface_visibility(self) -> None:
        """Hide the left-nav pages and the Alert Center tabs the trader skips.

        **HIDDEN IS NOT REMOVED.** Every page is still built, still in
        `self.pages`, still at the same index - `_select_page` and every stored
        index keep working - and every timer behind a hidden page stays
        visibility-gated exactly as snappiness packet 3 left it. What changes is
        one nav button's visibility.
        """
        show = self.show_unused_surfaces()
        for index, spec in enumerate(PAGE_SPECS):
            if spec.title in self.UNUSED_PAGE_TITLES and index < len(self.nav_buttons):
                self.nav_buttons[index].setVisible(show)
        try:
            self.trading_panel.alert_center.apply_unused_tab_visibility(show)
        except Exception:  # noqa: BLE001 - a hidden tab is never worth a broken desk
            pass

    def _start_tag_review_badge(self) -> None:
        """Count the trades awaiting tag review, off-thread, once at startup.

        Decision 0016 answer 10 makes the nightly tagger the thing that saves the
        trader time, and a review queue nobody can see saves none: the badge is
        how the Trades tab's Provisional filter gets asked for.

        Everything here fails to a PLAIN LABEL. A badge is a convenience, and a
        number nobody can compute is not worth a broken nav bar.
        """
        try:
            from ui.read_worker import ReadWorker
        except Exception:  # noqa: BLE001
            return

        def _count():
            from ai_jobs.journal_auto_tag import trades_awaiting_review

            return trades_awaiting_review()

        try:
            worker = ReadWorker(_count, self)
            worker.finished_with.connect(self._apply_tag_review_badge)
            self._tag_badge_worker = worker
            worker.start()
        except Exception:  # noqa: BLE001
            return

    def _join_tag_review_badge(self) -> None:
        """Wait for the badge reader, but never forever. Called from shutdown."""
        worker = getattr(self, "_tag_badge_worker", None)
        if worker is None:
            return
        try:
            from ui.read_worker import join_worker

            join_worker(worker)
        except Exception:  # noqa: BLE001
            pass

    def _apply_tag_review_badge(self, payload: object) -> None:
        """"Journal (12 to review)". Zero leaves the label exactly as it was."""
        try:
            count = int(payload)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        for index, spec in enumerate(PAGE_SPECS):
            if spec.title != "Journal":
                continue
            if index < len(self.nav_buttons):
                self.nav_buttons[index].setText(
                    f"Journal ({count} to review)" if count > 0 else "Journal"
                )
            return

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
            if (
                PAGE_SPECS[index].title == DAY_REVIEW_PAGE_TITLE
                and not self._opening_latest_day_review
            ):
                self._feed_away_recap()
                self._reload_day_review()
        finally:
            # Closed here rather than left open: a span that outlived its click
            # would attribute every later idle stall to the last page visited.
            interaction_trace.end()

    def _select_page_by_title(self, title: str) -> bool:
        """Select a page by its TITLE, never by an index a reorder can move."""
        for index, spec in enumerate(PAGE_SPECS):
            if spec.title == title:
                self._select_page(index)
                return True
        return False

    def show_latest_completed_day_review(self) -> bool:
        """Open the one Day Review page on its newest completed session.

        Selecting the page normally reloads its remembered date.  This route
        briefly holds that reload so the panel chooses its calendar-owned date
        first, then starts the existing single worker for that one session.
        """
        self._opening_latest_day_review = True
        try:
            if not self._select_page_by_title(DAY_REVIEW_PAGE_TITLE):
                return False
            return bool(self.day_review_panel.show_latest_completed_session())
        finally:
            self._opening_latest_day_review = False

    def _open_day_review_session(self, session: str) -> None:
        """Open the exact Week Review day through the existing one-page reader."""
        self._opening_latest_day_review = True
        try:
            if self._select_page_by_title(DAY_REVIEW_PAGE_TITLE):
                self.day_review_panel.show_session(str(session or "")[:10])
        finally:
            self._opening_latest_day_review = False

    def show_watchlist_positions(self) -> bool:
        """The Journal's "Positions on the Watchlist" (WS-WL item 4).

        A nav call: the Trading Desk page, then its Watchlist tab on the
        Positions view. No second list is built and nothing is read here.
        """
        if not self._select_page_by_title("Trading Desk"):
            return False
        import watchlist_views

        return bool(
            self.trading_panel.show_watchlist(watchlist_views.VIEW_POSITIONS)
        )
    def _chart_recap_row(self, symbol: str, side: str = "") -> None:
        """A Daily Recap row -> the board chart door, resolved at click time.

        Never `_enqueue_review_alert`: a board chart takes no place in the
        waiting list and is never skip-counted (CLAUDE.md, "Charts and boards").
        """
        try:
            self.trading_panel.alert_center.show_board_symbol(symbol, side)
        except Exception:  # noqa: BLE001 - a chart request never costs the page
            logging.exception("The Daily Recap could not chart %s.", symbol)

    def _add_staged_pick_to_focus(self, symbol: str, side: str) -> None:
        """The staged-pick add, performed by the store's own owner."""
        try:
            self.trading_panel.focus_service.add(symbol, side, "swing")
        except Exception:  # noqa: BLE001
            logging.exception("The staged pick %s could not be added to Focus.", symbol)

    def _reload_day_review(self) -> None:
        """Kick the Day Review page's worker. Quiet on failure: a day that
        cannot be read must never cost the page switch that asked for it."""
        try:
            self.day_review_panel.reload()
        except Exception:
            logging.exception("The Day Review page could not be reloaded.")

    def _open_journal_trade(self, trade_id: str) -> None:
        """Show the Journal page on one trade (ST5.5). Never a writer.

        A referral that cannot find its trade says so in the status line rather
        than leaving the trader on a Journal page wondering which row was meant.
        """
        for index, spec in enumerate(PAGE_SPECS):
            if spec.attribute == "journal_panel":
                self._select_page(index)
                break
        try:
            found = self.journal_panel.show_trade(str(trade_id or ""))
        except Exception as exc:  # noqa: BLE001 - a referral never costs the desk
            self._set_scan_status(f"could not open that trade: {exc}")
            return
        if not found:
            self._set_scan_status(
                "that trade is not in the Journal's current filter - widen the "
                "date range and try again"
            )

    def _record_auto_mode_flip(self, previous: str, current: str) -> None:
        """Say the flip in the Auto Pilot log. Not in the journal (TJ-1 item 1).

        It used to write a Market Journal row with SPY's tape attached, on the
        theory that the journal would read as one timeline. It read as noise
        instead: on the live desk 2026-09-17 those rows were **34 of 77** and
        the nightly narration repeated them back. The trader's answer was
        "i don't need to see the SPY auto modes pasted in there" (decision
        0021 answer 3), so the desk's own hand goes where the desk's own lines
        already live - one Auto Pilot log line, no journal write, no capture.

        The method keeps its name and its caller
        (`autoModeChanged.connect(self._record_auto_mode_flip)`), and it stays
        quiet on every failure path: the mode has already changed by the time
        this runs, and a log line may never cost the thing it records. The old
        rows are still on disk and are FILTERED at read time
        (`market_journal.is_machine_entry`), never deleted.
        """
        try:
            self.autopilot_panel.service.log(
                f"Auto mode {previous or 'UNKNOWN'} -> {current or 'UNKNOWN'}."
            )
        except Exception:
            logging.exception("The auto-mode flip could not be logged.")

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
            import working_lately
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
                        # ST6.6. The cell the M5 row already carries and the
                        # held x ran suffix already attached to it - travelling,
                        # never recomputed. The recap classifies nothing.
                        "cell": " ".join(working_lately.alert_priority_key(alert)).strip(),
                        "held_run_suffix": str(
                            getattr(alert, "held_run_suffix", "") or ""
                        ),
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
        # The Trade Mentor checkbox lives on this panel, so the "next prompt"
        # line beside it has to answer the switch the trader just flipped
        # rather than whatever it said when the window opened.
        self._sync_trade_mentor_label()

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
        self.day_review_panel.refresh_reader_measure()

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

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's own spelling
        """Start the tag-review badge the first time the window is actually shown.

        NOT in `__init__`, and the reason is measured rather than tidy: the count
        opens the journal on a background thread, and a thread that starts during
        construction runs while a test is still monkeypatching the journal's
        module globals. It made `test_migration_failure_stays_visible_instead_of_
        claiming_no_accounts` fail from a hundred tests away - green alone, red in
        the suite - which is the worst kind of failure to own.

        A window that is never shown is a window nobody is reading a badge on, so
        this costs production nothing: the desk always shows.
        """
        super().showEvent(event)
        if not getattr(self, "_tag_badge_started", False):
            self._tag_badge_started = True
            self._start_tag_review_badge()
        # ST6.3 trigger (a): once, after the window is actually on screen - for
        # the same reason the badge waits. The build opens three stores on a
        # worker, and a thread started during construction runs while a test is
        # still monkeypatching the module globals it reads.
        self.working_lately_service.start()
        # WISHLIST 10J, for the same reason: the first poll reads the Settings
        # flag and a state file, and a timer started during construction runs
        # while a test is still monkeypatching what it reads.
        self.trade_mentor_service.start()
        self._sync_trade_mentor_label()
        # Trader request 2026-09-14, kept by TJ-1: the day page reads today by
        # itself at 12:00 Pacific, and its post-close tick builds that session's
        # index. Same seam, same reason - the tick reads a setting.
        self.day_review_panel.start()

    # -- Trade Mentor (WISHLIST 10J) --------------------------------------
    def _trade_mentor_cached_bars(self, timeframe, symbols, *, now, timeout_seconds):
        """Read existing desk caches only; a miss is left for the service batch.

        The M5 call is the BounceBot's documented memory-only chart accessor.
        Daily CSVs are the scanner's local cache.  Neither request can start
        IB or alter a detector, and this callback runs on the context worker.
        """
        names = tuple(str(symbol or "").strip().upper() for symbol in symbols)
        if timeframe == "m5":
            try:
                bot = self.trading_panel.bounce_panel.service.current_bot()
            except Exception:
                bot = None
            if bot is None:
                return {}
            result = {}
            for symbol in names:
                try:
                    bars = bot.m5_chart_bars(symbol, max_sessions=2)
                except Exception:
                    bars = []
                if bars:
                    result[symbol] = bars
            return result
        if timeframe == "d1":
            try:
                from d1_environment_store import _cached_daily_bars

                return {symbol: _cached_daily_bars(symbol) for symbol in names}
            except Exception:
                logging.debug("Trade Mentor D1 cache unreadable.", exc_info=True)
        return {}

    def _previous_mentor_read(self, session: str):
        """The last read the Trade Mentor filed for this session, if any.

        Shown beside the new prompt so "Read unchanged" has something to name.
        One bounded read of a small JSONL, at most once an hour - not a paint
        path, and never in the 60-second poll (the service emits, this runs).

        The answer is the latest read of EACH timeframe, `{"M5": row, "D1":
        row}`, not the latest row full stop. "Read unchanged" reaffirms PER
        TIMEFRAME, and `rows[-1]` is the D1 row on every day the 08:00 card was
        answered - which is how a D1-timeframe entry came to be filed at 09:00
        carrying a rest-of-day call, a row true of neither timeframe.
        """
        try:
            from ui.services.market_journal_service import shared_journal_service

            rows = [
                row
                for row in shared_journal_service().entries_for(session)
                if str(row.get("origin") or "") == "trade_mentor"
            ]
        except Exception:  # noqa: BLE001 - a missing previous read is not an error
            logging.debug("Previous mentor read unreadable.", exc_info=True)
            return None
        latest: dict[str, dict] = {}
        for row in rows:
            timeframe = str(row.get("timeframe") or "").strip().upper()
            if timeframe in ("M5", "D1") and str(row.get("text") or "").strip():
                latest[timeframe] = row
        return latest or None

    def _show_trade_mentor_prompt(self, slot) -> None:
        """Show a due prompt in its reusable popup, with its question."""
        review = self.trading_panel.alert_center.chart_review
        try:
            review.show_mentor_slot(slot, previous=self._previous_mentor_read(str(slot.session)))
        except Exception:  # noqa: BLE001 - a prompt never costs the desk
            logging.debug("Trade Mentor prompt could not be shown.", exc_info=True)
            return
        # The Settings line says when the NEXT one is, so it moves every time a
        # prompt lands rather than telling the trader what was true at startup.
        self._sync_trade_mentor_label()
        # The 09:00 second section, and the RIDE. Everything from here down is
        # inside one guard on purpose: a trade check that cannot be built must
        # never cost the prompt above it, which is the read the trader is
        # actually being interrupted for. The kind test used to sit OUTSIDE it
        # and to read the constant off the wrong module, so every prompt raised
        # `AttributeError` in a Qt slot and no trade section ever appeared.
        try:
            import trade_mentor_trade_check as check
            from journal_store import JournalStore
            from trade_mentor_schedule import KIND_M5_TRADES

            card = review.mentor_card
            store = JournalStore()
            # The TASK is built FIRST and the pulls come after it (TJ-14B review,
            # item B): a raise anywhere in the pull path used to lose the forced
            # trade section that is built further down, which is the one thing on
            # this card the trader is not allowed to skip.
            task = check.build_task(store, slot.scheduled_at.date())
            # TJ-14B item 4: ONE card starts AT MOST ONE import. Everything
            # inside is guarded and returns rather than raises.
            self._mentor_journal_pull(slot, task)
            # TJ-14B item 3: the few questions the desk cannot work out on its
            # own, at most three, on EVERY card - the trade section below has
            # its own ride rule and its own early returns.
            self._show_mentor_questions(slot, store=store)

            is_check_slot = str(getattr(slot, "kind", "")) == KIND_M5_TRADES
            # EVERY delivered slot of the session hands the card the FRESH
            # task; the card MERGES it (`set_trade_check`), keeping the exact
            # widgets of a row the trader may already have touched, adding a
            # trade it does not hold yet, dropping one that is answered, and
            # rewriting the heading every time.
            #
            # The host used to return early whenever the card held ANSWER
            # WIDGETS, which was the only protection those half-set combos had.
            # Once the 09:00 not-ready card started drawing today's own fills
            # (TJ-14B) that early return fired on every later slot of the day:
            # the reviewed session's trades were never asked about at all, and
            # the card went on printing `journal not ready` and a freshness
            # date that was no longer true. The protection now lives in the
            # merge, which is where it can protect the widgets WITHOUT also
            # freezing the words above them.
            carrying = str(card.trade_check_session() or "") == str(slot.session)
            if not is_check_slot and not carrying and not self._trade_check_is_owed(check, slot):
                return

            card.set_trade_check(task, store=store, auto_mode=self._auto_mode_now())
        except Exception:  # noqa: BLE001 - the read still stands without it
            logging.debug("Trade Mentor trade check could not be built.", exc_info=True)

    def _trade_check_is_owed(self, check, slot) -> bool:
        """Does this ORDINARY slot have to carry the trade check?

        TJ-9 item 2: *"AWAY still prompts nothing; the first DESK slot after it
        carries the section."* The section used to be built only for the
        `m5_trades` kind, and only the 09:00 slot has that kind - so a 09:00
        that was away, idle, locked, skipped or expired took the whole day's
        questions with it, and a trader who sat down at 11:00 was asked nothing
        at all.

        It is owed when the reviewed session still has an unlabelled trade, or
        an EXIT nobody has explained (TJ-9E), or when its broker statement has
        not landed (item 6's line has to ride too). A session whose trades are
        all answered brings nothing back. AWAY needs no test here: the service
        records the absence and never emits `promptDue`, so an ordinary slot in
        AWAY does not reach this.

        The exit count is asked SEPARATELY and not folded into the unlabelled
        one, because they are two questions: a swing whose four entry fields
        were answered the morning after it opened is not unlabelled and can
        still have an exit nobody explained. Review 1 blocker 4 is what one
        number costs - a trader who was away at 09:00, or who dismissed the
        card, was never asked about that exit at all, and the next morning the
        reviewed session has moved on.
        """
        try:
            reviewed = check.previous_exchange_session(slot.scheduled_at.date())
            # Only what may still be ASKED: a trade asked once (`MENTOR_ASKED`,
            # 2026-09-23) keeps its blanks and never brings the section back.
            if self.trade_mentor_service.unlabelled_trades(reviewed, askable=True) > 0:
                return True
            if self.trade_mentor_service.unexplained_exits(reviewed, askable=True) > 0:
                return True
            from journal_store import JournalStore

            return check.fills_current_to(JournalStore()) != _date.fromisoformat(reviewed)
        except Exception:  # noqa: BLE001 - an unreadable journal asks nothing extra
            logging.debug("Trade check ride undecidable.", exc_info=True)
            return False

    def _show_mentor_questions(self, slot, store=None) -> None:
        """Put this card's budgeted questions on it (TJ-14B item 3).

        Everything `mentor_questions.pending` needs arrives already loaded - the
        registry is PURE and a trigger that opened a store would be a second
        opinion about it, on whatever thread the card happened to be built on.
        The reads here are bounded to the two sessions a question can be about.

        `carried` is kept on the window so the next card asks the remainder
        first: a question over budget is counted and carried, never dropped.
        """
        try:
            import mentor_questions

            card = self.trading_panel.alert_center.chart_review.mentor_card
            if store is None:
                from journal_store import JournalStore

                store = JournalStore()
            state = self._mentor_question_state(slot, store)
            result = mentor_questions.pending(state, slot)
            self._mentor_carried = tuple(result.carried)
            card.set_questions(result, store=store, service=self.trade_mentor_service)
        except Exception:  # noqa: BLE001 - a question never costs the prompt
            logging.debug("Trade Mentor questions could not be built.", exc_info=True)

    def _mentor_question_state(self, slot, store) -> dict:
        """Every lane `mentor_questions.pending` reads, loaded once, here."""
        import trade_mentor_trade_check as check

        session = str(getattr(slot, "session", "") or "")
        reviewed = check.previous_exchange_session(slot.scheduled_at.date())
        trades: list = []
        for day in (session, reviewed):
            try:
                trades.extend(store.list_trades(trade_date=day))
            except Exception:  # noqa: BLE001 - an unreadable day asks nothing
                logging.debug("Mentor trade lane unreadable for %s.", day, exc_info=True)
        return {
            "session": session,
            "now": slot.scheduled_at,
            "auto_mode": self._auto_mode_now(),
            "trades": trades,
            "open_positions": [
                row for row in trades if str(row.get("status") or "").upper() != "CLOSED"
            ],
            "likes": self._mentor_like_lane(trades, (session, reviewed)),
            # `grader_gap` is still DORMANT (plan.md §12.5 names TJ-10's small
            # follow-up), so its lane is still named and still empty.
            "grader_gaps": (),
            # TJ-12 woke `trade_origin` and `open_position_check`, so these are
            # really read now. An EMPTY lane is not neutral here: `planned_state`
            # answers `unplanned` when nothing was said, and an unread store
            # looks exactly like nothing said - the trader would be asked where
            # every trade came from.
            **self._mentor_origin_lanes((session, reviewed)),
            "ai_question": self._mentor_ai_question(),
            # TJ-9E, and it is a LANE like every other one here: the registry
            # is pure, so a trigger that opened a store would be a second
            # opinion about it on whatever thread the card was built on. The
            # kind shipped AWAKE with nothing feeding this key, so lead
            # decision 7's budget clause could never fire (review 1 advisory
            # 2). The window ends at the CARD's own session, never the reviewed
            # one, because a draft is offered on its own clock.
            #
            # MEASURED, on the Qt thread at card-show time, over a 201-trade
            # scratch journal carrying 20 drafts: **about 4 ms warm** for the
            # lane, 1.4-1.8 ms on a small journal, and **12.7 ms on the FIRST
            # call** of the process, where the imports and the first statement
            # are paid. ONE pack read: two `opportunity_events` queries cover
            # the whole five-session window, and a session nobody wrote a note
            # in costs no file read at all. Review 2 measured 14.7 ms against
            # the older shape, which read five pack files whatever the journal
            # said. `_mentor_annotation_lane` above already reads a file here,
            # so this is the same class of cost and not a new one; it stays on
            # this thread this round by decision.
            "exit_drafts": self._mentor_exit_drafts(store, session),
            "answered": self._mentor_answered(
                store,
                (session, reviewed),
                trade_ids=[str(row.get("trade_id") or "") for row in trades],
            ),
            "retired": self.trade_mentor_service.retired_subjects(),
            "carried": getattr(self, "_mentor_carried", ()),
        }

    @staticmethod
    def _mentor_exit_drafts(store, session: str) -> list:
        """The night's exit readings the trader has NOT signed off yet (TJ-9E).

        The lane behind `exit_draft_review`, and the reason the Confirm click
        exists at all: a draft is offered on its OWN clock
        (`check.EXIT_DRAFT_OFFER_SESSIONS`, walked on the exchange calendar),
        never on the session the card happens to be reviewing. The trade exits
        Monday, the note is typed on TUESDAY's card, TUESDAY NIGHT drafts it,
        and Wednesday's card reviews Tuesday - where Monday's trade is not a
        row at all.

        `session` is the CARD's own session, so the window is the trader's last
        five sessions ending today. The rule and the reads live in
        `trade_mentor_trade_check`; this is the seam that hands them to a pure
        registry, because a trigger that opened a store would be a second
        opinion about it on whatever thread the card was built on.

        Never raises: a lane never costs the card.
        """
        try:
            import trade_mentor_trade_check as check

            return list(check.waiting_exit_drafts(store, str(session or "")[:10]))
        except Exception:  # noqa: BLE001 - a lane never costs the card
            logging.debug("Mentor exit-draft lane unreadable.", exc_info=True)
            return []

    @staticmethod
    def _mentor_annotation_lane(days) -> list:
        """The sessions' like/claim annotations, bounded to those sessions.

        ONE reader for two lanes: the quick-like follow-up and TJ-12's
        planned-vs-unplanned question both ask this log the same bounded
        question, and two walks of an append-only file on the Qt thread is the
        shape every other log-walking read grew a stall out of.
        """
        from pathlib import Path

        from project_paths import TRADER_ANNOTATIONS_FILE
        from ui.annotations.store import EVENT_LIKE_CLAIM, load_annotations

        rows: list[dict] = []
        for day in [str(value)[:10] for value in days if str(value or "").strip()]:
            rows.extend(
                load_annotations(
                    Path(TRADER_ANNOTATIONS_FILE),
                    session_date=day,
                    event_types=(EVENT_LIKE_CLAIM,),
                )
            )
        return rows

    @staticmethod
    def _mentor_origin_lanes(days) -> dict:
        """The lanes `trade_origin.planned_state` reads, loaded once, bounded.

        Each lane is guarded on its own: an unreadable store asks MORE questions
        (nothing was said about that name, as far as the desk can tell) and
        never takes the Mentor card down.

        Which lanes are really read is `day_report_card.DESK_ORIGIN_LANES_READ`
        and is declared THERE, once, because the Day Review worker builds the
        same lanes and the card has to say which doors were opened. An unread
        lane is indistinguishable from "nothing was said" to
        `trade_origin.planned_state`, so the card names it rather than printing
        a bare `unplanned` (reviewer, 2026-09-20: 30 of the trader's 33 trades
        since 2026-08-20 read `unplanned` for exactly this reason).

        `focus_adds` and `armed` are named and EMPTY until **TJ-12F**: neither
        store has a public reader that hands back a row with the stamp key
        `trade_origin` reads, and inventing one is that packet's work. A trade
        planned only through a Focus add or an armed alert is therefore asked
        once - `CADENCE_ONCE` - and the trader's own answer is what the Process
        line then reads.
        """
        import day_report_card

        wanted = [str(value)[:10] for value in days if str(value or "").strip()]
        decisions: list[dict] = []
        try:
            decisions = list(MainWindow._mentor_annotation_lane(wanted))
        except Exception:  # noqa: BLE001 - an unreadable log says nothing
            logging.debug("Mentor decision lane unreadable.", exc_info=True)
        claims: list[dict] = []
        try:
            import claimed_picks

            claims = [
                row
                for row in claimed_picks.load_rows()
                if str(row.get("session_date") or "")[:10] in wanted
            ]
        except Exception:  # noqa: BLE001 - an unreadable store says nothing
            logging.debug("Mentor claim lane unreadable.", exc_info=True)
        loaded = {"decisions": decisions, "claims": claims}
        return {
            name: loaded.get(name, ())
            if name in day_report_card.DESK_ORIGIN_LANES_READ
            else ()
            for name in day_report_card.ORIGIN_LANES
        }

    @staticmethod
    def _mentor_like_lane(trades, days) -> list:
        """The sessions' QUICK likes, each told whether it was then traded.

        The join is by name and SIDE against the same two sessions' trades - a
        LONG like says nothing about a SHORT entry - and it is done here rather
        than in the registry so the trigger stays pure.

        **Bounded to the sessions a question can be about** (TJ-14B review, item
        E): the store's own `session_date` filter is asked once per session
        rather than the whole append-only log being walked on the Qt thread
        every prompt - 3.5 ms today and unbounded, which is the shape that grew
        every other log-walking read into a stall.
        """
        wanted = [str(day)[:10] for day in days if str(day or "").strip()]
        rows: list[dict] = []
        try:
            rows = list(MainWindow._mentor_annotation_lane(wanted))
        except Exception:  # noqa: BLE001 - a missing log asks nothing
            logging.debug("Like lane unreadable.", exc_info=True)
            return []
        traded: set[tuple[str, str]] = set()
        for trade in trades:
            symbol = str(trade.get("symbol") or "").strip().upper()
            side = str(trade.get("direction") or "").strip().upper()
            if symbol and side:
                traded.add((symbol, side))
        lane: list[dict] = []
        for row in rows:
            symbol = str(row.get("symbol") or "").strip().upper()
            side = str(row.get("side") or "").strip().upper()
            side = "LONG" if side.startswith("LONG") else "SHORT" if side.startswith("SHORT") else ""
            enriched = dict(row)
            if symbol and side and (symbol, side) in traded:
                enriched["matched_trade_id"] = f"{symbol}:{side}"
            lane.append(enriched)
        return lane

    @staticmethod
    def _mentor_ai_question() -> dict:
        """Last night's coaching question and its click options, or `{}`.

        ONE walk of the narrations folder - the card's own legacy line reads the
        same newest file, and it is hidden when this becomes a click.
        """
        try:
            import json
            from pathlib import Path

            from project_paths import MARKET_STORY_NARRATIONS_DIR

            for path in reversed(sorted(Path(MARKET_STORY_NARRATIONS_DIR).glob("*.json"))):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                narration = payload.get("narration") if isinstance(payload, dict) else None
                if not isinstance(narration, dict):
                    continue
                question = str(narration.get("mentor_question") or "").strip()
                if not question:
                    continue
                options = narration.get("mentor_question_options") or ()
                return {
                    "question": question,
                    "options": tuple(str(item) for item in options if str(item or "").strip()),
                }
        except Exception:  # noqa: BLE001 - a degraded night asks nothing
            logging.debug("Overnight Mentor question unreadable.", exc_info=True)
        return {}

    @staticmethod
    def _mentor_answered(store, days, trade_ids=()) -> dict:
        """Which questions already have an answer, so none is asked twice.

        Two stores, because two kinds file their answers as the trader's own
        dated Market Journal row (`day_close`, `ai_question`) and the rest as
        append-only annotation rows. Both reads are bounded to the sessions a
        question can be about.

        **An answer about a TRADE is found by the trade, never only by the day
        it was given on** (trader 2026-09-21: *"dont ask for it again just store
        that info"*). A trade's `trade_date` MOVES - an open position answered
        on Monday closes on Thursday and lands back on Thursday's card - and a
        read keyed on the answer's own date could not see Monday's row, so a
        `once` question came back. Read FIRST, so a same-day row still wins
        the stamp a daily kind compares.
        """
        answered: dict[str, dict] = {}
        import mentor_questions

        def _keep(rows) -> None:
            for row in rows:
                payload = row.get("payload") or {}
                kind = str(payload.get("mentor_question_kind") or "")
                subject_id = str(payload.get("subject_id") or "")
                if kind and subject_id:
                    answered[f"{kind}:{subject_id}"] = {
                        "answered_at": str(row.get("occurred_at") or "")[:10]
                    }

        for trade_id in dict.fromkeys(str(value or "").strip() for value in trade_ids):
            if not trade_id:
                continue
            try:
                _keep(
                    store.list_opportunity_events(
                        trade_id=trade_id,
                        event_type=mentor_questions.EVENT_MENTOR_ANSWER,
                        limit=1000,
                    )
                )
            except Exception:  # noqa: BLE001
                logging.debug("Mentor answers unreadable for %s.", trade_id, exc_info=True)

        for day in days:
            if not str(day or "").strip():
                continue
            try:
                rows = store.list_opportunity_events(
                    event_type=mentor_questions.EVENT_MENTOR_ANSWER,
                    trade_date=str(day)[:10],
                    limit=1000,
                )
            except Exception:  # noqa: BLE001
                logging.debug("Mentor answers unreadable for %s.", day, exc_info=True)
                continue
            _keep(rows)
        try:
            from ui.services.market_journal_service import shared_journal_service

            for day in days:
                if not str(day or "").strip():
                    continue
                for row in shared_journal_service().entries_for(str(day)[:10]):
                    payload = (row.get("mentor") or {}).get("mentor_question") or {}
                    kind = str(payload.get("mentor_question_kind") or "")
                    subject_id = str(payload.get("subject_id") or "")
                    if kind and subject_id:
                        answered[f"{kind}:{subject_id}"] = {
                            "answered_at": str(payload.get("answered_at") or "")[:10]
                            or str(day)[:10]
                        }
        except Exception:  # noqa: BLE001 - a missing journal answers nothing
            logging.debug("Mentor journal answers unreadable.", exc_info=True)
        return answered

    def _auto_mode_now(self) -> str:
        """The Auto Pilot mode, or `""` when it cannot be read.

        An unreadable mode is NOT an absence: the safe direction for a pull is
        the same as for a prompt, and the Mentor service has already decided
        the trader is present by the time this runs.
        """
        try:
            from autopilot_core import read_auto_pilot_mode

            return str(read_auto_pilot_mode() or "").upper()
        except Exception:  # noqa: BLE001 - an unreadable mode blocks nothing
            logging.debug("Auto mode unreadable for the pre-card pull.", exc_info=True)
            return ""

    def _mentor_journal_pull(self, slot, task) -> dict:
        """The ONE owner of the desk's day-time Questrade attempts (TJ-14B).

        **One card starts AT MOST ONE import** (review blocker 1). Two calls in
        one synchronous slot cannot both start: `JournalImportService.running`
        refuses the second, and the one that lost was the THREE-day morning
        catch-up, marked spent and never fired again that morning - so a Monday
        whose Friday-night import had failed never reached back to Friday.

        So they are ordered, never stacked:

        * when the morning catch-up is OWED - the reviewed session's statement
          has not landed and nothing has been retried today - it goes FIRST and
          the pre-card pull is skipped on this card. Its three days cover today
          too, and it does not spend the pre-card cap;
        * otherwise the pre-card pull runs, and only on one of the day's three
          RESERVED cards (`mentor_questions.pull_slot_ids`: the 09:00 card, the
          middle card, and the last one).

        The tally is persisted beside the Mentor's slot state, so a restart
        cannot spend the day's attempts twice; a corrupt one is read as empty
        and rewritten clean. The pull runs on `JournalImportService`'s own
        `QThread`, the desk's single caller of the Questrade refresh chain, and
        nothing here refreshes a token. **The card never waits for it**: a fill
        a late pull lands is asked about on the NEXT card, and a pull that
        raises, refuses or is busy costs the card nothing.
        """
        try:
            import mentor_questions
            import trade_mentor_trade_check as check

            service = self.trade_mentor_service
            tally = service.pull_tally()
            day = str(getattr(slot, "session", "") or "")
            auto_mode = self._auto_mode_now()
            last_retry = self._journal_retry_date or str(tally.get("last_retry") or "")
            if not getattr(task, "journal_ready", False) and str(last_retry)[:10] != day[:10]:
                outcome = check.morning_import_retry(
                    self._journal_import_service(),
                    task,
                    today=day,
                    # The once-a-morning date is kept in the PERSISTED tally
                    # too, so a restart before 10:00 no longer allows a second
                    # morning pull (TJ-9's advisory).
                    last_retry=last_retry,
                    tally=tally,
                    auto_mode=auto_mode,
                )
                self._journal_retry_date = str(outcome.get("last_retry") or "")
                persisted = dict(outcome.get("tally") or tally)
                persisted["last_retry"] = self._journal_retry_date
                service.set_pull_tally(persisted)
                if outcome.get("retried"):
                    logging.info(
                        "Trade Mentor: retrying the Questrade import before the "
                        "%s card (fills current to %s).",
                        getattr(slot, "slot_id", ""),
                        getattr(task, "fills_current_to", "") or "nothing yet",
                    )
                return outcome
            outcome = mentor_questions.pre_card_pull(
                self._journal_import_service(),
                today=day,
                tally=tally,
                auto_mode=auto_mode,
                slot=slot,
            )
            persisted = dict(outcome.get("tally") or {})
            if last_retry:
                persisted["last_retry"] = last_retry
            service.set_pull_tally(persisted)
            if outcome.get("pulled"):
                logging.info(
                    "Trade Mentor: pulling the Questrade journal before the %s card.",
                    getattr(slot, "slot_id", ""),
                )
            return outcome
        except Exception:  # noqa: BLE001 - a pull never costs the card
            logging.debug("Mentor journal pull failed.", exc_info=True)
            return {}

    def _journal_import_service(self):
        """The desk's ONE journal import owner, built on first need.

        `JournalImportService` owns its own `QThread` and is the single caller
        of the Questrade refresh chain; the Mentor's morning retry calls it and
        never refreshes a token itself.
        """
        service = getattr(self, "_journal_importer", None)
        if service is None:
            from ui.services.journal_import_service import JournalImportService

            service = JournalImportService(self)
            self._journal_importer = service
        return service

    def _pause_trade_mentor(self) -> None:
        self.trade_mentor_service.pause_today()
        self.trading_panel.alert_center.chart_review.hide_mentor_card()
        self._sync_trade_mentor_label()

    def _sync_trade_mentor_label(self) -> None:
        try:
            self.settings_panel.set_next_prompt_at(self.trade_mentor_service.next_prompt_at())
        except Exception:  # noqa: BLE001 - a label never breaks the window
            logging.debug("Trade Mentor label could not be refreshed.", exc_info=True)

    def _show_setup_tracker_page(self) -> None:
        """The Working-lately strip's click-through (ST6.4).

        The Setup Tracker is a TAB inside the Research page, so this switches
        the page by TITLE - never by index, which a reorder would silently
        unwire - and then asks the Research panel to raise its own tab. The
        banner there prints the same `snapshot_id` the strip just showed.
        """
        for index, spec in enumerate(PAGE_SPECS):
            if spec.title == "Research":
                self._select_page(index)
                break
        try:
            self.research_panel.show_setup_tracker()
        except Exception:  # noqa: BLE001 - a page switch never costs the desk
            logging.debug("Setup Tracker tab could not be raised.", exc_info=True)

    def closeEvent(self, event) -> None:
        # V2: the badge reader, before the panels. It is one bounded read and it
        # must not outlive the window that started it.
        self._join_tag_review_badge()
        for panel in (
            self.trading_panel,
            self.journal_panel,
            # TJ-1: ONE page where the Market Journal and the Daily Recap were
            # two. It owns a read worker and writes through the shared journal
            # service's capture threads, and a capture killed half-written is
            # what this list is for.
            self.day_review_panel,
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
        # Same reason, same list: the Working-lately service is owned by the
        # window (four surfaces read it) and holds one timer and one bounded
        # reader. ST6.3.
        try:
            self.working_lately_service.shutdown()
        except Exception:
            pass
        # Same list, same reason (WISHLIST 10J): one timer, owned here.
        try:
            self.trade_mentor_service.shutdown()
        except Exception:
            pass
        try:
            self.trade_mentor_context_service.shutdown(timeout_ms=250)
        except Exception:
            pass
        # TJ-9 item 6: the morning retry's worker, when one was ever built.
        try:
            if self._journal_importer is not None:
                self._journal_importer.shutdown()
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
    # yfinance's download threads each leave SQLite connections that only the
    # GUI-thread collector would free; close each one on its own thread instead.
    from peewee_thread_close import install as install_peewee_thread_close

    install_peewee_thread_close()
    # Scale first: every widget built below reads theme.px() at construction.
    apply_theme(
        app,
        state.theme_name,
        state.compact_density,
        theme.resolve_scale(state.ui_scale, _available_screen_size()),
    )

    window = MainWindow(state)
    window.show()
    # Snappiness packet 2, item 2: sweep the startup garbage once, then take the
    # survivors out of every future sweep's scan set.
    #
    # All cyclic collection runs on this thread by design (install_gui_thread_gc
    # above, and that design stays), so every sweep's cost is a GUI freeze. On
    # 2026-08-31 gen-0 sweeps averaged ~300 ms and full sweeps ~770 ms, and 6.5
    # minutes of that day's 78 minutes of freeze was the collector. Most of what
    # it walks is the startup graph - the widget tree, the theme, every imported
    # module - which lives for the whole process and can never be garbage.
    # `gc.freeze()` moves that graph to a permanent generation the collector does
    # not scan; the collect first is what makes sure only survivors get frozen.
    # Nothing about the controller's cadence, deadlines or disable/collect design
    # changes: the same sweeps happen, over a smaller heap.
    gc.collect(2)
    gc.freeze()
    # Off unless this machine asked for it. When on, every GUI-thread block
    # over the threshold is logged with the stack that caused it, which is
    # the only honest way to pick what to optimize next (Part C rule C1).
    from ui.stall_watchdog import install as install_stall_watchdog

    window.stall_watchdog = install_stall_watchdog(window)
    # Always on, one syscall per thread per minute: names any thread that is
    # holding the interpreter lock, which the stall watchdog cannot see
    # (2026-09-03: the M5 tee thread at 91% of GIL samples, unnamed for 8 h).
    from ui.thread_cpu_gauge import install as install_thread_cpu_gauge

    window.thread_cpu_gauge = install_thread_cpu_gauge()
    try:
        return app.exec()
    finally:
        _print_qt_message_tally()


if __name__ == "__main__":
    raise SystemExit(main())
