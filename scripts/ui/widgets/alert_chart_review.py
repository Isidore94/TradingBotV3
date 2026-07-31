from __future__ import annotations

"""Automatic D1/M5 visual review surface for Alert Center alerts."""

from typing import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from chart_watch import WATCH_KINDS
from ui.models.bounce import MANUAL_CHART_TAG, BounceAlert, is_auto_pick_alert
from ui.widgets.arm_bar import ArmBar
from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

_NO_M5_WATCH_REASON = (
    "No cached M5 bars for this symbol yet - arming still works: BounceBot "
    "folds armed names into its M5 scan set, so bars land within a scan "
    "cycle and the watch starts evaluating then."
)


class AlertChartReview(QWidget):
    """Chart + queue controls.

    Only three actions advance the review queue: "Remove for today", "Skip
    for now", and the type-matched focus add (M5 pick -> M5 Focus, swing
    pick -> Swing Focus). Everything on the second row is a TOGGLE that
    leaves the chart in place: the cross-focus button (M5 pick -> pin into
    the D1 Focus feed; swing pick -> add to the M5 Focus day-trade list) and
    the one-shot chart watches (click again to disarm).
    """

    removeTodayRequested = Signal(object)
    focusRequested = Signal(object)
    skipRequested = Signal(object)
    crossFocusToggled = Signal(object)
    watchToggled = Signal(object, str)  # (alert, chart-watch kind)
    d1EventToggled = Signal(object, str)  # (alert, D1 event watch kind)
    d1LevelAlertRequested = Signal(str, str, float, str)  # symbol, direction, level, candle date
    symbolRequested = Signal(str)  # type-a-ticker: chart it on demand
    levelArmRequested = Signal(str, str, float)  # symbol, direction, level
    levelDisarmRequested = Signal(str, str, float)  # symbol, direction, level
    autoPickApproved = Signal(object)  # DESK-mode auto pick: add to watchlist
    autoPickPassed = Signal(object)  # DESK-mode auto pick: decline for today

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.alert: BounceAlert | None = None
        self._cross_labels = ("Add to D1 Focus", "✓ In D1 Focus")

        self.title = QLabel("Visual Alert Review")
        self.title.setObjectName("SectionTitle")
        # The setup line: WHAT exactly fired/is being looked at. Styled large
        # via ReviewSetupText, and red (alertLive property) when a live alert
        # put this chart up - the trader reads it from across the desk.
        self.alert_text = QLabel("Waiting for the next ticker alert.")
        self.alert_text.setWordWrap(True)
        self.alert_text.setObjectName("ReviewSetupText")
        # Guidance line from the review-learning loop (take-prob, segment
        # edge, AI-policy notes like "Blind spot: ..."). Purely advisory -
        # it annotates the chart the trader is already looking at.
        self.guidance_label = QLabel("")
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setObjectName("GuidanceLabel")
        self.guidance_label.setVisible(False)
        self.queue_label = QLabel("")
        self.queue_label.setObjectName("MutedLabel")

        # compact: this pane is height-starved in the desk column, so legends
        # stay on one line and the candles get the pixels instead.
        self.snapshot = SymbolSnapshotWidget(self, compact=True)
        self.snapshot.setVisible(False)
        # Candle clicks on the embedded D1 chart arm persistent level alerts
        # through the hosting panel.
        self.snapshot.d1LevelAlertRequested.connect(self.d1LevelAlertRequested)

        self.remove_today_button = QPushButton("Remove for today")
        self.remove_today_button.setToolTip(
            "Remove this symbol from today's Alert Center feed and review queue. "
            "The BounceBot scanner and watchlists are untouched."
        )
        self.remove_today_button.clicked.connect(
            lambda: self.alert is not None and self.removeTodayRequested.emit(self.alert)
        )
        self.focus_button = QPushButton("Add to Focus Picks")
        self.focus_button.clicked.connect(
            lambda: self.alert is not None and self.focusRequested.emit(self.alert)
        )
        self.skip_button = QPushButton("Skip for now")
        self.skip_button.clicked.connect(
            lambda: self.alert is not None and self.skipRequested.emit(self.alert)
        )

        # DESK-mode auto-pick verdict verbs. Hidden except when an auto-populate
        # proposal occupies the chart; both advance the review queue.
        self.auto_approve_button = QPushButton("✓ Add to watchlist")
        self.auto_approve_button.setToolTip(
            "Approve this auto pick: it joins the auto-owned slice of the "
            "BounceBot watchlist and gets M5-scanned immediately."
        )
        self.auto_approve_button.clicked.connect(
            lambda: self.alert is not None and self.autoPickApproved.emit(self.alert)
        )
        self.auto_pass_button = QPushButton("✕ Pass")
        self.auto_pass_button.setToolTip(
            "Decline this auto pick for today - it will not be proposed again "
            "this session and the watchlists are untouched."
        )
        self.auto_pass_button.clicked.connect(
            lambda: self.alert is not None and self.autoPickPassed.emit(self.alert)
        )
        self.auto_approve_button.setVisible(False)
        self.auto_pass_button.setVisible(False)

        self.cross_focus_button = QPushButton(self._cross_labels[0])
        self.cross_focus_button.setCheckable(True)
        self.cross_focus_button.clicked.connect(
            lambda: self.alert is not None and self.crossFocusToggled.emit(self.alert)
        )
        # The four watch toggles live on the arm dock now, so this pane owns
        # one row of queue verbs instead of two rows of mixed controls.
        self.arm_bar = ArmBar(self)
        self.arm_bar.set_quick_fill_source(self.snapshot.quick_fill)
        self.arm_bar.watchToggled.connect(
            lambda kind: self.alert is not None and self.watchToggled.emit(self.alert, kind)
        )
        self.arm_bar.d1EventToggled.connect(
            lambda kind: self.alert is not None and self.d1EventToggled.emit(self.alert, kind)
        )
        self.arm_bar.symbolRequested.connect(self.symbolRequested)
        self.arm_bar.levelArmRequested.connect(self._emit_level_arm)
        self.arm_bar.levelDisarmRequested.connect(self._emit_level_disarm)
        self.snapshot.pricePicked.connect(self.arm_bar.set_level)
        # Kept for callers and tests that poke the toggles directly.
        self.watch_buttons = self.arm_bar.watch_buttons

        buttons = QHBoxLayout()
        buttons.addWidget(self.auto_approve_button)
        buttons.addWidget(self.auto_pass_button)
        buttons.addWidget(self.remove_today_button)
        buttons.addWidget(self.focus_button)
        buttons.addWidget(self.skip_button)
        buttons.addWidget(self.cross_focus_button)
        buttons.addStretch(1)
        buttons.addWidget(self.queue_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addWidget(self.title)
        layout.addWidget(self.alert_text)
        layout.addWidget(self.guidance_label)
        layout.addWidget(self.snapshot, 1)
        layout.addWidget(self.arm_bar)
        layout.addLayout(buttons)
        self._set_actions_enabled(False)

    def _emit_level_arm(self, direction: str, level: float) -> None:
        if self.alert is not None and self.alert.symbol:
            self.levelArmRequested.emit(self.alert.symbol, direction, float(level))

    def _emit_level_disarm(self, direction: str, level: float) -> None:
        if self.alert is not None and self.alert.symbol:
            self.levelDisarmRequested.emit(self.alert.symbol, direction, float(level))

    def set_alert(
        self,
        alert: BounceAlert,
        *,
        bot=None,
        focus_category: str = "m5",
        queued: int = 0,
        armed_kinds: Iterable[str] = (),
        cross_active: bool = False,
        armed_levels: Iterable = (),
        armed_d1_events: Iterable[str] = (),
        guidance_text: str = "",
    ) -> None:
        self.alert = alert
        guidance_text = str(guidance_text or "").strip()
        self.guidance_label.setText(guidance_text)
        self.guidance_label.setVisible(bool(guidance_text))
        side = f" · {alert.side}" if alert.side else ""
        timeframe = f" · {alert.timeframe}" if alert.timeframe else ""
        self.title.setText(f"{alert.symbol}{side}{timeframe}")
        self.alert_text.setText(alert.trigger or alert.raw_text)
        # Red setup text = a live alert put this chart up. A typed manual
        # chart and a staged auto pick stay muted, so a glance separates
        # "something fired" from "I was just looking / deciding".
        is_auto_pick = is_auto_pick_alert(alert)
        self._set_setup_text_live(alert.tag != MANUAL_CHART_TAG and not is_auto_pick)
        self.auto_approve_button.setVisible(is_auto_pick)
        self.auto_pass_button.setVisible(is_auto_pick)
        if focus_category == "swing":
            self.focus_button.setText("Add to Swing Focus")
            # Swing pick: the cross-promote is the M5 day-trade list.
            self._cross_labels = ("Add to M5 Focus", "✓ In M5 Focus")
            self.cross_focus_button.setToolTip(
                "Toggle this swing pick onto the M5 Focus day-trade list "
                "(BounceBot M5-scans it immediately). Click again to remove."
            )
        else:
            self.focus_button.setText("Add to M5 Focus")
            # M5 pick: the cross-promote files it as a swing name.
            self._cross_labels = ("Add to D1 Focus", "✓ In D1 Focus")
            self.cross_focus_button.setToolTip(
                "Toggle this pick into Swing Focus (it lands on the Focus "
                "Picks tab and the swing watchlists) and pin it in the D1 "
                "Focus feed below. Click again to remove both."
            )
        self.snapshot.set_symbol(alert.symbol, bot=bot)
        self.snapshot.setVisible(True)
        self.queue_label.setText(f"{queued} waiting" if queued else "queue clear")
        self._set_actions_enabled(True)
        self.set_armed_kinds(armed_kinds)
        self.set_armed_levels(armed_levels)
        self.set_armed_d1_events(armed_d1_events)
        self.set_cross_active(cross_active)
        # Seed the price box with the last traded price so the trader adjusts
        # from something real instead of typing a level from scratch.
        self.arm_bar.apply_quick_fill("last")
        # A session watch can only ever fire off cached M5 bars. Say so on the
        # buttons rather than letting the trader wait on a watch that has
        # nothing to evaluate against.
        has_m5 = bool((self.snapshot._m5 or {}).get("bars"))
        self.arm_bar.set_watch_availability(has_m5, _NO_M5_WATCH_REASON)

    def refresh_chart(self, *, bot=None) -> None:
        """Re-pull the visible D1/M5 charts from the local caches.

        The hosting panel calls this on its 30s watch tick, so an alert the
        trader gets to minutes after it fired shows the bars of NOW, not of
        when it landed in the pane. Only the snapshot re-renders (and only
        when a bar actually changed); the arm dock, typed levels, and queue
        buttons are never touched.
        """
        if self.alert is None:
            return
        if self.snapshot.refresh(bot=bot):
            # M5 bars can appear after the alert landed (e.g. the scan loop
            # reached the symbol) - the watch buttons unlock with them.
            has_m5 = bool((self.snapshot._m5 or {}).get("bars"))
            self.arm_bar.set_watch_availability(has_m5, _NO_M5_WATCH_REASON)

    def _set_setup_text_live(self, live: bool) -> None:
        """Flip the alertLive QSS property (red setup text) with a repolish."""
        live = bool(live)
        if bool(self.alert_text.property("alertLive")) == live:
            return
        self.alert_text.setProperty("alertLive", live)
        style = self.alert_text.style()
        style.unpolish(self.alert_text)
        style.polish(self.alert_text)

    def clear(self) -> None:
        self.alert = None
        self.title.setText("Visual Alert Review")
        self.alert_text.setText("Waiting for the next ticker alert.")
        self._set_setup_text_live(False)
        self.guidance_label.setText("")
        self.guidance_label.setVisible(False)
        self.snapshot.setVisible(False)
        self.queue_label.setText("")
        self.auto_approve_button.setVisible(False)
        self.auto_pass_button.setVisible(False)
        self._set_actions_enabled(False)
        self.set_armed_kinds(())
        self.set_armed_d1_events(())
        self.set_cross_active(False)

    def set_queued_count(self, count: int) -> None:
        self.queue_label.setText(f"{count} waiting" if count else "queue clear")

    def set_armed_kinds(self, kinds: Iterable[str]) -> None:
        """Reflect this symbol's armed watches; buttons stay clickable so a
        second click disarms."""
        self.arm_bar.set_armed_kinds(kinds)

    def set_armed_levels(self, levels: Iterable = ()) -> None:
        """Show this symbol's armed price levels as dismissable chips."""
        self.arm_bar.set_armed_levels(levels)

    def set_armed_d1_events(self, kinds: Iterable[str] = ()) -> None:
        """Reflect this symbol's armed D1 event watches on the dock's D1 row."""
        self.arm_bar.set_armed_d1_events(kinds)

    def set_cross_active(self, active: bool) -> None:
        self.cross_focus_button.setText(self._cross_labels[1 if active else 0])
        self.cross_focus_button.setChecked(bool(active))

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self.remove_today_button,
            self.focus_button,
            self.skip_button,
            self.cross_focus_button,
            self.auto_approve_button,
            self.auto_pass_button,
        ):
            button.setEnabled(enabled)
        # The symbol box stays live even with no alert on screen - typing a
        # ticker is how the trader breaks out of an empty queue.
        self.arm_bar.set_enabled_for_symbol(enabled)
