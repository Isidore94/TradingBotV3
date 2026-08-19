from __future__ import annotations

"""Automatic D1/M5 visual review surface for Alert Center alerts."""

from typing import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from chart_watch import WATCH_KINDS
from ui.models.bounce import (
    FOCUS_REVIEW_TAG,
    MANUAL_CHART_TAG,
    BounceAlert,
    is_auto_pick_alert,
)
from ui import theme
from ui.widgets.arm_bar import ArmBar
from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

_NO_M5_WATCH_REASON = (
    "No cached M5 bars for this symbol yet - arming still works: BounceBot "
    "folds armed names into its M5 scan set, so bars land within a scan "
    "cycle and the watch starts evaluating then."
)


class AlertChartReview(QWidget):
    """Chart + queue controls.

    ONE verb row for every chart, same three buttons in the same spots
    (2026-07-31 user rule: unified tabs, no shifting layouts):

    - Add (positive): type-matched focus add for a scanner alert (M5 pick ->
      M5 Focus, swing pick -> Swing Focus); "✓ Add to watchlist" when a
      DESK-mode auto pick occupies the chart.
    - "Skip for now": just shows the next chart. Nothing is recorded or
      removed - the name can come back today.
    - "✕ Not today": done with this name for the day. Scanner alert: removed
      from today's feed and chart queue. Auto pick: declined - it will not
      be proposed again today. Watchlists and scanning are never touched by
      a scanner-alert dismissal.

    All three advance the review queue. Everything on the arm dock is a
    TOGGLE that leaves the chart in place: the cross-focus button (M5 pick
    -> pin into the D1 Focus feed; swing pick -> add to the M5 Focus
    day-trade list) and the one-shot chart watches (click again to disarm).
    """

    removeTodayRequested = Signal(object)
    focusRequested = Signal(object)
    skipRequested = Signal(object)
    crossFocusToggled = Signal(object)
    watchToggled = Signal(object, str)  # (alert, chart-watch kind)
    d1EventToggled = Signal(object, str)  # (alert, D1 event watch kind)
    anyBounceToggled = Signal(object)  # (alert) - R5 section 4, whole level set
    d1LevelAlertRequested = Signal(str, str, float, str)  # symbol, direction, level, candle date
    symbolRequested = Signal(str)  # type-a-ticker: chart it on demand
    levelArmRequested = Signal(str, str, float)  # symbol, direction, level
    levelDisarmRequested = Signal(str, str, float)  # symbol, direction, level
    # (symbol, direction, level) - arm a PHONE price alert at the painted D1
    # level the trader picked. A request, never a write: price_alerts.json has
    # exactly one writer, the hosting panel that owns PriceAlertService
    # (plan.md sec 5; trader decision 2026-08-09). Nothing on this path mutes,
    # suppresses, scores, gates or reorders anything - it arms an alert.
    levelAlertRequested = Signal(str, str, float)

    def __init__(self, parent=None, *, annotations_path=None) -> None:
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
        # A4's painted D1 levels are clickable; remember which line was picked
        # so the phone-alert affordance has something to arm at.
        self._selected_level: tuple[str, str, str, float] | None = None
        self.snapshot.d1LevelSelected.connect(self._on_level_selected)
        # Charts build off-thread now, so anything that depends on the bars
        # they hold has to wait for them to land rather than reading straight
        # after set_symbol returns.
        self.snapshot.snapshotRendered.connect(self._on_snapshot_rendered)
        self._seed_quick_fill = False

        # The unified verb row: add | skip | not-today. Labels adapt to what
        # occupies the chart (scanner alert vs DESK auto pick) but every
        # button keeps its spot, so muscle memory never misfires.
        self.focus_button = QPushButton("Add to Focus Picks")
        self.focus_button.clicked.connect(
            lambda: self.alert is not None and self.focusRequested.emit(self.alert)
        )
        self.skip_button = QPushButton("Skip for now")
        self.skip_button.setToolTip(
            "Just shows the next chart. Nothing is recorded or removed - "
            "this name can chart again today."
        )
        self.skip_button.clicked.connect(
            lambda: self.alert is not None and self.skipRequested.emit(self.alert)
        )
        self.remove_today_button = QPushButton("✕ Not today")
        self.remove_today_button.setToolTip(
            "Done with this name for the day: removed from today's Alert "
            "Center feed and chart queue. The BounceBot scanner and "
            "watchlists are untouched."
        )
        self.remove_today_button.clicked.connect(
            lambda: self.alert is not None and self.removeTodayRequested.emit(self.alert)
        )

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
        self.arm_bar.anyBounceToggled.connect(
            lambda: self.alert is not None and self.anyBounceToggled.emit(self.alert)
        )
        self.arm_bar.symbolRequested.connect(self.symbolRequested)
        self.arm_bar.levelArmRequested.connect(self._emit_level_arm)
        self.arm_bar.levelDisarmRequested.connect(self._emit_level_disarm)
        self.arm_bar.levelAlertRequested.connect(self._emit_level_alert)
        self.snapshot.pricePicked.connect(self.arm_bar.set_level)
        # Kept for callers and tests that poke the toggles directly.
        self.watch_buttons = self.arm_bar.watch_buttons

        # R4 section 2.3: "I like the stock" on the Alert screen, as capture.
        #
        # The boundary this has to hold: CaptureRail LIKE writes ONE annotation
        # row and nothing else. It is not a placement verb. "Add to Focus
        # Picks" above stays the single explicit thing that puts a name on a
        # list - an earlier draft of the rail routed likes through
        # FocusService.add, which quietly gave a liked name Focus alert
        # privileges, and it had to be torn back out. Keep the two apart.
        from ui.widgets.capture_rail import CaptureRail

        self.capture_rail = CaptureRail(annotations_path=annotations_path)
        self.capture_rail.captured.connect(self._on_captured)
        self.snapshot.d1LevelSelected.connect(self._on_capture_level_selected)

        # R4 section 5, on the surface the trader stares at most.
        self.reviewed_badge = QLabel("")
        self.reviewed_badge.setObjectName("reviewedTodayBadge")
        self.reviewed_badge.setStyleSheet(
            f"color: {theme.color('caution')}; font-weight: 600;"
        )

        buttons = QHBoxLayout()
        buttons.addWidget(self.reviewed_badge)
        buttons.addWidget(self.focus_button)
        buttons.addWidget(self.skip_button)
        buttons.addWidget(self.remove_today_button)
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
        layout.addWidget(self.capture_rail)
        layout.addLayout(buttons)
        self._set_actions_enabled(False)

    # -- R4 sections 2.3 and 5 ------------------------------------------
    def _on_capture_level_selected(
        self, symbol: str, level_id: str, family: str, _price: float
    ) -> None:
        """A clicked paint-line becomes the capture's reference level."""
        if self.alert is None or symbol != self.alert.symbol:
            return
        self.capture_rail.set_context(
            symbol=symbol,
            timeframe="D1",
            ref_level_id=level_id,
            ref_level_family=family,
        )

    def _on_captured(self, _event_type: str, _row: dict) -> None:
        """Capture is a decision, so the badge updates without a re-chart.

        Deliberately does NOT advance the review queue: only the three queue
        verbs move it. A rail that skipped to the next chart would make every
        note cost the trader the chart they were writing it about.
        """
        self._refresh_reviewed_badge()

    def _reviewed_symbols(self) -> set:
        """Today's decided set. Seam so tests read a fixture, not live files."""
        from pick_feedback import reviewed_symbols_today

        return reviewed_symbols_today()

    def _refresh_reviewed_badge(self) -> None:
        symbol = self.alert.symbol if self.alert is not None else ""
        text = ""
        try:
            if symbol and symbol in self._reviewed_symbols():
                text = "● Reviewed today"
        except Exception:
            # Presentation only: a locked evidence file costs the badge, and
            # must never cost the chart the trader is trying to read.
            text = ""
        self.reviewed_badge.setText(text)
        self.reviewed_badge.setToolTip(
            "You already recorded a decision on this symbol today "
            "(dislike, favorite, veto, like or note)."
            if text
            else ""
        )

    def _emit_level_arm(self, direction: str, level: float) -> None:
        if self.alert is not None and self.alert.symbol:
            self.levelArmRequested.emit(self.alert.symbol, direction, float(level))

    def _emit_level_disarm(self, direction: str, level: float) -> None:
        if self.alert is not None and self.alert.symbol:
            self.levelDisarmRequested.emit(self.alert.symbol, direction, float(level))

    def _on_level_selected(
        self, symbol: str, level_id: str, family: str, price: float
    ) -> None:
        """Record the painted D1 level the trader just clicked."""
        try:
            price = float(price)
        except (TypeError, ValueError):
            return
        self._selected_level = (
            str(symbol or ""),
            str(level_id or ""),
            str(family or ""),
            price,
        )
        self.arm_bar.set_level_alert_available(price > 0)

    def selected_level(self) -> tuple[str, str, str, float] | None:
        """The painted level this pane last saw clicked, for capture rails."""
        return self._selected_level

    def _clear_selected_level(self) -> None:
        self._selected_level = None
        self.arm_bar.set_level_alert_available(False)

    def _emit_level_alert(self, direction: str) -> None:
        """Ask the hosting panel to arm a phone alert at the selected line.

        The CHART is the authority on which line is picked - clicking away
        from every line clears the highlight without emitting anything, so the
        recorded push-side tuple can be stale while ``selected_d1_level()``
        cannot. Nothing is written here; the panel that owns the store writes.
        """
        chosen = self.snapshot.selected_d1_level()
        if chosen is None or self.alert is None or not self.alert.symbol:
            return
        try:
            level = float(chosen.get("price"))
        except (TypeError, ValueError):
            return
        if not level > 0:
            return
        self.levelAlertRequested.emit(self.alert.symbol, direction, level)

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
        any_bounce_armed: bool = False,
        guidance_text: str = "",
        in_focus: bool = False,
        auto_adopted: bool = False,
    ) -> None:
        self.alert = alert
        # Re-point capture, clearing the previous chart's level reference: a
        # stale ref_level_id would attribute this alert's veto to a line the
        # trader clicked on a different symbol.
        self.capture_rail.set_context(
            symbol=alert.symbol,
            side=alert.side if alert.side in ("LONG", "SHORT") else None,
            ref_level_id="",
            ref_level_family="",
        )
        self._refresh_reviewed_badge()
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
        self._set_setup_text_live(
            alert.tag not in (MANUAL_CHART_TAG, FOCUS_REVIEW_TAG) and not is_auto_pick
        )
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
        # Same three buttons, same spots; only the words adapt to what is on
        # the chart. An auto pick's "yes" is the watchlist, its "no" retires
        # the proposal for the day.
        if is_auto_pick:
            self.focus_button.setText("✓ Add to watchlist")
            self.focus_button.setToolTip(
                "Approve this auto pick: it joins the auto-owned slice of the "
                "BounceBot watchlist and gets M5-scanned within a cycle."
            )
            self.remove_today_button.setText("✕ Not today")
            self.remove_today_button.setToolTip(
                "Decline this auto pick - it will not be proposed again today "
                "and the watchlists are untouched."
            )
        elif alert.tag == FOCUS_REVIEW_TAG:
            # Focus walkthrough: the row becomes keep / skip / delete-pick.
            self.focus_button.setText("★ Keep in Focus")
            self.focus_button.setToolTip(
                "Keep this Focus pick as-is and show the next chart."
            )
            self.remove_today_button.setText("✕ Remove from Focus")
            self.remove_today_button.setToolTip(
                "Delete this pick from Focus Picks (every bucket and side; "
                "its focus-injected watchlist entries go with it). The "
                "symbol itself is not muted - alerts still show."
            )
        elif in_focus:
            # The name is ALREADY the trader's, so "Add to ... Focus" is a
            # no-op wearing a verb's clothes. The useful action on a Focus
            # pick's own chart is dropping it - and the only removal here used
            # to be the checked-looking cross toggle, which reads as a status
            # badge, so a pick that had done its move looked unremovable
            # (trader, 2026-08-05: "there's no way of removing this pick").
            self.focus_button.setText("✕ Remove from Focus")
            self.focus_button.setToolTip(
                "Delete this pick from Focus Picks (every bucket and side; "
                "its focus-injected watchlist entries go with it). The symbol "
                "itself is not muted - ordinary alerts still show."
            )
            if auto_adopted:
                # The machine put this name here, so "Not today" can throw it
                # back. The label says so, because the same button on a name
                # the trader typed does something different and quieter - one
                # click must never mean two things with nothing on screen to
                # tell them apart (packet R2, trader decision 2026-08-15).
                self.remove_today_button.setText("✕ Not today - drop pick")
                self.remove_today_button.setToolTip(
                    "Throw this AUTO pick back: its M5 Focus entry goes (and "
                    "the watchlist line it injected), and the name leaves "
                    "today's feed. Only this M5 entry on this side - a swing "
                    "entry, the other side, and anything you added yourself "
                    "are untouched. Recorded as a not-today verdict."
                )
            else:
                self.remove_today_button.setText("✕ Not today")
                self.remove_today_button.setToolTip(
                    "Done with this name for the day: removed from today's Alert "
                    "Center feed and chart queue. Focus membership, the BounceBot "
                    "scanner and the watchlists are untouched."
                )
        else:
            self.focus_button.setToolTip(
                "File this pick into Focus (it gets the heavier alert "
                "treatment) and show the next chart."
            )
            self.remove_today_button.setText("✕ Not today")
            self.remove_today_button.setToolTip(
                "Done with this name for the day: removed from today's Alert "
                "Center feed and chart queue. The BounceBot scanner and "
                "watchlists are untouched."
            )
        # A different symbol's lines are about to be drawn: whatever was
        # picked on the old chart is not on this one.
        self._clear_selected_level()
        self.snapshot.set_symbol(alert.symbol, bot=bot)
        self.snapshot.setVisible(True)
        self.queue_label.setText(f"{queued} waiting" if queued else "queue clear")
        self._set_actions_enabled(True)
        self.set_armed_kinds(armed_kinds)
        self.set_armed_levels(armed_levels)
        self.set_armed_d1_events(armed_d1_events)
        self.set_any_bounce_armed(any_bounce_armed)
        self.set_cross_active(cross_active)
        # The price box seed and the watch-button availability both read the
        # drawn M5 series, which does not exist yet - _on_snapshot_rendered
        # applies them when the bars arrive.
        self._seed_quick_fill = True

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
        self.snapshot.refresh(bot=bot)

    def _on_snapshot_rendered(self, _symbol: str) -> None:
        """Apply everything that needed the bars, once the bars exist."""
        if self.alert is None:
            return
        if self._seed_quick_fill:
            # Seed the price box with the last traded price so the trader
            # adjusts from something real instead of typing from scratch.
            # Once only: a 30s refresh must not overwrite a typed level.
            self._seed_quick_fill = False
            self.arm_bar.apply_quick_fill("last")
        # A session watch can only ever fire off cached M5 bars. Say so on the
        # buttons rather than letting the trader wait on a watch that has
        # nothing to evaluate against. M5 bars can also appear after the alert
        # landed (the scan loop reached the symbol), unlocking the buttons.
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
        self._clear_selected_level()
        self._set_actions_enabled(False)
        self.set_armed_kinds(())
        self.set_armed_d1_events(())
        self.set_any_bounce_armed(False)
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

    def set_any_bounce_armed(self, armed: bool = False) -> None:
        """Reflect this symbol's any-bounce watch on the dock's D1 row."""
        self.arm_bar.set_any_bounce_armed(armed)

    def set_cross_active(self, active: bool) -> None:
        self.cross_focus_button.setText(self._cross_labels[1 if active else 0])
        self.cross_focus_button.setChecked(bool(active))

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self.remove_today_button,
            self.focus_button,
            self.skip_button,
            self.cross_focus_button,
        ):
            button.setEnabled(enabled)
        # The symbol box stays live even with no alert on screen - typing a
        # ticker is how the trader breaks out of an empty queue.
        self.arm_bar.set_enabled_for_symbol(enabled)
