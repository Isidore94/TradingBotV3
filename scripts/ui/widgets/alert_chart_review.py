from __future__ import annotations

"""Automatic D1/M5 visual review surface for Alert Center alerts."""

from typing import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.models.bounce import (
    FOCUS_REVIEW_TAG,
    MANUAL_CHART_TAG,
    BounceAlert,
    is_auto_pick_alert,
)
from ui import theme
from ui.annotations.store import EVENT_LIKE_CLAIM, EVENT_VETO
from ui.widgets.arm_bar import ArmBar
from ui.widgets.empty_state import EmptyState
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
    # A LIKE is finished with the chart but NOT finished with the symbol
    # (R9.2). Separate from `removeTodayRequested` because that verb parks the
    # name for the rest of the day, which a like must never do.
    likeAdvanceRequested = Signal(object)
    focusRequested = Signal(object)
    skipRequested = Signal(object)
    crossFocusToggled = Signal(object)
    watchToggled = Signal(object, str)  # (alert, chart-watch kind)
    d1EventToggled = Signal(object, str)  # (alert, D1 event watch kind)
    anyBounceToggled = Signal(object)  # (alert) - R5 section 4, whole level set
    externalChartRequested = Signal(str)  # symbol - deep-link out for external TA
    # The trader clicking "N hidden (inside yesterday's range) - show". A
    # request to REVEAL, never to change what was recorded: the host still owns
    # every store, and nothing was removed to begin with.
    revealHiddenRequested = Signal()
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
    # (count) - how many watches / D1 events / price levels are armed on the
    # charted symbol. Emitted so a host that took the arm bar onto a tab can
    # keep the armed state legible without the trader opening that tab.
    armedSummaryChanged = Signal(int)
    # (alert) - the trader vetoed this D1 chart but wants the name as a day
    # trade. Two things the host must do, in this order: place it on M5 Focus,
    # then retire it from today's review queue. A request, never a write.
    vetoDayTradeRequested = Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        annotations_path=None,
        dock_arm_bar: bool = True,
        dock_capture_rail: bool = True,
    ) -> None:
        super().__init__(parent)
        self.alert: BounceAlert | None = None
        self._cross_labels = ("Add to D1 Focus", "✓ In D1 Focus")
        # Where each control dock goes is the HOST's decision, not this
        # widget's, and the two are decided SEPARATELY because they cost very
        # different amounts of the thing the pane is short of. Measured at the
        # desk column's 420px: the arm bar is 131px, the capture rail is 697px.
        #
        # So the Alert Center keeps the arm bar welded under the chart where
        # the trader wants their M5/D1 hotbuttons and the type-a-ticker box
        # (trader, 2026-08-20: "I also need my m5 and D1 alert hotbuttons back
        # on the bottom of the visual chart"), and sends only the rail to a
        # tab. That is 84% of the reclaimed height kept and the fast controls
        # back within reach - the earlier all-or-nothing flag could not
        # express it.
        self._dock_arm_bar = bool(dock_arm_bar)
        self._dock_capture_rail = bool(dock_capture_rail)
        self._armed_watch_count = 0
        self._armed_level_count = 0
        self._armed_d1_event_count = 0
        self._any_bounce_armed = False

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
        self.arm_bar.externalChartRequested.connect(self.externalChartRequested)
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

        self.capture_rail = CaptureRail(
            annotations_path=annotations_path,
            # Undocked, the rail sits on a tab page that is hidden most of the
            # time, and a shortcut bound inside a hidden page never fires. The
            # host rebinds `action_shortcuts()` at a scope the trader reaches.
            bind_action_shortcuts=self._dock_capture_rail,
        )
        self.capture_rail.captured.connect(self._on_captured)
        self.capture_rail.vetoDayTradeRequested.connect(self._on_veto_day_trade)
        # A day-trade pass attaches the M5 bars this pane already drew, so the
        # chart can be read back as it stood. Memory-only and read at click
        # time; a pane with nothing cached simply writes the timestamp.
        self.capture_rail.set_m5_bars_provider(self.snapshot.cached_m5_bars)
        self.snapshot.d1LevelSelected.connect(self._on_capture_level_selected)

        # R4 section 5, on the surface the trader stares at most.
        self.reviewed_badge = QLabel("")
        self.reviewed_badge.setObjectName("reviewedTodayBadge")
        self.reviewed_badge.setStyleSheet(
            f"color: {theme.color('caution')}; font-weight: 600;"
        )
        # Trader rule 2026-08-19: the names actually moving are the ones beyond
        # yesterday's extreme. Same badge idiom as the Focus chips' BOUNCE/RRS
        # flag - a short uppercase word in the accent colour - rather than a new
        # visual language for one more piece of state.
        self.mover_badge = QLabel("")
        self.mover_badge.setObjectName("moverBadge")
        self.mover_badge.setVisible(False)
        # The withheld count. It is a BUTTON because it is an action: one click
        # shows the inside-range names for the rest of the session. It states a
        # number so "nothing is queued" can never be confused with "everything
        # was filtered away".
        self.hidden_button = QPushButton("")
        self.hidden_button.setObjectName("HiddenReviewsButton")
        self.hidden_button.setFlat(True)
        self.hidden_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hidden_button.setVisible(False)
        self.hidden_button.setToolTip(
            "These names are inside yesterday's range, so the review queue is "
            "holding them back. Nothing was deleted - they are still in the "
            "feed, the history and every store. One click shows them for the "
            "rest of today."
        )
        self.hidden_button.clicked.connect(self.revealHiddenRequested)

        # The arm bar's own "Nothing armed" line goes with it when the host
        # takes the bar onto a tab, so the state it carried has to survive on
        # the row that never hides. It is a COUNT, not the inventory: the
        # question a glance asks is "is anything live on this name", and the
        # tab (and the Armed inventory under it) answers the rest.
        self.armed_summary = QLabel("")
        self.armed_summary.setObjectName("MutedLabel")
        self.armed_summary.setToolTip(
            "Session watches, D1 event alerts and price levels armed on this "
            "symbol. Open the Armed tab to add or cancel one."
        )
        # Only earns its pixels when the arm bar is NOT under the chart. With
        # the bar docked its own "Nothing armed" line and chips are right
        # there, and two copies of one state is noise.
        self.armed_summary.setVisible(not self._dock_arm_bar)

        # What stands in the chart's slot when there is no chart.
        #
        # This is not decoration - it is the fix for a measured layout fault.
        # The snapshot carries this pane's only expanding stretch, so HIDING it
        # left Qt with four Preferred widgets and a column of slack, which it
        # smeared equally across all of them: at 2000x1900 the one-line title
        # got 346px, the "waiting" line got 346px, the arm bar got 346px and
        # the verb row got 346px, for ~170px of actual content. That is
        # ~1240px of a 4K screen spent on label padding, and it is the state
        # the desk sits in whenever the review queue is empty.
        #
        # An expanding placeholder keeps a stretch item in the layout at all
        # times, so the slack collects in ONE place that can say something
        # useful instead of being distributed into whitespace.
        self.empty_state = EmptyState(
            "No chart up",
            "The review queue is clear. Type a ticker in the box below to "
            "chart anything on demand - it does not have to have alerted - "
            "or wait for the next scanner alert to put one here.",
        )
        self.empty_state.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        buttons = QHBoxLayout()
        buttons.addWidget(self.reviewed_badge)
        buttons.addWidget(self.mover_badge)
        buttons.addWidget(self.focus_button)
        buttons.addWidget(self.skip_button)
        buttons.addWidget(self.remove_today_button)
        buttons.addWidget(self.cross_focus_button)
        buttons.addStretch(1)
        buttons.addWidget(self.hidden_button)
        buttons.addWidget(self.armed_summary)
        buttons.addWidget(self.queue_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        # Everything that is not the chart is pinned to its size hint. A
        # QLabel defaults to Preferred vertically, which means "I will happily
        # take more" - and more is exactly what it got every time the chart
        # was hidden.
        for fixed in (self.title, self.alert_text, self.guidance_label, self.arm_bar):
            fixed.setSizePolicy(
                fixed.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Maximum
            )
        layout.addWidget(self.title)
        layout.addWidget(self.alert_text)
        layout.addWidget(self.guidance_label)
        layout.addWidget(self.snapshot, 1)
        layout.addWidget(self.empty_state, 1)
        # Detached, not destroyed: this widget keeps the Python references (and
        # every signal already wired through them), and the host calls
        # addWidget to adopt what it took. Without the explicit unparenting an
        # undocked control would be a laid-out-less child painting over the
        # charts until the host got round to it.
        if self._dock_arm_bar:
            layout.addWidget(self.arm_bar)
        else:
            self.arm_bar.setParent(None)
        if self._dock_capture_rail:
            layout.addWidget(self.capture_rail)
        else:
            self.capture_rail.setParent(None)
        layout.addLayout(buttons)
        self._refresh_armed_summary()
        # Open in the empty state rather than falling into it on the first
        # clear(): a pane built with no alert had the placeholder hidden AND
        # the chart hidden, i.e. no stretch item at all.
        self._show_chart(False)
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

    def _on_captured(self, event_type: str, _row: dict) -> None:
        """Capture is a decision, so the badge updates without a re-chart.

        A VETO and a LIKE both MOVE ON, by trader rule (2026-08-20): "when I
        click veto it should just disappear as 'not for today'" and "when I
        pick a like and claim setup reason, we should just move onto the next
        chart". Neither is an annotation about a chart still being read.

        They move on by DIFFERENT routes (R9.2, 2026-08-22). A veto keeps the
        "Not today" verb, which retires the chart and parks the symbol. A like
        takes an advance-only route, because parking it was measurably wrong:
        over 2026-07-24..08-21, 40 of 52 likes put the symbol on the day's
        ignore list, which also silenced its `d1EventRecorded` - so on an AWAY
        day, liking a chart quietly dropped that name from the hourly D1 phone
        push. Liking a setup is the opposite of being done with the symbol.

        A NOTE deliberately does not. It is written ABOUT the chart in front
        of the trader, and a rail that skipped to the next one would make
        every note cost them the thing they were writing it about.

        The annotation is written first and the queue move is never
        conditional on it - `_record` has already returned by the time we get
        here, so a retired chart always has a row behind it.

        The day-trade veto does NOT come through here: it needs the Focus
        placement to happen before the chart is retired, so it has its own
        route (`_on_veto_day_trade`) and this deliberately ignores it.
        """
        self._refresh_reviewed_badge()
        if self.alert is None:
            return
        if event_type == EVENT_VETO and not self.capture_rail.veto_keeps_chart():
            self.removeTodayRequested.emit(self.alert)
        elif event_type == EVENT_LIKE_CLAIM:
            self.likeAdvanceRequested.emit(self.alert)

    def _on_veto_day_trade(self, _row: dict) -> None:
        """Vetoed the D1, keeping the name for an M5 trade.

        The plain-veto auto-advance is suppressed for this one commit: the
        host has to place the name on M5 Focus BEFORE the alert is retired
        from the queue, because retiring it is what drops the object both
        steps need. The host does both, in that order.
        """
        if self.alert is not None:
            self.vetoDayTradeRequested.emit(self.alert)

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
        mover_state: str = "",
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
        self._show_chart(True)
        self.queue_label.setText(f"{queued} waiting" if queued else "queue clear")
        self._set_actions_enabled(True)
        self.set_armed_kinds(armed_kinds)
        self.set_armed_levels(armed_levels)
        self.set_armed_d1_events(armed_d1_events)
        self.set_any_bounce_armed(any_bounce_armed)
        self.set_mover_state(mover_state)
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
        self._show_chart(False)
        self.queue_label.setText("")
        self._clear_selected_level()
        self._set_actions_enabled(False)
        self.set_armed_kinds(())
        self.set_armed_levels(())
        self.set_armed_d1_events(())
        self.set_any_bounce_armed(False)
        self.set_mover_state("")
        self.set_cross_active(False)

    def _show_chart(self, charted: bool) -> None:
        """Exactly one of the chart and the placeholder is ever in the layout.

        They share the pane's only expanding slot, so swapping them keeps a
        stretch item present at all times - which is what stops the slack
        being smeared into the labels and the arm bar (see ``empty_state``).
        The title and the setup line have nothing to say with no chart up, and
        the placeholder says it better, so they stand down together.
        """
        charted = bool(charted)
        self.snapshot.setVisible(charted)
        self.empty_state.setVisible(not charted)
        self.title.setVisible(charted)
        self.alert_text.setVisible(charted)

    def set_queued_count(self, count: int) -> None:
        self.queue_label.setText(f"{count} waiting" if count else "queue clear")

    def set_armed_kinds(self, kinds: Iterable[str]) -> None:
        """Reflect this symbol's armed watches; buttons stay clickable so a
        second click disarms."""
        kinds = list(kinds or ())
        self._armed_watch_count = len(set(kinds))
        self.arm_bar.set_armed_kinds(kinds)
        self._refresh_armed_summary()

    def set_armed_levels(self, levels: Iterable = ()) -> None:
        """Show this symbol's armed price levels as dismissable chips."""
        levels = list(levels or ())
        self._armed_level_count = len(levels)
        self.arm_bar.set_armed_levels(levels)
        self._refresh_armed_summary()

    def set_armed_d1_events(self, kinds: Iterable[str] = ()) -> None:
        """Reflect this symbol's armed D1 event watches on the dock's D1 row."""
        kinds = list(kinds or ())
        self._armed_d1_event_count = len(set(kinds))
        self.arm_bar.set_armed_d1_events(kinds)
        self._refresh_armed_summary()

    def armed_count(self) -> int:
        """Everything armed on the charted symbol, as one number."""
        return (
            self._armed_watch_count
            + self._armed_level_count
            + self._armed_d1_event_count
            + (1 if self._any_bounce_armed else 0)
        )

    def _refresh_armed_summary(self) -> None:
        """Keep the always-visible armed line (and the host's tab) honest."""
        count = self.armed_count()
        self.armed_summary.setText(f"⚡ {count} armed" if count else "Nothing armed")
        color = theme.color("favorite" if count else "text_muted")
        weight = 700 if count else 400
        self.armed_summary.setStyleSheet(f"color: {color}; font-weight: {weight};")
        self.armedSummaryChanged.emit(count)

    def set_mover_state(self, state: str = "") -> None:
        """Say which of the three answers this chart is showing.

        A verified break is the flag the trader asked for. An UNMEASURED name
        is labelled rather than dressed up either way: it is on the chart
        BECAUSE it could not be measured (missing data is uncertainty, never
        confirmation), and the tag is what stops that from reading as a
        breakout. A name verified inside the range only appears once the
        trader has revealed the hidden ones, and it says so.
        """
        text, color = {
            "open": ("MOVING", theme.color("favorite")),
            "unknown": ("unmeasured", theme.color("text_muted")),
            "closed": ("inside range", theme.color("text_muted")),
            # Trader rule 2026-08-27: a long under session VWAP (a short over
            # it) is hidden the same way, and says so once revealed.
            "wrong_side_vwap": ("wrong side of VWAP", theme.color("text_muted")),
            # Trader rule 3, same day: a D1 long under its SMA200 or a D1
            # short over its SMA50 is hidden too.
            "wrong_side_sma": ("wrong side of SMA", theme.color("text_muted")),
        }.get(str(state or "").strip().lower(), ("", ""))
        self.mover_badge.setText(text)
        self.mover_badge.setVisible(bool(text))
        if text:
            weight = 700 if text == "MOVING" else 500
            self.mover_badge.setStyleSheet(f"color: {color}; font-weight: {weight};")

    def set_hidden_count(self, count: int = 0) -> None:
        """The honest line about what the movers-only filter is holding back."""
        count = max(0, int(count or 0))
        self.hidden_button.setVisible(count > 0)
        if count:
            self.hidden_button.setText(
                f"{count} hidden (inside yesterday's range / wrong side of VWAP or SMA) - show"
            )

    def set_any_bounce_armed(self, armed: bool = False) -> None:
        """Reflect this symbol's any-bounce watch on the dock's D1 row."""
        self._any_bounce_armed = bool(armed)
        self.arm_bar.set_any_bounce_armed(armed)
        self._refresh_armed_summary()

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
