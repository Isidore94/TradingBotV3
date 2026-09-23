from __future__ import annotations

"""Automatic D1/M5 visual review surface for Alert Center alerts."""

from typing import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.models.bounce import (
    FOCUS_REVIEW_TAG,
    MANUAL_CHART_TAG,
    BounceAlert,
    capture_timeframe,
    is_auto_pick_alert,
)
from ui.models.alert_presentation import classify_alert
from ui import theme
from ui.annotations.store import (
    EVENT_LIKE_CLAIM,
    EVENT_VETO,
    LIKE_MODE_QUICK,
    SURFACE_CHART_REVIEW,
    like_mode_of,
)
from ui.widgets.arm_bar import ArmBar
from ui.widgets.empty_state import EmptyState
from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

_NO_M5_WATCH_REASON = (
    "No cached M5 bars for this symbol yet - arming still works: BounceBot "
    "folds armed names into its M5 scan set, so bars land within a scan "
    "cycle and the watch starts evaluating then."
)


class _MentorPopup(QDialog):
    """A normal modeless tool window whose close paths have one meaning."""

    dismissed = Signal()

    SIZE_SETTING = "qt_trade_mentor_popup_size_v1"
    DEFAULT_SIZE = (900, 820)
    MINIMUM_SIZE = (760, 720)

    def restore_saved_size(self) -> None:
        """Restore a useful size, bounded by the screen that owns the window."""

        from project_paths import get_local_setting

        raw = get_local_setting(self.SIZE_SETTING, list(self.DEFAULT_SIZE))
        try:
            width, height = int(raw[0]), int(raw[1])
        except (TypeError, ValueError, IndexError):
            width, height = self.DEFAULT_SIZE
        screen = self.screen()
        available = screen.availableGeometry() if screen is not None else None
        if available is not None:
            width = min(width, max(self.MINIMUM_SIZE[0], available.width() - 48))
            height = min(height, max(self.MINIMUM_SIZE[1], available.height() - 48))
        self.resize(max(self.MINIMUM_SIZE[0], width), max(self.MINIMUM_SIZE[1], height))

    def _save_size(self) -> None:
        from project_paths import save_local_setting

        try:
            save_local_setting(self.SIZE_SETTING, [self.width(), self.height()])
        except OSError:
            pass

    def keyPressEvent(self, event):  # noqa: N802 - Qt override
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):  # noqa: N802 - Qt override
        self._save_size()
        self.dismissed.emit()
        event.accept()


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

    All three advance the review queue, and so does a VETO from the capture
    rail. A LIKE does NOT (packet T1, trader 2026-09-04: the chart stays so
    the trader can arm their alerts on it), and neither does a note or a
    day-trade pass. Everything on the arm dock is a TOGGLE that leaves the
    chart in place: the cross-focus button (M5 pick -> pin into the D1 Focus
    feed; swing pick -> add to the M5 Focus day-trade list) and the one-shot
    chart watches (click again to disarm).
    """

    # The "✕ Not today" BUTTON's signal, and since packet T1 (2026-09-04) only
    # that button's. The verb behind it writes an UNCODED veto row and then
    # opens a note box, which is exactly what the trader asked to be rid of on
    # the capture window - so the rail's coded veto no longer borrows it.
    removeTodayRequested = Signal(object)
    # (alert) - the trader vetoed this chart FROM THE CAPTURE RAIL, with a
    # reason code and the why already written. The host retires the chart and
    # parks the symbol exactly as "Not today" does, and writes NO second row
    # and NO note box: the capture window IS the why (trader, 2026-09-04:
    # "when i double tap something in the capture window ... i shouldnt get a
    # pop up note box").
    vetoRetireRequested = Signal(object)
    # (alert, saved veto row).  The host owns watches, so a saved annotation
    # asks it to make the one narrow exception before the ordinary veto
    # retirement runs.
    savedVeto = Signal(object, object)
    # (alert) - a QUICK like was recorded. A REPORT, not a request: since
    # packet T1 (trader, 2026-09-04: "i still need time to enter alerts etc.")
    # the chart STAYS for this mode, so it asks the host for no movement.
    likeRecorded = Signal(object)
    # (alert) - a CLAIMED like: move to the next chart. Packet T2 (trader,
    # 2026-09-04, second pass: "double clicking that box should advance the
    # chart"). Two signals rather than one with a flag, because the host's two
    # answers really are different verbs and a flag would be read wrong once.
    likeAdvanceRequested = Signal(object)
    # (alert, claim_row) - a CLAIMED like on a D1 chart was SAVED as a pick.
    # Packet D1C-A (trader, 2026-09-14: *"A successful D1 'Like and claim' must
    # add the pick to Master AVWAP Setups immediately ... Save and confirm the
    # pick before removing its D1 item from Visual Chart Review"*). Emitted only
    # after the store has returned a row, so the host retires a chart that has a
    # pick behind it and never one that does not. A DUPLICATE claim comes this
    # way too - the pick exists, which is what the signal means.
    claimPlaced = Signal(object, object)
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
    scanReviewViewToggled = Signal()
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
        claim_writer=None,
        mentor_context_service=None,
    ) -> None:
        super().__init__(parent)
        self.alert: BounceAlert | None = None
        # Packet D1C-A. The ONE writer a claimed D1 like reaches. The HOST owns
        # the store and binds this, exactly as it owns the review-events and
        # parked-symbols files; None means this pane has no store behind it and
        # a claimed like takes the pre-packet route (advance, place nothing).
        # There is deliberately no default writer here: a widget that reached
        # for `claimed_picks.record_claim` itself would write the live store
        # from a pane nobody gave a store to.
        self._claim_writer = claim_writer
        #: The host's answer to "is this an M5 review alert?", handed in with
        #: the alert. The widget never imports the panel, and the horizon must
        #: not be guessed from the chart's own timeframe.
        self._alert_is_m5_review = False
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

        # P9's quick like, ON THE CHART (trader, 2026-09-02: "ensure we also
        # just have a button on the visual chart as well"). APPENDED to the verb
        # row rather than inserted among it: every existing button keeps its
        # spot, which is what that row's muscle-memory rule is for. It is still
        # ONE row - CLAUDE.md's limit between the charts and the tab strip.
        #
        # It calls the rail directly, like the rail's own buttons do, rather
        # than emitting a request the panel has to route: the capture rail owns
        # capture, and a second route to the same write is a second thing to
        # keep in step. The chart retires afterwards through `captured` ->
        # `_on_captured`, exactly as a claimed like does.
        self.quick_like_button = QPushButton("♥ Like")
        self.quick_like_button.setToolTip(
            "Something about this chart was good. Opens a box for an optional "
            "note; no setup to pick and no reason required. Alt+L does the same "
            "with no box. Nothing is added to Focus or any watchlist, and the "
            "chart stays up so you can arm your alerts."
        )
        # Late-bound: the rail is constructed further down this __init__, and
        # the same lambda idiom the other verb buttons use resolves at click
        # time rather than at wiring time.
        self.quick_like_button.clicked.connect(
            lambda: self.capture_rail.prompt_quick_like()
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
        # R4 A5: THIS pane is the screen, not the rail. `set_scan_context` has
        # carried a `surface` override since P10 B1 and nothing ever called it,
        # so every verdict the trader passed on a review chart filed as `rail` -
        # and a rollup asking "is the trader a better judge from the setups
        # table or from the chart?" could not tell the two apart. The rail's own
        # default stays `rail`, which is honest for a host that IS the rail.
        self.capture_rail.set_scan_context(surface=SURFACE_CHART_REVIEW)
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

        # AR-2B shares the existing verb row: ordinary D1 scan ideas are a
        # view choice, never a new alert state.  The host owns the backing
        # queue and tells this compact switch what is currently held.
        self.scan_view_button = QPushButton("")
        self.scan_view_button.setObjectName("ScanReviewViewButton")
        self.scan_view_button.setFlat(True)
        self.scan_view_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scan_view_button.setVisible(False)
        self.scan_view_button.clicked.connect(self.scanReviewViewToggled)

        # Packet D1C-A's count, beside the withheld one. A claimed chart is
        # ANSWERED, not withheld, so this is a muted label and not a button:
        # there is nothing to reveal, and the row it refers to is in the setups
        # table. It still states a number, so "the queue went quiet" can never
        # be confused with "the desk stopped charting".
        self.claimed_skipped_label = QLabel("")
        self.claimed_skipped_label.setObjectName("MutedLabel")
        self.claimed_skipped_label.setVisible(False)
        self.claimed_skipped_label.setToolTip(
            "D1 charts skipped because you already liked and claimed that "
            "setup. Nothing was deleted or muted - the pick is in Master "
            "AVWAP Setups, its alerts still fire, and its M5 alerts still "
            "reach the list on the left. Drop the claim to see the chart again."
        )

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

        # The Mentor is a small reusable modeless window, never a row under the
        # chart.  The arm bar therefore keeps its fixed home and scheduled
        # prompts cannot steal chart height.
        from ui.widgets.trade_mentor_card import TradeMentorCard

        self.mentor_popup = _MentorPopup(
            self,
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint,
        )
        self.mentor_popup.setObjectName("TradeMentorPopup")
        self.mentor_popup.setWindowTitle("Trade Mentor")
        self.mentor_popup.setModal(False)
        self.mentor_popup.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.mentor_popup.setMinimumSize(*_MentorPopup.MINIMUM_SIZE)
        self.mentor_popup.restore_saved_size()
        popup_layout = QVBoxLayout(self.mentor_popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        self.mentor_card = TradeMentorCard(
            self.mentor_popup, context_service=mentor_context_service
        )
        self.mentor_scroll = QScrollArea(self.mentor_popup)
        self.mentor_scroll.setObjectName("TradeMentorScroll")
        self.mentor_scroll.setWidgetResizable(True)
        self.mentor_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.mentor_scroll.setWidget(self.mentor_card)
        popup_layout.addWidget(self.mentor_scroll)
        self.mentor_card.setVisible(False)
        self.mentor_popup.dismissed.connect(self._dismiss_mentor_popup)
        self.mentor_card.answered.connect(lambda _slot_id: self._hide_mentor_popup_window())
        self.mentor_card.skipped.connect(lambda _record: self._hide_mentor_popup_window())
        # Always reachable, whether or not anything is due: "I want to write a
        # read now" must never require waiting for the top of an hour. It sits
        # in the existing verb row rather than adding a second one - CLAUDE.md
        # allows exactly one row between the charts and the tab strip.
        self.give_a_read_button = QPushButton("Give a read")
        self.give_a_read_button.setToolTip(
            "Write a market read right now and file it in the Market Journal. "
            "Always available - it does not need a scheduled prompt."
        )
        self.give_a_read_button.clicked.connect(self._on_give_a_read)

        buttons = QHBoxLayout()
        buttons.addWidget(self.reviewed_badge)
        buttons.addWidget(self.mover_badge)
        buttons.addWidget(self.focus_button)
        buttons.addWidget(self.skip_button)
        buttons.addWidget(self.remove_today_button)
        buttons.addWidget(self.cross_focus_button)
        buttons.addWidget(self.quick_like_button)
        buttons.addWidget(self.give_a_read_button)
        buttons.addStretch(1)
        buttons.addWidget(self.hidden_button)
        buttons.addWidget(self.scan_view_button)
        buttons.addWidget(self.claimed_skipped_label)
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

    def _on_captured(self, event_type: str, row: dict) -> None:
        """Capture is a decision, so the badge updates without a re-chart.

        A VETO moves on, by trader rule (2026-08-20): "when I click veto it
        should just disappear as 'not for today'". It does so through its OWN
        verb (`vetoRetireRequested`, packet T1) rather than the "✕ Not today"
        button's: that button writes a second, UNCODED veto row and opens a
        note box, and the trader cut both from the capture window
        (2026-09-04: "the point of the capture window is to quickly enter
        'WHY' I like or dislike something"). The retirement itself - the
        review event, the parking, the advance - is identical.

        A LIKE takes ONE of two routes, decided by its MODE (packet T2, trader
        2026-09-04 second pass: "for the 'like and claim' part of the capture
        tab, a double click of any of the setups there should be sufficient ...
        and then double clicking that box should advance the chart"):

        * a CLAIMED like has said everything it has to say - the setup is
          named - so it asks for the advance (`likeAdvanceRequested`);
        * a QUICK like has named nothing and the trader is still working on
          the chart, so it is merely reported (`likeRecorded`) and the chart
          stays (packet T1, 2026-09-04: "the 'like' button in the visual chart
          review should NOT advance the char to the next page because i still
          need time to enter alerts etc.").

        Absence of `like_mode` reads as CLAIMED - the P9 rule, because a claim
        was required until then; `like_mode_of` owns that answer so this is not
        a second copy of it.

        NEITHER route is the "Not today" verb (R9.2, 2026-08-22): parking a
        liked name was measurably wrong - over 2026-07-24..08-21, 40 of 52
        likes put the symbol on the day's ignore list, which also silenced its
        `d1EventRecorded`, so on an AWAY day liking a chart quietly dropped that
        name from the hourly D1 phone push. An advance is not a retirement:
        nothing is parked, nothing is dropped from Focus, and the symbol's other
        queued alerts keep their places.

        A NOTE deliberately does not move on either. It is written ABOUT the
        chart in front of the trader, and a rail that skipped to the next one
        would make every note cost them the thing they were writing it about.

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
        if event_type == EVENT_VETO:
            # A delayed rail delivery must never turn a row captured on AAPL
            # into a retirement of the next chart.  Both identity dimensions
            # are on every saved veto row, and a missing/mismatched one is no
            # request at all.
            row_symbol = str(row.get("symbol") or "").strip().upper()
            row_side = str(row.get("side") or "").strip().upper()
            alert_symbol = str(self.alert.symbol or "").strip().upper()
            alert_side = str(self.alert.side or "").strip().upper()
            if row_symbol != alert_symbol or row_side != alert_side:
                return
            self.savedVeto.emit(self.alert, dict(row))
            if not self.capture_rail.veto_keeps_chart():
                self.vetoRetireRequested.emit(self.alert)
        elif event_type == EVENT_LIKE_CLAIM:
            if like_mode_of(row) == LIKE_MODE_QUICK:
                self.likeRecorded.emit(self.alert)
            else:
                self._route_claimed_like(row)

    def _route_claimed_like(self, row: dict) -> None:
        """A CLAIMED like, routed by the HORIZON of the thesis it names.

        Packet D1C-A (trader, 2026-09-14). The desk has two sides and they get
        different answers:

        * **d1** - the right side of the desk. The pick is SAVED first, and
          only a saved pick retires the chart (`claimPlaced`). A store that
          could not be written keeps the chart and says so: the like still
          stands (its annotation row was written before we got here and is
          never conditional on the placement), but the trader is not going to
          lose a chart to a pick that does not exist.
        * **m5** - the left side. Unchanged: the claimed like advances and
          places nothing, because an intraday entry is not a swing pick.
        * **""** - the registry cannot name the claim (`none_of_these`, or an
          id it has never heard of). Unchanged route, plus a line that says
          why nothing was placed. A horizon nobody can resolve is not a pick
          anybody can rank.
        """
        import claimed_picks

        alert = self.alert
        if self._claim_writer is None:
            # This pane has no claim store behind it - a bare widget, or a host
            # that does not own one. It cannot place a pick, so it behaves
            # exactly as it did before packet D1C-A: the claimed like advances
            # and places nothing. Deliberately NOT the failure route: nothing
            # was lost, because nothing was ever going to be written, and a
            # pane that reached for the DEFAULT store would write the live
            # `claimed_picks.jsonl` from a widget that owns no store at all.
            self.likeAdvanceRequested.emit(alert)
            return
        claimed_setup_id = str(row.get("claimed_setup_id") or "").strip()
        horizon = claimed_picks.claim_horizon(
            alert, claimed_setup_id, is_m5_review=self._alert_is_m5_review
        )
        if horizon != claimed_picks.HORIZON_D1:
            if not horizon:
                self.capture_rail.set_capture_status(
                    "claimed; horizon unknown - not placed in Setups", ok=False
                )
            self.likeAdvanceRequested.emit(alert)
            return
        claim_row = self._place_claim(alert, row, claimed_setup_id)
        if claim_row is None:
            self.capture_rail.set_capture_status(
                "NOT PLACED - claimed_picks.jsonl could not be written; chart kept",
                ok=False,
            )
            self.likeRecorded.emit(alert)
            return
        self.claimPlaced.emit(alert, claim_row)

    def _place_claim(self, alert, row: dict, claimed_setup_id: str):
        """Write the claim. Returns the stored row, or None when it failed.

        Every failure is None, including an unexpected one: the caller's answer
        to "it did not save" is to keep the chart and say so, which is the
        right answer whatever went wrong.
        """
        import claimed_picks

        writer = self._claim_writer
        payload = getattr(alert, "payload", None)
        try:
            return writer(
                symbol=alert.symbol,
                # The LIKE ROW's side, not the alert's. They agree whenever an
                # alert put the chart up (`set_alert` points the rail at it),
                # and where they differ the row is right: a typed symbol has no
                # side at all, and the rail's selector is the trader's own
                # answer for the chart in front of them.
                side=str(row.get("side") or "").strip() or alert.side,
                horizon=claimed_picks.HORIZON_D1,
                claimed_setup_id=claimed_setup_id,
                source=claimed_picks.claim_source(
                    row.get("surface") or SURFACE_CHART_REVIEW, alert
                ),
                # The like row and the claim row join on this: two stores, one
                # gesture, and a reader has to be able to walk from the pick
                # back to the words the trader typed about it.
                annotation_ref=str(row.get("event_id") or row.get("created_at") or ""),
                known_at_claim=claimed_picks.known_at_claim_from_payload(payload),
                note=str(row.get("note") or ""),
            )
        except Exception:  # noqa: BLE001 - a failed placement keeps the chart
            return None

    def _on_veto_day_trade(self, row: dict) -> None:
        """Vetoed the D1, keeping the name for an M5 trade.

        The plain veto's own retire verb is suppressed for this one commit:
        the host has to place the name on M5 Focus BEFORE the alert is retired
        from the queue, because retiring it is what drops the object both
        steps need. The host does both, in that order - and since packet T1
        the retire it performs is the same box-free one a plain veto gets
        (lead ruling 2026-09-04: the trader's "either veto or like+claim ...
        no pop up note box" covers this verb too).
        """
        if self.alert is not None:
            row_symbol = str(row.get("symbol") or "").strip().upper()
            row_side = str(row.get("side") or "").strip().upper()
            alert_symbol = str(self.alert.symbol or "").strip().upper()
            alert_side = str(self.alert.side or "").strip().upper()
            if row_symbol != alert_symbol or row_side != alert_side:
                return
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

    # -- Trade Mentor (WISHLIST 10J) --------------------------------------
    def show_mentor_slot(self, slot, previous=None) -> None:
        """Put a due prompt in the modeless popup. Never steals focus.

        A new hour REPLACES whatever card was there; the card itself stashes any
        half-typed draft on the way out. Failure here is swallowed: a prompt is
        an interruption, and an interruption that throws would take the chart
        with it.
        """
        try:
            self.mentor_card.show_slot(slot, previous=previous)
            self.mentor_popup.show()
        except Exception:  # noqa: BLE001 - a prompt never costs the chart
            import logging

            logging.debug("Trade Mentor card could not be shown.", exc_info=True)

    def hide_mentor_popup(self) -> None:
        try:
            self.mentor_card.hide_card()
            self._hide_mentor_popup_window()
        except Exception:  # noqa: BLE001
            import logging

            logging.debug("Trade Mentor card could not be hidden.", exc_info=True)

    def _hide_mentor_popup_window(self) -> None:
        self.mentor_popup._save_size()
        self.mentor_popup.hide()

    # Existing scheduler callers use this name.  Both paths hide the same
    # reusable popup; expiry and Pause have already recorded their own state.
    hide_mentor_card = hide_mentor_popup

    def _dismiss_mentor_popup(self) -> None:
        """Escape and the title-bar X mean one explicit trader skip."""
        try:
            self.mentor_card.skip()
        finally:
            self.mentor_popup.hide()

    def _on_give_a_read(self) -> None:
        try:
            self.mentor_card.give_a_read()
            self.mentor_popup.show()
        except Exception:  # noqa: BLE001
            import logging

            logging.debug("Manual read could not be opened.", exc_info=True)

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
        is_m5_review: bool = False,
    ) -> None:
        self.alert = alert
        # The HOST's answer to "does this alert belong to the M5 bar?", handed
        # in rather than re-derived: `_is_m5_review_alert` lives on the panel
        # and this widget never imports the panel (packet D1C-A item 2).
        self._alert_is_m5_review = bool(is_m5_review)
        # Re-point capture, clearing the previous chart's level reference: a
        # stale ref_level_id would attribute this alert's veto to a line the
        # trader clicked on a different symbol.
        self.capture_rail.set_context(
            symbol=alert.symbol,
            side=alert.side if alert.side in ("LONG", "SHORT") else None,
            # Packet D1C-A, the correctness fix behind the trader's *"do not
            # rely on a stale chart timeframe"*: the rail is constructed on
            # "D1" and this call never re-pointed it, so every chart in the
            # queue left it saying D1 - and `_record_like` reads it to decide
            # whether to attach the M5 sidecar. An M5 chart's like was
            # therefore filed without the bars it was made on. The HORIZON does
            # not come from here (see `claimed_picks.claim_horizon`); this is
            # the sidecar's read, and it has to be the chart's own answer.
            #
            # NORMALISED, and never blank (reviewer blocker, 2026-09-14).
            # `set_context` is `if timeframe:`, so handing it the alert's raw
            # value left a typed symbol - whose alert names no timeframe - on
            # the PREVIOUS chart's answer, filing a daily look as `M5` with an
            # M5 sidecar behind it. And a live `from_callback` alert says
            # `"5m"`, which upper-cases to `"5M"` and misses `_record_like`'s
            # `== "M5"` compare, so the one path the attachment exists for was
            # the one losing its bars. One seam owns both: `capture_timeframe`,
            # beside `BounceAlert` because the alert's spelling is its own.
            timeframe=capture_timeframe(alert.timeframe),
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
        # Keep the established live/muted marker for existing callers, then
        # classify the exact reason without changing routing or membership.
        is_auto_pick = is_auto_pick_alert(alert)
        self._set_setup_text_live(
            alert.tag not in (MANUAL_CHART_TAG, FOCUS_REVIEW_TAG) and not is_auto_pick
        )
        self._set_alert_reason_tone(classify_alert(alert, in_focus=in_focus).reason_tone)
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

    def _set_alert_reason_tone(self, tone: str) -> None:
        """Repolish only the reason label when its live meaning changes."""
        tone = str(tone or "muted")
        if self.alert_text.property("alertReasonTone") == tone:
            return
        self.alert_text.setProperty("alertReasonTone", tone)
        style = self.alert_text.style()
        style.unpolish(self.alert_text)
        style.polish(self.alert_text)
        self.alert_text.update()

    def clear(self) -> None:
        self.alert = None
        self.title.setText("Visual Alert Review")
        self.alert_text.setText("Waiting for the next ticker alert.")
        self._set_setup_text_live(False)
        self._set_alert_reason_tone("muted")
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
            # 2026-09-23: within 1 ATR of an SMA in its path or a D1 trendline.
            "at_wall": ("at wall", theme.color("text_muted")),
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
                f"{count} hidden (inside yesterday's range / wrong side of VWAP or SMA"
                " / at a wall) - show"
            )

    def set_scan_review_view(self, *, show_all: bool, hidden_count: int = 0) -> None:
        """Reflect the AR-2B D1 scan view without owning its queue."""
        hidden_count = max(0, int(hidden_count or 0))
        self.scan_view_button.setVisible(bool(hidden_count) or bool(show_all))
        self.scan_view_button.setText(
            "My alerts" if show_all else f"Show all ({hidden_count})"
        )

    def set_claimed_skipped_count(self, count: int = 0) -> None:
        """The honest line about the repeat D1 charts a claim is holding back.

        Packet D1C-A, built the way `set_hidden_count` above is built and for
        the same reason: the review pane never goes quiet without saying why. A
        LABEL rather than a button, because there is nothing to reveal - the
        chart is not withheld, the thesis is answered, and the pick is sitting
        in the setups table where the trader put it. Nothing is deleted, muted
        or written to `review_policy.json`.
        """
        count = max(0, int(count or 0))
        self.claimed_skipped_label.setVisible(count > 0)
        if count:
            self.claimed_skipped_label.setText(
                f"{count} D1 chart(s) skipped - already claimed"
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
