"""The M5 alert bar - a list, not a queue (trader, 2026-08-27).

"A lot of my charts to review are M5 charts. If I can instead just get a list I
can copy and paste into TC2000 that would be faster... a little sidebar in
between the master AVWAP setups and the chart... the ticker and the alert type
(new HOD, VWAP bounce etc) and I can choose what to look at. Then we can
totally purge M5 alerts from the waiting list and keep those for D1 alerts."

One line per alert, newest on top, oldest at the bottom - the trader's own
ordering rule, stated when asked. Clicking a line charts that alert in the
Alert Center exactly as a feed-row click does. "Copy all" puts the tickers on
the clipboard one per line (TC2000 paste), each ticker once, in bar order;
"Clear all" empties the bar ON SCREEN, and a clicked line leaves the bar the
moment it is charted - looked-at is done. Nothing here deletes, mutes, records
or withholds: the alert list, History, the feed and every evidence stream are
written before any alert reaches this bar, and none of them reads it.

Since 2026-09-01 a REPEAT of the same symbol+side folds into the row it already
has, with a ×N badge, and the row returns to the top carrying the newest alert.
That is the main feed's own rule applied to this bar, and it is PRESENTATION
ONLY - the sentence above still holds exactly as written. N events are drawn on
one line instead of N lines; every one of them has already reached the review
queue door, the outcome CSV and the review-event store, and the tooltip says so
on any folded row. A row also carries the take rate the Alert Center measured
for it when there is one: context, never a filter.

Rows are plain QListWidget items with a foreground role for the side - no
per-widget stylesheet, no rebuild (fluidity rules, 2026-08-21).

Since R4 (2026-09-02) a row also carries ONE capture verb, on the right-click
menu: a quick like. `SURFACE_M5_ALERT_BAR` had been declared by P10 and never
written from anywhere, so the trader's opinion of a name on this bar had no
column to land in. It is a CAPTURE and not a control: the like writes one
annotation row and does nothing else - no Focus placement, no arm, no alert, no
change to what this bar shows or to what reaches the review queue. The paragraph
above still holds for the alert stream itself; what is new is that the trader can
now say something about a row, and be recorded saying it.

Since 2026-09-23 a row whose symbol AND side is also a D1 swing setup carries
``· D1 A ★`` - the swing grade, and a star when the trader claimed that pick
(`swing_context`). The desk hands the map in; nothing here reads a file. With
the prioritise switch on, swing-backed rows are DRAWN first; the arrival list is
untouched and the switch off restores arrival order exactly.

P1-6 6a: a row also carries its alert's entry state (``valid``, ``improved``,
``gone`` or ``unknown``), measured off the Qt thread by the Working-now strip
from the NEWEST alert on that row. Display only.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.models.bounce import REGIME_PAUSE_TRIGGER_PREFIX
from swallowed import note_swallowed

#: Oldest rows fall off past this; a session produced 72 M5 alerts in its
#: first 46 minutes on 2026-08-27, so this is a whole day with room.
MAX_ROWS = 400

_ALERT_ROLE = Qt.ItemDataRole.UserRole
#: How many alerts this row has folded. Display only - see `post`.
_REPEAT_ROLE = Qt.ItemDataRole.UserRole + 1


def alert_type_label(alert: Any) -> str:
    """The short words for what fired: "new HOD", "VWAP reclaim", "lrsi_cross_20".

    The trigger is what the feed already shows; this only drops the tier tag
    in front of it and the regime-pause preamble, so the bar reads as a list
    of tickers and types rather than a second feed.
    """
    trigger = str(getattr(alert, "trigger", "") or "").strip()
    if trigger.startswith(REGIME_PAUSE_TRIGGER_PREFIX):
        tail = trigger[len(REGIME_PAUSE_TRIGGER_PREFIX) :].lstrip(" ·:-")
        trigger = tail or "regime pause"
    if trigger.startswith("[") and "]" in trigger:
        trigger = trigger.split("]", 1)[1].strip() or trigger
    if not trigger:
        trigger = str(getattr(alert, "timeframe", "") or "alert").strip() or "alert"
    return trigger[:48]


def take_probability(alert: Any) -> float | None:
    """The take probability the Alert Center already computed, or None.

    READ ONLY. The host attaches it to the alert before posting; nothing here
    computes, looks up, or stats a file for it. An alert the desk has no
    guidance for carries nothing, and the row simply does not mention a take
    rate - which is the honest rendering of "not measured", where a 0% would
    be a claim.
    """
    value = getattr(alert, "review_take_prob", None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 1.0 else None


def alert_grade_label(alert: Any) -> str:
    """The Alert Center's existing grade, rendered at the row's left edge."""
    # `alert_center_panel` owns both readers and imports this widget, so defer the
    # import until a row is drawn rather than creating an import cycle at startup.
    from ui.panels.alert_center_panel import extract_alert_tier, is_proven_alert

    if is_proven_alert(alert):
        return "[PROVEN]"
    tier = extract_alert_tier(alert)
    return f"[{tier}]" if tier else "[—]"


def row_text(
    alert: Any,
    *,
    repeats: int = 1,
    grade: str | None = None,
    swing: str = "",
    entry: str = "",
    size: str = "",
) -> str:
    """One line: grade, time, side, ticker, what fired, and take context.

    The take-rate suffix is P(take | shown) for this alert's segments, measured
    from the trader's own review decisions. The held/ran suffix is decision 0016
    answer 4's day-trade headline for this alert's `held_run_score` cell - did
    the level hold, and then how far did it run. Both are CONTEXT on a row, never
    a filter: every alert is on the bar whatever they say, and both are BLANK
    rather than zero when the desk has not measured them.

    ``repeats`` is the ×N fold badge - see `M5AlertBar.post`. It counts what
    the bar has SHOWN, and the count is display only.

    ``swing`` is `swing_context.suffix` (``· D1 A ★``), or blank when the name
    and side are not a D1 swing setup.

    ``entry`` is the P1-6 entry chip (``valid`` / ``improved`` / ``gone``),
    blank until the state has been measured. ``size`` is the P1-6 size at the
    trader's fixed risk with its notional (``· 120 sh · $12.0k``), blank when
    sizing is off or the alert has no usable stop. A size, never an order.
    """
    time_text = str(getattr(alert, "time_text", "") or "")[:5]
    side = str(getattr(alert, "side", "") or "")
    mark = "▲" if side == "LONG" else "▼" if side == "SHORT" else "·"
    label = f"[{grade}]" if grade else alert_grade_label(alert)
    line = (
        f"{label}  {time_text}  {mark} "
        f"{getattr(alert, 'symbol', '')}  {alert_type_label(alert)}"
    )
    if swing:
        line += f"  {swing}"
    if entry:
        line += f"  · {entry}"
    if size:
        line += f"  · {size}"
    if repeats > 1:
        line += f"  ×{repeats}"
    probability = take_probability(alert)
    if probability is not None:
        line += f"  take {probability * 100:.0f}%"
    suffix = str(getattr(alert, "held_run_suffix", "") or "").strip()
    if suffix:
        line += f"  {suffix}"
    return line


class M5AlertBar(QWidget):
    """Newest alert on top. Click charts it; Copy/Clear act on the list only."""

    alertActivated = Signal(object)  # the BounceAlert behind the clicked row
    #: R4 A5 - one annotation row was written for this alert. Capture only; no
    #: listener may treat it as a placement, and nothing here reads it back.
    likeRecorded = Signal(object)

    def __init__(self, parent=None, *, annotations_path: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("M5AlertBar")
        # Where a quick like is written (R4 A5). None means the one live
        # stream, which is what the desk passes; the seam exists so a test can
        # exercise the real handler without touching the trader's file.
        self._annotations_path = annotations_path
        # ST6.5. `[(bounce_type, SIDE)]`, best first, off the desk's shared
        # Working-lately snapshot. Read AT SORT TIME and only when the switch is
        # ON: this REORDERS and never withholds - every row that was here is
        # still here, the repeat fold is computed before any sort, and turning
        # the switch off brings today's arrival order back exactly.
        self._working_lately_order: list[tuple[str, str]] = []
        #: The bar's own backing list, in ARRIVAL order (newest first). The
        #: QListWidget shows a VIEW of it. Blocker 1 of the ST6 re-review was
        #: that there was no such list: the widget was both the model and the
        #: display, so a sort applied while the switch was on could not be
        #: undone when it went off.
        self._arrival: list[QListWidgetItem] = []
        #: `{(SYMBOL, SIDE): {"grade", "family", "claimed"}}` from the desk's
        #: setups table (`swing_context`). Display and draw order only.
        self._swing_context: dict = {}
        #: `{(SYMBOL, SIDE): entry_state dict}` from the Working-now worker (6a).
        self._entry_states: dict = {}
        #: P1-6 6c: fixed risk per trade in dollars; None = sizing off.
        self._risk_dollars: float | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.title_label = QLabel("M5 alerts")
        self.title_label.setObjectName("SectionTitle")
        header.addWidget(self.title_label, 1)
        self.copy_button = QPushButton("Copy all")
        self.copy_button.setToolTip(
            "Copy every ticker in this bar to the clipboard, one per line, "
            "each once, newest first - paste straight into a TC2000 watchlist."
        )
        self.copy_button.clicked.connect(self.copy_all)
        self.clear_button = QPushButton("Clear all")
        self.clear_button.setToolTip(
            "Empty this bar on screen. Nothing is deleted anywhere else - the "
            "feed, History and the evidence files keep every alert."
        )
        self.clear_button.clicked.connect(self.clear_all)
        header.addWidget(self.copy_button, 0)
        header.addWidget(self.clear_button, 0)
        layout.addLayout(header)

        self.list = QListWidget()
        self.list.setObjectName("M5AlertList")
        self.list.setUniformItemSizes(True)
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.itemClicked.connect(self._on_item_clicked)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._open_row_menu)
        layout.addWidget(self.list, 1)
        self._refresh_title()

    # ---------------------------------------------------------------- data
    def post(self, alert: Any) -> None:
        """Show one alert. A repeat of the same name FOLDS into its own row.

        The fold is the main feed's rule (`alert_repetition.RepetitionLedger`,
        trader 2026-08-16: "less spam and more quality"), applied to this bar's
        far smaller question: one row per symbol+side, a ×N badge, and the row
        moves back to the top carrying the newest alert. A tier upgrade
        escalates - the row is rewritten with the stronger alert - because the
        thing that changed is exactly what the trader wanted to see.

        PRESENTATION ONLY, and this is the line that matters: every event has
        already reached `_enqueue_review_alert`, the outcome CSV and the
        review-event store before it arrives here. This bar still deletes
        nothing, mutes nothing, records nothing and withholds nothing - it
        draws N events on one line instead of N lines, and clicking that line
        charts the NEWEST of them. `alerts()` returns one object per row, so
        "Copy all" keeps listing one symbol per row exactly as before.
        """
        symbol = str(getattr(alert, "symbol", "") or "").strip().upper()
        if not symbol:
            return
        side = str(getattr(alert, "side", "") or "")
        existing = self._arrival_item_for(symbol, side)
        if existing is not None:
            repeats = int(existing.data(_REPEAT_ROLE) or 1) + 1
            self._write_item(existing, alert, repeats)
            self._arrival.remove(existing)
            self._arrival.insert(0, existing)
        else:
            item = QListWidgetItem()
            self._write_item(item, alert, 1)
            self._arrival.insert(0, item)
            self.list.addItem(item)
            # **The cap applies to the ARRIVAL list, never to the displayed
            # order** (ST6 re-review, blocker 2). Trimming the sorted list made
            # the switch decide WHICH rows survive: with MAX_ROWS at 3 and
            # AAA/BBB/CCC/DDD posted, OFF kept BBB/CCC/DDD and ON kept
            # AAA/CCC/DDD - a display preference deleting a different alert.
            while len(self._arrival) > MAX_ROWS:
                dropped = self._arrival.pop()
                row = self.list.row(dropped)
                if row >= 0:
                    self.list.takeItem(row)
        self._render_order()
        self._refresh_title()

    def mount_top_strip(self, widget) -> None:
        """Put one widget above the header - the Working-lately strip (ST6.4).

        A mounting point rather than a constructor argument: the strip is owned
        by the desk (it is fed by the desk's service and opens a desk page), and
        the bar simply gives it the top of the M5 column. Nothing about the bar
        changes; the splitter above it keeps its two children, so the M5 alerts
        widget is still `m5_column.widget(0)`.
        """
        self.layout().insertWidget(0, widget)

    def set_setup_grades(self, payload) -> None:
        """The day-trade grades from the Working-lately worker. Rewrites rows in place."""
        import setup_grades

        self._grades = setup_grades.daytrade_lookup(payload)
        for index in range(self.list.count()):
            item = self.list.item(index)
            alert = item.data(_ALERT_ROLE)
            if alert is not None:
                self._write_item(item, alert, int(item.data(_REPEAT_ROLE) or 1))

    @staticmethod
    def _bounce_types_of(alert: Any) -> str:
        import working_lately

        payload = getattr(alert, "payload", None)
        feedback = payload.get("feedback") if isinstance(payload, dict) else None
        bounce_types = str((feedback or {}).get("bounce_types") or "")
        return bounce_types or working_lately.alert_priority_key(alert)[0]

    def _grade_for(self, alert: Any) -> str | None:
        """The tracker grade badge for this alert, or None before grades load."""
        grades = getattr(self, "_grades", None)
        if not grades:
            return None
        import setup_grades

        side = str(getattr(alert, "side", "") or "")
        return setup_grades.badge(
            setup_grades.daytrade_grade_for_alert(grades, self._bounce_types_of(alert), side)
        )

    def _grade_line_for(self, alert: Any) -> str:
        """`setup_grades.cell_line` of the alert's best-graded type, "" when none."""
        grades = getattr(self, "_grades", None)
        if not grades:
            return ""
        import setup_grades

        side = str(getattr(alert, "side", "") or "")
        cell = setup_grades.daytrade_cell_for_alert(grades, self._bounce_types_of(alert), side)
        return setup_grades.cell_line(cell) if cell else ""

    def set_swing_context(self, mapping) -> None:
        """Which rows sit on a D1 swing setup. Rewrites rows in place.

        The whole map is replaced each time, so a name that left the setups
        table loses its suffix, and an empty map leaves every row clean.
        """
        self._swing_context = dict(mapping or {})
        for item in self._arrival:
            alert = item.data(_ALERT_ROLE)
            if alert is not None:
                self._write_item(item, alert, int(item.data(_REPEAT_ROLE) or 1))
        self._render_order()

    def set_entry_states(self, mapping) -> None:
        """The newest alert's entry state per (symbol, side). Rewrites changed rows only."""
        mapping = dict(mapping or {})
        old, self._entry_states = self._entry_states, mapping
        for item in self._arrival:
            alert = item.data(_ALERT_ROLE)
            if alert is None:
                continue
            key = self._entry_key(alert)
            if old.get(key) != mapping.get(key):
                self._write_item(item, alert, int(item.data(_REPEAT_ROLE) or 1))

    def set_risk_per_trade(self, value) -> None:
        """`risk_per_trade_dollars` changed (None = off). Rewrites every row."""
        import entry_plan

        risk = entry_plan.parse_risk_dollars(value)
        if risk == self._risk_dollars:
            return
        self._risk_dollars = risk
        for item in self._arrival:
            alert = item.data(_ALERT_ROLE)
            if alert is not None:
                self._write_item(item, alert, int(item.data(_REPEAT_ROLE) or 1))

    def _size_for(self, alert: Any) -> str:
        """`N sh · $notional` at the fixed risk from this alert's own entry and stop, or ''.

        A stop on the wrong side of the entry for the alert's side sizes nothing.
        """
        if self._risk_dollars is None:
            return ""
        import entry_plan

        payload = getattr(alert, "payload", None)
        feedback = payload.get("feedback") if isinstance(payload, dict) else None
        feedback = feedback if isinstance(feedback, dict) else {}
        side = getattr(alert, "side", "") or feedback.get("direction")
        entry = feedback.get("entry_price")
        shares = entry_plan.shares_for(self._risk_dollars, entry, feedback.get("stop_price"), side)
        return entry_plan.size_text(shares, entry)

    @staticmethod
    def _entry_key(alert: Any) -> tuple[str, str]:
        symbol = str(getattr(alert, "symbol", "") or "").strip().upper()
        side = str(getattr(alert, "side", "") or "").strip().upper()
        side = {"BUY": "LONG", "SELL": "SHORT"}.get(side, side)
        return (symbol, side)

    def _entry_for(self, alert: Any):
        """This row's entry state dict, or None before it is measured."""
        return self._entry_states.get(self._entry_key(alert)) if self._entry_states else None

    def _swing_for(self, alert: Any):
        """This alert's swing context, or None."""
        if not self._swing_context:
            return None
        import swing_context

        return swing_context.context_for(
            self._swing_context, getattr(alert, "symbol", ""), getattr(alert, "side", "")
        )

    def set_working_lately_order(self, order) -> None:
        """`[(bounce_type, SIDE)]`, best first. Presentation only (ST6.5)."""
        self._working_lately_order = [
            (str(cell), str(side)) for cell, side in (order or ())
        ]
        self._render_order()

    def _display_order(self) -> list:
        """The order to DRAW, computed from the arrival list. Never stored.

        **The backing list is never sorted** (ST6 re-review, blocker 1). It used
        to be: `_apply_priority_order` re-ordered the widget in place and
        returned early when the switch was off, so a list sorted while the
        switch was ON stayed sorted after it went OFF and today's arrival order
        was gone for the session. The priority order is a VIEW; turning the
        switch off is nothing more than drawing the same rows in the order they
        arrived, which is what "reorders and never withholds" has to mean if it
        is to be reversible.
        """
        import working_lately

        import swing_context

        rows = list(self._arrival)
        if len(rows) < 2 or not (self._working_lately_order or self._swing_context):
            return rows
        if not working_lately.prioritise_enabled():
            return rows
        # Swing-backed rows first (claimed, then swing grade), then the
        # Working-lately rank, then arrival order (2026-09-23).
        return [
            item
            for _swing, _rank, _index, item in sorted(
                (
                    (
                        swing_context.sort_key(self._swing_for(item.data(_ALERT_ROLE))),
                        working_lately.priority_rank(
                            self._working_lately_order,
                            working_lately.alert_priority_key(item.data(_ALERT_ROLE)),
                        ),
                        index,
                        item,
                    )
                    for index, item in enumerate(rows)
                ),
                key=lambda entry: (entry[0], entry[1], entry[2]),
            )
        ]

    def _render_order(self) -> None:
        """Draw the display order. A no-op when it already matches."""
        order = self._display_order()
        current = [self.list.item(index) for index in range(self.list.count())]
        if current == order:
            return
        while self.list.count():
            self.list.takeItem(0)
        for item in order:
            self.list.addItem(item)

    def _arrival_item_for(self, symbol: str, side: str):
        """The arrival-list row for this symbol+side, or None.

        Keyed on symbol AND side deliberately: a name that flips direction is a
        different claim, and folding the two would hide the flip - the one thing
        on this bar most worth seeing.
        """
        for item in self._arrival:
            held = item.data(_ALERT_ROLE)
            if held is None:
                continue
            if (
                str(getattr(held, "symbol", "") or "").strip().upper() == symbol
                and str(getattr(held, "side", "") or "") == side
            ):
                return item
        return None

    def _write_item(self, item: QListWidgetItem, alert: Any, repeats: int) -> None:
        """Fill one row IN PLACE - never a rebuilt widget (fluidity rules)."""
        import swing_context

        import entry_state

        swing = self._swing_for(alert)
        state = self._entry_for(alert)
        item.setText(
            row_text(
                alert,
                repeats=repeats,
                grade=self._grade_for(alert),
                swing=swing_context.suffix(swing),
                entry=entry_state.chip_text(state) if state is not None else "",
                size=self._size_for(alert),
            )
        )
        item.setData(_ALERT_ROLE, alert)
        item.setData(_REPEAT_ROLE, repeats)
        raw = str(getattr(alert, "raw_text", "") or "")
        grade_help = (
            "Grade: PROVEN, then S through D; — means the alert is ungraded.\n\n"
            if not getattr(self, "_grades", None)
            else "Grade from the Daytrade Tracker: how often this alert type reached "
            "+1R before -1R over the last 20 sessions (PROVEN, A, B, C, D; NEW = "
            "too few to grade).\n\n"
        )
        grade_line = self._grade_line_for(alert)
        if grade_line:
            grade_help = f"{grade_line}\n{grade_help}"
        if repeats > 1:
            raw = (
                f"{repeats} alerts on this name this session; the newest is shown.\n"
                "Every one of them is in the feed, History and the evidence "
                f"files - this row folds them, it does not drop them.\n\n{raw}"
            )
        swing_line = swing_context.tooltip_line(swing)
        if swing_line:
            grade_help = f"{swing_line}\n\n{grade_help}"
        if state is not None:
            grade_help = f"{entry_state.chip_detail(state)}\n\n{grade_help}"
        item.setToolTip(f"{grade_help}{raw}")
        side = str(getattr(alert, "side", "") or "")
        token = "long" if side == "LONG" else "short" if side == "SHORT" else "text_muted"
        try:
            item.setForeground(QColor(theme.color(token)))
        except Exception as exc:
            note_swallowed("M5 alert row colour not applied", exc, quiet=True)

    def alerts(self) -> list:
        """Top to bottom - newest first."""
        return [self.list.item(i).data(_ALERT_ROLE) for i in range(self.list.count())]

    def symbols(self) -> list[str]:
        """Each ticker once, in bar order (newest first)."""
        seen: set[str] = set()
        out: list[str] = []
        for alert in self.alerts():
            symbol = str(getattr(alert, "symbol", "") or "").strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                out.append(symbol)
        return out

    def count(self) -> int:
        return self.list.count()

    # ------------------------------------------------------------- actions
    def copy_all(self) -> str:
        """Tickers to the clipboard, one per line. Returns what was copied."""
        text = "\n".join(self.symbols())
        try:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)
        except Exception as exc:
            note_swallowed("M5 alert tickers not copied to the clipboard", exc)
        return text

    def clear_all(self) -> None:
        self._arrival.clear()
        self.list.clear()
        self._refresh_title()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Chart it, then take the line away (trader, 2026-08-27: "after I
        click on an alert it should go away"). Looked-at is done; the feed
        and History still have it."""
        alert = item.data(_ALERT_ROLE)
        row = self.list.row(item)
        if row >= 0:
            # Out of the arrival list too, or the next render would put it back.
            if item in self._arrival:
                self._arrival.remove(item)
            self.list.takeItem(row)
            self._refresh_title()
        if alert is not None:
            self.alertActivated.emit(alert)

    # ----------------------------------------------------------- capture
    def _open_row_menu(self, point) -> None:
        """Right-click a row: one quick like, and nothing else."""
        item = self.list.itemAt(point)
        if item is None:
            return
        menu = QMenu(self)
        like = menu.addAction("Quick like")
        like.setToolTip(
            "Records that something about this alert was good. It places "
            "nothing, arms nothing and changes nothing on this bar."
        )
        if menu.exec(self.list.viewport().mapToGlobal(point)) is like:
            self.quick_like(item)

    def quick_like(self, item: QListWidgetItem | None = None) -> dict | None:
        """One QUICK like for a row of this bar. Returns the written row.

        QUICK because this bar has no claim picklist and never asks for a why -
        P9's Alt+L path, from a different screen. A like carries zero privileges
        (plan.md P3.1): it records, and it does not act. The row stays on the bar
        so the trader can still click through to the chart.

        Every failure is swallowed and reported nowhere but the return value: an
        evidence store never costs the event it records, and there is nothing
        here for it to cost anyway.
        """
        if item is None:
            item = self.list.currentItem()
        alert = item.data(_ALERT_ROLE) if item is not None else None
        if alert is None:
            return None
        symbol = str(getattr(alert, "symbol", "") or "").strip().upper()
        if not symbol:
            return None
        side = str(getattr(alert, "side", "") or "").strip().upper()
        try:
            from ui.annotations import verdicts

            written = verdicts.record_like(
                symbol=symbol,
                side="SHORT" if side.startswith("SHORT") else "LONG",
                surface=verdicts.SURFACE_M5_ALERT_BAR,
                timeframe=str(getattr(alert, "timeframe", "") or "M5"),
                **({} if self._annotations_path is None else {"path": self._annotations_path}),
            )
        except Exception:
            return None
        if written is not None:
            self.likeRecorded.emit(alert)
        return written

    def _refresh_title(self) -> None:
        n = self.list.count()
        self.title_label.setText(f"M5 alerts ({n})" if n else "M5 alerts")
        self.copy_button.setEnabled(n > 0)
        self.clear_button.setEnabled(n > 0)
