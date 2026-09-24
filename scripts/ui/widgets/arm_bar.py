"""Arming controls welded under the desk's chart.

Before this, a level alert could only be armed by right-clicking a D1 candle
and taking that candle's literal high or low, and the four one-shot watches
were only reachable when the review queue happened to hand you a symbol. This
adds the things a scanner-driven desk needs: chart any symbol on demand, arm
an arbitrary price (clicking either chart fills the box), and arm persistent
D1 EVENT alerts - 15EMA rejection, new 5/20-day extremes, SMA breaks - whose
levels move with the daily store instead of freezing at arm time.

The old quick-fill button row (Last/HOD/LOD/VWAP/±1σ) is gone: click-to-price
on the charts already fills the level box with the line the trader is looking
at, so the row duplicated a click while D1 alerts had no home. The resolver
plumbing stays - hosts still seed the box programmatically ("last") and the
fill source still feeds the decision log.
"""

from __future__ import annotations

from typing import Callable, Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chart_watch import (
    ANY_BOUNCE_KINDS,
    D1_EVENT_KINDS,
    D1_LEGACY_KINDS,
    D1_MENU_GROUPS,
    D1_MENU_LABELS,
    PERSISTENT_WATCH_KINDS,
    WATCH_KINDS,
)
from ui import theme
from ui.widgets.flow_layout import FlowLayout

# Chart-watch kinds that live on the D1 menu (the Pullback alert), not the M5 one.
_D1_MENU_WATCH_KINDS = frozenset(
    kind for _title, kinds in D1_MENU_GROUPS for kind in kinds if kind in WATCH_KINDS
)
_LEGACY_HEADER = "ARMED EARLIER — click to disarm"


def _watch_label(kind: str) -> str:
    return D1_MENU_LABELS.get(kind, WATCH_KINDS[kind])


def quick_fill_value(source: str, bars, overlays) -> float | None:
    """Resolve a quick-fill source from the drawn bars and overlay series.

    Overlays follow the chart_snapshot contract: `values` aligns 1:1 with the
    bars and may contain None where the series has no value yet, so the last
    non-None entry is the line's current level.
    """
    bars = list(bars or [])
    if not bars:
        return None
    if source == "last":
        return _as_float(bars[-1].get("close"))
    if source == "hod":
        highs = [_as_float(bar.get("high")) for bar in bars]
        highs = [value for value in highs if value is not None]
        return max(highs) if highs else None
    if source == "lod":
        lows = [_as_float(bar.get("low")) for bar in bars]
        lows = [value for value in lows if value is not None]
        return min(lows) if lows else None

    label = {"vwap": "VWAP", "upper_1": "+1σ", "lower_1": "-1σ"}.get(source)
    if label is None:
        return None
    for overlay in overlays or []:
        if str(overlay.get("label") or "") != label:
            continue
        for value in reversed(list(overlay.get("values") or [])):
            resolved = _as_float(value)
            if resolved is not None:
                return resolved
    return None


def _as_float(value) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved == resolved else None  # drop NaN


class ArmBar(QFrame):
    """Symbol box, watch toggles, price level, D1 event toggles, armed chips."""

    symbolRequested = Signal(str)
    watchToggled = Signal(str)  # chart-watch kind
    d1EventToggled = Signal(str)  # D1 event watch kind
    # R5 section 4: one armed request covering the WHOLE level set, so it
    # carries no kind - "tell me when this name bounces off any of my
    # levels". The host owns the side (it knows the charted alert).
    anyBounceToggled = Signal()
    # WISHLIST 2026-08-18: hand the charted symbol to an external charting tool
    # for deep TA. The host owns the timeframe; this bar only says "that name".
    externalChartRequested = Signal(str)
    levelArmRequested = Signal(str, float)  # direction, level
    levelDisarmRequested = Signal(str, float)  # direction, level
    # direction - a PHONE price alert off the painted D1 level the trader
    # picked. No price rides along: the host owns the selection (the chart
    # knows which line is highlighted), this bar only supplies the direction
    # the trader chose, exactly as it does for levelArmRequested.
    levelAlertRequested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._quick_fill: Callable[[str], float | None] | None = None
        # The phone-alert button needs BOTH a charted symbol and a picked
        # painted level, so its enabled state is tracked separately from the
        # rest of the row.
        self._has_symbol = False
        self._level_alert_available = False
        # Which source produced the price in the level box ("vwap", "upper_1",
        # "chart_click", "manual", ...). Logged with each armed level so the
        # review-events learner can see e.g. "always arms off +1σ on shorts".
        self._last_fill_source = ""
        self._setting_level_programmatically = False
        # What is armed, and what is QUEUED (clicked, not yet saved), per row.
        self._armed_watch: set[str] = set()
        self._armed_d1: set[str] = set()
        self._armed_any = False
        self._pending_watch: set[str] = set()
        self._pending_d1: set[str] = set()
        self._pending_any = False

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Symbol ⏎")
        self.symbol_input.setToolTip(
            "Type a ticker and press Enter to chart it immediately - it does not "
            "have to have alerted, or even be in the current scan set."
        )
        self.symbol_input.returnPressed.connect(self._emit_symbol)

        self._watch_warning = ""
        self.watch_buttons: dict[str, QPushButton] = {}
        for kind in WATCH_KINDS:
            button = QPushButton(_watch_label(kind))
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, k=kind: self.watchToggled.emit(k))
            self.watch_buttons[kind] = button
        for kind in self.watch_buttons:
            self.watch_buttons[kind].setToolTip(self._tooltip_for(kind))

        self.level_input = QDoubleSpinBox()
        self.level_input.setDecimals(2)
        self.level_input.setRange(0.01, 1_000_000.0)
        self.level_input.setSingleStep(0.05)
        self.level_input.setToolTip("Price level for the break alert")
        # Any edit not made by apply_quick_fill/set_level is the trader typing
        # or nudging the spinner - that overrides the remembered fill source.
        self.level_input.valueChanged.connect(self._on_level_edited)

        self.direction_input = QComboBox()
        self.direction_input.addItem("Above", "above")
        self.direction_input.addItem("Below", "below")

        self.arm_level_button = QPushButton("Arm level")
        self.arm_level_button.setObjectName("PrimaryButton")
        self.arm_level_button.setToolTip(
            "Arm a persistent break alert at this price. It survives restarts "
            "and keeps watching even while the symbol is not being scanned."
        )
        self.arm_level_button.clicked.connect(self._emit_level)

        # Same control shape as "Arm level" - same direction combo, one click -
        # but it arms the PHONE price alert instead of the D1 level watch, at
        # the painted line the trader clicked rather than at the level box.
        self.phone_alert_button = QPushButton("Phone alert")
        self.phone_alert_button.setToolTip(
            "Click a painted level on the D1 chart, then arm a phone price "
            "alert at exactly that line. It fires once, pushes to your phone, "
            "and then stays off until it is re-armed on the Focus tab."
        )
        self.phone_alert_button.clicked.connect(
            lambda: self.levelAlertRequested.emit(
                str(self.direction_input.currentData() or "above")
            )
        )

        self.d1_event_buttons: dict[str, QPushButton] = {}
        for kind, label in D1_EVENT_KINDS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(self._d1_event_tooltip(kind))
            button.clicked.connect(
                lambda _checked=False, k=kind: self.d1EventToggled.emit(k)
            )
            self.d1_event_buttons[kind] = button

        self.external_chart_button = QPushButton("Open in TradingView")
        self.external_chart_button.setToolTip(
            "Open this symbol in your external charting tool for deep TA. The "
            "link is a setting (external_chart_url_template), so it can point at "
            "any tool that answers a URL. Nothing is read back - the desk never "
            "learns anything from the other window."
        )
        self.external_chart_button.clicked.connect(
            lambda: self.externalChartRequested.emit(self.symbol_input.text().strip().upper())
        )

        self.any_bounce_button = QPushButton("Any bounce")
        self.any_bounce_button.setCheckable(True)
        self.any_bounce_button.setToolTip(self._any_bounce_tooltip())
        self.any_bounce_button.clicked.connect(
            lambda _checked=False: self.anyBounceToggled.emit()
        )

        self.armed_row = QWidget()
        self.armed_layout = QHBoxLayout(self.armed_row)
        self.armed_layout.setContentsMargins(0, 0, 0, 0)
        self.armed_layout.setSpacing(6)
        self.armed_hint = QLabel("Nothing armed")
        self.armed_hint.setObjectName("MutedLabel")
        self.armed_layout.addWidget(self.armed_hint)
        self.armed_layout.addStretch(1)

        # The compact desk's one-row variant (built on first use). Its menus
        # only click the buttons above, so every signal path stays theirs.
        self._compact = False
        self.compact_row: QWidget | None = None
        self.m5_menu_button: QToolButton | None = None
        self.d1_menu_button: QToolButton | None = None
        self.m5_actions: dict[str, QAction] = {}
        self.d1_actions: dict[str, QAction] = {}
        self.d1_menu_headers: list[QAction] = []

        # One muted title per D1 menu group, for the classic row.
        self._d1_group_labels: list[QLabel] = []
        for title, _kinds in D1_MENU_GROUPS:
            group_label = QLabel(title.split(" — ", 1)[0].title() + ":")
            group_label.setObjectName("MutedLabel")
            group_label.setToolTip(title)
            self._d1_group_labels.append(group_label)
        # Off-menu buttons currently shown in the classic row (armed ones only).
        # The rest wait in the hidden parking widget the compact row also uses.
        self._legacy_in_row: list[QPushButton] = []
        self._compact_parking = QWidget(self)
        self._compact_parking.setObjectName("ArmBarParking")
        self._compact_parking.hide()
        for button in self._legacy_buttons().values():
            button.setParent(self._compact_parking)

        self.apply_scaled_metrics()
        self._build_layout()
        self.set_enabled_for_symbol(False)

    def _menu_button_for(self, kind: str) -> QPushButton:
        """The real button behind one D1 menu entry (a watch or a D1 event)."""
        if kind in WATCH_KINDS:
            return self.watch_buttons[kind]
        return self.d1_event_buttons[kind]

    def _legacy_buttons(self) -> dict:
        """Off-menu D1 buttons that still show while armed, so they can be disarmed."""
        return {
            **{kind: self.d1_event_buttons[kind] for kind in D1_LEGACY_KINDS},
            "any_bounce": self.any_bounce_button,
        }

    def _legacy_shown(self) -> list[QPushButton]:
        return [button for button in self._legacy_buttons().values() if button.isChecked()]

    def _classic_top_widgets(self) -> list:
        return [
            self.symbol_input,
            *(
                button
                for kind, button in self.watch_buttons.items()
                if kind not in _D1_MENU_WATCH_KINDS
            ),
            self.level_input,
            self.direction_input,
            self.arm_level_button,
            self.phone_alert_button,
        ]

    def _classic_d1_widgets(self) -> list:
        grouped: list = []
        for group_label, (_title, kinds) in zip(self._d1_group_labels, D1_MENU_GROUPS):
            grouped.append(group_label)
            grouped.extend(self._menu_button_for(kind) for kind in kinds)
        return [
            self._d1_label,
            *grouped,
            *self._legacy_shown(),
            self.external_chart_button,
            self.armed_row,
        ]

    def _sync_legacy_classic(self) -> None:
        """Show an off-menu D1 button in the classic row only while it is armed."""
        if self._compact or not hasattr(self, "_d1_flow"):
            return
        shown = self._legacy_shown()
        if shown != self._legacy_in_row:
            self._legacy_in_row = shown
            self._fill_classic_rows()
        for button in self._legacy_buttons().values():
            if button in shown:
                button.show()
            elif button.parentWidget() is not self._compact_parking:
                button.setParent(self._compact_parking)

    def _build_layout(self) -> None:
        # Both control rows WRAP rather than compress. A QHBoxLayout hands every
        # child an equal share of whatever width is left, so on a 1680px laptop
        # desk the watch and D1 buttons squeezed down to unreadable stubs
        # ("ew HC", "d hig") - the controls were all still there and none of
        # them could be identified. Flowing onto a second line costs vertical
        # space the alert column has and buys back every label.
        top = FlowLayout(margin=0, spacing=theme.px(6))
        self._top_flow = top

        d1_label = QLabel("D1:")
        d1_label.setObjectName("MutedLabel")
        d1_label.setToolTip(
            "Persistent D1 event alerts for this symbol. Levels re-derive "
            "from the daily store every poll, so they track the moving "
            "average / rolling extreme instead of a frozen price."
        )
        self._d1_label = d1_label
        d1_row = FlowLayout(margin=0, spacing=theme.px(4))
        self._d1_flow = d1_row
        self._fill_classic_rows()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.px(6), theme.px(4), theme.px(6), theme.px(4))
        layout.setSpacing(theme.px(4))
        layout.addLayout(top)
        layout.addLayout(d1_row)

    def _fill_classic_rows(self) -> None:
        """(Re)fill the two classic rows in their one canonical order."""
        for flow, widgets in (
            (self._top_flow, self._classic_top_widgets()),
            (self._d1_flow, self._classic_d1_widgets()),
        ):
            _empty_layout(flow)
            for widget in widgets:
                flow.addWidget(widget)

    # ------------------------------------------------------------ compact row
    def is_compact(self) -> bool:
        return self._compact

    def set_compact(self, compact: bool) -> None:
        """One row (compact desk) or the classic rows, live and reversible.

        Compact keeps the symbol box, level controls and armed chips, and puts
        the M5 and D1 alert buttons behind two menus whose actions click the
        real buttons. The buttons themselves are parked, never deleted.
        """
        compact = bool(compact)
        if compact == self._compact:
            return
        self._compact = compact
        root = self.layout()
        if compact:
            self._ensure_compact_row()
            _empty_layout(self._top_flow)
            _empty_layout(self._d1_flow)
            for widget in (
                self._d1_label,
                *self._d1_group_labels,
                *self.watch_buttons.values(),
                *self.d1_event_buttons.values(),
                self.any_bounce_button,
                self.external_chart_button,
            ):
                widget.setParent(self._compact_parking)
            row_layout = self.compact_row.layout()
            _empty_layout(row_layout)
            for widget in (
                self.symbol_input,
                self.m5_menu_button,
                self.d1_menu_button,
                self.level_input,
                self.direction_input,
                self.arm_level_button,
                self.phone_alert_button,
                self.armed_row,
            ):
                row_layout.addWidget(widget)
            root.insertWidget(0, self.compact_row)
            root.setContentsMargins(theme.px(6), theme.px(1), theme.px(6), theme.px(1))
            self.compact_row.setVisible(True)
            self.sync_compact_menus()
        else:
            root.removeWidget(self.compact_row)
            root.setContentsMargins(theme.px(6), theme.px(4), theme.px(6), theme.px(4))
            self.compact_row.setVisible(False)
            _empty_layout(self.compact_row.layout())
            self._legacy_in_row = self._legacy_shown()
            self._fill_classic_rows()
            self._sync_legacy_classic()

    def _ensure_compact_row(self) -> None:
        if self.compact_row is not None:
            return
        self.compact_row = QWidget(self)
        self.compact_row.setObjectName("ArmBarCompactRow")
        row_layout = QHBoxLayout(self.compact_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(theme.px(6))

        m5_menu = QMenu(self)
        m5_menu.setToolTipsVisible(True)
        for kind, button in self.watch_buttons.items():
            if kind in _D1_MENU_WATCH_KINDS:
                continue
            action = QAction(button.text(), self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked=False, b=button: self._click_from_menu(b))
            m5_menu.addAction(action)
            self.m5_actions[kind] = action
        m5_menu.aboutToShow.connect(self.sync_compact_menus)
        self.m5_menu_button = _menu_button(m5_menu)

        self._grouped_d1 = GroupedD1Menu(
            self,
            {kind: self._menu_button_for(kind) for _t, kinds in D1_MENU_GROUPS for kind in kinds},
            self._legacy_buttons(),
            on_click=self._click_from_menu,
            trailing={"external_chart": self.external_chart_button},
        )
        self.d1_actions = self._grouped_d1.actions
        self.d1_menu_headers = self._grouped_d1.headers
        self._grouped_d1.menu.aboutToShow.connect(self.sync_compact_menus)
        self.d1_menu_button = _menu_button(self._grouped_d1.menu)
        self.compact_row.setVisible(False)

    def _click_from_menu(self, button: QPushButton) -> None:
        button.click()
        self.sync_compact_menus()

    def sync_compact_menus(self) -> None:
        """Mirror each button's text, checked, enabled and tooltip onto its action."""
        if self.compact_row is None:
            return
        armed = _mirror_actions(self.m5_actions, self.watch_buttons)
        self.m5_menu_button.setText(f"M5 alert ({armed}) ▾" if armed else "M5 alert ▾")
        self.m5_menu_button.setToolTip("One-shot M5 chart watches for this symbol.")
        self.m5_menu_button.setEnabled(self._has_symbol)
        armed = self._grouped_d1.sync()
        self.d1_menu_button.setText(f"D1 alert ({armed}) ▾" if armed else "D1 alert ▾")
        self.d1_menu_button.setToolTip(
            "D1 alerts by kind: pullback, breakout, line break - and TradingView."
        )
        self.d1_menu_button.setEnabled(self._has_symbol)

    def apply_scaled_metrics(self) -> None:
        """Re-apply the input widths that are pixel budgets, not stylesheet."""
        self.symbol_input.setMinimumWidth(theme.px(96))
        self.symbol_input.setMaximumWidth(theme.px(140))
        self.level_input.setMaximumWidth(theme.px(110))
        self.direction_input.setMaximumWidth(theme.px(90))

    # ------------------------------------------------------------------
    def set_quick_fill_source(self, resolver: Callable[[str], float | None]) -> None:
        """Install the callback that resolves a quick-fill source to a price."""
        self._quick_fill = resolver

    def apply_quick_fill(self, source: str) -> bool:
        if self._quick_fill is None:
            return False
        value = self._quick_fill(source)
        if value is None:
            return False
        self._set_level_value(float(value), source)
        return True

    def set_level(self, level: float) -> None:
        """Used by click-to-price on the charts."""
        value = _as_float(level)
        if value is not None and value > 0:
            self._set_level_value(value, "chart_click")

    def _set_level_value(self, value: float, source: str) -> None:
        self._setting_level_programmatically = True
        try:
            self.level_input.setValue(value)
        finally:
            self._setting_level_programmatically = False
        self._last_fill_source = source

    def _on_level_edited(self, *_args) -> None:
        if not self._setting_level_programmatically:
            self._last_fill_source = "manual"

    def last_fill_source(self) -> str:
        """Where the current level-box price came from, for decision logging."""
        return self._last_fill_source

    def set_enabled_for_symbol(self, has_symbol: bool) -> None:
        for widget in (
            *self.watch_buttons.values(),
            *self.d1_event_buttons.values(),
            self.any_bounce_button,
            self.external_chart_button,
            self.level_input,
            self.direction_input,
            self.arm_level_button,
        ):
            widget.setEnabled(bool(has_symbol))
        self._has_symbol = bool(has_symbol)
        self._sync_level_alert_button()
        self.sync_compact_menus()

    def set_level_alert_available(self, available: bool) -> None:
        """Offer the phone-alert button only while a painted level is picked.

        Unlike the watch toggles (which stay permissive because a watch with
        no bars still works), this one has nothing to arm AT without a chosen
        line, so a click would be a silent no-op. Disabled says so honestly.
        """
        self._level_alert_available = bool(available)
        self._sync_level_alert_button()

    def _sync_level_alert_button(self) -> None:
        self.phone_alert_button.setEnabled(
            self._has_symbol and self._level_alert_available
        )

    def set_watch_availability(self, available: bool, reason: str = "") -> None:
        """Warn when a session watch has no bars to evaluate against.

        Deliberately does NOT disable the buttons. Arming stays permissive
        because a watch armed before the bot has cached the symbol still works
        - chart_watch adopts the first tracked bar as its baseline - so
        refusing the click would block a legitimate arm. The trader gets the
        caveat in the tooltip, and the armed-watch inventory carries the same
        state as an explicit health column.
        """
        self._watch_warning = "" if available else str(reason or "")
        for kind, button in self.watch_buttons.items():
            button.setToolTip(self._tooltip_for(kind))
        self.sync_compact_menus()

    _WATCH_KIND_DETAILS = {
        "pullback": (
            "Four ways a pullback can offer an entry, on one arm. (1) The "
            "hourly chart comes back to its 15-EMA and holds it: a completed "
            "H1 bar tags the line (within 0.25 ATR) and a later one - inside "
            "three bars - closes back through it, with the EMA moving your "
            "way; a touch and a reclaim on the SAME candle is ambiguous and "
            "does not fire, and a close a full ATR the wrong side of the line "
            "ends the watch. (2) A completed M15 close reclaims the 150-SMA "
            "(M30: the 75) with an LRSI cross up through 80 on that bar or "
            "the two before it. (3) The M30 has reclaimed and held the 75-SMA "
            "and the LRSI crosses 80 later, on the M30 or the M15. (4) A "
            "later bar retests the SMA - low within 0.25 ATR of it - and "
            "still closes on the right side. Every fire names its trigger and "
            "timeframe. It needs 45 completed H1 / 160 M15 / 85 M30 bars "
            "before each leg will answer at all."
        ),
        "hod_avwap": (
            "AVWAP anchored on whichever candle made today's HOD (re-anchors if "
            "a new HOD prints); fires when a completed bar tags the line from "
            "below and closes above it."
        ),
        "lod_avwap": (
            "AVWAP anchored on whichever candle made today's LOD (re-anchors if "
            "a new LOD prints); fires when a completed bar tags the line from "
            "above and closes below it."
        ),
    }

    def _tooltip_for(self, kind: str) -> str:
        label = WATCH_KINDS[kind]
        if kind in PERSISTENT_WATCH_KINDS:
            base = (
                f"Toggle a one-shot {label} watch for this symbol. It is NOT a "
                "session watch: it stays armed for ten trading days, across "
                "restarts, and is evaluated on every completed H1 bar. One "
                "fire, then it disarms - re-arming is a new watch. Click "
                "again to disarm."
            )
        else:
            base = (
                f"Toggle a one-shot {label} watch for this symbol. The first "
                "completed M5 bar that meets it fires a red alert in the Alert "
                "Center (bypasses the tier gate and sounds). Click again to disarm."
            )
        detail = self._WATCH_KIND_DETAILS.get(kind)
        if detail:
            base = f"{base}\n\n{detail}"
        warning = getattr(self, "_watch_warning", "")
        return f"{base}\n\n⚠ {warning}" if warning else base

    def set_armed_kinds(self, kinds: Iterable[str]) -> None:
        armed = set(kinds or ())
        self._armed_watch = armed
        for kind, button in self.watch_buttons.items():
            label = _watch_label(kind)
            queued = kind in self._pending_watch and kind not in armed
            button.setText(
                f"⏳ {label}" if queued else f"{label} ✓ armed" if kind in armed else label
            )
            button.setChecked(kind in armed or queued)
        self.sync_compact_menus()

    def set_armed_d1_events(self, kinds: Iterable[str]) -> None:
        """Reflect this symbol's armed D1 event watches; a second click disarms."""
        armed = set(kinds or ())
        self._armed_d1 = armed
        for kind, button in self.d1_event_buttons.items():
            label = D1_EVENT_KINDS[kind]
            queued = kind in self._pending_d1 and kind not in armed
            button.setText(f"⏳ {label}" if queued else f"{label} ✓" if kind in armed else label)
            button.setChecked(kind in armed or queued)
        self._sync_legacy_classic()
        self.sync_compact_menus()

    def set_any_bounce_armed(self, armed: bool) -> None:
        """Reflect this symbol's any-bounce watch; a second click disarms."""
        self._armed_any = bool(armed)
        queued = self._pending_any and not armed
        self.any_bounce_button.setText(
            "⏳ Any bounce" if queued else "Any bounce ✓" if armed else "Any bounce"
        )
        self.any_bounce_button.setChecked(bool(armed) or queued)
        self._sync_legacy_classic()
        self.sync_compact_menus()

    def set_pending_arms(
        self, watch_kinds: Iterable[str] = (), d1_kinds: Iterable[str] = (), any_bounce: bool = False
    ) -> None:
        """Show QUEUED arms (⏳, pressed) until they save; a click cancels one."""
        self._pending_watch = set(watch_kinds or ())
        self._pending_d1 = set(d1_kinds or ())
        self._pending_any = bool(any_bounce)
        self.set_armed_kinds(self._armed_watch)
        self.set_armed_d1_events(self._armed_d1)
        self.set_any_bounce_armed(self._armed_any)

    @staticmethod
    def _any_bounce_tooltip() -> str:
        levels = ", ".join(ANY_BOUNCE_KINDS.values())
        return (
            "Tell me when this name bounces off ANY of my levels: "
            f"{levels}. Two completed 5m bars confirm it, exactly as the "
            "D1 zone arms do. It fires once, names the level that held, "
            "and disarms - click again to re-arm. A level the data cannot "
            "supply is simply not watched."
        )

    @staticmethod
    def _d1_event_tooltip(kind: str) -> str:
        label = D1_EVENT_KINDS[kind]
        detail = {
            "ema15_reject": (
                "price tags the D1 15EMA and a completed M5 bar closes back "
                "on the other side of it (fires long or short as it happens)"
            ),
            "new_5d_high": "a completed bar trades above the prior 5 sessions' high",
            "new_5d_low": "a completed bar trades below the prior 5 sessions' low",
            "new_20d_high": "a completed bar trades above the prior 20 sessions' high",
            "new_20d_low": "a completed bar trades below the prior 20 sessions' low",
            "sma_break": (
                "a completed bar closes across the D1 SMA50/100/200 - any of "
                "the three, either direction"
            ),
            "avwape_bounce": (
                "price tags the AVWAPE line and closes back on the side it "
                "came from. Needs an earnings anchor in the cache"
            ),
            "avwape_break": (
                "a completed bar closes THROUGH the AVWAPE line, either "
                "direction. Needs an earnings anchor in the cache"
            ),
            "avwape_dev1_bounce": (
                "price tags the AVWAPE +1σ or -1σ band and closes back on "
                "the side it came from; the alert names the band. Needs an "
                "earnings anchor in the cache"
            ),
            "avwape_dev1_break": (
                "a completed bar closes THROUGH the AVWAPE +1σ or -1σ band, "
                "either direction; the alert names the band. Needs an "
                "earnings anchor in the cache"
            ),
            "trendline_break": (
                "a completed D1 bar closes through the exact scan trendline "
                "saved when you arm it"
            ),
            "trendline_break_retest": (
                "a completed D1 bar breaks the saved scan trendline, a later "
                "bar retests it, and a third completed bar confirms"
            ),
            "d1_line_pullback": (
                "price tags the D1 15EMA, the AVWAPE line or its 1σ band and "
                "closes back; the alert names the line"
            ),
            "range_breakout": (
                "a new 20-day high or low, but only out of a tight 20-day base "
                "(range at most 4x ATR14)"
            ),
            "sma_break_retest": (
                "a completed D1 close through the SMA50/100/200 your way, then "
                "within 10 sessions a later bar tags the D1 15EMA and closes back"
            ),
            "line_break": (
                "a completed bar closes through the SMA50/100/200, the AVWAPE "
                "line or its 1σ band; the alert names the line"
            ),
        }.get(kind, "")
        return (
            f"Toggle a persistent {label} alert for this symbol: {detail}. "
            "The reference level re-derives from the daily store every poll. "
            "One-shot, survives restarts and sessions, fires red in the "
            "Alert Center. Click again to disarm."
        )

    def set_armed_levels(self, levels: Iterable) -> None:
        """Render one dismissable chip per armed level for this symbol."""
        while self.armed_layout.count():
            item = self.armed_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        levels = list(levels or [])
        if not levels:
            self.armed_hint = QLabel("Nothing armed")
            self.armed_hint.setObjectName("MutedLabel")
            self.armed_layout.addWidget(self.armed_hint)
            self.armed_layout.addStretch(1)
            return
        for watch in levels:
            chip = _ArmedLevelChip(watch)
            chip.disarmRequested.connect(self.levelDisarmRequested)
            self.armed_layout.addWidget(chip)
        self.armed_layout.addStretch(1)

    # ------------------------------------------------------------------
    def _emit_symbol(self) -> None:
        symbol = self.symbol_input.text().strip().upper()
        if symbol:
            self.symbolRequested.emit(symbol)
            self.symbol_input.clear()

    def _emit_level(self) -> None:
        level = float(self.level_input.value())
        if level > 0:
            self.levelArmRequested.emit(
                str(self.direction_input.currentData() or "above"), level
            )


def _mirror_actions(actions: dict, sources: dict) -> int:
    """Copy each button's text/tooltip/enabled/checked onto its action; count checked."""
    armed = 0
    for key, action in actions.items():
        button = sources[key]
        action.setText(button.text())
        action.setToolTip(button.toolTip())
        action.setEnabled(button.isEnabled())
        if button.isCheckable():
            action.setChecked(button.isChecked())
            armed += int(button.isChecked())
    return armed


class GroupedD1Menu:
    """The D1 alert menu grouped by `D1_MENU_GROUPS`; its actions click real buttons.

    Shared by the arm bar and the snapshot popup. Off-menu (legacy) buttons get
    actions that show only while their button is checked (armed or queued).
    """

    def __init__(self, owner, grouped: dict, legacy: dict, *, on_click, trailing=None) -> None:
        self.menu = QMenu(owner)
        self.menu.setToolTipsVisible(True)
        self.actions: dict[str, QAction] = {}
        self.headers: list[QAction] = []
        self._sources = {**grouped, **legacy, **(trailing or {})}
        self._legacy = dict(legacy)

        def add_header(text: str) -> QAction:
            header = QAction(text, owner)
            header.setEnabled(False)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            self.menu.addAction(header)
            self.headers.append(header)
            return header

        def add_action(key: str, button: QPushButton) -> None:
            action = QAction(button.text(), owner)
            action.setCheckable(button.isCheckable())
            action.triggered.connect(lambda _checked=False, b=button: on_click(b))
            self.menu.addAction(action)
            self.actions[key] = action

        for index, (title, kinds) in enumerate(D1_MENU_GROUPS):
            if index:
                self.menu.addSeparator()
            add_header(title)
            for kind in kinds:
                add_action(kind, grouped[kind])
        self._legacy_separator = self.menu.addSeparator()
        self._legacy_header = add_header(_LEGACY_HEADER)
        for key, button in self._legacy.items():
            add_action(key, button)
        if trailing:
            self.menu.addSeparator()
            for key, button in trailing.items():
                add_action(key, button)

    def sync(self) -> int:
        """Mirror the buttons; show legacy rows only while checked. Returns the checked count."""
        armed = _mirror_actions(self.actions, self._sources)
        any_legacy = False
        for key, button in self._legacy.items():
            shown = button.isChecked()
            self.actions[key].setVisible(shown)
            any_legacy = any_legacy or shown
        self._legacy_header.setVisible(any_legacy)
        self._legacy_separator.setVisible(any_legacy)
        return armed


def _empty_layout(layout) -> None:
    """Take every item out of `layout` without deleting any widget."""
    while layout.count():
        layout.takeAt(0)


def _menu_button(menu: QMenu) -> QToolButton:
    button = QToolButton()
    button.setObjectName("CompactMenuButton")
    button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    button.setMenu(menu)
    return button


class _ArmedLevelChip(QFrame):
    disarmRequested = Signal(str, float)

    def __init__(self, watch, parent=None) -> None:
        super().__init__(parent)
        self.watch = watch
        arrow = "▲" if watch.direction == "above" else "▼"
        accent = theme.color("long" if watch.direction == "above" else "short")
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {theme.with_alpha(accent, 0.55)};"
            f" border-radius: 7px; }}"
        )
        label = QLabel(f"{arrow} {watch.level:.2f}")
        label.setStyleSheet(f"color: {accent}; border: none;")
        remove = QPushButton("✕")
        remove.setFlat(True)
        remove.setMaximumWidth(20)
        remove.setCursor(Qt.CursorShape.PointingHandCursor)
        remove.setToolTip("Disarm this level alert")
        remove.clicked.connect(
            lambda: self.disarmRequested.emit(self.watch.direction, float(self.watch.level))
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 2, 1)
        layout.setSpacing(2)
        layout.addWidget(label)
        layout.addWidget(remove)
