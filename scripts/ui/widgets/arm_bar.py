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
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from chart_watch import D1_EVENT_KINDS, WATCH_KINDS
from ui import theme


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
    levelArmRequested = Signal(str, float)  # direction, level
    levelDisarmRequested = Signal(str, float)  # direction, level

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._quick_fill: Callable[[str], float | None] | None = None
        # Which source produced the price in the level box ("vwap", "upper_1",
        # "chart_click", "manual", ...). Logged with each armed level so the
        # review-events learner can see e.g. "always arms off +1σ on shorts".
        self._last_fill_source = ""
        self._setting_level_programmatically = False

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Symbol ⏎")
        self.symbol_input.setMinimumWidth(96)
        self.symbol_input.setMaximumWidth(140)
        self.symbol_input.setToolTip(
            "Type a ticker and press Enter to chart it immediately - it does not "
            "have to have alerted, or even be in the current scan set."
        )
        self.symbol_input.returnPressed.connect(self._emit_symbol)

        self._watch_warning = ""
        self.watch_buttons: dict[str, QPushButton] = {}
        for kind, label in WATCH_KINDS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, k=kind: self.watchToggled.emit(k))
            self.watch_buttons[kind] = button
        for kind in self.watch_buttons:
            self.watch_buttons[kind].setToolTip(self._tooltip_for(kind))

        self.level_input = QDoubleSpinBox()
        self.level_input.setDecimals(2)
        self.level_input.setRange(0.01, 1_000_000.0)
        self.level_input.setSingleStep(0.05)
        self.level_input.setMaximumWidth(110)
        self.level_input.setToolTip("Price level for the break alert")
        # Any edit not made by apply_quick_fill/set_level is the trader typing
        # or nudging the spinner - that overrides the remembered fill source.
        self.level_input.valueChanged.connect(self._on_level_edited)

        self.direction_input = QComboBox()
        self.direction_input.addItem("Above", "above")
        self.direction_input.addItem("Below", "below")
        self.direction_input.setMaximumWidth(90)

        self.arm_level_button = QPushButton("Arm level")
        self.arm_level_button.setObjectName("PrimaryButton")
        self.arm_level_button.setToolTip(
            "Arm a persistent break alert at this price. It survives restarts "
            "and keeps watching even while the symbol is not being scanned."
        )
        self.arm_level_button.clicked.connect(self._emit_level)

        self.d1_event_buttons: dict[str, QPushButton] = {}
        for kind, label in D1_EVENT_KINDS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(self._d1_event_tooltip(kind))
            button.clicked.connect(
                lambda _checked=False, k=kind: self.d1EventToggled.emit(k)
            )
            self.d1_event_buttons[kind] = button

        self.armed_row = QWidget()
        self.armed_layout = QHBoxLayout(self.armed_row)
        self.armed_layout.setContentsMargins(0, 0, 0, 0)
        self.armed_layout.setSpacing(6)
        self.armed_hint = QLabel("Nothing armed")
        self.armed_hint.setObjectName("MutedLabel")
        self.armed_layout.addWidget(self.armed_hint)
        self.armed_layout.addStretch(1)

        self._build_layout()
        self.set_enabled_for_symbol(False)

    def _build_layout(self) -> None:
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        top.addWidget(self.symbol_input)
        for button in self.watch_buttons.values():
            top.addWidget(button)
        top.addStretch(1)
        top.addWidget(self.level_input)
        top.addWidget(self.direction_input)
        top.addWidget(self.arm_level_button)

        d1_row = QHBoxLayout()
        d1_row.setContentsMargins(0, 0, 0, 0)
        d1_row.setSpacing(4)
        d1_label = QLabel("D1:")
        d1_label.setObjectName("MutedLabel")
        d1_label.setToolTip(
            "Persistent D1 event alerts for this symbol. Levels re-derive "
            "from the daily store every poll, so they track the moving "
            "average / rolling extreme instead of a frozen price."
        )
        d1_row.addWidget(d1_label)
        for button in self.d1_event_buttons.values():
            d1_row.addWidget(button)
        d1_row.addStretch(1)
        d1_row.addWidget(self.armed_row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addLayout(d1_row)

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
            self.level_input,
            self.direction_input,
            self.arm_level_button,
        ):
            widget.setEnabled(bool(has_symbol))

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

    def _tooltip_for(self, kind: str) -> str:
        label = WATCH_KINDS[kind]
        base = (
            f"Toggle a one-shot {label} watch for this symbol. The first "
            "completed M5 bar that meets it fires a red alert in the Alert "
            "Center (bypasses the tier gate and sounds). Click again to disarm."
        )
        warning = getattr(self, "_watch_warning", "")
        return f"{base}\n\n⚠ {warning}" if warning else base

    def set_armed_kinds(self, kinds: Iterable[str]) -> None:
        armed = set(kinds or ())
        for kind, button in self.watch_buttons.items():
            label = WATCH_KINDS[kind]
            button.setText(f"{label} ✓ armed" if kind in armed else label)
            button.setChecked(kind in armed)

    def set_armed_d1_events(self, kinds: Iterable[str]) -> None:
        """Reflect this symbol's armed D1 event watches; a second click disarms."""
        armed = set(kinds or ())
        for kind, button in self.d1_event_buttons.items():
            label = D1_EVENT_KINDS[kind]
            button.setText(f"{label} ✓" if kind in armed else label)
            button.setChecked(kind in armed)

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
