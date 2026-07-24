from __future__ import annotations

"""Automatic D1/M5 visual review surface for Alert Center alerts."""

from typing import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from chart_watch import WATCH_KINDS
from ui.models.bounce import BounceAlert
from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget


class AlertChartReview(QWidget):
    removeTodayRequested = Signal(object)
    focusRequested = Signal(object)
    skipRequested = Signal(object)
    d1FocusRequested = Signal(object)
    watchRequested = Signal(object, str)  # (alert, chart-watch kind)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.alert: BounceAlert | None = None

        self.title = QLabel("Visual Alert Review")
        self.title.setObjectName("SectionTitle")
        self.alert_text = QLabel("Waiting for the next ticker alert.")
        self.alert_text.setWordWrap(True)
        self.alert_text.setObjectName("MutedLabel")
        self.queue_label = QLabel("")
        self.queue_label.setObjectName("MutedLabel")

        self.snapshot = SymbolSnapshotWidget(self)
        self.snapshot.setVisible(False)

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

        # Second row: actions that only exist on this visual chart. "Add to
        # D1 Focus" pins the pick into the D1 Focus feed; the watch buttons
        # arm one-shot conditions that flag RED in the Alert Center when a
        # completed M5 bar meets them.
        self.d1_focus_button = QPushButton("Add to D1 Focus")
        self.d1_focus_button.setToolTip(
            "Pin this symbol into the D1 Focus feed below so it sits with the "
            "confirmed favorite / high-conviction names."
        )
        self.d1_focus_button.clicked.connect(
            lambda: self.alert is not None and self.d1FocusRequested.emit(self.alert)
        )
        self.watch_buttons: dict[str, QPushButton] = {}
        for kind, label in WATCH_KINDS.items():
            button = QPushButton(label)
            button.setToolTip(
                f"Arm a one-shot {label} watch for this symbol. The first "
                "completed M5 bar that meets it fires a red alert in the "
                "Alert Center (bypasses the tier gate and sounds)."
            )
            button.clicked.connect(
                lambda _checked=False, k=kind: self.alert is not None
                and self.watchRequested.emit(self.alert, k)
            )
            self.watch_buttons[kind] = button

        buttons = QHBoxLayout()
        buttons.addWidget(self.remove_today_button)
        buttons.addWidget(self.focus_button)
        buttons.addWidget(self.skip_button)
        buttons.addStretch(1)
        buttons.addWidget(self.queue_label)

        watch_row = QHBoxLayout()
        watch_row.addWidget(self.d1_focus_button)
        for button in self.watch_buttons.values():
            watch_row.addWidget(button)
        watch_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        layout.addWidget(self.title)
        layout.addWidget(self.alert_text)
        layout.addWidget(self.snapshot, 1)
        layout.addLayout(buttons)
        layout.addLayout(watch_row)
        self._set_actions_enabled(False)

    def set_alert(
        self,
        alert: BounceAlert,
        *,
        bot=None,
        focus_category: str = "m5",
        queued: int = 0,
        armed_kinds: Iterable[str] = (),
    ) -> None:
        self.alert = alert
        side = f" · {alert.side}" if alert.side else ""
        timeframe = f" · {alert.timeframe}" if alert.timeframe else ""
        self.title.setText(f"{alert.symbol}{side}{timeframe}")
        self.alert_text.setText(alert.trigger or alert.raw_text)
        self.focus_button.setText(
            "Add to Swing Focus" if focus_category == "swing" else "Add to M5 Focus"
        )
        self.snapshot.set_symbol(alert.symbol, bot=bot)
        self.snapshot.setVisible(True)
        self.queue_label.setText(f"{queued} waiting" if queued else "queue clear")
        self._set_actions_enabled(True)
        self.set_armed_kinds(armed_kinds)

    def clear(self) -> None:
        self.alert = None
        self.title.setText("Visual Alert Review")
        self.alert_text.setText("Waiting for the next ticker alert.")
        self.snapshot.setVisible(False)
        self.queue_label.setText("")
        self._set_actions_enabled(False)
        self.set_armed_kinds(())

    def set_queued_count(self, count: int) -> None:
        self.queue_label.setText(f"{count} waiting" if count else "queue clear")

    def set_armed_kinds(self, kinds: Iterable[str]) -> None:
        """Reflect this symbol's already-armed watches: ✓ and locked."""
        armed = set(kinds)
        for kind, button in self.watch_buttons.items():
            label = WATCH_KINDS[kind]
            button.setText(f"{label} ✓ armed" if kind in armed else label)
            button.setEnabled(self.alert is not None and kind not in armed)

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self.remove_today_button,
            self.focus_button,
            self.skip_button,
            self.d1_focus_button,
            *self.watch_buttons.values(),
        ):
            button.setEnabled(enabled)
