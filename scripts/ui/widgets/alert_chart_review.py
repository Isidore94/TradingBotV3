from __future__ import annotations

"""Automatic D1/M5 visual review surface for Alert Center alerts."""

from typing import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from chart_watch import WATCH_KINDS
from ui.models.bounce import BounceAlert
from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget


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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.alert: BounceAlert | None = None
        self._cross_labels = ("Add to D1 Focus", "✓ In D1 Focus")

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

        self.cross_focus_button = QPushButton(self._cross_labels[0])
        self.cross_focus_button.setCheckable(True)
        self.cross_focus_button.clicked.connect(
            lambda: self.alert is not None and self.crossFocusToggled.emit(self.alert)
        )
        self.watch_buttons: dict[str, QPushButton] = {}
        for kind, label in WATCH_KINDS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(
                f"Toggle a one-shot {label} watch for this symbol. The first "
                "completed M5 bar that meets it fires a red alert in the "
                "Alert Center (bypasses the tier gate and sounds). Click "
                "again to disarm."
            )
            button.clicked.connect(
                lambda _checked=False, k=kind: self.alert is not None
                and self.watchToggled.emit(self.alert, k)
            )
            self.watch_buttons[kind] = button

        buttons = QHBoxLayout()
        buttons.addWidget(self.remove_today_button)
        buttons.addWidget(self.focus_button)
        buttons.addWidget(self.skip_button)
        buttons.addStretch(1)
        buttons.addWidget(self.queue_label)

        watch_row = QHBoxLayout()
        watch_row.addWidget(self.cross_focus_button)
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
        cross_active: bool = False,
    ) -> None:
        self.alert = alert
        side = f" · {alert.side}" if alert.side else ""
        timeframe = f" · {alert.timeframe}" if alert.timeframe else ""
        self.title.setText(f"{alert.symbol}{side}{timeframe}")
        self.alert_text.setText(alert.trigger or alert.raw_text)
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
            # M5 pick: the cross-promote is a pin into the D1 Focus feed.
            self._cross_labels = ("Add to D1 Focus", "✓ In D1 Focus")
            self.cross_focus_button.setToolTip(
                "Toggle a pin for this pick in the D1 Focus feed below, next "
                "to the confirmed favorite / high-conviction names. Click "
                "again to unpin."
            )
        self.snapshot.set_symbol(alert.symbol, bot=bot)
        self.snapshot.setVisible(True)
        self.queue_label.setText(f"{queued} waiting" if queued else "queue clear")
        self._set_actions_enabled(True)
        self.set_armed_kinds(armed_kinds)
        self.set_cross_active(cross_active)

    def clear(self) -> None:
        self.alert = None
        self.title.setText("Visual Alert Review")
        self.alert_text.setText("Waiting for the next ticker alert.")
        self.snapshot.setVisible(False)
        self.queue_label.setText("")
        self._set_actions_enabled(False)
        self.set_armed_kinds(())
        self.set_cross_active(False)

    def set_queued_count(self, count: int) -> None:
        self.queue_label.setText(f"{count} waiting" if count else "queue clear")

    def set_armed_kinds(self, kinds: Iterable[str]) -> None:
        """Reflect this symbol's armed watches; buttons stay clickable so a
        second click disarms."""
        armed = set(kinds)
        for kind, button in self.watch_buttons.items():
            label = WATCH_KINDS[kind]
            is_armed = kind in armed
            button.setText(f"{label} ✓ armed" if is_armed else label)
            button.setChecked(is_armed)

    def set_cross_active(self, active: bool) -> None:
        self.cross_focus_button.setText(self._cross_labels[1 if active else 0])
        self.cross_focus_button.setChecked(bool(active))

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self.remove_today_button,
            self.focus_button,
            self.skip_button,
            self.cross_focus_button,
            *self.watch_buttons.values(),
        ):
            button.setEnabled(enabled)
