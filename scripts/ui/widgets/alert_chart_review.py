from __future__ import annotations

"""Automatic D1/M5 visual review surface for Alert Center alerts."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui.models.bounce import BounceAlert
from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget


class AlertChartReview(QWidget):
    removeTodayRequested = Signal(object)
    focusRequested = Signal(object)
    skipRequested = Signal(object)

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

        buttons = QHBoxLayout()
        buttons.addWidget(self.remove_today_button)
        buttons.addWidget(self.focus_button)
        buttons.addWidget(self.skip_button)
        buttons.addStretch(1)
        buttons.addWidget(self.queue_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        layout.addWidget(self.title)
        layout.addWidget(self.alert_text)
        layout.addWidget(self.snapshot, 1)
        layout.addLayout(buttons)
        self._set_actions_enabled(False)

    def set_alert(
        self,
        alert: BounceAlert,
        *,
        bot=None,
        focus_category: str = "m5",
        queued: int = 0,
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

    def clear(self) -> None:
        self.alert = None
        self.title.setText("Visual Alert Review")
        self.alert_text.setText("Waiting for the next ticker alert.")
        self.snapshot.setVisible(False)
        self.queue_label.setText("")
        self._set_actions_enabled(False)

    def set_queued_count(self, count: int) -> None:
        self.queue_label.setText(f"{count} waiting" if count else "queue clear")

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (self.remove_today_button, self.focus_button, self.skip_button):
            button.setEnabled(enabled)
