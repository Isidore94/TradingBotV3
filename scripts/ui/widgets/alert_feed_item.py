from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QWidget

from ui import theme
from ui.models.bounce import BounceAlert
from ui.widgets.badge import Badge


class AlertFeedItem(QWidget):
    def __init__(self, alert: BounceAlert, parent=None, *, is_focus: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        tone = "long" if alert.side == "LONG" else "short" if alert.side == "SHORT" else "neutral"
        if is_focus:
            # Gold left stripe so handpicked names jump out of the feed.
            accent = theme.color("favorite")
            self.setStyleSheet(
                f"QWidget#Panel {{ border-left: 3px solid {accent}; "
                f"background: {theme.with_alpha(accent, 0.08)}; }}"
            )

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        time_label = QLabel(alert.time_text)
        time_label.setObjectName("MutedLabel")
        symbol_label = QLabel(alert.symbol or "Alert")
        symbol_label.setStyleSheet("font-weight: 700;")
        top.addWidget(time_label)
        top.addWidget(symbol_label)
        if is_focus:
            top.addWidget(Badge("★ FOCUS", "favorite"))
        top.addWidget(Badge(alert.side, tone))
        if alert.timeframe:
            top.addWidget(Badge(alert.timeframe, "info"))
        top.addStretch(1)

        trigger = QLabel(alert.trigger or alert.raw_text)
        trigger.setWordWrap(True)
        trigger.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)
        layout.addLayout(top)
        layout.addWidget(trigger)
        if alert.context:
            context = QLabel(alert.context)
            context.setObjectName("MutedLabel")
            context.setWordWrap(True)
            layout.addWidget(context)
