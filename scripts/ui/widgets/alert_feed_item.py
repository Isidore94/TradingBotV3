from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QHBoxLayout, QToolButton, QVBoxLayout, QWidget

from ui import theme
from ui.models.bounce import BounceAlert
from ui.widgets.badge import Badge


_FOCUS_BADGE_TEXT = {
    "swing": "★ SWING",
    "m5": "★ M5",
    "both": "★ SWING+M5",
}


class AlertFeedItem(QWidget):
    """One alert row in a feed.

    Focus (liked) names get the heavy treatment - full gold frame plus a
    category badge (★ SWING / ★ M5) - so a handpicked pick is unmissable when
    it fires again. ``show_favorite_button`` adds a star at the right edge of
    every row (a clickable favorite column): hollow ☆ to favorite the pick,
    lit gold ★ to unfavorite. The hosting panel decides the category (D1/H1
    alerts -> Swing, intraday -> M5) and wires it into Focus Picks. Next to it,
    ✕ records a dislike: the panel prompts for the reason and logs it to the
    AI-reviewable pick-feedback file.
    """

    favoriteToggled = Signal()
    dislikeRequested = Signal()

    def __init__(
        self,
        alert: BounceAlert,
        parent=None,
        *,
        focus_category: str = "",
        show_favorite_button: bool = False,
        favorite_hint: str = "",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        tone = "long" if alert.side == "LONG" else "short" if alert.side == "SHORT" else "neutral"
        is_focus = bool(focus_category)
        if is_focus:
            # Liked picks: gold frame all the way around, not just a stripe.
            accent = theme.color("favorite")
            self.setStyleSheet(
                f"QWidget#Panel {{ border: 1px solid {theme.with_alpha(accent, 0.85)}; "
                f"border-left: 4px solid {accent}; "
                f"background: {theme.with_alpha(accent, 0.14)}; }}"
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
            top.addWidget(Badge(_FOCUS_BADGE_TEXT.get(focus_category, "★ FOCUS"), "favorite"))
        top.addWidget(Badge(alert.side, tone))
        if alert.timeframe:
            top.addWidget(Badge(alert.timeframe, "info"))
        top.addStretch(1)
        if show_favorite_button and alert.symbol:
            star = QToolButton()
            star.setText("★" if is_focus else "☆")
            if is_focus:
                star.setToolTip(f"Unfavorite {alert.symbol}: remove it from Focus Picks.")
            else:
                star.setToolTip(
                    f"Favorite {alert.symbol}{f' into {favorite_hint}' if favorite_hint else ''}: "
                    "its alerts flag gold, skip the tier filter, and sound."
                )
            star.setCursor(Qt.CursorShape.PointingHandCursor)
            color = theme.color("favorite") if is_focus else theme.with_alpha(theme.color("favorite"), 0.55)
            star.setStyleSheet(
                f"QToolButton {{ color: {color}; font-weight: 700; font-size: 14pt; padding: 0 4px; }}"
            )
            star.clicked.connect(self.favoriteToggled.emit)
            top.addWidget(star)

            dislike = QToolButton()
            dislike.setText("✕")
            dislike.setToolTip(
                f"Dislike {alert.symbol}: you'll be asked why, and the reason is logged to "
                "pick_feedback.jsonl for AI review (also removes it from Focus Picks if starred)."
            )
            dislike.setCursor(Qt.CursorShape.PointingHandCursor)
            dislike.setStyleSheet(
                f"QToolButton {{ color: {theme.with_alpha(theme.color('short'), 0.65)}; "
                "font-weight: 700; font-size: 12pt; padding: 0 4px; }"
            )
            dislike.clicked.connect(self.dislikeRequested.emit)
            top.addWidget(dislike)

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
