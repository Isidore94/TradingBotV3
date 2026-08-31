from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QHBoxLayout, QToolButton, QVBoxLayout, QWidget

from chart_watch import D1_EVENT_KINDS, D1_LEVEL_KINDS, WATCH_KINDS
from ui.models.bounce import BounceAlert, is_chart_watch_alert
from ui.widgets.badge import Badge


_FOCUS_BADGE_TEXT = {
    "swing": "★ SWING",
    "m5": "★ M5",
    "both": "★ SWING+M5",
}


class _SymbolLabel(QLabel):
    """The ticker name as a click target for the D1+M5 snapshot popup.

    Accepts the press so it does not bubble to the row (whose click opens the
    setup detail) - ticker click and row click stay two distinct actions.
    """

    clicked = Signal()

    def __init__(self, symbol: str, parent=None) -> None:
        super().__init__(symbol, parent)
        # Styled from theme.qss by object name: this label is built once per
        # alert row and the feed rebuilds up to MAX_FEED_ITEMS of them at once.
        self.setObjectName("AlertSymbolLink")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{symbol}: D1 + M5 snapshot chart")

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        event.accept()
        self.clicked.emit()


class AlertFeedItem(QWidget):
    """One alert row in a feed.

    Focus (liked) names get the heavy treatment - full gold frame plus a
    category badge (★ SWING / ★ M5) - so a handpicked pick is unmissable when
    it fires again. ``show_favorite_button`` adds a star at the right edge of
    every row (a clickable favorite column): hollow ☆ to favorite the pick,
    lit gold ★ to unfavorite. The hosting panel decides the category (D1/H1
    alerts -> Swing, intraday -> M5) and wires it into Focus Picks. Next to it,
    ✕ records a dislike, removes the symbol from this visual feed, and logs it
    to the AI-reviewable pick-feedback file.
    """

    favoriteToggled = Signal()
    dislikeRequested = Signal()
    symbolClicked = Signal()

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
        is_watch_hit = is_chart_watch_alert(alert)
        if is_watch_hit:
            # A user-armed chart watch fired: full red frame - it outranks
            # even the gold focus treatment because the trader set this exact
            # alarm and is waiting on it.
            self.setProperty("alertKind", "watch")
        elif is_focus:
            # Liked picks: gold frame all the way around, not just a stripe.
            self.setProperty("alertKind", "focus")

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        time_label = QLabel(alert.time_text)
        time_label.setObjectName("MutedLabel")
        if alert.symbol:
            symbol_label = _SymbolLabel(alert.symbol)
            symbol_label.clicked.connect(self.symbolClicked.emit)
        else:
            symbol_label = QLabel("Alert")
            symbol_label.setObjectName("AlertSymbolPlain")
        top.addWidget(time_label)
        top.addWidget(symbol_label)
        # R4 section 6.3: a repeat updates this row in place and shows how many
        # times the name has come back, instead of stacking another row. The
        # time_label above deliberately keeps FIRST-seen time - "when did this
        # start" is the question a repeat count makes worth asking.
        self.repeat_badge = QLabel("")
        self.repeat_badge.setObjectName("MutedLabel")
        self.repeat_badge.setVisible(False)
        top.addWidget(self.repeat_badge)
        if is_watch_hit:
            kind = str((alert.payload or {}).get("chart_watch_kind") or "")
            label = (
                WATCH_KINDS.get(kind)
                or D1_LEVEL_KINDS.get(kind)
                or D1_EVENT_KINDS.get(kind)
                or "Chart watch"
            )
            top.addWidget(Badge(label.upper(), "short"))
        if is_focus:
            top.addWidget(Badge(_FOCUS_BADGE_TEXT.get(focus_category, "★ FOCUS"), "favorite"))
        top.addWidget(Badge(alert.side, tone))
        if alert.timeframe:
            top.addWidget(Badge(alert.timeframe, "info"))
        top.addStretch(1)
        # R4 section 6.2: the one verb on this row that places membership used
        # to be a bare glyph. It keeps the same semantics and the same signal -
        # only the words change, so it reads as the action it always was.
        #
        # This is NOT the CaptureRail LIKE on the chart pane. That one is
        # analysis-only and never writes Focus; this one is placement.
        self.favorite_button = None
        if show_favorite_button and alert.symbol:
            bucket = favorite_hint or "Focus"
            star = QToolButton()
            if is_focus:
                star.setText(f"★ In {bucket}")
                star.setToolTip(
                    f"{alert.symbol} is in {bucket}. Click to remove it from "
                    "Focus Picks."
                )
            else:
                star.setText(f"☆ Like → {bucket}")
                star.setToolTip(
                    f"Like {alert.symbol} into {bucket}: its alerts flag gold, "
                    "skip the tier filter, and sound."
                )
            star.setCursor(Qt.CursorShape.PointingHandCursor)
            star.setObjectName("AlertFavoriteButton")
            star.setProperty("focusOn", "true" if is_focus else "false")
            star.clicked.connect(self.favoriteToggled.emit)
            self.favorite_button = star
            top.addWidget(star)

            dislike = QToolButton()
            dislike.setText("✕")
            dislike.setToolTip(
                f"Dislike {alert.symbol}: you'll be asked why, and the reason is logged to "
                "pick_feedback.jsonl for AI review. The symbol is then removed from today's "
                "Alert Center review and removed from Focus Picks if starred."
            )
            dislike.setCursor(Qt.CursorShape.PointingHandCursor)
            dislike.setObjectName("AlertDislikeButton")
            dislike.clicked.connect(self.dislikeRequested.emit)
            top.addWidget(dislike)

        trigger = QLabel(alert.trigger or alert.raw_text)
        trigger.setWordWrap(True)
        trigger.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if is_watch_hit:
            # The requested red-font flag for a fired chart watch.
            trigger.setObjectName("AlertTriggerWatch")
        self.trigger_label = trigger

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

    def set_repeat_count(self, count: int, *, latest_trigger: str = "") -> None:
        """Fold a repeat into this row (R4 section 6.3).

        Display only: the alert itself is already in the feed's backing list,
        in History, and in whatever the AWAY push reads. This just stops a
        second row from appearing for a name the trader is already looking at.

        The first-seen time in the row header is deliberately left alone -
        "since when has this name been coming back" is the question a repeat
        count exists to answer.
        """
        try:
            count = int(count)
        except (TypeError, ValueError):
            return
        if count <= 1:
            self.repeat_badge.setVisible(False)
            return
        self.repeat_badge.setText(f"×{count}")
        self.repeat_badge.setToolTip(
            f"This name has alerted {count} times today. The row keeps its "
            "first-seen time; nothing was dropped - every hit is still in "
            "History and the evidence log."
        )
        self.repeat_badge.setVisible(True)
        latest_trigger = str(latest_trigger or "").strip()
        if latest_trigger:
            self.trigger_label.setText(latest_trigger)
