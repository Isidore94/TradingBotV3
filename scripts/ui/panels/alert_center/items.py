"""Alert Center feed-row widgets: the clickable wrapper around one `AlertFeedItem`."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout

from ui.models.bounce import BounceAlert
from ui.widgets.alert_feed_item import AlertFeedItem


class _ClickableItem(QFrame):
    clicked = Signal(object)
    favoriteToggled = Signal(object)  # alert
    dislikeRequested = Signal(object)  # alert
    symbolClicked = Signal(object)  # alert - ticker name click -> chart snapshot

    def __init__(
        self,
        alert: BounceAlert,
        *,
        focus_category: str = "",
        show_favorite_button: bool = False,
        favorite_hint: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.alert = alert
        feed_item = AlertFeedItem(
            alert,
            focus_category=focus_category,
            show_favorite_button=show_favorite_button,
            favorite_hint=favorite_hint,
        )
        feed_item.favoriteToggled.connect(lambda: self.favoriteToggled.emit(self.alert))
        feed_item.dislikeRequested.connect(lambda: self.dislikeRequested.emit(self.alert))
        feed_item.symbolClicked.connect(lambda: self.symbolClicked.emit(self.alert))
        self.feed_item = feed_item
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(feed_item)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_repeat_count(
        self,
        count: int,
        *,
        latest_trigger: str = "",
        latest_alert: BounceAlert | None = None,
    ) -> None:
        """Forward R4 section 6.3's fold to the row this wrapper contains.

        This class wraps an ``AlertFeedItem`` rather than subclassing it, so
        the repeat badge has to be forwarded explicitly - the feed only ever
        holds wrappers, so without this the fold silently fails over to a new
        row and the whole control does nothing.
        """
        self.feed_item.set_repeat_count(
            count,
            latest_trigger=latest_trigger,
            latest_alert=latest_alert,
        )

    @property
    def repeat_badge(self):
        return self.feed_item.repeat_badge

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.clicked.emit(self.alert)
        super().mousePressEvent(event)
