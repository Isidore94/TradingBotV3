"""The Day Review Show overlay: one slide at a time over the whole window (R1).

It paints a deck the Day Review worker already chose (`day_review_show.desk_deck`)
and computes nothing: Right/Space next, Left back, Esc close, A toggles an
8-second auto-advance. Fonts are set with `QFont` and sized through
`theme.px`; colours come from the one `QFrame#DayReviewShow` block in theme.qss.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PySide6.QtCore import QEvent, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui import theme

AUTO_ADVANCE_MS = 8_000

#: Font family lists, first installed wins (Qt substitutes if none is).
HEADLINE_FAMILIES = ["Bahnschrift SemiBold", "Bahnschrift", "Segoe UI"]
NUMBER_FAMILIES = ["Cascadia Mono SemiBold", "Cascadia Mono", "Consolas"]
PROSE_FAMILIES = ["Georgia", "Times New Roman"]
CAPTION_FAMILIES = ["Segoe UI"]
GLYPH_FAMILIES = ["Segoe UI Emoji"]


def _font(families: list[str], size: float, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setFamilies(families)
    font.setPixelSize(theme.px(size))
    font.setWeight(weight)
    return font


class _TapeLine(QWidget):
    """The session's SPY closes as one line; points are built once."""

    def __init__(self, bars: Sequence[Mapping[str, Any]], parent=None) -> None:
        super().__init__(parent)
        closes: list[float] = []
        for bar in bars or ():
            try:
                closes.append(float(bar.get("close")))
            except (TypeError, ValueError, AttributeError):
                continue
        self._closes = closes
        self.setMinimumHeight(theme.px(120))

    def has_data(self) -> bool:
        return len(self._closes) >= 2

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        if not self.has_data():
            return
        low, high = min(self._closes), max(self._closes)
        span = (high - low) or 1.0
        width, height = max(1, self.width() - 1), max(1, self.height() - 1)
        step = width / (len(self._closes) - 1)
        line = QPolygonF([
            QPointF(index * step, height - (value - low) / span * height)
            for index, value in enumerate(self._closes)
        ])
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self.palette().windowText().color(), theme.px(2)))
        painter.drawPolyline(line)
        painter.end()


class DayReviewShow(QFrame):
    """A full-window slide deck. Emits `closed` when the trader leaves it."""

    closed = Signal()

    def __init__(
        self,
        chosen: Mapping[str, Any],
        *,
        spy_bars: Sequence[Mapping[str, Any]] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        import day_review_show

        self.setObjectName("DayReviewShow")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._glyphs = day_review_show.KIND_GLYPHS
        deck = chosen.get("deck") if isinstance(chosen.get("deck"), Mapping) else {}
        self.slides: list[Mapping[str, Any]] = [
            slide for slide in deck.get("slides") or () if isinstance(slide, Mapping)
        ]
        self.facts_only = bool(chosen.get("facts_only"))
        self.model = str(chosen.get("model") or "")
        self.index = 0

        self._auto = QTimer(self)
        self._auto.setInterval(AUTO_ADVANCE_MS)
        self._auto.timeout.connect(self._auto_step)

        self.glyph = QLabel()
        self.glyph.setFont(_font(GLYPH_FAMILIES, 36))
        self.deck_title = QLabel(str(deck.get("title") or ""))
        self.deck_title.setObjectName("DayReviewShowCaption")
        self.deck_title.setFont(_font(CAPTION_FAMILIES, 15))
        self.badge = QLabel("facts only")
        self.badge.setObjectName("DayReviewShowBadge")
        self.badge.setFont(_font(CAPTION_FAMILIES, 13, QFont.Weight.DemiBold))
        self.badge.setToolTip(str(chosen.get("reason") or ""))
        self.badge.setVisible(self.facts_only)
        self.close_button = QPushButton("Close (Esc)")
        self.close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_button.clicked.connect(self.close_show)
        top = QHBoxLayout()
        top.addWidget(self.glyph)
        top.addWidget(self.deck_title)
        top.addWidget(self.badge)
        top.addStretch(1)
        top.addWidget(self.close_button)

        self.stripe = QFrame()
        self.stripe.setObjectName("DayReviewShowStripe")
        self.stripe.setFixedHeight(theme.px(6))
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setFont(_font(HEADLINE_FAMILIES, 44, QFont.Weight.DemiBold))
        self.stat_value = QLabel()
        self.stat_value.setObjectName("DayReviewShowNumber")
        self.stat_value.setFont(_font(NUMBER_FAMILIES, 80, QFont.Weight.DemiBold))
        self.stat_label = QLabel()
        self.stat_label.setObjectName("DayReviewShowCaption")
        self.stat_label.setFont(_font(CAPTION_FAMILIES, 16))
        self.tape = _TapeLine(spy_bars)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setFont(_font(PROSE_FAMILIES, 24))
        self.lines = QLabel()
        self.lines.setWordWrap(True)
        self.lines.setFont(_font(PROSE_FAMILIES, 19))
        self.footer = QLabel()
        self.footer.setObjectName("DayReviewShowCaption")
        self.footer.setFont(_font(CAPTION_FAMILIES, 14))

        content = QVBoxLayout()
        content.setContentsMargins(theme.px(80), theme.px(24), theme.px(80), theme.px(24))
        content.setSpacing(theme.px(16))
        content.addWidget(self.stripe)
        content.addWidget(self.title)
        content.addWidget(self.stat_value)
        content.addWidget(self.stat_label)
        content.addWidget(self.tape)
        content.addWidget(self.body)
        content.addWidget(self.lines)
        content.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(theme.px(24), theme.px(16), theme.px(24), theme.px(16))
        root.addLayout(top)
        root.addLayout(content, 1)
        root.addWidget(self.footer)
        self._paint()

    # -- the slide ---------------------------------------------------------
    def current(self) -> Mapping[str, Any]:
        return self.slides[self.index] if self.slides else {}

    def _paint(self) -> None:
        slide = self.current()
        kind = str(slide.get("kind") or "")
        self.glyph.setText(self._glyphs.get(kind, ""))
        self.stripe.setProperty("showKind", kind)
        self.stripe.style().unpolish(self.stripe)
        self.stripe.style().polish(self.stripe)
        self.title.setText(str(slide.get("title") or ""))
        stat = slide.get("stat") if isinstance(slide.get("stat"), Mapping) else None
        self.stat_value.setText(str(stat.get("value") or "") if stat else "")
        self.stat_label.setText(str(stat.get("label") or "") if stat else "")
        self.stat_value.setVisible(bool(stat))
        self.stat_label.setVisible(bool(stat))
        self.tape.setVisible(kind == "tape" and self.tape.has_data())
        self.body.setText(str(slide.get("body") or ""))
        self.body.setVisible(bool(self.body.text()))
        lines = [str(line) for line in slide.get("lines") or ()]
        self.lines.setText("\n".join(f"• {line}" for line in lines))
        self.lines.setVisible(bool(lines))
        sources = ", ".join(str(item) for item in slide.get("source_ids") or ()) or "none"
        for widget in (self.title, self.body, self.stat_value, self.lines):
            widget.setToolTip(f"Sources: {sources}")
        teller = "facts only" if self.facts_only else f"told by {self.model or 'the local model'}"
        auto = " - auto" if self._auto.isActive() else ""
        self.footer.setText(
            f"{self.index + 1 if self.slides else 0}/{len(self.slides)} - {teller} - "
            f"sources on hover{auto}"
        )

    def step(self, delta: int) -> None:
        if not self.slides:
            return
        self.index = max(0, min(len(self.slides) - 1, self.index + delta))
        self._paint()

    def _auto_step(self) -> None:
        if self.index >= len(self.slides) - 1:
            self._auto.stop()
            self._paint()
            return
        self.step(1)

    def toggle_auto(self) -> None:
        if self._auto.isActive():
            self._auto.stop()
        else:
            self._auto.start()
        self._paint()

    def close_show(self) -> None:
        self._auto.stop()
        self.hide()
        self.closed.emit()

    # -- keys and the window -----------------------------------------------
    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Space):
            self.step(1)
        elif key == Qt.Key.Key_Left:
            self.step(-1)
        elif key == Qt.Key.Key_Escape:
            self.close_show()
        elif key == Qt.Key.Key_A:
            self.toggle_auto()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def cover(self, window: QWidget) -> None:
        """Lie over `window` whole, and follow it when it resizes."""
        self.setParent(window)
        self.setGeometry(window.rect())
        window.installEventFilter(self)
        self.show()
        self.raise_()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def eventFilter(self, watched, event):  # noqa: N802 (Qt override)
        if watched is self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(watched.rect())
        return super().eventFilter(watched, event)


__all__ = ["AUTO_ADVANCE_MS", "DayReviewShow"]
