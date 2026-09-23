from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout


class FlowLayout(QLayout):
    """A left-to-right wrapping layout (chips flow onto the next line as needed)."""

    def __init__(
        self, parent=None, margin: int = 0, spacing: int = 6, *, fill: bool = False
    ) -> None:
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items: list = []
        # fill: when every item fits on ONE line, stretch them to the whole
        # rect - spare width shared evenly, full height each - instead of
        # leaving dead space right of and below them. Wrapped lines keep their
        # size hints, exactly as without it.
        self._fill = bool(fill)

    def addItem(self, item) -> None:  # noqa: N802 (Qt override)
        self._items.append(item)

    def insertWidget(self, index: int, widget) -> None:  # noqa: N802
        """Place ``widget`` at ``index`` instead of at the end.

        `QLayout` has no generic insert, which is why the Focus board used to
        empty itself and re-add every chip just to put one arrival in the right
        place. Built on `addWidget` so the reparenting and ownership are Qt's
        own; only the item's position in the list is ours.
        """
        self.addWidget(widget)
        item = self._items.pop()
        position = max(0, min(int(index), len(self._items)))
        self._items.insert(position, item)
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        if self._fill:
            return Qt.Orientation.Horizontal | Qt.Orientation.Vertical
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = effective.x()
        y = effective.y()
        line_height = 0
        spacing = self.spacing()

        if self._fill and not test_only and self._fits_one_line(effective, spacing):
            return self._fill_one_line(effective, spacing, rect, margins)

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + spacing
            if next_x - spacing > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + spacing
                next_x = x + hint.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())

        return y + line_height - rect.y() + margins.bottom()

    def _fits_one_line(self, effective, spacing: int) -> bool:
        if not self._items:
            return False
        total = sum(item.sizeHint().width() for item in self._items)
        total += spacing * (len(self._items) - 1)
        return total <= effective.width()

    def _fill_one_line(self, effective, spacing: int, rect, margins) -> int:
        hints = [item.sizeHint() for item in self._items]
        spare = effective.width() - sum(h.width() for h in hints)
        spare -= spacing * (len(self._items) - 1)
        extra, remainder = divmod(max(0, spare), len(self._items))
        height = max(effective.height(), max(h.height() for h in hints))
        x = effective.x()
        for position, (item, hint) in enumerate(zip(self._items, hints)):
            width = hint.width() + extra + (1 if position < remainder else 0)
            item.setGeometry(QRect(x, effective.y(), width, height))
            x += width + spacing
        return height - rect.y() + effective.y() + margins.bottom()
