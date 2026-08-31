"""A titled section the trader opens when they want it, closed by default.

Built to host the M5 Strength Board under the Desk's Strength window (trader,
2026-08-31). The constraint that shapes it is the desk's width budget: the
alert column has a 360 px floor and the charts own everything left of it, so a
section that claimed height or width on startup would take it from the chart
the trader is actually reading. Collapsed, this contributes one header row and
nothing else - the content widget is hidden, so it asks the layout for no
space at all and its minimum width never reaches the column.

No stylesheet of its own. The header is a plain `QToolButton` and its label a
`SectionTitle`, both already styled in `theme.qss` - the rule is that widget
variants live there keyed on object names, never in a `setStyleSheet` call.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """One header row; the content below it appears only when asked for."""

    #: True when the section has just been opened.
    toggled = Signal(bool)

    def __init__(self, title: str, parent=None, *, expanded: bool = False) -> None:
        super().__init__(parent)
        self._content: QWidget | None = None
        self._title = str(title)

        self.header = QToolButton(self)
        self.header.setObjectName("CollapsibleSectionHeader")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(False)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.ArrowType.RightArrow)
        self.header.setToolTip(self._title)
        # The header must never be the reason the alert column gets wider. A
        # QToolButton demands its whole text (315 px measured for this title
        # under `theme.qss`), which alone would have pushed the column past
        # the 360 px floor the charts are sized against. Ignored horizontally
        # + elided text means it takes the width it is given and says as much
        # of the title as fits.
        self.header.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.header.clicked.connect(self._on_header_clicked)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._layout.addWidget(self.header)

        if expanded:
            self.set_expanded(True)

    # ------------------------------------------------------------------ views
    def content(self) -> QWidget | None:
        return self._content

    def is_expanded(self) -> bool:
        return bool(self.header.isChecked())

    # ----------------------------------------------------------------- wiring
    def set_content(self, widget: QWidget) -> None:
        """Give the section its one body widget.

        Called once by the host. Hidden immediately unless the section is
        already open, so a body built at startup costs a `setVisible(False)`
        and never a layout pass of its own.
        """
        if self._content is not None:
            self._layout.removeWidget(self._content)
            self._content.setParent(None)
        self._content = widget
        self._layout.addWidget(widget, 1)
        widget.setVisible(self.is_expanded())

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        if expanded == self.is_expanded() and (
            self._content is None or self._content.isVisible() == expanded
        ):
            return
        self.header.setChecked(expanded)
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        if self._content is not None:
            self._content.setVisible(expanded)
        self.toggled.emit(expanded)

    # ------------------------------------------------------------------ slots
    def _on_header_clicked(self, checked: bool) -> None:
        self.set_expanded(checked)

    # ------------------------------------------------------------------ paint
    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._apply_elided_title()

    def _apply_elided_title(self) -> None:
        """Fit the title to the width the column actually gave the header.

        Cheap by construction: `QFontMetrics.elidedText` on one short string,
        and the text is only written back when it changed, so a resize storm
        does not turn into a relayout storm on the Qt thread.
        """
        # Room for the arrow indicator plus the button's own padding.
        available = max(0, self.header.width() - 48)
        text = QFontMetrics(self.header.font()).elidedText(
            self._title, Qt.TextElideMode.ElideRight, available
        )
        if text != self.header.text():
            self.header.setText(text)
