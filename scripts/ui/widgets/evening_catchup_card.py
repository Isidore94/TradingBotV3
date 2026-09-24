"""The EVENING catch-up card: one modeless window shown on the flip out of EVENING.

Five short sections of rows (strongest longs, weakest shorts, strong on
pullbacks, best swing setups, price alerts that fired). A row click asks for
the board chart door (`symbolClicked`), never the review queue. The content is
built off the Qt thread by `EveningCatchupService`; this only lays it out.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class EveningCatchupCard(QDialog):
    """Modeless and non-activating, like the Trade Mentor popup."""

    symbolClicked = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint,
        )
        self.setObjectName("EveningCatchupPopup")
        self.setWindowTitle("Evening catch-up")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.resize(520, 640)
        outer = QVBoxLayout(self)
        self.title_label = QLabel("")
        self.title_label.setObjectName("SectionTitle")
        self.title_label.setWordWrap(True)
        outer.addWidget(self.title_label)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.hide)
        buttons.addWidget(self.close_button)
        outer.addLayout(buttons)
        self._row_buttons: list[QPushButton] = []

    def row_buttons(self) -> list[QPushButton]:
        return list(self._row_buttons)

    def show_catchup(self, payload: dict[str, Any]) -> None:
        """Lay out one payload from `evening_catchup.build_catchup` and show."""
        payload = payload if isinstance(payload, dict) else {}
        self.title_label.setText(str(payload.get("title") or "Evening catch-up"))
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(4, 4, 4, 4)
        self._row_buttons = []
        for section in payload.get("sections") or []:
            header = QLabel(str(section.get("title") or ""))
            header.setObjectName("SectionTitle")
            layout.addWidget(header)
            rows = section.get("rows") or []
            if not rows:
                empty = QLabel("(none)")
                empty.setObjectName("MutedLabel")
                layout.addWidget(empty)
                continue
            for row in rows:
                symbol = str(row.get("symbol") or "")
                side = str(row.get("side") or "")
                side_text = f" {side.upper()}" if side else ""
                button = QPushButton(f"{symbol}{side_text}  -  {row.get('text') or ''}")
                button.setFlat(True)
                button.setToolTip("Chart it (board chart, not the review queue)")
                button.clicked.connect(
                    lambda _checked=False, s=symbol, d=side: self.symbolClicked.emit(s, d)
                )
                layout.addWidget(button)
                self._row_buttons.append(button)
        layout.addStretch(1)
        self._scroll.setWidget(body)
        self.show()
