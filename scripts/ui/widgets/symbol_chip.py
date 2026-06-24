from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton

from ui import theme


class SymbolChip(QFrame):
    """A ticker chip with an inline remove (×) button. Emits `removed(symbol)`."""

    removed = Signal(str)

    def __init__(self, symbol: str, tone: str = "favorite", parent=None) -> None:
        super().__init__(parent)
        self.symbol = symbol
        self.setObjectName("SymbolChip")

        color = theme.color(tone)
        bg = theme.with_alpha(color, 0.16)
        border = theme.with_alpha(color, 0.55)
        self.setStyleSheet(
            f"""
            QFrame#SymbolChip {{ background: {bg}; border: 1px solid {border}; border-radius: 10px; }}
            QFrame#SymbolChip QLabel {{ color: {color}; font-weight: 650; background: transparent; }}
            QFrame#SymbolChip QToolButton {{
                color: {color}; border: none; background: transparent; font-weight: 700; padding: 0 2px;
            }}
            QFrame#SymbolChip QToolButton:hover {{ color: {theme.color('text_primary')}; }}
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 2, 4, 2)
        layout.setSpacing(4)
        layout.addWidget(QLabel(symbol))
        remove_button = QToolButton()
        remove_button.setText("×")  # ×
        remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_button.setToolTip(f"Remove {symbol}")
        remove_button.clicked.connect(lambda: self.removed.emit(self.symbol))
        layout.addWidget(remove_button)
