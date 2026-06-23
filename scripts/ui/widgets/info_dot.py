from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from ui import theme


class InfoDot(QLabel):
    def __init__(self, tooltip: str, parent=None) -> None:
        super().__init__("i", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip(tooltip)
        self.setFixedSize(18, 18)
        accent = theme.color("accent")
        self.setStyleSheet(
            f"""
            QLabel {{
                border: 1px solid {accent};
                border-radius: 9px;
                color: {accent};
                font-weight: 700;
            }}
            """
        )
