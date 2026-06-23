from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui import theme


class KpiTile(QWidget):
    def __init__(self, label: str, value: str = "0", tone: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("TitleLabel")
        if tone:
            self.value_label.setStyleSheet(f"color: {theme.color(tone)};")
        self.label_label = QLabel(label)
        self.label_label.setObjectName("MutedLabel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)
        layout.addWidget(self.value_label)
        layout.addWidget(self.label_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)
