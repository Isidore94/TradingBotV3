from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class SectionHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent=None) -> None:
        super().__init__(parent)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionTitle")
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("SectionSubtitle")
        self.subtitle_label.setWordWrap(True)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        text_layout.addWidget(self.title_label)
        if subtitle:
            text_layout.addWidget(self.subtitle_label)

        self.action_layout = QHBoxLayout()
        self.action_layout.setContentsMargins(0, 0, 0, 0)
        self.action_layout.setSpacing(8)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addLayout(text_layout, 1)
        layout.addLayout(self.action_layout)

    def add_action(self, widget: QWidget) -> None:
        self.action_layout.addWidget(widget)
