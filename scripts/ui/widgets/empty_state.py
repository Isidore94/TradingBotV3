from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class EmptyState(QWidget):
    def __init__(self, title: str, message: str = "", action_text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label = QLabel(message)
        self.message_label.setObjectName("MutedLabel")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setWordWrap(True)
        self.action_button = QPushButton(action_text)
        self.action_button.setObjectName("PrimaryButton")
        self.action_button.setVisible(bool(action_text))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)
        layout.addStretch(1)
        layout.addWidget(self.title_label)
        if message:
            layout.addWidget(self.message_label)
        layout.addWidget(self.action_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
