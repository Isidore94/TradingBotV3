from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from project_paths import get_tracker_storage_details, open_path_in_file_manager
from ui.services.maintenance import WarmingService, format_warm_summary
from ui.state import UiState
from ui.widgets.section_header import SectionHeader


THEME_LABELS = {
    "Dark": "dark",
    "Light": "light",
}


class SettingsPanel(QFrame):
    stateChanged = Signal()

    def __init__(self, state: UiState, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.state = state

        self.theme_input = QComboBox()
        self.theme_input.addItems(THEME_LABELS)
        self.theme_input.setCurrentText(_theme_label(self.state.theme_name))
        self.theme_input.currentTextChanged.connect(self._save)

        self.mode_input = QComboBox()
        self.mode_input.addItems(["workspace", "tabs"])
        self.mode_input.setCurrentText(self.state.workspace_mode)
        self.mode_input.currentTextChanged.connect(self._save)

        self.explain_input = QCheckBox("Show inline explanations and extra tooltips")
        self.explain_input.setChecked(self.state.explain_mode)
        self.explain_input.toggled.connect(self._save)

        self.compact_input = QCheckBox("Use compact density")
        self.compact_input.setChecked(self.state.compact_density)
        self.compact_input.toggled.connect(self._save)

        details = get_tracker_storage_details()
        self.data_dir_label = QLabel(details.get("data_dir", ""))
        self.data_dir_label.setWordWrap(True)
        self.source_label = QLabel(details.get("source_label", ""))
        self.source_label.setWordWrap(True)

        open_data_button = QPushButton("Open Data Folder")
        open_data_button.clicked.connect(lambda: open_path_in_file_manager(details["data_dir"]))

        self.warming_service = WarmingService(self)
        self.warming_service.started.connect(self._on_warm_started)
        self.warming_service.finished.connect(self._on_warm_finished)
        self.warming_service.failed.connect(self._on_warm_failed)
        self.warm_button = QPushButton("Warm Durable Stores (daily + H1)")
        self.warm_button.clicked.connect(self.warming_service.warm)
        self.warm_status = QLabel("Pre-fetches bar history so the next scan's cold start only fetches the delta.")
        self.warm_status.setObjectName("MutedLabel")
        self.warm_status.setWordWrap(True)

        form = QFormLayout()
        form.setSpacing(10)
        form.addRow("Theme", self.theme_input)
        form.addRow("Trading Desk mode", self.mode_input)
        form.addRow("Explain mode", self.explain_input)
        form.addRow("Density", self.compact_input)
        form.addRow("Data folder", self.data_dir_label)
        form.addRow("Storage source", self.source_label)

        data_actions = QHBoxLayout()
        data_actions.setContentsMargins(0, 0, 0, 0)
        data_actions.setSpacing(8)
        data_actions.addWidget(open_data_button)
        data_actions.addWidget(self.warm_button)
        data_actions.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(SectionHeader("Settings", "Per-machine presentation and storage settings."))
        layout.addLayout(form)
        layout.addLayout(data_actions)
        layout.addWidget(self.warm_status)
        layout.addStretch(1)

    def _save(self) -> None:
        self.state.theme_name = THEME_LABELS.get(self.theme_input.currentText(), "dark")
        self.state.workspace_mode = self.mode_input.currentText()
        self.state.explain_mode = self.explain_input.isChecked()
        self.state.compact_density = self.compact_input.isChecked()
        self.state.save()
        self.stateChanged.emit()

    def _on_warm_started(self) -> None:
        self.warm_button.setEnabled(False)
        self.warm_status.setText("Warming durable stores… this can take a few minutes for a large watchlist.")

    def _on_warm_finished(self, summary: dict) -> None:
        self.warm_button.setEnabled(True)
        self.warm_status.setText(format_warm_summary(summary))

    def _on_warm_failed(self, message: str) -> None:
        self.warm_button.setEnabled(True)
        first_line = message.splitlines()[0] if message else "Warm failed."
        self.warm_status.setText(f"Warm failed: {first_line}")

    def shutdown(self) -> None:
        self.warming_service.shutdown()


def _theme_label(theme_name: str) -> str:
    for label, value in THEME_LABELS.items():
        if value == theme_name:
            return label
    return "Dark"
