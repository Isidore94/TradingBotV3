from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from project_paths import get_tracker_storage_details, open_path_in_file_manager
from ui import theme
from ui.services.maintenance import WarmingService, format_warm_summary
from ui.state import UiState
from ui.widgets.section_header import SectionHeader


THEME_LABELS = {
    "Dark": "dark",
    "Light": "light",
}

# Scales the whole shell - type, padding, row height, and the panel minimum
# widths that decide whether a column can shrink. Auto reads the screen, which
# is the point: a 4K desktop and a 1680px laptop cannot share one layout.
UI_SCALE_LABELS = {
    "Auto (fit this screen)": "auto",
    "80% - smallest": "0.80",
    "85%": "0.85",
    "90%": "0.90",
    "95%": "0.95",
    "100% - desktop": "1.00",
    "110%": "1.10",
    "125% - largest": "1.25",
}


class SettingsPanel(QFrame):
    stateChanged = Signal()

    def __init__(
        self,
        state: UiState,
        bounce_service=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.state = state
        self.bounce_service = bounce_service

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

        self.ui_scale_input = QComboBox()
        self.ui_scale_input.addItems(UI_SCALE_LABELS)
        self.ui_scale_input.setCurrentText(_ui_scale_label(self.state.ui_scale))
        self.ui_scale_input.currentTextChanged.connect(self._save)
        self.ui_scale_hint = QLabel()
        self.ui_scale_hint.setObjectName("MutedLabel")
        self.ui_scale_hint.setWordWrap(True)
        self._sync_scale_hint()

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
        form.addRow("UI scale", self.ui_scale_input)
        form.addRow("", self.ui_scale_hint)
        form.addRow("Data folder", self.data_dir_label)
        form.addRow("Storage source", self.source_label)

        data_actions = QHBoxLayout()
        data_actions.setContentsMargins(0, 0, 0, 0)
        data_actions.setSpacing(8)
        data_actions.addWidget(open_data_button)
        data_actions.addWidget(self.warm_button)
        data_actions.addStretch(1)

        general_page = QFrame()
        general_page.setObjectName("Panel")
        general_layout = QVBoxLayout(general_page)
        general_layout.setContentsMargins(12, 12, 12, 12)
        general_layout.setSpacing(10)
        general_layout.addWidget(
            SectionHeader("General", "Per-machine presentation and durable storage.")
        )
        general_layout.addLayout(form)
        general_layout.addLayout(data_actions)
        general_layout.addWidget(self.warm_status)
        general_layout.addStretch(1)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setDocumentMode(True)
        self.settings_tabs.addTab(_scrollable_tab(general_page), "General")
        if self.bounce_service is not None:
            self.settings_tabs.addTab(_scrollable_tab(self._build_bounce_section()), "BounceBot")
        # Always present: the testing plan is the one page the trader may need
        # at 6am on a morning when nothing else is behaving. Display only - it
        # renders a markdown file and owns no timer, state or engine hook.
        from ui.widgets.testing_plan_view import TestingPlanView

        self.testing_plan_view = TestingPlanView()
        self.settings_tabs.addTab(self.testing_plan_view, "Testing Plan")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(
            SectionHeader(
                "Settings",
                "Presentation, data, and live-engine controls.",
            )
        )
        layout.addWidget(self.settings_tabs, 1)

    def _build_bounce_section(self) -> QFrame:
        """Operational BounceBot controls, moved off the Trading Desk so the
        desk stays alert-first: connection management, RRS tuning, and the
        per-type bounce toggles."""
        from ui.panels.bounce_panel import BOUNCE_TOGGLE_ORDER
        from ui.services.bounce_service import load_bounce_config

        service = self.bounce_service
        config = load_bounce_config()
        section = QFrame()
        section.setObjectName("Panel")
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(12, 12, 12, 12)
        section_layout.setSpacing(10)
        section_layout.addWidget(
            SectionHeader("BounceBot", "Connection, RRS tuning, and bounce-type toggles.")
        )

        connect_row = QHBoxLayout()
        connect_row.setSpacing(8)
        connect_button = QPushButton("Connect")
        connect_button.clicked.connect(service.start)
        disconnect_button = QPushButton("Disconnect")
        disconnect_button.clicked.connect(service.stop)
        reconnect_button = QPushButton("Reconnect")
        reconnect_button.clicked.connect(service.restart)
        for button in (connect_button, disconnect_button, reconnect_button):
            connect_row.addWidget(button)
        connect_row.addStretch(1)
        section_layout.addLayout(connect_row)

        tuning_form = QFormLayout()
        tuning_form.setSpacing(8)
        self.rrs_threshold_input = QDoubleSpinBox()
        self.rrs_threshold_input.setRange(0.0, 5.0)
        self.rrs_threshold_input.setDecimals(1)
        self.rrs_threshold_input.setSingleStep(0.1)
        self.rrs_threshold_input.setValue(service.rrs_threshold)
        self.rrs_threshold_input.valueChanged.connect(service.set_rrs_threshold)
        tuning_form.addRow("RRS sensitivity", self.rrs_threshold_input)

        self.timeframe_input = QComboBox()
        for key, item in config["rrs_timeframes"].items():
            self.timeframe_input.addItem(str(item.get("label", key)), key)
        self.timeframe_input.setCurrentIndex(
            max(0, self.timeframe_input.findData(service.rrs_timeframe_key))
        )
        self.timeframe_input.currentIndexChanged.connect(
            lambda _index: service.set_rrs_timeframe(str(self.timeframe_input.currentData() or ""))
        )
        tuning_form.addRow("RRS timeframe", self.timeframe_input)
        section_layout.addLayout(tuning_form)

        toggles_label = QLabel(
            "Bounce types (defaults are evidence-based from the outcome tracker; "
            "disabled types keep recording as learning-only)."
        )
        toggles_label.setObjectName("MutedLabel")
        toggles_label.setWordWrap(True)
        section_layout.addWidget(toggles_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        labels = config["bounce_type_labels"]
        defaults = service.bounce_type_settings
        self.bounce_toggle_boxes: dict[str, QCheckBox] = {}
        for index, key in enumerate(BOUNCE_TOGGLE_ORDER):
            if key not in defaults:
                continue
            checkbox = QCheckBox(str(labels.get(key, key)))
            checkbox.setChecked(bool(defaults.get(key)))
            checkbox.toggled.connect(
                lambda checked, bounce_key=key: service.set_bounce_type_enabled(bounce_key, checked)
            )
            self.bounce_toggle_boxes[key] = checkbox
            grid.addWidget(checkbox, index // 4, index % 4)
        section_layout.addLayout(grid)
        return section

    def _save(self) -> None:
        self.state.theme_name = THEME_LABELS.get(self.theme_input.currentText(), "dark")
        self.state.workspace_mode = self.mode_input.currentText()
        self.state.explain_mode = self.explain_input.isChecked()
        self.state.compact_density = self.compact_input.isChecked()
        self.state.ui_scale = UI_SCALE_LABELS.get(
            self.ui_scale_input.currentText(), "auto"
        )
        self.state.save()
        self._sync_scale_hint()
        self.stateChanged.emit()

    def _sync_scale_hint(self) -> None:
        """Say what the scale resolves to, and that panes keep their own splits.

        Auto is a screen read, so the number it lands on is worth showing -
        otherwise "Auto" is the one option whose effect the trader cannot see.
        """
        screen = _screen_size()
        resolved = theme.resolve_scale(self.state.ui_scale, screen)
        source = "auto" if self.state.ui_scale == "auto" else "set here"
        self.ui_scale_hint.setText(
            f"Scaling type, spacing, and panel minimum widths to {resolved:.0%} "
            f"({source}; this screen offers {screen[0]}x{screen[1]}). Splits you have "
            "dragged are remembered separately and are not reset by this."
        )

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


def _ui_scale_label(value: str) -> str:
    for label, stored in UI_SCALE_LABELS.items():
        if stored == value:
            return label
    return "Auto (fit this screen)"


def _scrollable_tab(content: QWidget) -> QScrollArea:
    """Give each settings category its own viewport instead of compressing it."""
    scroll = QScrollArea()
    scroll.setObjectName("SettingsTabScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(content)
    return scroll


def _screen_size() -> tuple[int, int]:
    app = QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return (2560, 1440)
    size = screen.availableGeometry()
    return (size.width(), size.height())
