from __future__ import annotations

from PySide6.QtCore import Signal
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
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
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
        desk_link_service=None,
        desk_link_feed=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.state = state
        self.bounce_service = bounce_service
        self.desk_link_service = desk_link_service
        # Present only on a satellite desk (--satellite-desk). The pairing rows
        # below are shown either way: on a normal desk they save the connection
        # that a later satellite launch picks up.
        self.desk_link_feed = desk_link_feed

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(SectionHeader("Settings", "Per-machine presentation, storage, and BounceBot configuration."))
        layout.addLayout(form)
        layout.addLayout(data_actions)
        layout.addWidget(self.warm_status)
        if self.bounce_service is not None:
            layout.addWidget(self._build_bounce_section())
        if self.desk_link_service is not None:
            layout.addWidget(self._build_desk_link_section())
        layout.addStretch(1)

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

    def _build_desk_link_section(self) -> QFrame:
        """Desk Link relay controls (docs/MULTI_MACHINE_DESK_PROPOSAL.md).

        Everything the trader needs to serve satellites without touching
        local_settings.json: the enable toggle applies immediately, the token
        is visible/copyable for the satellite's first launch, and regenerate
        revokes a token by restarting the server.
        """
        service = self.desk_link_service
        section = QFrame()
        section.setObjectName("Panel")
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(12, 12, 12, 12)
        section_layout.setSpacing(10)
        section_layout.addWidget(
            SectionHeader(
                "Desk Link",
                "Serve view-only satellites on your local network with live alert chart popups.",
            )
        )

        self.desk_link_enable_input = QCheckBox("Serve satellites from this machine")
        self.desk_link_enable_input.setChecked(service.running)
        self.desk_link_enable_input.toggled.connect(self._on_desk_link_toggled)

        self.desk_link_port_input = QSpinBox()
        self.desk_link_port_input.setRange(1024, 65535)
        self.desk_link_port_input.setValue(service.configured_port())
        # Applied on editing finished (not per keystroke) so a half-typed
        # port never triggers a server restart.
        self.desk_link_port_input.editingFinished.connect(self._on_desk_link_port_changed)

        self.desk_link_token_view = QLineEdit(service.current_token())
        self.desk_link_token_view.setReadOnly(True)
        self.desk_link_token_view.setPlaceholderText("Generated when serving is first enabled")

        copy_button = QPushButton("Copy token")
        copy_button.clicked.connect(self._copy_desk_link_token)
        regenerate_button = QPushButton("Regenerate token")
        regenerate_button.clicked.connect(self._regenerate_desk_link_token)
        test_popup_button = QPushButton("Send test popup")
        test_popup_button.clicked.connect(self._send_desk_link_test_popup)

        form = QFormLayout()
        form.setSpacing(8)
        form.addRow("Serving", self.desk_link_enable_input)
        form.addRow("Port", self.desk_link_port_input)
        form.addRow("Link token", self.desk_link_token_view)
        section_layout.addLayout(form)

        token_row = QHBoxLayout()
        token_row.setSpacing(8)
        token_row.addWidget(copy_button)
        token_row.addWidget(regenerate_button)
        token_row.addWidget(test_popup_button)
        token_row.addStretch(1)
        section_layout.addLayout(token_row)

        self.desk_link_status = QLabel()
        self.desk_link_status.setObjectName("MutedLabel")
        self.desk_link_status.setWordWrap(True)
        section_layout.addWidget(self.desk_link_status)

        hint = QLabel(
            "On the satellite machine (same repo, no TWS needed): launch "
            "TradingBotV3_SatelliteDesk.cmd and fill in \"Connect to a main desk\" below, "
            "or run python scripts/gui.py --ui qt --satellite <this-machine> --link-token <token> "
            "for the small mirror window. The pairing is remembered after the first connect."
        )
        hint.setObjectName("MutedLabel")
        hint.setWordWrap(True)
        section_layout.addWidget(hint)

        section_layout.addWidget(self._build_main_desk_pairing())

        service.runningChanged.connect(self._refresh_desk_link_status)
        service.satellitesChanged.connect(lambda _machines: self._refresh_desk_link_status())
        self._refresh_desk_link_status()
        return section

    def _build_main_desk_pairing(self) -> QFrame:
        """The satellite half of Desk Link: which main desk THIS machine follows.

        Same host/port/token the satellite window's connect dialog writes, on
        the Settings page instead — so a satellite desk is re-pointed in place
        and never needs the separate popup window. Applies live when this desk
        is running in satellite mode; otherwise it just saves the pairing for
        the next satellite launch.
        """
        from desk_link.server import DEFAULT_PORT
        from ui.satellite import load_saved_connection

        host, port, token = load_saved_connection()

        block = QFrame()
        block.setObjectName("Panel")
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(12, 12, 12, 12)
        block_layout.setSpacing(8)
        block_layout.addWidget(
            SectionHeader(
                "Connect to a main desk",
                "Follow another machine's Desk Link relay instead of TWS on this one.",
            )
        )

        self.main_desk_host_input = QLineEdit(host)
        self.main_desk_host_input.setPlaceholderText("Main desk name or IP, e.g. 192.168.0.223")
        self.main_desk_port_input = QSpinBox()
        self.main_desk_port_input.setRange(1024, 65535)
        self.main_desk_port_input.setValue(port or DEFAULT_PORT)
        self.main_desk_token_input = QLineEdit(token)
        self.main_desk_token_input.setPlaceholderText("Link token - \"Copy token\" on the main desk")
        # Enter in either field connects, the way the dialog's Connect button did.
        self.main_desk_host_input.returnPressed.connect(self._connect_to_main_desk)
        self.main_desk_token_input.returnPressed.connect(self._connect_to_main_desk)

        form = QFormLayout()
        form.setSpacing(8)
        form.addRow("Main desk", self.main_desk_host_input)
        form.addRow("Port", self.main_desk_port_input)
        form.addRow("Link token", self.main_desk_token_input)
        block_layout.addLayout(form)

        connect_button = QPushButton("Connect")
        connect_button.clicked.connect(self._connect_to_main_desk)
        forget_button = QPushButton("Forget this desk")
        forget_button.clicked.connect(self._forget_main_desk)
        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addWidget(connect_button)
        button_row.addWidget(forget_button)
        button_row.addStretch(1)
        block_layout.addLayout(button_row)

        self.main_desk_link_status = QLabel()
        self.main_desk_link_status.setObjectName("MutedLabel")
        self.main_desk_link_status.setWordWrap(True)
        block_layout.addWidget(self.main_desk_link_status)

        if self.desk_link_feed is not None:
            self.desk_link_feed.linkStatusChanged.connect(self._on_main_desk_link_status)
        self._refresh_main_desk_status()
        return block

    def _connect_to_main_desk(self) -> None:
        from ui.satellite import save_connection

        host = self.main_desk_host_input.text().strip()
        token = self.main_desk_token_input.text().strip()
        port = int(self.main_desk_port_input.value())
        if not host or not token:
            self.main_desk_link_status.setText(
                "Enter the main desk's name/IP and its link token (Settings -> Desk Link -> "
                "Copy token, over there)."
            )
            return
        save_connection(host, port, token)
        feed = self.desk_link_feed
        if feed is None:
            self.main_desk_link_status.setText(
                f"Saved {host}:{port}. This desk runs on its own data - relaunch with "
                "TradingBotV3_SatelliteDesk.cmd (--satellite-desk) to be fed by that main."
            )
            return
        # Replace the live link in place: stop() drops the old client so start()
        # is not a no-op when re-pointing at a different main.
        feed.stop()
        feed.start(host=host, port=port, token=token, machine_name=_satellite_machine_name())
        self.main_desk_link_status.setText(f"Connecting to {host}:{port}...")

    def _forget_main_desk(self) -> None:
        from ui.satellite import save_connection

        if self.desk_link_feed is not None:
            self.desk_link_feed.stop()
        save_connection("", int(self.main_desk_port_input.value()), "")
        self.main_desk_host_input.clear()
        self.main_desk_token_input.clear()
        self.main_desk_link_status.setText("Pairing cleared. This desk follows no main.")

    def _on_main_desk_link_status(self, link_state: str, detail: str) -> None:
        texts = {
            "connecting": f"Connecting... ({detail})",
            "connected": f"Connected to {detail} - this desk is fed by that main's relay.",
            "disconnected": f"Link lost - {detail}. Retrying.",
            "rejected": (
                f"Token rejected: {detail}. Copy the current token from the main desk's "
                "Desk Link section and connect again."
            ),
            "stopped": "Not following a main desk.",
        }
        self.main_desk_link_status.setText(texts.get(link_state, f"{link_state}: {detail}"))

    def _refresh_main_desk_status(self) -> None:
        host = self.main_desk_host_input.text().strip()
        if self.desk_link_feed is None:
            self.main_desk_link_status.setText(
                f"Saved: {host}:{int(self.main_desk_port_input.value())}. Used when this machine "
                "launches as a satellite desk; this desk is running on its own data."
                if host
                else "Not paired. Used only when this machine launches as a satellite desk."
            )
            return
        current_status = getattr(self.desk_link_feed, "current_link_status", None)
        if callable(current_status):
            state, detail = current_status()
            if state != "stopped" or host:
                self._on_main_desk_link_status(state, detail)
                return
        self.main_desk_link_status.setText(
            "Satellite desk with no pairing yet - enter the main desk and token, then Connect."
        )

    def _on_desk_link_toggled(self, enabled: bool) -> None:
        service = self.desk_link_service
        ok = service.set_enabled(enabled)
        if enabled and not ok:
            # Port already in use (or bind failed): reflect reality, keep the
            # saved setting off so the next launch does not fail silently.
            service.set_enabled(False)
            self.desk_link_enable_input.blockSignals(True)
            self.desk_link_enable_input.setChecked(False)
            self.desk_link_enable_input.blockSignals(False)
            self.desk_link_status.setText(
                f"Could not serve on port {service.configured_port()} - is another desk already "
                "serving, or the port in use? Change the port and try again."
            )
            return
        self.desk_link_token_view.setText(service.current_token())
        self._refresh_desk_link_status()

    def _on_desk_link_port_changed(self) -> None:
        service = self.desk_link_service
        port = int(self.desk_link_port_input.value())
        if port == service.configured_port():
            return
        if not service.set_port(port):
            self.desk_link_status.setText(f"Port {port} could not be bound; Desk Link is stopped.")
            self.desk_link_enable_input.blockSignals(True)
            self.desk_link_enable_input.setChecked(False)
            self.desk_link_enable_input.blockSignals(False)
            return
        self._refresh_desk_link_status()

    def _copy_desk_link_token(self) -> None:
        token = self.desk_link_service.ensure_token()
        self.desk_link_token_view.setText(token)
        QApplication.clipboard().setText(token)
        self.desk_link_status.setText("Token copied to clipboard - paste it into the satellite's first launch.")

    def _send_desk_link_test_popup(self) -> None:
        if self.desk_link_service.send_test_popup():
            self.desk_link_status.setText(
                "Test popup sent - it should appear on every connected satellite now."
            )
        else:
            self.desk_link_status.setText(
                "No satellite is connected (or serving is off) - nothing to send the test popup to."
            )

    def _regenerate_desk_link_token(self) -> None:
        token = self.desk_link_service.regenerate_token()
        self.desk_link_token_view.setText(token)
        QApplication.clipboard().setText(token)
        self.desk_link_status.setText(
            "New token generated and copied. Connected satellites were disconnected and "
            "need the new token."
        )

    def _refresh_desk_link_status(self) -> None:
        service = self.desk_link_service
        if service is None:
            return
        if not service.running:
            self.desk_link_status.setText("Not serving. Satellites cannot connect.")
            return
        machines = service.connected_machines()
        if machines:
            self.desk_link_status.setText(
                f"Serving on port {service.configured_port()} - connected: {', '.join(machines)}."
            )
        else:
            self.desk_link_status.setText(
                f"Serving on port {service.configured_port()} - no satellites connected yet."
            )

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


def _satellite_machine_name() -> str:
    """Name this machine announces to the main desk (matches app.py's)."""
    import socket

    return socket.gethostname() or "satellite-desk"


def _screen_size() -> tuple[int, int]:
    app = QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return (2560, 1440)
    size = screen.availableGeometry()
    return (size.width(), size.height())
