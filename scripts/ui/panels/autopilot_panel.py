from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ui.services.autopilot_service import AutopilotService


class AutopilotPanel(QFrame):
    """Auto Pilot (mini PC mode): one big ON/OFF switch plus enough live
    status + activity log to check on the bot from work or the kitchen."""

    statusChanged = Signal(str)

    def __init__(self, bounce_service, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.service = AutopilotService(bounce_service, parent=self)

        title = QLabel("Auto Pilot - Mini PC Mode")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "Unattended trading day: swing scans at open+1h then hourly from the first full hour "
            "(tracker writes in the final-hour runs), self-built longs/shorts from the open's gaps "
            "and RS/RW vs SPY, near-HOD adds on regime pauses, and a phone digest on the shared Drive."
        )
        subtitle.setWordWrap(True)

        self.toggle_button = QPushButton("AUTO PILOT: OFF")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setMinimumHeight(44)
        self.toggle_button.clicked.connect(self._on_toggle)

        self.reconnect_button = QPushButton("Reconnect IB Now")
        self.reconnect_button.clicked.connect(self.service.force_reconnect)
        self.scan_now_button = QPushButton("Run Swing Scan Now")
        self.scan_now_button.clicked.connect(self.service.run_swing_scan_now)
        self.rebuild_button = QPushButton("Rebuild Watchlists Now")
        self.rebuild_button.clicked.connect(self.service.rebuild_watchlists_now)
        self.universe_button = QPushButton("Rebuild Universe Now")
        self.universe_button.clicked.connect(self.service.rebuild_universe_now)
        self.report_button = QPushButton("Write Report Now")
        self.report_button.clicked.connect(self.service.write_report_now)

        self.ib_value = QLabel("unknown")
        self.regime_value = QLabel("unknown")
        self.next_slot_value = QLabel("-")
        self.slots_value = QLabel("-")
        self.watchlist_value = QLabel("-")
        self.universe_value = QLabel("-")
        self.wrapup_value = QLabel("-")
        self.report_value = QLabel("-")
        self.report_value.setWordWrap(True)

        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(18)
        status_grid.setVerticalSpacing(6)
        for row, (label_text, value_label) in enumerate(
            (
                ("IB connection", self.ib_value),
                ("Market regime", self.regime_value),
                ("Next swing slot", self.next_slot_value),
                ("Slots done", self.slots_value),
                ("Watchlists", self.watchlist_value),
                ("Universe", self.universe_value),
                ("After-close wrap-up", self.wrapup_value),
                ("Away report", self.report_value),
            )
        ):
            key = QLabel(label_text)
            key.setObjectName("MutedLabel")
            status_grid.addWidget(key, row, 0, Qt.AlignmentFlag.AlignTop)
            status_grid.addWidget(value_label, row, 1)
        status_grid.setColumnStretch(1, 1)

        buttons = QHBoxLayout()
        buttons.addWidget(self.reconnect_button)
        buttons.addWidget(self.scan_now_button)
        buttons.addWidget(self.rebuild_button)
        buttons.addWidget(self.universe_button)
        buttons.addWidget(self.report_button)
        buttons.addStretch(1)

        log_title = QLabel("Activity log")
        log_title.setObjectName("SectionTitle")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("Auto Pilot activity shows up here (also written to logs/autopilot.log).")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.toggle_button)
        layout.addLayout(status_grid)
        layout.addLayout(buttons)
        layout.addWidget(log_title)
        layout.addWidget(self.log_view, 1)

        self.service.logMessage.connect(self._append_log)
        self.service.enabledChanged.connect(self._sync_toggle)
        self.service.statusChanged.connect(self._apply_status)
        for line in self.service.log_lines():
            self.log_view.appendPlainText(line)
        self._sync_toggle(self.service.enabled)
        self._apply_status(self.service.status_snapshot())

        # Keep the status row fresh even between service ticks.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(5000)
        self._refresh_timer.timeout.connect(self._refresh_status)
        self._refresh_timer.start()

    # ------------------------------------------------------------------
    @Slot()
    def _on_toggle(self) -> None:
        self.service.set_enabled(self.toggle_button.isChecked())

    @Slot(bool)
    def _sync_toggle(self, enabled: bool) -> None:
        self.toggle_button.blockSignals(True)
        self.toggle_button.setChecked(enabled)
        self.toggle_button.setText("AUTO PILOT: ON" if enabled else "AUTO PILOT: OFF")
        self.toggle_button.blockSignals(False)
        self.statusChanged.emit(f"Auto Pilot {'on' if enabled else 'off'}")

    @Slot(str)
    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    @Slot()
    def _refresh_status(self) -> None:
        self._apply_status(self.service.status_snapshot())

    @Slot(dict)
    def _apply_status(self, snapshot: dict) -> None:
        ib_text = str(snapshot.get("ib_status", "unknown"))
        self.ib_value.setText(ib_text)
        self.ib_value.setStyleSheet(
            "color: #58C777;" if ib_text.startswith("connected") else "color: #E06C75;"
        )
        self.regime_value.setText(str(snapshot.get("regime", "unknown")))
        scan_note = " (scan running)" if snapshot.get("scan_running") else ""
        self.next_slot_value.setText(f"{snapshot.get('next_slot') or '(none left today)'}{scan_note}")
        done = snapshot.get("slots_done") or []
        self.slots_value.setText(", ".join(done) if done else "(none yet)")
        built = snapshot.get("watchlist_built_at") or "not built today"
        self.watchlist_value.setText(
            f"{snapshot.get('longs_count', 0)} longs / {snapshot.get('shorts_count', 0)} shorts "
            f"+ bot picks {snapshot.get('auto_longs_count', 0)}/{snapshot.get('auto_shorts_count', 0)} "
            f"(auto-build: {built})"
        )
        universe_text = str(snapshot.get("universe_line", "-")).replace("Universe: ", "")
        self.universe_value.setText(universe_text)
        self.universe_value.setStyleSheet(
            "color: #E06C75;" if "stale" in universe_text or "MISSING" in universe_text else ""
        )
        if snapshot.get("wrapup_running"):
            self.wrapup_value.setText("running...")
        else:
            done_at = snapshot.get("wrapup_done_at") or ""
            self.wrapup_value.setText(f"done at {done_at}" if done_at else "pending (after the last slot)")
        self.report_value.setText(str(snapshot.get("report_path", "")))

    def shutdown(self) -> None:
        self._refresh_timer.stop()
        self.service.shutdown()
