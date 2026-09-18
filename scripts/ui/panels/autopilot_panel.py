from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from project_paths import get_local_setting, save_local_setting
from ui.read_worker import ReadWorker, join_worker
from ui.services.autopilot_service import AutopilotService

#: How often the staged-picks table re-reads its file, and only while the page is
#: visible. A minute rather than the status row's five seconds: a staged pick
#: appears when a scan finishes, not between two ticks of a status label.
STAGED_POLL_INTERVAL_MS = 60_000


def _read_staged_picks() -> dict:
    """Today's staged picks, off the Qt thread. ONE normalisation, not a second."""
    import daily_recap_reader

    return dict(daily_recap_reader.staged_picks())


class AutopilotPanel(QFrame):
    """Auto Pilot (mini PC mode): one big ON/OFF switch plus enough live
    status + activity log to check on the bot from work or the kitchen."""

    statusChanged = Signal(str)
    #: (symbol, side) - the host performs the Focus add through FocusService.
    #: TJ-1 item 6(b): the staged-picks table moved here from the Daily Recap
    #: page, keeping this signal, so the add is still performed by the store's
    #: own owner (ground rule 8) and this page still only ASKS.
    focusAddRequested = Signal(str, str)

    def __init__(self, bounce_service, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.service = AutopilotService(bounce_service, parent=self)
        self._staged_rows: list[tuple[str, str]] = []
        self._staged_worker: ReadWorker | None = None
        #: widget id -> (text, style) last applied. A setStyleSheet is a Qt
        #: style recalculation, and five of them ran unconditionally on every
        #: 5 s refresh whether or not anything had changed.
        self._status_signatures: dict[int, tuple[str, str]] = {}

        title = QLabel("Auto Pilot - Mini PC Mode")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "Unattended trading day: swing scans at open+1h then hourly from the first full hour "
            "(tracker writes in the final-hour runs), self-built longs/shorts from the open's gaps "
            "and RS/RW vs SPY, near-HOD adds on regime pauses, and a phone digest in the home folder."
        )
        subtitle.setWordWrap(True)

        self.toggle_button = QPushButton("AUTO PILOT: OFF")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setMinimumHeight(44)
        self.toggle_button.clicked.connect(self._on_toggle)

        self.auto_arm_input = QCheckBox("Auto-arm every weekday at 07:00 (hands-off default)")
        self.auto_arm_input.setChecked(bool(get_local_setting("qt_autopilot_auto_arm", True)))
        self.auto_arm_input.setToolTip(
            "Auto Pilot switches itself ON once per weekday at/after 07:00 local (immediately if the "
            "app launches later). Flipping the big button OFF sticks for the rest of that day."
        )
        self.auto_arm_input.toggled.connect(
            lambda checked: save_local_setting("qt_autopilot_auto_arm", bool(checked))
        )

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
        self.industry_value = QLabel("-")
        self.industry_value.setWordWrap(True)
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
                ("Industry evidence", self.industry_value),
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

        # The staged-pick block, moved here from the Daily Recap page by TJ-1
        # item 6(b) unchanged in behaviour: AWAY still STAGES and never adopts,
        # this page still only ASKS, and the R2 adoption gate is SHOWN at click
        # time rather than enforced.
        self.staged_heading = QLabel("Staged picks - never adopted while AWAY")
        self.staged_heading.setObjectName("SectionTitle")
        self.staged = QTableWidget(0, 3)
        self.staged.setHorizontalHeaderLabels(["Symbol", "Side", "Gate at click time"])
        self.staged.setEditTriggers(QTableWidget.NoEditTriggers)
        self.staged.setSelectionBehavior(QTableWidget.SelectRows)
        self.staged.setMinimumHeight(84)
        self.staged.setMaximumHeight(150)
        self.add_button = QPushButton("Add selected staged pick to Focus")
        self.add_button.clicked.connect(self._add_selected)
        self.gate_note = QLabel("")
        self.gate_note.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.auto_arm_input)
        layout.addLayout(status_grid)
        layout.addLayout(buttons)
        layout.addWidget(log_title)
        layout.addWidget(self.log_view, 1)
        layout.addWidget(self.staged_heading)
        layout.addWidget(self.staged)
        layout.addWidget(self.add_button)
        layout.addWidget(self.gate_note)

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

        # The staged file, on its own slower clock and only while visible. The
        # READ is on a worker: it is a small JSON, and "small" is not "free" on
        # the Qt thread (ground rule 9).
        self._staged_timer = QTimer(self)
        self._staged_timer.setInterval(STAGED_POLL_INTERVAL_MS)
        self._staged_timer.timeout.connect(self._refresh_staged_if_visible)
        self._staged_timer.start()

    # -- staged picks (moved here by TJ-1 item 6(b)) --------------------
    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Read the staged picks the first time the page is looked at."""
        super().showEvent(event)
        self.refresh_staged()

    def _refresh_staged_if_visible(self) -> None:
        if self.isVisible():
            self.refresh_staged()

    def refresh_staged(self) -> None:
        """Re-read today's staged picks on a worker. Never blocks the page."""
        if self._staged_worker is not None and self._staged_worker.isRunning():
            return
        worker = ReadWorker(_read_staged_picks, self)
        worker.finished_with.connect(self._on_staged_loaded)
        worker.failed.connect(self._on_staged_failed)
        self._staged_worker = worker
        worker.start()

    def _on_staged_failed(self, message: str) -> None:
        self._staged_worker = None
        logging.debug("The staged picks could not be read: %s", message)

    def _on_staged_loaded(self, payload: object) -> None:
        self._staged_worker = None
        if isinstance(payload, dict):
            self.render_staged(payload)

    def render_staged(self, staged: dict) -> None:
        """Draw the staged names. Formatting only - it adopts nothing."""
        rows: list[tuple[str, str]] = []
        for side in ("long", "short"):
            for symbol in (staged or {}).get(side) or ():
                rows.append((str(symbol), side.upper()))
        self._staged_rows = rows
        self.staged.setRowCount(len(rows))
        for index, (symbol, side) in enumerate(rows):
            self.staged.setItem(index, 0, QTableWidgetItem(symbol))
            self.staged.setItem(index, 1, QTableWidgetItem(side))
            self.staged.setItem(index, 2, QTableWidgetItem(""))

    def _add_selected(self) -> None:
        """Adopt one staged pick. The GATE IS SHOWN, never enforced.

        Unchanged from the Daily Recap and the AWAY Recap before it: the R2
        adoption gate governs the MACHINE's adoptions, and a surface that blocked
        the trader on it would substitute the machine's judgement for theirs.
        """
        index = self.staged.currentRow()
        if index < 0 or index >= len(self._staged_rows):
            self.gate_note.setText("select a staged pick first")
            return
        symbol, side = self._staged_rows[index]
        self.gate_note.setText(
            f"R2 adoption gate: not measured on this page for {symbol} - this page "
            "reads stores, not bars, so it has no completed M5 bar or previous-day "
            "extreme to measure against. It governs the machine's adoptions, never "
            "yours, so your action is unaffected."
        )
        self.focusAddRequested.emit(symbol, side)
        self.statusChanged.emit(f"asked the desk to add {symbol} ({side}) to Focus")

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
        # Hidden, the 5 s poll does no work - the timer keeps running so the
        # first visible tick is at most 5 s away (the chart_review_panel
        # pattern). The service's own 30 s tick still emits statusChanged
        # into _apply_status, so nothing is lost while the page is closed.
        if not self.isVisible():
            return
        self._apply_status(self.service.status_snapshot())

    def _apply_styled(self, widget: QLabel, text: str, style: str) -> None:
        """setText/setStyleSheet only when the text or tone actually changed."""
        key = id(widget)
        previous = self._status_signatures.get(key)
        if previous == (text, style):
            return
        widget.setText(text)
        if previous is None or previous[1] != style:
            widget.setStyleSheet(style)
        self._status_signatures[key] = (text, style)

    @Slot(dict)
    def _apply_status(self, snapshot: dict) -> None:
        ib_text = str(snapshot.get("ib_status", "unknown"))
        self._apply_styled(
            self.ib_value,
            ib_text,
            "color: #58C777;" if ib_text.startswith("connected") else "color: #E06C75;",
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
        self._apply_styled(
            self.universe_value,
            universe_text,
            "color: #E06C75;" if "stale" in universe_text or "MISSING" in universe_text else "",
        )
        industry_text = str(snapshot.get("industry_line") or "Industry Board: unavailable")
        self._apply_styled(
            self.industry_value,
            industry_text,
            "color: #E06C75;"
            if "unavailable" in industry_text.lower() or "mismatch" in industry_text.lower()
            else "",
        )
        if snapshot.get("wrapup_running"):
            self.wrapup_value.setText("running...")
        else:
            done_at = snapshot.get("wrapup_done_at") or ""
            self.wrapup_value.setText(f"done at {done_at}" if done_at else "pending (after the last slot)")
        report_path = str(snapshot.get("report_path", ""))
        report_error = str(snapshot.get("report_error") or "")
        report_attempt = str(snapshot.get("report_last_attempt") or "")
        report_verified = str(snapshot.get("report_last_verified") or "")
        if report_error:
            verified_note = f"; last verified {report_verified}" if report_verified else "; no verified write this run"
            report_text = (
                f"{report_path}\nFAILED at {report_attempt or 'unknown'}: {report_error}{verified_note}"
            )
            report_style = "color: #E06C75;"
        elif report_verified:
            report_text = f"{report_path}\nverified {report_verified}"
            report_style = "color: #58C777;"
        else:
            report_text = f"{report_path}\nnot verified in this app run"
            report_style = "color: #E5C07B;"
        self._apply_styled(self.report_value, report_text, report_style)

    def shutdown(self) -> None:
        self._refresh_timer.stop()
        self._staged_timer.stop()
        join_worker(self._staged_worker)
        self._staged_worker = None
        self.service.shutdown()
