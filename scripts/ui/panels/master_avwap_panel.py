from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QItemSelection, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from project_paths import MASTER_AVWAP_FOCUS_FILE, MASTER_AVWAP_PRIORITY_SETUPS_FILE
from market_session import get_default_hourly_scan_schedule, get_default_stop_time_label, get_market_session_window
from ui.models.setup import SetupRow
from ui.models.setup_table_model import ROW_ROLE, SetupFilterProxyModel, SetupTableModel
from ui.services.data_feed import copy_symbols, load_latest_setup_rows
from ui.services.scan_service import ScanService
from ui.widgets.data_table import DataTable
from ui.widgets.setup_delegate import SetupTableDelegate
from ui.widgets.empty_state import EmptyState
from ui.widgets.kpi_tile import KpiTile
from ui.widgets.section_header import SectionHeader


class MasterAvwapPanel(QWidget):
    setupSelected = Signal(object)
    rowsChanged = Signal(int, int, int)
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scan_service = ScanService(self)
        self.scan_service.started.connect(self._on_scan_started)
        self.scan_service.finished.connect(self._on_scan_finished)
        self.scan_service.failed.connect(self._on_scan_failed)
        self.scheduler_enabled = False
        self.scheduler_active_slot = ""
        self.scheduler_day = ""
        self.scheduler_slots_state: dict[str, str] = {}
        self.scheduler_note = "Hourly shared-watchlist scheduler is off."
        self.scheduler_covered_slots: list[str] = []

        self.model = SetupTableModel()
        self.proxy = SetupFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)

        self.table = DataTable()
        self.table.setModel(self.proxy)
        self.table.setItemDelegate(SetupTableDelegate(self.table))
        self.table.setShowGrid(False)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        self.empty_state = EmptyState(
            "Run a scan to see setups",
            "Master AVWAP results will appear here as sortable rows with side, score, bucket, support stack, theta, and expected-R context.",
            "Run Shared Scan",
        )
        self.empty_state.action_button.clicked.connect(self.run_shared_scan)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty_state)
        self.stack.addWidget(self.table)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("MutedLabel")
        self.last_run_label = QLabel("Last run: never")
        self.last_run_label.setObjectName("MutedLabel")
        self.scheduler_status_label = QLabel("")
        self.scheduler_status_label.setObjectName("MutedLabel")
        self.scheduler_status_label.setWordWrap(True)
        self.scheduler_button = QPushButton("Start Scheduler")
        self.scheduler_button.clicked.connect(self.toggle_scheduler)

        self.total_tile = KpiTile("Setups", "0")
        self.favorite_tile = KpiTile("Favorites", "0", tone="favorite")
        self.near_tile = KpiTile("Near Zones", "0", tone="near")

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter symbol, tag, level")
        self.search_input.textChanged.connect(self._apply_filters)

        self.min_score_input = QDoubleSpinBox()
        self.min_score_input.setRange(0.0, 200.0)
        self.min_score_input.setDecimals(1)
        self.min_score_input.setSingleStep(5.0)
        self.min_score_input.setPrefix("Min score ")
        self.min_score_input.valueChanged.connect(self._apply_filters)

        self.side_input = QComboBox()
        self.side_input.addItems(["ALL", "LONG", "SHORT"])
        self.side_input.currentTextChanged.connect(self._apply_filters)

        self.bucket_input = QComboBox()
        self.bucket_input.addItem("ALL")
        self.bucket_input.currentTextChanged.connect(self._apply_filters)

        self.max_dte_input = QSpinBox()
        self.max_dte_input.setRange(0, 365)
        self.max_dte_input.setSpecialValueText("Any DTE")
        self.max_dte_input.setValue(0)
        self.max_dte_input.valueChanged.connect(self._apply_filters)

        self._build_layout()
        self._configure_report_watcher()
        self.refresh_from_reports(emit_empty=False)
        self.scheduler_timer = QTimer(self)
        self.scheduler_timer.setInterval(15_000)
        self.scheduler_timer.timeout.connect(self._scheduler_tick)
        self.scheduler_timer.start()
        self._refresh_scheduler_status()

    def _build_layout(self) -> None:
        header = SectionHeader(
            "Master AVWAP Setups",
            "Ranked favorite, high-conviction, and near-zone setups.",
        )
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_from_reports)
        header.add_action(refresh_button)

        run_shared_button = QPushButton("Run Shared Scan")
        run_shared_button.setObjectName("PrimaryButton")
        run_shared_button.clicked.connect(self.run_shared_scan)
        header.add_action(run_shared_button)

        run_local_button = QPushButton("Run Local Scan")
        run_local_button.clicked.connect(self.run_local_scan)
        header.add_action(run_local_button)

        header.add_action(self.scheduler_button)

        kpi_row = QHBoxLayout()
        kpi_row.setContentsMargins(0, 0, 0, 0)
        kpi_row.setSpacing(8)
        kpi_row.addWidget(self.total_tile)
        kpi_row.addWidget(self.favorite_tile)
        kpi_row.addWidget(self.near_tile)
        kpi_row.addStretch(1)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(8)
        filter_row.addWidget(self.search_input, 2)
        filter_row.addWidget(self.min_score_input)
        filter_row.addWidget(self.side_input)
        filter_row.addWidget(self.bucket_input)
        filter_row.addWidget(self.max_dte_input)

        copy_row = QHBoxLayout()
        copy_row.setContentsMargins(0, 0, 0, 0)
        copy_row.setSpacing(8)
        copy_label = QLabel("Copy visible:")
        copy_label.setObjectName("MutedLabel")
        copy_row.addWidget(copy_label)
        for label, kind in (
            ("Longs", "longs"),
            ("Shorts", "shorts"),
            ("Favorites", "favorites"),
            ("Active", "active"),
            ("Ranked", "ranked"),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, copy_kind=kind: self.copy_list(copy_kind))
            copy_row.addWidget(button)
        copy_row.addStretch(1)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        status_row.addWidget(self.last_run_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addLayout(kpi_row)
        layout.addLayout(filter_row)
        layout.addLayout(copy_row)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.scheduler_status_label)
        layout.addLayout(status_row)

    def _configure_report_watcher(self) -> None:
        self.watcher = QFileSystemWatcher(self)
        for path in (MASTER_AVWAP_FOCUS_FILE, MASTER_AVWAP_PRIORITY_SETUPS_FILE):
            if Path(path).exists():
                self.watcher.addPath(str(path))
        self.watcher.fileChanged.connect(lambda _path: self.refresh_from_reports(emit_empty=False))

    def run_shared_scan(self) -> None:
        self.scan_service.run_shared_watchlist_scan()

    def run_local_scan(self) -> None:
        self.scan_service.run_local_watchlist_scan()

    def toggle_scheduler(self) -> None:
        self.scheduler_enabled = not self.scheduler_enabled
        note = (
            "Hourly shared-watchlist scheduler started."
            if self.scheduler_enabled
            else "Hourly shared-watchlist scheduler stopped."
        )
        self._refresh_scheduler_status(note=note)
        self.statusChanged.emit(note)
        if self.scheduler_enabled:
            self._scheduler_tick()

    def _scheduler_slot_datetime(self, slot: str, reference: datetime | None = None) -> datetime:
        now = reference or datetime.now()
        slot_time = datetime.strptime(str(slot).strip(), "%H:%M").time()
        return datetime.combine(now.date(), slot_time)

    def _reset_scheduler_state_for_day(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        today_iso = now.date().isoformat()
        if self.scheduler_day == today_iso and self.scheduler_slots_state:
            return False
        schedule = list(get_default_hourly_scan_schedule(reference=now))
        self.scheduler_day = today_iso
        self.scheduler_slots_state = {slot: "pending" for slot in schedule}
        self.scheduler_note = (
            "Scheduler ready for today's market window."
            if self.scheduler_enabled
            else "Hourly shared-watchlist scheduler is off."
        )
        return True

    def _scheduler_schedule(self, now: datetime | None = None) -> list[str]:
        self._reset_scheduler_state_for_day(now)
        return list(self.scheduler_slots_state.keys())

    def _due_scheduler_slots(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now()
        due: list[str] = []
        for slot in self._scheduler_schedule(now):
            if self.scheduler_slots_state.get(slot) != "pending":
                continue
            if self._scheduler_slot_datetime(slot, reference=now) <= now:
                due.append(slot)
        return due

    def _next_scheduler_slot(self, now: datetime | None = None) -> str | None:
        now = now or datetime.now()
        for slot in self._scheduler_schedule(now):
            if self.scheduler_slots_state.get(slot) != "pending":
                continue
            if self._scheduler_slot_datetime(slot, reference=now) > now:
                return slot
        return None

    def _refresh_scheduler_status(self, note: str | None = None) -> None:
        now = datetime.now()
        self._reset_scheduler_state_for_day(now)
        if note is not None:
            self.scheduler_note = note

        schedule = self._scheduler_schedule(now)
        next_slot = self._next_scheduler_slot(now)
        stop_at = get_default_stop_time_label(reference=now)
        session = get_market_session_window(reference=now)
        completed = [slot for slot, status in self.scheduler_slots_state.items() if status == "completed"]
        failed = [slot for slot, status in self.scheduler_slots_state.items() if status == "failed"]
        state = "running" if self.scheduler_enabled else "stopped"
        active_task = self.scheduler_active_slot or ("manual scan" if self.scan_service.running else "None")
        self.scheduler_button.setText("Stop Scheduler" if self.scheduler_enabled else "Start Scheduler")
        self.scheduler_status_label.setText(
            (
                f"Hourly shared-watchlist scheduler: {state} | "
                f"Market session: {session.session_label} | "
                f"Today's slots: {', '.join(schedule) if schedule else 'None'} | Stop at: {stop_at}\n"
                f"Next slot: {next_slot or 'None'} | Completed: {len(completed)} | "
                f"Failed: {len(failed)} | Active task: {active_task}\n"
                f"Note: {self.scheduler_note}"
            )
        )

    def _scheduler_tick(self) -> None:
        now = datetime.now()
        self._reset_scheduler_state_for_day(now)
        if self.scan_service.running:
            self._refresh_scheduler_status()
            return
        if not self.scheduler_enabled:
            self._refresh_scheduler_status()
            return
        stop_dt = self._scheduler_slot_datetime(get_default_stop_time_label(reference=now), reference=now)
        if now >= stop_dt:
            self._refresh_scheduler_status(note=f"Today's scheduler window ended at {stop_dt.strftime('%H:%M')}.")
            return

        due_slots = self._due_scheduler_slots(now)
        if due_slots:
            self._start_scheduled_shared_scan(due_slots[-1], due_slots)
            return

        next_slot = self._next_scheduler_slot(now)
        self._refresh_scheduler_status(
            note=(
                f"Waiting for next hourly slot {next_slot}."
                if next_slot
                else "No pending hourly shared-watchlist slots remain for today."
            )
        )

    def _start_scheduled_shared_scan(self, trigger_slot: str, covered_slots: list[str]) -> None:
        self.scheduler_active_slot = trigger_slot
        self.scheduler_covered_slots = list(covered_slots)
        label = f"Running scheduled shared-watchlist scan for {trigger_slot}..."
        if not self.scan_service.run_shared_watchlist_scan(label):
            self.scheduler_active_slot = ""
            self.scheduler_covered_slots = []
            self._refresh_scheduler_status(note="Scheduler found a due slot, but another scan is already running.")
            return
        self._refresh_scheduler_status(note=label)

    def _finish_scheduler_run(self, success: bool, error_text: str = "") -> None:
        if not self.scheduler_active_slot:
            return
        trigger_slot = self.scheduler_active_slot
        for slot in self.scheduler_covered_slots:
            if slot in self.scheduler_slots_state:
                self.scheduler_slots_state[slot] = "completed" if success else "failed"
        self.scheduler_active_slot = ""
        self.scheduler_covered_slots = []
        note = (
            f"Scheduled shared-watchlist scan for {trigger_slot} completed."
            if success
            else f"Scheduled shared-watchlist scan for {trigger_slot} failed: {error_text}"
        )
        self._refresh_scheduler_status(note=note)

    def refresh_from_reports(self, emit_empty: bool = True) -> None:
        rows = load_latest_setup_rows()
        if rows or emit_empty:
            self.set_rows(rows)
            self.status_label.setText("Loaded latest report rows." if rows else "No report rows found.")
            self.statusChanged.emit(self.status_label.text())
        self._refresh_watcher_paths()

    def set_rows(self, rows: list[SetupRow]) -> None:
        self.model.set_rows(rows)
        self._refresh_bucket_filter(rows)
        self._apply_filters()
        self.stack.setCurrentWidget(self.table if rows else self.empty_state)
        if rows:
            # Rows arrive pre-ranked (conviction bucket, then tracker-led
            # Expected-R). Preserve that order instead of forcing a score sort so
            # the headline ranking is what the trader sees first; column headers
            # remain click-sortable.
            self.table.fit_columns()
        self._update_kpis(rows)
        self.rowsChanged.emit(
            len(rows),
            sum(1 for row in rows if row.bucket.strip().lower() in {"favorite_setup", "high_conviction"}),
            sum(1 for row in rows if row.bucket.strip().lower() == "near_favorite_zone"),
        )

    def filtered_rows(self) -> list[SetupRow]:
        rows: list[SetupRow] = []
        for proxy_row in range(self.proxy.rowCount()):
            proxy_index = self.proxy.index(proxy_row, 0)
            source_index = self.proxy.mapToSource(proxy_index)
            row = self.model.data(source_index, ROW_ROLE)
            if isinstance(row, SetupRow):
                rows.append(row)
        return rows

    def all_rows(self) -> list[SetupRow]:
        return self.model.rows()

    def copy_list(self, kind: str) -> None:
        rows = self.filtered_rows()
        text = copy_symbols(rows, kind)
        QApplication.clipboard().setText(text)
        self.status_label.setText(f"Copied {kind.lower()} list: {len([s for s in text.split(',') if s.strip()])} symbol(s).")
        self.statusChanged.emit(self.status_label.text())

    def _apply_filters(self) -> None:
        max_dte = self.max_dte_input.value() or None
        self.proxy.set_filters(
            min_score=self.min_score_input.value(),
            side=self.side_input.currentText(),
            bucket=self.bucket_input.currentText(),
            max_dte=max_dte,
            search_text=self.search_input.text(),
        )

    def _refresh_bucket_filter(self, rows: list[SetupRow]) -> None:
        current = self.bucket_input.currentText()
        labels = sorted({row.bucket_label for row in rows if row.bucket_label})
        self.bucket_input.blockSignals(True)
        self.bucket_input.clear()
        self.bucket_input.addItem("ALL")
        self.bucket_input.addItems(labels)
        index = self.bucket_input.findText(current)
        self.bucket_input.setCurrentIndex(index if index >= 0 else 0)
        self.bucket_input.blockSignals(False)

    def _update_kpis(self, rows: list[SetupRow]) -> None:
        favorite_count = sum(1 for row in rows if row.bucket.strip().lower() in {"favorite_setup", "high_conviction"})
        near_count = sum(1 for row in rows if row.bucket.strip().lower() == "near_favorite_zone")
        self.total_tile.set_value(str(len(rows)))
        self.favorite_tile.set_value(str(favorite_count))
        self.near_tile.set_value(str(near_count))

    def _refresh_watcher_paths(self) -> None:
        watched = set(self.watcher.files())
        for path in (MASTER_AVWAP_FOCUS_FILE, MASTER_AVWAP_PRIORITY_SETUPS_FILE):
            path_text = str(path)
            if Path(path).exists() and path_text not in watched:
                self.watcher.addPath(path_text)

    def _on_scan_started(self, label: str) -> None:
        self.status_label.setText(label)
        self.statusChanged.emit(label)

    def _on_scan_finished(self, _run_result: dict, rows: list[SetupRow], stamp: str) -> None:
        self.set_rows(rows)
        self.last_run_label.setText(f"Last run: {stamp}")
        message = f"Scan complete at {stamp}; loaded {len(rows)} setup row(s)."
        self.status_label.setText(message)
        self.statusChanged.emit(message)
        self._refresh_watcher_paths()
        self._finish_scheduler_run(success=True)

    def _on_scan_failed(self, message: str) -> None:
        summary = message.splitlines()[0] if message else "Scan failed."
        self.status_label.setText(f"Error: {summary}")
        self.statusChanged.emit(self.status_label.text())
        self._finish_scheduler_run(success=False, error_text=summary)
        QMessageBox.critical(self, "Master AVWAP Scan Failed", message)

    def _on_selection_changed(self, selected: QItemSelection, _deselected: QItemSelection) -> None:
        indexes = selected.indexes()
        if not indexes:
            return
        source_index = self.proxy.mapToSource(indexes[0])
        row = self.model.row_at(source_index.row())
        if row is not None:
            self.setupSelected.emit(row)
