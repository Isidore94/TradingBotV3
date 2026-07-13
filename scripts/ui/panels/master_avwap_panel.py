from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QItemSelection, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from project_paths import MASTER_AVWAP_FOCUS_FILE, MASTER_AVWAP_PRIORITY_SETUPS_FILE
from market_session import get_default_hourly_scan_schedule, get_default_stop_time_label, get_market_session_window
from ui.models.setup import DEFAULT_SETUP_BUCKET_FILTER_LABELS, SetupRow
from ui.models.setup_table_model import ROW_ROLE, SetupFilterProxyModel, SetupTableModel
from ui.services.data_feed import copy_symbols, load_latest_setup_rows_with_meta
from ui.services.scan_service import ScanService
from ui.widgets.data_table import DataTable
from ui.widgets.setup_delegate import SetupTableDelegate
from ui.widgets.empty_state import EmptyState
from ui.widgets.kpi_tile import KpiTile
from ui.widgets.section_header import SectionHeader
from ui.widgets.setup_detail_view import SetupDetailView


def _row_context(row: SetupRow) -> str:
    """One-line setup-row summary stored with a verdict for later AI review."""
    parts = [f"bucket={row.bucket_label or row.bucket}"]
    if row.score is not None:
        parts.append(f"score={row.score:.1f}")
    if row.tags_text:
        parts.append(f"tags={row.tags_text}")
    if row.key_level:
        parts.append(f"level={row.key_level}")
    if row.expected_r is not None:
        parts.append(f"expected_r={row.expected_r:.2f}")
    return "; ".join(parts)


class MasterAvwapPanel(QWidget):
    setupSelected = Signal(object)
    rowsChanged = Signal(int, int, int)
    statusChanged = Signal(str)

    def __init__(self, focus_service=None, parent=None) -> None:
        super().__init__(parent)
        self.focus_service = focus_service
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
        self.external_scheduler_owner = ""

        self.model = SetupTableModel()
        self.proxy = SetupFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)

        self.table = DataTable()
        self.table.setModel(self.proxy)
        self.delegate = SetupTableDelegate(self.table)
        self.table.setItemDelegate(self.delegate)
        self.table.setShowGrid(False)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        if self.focus_service is not None:
            self.delegate.set_focus_lookup(self.focus_service.is_focus)
            # The ★ column: click to favorite into Swing Focus / click again to remove.
            self.table.clicked.connect(self._on_table_clicked)
            self.table.add_row_action(
                "Add to Swing Focus Picks",
                lambda proxy_index: self._add_row_to_focus(proxy_index, "swing"),
            )
            self.table.add_row_action(
                "Add to M5 Focus Picks",
                lambda proxy_index: self._add_row_to_focus(proxy_index, "m5"),
            )
            self.focus_service.focusChanged.connect(lambda: self.table.viewport().update())

        self.empty_state = EmptyState(
            "Run a scan to see setups",
            "Master AVWAP results will appear here as sortable rows with side, score, bucket, support stack, theta, and expected-R context.",
            "Run Shared Scan",
        )
        self.empty_state.action_button.clicked.connect(self.run_shared_scan)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty_state)
        self.stack.addWidget(self.table)

        # Clicking a setup opens this pane to the right of the table with the
        # family mechanics and the symbol's concrete stop/TP levels.
        self.detail_view = SetupDetailView(self)
        self.detail_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.detail_splitter.addWidget(self.stack)
        self.detail_splitter.addWidget(self.detail_view)
        self.detail_splitter.setStretchFactor(0, 3)
        self.detail_splitter.setStretchFactor(1, 2)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("MutedLabel")
        self.data_as_of_label = QLabel("")
        self.data_as_of_label.setObjectName("MutedLabel")
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
        # QFileSystemWatcher can miss atomic replacements on synced/network
        # drives.  Poll file metadata as a cheap fallback so Auto Pilot scans
        # (which use a separate service) still refresh this page promptly.
        self.report_poll_timer = QTimer(self)
        self.report_poll_timer.setInterval(30_000)
        self.report_poll_timer.timeout.connect(self._poll_report_changes)
        self.report_poll_timer.start()
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
        status_row.addWidget(self.data_as_of_label)
        status_row.addWidget(self.last_run_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addLayout(kpi_row)
        layout.addLayout(filter_row)
        layout.addLayout(copy_row)
        layout.addWidget(self.detail_splitter, 1)
        layout.addWidget(self.scheduler_status_label)
        layout.addLayout(status_row)

    def _configure_report_watcher(self) -> None:
        self.watcher = QFileSystemWatcher(self)
        for path in (MASTER_AVWAP_FOCUS_FILE, MASTER_AVWAP_PRIORITY_SETUPS_FILE):
            if Path(path).exists():
                self.watcher.addPath(str(path))
        self.watcher.fileChanged.connect(lambda _path: self.refresh_from_reports(emit_empty=False))
        self._report_signatures = self._current_report_signatures()

    @staticmethod
    def _path_signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _current_report_signatures(self) -> dict[str, tuple[int, int] | None]:
        return {
            str(path): self._path_signature(Path(path))
            for path in (MASTER_AVWAP_FOCUS_FILE, MASTER_AVWAP_PRIORITY_SETUPS_FILE)
        }

    def _poll_report_changes(self) -> None:
        signatures = self._current_report_signatures()
        if signatures != getattr(self, "_report_signatures", {}):
            self.refresh_from_reports(emit_empty=False)

    def run_shared_scan(self) -> None:
        self.scan_service.run_shared_watchlist_scan()

    def run_local_scan(self) -> None:
        self.scan_service.run_local_watchlist_scan()

    def toggle_scheduler(self) -> None:
        if self.external_scheduler_owner:
            note = f"{self.external_scheduler_owner} owns scheduled scans while Auto mode is active."
            self._refresh_scheduler_status(note=note)
            self.statusChanged.emit(note)
            return
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

    def set_external_scheduler_owner(self, owner: str = "") -> None:
        """Make scheduled-scan ownership explicit across the main GUI.

        AutoPilot is the canonical scheduler while Auto mode is active.  The
        Setups-page scheduler remains available for manual-mode days, but the
        two can never both own the hourly slots.
        """
        owner = str(owner or "").strip()
        if owner == self.external_scheduler_owner:
            return
        self.external_scheduler_owner = owner
        if owner:
            self.scheduler_enabled = False
            note = f"{owner} owns hourly scans; Setups-page scheduler disabled."
        else:
            note = "AutoPilot is off; the Setups-page scheduler is available if needed."
        self._refresh_scheduler_status(note=note)

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
        state = "externally owned" if self.external_scheduler_owner else ("running" if self.scheduler_enabled else "stopped")
        active_task = self.scheduler_active_slot or ("manual scan" if self.scan_service.running else "None")
        self.scheduler_button.setEnabled(not bool(self.external_scheduler_owner))
        self.scheduler_button.setText(
            f"Owned by {self.external_scheduler_owner}"
            if self.external_scheduler_owner
            else ("Stop Scheduler" if self.scheduler_enabled else "Start Scheduler")
        )
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
        if self.external_scheduler_owner:
            self.scheduler_enabled = False
            self._refresh_scheduler_status(
                note=f"{self.external_scheduler_owner} owns scheduled scans; no duplicate launched."
            )
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
        coverage = (
            f" (one catch-up scan covering {len(covered_slots)} due slots)"
            if len(covered_slots) > 1
            else ""
        )
        label = f"Running scheduled shared-watchlist scan for {trigger_slot}{coverage}..."
        if not self.scan_service.run_shared_watchlist_scan(label, scheduled_slot=trigger_slot):
            rejection = self.scan_service.last_rejection_reason
            if rejection == "scheduled slot already completed":
                for slot in self.scheduler_covered_slots:
                    if slot in self.scheduler_slots_state:
                        self.scheduler_slots_state[slot] = "completed"
                self.scheduler_active_slot = ""
                self.scheduler_covered_slots = []
                self._refresh_scheduler_status(
                    note=f"Scheduled slot {trigger_slot} was already completed; no duplicate scan launched."
                )
                return
            self.scheduler_active_slot = ""
            self.scheduler_covered_slots = []
            self._refresh_scheduler_status(note="Scheduler found a due slot, but another scan is already running.")
            return
        self._refresh_scheduler_status(note=label)

    def _finish_scheduler_run(self, success: bool, error_text: str = "") -> None:
        if not self.scheduler_active_slot:
            return
        trigger_slot = self.scheduler_active_slot
        covered_count = len(self.scheduler_covered_slots)
        for slot in self.scheduler_covered_slots:
            if slot in self.scheduler_slots_state:
                self.scheduler_slots_state[slot] = "completed" if success else "failed"
        self.scheduler_active_slot = ""
        self.scheduler_covered_slots = []
        note = (
            (
                f"Scheduled shared-watchlist scan for {trigger_slot} completed; "
                f"one scan covered {covered_count} due slots."
                if covered_count > 1
                else f"Scheduled shared-watchlist scan for {trigger_slot} completed."
            )
            if success
            else f"Scheduled shared-watchlist scan for {trigger_slot} failed: {error_text}"
        )
        self._refresh_scheduler_status(note=note)

    def refresh_from_reports(self, emit_empty: bool = True) -> None:
        meta = load_latest_setup_rows_with_meta()
        rows = meta["rows"]
        if rows or emit_empty:
            self.set_rows(rows)
            self.status_label.setText("Loaded latest report rows." if rows else "No report rows found.")
            self.statusChanged.emit(self.status_label.text())
        self._apply_data_as_of(meta)
        self._refresh_watcher_paths()
        self._report_signatures = self._current_report_signatures()

    def _apply_data_as_of(self, meta: dict) -> None:
        data_date = meta.get("data_date")
        source = meta.get("source") or ""
        is_stale = bool(meta.get("is_stale"))
        if not data_date:
            self.data_as_of_label.setText("")
        else:
            source_note = " · priority report" if source == "priority_report" else ""
            if is_stale:
                text = f"⚠ Setups as of {data_date} — stale; run an after-close scan to refresh{source_note}"
            else:
                text = f"Setups as of {data_date}{source_note}"
            self.data_as_of_label.setText(text)
        object_name = "CautionLabel" if is_stale else "MutedLabel"
        if self.data_as_of_label.objectName() != object_name:
            self.data_as_of_label.setObjectName(object_name)
            # Re-apply the stylesheet so the objectName-scoped rule takes effect.
            style = self.data_as_of_label.style()
            style.unpolish(self.data_as_of_label)
            style.polish(self.data_as_of_label)

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
            # ★ / ✕ verdict columns stay icon-width.
            self.table.horizontalHeader().resizeSection(0, 36)
            self.table.horizontalHeader().resizeSection(1, 36)
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
        labels = list(DEFAULT_SETUP_BUCKET_FILTER_LABELS)
        extras = sorted({row.bucket_label for row in rows if row.bucket_label and row.bucket_label not in labels})
        self.bucket_input.blockSignals(True)
        self.bucket_input.clear()
        self.bucket_input.addItem("ALL")
        self.bucket_input.addItems(labels)
        self.bucket_input.addItems(extras)
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
        self._apply_data_as_of(load_latest_setup_rows_with_meta())
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
            self._show_setup_detail(row)

    def _show_setup_detail(self, row: SetupRow) -> None:
        from setup_docs import resolve_setup_family_from_candidates

        raw = row.raw if isinstance(row.raw, dict) else {}
        signals = raw.get("favorite_signals") or row.setup_tags or []
        # Priority-report rows carry the family as a display label (and often
        # only inside the tags); resolve against the docs registry instead of
        # falling back to the bucket, which is not a setup family.
        family = resolve_setup_family_from_candidates(
            [raw.get("setup_family"), *row.setup_tags]
        )
        self.detail_view.show_setup(
            symbol=row.symbol,
            side=row.side or str(raw.get("side") or "LONG"),
            setup_family=family,
            favorite_signals=signals,
            last_close=raw.get("last_close") or raw.get("previous_close"),
        )

    def _on_table_clicked(self, proxy_index) -> None:
        """★ column clicks toggle Swing Focus; ✕ column clicks log a dislike."""
        if self.focus_service is None or not proxy_index.isValid():
            return
        source_index = self.proxy.mapToSource(proxy_index)
        key = self.model.COLUMNS[source_index.column()][0]
        if key not in {"favorite", "dislike"}:
            return
        row = self.model.row_at(source_index.row())
        if row is None or not row.symbol:
            return
        if key == "dislike":
            self._dislike_row(row)
        elif self.focus_service.is_focus(row.symbol):
            self.focus_service.remove_everywhere(row.symbol, origin="setups", context=_row_context(row))
            message = f"Unfavorited {row.symbol}: removed from focus picks."
            self.status_label.setText(message)
            self.statusChanged.emit(message)
        else:
            self._add_row_to_focus(proxy_index, "swing")

    def _dislike_row(self, row: SetupRow) -> None:
        reason, accepted = QInputDialog.getMultiLineText(
            self,
            f"Dislike {row.symbol}",
            "Why is this a bad pick? Saved to pick_feedback.jsonl so an AI can\n"
            "review your dislikes and suggest scan/scoring changes.",
        )
        if not accepted:
            return
        self._record_dislike(row, reason)

    def _record_dislike(self, row: SetupRow, reason: str) -> None:
        self.focus_service.record_feedback(
            row.symbol,
            row.side,
            "dislike",
            category=self.focus_service.focus_category(row.symbol) or "swing",
            origin="setups",
            reason=reason,
            context=_row_context(row),
        )
        message = f"✕ {row.symbol}: dislike logged for AI review."
        if self.focus_service.is_focus(row.symbol):
            self.focus_service.remove_everywhere(row.symbol)
            message = f"✕ {row.symbol}: dislike logged and removed from focus picks."
        self.status_label.setText(message)
        self.statusChanged.emit(message)

    def _add_row_to_focus(self, proxy_index, category: str = "swing") -> None:
        if self.focus_service is None or not proxy_index.isValid():
            return
        row = self.model.row_at(self.proxy.mapToSource(proxy_index).row())
        bucket = "Swing" if category == "swing" else "M5"
        if row is None or row.side not in {"LONG", "SHORT"}:
            message = "Add to Focus needs a LONG or SHORT row."
        else:
            side = "long" if row.side == "LONG" else "short"
            added = self.focus_service.add(row.symbol, side, category, origin="setups", context=_row_context(row))
            message = (
                f"Liked {row.symbol}: added to {bucket} Focus {side}s - its alerts now flag gold in the Alert Center."
                if added
                else f"{row.symbol} already in {bucket} Focus {side}s."
            )
        self.status_label.setText(message)
        self.statusChanged.emit(message)
