from __future__ import annotations

import json
import threading
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from operations_audit import build_operations_audit
from ui import theme
from ui.widgets.kpi_tile import KpiTile
from ui.widgets.section_header import SectionHeader


# UNKNOWN is its own tone on purpose: "we never measured this" must not look
# like "we measured this and it is bad" (plan.md sec 6.3 - the page must show
# UNKNOWN when evidence is absent, and must not convert missing telemetry into
# a green state either).
_STATUS_TONES = {
    "healthy": "long",
    "degraded": "caution",
    "unhealthy": "short",
    "unknown": "study",
}
#: Anything that is not one of the four statuses is itself unknown.
_UNKNOWN = "unknown"

#: Job states that colour the per-job rows. Everything else renders neutral.
_JOB_STATE_TONES = {
    "COMPLETED": "long",
    "RUNNING": "study",
    "QUEUED": "study",
    "FAILED": "short",
    "STALE": "short",
    "SKIPPED": "caution",
}


def _table(columns: tuple[str, ...], *, stretch_column: int) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().setVisible(False)
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    for index in range(len(columns)):
        header.setSectionResizeMode(
            index,
            header.ResizeMode.Stretch if index == stretch_column else header.ResizeMode.ResizeToContents,
        )
    return table


def _fill(table: QTableWidget, rows: list[tuple[str, ...]], *, tones: list[str] | None = None) -> None:
    table.setRowCount(len(rows))
    for row_index, values in enumerate(rows):
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column == 0 and tones:
                item.setForeground(QColor(theme.color(tones[row_index])))
            table.setItem(row_index, column, item)


class HealthPanel(QFrame):
    """Live Sol3 operational evidence without touching the large tracker."""

    statusChanged = Signal(str)
    #: Worker-thread audit results arrive here; the queued connection delivers
    #: them on the GUI thread.
    _audit_ready = Signal(dict)

    def __init__(self, parent=None, *, refresh_interval_ms: int = 15_000) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._payload: dict[str, Any] = {}

        self.overall_tile = KpiTile("Overall Sol3 health", "CHECKING")
        self.healthy_tile = KpiTile("Healthy checks", "0", "long")
        self.degraded_tile = KpiTile("Degraded checks", "0", "caution")
        self.unhealthy_tile = KpiTile("Unhealthy checks", "0", "short")
        self.unknown_tile = KpiTile("Unknown (unmeasured)", "0", "study")

        self.meta_label = QLabel("Waiting for the first audit...")
        self.meta_label.setObjectName("MutedLabel")

        refresh_button = QPushButton("Refresh Now")
        refresh_button.clicked.connect(self.refresh)
        header = SectionHeader(
            "System Health",
            "Sol3 writer identity/lease, heartbeat, scheduler (with retry budgets), scan manifests and "
            "per-phase timings, owned processes/threads, universe and market-data age, disk/storage, "
            "SPY/Greatness shadows, candidate registry, and learning capture readiness. UNKNOWN rows are "
            "required evidence nobody has measured yet - they are not green. The large setup-tracker file "
            "is intentionally excluded.",
        )
        header.add_action(refresh_button)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(("Status", "Component", "Summary", "Updated"))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, self.table.horizontalHeader().ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, self.table.horizontalHeader().ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, self.table.horizontalHeader().ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, self.table.horizontalHeader().ResizeMode.ResizeToContents)
        self.table.currentCellChanged.connect(self._show_selected_check)

        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        self.details.setPlaceholderText("Select a health check to see its evidence and source path.")

        # plan.md sec 6.3 bullets 4 and 11 (per-job attempts/success, per-phase
        # timings) were already computed by the audit and then dropped here:
        # set_payload read only payload["checks"], so payload["jobs"] and the
        # manifest's phase list never reached the trader and only one aggregate
        # total was visible. They render as structured rows, not a JSON dump.
        self.jobs_table = _table(
            ("State", "Job", "Slot", "Attempt", "Started", "Ended", "Detail"), stretch_column=6
        )
        self.phases_table = _table(("Phase", "Minutes", "Seconds", "Share"), stretch_column=0)

        self.detail_tabs = QTabWidget()
        self.detail_tabs.addTab(self.details, "Evidence")
        self.detail_tabs.addTab(self.jobs_table, "Jobs")
        self.detail_tabs.addTab(self.phases_table, "Phase timings")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(self.detail_tabs)
        splitter.setSizes([440, 260])

        tiles = QHBoxLayout()
        for tile in (
            self.overall_tile,
            self.healthy_tile,
            self.degraded_tile,
            self.unhealthy_tile,
            self.unknown_tile,
        ):
            tiles.addWidget(tile, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(header)
        layout.addLayout(tiles)
        layout.addWidget(self.meta_label)
        layout.addWidget(splitter, 1)

        self._audit_thread: threading.Thread | None = None
        self._audit_ready.connect(self.set_payload)
        self._timer = QTimer(self)
        self._timer.setInterval(max(5_000, int(refresh_interval_ms)))
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        QTimer.singleShot(0, self.refresh)

    def refresh(self) -> None:
        """Kick off one audit on a worker thread; never block the GUI.

        build_operations_audit streams the multi-megabyte shadow JSONLs, walks
        the diagnostics footprint and probes the disk - measured at ~0.4s on
        the live machine, and worse while a scan is churning the log mtimes
        (each refresh re-streams the whole greatness log). Running that
        synchronously here froze the GUI for that long every refresh interval,
        which is exactly the recurring stutter the trader reported. One audit
        runs at a time; a tick that lands while one is in flight is skipped -
        the running audit's result is at most seconds away.
        """
        if self._audit_thread is not None and self._audit_thread.is_alive():
            return
        self._audit_thread = threading.Thread(
            target=self._build_audit_payload,
            name="qt-health-audit",
            daemon=True,
        )
        self._audit_thread.start()

    def _build_audit_payload(self) -> None:
        try:
            payload = build_operations_audit()
        except Exception as exc:
            payload = {
                "status": "unhealthy",
                "generated_at": "",
                "market_phase": "unknown",
                "market_session": "",
                "summary": {"healthy": 0, "degraded": 0, "unhealthy": 1, "unknown": 0, "total": 1},
                "checks": [
                    {
                        "id": "audit_error",
                        "label": "Operations audit",
                        "status": "unhealthy",
                        "summary": str(exc),
                        "updated_at": "",
                        "source": "operations_audit.py",
                        "details": {},
                    }
                ],
            }
        try:
            self._audit_ready.emit(payload)
        except RuntimeError:
            # The panel's C++ half was deleted while the audit ran (app
            # shutdown). Nothing to update, nothing to leak.
            pass

    def wait_for_audit(self, timeout: float = 10.0) -> None:
        """Test/shutdown helper: join the in-flight audit thread, if any."""
        thread = self._audit_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def set_payload(self, payload: dict[str, Any]) -> None:
        self._payload = payload if isinstance(payload, dict) else {}
        # An audit with no status at all is UNKNOWN, not broken and certainly
        # not green.
        status = str(self._payload.get("status") or _UNKNOWN).lower()
        if status not in _STATUS_TONES:
            status = _UNKNOWN
        summary = self._payload.get("summary") if isinstance(self._payload.get("summary"), dict) else {}
        unknown_count = int(summary.get("unknown", 0) or 0)
        self.overall_tile.set_value(status.upper())
        self.overall_tile.value_label.setStyleSheet(f"color: {theme.color(_STATUS_TONES.get(status, 'neutral'))};")
        self.healthy_tile.set_value(str(int(summary.get("healthy", 0) or 0)))
        self.degraded_tile.set_value(str(int(summary.get("degraded", 0) or 0)))
        self.unhealthy_tile.set_value(str(int(summary.get("unhealthy", 0) or 0)))
        self.unknown_tile.set_value(str(unknown_count))
        meta_text = (
            f"Audit {self._payload.get('generated_at') or 'unknown time'} | "
            f"market {self._payload.get('market_phase') or 'unknown'} "
            f"({self._payload.get('market_session') or '?'})"
        )
        if unknown_count:
            unmeasured = [
                str(row.get("label") or row.get("id") or "")
                for row in self._payload.get("required_checks", [])
                if isinstance(row, dict) and not row.get("implemented")
            ]
            meta_text += f" | {unknown_count} check(s) UNKNOWN"
            if unmeasured:
                meta_text += f"; {len(unmeasured)} required dimension(s) unmeasured: " + ", ".join(unmeasured)
        # Phase 0 task 8: every learning artifact on this page is pre-v2
        # evidence, and the page must say so rather than let a reader assume
        # the numbers are promotable.
        label = str(self._payload.get("evidence_label") or "").strip()
        if label:
            meta_text += f" | learning evidence: {label}"
        self.meta_label.setText(meta_text)

        checks = [item for item in self._payload.get("checks", []) if isinstance(item, dict)]
        selected_id = ""
        if 0 <= self.table.currentRow() < len(getattr(self, "_checks", [])):
            selected_id = str(self._checks[self.table.currentRow()].get("id") or "")
        self._checks = checks
        self.table.setRowCount(len(checks))
        selected_row = 0 if checks else -1
        for row_index, check in enumerate(checks):
            check_status = str(check.get("status") or _UNKNOWN).lower()
            if check_status not in _STATUS_TONES:
                check_status = _UNKNOWN
            values = (
                check_status.upper(),
                str(check.get("label") or check.get("id") or ""),
                str(check.get("summary") or ""),
                str(check.get("updated_at") or ""),
            )
            foreground = QColor(theme.color(_STATUS_TONES.get(check_status, "neutral")))
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setForeground(foreground)
                    font = item.font()
                    font.setBold(True)
                    # Colour alone is not a distinction: UNKNOWN also reads as
                    # italic so an unmeasured row is obvious without relying on
                    # hue discrimination.
                    font.setItalic(check_status == _UNKNOWN)
                    item.setFont(font)
                self.table.setItem(row_index, column, item)
            if str(check.get("id") or "") == selected_id:
                selected_row = row_index
        if selected_row >= 0:
            self.table.selectRow(selected_row)
            self._show_selected_check(selected_row)
        else:
            self.details.clear()
        self._render_jobs()
        self._render_phases()
        self.statusChanged.emit(status)

    def _render_jobs(self) -> None:
        """Per-job attempt / last-success detail (plan.md sec 6.3 bullet 4)."""
        jobs = [item for item in self._payload.get("jobs", []) if isinstance(item, dict)]
        rows: list[tuple[str, ...]] = []
        tones: list[str] = []
        for job in jobs:
            state = str(job.get("state") or "").upper()
            detail = str(job.get("error") or "")
            if job.get("error_class"):
                detail = f"[{job.get('error_class')}] {detail}".strip()
            if not detail:
                detail = str(job.get("run_id") or "")
            rows.append(
                (
                    state or "UNKNOWN",
                    str(job.get("job_type") or ""),
                    str(job.get("slot") or ""),
                    str(job.get("attempt", 0)),
                    str(job.get("started_at") or ""),
                    str(job.get("ended_at") or ""),
                    detail,
                )
            )
            tones.append(_JOB_STATE_TONES.get(state, "neutral"))
        _fill(self.jobs_table, rows, tones=tones)
        self.detail_tabs.setTabText(1, f"Jobs ({len(rows)})")

    def _render_phases(self) -> None:
        """Per-phase timings of the latest manifest (plan.md sec 6.3 bullet 11)."""
        manifest = self._payload.get("latest_manifest")
        manifest = manifest if isinstance(manifest, dict) else {}
        checks = {str(item.get("id")): item for item in self._payload.get("checks", []) if isinstance(item, dict)}
        details = checks.get("run_manifest", {}).get("details") if checks.get("run_manifest") else None
        phases = (details or {}).get("phases")
        if not isinstance(phases, list) or not phases:
            phases = manifest.get("phases") if isinstance(manifest.get("phases"), list) else []
        rows: list[tuple[str, ...]] = []
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            try:
                seconds = float(phase.get("seconds") or 0.0)
            except (TypeError, ValueError):
                seconds = 0.0
            share = phase.get("share_pct")
            rows.append(
                (
                    str(phase.get("label") or ""),
                    f"{seconds / 60.0:.1f}",
                    f"{seconds:.1f}",
                    "-" if share is None else f"{float(share):.1f}%",
                )
            )
        _fill(self.phases_table, rows)
        self.detail_tabs.setTabText(2, f"Phase timings ({len(rows)})")

    def _show_selected_check(self, current_row: int, *_args) -> None:
        if not (0 <= current_row < len(getattr(self, "_checks", []))):
            self.details.clear()
            return
        check = self._checks[current_row]
        details = json.dumps(check.get("details") or {}, indent=2, sort_keys=True, default=str)
        self.details.setPlainText(
            f"{check.get('label') or check.get('id')}\n"
            f"Status: {str(check.get('status') or '').upper()}\n"
            f"Summary: {check.get('summary') or ''}\n"
            f"Updated: {check.get('updated_at') or ''}\n"
            f"Source: {check.get('source') or ''}\n\n"
            f"Evidence\n{details}"
        )

    def shutdown(self) -> None:
        self._timer.stop()
