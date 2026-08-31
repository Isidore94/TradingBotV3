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
)

from operations_audit import build_operations_audit
from ui import theme
from ui.timer_utils import start_staggered
from ui.widgets.kpi_tile import KpiTile
from ui.widgets.section_header import SectionHeader

#: Warehouse tile status -> this page's four-status vocabulary. OFF is UNKNOWN,
#: not green: "no research store configured" is an unmeasured dimension, and
#: this page's whole discipline is that absent evidence never reads as healthy.
_WAREHOUSE_TONES = {"OK": "healthy", "WARN": "degraded", "RED": "unhealthy", "OFF": "unknown"}


def warehouse_checks(now=None) -> list[dict[str, Any]]:
    """The six warehouse tiles as Health check rows (plan sec 18, BD-20).

    Computed on the audit worker thread, never on the GUI thread: the tiles
    stat the DAS and read the gap ledger, and this page's own history is that
    synchronous evidence work here stutters the desk.
    """
    try:
        from ui.services.warehouse_service import warehouse_health_tiles

        tiles = warehouse_health_tiles(now=now)
    except Exception as exc:
        return [
            {
                "id": "warehouse_error",
                "label": "Research warehouse",
                "status": "unknown",
                "summary": f"tiles unavailable: {exc}",
                "updated_at": "",
                "source": "ui/services/warehouse_service.py",
                "details": {},
            }
        ]
    return [  # noqa: RET504 - the comprehension is the return value
        {
            "id": f"warehouse_{tile.key}",
            "label": f"Warehouse: {tile.label}",
            "status": _WAREHOUSE_TONES.get(str(tile.status).upper(), "unknown"),
            "summary": f"{tile.value} - {tile.detail}" if tile.detail else str(tile.value),
            "updated_at": "",
            "source": "ui/services/warehouse_service.py",
            "details": dict(tile.metrics or {}),
        }
        for tile in tiles
    ]


def _with_warehouse_checks(payload: dict[str, Any]) -> dict[str, Any]:
    """Append the warehouse rows and keep the summary counters consistent."""
    if not isinstance(payload, dict):
        return payload
    rows = warehouse_checks()
    if not rows:
        return payload
    merged = dict(payload)
    checks = [item for item in merged.get("checks", []) if isinstance(item, dict)]
    merged["checks"] = checks + rows
    summary = dict(merged.get("summary") or {})
    for row in rows:
        key = str(row.get("status") or _UNKNOWN)
        summary[key] = int(summary.get(key, 0) or 0) + 1
    summary["total"] = int(summary.get("total", 0) or 0) + len(rows)
    merged["summary"] = summary
    # A red warehouse tile must be able to move the page's overall verdict; a
    # green one must never improve it.
    statuses = {str(row.get("status")) for row in rows}
    current = str(merged.get("status") or _UNKNOWN).lower()
    if "unhealthy" in statuses:
        merged["status"] = "unhealthy"
    elif "degraded" in statuses and current == "healthy":
        merged["status"] = "degraded"
    return merged


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


def _cell(table: QTableWidget, row: int, column: int) -> QTableWidgetItem:
    """The cell that is already there, or a new one if there genuinely is none."""
    item = table.item(row, column)
    if item is None:
        item = QTableWidgetItem()
        table.setItem(row, column, item)
    return item


class _KeepView:
    """Hold a table's scroll position across an update.

    These tables refresh on a 15-second timer. Rebuilding one sends it back to
    the top, so a trader reading the bottom of the jobs list was pulled away
    from it every fifteen seconds with nothing on screen to explain why.
    """

    def __init__(self, table: QTableWidget) -> None:
        self._table = table
        self._vertical = 0
        self._horizontal = 0

    def __enter__(self) -> "_KeepView":
        self._vertical = self._table.verticalScrollBar().value()
        self._horizontal = self._table.horizontalScrollBar().value()
        return self

    def __exit__(self, *_exc) -> None:
        self._table.verticalScrollBar().setValue(self._vertical)
        self._table.horizontalScrollBar().setValue(self._horizontal)
        return None


def _fill(table: QTableWidget, rows: list[tuple[str, ...]], *, tones: list[str] | None = None) -> None:
    """Write into the cells that exist; create only what is genuinely new.

    G-P1.4. This built a fresh `QTableWidgetItem` for every cell of every table
    on every refresh. Same rows, same text, same colours - and a steady churn of
    Qt objects on a timer, which is also where the scroll position went.
    """
    with _KeepView(table):
        if table.rowCount() != len(rows):
            table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                item = _cell(table, row_index, column)
                text = str(value)
                if item.text() != text:
                    item.setText(text)
                if column == 0 and tones:
                    item.setForeground(QColor(theme.color(tones[row_index])))


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
        #: Set by `shutdown`. Construction schedules a refresh with
        #: `singleShot(0, ...)`, so without this a refresh queued before
        #: shutdown can fire after it and start a fresh audit thread into a
        #: panel that is going away.
        self._closing = False
        self._audit_ready.connect(self.set_payload)
        self._timer = QTimer(self)
        self._timer.setInterval(max(5_000, int(refresh_interval_ms)))
        self._timer.timeout.connect(self.refresh)
        start_staggered(self._timer, 81_000)
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
        if self._closing:
            return
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
            payload = _with_warehouse_checks(payload)
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
        # G-P1.4: updated in place rather than rebuilt - see `_fill`.
        with _KeepView(self.table):
            if self.table.rowCount() != len(checks):
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
                    item = _cell(self.table, row_index, column)
                    if item.text() != value:
                        item.setText(value)
                    if column == 0:
                        item.setForeground(foreground)
                        font = item.font()
                        font.setBold(True)
                        # Colour alone is not a distinction: UNKNOWN also reads
                        # as italic so an unmeasured row is obvious without
                        # relying on hue discrimination.
                        font.setItalic(check_status == _UNKNOWN)
                        item.setFont(font)
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
        self._closing = True
        self._timer.stop()
        # The audit runs on a plain daemon thread that emits a Qt signal back
        # into this panel. Left unjoined it can emit into a panel whose C++
        # half has already been freed, and THAT is an access violation rather
        # than a Python RuntimeError - so the `except RuntimeError` at the emit
        # cannot catch it. Joining here is what makes that guard's job
        # possible. Intermittent, and it took a segfault two test files later
        # to find: 4 runs in 6 crashed `test_qt_alert_capture` merely because a
        # HealthPanel had been constructed earlier in the same process.
        thread, self._audit_thread = self._audit_thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
