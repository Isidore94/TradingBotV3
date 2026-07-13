from __future__ import annotations

import json
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
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from operations_audit import build_operations_audit
from ui import theme
from ui.widgets.kpi_tile import KpiTile
from ui.widgets.section_header import SectionHeader


_STATUS_TONES = {
    "healthy": "long",
    "degraded": "caution",
    "unhealthy": "short",
}


class HealthPanel(QFrame):
    """Live Sol3 operational evidence without touching the large tracker."""

    statusChanged = Signal(str)

    def __init__(self, parent=None, *, refresh_interval_ms: int = 15_000) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._payload: dict[str, Any] = {}

        self.overall_tile = KpiTile("Overall Sol3 health", "CHECKING")
        self.healthy_tile = KpiTile("Healthy checks", "0", "long")
        self.degraded_tile = KpiTile("Degraded checks", "0", "caution")
        self.unhealthy_tile = KpiTile("Unhealthy checks", "0", "short")

        self.meta_label = QLabel("Waiting for the first audit...")
        self.meta_label.setObjectName("MutedLabel")

        refresh_button = QPushButton("Refresh Now")
        refresh_button.clicked.connect(self.refresh)
        header = SectionHeader(
            "System Health",
            "Sol3 heartbeat, scheduler, scan manifests, SPY/Greatness shadows, and candidate registry. "
            "The large setup-tracker file is intentionally excluded.",
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

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(self.details)
        splitter.setSizes([440, 260])

        tiles = QHBoxLayout()
        for tile in (self.overall_tile, self.healthy_tile, self.degraded_tile, self.unhealthy_tile):
            tiles.addWidget(tile, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(header)
        layout.addLayout(tiles)
        layout.addWidget(self.meta_label)
        layout.addWidget(splitter, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(max(5_000, int(refresh_interval_ms)))
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        QTimer.singleShot(0, self.refresh)

    def refresh(self) -> None:
        try:
            payload = build_operations_audit()
        except Exception as exc:
            payload = {
                "status": "unhealthy",
                "generated_at": "",
                "market_phase": "unknown",
                "market_session": "",
                "summary": {"healthy": 0, "degraded": 0, "unhealthy": 1, "total": 1},
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
        self.set_payload(payload)

    def set_payload(self, payload: dict[str, Any]) -> None:
        self._payload = payload if isinstance(payload, dict) else {}
        status = str(self._payload.get("status") or "unhealthy").lower()
        summary = self._payload.get("summary") if isinstance(self._payload.get("summary"), dict) else {}
        self.overall_tile.set_value(status.upper())
        self.overall_tile.value_label.setStyleSheet(f"color: {theme.color(_STATUS_TONES.get(status, 'neutral'))};")
        self.healthy_tile.set_value(str(int(summary.get("healthy", 0) or 0)))
        self.degraded_tile.set_value(str(int(summary.get("degraded", 0) or 0)))
        self.unhealthy_tile.set_value(str(int(summary.get("unhealthy", 0) or 0)))
        self.meta_label.setText(
            f"Audit {self._payload.get('generated_at') or 'unknown time'} | "
            f"market {self._payload.get('market_phase') or 'unknown'} "
            f"({self._payload.get('market_session') or '?'})"
        )

        checks = [item for item in self._payload.get("checks", []) if isinstance(item, dict)]
        selected_id = ""
        if 0 <= self.table.currentRow() < len(getattr(self, "_checks", [])):
            selected_id = str(self._checks[self.table.currentRow()].get("id") or "")
        self._checks = checks
        self.table.setRowCount(len(checks))
        selected_row = 0 if checks else -1
        for row_index, check in enumerate(checks):
            check_status = str(check.get("status") or "unhealthy").lower()
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
                    item.setFont(font)
                self.table.setItem(row_index, column, item)
            if str(check.get("id") or "") == selected_id:
                selected_row = row_index
        if selected_row >= 0:
            self.table.selectRow(selected_row)
            self._show_selected_check(selected_row)
        else:
            self.details.clear()
        self.statusChanged.emit(status)

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
