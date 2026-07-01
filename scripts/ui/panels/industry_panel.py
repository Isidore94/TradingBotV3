from __future__ import annotations

import csv
import threading
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from industry_scanner import (
    INDUSTRY_BOARD_CSV_FILE,
    SECTOR_BOARD_CSV_FILE,
    run_industry_scan,
)
from ui.widgets.section_header import SectionHeader

POSITIVE_COLOR = QColor("#4CAF50")
NEGATIVE_COLOR = QColor("#EF5350")

SECTOR_COLUMNS = (
    ("rs_rank", "Rank"),
    ("etf", "ETF"),
    ("sector", "Sector"),
    ("pct_change_1d", "Today"),
    ("rs_score", "RS"),
    ("return_5d_pct", "5d"),
    ("return_20d_pct", "20d"),
    ("return_65d_pct", "65d"),
    ("volume_buzz_pct", "Vol Buzz"),
)
INDUSTRY_COLUMNS = (
    ("rs_rank", "Rank"),
    ("industry", "Industry"),
    ("member_count", "N"),
    ("pct_change_1d", "Today"),
    ("rs_score", "RS"),
    ("return_5d_pct", "5d"),
    ("return_20d_pct", "20d"),
    ("volume_buzz_pct", "Vol Buzz"),
    ("top_movers", "Top Movers"),
)
PCT_KEYS = {"pct_change_1d", "return_5d_pct", "return_20d_pct", "return_65d_pct", "volume_buzz_pct"}
SIGNED_KEYS = PCT_KEYS | {"rs_score"}


class IndustryPanel(QFrame):
    """TC2000-style "Industry Indexes" board on the Trading Desk.

    Shows the last scan instantly from the CSVs the scanner writes, and the
    Refresh button re-runs the yfinance board on a background thread (no IBKR
    pacing spent), so "what sector is moving / which industry is hot" lives one
    tab away from the setups table.
    """

    statusChanged = Signal(str)
    _refreshFinished = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._refresh_thread: threading.Thread | None = None

        self.refresh_button = QPushButton("Refresh Board (yfinance)")
        self.refresh_button.setObjectName("PrimaryButton")
        self.refresh_button.clicked.connect(self.start_refresh)
        reload_button = QPushButton("Reload From Disk")
        reload_button.clicked.connect(self.reload_from_disk)
        self.as_of_label = QLabel("")
        self.as_of_label.setObjectName("MutedLabel")

        self.sector_table = _make_table(SECTOR_COLUMNS)
        self.industry_table = _make_table(INDUSTRY_COLUMNS)

        self._refreshFinished.connect(self._on_refresh_finished)
        self._build_layout(reload_button)
        self.reload_from_disk()

    def _build_layout(self, reload_button: QPushButton) -> None:
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        action_row.addWidget(self.refresh_button)
        action_row.addWidget(reload_button)
        action_row.addWidget(self.as_of_label)
        action_row.addStretch(1)

        sectors_box = QWidget()
        sectors_layout = QVBoxLayout(sectors_box)
        sectors_layout.setContentsMargins(0, 0, 0, 0)
        sectors_layout.setSpacing(4)
        sector_title = QLabel("Sectors (SPDR ETFs vs SPY)")
        sector_title.setObjectName("SectionTitle")
        sectors_layout.addWidget(sector_title)
        sectors_layout.addWidget(self.sector_table)

        industries_box = QWidget()
        industries_layout = QVBoxLayout(industries_box)
        industries_layout.setContentsMargins(0, 0, 0, 0)
        industries_layout.setSpacing(4)
        industry_title = QLabel("Industry Indexes (composite of members; * = custom group)")
        industry_title.setObjectName("SectionTitle")
        industries_layout.addWidget(industry_title)
        industries_layout.addWidget(self.industry_table)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(sectors_box)
        splitter.addWidget(industries_box)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([300, 460])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(
            SectionHeader(
                "Industry Indexes",
                "What's moving: sector ETFs and composite industry groups ranked by blended RS vs SPY.",
            )
        )
        layout.addLayout(action_row)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def reload_from_disk(self) -> None:
        sector_rows = _read_csv_rows(SECTOR_BOARD_CSV_FILE)
        industry_rows = _read_csv_rows(INDUSTRY_BOARD_CSV_FILE)
        _fill_table(self.sector_table, SECTOR_COLUMNS, sector_rows)
        _fill_table(self.industry_table, INDUSTRY_COLUMNS, industry_rows)
        if SECTOR_BOARD_CSV_FILE.exists():
            as_of = datetime.fromtimestamp(SECTOR_BOARD_CSV_FILE.stat().st_mtime)
            self.as_of_label.setText(f"Board as of {as_of.isoformat(timespec='minutes')}")
        else:
            self.as_of_label.setText("No board yet -- click Refresh to build it.")

    def start_refresh(self) -> None:
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return
        self.refresh_button.setEnabled(False)
        self.as_of_label.setText("Refreshing board (batched yfinance download)...")
        self._refresh_thread = threading.Thread(target=self._refresh_worker, daemon=True)
        self._refresh_thread.start()

    def _refresh_worker(self) -> None:
        try:
            result = run_industry_scan(write_outputs=True)
            message = (
                f"Industry board refreshed: {len(result['sector_rows'])} sectors, "
                f"{len(result['industry_rows'])} industries from {result['symbol_count']} symbols."
            )
        except Exception as exc:  # surfaced in the label, never crashes the desk
            traceback.print_exc()
            message = f"Industry board refresh failed: {exc}"
        self._refreshFinished.emit(message)

    def _on_refresh_finished(self, message: str) -> None:
        self.refresh_button.setEnabled(True)
        self.reload_from_disk()
        self.statusChanged.emit(message)

    def shutdown(self) -> None:
        # Daemon refresh thread finishes on its own; nothing to stop.
        pass


def _make_table(columns: tuple[tuple[str, str], ...]) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels([label for _key, label in columns])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _format_cell(key: str, value) -> str:
    text = str(value if value is not None else "").strip()
    if not text or text.lower() == "none":
        return ""
    if key in SIGNED_KEYS:
        try:
            number = float(text)
        except ValueError:
            return text
        suffix = "%" if key in PCT_KEYS else ""
        return f"{number:+.1f}{suffix}" if key in PCT_KEYS else f"{number:+.2f}"
    return text


def _fill_table(table: QTableWidget, columns: tuple[tuple[str, str], ...], rows: list[dict]) -> None:
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for col_index, (key, _label) in enumerate(columns):
            text = _format_cell(key, row.get(key))
            item = QTableWidgetItem(text)
            if key in SIGNED_KEYS and text.startswith(("+", "-")):
                item.setForeground(QBrush(POSITIVE_COLOR if text.startswith("+") else NEGATIVE_COLOR))
            if key not in {"sector", "industry", "top_movers"}:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row_index, col_index, item)
    table.resizeColumnsToContents()
