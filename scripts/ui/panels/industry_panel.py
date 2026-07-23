from __future__ import annotations

import csv
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
)
from ui.services.industry_board_service import IndustryBoardService, inspect_industry_snapshot
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
NUMERIC_KEYS = SIGNED_KEYS | {"rs_rank", "member_count"}


class IndustryPanel(QFrame):
    """TC2000-style "Industry Indexes" board on the Trading Desk.

    Shows the last scan instantly from the CSVs the scanner writes, and the
    Refresh button re-runs the yfinance board on a background thread (no IBKR
    pacing spent), so "what sector is moving / which industry is hot" lives one
    tab away from the setups table.
    """

    statusChanged = Signal(str)
    def __init__(self, parent=None, *, service: IndustryBoardService | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.service = service or IndustryBoardService(self)
        self._bounce_service = None

        self.refresh_button = QPushButton("Refresh Board (yfinance)")
        self.refresh_button.setObjectName("PrimaryButton")
        self.refresh_button.clicked.connect(lambda: self.service.request_refresh(force=True))
        reload_button = QPushButton("Reload From Disk")
        reload_button.clicked.connect(self.reload_from_disk)
        strongest_button = QPushButton("Strongest")
        strongest_button.clicked.connect(
            lambda: self._sort_strength(Qt.SortOrder.DescendingOrder)
        )
        weakest_button = QPushButton("Weakest")
        weakest_button.clicked.connect(
            lambda: self._sort_strength(Qt.SortOrder.AscendingOrder)
        )
        self.as_of_label = QLabel("")
        self.as_of_label.setObjectName("MutedLabel")

        self.sector_table = _make_table(SECTOR_COLUMNS)
        self.industry_table = _make_table(INDUSTRY_COLUMNS)
        self.sector_table.cellClicked.connect(self._on_sector_table_clicked)

        self.service.refreshStarted.connect(self._on_refresh_started)
        self.service.refreshFinished.connect(self._on_refresh_finished)
        self.service.snapshotChanged.connect(self._on_snapshot_changed)
        self._build_layout(reload_button, strongest_button, weakest_button)
        self.reload_from_disk()
        self.service.start()

    def _build_layout(
        self,
        reload_button: QPushButton,
        strongest_button: QPushButton,
        weakest_button: QPushButton,
    ) -> None:
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        action_row.addWidget(self.refresh_button)
        action_row.addWidget(reload_button)
        action_row.addWidget(strongest_button)
        action_row.addWidget(weakest_button)
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
    def reload_from_disk(self, snapshot: dict | None = None) -> None:
        sector_rows = _read_csv_rows(SECTOR_BOARD_CSV_FILE)
        industry_rows = _read_csv_rows(INDUSTRY_BOARD_CSV_FILE)
        _fill_table(self.sector_table, SECTOR_COLUMNS, sector_rows)
        _fill_table(self.industry_table, INDUSTRY_COLUMNS, industry_rows)
        snapshot = snapshot or inspect_industry_snapshot()
        as_of = snapshot.get("as_of")
        if isinstance(as_of, datetime):
            state = str(snapshot.get("state") or "unknown").upper()
            snapshot_id = str(snapshot.get("snapshot_id") or "")
            self.as_of_label.setText(
                f"{state} | board as of {as_of.isoformat(timespec='minutes')} | "
                f"snapshot {snapshot_id or 'unknown'}"
            )
        else:
            self.as_of_label.setText("No board yet -- click Refresh to build it.")

    def start_refresh(self) -> None:
        self.service.request_refresh(force=True)

    def set_bounce_service(self, service) -> None:
        self._bounce_service = service

    def _on_sector_table_clicked(self, row: int, column: int) -> None:
        etf_column = next(
            index for index, (key, _label) in enumerate(SECTOR_COLUMNS) if key == "etf"
        )
        if column != etf_column:
            return
        item = self.sector_table.item(row, etf_column)
        symbol = str(item.text() if item is not None else "").strip().upper()
        if not symbol:
            return
        rs_column = next(
            index for index, (key, _label) in enumerate(SECTOR_COLUMNS) if key == "rs_score"
        )
        rs_item = self.sector_table.item(row, rs_column)
        try:
            rs_score = float(rs_item.text()) if rs_item is not None else 0.0
        except ValueError:
            rs_score = 0.0
        side = "LONG" if rs_score > 0 else "SHORT" if rs_score < 0 else ""
        bot = None
        if self._bounce_service is not None:
            try:
                bot = self._bounce_service.current_bot()
            except Exception:
                bot = None
        from ui.widgets.symbol_snapshot_dialog import show_symbol_snapshot

        show_symbol_snapshot(self, symbol, bot=bot, side=side)

    def _on_refresh_started(self) -> None:
        self.refresh_button.setEnabled(False)
        self.as_of_label.setText("Refreshing board (batched yfinance download)...")

    def _on_refresh_finished(self, result: dict) -> None:
        self.refresh_button.setEnabled(True)
        self.reload_from_disk(result.get("snapshot") if isinstance(result, dict) else None)
        if result.get("ok"):
            message = (
                f"Industry board refreshed: {result.get('sector_count', 0)} sectors, "
                f"{result.get('industry_count', 0)} industries from "
                f"{result.get('symbol_count', 0)} symbols."
            )
        else:
            message = (
                f"Industry board refresh failed: {result.get('error') or 'unknown error'}. "
                "The last good snapshot remains visible."
            )
        self.statusChanged.emit(message)

    def _on_snapshot_changed(self, snapshot: dict) -> None:
        if not self.service.running:
            self.reload_from_disk(snapshot)

    def _sort_strength(self, order: Qt.SortOrder) -> None:
        for table, columns in (
            (self.sector_table, SECTOR_COLUMNS),
            (self.industry_table, INDUSTRY_COLUMNS),
        ):
            rs_column = next(index for index, (key, _label) in enumerate(columns) if key == "rs_score")
            table.sortItems(rs_column, order)

    def shutdown(self) -> None:
        self.service.shutdown()


def _make_table(columns: tuple[tuple[str, str], ...]) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels([label for _key, label in columns])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    table.horizontalHeader().setStretchLastSection(True)
    table.setSortingEnabled(True)
    rs_column = next(index for index, (key, _label) in enumerate(columns) if key == "rs_score")
    table.sortItems(rs_column, Qt.SortOrder.DescendingOrder)
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
    sorting_enabled = table.isSortingEnabled()
    header = table.horizontalHeader()
    sort_column = header.sortIndicatorSection()
    sort_order = header.sortIndicatorOrder()
    table.setSortingEnabled(False)
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for col_index, (key, _label) in enumerate(columns):
            text = _format_cell(key, row.get(key))
            sort_value = _numeric_value(row.get(key)) if key in NUMERIC_KEYS else None
            item = _SortableTableItem(text, sort_value=sort_value)
            if key in SIGNED_KEYS and text.startswith(("+", "-")):
                item.setForeground(QBrush(POSITIVE_COLOR if text.startswith("+") else NEGATIVE_COLOR))
            if key not in {"sector", "industry", "top_movers"}:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row_index, col_index, item)
    table.resizeColumnsToContents()
    table.setSortingEnabled(sorting_enabled)
    if sorting_enabled and sort_column >= 0:
        table.sortItems(sort_column, sort_order)


def _numeric_value(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class _SortableTableItem(QTableWidgetItem):
    def __init__(self, text: str, *, sort_value: float | None = None) -> None:
        super().__init__(text)
        self._sort_value = sort_value

    def __lt__(self, other) -> bool:
        if isinstance(other, _SortableTableItem):
            if self._sort_value is not None and other._sort_value is not None:
                return self._sort_value < other._sort_value
            if self._sort_value is None and other._sort_value is not None:
                return True
            if self._sort_value is not None and other._sort_value is None:
                return False
        return super().__lt__(other)
