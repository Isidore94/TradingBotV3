from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor

from ui import theme
from ui.models.theta import ThetaRow


SORT_ROLE = Qt.ItemDataRole.UserRole + 1
ROW_ROLE = Qt.ItemDataRole.UserRole + 2


class ThetaTableModel(QAbstractTableModel):
    COLUMNS = (
        ("symbol", "Symbol"),
        ("play_type", "Play"),
        ("score", "Score"),
        ("support_count", "Supports"),
        ("close", "Close"),
        ("next_earnings_days", "Earnings DTE"),
        ("recommended_strike", "Strike"),
        ("recommended_credit", "Credit"),
        ("primary_strike_band", "Strike Band"),
        ("liquidity_score", "Liquidity"),
    )

    def __init__(self, rows: list[ThetaRow] | None = None, parent=None) -> None:
        super().__init__(parent)
        self._rows = list(rows or [])

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        key = self.COLUMNS[index.column()][0]
        if role == ROW_ROLE:
            return row
        if role == SORT_ROLE:
            return _sort_value(row, key)
        if role == Qt.ItemDataRole.DisplayRole:
            return _display_value(row, key)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if key in {"score", "support_count", "close", "next_earnings_days", "recommended_strike", "recommended_credit"}:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole:
            if key == "play_type":
                return QColor(theme.color("near" if row.play_type == "pcs" else "long"))
            if key == "recommended_credit" and row.recommended_credit:
                return QColor(theme.color("long"))
        if role == Qt.ItemDataRole.ToolTipRole:
            return _tooltip(row, key)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return self.COLUMNS[section][1]

    def set_rows(self, rows: list[ThetaRow]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, row_index: int) -> ThetaRow | None:
        if 0 <= row_index < len(self._rows):
            return self._rows[row_index]
        return None

    def rows(self) -> list[ThetaRow]:
        return list(self._rows)


class ThetaFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.min_score: int | None = None
        self.min_supports: int | None = None
        self.max_dte: int | None = None
        self.play_type: str = "ALL"
        self.search_text: str = ""
        self.setSortRole(SORT_ROLE)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def set_filters(
        self,
        *,
        min_score: int | None,
        min_supports: int | None,
        max_dte: int | None,
        play_type: str,
        search_text: str,
    ) -> None:
        self.min_score = min_score
        self.min_supports = min_supports
        self.max_dte = max_dte
        self.play_type = play_type
        self.search_text = search_text.strip().upper()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return True
        index = model.index(source_row, 0, source_parent)
        row = model.data(index, ROW_ROLE)
        if not isinstance(row, ThetaRow):
            return True
        if self.play_type != "ALL" and row.play_type != self.play_type:
            return False
        if self.min_score is not None and (row.score is None or row.score < self.min_score):
            return False
        if self.min_supports is not None and (row.support_count is None or row.support_count < self.min_supports):
            return False
        if self.max_dte is not None and row.next_earnings_days is not None and row.next_earnings_days > self.max_dte:
            return False
        if self.search_text and self.search_text not in row.symbol:
            return False
        return True


def _display_value(row: ThetaRow, key: str) -> str:
    if key == "symbol":
        return row.symbol
    if key == "play_type":
        return row.play_label
    if key == "score":
        return "" if row.score is None else str(row.score)
    if key == "support_count":
        return "" if row.support_count is None else str(row.support_count)
    if key == "close":
        return "" if row.close is None else f"{row.close:.2f}"
    if key == "next_earnings_days":
        return "" if row.next_earnings_days is None else str(row.next_earnings_days)
    if key == "recommended_strike":
        return "" if row.recommended_strike is None else f"{row.recommended_strike:.2f}"
    if key == "recommended_credit":
        return "" if row.recommended_credit is None else f"{row.recommended_credit:.2f}"
    if key == "primary_strike_band":
        return row.primary_strike_band
    if key == "liquidity_score":
        return row.liquidity_score
    return ""


def _sort_value(row: ThetaRow, key: str) -> Any:
    if key in {"score", "support_count", "next_earnings_days"}:
        value = getattr(row, key)
        return value if value is not None else -999999
    if key in {"close", "recommended_strike", "recommended_credit"}:
        value = getattr(row, key)
        return value if value is not None else -999999.0
    return _display_value(row, key)


def _tooltip(row: ThetaRow, key: str) -> str:
    if key == "recommended_credit" and row.recommended_credit_source:
        return f"Credit source: {row.recommended_credit_source}"
    if key == "recommended_strike" and row.recommended_long_strike is not None:
        return f"Long leg strike: {row.recommended_long_strike:.2f}"
    if key in {"next_earnings_days", "next_earnings_label"} and row.next_earnings_label:
        return row.next_earnings_label
    if key == "primary_strike_band":
        return row.primary_strike_band
    if key == "liquidity_score":
        return row.liquidity_score
    return ""
