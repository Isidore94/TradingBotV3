from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor

from ui import theme
from ui.models.journal import JournalTrade


SORT_ROLE = Qt.ItemDataRole.UserRole + 1
ROW_ROLE = Qt.ItemDataRole.UserRole + 2


class JournalTableModel(QAbstractTableModel):
    COLUMNS = (
        ("trade_date", "Date"),
        ("symbol", "Symbol"),
        ("direction", "Dir"),
        ("status", "Status"),
        ("quantity", "Qty"),
        ("entry_price", "Entry"),
        ("exit_price", "Exit"),
        ("net_pnl", "Net P&L"),
        ("fees", "Costs"),
        ("tags", "Tags"),
        ("account", "Account"),
    )

    def __init__(self, rows: list[JournalTrade] | None = None, parent=None) -> None:
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
            if key in {"quantity", "entry_price", "exit_price", "net_pnl", "fees"}:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole:
            if key == "net_pnl" and row.net_pnl is not None and row.is_closed:
                return QColor(theme.color("long" if row.net_pnl >= 0 else "short"))
            if key == "direction" and row.direction in {"LONG", "SHORT"}:
                return QColor(theme.color("long" if row.direction == "LONG" else "short"))
            if key == "status":
                return QColor(theme.color("text_secondary" if row.is_closed else "caution"))
        if role == Qt.ItemDataRole.ToolTipRole:
            if key == "tags":
                return row.tags
            if key == "account":
                return f"{row.broker} {row.account}".strip()
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return self.COLUMNS[section][1]

    def set_rows(self, rows: list[JournalTrade]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, source_row: int) -> JournalTrade | None:
        if 0 <= source_row < len(self._rows):
            return self._rows[source_row]
        return None

    def rows(self) -> list[JournalTrade]:
        return list(self._rows)


class JournalFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.status: str = "ALL"
        self.direction: str = "ALL"
        self.search_text: str = ""
        self.setSortRole(SORT_ROLE)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def set_filters(self, *, status: str, direction: str, search_text: str) -> None:
        self.status = status
        self.direction = direction
        self.search_text = search_text.strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return True
        row = model.data(model.index(source_row, 0, source_parent), ROW_ROLE)
        if not isinstance(row, JournalTrade):
            return True
        if self.status != "ALL" and row.status != self.status:
            return False
        if self.direction != "ALL" and row.direction != self.direction:
            return False
        if self.search_text:
            haystack = " ".join([row.symbol, row.tags, row.account, row.broker, row.notes]).lower()
            if self.search_text not in haystack:
                return False
        return True


def _display_value(row: JournalTrade, key: str) -> str:
    if key == "trade_date":
        return row.trade_date
    if key == "symbol":
        return row.symbol
    if key == "direction":
        return row.direction
    if key == "status":
        return row.status.title()
    if key == "quantity":
        return _fmt_qty(row.quantity)
    if key == "entry_price":
        return _fmt_price(row.entry_price)
    if key == "exit_price":
        return _fmt_price(row.exit_price)
    if key == "net_pnl":
        return _fmt_money(row.net_pnl)
    if key == "fees":
        return _fmt_money(row.fees)
    if key == "tags":
        return row.tags
    if key == "account":
        return row.account
    return ""


def _sort_value(row: JournalTrade, key: str) -> Any:
    if key in {"quantity", "entry_price", "exit_price", "net_pnl", "fees"}:
        value = getattr(row, key)
        return value if value is not None else -1e18
    return _display_value(row, key)


def _fmt_qty(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def _fmt_price(value: float | None) -> str:
    return "" if value is None else f"{value:,.2f}"


def _fmt_money(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:,.2f}"
