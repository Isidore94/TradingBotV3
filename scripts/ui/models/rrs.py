from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor

from ui import theme


SORT_ROLE = Qt.ItemDataRole.UserRole + 1
ROW_ROLE = Qt.ItemDataRole.UserRole + 2

SCOPE_KEYS = {"SPY": "results", "Sector": "results_sector", "Industry": "results_industry"}


@dataclass
class RrsRow:
    symbol: str
    side: str  # "RS" (relative strength) or "RW" (relative weakness)
    rrs: float
    power: float | None


def rrs_rows(payload: dict[str, Any] | None, scope: str, *, apply_threshold: bool = True) -> list[RrsRow]:
    """Parse one results list (vs SPY / Sector / Industry) into ranked rows.

    Mirrors the legacy filter: only RS/RW rows whose |RRS| clears the snapshot
    threshold, sorted strongest (most positive) to weakest (most negative).
    """
    if not isinstance(payload, dict):
        return []
    items = payload.get(SCOPE_KEYS.get(scope, "results"))
    if not isinstance(items, list):
        return []
    threshold = _float(payload.get("threshold")) or 0.0
    rows: list[RrsRow] = []
    for item in items:
        parsed = _result_fields(item)
        if parsed is None:
            continue
        side, symbol, rrs, power = parsed
        if not symbol or side not in {"RS", "RW"} or rrs is None:
            continue
        if apply_threshold and abs(rrs) < threshold:
            continue
        rows.append(RrsRow(symbol=symbol, side=side, rrs=rrs, power=power))
    rows.sort(key=lambda row: -row.rrs)
    return rows


def _result_fields(item: Any) -> tuple[str, str, float | None, float | None] | None:
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        power = _float(item[3]) if len(item) > 3 else None
        return str(item[0] or "").upper(), str(item[1] or "").upper(), _float(item[2]), power
    if isinstance(item, dict):
        side = str(item.get("signal") or item.get("side") or "").upper()
        symbol = str(item.get("symbol") or item.get("ticker") or "").upper()
        power = _float(item.get("power", item.get("power_index")))
        return side, symbol, _float(item.get("rrs")), power
    return None


class RrsTableModel(QAbstractTableModel):
    COLUMNS = (
        ("symbol", "Symbol"),
        ("side", "Side"),
        ("rrs", "RRS"),
        ("power", "Power"),
    )

    def __init__(self, rows: list[RrsRow] | None = None, parent=None) -> None:
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
            if key in {"rrs", "power"}:
                value = getattr(row, key)
                return value if value is not None else -1e18
            return getattr(row, key)
        if role == Qt.ItemDataRole.DisplayRole:
            if key == "symbol":
                return row.symbol
            if key == "side":
                return row.side
            if key == "rrs":
                return f"{row.rrs:+.2f}" if row.rrs is not None else ""
            if key == "power":
                return f"{row.power:+.2f}" if row.power is not None else ""
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if key in {"rrs", "power"}:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole:
            if key == "side":
                return QColor(theme.color("long" if row.side == "RS" else "short"))
            if key == "rrs" and row.rrs is not None:
                return QColor(theme.color("long" if row.rrs >= 0 else "short"))
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return self.COLUMNS[section][1]

    def set_rows(self, rows: list[RrsRow]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def rows(self) -> list[RrsRow]:
        return list(self._rows)


class RrsSortProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSortRole(SORT_ROLE)


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
