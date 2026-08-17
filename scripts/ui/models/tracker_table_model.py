from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor

from ui import theme


SORT_ROLE = Qt.ItemDataRole.UserRole + 1
ROW_ROLE = Qt.ItemDataRole.UserRole + 2


class TrackerTableModel(QAbstractTableModel):
    def __init__(
        self,
        columns: tuple[tuple[str, str], ...],
        rows: list[dict[str, Any]] | None = None,
        *,
        percent_keys: set[str] | None = None,
        signed_keys: set[str] | None = None,
        numeric_keys: set[str] | None = None,
        tooltip_keys: set[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.columns = columns
        self._rows = list(rows or [])
        self.percent_keys = set(percent_keys or set())
        self.signed_keys = set(signed_keys or set())
        self.numeric_keys = set(numeric_keys or set()) | self.percent_keys | self.signed_keys
        self.tooltip_keys = set(tooltip_keys or set())
        # R4 section 5: "very obvious I have already checked that chart today".
        # Presentation only - it never filters, reorders, or feeds scoring, and
        # it is built from RECORDED DECISIONS only (trader decision 2026-08-15:
        # no view tracking, zero new capture).
        self._reviewed_symbols: set[str] = set()

    def set_reviewed_symbols(self, symbols) -> None:
        """Mark today's already-decided names. Repaints, never re-sorts."""
        reviewed = {str(symbol or "").strip().upper() for symbol in symbols or ()} - {""}
        if reviewed == self._reviewed_symbols:
            return
        self._reviewed_symbols = reviewed
        if self._rows:
            top = self.index(0, 0)
            bottom = self.index(len(self._rows) - 1, max(0, len(self.columns) - 1))
            self.dataChanged.emit(top, bottom)

    def _is_reviewed(self, key: str, value) -> bool:
        return (
            key == "symbol"
            and bool(self._reviewed_symbols)
            and str(value or "").strip().upper() in self._reviewed_symbols
        )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        key = self.columns[index.column()][0]
        value = row.get(key, "")
        if role == ROW_ROLE:
            return row
        if role == SORT_ROLE:
            if key in self.numeric_keys:
                numeric = _float(value)
                return numeric if numeric is not None else -999999999.0
            return str(value or "")
        if role == Qt.ItemDataRole.DisplayRole:
            text = _format_value(
                value, key, percent_keys=self.percent_keys, signed_keys=self.signed_keys
            )
            # The marker rides the DISPLAY text only. SORT_ROLE above is
            # untouched, so a reviewed name keeps its exact place in the
            # ordering - the badge must never become a ranking.
            if self._is_reviewed(key, value):
                return f"● {text}"
            return text
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if key in self.numeric_keys:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole:
            if key == "side":
                side = str(value or "").upper()
                if side in {"LONG", "SHORT"}:
                    return QColor(theme.color("long" if side == "LONG" else "short"))
            if key == "tier":
                tier = str(value or "").upper()
                if tier == "S":
                    return QColor(theme.color("favorite"))
                if tier == "A":
                    return QColor(theme.color("long"))
            if key in self.signed_keys:
                numeric = _float(value)
                if numeric is not None:
                    return QColor(theme.color("long" if numeric > 0 else "short" if numeric < 0 else "text_secondary"))
        if role == Qt.ItemDataRole.ToolTipRole:
            if self._is_reviewed(key, value):
                return (
                    "You already recorded a decision on this symbol today "
                    "(dislike, favorite, veto, like or note)."
                )
            if key in self.tooltip_keys:
                return str(value or "")
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return self.columns[section][1]

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def row_at(self, row_index: int) -> dict[str, Any] | None:
        if 0 <= row_index < len(self._rows):
            return self._rows[row_index]
        return None


class TrackerSortProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSortRole(SORT_ROLE)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        """Numeric missing values stay last in both sort directions."""
        source = self.sourceModel()
        if (
            isinstance(source, TrackerTableModel)
            and left.column() == right.column()
            and 0 <= left.column() < len(source.columns)
        ):
            key = source.columns[left.column()][0]
            if key in source.numeric_keys:
                left_row = source.row_at(left.row()) or {}
                right_row = source.row_at(right.row()) or {}
                left_missing = _float(left_row.get(key)) is None
                right_missing = _float(right_row.get(key)) is None
                if left_missing != right_missing:
                    if self.sortOrder() == Qt.SortOrder.AscendingOrder:
                        return not left_missing
                    return left_missing
        return super().lessThan(left, right)


def _format_value(value: Any, key: str, *, percent_keys: set[str], signed_keys: set[str]) -> str:
    if value in (None, ""):
        return ""
    numeric = _float(value)
    if key in percent_keys and numeric is not None:
        return f"{numeric * 100:.1f}%"
    if key in signed_keys and numeric is not None and key.endswith("_pct"):
        return f"{numeric:+.2f}%"
    if key in signed_keys and numeric is not None:
        return f"{numeric:+.2f}"
    if numeric is not None and key.endswith("_pct"):
        return f"{numeric:+.2f}%"
    if numeric is not None and (
        key.endswith("_r")
        or key.endswith("_edge")
        or key.endswith("_score")
        or key in {"priority_score", "ranking_score", "robust_closed_r", "median_closed_r"}
    ):
        return f"{numeric:.2f}"
    return str(value)


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
