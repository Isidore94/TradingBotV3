from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor

from ui import theme
from ui.models.setup import SetupRow


SORT_ROLE = Qt.ItemDataRole.UserRole + 1
ROW_ROLE = Qt.ItemDataRole.UserRole + 2


class SetupTableModel(QAbstractTableModel):
    COLUMNS = (
        ("symbol", "Symbol"),
        ("side", "Side"),
        ("score", "Score"),
        ("bucket", "Bucket"),
        ("setup_tags", "Setup Tags"),
        ("key_level", "Key Level / Entry"),
        ("supports", "Supports"),
        ("theta", "Theta"),
        ("expected_r", "Expected R"),
        ("days_to_earnings", "DTE"),
        ("last_trade_date", "Last Bar"),
    )

    def __init__(self, rows: list[SetupRow] | None = None, parent=None) -> None:
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
            return self._sort_value(row, key)
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_value(row, key)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if key in {"score", "supports", "expected_r", "days_to_earnings"}:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole:
            if key == "side" and row.side in {"LONG", "SHORT"}:
                return QColor(theme.color("long" if row.side == "LONG" else "short"))
            if key == "bucket":
                return QColor(_bucket_color(row.bucket))
            if key == "score" and row.score is not None:
                if row.score >= 80:
                    return QColor(theme.color("long"))
                if row.score < 45:
                    return QColor(theme.color("caution"))
            return None
        if role == Qt.ItemDataRole.ToolTipRole:
            return _tooltip(row, key)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return self.COLUMNS[section][1]

    def set_rows(self, rows: list[SetupRow]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, source_row: int) -> SetupRow | None:
        if 0 <= source_row < len(self._rows):
            return self._rows[source_row]
        return None

    def rows(self) -> list[SetupRow]:
        return list(self._rows)

    def _display_value(self, row: SetupRow, key: str) -> str:
        if key == "symbol":
            return row.symbol
        if key == "side":
            return row.side
        if key == "score":
            return "" if row.score is None else f"{row.score:.1f}"
        if key == "bucket":
            return row.bucket_label
        if key == "setup_tags":
            return row.tags_text
        if key == "key_level":
            return row.key_level
        if key == "supports":
            return row.supports_text
        if key == "theta":
            return row.theta
        if key == "expected_r":
            return row.expected_r_text
        if key == "days_to_earnings":
            return "" if row.days_to_earnings is None else str(row.days_to_earnings)
        if key == "last_trade_date":
            return row.last_trade_date
        return ""

    def _sort_value(self, row: SetupRow, key: str) -> Any:
        if key == "score":
            return row.score if row.score is not None else -999999.0
        if key == "supports":
            return row.supports if row.supports is not None else -1
        if key == "expected_r":
            return row.expected_r if row.expected_r is not None else -999999.0
        if key == "days_to_earnings":
            return row.days_to_earnings if row.days_to_earnings is not None else 999999
        return self._display_value(row, key)


class SetupFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.min_score: float = 0.0
        self.side: str = "ALL"
        self.bucket: str = "ALL"
        self.max_dte: int | None = None
        self.search_text: str = ""
        self.setSortRole(SORT_ROLE)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def set_filters(
        self,
        *,
        min_score: float | None = None,
        side: str | None = None,
        bucket: str | None = None,
        max_dte: int | None = None,
        search_text: str | None = None,
    ) -> None:
        if min_score is not None:
            self.min_score = float(min_score)
        if side is not None:
            self.side = side
        if bucket is not None:
            self.bucket = bucket
        self.max_dte = max_dte
        if search_text is not None:
            self.search_text = search_text.strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return True
        index = model.index(source_row, 0, source_parent)
        row = model.data(index, ROW_ROLE)
        if not isinstance(row, SetupRow):
            return True

        if row.score is not None and row.score < self.min_score:
            return False
        if self.side != "ALL" and row.side != self.side:
            return False
        if self.bucket != "ALL" and row.bucket_label != self.bucket:
            return False
        if self.max_dte is not None and row.days_to_earnings is not None and row.days_to_earnings > self.max_dte:
            return False
        if self.search_text:
            haystack = " ".join(
                [
                    row.symbol,
                    row.side,
                    row.bucket_label,
                    row.tags_text,
                    row.key_level,
                    row.theta,
                ]
            ).lower()
            if self.search_text not in haystack:
                return False
        return True


def _bucket_color(bucket: str) -> str:
    normalized = bucket.strip().lower()
    if normalized in {"favorite_setup", "high_conviction"}:
        return theme.color("favorite")
    if normalized == "near_favorite_zone":
        return theme.color("near")
    if "study" in normalized:
        return theme.color("study")
    return theme.color("text_secondary")


def _tooltip(row: SetupRow, key: str) -> str:
    if key == "supports" and row.raw.get("hv_level_note"):
        return str(row.raw.get("hv_level_note"))
    if key == "expected_r" and row.raw.get("expected_r_note"):
        return str(row.raw.get("expected_r_note"))
    if key == "setup_tags":
        return row.tags_text
    return ""
