"""The Movers board: what is popping now, and who is strong or weak in a SPY turn.

Sits at the top of the Alert Center's lower-right column (trader, 2026-09-23).
Two modes. Pop stacks three tables (trader, 2026-09-24): Pop (longs and shorts
together), then Dip-strong and Dip-weak, lit by a SPY pullback or bounce. My
names is one table with a Long/Short toggle. A SPY state banner tops both.
Names new to a list get a tinted Sym cell. Header clicks sort (third click = board order); the trader can hide a
row for the day (right-click or Delete) and bring hidden rows back. The "Review" menu holds the Focus pick and
Faded review doors; "Deep read" shows the old Strength page (Focus strength,
entry board, RRS snapshot, M5 Strength Board) underneath.

Display only. It owns no data, timer or fetch: `MoversService` publishes and
this widget renders, coalesced. No stylesheets are set here; colours ride the
model's roles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, Signal

#: The invalid (root) index used as the default parent.
_NO_PARENT = QModelIndex()
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QSizePolicy,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import movers_scan
from ui import theme
from ui.timer_utils import SignalCoalescer
from swallowed import note_swallowed

#: Same floor as the Strength page it sits above (alert column budget: 360 px).
MIN_BOARD_WIDTH = 170
#: Rows each table shows without scrolling: the main table, and each dip table.
VISIBLE_ROWS = 6
DIP_VISIBLE_ROWS = 4
#: Width that one numeric column needs; narrower tables show fewer columns.
COLUMN_MIN_PX = 48
SYMBOL_COLUMN_PX = 88
LVL_COLUMN_PX = 70

MODES = ("pop", "mine")
MODE_LABELS = {"pop": "Pop + Dip", "mine": "My names"}
MODE_SHORT = {"pop": "Pop", "mine": "Mine"}
#: Below this width the chips and header buttons use short labels.
NARROW_PX = 300
MOVERS_MODE_SETTING = "movers_board_mode"
MOVERS_SIDE_SETTING = "movers_board_side"
MOVERS_DEEP_READ_SETTING = "movers_board_deep_read"
#: {"day": "YYYY-MM-DD", "keys": ["SYM|side", ...]}: rows the trader hid today.
MOVERS_HIDDEN_SETTING = "movers_board_hidden"
#: Raw value a column sorts by (None = unmeasured, always last).
SORT_ROLE = int(Qt.ItemDataRole.UserRole) + 1
_TEXT_SORT_KEYS = {"symbol", "group"}

#: (key, header) per mode. Priority order: Sym, main score, RVOL, Lvl, then the
#: rest; narrow widths drop columns from the end.
COLUMNS = {
    "pop": (("symbol", "Sym"), ("move15_pct", "15m"), ("rvol", "RVOL"), ("lvl", "Lvl"),
            ("vs_spy15_pct", "vSPY"), ("move30_pct", "30m"), ("day_pct", "Day"),
            ("group", "Grp")),
    "dip": (("symbol", "Sym"), ("dip_score", "xSPY"), ("rvol", "RVOL"), ("lvl", "Lvl"),
            ("since_start_pct", "Since"), ("day_pct", "Day"), ("move15_pct", "15m"),
            ("group", "Grp")),
    "mine": (("symbol", "Sym"), ("move15_pct", "15m"), ("rvol", "RVOL"), ("lvl", "Lvl"),
             ("day_pct", "Day"), ("vs_spy15_pct", "vSPY"), ("since_start_pct", "Since"),
             ("group", "Grp")),
}
_PCT_KEYS = {"move15_pct", "move30_pct", "day_pct", "vs_spy15_pct", "since_start_pct"}


def symbol_text(row: dict[str, Any]) -> str:
    """Symbol, an ER tag, and the rank change since the last tick (lists only)."""
    parts = [str(row.get("symbol") or "")]
    if row.get("er"):
        parts.append("ER")
    if "streak" in row:
        change = row.get("rank_change")
        if change is None:
            parts.append("new")
        elif change > 0:
            parts.append(f"▲{change}")
        elif change < 0:
            parts.append(f"▼{-change}")
    return " ".join(parts)


def is_new(row: dict[str, Any]) -> bool:
    """First tick on this list (lists only; My names carries no streak)."""
    return "streak" in row and row.get("rank_change") is None


def level_text(row: dict[str, Any], side: str) -> str:
    """Compact Lvl cell: a break/extension tag, else ATRs from the day's extreme."""
    long_side = side != "short"
    brk = row.get("hod_break") if long_side else row.get("lod_break")
    ext = row.get("ext_up") if long_side else row.get("ext_down")
    if brk and ext:
        return "brk ext"
    if brk:
        return "HOD brk" if long_side else "LOD brk"
    if ext:
        return "ext"
    value = row.get("from_hod_atr") if long_side else row.get("from_lod_atr")
    if value is None:
        return "—"
    return f"{float(value):+.1f}{'H' if long_side else 'L'}"


def format_cell(key: str, value: Any) -> str:
    if key == "symbol":
        return str(value or "")
    if key == "group":
        return str(value or "")
    if value is None:
        return "—"
    if key == "rvol":
        return f"{float(value):.1f}x"
    if key == "dip_score":
        return f"{float(value):+.1f}"
    if key in _PCT_KEYS:
        return f"{float(value):+.2f}"
    return str(value)


def banner_text(state: dict[str, Any] | None) -> str:
    """One line on SPY: pullback/bounce, plain up/down day, or unknown."""
    state = state or {}
    kind = state.get("state") or "unknown"
    if kind == "unknown":
        return "SPY: unknown (no completed bars)"
    off = state.get("spy_from_extreme_pct")
    when = _local_clock(state.get("start_dt")) or state.get("extreme_time") or ""
    if state.get("pullback") and off is not None:
        return f"SPY {off:+.2f}% from {when} high · up day · PULLBACK"
    if state.get("bounce") and off is not None:
        return f"SPY {off:+.2f}% from {when} low · down day · BOUNCE"
    day = state.get("spy_day_pct")
    day_text = f" · SPY {day:+.2f}% on the day" if day is not None else ""
    label = {"up_day": "up day", "down_day": "down day", "flat": "flat"}.get(kind, kind)
    return f"{label} · no pullback{day_text}" if kind == "up_day" else (
        f"{label} · no bounce{day_text}" if kind == "down_day" else f"{label}{day_text}"
    )


def rows_for(board: dict[str, Any] | None, mode: str, side: str) -> list[dict[str, Any]]:
    """Rows for one list, each tagged `_side`. Pop shows both sides, biggest move first;
    "strong"/"weak" are the dip lists (beating / lagging SPY since the turn)."""
    board = board or {}
    if mode in ("strong", "weak"):
        side = "long" if mode == "strong" else "short"
        return [dict(row, _side=side) for row in (((board.get("dip") or {}).get(side)) or [])]
    if mode == "pop":
        both = [dict(row, _side=s) for s in ("long", "short")
                for row in (((board.get(mode) or {}).get(s)) or [])]
        return sorted(both, key=lambda r: -abs(float(r.get("pop_score") or 0.0)))
    rows = list(((board.get(mode) or {}).get(side)) or [])
    if mode == "mine":
        dip_live = bool((board.get("state") or {}).get("pullback" if side == "long" else "bounce"))
        rows = movers_scan.sort_mine(rows, "dip" if dip_live else "pop", side)
    return [dict(row, _side=side) for row in rows]


def hidden_key(row: dict[str, Any]) -> str:
    return f"{str(row.get('symbol') or '').strip().upper()}|{row.get('_side') or 'long'}"


def sort_value(row: dict[str, Any], key: str) -> Any:
    """What a column sorts by; None sorts last either way."""
    if key in _TEXT_SORT_KEYS:
        return str(row.get(key) or "").upper() or None
    if key == "lvl":
        long_side = row.get("_side") != "short"
        brk = row.get("hod_break") if long_side else row.get("lod_break")
        ext = row.get("ext_up") if long_side else row.get("ext_down")
        if brk or ext:
            return 1.0 + (2.0 if brk else 0.0) + (1.0 if ext else 0.0)
        value = row.get("from_hod_atr") if long_side else row.get("from_lod_atr")
        return None if value is None else -abs(float(value))
    value = row.get(key)
    return None if value is None else float(value)


class MoversSortProxy(QSortFilterProxyModel):
    """Sorts on SORT_ROLE with unmeasured (None) rows last in both directions."""

    def lessThan(self, left, right) -> bool:  # noqa: N802 - Qt API
        a = self.sourceModel().data(left, SORT_ROLE)
        b = self.sourceModel().data(right, SORT_ROLE)
        if a is None or b is None:
            if a is None and b is None:
                return left.row() < right.row()
            # Qt flips the comparison for descending; keep None at the bottom anyway.
            none_last = b is None
            return none_last if self.sortOrder() == Qt.SortOrder.AscendingOrder else not none_last
        if a == b:
            return left.row() < right.row()
        return a < b


class MoversTableModel(QAbstractTableModel):
    """Rows as dicts; updates in place (diff), never a model reset."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[dict[str, Any]] = []
        self._columns = COLUMNS["pop"]
        self._side = "long"

    def rowCount(self, parent=_NO_PARENT) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=_NO_PARENT) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._columns)

    def row(self, index: int) -> dict[str, Any] | None:
        return self._rows[index] if 0 <= index < len(self._rows) else None

    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def set_rows(self, rows: list[dict[str, Any]], mode: str, side: str) -> None:
        columns = COLUMNS.get(mode, COLUMNS["pop"])
        if columns != self._columns:
            self._columns = columns
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(columns) - 1)
        self._side = side
        old, new = len(self._rows), len(rows)
        if new < old:
            self.beginRemoveRows(QModelIndex(), new, old - 1)
            self._rows = self._rows[:new]
            self.endRemoveRows()
        elif new > old:
            self.beginInsertRows(QModelIndex(), old, new - 1)
            self._rows = self._rows + [dict(r) for r in rows[old:]]
            self.endInsertRows()
        self._rows = [dict(r) for r in rows]
        if new:
            self.dataChanged.emit(self.index(0, 0), self.index(new - 1, len(columns) - 1))

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if 0 <= section < len(self._columns):
                return self._columns[section][1]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        key = self._columns[index.column()][0]
        value = row.get(key)
        side = row.get("_side") or self._side
        if role == SORT_ROLE:
            return sort_value(row, key)
        if role == Qt.ItemDataRole.DisplayRole:
            if key == "symbol":
                return symbol_text(row)
            if key == "lvl":
                return level_text(row, side)
            return format_cell(key, value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if key == "symbol":
                return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole:
            if key == "symbol":
                return QColor(theme.color("long" if side == "long" else "short"))
            if key in _PCT_KEYS or key == "dip_score":
                if value is None:
                    return QColor(theme.color("text_secondary"))
                return QColor(theme.color("long" if float(value) >= 0 else "short"))
            if key == "rvol" and value is None:
                return QColor(theme.color("text_secondary"))
        if role == Qt.ItemDataRole.BackgroundRole and key == "symbol" and is_new(row):
            color = QColor(theme.color("accent"))
            color.setAlphaF(0.35)
            return color
        if role == Qt.ItemDataRole.BackgroundRole and key == "rvol" and value is not None:
            # Busier than usual reads warmer: alpha grows from 1x to 3x.
            strength = max(0.0, min(1.0, (float(value) - 1.0) / 2.0))
            if strength > 0:
                color = QColor(theme.color("caution"))
                color.setAlphaF(0.12 + 0.45 * strength)
                return color
        if role == Qt.ItemDataRole.ToolTipRole:
            return _row_tooltip(row)
        return None


def _row_tooltip(row: dict[str, Any]) -> str:
    parts = [str(row.get("symbol") or "")]
    for key, label in (("move15_pct", "15m %"), ("move30_pct", "30m %"), ("day_pct", "day %"),
                       ("vs_spy15_pct", "vs SPY 15m"), ("since_start_pct", "since start %")):
        parts.append(f"{label} {format_cell(key, row.get(key))}")
    parts.append(f"RVOL {format_cell('rvol', row.get('rvol'))}")
    for key, label in (("from_hod_atr", "from HOD"), ("from_lod_atr", "from LOD"),
                       ("from_vwap_atr", "from VWAP")):
        value = row.get(key)
        parts.append(f"{label} {'—' if value is None else f'{float(value):+.1f} ATR'}")
    if row.get("streak"):
        parts.append(f"on this list {row['streak']} tick(s)")
    if row.get("group"):
        parts.append(f"group {row['group']}")
    if row.get("er"):
        parts.append("earnings today / after last close")
    if row.get("stale"):
        parts.append("stale bars")
    if row.get("note"):
        parts.append(str(row["note"]))
    return " · ".join(parts)


class MoversSection(QWidget):
    """One titled table with its own model, sort proxy and header-click sort."""

    def __init__(self, parent=None, *, titled: bool = False) -> None:
        super().__init__(parent)
        self.columns_mode = "pop"
        # Column sort (key, order) or None = board order.
        self.sort: tuple[str, Qt.SortOrder] | None = None
        self.title_label = QLabel("")
        self.title_label.setObjectName("MutedLabel")
        self.title_label.setVisible(titled)
        self.model = MoversTableModel(self)
        self.proxy = MoversSortProxy(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(SORT_ROLE)
        self.table = QTableView()
        self.table.setObjectName("MoversTable")
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setWordWrap(False)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setMinimumSectionSize(theme.px(30))
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().setSortIndicatorShown(False)
        self.table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.empty_label = QLabel("")
        self.empty_label.setObjectName("MutedLabel")
        self.empty_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.title_label)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.empty_label)

    def set_rows(self, rows: list[dict[str, Any]], columns_mode: str, side: str) -> None:
        self.columns_mode = columns_mode
        self.model.set_rows(rows, columns_mode, side)
        self.apply_sort()

    def columns(self):
        return COLUMNS.get(self.columns_mode, COLUMNS["pop"])

    def _on_header_clicked(self, column: int) -> None:
        """Numbers sort biggest first, text A-Z; the next click flips; the third restores board order."""
        columns = self.columns()
        if not 0 <= column < len(columns):
            return
        key = columns[column][0]
        first = (Qt.SortOrder.AscendingOrder if key in _TEXT_SORT_KEYS
                 else Qt.SortOrder.DescendingOrder)
        if self.sort is None or self.sort[0] != key:
            self.sort = (key, first)
        elif self.sort[1] == first:
            flipped = (Qt.SortOrder.DescendingOrder if first == Qt.SortOrder.AscendingOrder
                       else Qt.SortOrder.AscendingOrder)
            self.sort = (key, flipped)
        else:
            self.sort = None
        self.apply_sort()

    def apply_sort(self) -> None:
        header = self.table.horizontalHeader()
        keys = [k for k, _h in self.columns()]
        if self.sort is None or self.sort[0] not in keys:
            self.proxy.sort(-1)
            header.setSortIndicatorShown(False)
            return
        column = keys.index(self.sort[0])
        self.proxy.sort(column, self.sort[1])
        header.setSortIndicatorShown(True)
        header.setSortIndicator(column, self.sort[1])

    def visible_rows(self) -> list[dict[str, Any]]:
        """Rows in the order the table shows them."""
        rows = []
        for r in range(self.proxy.rowCount()):
            row = self.model.row(self.proxy.mapToSource(self.proxy.index(r, 0)).row())
            if row is not None:
                rows.append(row)
        return rows

    def owns(self, index) -> bool:
        return index.isValid() and index.model() in (self.proxy, self.model)

    def source_row(self, index) -> dict[str, Any] | None:
        if not self.owns(index):
            return None
        if index.model() is self.proxy:
            index = self.proxy.mapToSource(index)
        return self.model.row(index.row())

    def selected_row(self) -> dict[str, Any] | None:
        selection = self.table.selectionModel()
        rows = selection.selectedRows() if selection is not None else []
        return self.source_row(rows[0]) if rows else None

    def set_min_rows(self, rows: int) -> None:
        row_height = self.table.fontMetrics().height() + theme.px(6)
        self.table.verticalHeader().setDefaultSectionSize(row_height)
        chrome = self.table.horizontalHeader().sizeHint().height() + 2 * self.table.frameWidth()
        self.table.setMinimumHeight(chrome + rows * row_height)

    def fit_columns(self, count: int, column_px) -> None:
        header = self.table.horizontalHeader()
        for column, (key, _h) in enumerate(self.columns()):
            hidden = column >= count
            if self.table.isColumnHidden(column) != hidden:
                self.table.setColumnHidden(column, hidden)
            if key in ("symbol", "lvl") and column < self.model.columnCount():
                if header.sectionResizeMode(column) != QHeaderView.ResizeMode.Fixed:
                    header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
                if header.sectionSize(column) != column_px(key):
                    header.resizeSection(column, column_px(key))


class MoversBoard(QWidget):
    """Header, banner, and the Pop / Dip-strong / Dip-weak tables. The Alert Center wires its signals."""

    symbolActivated = Signal(str, str)
    reviewAllRequested = Signal()
    fadedReviewRequested = Signal()
    deepReadToggled = Signal(bool)
    #: The trader's explicit "+F" click: (symbol, "long"|"short"). Never automatic.
    focusAddRequested = Signal(str, str)

    def __init__(self, parent=None, *, persist: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("MoversBoard")
        self._persist = persist
        self._board: dict[str, Any] = {}
        self._mode = self._setting(MOVERS_MODE_SETTING, "pop")
        if self._mode not in MODES:
            self._mode = "pop"
        self._side = self._setting(MOVERS_SIDE_SETTING, "long")
        if self._side not in ("long", "short"):
            self._side = "long"
        self._auto_switched_episode = ""
        # Session date + day state the side last followed (or the trader overrode).
        self._day_followed = ""
        self._focus_service = None
        # Rows the trader hid today ("SYM|side").
        self._hidden_day, self._hidden = self._load_hidden()
        self._render_coalescer = SignalCoalescer(self._render, parent=self)
        self._counts_coalescer = SignalCoalescer(self._render_counts, parent=self)

        self.title_label = QLabel("MOVERS")
        self.title_label.setObjectName("SectionTitle")
        self.meta_label = QLabel("--:--")
        self.meta_label.setObjectName("MutedLabel")

        self.review_button = QToolButton()
        self.review_button.setObjectName("MoversChip")
        self.review_button.setText("Review ▾")
        self.review_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.review_menu = QMenu(self.review_button)
        self.focus_review_action = QAction("Focus pick review (0)", self)
        self.focus_review_action.triggered.connect(self.reviewAllRequested)
        self.faded_review_action = QAction("Faded review (0)", self)
        self.faded_review_action.triggered.connect(self.fadedReviewRequested)
        self.review_menu.addAction(self.focus_review_action)
        self.review_menu.addAction(self.faded_review_action)
        self.review_button.setMenu(self.review_menu)

        self.deep_read_button = QToolButton()
        self.deep_read_button.setObjectName("MoversChip")
        self.deep_read_button.setText("Deep read")
        self.deep_read_button.setCheckable(True)
        self.deep_read_button.setToolTip(
            "Show the Focus strength lane, entry board, RRS snapshot and M5 Strength Board."
        )
        self.deep_read_button.toggled.connect(self._on_deep_read)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.review_button)
        header.addWidget(self.deep_read_button)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QToolButton] = {}
        modes_row = QHBoxLayout()
        modes_row.setContentsMargins(0, 0, 0, 0)
        modes_row.setSpacing(2)
        for mode in MODES:
            button = QToolButton()
            button.setObjectName("MoversChip")
            button.setCheckable(True)
            button.setText(MODE_LABELS[mode])
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _checked=False, m=mode: self.set_mode(m, user=True))
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            modes_row.addWidget(button)
        self.side_button = QToolButton()
        self.side_button.setObjectName("MoversChip")
        self.side_button.setCheckable(True)
        self.side_button.clicked.connect(self._on_side_clicked)
        modes_row.addWidget(self.side_button)
        self.add_focus_button = QToolButton()
        self.add_focus_button.setObjectName("MoversChip")
        self.add_focus_button.setText("+F")
        self.add_focus_button.setToolTip(
            "Add the selected row to M5 Focus on this side (through the adoption gate)."
        )
        self.add_focus_button.setEnabled(False)
        self.add_focus_button.clicked.connect(self._add_selected_to_focus)
        modes_row.addWidget(self.add_focus_button)
        self.unhide_button = QToolButton()
        self.unhide_button.setObjectName("MoversChip")
        self.unhide_button.setToolTip("Show the rows you hid today again.")
        self.unhide_button.setVisible(False)
        self.unhide_button.clicked.connect(self.unhide_all)
        modes_row.addWidget(self.unhide_button)

        self.banner = QLabel(banner_text(None))
        self.banner.setObjectName("MutedLabel")
        self.banner.setWordWrap(True)
        banner_row = QHBoxLayout()
        banner_row.setContentsMargins(0, 0, 0, 0)
        banner_row.setSpacing(6)
        banner_row.addWidget(self.meta_label, 0, Qt.AlignmentFlag.AlignTop)
        banner_row.addWidget(self.banner, 1)

        # Main table (Pop or My names), then the two dip tables under it.
        self.main = MoversSection(self)
        self.strong = MoversSection(self, titled=True)
        self.weak = MoversSection(self, titled=True)
        self.sections = (self.main, self.strong, self.weak)
        self.model = self.main.model
        self.proxy = self.main.proxy
        self.table = self.main.table
        self.empty_label = self.main.empty_label
        for section in self.sections:
            table = section.table
            hide_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), table)
            hide_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            hide_shortcut.activated.connect(self._hide_selected)
            table.clicked.connect(self._on_clicked)
            table.customContextMenuRequested.connect(
                lambda pos, t=table: self._on_context_menu(t, pos)
            )
            table.selectionModel().selectionChanged.connect(
                lambda *_a, s=section: self._on_selection(s)
            )

        self.dip_hint = QLabel("")
        self.dip_hint.setObjectName("MutedLabel")
        self.dip_hint.setWordWrap(True)
        self.dip_hint.setVisible(False)

        self.groups_label = QLabel("")
        self.groups_label.setObjectName("MutedLabel")
        self.groups_label.setWordWrap(True)
        self.groups_label.setVisible(False)
        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(3)
        layout.addLayout(header)
        layout.addLayout(modes_row)
        layout.addLayout(banner_row)
        layout.addWidget(self.groups_label)
        layout.addWidget(self.main, 1)
        layout.addWidget(self.dip_hint)
        layout.addWidget(self.strong, 1)
        layout.addWidget(self.weak, 1)
        layout.addWidget(self.status_label)

        self.apply_scaled_metrics()
        self._sync_controls()
        self._render()
        self.deep_read_button.setChecked(bool(self._setting(MOVERS_DEEP_READ_SETTING, False)))

    # ------------------------------------------------------------ settings
    def _setting(self, key: str, default):
        if not self._persist:
            return default
        try:
            from project_paths import get_local_setting

            return get_local_setting(key, default)
        except Exception:
            return default

    def _save(self, key: str, value) -> None:
        if not self._persist:
            return
        try:
            from project_paths import save_local_setting

            save_local_setting(key, value)
        except Exception as exc:
            note_swallowed("movers board setting write failed", exc)

    def _load_hidden(self) -> tuple[str, set[str]]:
        saved = self._setting(MOVERS_HIDDEN_SETTING, {})
        if not isinstance(saved, dict):
            return "", set()
        return str(saved.get("day") or ""), {str(k) for k in saved.get("keys") or [] if k}

    def _board_day(self) -> str:
        return str(self._board.get("as_of") or "")[:10] or datetime.now().date().isoformat()

    # ------------------------------------------------------------ hide for today
    def hidden_keys(self) -> set[str]:
        return set(self._hidden) if self._hidden_day == self._board_day() else set()

    def hide_row(self, row: dict[str, Any]) -> None:
        """Hide one symbol/side from the board for today (display only)."""
        key = hidden_key(row)
        if key.startswith("|"):
            return
        day = self._board_day()
        if self._hidden_day != day:
            self._hidden_day, self._hidden = day, set()
        self._hidden.add(key)
        self._save(MOVERS_HIDDEN_SETTING, {"day": day, "keys": sorted(self._hidden)})
        self._render()

    def unhide_all(self) -> None:
        self._hidden = set()
        self._save(MOVERS_HIDDEN_SETTING, {"day": self._hidden_day, "keys": []})
        self._render()

    def _hide_selected(self) -> None:
        row = self._selected_row()
        if row:
            self.hide_row(row)

    # ------------------------------------------------------------ rows
    def visible_rows(self) -> list[dict[str, Any]]:
        """Main-table rows in the order the table shows them."""
        return self.main.visible_rows()

    def _source_row(self, index) -> dict[str, Any] | None:
        for section in self.sections:
            if section.owns(index):
                return section.source_row(index)
        return None

    def _lists_in_view(self) -> list[tuple[MoversSection, str]]:
        """(section, list) pairs the current mode shows."""
        if self._mode == "pop":
            return [(self.main, "pop"), (self.strong, "strong"), (self.weak, "weak")]
        return [(self.main, self._mode)]

    # ------------------------------------------------------------ metrics
    def apply_scaled_metrics(self) -> None:
        self.setMinimumWidth(theme.px(MIN_BOARD_WIDTH))
        self.main.set_min_rows(VISIBLE_ROWS)
        self.strong.set_min_rows(DIP_VISIBLE_ROWS)
        self.weak.set_min_rows(DIP_VISIBLE_ROWS)
        self._fit_columns()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._fit_columns()

    def _column_px(self, key: str) -> int:
        if key == "symbol":
            return theme.px(SYMBOL_COLUMN_PX)
        if key == "lvl":
            return theme.px(LVL_COLUMN_PX)
        return theme.px(COLUMN_MIN_PX)

    def visible_column_count(self, mode: str | None = None) -> int:
        """Columns that fit the board's width, in priority order (at least 2)."""
        margins = self.layout().contentsMargins() if self.layout() is not None else None
        width = self.width() - (margins.left() + margins.right() if margins else 0)
        used = count = 0
        for key, _header in COLUMNS[mode or self._mode]:
            used += self._column_px(key)
            if used > width:
                break
            count += 1
        return max(2, count)

    def _fit_columns(self) -> None:
        for section in self.sections:
            section.fit_columns(self.visible_column_count(section.columns_mode), self._column_px)
        narrow = self.width() < theme.px(NARROW_PX)
        # The +F chip is the first control to go on a narrow board (the row menu stays).
        if self.add_focus_button.isHidden() != narrow:
            self.add_focus_button.setVisible(not narrow)
        labels = MODE_SHORT if narrow else MODE_LABELS
        for mode, button in self.mode_buttons.items():
            text = labels[mode] + (" ●" if mode == "pop" and self._dip_live() else "")
            if button.text() != text:
                button.setText(text)
        long_side = self._side == "long"
        side = ("L" if long_side else "S") if narrow else ("Long" if long_side else "Short")
        hidden = len(self._hidden_in_view())
        unhide = f"↺{hidden}" if narrow else f"Unhide {hidden}"
        if self.unhide_button.text() != unhide:
            self.unhide_button.setText(unhide)
        for button, text in (
            (self.side_button, side),
            (self.review_button, "Rev ▾" if narrow else "Review ▾"),
            (self.deep_read_button, "Deep" if narrow else "Deep read"),
        ):
            if button.text() != text:
                button.setText(text)

    # ------------------------------------------------------------ focus counts
    def set_focus_service(self, service) -> None:
        self._focus_service = service
        if service is not None:
            service.focusChanged.connect(self._counts_coalescer.request)
            faded = getattr(service, "picksFaded", None)
            if faded is not None:
                faded.connect(lambda *_: self._counts_coalescer.request())
        self._render_counts()

    def request_counts_refresh(self) -> None:
        self._counts_coalescer.request()

    def _render_counts(self) -> None:
        focus_count = faded_count = 0
        if self._focus_service is not None:
            try:
                seen: set[str] = set()
                for sides in self._focus_service.all_focus_by_category().values():
                    for names in sides.values():
                        seen.update(names)
                focus_count = len(seen)
            except Exception:
                focus_count = 0
            try:
                faded_count = len(self._focus_service.faded_picks())
            except Exception:
                faded_count = 0
        self.focus_review_action.setText(f"Focus pick review ({focus_count})")
        self.focus_review_action.setEnabled(focus_count > 0)
        self.faded_review_action.setText(f"Faded review ({faded_count})")
        self.faded_review_action.setEnabled(faded_count > 0)

    # ------------------------------------------------------------ data
    def update_board(self, board: Any) -> None:
        """New board from the service. Coalesced: a burst is one render."""
        self._board = board if isinstance(board, dict) else {}
        self._maybe_follow_day()
        self._maybe_auto_switch()
        self._render_coalescer.request()

    def flush_pending_refresh(self) -> None:
        self._render_coalescer.flush()
        self._counts_coalescer.flush()

    def board(self) -> dict[str, Any]:
        return dict(self._board)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def side(self) -> str:
        return self._side

    def _state(self) -> dict[str, Any]:
        return dict(self._board.get("state") or {})

    def _dip_live(self) -> bool:
        state = self._state()
        return bool(state.get("pullback") or state.get("bounce"))

    def _day_key(self) -> str:
        state = self._state().get("state")
        if state not in ("up_day", "down_day"):
            return ""
        return f"{str(self._board.get('as_of') or '')[:10]}|{state}"

    def _maybe_follow_day(self) -> None:
        """Long on an up day, Short on a down day, once per day state; a tap holds until it changes."""
        key = self._day_key()
        if not key or key == self._day_followed:
            return
        self._day_followed = key
        self._side = "long" if key.endswith("up_day") else "short"
        self._sync_controls()

    def _maybe_auto_switch(self) -> None:
        """Jump to Pop + Dip once per pullback/bounce episode; never fight the trader."""
        state = self._state()
        if not (state.get("pullback") or state.get("bounce")):
            return
        episode = f"{state.get('start_dt') or state.get('extreme_time')}|{state.get('state')}"
        if episode == self._auto_switched_episode:
            return
        self._auto_switched_episode = episode
        self._mode = "pop"
        self._sync_controls()

    def set_mode(self, mode: str, *, user: bool = False) -> None:
        if mode not in MODES:
            return
        self._mode = mode
        if user:
            self._save(MOVERS_MODE_SETTING, mode)
        self._sync_controls()
        self._render()

    def set_side(self, side: str, *, user: bool = False) -> None:
        if side not in ("long", "short"):
            return
        self._side = side
        if user:
            self._save(MOVERS_SIDE_SETTING, side)
        self._sync_controls()
        self._render()

    def _on_side_clicked(self) -> None:
        self._day_followed = self._day_key() or self._day_followed
        self.set_side("short" if self._side == "long" else "long", user=True)

    def _on_deep_read(self, checked: bool) -> None:
        self._save(MOVERS_DEEP_READ_SETTING, bool(checked))
        self.deepReadToggled.emit(bool(checked))

    def _sync_controls(self) -> None:
        button = self.mode_buttons[self._mode]
        if not button.isChecked():
            button.setChecked(True)
        self.side_button.setToolTip("Showing longs" if self._side == "long" else "Showing shorts")
        self.side_button.setChecked(self._side == "short")
        # Pop shows both sides (and the dip tables are one per side), so the toggle is Mine's only.
        if self.side_button.isHidden() != (self._mode == "pop"):
            self.side_button.setVisible(self._mode != "pop")
        self._fit_columns()

    def _hidden_in_view(self) -> list[dict[str, Any]]:
        hidden = self.hidden_keys()
        if not hidden:
            return []
        seen: dict[str, dict[str, Any]] = {}
        for _section, name in self._lists_in_view():
            for row in rows_for(self._board, name, self._side):
                key = hidden_key(row)
                if key in hidden:
                    seen.setdefault(key, row)
        return list(seen.values())

    def _render(self) -> None:
        hidden = self.hidden_keys()
        pop_mode = self._mode == "pop"
        dip_live = pop_mode and self._dip_live()
        for section, name in self._lists_in_view():
            rows = [r for r in rows_for(self._board, name, self._side)
                    if hidden_key(r) not in hidden]
            columns_mode = "dip" if name in ("strong", "weak") else self._mode
            section.set_rows(rows, columns_mode, "short" if name == "weak" else self._side)
            section.empty_label.setText(self._empty_text(rows, name))
            section.empty_label.setVisible(not rows)
            # Height follows the row counts, so no table idles half empty.
            floor = VISIBLE_ROWS if section is self.main else DIP_VISIBLE_ROWS
            self.layout().setStretchFactor(section, max(len(rows), floor) + 2)
        for section in (self.strong, self.weak):
            if section.isHidden() != (not dip_live):
                section.setVisible(dip_live)
        state = self._state()
        pullback = bool(state.get("pullback"))
        when = _local_clock(state.get("start_dt")) or state.get("extreme_time") or ""
        turn = f"since the {when} {'high' if pullback else 'low'}" if when else "since the turn"
        word = "Dip" if pullback else "Bounce"
        self.strong.title_label.setText(f"{word}-strong ● · beating SPY {turn}")
        self.weak.title_label.setText(f"{word}-weak ● · lagging SPY {turn}")
        hint = ""
        if pop_mode and not dip_live and self._board:
            hint = ("Dip-strong / Dip-weak: no SPY pullback or bounce now. "
                    f"They light at {movers_scan.PULLBACK_MIN_PCT:.2f}% off the high or low.")
        if self.dip_hint.text() != hint:
            self.dip_hint.setText(hint)
        self.dip_hint.setVisible(bool(hint))
        hidden_count = len(self._hidden_in_view())
        if self.unhide_button.isHidden() != (hidden_count == 0):
            self.unhide_button.setVisible(hidden_count > 0)
        self._fit_columns()
        banner = banner_text(self._board.get("state") if self._board else None)
        if self._board.get("offered"):
            banner += f" · {int(self._board.get('fresh') or 0)} of {int(self._board['offered'])} fresh"
        self.banner.setText(banner)
        stamp = _local_clock(self._board.get("as_of")) or "--:--"
        if self._board and self._board.get("as_of_stale"):
            stamp += " stale"
        self.meta_label.setText(stamp)
        by_side = (self._board.get("groups") or {}).get(self._mode) or {}
        if pop_mode:
            labels = [f"{name} ×{count}{tag}" for s, tag in (("long", ""), ("short", " S"))
                      for name, count in (by_side.get(s) or [])]
        else:
            labels = [f"{name} ×{count}" for name, count in (by_side.get(self._side) or [])]
        text = "Groups: " + ", ".join(labels) if labels else ""
        if self.groups_label.text() != text:
            self.groups_label.setText(text)
        self.groups_label.setVisible(bool(text))
        self._sync_add_button()

    # ------------------------------------------------------------ +Focus
    def _on_selection(self, section: MoversSection) -> None:
        """One selection across the tables: selecting in one clears the others."""
        if section.selected_row() is not None:
            for other in self.sections:
                if other is not section and other.table.selectionModel().hasSelection():
                    other.table.clearSelection()
        self._sync_add_button()

    def _selected_row(self) -> dict[str, Any] | None:
        for section in self.sections:
            if not section.isHidden():
                row = section.selected_row()
                if row is not None:
                    return row
        return None

    def _sync_add_button(self, *_args) -> None:
        self.add_focus_button.setEnabled(self._selected_row() is not None)

    def _add_selected_to_focus(self) -> None:
        row = self._selected_row()
        if row:
            self._request_focus(row)

    def _request_focus(self, row: dict[str, Any]) -> None:
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            self.focusAddRequested.emit(symbol, row.get("_side") or self._side)

    def row_menu(self, index) -> QMenu:
        """The right-click menu for one row (built on demand)."""
        menu = QMenu(self)
        row = self._source_row(index)
        if row:
            symbol = str(row.get("symbol") or "")
            side = row.get("_side") or self._side
            action = menu.addAction(f"+F  Add {symbol} to M5 Focus ({side})")
            action.triggered.connect(lambda _checked=False, r=dict(row): self._request_focus(r))
            hide = menu.addAction(f"Hide {symbol} for today (Del)")
            hide.triggered.connect(lambda _checked=False, r=dict(row): self.hide_row(r))
        hidden = len(self._hidden_in_view())
        if hidden:
            menu.addAction(f"Unhide {hidden} hidden").triggered.connect(self.unhide_all)
        return menu

    def _on_context_menu(self, table: QTableView, pos) -> None:
        index = table.indexAt(pos)
        if not index.isValid():
            return
        self.row_menu(index).exec(table.viewport().mapToGlobal(pos))

    def show_status(self, text: str) -> None:
        """One line under the tables (e.g. the +Focus result)."""
        self.status_label.setText(str(text or ""))
        self.status_label.setVisible(bool(text))

    def _empty_text(self, rows, name: str) -> str:
        if rows:
            return ""
        if not self._board:
            return "No Movers read yet. It refreshes every 5-minute bar in market hours."
        if name == "strong":
            return "No name is beating SPY since the turn."
        if name == "weak":
            return "No name is lagging SPY since the turn."
        if name == "mine":
            return "No Focus names on this side."
        if self._hidden_in_view():
            return "Every popping name is hidden. Tap Unhide to see them."
        return "Nothing is popping."

    def _on_clicked(self, index) -> None:
        row = self._source_row(index)
        if not row:
            return
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            self.symbolActivated.emit(symbol, str(row.get("_side") or self._side).upper())


def _local_clock(value: Any) -> str:
    """HH:MM on the desk's clock for an aware ISO stamp ('' when absent)."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return ""
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    return moment.strftime("%H:%M")
