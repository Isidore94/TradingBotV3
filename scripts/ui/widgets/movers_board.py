"""The Movers board: what is popping now, and what holds up in a SPY pullback.

Sits at the top of the Alert Center's lower-right column (trader, 2026-09-23).
Three modes - Pop, Dip-strong, My names - a Long/Short toggle, a SPY state
banner and one model/view table. The "Review" menu holds the Focus pick and
Faded review doors; "Deep read" shows the old Strength page (Focus strength,
entry board, RRS snapshot, M5 Strength Board) underneath.

Display only. It owns no data, timer or fetch: `MoversService` publishes and
this widget renders, coalesced. No stylesheets are set here; colours ride the
model's roles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QAction, QColor
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

#: Same floor as the Strength page it sits above (alert column budget: 360 px).
MIN_BOARD_WIDTH = 170
#: Rows the table shows without scrolling.
VISIBLE_ROWS = 10
#: Width that one numeric column needs; narrower tables show fewer columns.
COLUMN_MIN_PX = 48
SYMBOL_COLUMN_PX = 88
LVL_COLUMN_PX = 70

MODES = ("pop", "dip", "mine")
MODE_LABELS = {"pop": "Pop", "dip": "Dip-strong", "mine": "My names"}
MODE_SHORT = {"pop": "Pop", "dip": "Dip", "mine": "Mine"}
#: Below this width the chips and header buttons use short labels.
NARROW_PX = 300
MOVERS_MODE_SETTING = "movers_board_mode"
MOVERS_SIDE_SETTING = "movers_board_side"
MOVERS_DEEP_READ_SETTING = "movers_board_deep_read"

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
    board = board or {}
    rows = list(((board.get(mode) or {}).get(side)) or [])
    if mode == "mine":
        dip_live = bool((board.get("state") or {}).get("pullback" if side == "long" else "bounce"))
        rows = movers_scan.sort_mine(rows, "dip" if dip_live else "pop", side)
    return rows


class MoversTableModel(QAbstractTableModel):
    """Rows as dicts; updates in place (diff), never a model reset."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[dict[str, Any]] = []
        self._columns = COLUMNS["pop"]
        self._side = "long"

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
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
        if role == Qt.ItemDataRole.DisplayRole:
            if key == "symbol":
                return symbol_text(row)
            if key == "lvl":
                return level_text(row, self._side)
            return format_cell(key, value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if key == "symbol":
                return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole:
            if key == "symbol":
                return QColor(theme.color("long" if self._side == "long" else "short"))
            if key in _PCT_KEYS or key == "dip_score":
                if value is None:
                    return QColor(theme.color("text_secondary"))
                return QColor(theme.color("long" if float(value) >= 0 else "short"))
            if key == "rvol" and value is None:
                return QColor(theme.color("text_secondary"))
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


class MoversBoard(QWidget):
    """Header, banner, table. The Alert Center wires its signals."""

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

        self.banner = QLabel(banner_text(None))
        self.banner.setObjectName("MutedLabel")
        self.banner.setWordWrap(True)
        banner_row = QHBoxLayout()
        banner_row.setContentsMargins(0, 0, 0, 0)
        banner_row.setSpacing(6)
        banner_row.addWidget(self.meta_label, 0, Qt.AlignmentFlag.AlignTop)
        banner_row.addWidget(self.banner, 1)

        self.model = MoversTableModel(self)
        self.table = QTableView()
        self.table.setObjectName("MoversTable")
        self.table.setModel(self.model)
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
        self.table.clicked.connect(self._on_clicked)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        self.table.selectionModel().selectionChanged.connect(self._sync_add_button)

        self.groups_label = QLabel("")
        self.groups_label.setObjectName("MutedLabel")
        self.groups_label.setWordWrap(True)
        self.groups_label.setVisible(False)
        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)

        self.empty_label = QLabel("")
        self.empty_label.setObjectName("MutedLabel")
        self.empty_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(3)
        layout.addLayout(header)
        layout.addLayout(modes_row)
        layout.addLayout(banner_row)
        layout.addWidget(self.groups_label)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.status_label)
        layout.addWidget(self.empty_label)

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
        except Exception:
            pass

    # ------------------------------------------------------------ metrics
    def apply_scaled_metrics(self) -> None:
        self.setMinimumWidth(theme.px(MIN_BOARD_WIDTH))
        row_height = self.table.fontMetrics().height() + theme.px(6)
        self.table.verticalHeader().setDefaultSectionSize(row_height)
        chrome = self.table.horizontalHeader().sizeHint().height() + 2 * self.table.frameWidth()
        self.table.setMinimumHeight(chrome + VISIBLE_ROWS * row_height)
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

    def visible_column_count(self) -> int:
        """Columns that fit the board's width, in priority order (at least 2)."""
        margins = self.layout().contentsMargins() if self.layout() is not None else None
        width = self.width() - (margins.left() + margins.right() if margins else 0)
        used = count = 0
        for key, _header in COLUMNS[self._mode]:
            used += self._column_px(key)
            if used > width:
                break
            count += 1
        return max(2, count)

    def _fit_columns(self) -> None:
        count = self.visible_column_count()
        header = self.table.horizontalHeader()
        for column, (key, _h) in enumerate(COLUMNS[self._mode]):
            hidden = column >= count
            if self.table.isColumnHidden(column) != hidden:
                self.table.setColumnHidden(column, hidden)
            if key in ("symbol", "lvl") and column < self.model.columnCount():
                if header.sectionResizeMode(column) != QHeaderView.ResizeMode.Fixed:
                    header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
                if header.sectionSize(column) != self._column_px(key):
                    header.resizeSection(column, self._column_px(key))
        narrow = self.width() < theme.px(NARROW_PX)
        # The +F chip is the first control to go on a narrow board (the row menu stays).
        if self.add_focus_button.isHidden() != narrow:
            self.add_focus_button.setVisible(not narrow)
        labels = MODE_SHORT if narrow else MODE_LABELS
        for mode, button in self.mode_buttons.items():
            text = labels[mode] + (" ●" if mode == "dip" and self._dip_live() else "")
            if button.text() != text:
                button.setText(text)
        long_side = self._side == "long"
        side = ("L" if long_side else "S") if narrow else ("Long" if long_side else "Short")
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
        """Jump to Dip-strong once per pullback/bounce episode; never fight the trader."""
        state = self._state()
        if not (state.get("pullback") or state.get("bounce")):
            return
        episode = f"{state.get('start_dt') or state.get('extreme_time')}|{state.get('state')}"
        if episode == self._auto_switched_episode:
            return
        self._auto_switched_episode = episode
        self._mode = "dip"
        self._side = "long" if state.get("pullback") else "short"
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
        self._fit_columns()

    def _render(self) -> None:
        rows = rows_for(self._board, self._mode, self._side)
        self.model.set_rows(rows, self._mode, self._side)
        self._fit_columns()
        banner = banner_text(self._board.get("state") if self._board else None)
        if self._board.get("offered"):
            banner += f" · {int(self._board.get('fresh') or 0)} of {int(self._board['offered'])} fresh"
        self.banner.setText(banner)
        stamp = _local_clock(self._board.get("as_of")) or "--:--"
        if self._board and self._board.get("as_of_stale"):
            stamp += " stale"
        self.meta_label.setText(stamp)
        self.empty_label.setText(self._empty_text(rows))
        self.empty_label.setVisible(not rows)
        groups = ((self._board.get("groups") or {}).get(self._mode) or {}).get(self._side) or []
        text = "Groups: " + ", ".join(f"{name} ×{count}" for name, count in groups) if groups else ""
        if self.groups_label.text() != text:
            self.groups_label.setText(text)
        self.groups_label.setVisible(bool(text))
        self._sync_add_button()

    # ------------------------------------------------------------ +Focus
    def _selected_row(self) -> dict[str, Any] | None:
        selection = self.table.selectionModel()
        if selection is None:
            return None
        rows = selection.selectedRows()
        return self.model.row(rows[0].row()) if rows else None

    def _sync_add_button(self, *_args) -> None:
        self.add_focus_button.setEnabled(self._selected_row() is not None)

    def _add_selected_to_focus(self) -> None:
        row = self._selected_row()
        if row:
            self._request_focus(row)

    def _request_focus(self, row: dict[str, Any]) -> None:
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            self.focusAddRequested.emit(symbol, self._side)

    def row_menu(self, index) -> QMenu:
        """The right-click menu for one row (built on demand)."""
        menu = QMenu(self)
        row = self.model.row(index.row()) if index.isValid() else None
        if row:
            symbol = str(row.get("symbol") or "")
            action = menu.addAction(f"+F  Add {symbol} to M5 Focus ({self._side})")
            action.triggered.connect(lambda _checked=False, r=dict(row): self._request_focus(r))
        return menu

    def _on_context_menu(self, pos) -> None:
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        self.row_menu(index).exec(self.table.viewport().mapToGlobal(pos))

    def show_status(self, text: str) -> None:
        """One line under the table (e.g. the +Focus result)."""
        self.status_label.setText(str(text or ""))
        self.status_label.setVisible(bool(text))

    def _empty_text(self, rows) -> str:
        if rows:
            return ""
        if not self._board:
            return "No Movers read yet. It refreshes every 5-minute bar in market hours."
        if self._mode == "dip":
            which = "pullback" if self._side == "long" else "bounce"
            if not self._state().get("pullback" if self._side == "long" else "bounce"):
                return f"No SPY {which} right now. Dip-strong lights when one starts."
            return "No name is beating SPY since the turn."
        if self._mode == "mine":
            return "No Focus names on this side."
        return "Nothing is popping on this side."

    def _on_clicked(self, index) -> None:
        row = self.model.row(index.row())
        if not row:
            return
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            self.symbolActivated.emit(symbol, self._side.upper())


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
