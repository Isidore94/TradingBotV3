"""The Movers board: what is popping now, and who is strong or weak in a SPY turn.

Sits at the top of the Alert Center's lower-right column (trader, 2026-09-23).
Two modes. Pop stacks three boxes (trader, 2026-09-24; always all three,
2026-09-28): Movers on top (longs and shorts together, half the height), then
Dip-strong and Dip-weak (a quarter each) for the live SPY turn: Dip-* in a
pullback, Bounce-* in a bounce, Rip-* in a rally; unlit, they sit empty. A SPY
state banner tops them. Sym cells carry the side colour in the mixed Pop table
(long green, short red); names new to a list get a stronger tint. A row whose D1
trend is unknown is tagged "D1?" (the ranked lists drop a known miss).
M30 and Daily (trader, 2026-09-29) show the same three boxes from the
once-a-day `MoversTimeframeService` boards; each box title's hover carries its
outcome line. PB / Line chips (and the row menu) arm D1 Pullback (fast) or
Pullback to D1 line on the selected row's side: an explicit click only. Header clicks sort (third click = board order); the trader can hide a
row for the day (right-click or Delete) and bring hidden rows back. The "Review" menu holds the Focus pick and
Faded review doors; "Deep read" shows the old Strength page (Focus strength,
entry board, RRS snapshot, M5 Strength Board) underneath.

Display only. It owns no data, timer or fetch: `MoversService` publishes and
this widget renders, coalesced. No stylesheets are set here; colours ride the
model's roles.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Signal,
)

#: The invalid (root) index used as the default parent.
_NO_PARENT = QModelIndex()
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
import movers_timeframe
import options_chase
from ui import theme
from ui.timer_utils import SignalCoalescer
from swallowed import note_swallowed

#: Same floor as the Strength page it sits above (alert column budget: 360 px).
MIN_BOARD_WIDTH = 170
#: Rows each table keeps at its smallest: the main table, and each dip table.
#: The Dip boxes are always shown, so their floor is the header alone and the
#: Movers board never pushes the chart pane above it smaller than the feed.
VISIBLE_ROWS = 6
DIP_VISIBLE_ROWS = 0
#: Width that one numeric column needs; narrower tables show fewer columns.
COLUMN_MIN_PX = 48
SYMBOL_COLUMN_PX = 88
LVL_COLUMN_PX = 70
#: The Opt (options chase) cell: elided here, the full text is in its hover.
OPT_COLUMN_PX = 96

MODES = ("pop", "m30", "d1")
MODE_LABELS = {"pop": "Pop + Dip", "m30": "M30", "d1": "Daily"}
MODE_SHORT = {"pop": "Pop", "m30": "M30", "d1": "D1"}
#: The once-a-day timeframe tabs.
TF_MODES = ("m30", "d1")
#: Alert kinds the PB / Line chips arm (the D1 menu's "Pullback (fast)" and
#: "Pullback to D1 line"); explicit clicks only (trader rule 2026-09-17).
ARM_PULLBACK = "pullback"
ARM_LINE = "d1_line_pullback"
ARM_LABELS = {ARM_PULLBACK: "D1 Pullback (fast)", ARM_LINE: "Pullback to D1 line"}
#: Below this width the chips and header buttons use short labels.
NARROW_PX = 300
#: Each box's Copy chip, and how long it shows "Copied N" after a click.
COPY_LABEL = "Copy"
COPY_FEEDBACK_MS = 1500
MOVERS_MODE_SETTING = "movers_board_mode"
MOVERS_DEEP_READ_SETTING = "movers_board_deep_read"
#: {"day": "YYYY-MM-DD", "keys": ["SYM|side", ...]}: rows the trader hid today.
MOVERS_HIDDEN_SETTING = "movers_board_hidden"
#: Raw value a column sorts by (None = unmeasured, always last).
SORT_ROLE = int(Qt.ItemDataRole.UserRole) + 1
_TEXT_SORT_KEYS = {"symbol", "group"}

#: (key, header) per mode. Priority order: Sym, main score, RVOL, Lvl, then the
#: rest (Pop: the options chase Opt cell next); narrow widths drop columns from the end.
COLUMNS = {
    "pop": (("symbol", "Sym"), ("move15_pct", "15m"), ("rvol", "RVOL"), ("lvl", "Lvl"),
            ("opt", "Opt"), ("vs_spy15_pct", "vSPY"), ("move30_pct", "30m"),
            ("day_pct", "Day"), ("group", "Grp")),
    "dip": (("symbol", "Sym"), ("dip_score", "xSPY"), ("rvol", "RVOL"), ("lvl", "Lvl"),
            ("since_start_pct", "Since"), ("day_pct", "Day"), ("move15_pct", "15m"),
            ("group", "Grp")),
    # M30 / Daily: moves over 3 and 6 bars of that timeframe; no Lvl or Opt.
    "m30_pop": (("symbol", "Sym"), ("move15_pct", "90m"), ("rvol", "RVOL"),
                ("vs_spy15_pct", "vSPY"), ("move30_pct", "3h"), ("day_pct", "Day"),
                ("group", "Grp")),
    "m30_dip": (("symbol", "Sym"), ("dip_score", "xSPY"), ("rvol", "RVOL"),
                ("since_start_pct", "Since"), ("day_pct", "Day"), ("move15_pct", "90m"),
                ("group", "Grp")),
    "d1_pop": (("symbol", "Sym"), ("move15_pct", "3d"), ("rvol", "RVOL"),
               ("vs_spy15_pct", "vSPY"), ("move30_pct", "6d"), ("day_pct", "1d"),
               ("group", "Grp")),
    "d1_dip": (("symbol", "Sym"), ("dip_score", "xSPY"), ("rvol", "RVOL"),
               ("since_start_pct", "Since"), ("day_pct", "1d"), ("move15_pct", "3d"),
               ("group", "Grp")),
}
_PCT_KEYS = {"move15_pct", "move30_pct", "day_pct", "vs_spy15_pct", "since_start_pct"}


def symbol_text(row: dict[str, Any]) -> str:
    """Symbol, an ER tag, and the rank change since the last tick (lists only)."""
    parts = [str(row.get("symbol") or "")]
    if row.get("er"):
        parts.append("ER")
    if row.get("earnings_badge"):
        parts.append(f"⚠{row['earnings_badge']}")
    if "streak" in row:
        change = row.get("rank_change")
        if change is None:
            parts.append("new")
        elif change > 0:
            parts.append(f"▲{change}")
        elif change < 0:
            parts.append(f"▼{-change}")
    trend = trend_flag(row)
    if trend is None and "trend_long" in row and row.get("last") is not None:
        parts.append("D1?")
    elif trend is False:
        parts.append("D1✗")
    return " ".join(parts)


def _swing_title(side: str, anchor: dict[str, Any] | None) -> str:
    """A Dip box title naming its own SPY anchor (local clock), or that none exists yet."""
    name = "Dip-strong" if side == "long" else "Dip-weak"
    if not anchor:
        return f"{name} · no SPY bars today yet"
    when = _local_clock(anchor.get("dt")) or anchor.get("time") or ""
    verb = "beating" if side == "long" else "lagging"
    point = "high" if side == "long" else "low"
    kind = anchor.get("kind")
    if kind == "open":
        since = "the open"
    elif kind in ("lod", "hod"):
        since = f"the {when} {point} of day so far"
    else:
        since = f"SPY's {point} {when}"
    held = " (held)" if anchor.get("held_from_previous") else ""
    return f"{name} · {verb} SPY since {since}{held}"


def _short_date(value: Any) -> str:
    """'9/22' from an ISO date or stamp ('' when absent)."""
    text = str(value or "")[:10]
    try:
        day = datetime.fromisoformat(text).date()
    except ValueError:
        return ""
    return f"{day.month}/{day.day}"


def _tf_measured_at(board: dict[str, Any]) -> str:
    """When an M30 board was measured on the desk clock (its last bar's end)."""
    try:
        start = datetime.fromisoformat(str(board.get("as_of") or ""))
    except ValueError:
        return ""
    return _local_clock((start + timedelta(minutes=movers_timeframe.M30_MINUTES)).isoformat())


def tf_main_title(tf: str, board: dict[str, Any] | None) -> str:
    """'M30 Movers · 3-bar moves at 09:00' / 'Daily Movers · 3-day moves to the 9/22 close'."""
    board = board or {}
    label = movers_timeframe.TF_LABELS.get(tf, tf)
    if not board:
        return f"{label} Movers · not scanned yet"
    if tf == "d1":
        return f"{label} Movers · 3-day moves to the {_short_date(board.get('session'))} close"
    return f"{label} Movers · 3-bar moves at {_tf_measured_at(board)}"


def tf_swing_title(tf: str, side: str, anchor: dict[str, Any] | None) -> str:
    """'Daily Dip-strong · beating SPY since 9/22 low' (the anchor's own date / time)."""
    label = movers_timeframe.TF_LABELS.get(tf, tf)
    name = "Dip-strong" if side == "long" else "Dip-weak"
    if not anchor:
        return f"{label} {name} · no SPY anchor yet"
    verb = "beating" if side == "long" else "lagging"
    # M30: longs from SPY's high before its last big dip, shorts from the low (trader 2026-09-30).
    point = ("high" if side == "long" else "low") if tf == "m30" else (
        "low" if side == "long" else "high")
    when = _short_date(anchor.get("date") or anchor.get("dt"))
    if tf == "m30":
        when = f"{when} {_local_clock(anchor.get('dt')) or anchor.get('time') or ''}".strip()
    return f"{label} {name} · {verb} SPY since {when} {point}"


def outcome_tip(board: dict[str, Any] | None, box: str) -> str:
    """The box title's hover: its outcome line(s) from the board's summaries."""
    import movers_timeframe_outcomes as outcomes

    summaries = (board or {}).get("summaries") or {}
    if box == "pop":
        sides = summaries.get("pop") or {}
        return (f"Longs: {outcomes.summary_line(sides.get('long'))}\n"
                f"Shorts: {outcomes.summary_line(sides.get('short'))}")
    side = "long" if box == "dip_strong" else "short"
    return outcomes.summary_line((summaries.get(box) or {}).get(side))


def tf_banner_text(tf: str, board: dict[str, Any] | None) -> str:
    """'M30 scanned 9/22 09:05 · SPY +0.42% on the day · 812 of 1400 measured', or 'not scanned yet'."""
    board = board or {}
    label = movers_timeframe.TF_LABELS.get(tf, tf)
    when = "runs at 09:00 on trading days" if tf == "m30" else "runs after the close"
    if not board:
        return f"{label}: not scanned yet ({when})"
    stamp = f"{_short_date(board.get('scanned_at'))} {_local_clock(board.get('scanned_at'))}"
    parts = [f"{label} scanned {stamp.strip()}"]
    if board.get("stale"):
        parts.append("stale")
    day = (board.get("state") or {}).get("spy_day_pct")
    if day is not None:
        parts.append(f"SPY {float(day):+.2f}% " + ("on the day" if tf == "m30" else "last session"))
    if board.get("offered"):
        parts.append(f"{int(board.get('measured') or 0)} of {int(board['offered'])} measured")
    if board.get("last_error"):
        parts.append(f"last scan FAILED: {board['last_error']}")
    return " · ".join(parts)


def trend_flag(row: dict[str, Any]) -> bool | None:
    """The D1 trend verdict for the row's side; None when unknown or untagged."""
    return row.get("trend_short") if row.get("_side") == "short" else row.get("trend_long")


def _tag_short_earnings(rows: list[dict[str, Any]]) -> None:
    """S10b: tag each weak (short) row 0-14 days before earnings. Memory only; display only."""
    try:
        import earnings_warning

        for row in rows:
            days = earnings_warning.cached_days_to_next_earnings(row.get("symbol"))
            text = earnings_warning.warning_for_symbol(row.get("symbol"), "SHORT")
            if text:
                row["earnings_warning"] = text
                row["earnings_badge"] = f"{days}d"
    except Exception as exc:  # noqa: BLE001 - a warning never costs the board
        note_swallowed("movers earnings warning not tagged", exc, quiet=True)


def is_new(row: dict[str, Any]) -> bool:
    """First tick on this list (rows without a streak are never new)."""
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
    label = {"up_day": "up day", "down_day": "down day", "flat": "flat"}.get(kind, kind)
    if state.get("rally") and off is not None:
        return f"SPY {off:+.2f}% from {when} low · {label} · RALLY"
    day = state.get("spy_day_pct")
    day_text = f" · SPY {day:+.2f}% on the day" if day is not None else ""
    return f"{label} · no pullback{day_text}" if kind == "up_day" else (
        f"{label} · no bounce{day_text}" if kind == "down_day" else f"{label}{day_text}"
    )


def rows_for(board: dict[str, Any] | None, mode: str, side: str) -> list[dict[str, Any]]:
    """Rows for one list, each tagged `_side`. Pop shows both sides, biggest move first;
    "strong"/"weak" are the live turn's lists (beating / lagging SPY since the turn):
    the rip lists in a rally, else the dip lists."""
    board = board or {}
    if mode in ("strong", "weak"):
        side = "long" if mode == "strong" else "short"
        # The swing-anchored lists when the board has them, else the turn's lists.
        key = "swing" if "swing" in board else (
            "rip" if (board.get("state") or {}).get("rally") else "dip")
        return [dict(row, _side=side) for row in (((board.get(key) or {}).get(side)) or [])]
    if mode == "pop":
        both = [dict(row, _side=s) for s in ("long", "short")
                for row in (((board.get(mode) or {}).get(s)) or [])]
        return sorted(both, key=lambda r: -abs(float(r.get("pop_score") or 0.0)))
    rows = list(((board.get(mode) or {}).get(side)) or [])
    return [dict(row, _side=side) for row in rows]


def hidden_key(row: dict[str, Any]) -> str:
    return f"{str(row.get('symbol') or '').strip().upper()}|{row.get('_side') or 'long'}"


def sort_value(row: dict[str, Any], key: str) -> Any:
    """What a column sorts by; None sorts last either way."""
    if key in _TEXT_SORT_KEYS:
        return str(row.get(key) or "").upper() or None
    if key == "opt":
        # Candidates first (tightest spread first), then refusals and no data, then unchecked.
        result = row.get("opt") or {}
        if result.get("status") == options_chase.STATUS_CANDIDATE:
            return 100.0 - float(result.get("spread_pct") or 0.0)
        return {options_chase.STATUS_REFUSED: -1.0, options_chase.STATUS_NO_DATA: -2.0}.get(
            result.get("status"))
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
        self._mixed = False  # the Pop table holds both sides

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
        self._mixed = mode.endswith("pop")  # Pop, M30 and Daily Movers hold both sides
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
            if key == "opt":
                return options_chase.cell_text(value)
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
            if key == "opt" and (value or {}).get("status") != options_chase.STATUS_CANDIDATE:
                return QColor(theme.color("text_secondary"))
        if role == Qt.ItemDataRole.BackgroundRole and key == "symbol":
            # The side colour, so a mixed table reads at a glance; stronger when new.
            new = is_new(row)
            if new or self._mixed:
                color = QColor(theme.color("long" if side == "long" else "short"))
                color.setAlphaF(0.30 if new else 0.10)
                return color
        if role == Qt.ItemDataRole.BackgroundRole and key == "rvol" and value is not None:
            # Busier than usual reads warmer: alpha grows from 1x to 3x.
            strength = max(0.0, min(1.0, (float(value) - 1.0) / 2.0))
            if strength > 0:
                color = QColor(theme.color("caution"))
                color.setAlphaF(0.12 + 0.45 * strength)
                return color
        if role == Qt.ItemDataRole.ToolTipRole:
            if key == "opt":
                return options_chase.detail_text(value)
            return _row_tooltip(row)
        return None


def _row_tooltip(row: dict[str, Any]) -> str:
    parts = [str(row.get("symbol") or "")]
    short, long = movers_timeframe.MOVE_LABELS.get(row.get("_tf") or "", ("15m", "30m"))
    for key, label in (("move15_pct", f"{short} %"), ("move30_pct", f"{long} %"),
                       ("day_pct", "day %"), ("vs_spy15_pct", f"vs SPY {short}"),
                       ("since_start_pct", "since start %")):
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
    if row.get("earnings_warning"):
        parts.append(str(row["earnings_warning"]))
    if "quality_ok" in row:
        cap, volume = row.get("market_cap_m"), row.get("avg_volume_20d")
        cap_text = "?" if cap is None else f"${float(cap) / 1000:.1f}B"
        volume_text = "?" if volume is None else f"{float(volume) / 1e6:.1f}M"
        parts.append(f"cap {cap_text} · 20d vol {volume_text}")
    if "trend_long" in row:
        trend = trend_flag(row)
        want = ("below D1 50/100 SMA" if row.get("_side") == "short"
                else "above D1 100/200 SMA")
        verdict = ("yes" if trend else "NO" if trend is False
                   else f"unknown ({int(row.get('daily_bars') or 0)} daily bars)")
        parts.append(f"{want}: {verdict}")
    if row.get("avwap") is not None:
        parts.append(f"VWAP from SPY's anchor {float(row['avwap']):.2f}")
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
        self.copy_button = QToolButton()
        self.copy_button.setObjectName("MoversChip")
        self.copy_button.setText(COPY_LABEL)
        self.copy_button.setToolTip(
            "Copy this box's symbols, in the order shown, as one comma list "
            "(pastes into a TC2000 or TradingView watchlist)."
        )
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_symbols)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(4)
        layout.addLayout(title_row)
        title_row.addWidget(self.title_label, 1)
        title_row.addWidget(self.copy_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.empty_label)

    def set_rows(self, rows: list[dict[str, Any]], columns_mode: str, side: str) -> None:
        self.columns_mode = columns_mode
        self.model.set_rows(rows, columns_mode, side)
        self.apply_sort()
        has_rows = bool(self.copy_symbols_text())
        if self.copy_button.isEnabled() != has_rows:
            self.copy_button.setEnabled(has_rows)

    def copy_symbols_text(self) -> str:
        """The shown symbols in view order: upper case, no blanks, no repeats, comma-joined."""
        seen: dict[str, None] = {}
        for row in self.visible_rows():
            symbol = str(row.get("symbol") or "").strip().upper()
            if symbol:
                seen.setdefault(symbol, None)
        return ",".join(seen)

    def copy_symbols(self) -> str:
        """Put this box's symbols on the clipboard; an empty box copies nothing."""
        text = self.copy_symbols_text()
        if not text:
            return ""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self.copy_button.setText(f"Copied {text.count(',') + 1}")
        QTimer.singleShot(COPY_FEEDBACK_MS, self._reset_copy_label)
        return text

    def _reset_copy_label(self) -> None:
        self.copy_button.setText(COPY_LABEL)

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
            if key in ("symbol", "lvl", "opt") and column < self.model.columnCount():
                if header.sectionResizeMode(column) != QHeaderView.ResizeMode.Fixed:
                    header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
                if header.sectionSize(column) != column_px(key):
                    header.resizeSection(column, column_px(key))


class MoversBoard(QWidget):
    """Header, banner, and the Pop / strong / weak tables. The Alert Center wires its signals."""

    symbolActivated = Signal(str, str)
    reviewAllRequested = Signal()
    fadedReviewRequested = Signal()
    deepReadToggled = Signal(bool)
    #: The trader's explicit "+F" click: (symbol, "long"|"short"). Never automatic.
    focusAddRequested = Signal(str, str)
    #: The trader's explicit PB / Line click: (symbol, "long"|"short", kind). Never automatic.
    alertArmRequested = Signal(str, str, str)

    def __init__(self, parent=None, *, persist: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("MoversBoard")
        try:
            import earnings_warning

            # S10b: load earnings dates on its background thread before the first tick.
            earnings_warning.request_warm()
        except Exception as exc:  # noqa: BLE001
            note_swallowed("earnings warning warm not started", exc, quiet=True)
        self._persist = persist
        self._board: dict[str, Any] = {}
        # The once-a-day M30 / Daily boards by timeframe.
        self._tf_boards: dict[str, dict[str, Any]] = {}
        # symbol -> armed alert kinds (the Alert Center's in-memory lists).
        self._armed_provider = None
        self._mode = self._setting(MOVERS_MODE_SETTING, "pop")
        if self._mode not in MODES:
            self._mode = "pop"
        # Every list carries its own side; this is only the fallback for an untagged row.
        self._side = "long"
        self._auto_switched_episode = ""
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
        self.add_focus_button = QToolButton()
        self.add_focus_button.setObjectName("MoversChip")
        self.add_focus_button.setText("+F")
        self.add_focus_button.setToolTip(
            "Add the selected row to M5 Focus on this side (through the adoption gate)."
        )
        self.add_focus_button.setEnabled(False)
        self.add_focus_button.clicked.connect(self._add_selected_to_focus)
        modes_row.addWidget(self.add_focus_button)
        self.arm_buttons: dict[str, QToolButton] = {}
        for kind, text in ((ARM_PULLBACK, "PB"), (ARM_LINE, "Line")):
            button = QToolButton()
            button.setObjectName("MoversChip")
            button.setText(text)
            button.setToolTip(f"Arm {ARM_LABELS[kind]} on the selected row's side "
                              "(click again to disarm).")
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, k=kind: self._arm_selected(k))
            self.arm_buttons[kind] = button
            modes_row.addWidget(button)
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

        # Main table (Movers), then the two dip tables under it (every mode).
        self.main = MoversSection(self, titled=True)
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
        """(section, list) pairs the current mode shows: always the three boxes."""
        return [(self.main, "pop"), (self.strong, "strong"), (self.weak, "weak")]

    def _view_board(self) -> dict[str, Any]:
        """The M5 board in Pop + Dip, else that timeframe's board."""
        return self._board if self._mode == "pop" else (self._tf_boards.get(self._mode) or {})

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
        if key == "opt":
            return theme.px(OPT_COLUMN_PX)
        return theme.px(COLUMN_MIN_PX)

    def visible_column_count(self, mode: str | None = None) -> int:
        """Columns that fit the board's width, in priority order (at least 2)."""
        margins = self.layout().contentsMargins() if self.layout() is not None else None
        width = self.width() - (margins.left() + margins.right() if margins else 0)
        used = count = 0
        for key, _header in COLUMNS.get(mode or self._mode, COLUMNS["pop"]):
            used += self._column_px(key)
            if used > width:
                break
            count += 1
        return max(2, count)

    def _fit_columns(self) -> None:
        for section in self.sections:
            section.fit_columns(self.visible_column_count(section.columns_mode), self._column_px)
        narrow = self.width() < theme.px(NARROW_PX)
        # The +F and arm chips are the first controls to go on a narrow board (the row menu stays).
        for chip in (self.add_focus_button, *self.arm_buttons.values()):
            if chip.isHidden() != narrow:
                chip.setVisible(not narrow)
        labels = MODE_SHORT if narrow else MODE_LABELS
        for mode, button in self.mode_buttons.items():
            text = labels[mode] + (" ●" if mode == "pop" and self._dip_live() else "")
            if button.text() != text:
                button.setText(text)
        hidden = len(self._hidden_in_view())
        unhide = f"↺{hidden}" if narrow else f"Unhide {hidden}"
        if self.unhide_button.text() != unhide:
            self.unhide_button.setText(unhide)
        for button, text in (
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
        self._maybe_auto_switch()
        self._render_coalescer.request()

    def update_timeframe_board(self, tf: str, board: Any) -> None:
        """A new M30 / Daily board from `MoversTimeframeService`. Coalesced."""
        if tf not in TF_MODES:
            return
        self._tf_boards[tf] = board if isinstance(board, dict) else {}
        self._render_coalescer.request()

    def timeframe_board(self, tf: str) -> dict[str, Any]:
        return dict(self._tf_boards.get(tf) or {})

    def flush_pending_refresh(self) -> None:
        self._render_coalescer.flush()
        self._counts_coalescer.flush()

    def board(self) -> dict[str, Any]:
        return dict(self._board)

    @property
    def mode(self) -> str:
        return self._mode

    def _state(self) -> dict[str, Any]:
        return dict(self._board.get("state") or {})

    def _dip_live(self) -> bool:
        """A SPY turn is live (pullback, bounce or rally): the strong/weak tables show."""
        state = self._state()
        return bool(state.get("pullback") or state.get("bounce") or state.get("rally"))

    def _maybe_auto_switch(self) -> None:
        """Jump to Pop + Dip once per pullback/bounce episode (not a rally); never fight the trader."""
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

    def _on_deep_read(self, checked: bool) -> None:
        self._save(MOVERS_DEEP_READ_SETTING, bool(checked))
        self.deepReadToggled.emit(bool(checked))

    def _sync_controls(self) -> None:
        button = self.mode_buttons[self._mode]
        if not button.isChecked():
            button.setChecked(True)
        self._fit_columns()

    def _hidden_in_view(self) -> list[dict[str, Any]]:
        hidden = self.hidden_keys()
        if not hidden:
            return []
        seen: dict[str, dict[str, Any]] = {}
        for _section, name in self._lists_in_view():
            for row in rows_for(self._view_board(), name, self._side):
                key = hidden_key(row)
                if key in hidden:
                    seen.setdefault(key, row)
        return list(seen.values())

    def _render(self) -> None:
        hidden = self.hidden_keys()
        pop_mode = self._mode == "pop"
        board = self._view_board()
        # Every mode shows three boxes: Movers, Dip-strong, Dip-weak. In Pop + Dip
        # the Dip tables fill only while SPY is in a pullback, bounce or rally.
        for section in (self.strong, self.weak):
            if section.isHidden():
                section.setVisible(True)
        main_title = ("Movers · biggest 15-minute moves now" if pop_mode
                      else tf_main_title(self._mode, board))
        if self.main.title_label.text() != main_title:
            self.main.title_label.setText(main_title)
        if self.main.title_label.isHidden():
            self.main.title_label.setVisible(True)
        main_tip = "" if pop_mode else outcome_tip(board, "pop")
        if self.main.title_label.toolTip() != main_tip:
            self.main.title_label.setToolTip(main_tip)
        prefix = "" if pop_mode else f"{self._mode}_"
        for section, name in self._lists_in_view():
            rows = [r for r in rows_for(board, name, self._side)
                    if hidden_key(r) not in hidden]
            if not pop_mode:
                rows = [dict(r, _tf=self._mode) for r in rows]
            if name == "weak":
                _tag_short_earnings(rows)
            columns_mode = prefix + ("dip" if name in ("strong", "weak") else "pop")
            section.set_rows(rows, columns_mode, "short" if name == "weak" else self._side)
            section.empty_label.setText(self._empty_text(rows, name))
            section.empty_label.setVisible(not rows and bool(section.empty_label.text()))
            # Fixed shares: Movers half the height, each Dip box a quarter.
            self.layout().setStretchFactor(section, 2 if section is self.main else 1)
        if pop_mode:
            self._render_pop_titles()
        else:
            self._render_tf_titles(board)
        hidden_count = len(self._hidden_in_view())
        if self.unhide_button.isHidden() != (hidden_count == 0):
            self.unhide_button.setVisible(hidden_count > 0)
        self._fit_columns()
        if pop_mode:
            banner = banner_text(self._board.get("state") if self._board else None)
            if self._board.get("offered"):
                banner += (f" · {int(self._board.get('fresh') or 0)} of "
                           f"{int(self._board['offered'])} fresh")
            stamp = _local_clock(self._board.get("as_of")) or "--:--"
            if self._board and self._board.get("as_of_stale"):
                stamp += " stale"
        else:
            banner = tf_banner_text(self._mode, board)
            stamp = _short_date(board.get("session")) or "--"
            if board.get("stale"):
                stamp += " stale"
        self.banner.setText(banner)
        self.meta_label.setText(stamp)
        by_side = (board.get("groups") or {}).get("pop") or {}
        labels = [f"{name} ×{count}{tag}" for s, tag in (("long", ""), ("short", " S"))
                  for name, count in (by_side.get(s) or [])]
        text = "Groups: " + ", ".join(labels) if labels else ""
        if self.groups_label.text() != text:
            self.groups_label.setText(text)
        self.groups_label.setVisible(bool(text))
        self._sync_add_button()

    def _render_tf_titles(self, board: dict[str, Any]) -> None:
        """M30 / Daily Dip-box titles name their SPY anchor; the hover is the outcome line."""
        anchors = board.get("swing_anchor") or {}
        for section, side, box in ((self.strong, "long", "dip_strong"),
                                   (self.weak, "short", "dip_weak")):
            title = tf_swing_title(self._mode, side, anchors.get(side))
            if section.title_label.text() != title:
                section.title_label.setText(title)
            tip = outcome_tip(board, box)
            if section.title_label.toolTip() != tip:
                section.title_label.setToolTip(tip)

    def _render_pop_titles(self) -> None:
        """Pop + Dip's Dip-box titles and hover (the M5 SPY turn or swing anchor)."""
        dip_live = self._dip_live()
        state = self._state()
        pullback = bool(state.get("pullback"))
        when = _local_clock(state.get("start_dt")) or state.get("extreme_time") or ""
        turn = f"since the {when} {'high' if pullback else 'low'}" if when else "since the turn"
        word = "Rip" if state.get("rally") else "Dip" if pullback else "Bounce"
        swing_anchor = (self._board or {}).get("swing_anchor") if "swing" in (self._board or {}) else None
        if swing_anchor is not None:
            self.strong.title_label.setText(_swing_title("long", swing_anchor.get("long")))
            self.weak.title_label.setText(_swing_title("short", swing_anchor.get("short")))
            tip = ("Dip-strong measures from SPY's low since its last big M5 drop; "
                   "Dip-weak from its high since its last big M5 bounce. Big = "
                   f"{movers_scan.SWING_HA_RUN}+ Heikin-Ashi candles in a row; with none yet, "
                   f"the low or high of day. A point under {movers_scan.SWING_MIN_AGE_MIN} "
                   "minutes old keeps the last one (held), else the low/high of day, "
                   "else the open. Dip-strong lists only names above VWAP, yesterday's "
                   "high and the daily 100 and 200 SMA; Dip-weak only names below "
                   "yesterday's low, VWAP and the daily 50 SMA. A name listed earlier "
                   "today stays while it still qualifies.")
        elif dip_live:
            self.strong.title_label.setText(f"{word}-strong ● · beating SPY {turn}")
            self.weak.title_label.setText(f"{word}-weak ● · lagging SPY {turn}")
            tip = ""
        else:
            # Short titles (a long unwrapped label would widen the column); the why is the tip.
            self.strong.title_label.setText("Dip-strong · not lit")
            self.weak.title_label.setText("Dip-weak · not lit")
            tip = ("No SPY pullback, bounce or rally now. The Dip boxes fill at "
                   f"{movers_scan.PULLBACK_MIN_PCT:.2f}% off the high or low.")
        for section in (self.strong, self.weak):
            if section.title_label.toolTip() != tip:
                section.title_label.setToolTip(tip)

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
        row = self._selected_row()
        self.add_focus_button.setEnabled(row is not None)
        armed = self._armed_kinds(row) if row else set()
        for kind, button in self.arm_buttons.items():
            if button.isEnabled() != (row is not None):
                button.setEnabled(row is not None)
            text = ("PB" if kind == ARM_PULLBACK else "Line") + (" ✓" if kind in armed else "")
            if button.text() != text:
                button.setText(text)

    # ------------------------------------------------------------ alert arming
    def set_armed_kinds_provider(self, provider) -> None:
        """`provider(symbol) -> set of armed kinds` (an in-memory read, Qt thread)."""
        self._armed_provider = provider
        self._sync_add_button()

    def refresh_arm_state(self, *_args) -> None:
        self._sync_add_button()

    def _armed_kinds(self, row: dict[str, Any] | None) -> set[str]:
        symbol = str((row or {}).get("symbol") or "").strip().upper()
        if not symbol or self._armed_provider is None:
            return set()
        try:
            return set(self._armed_provider(symbol) or ())
        except Exception as exc:  # noqa: BLE001 - a failed read shows "not armed"
            note_swallowed("movers armed-kinds read failed", exc, quiet=True)
            return set()

    def _arm_selected(self, kind: str) -> None:
        row = self._selected_row()
        if row:
            self._request_arm(row, kind)

    def _request_arm(self, row: dict[str, Any], kind: str) -> None:
        """The trader's explicit arm / disarm click for one row, on that row's side."""
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol and kind in ARM_LABELS:
            side = "short" if row.get("_side") == "short" else "long"
            self.alertArmRequested.emit(symbol, side, kind)

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
            armed = self._armed_kinds(row)
            for kind in (ARM_PULLBACK, ARM_LINE):
                label = ARM_LABELS[kind]
                text = (f"✓ Disarm {label} {symbol}" if kind in armed
                        else f"Arm {label} {symbol} ({side})")
                arm = menu.addAction(text)
                arm.triggered.connect(
                    lambda _checked=False, r=dict(row), k=kind: self._request_arm(r, k))
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
        if self._mode in TF_MODES:
            board = self._view_board()
            if not board:
                return "" if name != "pop" else tf_banner_text(self._mode, board)
            m30 = self._mode == "m30"
            if name == "strong":
                return f"No name is beating SPY since the {'high' if m30 else 'low'}."
            if name == "weak":
                return f"No name is lagging SPY since the {'low' if m30 else 'high'}."
            if self._hidden_in_view():
                return "Every name is hidden. Tap Unhide to see them."
            return "Nothing moved enough."
        if not self._board:
            return "No Movers read yet. It refreshes every 5-minute bar in market hours."
        if "swing" in (self._board or {}):
            if name == "strong":
                return "" if (self._board.get("swing_anchor") or {}).get("long") is None else (
                    "No name is beating SPY since the high.")
            if name == "weak":
                return "" if (self._board.get("swing_anchor") or {}).get("short") is None else (
                    "No name is lagging SPY since the low.")
        if name in ("strong", "weak") and not self._dip_live():
            return ""  # the box title says it is not lit
        if name == "strong":
            return "No name is beating SPY since the turn."
        if name == "weak":
            return "No name is lagging SPY since the turn."
        if self._hidden_in_view():
            return "Every popping name is hidden. Tap Unhide to see them."
        return "Nothing is popping."

    def copy_opt_text(self, row: dict[str, Any]) -> str:
        """The Opt cell's only action: its text goes to the clipboard."""
        text = options_chase.cell_text(row.get("opt"))
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self.show_status(f"Copied: {row.get('symbol') or ''} {text}".strip())
        return text

    def _column_key(self, index) -> str:
        for section in self.sections:
            if section.owns(index):
                columns = section.columns()
                return columns[index.column()][0] if 0 <= index.column() < len(columns) else ""
        return ""

    def _on_clicked(self, index) -> None:
        row = self._source_row(index)
        if not row:
            return
        if self._column_key(index) == "opt":
            self.copy_opt_text(row)
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
