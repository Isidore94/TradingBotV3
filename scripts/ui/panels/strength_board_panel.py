"""The M5 strength board (plan.md Phase 0.5, packet R2 Part B.3.4).

Side-split rows from `StrengthBoardService`, with one-click **Add to M5 Focus**
per row and a side-aware **Add all shown**. Every add passes through packet R2
Part A's adoption gate, and a row that fails it at click time is refused with
the reason shown rather than silently dropped - the trader asked why a name is
not there often enough that "nothing happened" is not an acceptable answer.

Every column sorts on click (trader request 2026-08-19) and selecting a row
opens that symbol in the desk's existing snapshot popup, so the board can be
read as charts rather than as a list of tickers.

Decision support only: no alerts, no watchlist writes beyond the Focus adds the
trader explicitly clicks, and no influence on any champion path. Sorting is
presentation and nothing else - it re-orders rows the service already fetched
and can never trigger a refetch, so a click here costs no network call and no
IB traffic.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import focus_adoption_gate
from ui.widgets.section_header import SectionHeader

_COLUMNS = ("Symbol", "Strength", "Day %", "vs VWAP", "Last")
#: Which row field each column shows, so a header click sorts on the NUMBER
#: rather than on the formatted text. "+1.20%" and "-11.00%" sort correctly as
#: floats and backwards as strings, and "—" is not a small number at all.
_COLUMN_KEYS = ("symbol", "strength", "day_pct", "vwap_distance_pct", "last")


def _sort_value(row: dict, key: str):
    """(missing?, value) - so an unmeasured field sorts LAST in both directions.

    A blank cell is not a zero and not the smallest number; it is an absence.
    Ranking it against real readings in either direction would put a name the
    board could not measure at the top of a list the trader reads as a ranking.
    """
    value = row.get(key)
    if key == "symbol":
        text = str(value or "").strip().upper()
        return (not text, text)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return (True, 0.0)
    if number != number:  # NaN
        return (True, 0.0)
    return (False, number)


def sort_rows(rows: list[dict], column: int, descending: bool) -> list[dict]:
    """Rows ordered by one column. Pure, so the ordering is testable alone.

    Missing values stay last whichever way the arrow points, which is why this
    cannot be expressed as a plain `reverse=True`.
    """
    if not (0 <= column < len(_COLUMN_KEYS)):
        return list(rows)
    key = _COLUMN_KEYS[column]
    present = [row for row in rows if not _sort_value(row, key)[0]]
    missing = [row for row in rows if _sort_value(row, key)[0]]
    present.sort(key=lambda row: _sort_value(row, key)[1], reverse=bool(descending))
    return present + missing


class _SideTable(QWidget):
    """One side's rows plus its add buttons."""

    addRequested = Signal(str, str)      # symbol, side
    addAllRequested = Signal(str)        # side
    symbolActivated = Signal(str)        # symbol (open the chart popup)

    def __init__(self, side: str, parent=None) -> None:
        super().__init__(parent)
        self._side = side
        self._rows: list[dict] = []
        # The board arrives already ranked "strongest for this side first"
        # (`strength_scan.top_fraction`: descending for longs, ascending for
        # shorts). The indicator states that rather than inventing a default -
        # a board whose order is unexplained is a board the trader has to
        # re-derive every morning.
        self._sort_column = 1
        self._sort_descending = side != "short"
        self._selected_symbol = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        self._title = QLabel(f"{side.title()}s (0)")
        self._title.setStyleSheet("font-weight: 600;")
        header.addWidget(self._title)
        header.addStretch(1)
        self._add_all = QPushButton("Add all shown")
        self._add_all.setToolTip(
            "Add every row shown here to M5 Focus. Each one is re-checked "
            "against the adoption gate at click time, so a name that has "
            "fallen back through VWAP or yesterday's level is refused."
        )
        self._add_all.clicked.connect(lambda: self.addAllRequested.emit(self._side))
        header.addWidget(self._add_all)
        layout.addLayout(header)

        self.table = QTableWidget(0, len(_COLUMNS) + 1)
        self.table.setHorizontalHeaderLabels([*_COLUMNS, ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        # Selecting a row IS the request to see it (trader 2026-08-19: "I need
        # to see my charts in there to make decisions"). Double-click still
        # works for the trader whose hands already know it.
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        # Qt's own sorting is deliberately NOT enabled: the last column holds a
        # cell WIDGET per row, and QTableWidget moves items when it sorts while
        # leaving cell widgets where they are - the button would end up on
        # another symbol's row. Owning the order here also keeps blanks last in
        # both directions, which `setSortingEnabled` cannot express.
        #
        # It is set BEFORE the indicator: `setSortingEnabled(False)` clears the
        # indicator, so asking for the arrow first would silently lose it.
        self.table.setSortingEnabled(False)
        header = self.table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicator(self._sort_column, self._indicator_order())
        header.setSortIndicatorShown(True)
        header.sectionClicked.connect(self._on_header_clicked)
        layout.addWidget(self.table)

    def _indicator_order(self):
        return Qt.DescendingOrder if self._sort_descending else Qt.AscendingOrder

    def _on_header_clicked(self, column: int) -> None:
        if not (0 <= column < len(_COLUMNS)):
            return  # the button column sorts nothing
        if column == self._sort_column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column = column
            # Numbers open biggest-first because that is what a ranking means;
            # the symbol column opens A-Z.
            self._sort_descending = column != 0
        self.table.horizontalHeader().setSortIndicator(column, self._indicator_order())
        # Re-render from the rows already in hand. No service call, no fetch.
        self._render()

    def sort_state(self) -> tuple[int, bool]:
        """(column, descending) - for tests and for the panel's status line."""
        return self._sort_column, self._sort_descending

    def _on_double_click(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.symbolActivated.emit(item.text())

    def _on_selection_changed(self) -> None:
        symbol = self.selected_symbol()
        if not symbol or symbol == self._selected_symbol:
            # Re-emitting on every repaint would re-point the chart while the
            # trader is reading it; a refresh that keeps the same row selected
            # is not a new request.
            return
        self._selected_symbol = symbol
        self.symbolActivated.emit(symbol)

    def selected_symbol(self) -> str:
        items = self.table.selectedItems()
        if not items:
            return ""
        item = self.table.item(items[0].row(), 0)
        return item.text() if item is not None else ""

    def clear_selection(self) -> None:
        """Used when the other side takes over the chart."""
        self.table.clearSelection()
        self._selected_symbol = ""

    def set_rows(self, rows: list[dict]) -> None:
        self._rows = [dict(row) for row in rows]
        self._render()

    def _render(self) -> None:
        rows = sort_rows(self._rows, self._sort_column, self._sort_descending)
        self._title.setText(f"{self._side.title()}s ({len(rows)})")
        keep = self._selected_symbol
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = (
                str(row.get("symbol") or ""),
                _fmt(row.get("strength"), 1),
                _fmt(row.get("day_pct"), 2, signed=True, suffix="%"),
                _fmt(row.get("vwap_distance_pct"), 2, signed=True, suffix="%"),
                _fmt(row.get("last"), 2),
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(index, column, item)
            button = QPushButton("Add to Focus")
            symbol = str(row.get("symbol") or "")
            # Bound to the SYMBOL, not the row index, so a re-sort cannot point
            # a button at its neighbour. Every add still re-runs the adoption
            # gate at click time (`_gate_row`), whatever order the rows are in.
            button.clicked.connect(
                lambda _checked=False, sym=symbol: self.addRequested.emit(sym, self._side)
            )
            self.table.setCellWidget(index, len(_COLUMNS), button)
        self.table.blockSignals(False)
        if keep:
            # A sort or a refresh must not silently move the chart to whatever
            # name landed on the old row number.
            self._reselect(keep)

    def _reselect(self, symbol: str) -> None:
        for index in range(self.table.rowCount()):
            item = self.table.item(index, 0)
            if item is not None and item.text() == symbol:
                self.table.selectRow(index)
                return
        # The name left the board on this refresh; the chart stays where it is
        # and the next explicit selection re-points it.
        self._selected_symbol = ""


def _fmt(value, places: int, *, signed: bool = False, suffix: str = "") -> str:
    """Blank rather than 0.00 for a missing number: an unmeasured field must
    not read as a measured zero."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    text = f"{number:+.{places}f}" if signed else f"{number:.{places}f}"
    return f"{text}{suffix}"


class StrengthBoardPanel(QWidget):
    """The board surface. Owns no data - `StrengthBoardService` does."""

    statusChanged = Signal(str)
    symbolActivated = Signal(str)

    def __init__(self, service=None, focus_service=None, parent=None) -> None:
        super().__init__(parent)
        self.service = service
        self.focus_service = focus_service

        layout = QVBoxLayout(self)
        layout.addWidget(SectionHeader("M5 Strength Board"))

        controls = QHBoxLayout()
        self.status = QLabel("Strength board: never refreshed")
        controls.addWidget(self.status)
        controls.addStretch(1)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip(
            "Refresh now. Manual refreshes are never gated on quiet hours."
        )
        self.refresh_button.clicked.connect(self._refresh)
        controls.addWidget(self.refresh_button)
        layout.addLayout(controls)

        self.hint = QLabel(
            "Click any column heading to sort. Select a row to chart it - the "
            "chart opens in the desk's usual snapshot popup, with the same "
            "levels and capture buttons as every other board."
        )
        self.hint.setObjectName("MutedLabel")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        tables = QHBoxLayout()
        self.longs = _SideTable("long")
        self.shorts = _SideTable("short")
        for table in (self.longs, self.shorts):
            table.addRequested.connect(self._add_one)
            table.addAllRequested.connect(self._add_all)
            tables.addWidget(table)
        # One chart at a time: selecting on one side drops the other side's
        # selection, so "the charted name" is never ambiguous.
        self.longs.symbolActivated.connect(
            lambda symbol: self._on_symbol_activated(symbol, self.shorts)
        )
        self.shorts.symbolActivated.connect(
            lambda symbol: self._on_symbol_activated(symbol, self.longs)
        )
        layout.addLayout(tables)

        if service is not None:
            service.boardChanged.connect(self.set_board)
            service.statusChanged.connect(self._on_status)
            self.set_board(service.board())
            self.status.setText(service.status_text())

    def _on_symbol_activated(self, symbol: str, other: "_SideTable") -> None:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        other.clear_selection()
        # The host (the Alert Center, as for every other board) owns the popup,
        # so the chart carries the same bot-backed series, painted levels and
        # capture rail as the RS/RW and Industry boards. Nothing new is drawn
        # here and nothing is fetched: this is R4's unification, not a second
        # chart widget.
        self.symbolActivated.emit(symbol)

    # ------------------------------------------------------------------ views
    def set_board(self, board: dict) -> None:
        self.longs.set_rows(list(board.get("long") or []))
        self.shorts.set_rows(list(board.get("short") or []))

    def _on_status(self, text: str) -> None:
        self.status.setText(text)
        self.statusChanged.emit(text)

    def _refresh(self) -> None:
        if self.service is not None:
            self.service.refresh_now()

    # ------------------------------------------------------------------ adds
    def _row_for(self, symbol: str, side: str) -> dict | None:
        board = self.service.board() if self.service is not None else {}
        for row in board.get(side) or []:
            if str(row.get("symbol") or "").upper() == str(symbol or "").upper():
                return row
        return None

    def _gate_row(self, row: dict, side: str) -> tuple[bool, str]:
        """Re-run the Part A gate on this row's own numbers at click time.

        The board refreshes on a 15-minute cadence, so a row can be several
        minutes stale by the time the trader clicks it. The gate is the same
        one the auto path uses - one definition, three call sites.
        """
        return focus_adoption_gate.passes_focus_adoption_gate(
            side,
            row.get("last"),
            row.get("prev_high"),
            row.get("prev_low"),
            row.get("session_vwap"),
        )

    def _add_one(self, symbol: str, side: str) -> None:
        added, refused = self._add_symbols([symbol], side)
        if refused:
            self.statusChanged.emit(f"✕ {refused[0]}")
        elif added:
            self.statusChanged.emit(f"★ {added[0]} added to M5 Focus ({side}).")

    def _add_all(self, side: str) -> None:
        board = self.service.board() if self.service is not None else {}
        symbols = [str(row.get("symbol") or "") for row in (board.get(side) or [])]
        added, refused = self._add_symbols(symbols, side)
        parts = []
        if added:
            parts.append(f"★ {len(added)} added to M5 Focus ({side})")
        if refused:
            # Named, not counted: "3 refused" tells the trader nothing they can
            # act on, and the reason is the whole value of the gate.
            parts.append(
                f"✕ refused {len(refused)}: "
                f"{'; '.join(refused[:4])}{'...' if len(refused) > 4 else ''}"
            )
        self.statusChanged.emit(" · ".join(parts) if parts else f"No {side} rows to add.")

    def _add_symbols(self, symbols: list[str], side: str) -> tuple[list[str], list[str]]:
        if self.focus_service is None:
            return [], ["no Focus service on this desk"]
        added: list[str] = []
        refused: list[str] = []
        for symbol in symbols:
            symbol = str(symbol or "").strip().upper()
            if not symbol:
                continue
            row = self._row_for(symbol, side)
            if row is None:
                refused.append(f"{symbol} (no longer on the board)")
                continue
            passes, reason = self._gate_row(row, side)
            if not passes:
                refused.append(f"{symbol} ({reason})")
                continue
            try:
                # Through the SERVICE, not the store: this add IS the trader
                # liking the name, so it belongs in the pick-feedback log - the
                # opposite of the machine's auto-adoption, which writes through
                # the store precisely to stay out of it.
                if self.focus_service.add(
                    symbol, side, "m5", origin="strength_board",
                    context=f"strength {_fmt(row.get('strength'), 1)}",
                ):
                    added.append(symbol)
            except Exception:
                logging.warning("Strength board could not add %s.", symbol, exc_info=True)
                refused.append(f"{symbol} (add failed)")
        return added, refused
