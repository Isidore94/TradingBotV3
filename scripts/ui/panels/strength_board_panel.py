"""The M5 strength board (plan.md Phase 0.5, packet R2 Part B.3.4).

Side-split rows from `StrengthBoardService`, with one-click **Add to M5 Focus**
per row and a side-aware **Add all shown**. Every add passes through packet R2
Part A's adoption gate, and a row that fails it at click time is refused with
the reason shown rather than silently dropped - the trader asked why a name is
not there often enough that "nothing happened" is not an acceptable answer.

Decision support only: no alerts, no watchlist writes beyond the Focus adds the
trader explicitly clicks, and no influence on any champion path.
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


class _SideTable(QWidget):
    """One side's rows plus its add buttons."""

    addRequested = Signal(str, str)      # symbol, side
    addAllRequested = Signal(str)        # side
    symbolActivated = Signal(str)        # symbol (open the chart popup)

    def __init__(self, side: str, parent=None) -> None:
        super().__init__(parent)
        self._side = side
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
        layout.addWidget(self.table)

    def _on_double_click(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.symbolActivated.emit(item.text())

    def set_rows(self, rows: list[dict]) -> None:
        self._title.setText(f"{self._side.title()}s ({len(rows)})")
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
            button.clicked.connect(
                lambda _checked=False, sym=symbol: self.addRequested.emit(sym, self._side)
            )
            self.table.setCellWidget(index, len(_COLUMNS), button)


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

        tables = QHBoxLayout()
        self.longs = _SideTable("long")
        self.shorts = _SideTable("short")
        for table in (self.longs, self.shorts):
            table.addRequested.connect(self._add_one)
            table.addAllRequested.connect(self._add_all)
            table.symbolActivated.connect(self.symbolActivated.emit)
            tables.addWidget(table)
        layout.addLayout(tables)

        if service is not None:
            service.boardChanged.connect(self.set_board)
            service.statusChanged.connect(self._on_status)
            self.set_board(service.board())
            self.status.setText(service.status_text())

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
