"""Today's swing picks - the trader's own list, under the M5 alert list.

Trader, 2026-08-31: *"at the end of the day I have a list of my top swing
targets. I want a place to put them in so the bot knows my personal favourite
picks... put it at the very bottom of the M5 alerts tab, the tab is so long and
I never use all of it."*

So it is a strip, not a panel: one input, a Long/Short toggle, and a chip per
pick with an x. It shares a column with the alert list and must never take
space that list wants, which is why the chips live in a short scroll area
rather than growing the strip.

Fluidity rules (2026-08-21 / 2026-08-31): the chip area is **diffed, never
rebuilt** - an add costs one insert and a removal one take, and the common case
of "nothing changed" costs zero layout operations. Every variant (side colour,
the taken mark) is a dynamic property answered by `theme.qss`; there is no
per-widget stylesheet in this file.

The "taken" mark is display only. It says a journal trade for that symbol was
opened on or after the day the trader picked it, and it derives nothing else -
no rate, no grade, no statistic. Nothing here reaches a detector, a score, an
alert, a watchlist ranking or `review_policy.json`.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.widgets.flow_layout import FlowLayout

#: How tall the chip area is allowed to get before it scrolls. The alert list
#: above it is the thing the trader is actually watching; this strip borrows
#: the bottom of that column and gives the rest back.
MAX_CHIP_HEIGHT = 132


class SwingFavoriteChip(QFrame):
    """One pick. Built once, then updated in place by :meth:`set_taken`."""

    removed = Signal(str, str)

    def __init__(self, symbol: str, side: str, *, taken: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.symbol = symbol
        self.side = side
        self.setObjectName("SwingFavoriteChip")
        self.setProperty("side", side)

        self.name_label = QLabel(symbol)
        self.name_label.setObjectName("SwingFavoriteName")
        self.name_label.setProperty("side", side)

        # Always present, hidden when it does not apply: adding and removing a
        # label would put layout work back on every refresh, which is the cost
        # the chip exists to avoid.
        self.taken_label = QLabel("took")
        self.taken_label.setObjectName("SwingFavoriteTaken")
        self.taken_label.setToolTip(
            "A journal trade for this symbol was opened on or after the day "
            "you picked it. Display only - nothing is graded or scored here."
        )

        remove_button = QToolButton()
        remove_button.setText("x")
        remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_button.setProperty("rowChrome", True)
        remove_button.setToolTip(f"Drop {symbol} from today's swing picks")
        remove_button.clicked.connect(lambda: self.removed.emit(self.symbol, self.side))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 4, 2)
        layout.setSpacing(4)
        layout.addWidget(self.name_label)
        layout.addWidget(self.taken_label)
        layout.addWidget(remove_button)

        self._taken: bool | None = None
        self.set_taken(taken)

    def set_taken(self, taken: bool) -> None:
        """Show or hide the taken mark. A no-op when it has not changed."""
        taken = bool(taken)
        if taken == self._taken:
            return
        self._taken = taken
        self.taken_label.setVisible(taken)
        self.setProperty("taken", "true" if taken else "false")
        # Re-polish only when the look actually changed; on a stable strip
        # that is never.
        self.style().unpolish(self)
        self.style().polish(self)


class SwingFavoritesBar(QWidget):
    """The strip: type or paste, pick a side, and the day's chips."""

    #: (text, side) - the raw input, so one paste can carry many symbols.
    addRequested = Signal(str, str)
    #: (symbol, side)
    removeRequested = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("SwingFavoritesBar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.title_label = QLabel("Today's swing picks")
        self.title_label.setObjectName("SectionTitle")
        self.title_label.setToolTip(
            "Your own vetted swing targets for today. Each one also joins the "
            "swing Focus list as YOUR pick, so nothing automatic removes it."
        )
        header.addWidget(self.title_label, 1)
        self.count_label = QLabel("0")
        self.count_label.setObjectName("MutedLabel")
        header.addWidget(self.count_label, 0)
        layout.addLayout(header)

        self.input = QLineEdit()
        self.input.setObjectName("SwingFavoriteInput")
        self.input.setPlaceholderText("Ticker(s), then Enter")
        self.input.setClearButtonEnabled(True)
        self.input.returnPressed.connect(self._emit_add)
        layout.addWidget(self.input)

        side_row = QHBoxLayout()
        side_row.setContentsMargins(0, 0, 0, 0)
        side_row.setSpacing(4)
        self.long_button = QToolButton()
        self.long_button.setText("Long")
        self.long_button.setCheckable(True)
        self.long_button.setChecked(True)
        self.long_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.long_button.clicked.connect(lambda: self.set_side("long"))
        self.short_button = QToolButton()
        self.short_button.setText("Short")
        self.short_button.setCheckable(True)
        self.short_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.short_button.clicked.connect(lambda: self.set_side("short"))
        self.add_button = QPushButton("Add")
        self.add_button.setToolTip(
            "Add these to today's swing picks and to the swing Focus list."
        )
        self.add_button.clicked.connect(self._emit_add)
        side_row.addWidget(self.long_button, 0)
        side_row.addWidget(self.short_button, 0)
        side_row.addWidget(self.add_button, 1)
        layout.addLayout(side_row)

        self.chip_host = QWidget()
        self.chip_flow = FlowLayout(self.chip_host, margin=2, spacing=4)
        self.chip_scroll = QScrollArea()
        self.chip_scroll.setObjectName("SwingFavoriteChips")
        self.chip_scroll.setWidgetResizable(True)
        self.chip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.chip_scroll.setMaximumHeight(theme.px(MAX_CHIP_HEIGHT))
        self.chip_scroll.setWidget(self.chip_host)
        layout.addWidget(self.chip_scroll)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._side = "long"
        self._taken: set[tuple[str, str]] = set()

    # ------------------------------------------------------------- input
    def side(self) -> str:
        return self._side

    def set_side(self, side: str) -> None:
        side = "short" if str(side).lower().startswith("short") else "long"
        self._side = side
        self.long_button.setChecked(side == "long")
        self.short_button.setChecked(side == "short")

    def _emit_add(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.addRequested.emit(text, self._side)

    def set_status(self, text: str) -> None:
        self.status_label.setText(str(text or ""))

    # -------------------------------------------------------------- chips
    def symbols(self) -> list[tuple[str, str]]:
        """(symbol, side) in the order the strip holds them."""
        return [(chip.symbol, chip.side) for chip in self._current_chips()]

    def _current_chips(self) -> list[SwingFavoriteChip]:
        chips = []
        for index in range(self.chip_flow.count()):
            item = self.chip_flow.itemAt(index)
            widget = item.widget() if item is not None else None
            if isinstance(widget, SwingFavoriteChip):
                chips.append(widget)
        return chips

    def set_favorites(self, favorites: Iterable[Mapping[str, Any]]) -> None:
        """Show this list. Diffed against what is on screen, never rebuilt."""
        wanted: list[tuple[str, str]] = []
        for row in favorites or ():
            symbol = str(row.get("symbol") or "").strip().upper()
            side = str(row.get("side") or "").strip().lower()
            if symbol and side in {"long", "short"} and (symbol, side) not in wanted:
                wanted.append((symbol, side))

        current = self._current_chips()
        if [(chip.symbol, chip.side) for chip in current] != wanted:
            self._apply_diff(wanted)
        for chip in self._current_chips():
            chip.set_taken((chip.symbol, chip.side) in self._taken)
        self.count_label.setText(str(len(wanted)))

    def set_taken(self, taken: Iterable[tuple[str, str]]) -> None:
        """Re-point the taken marks. No layout work, ever."""
        self._taken = {
            (str(symbol).strip().upper(), str(side).strip().lower())
            for symbol, side in (taken or ())
        }
        for chip in self._current_chips():
            chip.set_taken((chip.symbol, chip.side) in self._taken)

    def _apply_diff(self, wanted: list[tuple[str, str]]) -> None:
        wanted_set = set(wanted)
        seen: set[tuple[str, str]] = set()
        departures: list[int] = []
        for index in range(self.chip_flow.count()):
            item = self.chip_flow.itemAt(index)
            widget = item.widget() if item is not None else None
            if not isinstance(widget, SwingFavoriteChip):
                departures.append(index)
                continue
            key = (widget.symbol, widget.side)
            if key not in wanted_set or key in seen:
                departures.append(index)
                continue
            seen.add(key)
        # Highest index first, so the ones still owed keep their positions.
        for index in reversed(departures):
            item = self.chip_flow.takeAt(index)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        for position, key in enumerate(wanted):
            item = self.chip_flow.itemAt(position)
            widget = item.widget() if item is not None else None
            if isinstance(widget, SwingFavoriteChip) and (widget.symbol, widget.side) == key:
                continue
            chip = self._take_chip(key)
            if chip is None:
                chip = SwingFavoriteChip(key[0], key[1], taken=key in self._taken)
                chip.removed.connect(self.removeRequested)
            self.chip_flow.insertWidget(position, chip)
        self.chip_flow.invalidate()

    def _take_chip(self, key: tuple[str, str]) -> SwingFavoriteChip | None:
        for index in range(self.chip_flow.count()):
            item = self.chip_flow.itemAt(index)
            widget = item.widget() if item is not None else None
            if isinstance(widget, SwingFavoriteChip) and (widget.symbol, widget.side) == key:
                self.chip_flow.takeAt(index)
                return widget
        return None
