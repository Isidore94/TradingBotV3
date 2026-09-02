"""Today's swing picks - the trader's own list, under the M5 alert list.

Trader, 2026-08-31: *"at the end of the day I have a list of my top swing
targets. I want a place to put them in so the bot knows my personal favourite
picks... put it at the very bottom of the M5 alerts tab, the tab is so long and
I never use all of it."*

So it is a strip, not a panel: one input, a Long/Short toggle, Copy/Paste for
the trader's TC2000 list, and a chip per pick with an x. It shares a DRAGGABLE
split with the alert list above it (trader, 2026-08-31: *"the tab needs to be
resizable relative to the M5 alerts tab, I should be able to drag it up to see
more"*), so the chip area has a floor and no ceiling - how much of the column
this strip gets is the trader's decision, saved across restarts.

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
    QApplication,
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

#: The chip area never goes below this, so the strip cannot be dragged into a
#: sliver with no visible picks. It has no ceiling: the trader asked to be able
#: to "drag it up to see more" (2026-08-31), and a maximum height would make the
#: drag do nothing past the ceiling.
MIN_CHIP_HEIGHT = 44


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
        self._taken_trade_id: str = ""
        self.set_taken(taken)

    def set_taken(self, taken: bool, trade_id: str = "") -> None:
        """Show or hide the taken mark. A no-op when it has not changed.

        `trade_id` names the journal row the badge is about (P6), and rides on
        the mark's TOOLTIP: the badge stays display-only and computes nothing,
        and an id in a tooltip is a pointer the trader can follow rather than a
        number that could be mistaken for a result.
        """
        taken = bool(taken)
        trade_id = str(trade_id or "")
        if taken == self._taken and trade_id == getattr(self, "_taken_trade_id", ""):
            return
        self._taken = taken
        self._taken_trade_id = trade_id
        self.taken_label.setVisible(taken)
        self.taken_label.setToolTip(
            f"Traded - journal trade {trade_id}" if taken and trade_id
            else ("Traded (the journal row could not be identified)" if taken else "")
        )
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
    #: Emitted once, the first time the strip is actually on screen. The
    #: journal read behind the "took" badge hangs off this rather than off
    #: construction: a strip nobody has looked at yet has no reason to open the
    #: journal, and a worker started during __init__ outlives any host that is
    #: torn down without calling shutdown().
    firstShown = Signal()

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
        self.copy_button = QToolButton()
        self.copy_button.setText("Copy")
        self.copy_button.setProperty("rowChrome", True)
        self.copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_button.setToolTip(
            "Copy today's picks to the clipboard, one ticker per line, each "
            "once, in list order - paste straight into a TC2000 watchlist."
        )
        self.copy_button.clicked.connect(self.copy_all)
        header.addWidget(self.copy_button, 0)
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
        self.paste_button = QPushButton("Paste")
        self.paste_button.setToolTip(
            "Add every ticker on the clipboard to today's swing picks, on the "
            "side selected here - paste a TC2000 list straight in."
        )
        self.paste_button.clicked.connect(self.paste)
        side_row.addWidget(self.long_button, 0)
        side_row.addWidget(self.short_button, 0)
        side_row.addWidget(self.add_button, 1)
        side_row.addWidget(self.paste_button, 1)
        layout.addLayout(side_row)

        self.chip_host = QWidget()
        self.chip_flow = FlowLayout(self.chip_host, margin=2, spacing=4)
        self.chip_scroll = QScrollArea()
        self.chip_scroll.setObjectName("SwingFavoriteChips")
        self.chip_scroll.setWidgetResizable(True)
        self.chip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # A floor, not a ceiling: the strip shares a draggable split with the
        # alert list, so the trader decides how much of it they want.
        self.chip_scroll.setMinimumHeight(theme.px(MIN_CHIP_HEIGHT))
        self.chip_scroll.setWidget(self.chip_host)
        layout.addWidget(self.chip_scroll, 1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._side = "long"
        self._taken: set[tuple[str, str]] = set()
        #: (symbol, side) -> journal trade_id, for the badge's tooltip.
        self._taken_trade_ids: dict[tuple[str, str], str] = {}
        self._announced_shown = False

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if not self._announced_shown:
            self._announced_shown = True
            self.firstShown.emit()

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

    def paste(self) -> None:
        """Add whatever is on the clipboard, on the currently selected side."""
        text = QApplication.clipboard().text().strip()
        if not text:
            self.set_status("Clipboard is empty.")
            return
        self.addRequested.emit(text, self._side)

    def copy_all(self) -> str:
        """Put today's tickers on the clipboard, one per line. Returns the text."""
        symbols = list(dict.fromkeys(symbol for symbol, _side in self.symbols()))
        text = "\n".join(symbols)
        QApplication.clipboard().setText(text)
        self.set_status(
            f"Copied {len(symbols)} ticker{'' if len(symbols) == 1 else 's'}."
            if symbols
            else "Nothing to copy yet."
        )
        return text

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
            key = (chip.symbol, chip.side)
            chip.set_taken(key in self._taken, self._taken_trade_ids.get(key, ""))
        self.count_label.setText(str(len(wanted)))

    def set_taken(self, taken) -> None:
        """Re-point the taken marks. No layout work, ever.

        Accepts either the historical SET of (symbol, side) or a MAPPING of
        those to a journal trade_id (P6). Both are supported deliberately: the
        set is what every existing caller and test passes, and a mapping only
        adds the id the badge's tooltip names. Which one arrived changes
        nothing about which chips are marked.
        """
        pairs = list((taken or {}).items()) if hasattr(taken, "items") else [
            (pair, "") for pair in (taken or ())
        ]
        self._taken = set()
        self._taken_trade_ids = {}
        for pair, trade_id in pairs:
            symbol, side = pair
            key = (str(symbol).strip().upper(), str(side).strip().lower())
            self._taken.add(key)
            if trade_id:
                self._taken_trade_ids[key] = str(trade_id)
        for chip in self._current_chips():
            key = (chip.symbol, chip.side)
            chip.set_taken(key in self._taken, self._taken_trade_ids.get(key, ""))

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
