"""The Trading Desk's ONE Watchlist tab (WS-WL items 2-5, WISHLIST 10G).

The trader's brief: *"the main Watchlist tab belongs on Trading Desk"* - one
list with five views (My watchlist | M5/TC2000 | Swing favorites | Open
positions | All), source badges and side/horizon shown **without turning source
into priority**, single-symbol add and paste-many with duplicate handling, and
the Journal linking to its Positions view.

This panel is a READER plus a router. It computes nothing: `watchlist_views`
builds the rows and `ui.services.watchlist_tab_service` reads the stores. What
it owns is the table, the view selector, the add box, and the routing of every
verb to **the store that already owned it**:

=========================  ===================================================
verb                       owner it asks
=========================  ===================================================
add / paste (manual)       `WatchlistEditorPanel` (WS-5D's intent seam)
remove, manual name        `WatchlistEditorPanel.remove_symbols`
remove, Focus pick         `FocusService.remove_everywhere` (records `unfavorite`)
remove, swing favorite     a `swing_favorites` RETRACTION row
remove, position           **nothing** - a broker position is not a list entry
like / not today           `ui.annotations.verdicts` + `remove_if_auto_adopted`
restore a faded pick       `FocusPickStore.restore_faded`
arm / disarm / re-arm      `PriceAlertService.save_entries`
chart                      the Alert Center's `chart_symbol` (a MANUAL look)
=========================  ===================================================

Nothing new writes anything. The consolidation is a SURFACE: the same four
writers, one screen.

Two rules the retirement of the Chart Review and Focus Picks pages puts here:

* **`Ctrl+L` is bound at this panel's scope.** A `QShortcut` in a hidden tab
  never fires and two bindings for one sequence fire neither, so the window
  never binds it and raising this tab moves focus INTO it - otherwise the focus
  widget is the tab BAR, which is not a child of this panel and the shortcut
  would not match.
* **A position row has no Remove and arms nothing.** `can_remove()` answers
  before the button is drawn, and a position is never a reason to write a price
  alert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

import watchlist_views
from ui import theme
from ui.widgets.section_header import SectionHeader

logger = logging.getLogger(__name__)

#: The column order. Symbol first, then the two facts that are not a source,
#: then the badges. Source is a BADGE, never a rank - it does not sort anything.
COLUMNS = (
    ("symbol", "Symbol"),
    ("side", "Side"),
    ("horizon", "Horizon"),
    ("sources", "Where from"),
    ("position", "Position"),
    ("alerts", "Alerts"),
    ("marks", "Today"),
)

SOURCES_COLUMN = 3

#: The role the chip delegate reads. A list of (label, colour-token) pairs.
CHIPS_ROLE = Qt.ItemDataRole.UserRole + 11
#: `(symbol, side)` - the row identity `watchlist_views` uses.
KEY_ROLE = Qt.ItemDataRole.UserRole + 12

#: Badge -> theme colour token. An unknown token paints `neutral` (theme.color
#: answers an unknown name with it), so a new source is grey, never a crash.
_CHIP_COLORS = {
    watchlist_views.SOURCE_MANUAL: "accent",
    watchlist_views.SOURCE_FOCUS_TRADER: "favorite",
    watchlist_views.SOURCE_FOCUS_AUTO: "study",
    watchlist_views.SOURCE_BOARD: "near",
    watchlist_views.SOURCE_SWING_FAVORITE: "long",
    watchlist_views.SOURCE_POSITION: "caution",
    watchlist_views.SOURCE_ALERT: "info",
}

_CHIP_HEIGHT = 18
_CHIP_GAP = 4
_CHIP_PAD = 6


@dataclass
class AddResult:
    """What one add or paste did. `duplicates` counts OCCURRENCES, not names.

    A name typed twice in one paste is one add and one duplicate, which is what
    the trader means by "it told me it was already there".
    """

    added: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()
    message: str = ""

    def __bool__(self) -> bool:
        return bool(self.added)


class ChipDelegate(QStyledItemDelegate):
    """Paints the source badges as chips (the G2b delegate pattern).

    A delegate rather than per-cell widgets on purpose: the desk learned in
    2026-08 that a stylesheet per chip is expensive work on the Qt thread, and
    a table of 60 names would be 200 styled widgets rebuilt on every refresh.
    """

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: N802 - Qt
        chips = index.data(CHIPS_ROLE) or ()
        if not chips:
            super().paint(painter, option, index)
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QColor(theme.color("selection")))
        metrics = option.fontMetrics
        x = option.rect.left() + _CHIP_PAD
        top = option.rect.top() + max(0, (option.rect.height() - _CHIP_HEIGHT) // 2)
        for label, token in chips:
            width = metrics.horizontalAdvance(str(label)) + 2 * _CHIP_PAD
            if x + width > option.rect.right():
                break
            rect = QRect(x, top, width, _CHIP_HEIGHT)
            colour = QColor(theme.color(token))
            fill = QColor(colour)
            fill.setAlpha(46)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(colour)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(label))
            x += width + _CHIP_GAP
        painter.restore()


class WatchlistTabPanel(QFrame):
    """One watchlist, many owners. See the module docstring."""

    statusChanged = Signal(str)
    #: The trader asked for a name on the shared chart. The desk wires this to
    #: the Alert Center's `chart_symbol` through `set_chart_sink`.
    symbolActivated = Signal(str)

    def __init__(
        self,
        *,
        service: Any = None,
        focus_service: Any = None,
        price_alert_service: Any = None,
        watchlists_panel: Any = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._service = service
        self._focus_service = focus_service
        self._price_alert_service = price_alert_service
        self._watchlists_panel = watchlists_panel
        self._chart_sink: Callable[..., Any] | None = None
        self._rows: tuple[watchlist_views.WatchRow, ...] = ()
        self._visible: tuple[watchlist_views.WatchRow, ...] = ()
        self._view = watchlist_views.VIEW_ALL
        self._first_shown = False

        self._build_layout()
        self._bind_shortcuts()
        if self._service is not None:
            try:
                self._service.rowsChanged.connect(self.set_rows)
                self._service.statusChanged.connect(self.statusChanged)
            except Exception:  # noqa: BLE001
                logger.debug("watchlist tab: service not connected", exc_info=True)

    # ------------------------------------------------------------------ build
    def _build_layout(self) -> None:
        self.view_selector = QComboBox()
        for name in watchlist_views.VIEWS:
            self.view_selector.addItem(watchlist_views.VIEW_LABELS[name], name)
        self.view_selector.setCurrentIndex(len(watchlist_views.VIEWS) - 1)
        self.view_selector.currentIndexChanged.connect(
            lambda _index: self.set_view(self.view_selector.currentData())
        )

        self.add_input = QLineEdit()
        self.add_input.setPlaceholderText("Add ticker (Ctrl+L)")
        self.add_input.returnPressed.connect(self._add_from_input)
        self.side_selector = QComboBox()
        self.side_selector.addItems(["long", "short"])
        self.horizon_selector = QComboBox()
        self.horizon_selector.addItems([watchlist_views.HORIZON_DAY, watchlist_views.HORIZON_SWING])

        self.add_button = QPushButton("Add")
        self.paste_button = QPushButton("Paste many")
        self.copy_button = QPushButton("Copy")
        self.refresh_button = QPushButton("Refresh")
        self.add_button.clicked.connect(self._add_from_input)
        self.paste_button.clicked.connect(self._paste_from_clipboard)
        self.copy_button.clicked.connect(self.copy_visible)
        self.refresh_button.clicked.connect(self._manual_refresh)

        self.chart_button = QPushButton("Chart")
        self.like_button = QPushButton("Like")
        self.not_today_button = QPushButton("Not today")
        self.remove_button = QPushButton("Remove")
        self.restore_button = QPushButton("Restore")
        self.arm_button = QPushButton("Arm alert")
        self.disarm_button = QPushButton("Disarm")
        self.rearm_button = QPushButton("Re-arm")
        self.above_input = QLineEdit()
        self.above_input.setPlaceholderText("Cross up")
        self.below_input = QLineEdit()
        self.below_input.setPlaceholderText("Cross down")
        self.chart_button.clicked.connect(self.chart_selected)
        self.like_button.clicked.connect(self.like_selected)
        self.not_today_button.clicked.connect(self.not_today_selected)
        self.remove_button.clicked.connect(self.remove_selected)
        self.restore_button.clicked.connect(self.restore_selected)
        self.arm_button.clicked.connect(self._arm_from_inputs)
        self.disarm_button.clicked.connect(self.disarm_selected)
        self.rearm_button.clicked.connect(self.rearm_selected)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _key, label in COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setItemDelegateForColumn(SOURCES_COLUMN, ChipDelegate(self.table))
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(theme.px(44))
        # Every section stretches, so the header is exactly as wide as the
        # viewport at any desk size. Half a table behind a horizontal scrollbar
        # is the 2026-08 setups defect and this column can be narrow.
        for index in range(len(COLUMNS)):
            header.setSectionResizeMode(index, QHeaderView.ResizeMode.Stretch)
        self.table.doubleClicked.connect(lambda _index: self.chart_selected())

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        top.addWidget(QLabel("View"))
        top.addWidget(self.view_selector)
        top.addStretch(1)
        top.addWidget(self.refresh_button)

        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        add_row.setSpacing(6)
        add_row.addWidget(self.add_input, 1)
        add_row.addWidget(self.side_selector)
        add_row.addWidget(self.horizon_selector)
        add_row.addWidget(self.add_button)
        add_row.addWidget(self.paste_button)
        add_row.addWidget(self.copy_button)

        verbs = QHBoxLayout()
        verbs.setContentsMargins(0, 0, 0, 0)
        verbs.setSpacing(6)
        for button in (
            self.chart_button,
            self.like_button,
            self.not_today_button,
            self.remove_button,
            self.restore_button,
        ):
            verbs.addWidget(button)
        verbs.addStretch(1)

        alerts = QHBoxLayout()
        alerts.setContentsMargins(0, 0, 0, 0)
        alerts.setSpacing(6)
        alerts.addWidget(self.above_input)
        alerts.addWidget(self.below_input)
        alerts.addWidget(self.arm_button)
        alerts.addWidget(self.disarm_button)
        alerts.addWidget(self.rearm_button)
        alerts.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(
            SectionHeader(
                "Watchlist",
                "Every name the desk is holding an opinion about. The badge says "
                "where it came from - it never changes the order.",
            )
        )
        layout.addLayout(top)
        layout.addLayout(add_row)
        layout.addLayout(verbs)
        layout.addLayout(alerts)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table, 1)

    def _bind_shortcuts(self) -> None:
        """`Ctrl+L` - the Chart Review lookup box's key, on the add box now.

        Panel scope, exactly one binding. See the module docstring.
        """
        shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self.focus_lookup)
        self._lookup_shortcut = shortcut

    # ------------------------------------------------------------------ wiring
    def set_chart_sink(self, sink) -> None:
        """Point a row click at the desk's centre chart (`watchlists_panel`'s
        pattern). `None` falls back to the `symbolActivated` signal alone."""
        self._chart_sink = sink

    def set_service(self, service) -> None:
        self._service = service
        if service is not None:
            service.rowsChanged.connect(self.set_rows)
            service.statusChanged.connect(self.statusChanged)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        # Hidden tabs pay nothing: the first time this one is actually on
        # screen it catches up, and the focus moves INSIDE the panel so the
        # panel-scoped shortcut can match (the tab bar is not a child of ours).
        if not self._first_shown:
            self._first_shown = True
            self.refresh_now()
        self.take_focus()

    def take_focus(self) -> None:
        """Put the keyboard inside this panel without stealing the caret."""
        if self.add_input.hasFocus():
            return
        self.table.setFocus(Qt.FocusReason.OtherFocusReason)

    # ------------------------------------------------------------------- rows
    def rows(self) -> tuple[watchlist_views.WatchRow, ...]:
        """Every row, before any view filter. Written BEFORE the filter runs."""
        return self._rows

    def visible_rows(self) -> tuple[watchlist_views.WatchRow, ...]:
        """The rows the table is showing, in order."""
        return self._visible

    def view(self) -> str:
        return self._view

    def set_view(self, view: str) -> None:
        name = str(view or "").strip().lower() or watchlist_views.VIEW_ALL
        if name not in watchlist_views.VIEWS:
            return
        self._view = name
        index = self.view_selector.findData(name)
        if index >= 0 and self.view_selector.currentIndex() != index:
            self.view_selector.blockSignals(True)
            self.view_selector.setCurrentIndex(index)
            self.view_selector.blockSignals(False)
        self._repaint_table()

    def set_rows(self, rows: Iterable[watchlist_views.WatchRow]) -> None:
        """Take a row set and show it. The backing list is written FIRST."""
        self._rows = tuple(rows or ())
        self._repaint_table()

    def refresh_now(self) -> bool:
        """Rebuild from the stores NOW, and ask the service for the journal.

        Synchronous over the cheap stores (four text files, two JSONL, one
        JSON) plus the service's cached journal snapshot; the sqlite read that
        the Positions view needs runs on the service's worker and lands after.
        """
        from ui.services import watchlist_tab_service as svc

        service = self._service
        try:
            payload = svc.gather(
                focus=self._focus_snapshot(),
                armed_alerts=self._armed_entries(),
                board_rows=service.board_rows() if service is not None else {},
                journal=service.journal_snapshot() if service is not None else None,
            )
            self.set_rows(watchlist_views.build_watchlist_rows(**payload))
        except Exception as exc:  # noqa: BLE001 - the tab outlives a torn store
            logger.exception("watchlist tab refresh failed")
            self._set_status(f"Could not read the watchlist: {exc}")
            return False
        if service is not None:
            service.refresh_now()
        return True

    def _manual_refresh(self) -> None:
        self.refresh_now()
        self._set_status(f"{len(self._visible)} name(s) on this view.")

    def _focus_snapshot(self):
        store = getattr(self._focus_service, "store", None)
        if store is None:
            return None
        return watchlist_views.focus_snapshot(store)

    def _armed_entries(self) -> tuple:
        if self._price_alert_service is not None:
            try:
                return tuple(self._price_alert_service.entries())
            except Exception:  # noqa: BLE001
                return ()
        try:
            import price_alerts

            return tuple(price_alerts.load_price_alerts())
        except Exception:  # noqa: BLE001
            return ()

    # ------------------------------------------------------------------ table
    def _repaint_table(self) -> None:
        """Diff the table onto the filtered rows. Selection survives.

        Lists diff, never rebuild: the table keeps its row count where it can,
        rewrites text in place, and re-selects by (symbol, side) rather than by
        the row number - a name inserted above the selection must not move it.
        """
        selected = self.selected_key()
        rows = watchlist_views.filter_rows(self._rows, self._view)
        self._visible = rows
        self.table.setUpdatesEnabled(False)
        try:
            if self.table.rowCount() != len(rows):
                self.table.setRowCount(len(rows))
            for index, row in enumerate(rows):
                self._write_row(index, row)
        finally:
            self.table.setUpdatesEnabled(True)
        if selected is not None:
            self.select_symbol(selected[0], side=selected[1])

    def _write_row(self, index: int, row: watchlist_views.WatchRow) -> None:
        values = (
            row.symbol,
            row.side,
            row.horizon_text,
            "",  # the chip column paints from CHIPS_ROLE
            _position_text(row),
            str(row.armed_alerts) if row.armed_alerts else "",
            _marks_text(row),
        )
        chips = tuple(
            (watchlist_views.SOURCE_LABELS[name], _CHIP_COLORS.get(name, "neutral"))
            for name in watchlist_views.SOURCES
            if name in row.sources
        )
        for column, text in enumerate(values):
            item = self.table.item(index, column)
            if item is None:
                item = QTableWidgetItem()
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                self.table.setItem(index, column, item)
            if item.text() != text:
                item.setText(text)
            if column == SOURCES_COLUMN:
                item.setData(CHIPS_ROLE, chips)
                item.setToolTip(", ".join(label for label, _token in chips))
            if column == 0:
                item.setData(KEY_ROLE, (row.symbol, row.side))
                item.setToolTip(_row_tooltip(row))
            if column == 4 and row.positions and any(p.stale for p in row.positions):
                # A stale sync is SHOWN, greyed. Never removed - uncertainty
                # does not delete a position.
                item.setForeground(QColor(theme.color("text_muted")))

    # -------------------------------------------------------------- selection
    def selected_key(self) -> tuple[str, str] | None:
        rows = self.table.selectionModel().selectedRows() if self.table.model() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        if item is None:
            return None
        key = item.data(KEY_ROLE)
        return (str(key[0]), str(key[1])) if key else None

    def selected_symbol(self) -> str:
        key = self.selected_key()
        return key[0] if key else ""

    def selected_row(self) -> watchlist_views.WatchRow | None:
        key = self.selected_key()
        if key is None:
            return None
        for row in self._visible:
            if (row.symbol, row.side) == key:
                return row
        return None

    def select_symbol(self, symbol: object, side: str = "") -> bool:
        """Select one row by identity. Returns False when it is not showing."""
        want_symbol = str(symbol or "").strip().upper()
        want_side = str(side or "").strip().lower()
        for index, row in enumerate(self._visible):
            if row.symbol != want_symbol:
                continue
            if want_side and row.side != want_side:
                continue
            self.table.selectRow(index)
            return True
        return False

    def _on_selection_changed(self) -> None:
        row = self.selected_row()
        can_remove = self.can_remove()
        self.remove_button.setEnabled(can_remove)
        self.restore_button.setEnabled(bool(row is not None and row.faded))

    # ------------------------------------------------------------------ verbs
    def add_symbol(self, text: object, side: str = "", horizon: str = "") -> AddResult:
        """One name onto the trader's own list, through its existing writer."""
        return self._write_manual(text, side=side, horizon=horizon, pasted=False)

    def paste_many(self, text: object, side: str = "", horizon: str = "") -> AddResult:
        """Many names at once. Duplicates are REPORTED and never doubled."""
        return self._write_manual(text, side=side, horizon=horizon, pasted=True)

    def copy_visible(self) -> str:
        """The names on this view, comma separated, on the clipboard."""
        from PySide6.QtWidgets import QApplication

        text = ", ".join(row.symbol for row in self._visible)
        QApplication.clipboard().setText(text)
        self._set_status(f"Copied {len(self._visible)} name(s).")
        return text

    def clear_view(self) -> int:
        """Take every name on THIS view off, each through its own owner.

        The Focus Picks page's "Clear All" cleared one side of one category.
        Here it clears what is in front of the trader, and it still cannot
        remove a name it does not own - a position row is skipped.
        """
        removed = 0
        for row in tuple(self._visible):
            if self._remove_row(row):
                removed += 1
        self.refresh_now()
        self._set_status(f"Removed {removed} name(s) from this view.")
        return removed

    def chart_selected(self) -> bool:
        """The shared chart. A board look - never a queue entry, never a skip."""
        symbol = self.selected_symbol()
        if not symbol:
            return False
        self.symbolActivated.emit(symbol)
        if self._chart_sink is None:
            return False
        row = self.selected_row()
        self._chart_sink(
            symbol, side=(row.side.upper() if row and row.side else ""), origin="the Watchlist"
        )
        return True

    def focus_lookup(self) -> None:
        """`Ctrl+L`: the add box takes the caret and selects what is in it."""
        self.add_input.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.add_input.selectAll()

    def can_remove(self) -> bool:
        """False for a position-only row: the broker's answer is not a list."""
        row = self.selected_row()
        if row is None:
            return False
        return bool(
            row.sources
            & {
                watchlist_views.SOURCE_MANUAL,
                watchlist_views.SOURCE_FOCUS_TRADER,
                watchlist_views.SOURCE_FOCUS_AUTO,
                watchlist_views.SOURCE_SWING_FAVORITE,
            }
        )

    def remove_selected(self) -> bool:
        row = self.selected_row()
        if row is None or not self.can_remove():
            return False
        removed = self._remove_row(row)
        if removed:
            self.refresh_now()
            self._set_status(f"Removed {row.symbol}.")
        return removed

    def restore_selected(self) -> bool:
        """A faded pick back in Focus, with a fresh clock.

        `restore_faded`, never `discard_faded`: discard clears the entry
        WITHOUT putting the pick back, so a Restore wired to it would quietly
        lose the name.
        """
        row = self.selected_row()
        store = getattr(self._focus_service, "store", None)
        if row is None or store is None:
            return False
        restored = False
        for entry in store.faded_picks():
            if str(entry.get("symbol") or "").upper() != row.symbol:
                continue
            if row.side and str(entry.get("side") or "").lower() != row.side:
                continue
            try:
                restored = bool(
                    store.restore_faded(
                        entry.get("symbol"),
                        entry.get("side"),
                        entry.get("category") or "m5",
                    )
                )
            except Exception:  # noqa: BLE001
                logger.exception("watchlist tab: restore failed")
                restored = False
            break
        if restored:
            self.refresh_now()
            self._set_status(f"Restored {row.symbol} to Focus.")
        return restored

    def like_selected(self) -> bool:
        """A like carries ZERO privileges (P9): nothing moves, nothing is armed."""
        row = self.selected_row()
        if row is None:
            return False
        try:
            from ui.annotations import verdicts

            written = verdicts.record_like(
                symbol=row.symbol,
                side=self._annotation_side(row),
                surface=verdicts.SURFACE_FOCUS_PANEL,
                timeframe=self._annotation_timeframe(row),
            )
        except Exception:  # noqa: BLE001
            self._set_status(f"{row.symbol}: like NOT saved.")
            return False
        if written is None:
            self._set_status(f"{row.symbol}: like NOT saved.")
            return False
        self._set_status(f"Liked {row.symbol}. Nothing was placed or armed.")
        return True

    def not_today_selected(self) -> bool:
        """The verdict is recorded FIRST; only a MACHINE-placed pick leaves.

        `remove_if_auto_adopted` is the seam that decides, and the absence of
        an auto marker means the trader owns the name - so the same click on
        their own pick records the decision and changes nothing.
        """
        row = self.selected_row()
        if row is None:
            return False
        try:
            from ui.annotations import verdicts

            verdicts.record_dislike(
                symbol=row.symbol,
                side=self._annotation_side(row),
                surface=verdicts.SURFACE_FOCUS_PANEL,
                timeframe=self._annotation_timeframe(row),
            )
        except Exception:  # noqa: BLE001
            logger.debug("watchlist tab: dislike not recorded", exc_info=True)
        dropped = False
        service = self._focus_service
        if service is not None:
            for category in self._focus_categories(row):
                try:
                    dropped = bool(
                        service.remove_if_auto_adopted(row.symbol, row.side or "long", category)
                    ) or dropped
                except Exception:  # noqa: BLE001
                    logger.debug("watchlist tab: not-today removal failed", exc_info=True)
        self.refresh_now()
        if dropped:
            self._set_status(f"Not today: {row.symbol} dropped from this list.")
        else:
            self._set_status(
                f"Not today: {row.symbol} recorded. It stays on the list - "
                "you added it, so only you take it off."
            )
        return True

    # ------------------------------------------------------------ price alerts
    def arm_selected(self, above: float | None = None, below: float | None = None) -> bool:
        """One entry through `PriceAlertService.save_entries` - the existing
        identity, the existing store, the existing poller."""
        row = self.selected_row()
        service = self._price_alert_service
        if row is None or service is None:
            return False
        above = _level(above)
        below = _level(below)
        if above is None and below is None:
            self._set_status("Enter a cross-up or cross-down price first.")
            return False
        import price_alerts

        entries = service.entries()
        existing = next(
            (entry for entry in entries if entry.get("symbol") == row.symbol), None
        )
        if existing is None:
            entries.append(
                price_alerts.mark_armed_now(
                    {
                        "symbol": row.symbol,
                        "above": above,
                        "below": below,
                        "armed_above": above is not None,
                        "armed_below": below is not None,
                        "note": "",
                        "history": [],
                    }
                )
            )
        else:
            if above is not None:
                existing["above"] = above
                existing["armed_above"] = True
            if below is not None:
                existing["below"] = below
                existing["armed_below"] = True
            price_alerts.mark_armed_now(existing)
        saved = bool(service.save_entries(entries))
        if saved:
            self.refresh_now()
            self._set_status(f"{row.symbol}: price alert armed.")
        return saved

    def _arm_from_inputs(self) -> bool:
        return self.arm_selected(
            above=_level(self.above_input.text()), below=_level(self.below_input.text())
        )

    def disarm_selected(self) -> bool:
        """DISARMED, never deleted (A2). The entry and its history stay."""
        row = self.selected_row()
        service = self._price_alert_service
        if row is None or service is None:
            return False
        entries = service.entries()
        touched = False
        for entry in entries:
            if entry.get("symbol") != row.symbol:
                continue
            if entry.get("armed_above") or entry.get("armed_below"):
                entry["armed_above"] = False
                entry["armed_below"] = False
                touched = True
        if not touched:
            return False
        saved = bool(service.save_entries(entries))
        if saved:
            self.refresh_now()
            self._set_status(f"{row.symbol}: price alert disarmed (the levels stay).")
        return saved

    def rearm_selected(self) -> bool:
        row = self.selected_row()
        service = self._price_alert_service
        if row is None or service is None:
            return False
        import price_alerts

        entries = service.entries()
        touched = False
        for entry in entries:
            if entry.get("symbol") != row.symbol:
                continue
            entry["armed_above"] = entry.get("above") is not None
            entry["armed_below"] = entry.get("below") is not None
            price_alerts.mark_armed_now(entry)
            touched = touched or entry["armed_above"] or entry["armed_below"]
        if not touched:
            return False
        saved = bool(service.save_entries(entries))
        if saved:
            self.refresh_now()
            self._set_status(f"{row.symbol}: price alert re-armed.")
        return saved

    def snapshot_today(self, *, force: bool = False) -> Any:
        """The Focus Picks page's "Snapshot Today", asking the same writer.

        A custom (test) focus store is refused for the same reason it always
        was: the daily snapshot names the trader's real picks and a sandbox
        store would write a day of fiction into the tracking file.
        """
        service = self._focus_service
        store = getattr(service, "store", None)
        if store is None or not getattr(store, "uses_default_paths", lambda: False)():
            self._set_status("Snapshot: custom focus store")
            return None
        try:
            from human_focus_tracking import snapshot_human_focus_picks
            from ui.panels.focus_picks_panel import latest_like_origins

            result = snapshot_human_focus_picks(
                focus_maps_by_category=service.all_focus_by_category(),
                like_origins=latest_like_origins(),
                force=force,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Snapshot failed: {exc}")
            return None
        trade_date = result.get("trade_date", "today")
        total = int(result.get("total_for_date") or 0)
        added = int(result.get("added") or 0)
        self._set_status(
            f"Snapshot {trade_date}: {total} pick(s), {added} new."
            if result.get("snapshotted")
            else f"Snapshot {trade_date}: already captured ({total} pick(s))."
        )
        return result

    # --------------------------------------------------------------- internals
    def _add_from_input(self) -> AddResult:
        result = self.add_symbol(
            self.add_input.text(),
            side=self.side_selector.currentText(),
            horizon=self.horizon_selector.currentText(),
        )
        if result.added:
            self.add_input.clear()
        return result

    def _paste_from_clipboard(self) -> AddResult:
        from PySide6.QtWidgets import QApplication

        return self.paste_many(
            QApplication.clipboard().text(),
            side=self.side_selector.currentText(),
            horizon=self.horizon_selector.currentText(),
        )

    def _write_manual(
        self, text: object, *, side: str, horizon: str, pasted: bool
    ) -> AddResult:
        from watchlist_utils import extract_watchlist_symbols

        side_text = str(side or self.side_selector.currentText() or "long").strip().lower()
        side_text = "short" if side_text.startswith("short") else "long"
        horizon_text = (
            watchlist_views.HORIZON_SWING
            if str(horizon or "").strip().lower().startswith("swing")
            else watchlist_views.HORIZON_DAY
        )
        list_name = _list_for(side_text, horizon_text)
        editor = self._editor_for(list_name)
        if editor is None:
            result = AddResult(message="No watchlist editor is wired to this tab.")
            self._set_status(result.message)
            return result

        incoming = extract_watchlist_symbols(str(text or ""))
        if not incoming:
            result = AddResult(message="Nothing to add.")
            self._set_status(result.message)
            return result

        # The file on disk, not the editor's last paint: another writer (the
        # AWAY auto-populate, a Notepad edit, a Focus injection) may have moved
        # it since this desk drew that tab.
        editor.refresh_from_disk()
        existing = list(editor.current_symbols())
        seen = {symbol for symbol in existing}
        added: list[str] = []
        duplicates: list[str] = []
        for symbol in incoming:
            if symbol in seen:
                duplicates.append(symbol)
                continue
            seen.add(symbol)
            added.append(symbol)
        if added:
            editor.add_symbols(added, pasted=pasted)
        self.refresh_now()
        message = (
            f"Added {', '.join(added)}." if added else "Nothing new to add."
        )
        if duplicates:
            message += f" Already there: {', '.join(sorted(set(duplicates)))}."
        self._set_status(message)
        return AddResult(
            added=tuple(added), duplicates=tuple(duplicates), message=message
        )

    def _editor_for(self, list_name: str):
        panel = self._watchlists_panel
        if panel is None:
            return None
        try:
            return panel.editor_for_list(list_name)
        except Exception:  # noqa: BLE001
            return None

    def _remove_row(self, row: watchlist_views.WatchRow) -> bool:
        """Ask the OWNER of each badge. Nothing here removes what it does not own."""
        removed = False
        if watchlist_views.SOURCE_MANUAL in row.sources:
            for horizon in row.horizons or {watchlist_views.HORIZON_DAY}:
                editor = self._editor_for(_list_for(row.side or "long", horizon))
                if editor is None:
                    continue
                try:
                    editor.refresh_from_disk()
                    if row.symbol in editor.current_symbols():
                        editor.remove_symbols(
                            {row.symbol}, reason="removed from the Watchlist tab"
                        )
                        removed = True
                except Exception:  # noqa: BLE001
                    logger.exception("watchlist tab: manual removal failed")
        if row.sources & {
            watchlist_views.SOURCE_FOCUS_TRADER,
            watchlist_views.SOURCE_FOCUS_AUTO,
        }:
            service = self._focus_service
            if service is not None:
                try:
                    removed = bool(service.remove_everywhere(row.symbol)) or removed
                except Exception:  # noqa: BLE001
                    logger.exception("watchlist tab: focus removal failed")
        if watchlist_views.SOURCE_SWING_FAVORITE in row.sources:
            try:
                import swing_favorites
                from project_paths import SWING_FAVORITES_FILE

                written = swing_favorites.record_favorite(
                    symbol=row.symbol,
                    side=row.side or "long",
                    action=swing_favorites.ACTION_REMOVE,
                    path=SWING_FAVORITES_FILE,
                )
                removed = bool(written) or removed
            except Exception:  # noqa: BLE001
                logger.exception("watchlist tab: swing retraction failed")
        return removed

    def _focus_categories(self, row: watchlist_views.WatchRow) -> tuple[str, ...]:
        store = getattr(self._focus_service, "store", None)
        if store is None:
            return ()
        out = []
        for category in ("m5", "swing"):
            try:
                if store.is_focus(row.symbol, row.side or None, category):
                    out.append(category)
            except Exception:  # noqa: BLE001
                continue
        return tuple(out)

    def _annotation_side(self, row: watchlist_views.WatchRow) -> str:
        return "SHORT" if str(row.side).startswith("short") else "LONG"

    def _annotation_timeframe(self, row: watchlist_views.WatchRow) -> str:
        return "D1" if watchlist_views.HORIZON_SWING in row.horizons else "M5"

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"Watchlist: {message}")


# ---------------------------------------------------------------------------
def _list_for(side: str, horizon: str) -> str:
    swing = str(horizon or "").strip().lower().startswith("swing")
    short = str(side or "").strip().lower().startswith("short")
    if swing:
        return "shortswings" if short else "swinglongs"
    return "shorts" if short else "longs"


def _level(value: object) -> float | None:
    text = str(value if value is not None else "").replace("$", "").replace(",", "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number > 0 else None


def _position_text(row: watchlist_views.WatchRow) -> str:
    """One line per account. A stale row says so, and is still shown."""
    if not row.positions:
        return ""
    parts = []
    for position in row.positions:
        head = f"{position.account or position.broker or 'position'} {position.quantity:g}"
        if position.status != watchlist_views.STATUS_OPEN:
            head += f" ({position.status})"
        if position.stale:
            head += " · sync stale"
        parts.append(head)
    return "; ".join(parts)


def _marks_text(row: watchlist_views.WatchRow) -> str:
    marks = []
    if row.liked_today:
        marks.append("★")
    if row.rejected_today:
        marks.append("✕")
    if row.faded:
        marks.append("faded")
    if row.adoption:
        marks.append(row.adoption)
    return " ".join(marks)


def _row_tooltip(row: watchlist_views.WatchRow) -> str:
    lines = [f"{row.symbol} {row.side}".strip()]
    if row.horizon_text:
        lines.append(f"horizon: {row.horizon_text}")
    if row.first_seen:
        lines.append(f"first seen: {row.first_seen}")
    for position in row.positions:
        stamp = position.last_sync.strftime("%Y-%m-%d %H:%M") if position.last_sync else "never"
        lines.append(
            f"{position.broker} {position.account}: {position.quantity:g} "
            f"{position.instrument} ({position.status}), last sync {stamp}"
            + (" - STALE" if position.stale else "")
        )
    return "\n".join(lines)
