"""The shared table shell, and the one implementation of the §12 width rule.

The rule, verbatim from `docs/GUI_REDESIGN_PLAN_2026-08-25.md` §12:

    Tables stretch to the available width. A table never hugs the left edge of
    an otherwise empty page: the widest TEXT column takes the slack, numeric and
    badge columns keep their measured width, and the last section is not the
    only one that stretches.

    Long identifiers elide in the MIDDLE, never at the end, so the
    distinguishing tail survives (`human_f…tracking`, not `human_foc…`) … the
    full value is the tooltip and the head/tail split is deterministic. An
    elision that leaves every row reading the same is a rendering defect.

`apply_width_rule` is module level, not a `DataTable` method, because two of the
pages the rule was learned on - AWAY Recap and Weekend Prep ▸ Focus pick review
- build raw `QTableWidget`s and are not going to become `DataTable`s inside a
presentation packet. A rule applied through a shell that half the tables do not
use is not a rule.

**Presentation only.** Nothing here reads a store, starts a thread, or changes
what a table contains: it sets resize modes, section widths, an item delegate
and tooltips.

Which column takes the slack
----------------------------
A caller that knows names them (`text_columns=`). A caller that does not gets a
deterministic answer measured from the table itself:

* a column whose every non-empty sampled value parses as a number is NUMERIC and
  keeps its measured width - that is the "numeric and badge columns" half;
* of what is left, the one whose measured content is widest takes the slack,
  ties broken by the lowest column index.

That is literally "the widest TEXT column", decided by measurement rather than
by a hand-maintained list per panel - the shape that fell behind three times in
one week for the shutdown child lists.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QMenu,
    QStyledItemDelegate,
    QTableView,
    QTableWidget,
    QToolTip,
)

#: The floor and ceiling `fit_columns` has always applied. Kept: the rule
#: changes which column absorbs the slack, not how wide a measured column may
#: get.
MIN_COLUMN_WIDTH = 80
MAX_COLUMN_WIDTH = 260

#: How many rows the numeric/text classification looks at. Classification only
#: decides which column may stretch, so a sample is enough and a 5,000-row model
#: must never be walked for it.
CLASSIFY_SAMPLE_ROWS = 50

_NUMBER = re.compile(r"^[+-]?[$]?\(?[0-9][0-9,]*(\.[0-9]+)?\)?[%x×]?$")


def looks_numeric(value: str) -> bool:
    """True for the shapes a numeric or badge column holds.

    Deliberately narrow: `12`, `-3.5`, `$1,204`, `(0.42)`, `88%`, `2.3x`. What
    is not one of those is text, and text is what may take the slack.
    """
    text = (value or "").strip()
    if not text:
        return False
    return bool(_NUMBER.match(text))


class MiddleElideDelegate(QStyledItemDelegate):
    """Elides in the MIDDLE and offers the whole value as the tooltip.

    Qt's own elision is end-first, which is what produced `human_foc…` on every
    row of three Weekend Prep tables: an identifier's distinguishing part is its
    TAIL (`human_focus_tracking` vs `human_focus_review`), so an end elision
    renders mutually indistinguishable rows. `QStyleOptionViewItem.textElideMode`
    is per item, so this applies to the identifier column alone and leaves the
    rest of the table reading normally.

    The tooltip is served from here rather than written into every cell so that
    model-backed views get it without their models growing a `ToolTipRole`; a
    model that already supplies one wins.
    """

    def initStyleOption(self, option, index) -> None:  # noqa: N802 - Qt override
        super().initStyleOption(option, index)
        option.textElideMode = Qt.TextElideMode.ElideMiddle

    def helpEvent(self, event, view, option, index) -> bool:  # noqa: N802 - Qt override
        if event is not None and event.type() == QEvent.Type.ToolTip and index.isValid():
            existing = index.data(Qt.ItemDataRole.ToolTipRole)
            full = existing if existing else index.data(Qt.ItemDataRole.DisplayRole)
            if full:
                QToolTip.showText(event.globalPos(), str(full), view)
                return True
        return super().helpEvent(event, view, option, index)


def elide_middle(text: str, metrics, width: int) -> str:
    """The head/tail split, in one place so a test can assert on it.

    Deterministic for a given font and width, and the tail always survives -
    the two properties §12 asks for and the two a caller can check.
    """
    return metrics.elidedText(str(text or ""), Qt.TextElideMode.ElideMiddle, int(width))


def _row_count(view) -> int:
    model = view.model()
    return 0 if model is None else int(model.rowCount())


def _column_count(view) -> int:
    model = view.model()
    return 0 if model is None else int(model.columnCount())


def _display(view, row: int, column: int) -> str:
    model = view.model()
    if model is None:
        return ""
    value = model.data(model.index(row, column), Qt.ItemDataRole.DisplayRole)
    return "" if value is None else str(value)


def classify_columns(view, sample_rows: int = CLASSIFY_SAMPLE_ROWS) -> list[bool]:
    """One bool per column: True where the column reads as text.

    A column with nothing in it counts as text: a column that is empty today and
    fills with identifiers tomorrow must not be frozen narrow by its emptiness.
    """
    columns = _column_count(view)
    rows = min(_row_count(view), max(0, int(sample_rows)))
    verdicts: list[bool] = []
    for column in range(columns):
        seen = 0
        numeric = 0
        for row in range(rows):
            value = _display(view, row, column).strip()
            if not value:
                continue
            seen += 1
            if looks_numeric(value):
                numeric += 1
        verdicts.append(not seen or numeric != seen)
    return verdicts


def measure_column_widths(view) -> list[int]:
    """What each column wants, before the rule decides who gets the slack.

    `resizeColumnsToContents` is what `fit_columns` has always used. It is also
    the 7.9% / 115 s site of the 2026-08-26 measurement (worst stall 23.9 s), so
    this is the single place a bounded measurement replaces it - every caller
    goes through here.
    """
    view.resizeColumnsToContents()
    header = view.horizontalHeader()
    return [int(header.sectionSize(column)) for column in range(_column_count(view))]


def _widest_text_column(verdicts: list[bool], widths: list[int]):
    best = None
    for column, is_text in enumerate(verdicts):
        if not is_text:
            continue
        if best is None or widths[column] > widths[best]:
            best = column
    return best


def apply_width_rule(
    view,
    *,
    text_columns=None,
    elide_columns=(),
    min_width: int = MIN_COLUMN_WIDTH,
    max_width: int = MAX_COLUMN_WIDTH,
    sample_rows: int = CLASSIFY_SAMPLE_ROWS,
) -> None:
    """Apply §12 to one `QTableView`/`QTableWidget`.

    `text_columns` names the columns allowed to take the slack; omit it and the
    widest measured text column is chosen. `elide_columns` gets the middle
    elision - identifier columns, whose tail carries the identity.

    Safe on an empty table and safe to call repeatedly: it is the same call
    `fit_columns` already was.
    """
    columns = _column_count(view)
    header = view.horizontalHeader()
    header.setStretchLastSection(False)
    if columns <= 0:
        return

    widths = measure_column_widths(view)
    named = None
    if text_columns:
        named = [int(column) for column in text_columns if 0 <= int(column) < columns]
    if named:
        stretching = set(named)
    else:
        widest = _widest_text_column(classify_columns(view, sample_rows), widths)
        stretching = set() if widest is None else {widest}

    for column in range(columns):
        if column in stretching:
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
            continue
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(column, max(min_width, min(widths[column], max_width)))

    # The last section stretches only when nothing else can. Qt's
    # `stretchLastSection` alone is what pinned every other column narrow while
    # the right-hand third of a 4K window sat empty.
    if not stretching:
        header.setStretchLastSection(True)

    for column in elide_columns:
        column = int(column)
        if 0 <= column < columns:
            view.setItemDelegateForColumn(column, _shared_elide_delegate(view))


def _shared_elide_delegate(view) -> MiddleElideDelegate:
    """One delegate per view, parented to it - never one per call.

    `apply_width_rule` runs on every refresh; a fresh delegate per refresh would
    leave one object per refresh behind and re-parent the column each time.
    """
    delegate = getattr(view, "_middle_elide_delegate", None)
    if delegate is None:
        delegate = MiddleElideDelegate(view)
        view._middle_elide_delegate = delegate
    return delegate


def apply_width_rule_to_table_widget(
    table: QTableWidget,
    *,
    text_columns=None,
    elide_columns=(),
    **kwargs,
) -> None:
    """The same rule for a raw `QTableWidget`, plus the full-value tooltips.

    A `QTableWidget` owns its items, so the full value is attached directly and
    survives wherever a tooltip event never reaches the delegate.
    """
    for column in elide_columns:
        column = int(column)
        if column < 0 or column >= table.columnCount():
            continue
        for row in range(table.rowCount()):
            item = table.item(row, column)
            if item is not None and not item.toolTip():
                item.setToolTip(item.text())
    apply_width_rule(
        table, text_columns=text_columns, elide_columns=elide_columns, **kwargs
    )


class DataTable(QTableView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setWordWrap(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        copy_action = QAction("Copy Selection", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self.copy_selection)
        self.addAction(copy_action)

        # Extra per-row context-menu actions; each callback receives the clicked
        # (proxy) QModelIndex. Used e.g. for "Add to Focus".
        self._row_actions: list[tuple[str, object]] = []
        # §12: which columns may take the slack and which carry an identifier in
        # their tail. Empty means "measure it" - see `apply_width_rule`.
        self._text_columns: list[int] | None = None
        self._elide_columns: tuple[int, ...] = ()

    def add_row_action(self, label: str, callback) -> None:
        self._row_actions.append((label, callback))

    def set_width_rule(self, *, text_columns=None, elide_columns=()) -> None:
        """Declare the §12 roles for this table. Applied by `fit_columns`."""
        self._text_columns = None if text_columns is None else [int(c) for c in text_columns]
        self._elide_columns = tuple(int(c) for c in elide_columns)

    def fit_columns(self) -> None:
        apply_width_rule(
            self,
            text_columns=self._text_columns,
            elide_columns=self._elide_columns,
        )

    def copy_selection(self) -> None:
        model = self.model()
        if model is None:
            return
        indexes = sorted(self.selectedIndexes(), key=lambda item: (item.row(), item.column()))
        if not indexes:
            return

        rows: dict[int, list[str]] = {}
        for index in indexes:
            rows.setdefault(index.row(), []).append(str(model.data(index, Qt.ItemDataRole.DisplayRole) or ""))
        text = "\n".join("\t".join(values) for _, values in sorted(rows.items()))
        QApplication.clipboard().setText(text)

    def _show_context_menu(self, point) -> None:
        menu = QMenu(self)
        index = self.indexAt(point)
        if self._row_actions and index.isValid():
            for label, callback in self._row_actions:
                action = menu.addAction(label)
                action.triggered.connect(
                    lambda _checked=False, cb=callback, idx=index: cb(idx)
                )
            menu.addSeparator()
        copy_action = menu.addAction("Copy Selection")
        copy_action.triggered.connect(self.copy_selection)
        menu.exec(self.viewport().mapToGlobal(point))
