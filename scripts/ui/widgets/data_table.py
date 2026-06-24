from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QAbstractItemView, QHeaderView, QMenu, QTableView


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

    def add_row_action(self, label: str, callback) -> None:
        self._row_actions.append((label, callback))

    def fit_columns(self) -> None:
        self.resizeColumnsToContents()
        header = self.horizontalHeader()
        for column in range(self.model().columnCount() if self.model() else 0):
            width = max(80, min(header.sectionSize(column), 260))
            header.resizeSection(column, width)
        header.setStretchLastSection(True)

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
