"""FlowLayout never hands out a QWidgetItem whose C++ object Qt already deleted."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("PySide6")
import shiboken6  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QRect  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _board(fill: bool = False):
    from ui.widgets.flow_layout import FlowLayout

    host = QWidget()
    layout = FlowLayout(host, fill=fill)
    labels = [QLabel(text) for text in ("AAPL", "MSFT", "NVDA")]
    for label in labels:
        layout.addWidget(label)
    return host, layout, labels


@pytest.mark.parametrize("fill", [False, True])
def test_a_dead_item_is_dropped_from_every_layout_method(qapp, fill):
    host, layout, labels = _board(fill)
    try:
        dead = layout.itemAt(0)
        shiboken6.delete(dead)
        assert not shiboken6.isValid(dead)

        assert layout.count() == 2
        for index in range(layout.count()):
            assert shiboken6.isValid(layout.itemAt(index))
        assert layout.itemAt(2) is None
        layout.minimumSize()
        layout.sizeHint()
        layout.heightForWidth(300)
        layout.setGeometry(QRect(0, 0, 400, 40))
        taken = layout.takeAt(0)
        assert taken is not None and shiboken6.isValid(taken)
        assert taken.widget() is labels[1]
        assert layout.count() == 1
    finally:
        host.deleteLater()
        qapp.processEvents()


def test_a_dead_item_found_while_laying_out_is_skipped(qapp):
    host, layout, _labels = _board()
    try:
        # Kill an item without going through count/itemAt first.
        shiboken6.delete(layout._items[1])
        layout.setGeometry(QRect(0, 0, 400, 40))
        assert layout.heightForWidth(50) > 0
        assert all(shiboken6.isValid(item) for item in layout._items)
    finally:
        host.deleteLater()
        qapp.processEvents()


def test_a_deleted_child_widget_leaves_no_dead_item(qapp):
    host, layout, labels = _board()
    try:
        labels[0].deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not shiboken6.isValid(labels[0])
        assert layout.count() == 2
        assert all(shiboken6.isValid(layout.itemAt(i)) for i in range(layout.count()))
        layout.setGeometry(QRect(0, 0, 400, 40))
        layout.minimumSize()
    finally:
        host.deleteLater()
        qapp.processEvents()
