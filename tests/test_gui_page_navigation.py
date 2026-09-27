"""The compact page bar shows the current destination and supports keyboard use."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest.importorskip("PySide6")
pytestmark = pytest.mark.qt

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.widgets.page_tab_row import MORE_LABEL, PageTabRow  # noqa: E402


def test_secondary_page_name_tracks_selection_and_badge_without_navigation():
    app = QApplication.instance() or QApplication([])
    row = PageTabRow(["Trading Desk", "Journal", "Weekend Prep", "Settings"])
    requests = []
    row.pageRequested.connect(requests.append)
    try:
        row.set_current(2)
        assert row.more_button.text() == "Weekend Prep ▾"
        row.set_label(2, "Weekend Prep (2 to review)")
        assert row.more_button.text() == "Weekend Prep (2 to review) ▾"
        assert requests == []
        row.set_current(0)
        assert row.more_button.text() == MORE_LABEL
        assert not row.more_button.isChecked()
        assert sorted(row.page_indices()) == [0, 1, 2, 3]
    finally:
        row.deleteLater()
        app.processEvents()


def test_page_tabs_accept_keyboard_focus_and_request_once():
    app = QApplication.instance() or QApplication([])
    row = PageTabRow(["Trading Desk", "Journal", "Settings"])
    requests = []
    row.pageRequested.connect(requests.append)
    try:
        row.set_current(0)
        row.show()
        journal = row.tab_buttons[1]
        assert journal.focusPolicy() == Qt.TabFocus
        assert row.more_button.focusPolicy() == Qt.TabFocus
        journal.setFocus()
        app.processEvents()
        QTest.keyClick(journal, Qt.Key_Space)
        assert requests == [1]
        assert row.tab_buttons[0].isChecked(), "the window still owns navigation"
        assert not journal.isChecked()
    finally:
        row.close()
        row.deleteLater()
        app.processEvents()


def test_more_groups_keep_each_page_once_and_hide_empty_groups():
    app = QApplication.instance() or QApplication([])
    row = PageTabRow(["Trading Desk", "Weekend Prep", "Universe", "Settings", "New page"])
    try:
        assert [section.text() for section, _ in row._more_sections] == [
            "Review", "Explore", "System", "Other pages"
        ]
        actions = [action for _, members in row._more_sections for action in members]
        assert actions == list(row.more_actions.values())
        row.set_page_visible(2, False)
        assert not row._more_sections[1][0].isVisible()
        assert row._more_sections[2][0].isVisible()
        row.set_page_visible(2, True)
        assert row._more_sections[1][0].isVisible()
    finally:
        row.deleteLater()
        app.processEvents()
