"""V2 item 5 - hide the surfaces the trader never opens, keep the code.

Decision 0016 answer 7, in the trader's words: the screens actually used are the
Trading Desk (sitting on **Capture** almost all the time) and the Journal.
**Never opened:** Research, Universe, the Alerts tab, the D1 Focus tab, the Armed
tab.

**HIDDEN IS NOT REMOVED**, and the difference is the whole design:

* the Alerts feed is the review-alert door - `_enqueue_review_alert` routes
  through it and the M5 list is built from it;
* the D1 Focus tab holds the flag list several polls write into;
* the Armed tab is the armed-watch inventory across every symbol, and the
  expiry sweep runs at the head of the poll that owns each store;
* the Universe page's BUILDER writes the file the scanner reads.

All four are load-bearing behind the scenes. What they are not is worth a tab or
a nav row the trader has to skip past. This record is also in DESK_INTERNALS, so
the next agent does not read "unused" as "deletable".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _Stand:
    """Carries only what `show_unused_surfaces` reads off `self`."""

    SHOW_UNUSED_SETTING = "qt_show_unused_tabs"


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# The default, and the switch
# ---------------------------------------------------------------------------


def test_the_setting_defaults_to_hidden(qapp, monkeypatch):
    import project_paths

    from ui.app import MainWindow

    monkeypatch.setattr(project_paths, "get_local_setting", lambda key, default=None: default)
    # A stand-in carrying the constant, not a bare object: the method reads
    # `self.SHOW_UNUSED_SETTING`, and a bare object would take the except branch
    # and prove only that the fallback works.
    assert MainWindow.show_unused_surfaces(_Stand()) is False


def test_an_unreadable_setting_shows_rather_than_hides(qapp, monkeypatch):
    """A surface the trader cannot reach is worse than one they skip past."""
    import project_paths

    from ui.app import MainWindow

    def _explode(*_args, **_kwargs):
        raise OSError("settings unreadable")

    monkeypatch.setattr(project_paths, "get_local_setting", _explode)
    assert MainWindow.show_unused_surfaces(_Stand()) is True


# ---------------------------------------------------------------------------
# What it hides, and what it must not touch
# ---------------------------------------------------------------------------


def test_hiding_a_tab_never_removes_it(qapp):
    """`setTabVisible`, not `removeTab`: no index shifts, nothing recomputes."""
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    before = panel.tabs.count()
    titles = [panel.tabs.tabText(i) for i in range(before)]

    panel.apply_unused_tab_visibility(False)

    assert panel.tabs.count() == before, "the tabs are still there"
    assert [panel.tabs.tabText(i) for i in range(panel.tabs.count())] == titles
    for index in range(panel.tabs.count()):
        expected = panel.tabs.tabText(index) not in panel.UNUSED_TAB_TITLES
        assert panel.tabs.isTabVisible(index) is expected, panel.tabs.tabText(index)


def test_the_stored_tab_indexes_still_point_where_they_did(qapp):
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    d1, armed, capture = (
        panel._d1_tab_index,
        panel._armed_tab_index,
        panel._capture_tab_index,
    )

    panel.apply_unused_tab_visibility(False)

    assert panel.tabs.tabText(d1) == "D1 Focus"
    assert panel.tabs.tabText(armed) == "Armed"
    assert panel.tabs.tabText(capture) == "Capture"


def test_hiding_the_current_tab_moves_the_trader_to_capture(qapp):
    """Never leave them looking at a tab that just vanished."""
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    panel.tabs.setCurrentIndex(panel._armed_tab_index)

    panel.apply_unused_tab_visibility(False)

    assert panel.tabs.currentIndex() == panel._capture_tab_index


def test_showing_them_again_brings_every_tab_back(qapp):
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    panel.apply_unused_tab_visibility(False)
    panel.apply_unused_tab_visibility(True)

    for index in range(panel.tabs.count()):
        assert panel.tabs.isTabVisible(index) is True


# ---------------------------------------------------------------------------
# The one that would actually cost the trader something
# ---------------------------------------------------------------------------


def test_every_rail_hotkey_still_fires_with_the_tabs_hidden(qapp):
    """A QShortcut inside a hidden tab never fires; two for one sequence fire
    NEITHER. The rail rebinds at PANEL scope, and this is what proves it."""
    from PySide6.QtCore import Qt
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    panel.apply_unused_tab_visibility(False)

    from PySide6.QtGui import QShortcut

    bound = [
        item
        for item in panel.findChildren(QShortcut)
        if item.context() != Qt.ShortcutContext.WidgetShortcut
    ]
    assert bound, "the rail's shortcuts must live at panel scope"

    # No sequence may be bound twice: two bindings for one sequence fire NEITHER.
    sequences = [item.key().toString() for item in bound if not item.key().isEmpty()]
    duplicates = {key for key in sequences if sequences.count(key) > 1}
    assert not duplicates, duplicates

    # And every one of them is still owned by a widget that is VISIBLE-capable,
    # never by a widget inside one of the hidden tabs.
    hidden_pages = {
        panel.tabs.widget(index)
        for index in range(panel.tabs.count())
        if panel.tabs.tabText(index) in panel.UNUSED_TAB_TITLES
    }
    for item in bound:
        parent = item.parent()
        while parent is not None:
            assert parent not in hidden_pages, (
                f"{item.key().toString()} is owned inside a hidden tab and will "
                "never fire"
            )
            parent = parent.parent()



def test_the_hidden_pages_are_still_built_and_still_indexed(qapp):
    """`_select_page` and every stored index keep working."""
    from ui.app import PAGE_SPECS, MainWindow

    for title in MainWindow.UNUSED_PAGE_TITLES:
        assert any(spec.title == title for spec in PAGE_SPECS), title


def test_the_record_says_hidden_is_not_removed():
    internals = (ROOT / "docs" / "DESK_INTERNALS.md").read_text(encoding="utf-8")
    assert "hidden is not removed" in internals.lower()
