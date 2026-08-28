"""G-P2.2: one keyboard route to the Desk Journal (§5.3, decision 10).

The trader could not find the Desk quick-journal on 2026-08-26. It is the SIXTH
tab of the Trading Desk's lower strip (`Alerts | D1 Focus | RS/RW Board | Armed |
Capture | Journal`) and was reachable only by clicking that tab; §6.3 had assumed
otherwise.

The fix had to obey a rule that predates it: the review pane carries AT MOST ONE
slim row between the charts and the tab strip (trader instruction, 2026-08-20).
A second row is therefore rejected, and a verb-row verb spends that one row - so
the route is a keyboard one, which costs no row at all, plus a hint on the tab
label. A mouse route is the trader's to ask for.

Two Qt rules this shares with the capture keys, both paid for once already:

* a `QShortcut` bound inside a hidden tab page NEVER fires - so the binding is at
  PANEL scope with `WidgetWithChildrenShortcut`;
* two live bindings for one sequence is an ambiguous shortcut and Qt fires
  NEITHER - so the sequence must be unbound everywhere else, which the last test
  here checks at source level rather than trusting a reading of the code.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

JOURNAL_SEQUENCE = "Ctrl+J"
PANEL_SOURCE = SCRIPTS_DIR / "ui" / "panels" / "alert_center_panel.py"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app, tmp_path):
    from ui.panels.alert_center_panel import AlertCenterPanel

    made = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    yield made
    if hasattr(made, "shutdown"):
        made.shutdown()


def _journal_shortcuts(panel):
    from PySide6.QtGui import QShortcut

    wanted = QKeySequence(JOURNAL_SEQUENCE)
    return [
        shortcut
        for shortcut in panel.findChildren(QShortcut)
        if shortcut.key() == wanted
    ]


def test_the_route_is_registered_once_at_panel_scope(panel):
    """Once, and on the panel - not on the page, which is hidden when it matters."""
    found = _journal_shortcuts(panel)

    assert len(found) == 1, f"{len(found)} bindings for {JOURNAL_SEQUENCE}; Qt fires NEITHER when there are two"
    shortcut = found[0]
    assert shortcut.parent() is panel
    assert shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut


def test_firing_it_selects_the_journal_tab_and_focuses_the_composer(panel):
    panel.tabs.setCurrentIndex(0)
    assert panel.tabs.currentIndex() != panel._journal_tab_index

    panel._focus_journal_composer()

    assert panel.tabs.currentIndex() == panel._journal_tab_index
    # focusWidget, not hasFocus: an offscreen panel that is never shown has no
    # ACTIVE window, so hasFocus() is False for every widget in it. focusWidget()
    # is what the window would hand the keystrokes to, which is the claim.
    assert panel.focusWidget() is panel._journal_text


def test_the_shortcut_itself_is_wired_to_that_handler(panel):
    """Not just the handler: the key has to reach it."""
    panel.tabs.setCurrentIndex(0)

    _journal_shortcuts(panel)[0].activated.emit()

    assert panel.tabs.currentIndex() == panel._journal_tab_index


def test_the_tab_label_carries_the_hint(panel):
    label = panel.tabs.tabText(panel._journal_tab_index)

    assert "Journal" in label
    assert JOURNAL_SEQUENCE in label


def test_no_other_binding_of_the_same_sequence_exists_anywhere_in_the_ui():
    """Source-level, because an ambiguous shortcut fires NOTHING and is silent.

    Every `QShortcut(...)`/`QKeySequence(...)` under `scripts/ui` is read; the
    sequence may appear exactly once, in the panel that owns the route.
    """
    hits: list[str] = []
    for path in sorted((SCRIPTS_DIR / "ui").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"QKeySequence\(\s*[\"']([^\"']+)[\"']\s*\)", source):
            if match.group(1).strip().lower() == JOURNAL_SEQUENCE.lower():
                hits.append(str(path.relative_to(ROOT_DIR)))

    assert hits == [str(PANEL_SOURCE.relative_to(ROOT_DIR))], hits


def test_the_review_pane_gains_no_second_row():
    """The 2026-08-20 rule: at most ONE slim row under the charts.

    The route was chosen because it costs no row. If a later change spends one
    here, this is the test that has to be argued with.
    """
    source = PANEL_SOURCE.read_text(encoding="utf-8")

    assert "dock_arm_bar=True" in source, "the arm bar stays under the chart"
    assert "dock_capture_rail=False" in source
    assert "_focus_journal_composer" in source
