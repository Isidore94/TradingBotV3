"""Lists update in place; they are not destroyed and rebuilt.

Two costs sat behind the 56% of stalls the 2026-08-21 log could only attribute
to ``app.exec()`` - Qt's own C++ with no Python frame below it:

* ``FocusPicksPanel`` emptied its flow layout and constructed a chip per symbol
  on every refresh (105 on the D1 Focus tab), and every chip ran
  ``setStyleSheet`` in its constructor, which is a CSS parse and a style
  recomputation per widget;
* ``AlertFeedItem`` called ``setStyleSheet`` seven times per row, and the feed
  rebuilds up to ``MAX_FEED_ITEMS`` of them at once.

The teardown half mattered too: 105 QWidget trees dropped per refresh is
exactly the cyclic garbage that the starved collector later had to walk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


def _editor(symbols):
    """A focus side editor backed by a stub service holding ``symbols``.

    ``symbols`` is captured by reference so a test can edit the list in place
    and call refresh() again - which is exactly what the service does when the
    trader adds or removes a name.
    """
    from ui.panels.focus_picks_panel import FocusSideEditor
    from ui.services.focus_service import FocusService

    service = FocusService()
    service.focus_symbols = lambda side, category: list(symbols)
    editor = FocusSideEditor(
        "Longs",
        "long",
        "m5",
        service,
        lambda symbol, side="": {},
        tone="long",
    )
    editor.refresh()
    return editor


def _chips(editor):
    from ui.panels.focus_picks_panel import FocusStatusChip

    return [
        editor.chip_flow.itemAt(index).widget()
        for index in range(editor.chip_flow.count())
        if isinstance(editor.chip_flow.itemAt(index).widget(), FocusStatusChip)
    ]


def test_an_unchanged_list_reuses_every_chip():
    """The common case: a refresh where nothing about the board changed."""
    symbols = ["AAA", "BBB", "CCC"]
    editor = _editor(symbols)
    before = _chips(editor)
    editor.refresh()
    after = _chips(editor)
    assert [chip.symbol for chip in after] == symbols
    assert all(one is two for one, two in zip(before, after)), "chips were rebuilt"


def test_only_the_arrival_is_constructed():
    symbols = ["AAA", "BBB"]
    editor = _editor(symbols)
    before = {chip.symbol: chip for chip in _chips(editor)}
    symbols.append("CCC")
    editor.refresh()
    after = {chip.symbol: chip for chip in _chips(editor)}
    assert set(after) == {"AAA", "BBB", "CCC"}
    assert after["AAA"] is before["AAA"] and after["BBB"] is before["BBB"]


def test_a_departed_symbol_leaves_the_board():
    symbols = ["AAA", "BBB", "CCC"]
    editor = _editor(symbols)
    symbols.remove("BBB")
    editor.refresh()
    assert [chip.symbol for chip in _chips(editor)] == ["AAA", "CCC"]
    assert editor.count_label.text() == "2"


def test_the_service_order_is_what_the_board_shows():
    """The trader reads it as a list, so a reordered service reorders the row."""
    symbols = ["AAA", "BBB", "CCC"]
    editor = _editor(symbols)
    symbols[:] = ["CCC", "AAA", "BBB"]
    editor.refresh()
    assert [chip.symbol for chip in _chips(editor)] == ["CCC", "AAA", "BBB"]


def test_a_chip_restyles_only_when_its_accent_actually_moves():
    """setStyleSheet is the expensive call; state changes that do not change
    the accent must not make it."""
    from ui.panels.focus_picks_panel import FocusStatusChip

    chip = FocusStatusChip("AAA", tone="long", state={})
    calls = {"count": 0}
    original = chip._apply_look

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    chip._apply_look = counting

    chip.update_state({"mover": "open"})
    chip.update_state({"mover": "closed"})
    assert calls["count"] == 0, "a mover flag is not an accent change"

    chip.update_state({"bounce": {"text": "held VWAP", "tone": "long"}})
    assert calls["count"] == 1, "a BOUNCE arriving must restyle"
    chip.update_state({"bounce": {"text": "held VWAP again", "tone": "long"}})
    assert calls["count"] == 1, "same accent, no second parse"


def test_a_chip_shows_the_state_it_was_last_given():
    from ui.panels.focus_picks_panel import FocusStatusChip

    chip = FocusStatusChip("AAA", tone="long", state={})
    assert not chip.live_flag.isVisible() or chip.live_flag.text() == ""

    chip.update_state({"bounce": {"text": "held VWAP", "tone": "long"}})
    assert chip.live_flag.text() == "BOUNCE"
    assert chip.status_labels[0].text() == "held VWAP"

    chip.update_state({})
    assert chip.status_labels[0].isHidden()


def test_an_alert_row_parses_no_stylesheet_of_its_own():
    """Every variant is a theme.qss rule now, selected by name or property."""
    from ui.models.bounce import BounceAlert
    from ui.widgets.alert_feed_item import AlertFeedItem

    alert = BounceAlert(
        time_text="09:30:00", symbol="NVDA", side="LONG", trigger="VWAP reclaim"
    )
    for kwargs in ({}, {"focus_category": "swing"}, {"show_favorite_button": True}):
        item = AlertFeedItem(alert, **kwargs)
        # Badge paints its own tier colour and is out of scope here - it is
        # one widget per row rather than seven, and its colour is per tier
        # rather than per row.
        offenders = [
            child
            for child in item.findChildren(_QT.QWidget)
            if child.styleSheet() and child.metaObject().className() != "Badge"
        ]
        assert not offenders, [child.objectName() for child in offenders]
        assert item.styleSheet() == ""
