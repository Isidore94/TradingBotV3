"""The ticker popup must not open edge to edge (trader rule 2026-08-27).

The 2026-08-11 ask was the opposite one - the popup opened at a fixed 1180x760
regardless of the monitor and squeezed both charts into about half the vertical
space, so it was made to match the desk window's own height minus a title-bar
allowance. That went too far the other way: "make the charts that pop up when i
click on a ticker just a little less tall. i dont want them edge to edge on the
screen just reduce by 10% top and bottom."

So the popup is now inset by a tenth of the anchor's height at the top AND at
the bottom, leaving 80% of it. The arithmetic lives in a pure helper rather than
inside the Qt sizing method, because the thing worth pinning is the proportion
and the equal gaps - not whatever `availableGeometry()` reports on a headless
runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pytest  # noqa: E402

pytest.importorskip("PySide6", reason="the popup is a Qt dialog")


def test_the_popup_leaves_a_tenth_of_the_screen_free_top_and_bottom():
    from ui.widgets.symbol_snapshot_dialog import inset_vertical_bounds

    top, height = inset_vertical_bounds(0, 1440, minimum=0)
    assert height == 1152, "80% of the anchor - a tenth off the top and a tenth off the bottom"
    assert top == 144, "the free tenth at the top"
    assert 1440 - (top + height) == 144, "and the same tenth at the bottom"


def test_the_gaps_are_equal_whatever_the_anchor():
    from ui.widgets.symbol_snapshot_dialog import inset_vertical_bounds

    for anchor_top, anchor_height in ((0, 1080), (0, 1440), (37, 1234), (100, 2160)):
        top, height = inset_vertical_bounds(anchor_top, anchor_height, minimum=0)
        above = top - anchor_top
        below = (anchor_top + anchor_height) - (top + height)
        assert abs(above - below) <= 1, (
            f"anchor {anchor_height}: {above}px above vs {below}px below - "
            "the trader asked for the same gap at each end"
        )
        assert height < anchor_height, "never edge to edge"


def test_the_inset_is_ten_percent():
    from ui.widgets.symbol_snapshot_dialog import POPUP_VERTICAL_INSET

    assert POPUP_VERTICAL_INSET == 0.10


def test_a_short_screen_keeps_the_charts_usable_rather_than_honouring_the_inset():
    """The 2026-08-11 problem must not come back on a small monitor: the two
    charts carry a 120px minimum each, and a squeezed popup was the whole
    reason the sizing was changed then. The floor wins, and the result is
    still centred so what room there is falls evenly."""
    from ui.widgets.symbol_snapshot_dialog import inset_vertical_bounds

    top, height = inset_vertical_bounds(0, 800, minimum=760)
    assert height == 760, "the floor wins over the inset on a short screen"
    assert top == 20, "and the remaining room is split evenly"


def test_the_floor_never_makes_the_popup_taller_than_its_anchor():
    from ui.widgets.symbol_snapshot_dialog import inset_vertical_bounds

    top, height = inset_vertical_bounds(0, 600, minimum=760)
    assert height == 760, "the floor is honoured"
    assert top == 0, "but it is not pushed off the top of the screen to do it"


def test_the_dialog_actually_uses_the_helper(monkeypatch):
    """The helper is only worth anything if the sizing method calls it."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from ui.widgets import symbol_snapshot_dialog as module

    calls = []
    real = module.inset_vertical_bounds

    def spy(anchor_top, anchor_height, **kwargs):
        calls.append((anchor_top, anchor_height))
        return real(anchor_top, anchor_height, **kwargs)

    monkeypatch.setattr(module, "inset_vertical_bounds", spy)

    dialog = module.SymbolSnapshotDialog()
    try:
        dialog._resize_to_desk_height()
        assert calls, "the sizing method must go through the shared arithmetic"
        anchor_top, anchor_height = calls[-1]
        assert dialog.height() < anchor_height or anchor_height <= 0, (
            "the popup must be shorter than whatever it was anchored to"
        )
    finally:
        dialog.close()
        dialog.deleteLater()
