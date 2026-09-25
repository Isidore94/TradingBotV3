"""P2-11e: Focus board chips parse no stylesheet of their own; theme.qss has the variants."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_app = QApplication.instance() or QApplication([])

from ui import theme  # noqa: E402
from ui.panels.focus_picks_panel import FocusStatusChip  # noqa: E402


def _own_stylesheets(chip):
    widgets = [chip, *chip.findChildren(QWidget)]
    return [w.objectName() or type(w).__name__ for w in widgets if w.styleSheet()]


def test_a_chip_through_every_look_sets_no_stylesheet_of_its_own():
    chip = FocusStatusChip("NVDA", tone="short", state={})
    for state in (
        {},
        {"bounce": {"text": "held VWAP", "tone": "short"}},
        {"rrs": {"text": "RW -1.1", "tone": "short"}},
        {"rrs": {"text": "RS +1.1", "tone": "long"}},
        {"mover": "open"},
    ):
        chip.update_state(state)
    assert _own_stylesheets(chip) == []
    assert chip.property("chipSide") == "short"
    assert chip.property("chipLook") == "plain-short"


def test_every_chip_variant_is_in_the_rendered_theme():
    sheet = theme.build_stylesheet("dark")
    assert "@focus_chip" not in sheet, "an unreplaced token"
    for look in ("plain-long", "plain-short", "rrs-long", "rrs-short", "bounce"):
        assert f'QFrame#FocusStatusChip[chipLook="{look}"]' in sheet or look == "plain-long"
    for selector in (
        "QLabel#FocusChipTitle",
        "QLabel#FocusMovingFlag",
        'QLabel#FocusLiveFlag[liveTone="favorite"]',
        'QLabel#SectionTitle[accent="favorite"]',
    ):
        assert selector in sheet


def test_the_title_takes_its_side_colour_from_the_theme():
    previous = _app.styleSheet()
    _app.setStyleSheet(theme.build_stylesheet("dark"))
    try:
        chip = FocusStatusChip("NVDA", tone="short", state={})
        chip.ensurePolished()
        chip.title.ensurePolished()
        colour = chip.title.palette().color(QPalette.ColorRole.WindowText).name().lower()
        assert colour == theme.color("short", "dark").lower()
    finally:
        _app.setStyleSheet(previous)
