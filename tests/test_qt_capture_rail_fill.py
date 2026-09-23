"""Capture tab fills its space, and veto v4 offers the trader's three new reasons.

Trader, 2026-09-23: "make the capture tab fill out theres a lot of dead space
at the bottom", then add Bad industry, Too early and Not remotely good.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")

NEW_V4 = {"bad_industry": "q", "too_early": "w", "not_remotely_good": "e"}


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


@pytest.fixture
def rail(tmp_path):
    from ui.widgets.capture_rail import CaptureRail

    widget = CaptureRail(annotations_path=tmp_path / "trader_annotations.jsonl")
    yield widget
    widget.deleteLater()


def _settle(widget, width: int, height: int) -> None:
    widget.resize(width, height)
    widget.show()
    for _ in range(20):
        _QT.QApplication.processEvents()


def test_v4_adds_the_three_reasons_and_keeps_every_v3_reason():
    from ui.annotations.vocabulary import load_veto_vocabulary

    v3 = load_veto_vocabulary(version=3)
    v4 = load_veto_vocabulary(version=4)
    for reason in v3.reasons:
        twin = v4.reason(reason.code)
        assert twin == reason
    for code, hotkey in NEW_V4.items():
        assert v4.reason(code).hotkey == hotkey
        assert v3.reason(code) is None
    assert v4.reasons[-1].code == "other"


def test_the_rail_offers_the_new_reasons_on_their_keys(rail):
    from PySide6.QtCore import Qt

    codes = [
        rail.reason_list.item(row).data(Qt.ItemDataRole.UserRole)
        for row in range(rail.reason_list.count())
    ]
    for code, hotkey in NEW_V4.items():
        assert code in codes
        row = codes.index(code)
        assert rail.reason_list.item(row).text().startswith(f"{hotkey}  ")


def test_a_letter_picks_its_reason_while_the_list_has_focus(rail):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    _settle(rail, 1030, 540)
    rail.reason_list.setFocus()
    _QT.QApplication.processEvents()
    QTest.keyClick(rail.reason_list, Qt.Key.Key_W)
    assert rail.selected_reason_code() == "too_early"


def test_one_line_of_sections_fills_the_whole_height(rail):
    _settle(rail, 2400, 900)
    veto_section = rail.reason_list.parentWidget()
    assert veto_section.geometry().bottom() > 700
    # every reason fits without scrolling once the list is given the height
    assert rail.reason_list.verticalScrollBar().maximum() == 0


def test_a_narrow_host_still_wraps_the_sections(rail):
    _settle(rail, 420, 1400)
    veto = rail.reason_list.parentWidget().geometry()
    like = rail.setup_list.parentWidget().geometry()
    assert like.top() > veto.bottom()
