"""Veto reason "Hammering": a hammer against the trade's way.

Trader, 2026-09-29: "add a new veto reason for trades. "hammering" which means
the stock is making a hammer against the way we want to go. so for shorts its
a bullish hammer for longs its a bearish ones. this indicates support/resistance
is right there so we cant get into that trade."
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

CODE = "hammering"
HOTKEY = "r"


def _newest_and_previous():
    from ui.annotations.vocabulary import available_veto_versions, load_veto_vocabulary

    versions = available_veto_versions()
    return load_veto_vocabulary(), load_veto_vocabulary(version=versions[-2])


def test_the_newest_vocabulary_has_hammering_with_both_sides_words():
    newest, _ = _newest_and_previous()
    reason = newest.reason(CODE)
    assert reason is not None
    assert reason.hotkey == HOTKEY
    assert reason.note_required is False
    assert reason.label_for("LONG") == "Hammering (bearish hammer)"
    assert reason.label_for("SHORT") == "Hammering (bullish hammer)"
    assert "bearish" in reason.hint_for("LONG") and "resistance" in reason.hint_for("LONG")
    assert "bullish" in reason.hint_for("SHORT") and "support" in reason.hint_for("SHORT")
    assert newest.reasons[-1].code == "other"


def test_the_newest_vocabulary_carries_the_previous_one_byte_identical():
    newest, previous = _newest_and_previous()
    assert previous.reason(CODE) is None
    for reason in previous.reasons:
        assert newest.reason(reason.code) == reason, reason.code
    assert set(newest.codes) - set(previous.codes) == {CODE}
    hotkeys = [reason.hotkey for reason in newest.reasons]
    assert len(hotkeys) == len(set(hotkeys))


def test_a_hammering_veto_round_trips_through_the_store_and_its_own_cohort(tmp_path):
    from ui.annotations import veto_cohort
    from ui.annotations.store import EVENT_VETO, load_annotations, record_annotation

    newest, previous = _newest_and_previous()
    path = tmp_path / "trader_annotations.jsonl"
    row = record_annotation(
        EVENT_VETO, path=path, symbol="ABC", side="SHORT", session_date="2026-09-29", reason_code=CODE
    )
    assert row is not None
    loaded = load_annotations(path, event_types=(EVENT_VETO,))
    assert loaded[-1]["reason_code"] == CODE
    assert loaded[-1]["vocab_version"] == newest.vocab_version

    picks, skipped = veto_cohort.veto_pick_rows(loaded)
    assert skipped == 0
    source = veto_cohort.veto_cohort_source(CODE, newest.vocab_version)
    assert picks[0]["source"] == source
    veto_cohort._canonical_cohort_map.cache_clear()
    try:
        # A new code grades on its own; every carried reason still pools back.
        assert veto_cohort.canonical_veto_cohort(source) == source
        assert veto_cohort.canonical_veto_cohort(veto_cohort.veto_cohort_source(CODE)) == source
        for reason in previous.reasons:
            carried = veto_cohort.veto_cohort_source(reason.code, newest.vocab_version)
            assert veto_cohort.canonical_veto_cohort(carried) == veto_cohort.canonical_veto_cohort(
                veto_cohort.veto_cohort_source(reason.code, previous.vocab_version)
            ), reason.code
    finally:
        veto_cohort._canonical_cohort_map.cache_clear()


@pytest.mark.qt
def test_the_rail_key_picks_hammering_in_the_sides_words(tmp_path):
    qt = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from ui.widgets.capture_rail import CaptureRail

    app = qt.QApplication.instance() or qt.QApplication([])
    rail = CaptureRail(annotations_path=tmp_path / "trader_annotations.jsonl")
    try:
        rail.resize(1030, 540)
        rail.show()
        for _ in range(20):
            app.processEvents()

        def row_text() -> str:
            for row in range(rail.reason_list.count()):
                item = rail.reason_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == CODE:
                    return item.text()
            raise AssertionError("hammering missing from the rail")

        rail.set_context(symbol="ABC", side="SHORT")
        assert row_text() == f"{HOTKEY}  Hammering (bullish hammer)"
        rail.side_input.setCurrentText("LONG")
        assert row_text() == f"{HOTKEY}  Hammering (bearish hammer)"
        rail.reason_list.setFocus()
        app.processEvents()
        QTest.keyClick(rail.reason_list, Qt.Key.Key_R)
        assert rail.selected_reason_code() == CODE
    finally:
        rail.deleteLater()


@dataclass
class _Row:
    symbol: str
    side: str


class _FakePanel:
    def __init__(self) -> None:
        self.recorded: list[tuple] = []

    def _record_dislike(self, row, detail, *, reason_code="", vocab_version=None):
        self.recorded.append((row.symbol, reason_code, vocab_version))


@pytest.mark.qt
@pytest.mark.parametrize(
    ("side", "wording"),
    [("SHORT", "Hammering (bullish hammer)"), ("LONG", "Hammering (bearish hammer)")],
)
def test_the_setups_dislike_dialog_offers_hammering_in_the_rows_words(monkeypatch, side, wording):
    qt = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")
    qt.QApplication.instance() or qt.QApplication([])
    from ui.panels import master_avwap_panel

    offered: list[str] = []

    def fake_get_item(_parent, _title, _prompt, labels, *_args):
        offered.extend(labels)
        return next(label for label in labels if label.endswith(f"[{CODE}]")), True

    monkeypatch.setattr(master_avwap_panel.QInputDialog, "getItem", staticmethod(fake_get_item))
    monkeypatch.setattr(
        master_avwap_panel.QInputDialog, "getMultiLineText", staticmethod(lambda *_a, **_k: ("", True))
    )
    panel = _FakePanel()
    assert master_avwap_panel.MasterAvwapPanel._dislike_row(panel, _Row("ABC", side)) is True
    assert f"{HOTKEY}. {wording} [{CODE}]" in offered
    newest, _ = _newest_and_previous()
    assert panel.recorded == [("ABC", CODE, newest.vocab_version)]
