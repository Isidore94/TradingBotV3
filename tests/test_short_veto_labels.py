"""Veto reasons read inverted on a SHORT chart.

Trader, 2026-09-29: "invert the rules for shorts. SMA incoming and horizontal
overhead should mean the same thing but inverted." Codes stay the same so the
forward record pools; only the words shown for a SHORT change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

INVERTED = ("sma_incoming", "overhead_horizontal", "incoming_trendline", "too_extended_from_base")


def test_the_newest_vocabulary_words_the_level_reasons_for_shorts():
    from ui.annotations.vocabulary import load_veto_vocabulary

    vocabulary = load_veto_vocabulary()
    for code in INVERTED:
        reason = vocabulary.reason(code)
        assert reason.label_for("SHORT") != reason.label_for("LONG"), code
        assert reason.hint_for("SHORT") != reason.hint_for("LONG"), code
    assert "below" in vocabulary.reason("overhead_horizontal").label_for("SHORT").lower()
    assert "below" in vocabulary.reason("sma_incoming").label_for("SHORT").lower()


def test_long_words_and_every_code_and_hotkey_carry_over_unchanged():
    from ui.annotations.vocabulary import available_veto_versions, load_veto_vocabulary

    versions = available_veto_versions()
    newest = load_veto_vocabulary()
    previous = load_veto_vocabulary(version=versions[-2])
    # A later bump may add a reason (v6 added hammering); it never drops one.
    assert set(previous.codes) <= set(newest.codes)
    for reason in previous.reasons:
        twin = newest.reason(reason.code)
        assert (twin.label, twin.hint, twin.hotkey, twin.note_required) == (
            reason.label,
            reason.hint,
            reason.hotkey,
            reason.note_required,
        )
        assert twin.label_for("LONG") == reason.label


def test_a_reason_without_short_words_shows_its_long_words(tmp_path):
    from ui.annotations.vocabulary import clear_vocabulary_cache, load_veto_vocabulary

    payload = {
        "vocabulary_id": "veto_reasons",
        "vocab_version": 1,
        "reasons": [
            {"code": "compressed", "label": "Compressed", "hotkey": "3", "note_required": False, "hint": "tight"},
        ],
    }
    (tmp_path / "veto_reasons_v1.json").write_text(json.dumps(payload), encoding="utf-8")
    clear_vocabulary_cache()
    reason = load_veto_vocabulary(directory=tmp_path).reason("compressed")
    assert reason.label_for("SHORT") == "Compressed"
    assert reason.hint_for("SHORT") == "tight"
    clear_vocabulary_cache()


def test_a_non_text_short_label_fails_closed(tmp_path):
    from ui.annotations.vocabulary import VocabularyError, clear_vocabulary_cache, load_veto_vocabulary

    payload = {
        "vocabulary_id": "veto_reasons",
        "vocab_version": 1,
        "reasons": [
            {"code": "compressed", "label": "Compressed", "hotkey": "3", "note_required": False, "short_label": 5},
        ],
    }
    (tmp_path / "veto_reasons_v1.json").write_text(json.dumps(payload), encoding="utf-8")
    clear_vocabulary_cache()
    with pytest.raises(VocabularyError):
        load_veto_vocabulary(directory=tmp_path)
    clear_vocabulary_cache()


def test_the_newest_version_pools_with_the_reasons_it_carries():
    from ui.annotations import veto_cohort
    from ui.annotations.vocabulary import available_veto_versions

    newest = available_veto_versions()[-1]
    veto_cohort._canonical_cohort_map.cache_clear()
    try:
        for code in INVERTED:
            source = veto_cohort.veto_cohort_source(code, newest)
            assert veto_cohort.canonical_veto_cohort(source) != source, code
    finally:
        veto_cohort._canonical_cohort_map.cache_clear()


@pytest.mark.qt
def test_the_rail_relabels_the_veto_list_when_the_side_flips(tmp_path):
    qt = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")
    from PySide6.QtCore import Qt

    from ui.annotations.vocabulary import load_veto_vocabulary
    from ui.widgets.capture_rail import CaptureRail

    qt.QApplication.instance() or qt.QApplication([])
    rail = CaptureRail(annotations_path=tmp_path / "trader_annotations.jsonl")
    try:
        reason = load_veto_vocabulary().reason("overhead_horizontal")

        def row_text() -> str:
            for row in range(rail.reason_list.count()):
                item = rail.reason_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == "overhead_horizontal":
                    return item.text()
            raise AssertionError("reason missing")

        rail.set_context(symbol="ABC", side="SHORT")
        assert row_text() == f"{reason.hotkey}  {reason.label_for('SHORT')}"
        rail.side_input.setCurrentText("LONG")
        assert row_text() == f"{reason.hotkey}  {reason.label}"
        rail.side_input.setCurrentText("SHORT")
        assert row_text() == f"{reason.hotkey}  {reason.label_for('SHORT')}"
    finally:
        rail.deleteLater()
