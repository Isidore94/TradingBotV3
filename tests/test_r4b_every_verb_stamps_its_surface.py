"""R4 Part B item B5 - every capture verb writes the screen it came from.

The V3 item-4 test asserted on the SOURCE TEXT of `_record`
(`'common.setdefault("surface", self._surface)' in record`). That proves a line
exists in one method; it proves nothing about the verbs that never reach that
method, and `commit_pass` was exactly such a verb - it called
`record_pass_annotation` directly, so every day-trade pass landed with no
`surface` and no `scan_context` while the veto, the like and the note beside it
carried both. A rollup by screen read as "the trader never passes from the chart".

So this file asks the question behaviourally, once per verb: perform the real
click handler on a real rail bound to a temp file, then read the row off disk.
A test that reads the written row cannot be satisfied by a line of source that
does not run.

Offline: every write goes to `tmp_path`. The live annotation log is never opened.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6", reason="the capture rail is a Qt widget")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def rail(app, tmp_path, monkeypatch):
    """A rail on a temp log, told it is serving the chart-review screen.

    `chart_review` rather than the `rail` default precisely because the default
    would pass by accident: a verb that stamps nothing and a verb that stamps the
    default are indistinguishable when the default is what you assert.
    """
    from ui.annotations import verdicts
    from ui.widgets.capture_rail import CaptureRail

    widget = CaptureRail(annotations_path=tmp_path / "trader_annotations.jsonl")
    # Real `SCAN_CONTEXT_FIELDS` keys: `build_annotation` FLATTENS the context
    # onto the row under those names rather than nesting it, so asserting on a
    # made-up key would fail for the wrong reason.
    widget.set_scan_context(
        {"scan_date": "2026-09-01", "tracker_setup_id": "T-42"},
        surface=verdicts.SURFACE_CHART_REVIEW,
    )
    widget.set_context(symbol="NVDA", side="LONG", last_price=100.0, timeframe="M5")
    # The cohort merges read and write CSVs under the home folder; this test is
    # about the annotation row, and the merge already degrades to a status
    # suffix by design.
    monkeypatch.setattr(widget, "_merge_cohort_safely", lambda merge: "")
    monkeypatch.setattr(widget, "_merge_pass_cohort_safely", lambda: "")
    monkeypatch.setattr(widget, "_merge_like_cohort_safely", lambda: "")
    yield widget, tmp_path / "trader_annotations.jsonl"
    widget.deleteLater()


def _assert_stamped(row: dict, event_type: str) -> None:
    assert row["event_type"] == event_type
    assert row.get("surface") == "chart_review", (
        f"{event_type} wrote no surface, so a rollup by screen drops it"
    )
    assert row.get("tracker_setup_id") == "T-42", (
        f"{event_type} wrote no scan_context, so the row it judged is unknowable"
    )
    assert row.get("scan_date") == "2026-09-01", event_type


def test_a_veto_carries_the_screen_and_the_scanner_row(rail):
    widget, path = rail
    widget.reason_list.setCurrentRow(0)
    assert widget.selected_reason_code(), "the vocabulary gave the rail no reasons"
    widget.veto_note_input.setText("because")
    assert widget.commit_veto() is not None, widget.status_text()

    rows = _rows(path)
    assert len(rows) == 1
    _assert_stamped(rows[0], "veto")


def test_a_claimed_like_carries_the_screen_and_the_scanner_row(rail):
    widget, path = rail
    widget.setup_list.setCurrentRow(0)
    assert widget.selected_setup_id(), "the claim picklist gave the rail no setups"
    widget.like_note_input.setText("held the level twice")
    assert widget.commit_like() is not None, widget.status_text()

    rows = _rows(path)
    assert len(rows) == 1
    _assert_stamped(rows[0], "like_claim")


def test_a_quick_like_carries_the_screen_and_the_scanner_row(rail):
    """Alt+L: no claim, no why - and still the screen it came from."""
    widget, path = rail
    assert widget.commit_quick_like() is not None, widget.status_text()

    rows = _rows(path)
    assert len(rows) == 1
    _assert_stamped(rows[0], "like_claim")
    assert rows[0].get("like_mode") == "quick"


def test_a_day_trade_pass_carries_the_screen_and_the_scanner_row(rail):
    """B5's defect. This verb bypassed `_record` entirely."""
    widget, path = rail
    if widget._pass_vocabulary is None:
        pytest.skip(f"pass vocabulary unavailable: {widget._pass_vocabulary_error}")
    code = next(iter(widget.pass_checkboxes))
    widget.toggle_pass_reason(code)
    assert widget.selected_pass_codes(), "the pass vocabulary gave the rail no reasons"
    widget.note_input.setText("liked it, one issue")
    assert widget.commit_pass() is not None, widget.status_text()

    rows = _rows(path)
    assert len(rows) == 1
    _assert_stamped(rows[0], "pass")
    assert rows[0].get("reason_codes"), "a pass without its codes is not a pass"


def test_a_note_carries_the_screen_and_the_scanner_row(rail):
    widget, path = rail
    widget.note_input.setText("watching the retest")
    assert widget.commit_note() is not None, widget.status_text()

    rows = _rows(path)
    assert len(rows) == 1
    _assert_stamped(rows[0], "note")
