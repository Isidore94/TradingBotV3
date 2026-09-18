"""TJ-1 item 6 - move, not delete: two blocks leave the Daily Recap intact.

The Day Review page holds the six sections of plan.md §12.2 and nothing else, so
the two parts of the Daily Recap that are NOT about one trading day go where they
belong rather than being deleted:

(a) the **Review** tab - the published measured report's cells, the Next test card
    and Copy/Export - becomes a **Measured report** section on
    **Research > Results**, driven by that page's own `ReadWorker`. Same cells,
    same `report_id`, same parity seam (`review_cells()`), and
    `entryQualityProposalChanged` travels with it.
(b) the **Staged picks** table and "Add selected staged pick to Focus" move to the
    **Auto Pilot** page, under its log, keeping the `focusAddRequested` signal so
    `MainWindow` performs the add through the store's own owner.

Neither move may change what the widgets DO: a staged pick is still only ASKED
for (the R2 adoption gate is SHOWN, never enforced - it governs the machine's
adoptions, never the trader's), and the report section still computes nothing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="both pages are Qt panels")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QTableWidget  # noqa: E402

SESSION = "2026-09-10"
REPORT_ID = "mr-2026-09-10-abc123"

#: A published measured report, by hand. Two cells, one measured and one not, so
#: the section has to print a state rather than a number for the second.
REPORT_PAYLOAD = {
    "session_date": SESSION,
    "as_of": f"{SESSION}T21:05:00-07:00",
    "report_id": REPORT_ID,
    "generated_at": f"{SESSION}T21:05:00-07:00",
    "cells": [
        {
            "cell_id": "day_quick_median_mfe_r",
            "metric": "median MFE R",
            "unit": "R",
            "value": 0.85,
            "n": 41,
            "distinct_sessions": 12,
            "distinct_symbols": 30,
            "population": "bot_day",
            "window": (SESSION, SESSION),
            "reference_clock": "America/New_York",
            "exit_policy": "last_measured",
            "state": "measured",
            "sources": ("intraday_bounce_outcomes.csv",),
            "section": "measured_context",
        },
        {
            "cell_id": "swing_win_rate_5s",
            "metric": "win rate",
            "unit": "%",
            "value": None,
            "n": 0,
            "distinct_sessions": 0,
            "distinct_symbols": 0,
            "population": "bot_swing",
            "window": (SESSION, SESSION),
            "reference_clock": "America/New_York",
            "exit_policy": "eod_hold",
            "state": "unknown",
            "sources": (),
            "unavailable": "the warehouse is not configured on this machine",
            "section": "measured_context",
        },
    ],
    "source_paths": [],
    "policy": ["one Wilson, z 1.96"],
    # Present because `measured_report.report_from_payload` RAISES without it:
    # `entry_quality.get("available_windows")` is iterated with no `or ()` guard
    # (`scripts/measured_report.py:1704-1712`). Latent, outside this packet.
    "entry_quality": {"primary_window": "", "available_windows": []},
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _report():
    import measured_report

    return measured_report.report_from_payload(REPORT_PAYLOAD)


# ==========================================================================
# (a) the measured report, on Research > Results
# ==========================================================================
@pytest.fixture()
def results(qapp):
    from ui.panels.research_results_panel import ResearchResultsPanel

    widget = ResearchResultsPanel()
    yield widget
    widget.deleteLater()


def test_the_results_page_declares_the_measured_report_section(results):
    import ui.panels.research_results_panel as module

    assert module.MEASURED_REPORT_TITLE == "Measured report"
    assert module.REVIEW_COLUMNS == (
        "Cell", "Value", "Unit", "State", "n", "Population", "Window", "Why not",
    )
    assert module.NO_REVIEW_YET == "no review yet"
    assert results.review_table.columnCount() == len(module.REVIEW_COLUMNS)
    headers = [
        results.review_table.horizontalHeaderItem(i).text()
        for i in range(results.review_table.columnCount())
    ]
    assert tuple(headers) == module.REVIEW_COLUMNS


def test_before_any_report_is_published_the_section_says_so(results):
    assert "no measured report" in results.report_id_label.text().lower()
    assert results.review_table.rowCount() == 0


def test_the_section_prints_the_same_cells_and_the_same_report_id_as_the_export(results):
    """The parity seam moves with the widgets: the page and the export read one
    published file, so they cannot disagree by construction."""
    report = _report()
    results.render_report(report)

    assert REPORT_ID in results.report_id_label.text()
    assert tuple(results.review_cells()) == tuple(
        (cell.cell_id, cell.value) for cell in report.cells()
    )
    assert results.review_table.rowCount() == 2


def test_an_unmeasured_cell_prints_its_state_and_its_reason_never_a_zero(results):
    """"Unknown" is not zero (plan.md sec 5). The `Why not` column is why the
    column exists."""
    results.render_report(_report())
    row_text = [
        results.review_table.item(1, column).text()
        for column in range(results.review_table.columnCount())
    ]
    assert "unknown" in row_text
    assert "0" not in row_text[1], row_text
    assert "warehouse is not configured" in row_text[-1]


def test_the_local_review_area_says_nothing_rather_than_inventing_a_review(results):
    import ui.panels.research_results_panel as module

    report = _report()
    results.render_report(report, narration=None)
    assert results.review_note.text().strip() == module.NO_REVIEW_YET

    results.render_report(report, narration="day_quick_median_mfe_r is the slow half.")
    assert "day_quick_median_mfe_r" in results.review_note.text()
    # The narration did not become the numbers.
    assert tuple(results.review_cells()) == tuple(
        (cell.cell_id, cell.value) for cell in report.cells()
    )


def test_the_next_test_card_and_the_copy_export_verbs_came_too(results):
    assert "Next test" in results.next_test_card.text()
    assert results.copy_next_test_button is not None
    assert results.copy_handoff_button is not None
    assert results.export_handoff_button is not None


def test_the_handoff_export_is_still_the_traders_click_and_writes_three_files(results, tmp_path):
    results.render_report(_report())
    written = results.export_handoff(tmp_path)

    assert set(written) == {"markdown", "payload", "manifest"}
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == sorted([
        f"frontier_handoff_{SESSION}.json",
        f"frontier_handoff_{SESSION}.md",
        "manifest.json",
    ])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["report_id"] == REPORT_ID


def test_the_entry_quality_proposal_signal_moved_with_the_section(results):
    """`app.py` rewires this connect; the Tracker route still receives the
    display object the page's own worker already read."""
    seen: list[object] = []
    results.entryQualityProposalChanged.connect(seen.append)
    results.entryQualityProposalChanged.emit({"proposal": "stub"})
    assert seen == [{"proposal": "stub"}]


# ==========================================================================
# (b) the staged picks, on Auto Pilot
# ==========================================================================
class _Signal:
    def connect(self, *_args, **_kwargs):
        return None

    def emit(self, *_args, **_kwargs):
        return None


class _FakeBounceService:
    """Enough of the bounce service for `AutopilotService.__init__`."""

    def __init__(self) -> None:
        self.running = False
        self.scanning_enabled = False
        self.alertReceived = _Signal()
        self.connectionChanged = _Signal()

    def start(self):
        self.running = True
        return True

    def set_scanning_enabled(self, enabled):
        self.scanning_enabled = bool(enabled)


@pytest.fixture()
def autopilot(qapp, monkeypatch):
    from ui.services.autopilot_service import AutopilotService

    monkeypatch.setattr(AutopilotService, "_load_state", lambda self: {"enabled": False})
    monkeypatch.setattr(AutopilotService, "_save_state", lambda self: None)
    monkeypatch.setattr("job_ledger.get_default_ledger", lambda: None)

    from ui.panels.autopilot_panel import AutopilotPanel

    widget = AutopilotPanel(bounce_service=_FakeBounceService())
    yield widget
    # Every timer this page and its service own, stopped: the suite never leaves
    # an Auto Pilot tick running (`test_the_suite_never_starts_a_real_scan`).
    from PySide6.QtCore import QTimer

    for timer in widget.findChildren(QTimer) + widget.service.findChildren(QTimer):
        try:
            timer.stop()
        except Exception:
            pass
    widget.deleteLater()


def test_the_auto_pilot_page_now_holds_the_staged_picks_table(autopilot):
    assert autopilot.staged.columnCount() == 3
    headers = [
        autopilot.staged.horizontalHeaderItem(i).text()
        for i in range(autopilot.staged.columnCount())
    ]
    assert headers == ["Symbol", "Side", "Gate at click time"]
    assert autopilot.staged.editTriggers() == QTableWidget.EditTrigger.NoEditTriggers
    assert autopilot.add_button.text() == "Add selected staged pick to Focus"


def test_the_staged_table_is_filled_from_the_staged_picks_it_is_handed(autopilot):
    autopilot.render_staged({"long": ("NVDA", "AMD"), "short": ("TSLA",)})
    assert autopilot.staged.rowCount() == 3
    rows = [
        (autopilot.staged.item(i, 0).text(), autopilot.staged.item(i, 1).text())
        for i in range(autopilot.staged.rowCount())
    ]
    assert rows == [("NVDA", "LONG"), ("AMD", "LONG"), ("TSLA", "SHORT")]


def test_adding_a_selected_pick_asks_the_desk_rather_than_writing_focus_itself(autopilot):
    """The signal, not a store write: `MainWindow._add_staged_pick_to_focus`
    performs it through `FocusService`, the store's own owner (ground rule 8)."""
    asked: list[tuple[str, str]] = []
    autopilot.focusAddRequested.connect(lambda symbol, side: asked.append((symbol, side)))

    autopilot.render_staged({"long": ("NVDA",), "short": ("TSLA",)})
    autopilot.staged.setCurrentCell(1, 0)
    autopilot.add_button.click()

    assert asked == [("TSLA", "SHORT")]


def test_with_nothing_selected_the_add_button_asks_for_nothing(autopilot):
    asked: list[tuple[str, str]] = []
    autopilot.focusAddRequested.connect(lambda symbol, side: asked.append((symbol, side)))

    autopilot.render_staged({"long": ("NVDA",), "short": ()})
    autopilot.staged.setCurrentCell(-1, -1)
    autopilot.add_button.click()

    assert asked == []


def test_the_adoption_gate_is_shown_and_never_enforced(autopilot):
    """Unchanged from the AWAY Recap and the Daily Recap: the gate governs the
    MACHINE's adoptions, so blocking the trader on it would substitute the
    machine's judgement for theirs."""
    autopilot.render_staged({"long": ("NVDA",), "short": ()})
    autopilot.staged.setCurrentCell(0, 0)
    autopilot.add_button.click()
    note = autopilot.gate_note.text()
    assert "NVDA" in note
    assert "adoption gate" in note.lower()
    assert "never yours" in note.lower() or "unaffected" in note.lower()
