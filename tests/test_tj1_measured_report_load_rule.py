"""TJ-1 item 6(a) - the moved section obeys the page it moved ONTO (G7.1).

The Measured report came off the Daily Recap's fifth tab, where it was read by
that page's own worker when the page was opened. On Research > Results it is a
SECTION of a page the desk builds at startup, and G7.1's rule for that page is
"Research's first paint costs ONE child's load, not nine": each child reads when
its tab is opened.

The first cut of the move started the report read in `ResearchResultsPanel
.__init__`, which broke that rule in two ways worth writing down: the desk got a
thread at startup for a section nobody had looked at, and in a test that only
calls `deleteLater()` the read outlived the panel it was going to update. It also
coincided with an intermittent failure of
`tests/test_g7_speed_pass.py::test_showing_the_research_tab_loads_only_the_child_whose_tab_is_open`
(2 of 4 full-suite runs, never standalone) - the section's read is not what that
test counts, but a background import inside its 0.5 s drain window is exactly the
kind of thing that moves a race.

So the read waits for the first SHOW, and a page switch is not a re-read. This
file pins that, because nothing else does: the tester's `test_tj1_moves.py`
renders the report directly.
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

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="Research > Results is a Qt panel")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_the_measured_report_read_is_started_by_the_first_show_not_the_constructor(
    qapp, monkeypatch
):
    """Asserted on WHO STARTS the read, not on when it lands.

    The read runs on a `ReadWorker`, so "has it happened yet" is a race even when
    the answer is wrong; "did the constructor ask for it" is a fact.
    """
    import ui.panels.research_results_panel as module

    started: list[int] = []
    monkeypatch.setattr(
        module.ResearchResultsPanel, "refresh_report", lambda self: started.append(1)
    )

    widget = module.ResearchResultsPanel()
    try:
        assert started == [], "the constructor started the report read"

        widget.show()
        qapp.processEvents()
        assert started == [1], "the first show did not start the report read"

        widget.hide()
        widget.show()
        qapp.processEvents()
        assert started == [1], "a page switch is not a re-read"
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()


def test_the_read_that_the_show_starts_really_reaches_the_published_file(qapp, monkeypatch):
    """...and the seam it starts is the one that reads the report."""
    import ui.panels.research_results_panel as module

    reads: list[int] = []
    monkeypatch.setattr(module, "_read_measured_report", lambda: reads.append(1) or {})

    widget = module.ResearchResultsPanel()
    try:
        widget.show()
        for _ in range(40):
            qapp.processEvents()
            if reads:
                break
        assert reads == [1]
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()


def test_the_section_still_says_so_before_anything_is_published(qapp):
    """Whatever the read does later, the page opens with a sentence rather than
    an empty table - the state the trader sees on a desk whose overnight run has
    not written a report yet."""
    import ui.panels.research_results_panel as module

    widget = module.ResearchResultsPanel()
    try:
        assert "no measured report" in widget.report_id_label.text().lower()
        assert widget.review_table.rowCount() == 0
        assert widget.review_note.text().strip() == module.NO_REVIEW_YET
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()


def test_a_report_read_that_fails_says_so_and_costs_the_page_nothing(qapp, monkeypatch):
    import ui.panels.research_results_panel as module

    def _refuse():
        raise OSError("the ai_store is not mounted")

    monkeypatch.setattr(module, "_read_measured_report", _refuse)
    widget = module.ResearchResultsPanel()
    try:
        widget.show()
        for _ in range(40):
            qapp.processEvents()
            if "not mounted" in widget.report_id_label.text():
                break
        assert "not mounted" in widget.report_id_label.text()
        # The four populations are untouched: a failed report read is one label.
        assert widget.selection()
        assert widget.review_table.rowCount() == 0
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()
