"""Packet G4.1 - `SetupDetailView` gains the same two additions as the
research pane, ahead of the caller that will use them.

G4.3 (the Setup Tracker's own clearing rule) is DEFERRED to packet G4b because
`setup_tracker_panel.py` is being rewritten on another branch, so nothing calls
these yet. The widget file is not touched by that rewrite, so the API lands here
and this test is what keeps it honest until the caller arrives: a `clear()` that
only emptied the document would leave an empty pane standing where the setup
was, and an identity computed from the display text would be a guess.

Visibility is asserted with `isHidden()` - the widget's own explicit hide flag,
which is what `setVisible` moves - because a never-`show()`n widget reports
`isVisible() == False` regardless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6", reason="the setup detail pane is a Qt widget")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_setup_detail_view_clears_and_names_what_it_shows(qapp):
    from ui.widgets.setup_detail_view import SetupDetailView

    view = SetupDetailView()
    try:
        # A family with no symbol: `show_setup`'s level read only starts for a
        # symbol, so this stays on the calling thread and opens no file.
        view.show_family("earnings_avwap_reclaim", side="SHORT")
        assert not view.isHidden()
        assert view.shown_identity == (
            "family",
            "SHORT",
            "earnings_avwap_reclaim",
            "",
            "",
        )

        view.clear()
        assert view.toPlainText().strip() == "", "clear() left text behind"
        assert view.isHidden(), "clear() left the empty pane on screen"
        assert view.shown_identity is None, (
            "a cleared pane still claims to be showing something"
        )

        # A research row names its own dimension, and the identity carries it:
        # the same family can be shown from two different aggregate tables.
        view.show_research_row(
            "setup_family_performance",
            {
                "dimension": "setup_family",
                "side": "LONG",
                "setup_family": "earnings_avwap_reclaim",
                "sample_count": "12",
            },
        )
        assert not view.isHidden()
        assert view.shown_identity == (
            "setup_family_performance",
            "LONG",
            "earnings_avwap_reclaim",
            "",
            "setup_family",
        )
    finally:
        view.deleteLater()
        qapp.processEvents()
