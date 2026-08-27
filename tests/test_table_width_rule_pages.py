"""The two pages the §12 width rule was learned on, checked through the pages.

Split from `test_table_width_rule.py` on purpose: that file imports the helper
at module scope, so on un-fixed code it fails at collection, which proves the
helper is absent and nothing about the pages. These two import nothing new at
module scope, so on un-fixed code they fail on the ASSERTION - the cohort column
does not stretch, the `Line` column does not stretch - which is the defect the
trader reported in their own words on 2026-08-26.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QHeaderView  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application

def test_weekend_prep_focus_review_cohort_columns_stretch_and_elide(app, tmp_path):
    """§8.4: the page where the rule was learned. Three tables, one identity column."""
    from ui.panels import weekend_prep_panel
    from ui.widgets.data_table import MiddleElideDelegate
    from ui.services.weekend_prep_service import WeekendPrepService

    service = WeekendPrepService(
        state_path=tmp_path / "state.json", now=datetime(2026, 8, 15, 10, 0)
    )
    page = weekend_prep_panel.FocusReviewPage(service)
    page._render_cohort(
        [
            {"cohort": "veto_v3_sma_incoming", "side": "LONG", "n": 8,
             "win_rate": "0.50", "avg_return": "0.10", "profit_factor": "1.1"},
        ]
    )
    page._render_like_cohort(
        [
            {"cohort": "human_focus_tracking", "side": "LONG", "n": 8,
             "win_rate": "0.50", "avg_return": "0.10", "profit_factor": "1.1"},
        ]
    )
    page._render_performance(
        [
            {"cohort": "human_focus_tracking", "side": "LONG", "horizon": "h3",
             "n": 8, "win_rate": "0.50", "avg_return": "0.10", "median": "0.05",
             "profit_factor": "1.1", "symbols": 4, "sessions": 3, "ci": "",
             "updated_at": "2026-08-25T21:00:00"},
        ]
    )

    for table in (page.cohort_table, page.like_table, page.performance_table):
        assert table.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
        assert isinstance(table.itemDelegateForColumn(0), MiddleElideDelegate)
        assert table.item(0, 0).toolTip() == table.item(0, 0).text()
    page.shutdown() if hasattr(page, "shutdown") else None


def test_away_recap_tables_give_the_line_column_the_slack(app):
    """§3.4 A: `1. FROG …` with two thirds of the window empty."""
    from ui.panels.away_recap_panel import AwayRecapPanel

    panel = AwayRecapPanel()
    panel._render(
        {
            "summary": "one day",
            "best_swings": [
                {"rank": 1, "symbol": "FROG", "side": "LONG",
                 "text": "1. FROG LONG - above the previous day's high, holding it"},
            ],
            "classified_alerts": [
                {"time_text": "09:31", "symbol": "OKTA", "side": "LONG",
                 "tier": "A", "is_d1": False,
                 "trigger": "M5 bounce off the anchored VWAP with volume"},
            ],
            "staged_picks": [{"symbol": "MRK", "side": "LONG"}],
            "focus_to_manage": [{"symbol": "GFS", "side": "SHORT"}],
        }
    )

    assert panel.swings.horizontalHeader().sectionResizeMode(3) == QHeaderView.ResizeMode.Stretch
    assert panel.alerts.horizontalHeader().sectionResizeMode(5) == QHeaderView.ResizeMode.Stretch
    assert panel.staged.horizontalHeader().sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch
    assert panel.swings.horizontalHeader().stretchLastSection() is False
