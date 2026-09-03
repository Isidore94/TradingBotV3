"""V3 item 2 - MFE after a held level is the day-trade headline.

Decision 0016 answer 4, in the trader's words: *"the intraday level holds, then
the name runs. Rank by maximum favourable excursion - the most the move offered -
not by any exit; exiting well is the trader's job."*

The champion tier stays as a column and keeps gating alerts. These two say what
the alert OFFERED once the level held, which is a different question.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_held_and_ran_lead_the_table_and_the_tier_columns_stay():
    from ui.panels.daytrade_tracker_panel import PERFORMANCE_COLUMNS

    keys = [key for key, _label in PERFORMANCE_COLUMNS]
    assert keys.index("held_rate") < keys.index("avg_close_r")
    assert keys.index("held_run_score") < keys.index("avg_close_r")
    # Never replaced: the tier's own statistics are still there.
    for kept in ("sample_count", "avg_close_r", "stop_rate", "recommendation"):
        assert kept in keys, kept


def test_the_default_sort_is_the_headline_not_the_sample_count():
    from ui.panels.daytrade_tracker_panel import (
        DEFAULT_PERFORMANCE_SORT_KEY,
        PERFORMANCE_COLUMNS,
    )

    assert DEFAULT_PERFORMANCE_SORT_KEY == "held_run_score"
    assert DEFAULT_PERFORMANCE_SORT_KEY in [key for key, _ in PERFORMANCE_COLUMNS]


def test_the_panel_computes_nothing_and_joins_the_modules_own_answer():
    """R4 A10: V3 shipped a SECOND FORMULA under the headline column key.

    `1 - stop_rate` times `avg_mfe_r`, both from the aggregator over its own
    window and over ALL rows rather than the held ones, with no thirty-minute
    question in it anywhere. Two different numbers under one heading is worse
    than one blank, because the trader reads the column as an ordering.
    """
    from ui.panels import daytrade_tracker_panel as panel

    assert not hasattr(panel, "_add_held_and_ran"), "the second formula is back"

    summaries = {
        ("bounce_type", "long", "ema_15"): {"hold_rate": 0.75, "held_run_score": 1.5},
    }
    rows = panel.apply_held_and_ran(
        [
            {"dimension": "bounce_type", "direction": "long", "segment": "ema_15",
             "stop_rate": 0.9, "avg_mfe_r": 0.1},
        ],
        summaries,
    )

    assert rows[0]["held_rate"] == 0.75
    assert rows[0]["held_run_score"] == 1.5, (
        "the aggregate's own stop_rate/avg_mfe_r must not reach this column"
    )


def test_a_dimension_the_outcome_log_cannot_answer_is_blank_and_never_zero():
    """Six of the nine tabs are cut on context the outcome log does not record.

    A zero would rank a segment we could not measure at the bottom of a list the
    trader reads as an ordering; a substitute formula would rank it wrongly and
    invisibly. Blank is the answer.
    """
    from ui.panels.daytrade_tracker_panel import apply_held_and_ran

    rows = apply_held_and_ran(
        [{"dimension": "master_avwap_setup_family", "direction": "long", "segment": "AVWAPE"}],
        {("bounce_type", "long", "ema_15"): {"hold_rate": 0.5, "held_run_score": 1.0}},
    )

    assert rows[0]["held_rate"] is None
    assert rows[0]["held_run_score"] is None


def test_an_unmeasured_row_sorts_last_rather_than_at_the_bottom_of_the_scale():
    from ui.panels.daytrade_tracker_panel import _by_headline, apply_held_and_ran

    summaries = {
        ("bounce_type", "long", "weak"): {"hold_rate": 0.1, "held_run_score": 0.02},
        ("bounce_type", "long", "strong"): {"hold_rate": 0.8, "held_run_score": 2.4},
    }
    rows = apply_held_and_ran(
        [
            {"dimension": "bounce_type", "direction": "long", "segment": "blank"},
            {"dimension": "bounce_type", "direction": "long", "segment": "weak"},
            {"dimension": "bounce_type", "direction": "long", "segment": "strong"},
        ],
        summaries,
    )

    assert [row["segment"] for row in _by_headline(rows)] == ["strong", "weak", "blank"]


def test_the_expensive_read_never_runs_on_the_qt_thread():
    """~300,000 outcome rows and a 19 MB snapshot; the panel opens on the GUI."""
    source = (ROOT / "scripts" / "ui" / "panels" / "daytrade_tracker_panel.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def reload_from_disk(", 1)[1].split("\n    def ", 1)[0]
    assert "load_held_run_summaries()" not in body, "the read moved onto the paint path"
    assert "_start_held_run_read()" in body

    worker = source.split("def _held_run_worker(", 1)[1].split("\n    def ", 1)[0]
    assert "load_held_run_summaries()" in worker


def test_the_column_says_what_it_measures_now_that_it_measures_it():
    """V3 called it "Held" because it was NOT the thirty-minute question."""
    from ui.panels.daytrade_tracker_panel import PERFORMANCE_COLUMNS

    labels = dict(PERFORMANCE_COLUMNS)
    assert labels["held_rate"] == "Held 30m"
    assert labels["held_run_score"] == "Held x Ran"
