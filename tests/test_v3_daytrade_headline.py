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


def test_the_score_is_the_hold_rate_times_what_the_held_ones_offered():
    from ui.panels.daytrade_tracker_panel import _add_held_and_ran

    rows = _add_held_and_ran([{"stop_rate": 0.25, "avg_mfe_r": 2.0}])

    assert rows[0]["held_rate"] == 0.75
    assert rows[0]["held_run_score"] == 1.5


def test_a_row_missing_an_input_is_blank_and_never_zero():
    """A zero ranks a segment we could not measure at the bottom of an ordering."""
    from ui.panels.daytrade_tracker_panel import _add_held_and_ran

    rows = _add_held_and_ran(
        [{"stop_rate": None, "avg_mfe_r": 2.0}, {"stop_rate": 0.1, "avg_mfe_r": ""}]
    )

    for row in rows:
        assert row["held_rate"] is None
        assert row["held_run_score"] is None


def test_an_unmeasured_row_sorts_last_rather_than_at_the_bottom_of_the_scale():
    from ui.panels.daytrade_tracker_panel import _add_held_and_ran, _float

    rows = _add_held_and_ran(
        [
            {"segment": "blank", "stop_rate": None, "avg_mfe_r": None},
            {"segment": "weak", "stop_rate": 0.9, "avg_mfe_r": 0.2},
            {"segment": "strong", "stop_rate": 0.2, "avg_mfe_r": 3.0},
        ]
    )
    ordered = sorted(
        rows,
        key=lambda r: (r.get("held_run_score") is None, -_float(r.get("held_run_score"), -999.0)),
    )
    assert [row["segment"] for row in ordered] == ["strong", "weak", "blank"]


def test_the_hold_rate_is_clamped_to_a_rate():
    """A stop rate over 1 is bad data, not a negative hold rate."""
    from ui.panels.daytrade_tracker_panel import _add_held_and_ran

    rows = _add_held_and_ran([{"stop_rate": 1.4, "avg_mfe_r": 2.0}])
    assert rows[0]["held_rate"] == 0.0
    assert rows[0]["held_run_score"] == 0.0


def test_the_column_is_labelled_held_and_not_held_in_thirty_minutes():
    """The tracker's stop rate is over ITS window, not the 30-minute question.

    `held_run_score.build_segments` computes the precise version from the raw
    outcome log; this column is the same SHAPE from the aggregate the panel
    already has, and the label must not claim otherwise.
    """
    from ui.panels.daytrade_tracker_panel import PERFORMANCE_COLUMNS

    labels = dict(PERFORMANCE_COLUMNS)
    assert labels["held_rate"] == "Held"
    assert "30" not in labels["held_rate"]

    source = (ROOT / "scripts" / "ui" / "panels" / "daytrade_tracker_panel.py").read_text(
        encoding="utf-8"
    )
    assert "30-minute question" in source
    assert "approximation of the same shape" in source
