"""R4 Part B item B4 - the Daytrade Tracker says which number is which.

Three things the panel's own header comment promised and the table did not do.

1. *"the champion tier stays as a column"* - there was no tier column. The tier
   and the headline answer different questions: the tier says whether the desk
   should alert on the segment at all, and Held x Ran says what the alert offered
   once the level held. Without the tier beside it a reader has one number and
   two meanings.
2. *"the Verdict column already says which is which"* - it did not. `Verdict` is
   the aggregator's `edge_score` recommendation, computed from average R, and it
   sat unlabelled next to a headline computed from something else entirely. Two
   verdicts under one table, neither naming its own basis.
3. The **My Decisions** tabs - what the trader took and passed - were graded in
   mean R alone, on the DAY-TRADE side, where decision 0016 answer 4 says the
   headline is MFE after a held level. They now carry Held 30m and Held x Ran
   from the one helper, joined on the same key.

Offline and pure. No store, no widget: every assertion is against the joining
functions and the column tuples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("PySide6", reason="the tracker panel is a Qt widget")


# ---------------------------------------------------------------------------
# The champion tier column
# ---------------------------------------------------------------------------


def test_the_performance_table_carries_the_champion_tier():
    from ui.panels.daytrade_tracker_panel import PERFORMANCE_COLUMNS

    keys = [key for key, _label in PERFORMANCE_COLUMNS]
    assert "champion_tier" in keys, "the header comment promises a tier column"
    # Beside the headline, not in front of it: Held 30m and Held x Ran still lead.
    assert keys.index("held_rate") < keys.index("champion_tier")


def test_the_tier_is_joined_from_the_learning_state_on_the_same_key():
    from ui.panels.daytrade_tracker_panel import apply_champion_tier

    state = {
        "segments": {
            "bounce_type": {
                "long|vwap": {"proven": True, "muted": False},
                "short|vwap": {"proven": False, "muted": True},
                "long|ema9": {"proven": False, "muted": False},
            }
        }
    }
    rows = apply_champion_tier(
        [
            {"dimension": "bounce_type", "direction": "long", "segment": "vwap"},
            {"dimension": "bounce_type", "direction": "short", "segment": "vwap"},
            {"dimension": "bounce_type", "direction": "long", "segment": "ema9"},
            {"dimension": "bounce_type", "direction": "long", "segment": "unheard_of"},
        ],
        state,
    )
    assert [row["champion_tier"] for row in rows] == ["PROVEN", "MUTED", "active", ""]


def test_a_segment_the_learning_state_never_saw_is_blank_not_active():
    """"Not tracked" and "tracked and unremarkable" are different facts."""
    from ui.panels.daytrade_tracker_panel import apply_champion_tier

    rows = apply_champion_tier(
        [{"dimension": "bounce_type", "direction": "long", "segment": "x"}], {}
    )
    assert rows[0]["champion_tier"] == ""


# ---------------------------------------------------------------------------
# The Verdict names its basis
# ---------------------------------------------------------------------------


def test_the_verdict_header_names_the_score_it_comes_from():
    from ui.panels.daytrade_tracker_panel import PERFORMANCE_COLUMNS

    label = dict(PERFORMANCE_COLUMNS)["recommendation"]
    assert "edge" in label.lower(), (
        "an unlabelled Verdict beside an unlabelled headline is two verdicts and "
        "no way to tell which is which"
    )


# ---------------------------------------------------------------------------
# The decision readouts gain the headline
# ---------------------------------------------------------------------------


def test_the_pooled_direction_cell_is_measured_and_not_an_average_of_averages():
    """The decisions state does not carry a side, so the join needs a pooled cell.

    It is built from the EPISODES, once, by the same `Segment.summary` - never by
    averaging the long cell and the short cell, because a mean of trimmed means
    is not a trimmed mean.
    """
    import held_run_score as hrs

    episodes = [
        hrs.Episode(
            event_id=f"e{index}",
            trade_date="2026-09-01",
            symbol=f"S{index}",
            direction="long" if index % 2 == 0 else "short",
            bounce_type="vwap",
            entry_time="10:00",
            market_environment="BULLISH",
            measurement=hrs.MEASURED_HELD,
            mfe_r=1.0 + index,
        )
        for index in range(8)
    ]
    summaries = hrs.dimension_summaries(episodes, as_of="2026-09-01")
    pooled = summaries.get(("bounce_type", hrs.ALL_DIRECTIONS, "vwap"))
    assert pooled is not None, "no pooled-direction cell to join a sideless row to"
    assert pooled["n"] == 8
    longs = summaries[("bounce_type", "long", "vwap")]
    shorts = summaries[("bounce_type", "short", "vwap")]
    assert pooled["n"] == longs["n"] + shorts["n"]


def test_the_decision_columns_carry_the_day_trade_headline():
    from ui.panels.daytrade_tracker_panel import DECISION_COLUMNS

    keys = [key for key, _label in DECISION_COLUMNS]
    assert "held_rate" in keys and "held_run_score" in keys


def test_a_sideless_decision_row_joins_the_pooled_cell():
    """The one helper, not a second one written for this table."""
    import held_run_score as hrs
    from ui.panels.daytrade_tracker_panel import apply_held_and_ran

    summaries = {
        ("bounce_type", "long", "vwap"): {"hold_rate": 0.9, "held_run_score": 2.0},
        ("bounce_type", hrs.ALL_DIRECTIONS, "vwap"): {
            "hold_rate": 0.71,
            "held_run_score": 1.4,
        },
    }
    rows = apply_held_and_ran(
        [{"dimension": "bounce_type", "segment": "vwap"}], summaries
    )
    assert rows[0]["held_rate"] == 0.71
    assert rows[0]["held_run_score"] == 1.4


def test_a_row_that_names_its_side_still_gets_that_sides_cell():
    from ui.panels.daytrade_tracker_panel import apply_held_and_ran

    summaries = {
        ("bounce_type", "long", "vwap"): {"hold_rate": 0.9, "held_run_score": 2.0},
        ("bounce_type", "all", "vwap"): {"hold_rate": 0.71, "held_run_score": 1.4},
    }
    rows = apply_held_and_ran(
        [{"dimension": "bounce_type", "direction": "LONG", "segment": "vwap"}],
        summaries,
    )
    assert rows[0]["held_rate"] == 0.9


def test_an_unanswerable_decision_row_is_blank_rather_than_zero():
    from ui.panels.daytrade_tracker_panel import apply_held_and_ran

    rows = apply_held_and_ran([{"dimension": "tier", "segment": "A"}], {})
    assert rows[0]["held_rate"] is None
    assert rows[0]["held_run_score"] is None
