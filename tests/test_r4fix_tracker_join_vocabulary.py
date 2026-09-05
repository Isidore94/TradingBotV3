"""R4 fix round 1, blocker 1 - the tracker join was a string match across two
vocabularies that do not agree.

`daytrade_tracker_panel.apply_held_and_ran` keys on `(dimension, direction,
segment)` raw text, and `held_run_score` spelled all three differently from the
aggregator that writes `intraday_bounce_performance.csv`. Measured on the live
stores, the four dimensions this module claims read **28/36, 0/59, 2/10 and
10/10** - so rows the data CAN answer went blank for a SPELLING reason while
`CLAUDE.md`, `plan.md` and the gate all said the tab was unanswerable.

Three separate mismatches, and the first is a real defect on its own:

* `time_bucket` compared raw wall-clock hours against Eastern cutoffs while
  `entry_time` in the outcome log is DESK-LOCAL. That is the exact bug
  `bounce_bot_lib.learning.time_bucket_for` records itself as having fixed - "on
  a Pacific machine that mislabeled nearly the entire session" - so the module
  was not merely spelling the buckets differently, it was putting episodes in
  the wrong ones.
* the aggregator counts an episode under EACH of its bounce types; this module
  never split a combination, so eight of the 36 types never matched.
* the combination itself is `+`-joined there and `-`-joined here.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import held_run_score as hrs  # noqa: E402


# ---------------------------------------------------------------------------
# One time-bucket vocabulary, and it is the champion's
# ---------------------------------------------------------------------------


def test_the_time_bucket_is_the_champions_own_answer():
    """Called, not copied. A second definition is what broke the join."""
    from bounce_bot_lib.learning import time_bucket_for

    for text in (
        "2026-09-02T06:35:00",
        "2026-09-02T07:45:00",
        "2026-09-02T09:15:00",
        "2026-09-02T11:30:00",
        "2026-09-02T12:45:00",
    ):
        assert hrs.time_bucket(text) == time_bucket_for(datetime.fromisoformat(text)), text


def test_the_declared_buckets_are_the_aggregators_five():
    assert hrs.TIME_BUCKETS == (
        "opening_drive",
        "late_morning",
        "midday",
        "afternoon",
        "closing_window",
    )


def test_a_desk_local_stamp_is_measured_from_its_own_session_open():
    """The Pacific bug: 10:40 PT is 13:40 in New York, not the late morning.

    The old rule read the hour off the clock and compared it with 10:00 / 11:30 /
    15:00 - boundaries that only mean anything in New York - so on this desk an
    afternoon entry was labelled `morning`.
    """
    from bounce_bot_lib.learning import time_bucket_for

    pacific = ZoneInfo("America/Los_Angeles")
    afternoon = datetime(2026, 9, 2, 10, 40, tzinfo=pacific)

    assert hrs.time_bucket(afternoon) == time_bucket_for(afternoon)
    assert hrs.time_bucket(afternoon) not in ("morning", "open_30m")


def test_an_unreadable_time_is_unknown_and_never_dropped():
    assert hrs.time_bucket("") == hrs.UNKNOWN
    assert hrs.time_bucket("not a time") == hrs.UNKNOWN


# ---------------------------------------------------------------------------
# The bounce type and the combination
# ---------------------------------------------------------------------------


def test_a_combination_splits_into_the_types_the_aggregator_counts():
    assert hrs.bounce_components("eod_vwap-impulse_retest_vwap_eod-vwap") == (
        "eod_vwap",
        "impulse_retest_vwap_eod",
        "vwap",
    )
    assert hrs.bounce_components("ema_15") == ("ema_15",)
    assert hrs.bounce_components("") == ()


def test_the_ten_candle_sides_are_one_type_to_the_aggregator():
    """`10_candle_high` and `10_candle_low` are both recorded as `10_candle`."""
    assert hrs.bounce_components("10_candle_low") == ("10_candle",)
    assert hrs.bounce_components("10_candle_high") == ("10_candle",)
    assert hrs.bounce_components("10_candle_low-ema_21") == ("10_candle", "ema_21")


def test_the_combination_is_spelled_the_way_the_aggregator_spells_it():
    """One separator is why the Combos tab matched 0 of 59 live rows."""
    assert hrs.bounce_combo("10_candle_low-ema_21") == "10_candle+ema_21"
    assert hrs.bounce_combo("ema_15") == "ema_15"


# ---------------------------------------------------------------------------
# What the join can now answer
# ---------------------------------------------------------------------------


def _rows(symbol, *, trade_date, bounce, held=True, mfe=2.0, n=1, entry="09:35:00"):
    out = []
    for index in range(n):
        event_id = f"{symbol}x{index}_long_{trade_date.replace('-', '')}_09_35_00_{bounce}"
        base = {
            "event_id": event_id,
            "trade_date": trade_date,
            "symbol": f"{symbol}x{index}",
            "direction": "long",
            "entry_time": f"{trade_date}T{entry}",
            "context_json": '{"market_environment": "trend_up"}',
        }
        out.append({**base, "event_type": "registered", "mfe_r": "", "stop_hit": "False"})
        out.append(
            {
                **base,
                "event_type": "final",
                "mfe_r": f"{mfe}",
                "stop_hit": "False" if held else "True",
                "minutes_elapsed": "60" if held else "5",
            }
        )
    return out


def test_the_four_measurable_dimensions_are_declared_and_the_rest_are_split_by_reason():
    assert hrs.MEASURABLE_DIMENSIONS == (
        "bounce_type",
        "bounce_combo",
        "time_bucket",
        "market_environment",
    )
    # "the log cannot be asked" and "we have not derived it" are different
    # promises and are no longer written as one.
    assert hrs.UNDERIVED_DIMENSIONS == ("rrs_alignment",)


def test_an_episode_counts_under_every_type_it_carries_and_once_under_its_combo():
    episodes = hrs.build_episodes(
        _rows("SYM", trade_date="2026-09-01", bounce="ema_15-vwap", n=40)
    )
    summaries = hrs.dimension_summaries(episodes)

    assert summaries[("bounce_type", "long", "ema_15")]["n"] == 40
    assert summaries[("bounce_type", "long", "vwap")]["n"] == 40
    assert summaries[("bounce_combo", "long", "ema_15+vwap")]["n"] == 40
    # And the combination is ONE VALUE, not two: `ema_15+vwap`, never a cell per
    # component. R4 B4 added a second DIRECTION SLOT per cell - the pooled
    # `ALL_DIRECTIONS` one a sideless "My Decisions" row joins - so the assertion
    # is on the segment values rather than on the raw key count, which would now
    # be counting slots.
    combos = {key[2] for key in summaries if key[0] == "bounce_combo"}
    assert combos == {"ema_15+vwap"}
    assert summaries[("bounce_combo", hrs.ALL_DIRECTIONS, "ema_15+vwap")]["n"] == 40


def test_the_panel_join_fills_a_row_written_in_the_aggregators_spelling():
    """End to end through the real join, on the real key."""
    from ui.panels.daytrade_tracker_panel import apply_held_and_ran

    episodes = hrs.build_episodes(
        _rows("SYM", trade_date="2026-09-01", bounce="10_candle_low-ema_21", n=40)
    )
    summaries = hrs.dimension_summaries(episodes)

    rows = apply_held_and_ran(
        [
            {"dimension": "bounce_type", "direction": "long", "segment": "10_candle"},
            {"dimension": "bounce_type", "direction": "long", "segment": "ema_21"},
            {"dimension": "bounce_combo", "direction": "long", "segment": "10_candle+ema_21"},
            {"dimension": "time_bucket", "direction": "long", "segment": hrs.time_bucket("2026-09-01T09:35:00")},
            {"dimension": "market_environment", "direction": "long", "segment": "trend_up"},
        ],
        summaries,
    )

    for row in rows:
        assert row["held_run_score"] is not None, row["dimension"] + "/" + row["segment"]
        assert row["held_rate"] == pytest.approx(1.0)


def test_a_dimension_the_log_genuinely_lacks_is_still_blank():
    """The four Swing tabs are not in the outcome log at all - not a spelling."""
    from ui.panels.daytrade_tracker_panel import apply_held_and_ran

    episodes = hrs.build_episodes(_rows("SYM", trade_date="2026-09-01", bounce="ema_15", n=40))
    rows = apply_held_and_ran(
        [{"dimension": "master_avwap_setup_family", "direction": "long", "segment": "AVWAPE"}],
        hrs.dimension_summaries(episodes),
    )

    assert rows[0]["held_run_score"] is None
    assert rows[0]["held_rate"] is None
