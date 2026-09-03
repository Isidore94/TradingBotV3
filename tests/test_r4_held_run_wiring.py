"""R4 A9/A10 - the D1 dimension is fed, and one formula reaches every surface.

A9: `d1_setup_present` had NO caller. Every one of the live segments read False,
so decision 0016 answer 4 - *"an M5 alert on a name that also carries a D1 setup
outranks the same alert on a name that does not"* - was a dimension in the schema
and a constant in the data.

A10: the Daytrade Tracker shipped a SECOND FORMULA under the headline column key,
and `segment_index`, which says it exists "for a per-alert lookup", had no caller
either - so the M5 alert row carried no held/ran suffix at all.
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


def _outcome_rows(symbol, *, trade_date, held, mfe, bounce="ema_15", n=1):
    """`n` complete episodes, one `registered` + one `final` row each."""
    rows = []
    for index in range(n):
        # The trailing index rides in the SYMBOL, never after the bounce type:
        # `bounce_type_from_event_id` reads everything after the timestamp.
        event_id = f"{symbol}x{index}_long_{trade_date.replace('-', '')}_09_35_00_{bounce}"
        base = {
            "event_id": event_id,
            "trade_date": trade_date,
            "symbol": symbol,
            "direction": "long",
            "entry_time": f"{trade_date}T09:35:00",
            "context_json": json.dumps({"market_environment": "trend_up"}),
        }
        rows.append({**base, "event_type": "registered", "mfe_r": "", "stop_hit": "False"})
        rows.append(
            {
                **base,
                "event_type": "final",
                "mfe_r": f"{mfe}",
                "stop_hit": "False" if held else "True",
                "minutes_elapsed": "5",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# A9 - the D1 dimension
# ---------------------------------------------------------------------------


def test_the_d1_dimension_is_read_from_the_scanners_own_snapshot(tmp_path):
    import held_run_score

    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "setups": {
                    "2026-09-01:NVDA:LONG:2026-08-01:favorite_setup": {
                        "scan_date": "2026-09-01",
                        "symbol": "NVDA",
                        "priority_bucket": "favorite_setup",
                    },
                    "2026-09-01:AMD:LONG:2026-08-01:watch": {
                        "scan_date": "2026-09-01",
                        "symbol": "AMD",
                        "priority_bucket": "watch",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    rows = held_run_score.d1_setup_rows(snapshot)
    by_session = held_run_score.d1_setups_by_session(rows)

    assert by_session == {"2026-09-01": {"NVDA"}}, (
        "only the favorite / near-favorite buckets count as a D1 setup"
    )


def test_an_episode_on_a_name_with_a_d1_setup_says_so(tmp_path):
    """The whole point of A9: this field read False for every live segment."""
    import held_run_score

    rows = _outcome_rows("NVDA", trade_date="2026-09-01", held=True, mfe=2.0)
    episodes = held_run_score.build_episodes(
        rows, d1_setups_by_session={"2026-09-01": {"NVDA"}}
    )

    assert [episode.d1_setup_present for episode in episodes] == [True]
    assert episodes[0].segment()[3] is True


def test_a_missing_snapshot_is_no_rows_and_never_an_error(tmp_path):
    """An absent file degrades to the old behaviour, which is False everywhere."""
    import held_run_score

    assert held_run_score.d1_setup_rows(tmp_path / "nope.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert held_run_score.d1_setup_rows(bad) == []


def test_the_build_path_reads_the_snapshot_and_not_the_gigabyte_tracker(tmp_path):
    """`master_avwap_setup_tracker.json` is 1.1 GB and holds the same fields.

    `json.loads` on that file is one of the three measured causes of the 10 GB
    desk (2026-08-27), so the build path must reach for the 19 MB sibling.
    """
    source = (ROOT / "scripts" / "held_run_score.py").read_text(encoding="utf-8")
    body = source.split("def load_episodes(", 1)[1]
    assert "MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE" in body
    assert "MASTER_AVWAP_SETUP_TRACKER_FILE" not in source


# ---------------------------------------------------------------------------
# A10 - one formula, on every surface
# ---------------------------------------------------------------------------


def test_the_marginal_uses_the_same_arithmetic_as_the_cross_cell():
    """A mean of trimmed means is not a trimmed mean, so it is not built that way."""
    import held_run_score

    rows = []
    for index in range(40):
        rows += _outcome_rows(
            f"SYM{index}",
            trade_date="2026-09-01",
            held=index % 4 != 0,
            mfe=2.0,
        )
    episodes = held_run_score.build_episodes(rows)

    summaries = held_run_score.dimension_summaries(episodes)
    cell = summaries[("bounce_type", "long", "ema_15")]

    assert cell["n"] == 40
    assert cell["hold_rate"] == pytest.approx(0.75)
    assert cell["mean_mfe_r_of_held"] == pytest.approx(2.0)
    assert cell["held_run_score"] == pytest.approx(1.5)
    assert cell["meets_floor"] is True


def test_only_the_dimensions_the_outcome_log_carries_are_offered():
    import held_run_score

    assert held_run_score.MEASURABLE_DIMENSIONS == (
        "bounce_type",
        "time_bucket",
        "market_environment",
    )
    episodes = held_run_score.build_episodes(
        _outcome_rows("NVDA", trade_date="2026-09-01", held=True, mfe=1.0)
    )
    keys = {key[0] for key in held_run_score.dimension_summaries(episodes)}
    assert keys == set(held_run_score.MEASURABLE_DIMENSIONS)


def test_the_alert_row_carries_the_held_and_ran_suffix():
    """`segment_index` promised a per-alert lookup and nothing built the key."""
    import held_run_score
    from ui.models.bounce import BounceAlert
    from ui.widgets.m5_alert_bar import row_text

    rows = []
    for index in range(40):
        rows += _outcome_rows(f"SYM{index}", trade_date="2026-09-01", held=index % 4 != 0, mfe=2.0)
    episodes = held_run_score.build_episodes(rows)
    index_map = held_run_score.segment_index(held_run_score.build_segments(episodes))

    cell = held_run_score.alert_cell(
        index_map,
        bounce_type="ema_15",
        entry_time="2026-09-01T09:35:00",
        market_environment="trend_up",
        d1_setup_present=False,
    )
    assert cell is not None

    alert = BounceAlert(time_text="09:35:00", symbol="NVDA", side="LONG", trigger="ema_15")
    alert.held_run_suffix = held_run_score.alert_suffix(cell)

    assert alert.held_run_suffix == "held 75% / ran 2.0R"
    assert "held 75% / ran 2.0R" in row_text(alert)


def test_a_thin_cell_leaves_the_row_silent_rather_than_bracketed():
    """"held 100% / ran 3.2R (n=2)" reads as a strong segment at a glance."""
    import held_run_score
    from ui.models.bounce import BounceAlert
    from ui.widgets.m5_alert_bar import row_text

    episodes = held_run_score.build_episodes(
        _outcome_rows("NVDA", trade_date="2026-09-01", held=True, mfe=3.2, n=2)
    )
    index_map = held_run_score.segment_index(held_run_score.build_segments(episodes))
    cell = held_run_score.alert_cell(
        index_map,
        bounce_type="ema_15",
        entry_time="2026-09-01T09:35:00",
        market_environment="trend_up",
    )

    assert cell is not None and cell["meets_floor"] is False
    alert = BounceAlert(time_text="09:35:00", symbol="NVDA", side="LONG")
    alert.held_run_suffix = held_run_score.alert_suffix(cell)
    assert alert.held_run_suffix == ""
    assert "held" not in row_text(alert)


def test_an_unmeasured_alert_says_nothing_at_all():
    from ui.models.bounce import BounceAlert
    from ui.widgets.m5_alert_bar import row_text

    alert = BounceAlert(time_text="09:35:00", symbol="NVDA", side="LONG", trigger="ema_15")

    assert alert.held_run_suffix == ""
    assert row_text(alert).strip().endswith("ema_15")


def test_the_alert_path_never_reads_a_file_for_the_suffix():
    """The take-rate suffix's rule, applied to this one: a dict read, never more."""
    source = (ROOT / "scripts" / "ui" / "panels" / "alert_center_panel.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def _attach_held_run_suffix(", 1)[1].split("\n    def ", 1)[0]
    assert "load_episodes()" not in body
    assert "read_outcome_rows" not in body
    assert "self._held_run_index" in body

    worker = source.split("def _held_run_index_worker(", 1)[1].split("\n    def ", 1)[0]
    assert "load_episodes()" in worker
