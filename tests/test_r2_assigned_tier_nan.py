"""R2 item 1 - an EMPTY `assigned_tier` cell must not become a tier called NAN.

`tier_for_tracker_row` did `str(getter(field, "") or "").strip().upper()` and
then `if assigned:`. A pandas NaN is TRUTHY and `str(nan)` is `"nan"`, so the
moment the feature-history file grows the column - which the first scan after
P4 does - every row written before it reads as a tier named `"NAN"` whose source
is `"assigned"`. The tier list, the tier outcomes and the S/A performance
aggregate then fill with rows for a tier that does not exist.

The existing unit test models an old row with the key ABSENT. The real file has
it PRESENT AND EMPTY, which is a different value and the one that breaks. Both
cases are kept below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _history_row(symbol: str, scan_date: str, **extra) -> dict:
    row = {
        "run_id": f"run-{scan_date}",
        "run_timestamp": f"{scan_date}T13:00:00",
        "run_date": scan_date,
        "watchlist_label": "swing_longs",
        "scan_date": scan_date,
        "symbol": symbol,
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "priority_score": 120.0,
        "setup_family": "avwape_to_first_dev",
        "favorite_zone": "AVWAPE to UPPER_1",
        "current_band_zone": "AVWAPE to UPPER_1",
        "trend_20d": "up",
        "last_close": 100.0,
    }
    row.update(extra)
    return row


def _widened_history(tmp_path: Path) -> pd.DataFrame:
    """The real shape: a file written WITHOUT the column, then widened.

    This is what the 07:30 scan produces - the old rows keep their values and
    gain an empty cell for the new column, which `pd.read_csv` reads back as
    NaN rather than as an absent key.
    """
    from master_avwap_lib.legacy import ASSIGNED_TIER_FIELD

    target = tmp_path / "d1_feature_history.csv"
    old = pd.DataFrame([_history_row("OLD1", "2026-09-01"), _history_row("OLD2", "2026-09-01")])
    old.to_csv(target, index=False)

    # The widening: a new run whose frame carries the column.
    new = pd.DataFrame(
        [
            _history_row("NEW1", "2026-09-02", **{ASSIGNED_TIER_FIELD: "S"}),
            _history_row("NEW2", "2026-09-02", **{ASSIGNED_TIER_FIELD: "A"}),
        ]
    )
    combined = pd.concat([old, new], ignore_index=True)
    combined.to_csv(target, index=False)

    return pd.read_csv(target, low_memory=False)


def test_an_empty_cell_is_absent_and_never_a_tier_named_nan(tmp_path):
    from master_avwap_lib.legacy import ASSIGNED_TIER_FIELD, tier_for_tracker_row

    frame = _widened_history(tmp_path)
    old_rows = frame[frame["scan_date"] == "2026-09-01"].to_dict("records")
    assert old_rows, "the fixture must contain rows written before the widening"
    # The value really is NaN, not an absent key - that is the whole point.
    assert pd.isna(old_rows[0][ASSIGNED_TIER_FIELD])

    for row in old_rows:
        tier, source = tier_for_tracker_row(row)
        assert source == "derived_from_bucket", row
        assert tier != "NAN"
        assert tier in {"S", "A", ""}


def test_the_absent_key_case_still_works():
    """The shape the existing unit test models. Both are real."""
    from master_avwap_lib.legacy import tier_for_tracker_row

    tier, source = tier_for_tracker_row({"priority_bucket": "favorite_setup"})
    assert (tier, source) == ("S", "derived_from_bucket")


def test_only_the_vocabulary_the_stamper_writes_counts_as_assigned():
    """`_priority_partition_tier_rows` writes exactly S, A and B."""
    from master_avwap_lib.legacy import (
        ASSIGNED_TIER_FIELD,
        ASSIGNED_TIER_VALUES,
        tier_for_tracker_row,
    )

    assert ASSIGNED_TIER_VALUES == frozenset({"S", "A", "B"})
    for value in ASSIGNED_TIER_VALUES:
        tier, source = tier_for_tracker_row(
            {ASSIGNED_TIER_FIELD: value, "priority_bucket": "favorite_setup"}
        )
        assert (tier, source) == (value, "assigned")

    # Everything else is ABSENT, not a tier.
    for junk in (float("nan"), "nan", "NaN", "", None, "  ", "Z", 0, "S/A"):
        tier, source = tier_for_tracker_row(
            {ASSIGNED_TIER_FIELD: junk, "priority_bucket": "favorite_setup"}
        )
        assert source == "derived_from_bucket", junk
        assert tier == "S", junk


def test_no_nan_tier_reaches_the_pick_or_outcome_rows(tmp_path):
    """The reproduction that matters: the files a reader actually opens."""
    from master_avwap_lib import legacy

    frame = _widened_history(tmp_path)

    picks = legacy.build_bot_tier_pick_rows(frame)
    assert picks, "the builder must produce something to be worth asserting on"
    tiers = {str(row.get("tier") or "") for row in picks}
    assert "NAN" not in tiers, picks
    assert tiers <= {"S", "A", "B"}

    # Only the rows the latest scan really assigned may say `assigned`.
    by_symbol = {str(row.get("symbol") or ""): row for row in picks}
    for symbol, expected in (("NEW1", "assigned"), ("NEW2", "assigned")):
        if symbol in by_symbol:
            assert by_symbol[symbol]["tier_source"] == expected

    observations = [
        {
            "observation_id": f"obs-{index}",
            "scan_row_id": str(row.get("_scan_row_id") or ""),
            "horizon_sessions": 5,
            "side": "LONG",
            "run_id": row.get("run_id"),
            "run_timestamp": row.get("run_timestamp"),
            "run_date": row.get("run_date"),
            "watchlist_label": row.get("watchlist_label"),
            "scan_date": row.get("scan_date"),
            "future_scan_date": "2026-09-08",
        }
        for index, row in enumerate(legacy._prepare_scan_factor_history_frame(frame).to_dict("records"))
    ]
    outcomes = legacy.build_bot_tier_outcome_rows(frame, observations)
    outcome_tiers = {str(row.get("tier") or "") for row in outcomes}
    assert "NAN" not in outcome_tiers, outcomes
