"""The favourite zone is SHORT-only (trader 2026-09-26): LONG favourite rows are retired, not lost."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

from master_avwap_lib import legacy  # noqa: E402


def _zone_row(side: str) -> dict:
    zone = "AVWAPE to UPPER_1" if side == "LONG" else "LOWER_1 to AVWAPE"
    return {"symbol": f"Z{side[0]}", "side": side, "score": 40.0, "favorite_zone": zone}


def test_long_zone_row_keeps_no_bucket_and_says_why_everywhere():
    long_row, short_row = _zone_row("LONG"), _zone_row("SHORT")
    ai_state = {"symbols": {"ZL": {}, "ZS": {}}}
    features = {"ZL": {}, "ZS": {}}
    records = [{"symbol": "ZL"}, {"symbol": "ZS"}]
    legacy.apply_final_priority_buckets([long_row, short_row], ai_state, records, features)

    assert short_row["priority_bucket"] == "near_favorite_zone"
    assert legacy.FAVZONE_LONG_RETIRED not in short_row
    assert long_row["priority_bucket"] == ""
    assert long_row[legacy.FAVZONE_LONG_RETIRED] == "near_favorite_zone"
    assert ai_state["symbols"]["ZL"][legacy.FAVZONE_LONG_RETIRED] == "near_favorite_zone"
    assert ai_state["symbols"]["ZL"]["is_near_favorite_zone"] is False
    assert features["ZL"]["priority_bucket"] == ""
    assert records[0]["priority_bucket"] == "" and records[1]["priority_bucket"] == "near_favorite_zone"


def test_retired_marker_clears_when_the_row_no_longer_qualifies():
    row = _zone_row("LONG")
    ai_state = {"symbols": {"ZL": {}}}
    legacy.apply_final_priority_buckets([row], ai_state, [], {})
    assert row[legacy.FAVZONE_LONG_RETIRED]
    row["favorite_zone"] = None
    legacy.apply_final_priority_buckets([row], ai_state, [], {})
    assert legacy.FAVZONE_LONG_RETIRED not in row
    assert legacy.FAVZONE_LONG_RETIRED not in ai_state["symbols"]["ZL"]


def test_retired_rows_are_tracker_controls_even_with_sampling_off(monkeypatch):
    row = _zone_row("LONG")
    legacy.apply_final_priority_buckets([row], {"symbols": {"ZL": {}}}, [], {})
    monkeypatch.setattr(legacy, "TRACKER_CONTROL_SAMPLING_ENABLED", False)
    assert legacy.select_tracker_control_rows([row], [], scan_date="2026-09-25") == [row]
    assert row["is_control"] is True
    assert row["control_reason"] == "favzone_long_retired"


def test_long_favourite_zone_arms_no_d1_trigger_but_the_short_one_does():
    anchor = {"vwap": 100.0, "upper_1": 104.0, "lower_1": 96.0, "date": "2026-09-01"}
    for side, close, zone in (("LONG", 102.0, "AVWAPE to UPPER_1"), ("SHORT", 98.0, "LOWER_1 to AVWAPE")):
        state = {"current_anchor": anchor, "last_close": close, "current_band_zone":
                 "VWAP to UPPER_1" if side == "LONG" else "VWAP to LOWER_1"}
        row = {"symbol": "Z", "side": side, "favorite_zone": zone}
        sources = {item["source"] for item in legacy._build_d1_watchlist_trigger_levels(row, state, today_iso="2026-09-25")}
        if side == "LONG":
            assert not sources & {"favorite_zone", "current_band_zone"}
        else:
            assert "favorite_zone" in sources
