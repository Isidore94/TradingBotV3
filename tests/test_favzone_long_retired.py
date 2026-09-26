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


def test_retired_controls_never_evict_the_sampled_controls(monkeypatch):
    monkeypatch.setattr(legacy, "CONTROL_SETUP_MAX_RECORDS", 3)
    monkeypatch.setattr(legacy, "SETUP_MAX_RECORDS", 3)
    controls = {f"r{i}": {"scan_date": f"2026-09-0{i + 1}", "control_reason": "random"} for i in range(3)}
    controls.update({f"f{i}": {"scan_date": "2026-09-09", "control_reason": "favzone_long_retired"} for i in range(5)})
    legacy._prune_control_setups(controls, reference_scan_date="2026-09-10")
    assert {key for key in controls if key.startswith("r")} == {"r0", "r1", "r2"}
    assert sum(1 for key in controls if key.startswith("f")) == 3


def _retired(day: str, *, open_: bool) -> dict:
    return {"scan_date": day, "control_reason": "favzone_long_retired", "open_scenario_count": int(open_)}


def test_an_open_retired_row_survives_to_its_time_stop_at_peak_volume():
    """Reviewer 2026-09-26: 350 retired rows a day for 28 days (18 hold + 10 grade sessions) must not
    evict a row opened 20 sessions ago - it stays until it closes, under the main namespace's limits."""
    from datetime import date, timedelta

    start = date(2026, 8, 3)
    controls = {"target": _retired(start.isoformat(), open_=True)}
    for offset in range(1, 41):
        day = start + timedelta(days=offset)
        for index in range(350):
            controls[f"{day}:{index}"] = _retired(day.isoformat(), open_=offset > 30)
        legacy._prune_control_setups(controls, reference_scan_date=day.isoformat())
        assert "target" in controls, day
    controls["target"]["open_scenario_count"] = 0  # closed: now an ordinary old record
    assert (start + timedelta(days=40) - start).days < legacy.SETUP_KEEP_DAYS
    assert len(controls) <= legacy.SETUP_MAX_RECORDS + 1


def test_count_cap_drops_closed_retired_rows_never_open_ones(monkeypatch):
    monkeypatch.setattr(legacy, "SETUP_MAX_RECORDS", 2)
    controls = {
        "old_open": _retired("2026-09-01", open_=True),
        "old_closed": _retired("2026-09-02", open_=False),
        "new_closed": _retired("2026-09-09", open_=False),
    }
    legacy._prune_control_setups(controls, reference_scan_date="2026-09-10")
    assert set(controls) == {"old_open", "new_closed"}


def test_control_discovery_reports_retired_longs_as_their_own_cohort(monkeypatch):
    observations = [
        {"side": "SHORT", "setup_family": "avwap_breakout", "reason": "random", "closed_r": 1.0, "scan_date": "d"},
        {"side": "LONG", "setup_family": "favorite_zone_watch", "reason": "favzone_long_retired",
         "closed_r": -1.0, "scan_date": "d"},
        {"side": "LONG", "setup_family": "favorite_zone_watch", "reason": "favzone_long_retired",
         "closed_r": -0.5, "scan_date": "d"},
    ]
    monkeypatch.setattr(legacy, "_collect_control_episode_observations",
                        lambda setups, default_reason="random": list(observations) if setups == "control" else [])
    discovery = legacy.build_control_discovery_rows({"setups": "main", "control_setups": "control"})
    cohorts = {row["cohort"]: row for row in discovery["cohorts"]}
    assert cohorts["random"]["closed_episodes"] == 1
    assert cohorts["favzone_long_retired"]["closed_episodes"] == 2
    assert [(row["side"], row["setup_family"]) for row in discovery["families"]] == [("SHORT", "avwap_breakout")]
