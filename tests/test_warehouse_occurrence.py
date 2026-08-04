"""Occurrence identity and the rescan rule (plan Phase 6, sec 7.3, risk R9).

The exit criterion pinned here: **a rescan updates, it never appends an
episode.** Also pinned: the deterministic key, that long/short and swing/M5
theses stay distinct, that the warehouse records detector output rather than
re-detecting, and that evidence counts episodes rather than rows.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.research_warehouse import occurrences, schemas
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc
TRIGGER = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
NOW = datetime(2026, 8, 3, 20, 5, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def _detected(**overrides):
    row = {
        "symbol": "AAPL",
        "canonical_setup_id": "AVWAPE_TO_FIRST_DEV",
        "side": "LONG",
        "structural_timeframe": "D1",
        "trigger_timeframe": "D1",
        "anchor_instance_id": "anchor-1",
        "status": "TRIGGERED",
        "trigger_at": TRIGGER,
        "entry_price_ref": 210.0,
        "stop_price_ref": 205.0,
        "detector_version": "master_scan_v7",
        "run_id": "scan-0900",
        "event_at": TRIGGER,
        "observed_at": TRIGGER,
    }
    row.update(overrides)
    return row


def test_the_occurrence_key_is_deterministic_and_side_aware():
    long_row = occurrences.build_occurrence_row(_detected(), now=NOW)
    short_row = occurrences.build_occurrence_row(_detected(side="SHORT"), now=NOW)
    intraday = occurrences.build_occurrence_row(_detected(structural_timeframe="M5"), now=NOW)
    other_anchor = occurrences.build_occurrence_row(_detected(anchor_instance_id="anchor-2"), now=NOW)

    assert long_row["occurrence_id"] == schemas.occurrence_id("AAPL", "AVWAPE_TO_FIRST_DEV", "LONG", "D1", "anchor-1")
    # Long vs short, swing vs intraday, and two anchors are all distinct.
    identities = {row["occurrence_id"] for row in (long_row, short_row, intraday, other_anchor)}
    assert len(identities) == 4
    # Recomputing from the same detection gives the same key, every time.
    assert occurrences.build_occurrence_row(_detected(), now=NOW)["occurrence_id"] == long_row["occurrence_id"]


def test_an_hourly_rescan_updates_and_never_appends_an_episode(store):
    first = occurrences.record_occurrences(store, [_detected()], run_id="scan-0900", now=NOW)
    assert first.created == 1 and first.rows == 1

    # 10:00 rescan: identical detection, nothing new to say.
    quiet = occurrences.record_occurrences(store, [_detected()], run_id="scan-1000", now=NOW)
    assert quiet.unchanged == 1 and quiet.rows == 0
    assert store.read_table("setup_occurrence").num_rows == 1

    # 11:00 rescan: the detector moved the stop and advanced the status.
    moved = occurrences.record_occurrences(
        store, [_detected(status="MANAGING", stop_price_ref=207.5, run_id="scan-1100")], run_id="scan-1100", now=NOW
    )
    assert moved.revised == 1 and moved.created == 0

    counts = occurrences.episode_counts(store, 2026)
    assert counts["rows"] == 2  # two revisions...
    assert counts["occurrences"] == 1 and counts["episodes"] == 1  # ...of ONE episode

    latest = occurrences.latest_occurrences(store, 2026)
    current = next(iter(latest.values()))
    assert current["revision_id"] == "rev-2" and current["supersedes_revision_id"] == "rev-1"
    assert current["status"] == "MANAGING" and current["stop_price_ref"] == 207.5
    # The first sighting is a historical fact, carried across revisions.
    assert current["first_detected_run_id"] == "scan-0900"
    assert current["last_updated_run_id"] == "scan-1100"


def test_a_hundred_rescans_still_count_one_episode(store):
    for hour in range(100):
        occurrences.record_occurrences(store, [_detected()], run_id=f"scan-{hour}", now=NOW)
    counts = occurrences.episode_counts(store, 2026)
    assert counts["rows"] == 1 and counts["occurrences"] == 1 and counts["episodes"] == 1


def test_variants_of_one_move_share_a_dependency_cluster(store):
    detections = [
        _detected(canonical_setup_id="AVWAPE_TO_FIRST_DEV"),
        _detected(canonical_setup_id="AVWAP_BAND_BOUNCE"),
        _detected(canonical_setup_id="EXTREME_MOVE_RETEST"),
    ]
    occurrences.record_occurrences(store, detections, run_id="scan", now=NOW)
    counts = occurrences.episode_counts(store, 2026)
    # Three hypotheses about one underlying move: three occurrences, ONE
    # episode. Evidence floors count the episode.
    assert counts["occurrences"] == 3 and counts["episodes"] == 1


def test_opposite_sides_are_different_episodes(store):
    occurrences.record_occurrences(store, [_detected(), _detected(side="SHORT")], run_id="scan", now=NOW)
    assert occurrences.episode_counts(store, 2026)["episodes"] == 2


def test_the_warehouse_records_detector_state_verbatim(store):
    occurrences.record_occurrences(
        store,
        [_detected(status="STRUCTURE_ABSENT", trigger_at=None, entry_price_ref=None, tags="banger?")],
        run_id="scan",
        now=NOW,
    )
    row = store.read_table("setup_occurrence").to_pylist()[0]
    assert row["status"] == "STRUCTURE_ABSENT"  # not re-derived, not corrected
    assert row["trigger_at"] is None and row["entry_price_ref"] is None
    assert row["detector_version"] == "master_scan_v7"
    # "banger" has no schema meaning; it rides in free-text tags (LD-27).
    assert row["tags"] == "banger?"


def test_incomplete_detections_are_skipped_not_invented(store):
    report = occurrences.record_occurrences(
        store,
        [_detected(symbol=""), _detected(side="MAYBE"), _detected(canonical_setup_id="")],
        run_id="scan",
        now=NOW,
    )
    assert report.status == "NOTHING_TO_RECORD" and report.skipped == {"INCOMPLETE_DETECTION": 3}
    assert store.read_table("setup_occurrence").num_rows == 0


def test_the_slice_setups_are_the_canonical_ids():
    assert occurrences.SLICE_SETUPS == {
        "AVWAPE_TO_FIRST_DEV": "LONG",
        "POST_EARNINGS_CANDLE_BREAK": "SHORT",
    }
    from master_avwap_lib.setup_tagging import _FAMILY_TAGS

    assert set(occurrences.SLICE_SETUPS) <= set(_FAMILY_TAGS.values())


def test_bounce_events_link_only_inside_the_window():
    rows = [occurrences.build_occurrence_row(_detected(trigger_at=datetime(2026, 8, 3, 17, 0, tzinfo=UTC)), now=NOW)]
    events = [
        {"symbol": "AAPL", "bounce_at": datetime(2026, 8, 3, 17, 20, tzinfo=UTC), "bounce_type": "vwap_band"},
        {"symbol": "AAPL", "bounce_at": datetime(2026, 8, 3, 14, 0, tzinfo=UTC), "bounce_type": "ema8"},
        {"symbol": "MSFT", "bounce_at": datetime(2026, 8, 3, 17, 10, tzinfo=UTC), "bounce_type": "pdh"},
    ]
    linked = occurrences.link_bounce_events(rows, events, window_minutes=60)
    assert len(linked) == 1
    assert linked[rows[0]["occurrence_id"]]["bounce_type"] == "vwap_band"

    # A different session never links, however close the clock time looks.
    other_day = [
        {"symbol": "AAPL", "bounce_at": datetime(2026, 8, 4, 17, 5, tzinfo=UTC), "bounce_type": "vwap_band"}
    ]
    assert occurrences.link_bounce_events(rows, other_day) == {}


def test_a_detection_with_no_episode_anchor_is_rejected_not_collapsed():
    """D16: an empty episode token hashed two unrelated theses into one id.

    ``_identity_token(None)`` is "", so a March thesis and a November thesis on
    the same (symbol, setup, side, timeframe) collapsed into one occurrence_id
    and one episode, permanently.
    """
    march = _detected(
        anchor_instance_id=None,
        episode_start=None,
        trigger_at=datetime(2026, 3, 10, 20, 0, tzinfo=UTC),
        event_at=datetime(2026, 3, 10, 20, 0, tzinfo=UTC),
    )
    november = dict(march, trigger_at=datetime(2026, 11, 10, 21, 0, tzinfo=UTC))
    assert occurrences.build_occurrence_row(march, now=NOW) is None
    assert occurrences.build_occurrence_row(november, now=NOW) is None

    # Either identity source is enough; neither is invented.
    by_anchor = occurrences.build_occurrence_row(dict(march, anchor_instance_id="anchor-9"), now=NOW)
    by_episode = occurrences.build_occurrence_row(
        dict(march, episode_start=datetime(2026, 3, 2, tzinfo=UTC)), now=NOW
    )
    assert by_anchor is not None and by_episode is not None
    assert by_anchor["occurrence_id"] != by_episode["occurrence_id"]


def test_a_december_occurrence_rescanned_in_january_revises_not_duplicates(store):
    """D15: resolving from one year partition inflated episodes at the boundary."""
    december = datetime(2026, 12, 29, 21, 0, tzinfo=UTC)
    detection = _detected(trigger_at=december, event_at=december, observed_at=december)

    first = occurrences.record_occurrences(store, [detection], now=december, run_id="dec")
    assert first.created == 1

    # January rescan: same episode, a moved stop. Partitioned on event_at, the
    # existing rev-1 lives in year=2026 while the rescan runs in 2027.
    january = datetime(2027, 1, 5, 21, 0, tzinfo=UTC)
    rescan = dict(detection, stop_price_ref=207.5, run_id="jan")
    second = occurrences.record_occurrences(store, [rescan], year=2027, now=january, run_id="jan")
    assert second.created == 0 and second.revised == 1

    rows = store.read_table("setup_occurrence").to_pylist()
    assert len({row["occurrence_id"] for row in rows}) == 1, "one episode, not two"
    assert sorted(row["revision_id"] for row in rows) == ["rev-1", "rev-2"]
    current = occurrences.latest_occurrences(store, 2027)
    assert len(current) == 1
    assert next(iter(current.values()))["stop_price_ref"] == 207.5


def test_occurrences_are_disabled_without_a_store():
    assert occurrences.record_occurrences(None, [_detected()]).status == "DISABLED"
