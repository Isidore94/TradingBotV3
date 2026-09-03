"""P10 B2/B3 - a like knows which occurrence it was, and what it looked like.

Trader, 2026-09-02: *"anytime I like a D1 it should be treated with respect by
the bot in regards to finding out what's good about it, how we can replicate
those searches, and then how we can improve the entries. if I like a stock one
day it may not be for 3-5 days later that the best entry is."*

Two joins that did not exist on 2026-09-02: a like to a warehouse occurrence, and
an occurrence to the feature snapshot of its own session (the round-1 audit's
item 6). Both are READ-ONLY over the gold datasets and shadow throughout.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.research_warehouse import like_links, queries  # noqa: E402
from scripts.research_warehouse.schemas import SCHEMA_VERSION  # noqa: E402
from scripts.research_warehouse.store import ResearchStore  # noqa: E402

UTC = timezone.utc


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def _occurrence(
    occurrence_id,
    *,
    symbol="NVDA",
    setup="AVWAP_BREAKOUT",
    side="LONG",
    trigger,
):
    return {
        "occurrence_id": occurrence_id,
        "symbol": symbol,
        "canonical_setup_id": setup,
        "side": side,
        "structural_timeframe": "D1",
        "trigger_timeframe": "D1",
        "anchor_instance_id": f"anchor-{occurrence_id}",
        "dependency_cluster_id": f"cluster-{symbol}",
        "status": "TRIGGERED",
        "trigger_at": trigger,
        "trigger_bar_interval_start": trigger,
        "entry_price_ref": 100.0,
        "stop_price_ref": 95.0,
        "detector_version": "test",
        "first_detected_run_id": "run-1",
        "last_updated_run_id": "run-1",
        "tags": "",
        "event_at": trigger,
        "observed_at": trigger,
        "computed_at": trigger,
        "revision_id": f"rev-{occurrence_id}",
        "supersedes_revision_id": "",
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-1",
    }


def _like(event_id="e1", *, symbol="NVDA", side="LONG", date="2026-09-02", family=""):
    row = {
        "event_id": event_id,
        "event_type": "like_claim",
        "symbol": symbol,
        "side": side,
        "session_date": date,
    }
    if family:
        row["canonical_setup_id"] = family
    return row


# ---------------------------------------------------------------------------
# B2 - the like -> occurrence link
# ---------------------------------------------------------------------------


def test_the_same_family_wins_over_a_nearer_different_one(store):
    """The click recorded which search found it; that beats proximity."""
    trigger_day = datetime(2026, 9, 2, 14, 30, tzinfo=UTC)
    store.publish(
        "setup_occurrence",
        [
            _occurrence("near-other", setup="PULLBACK_TO_SMA50", trigger=trigger_day),
            _occurrence(
                "far-same",
                setup="AVWAP_BREAKOUT",
                trigger=trigger_day + timedelta(days=3),
            ),
        ],
        job_id="p10",
    )

    link = like_links.link_one_like(store, _like(family="AVWAP_BREAKOUT"))

    assert link.match_basis == like_links.BASIS_EXACT_FAMILY
    assert link.occurrence_id == "far-same"
    assert link.candidates_in_window == 2


def test_with_no_family_the_nearest_trigger_wins(store):
    """The trader liked the CHART; the family is the scanner's opinion."""
    like_day = datetime(2026, 9, 2, 14, 30, tzinfo=UTC)
    store.publish(
        "setup_occurrence",
        [
            _occurrence("same-day", setup="PULLBACK_TO_SMA50", trigger=like_day),
            _occurrence("later", setup="AVWAP_BREAKOUT", trigger=like_day + timedelta(days=4)),
        ],
        job_id="p10",
    )

    link = like_links.link_one_like(store, _like())

    assert link.match_basis == like_links.BASIS_ANY_FAMILY
    assert link.occurrence_id == "same-day"


def test_a_like_with_no_occurrence_is_written_with_basis_none(store):
    """Absence is a first-class fact.

    A study that silently dropped the unmatched likes would report on the subset
    the scanner happened to find - which is precisely the population whose
    behaviour differs from the rest.
    """
    store.publish(
        "setup_occurrence",
        [_occurrence("far-away", trigger=datetime(2026, 12, 1, 14, 30, tzinfo=UTC))],
        job_id="p10",
    )

    link = like_links.link_one_like(store, _like())

    assert link is not None, "a like with no match is still a row"
    assert link.match_basis == like_links.BASIS_NONE
    assert link.occurrence_id == ""
    assert link.candidates_in_window == 0


def test_the_window_reaches_one_session_back_and_five_forward(store):
    """*"if I like a stock one day it may not be for 3-5 days later."*"""
    like_day = datetime(2026, 9, 2, 14, 30, tzinfo=UTC)
    store.publish(
        "setup_occurrence",
        [
            _occurrence("too-early", trigger=like_day - timedelta(days=3)),
            _occurrence("too-late", trigger=like_day + timedelta(days=9)),
        ],
        job_id="p10",
    )
    assert like_links.link_one_like(store, _like()).match_basis == "none"

    store.publish(
        "setup_occurrence",
        [_occurrence("in-window", trigger=like_day + timedelta(days=5))],
        job_id="p10",
    )
    assert like_links.link_one_like(store, _like()).occurrence_id == "in-window"


def test_the_other_side_is_never_matched(store):
    """A long like and a short occurrence are two different theses."""
    store.publish(
        "setup_occurrence",
        [_occurrence("short-one", side="SHORT", trigger=datetime(2026, 9, 2, 14, 30, tzinfo=UTC))],
        job_id="p10",
    )

    assert like_links.link_one_like(store, _like()).match_basis == "none"


def test_a_like_this_join_cannot_address_is_skipped_not_invented(store):
    """Skipped is not the same as unmatched, and must not enter the denominator."""
    links = like_links.link_likes(
        store,
        [
            _like("e1", side=""),
            _like("e2", symbol=""),
            {"event_id": "", "symbol": "NVDA", "side": "LONG", "session_date": "2026-09-02"},
            _like("e4"),
        ],
    )

    assert [link.event_id for link in links] == ["e4"]


def test_the_basis_counts_always_name_every_basis(store):
    """A missing key cannot distinguish "none tonight" from "never computed"."""
    counts = like_links.basis_counts([])
    assert set(counts) == set(like_links.MATCH_BASES)
    assert all(value == 0 for value in counts.values())


def test_the_bronze_rows_hash_their_payload_so_a_rerun_is_idempotent():
    link = like_links.LikeLink(
        event_id="e1",
        symbol="NVDA",
        side="LONG",
        like_date="2026-09-02",
        occurrence_id="occ-1",
        canonical_setup_id="AVWAP_BREAKOUT",
        trigger_at="2026-09-02T14:30:00+00:00",
        match_basis=like_links.BASIS_EXACT_FAMILY,
        candidates_in_window=2,
    )
    moment = datetime(2026, 9, 3, 4, 0, tzinfo=UTC)

    first = like_links.link_rows_for_bronze([link], observed_at=moment)
    again = like_links.link_rows_for_bronze([link], observed_at=moment)

    assert first == again
    assert first[0]["record_hash"] == again[0]["record_hash"]
    assert first[0]["source_artifact"] == "like_occurrence_link"


def test_the_read_is_narrowed_arrow_side_never_a_materialised_list(store, monkeypatch):
    """BD-74. A month-keyed partition read whole is what put the desk at 10 GB."""
    seen = {}
    original = ResearchStore.read_rows

    def _watched(self, dataset, partition=None, **kwargs):
        seen[dataset] = kwargs
        return original(self, dataset, partition, **kwargs)

    monkeypatch.setattr(ResearchStore, "read_rows", _watched)
    like_links.link_one_like(store, _like())

    assert seen["setup_occurrence"]["symbols"] == ["NVDA"]
    assert seen["setup_occurrence"]["interval_start_range"] is not None


# ---------------------------------------------------------------------------
# B3 - the occurrence -> feature snapshot join
# ---------------------------------------------------------------------------


def _snapshot(session_date, *, symbol="NVDA", close=100.0, computed_at=None):
    return {
        "symbol": symbol,
        "session_date": session_date,
        "feature_set_version": "v1",
        "close": close,
        "atr14": 2.5,
        "dist_sma50_atr": 1.2,
        "dist_sma200_atr": 3.4,
        "spy_regime_state": "bullish",
        "input_manifest_hash": "",
        "computed_at": computed_at or datetime.combine(session_date, datetime.min.time()).replace(tzinfo=UTC),
        "event_at": datetime.combine(session_date, datetime.min.time()).replace(tzinfo=UTC),
        "input_capture_mode_worst": "LIVE",
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-1",
    }


def test_it_takes_the_occurrences_own_session_and_never_a_later_one(store):
    """A snapshot from the day after the trigger knows how it turned out."""
    from datetime import date

    trigger = datetime(2026, 9, 2, 14, 30, tzinfo=UTC)
    store.publish("setup_occurrence", [_occurrence("occ-1", trigger=trigger)], job_id="p10")
    store.publish(
        "feature_snapshot_daily",
        [
            _snapshot(date(2026, 9, 1), close=99.0),
            _snapshot(date(2026, 9, 2), close=100.0),
            _snapshot(date(2026, 9, 3), close=115.0),
        ],
        job_id="p10",
    )

    features = queries.occurrence_features(store, ["occ-1"])

    assert features["occ-1"]["close"] == 100.0, "the trigger session, not the next"
    assert features["occ-1"]["atr14"] == 2.5
    assert features["occ-1"]["spy_regime_state"] == "bullish"


def test_a_later_revision_of_the_right_session_is_refused_too(store):
    """A revision computed after the trigger is as much of a leak as a later day."""
    from datetime import date

    trigger = datetime(2026, 9, 2, 14, 30, tzinfo=UTC)
    store.publish("setup_occurrence", [_occurrence("occ-1", trigger=trigger)], job_id="p10")
    store.publish(
        "feature_snapshot_daily",
        [
            _snapshot(
                date(2026, 9, 2),
                close=100.0,
                computed_at=datetime(2026, 9, 2, 9, 0, tzinfo=UTC),
            ),
            _snapshot(
                date(2026, 9, 2),
                close=140.0,
                computed_at=datetime(2026, 9, 4, 4, 0, tzinfo=UTC),
            ),
        ],
        job_id="p10",
    )

    features = queries.occurrence_features(store, ["occ-1"])

    assert features["occ-1"]["close"] == 100.0, "the revision that knew the future won"


def test_an_occurrence_with_no_usable_snapshot_is_absent_not_null(store):
    """"Measured and flat" and "never measured" must not look the same."""
    from datetime import date

    trigger = datetime(2026, 9, 2, 14, 30, tzinfo=UTC)
    store.publish("setup_occurrence", [_occurrence("occ-1", trigger=trigger)], job_id="p10")
    store.publish(
        "feature_snapshot_daily", [_snapshot(date(2026, 9, 10))], job_id="p10"
    )

    assert queries.occurrence_features(store, ["occ-1"]) == {}


def test_no_ids_reads_nothing_at_all(store, monkeypatch):
    reads = []
    original = ResearchStore.read_rows
    monkeypatch.setattr(
        ResearchStore,
        "read_rows",
        lambda self, dataset, partition=None, **kwargs: reads.append(dataset)
        or original(self, dataset, partition, **kwargs),
    )

    assert queries.occurrence_features(store, []) == {}
    assert reads == []


def test_the_nightly_pass_publishes_the_link_dataset_it_is_documented_to_write(
    store, tmp_path, monkeypatch
):
    """R4 A4: `link_rows_for_bronze` had no production caller.

    The ERD, the CHANGELOG and gate 42 all say `bronze_like_occurrence_link` is
    written nightly, and BD-92 makes it the ONLY route from an after-like
    outcome row back to the setup family behind it - those rows are keyed by the
    like episode and carry no family. The dataset was never written, so the
    documented join did not exist. Do not retract the claim; make it true.
    """
    import json
    from datetime import datetime, timedelta, timezone

    import project_paths
    from scripts.research_warehouse import cli

    utc = timezone.utc
    trigger = datetime(2026, 9, 2, 14, 30, tzinfo=utc)
    store.publish(
        "setup_occurrence",
        [_occurrence("occ-1", trigger=trigger)],
        job_id="tee",
    )

    log = tmp_path / "trader_annotations.jsonl"
    rows = [
        dict(_like("e1"), schema="trader_annotation_v1", ts=trigger.isoformat()),
        dict(_like("e2", symbol="AMD"), schema="trader_annotation_v1", ts=trigger.isoformat()),
    ]
    log.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(project_paths, "TRADER_ANNOTATIONS_FILE", log, raising=False)

    step = cli._run_after_like_pass(
        store,
        {},
        stamp=trigger + timedelta(days=1),
        run_id="r4",
    )

    assert step["status"] == "ok", step
    assert step["link_status"] == "ok", step
    assert step["link_rows"] == 2, step

    published = store.read_table("bronze_like_occurrence_link").to_pylist()
    assert len(published) == 2
    payloads = {json.loads(row["payload"])["event_id"] for row in published}
    assert payloads == {"e1", "e2"}
    # A like the scanner never found is written with basis `none`, not dropped -
    # that population is precisely the one whose behaviour differs.
    bases = {json.loads(row["payload"])["match_basis"] for row in published}
    assert "none" in bases


def test_a_second_night_on_an_unchanged_lake_writes_no_duplicate_link(
    store, tmp_path, monkeypatch
):
    """The record hash is over the payload, so a re-run is a no-op."""
    import json
    from datetime import datetime, timedelta, timezone

    import project_paths
    from scripts.research_warehouse import cli

    utc = timezone.utc
    trigger = datetime(2026, 9, 2, 14, 30, tzinfo=utc)
    log = tmp_path / "trader_annotations.jsonl"
    log.write_text(
        json.dumps(dict(_like("e1"), schema="trader_annotation_v1", ts=trigger.isoformat()))
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(project_paths, "TRADER_ANNOTATIONS_FILE", log, raising=False)

    stamp = trigger + timedelta(days=1)
    cli._run_after_like_pass(store, {}, stamp=stamp, run_id="r4")
    first = store.read_table("bronze_like_occurrence_link").num_rows
    cli._run_after_like_pass(store, {}, stamp=stamp, run_id="r4")

    assert store.read_table("bronze_like_occurrence_link").num_rows == first == 1
