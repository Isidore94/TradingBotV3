"""R10.E - Focus membership as episodes, not snapshots.

Audit F5: 244 of 499 (symbol, side) pairs - **49%** - appear on two or more
distinct sessions, DOCN SHORT on seven. The snapshot store cannot say whether
that is a name surviving the day roll or the trader re-adding it each morning,
and the two mean opposite things: the first is a bug in `expire_m5_if_new_day`,
the second is conviction. Episodes are what make it answerable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import focus_membership_events as fme  # noqa: E402


# ==========================================================================
# identity
# ==========================================================================
def test_the_pick_key_includes_the_category():
    """F3. `human_focus_tracking._pick_key` returns (trade_date, symbol, side)
    and callers build dicts on it, so a name on BOTH the swing and the M5 list
    silently loses one row. The CSV shows 0 multi-source keys, and that absence
    is the signature of the collision, not evidence against it."""
    swing = fme.membership_key("DOCN", "short", "swing")
    m5 = fme.membership_key("DOCN", "short", "m5")

    assert swing != m5
    assert "swing" in swing and "m5" in m5


def test_a_re_add_after_a_departure_is_a_new_episode():
    """The distinction F5 could not draw."""
    key = fme.membership_key("DOCN", "short", "m5")
    first = fme.episode_id(key, "2026-08-20T06:40:00-07:00")
    second = fme.episode_id(key, "2026-08-21T06:40:00-07:00")

    assert first != second
    assert first.startswith("DOCN-") and second.startswith("DOCN-")


def test_the_same_episode_is_stable_across_calls():
    key = fme.membership_key("AAA", "long", "m5")
    stamp = "2026-08-20T06:40:00-07:00"
    assert fme.episode_id(key, stamp) == fme.episode_id(key, stamp)


# ==========================================================================
# ownership (F4)
# ==========================================================================
def test_a_marker_means_the_machine_owns_it():
    assert fme.owner_for({"adopted_at": "x"}, markers_present=True) == fme.OWNER_MACHINE


def test_no_marker_in_a_store_that_has_markers_means_the_trader_owns_it():
    """This is what makes "user-entered names are never auto-removed"
    structural rather than aspirational."""
    assert fme.owner_for(None, markers_present=True) == fme.OWNER_TRADER


def test_no_marker_in_a_store_with_no_markers_is_unknown_legacy_never_trader():
    """F4: `focus_auto_picks.json` exists for NO historical date, so no owner is
    recoverable for any past pick. Calling those the trader's would invent
    provenance the system never recorded."""
    assert fme.owner_for(None, markers_present=False) == fme.OWNER_UNKNOWN_LEGACY


# ==========================================================================
# the events
# ==========================================================================
def test_an_expiry_is_one_row_per_name_not_a_count():
    """A single "cleared N" row would leave a survivor invisible: a name still
    on the list tomorrow would look like one that was never cleared."""
    row = fme.expired_event(
        symbol="AAA", side="long", category="m5", owner=fme.OWNER_MACHINE,
        episode="", joined_at="2026-08-20T06:40:00-07:00", at="2026-08-21T06:30:00-07:00",
    )

    assert row["event_type"] == fme.EVENT_EXPIRED
    assert row["symbol"] == "AAA"
    assert row["reason"] == "day_roll"


def test_a_departure_carries_its_age_and_bucket():
    row = fme.left_event(
        symbol="AAA", side="long", category="swing", owner=fme.OWNER_TRADER,
        episode="", joined_at="2026-08-17", left_at="2026-08-21",
        reason="focus_store.remove", sessions_on_list=5,
    )
    assert row["days_on_list"] == 5
    assert row["age_bucket"] == "3-5_sessions"


def test_an_unmeasurable_age_is_unknown_never_zero():
    assert fme.age_bucket(None) == "unknown"
    assert fme.sessions_between("", "2026-08-21") is None


def test_a_missed_snapshot_is_a_gap_never_a_reconstruction():
    """Membership is never inferred from current state. Inferring "it must have
    been there" would manufacture the very history this store exists to
    establish."""
    row = fme.observation_gap_event(
        expected_session="2026-08-19", reason="desk was off", seen_session="2026-08-18"
    )

    assert row["event_type"] == fme.EVENT_OBSERVATION_GAP
    assert "deliberately not reconstructed" in row["note"]


def test_enrichment_is_a_later_revision_not_a_rewrite():
    """Append-only stores exist so a fact learned later is recorded later
    rather than retro-fitted onto a row that predates it."""
    row = fme.enriched_event(
        episode="AAA-abc123", membership_key="AAA|long|m5", fields={"rvol": 2.4}
    )
    assert row["event_type"] == fme.EVENT_ENRICHED
    assert row["enriched_rvol"] == 2.4


def test_the_schema_is_named_never_numbered():
    assert fme.SCHEMA_FOCUS_MEMBERSHIP_EVENT == "focus_membership_event_v1"


# ==========================================================================
# the wiring: the ONE Focus writer emits, and never at the pick's expense
# ==========================================================================
@pytest.fixture
def store(tmp_path, monkeypatch):
    import focus_picks

    for name in (
        "FOCUS_LONGS_FILE", "FOCUS_SHORTS_FILE",
        "FOCUS_SWING_LONGS_FILE", "FOCUS_SWING_SHORTS_FILE",
        "LONGS_FILE", "SHORTS_FILE", "SWING_LONGS_FILE", "SHORT_SWINGS_FILE",
        "FOCUS_PICK_MEMBERSHIP_FILE", "FOCUS_AUTO_PICKS_FILE", "FOCUS_M5_STATE_FILE",
    ):
        if hasattr(focus_picks, name):
            monkeypatch.setattr(focus_picks, name, tmp_path / f"{name.lower()}.txt")
    emitted: list[dict] = []
    monkeypatch.setattr(
        focus_picks.FocusPickStore, "_emit_membership",
        lambda _self, event: emitted.append(event),
    )
    instance = focus_picks.FocusPickStore()
    instance.emitted = emitted
    return instance


def test_adding_a_pick_emits_a_joined_event(store):
    store.add("AAPL", "long", "m5")

    joined = [event for event in store.emitted if event["event_type"] == fme.EVENT_JOINED]
    assert len(joined) == 1
    assert joined[0]["symbol"] == "AAPL"
    assert joined[0]["category"] == "m5"
    assert joined[0]["membership_episode_id"].startswith("AAPL-")


def test_removing_a_pick_emits_a_left_event(store):
    store.add("AAPL", "long", "m5")
    store.emitted.clear()
    store.remove("AAPL", "long", "m5")

    left = [event for event in store.emitted if event["event_type"] == fme.EVENT_LEFT]
    assert len(left) == 1
    assert left[0]["reason"] == "focus_store.remove"


def test_the_day_roll_emits_one_expiry_per_name(store):
    from datetime import date, timedelta

    store.add("AAA", "long", "m5")
    store.add("BBB", "short", "m5")
    store.add("CCC", "long", "swing")
    store.emitted.clear()

    cleared = store.expire_m5_if_new_day(date.today() + timedelta(days=1))

    expired = [event for event in store.emitted if event["event_type"] == fme.EVENT_EXPIRED]
    assert cleared == 2
    assert sorted(event["symbol"] for event in expired) == ["AAA", "BBB"]
    # Swing is multi-day by definition and is never expired by the roll.
    assert all(event["category"] == "m5" for event in expired)


def test_an_event_failure_never_costs_the_pick(tmp_path, monkeypatch):
    """The trader's list is the product; this is evidence about it. A store
    that could cost a pick would be worse than no store."""
    import focus_picks

    for name in (
        "FOCUS_LONGS_FILE", "FOCUS_SHORTS_FILE",
        "FOCUS_SWING_LONGS_FILE", "FOCUS_SWING_SHORTS_FILE",
        "LONGS_FILE", "SHORTS_FILE", "SWING_LONGS_FILE", "SHORT_SWINGS_FILE",
        "FOCUS_PICK_MEMBERSHIP_FILE", "FOCUS_AUTO_PICKS_FILE", "FOCUS_M5_STATE_FILE",
    ):
        if hasattr(focus_picks, name):
            monkeypatch.setattr(focus_picks, name, tmp_path / f"{name.lower()}.txt")
    monkeypatch.setattr(
        "evidence_ledger.EvidenceLedger",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("ledger down")),
    )
    instance = focus_picks.FocusPickStore()

    assert instance.add("AAPL", "long", "m5") is True
    assert "AAPL" in instance.focus_symbols("long", "m5")
