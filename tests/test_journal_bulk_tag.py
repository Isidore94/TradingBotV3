"""Phase 0.13 packet P6a - the bulk tagger, and the boundary around it.

The packet's four named tests are here, plus the ones the boundary needs: this
is the single authorized machine writer into a table R7 invariant I7 gives the
trader, so the tests that matter most are the ones about what it REFUSES to do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _store(tmp_path):
    from journal_store import JournalStore

    return JournalStore(tmp_path / "journal.sqlite3")


def _seed_trade(store, trade_id="T1", symbol="AAA", status="CLOSED", trade_date="2026-08-20"):
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO trades(
                trade_id, broker, account_number, symbol, direction, status,
                opened_at, closed_at, trade_date, updated_at
            ) VALUES(?, 'QUESTRADE', '123', ?, 'LONG', ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                symbol,
                status,
                f"{trade_date}T09:31:00-07:00",
                f"{trade_date}T12:00:00-07:00",
                trade_date,
                "2026-08-20T00:00:00",
            ),
        )
    return trade_id


def _seed_candidate(store, trade_id, tag, confidence, source="setup_tracker"):
    with store.connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO auto_tag_candidates(
                trade_id, tag, confidence, source, rationale, created_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (trade_id, tag, float(confidence), source, "seeded", "2026-08-20T00:00:00"),
        )


# ---------------------------------------------------------------------------
# The four tests the packet names
# ---------------------------------------------------------------------------


def test_a_confirmed_tag_is_never_touched_by_the_bulk_apply(tmp_path):
    """I7's whole point. The trader's answer is the answer."""
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    store.save_trade_annotation(trade_id, setup_tags="my own words", notes="mine")
    _seed_candidate(store, trade_id, "machine-tag", 0.95)

    plan = bulk.build_plan(store, refresh=False)
    bulk.apply_plan(store, plan)

    state = store.annotation_state(trade_id)
    assert state["setup_tags"] == "my own words"
    assert state["tag_status"] == "confirmed"
    assert plan.already_confirmed == 1
    assert plan.to_apply == []


def test_the_store_itself_refuses_even_when_a_caller_insists(tmp_path):
    """The refusal is a boundary, not a convention the caller has to remember."""
    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    store.save_trade_annotation(trade_id, setup_tags="mine", notes="")

    assert store.apply_provisional_tags(trade_id, "machine-tag") is False
    assert store.annotation_state(trade_id)["setup_tags"] == "mine"


def test_running_it_twice_applies_nothing_new(tmp_path):
    """Idempotent over unchanged evidence: no second tag, no second audit row."""
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    _seed_candidate(store, trade_id, "avwap-reclaim", 0.90)

    first = bulk.apply_plan(store, bulk.build_plan(store, refresh=False))
    assert first["applied"] == 1
    adjustments_after_first = len(store.list_adjustments(limit=100))

    second_plan = bulk.build_plan(store, refresh=False)
    second = bulk.apply_plan(store, second_plan)

    assert second_plan.to_apply == []
    assert second == {"applied": 0, "marked": 0, "refused": 0}
    assert len(store.list_adjustments(limit=100)) == adjustments_after_first


def test_a_below_threshold_trade_gets_a_marker_and_no_setup_tag(tmp_path):
    """The refusal to guess is recorded, and it is not recorded as a tag."""
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    _seed_candidate(store, trade_id, "weak-guess", 0.31)

    plan = bulk.build_plan(store, refresh=False)
    bulk.apply_plan(store, plan)

    state = store.annotation_state(trade_id)
    assert state["setup_tags"] == ""
    assert state["tag_status"] == "needs_review"
    assert [item.action for item in plan.decisions] == ["needs_review"]


def test_the_analytics_summary_never_counts_a_provisional_tag_under_my_setups(tmp_path):
    """The one number this whole packet must not corrupt."""
    from journal_analytics import build_analytics_summary

    rows = [
        {
            "status": "CLOSED", "net_pnl": 100.0, "currency": "USD",
            "setup_tags": "machine-tag", "tag_status": "provisional",
        },
        {
            "status": "CLOSED", "net_pnl": 50.0, "currency": "USD",
            "setup_tags": "my own words", "tag_status": "confirmed",
        },
    ]
    summary = build_analytics_summary(rows)
    mine = {row["label"] for row in summary["groups"]["my setups"]}
    provisional = {row["label"] for row in summary["groups"]["provisional setups"]}

    assert "machine-tag" not in mine
    assert mine == {"my own words", "untagged"}
    assert provisional == {"machine-tag"}
    assert "provisional setups" in summary["provisional_groups"]


# ---------------------------------------------------------------------------
# The rest of the boundary
# ---------------------------------------------------------------------------


def test_the_bulk_apply_never_writes_tag_corrections(tmp_path):
    """That table is the trader's feedback. A machine writing it teaches itself."""
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    _seed_candidate(store, trade_id, "avwap-reclaim", 0.90)

    bulk.apply_plan(store, bulk.build_plan(store, refresh=False))

    assert store.list_tag_corrections() == []


def test_the_module_never_reads_an_outcome(tmp_path):
    """No tag may be derived from whether the trade made money."""
    source = (ROOT / "scripts" / "journal_bulk_tag.py").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    for banned in ("net_pnl", "gross_pnl", "pnl_usd", "r_multiple", "win_rate"):
        assert banned not in body, banned


def test_every_application_leaves_an_audit_row_naming_the_candidate(tmp_path):
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    _seed_candidate(store, trade_id, "avwap-reclaim", 0.90, source="setup_tracker")

    bulk.apply_plan(store, bulk.build_plan(store, refresh=False))

    rows = store.list_adjustments(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["action"] == "APPLY_PROVISIONAL_TAG"
    assert row["target_uid"] == trade_id
    assert row["target_kind"] == "TRADE"
    assert row["payload"]["tag"] == "avwap-reclaim"
    assert row["payload"]["confidence"] == pytest.approx(0.90)
    assert row["payload"]["packet"] == "P6a"
    assert "0.90" in row["reason"]


def test_the_audit_row_is_inert_for_assembly(tmp_path):
    """It must be readable as history and invisible to the rebuild."""
    from journal_store import PROVISIONAL_TAG_ADJUSTMENT, JournalStore

    assert PROVISIONAL_TAG_ADJUSTMENT not in JournalStore.EXECUTION_ADJUSTMENT_ACTIONS
    assert PROVISIONAL_TAG_ADJUSTMENT != "FORCE_CLOSE"
    assert PROVISIONAL_TAG_ADJUSTMENT in JournalStore.ADJUSTMENT_ACTIONS


def test_a_shape_tag_is_never_promoted_into_setup_tags(tmp_path):
    """`midday` is a fact about the clock, not an answer to "which setup?"."""
    import journal_bulk_tag as bulk
    from journal_store import TRADE_SHAPE_SOURCE

    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    _seed_candidate(store, trade_id, "midday", 1.0, source=f"{TRADE_SHAPE_SOURCE}:session")

    plan = bulk.build_plan(store, refresh=False)
    bulk.apply_plan(store, plan)

    assert store.annotation_state(trade_id)["setup_tags"] == ""
    assert plan.no_candidate == 1


def test_an_open_trade_is_left_alone(tmp_path):
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store, status="OPEN")
    _seed_candidate(store, trade_id, "avwap-reclaim", 0.95)

    plan = bulk.build_plan(store, refresh=False)
    bulk.apply_plan(store, plan)

    assert plan.considered == 0
    assert store.annotation_state(trade_id)["setup_tags"] == ""


def test_confirming_changes_the_lane_and_not_the_words(tmp_path):
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    _seed_candidate(store, trade_id, "avwap-reclaim", 0.90)
    bulk.apply_plan(store, bulk.build_plan(store, refresh=False))

    assert store.confirm_tags(trade_id) is True
    state = store.annotation_state(trade_id)
    assert state == {"setup_tags": "avwap-reclaim", "tag_status": "confirmed"}
    # And the audit row that explains where the words came from still stands.
    assert len(store.list_adjustments(limit=10)) == 1


def test_a_provisional_tag_is_not_counted_as_the_traders_vocabulary(tmp_path):
    """`distinct_tags` keeps the lanes apart, which is what the rename tool reads."""
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    _seed_candidate(store, trade_id, "avwap-reclaim", 0.90)
    bulk.apply_plan(store, bulk.build_plan(store, refresh=False))

    entry = {row["tag"]: row for row in store.distinct_tags()}["avwap-reclaim"]
    assert entry["own"] == 0
    assert entry["provisional"] == 1


def test_the_traders_save_confirms_the_row(tmp_path):
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    _seed_candidate(store, trade_id, "avwap-reclaim", 0.90)
    bulk.apply_plan(store, bulk.build_plan(store, refresh=False))

    store.save_trade_annotation(trade_id, setup_tags="my own words", notes="")
    assert store.annotation_state(trade_id) == {
        "setup_tags": "my own words",
        "tag_status": "confirmed",
    }


def test_the_dry_run_writes_no_trader_owned_row(tmp_path):
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store)
    _seed_candidate(store, trade_id, "avwap-reclaim", 0.90)

    plan = bulk.build_plan(store, refresh=False)

    assert len(plan.to_apply) == 1
    assert store.annotation_state(trade_id)["setup_tags"] == ""
    assert store.list_adjustments(limit=10) == []
    assert "Dry run" in bulk.format_plan(plan)


def test_the_histogram_is_printed_and_marks_the_threshold(tmp_path):
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    _seed_candidate(store, _seed_trade(store, "T1", "AAA"), "a", 0.90)
    _seed_candidate(store, _seed_trade(store, "T2", "BBB"), "b", 0.31)

    text = bulk.format_plan(bulk.build_plan(store, refresh=False))

    assert "0.90" in text and "0.30" in text
    assert "----- threshold 0.70 -----" in text
    # The line sits BETWEEN the two buckets, which is the whole point of drawing
    # it: the cut has to be visible in the distribution it was chosen from.
    assert text.index("0.30") < text.index("----- threshold") < text.index("0.90")


def test_the_threshold_encodes_a_same_day_same_side_match():
    """Not a round number: the score arithmetic behind it is the justification.

    The tracker or a focus favourite naming the symbol on the trade's own day
    and side clears it. The same tracker row one day later does not, and neither
    does a weaker source on the right day with no priority score behind it -
    those need more evidence, not a lower bar.
    """
    import journal_bulk_tag as bulk

    tracker_same_day_same_side = 0.28 + 0.28 + 0.16
    favourite_same_day_same_side = 0.24 + 0.28 + 0.16 + 0.08
    tracker_one_day_later_nothing_else = 0.28 + 0.22 + 0.16
    weak_source_same_day_no_priority = 0.18 + 0.28 + 0.16

    assert tracker_same_day_same_side >= bulk.DEFAULT_CONFIDENCE_THRESHOLD
    assert favourite_same_day_same_side >= bulk.DEFAULT_CONFIDENCE_THRESHOLD
    assert tracker_one_day_later_nothing_else < bulk.DEFAULT_CONFIDENCE_THRESHOLD
    assert weak_source_same_day_no_priority < bulk.DEFAULT_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Review round R1
# ---------------------------------------------------------------------------


def test_a_partly_closed_trade_is_marked_and_never_tagged(tmp_path):
    """Six live trades are CLOSED_PARTIAL and were falling through entirely.

    The OPEN rule applies - the position is not finished, so a tag is a claim
    about something still running - but they were not being MARKED either, so
    they left no trace at all and the counts silently did not add up.
    """
    import journal_bulk_tag as bulk

    store = _store(tmp_path)
    trade_id = _seed_trade(store, status="CLOSED_PARTIAL")
    _seed_candidate(store, trade_id, "avwap-reclaim", 0.95)

    plan = bulk.build_plan(store, refresh=False)
    bulk.apply_plan(store, plan)

    state = store.annotation_state(trade_id)
    assert state["setup_tags"] == "", "a half-closed trade is never tagged"
    assert state["tag_status"] == "needs_review", "but it is not invisible either"
    assert plan.partial == 1
    assert "Partly closed" in bulk.format_plan(plan)


def test_the_tagger_can_be_pointed_at_another_database(tmp_path):
    """A dry run against a copy is the safe way to try a threshold."""
    import journal_bulk_tag as bulk

    # `_store` appends its own filename, so the database is built directly here.
    from journal_store import JournalStore

    target = tmp_path / "copy.sqlite3"
    store = JournalStore(target)
    _seed_candidate(store, _seed_trade(store), "avwap-reclaim", 0.95)

    assert bulk.main(["--db", str(target), "--no-refresh"]) == 0
    # A dry run wrote nothing.
    assert store.annotation_state("T1")["setup_tags"] == ""
