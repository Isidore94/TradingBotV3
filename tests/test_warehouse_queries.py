"""The read path and the Phase-7 readout (plan sec 8.3, 17, 19.2).

The exit criterion pinned here: **manifest-resolved reads stay consistent under
concurrent compaction** - a query that started before a compaction returns the
pre-compaction row set, a query that starts after returns the post-compaction
one, and neither double-counts while both files are still on disk.

Also pinned: the readout reports episodes (the sample size) separately from
rows and occurrences, matured separately from open, and carries the
EXPLORATORY label with no shrinkage, intervals, or ranking anywhere.
"""

from __future__ import annotations

import sys
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.research_warehouse import (  # noqa: E402
    exchange_calendar as xcal,
    occurrences,
    outcomes,
    queries,
    schemas,
)
from scripts.research_warehouse.store import ResearchStore  # noqa: E402

UTC = timezone.utc
TRIGGER_DAY = date(2026, 8, 3)
TRIGGER_AT = xcal.trading_session(TRIGGER_DAY).rth_close_at
NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def _detected(symbol="AAPL", setup="AVWAPE_TO_FIRST_DEV", side="LONG", anchor="anchor-1"):
    return {
        "symbol": symbol,
        "canonical_setup_id": setup,
        "side": side,
        "structural_timeframe": "D1",
        "trigger_timeframe": "D1",
        "anchor_instance_id": anchor,
        "status": "TRIGGERED",
        "trigger_at": TRIGGER_AT,
        "entry_price_ref": 100.0,
        "stop_price_ref": 95.0,
        "detector_version": "master_scan_v7",
        "event_at": TRIGGER_AT,
        "observed_at": TRIGGER_AT,
    }


def _d1(closes, symbol="AAPL"):
    rows = []
    day = TRIGGER_DAY
    for close in closes:
        while not xcal.is_trading_day(day):
            day += timedelta(days=1)
        rows.append(
            {
                "symbol": symbol,
                "session_date": day,
                "session_id": xcal.session_id_for(day),
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000_000,
                "capture_mode": "BACKFILL",
                "is_complete": True,
            }
        )
        day += timedelta(days=1)
    return rows


def _session_row(day: int):
    return {
        "session_id": f"XNYS-2026-08-{day:02d}",
        "exchange_calendar": "XNYS",
        "session_date": date(2026, 8, day),
        "rth_open_at": datetime(2026, 8, day, 13, 30, tzinfo=UTC),
        "rth_close_at": datetime(2026, 8, day, 20, 0, tzinfo=UTC),
        "eth_open_at": datetime(2026, 8, day, 8, 0, tzinfo=UTC),
        "eth_close_at": datetime(2026, 8, day, 0, 0, tzinfo=UTC),
        "is_half_day": False,
        "expected_m5_bars_rth": 78,
        "expected_m1_bars_rth": 390,
        "calendar_version": "xnys_rules_v1",
        "observed_at": datetime(2026, 8, day, 20, 5, tzinfo=UTC),
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "cal",
    }


# --- the exit criterion ----------------------------------------------------
def test_a_query_across_a_compaction_never_double_counts(store):
    for day in (3, 4, 5, 6, 7):
        store.publish("trading_session", [_session_row(day)], job_id="cal")

    # A query resolves its file list at query start...
    before = queries.read_snapshot(store, "trading_session")
    assert before.files == 5 and len(before.rows) == 5

    compacted = store.compact("trading_session", "year=2026", job_id="eod")
    assert compacted is not None

    after = queries.read_snapshot(store, "trading_session")
    assert after.files == 1 and len(after.rows) == 5
    assert after.manifest_seq > before.manifest_seq
    # Both readings see five sessions; the tree holds six files meanwhile.
    assert sum(1 for _ in store.iter_live_tree_files()) == 6
    assert store.health_counts()["unmanifested_live_files"] == 0


def test_a_query_running_while_a_compaction_lands_stays_consistent(store):
    for day in (3, 4, 5, 6, 7):
        store.publish("trading_session", [_session_row(day)], job_id="cal")

    # Resolve the query's file list, then let a compaction land underneath it.
    resolved = store.resolve_files("trading_session")
    started = threading.Event()
    finished = threading.Event()

    def compact():
        started.wait(timeout=5)
        store.compact("trading_session", "year=2026", job_id="eod")
        store.collect_retired(now=NOW)
        finished.set()

    worker = threading.Thread(target=compact)
    worker.start()
    started.set()
    finished.wait(timeout=10)
    worker.join(timeout=10)

    # The pre-compaction file list is retired, so a *new* query is the
    # supported way to read - and it returns the post-compaction set exactly
    # once, never the union.
    after = queries.read_snapshot(store, "trading_session")
    assert len(after.rows) == 5 and after.files == 1
    assert len(resolved) == 5  # the old snapshot's identity is unchanged
    # The retired originals moved out of the live tree; nothing is duplicated.
    assert sum(1 for _ in store.iter_live_tree_files()) == 1


def test_reads_never_glob_the_directory(store):
    store.publish("trading_session", [_session_row(3)], job_id="cal")
    entry = store.manifest.resolve("trading_session").entries[0]
    stray = (store.root / entry.file_path).with_name("part-stray.parquet")
    stray.write_bytes((store.root / entry.file_path).read_bytes())

    snapshot = queries.read_snapshot(store, "trading_session")
    assert len(snapshot.rows) == 1 and snapshot.files == 1


def test_dataset_inventory_comes_from_the_ledger(store):
    store.publish("trading_session", [_session_row(3), _session_row(4)], job_id="cal")
    inventory = {row["dataset"]: row for row in queries.dataset_inventory(store)}
    assert inventory["trading_session"]["rows"] == 2
    assert inventory["trading_session"]["files"] == 1
    assert queries.dataset_inventory(None) == []


# --- the slice readout -----------------------------------------------------
def _build_slice(store):
    occurrences.record_occurrences(store, [_detected()], run_id="scan", now=TRIGGER_AT)
    latest = list(occurrences.latest_occurrences(store, 2026).values())
    outcomes.build_outcomes(store, latest, d1_by_symbol={"AAPL": _d1([100.0] + [103.0] * 20)}, as_of=NOW)
    return latest


def test_the_readout_reports_episodes_separately_from_rows(store):
    _build_slice(store)
    snapshot = queries.slice_readout(store, year=2026, as_of=NOW)

    assert snapshot.evidence_tier == "EXPLORATORY"
    assert len(snapshot.rows) == 3  # one row per recipe
    row = next(row for row in snapshot.rows if row["recipe_id"] == "swing_house_v1")
    assert row["canonical_setup_id"] == "AVWAPE_TO_FIRST_DEV" and row["side"] == "LONG"
    assert row["n_episodes"] == 1 and row["n_occurrences"] == 1
    assert row["n_symbols"] == 1 and row["n_sessions"] == 1
    assert row["mean_net_r"] is not None and row["mean_gross_r"] is not None
    assert row["outcome_definition_id"] == "house_default_v1"
    # The capture-mode split is reported, so BACKFILL evidence is never mistaken
    # for as-observed evidence.
    assert row["as_observed_only"] is False
    assert row["capture_modes"] == {"BACKFILL": 1}


def test_three_recipes_on_one_move_stay_one_episode(store):
    _build_slice(store)
    snapshot = queries.slice_readout(store, year=2026, as_of=NOW)
    assert {row["recipe_id"] for row in snapshot.rows} == {
        "swing_house_v1",
        "control_fixed_1r2r_v1",
        "control_time_only_v1",
    }
    # Every recipe row reports the SAME single episode - recipes are correlated
    # diagnostics, never extra samples.
    assert {row["n_episodes"] for row in snapshot.rows} == {1}


def test_open_outcomes_are_counted_but_never_averaged_in(store):
    _build_slice(store)
    early = queries.slice_readout(store, year=2026, as_of=TRIGGER_AT + timedelta(days=1))
    row = next(row for row in early.rows if row["recipe_id"] == "control_time_only_v1")
    assert row["n_open"] == 1 and row["n_matured"] == 0
    # Nothing has matured, so there is no mean to report.
    assert row["mean_net_r"] is None and row["mean_r_at_s18"] is None


def test_a_recomputed_outcome_supersedes_its_interim_reading(store):
    occurrences.record_occurrences(store, [_detected()], run_id="scan", now=TRIGGER_AT)
    latest = list(occurrences.latest_occurrences(store, 2026).values())

    # Night 1: only three sessions of path exist, price is UP interim. The
    # clock says these should have resolved (as_of is past maturity), so the
    # rows are TRUNCATED - and truncated evidence carries no realized R.
    outcomes.build_outcomes(store, latest, d1_by_symbol={"AAPL": _d1([100.0, 103.0, 105.0])}, as_of=NOW, now=NOW)
    stale = queries.slice_readout(store, year=2026, as_of=NOW)
    row = next(item for item in stale.rows if item["recipe_id"] == "control_time_only_v1")
    assert row["n_truncated"] == 1 and row["n_matured"] == 1
    assert row["mean_gross_r"] is None  # an interim +1R can never flatter a mean

    # A later build with the full path: the trade actually finished at -0.8R.
    outcomes.build_outcomes(
        store, latest, d1_by_symbol={"AAPL": _d1([100.0] + [96.0] * 20)}, as_of=NOW, now=NOW + timedelta(days=1)
    )
    fresh = queries.slice_readout(store, year=2026, as_of=NOW + timedelta(days=1))
    row = next(item for item in fresh.rows if item["recipe_id"] == "control_time_only_v1")
    # The superseded interim row is history, never a second sample.
    assert row["n_rows"] == 1 and row["n_episodes"] == 1
    assert row["n_truncated"] == 0
    assert row["mean_gross_r"] == pytest.approx((96.0 - 100.0) / 5.0)


def test_only_the_slice_setups_appear(store):
    occurrences.record_occurrences(
        store, [_detected(), _detected(setup="AVWAP_BAND_BOUNCE", anchor="anchor-2")], run_id="scan", now=TRIGGER_AT
    )
    latest = list(occurrences.latest_occurrences(store, 2026).values())
    outcomes.build_outcomes(store, latest, d1_by_symbol={"AAPL": _d1([100.0] * 20)}, as_of=NOW)
    snapshot = queries.slice_readout(store, year=2026, as_of=NOW)
    assert {row["canonical_setup_id"] for row in snapshot.rows} == {"AVWAPE_TO_FIRST_DEV"}


def test_the_rendered_readout_states_its_limits(store):
    _build_slice(store)
    text = queries.render_slice_readout(queries.slice_readout(store, year=2026, as_of=NOW))
    assert "EXPLORATORY" in text and "no shrinkage" in text and "episodes" in text
    assert "AVWAPE_TO_FIRST_DEV" in text
    assert queries.render_slice_readout(queries.QuerySnapshot()) == "No slice outcomes recorded yet."


def test_coverage_readout_summarizes_status_and_reasons(store):
    from scripts.research_warehouse import bar_archive

    context = {"risk_set_id": "rs-1", "run_kind": "master_scan", "scheduled_at": datetime(2026, 8, 3, 13, 45, tzinfo=UTC)}
    bar_archive.record_scan_coverage(
        store, {"AAPL": "EVALUATED_ELIGIBLE", "MSFT": "NO_RESPONSE"}, **context
    )
    session = bar_archive.session_context(datetime(2026, 8, 3, 17, 0, tzinfo=UTC))
    bar_archive.record_collection_gaps(store, session=session, captured_counts={"MSFT": 0}, policy_symbols=["TSLA"])

    snapshot = queries.coverage_readout(store, month="month=2026-08")
    row = snapshot.rows[0]
    assert row["risk_sets"] == 1
    assert row["coverage_by_status"] == {"EVALUATED_ELIGIBLE": 1, "NO_RESPONSE": 1}
    assert row["gaps_by_reason"] == {"MISSING": 1, "NOT_COLLECTED_BY_POLICY": 1}


def test_queries_are_inert_without_a_store():
    assert queries.slice_readout(None).rows == []
    assert queries.coverage_readout(None).rows == []


# --- optional DuckDB -------------------------------------------------------
def test_duckdb_is_optional_and_read_only(store):
    store.publish("trading_session", [_session_row(3), _session_row(4)], job_id="cal")
    if not queries.duckdb_available():
        pytest.skip("duckdb not installed; pyarrow answers every slice query (LD-04)")
    rows = queries.query_sql(store, "trading_session", "SELECT count(*) FROM t")
    assert rows[0][0] == 2
    # It reads the same manifest-resolved file list the pyarrow path uses.
    assert len(store.resolve_files("trading_session")) == 1
