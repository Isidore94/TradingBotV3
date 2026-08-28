"""Backup classes, restore verification, and the build job (plan Phase 8).

Pinned here: a Class B copy never propagates a deletion, the scripted restore
check re-verifies every file against the hash the manifest recorded and runs a
canned query against the restored copy, the build job refuses a second
concurrent run, and the six Health tiles report the plan's six things and
nothing more.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.research_warehouse import backup, cli, schemas  # noqa: E402
from scripts.research_warehouse.store import ResearchStore  # noqa: E402

UTC = timezone.utc
NOW = datetime(2026, 8, 4, 2, 0, tzinfo=UTC)


def _bar(symbol="AAPL", minute=0):
    start = datetime(2026, 8, 3, 13, 30, tzinfo=UTC) + timedelta(minutes=minute)
    return {
        "symbol": symbol,
        "interval_start": start,
        "interval_end": start + timedelta(minutes=5),
        "session_id": "XNYS-2026-08-03",
        "session_phase": "RTH",
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 1000, "vwap": None, "trade_count": None,
        "provider": "IBKR", "is_complete": True, "quality": "COMPLETE", "source_hash": "",
        "event_at": start + timedelta(minutes=5), "observed_at": start + timedelta(minutes=6),
        "capture_mode": "LIVE", "revision_id": "", "supersedes_revision_id": "",
        "schema_version": schemas.SCHEMA_VERSION, "run_id": "tee",
    }


@pytest.fixture()
def store(tmp_path):
    target = ResearchStore.open(tmp_path / "lake")
    target.publish("bar_m5", [_bar(minute=index) for index in range(6)], job_id="tee")
    return target


# --- backup ---------------------------------------------------------------
def test_class_a_mirrors_to_every_target(store, tmp_path):
    disk, drive = tmp_path / "backup_disk", tmp_path / "drive_mirror"
    report = backup.backup_class_a(store, [disk, drive], now=NOW)

    assert report.status == "OK" and report.files_copied >= 2
    for destination in (disk, drive):
        assert (destination / "20260804" / "manifest_log.jsonl").is_file()
    assert report.deleted_from_target == 0


def test_class_b_is_append_only_and_never_propagates_a_deletion(store, tmp_path):
    target = tmp_path / "second_disk"
    first = backup.backup_class_b(store, target, now=NOW)
    assert first.files_copied >= 1
    copied = sorted(target.rglob("*.parquet"))
    assert copied

    # The source loses a file (mistake, corruption, bad cleanup)...
    for path in store.root.rglob("*.parquet"):
        path.unlink()
    second = backup.backup_class_b(store, target, now=NOW)

    # ...and the backup still has it. That is the whole point of a backup.
    assert sorted(target.rglob("*.parquet")) == copied
    assert second.deleted_from_target == 0


def test_a_second_backup_run_skips_unchanged_files(store, tmp_path):
    target = tmp_path / "disk"
    backup.backup_class_b(store, target, now=NOW)
    again = backup.backup_class_b(store, target, now=NOW)
    assert again.files_copied == 0 and again.files_skipped > 0


def test_backup_is_disabled_without_a_store_or_target(store, tmp_path):
    assert backup.backup_class_a(None, [tmp_path]).status == "DISABLED"
    assert backup.backup_class_b(store, None).status == "NO_TARGET"


# --- the scripted restore check -------------------------------------------
def test_restore_check_verifies_hashes_and_runs_a_canned_query(store, tmp_path):
    report = backup.restore_check(store, tmp_path / "restored", dataset="bar_m5", partition="month=2026-08")

    assert report.passed and report.status == "OK"
    assert report.files == 1 and report.rows == 6
    assert report.query_rows == 6  # the canned query ran against the restored copy
    assert not report.hash_mismatches and not report.missing
    # Restores go to a NEW root; the live lake is untouched.
    assert Path(report.restored_to) != store.root
    assert store.read_table("bar_m5").num_rows == 6


def test_a_corrupted_file_fails_the_restore_check(store, tmp_path, monkeypatch):
    original = backup._sha256
    entry = store.manifest.resolve("bar_m5").entries[0]

    def wrong_hash(path):
        return "0" * 64 if path.name == Path(entry.file_path).name else original(path)

    monkeypatch.setattr(backup, "_sha256", wrong_hash)
    report = backup.restore_check(store, tmp_path / "restored", dataset="bar_m5")
    assert report.status == "FAILED" and report.hash_mismatches and not report.passed


def test_a_missing_file_fails_the_restore_check(store, tmp_path):
    entry = store.manifest.resolve("bar_m5").entries[0]
    (store.root / entry.file_path).unlink()
    report = backup.restore_check(store, tmp_path / "restored", dataset="bar_m5")
    assert report.status == "FAILED" and report.missing == [entry.file_path]


def test_restore_check_on_an_empty_dataset_says_so(store, tmp_path):
    report = backup.restore_check(store, tmp_path / "restored", dataset="bar_d1")
    assert report.status == "NOTHING_TO_RESTORE" and not report.passed


# --- the build job ---------------------------------------------------------
def test_a_second_build_refuses_rather_than_racing(store, tmp_path):
    lock = tmp_path / "build.lock"
    with cli.single_flight(lock):
        report = cli.run_build(store, session_date=date(2026, 8, 3), now=NOW, lock_path=lock)
    assert report.status == "REFUSED" and "already running" in report.message


def test_a_dead_holders_lock_is_reclaimed_not_obeyed(store, tmp_path):
    lock = tmp_path / "build.lock"
    lock.write_text(json.dumps({"pid": 2**22, "started_at": "2026-08-04T01:00:00+00:00"}), encoding="utf-8")

    with cli.single_flight(lock):
        assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert not lock.exists()


def test_the_build_job_is_idempotent(store, tmp_path):
    lock = tmp_path / "build.lock"
    first = cli.run_build(store, session_date=date(2026, 8, 3), now=NOW, lock_path=lock)
    assert first.status == "OK"
    rows_after_first = store.read_table("bar_derived").num_rows

    second = cli.run_build(store, session_date=date(2026, 8, 3), now=NOW, lock_path=lock)
    assert second.status == "OK"
    assert store.read_table("bar_derived").num_rows == rows_after_first


def test_the_build_job_runs_the_whole_step_list(store, tmp_path, monkeypatch):
    """D19: the EOD build never ran the D1 wrap, features, outcomes or backups.

    As shipped it stopped at derived/weekly, so a night of capture produced no
    ``bar_d1``, no feature snapshots, no outcomes and no backup at all -- an
    undeclared gap, unlike the tee (BD-20) and the adapter (BD-44).
    """
    lock = tmp_path / "build.lock"
    day = date(2026, 8, 3)
    class_a = tmp_path / "backup_a"
    class_b = tmp_path / "backup_b"
    monkeypatch.setattr(cli.config, "backup_class_a_dirs", lambda: [class_a])
    monkeypatch.setattr(cli.config, "backup_class_b_dir", lambda: class_b)

    report = cli.run_build(store, session_date=day, now=NOW, lock_path=lock)
    assert report.status == "OK"

    expected = [
        "reconcile", "spool", "bronze", "snapshots", "bar_d1", "sessions",
            "derived", "weekly", "anchors", "features_daily", "features_intraday",
            "occurrences", "outcomes", "backups", "retired",
    ]
    assert list(report.steps) == expected, "the step list is a dependency order"

    # Each new step actually ran rather than being silently absent.
    # NO_SOURCE here: the durable D1 store is a desk artifact, absent in tests.
    assert report.steps["bar_d1"]["status"] in {"OK", "NOTHING_TO_COMPUTE", "NO_SOURCE"}
    assert report.steps["bar_d1"]["dataset"] == "bar_d1"
    assert report.steps["anchors"]["dataset"] == "anchor_instance"
    assert report.steps["features_daily"]["dataset"] == "feature_snapshot_daily"
    assert report.steps["features_intraday"]["dataset"] == "feature_snapshot_intraday"
    # Occurrence ingestion stays blocked on BD-44 -- skipped cleanly, and said so.
    assert report.steps["outcomes"]["status"] == "NO_OCCURRENCES"
    assert "BD-44" in report.steps["outcomes"]["message"]
    # Backups ran against the configured targets.
    assert report.steps["backups"]["class_a"]["status"] == "OK"
    assert report.steps["backups"]["class_b"]["status"] == "OK"
    assert class_b.exists()

    # And the whole enlarged list is still a no-op on a re-run.
    before = {name: store.read_table(name).num_rows for name in ("bar_derived", "bar_d1", "anchor_instance")}
    again = cli.run_build(store, session_date=day, now=NOW, lock_path=lock)
    assert again.status == "OK"
    assert {name: store.read_table(name).num_rows for name in before} == before


def test_backups_no_op_with_a_clear_message_when_unconfigured(store, tmp_path, monkeypatch):
    """A backup written somewhere nobody chose is not a backup."""
    monkeypatch.setattr(cli.config, "backup_class_a_dirs", lambda: [])
    monkeypatch.setattr(cli.config, "backup_class_b_dir", lambda: None)

    report = cli.run_build(store, session_date=date(2026, 8, 3), now=NOW, lock_path=tmp_path / "lock")
    assert report.status == "OK"
    for cls in ("class_a", "class_b"):
        step = report.steps["backups"][cls]
        assert step["status"] == "NO_TARGET"
        assert "local_settings.json" in step["message"]


def test_the_anchor_step_reads_current_and_previous_from_bronze(store):
    """LD-09 scopes the slice to current + previous earnings anchors."""
    rows = []
    for offset, anchor_day in enumerate(("2026-01-28", "2026-04-29", "2026-07-30")):
        rows.append(
            {
                "source_artifact": "earnings_avwap_anchors",
                "source_path": "earnings_avwap_anchors.csv",
                "source_sha256": f"sha{offset}",
                "source_offset": offset,
                "record_hash": f"rec{offset}",
                "legacy_id": f"AAPL|{anchor_day}",
                "payload": json.dumps({"ticker": "AAPL", "anchor_date": anchor_day, "side": "LONG"}),
                # A CSV row is wrapped as CSV_ROW with a JSON payload text.
                "payload_format": schemas.BRONZE_FORMAT_CSV_ROW,
                "quality": "COMPLETE",
                "event_at": NOW,
                "observed_at": NOW,
                "partition_ts": NOW,
                "capture_mode": "RECONSTRUCTED",
                "schema_version": schemas.SCHEMA_VERSION,
                "run_id": "bronze",
            }
        )
    store.publish("bronze_earnings_avwap_anchors", rows, job_id="bronze")

    anchors = cli.anchors_from_bronze(store)
    assert [(item["anchor_type"], item["anchor_bar_date"].isoformat()) for item in anchors] == [
        ("EARNINGS_CURRENT", "2026-07-30"),
        ("EARNINGS_PREVIOUS", "2026-04-29"),
    ]


def test_the_build_job_is_a_no_op_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.ResearchStore, "open", classmethod(lambda cls, root=None: None))
    report = cli.run_build(lock_path=tmp_path / "lock")
    assert report.status == "DISABLED" and "no-op" in report.message
    assert cli.run_status()["enabled"] is False


def test_status_reports_the_ledger_not_the_bars(store):
    status = cli.run_status(store)
    assert status["enabled"] is True
    assert status["health"]["live_files"] >= 1
    assert any(row["dataset"] == "bar_m5" for row in status["datasets"])


# --- the six Health tiles --------------------------------------------------
def test_there_are_exactly_six_tiles_and_they_are_the_plans_six(store, tmp_path):
    from ui.services.warehouse_service import warehouse_health_tiles

    tiles = warehouse_health_tiles(store, now=NOW, backup_root=tmp_path / "absent")
    assert [tile.key for tile in tiles] == [
        "das_mount",
        "backup",
        "coverage",
        "spool",
        "last_seal",
        "manifest_integrity",
    ]
    integrity = tiles[-1]
    assert integrity.status == "OK" and integrity.metrics["unmanifested"] == 0
    seal = tiles[4]
    assert seal.value == "bar_m5"


def test_a_never_backed_up_lake_shows_red(store, tmp_path):
    from ui.services.warehouse_service import warehouse_health_tiles

    tiles = {tile.key: tile for tile in warehouse_health_tiles(store, now=NOW, backup_root=None)}
    assert tiles["backup"].status == "RED" and tiles["backup"].value == "never"


def test_an_unmanifested_file_turns_the_integrity_tile_red(store, tmp_path):
    from ui.services.warehouse_service import warehouse_health_tiles

    entry = store.manifest.resolve("bar_m5").entries[0]
    (store.root / entry.file_path).with_name("part-stray.parquet").write_bytes(b"not parquet")
    tiles = {tile.key: tile for tile in warehouse_health_tiles(store, now=NOW)}
    assert tiles["manifest_integrity"].status == "RED"
    assert tiles["manifest_integrity"].metrics["unmanifested"] == 1


def test_policy_absence_is_not_a_coverage_defect(store):
    from scripts.research_warehouse import bar_archive
    from ui.services.warehouse_service import warehouse_health_tiles

    session = bar_archive.session_context(datetime(2026, 8, 3, 17, 0, tzinfo=UTC))
    bar_archive.record_collection_gaps(store, session=session, policy_symbols=["TSLA", "AMD"])
    tiles = {tile.key: tile for tile in warehouse_health_tiles(store, now=NOW)}
    # Two policy gaps, zero coverage defects: they are a declared decision.
    assert tiles["coverage"].status == "OK" and tiles["coverage"].value == "complete"
    assert tiles["coverage"].metrics["by_reason"]["NOT_COLLECTED_BY_POLICY"] == 2


def test_tiles_collapse_to_one_message_when_the_warehouse_is_off(monkeypatch):
    import research_warehouse.store as store_module
    from ui.services.warehouse_service import warehouse_health_tiles

    monkeypatch.setattr(store_module.ResearchStore, "open", classmethod(lambda cls, root=None: None))
    tiles = warehouse_health_tiles(now=NOW)
    assert len(tiles) == 1 and tiles[0].status == "OFF"
    assert "not configured" in tiles[0].value


def test_the_build_job_adds_no_process_of_its_own():
    from ui.services.warehouse_service import register_build_job

    registered = []
    descriptor = register_build_job(type("S", (), {"register_job": lambda self, d: registered.append(d)})())
    assert descriptor["single_flight"] is True and descriptor["owner"] == "main_desktop"
    assert descriptor["entry_point"].startswith("python -m scripts.research_warehouse.cli")
    assert registered == [descriptor]


def test_outcome_simulation_reads_each_occurrences_own_month(store):
    """BD-69: M5 bars came from the build day's month alone.

    `known` spans two years of occurrences and BD-53 re-simulates every
    non-terminal one on every build, so an intraday occurrence from an earlier
    month was fed an empty archive each night and concluded from that absence.
    """
    build_day = date(2026, 8, 3)
    known = {
        "occ-old": {"symbol": "AAPL", "trigger_at": datetime(2026, 5, 12, 14, 0, tzinfo=UTC)},
        "occ-new": {"symbol": "MSFT", "trigger_at": datetime(2026, 8, 3, 14, 0, tzinfo=UTC)},
        # A winter trigger whose ETH tail lives in the following month (BD-66).
        "occ-dec": {"symbol": "NVDA", "trigger_at": datetime(2026, 12, 31, 23, 0, tzinfo=UTC)},
        "occ-none": {"symbol": "TSLA", "trigger_at": None},
    }
    partitions = cli._m5_partitions_for(known, build_day)

    assert "month=2026-05" in partitions, "the old occurrence's own month is read"
    assert "month=2026-04" in partitions, "the ATR warm-up month is read"
    assert "month=2026-06" in partitions, "the forward outcome month is read"
    assert "month=2026-08" in partitions
    assert "month=2026-12" in partitions
    assert "month=2027-01" in partitions, "the ETH tail month is read too"
    # Bounded: only months an occurrence can actually need, no full-range sweep.
    assert "month=2026-03" not in partitions
    assert partitions == sorted(set(partitions))


def test_outcome_bucket_covers_all_symbols_over_32_days():
    start = date(2026, 8, 3)
    buckets = {
        cli._outcome_bucket(start + timedelta(days=offset), datetime(2026, 8, 3, 18, tzinfo=UTC))
        for offset in range(cli.OUTCOME_BUCKETS)
    }
    assert buckets == set(range(cli.OUTCOME_BUCKETS))


# --- the backfill entry point ---------------------------------------------
def test_the_backfill_job_shares_the_builds_single_flight_lock(store, tmp_path):
    """Both write the lake, and LD-01 allows exactly one writer."""
    lock = tmp_path / "build.lock"
    with cli.single_flight(lock):
        report = cli.run_backfill_job(store, session_date=date(2026, 8, 3), lock_path=lock)
    assert report["status"] == "REFUSED" and "already running" in report["message"]


def test_the_backfill_job_uses_the_sessions_point_in_time_cohort(store, tmp_path, monkeypatch):
    """The cohort is universe_membership_daily, the same source the D1 wrap uses."""
    day = date(2026, 8, 3)
    store.publish(
        "universe_membership_daily",
        [
            {
                "session_date": day,
                "list_name": "longs",
                "symbol": symbol,
                "rank_in_list": index,
                "inclusion_reason": "watchlist_file",
                "snapshot_at": NOW,
                "schema_version": schemas.SCHEMA_VERSION,
                "run_id": "universe",
            }
            for index, symbol in enumerate(("AAPL", "MSFT"))
        ],
        job_id="universe",
    )

    seen = {}

    def fetcher(symbol, fetch_day, *, timeframe, use_rth):
        seen.setdefault("calls", []).append((symbol, fetch_day, use_rth))
        from scripts.research_warehouse.backfill import FetchResult

        return FetchResult(bars=[])

    report = cli.run_backfill_job(
        store,
        session_date=day,
        fetcher=fetcher,
        time_budget_seconds=0,
        now=NOW,
        lock_path=tmp_path / "lock",
    )
    assert report["cohort"] == 2
    assert sorted({call[0] for call in seen["calls"]}) == ["AAPL", "MSFT"]
    # ETH-inclusive, per LD-03.
    assert {call[2] for call in seen["calls"]} == {False}


def test_the_backfill_job_says_so_when_there_is_no_cohort(store, tmp_path):
    report = cli.run_backfill_job(
        store, session_date=date(2026, 8, 3), fetcher=lambda *a, **k: None,
        lock_path=tmp_path / "lock",
    )
    assert report["status"] == "NO_COHORT"
    assert "universe_membership_daily" in report["message"]


def test_the_backfill_job_reports_a_missing_provider_rather_than_raising(store, tmp_path, monkeypatch):
    """The BD-25 transport has no offline coverage; its absence is a status."""
    day = date(2026, 8, 3)
    monkeypatch.setattr(cli, "_backfill_cohort", lambda *a, **k: ["AAPL"])
    monkeypatch.setattr(
        cli, "_capture_fetcher", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no TWS"))
    )
    report = cli.run_backfill_job(store, session_date=day, lock_path=tmp_path / "lock")
    assert report["status"] == "NO_PROVIDER" and "no TWS" in report["message"]


def test_the_backfill_job_is_a_no_op_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.ResearchStore, "open", classmethod(lambda cls, root=None: None))
    report = cli.run_backfill_job(lock_path=tmp_path / "lock")
    assert report["status"] == "DISABLED"


def test_the_cli_exposes_build_backfill_status_and_restore_check(monkeypatch, capsys):
    """The entry points a scheduled desk actually calls."""
    monkeypatch.setattr(cli.ResearchStore, "open", classmethod(lambda cls, root=None: None))
    assert cli.main(["backfill", "--job", "nightly"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DISABLED"

    assert cli.main(["build"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "DISABLED"


def test_the_job_descriptor_names_a_real_invoker():
    """The old descriptor probed a `register_job` API that exists nowhere."""
    from ui.services.warehouse_service import register_build_job

    descriptor = register_build_job()
    assert descriptor["invoked_by"].endswith("ScanService.start_warehouse_build")
    assert "backfill" in descriptor["backfill_entry_point"]
    # And that invoker really exists, with that name.
    from ui.services.scan_service import ScanService

    assert callable(ScanService.start_warehouse_build)
