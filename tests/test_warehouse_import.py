"""Bronze wraps and the daily snapshots (plan Phase 2; sec 19.0, 19.5, 20.4).

The exit criterion pinned here: every wrap-as-bronze / daily-ingest artifact is
ingested **with hashes**, and a re-run is a no-op - byte-identical lake, no new
manifest lines, no new files. Reuse-as-is stores are read through their own
loaders and projected into silver; no copy of them is made.

Also pinned: the legacy artifact is never modified, a malformed record is
preserved rather than dropped, a missing source is a clean skip, and the whole
module is inert when the warehouse is disabled.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from scripts.research_warehouse import ingest_existing as ingest
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc
NOW = datetime(2026, 8, 4, 2, 30, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def _log_artifact(name="alert_review_events"):
    return ingest.BronzeArtifact(
        name,
        ingest.MODE_APPEND_LOG,
        event_keys=("ts",),
        id_keys=("event_id",),
        class_a=True,
    )


def _write_log(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_append_log_wrap_records_hashes_and_payloads(store, tmp_path):
    source = tmp_path / "alert_review_events.jsonl"
    _write_log(
        source,
        [
            {"event_id": "a1", "ts": "2026-08-03T13:40:00+00:00", "decision": "TAKEN"},
            {"event_id": "a2", "ts": "2026-08-03T14:10:00+00:00", "decision": "PASSED"},
        ],
    )
    before = source.read_bytes()

    report = ingest.ingest_artifact(store, _log_artifact(), path=source, run_id="run-1", now=NOW)

    assert report.status == "OK" and report.rows_ingested == 2
    assert len(report.source_sha256) == 64
    table = store.read_table("bronze_alert_review_events")
    assert table.num_rows == 2
    rows = table.to_pylist()
    assert [row["legacy_id"] for row in rows] == ["a1", "a2"]
    assert [row["source_offset"] for row in rows] == [1, 2]
    assert json.loads(rows[0]["payload"])["decision"] == "TAKEN"
    assert rows[0]["event_at"] == datetime(2026, 8, 3, 13, 40, tzinfo=UTC)
    assert rows[0]["observed_at"] == NOW
    # Wrapped legacy evidence is BACKFILL: excluded from AS_OBSERVED claims.
    assert {row["capture_mode"] for row in rows} == {"BACKFILL"}
    assert {row["source_sha256"] for row in rows} == {report.source_sha256}

    # The manifest line carries the source path + file hash (the exit criterion).
    entry = store.manifest.resolve("bronze_alert_review_events").entries[0]
    assert entry.extra[ingest.EXTRA_SOURCE_PATH] == str(source)
    assert entry.extra[ingest.EXTRA_SOURCE_SHA] == report.source_sha256
    assert entry.extra[ingest.EXTRA_MAX_OFFSET] == 2
    assert entry.extra["class_a"] is True
    # The legacy writer keeps writing; its file is never touched.
    assert source.read_bytes() == before


def test_rerun_is_a_no_op(store, tmp_path):
    source = tmp_path / "events.jsonl"
    _write_log(source, [{"event_id": "a1", "ts": "2026-08-03T13:40:00+00:00"}])
    ingest.ingest_artifact(store, _log_artifact(), path=source, now=NOW)

    manifest_before = store.manifest.path.read_bytes()
    files_before = {path: path.read_bytes() for path in store.iter_live_tree_files()}

    report = ingest.ingest_artifact(store, _log_artifact(), path=source, now=NOW)

    assert report.status == "UNCHANGED" and report.rows_ingested == 0
    assert store.manifest.path.read_bytes() == manifest_before
    assert {path: path.read_bytes() for path in store.iter_live_tree_files()} == files_before


def test_appended_records_ingest_from_the_watermark(store, tmp_path):
    source = tmp_path / "events.jsonl"
    _write_log(source, [{"event_id": "a1", "ts": "2026-08-03T13:40:00+00:00"}])
    ingest.ingest_artifact(store, _log_artifact(), path=source, now=NOW)

    with open(source, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_id": "a2", "ts": "2026-08-03T15:00:00+00:00"}) + "\n")
    report = ingest.ingest_artifact(store, _log_artifact(), path=source, now=NOW)

    assert report.rows_ingested == 1
    table = store.read_table("bronze_alert_review_events")
    assert table.num_rows == 2
    assert sorted(table.column("legacy_id").to_pylist()) == ["a1", "a2"]
    assert sorted(table.column("source_offset").to_pylist()) == [1, 2]


def test_malformed_record_is_preserved_not_dropped(store, tmp_path):
    source = tmp_path / "events.jsonl"
    source.write_text(
        json.dumps({"event_id": "a1", "ts": "2026-08-03T13:40:00+00:00"}) + "\n{ truncated json\n",
        encoding="utf-8",
    )
    report = ingest.ingest_artifact(store, _log_artifact(), path=source, now=NOW)

    assert report.rows_ingested == 2
    rows = store.read_table("bronze_alert_review_events").to_pylist()
    broken = [row for row in rows if row["quality"] == "INVALID_DATA"]
    assert len(broken) == 1 and broken[0]["payload"] == "{ truncated json"
    # Its partition still resolves: partition_ts falls back to observed_at.
    assert broken[0]["partition_ts"] == NOW


def test_snapshot_artifact_versions_by_content_hash(store, tmp_path):
    source = tmp_path / "price_alerts.json"
    artifact = ingest.BronzeArtifact("price_alerts", ingest.MODE_SNAPSHOT, class_a=True)
    source.write_text(json.dumps({"entries": [{"symbol": "AAPL", "above": 210.0}]}), encoding="utf-8")

    first = ingest.ingest_artifact(store, artifact, path=source, now=NOW)
    unchanged = ingest.ingest_artifact(store, artifact, path=source, now=NOW)
    source.write_text(json.dumps({"entries": [{"symbol": "AAPL", "above": 215.0}]}), encoding="utf-8")
    second = ingest.ingest_artifact(store, artifact, path=source, now=NOW)

    assert first.rows_ingested == 1
    assert unchanged.status == "UNCHANGED" and unchanged.rows_ingested == 0
    assert second.rows_ingested == 1
    table = store.read_table("bronze_price_alerts")
    assert table.num_rows == 2  # two document versions, both retained
    assert len(set(table.column("source_sha256").to_pylist())) == 2


def test_csv_rows_wrap_with_header_names(store, tmp_path):
    source = tmp_path / "intraday_bounces.csv"
    source.write_text("symbol,bounce_type,ts\nAAPL,vwap_band,2026-08-03T14:00:00+00:00\n", encoding="utf-8")
    artifact = ingest.BronzeArtifact("intraday_bounces", ingest.MODE_CSV_ROWS, event_keys=("ts",))

    report = ingest.ingest_artifact(store, artifact, path=source, now=NOW)

    assert report.rows_ingested == 1
    row = store.read_table("bronze_intraday_bounces").to_pylist()[0]
    assert json.loads(row["payload"]) == {
        "symbol": "AAPL",
        "bounce_type": "vwap_band",
        "ts": "2026-08-03T14:00:00+00:00",
    }
    assert row["payload_format"] == "CSV_ROW"
    assert row["event_at"] == datetime(2026, 8, 3, 14, 0, tzinfo=UTC)


def test_snapshot_directory_ingests_each_file_once(store, tmp_path):
    manifests = tmp_path / "run_manifests"
    manifests.mkdir()
    (manifests / "run-1.json").write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")
    artifact = ingest.BronzeArtifact("run_manifests", ingest.MODE_SNAPSHOT)

    first = ingest.ingest_artifact(store, artifact, path=manifests, now=NOW)
    (manifests / "run-2.json").write_text(json.dumps({"run_id": "run-2"}), encoding="utf-8")
    second = ingest.ingest_artifact(store, artifact, path=manifests, now=NOW)
    third = ingest.ingest_artifact(store, artifact, path=manifests, now=NOW)

    assert first.rows_ingested == 1 and second.rows_ingested == 1
    assert third.status == "UNCHANGED"
    assert store.read_table("bronze_run_manifests").num_rows == 2


def test_missing_source_is_a_clean_skip(store, tmp_path):
    report = ingest.ingest_artifact(store, _log_artifact(), path=tmp_path / "never_written.jsonl")
    assert report.status == "MISSING_SOURCE" and report.rows_ingested == 0
    assert store.manifest.read_entries() == []


def test_everything_is_inert_when_the_warehouse_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.config, "warehouse_enabled", lambda: False)
    assert ingest.ingest_artifact(None, _log_artifact()).status == "DISABLED"
    assert ingest.run_bronze_ingest(None) == []
    assert ingest.run_daily_snapshots(None) == []
    assert ingest.ingest_everything() == {"enabled": False, "bronze": [], "snapshots": []}


def test_registry_covers_the_wrap_as_bronze_inventory():
    names = {artifact.artifact for artifact in ingest.BRONZE_ARTIFACTS}
    # Section 19.0 rows whose disposition is wrap-as-bronze or daily ingest.
    assert {
        "setup_tracker",
        "setup_scenarios",
        "intraday_bounces",
        "intraday_bounce_outcomes",
        "alert_review_events",
        "spy_state_shadow",
        "greatness_shadow",
        "technical_integrity_events",
        "job_ledger",
        "heartbeat",
        "earnings_avwap_anchors",
        "earnings_calendar_history",
        "d1_level_watches",
        "d1_event_watches",
        "alert_chart_watches",
        "price_alerts",
    } <= names
    assert ingest.RUN_MANIFEST_ARTIFACT.artifact == "run_manifests"
    # The four trader geometry JSONs are Class A (mirrored to Drive + backup).
    assert {"d1_level_watches", "d1_event_watches", "alert_chart_watches", "price_alerts"} <= set(
        ingest.CLASS_A_ARTIFACTS
    )
    # Bronze raw is never a compaction input.
    from scripts.research_warehouse import schemas

    assert schemas.dataset_spec("bronze_setup_tracker").compactable is False
    assert schemas.dataset_spec("bronze_setup_tracker").layer == "bronze"


def test_every_registered_artifact_resolves_a_path_without_reading_it():
    """Path resolution must never raise, even where the artifact is absent."""
    for artifact in ingest.BRONZE_ARTIFACTS + (ingest.RUN_MANIFEST_ARTIFACT,):
        resolved = artifact.resolve_path()
        assert resolved is None or str(resolved), artifact.artifact
        assert artifact.mode in {ingest.MODE_APPEND_LOG, ingest.MODE_SNAPSHOT, ingest.MODE_CSV_ROWS}
        assert artifact.dataset == f"bronze_{artifact.artifact}"


def test_universe_snapshot_is_first_capture_and_never_backfilled(store, tmp_path, monkeypatch):
    longs = tmp_path / "longs.txt"
    longs.write_text("AAPL\nMSFT  # keeper\n\nNVDA\n", encoding="utf-8")
    monkeypatch.setattr(ingest._paths(), "LONGS_FILE", longs, raising=False)
    lists = (("longs", "LONGS_FILE", "watchlist_file"),)
    day = date(2026, 8, 3)

    report = ingest.snapshot_universe_membership(store, session_date=day, lists=lists, now=NOW, run_id="run-1")
    assert report.rows == 3
    rows = store.read_table("universe_membership_daily").to_pylist()
    assert [row["symbol"] for row in rows] == ["AAPL", "MSFT", "NVDA"]
    assert [row["rank_in_list"] for row in rows] == [1, 2, 3]
    assert {row["snapshot_at"] for row in rows} == {NOW}

    # A later edit the same day cannot rewrite who was a member (LD-05).
    longs.write_text("TSLA\n", encoding="utf-8")
    again = ingest.snapshot_universe_membership(store, session_date=day, lists=lists, now=NOW)
    assert again.status == "ALREADY_CAPTURED" and again.rows == 0
    assert store.read_table("universe_membership_daily").num_rows == 3

    # The next session captures the new list under its own date.
    tomorrow = ingest.snapshot_universe_membership(store, session_date=date(2026, 8, 4), lists=lists, now=NOW)
    assert tomorrow.rows == 1
    assert store.read_table("universe_membership_daily").num_rows == 4


def test_exploration_cohort_file_is_empty_until_the_trader_confirms_it():
    # An agent-invented exploration list would silently become the research
    # denominator; item 5 of the confirmation register fills this file.
    assert ingest.EXPLORATION_COHORT_FILE.is_file()
    assert ingest.load_exploration_cohort() == []
    text = ingest.EXPLORATION_COHORT_FILE.read_text(encoding="utf-8")
    assert "BACKFILL-only" in text and "confirmation register" in text


def test_exploration_cohort_snapshots_when_present(store, tmp_path, monkeypatch):
    cohort = tmp_path / "exploration_cohort.txt"
    cohort.write_text("# fixed cohort\nPLUG\nRIOT\n", encoding="utf-8")
    report = ingest.snapshot_universe_membership(
        store,
        session_date=date(2026, 8, 3),
        lists=(),
        exploration_path=cohort,
        now=NOW,
    )
    assert report.rows == 2
    rows = store.read_table("universe_membership_daily").to_pylist()
    assert {row["list_name"] for row in rows} == {"exploration_fixed"}
    assert {row["inclusion_reason"] for row in rows} == {"exploration_cohort_file"}


def test_geometry_snapshot_wraps_the_existing_level_sources(store, tmp_path, monkeypatch):
    paths = ingest._paths()
    levels_dir = tmp_path / "levels"
    levels_dir.mkdir()
    (levels_dir / "AAPL.json").write_text(
        json.dumps(
            {
                "symbol": "AAPL",
                "levels": [
                    {"kind": "hv_horizontal", "price": 205.5, "strength": 3.2},
                    {"kind": "cloud_flat", "price": 198.0, "strength": 1.5, "effective_range": ["2026-07-01", "2026-09-01"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    watches = tmp_path / "d1_level_watches.json"
    watches.write_text(
        json.dumps(
            {
                "watches": [
                    {
                        "symbol": "AAPL",
                        "direction": "above",
                        "level": 212.25,
                        "armed_at": "2026-07-30T18:05:00",
                        "candle_date": "2026-07-30",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    alerts = tmp_path / "price_alerts.json"
    alerts.write_text(
        json.dumps({"entries": [{"symbol": "MSFT", "above": 500.0, "below": 480.0, "armed_above": True}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "D1_LEVEL_WATCHES_FILE", watches, raising=False)
    monkeypatch.setattr(paths, "PRICE_ALERTS_FILE", alerts, raising=False)
    monkeypatch.setattr(paths, "ALERT_CHART_WATCHES_FILE", tmp_path / "absent.json", raising=False)
    monkeypatch.setattr(ingest, "_d1_feed_levels", lambda symbols: {})

    report = ingest.snapshot_level_geometry(
        store,
        session_date=date(2026, 8, 3),
        symbols=["AAPL"],
        levels_dir=levels_dir,
        now=NOW,
        run_id="run-1",
    )

    assert report.rows == 5  # 2 HV store + 1 armed D1 watch + 2 price-alert sides
    rows = store.read_table("level_state_daily").to_pylist()
    families = {(row["level_family"], row["source_store"]) for row in rows}
    assert families == {
        ("HORIZONTAL_STORE", "hv_level_store"),
        ("WATCH_JSON", "d1_level_watches.json"),
        ("WATCH_JSON", "price_alerts.json"),
    }
    # Human geometry keeps its own arm time as known_at.
    watch_row = next(row for row in rows if row["source_store"] == "d1_level_watches.json")
    assert watch_row["known_at"] == datetime(2026, 7, 30, 18, 5, tzinfo=UTC)
    # A disarmed alert side is recorded, not omitted - the denominator matters.
    below = next(row for row in rows if row["source_store"] == "price_alerts.json" and row["level_price"] == 480.0)
    assert below["is_active"] is False
    # Level identities are deterministic and distinct per source store.
    assert len({row["level_id"] for row in rows}) == 5

    repeat = ingest.snapshot_level_geometry(
        store,
        session_date=date(2026, 8, 3),
        symbols=["AAPL"],
        levels_dir=levels_dir,
        now=NOW,
    )
    assert repeat.status == "ALREADY_CAPTURED" and repeat.rows == 0
    assert store.read_table("level_state_daily").num_rows == 5


def test_geometry_snapshot_reads_d1_level_feed_state(store, tmp_path, monkeypatch):
    paths = ingest._paths()
    for attr in ("D1_LEVEL_WATCHES_FILE", "PRICE_ALERTS_FILE", "ALERT_CHART_WATCHES_FILE"):
        monkeypatch.setattr(paths, attr, tmp_path / f"{attr}.absent.json", raising=False)
    monkeypatch.setattr(
        ingest,
        "_d1_feed_levels",
        lambda symbols: {"AAPL": {"smas": {"sma50": 200.0}, "trendlines": [190.5], "last_trade_date": "2026-08-03"}},
    )

    report = ingest.snapshot_level_geometry(
        store,
        session_date=date(2026, 8, 3),
        symbols=["AAPL"],
        levels_dir=tmp_path / "no_levels",
        now=NOW,
    )
    assert report.rows == 2
    rows = store.read_table("level_state_daily").to_pylist()
    assert {row["level_family"] for row in rows} == {"MA_LEVEL", "TRENDLINE"}
    assert {row["source_store"] for row in rows} == {"d1_level_feed"}


def test_durable_d1_store_is_read_not_copied(store, tmp_path):
    import pandas as pd

    bars_dir = tmp_path / "daily_bars"
    bars_dir.mkdir()
    frame = pd.DataFrame(
        {
            "datetime": [datetime(2026, 7, 31), datetime(2026, 8, 3), datetime(2026, 8, 4)],
            "open": [100.0, 101.0, 102.0],
            "high": [103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1_000_000, 1_100_000, 900_000],
        }
    )
    source = ingest.durable_daily_bar_file("AAPL", bars_dir)
    frame.to_parquet(source, index=False)
    before = source.read_bytes()

    report = ingest.ingest_daily_bars(
        store, ["AAPL", "NOSUCH"], as_of=date(2026, 8, 4), bars_dir=bars_dir, now=NOW, run_id="run-1"
    )

    # 2026-08-04 is the current session: a forming D1 bar is never evidence.
    assert report.rows == 2
    rows = store.read_table("bar_d1").to_pylist()
    assert [str(row["session_date"]) for row in rows] == ["2026-07-31", "2026-08-03"]
    assert {row["session_id"] for row in rows} == {"XNYS-2026-07-31", "XNYS-2026-08-03"}
    assert {row["is_complete"] for row in rows} == {True}
    # That store never recorded which provider produced a row: say so.
    assert {row["provider"] for row in rows} == {"UNKNOWN"}
    assert {row["capture_mode"] for row in rows} == {"BACKFILL"}
    # The legacy store is read-only to us.
    assert source.read_bytes() == before

    repeat = ingest.ingest_daily_bars(store, ["AAPL"], as_of=date(2026, 8, 4), bars_dir=bars_dir, now=NOW)
    assert repeat.status == "ALREADY_CAPTURED" and repeat.rows == 0
    assert store.read_table("bar_d1").num_rows == 2

    later = ingest.ingest_daily_bars(store, ["AAPL"], as_of=date(2026, 8, 5), bars_dir=bars_dir, now=NOW)
    assert later.rows == 1  # yesterday's bar completed
    assert store.read_table("bar_d1").num_rows == 3


def test_durable_d1_filename_matches_the_scanner_sanitizer(tmp_path):
    assert ingest.durable_daily_bar_file("BF.B", tmp_path).name == "BF.B.parquet"
    assert ingest.durable_daily_bar_file("con", tmp_path).name == "CON_.parquet"
    assert ingest.read_durable_daily_bars("MISSING", tmp_path) is None


def test_ingest_everything_runs_bronze_and_snapshots(store, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "BRONZE_ARTIFACTS", (_log_artifact("job_ledger"),))
    monkeypatch.setattr(ingest, "RUN_MANIFEST_ARTIFACT", _log_artifact("run_manifests"))
    monkeypatch.setattr(ingest, "_d1_feed_levels", lambda symbols: {})
    monkeypatch.setattr(ingest, "snapshot_universe_membership", lambda *a, **k: ingest.SnapshotReport("universe"))
    monkeypatch.setattr(ingest, "snapshot_level_geometry", lambda *a, **k: ingest.SnapshotReport("levels"))

    result = ingest.ingest_everything(store, session_date=date(2026, 8, 3), now=NOW)

    assert result["enabled"] is True
    assert [report.artifact for report in result["bronze"]] == ["job_ledger", "run_manifests"]
    assert [report.dataset for report in result["snapshots"]] == ["universe", "levels"]
