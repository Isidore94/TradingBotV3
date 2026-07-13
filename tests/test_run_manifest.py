"""Phase 1 (plan.md sec 6.1): structured run manifests."""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from diagnostics import (  # noqa: E402
    ManifestRecorder,
    clear_active_recorder,
    get_active_recorder,
    load_recent_manifests,
    prune_manifests,
    set_active_recorder,
)


def test_manifest_records_phases_counters_and_saves_atomically(tmp_path):
    recorder = ManifestRecorder(job_type="master_scan", trigger="test")
    recorder.record_phase("earnings", 1.25)
    recorder.record_phase("symbols", 10.5)
    recorder.incr("provider_calls", 3)
    recorder.set_counter("symbols_processed", 42)
    recorder.finalize(status="ok")
    path = recorder.save(tmp_path)

    assert path.exists()
    loaded = load_recent_manifests(tmp_path, limit=5)
    assert len(loaded) == 1
    manifest = loaded[0]
    assert manifest["status"] == "ok"
    assert manifest["total_seconds"] == pytest.approx(11.75)
    assert manifest["counters"]["symbols_processed"] == 42
    assert [p["label"] for p in manifest["phases"]] == ["earnings", "symbols"]


def test_manifest_aggregate_total_is_not_double_counted():
    recorder = ManifestRecorder(job_type="master_scan", trigger="test")
    recorder.record_phase("prep", 4.0)
    recorder.record_phase("output", 6.0)
    recorder.record_phase("TOTAL (theta enrichment deferred)", 10.0)
    recorder.finalize(status="ok")

    assert recorder.to_dict()["total_seconds"] == pytest.approx(10.0)


def test_prune_keeps_bounded_history(tmp_path):
    for i in range(8):
        rec = ManifestRecorder(job_type="master_scan", run_id=f"run-{i}")
        rec.finalize("ok")
        rec.save(tmp_path, keep=100)
    removed = prune_manifests(tmp_path, keep=5)
    assert removed == 3
    assert len(list(tmp_path.glob("*.json"))) == 5


def test_active_recorder_is_thread_scoped():
    rec = ManifestRecorder(job_type="master_scan")
    set_active_recorder(rec)
    try:
        assert get_active_recorder() is rec
        seen = []
        import threading

        thread = threading.Thread(target=lambda: seen.append(get_active_recorder()))
        thread.start()
        thread.join()
        assert seen == [None], "another thread must not inherit this run's recorder"
    finally:
        clear_active_recorder()
    assert get_active_recorder() is None


def test_run_master_writes_manifest_even_on_failure(tmp_path, monkeypatch):
    import diagnostics.run_manifest as rm
    from master_avwap_lib import runner

    monkeypatch.setattr(rm, "default_manifest_dir", lambda: tmp_path)

    def boom(**kwargs):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(runner, "_run_master_impl", boom)

    with pytest.raises(RuntimeError, match="scan exploded"):
        runner.run_master(use_shared_watchlists=True)

    manifests = load_recent_manifests(tmp_path, limit=5)
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest["status"] == "failed"
    assert "scan exploded" in manifest["error"]
    assert manifest["counters"]["use_shared_watchlists"] is True
    assert get_active_recorder() is None


def test_run_master_manifest_records_success_counters(tmp_path, monkeypatch):
    import diagnostics.run_manifest as rm
    from master_avwap_lib import runner

    monkeypatch.setattr(rm, "default_manifest_dir", lambda: tmp_path)

    fake_result = {
        "watchlist_label": "test lists",
        "daily_frames_by_symbol": {"AAPL": object(), "NVDA": object()},
        "tracked_rows": [{"symbol": "AAPL"}],
        "setup_tracker_updated": False,
    }
    monkeypatch.setattr(runner, "_run_master_impl", lambda **kwargs: fake_result)

    result = runner.run_master()
    assert result is fake_result

    manifest = load_recent_manifests(tmp_path, limit=1)[0]
    assert manifest["status"] == "ok"
    assert manifest["counters"]["symbols_processed"] == 2
    assert manifest["counters"]["tracked_rows"] == 1
    assert manifest["outputs"]["watchlist_label"] == "test lists"


def test_run_master_uses_parent_scheduler_identity(tmp_path, monkeypatch):
    import diagnostics.run_manifest as rm
    from master_avwap_lib import runner

    monkeypatch.setattr(rm, "default_manifest_dir", lambda: tmp_path)
    monkeypatch.setenv("TRADINGBOT_RUN_ID", "master_scan-parent-linked")
    monkeypatch.setenv("TRADINGBOT_RUN_TRIGGER", "Auto Pilot swing scan (11:00)")
    monkeypatch.setattr(runner, "_run_master_impl", lambda **kwargs: {})

    runner.run_master(use_shared_watchlists=True)

    manifest = load_recent_manifests(tmp_path, limit=1)[0]
    assert manifest["run_id"] == "master_scan-parent-linked"
    assert manifest["trigger"] == "Auto Pilot swing scan (11:00)"
