"""The nightly `permutation_report` slot (trader 2026-09-30: every night, no by-hand step).

It wraps the backfill and the search (both untouched), publishes by temp-and-rename,
names the live outcomes parquet as the report's source, keeps the last good report
when a step fails, sits in stage 1, and `--force` re-runs it. `setup_keys_narration`
is on the weeknight slate now.
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import project_paths  # noqa: E402
import setup_permutation_backfill as bf  # noqa: E402
import setup_permutation_search as ps  # noqa: E402
from ai_jobs import ledger, runner  # noqa: E402
from ai_jobs import permutation_report as slot_mod  # noqa: E402
from research_warehouse import trial_ledger  # noqa: E402

SLOT = "permutation_report"
REAL_RUN_BACKFILL = slot_mod.run_backfill
REAL_LEDGER_ROOT = slot_mod.ledger_root
SESSIONS = [(date(2026, 6, 1) + timedelta(days=i)).isoformat() for i in range(60)]


def _rows(seed: int = 7) -> list[dict]:
    """A swing population with one planted key (``ma_support``), in the backfill's row shape."""
    rng = random.Random(seed)
    holdout = set(SESSIONS[-ps.HOLDOUT_SESSIONS:])
    rows = []
    for day in SESSIONS:
        for n in range(20):
            key_on = rng.random() < 0.3
            win = rng.random() < (0.85 if key_on else 0.4)
            rows.append({
                "population": "swing", "episode_id": f"{day}:{n}", "symbol": f"S{n}", "side": "LONG",
                "family": "avwap_band_bounce", "session": day, "horizon": 1, "horizon_name": "1_sessions",
                "win": win, "r": 1.0 if win else -1.0, "r_unit": "atr20",
                "f_ma_support": "sma100_support" if key_on else "no_ma_support",
                "f_weekday": "mon" if day not in holdout else "tue",
            })
    return rows


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Stand-in live inputs and outputs under tmp, and a fake backfill (the real one has its own tests)."""
    live = tmp_path / "live"
    (live / "daily_bars").mkdir(parents=True)
    (live / "daily_bars" / "SPY.csv").write_text("datetime,close\n2026-06-01,500\n", encoding="utf-8")
    (live / "d1_features_history.csv").write_text("symbol,side\n", encoding="utf-8")
    inputs = slot_mod.Inputs(
        features=live / "d1_features_history.csv", daily_bars=live / "daily_bars",
        m5_outcomes=live / "missing_outcomes.csv", m5_candidates=live / "missing_candidates.csv",
        m5_stamps=live / "missing_stamps.jsonl", journal_db=live / "missing.sqlite3",
        environment=live / "missing_env.jsonl", review_events_dir=live / "missing_events",
        review_events_file=live / "missing_events.jsonl", scan_reports=live / "missing_reports",
    )
    runtime = tmp_path / "runtime"
    outputs = slot_mod.Outputs(
        outcomes=runtime / "permutation_outcomes.parquet", report=runtime / "permutation_report.json",
        history_dir=runtime / "permutation_report_history", verdicts=runtime / "permutation_verdicts.json",
    )
    lake = tmp_path / "lake"
    staging = tmp_path / "staging"
    staging.mkdir()
    calls = []

    def fake_backfill(staged, last_completed):
        calls.append({"staged": dict(staged), "last_completed": last_completed})
        return _rows(), {"swing_rows": 1200, "m5_rows": 0}

    monkeypatch.setattr(slot_mod, "run_backfill", fake_backfill)
    monkeypatch.setattr(slot_mod, "live_inputs", lambda: inputs)
    monkeypatch.setattr(slot_mod, "live_outputs", lambda: outputs)
    monkeypatch.setattr(slot_mod, "ledger_root", lambda: lake)
    monkeypatch.setattr(slot_mod.tempfile, "tempdir", str(staging))
    return {"inputs": inputs, "outputs": outputs, "lake": lake, "staging": staging, "calls": calls,
            "tmp": tmp_path}


def _run(world, **kwargs):
    return slot_mod.run_permutation_report(session_date="2026-08-24", staging_parent=world["staging"], **kwargs)


# --- where it sits


def test_the_slot_is_stage_one_after_the_lake_and_before_family_side_evidence():
    names = [slot.name for slot in runner.default_slots()]
    assert names.index(SLOT) > names.index("lake_history_topup")
    assert names.index(SLOT) == names.index("market_regime_daily") + 1
    assert names.index(SLOT) < names.index("family_side_evidence")
    assert SLOT in [slot.name for slot in runner._deterministic_stage(runner.default_slots())]
    by_name = {slot.name: slot for slot in runner.default_slots()}
    assert by_name[SLOT].goal == "permutations"
    assert by_name[SLOT].uses_model is False
    assert by_name[SLOT].reserve_minutes == slot_mod.RESERVE_MINUTES
    for kind in (runner.NIGHT_WEEKNIGHT, runner.NIGHT_SATURDAY):
        assert SLOT in [slot.name for slot in runner.slots_for(kind)]
    assert SLOT in [slot.name for slot in runner.slots_for(runner.NIGHT_SUNDAY, session_date="")]


def test_setup_keys_narration_runs_on_weeknights_and_the_heavy_two_stay_weekend_only():
    weeknight = [slot.name for slot in runner.slots_for(runner.NIGHT_WEEKNIGHT)]
    assert "setup_keys_narration" in weeknight
    assert "setup_keys_narration" not in runner.WEEKEND_ONLY_SLOTS
    assert weeknight.index("setup_keys_narration") > weeknight.index(SLOT)
    for name in ("ai_summary", "week_review_narration"):
        assert name in runner.WEEKEND_ONLY_SLOTS
        assert name not in weeknight


def test_the_source_is_the_live_outcomes_parquet_beside_the_live_report():
    assert project_paths.SETUP_PERMUTATION_OUTCOMES_FILE.parent == project_paths.SETUP_PERMUTATION_REPORT_FILE.parent
    assert slot_mod.live_outputs().outcomes == project_paths.SETUP_PERMUTATION_OUTCOMES_FILE
    assert slot_mod.live_outputs().report == project_paths.SETUP_PERMUTATION_REPORT_FILE
    assert slot_mod.live_outputs().history_dir == project_paths.SETUP_PERMUTATION_REPORT_HISTORY_DIR
    assert slot_mod.live_outputs().verdicts == project_paths.SETUP_PERMUTATION_VERDICTS_FILE


# --- the run


def test_a_good_run_publishes_by_temp_and_rename_and_names_the_live_parquet(world, monkeypatch):
    renames = []
    real_replace = os.replace

    def spy(src, dst):
        renames.append((Path(src).name, Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    result = _run(world)
    out = world["outputs"]
    assert result["status"] == ledger.STATUS_OK, result
    report = json.loads(out.report.read_text(encoding="utf-8"))
    assert report["source"] == str(out.outcomes)
    assert out.outcomes.is_file()
    assert ("permutation_report.json.tmp", out.report) in renames
    assert ("permutation_outcomes.parquet.tmp", out.outcomes) in renames
    assert not list(out.report.parent.glob("*.tmp"))
    assert [path.name for _day, path in ps.history_files(out.history_dir)]
    assert out.verdicts.is_file()
    families = report["populations"]["swing"]["horizons"]["1"]["families"]
    assert families["avwap_band_bounce LONG"]["keys"][0]["facets"] == {"ma_support": "sma100_support"}
    # Every grid the search read was registered first, in the ledger it was given.
    ids = {row["trial_id"] for row in trial_ledger.load(world["lake"])}
    assert set(families["avwap_band_bounce LONG"]["trial_ids"]) <= ids
    assert not list(world["staging"].iterdir()), "the staging copy is removed"


def test_the_backfill_reads_staged_copies_never_the_live_files(world, monkeypatch):
    monkeypatch.setattr(slot_mod, "run_backfill", REAL_RUN_BACKFILL)
    seen = {}
    inputs, outputs, lake = world["inputs"], world["outputs"], world["lake"]

    def spy(features, **kwargs):
        seen["features"] = Path(features)
        seen.update(kwargs)
        return bf.BackfillResult(rows=_rows(), counts={})

    monkeypatch.setattr(bf, "build_permutation_outcomes", spy)
    result = slot_mod.run_permutation_report(session_date="2026-08-24", inputs=inputs, outputs=outputs,
                                             trial_root=lake, staging_parent=world["staging"])
    assert result["status"] == ledger.STATUS_OK, result
    assert seen["features"] != inputs.features and seen["features"].name == inputs.features.name
    assert Path(seen["daily_bars"]) != inputs.daily_bars
    assert str(seen["spy_bars"]).endswith(os.path.join("daily_bars", "SPY.csv"))
    assert seen["last_completed"] == date(2026, 8, 24)
    # Missing optional inputs stay unknown, never an empty stand-in.
    assert seen["m5_outcomes"] is None and seen["m5_candidates"] is None and seen["structural_regime"] is None


def test_a_failed_search_keeps_the_last_good_report_and_says_why(world, monkeypatch):
    out = world["outputs"]
    out.report.parent.mkdir(parents=True)
    out.report.write_text('{"old": true}', encoding="utf-8")
    out.outcomes.write_bytes(b"old parquet")

    def broken(*_args, **_kwargs):
        raise RuntimeError("ledger share offline")

    monkeypatch.setattr(ps, "build_report", broken)
    result = _run(world)
    assert result["status"] == ledger.STATUS_FAILED
    assert "search failed" in result["reason"] and "ledger share offline" in result["reason"]
    assert out.report.read_text(encoding="utf-8") == '{"old": true}'
    assert out.outcomes.read_bytes() == b"old parquet"
    assert not out.history_dir.exists() and not out.verdicts.exists()
    assert not list(world["staging"].iterdir())


def test_a_failed_backfill_keeps_the_last_good_report(world, monkeypatch):
    out = world["outputs"]
    out.report.parent.mkdir(parents=True)
    out.report.write_text('{"old": true}', encoding="utf-8")

    def broken(_staged, _last):
        raise ValueError("bad scan row")

    monkeypatch.setattr(slot_mod, "run_backfill", broken)
    result = _run(world)
    assert result["status"] == ledger.STATUS_FAILED and "backfill failed" in result["reason"]
    assert out.report.read_text(encoding="utf-8") == '{"old": true}'


def test_no_research_lake_is_a_failed_row_not_a_crash(world, monkeypatch):
    from research_warehouse import config

    monkeypatch.setattr(slot_mod, "ledger_root", REAL_LEDGER_ROOT)
    monkeypatch.setattr(config, "get_research_store_dir", lambda: None)
    result = _run(world)
    assert result["status"] == ledger.STATUS_FAILED and "trial ledger" in result["reason"]
    assert not world["outputs"].report.exists()


def test_missing_scan_history_fails_before_any_search(world, monkeypatch):
    world["inputs"].features.unlink()
    result = _run(world)
    assert result["status"] == ledger.STATUS_FAILED and "copy the inputs" in result["reason"]
    assert world["calls"] == []


# --- the runner: once a night, and --force re-runs it

ET = ZoneInfo("America/New_York")
OVERNIGHT = datetime(2026, 8, 12, 2, 0, tzinfo=ET)


def test_force_re_runs_the_slot(world, tmp_path):
    from ai_jobs import store, window

    slot = next(item for item in runner.default_slots() if item.name == SLOT)
    led = tmp_path / "ledger.jsonl"
    with mock.patch.object(store, "store_available", return_value=(True, "ready")), \
            mock.patch.object(window, "market_session_block", return_value=""), \
            mock.patch.object(window, "launch_allowed", return_value=(True, "window open")):
        first = runner.run_slots([slot], now=OVERNIGHT, ledger_path=led)
        again = runner.run_slots([slot], now=OVERNIGHT, ledger_path=led)
        forced = runner.run_slots([slot], now=OVERNIGHT, force=True, ledger_path=led)
    assert [row["status"] for row in first.results] == [ledger.STATUS_OK]
    assert len(world["calls"]) == 2, "one nightly run, then one forced re-run"
    assert not [row for row in again.results if row.get("status") == ledger.STATUS_OK]
    assert [row["status"] for row in forced.results] in ([ledger.STATUS_OK], [ledger.STATUS_MANUAL])


def test_live_inputs_read_the_scan_snapshots_where_scan_manifest_writes_them():
    """The slot spells the folder itself (no detector import); it must stay the manifest's."""
    from master_avwap_lib import scan_manifest

    from ai_jobs import permutation_report

    assert permutation_report.live_inputs().scan_reports == Path(scan_manifest.scan_reports_dir())
