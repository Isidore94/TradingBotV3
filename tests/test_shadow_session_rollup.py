"""W08/W09: durable shadow-session summaries, rotation, and floor counters."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from diagnostics.shadow_session_rollup import (  # noqa: E402
    GREATNESS_ENGINE,
    RETENTION_POLICY,
    SPY_ENGINE,
    apply_retention,
    audit_session_summaries,
    evidence_directories,
    finalize_session,
    reset_audit_cache,
)

NOW = datetime(2026, 7, 14, 9, 30, tzinfo=timezone.utc)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _spy_row(schema: str, *, session="2026-07-13", config="spy-cfg", **extra):
    row = {
        "schema": schema,
        "session_date": session,
        "config_hash": config,
        "engine_version": "spy-v1",
        "machine": "desk",
        "complete_bar_ts": "2026-07-13T16:00:00+00:00",
    }
    row.update(extra)
    return row


def _greatness_row(event: str, *, candidate="NVDA|LONG|x", **extra):
    row = {
        "schema": "greatness_shadow_v4",
        "session_date": "2026-07-13",
        "config_hash": "great-cfg",
        "engine_version": "great-v1",
        "machine": "desk",
        "candidate_id": candidate,
        "event": event,
        "side": "LONG",
        "setup_family": "breakout",
        "bar": {"complete": True},
    }
    row.update(extra)
    return row


def test_spy_finalize_is_idempotent_and_reconciles_to_raw(tmp_path):
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_rows(
        log,
        [
            _spy_row(
                "spy_state_shadow_v4",
                state="IMPULSE",
                evaluated_at="2026-07-13T16:00:00+00:00",
            ),
            _spy_row(
                "spy_state_shadow_v4",
                state="COUNTER_MOVE",
                complete_bar_ts="2026-07-13T16:05:00+00:00",
                evaluated_at="2026-07-13T16:05:00+00:00",
            ),
            _spy_row(
                "spy_episode_shadow_v1",
                episode_uid="SPY|2026-07-13|ep-1",
                episode_id="ep-1",
                outcome="RESUMED",
                direction="BULLISH",
                derived_from_completed_bars=True,
            ),
        ],
    )
    coverage = {
        "session_date": "2026-07-13",
        "config_hash": "spy-cfg",
        "evaluations": 4,
        "usable_evaluations": 4,
        "errors": 0,
        "last_evaluation_at": "2026-07-13T16:10:00+00:00",
    }

    summary = finalize_session(
        engine=SPY_ENGINE,
        log_path=log,
        coverage=coverage,
        finalized_at=NOW,
        reason="session_rollover",
        engine_version="spy-v1",
        machine="desk",
        timezone="UTC",
        configuration="spy-cfg",
    )
    first_bytes = summary.read_bytes()
    raw_dir, _ = evidence_directories(log, SPY_ENGINE)
    archives = list(raw_dir.glob("*.jsonl"))

    assert not log.exists()
    assert len(archives) == 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["coverage"] == coverage
    assert payload["raw_stats"]["distinct_chains"] == 1
    assert payload["raw_stats"]["complete_chains"] == 1
    assert payload["session_metrics"]["state_observations"] == {
        "COUNTER_MOVE": 1,
        "IMPULSE": 1,
    }
    assert payload["session_metrics"]["state_transitions"] == {
        "IMPULSE->COUNTER_MOVE": 1,
    }
    assert payload["session_metrics"]["state_duration_seconds"] == {
        "COUNTER_MOVE": 300,
        "IMPULSE": 300,
    }

    # Crash/restart replay: no second archive or summary and no byte churn.
    again = finalize_session(
        engine=SPY_ENGINE,
        log_path=log,
        coverage=coverage,
        finalized_at=datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc),
        reason="session_rollover",
        engine_version="spy-v1",
        machine="desk",
        timezone="UTC",
        configuration="spy-cfg",
    )
    assert again == summary
    assert summary.read_bytes() == first_bytes
    assert list(raw_dir.glob("*.jsonl")) == archives

    reset_audit_cache()
    progress = audit_session_summaries(log, SPY_ENGINE)
    assert progress["eligible_sessions"] == 1
    assert progress["incomplete_sessions"] == 0
    assert progress["complete_chains"] == 1
    assert progress["section_7_floor_progress"]["eligible_sessions"] == {
        "count": 1,
        "floor": 10,
    }
    assert progress["manual_reviewed_chains"] == 0
    assert progress["promotion_decision"] == "NONE"


def test_finalize_recovers_after_crash_between_rotation_and_summary(
    tmp_path, monkeypatch
):
    import diagnostics.shadow_session_rollup as rollup

    log = tmp_path / "spy_state_shadow.jsonl"
    _write_rows(log, [_spy_row("spy_state_shadow_v4")])
    coverage = {
        "session_date": "2026-07-13",
        "config_hash": "spy-cfg",
        "evaluations": 1,
        "usable_evaluations": 1,
        "errors": 0,
    }
    original = rollup.atomic_write_json
    monkeypatch.setattr(
        rollup,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("power loss")),
    )
    with pytest.raises(OSError, match="power loss"):
        finalize_session(
            engine=SPY_ENGINE,
            log_path=log,
            coverage=coverage,
            finalized_at=NOW,
            reason="session_rollover",
            engine_version="spy-v1",
            machine="desk",
            timezone="UTC",
            configuration="spy-cfg",
        )
    raw_dir, summary_dir = evidence_directories(log, SPY_ENGINE)
    assert not log.exists()
    assert len(list(raw_dir.glob("*.jsonl"))) == 1
    assert not list(summary_dir.glob("*.json"))

    monkeypatch.setattr(rollup, "atomic_write_json", original)
    summary = finalize_session(
        engine=SPY_ENGINE,
        log_path=log,
        coverage=coverage,
        finalized_at=NOW,
        reason="session_rollover",
        engine_version="spy-v1",
        machine="desk",
        timezone="UTC",
        configuration="spy-cfg",
    )
    assert summary.exists()


def test_backfilled_session_without_coverage_is_explicitly_incomplete(tmp_path):
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_rows(
        log,
        [
            _spy_row("spy_state_shadow_v4", session="2026-07-12"),
            _spy_row("spy_state_shadow_v4", session="2026-07-13"),
        ],
    )
    finalize_session(
        engine=SPY_ENGINE,
        log_path=log,
        coverage={
            "session_date": "2026-07-13",
            "config_hash": "spy-cfg",
            "evaluations": 1,
            "usable_evaluations": 1,
            "errors": 0,
        },
        finalized_at=NOW,
        reason="session_rollover",
        engine_version="spy-v1",
        machine="desk",
        timezone="UTC",
        configuration="spy-cfg",
    )

    reset_audit_cache()
    progress = audit_session_summaries(log, SPY_ENGINE)
    assert progress["summary_count"] == 2
    assert progress["eligible_sessions"] == 1
    assert progress["incomplete_sessions"] == 1
    assert "coverage counters were unavailable" in " ".join(
        progress["incomplete_session_details"][0]["reasons"]
    )


def test_multiple_eligible_config_scopes_count_as_one_session(tmp_path):
    log = tmp_path / "spy_state_shadow.jsonl"
    for configuration in ("spy-cfg-a", "spy-cfg-b"):
        _write_rows(
            log,
            [
                _spy_row(
                    "spy_state_shadow_v4",
                    config=configuration,
                    state="IDLE",
                    evaluated_at="2026-07-13T16:00:00+00:00",
                )
            ],
        )
        finalize_session(
            engine=SPY_ENGINE,
            log_path=log,
            coverage={
                "session_date": "2026-07-13",
                "config_hash": configuration,
                "evaluations": 1,
                "usable_evaluations": 1,
                "errors": 0,
                "last_evaluation_at": "2026-07-13T16:05:00+00:00",
            },
            finalized_at=NOW,
            reason="configuration_changed",
            engine_version="spy-v1",
            machine="desk",
            timezone="UTC",
            configuration=configuration,
        )

    reset_audit_cache()
    progress = audit_session_summaries(log, SPY_ENGINE)
    assert progress["summary_count"] == 2
    assert progress["eligible_sessions"] == 1
    assert progress["incomplete_sessions"] == 0


def test_repeated_finalized_scope_fails_closed_and_preserves_active_rows(tmp_path):
    log = tmp_path / "spy_state_shadow.jsonl"
    coverage = {
        "session_date": "2026-07-13",
        "config_hash": "spy-cfg-a",
        "evaluations": 1,
        "usable_evaluations": 1,
        "errors": 0,
    }
    _write_rows(log, [_spy_row("spy_state_shadow_v4", config="spy-cfg-a")])
    finalize_session(
        engine=SPY_ENGINE,
        log_path=log,
        coverage=coverage,
        finalized_at=NOW,
        reason="configuration_changed",
        engine_version="spy-v1",
        machine="desk",
        timezone="UTC",
        configuration="spy-cfg-a",
    )

    # A later A -> B -> A reversion must not overwrite A's first replay
    # archive or strand the second A segment in a mixed next-session log.
    _write_rows(log, [_spy_row("spy_state_shadow_v4", config="spy-cfg-a")])
    active_bytes = log.read_bytes()
    with pytest.raises(RuntimeError, match="already-finalized"):
        finalize_session(
            engine=SPY_ENGINE,
            log_path=log,
            coverage=coverage,
            finalized_at=NOW,
            reason="configuration_changed",
            engine_version="spy-v1",
            machine="desk",
            timezone="UTC",
            configuration="spy-cfg-a",
        )
    assert log.read_bytes() == active_bytes


def test_greatness_floor_counters_are_raw_counts_not_promotion(tmp_path):
    log = tmp_path / "greatness_shadow.jsonl"
    _write_rows(
        log,
        [
            _greatness_row("LEVEL_TOUCHED"),
            _greatness_row("FAILED_ATTEMPT"),
            _greatness_row("REARMED"),
            _greatness_row("READY"),
        ],
    )
    finalize_session(
        engine=GREATNESS_ENGINE,
        log_path=log,
        coverage={
            "session_date": "2026-07-13",
            "config_hash": "great-cfg",
            "evaluations": 3,
            "bars_consumed": 4,
            "errors": 0,
        },
        finalized_at=NOW,
        reason="session_rollover",
        engine_version="great-v1",
        machine="desk",
        timezone="UTC",
        configuration="great-cfg",
    )

    reset_audit_cache()
    progress = audit_session_summaries(log, GREATNESS_ENGINE)
    floors = progress["section_7_floor_progress"]
    assert progress["eligible_sessions"] == 1
    assert progress["complete_chains"] == 1
    assert floors["meaningful_level_interactions"]["count"] == 2
    assert floors["confirm_fail_rearm_outcomes"]["count"] == 3
    assert floors["manually_reviewed_transition_chains"]["count"] == 0
    assert progress["affects_promotion"] is False


def test_archive_tampering_breaks_reconciliation_not_the_counter(tmp_path):
    log = tmp_path / "greatness_shadow.jsonl"
    _write_rows(log, [_greatness_row("READY")])
    finalize_session(
        engine=GREATNESS_ENGINE,
        log_path=log,
        coverage={
            "session_date": "2026-07-13",
            "config_hash": "great-cfg",
            "evaluations": 1,
            "bars_consumed": 1,
            "errors": 0,
        },
        finalized_at=NOW,
        reason="session_rollover",
        engine_version="great-v1",
        machine="desk",
        timezone="UTC",
        configuration="great-cfg",
    )
    raw_dir, _ = evidence_directories(log, GREATNESS_ENGINE)
    archive = next(raw_dir.glob("*.jsonl"))
    archive.write_text(archive.read_text(encoding="utf-8") + "{broken\n", encoding="utf-8")

    reset_audit_cache()
    progress = audit_session_summaries(log, GREATNESS_ENGINE)
    assert progress["eligible_sessions"] == 0
    assert progress["incomplete_sessions"] == 1
    reasons = " ".join(progress["incomplete_session_details"][0]["reasons"])
    assert "checksum" in reasons
    assert "malformed" in reasons


def test_retention_keeps_the_newest_safety_floor(tmp_path):
    log = tmp_path / "spy_state_shadow.jsonl"
    raw_dir, _ = evidence_directories(log, SPY_ENGINE)
    raw_dir.mkdir(parents=True)
    count = RETENTION_POLICY["raw_keep_newest"] + 5
    for index in range(count):
        path = raw_dir / f"archive-{index:02d}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        old = 1_600_000_000 + index
        os.utime(path, (old, old))

    removed = apply_retention(log, SPY_ENGINE)

    assert removed["raw_age_pruned"] == 5
    assert len(list(raw_dir.glob("*.jsonl"))) == RETENTION_POLICY["raw_keep_newest"]
