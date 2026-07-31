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
    scan_raw_archive,
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


# ---------------------------------------------------------------------------
# P0: total timestamp handling.  The real SPY log's first row (2026-07-13,
# written before tz normalization) is NAIVE and carries no timezone field;
# every later row is aware.  scan_raw_archive() used to raise TypeError on the
# first naive/aware comparison, which killed the very first live rollover and
# silently froze SPY shadow recording.  These tests are real-shaped: same
# fields, same one-naive-then-aware sequence.
# ---------------------------------------------------------------------------
def _real_shaped_mixed_log(tmp_path: Path) -> Path:
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_rows(
        log,
        [
            # The legacy row: naive stamp, v2 schema, NO timezone field, no
            # complete_bar_ts (exactly like the live 2026-07-13 first row).
            {
                "schema": "spy_state_shadow_v2",
                "session_date": "2026-07-13",
                "config_hash": "spy-cfg",
                "engine_version": "spy-v1",
                "machine": "desk",
                "state": "BEAR_IMPULSE",
                "evaluated_at": "2026-07-13T12:00:09",
            },
            _spy_row(
                "spy_state_shadow_v2",
                state="COUNTERMOVE_ARMED",
                evaluated_at="2026-07-13T12:20:09-07:00",
                timezone="Pacific Daylight Time",  # NOT zoneinfo-resolvable
            ),
            _spy_row(
                "spy_state_shadow_v2",
                state="RANGE",
                evaluated_at="2026-07-13T13:18:35-07:00",
                timezone="Pacific Daylight Time",
            ),
            _spy_row(
                "spy_state_shadow_v4",
                session="2026-07-14",
                state="RANGE",
                evaluated_at="2026-07-14T06:35:00-07:00",
            ),
        ],
    )
    return log


def test_naive_legacy_timestamp_never_raises_and_scans_every_group(tmp_path):
    log = _real_shaped_mixed_log(tmp_path)

    scan = scan_raw_archive(log, SPY_ENGINE)  # must not raise

    assert set(scan["groups"]) == {"2026-07-13|spy-cfg", "2026-07-14|spy-cfg"}
    day1 = scan["groups"]["2026-07-13|spy-cfg"]
    # The naive stamp was normalized from RECORDED evidence: the sibling rows'
    # unanimous -07:00 offset ("Pacific Daylight Time" does not resolve via
    # zoneinfo, so the row-timezone path correctly does not fire).
    assert day1["timestamps_legacy_naive"] == 1
    assert day1["timestamps_naive_normalized"] == 1
    assert day1["timestamps_unresolved"] == 0
    assert day1["naive_timezone_source"].startswith("sibling_offset:")
    # With the naive row normalized, the 12:00:09 -> 12:20:09 boundary is a
    # real 1200s duration attributed to the naive row's state.
    assert day1["state_duration_seconds_observed"]["BEAR_IMPULSE"] == 1200
    assert scan["timestamps_legacy_naive"] == 1
    assert scan["timestamps_unresolved"] == 0


def test_unknown_timezone_is_an_explicit_anomaly_with_no_fabricated_duration(
    tmp_path, monkeypatch
):
    # No sibling aware rows, no row timezone, and no configured market tz:
    # nothing trustworthy exists, so nothing may be invented.
    monkeypatch.delenv("TRADINGBOT_MARKET_TIMEZONE", raising=False)
    import market_session

    monkeypatch.setattr(
        market_session, "_resolve_configured_timezone_name", lambda *a: None
    )
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_rows(
        log,
        [
            _spy_row("spy_state_shadow_v2", state="RANGE", evaluated_at="2026-07-13T09:35:00"),
            _spy_row("spy_state_shadow_v2", state="BULL_IMPULSE", evaluated_at="2026-07-13T09:40:00"),
        ],
    )

    scan = scan_raw_archive(log, SPY_ENGINE)

    day = scan["groups"]["2026-07-13|spy-cfg"]
    assert day["timestamps_legacy_naive"] == 2
    assert day["timestamps_naive_normalized"] == 0
    assert day["timestamps_unresolved"] == 2
    # Rows are retained and counted; no duration is invented anywhere.
    assert day["valid_rows"] == 2
    assert day["state_observations"] == {"RANGE": 1, "BULL_IMPULSE": 1}
    assert day["state_transitions"] == {"RANGE->BULL_IMPULSE": 1}
    assert day["state_duration_seconds_observed"] == {}


def test_configured_market_timezone_normalizes_deterministically(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGBOT_MARKET_TIMEZONE", "America/New_York")
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_rows(
        log,
        [
            _spy_row("spy_state_shadow_v2", state="RANGE", evaluated_at="2026-07-13T09:35:00"),
            _spy_row("spy_state_shadow_v2", state="BULL_IMPULSE", evaluated_at="2026-07-13T09:50:00"),
        ],
    )

    scan = scan_raw_archive(log, SPY_ENGINE)

    day = scan["groups"]["2026-07-13|spy-cfg"]
    assert day["timestamps_naive_normalized"] == 2
    assert day["naive_timezone_source"] == "configured:America/New_York"
    # Both stamps share the configured zone, so the duration is exact.
    assert day["state_duration_seconds_observed"] == {"RANGE": 900}


def test_mixed_utc_offsets_compare_correctly(tmp_path):
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_rows(
        log,
        [
            _spy_row("spy_state_shadow_v4", state="RANGE", evaluated_at="2026-07-13T09:35:00-07:00"),
            # Same instant expressed in a different offset, plus 10 minutes.
            _spy_row("spy_state_shadow_v4", state="BULL_IMPULSE", evaluated_at="2026-07-13T12:45:00-04:00"),
        ],
    )

    scan = scan_raw_archive(log, SPY_ENGINE)

    day = scan["groups"]["2026-07-13|spy-cfg"]
    assert day["timestamps_legacy_naive"] == 0
    assert day["state_duration_seconds_observed"] == {"RANGE": 600}


def test_malformed_timestamp_breaks_the_duration_boundary(tmp_path):
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_rows(
        log,
        [
            _spy_row("spy_state_shadow_v4", state="RANGE", evaluated_at="2026-07-13T09:35:00-07:00"),
            _spy_row("spy_state_shadow_v4", state="BULL_IMPULSE", evaluated_at="not-a-time"),
            _spy_row("spy_state_shadow_v4", state="STABILIZING", evaluated_at="2026-07-13T10:35:00-07:00"),
        ],
    )

    scan = scan_raw_archive(log, SPY_ENGINE)

    day = scan["groups"]["2026-07-13|spy-cfg"]
    assert day["timestamps_malformed"] == 1
    # The old walker kept the pre-anomaly stamp and attributed the whole
    # 09:35 -> 10:35 hour to the malformed row's state.  Elapsed time across an
    # unknown clock boundary is invented time: the chain must restart instead.
    assert day["state_duration_seconds_observed"] == {}
    assert day["state_transitions"] == {
        "RANGE->BULL_IMPULSE": 1,
        "BULL_IMPULSE->STABILIZING": 1,
    }


def test_normalized_naive_final_row_keeps_the_duration_tail(tmp_path):
    """Finding 2: the LAST state row is naive but normalized from trustworthy
    sibling evidence; the tail duration against aware coverage must survive.

    Before the fix, last_state_at stored the ORIGINAL naive text, and
    _session_metrics' aware-vs-naive comparison raised TypeError, silently
    dropping the final state's duration."""
    from diagnostics.shadow_session_rollup import _session_metrics

    log = tmp_path / "spy_state_shadow.jsonl"
    _write_rows(
        log,
        [
            _spy_row(
                "spy_state_shadow_v2",
                state="RANGE",
                evaluated_at="2026-07-13T12:40:00-07:00",
                timezone="Pacific Daylight Time",
            ),
            # The naive LAST row, normalizable from the sibling's -07:00 offset.
            {
                "schema": "spy_state_shadow_v2",
                "session_date": "2026-07-13",
                "config_hash": "spy-cfg",
                "engine_version": "spy-v1",
                "machine": "desk",
                "state": "BULL_IMPULSE",
                "evaluated_at": "2026-07-13T12:55:00",
            },
        ],
    )
    scan = scan_raw_archive(log, SPY_ENGINE)
    day = scan["groups"]["2026-07-13|spy-cfg"]
    # The aware instant feeds durations; the recorded text stays as provenance.
    assert day["last_state_at"] == "2026-07-13T12:55:00-07:00"
    assert day["last_state_at_recorded"] == "2026-07-13T12:55:00"
    assert day["timestamps_naive_normalized"] == 1

    metrics = _session_metrics(
        SPY_ENGINE, day, {"last_evaluation_at": "2026-07-13T13:10:00-07:00"}
    )
    # 12:55 -> 13:10 aware-vs-aware: the final state's 900s tail is counted.
    assert metrics["state_duration_seconds"]["BULL_IMPULSE"] == 900


def test_out_of_order_timestamps_are_counted_and_block_eligibility(tmp_path):
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_rows(
        log,
        [
            _spy_row("spy_state_shadow_v4", state="RANGE", evaluated_at="2026-07-13T10:00:00-07:00"),
            # Time runs BACKWARDS between two trusted stamps.
            _spy_row("spy_state_shadow_v4", state="BULL_IMPULSE", evaluated_at="2026-07-13T09:30:00-07:00"),
            _spy_row("spy_state_shadow_v4", state="STABILIZING", evaluated_at="2026-07-13T09:45:00-07:00"),
        ],
    )
    scan = scan_raw_archive(log, SPY_ENGINE)
    day = scan["groups"]["2026-07-13|spy-cfg"]
    assert day["timestamps_out_of_order"] == 1
    # No duration across the backwards boundary; the chain restarts AT the
    # anomalous stamp, so the later ordered pair still measures honestly.
    assert day["state_duration_seconds_observed"] == {"BULL_IMPULSE": 900}

    finalize_session(
        engine=SPY_ENGINE,
        log_path=log,
        coverage={
            "session_date": "2026-07-13",
            "config_hash": "spy-cfg",
            "evaluations": 3,
            "usable_evaluations": 3,
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
    day1 = next(
        item
        for item in progress["incomplete_session_details"]
        if item["session_date"] == "2026-07-13"
    )
    assert any("out-of-order raw timestamps" in reason for reason in day1["reasons"])


def test_timestamp_anomalies_block_eligibility_and_tampering_is_detected(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGBOT_MARKET_TIMEZONE", raising=False)
    import market_session

    monkeypatch.setattr(
        market_session, "_resolve_configured_timezone_name", lambda *a: None
    )
    log = tmp_path / "spy_state_shadow.jsonl"
    # Unresolvable naive stamp in an otherwise complete-looking session.
    _write_rows(
        log,
        [
            _spy_row("spy_state_shadow_v2", state="RANGE", evaluated_at="2026-07-13T09:35:00"),
            _spy_row(
                "spy_state_shadow_v4",
                state="BULL_IMPULSE",
                evaluated_at="2026-07-14T09:40:00-07:00",
                session="2026-07-14",
            ),
        ],
    )
    coverage = {
        "session_date": "2026-07-13",
        "config_hash": "spy-cfg",
        "evaluations": 1,
        "usable_evaluations": 1,
        "errors": 0,
    }
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

    reset_audit_cache()
    progress = audit_session_summaries(log, SPY_ENGINE)
    day1 = next(
        item
        for item in progress["incomplete_session_details"]
        if item["session_date"] == "2026-07-13"
    )
    assert any("unresolvable raw timestamps" in reason for reason in day1["reasons"])
    assert progress["eligible_sessions"] == 0

    # Tampering with the stored timestamp counters must break reconciliation.
    _, summary_dir = evidence_directories(log, SPY_ENGINE)
    target = next(
        path
        for path in summary_dir.glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["session_date"] == "2026-07-13"
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["raw_stats"]["timestamps_unresolved"] = 0
    target.write_text(json.dumps(payload), encoding="utf-8")

    reset_audit_cache()
    tampered = audit_session_summaries(log, SPY_ENGINE)
    day1 = next(
        item
        for item in tampered["incomplete_session_details"]
        if item["session_date"] == "2026-07-13"
    )
    assert any(
        "summary counters do not reconcile" in reason for reason in day1["reasons"]
    )
