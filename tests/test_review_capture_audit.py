"""Phase 0 capture readiness: the audit must make silent failures visible.

Covers GUI_TRADE_DISCOVERY_LEARNING_PLAN.md Phase 0 tasks 3, 4, 7, and 8 -
decision-log durability and schema, scoreboard/outcome/policy visibility, the
champion scoring snapshot, and the Exploratory / Non-Promotable label.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from review_capture_audit import (  # noqa: E402
    CAPTURE_SESSION_FLOOR,
    EVIDENCE_LABEL,
    build_review_capture_audit,
    evidence_label_check,
    outcome_join_check,
    policy_gate_check,
    review_log_check,
    scan_review_log,
    scoreboard_check,
    scoring_config_check,
)

NOW = datetime(2026, 7, 28, 16, 34, 0)


def _event(**overrides) -> dict:
    row = {
        "schema": "review_events_v1",
        "ts": "2026-07-28T10:15:00",
        "trade_date": "2026-07-28",
        "machine": "desk",
        "action": "shown",
        "symbol": "NVDA",
        "side": "LONG",
    }
    row.update(overrides)
    return row


def _write_log(path: Path, rows: list, *, extra_lines: tuple[str, ...] = ()) -> Path:
    lines = [json.dumps(row) for row in rows]
    lines.extend(extra_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Task 3 - decision-log durability and schema
# ---------------------------------------------------------------------------
def test_missing_log_is_degraded_not_a_failure(tmp_path):
    check = review_log_check(tmp_path / "absent.jsonl", now=NOW)
    assert check["status"] == "degraded"
    assert check["details"]["exists"] is False
    assert check["details"]["session_floor"] == CAPTURE_SESSION_FLOOR


def test_growing_log_reports_sessions_actions_and_writer(tmp_path):
    path = _write_log(
        tmp_path / "events.jsonl",
        [
            _event(),
            _event(action="add_focus", ts="2026-07-28T10:16:00"),
            _event(trade_date="2026-07-27", ts="2026-07-27T11:00:00", symbol="AMD"),
        ],
    )
    check = review_log_check(path, now=NOW)
    assert check["status"] == "healthy"
    assert check["details"]["rows"] == 3
    assert check["details"]["sessions"] == 2
    assert check["details"]["actions"] == {"shown": 2, "add_focus": 1}
    assert check["details"]["writers"] == ["desk"]
    assert check["details"]["problems"] == []
    assert f"3 decision(s) over 2/{CAPTURE_SESSION_FLOOR}" in check["summary"]


def test_malformed_lines_are_counted_not_silently_skipped(tmp_path):
    """The runtime reader drops bad lines on purpose; the audit must not."""
    path = _write_log(
        tmp_path / "events.jsonl",
        [_event()],
        extra_lines=("{not json", '"a bare string"'),
    )
    from review_events import load_review_events

    assert len(load_review_events(path)) == 1  # reader is unchanged and forgiving

    check = review_log_check(path, now=NOW)
    assert check["status"] == "unhealthy"
    assert check["details"]["malformed_lines"] == 2
    assert "malformed" in check["summary"]


def test_unexpected_schema_and_second_writer_are_both_flagged(tmp_path):
    path = _write_log(
        tmp_path / "events.jsonl",
        [_event(), _event(schema="review_events_v3", machine="mini-pc", symbol="AMD")],
    )
    check = review_log_check(path, now=NOW)
    assert check["status"] == "unhealthy"
    problems = " ".join(check["details"]["problems"])
    assert "review_events_v3" in problems
    assert "2 machines appended" in problems
    assert check["details"]["writers"] == ["desk", "mini-pc"]


def test_partitioned_installations_are_safe_and_hostname_renames_are_diagnostic(tmp_path):
    legacy = tmp_path / "alert_review_events.jsonl"
    shards = tmp_path / "alert_review_events"
    shards.mkdir()
    desk_id = "1" * 32
    mini_id = "2" * 32
    _write_log(
        shards / f"review-events-{desk_id}.jsonl",
        [
            _event(
                schema="review_events_v2",
                installation_id=desk_id,
                review_record_id="desk-1",
                machine="OLD-DESK",
            ),
            _event(
                schema="review_events_v2",
                installation_id=desk_id,
                review_record_id="desk-2",
                machine="NEW-DESK",
                symbol="AMD",
            ),
        ],
    )
    _write_log(
        shards / f"review-events-{mini_id}.jsonl",
        [
            _event(
                schema="review_events_v2",
                installation_id=mini_id,
                review_record_id="mini-1",
                machine="MainPC",
                symbol="TSLA",
            )
        ],
    )

    check = review_log_check(legacy, shards_dir=shards, now=NOW)

    assert check["status"] == "healthy"
    assert check["details"]["partitioned_rows"] == 3
    assert check["details"]["shard_files"] == 2
    assert check["details"]["installation_writers"] == [desk_id, mini_id]
    assert check["details"]["renamed_installations"] == {
        desk_id: ["NEW-DESK", "OLD-DESK"]
    }
    assert check["details"]["problems"] == []


def test_shard_identity_mismatch_is_unhealthy(tmp_path):
    legacy = tmp_path / "alert_review_events.jsonl"
    shards = tmp_path / "alert_review_events"
    shards.mkdir()
    filename_id = "1" * 32
    row_id = "2" * 32
    _write_log(
        shards / f"review-events-{filename_id}.jsonl",
        [
            _event(
                schema="review_events_v2",
                installation_id=row_id,
                review_record_id="wrong-owner",
            )
        ],
    )

    check = review_log_check(legacy, shards_dir=shards, now=NOW)

    assert check["status"] == "unhealthy"
    assert check["details"]["shard_identity_mismatches"] == 1
    assert "filename installation identity" in check["summary"]


def test_legacy_multi_machine_history_is_readable_but_not_live_validated(tmp_path):
    legacy = _write_log(
        tmp_path / "alert_review_events.jsonl",
        [_event(machine="MainPC"), _event(machine="DESK", symbol="AMD")],
    )
    shards = tmp_path / "alert_review_events"

    check = review_log_check(legacy, shards_dir=shards, now=NOW)

    assert check["status"] == "degraded"
    assert check["details"]["legacy_writers"] == ["DESK", "MainPC"]
    assert check["details"]["problems"] == []
    assert "cannot prove no rows were lost" in check["summary"]


def test_scan_reports_rows_missing_a_symbol(tmp_path):
    path = _write_log(tmp_path / "events.jsonl", [_event(symbol="")])
    stats = scan_review_log(path)
    assert stats["rows"] == 1
    assert stats["rows_missing_symbol"] == 1


# ---------------------------------------------------------------------------
# Task 4 - scoreboard, outcome join, and policy visibility
# ---------------------------------------------------------------------------
def _state(**overrides) -> dict:
    state = {
        "schema": "review_learning_v1",
        "generated_at": "2026-07-28T09:00:00",
        "window_days": 90,
        "event_rows": 40,
        "episodes": 20,
        "shown": 18,
        "takes": 6,
        "outcome_matches": 14,
        "forward_matches": 3,
        "overall_take_rate": 0.33,
        "dimensions": {
            "tier": {
                "A": {"shown": 12},
                "S": {"shown": 2},
            }
        },
    }
    state.update(overrides)
    return state


def test_scoreboard_reports_segment_floor_progress(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    check = scoreboard_check(state_path, tmp_path / "absent.jsonl", now=NOW)
    assert check["status"] == "healthy"
    assert check["details"]["segment_count"] == 2
    assert check["details"]["segments_meeting_floor"] == 1
    # The metric's real meaning must travel with it (plan section 4.2).
    assert "engagement probability" in check["details"]["take_metric_meaning"]


def test_scoreboard_is_degraded_when_the_log_has_moved_past_it(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    log_path = _write_log(tmp_path / "events.jsonl", [_event()])
    check = scoreboard_check(state_path, log_path, now=NOW)
    assert check["status"] == "degraded"
    assert check["details"]["behind_decision_log"] is True


def test_outcome_join_rate_is_reported_with_its_definition(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    check = outcome_join_check(state_path, now=NOW)
    assert check["details"]["join_rate"] == 0.7
    assert check["status"] == "healthy"
    assert "day-trade" in check["details"]["outcome_definition"]

    thin_path = tmp_path / "thin.json"
    thin_path.write_text(
        json.dumps(_state(outcome_matches=2, forward_matches=1)), encoding="utf-8"
    )
    assert outcome_join_check(thin_path, now=NOW)["status"] == "degraded"


def test_policy_gate_is_healthy_while_ordering_is_annotation_only(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "review_policy_v1",
                "generated_at": "2026-07-27T20:00:00",
                "author": "fable",
                "rules": [
                    {"dimension": "tier", "segment": "B", "priority_delta": 3},
                    {"dimension": "side", "segment": "SHORT", "priority_delta": -2},
                ],
            }
        ),
        encoding="utf-8",
    )
    check = policy_gate_check(
        policy, tmp_path / "draft.json", ordering_mode="annotation_only", now=NOW
    )
    assert check["status"] == "healthy"
    assert check["details"]["active_rules"] == 2
    assert check["details"]["max_priority_delta"] == 3
    assert check["details"]["orders_active_queue"] is False


def test_policy_gate_is_unhealthy_when_preference_can_reorder_the_queue(tmp_path):
    check = policy_gate_check(
        tmp_path / "policy.json",
        tmp_path / "draft.json",
        ordering_mode="preference",
        now=NOW,
    )
    assert check["status"] == "unhealthy"
    assert check["details"]["orders_active_queue"] is True
    assert "ACTIVE" in check["summary"]


# ---------------------------------------------------------------------------
# Task 7 - champion scoring snapshot and tuner characterization
# ---------------------------------------------------------------------------
def test_scoring_config_snapshot_hashes_and_separates_tuner_rules(tmp_path):
    config = tmp_path / "scoring.json"
    config.write_text(
        json.dumps(
            {
                "signal_weights": {"current": {"LONG": {"rs": 3}, "SHORT": {"rs": -1}}},
                "attribute_adjustments": [
                    {"source": "auto_tuner", "score_delta": 2},
                    {"source": "auto_tuner", "score_delta": -1},
                    {"source": "user_preference", "score_delta": 4},
                ],
            }
        ),
        encoding="utf-8",
    )
    check = scoring_config_check(config, now=NOW)
    assert check["status"] == "healthy"
    assert len(check["details"]["sha256"]) == 64
    assert check["details"]["auto_tuner_rules"] == 2
    assert check["details"]["rules_by_source"]["user_preference"] == 1
    assert check["details"]["signal_weight_entries"] == 2
    # Automatic scan/backfill sites are recommendation-only; the GUI apply
    # action is the only live mutation path.
    assert any("auto_tune=True" in site for site in check["details"]["tuner_run_sites"])
    assert "recommendation-only" in check["details"]["tuner_status"]
    assert "explicit trader-operated GUI apply" in check["details"]["tuner_status"]


def test_missing_scoring_config_is_degraded_not_invented(tmp_path):
    check = scoring_config_check(tmp_path / "absent.json", now=NOW)
    assert check["status"] == "degraded"
    assert check["details"]["exists"] is False


# ---------------------------------------------------------------------------
# Task 8 - the promotability label
# ---------------------------------------------------------------------------
def test_evidence_label_states_the_reasons_and_that_the_clock_has_not_started():
    check = evidence_label_check(ordering_mode="annotation_only")
    assert check["status"] == "healthy"
    assert check["details"]["label"] == EVIDENCE_LABEL
    assert check["details"]["promotion_clock_started"] is False
    reasons = " ".join(check["details"]["reasons"])
    assert "(trade_date, symbol)" in reasons
    assert "Engagement is not entry" in reasons

    ungated = evidence_label_check(ordering_mode="preference")
    assert ungated["status"] == "unhealthy"


def test_full_audit_composes_every_phase0_check(tmp_path, monkeypatch):
    from review_guidance import ORDERING_MODE_ENV_VAR

    monkeypatch.delenv(ORDERING_MODE_ENV_VAR, raising=False)
    payload = build_review_capture_audit(
        now=NOW,
        review_events_path=tmp_path / "events.jsonl",
        preference_state_path=tmp_path / "state.json",
        policy_path=tmp_path / "policy.json",
        policy_draft_path=tmp_path / "draft.json",
        scoring_config_path=tmp_path / "scoring.json",
    )
    assert payload["schema"] == "review_capture_audit_v1"
    assert payload["evidence_label"] == EVIDENCE_LABEL
    assert {check["id"] for check in payload["checks"]} == {
        "review_event_log",
        "review_scoreboard",
        "review_outcome_join",
        "review_policy_gate",
        "setup_scoring_config",
        "review_evidence_label",
    }
    # Nothing on disk: everything unbuilt reads degraded, nothing reads broken.
    assert payload["status"] == "degraded"
    assert payload["summary"]["unhealthy"] == 0


def test_audit_never_writes_to_the_artifacts_it_reads(tmp_path):
    """Phase 0 is observation only: no repair, no rebuild, no tuner run."""
    log = _write_log(tmp_path / "events.jsonl", [_event()])
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    build_review_capture_audit(
        now=NOW,
        review_events_path=log,
        preference_state_path=tmp_path / "state.json",
        policy_path=tmp_path / "policy.json",
        policy_draft_path=tmp_path / "draft.json",
        scoring_config_path=tmp_path / "scoring.json",
    )
    after = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    assert after == before
