#!/usr/bin/env python3
"""Phase 0 capture-readiness evidence for the trade-discovery learning program.

`GUI_TRADE_DISCOVERY_LEARNING_PLAN.md` Phase 0 will not let any of the later
phases start until the desk can prove four things about its own learning
inputs:

- the review-decision ledger is being created, is growing, and its failures
  are VISIBLE rather than silently swallowed (task 3);
- the scoreboard, policy, outcome-join, writer, and snapshot state are all
  inspectable from System Health (task 4);
- the champion setup-scoring configuration is snapshotted, with automatic
  tuner runs recommendation-only and the explicit trader action visible (task 7);
- every pre-v2 artifact is labeled Exploratory / Non-Promotable, and the
  preference-ordering gate is actually holding (tasks 6 and 8).

This module answers those questions and nothing else. It reads; it never
repairs, rebuilds, promotes, or invokes the tuner. Every function returns the
same check shape ``operations_audit`` uses, so the Health page renders them
with no new panel code.

Import-light on purpose (no Qt, no pandas): System Health calls it on a timer
and the tests drive it headless.

Run:
    .venv/Scripts/python.exe scripts/review_capture_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import (  # noqa: E402
    ALERT_REVIEW_EVENTS_FILE,
    MASTER_AVWAP_SCORING_CONFIG_FILE,
    REVIEW_POLICY_DRAFT_FILE,
    REVIEW_POLICY_FILE,
    REVIEW_PREFERENCE_STATE_FILE,
)
from review_events import (  # noqa: E402
    LEGACY_REVIEW_EVENTS_SCHEMA,
    REVIEW_EVENTS_SCHEMA,
    SUPPORTED_REVIEW_EVENTS_SCHEMAS,
    review_event_shard_installation_id,
    review_event_sources,
    review_event_store_mtime,
)
from review_guidance import (  # noqa: E402
    MIN_SEGMENT_SHOWN,
    ORDERING_PREFERENCE,
    resolve_ordering_mode,
)

CAPTURE_AUDIT_SCHEMA = "review_capture_audit_v1"

# The plan's cold-start instruction: collect roughly two to three weeks of
# normal sessions before anything is tuned. Ten distinct trade dates is the
# floor this audit reports progress against; it is an observability target,
# not a promotion gate (the promotion floors live in the plan's manifests).
CAPTURE_SESSION_FLOOR = 10

# Until the Phase 3 identity/action-parity gate lands, every episode in the
# ledger is folded by (trade_date, symbol) and every "take" includes arming a
# watch. Neither is promotable evidence, and the GUI must say so.
EVIDENCE_LABEL = "Exploratory / Non-Promotable"
EVIDENCE_REASONS = (
    "Episodes are keyed by (trade_date, symbol): Swing and M5, long and short, "
    "and separate attempts on one ticker still collapse into one sample.",
    'Engagement is not entry: add-to-focus, favorite, and arm-watch all count '
    'as a "take", so P(take|shown) is an engagement probability.',
    "Setup-chart reviews have no impression denominator, so likes/dislikes are "
    "not a representative sample of what was seen.",
    "Swing and M5 share one forward-return outcome definition; day-trade "
    "expectancy is not established by daily closes.",
)

_STATUS_ORDER = {"healthy": 0, "degraded": 1, "unhealthy": 2}


# ---------------------------------------------------------------------------
# Shared helpers (same check shape as operations_audit.py)
# ---------------------------------------------------------------------------
def _check(
    check_id: str,
    label: str,
    status: str,
    summary: str,
    *,
    source: Path,
    updated_at: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "status": status,
        "summary": summary,
        "updated_at": str(updated_at or ""),
        "source": str(source),
        "details": details or {},
    }


def _mtime_text(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def _review_store_mtime_text(path: Path, *, shards_dir: Path | None = None) -> str:
    value = review_event_store_mtime(
        path,
        shards_dir=shards_dir,
        include_shards=(Path(path) == Path(ALERT_REVIEW_EVENTS_FILE) or shards_dir is not None),
    )
    return (
        datetime.fromtimestamp(value).isoformat(timespec="seconds")
        if value is not None
        else ""
    )


def _age_days(value: str, now: datetime) -> float | None:
    """Age in days, comparing both sides as local wall clock.

    These artifacts are written with naive local timestamps while the audit's
    ``now`` arrives market-local and aware, so both are flattened to wall
    clock rather than subtracted across that boundary.
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    reference = now.replace(tzinfo=None) if now.tzinfo is not None else now
    return max(0.0, (reference - parsed).total_seconds() / 86400.0)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Task 3 - the review-decision ledger
# ---------------------------------------------------------------------------
def scan_review_log(path: Path, *, shards_dir: Path | None = None) -> dict[str, Any]:
    """Parse the ledger the way an auditor would, not the way a reader does.

    ``load_review_events`` skips malformed lines so one corrupt row can never
    cost the trader a session. That is right for the runtime and wrong for the
    audit: here every skipped line is counted and reported, because a ledger
    that quietly drops rows is exactly the silent failure Phase 0 exists to
    surface.
    """
    path = Path(path)
    include_shards = path == Path(ALERT_REVIEW_EVENTS_FILE) or shards_dir is not None
    sources = review_event_sources(
        path,
        shards_dir=shards_dir,
        include_shards=include_shards,
    )
    stats: dict[str, Any] = {
        "exists": bool(sources),
        "readable": True,
        "bytes": 0,
        "rows": 0,
        "legacy_rows": 0,
        "partitioned_rows": 0,
        "malformed_lines": 0,
        "schemas": {},
        "actions": {},
        "sessions": 0,
        "machines": {},
        "installations": {},
        "installation_machines": {},
        "source_files": [str(source) for source in sources],
        "shard_files": 0,
        "legacy_exists": path.exists(),
        "legacy_v2_rows": 0,
        "shard_legacy_rows": 0,
        "shard_identity_mismatches": 0,
        "duplicate_record_ids": 0,
        "source_details": [],
        "first_ts": "",
        "last_ts": "",
        "first_trade_date": "",
        "last_trade_date": "",
        "rows_missing_symbol": 0,
    }
    if not sources:
        return stats

    schemas: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    machines: Counter[str] = Counter()
    installations: Counter[str] = Counter()
    installation_machines: dict[str, set[str]] = {}
    trade_dates: set[str] = set()
    timestamps: list[str] = []
    record_ids: set[str] = set()
    for source in sources:
        is_legacy = source == path
        encoded_installation = (
            "" if is_legacy else review_event_shard_installation_id(source)
        )
        try:
            source_bytes = source.stat().st_size
            text = source.read_text(encoding="utf-8")
        except OSError:
            stats["readable"] = False
            continue
        stats["bytes"] += source_bytes
        if not is_legacy:
            stats["shard_files"] += 1
        source_rows = 0
        source_malformed = 0
        source_installations: Counter[str] = Counter()
        source_machines: Counter[str] = Counter()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                stats["malformed_lines"] += 1
                source_malformed += 1
                continue
            if not isinstance(row, dict):
                stats["malformed_lines"] += 1
                source_malformed += 1
                continue
            stats["rows"] += 1
            source_rows += 1
            if is_legacy:
                stats["legacy_rows"] += 1
            else:
                stats["partitioned_rows"] += 1
            schema = str(row.get("schema") or "(missing)")
            schemas[schema] += 1
            actions[str(row.get("action") or "(missing)")] += 1
            machine = str(row.get("machine") or "(unrecorded)")
            machines[machine] += 1
            source_machines[machine] += 1
            installation_id = str(row.get("installation_id") or "").strip().lower()
            if installation_id:
                installations[installation_id] += 1
                source_installations[installation_id] += 1
                installation_machines.setdefault(installation_id, set()).add(machine)
            if is_legacy and schema == REVIEW_EVENTS_SCHEMA:
                stats["legacy_v2_rows"] += 1
            if not is_legacy and schema == LEGACY_REVIEW_EVENTS_SCHEMA:
                stats["shard_legacy_rows"] += 1
            if not is_legacy and (
                not encoded_installation or installation_id != encoded_installation
            ):
                stats["shard_identity_mismatches"] += 1
            record_id = str(row.get("review_record_id") or "").strip()
            if record_id:
                if record_id in record_ids:
                    stats["duplicate_record_ids"] += 1
                record_ids.add(record_id)
            if not str(row.get("symbol") or "").strip():
                stats["rows_missing_symbol"] += 1
            trade_date = str(row.get("trade_date") or "").strip()
            if trade_date:
                trade_dates.add(trade_date)
            ts = str(row.get("ts") or "").strip()
            if ts:
                timestamps.append(ts)
        stats["source_details"].append(
            {
                "path": str(source),
                "kind": "legacy" if is_legacy else "installation_shard",
                "encoded_installation_id": encoded_installation,
                "rows": source_rows,
                "malformed_lines": source_malformed,
                "installations": dict(source_installations),
                "machines": dict(source_machines),
            }
        )

    stats["schemas"] = dict(schemas)
    stats["actions"] = dict(actions)
    stats["machines"] = dict(machines)
    stats["installations"] = dict(installations)
    stats["installation_machines"] = {
        identity: sorted(names)
        for identity, names in sorted(installation_machines.items())
    }
    stats["sessions"] = len(trade_dates)
    if trade_dates:
        stats["first_trade_date"] = min(trade_dates)
        stats["last_trade_date"] = max(trade_dates)
    if timestamps:
        stats["first_ts"] = min(timestamps)
        stats["last_ts"] = max(timestamps)
    return stats


def review_log_check(
    path: Path = ALERT_REVIEW_EVENTS_FILE,
    *,
    now: datetime | None = None,
    shards_dir: Path | None = None,
) -> dict[str, Any]:
    moment = now or datetime.now()
    path = Path(path)
    partitioned_store = path == Path(ALERT_REVIEW_EVENTS_FILE) or shards_dir is not None
    stats = scan_review_log(path, shards_dir=shards_dir)
    details = dict(stats)
    details["session_floor"] = CAPTURE_SESSION_FLOOR
    details["expected_schema"] = REVIEW_EVENTS_SCHEMA
    details["supported_schemas"] = sorted(SUPPORTED_REVIEW_EVENTS_SCHEMAS)
    details["partitioned_store"] = partitioned_store

    if not stats["exists"]:
        return _check(
            "review_event_log",
            "Review decision log",
            "degraded",
            "No review-decision log yet; it is created by the first reviewed alert.",
            source=path,
            details=details,
        )
    if not stats["readable"]:
        return _check(
            "review_event_log",
            "Review decision log",
            "unhealthy",
            "Review-decision log exists but could not be read.",
            source=path,
            details=details,
        )

    writers = [name for name in stats["machines"] if name != "(unrecorded)"]
    problems: list[str] = []
    warnings: list[str] = []
    if stats["malformed_lines"]:
        problems.append(f"{stats['malformed_lines']} malformed line(s) skipped by readers")
    if stats["rows_missing_symbol"]:
        problems.append(f"{stats['rows_missing_symbol']} row(s) with no symbol")
    unexpected = sorted(
        name for name in stats["schemas"] if name not in SUPPORTED_REVIEW_EVENTS_SCHEMAS
    )
    if unexpected:
        problems.append(f"unexpected schema(s): {', '.join(unexpected)}")
    if stats["shard_identity_mismatches"]:
        problems.append(
            f"{stats['shard_identity_mismatches']} shard row(s) do not match "
            "their filename installation identity"
        )
    if stats["shard_legacy_rows"]:
        problems.append(
            f"{stats['shard_legacy_rows']} legacy-schema row(s) were written into installation shards"
        )
    if stats["duplicate_record_ids"]:
        problems.append(
            f"{stats['duplicate_record_ids']} duplicate review record id(s) across sources"
        )
    if partitioned_store and stats["legacy_v2_rows"]:
        problems.append(
            f"{stats['legacy_v2_rows']} current-schema row(s) were appended to the legacy shared file"
        )
    if not partitioned_store and len(writers) > 1:
        problems.append(f"{len(writers)} machines appended: {', '.join(sorted(writers))}")
    if partitioned_store:
        legacy_writers = sorted(
            {
                name
                for source in stats["source_details"]
                if source["kind"] == "legacy"
                for name in source["machines"]
                if name != "(unrecorded)"
            }
        )
        if len(legacy_writers) > 1:
            warnings.append(
                f"legacy unpartitioned history contains {len(legacy_writers)} machine names "
                f"({', '.join(legacy_writers)}); it remains readable but cannot prove no rows were lost"
            )
        if stats["legacy_rows"] and not stats["partitioned_rows"]:
            warnings.append(
                "partitioned review capture has no live row yet; storage migration is not live-validated"
            )
        details["legacy_writers"] = legacy_writers
    details["problems"] = problems
    details["warnings"] = warnings
    details["writers"] = sorted(writers)
    details["installation_writers"] = sorted(stats["installations"])
    details["renamed_installations"] = {
        identity: machines
        for identity, machines in stats["installation_machines"].items()
        if len(machines) > 1
    }

    last_age = _age_days(stats["last_ts"], moment)
    details["last_event_age_days"] = round(last_age, 2) if last_age is not None else None

    if problems:
        status = "unhealthy"
    elif warnings or stats["rows"] == 0:
        status = "degraded"
    else:
        status = "healthy"
    summary = (
        f"{stats['rows']} decision(s) over {stats['sessions']}/{CAPTURE_SESSION_FLOOR} "
        f"session(s), {len(stats['installations'])} partitioned installation(s); "
        f"{stats['malformed_lines']} malformed."
    )
    if problems:
        summary += " " + "; ".join(problems) + "."
    elif warnings:
        summary += " " + "; ".join(warnings) + "."
    elif stats["rows"] == 0:
        summary += " Log is present but still empty."
    return _check(
        "review_event_log",
        "Review decision log",
        status,
        summary,
        source=path,
        updated_at=stats["last_ts"] or _review_store_mtime_text(path, shards_dir=shards_dir),
        details=details,
    )


# ---------------------------------------------------------------------------
# Task 4 - scoreboard, outcome join, and policy visibility
# ---------------------------------------------------------------------------
def scoreboard_check(
    state_path: Path = REVIEW_PREFERENCE_STATE_FILE,
    log_path: Path = ALERT_REVIEW_EVENTS_FILE,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    moment = now or datetime.now()
    state = _read_json(Path(state_path))
    if state is None:
        return _check(
            "review_scoreboard",
            "Review preference scoreboard",
            "degraded",
            "No scoreboard yet; review_learning.py rebuilds it from the decision log.",
            source=state_path,
            details={"exists": Path(state_path).exists()},
        )

    dimensions = state.get("dimensions") if isinstance(state.get("dimensions"), dict) else {}
    segments = 0
    qualified = 0
    for table in dimensions.values():
        if not isinstance(table, dict):
            continue
        for stats in table.values():
            if not isinstance(stats, dict):
                continue
            segments += 1
            if int(stats.get("shown", 0) or 0) >= MIN_SEGMENT_SHOWN:
                qualified += 1

    generated_at = str(state.get("generated_at") or "")
    age = _age_days(generated_at, moment)
    # Stale means the ledger has grown past the scoreboard, not merely that
    # some hours passed: an unrebuilt scoreboard is guidance from yesterday.
    log_mtime = _review_store_mtime_text(Path(log_path))
    behind_log = bool(log_mtime and generated_at and log_mtime > generated_at)
    details = {
        "generated_at": generated_at,
        "age_days": round(age, 2) if age is not None else None,
        "window_days": state.get("window_days"),
        "event_rows": int(state.get("event_rows", 0) or 0),
        "episodes": int(state.get("episodes", 0) or 0),
        "shown": int(state.get("shown", 0) or 0),
        "takes": int(state.get("takes", 0) or 0),
        "overall_take_rate": state.get("overall_take_rate"),
        "dimension_count": len(dimensions),
        "segment_count": segments,
        "segments_meeting_floor": qualified,
        "min_segment_shown": MIN_SEGMENT_SHOWN,
        "behind_decision_log": behind_log,
        "decision_log_modified_at": log_mtime,
        "take_metric_meaning": "engagement probability (arming a watch counts as a take)",
    }
    status = "degraded" if behind_log or age is None else "healthy"
    summary = (
        f"{details['episodes']} episode(s), {qualified}/{segments} segment(s) at the "
        f"n>={MIN_SEGMENT_SHOWN} floor."
    )
    if behind_log:
        summary += " Decision log is newer than the scoreboard."
    return _check(
        "review_scoreboard",
        "Review preference scoreboard",
        status,
        summary,
        source=state_path,
        updated_at=generated_at,
        details=details,
    )


def outcome_join_check(
    state_path: Path = REVIEW_PREFERENCE_STATE_FILE, *, now: datetime | None = None
) -> dict[str, Any]:
    """How many reviewed episodes actually carry an outcome.

    An unjoined episode is a decision the desk can never grade, so this is the
    single number that says whether the learning loop is closed at all.
    """
    state = _read_json(Path(state_path))
    if state is None:
        return _check(
            "review_outcome_join",
            "Review outcome join",
            "degraded",
            "No scoreboard yet, so no outcome-join rate can be reported.",
            source=state_path,
            details={},
        )
    episodes = int(state.get("episodes", 0) or 0)
    outcome_matches = int(state.get("outcome_matches", 0) or 0)
    forward_matches = int(state.get("forward_matches", 0) or 0)
    joined = max(outcome_matches, forward_matches)
    rate = round(joined / episodes, 3) if episodes else None
    details = {
        "episodes": episodes,
        "outcome_matches": outcome_matches,
        "forward_matches": forward_matches,
        "join_rate": rate,
        "outcome_definition": (
            "Swing and M5 share one daily forward-return definition; day-trade "
            "expectancy is NOT established by these numbers."
        ),
        "evidence_label": EVIDENCE_LABEL,
    }
    if not episodes:
        status, summary = "degraded", "No graded episodes yet."
    elif rate is not None and rate < 0.5:
        status = "degraded"
        summary = f"{joined}/{episodes} episode(s) joined to an outcome ({rate * 100:.0f}%)."
    else:
        status = "healthy"
        summary = f"{joined}/{episodes} episode(s) joined to an outcome ({rate * 100:.0f}%)."
    return _check(
        "review_outcome_join",
        "Review outcome join",
        status,
        summary,
        source=state_path,
        updated_at=str(state.get("generated_at") or ""),
        details=details,
    )


def policy_gate_check(
    policy_path: Path = REVIEW_POLICY_FILE,
    draft_path: Path = REVIEW_POLICY_DRAFT_FILE,
    *,
    ordering_mode: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The active policy AND whether its ordering power is still gated.

    Phase 0 task 6 forces preference guidance to annotation-only until the
    identity/parity gates pass. A policy on disk is fine; a policy on disk
    that is allowed to reorder the active queue before the gate is a
    production-influence violation, so it reads unhealthy here.
    """
    moment = now or datetime.now()
    mode = resolve_ordering_mode(ordering_mode)
    payload = _read_json(Path(policy_path)) or {}
    rules = [rule for rule in (payload.get("rules") or []) if isinstance(rule, dict)]
    draft = _read_json(Path(draft_path)) or {}
    draft_rules = [rule for rule in (draft.get("rules") or []) if isinstance(rule, dict)]
    generated_at = str(payload.get("generated_at") or "")
    age = _age_days(generated_at, moment)
    max_delta = max(
        (abs(int(rule.get("priority_delta", 0) or 0)) for rule in rules), default=0
    )
    details = {
        "queue_ordering_mode": mode,
        "orders_active_queue": mode == ORDERING_PREFERENCE,
        "active_rules": len(rules),
        "draft_rules": len(draft_rules),
        "author": str(payload.get("author") or ""),
        "generated_at": generated_at,
        "age_days": round(age, 2) if age is not None else None,
        "max_priority_delta": max_delta,
        "draft_path": str(draft_path),
        "gate": (
            "Preference may annotate and stamp impressions; it may not reorder "
            "the active queue, change severity, sound, budgets, or eligibility."
        ),
    }
    if mode == ORDERING_PREFERENCE:
        status = "unhealthy"
        summary = (
            f"Preference ordering is ACTIVE with {len(rules)} rule(s) before the "
            "identity/parity gate has passed."
        )
    else:
        status = "healthy"
        summary = (
            f"{len(rules)} active rule(s), {len(draft_rules)} draft; annotation-only "
            "- the review queue stays FIFO."
        )
    return _check(
        "review_policy_gate",
        "Review policy and ordering gate",
        status,
        summary,
        source=policy_path,
        updated_at=generated_at,
        details=details,
    )


# ---------------------------------------------------------------------------
# Task 7 - champion setup-scoring snapshot and tuner characterization
# ---------------------------------------------------------------------------
# Automatic tracker refresh/backfill sites now generate recommendations only.
# The explicit GUI apply button remains the sole live mutation path and requires
# a deliberate trader action.
TUNER_RUN_SITES = (
    "update_setup_tracker_from_scan(auto_tune=True) - generates recommendation "
    "artifacts without applying them (scripts/master_avwap_lib/legacy.py).",
    "backfill_setup_tracker_from_recent_sessions() - generates recommendations "
    "once after the backfill loop without applying them "
    "(scripts/master_avwap_lib/legacy.py).",
    'Master AVWAP GUI "apply" tuner button '
    "(scripts/master_avwap_lib/gui.py), explicit trader action.",
)


def scoring_config_check(
    path: Path = MASTER_AVWAP_SCORING_CONFIG_FILE, *, now: datetime | None = None
) -> dict[str, Any]:
    config_path = Path(path)
    payload = _read_json(config_path)
    if payload is None:
        return _check(
            "setup_scoring_config",
            "Setup scoring champion",
            "degraded",
            "No setup-scoring configuration on disk; the built-in defaults are active.",
            source=config_path,
            details={"exists": config_path.exists(), "tuner_run_sites": list(TUNER_RUN_SITES)},
        )
    adjustments = [
        rule for rule in (payload.get("attribute_adjustments") or []) if isinstance(rule, dict)
    ]
    by_source: Counter[str] = Counter(
        str(rule.get("source") or "(unset)") for rule in adjustments
    )
    weights = payload.get("signal_weights") if isinstance(payload.get("signal_weights"), dict) else {}
    weight_count = sum(
        len(side_map)
        for bucket in weights.values()
        if isinstance(bucket, dict)
        for side_map in bucket.values()
        if isinstance(side_map, dict)
    )
    updated_at = _mtime_text(config_path)
    details = {
        "sha256": _sha256(config_path),
        "bytes": config_path.stat().st_size if config_path.exists() else 0,
        "attribute_rules": len(adjustments),
        "rules_by_source": dict(by_source),
        "auto_tuner_rules": int(by_source.get("auto_tuner", 0)),
        "signal_weight_entries": weight_count,
        "modified_at": updated_at,
        "tuner_run_sites": list(TUNER_RUN_SITES),
        "tuner_status": (
            "Automatic tuner runs are recommendation-only. Live config mutation "
            "requires the explicit trader-operated GUI apply action."
        ),
    }
    return _check(
        "setup_scoring_config",
        "Setup scoring champion",
        "healthy",
        (
            f"Config {details['sha256'][:12] or '?'}; {len(adjustments)} attribute rule(s) "
            f"({details['auto_tuner_rules']} from the auto-tuner)."
        ),
        source=config_path,
        updated_at=updated_at,
        details=details,
    )


# ---------------------------------------------------------------------------
# Task 8 - the promotability label
# ---------------------------------------------------------------------------
def evidence_label_check(
    *, ordering_mode: str | None = None, source: Path = Path(__file__)
) -> dict[str, Any]:
    mode = resolve_ordering_mode(ordering_mode)
    gated = mode != ORDERING_PREFERENCE
    return _check(
        "review_evidence_label",
        "Learning evidence status",
        "healthy" if gated else "unhealthy",
        (
            f"{EVIDENCE_LABEL}: pre-v2 identity and action semantics. "
            + (
                "Preference ordering is gated to annotation-only."
                if gated
                else "Preference ordering is ACTIVE despite the label."
            )
        ),
        source=source,
        details={
            "label": EVIDENCE_LABEL,
            "reasons": list(EVIDENCE_REASONS),
            "queue_ordering_mode": mode,
            "promotion_clock_started": False,
            "clock_starts_when": (
                "Corrected v2 identity and action semantics produce stable, "
                "independently attributable episodes (plan Phase 3 exit gate)."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_review_capture_checks(
    *,
    now: datetime | None = None,
    review_events_path: Path | str | None = None,
    preference_state_path: Path | str | None = None,
    policy_path: Path | str | None = None,
    policy_draft_path: Path | str | None = None,
    scoring_config_path: Path | str | None = None,
    ordering_mode: str | None = None,
) -> list[dict[str, Any]]:
    """The Phase 0 checks, in the shape ``operations_audit`` publishes."""
    moment = now or datetime.now()
    events = Path(review_events_path) if review_events_path is not None else ALERT_REVIEW_EVENTS_FILE
    state = (
        Path(preference_state_path)
        if preference_state_path is not None
        else REVIEW_PREFERENCE_STATE_FILE
    )
    policy = Path(policy_path) if policy_path is not None else REVIEW_POLICY_FILE
    draft = Path(policy_draft_path) if policy_draft_path is not None else REVIEW_POLICY_DRAFT_FILE
    scoring = (
        Path(scoring_config_path)
        if scoring_config_path is not None
        else MASTER_AVWAP_SCORING_CONFIG_FILE
    )
    return [
        review_log_check(events, now=moment),
        scoreboard_check(state, events, now=moment),
        outcome_join_check(state, now=moment),
        policy_gate_check(policy, draft, ordering_mode=ordering_mode, now=moment),
        scoring_config_check(scoring, now=moment),
        evidence_label_check(ordering_mode=ordering_mode),
    ]


def build_review_capture_audit(**kwargs) -> dict[str, Any]:
    """Standalone payload for the CLI and for tests."""
    checks = build_review_capture_checks(**kwargs)
    counts = Counter(check["status"] for check in checks)
    overall = max((check["status"] for check in checks), key=lambda item: _STATUS_ORDER[item])
    return {
        "schema": CAPTURE_AUDIT_SCHEMA,
        "generated_at": (kwargs.get("now") or datetime.now()).isoformat(timespec="seconds"),
        "status": overall,
        "evidence_label": EVIDENCE_LABEL,
        "summary": {
            "healthy": int(counts.get("healthy", 0)),
            "degraded": int(counts.get("degraded", 0)),
            "unhealthy": int(counts.get("unhealthy", 0)),
            "total": len(checks),
        },
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 0 capture-readiness audit for the GUI learning program."
    )
    parser.add_argument("--json", action="store_true", help="Print the complete JSON payload.")
    args = parser.parse_args(argv)
    payload = build_review_capture_audit()
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload["status"] != "unhealthy" else 1
    print(f"{payload['status'].upper()} - evidence is {payload['evidence_label']}")
    for check in payload["checks"]:
        print(f"{check['status'].upper():9} {check['label']}: {check['summary']}")
    return 0 if payload["status"] != "unhealthy" else 1


if __name__ == "__main__":
    raise SystemExit(main())
