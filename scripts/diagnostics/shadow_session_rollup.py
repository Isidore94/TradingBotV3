"""Crash-safe per-session shadow summaries and bounded raw retention.

The active SPY and Greatness JSONLs are single-scope working files.  On a
session or configuration rollover this module atomically moves the active log
to a deterministic archive, derives one immutable summary per
``(engine, session_date, config_hash)``, and only then lets the writer reset its
coverage counters.  Repeating any step is safe after a crash.

The audit helpers are observational.  They rescan retained archives,
reconcile the stored summary counters, and report Section 7 floor progress;
they never make a promotion decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from diagnostics.artifact_io import atomic_write_json, config_hash, prune_by_age, prune_by_size

SESSION_SUMMARY_SCHEMA = "shadow_session_summary_v1"
RETENTION_POLICY = {
    "raw_max_age_days": 180,
    "raw_keep_newest": 30,
    "raw_max_bytes": 1024 * 1024 * 1024,
    "summary_max_age_days": 365,
    "summary_keep_newest": 60,
    "summary_max_bytes": 20 * 1024 * 1024,
}

SPY_ENGINE = "spy_state_shadow"
GREATNESS_ENGINE = "greatness_shadow"
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SPY_TERMINAL = {"RESUMED", "FAILED", "ABORTED"}
_GREATNESS_INTERACTIONS = {
    "LEVEL_TOUCHED",
    "WICK_THROUGH",
    "CLOSED_THROUGH",
    "ACCEPTED",
    "RETEST_HELD",
    "FAILED_ATTEMPT",
}
_GREATNESS_OUTCOMES = {"READY", "FAILED_ATTEMPT", "REARMED"}
_GREATNESS_TERMINAL = {"READY", "INVALIDATED"}
_AUDIT_CACHE: dict[tuple[str, str, int, int], dict[str, Any]] = {}


def _safe(value: object, fallback: str = "unknown") -> str:
    text = _SAFE_RE.sub("-", str(value or "").strip()).strip("-.")
    return text or fallback


def evidence_directories(log_path: Path | str, engine: str) -> tuple[Path, Path]:
    root = Path(log_path).parent / "shadow_evidence" / _safe(engine)
    return root / "raw", root / "summaries"


def _group_key(session_date: object, configuration: object) -> str:
    return f"{str(session_date or '').strip()}|{str(configuration or '').strip()}"


def _empty_group(session_date: str, configuration: str) -> dict[str, Any]:
    return {
        "session_date": session_date,
        "config_hash": configuration,
        "valid_rows": 0,
        "primary_rows": 0,
        "episode_rows": 0,
        "completed_bar_rows": 0,
        "incomplete_bar_rows": 0,
        "schemas": {},
        "machines": {},
        "engine_versions": {},
        "sides": {},
        "setup_families": {},
        "event_counts": {},
        "distinct_chains": 0,
        "complete_chains": 0,
        "meaningful_interactions": 0,
        "confirm_fail_rearm_outcomes": 0,
    }


def scan_raw_archive(path: Path | str, engine: str) -> dict[str, Any]:
    """Stream one raw archive and derive bounded per-scope counters."""

    path = Path(path)
    digest = hashlib.sha256()
    groups: dict[str, dict[str, Any]] = {}
    schema_counts: dict[str, Counter[str]] = defaultdict(Counter)
    machine_counts: dict[str, Counter[str]] = defaultdict(Counter)
    version_counts: dict[str, Counter[str]] = defaultdict(Counter)
    side_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    event_counts: dict[str, Counter[str]] = defaultdict(Counter)
    state_counts: dict[str, Counter[str]] = defaultdict(Counter)
    state_records: dict[str, list[tuple[str, str]]] = defaultdict(list)
    spy_chains: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    greatness_chains: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    malformed = 0
    lines = 0
    try:
        handle = path.open("rb")
    except OSError as exc:
        return {
            "path": str(path),
            "readable": False,
            "read_error": f"{type(exc).__name__}: {exc}",
            "bytes": 0,
            "sha256": "",
            "lines": 0,
            "malformed_lines": 0,
            "groups": {},
        }
    with handle:
        for raw in handle:
            lines += 1
            digest.update(raw)
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError, RecursionError):
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
                continue
            session_date = str(row.get("session_date") or "").strip()
            configuration = str(row.get("config_hash") or "").strip()
            if not session_date:
                malformed += 1
                continue
            key = _group_key(session_date, configuration)
            group = groups.setdefault(key, _empty_group(session_date, configuration))
            group["valid_rows"] += 1
            schema = str(row.get("schema") or "(missing)")
            schema_counts[key][schema] += 1
            machine_counts[key][str(row.get("machine") or "(missing)")] += 1
            version_counts[key][str(row.get("engine_version") or "(missing)")] += 1

            complete: bool | None = None
            if engine == SPY_ENGINE:
                if schema.startswith("spy_episode_"):
                    group["episode_rows"] += 1
                    complete = bool(row.get("derived_from_completed_bars", True))
                    chain_id = str(row.get("episode_uid") or row.get("episode_id") or "").strip()
                    if chain_id:
                        spy_chains[key][chain_id] = {
                            "outcome": str(row.get("outcome") or "OPEN"),
                            "direction": str(row.get("direction") or row.get("side_sign") or ""),
                        }
                else:
                    group["primary_rows"] += 1
                    complete = bool(str(row.get("complete_bar_ts") or "").strip())
                    state = str(row.get("state") or "").strip()
                    if state:
                        state_counts[key][state] += 1
                        state_records[key].append(
                            (
                                str(row.get("evaluated_at") or row.get("ts") or ""),
                                state,
                            )
                        )
                side = str(row.get("direction") or row.get("side_sign") or "").strip()
            else:
                group["primary_rows"] += 1
                bar = row.get("bar")
                complete = bool(bar.get("complete")) if isinstance(bar, dict) and "complete" in bar else None
                event = str(row.get("event") or "").strip()
                event_counts[key][event or "(missing)"] += 1
                candidate_id = str(row.get("candidate_id") or "").strip()
                if candidate_id:
                    chain = greatness_chains[key].setdefault(
                        candidate_id,
                        {
                            "events": set(),
                            "side": str(row.get("side") or ""),
                            "setup_family": str(row.get("setup_family") or ""),
                        },
                    )
                    chain["events"].add(event)
                side = str(row.get("side") or "").strip()
                family = str(row.get("setup_family") or "").strip()
                if family:
                    family_counts[key][family] += 1
            if side:
                side_counts[key][side] += 1
            if complete is True:
                group["completed_bar_rows"] += 1
            elif complete is False:
                group["incomplete_bar_rows"] += 1

    for key, group in groups.items():
        group["schemas"] = dict(schema_counts[key])
        group["machines"] = dict(machine_counts[key])
        group["engine_versions"] = dict(version_counts[key])
        group["sides"] = dict(side_counts[key])
        group["setup_families"] = dict(family_counts[key])
        group["event_counts"] = dict(event_counts[key])
        if engine == SPY_ENGINE:
            transitions: Counter[str] = Counter()
            durations: Counter[str] = Counter()
            previous_stamp: datetime | None = None
            previous_state = ""
            last_state_at = ""
            for stamp_text, state in state_records[key]:
                try:
                    stamp = datetime.fromisoformat(stamp_text.replace("Z", "+00:00"))
                except ValueError:
                    stamp = None
                if previous_state and state != previous_state:
                    transitions[f"{previous_state}->{state}"] += 1
                if (
                    previous_stamp is not None
                    and stamp is not None
                    and stamp >= previous_stamp
                ):
                    durations[previous_state] += int(
                        (stamp - previous_stamp).total_seconds()
                    )
                if stamp is not None:
                    previous_stamp = stamp
                    last_state_at = stamp_text
                previous_state = state
            group["state_observations"] = dict(state_counts[key])
            group["state_transitions"] = dict(transitions)
            group["state_duration_seconds_observed"] = dict(durations)
            group["last_state"] = previous_state
            group["last_state_at"] = last_state_at
            chains = spy_chains[key]
            group["distinct_chains"] = len(chains)
            group["complete_chains"] = sum(
                1 for chain in chains.values() if chain["outcome"] in _SPY_TERMINAL
            )
            outcomes = Counter(chain["outcome"] for chain in chains.values())
            group["chain_outcomes"] = dict(outcomes)
        else:
            chains = greatness_chains[key]
            group["distinct_chains"] = len(chains)
            group["complete_chains"] = sum(
                1 for chain in chains.values() if chain["events"] & _GREATNESS_TERMINAL
            )
            counts = event_counts[key]
            group["meaningful_interactions"] = sum(
                int(counts.get(event, 0)) for event in _GREATNESS_INTERACTIONS
            )
            group["confirm_fail_rearm_outcomes"] = sum(
                int(counts.get(event, 0)) for event in _GREATNESS_OUTCOMES
            )
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {
        "path": str(path),
        "readable": True,
        "read_error": "",
        "bytes": int(size),
        "sha256": digest.hexdigest(),
        "lines": lines,
        "malformed_lines": malformed,
        "groups": groups,
    }


def _session_metrics(
    engine: str,
    raw_stats: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    if engine != SPY_ENGINE:
        return {
            "event_counts": dict(raw_stats.get("event_counts") or {}),
            "distinct_candidate_chains": int(raw_stats.get("distinct_chains", 0) or 0),
            "complete_candidate_chains": int(raw_stats.get("complete_chains", 0) or 0),
            "meaningful_level_interactions": int(
                raw_stats.get("meaningful_interactions", 0) or 0
            ),
            "confirm_fail_rearm_outcomes": int(
                raw_stats.get("confirm_fail_rearm_outcomes", 0) or 0
            ),
        }
    durations = Counter(
        {
            str(state): int(seconds or 0)
            for state, seconds in (
                raw_stats.get("state_duration_seconds_observed") or {}
            ).items()
        }
    )
    last_state = str(raw_stats.get("last_state") or "")
    last_state_at = str(raw_stats.get("last_state_at") or "")
    last_evaluation = str(coverage.get("last_evaluation_at") or "")
    if last_state and last_state_at and last_evaluation:
        try:
            start = datetime.fromisoformat(last_state_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(last_evaluation.replace("Z", "+00:00"))
            if end >= start:
                durations[last_state] += int((end - start).total_seconds())
        except (TypeError, ValueError):
            pass
    return {
        "state_observations": dict(raw_stats.get("state_observations") or {}),
        "state_transitions": dict(raw_stats.get("state_transitions") or {}),
        "state_duration_seconds": dict(durations),
        "distinct_episode_chains": int(raw_stats.get("distinct_chains", 0) or 0),
        "complete_episode_chains": int(raw_stats.get("complete_chains", 0) or 0),
        "chain_outcomes": dict(raw_stats.get("chain_outcomes") or {}),
    }


def _archive_path(log_path: Path, engine: str, session_date: str, configuration: str) -> Path:
    raw_dir, _ = evidence_directories(log_path, engine)
    return raw_dir / (
        f"{log_path.stem}-through-{_safe(session_date)}-"
        f"{_safe(configuration[:12], 'no-config')}.jsonl"
    )


def _summary_path(
    log_path: Path,
    engine: str,
    session_date: str,
    configuration: str,
) -> Path:
    _, summary_dir = evidence_directories(log_path, engine)
    return summary_dir / (
        f"{_safe(session_date)}-{_safe(configuration[:12], 'no-config')}.json"
    )


def _read_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def apply_retention(log_path: Path | str, engine: str) -> dict[str, int]:
    raw_dir, summary_dir = evidence_directories(log_path, engine)
    return {
        "raw_age_pruned": prune_by_age(
            raw_dir,
            RETENTION_POLICY["raw_max_age_days"],
            pattern="*.jsonl",
            keep_newest=RETENTION_POLICY["raw_keep_newest"],
        ),
        "raw_size_pruned": prune_by_size(
            raw_dir,
            RETENTION_POLICY["raw_max_bytes"],
            pattern="*.jsonl",
            keep_newest=RETENTION_POLICY["raw_keep_newest"],
        ),
        "summary_age_pruned": prune_by_age(
            summary_dir,
            RETENTION_POLICY["summary_max_age_days"],
            pattern="*.json",
            keep_newest=RETENTION_POLICY["summary_keep_newest"],
        ),
        "summary_size_pruned": prune_by_size(
            summary_dir,
            RETENTION_POLICY["summary_max_bytes"],
            pattern="*.json",
            keep_newest=RETENTION_POLICY["summary_keep_newest"],
        ),
    }


def finalize_session(
    *,
    engine: str,
    log_path: Path | str,
    coverage: dict[str, Any],
    finalized_at: datetime,
    reason: str,
    engine_version: str,
    machine: str,
    timezone: str,
    configuration: str,
) -> Path:
    """Rotate raw evidence and atomically publish the finalized coverage.

    The deterministic archive path is the recovery marker.  If a process dies
    after ``os.replace`` but before the summary write, the next call sees the
    archive and resumes without moving or duplicating anything.
    """

    log_path = Path(log_path)
    session_date = str(coverage.get("session_date") or "").strip()
    if not session_date:
        raise ValueError("coverage has no session_date")
    configuration = str(
        coverage.get("config_hash") or configuration or ""
    ).strip()
    archive = _archive_path(log_path, engine, session_date, configuration)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() and log_path.exists() and log_path.stat().st_size:
        os.replace(log_path, archive)

    scan = scan_raw_archive(archive, engine)
    groups = dict(scan.get("groups") or {})
    coverage_key = _group_key(session_date, configuration)
    groups.setdefault(coverage_key, _empty_group(session_date, configuration))
    written: dict[str, Path] = {}
    for key, raw_stats in groups.items():
        group_session = str(raw_stats.get("session_date") or "")
        group_config = str(raw_stats.get("config_hash") or "")
        target = _summary_path(log_path, engine, group_session, group_config)
        existing = _read_dict(target)
        group_coverage = dict(coverage) if key == coverage_key else {}
        summary_id = config_hash(
            {
                "engine": engine,
                "session_date": group_session,
                "config_hash": group_config,
                "machine": machine,
            }
        )
        payload = {
            "schema": SESSION_SUMMARY_SCHEMA,
            "summary_id": summary_id,
            "engine": engine,
            "session_date": group_session,
            "engine_version": engine_version,
            "config_hash": group_config,
            "machine": machine,
            "timezone": timezone,
            "finalized_at": str(
                existing.get("finalized_at")
                or finalized_at.isoformat(timespec="seconds")
            ),
            "finalization_reason": str(existing.get("finalization_reason") or reason),
            "coverage_present": bool(group_coverage),
            "coverage": group_coverage,
            "raw_archive": str(archive),
            "raw_archive_exists": bool(scan.get("readable")),
            "raw_archive_bytes": int(scan.get("bytes", 0) or 0),
            "raw_archive_sha256": str(scan.get("sha256") or ""),
            "raw_archive_malformed_lines": int(scan.get("malformed_lines", 0) or 0),
            "raw_stats": raw_stats,
            "session_metrics": _session_metrics(engine, raw_stats, group_coverage),
            "retention_policy": dict(RETENTION_POLICY),
            "manual_reviewed_chains": int(existing.get("manual_reviewed_chains", 0) or 0),
            "promotion_decision": "NONE",
        }
        if payload != existing:
            atomic_write_json(target, payload)
        written[key] = target

    coverage_summary = written[coverage_key]
    apply_retention(log_path, engine)
    return coverage_summary


def _cached_scan(path: Path, engine: str) -> dict[str, Any]:
    try:
        stat = path.stat()
        key = (engine, str(path), int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        return scan_raw_archive(path, engine)
    cached = _AUDIT_CACHE.get(key)
    if cached is None:
        cached = scan_raw_archive(path, engine)
        if len(_AUDIT_CACHE) > 128:
            _AUDIT_CACHE.clear()
        _AUDIT_CACHE[key] = cached
    return cached


def audit_session_summaries(log_path: Path | str, engine: str) -> dict[str, Any]:
    """Reconcile summaries to retained raw archives and count evidence floors."""

    _, summary_dir = evidence_directories(log_path, engine)
    summaries: list[dict[str, Any]] = []
    try:
        paths = sorted(summary_dir.glob("*.json"))
    except OSError:
        paths = []
    archive_scans: dict[str, dict[str, Any]] = {}
    session_scope_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    totals = Counter()
    side_totals = Counter()
    family_totals = Counter()
    manual_reviewed = 0
    for path in paths:
        payload = _read_dict(path)
        reasons: list[str] = []
        if payload.get("schema") != SESSION_SUMMARY_SCHEMA:
            reasons.append("unexpected summary schema")
        archive_text = str(payload.get("raw_archive") or "")
        archive = Path(archive_text) if archive_text else Path()
        scan = archive_scans.get(archive_text)
        if scan is None:
            scan = _cached_scan(archive, engine) if archive_text else {
                "readable": False,
                "groups": {},
                "malformed_lines": 0,
            }
            archive_scans[archive_text] = scan
        key = _group_key(payload.get("session_date"), payload.get("config_hash"))
        observed = (scan.get("groups") or {}).get(key)
        if not scan.get("readable"):
            reasons.append("raw archive missing or unreadable")
        elif str(scan.get("sha256") or "") != str(payload.get("raw_archive_sha256") or ""):
            reasons.append("raw archive checksum differs from summary")
        if observed != payload.get("raw_stats"):
            reasons.append("summary counters do not reconcile to raw archive")
        if payload.get("session_metrics") != _session_metrics(
            engine,
            observed or payload.get("raw_stats") or {},
            payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {},
        ):
            reasons.append("session metrics do not reconcile to raw counters and coverage")
        if int(scan.get("malformed_lines", 0) or 0):
            reasons.append("raw archive contains malformed lines")
        coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
        if not payload.get("coverage_present") or not coverage:
            reasons.append("coverage counters were unavailable at finalization")
        elif int(coverage.get("errors", 0) or 0):
            reasons.append("coverage recorded errors")
        elif int(coverage.get("evaluations", 0) or 0) <= 0:
            reasons.append("no evaluations")
        usable = int(
            coverage.get("usable_evaluations", coverage.get("bars_consumed", 0)) or 0
        )
        if coverage and usable <= 0:
            reasons.append("no usable completed-bar evaluations")
        raw_stats = observed or payload.get("raw_stats") or {}
        if int(raw_stats.get("valid_rows", 0) or 0) <= 0:
            reasons.append("no raw rows")
        if int(raw_stats.get("completed_bar_rows", 0) or 0) <= 0:
            reasons.append("no completed-bar raw evidence")
        is_eligible = not reasons
        session_scope_results[str(payload.get("session_date") or "")].append(
            {
                "config_hash": payload.get("config_hash") or "",
                "eligible": is_eligible,
                "reasons": reasons,
            }
        )
        totals["recorded_chains"] += int(raw_stats.get("distinct_chains", 0) or 0)
        totals["complete_chains"] += int(raw_stats.get("complete_chains", 0) or 0)
        totals["meaningful_interactions"] += int(
            raw_stats.get("meaningful_interactions", 0) or 0
        )
        totals["confirm_fail_rearm_outcomes"] += int(
            raw_stats.get("confirm_fail_rearm_outcomes", 0) or 0
        )
        side_totals.update(raw_stats.get("sides") or {})
        family_totals.update(raw_stats.get("setup_families") or {})
        manual_reviewed += int(payload.get("manual_reviewed_chains", 0) or 0)
        summaries.append(payload)

    incomplete: list[dict[str, Any]] = []
    eligible = 0
    for session_date, scopes in sorted(session_scope_results.items()):
        session_reasons = [
            f"{scope['config_hash'] or 'no-config'}: {reason}"
            for scope in scopes
            for reason in scope["reasons"]
        ]
        if session_reasons:
            incomplete.append(
                {
                    "session_date": session_date,
                    "scope_count": len(scopes),
                    "reasons": session_reasons,
                }
            )
        else:
            eligible += 1

    if engine == SPY_ENGINE:
        floors = {
            "eligible_sessions": {"count": eligible, "floor": 10},
            "recorded_complete_chains": {
                "count": int(totals["complete_chains"]),
                "floor": 30,
                "note": "Recorded replayable chains; not a substitute for manual review.",
            },
            "manually_reviewed_meaningful_episodes": {
                "count": manual_reviewed,
                "floor": 30,
            },
        }
    else:
        floors = {
            "manually_reviewed_transition_chains": {
                "count": manual_reviewed,
                "floor": 25,
            },
            "meaningful_level_interactions": {
                "count": int(totals["meaningful_interactions"]),
                "floor": 50,
            },
            "confirm_fail_rearm_outcomes": {
                "count": int(totals["confirm_fail_rearm_outcomes"]),
                "floor": 20,
            },
            "recorded_complete_chains": {
                "count": int(totals["complete_chains"]),
                "floor": 25,
                "note": "Recorded chains; manual review remains separately zero until recorded.",
            },
        }
    return {
        "schema": "shadow_session_progress_v1",
        "engine": engine,
        "summary_count": len(summaries),
        "eligible_sessions": eligible,
        "incomplete_sessions": len(incomplete),
        "incomplete_session_details": incomplete,
        "recorded_chains": int(totals["recorded_chains"]),
        "complete_chains": int(totals["complete_chains"]),
        "manual_reviewed_chains": manual_reviewed,
        "sides": dict(side_totals),
        "setup_families": dict(family_totals),
        "section_7_floor_progress": floors,
        "retention_policy": dict(RETENTION_POLICY),
        "promotion_decision": "NONE",
        "affects_promotion": False,
    }


def reset_audit_cache() -> None:
    _AUDIT_CACHE.clear()
