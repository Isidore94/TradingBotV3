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
from datetime import datetime, timezone
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
        # Timestamp anomaly accounting (never hidden, never fatal).  Legacy rows
        # written before timezone normalization carry naive stamps; a naive
        # stamp compared against an aware one used to raise TypeError and kill
        # the whole scan.  Normalization is only ever done from RECORDED
        # evidence or explicit configuration - never the host's current clock.
        "timestamps_legacy_naive": 0,
        "timestamps_naive_normalized": 0,
        "timestamps_unresolved": 0,
        "timestamps_malformed": 0,
        "timestamps_out_of_order": 0,
        "naive_timezone_source": "",
    }


def _configured_market_zoneinfo():
    """The explicitly configured market-local timezone, or None.

    Deliberately uses only the CONFIGURED half of market_session's resolution
    chain (env var / local settings).  The system-timezone fallback is exactly
    the "silently assume the host's current timezone" behaviour that legacy
    naive stamps must never inherit: the host's zone today says nothing about
    the writer's clock on the day the row was written.
    """
    try:
        from market_session import _coerce_zoneinfo, _resolve_configured_timezone_name

        return _coerce_zoneinfo(_resolve_configured_timezone_name())
    except Exception:
        return None


def _parse_stamp(text: str) -> tuple[datetime | None, str]:
    """Parse one recorded timestamp; total over every input shape.

    Returns ``(stamp, kind)`` where kind is ``aware``, ``naive`` or
    ``malformed``.  Never raises.
    """
    value = str(text or "").strip()
    if not value:
        return None, "malformed"
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None, "malformed"
    if stamp.tzinfo is None:
        return stamp, "naive"
    return stamp, "aware"


def _resolve_naive_stamps(
    group: dict[str, Any],
    parsed: list[list[Any]],
) -> None:
    """Attach a timezone to naive stamps from trustworthy evidence, in place.

    Resolution order (first hit wins, and the source is recorded on the group):

    1. the row's own recorded ``timezone`` name, when zoneinfo can resolve it;
    2. the UNANIMOUS utc offset of the aware rows in the same session group -
       recorded evidence from the same writer on the same day;
    3. an explicitly configured market-local timezone (env / local settings).

    A naive stamp with none of the above stays unresolved: the row itself is
    kept and counted, but its duration boundary is unknown and is never used.
    ``parsed`` entries are ``[stamp, kind, state, stamp_text, row_tz_name]``.
    """
    sibling_offsets = {
        entry[0].utcoffset() for entry in parsed if entry[1] == "aware"
    }
    unanimous_offset = (
        next(iter(sibling_offsets)) if len(sibling_offsets) == 1 else None
    )
    configured = None
    configured_checked = False
    for entry in parsed:
        if entry[1] != "naive":
            continue
        group["timestamps_legacy_naive"] += 1
        stamp, row_tz_name = entry[0], entry[4]
        row_zone = None
        if row_tz_name:
            try:
                import zoneinfo

                row_zone = zoneinfo.ZoneInfo(str(row_tz_name))
            except Exception:
                row_zone = None
        if row_zone is not None:
            entry[0] = stamp.replace(tzinfo=row_zone)
            entry[1] = "aware"
            group["timestamps_naive_normalized"] += 1
            group["naive_timezone_source"] = f"row_timezone:{row_tz_name}"
            continue
        if unanimous_offset is not None:
            zone = timezone(unanimous_offset)
            entry[0] = stamp.replace(tzinfo=zone)
            entry[1] = "aware"
            group["timestamps_naive_normalized"] += 1
            # UTC-offset formatting ("-07:00"), not timedelta repr.
            group["naive_timezone_source"] = (
                f"sibling_offset:{stamp.replace(tzinfo=zone).strftime('%z')[:3]}:"
                f"{stamp.replace(tzinfo=zone).strftime('%z')[3:]}"
            )
            continue
        if not configured_checked:
            configured = _configured_market_zoneinfo()
            configured_checked = True
        if configured is not None:
            entry[0] = stamp.replace(tzinfo=configured)
            entry[1] = "aware"
            group["timestamps_naive_normalized"] += 1
            group["naive_timezone_source"] = f"configured:{configured.key}"
            continue
        # No trustworthy timezone: keep the row, refuse to invent a clock.
        entry[0] = None
        entry[1] = "unresolved"
        group["timestamps_unresolved"] += 1


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
                                str(row.get("timezone") or "").strip(),
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
            parsed: list[list[Any]] = []
            for stamp_text, state, row_tz in state_records[key]:
                stamp, kind = _parse_stamp(stamp_text)
                if kind == "malformed":
                    group["timestamps_malformed"] += 1
                parsed.append([stamp, kind, state, stamp_text, row_tz])
            _resolve_naive_stamps(group, parsed)
            previous_stamp: datetime | None = None
            previous_state = ""
            last_state_at = ""
            last_state_at_recorded = ""
            for stamp, kind, state, stamp_text, _row_tz in parsed:
                if previous_state and state != previous_state:
                    transitions[f"{previous_state}->{state}"] += 1
                usable = kind == "aware" and stamp is not None
                if previous_stamp is not None and usable:
                    try:
                        ordered = stamp >= previous_stamp
                    except TypeError:  # belt-and-braces: never let a scan raise
                        ordered = False
                        group["timestamps_unresolved"] += 1
                        usable = False
                    else:
                        if not ordered:
                            # Time ran backwards between two trusted stamps.
                            # Count it, never compute a duration across it, and
                            # restart the chain AT this stamp - the anomaly is
                            # the boundary, not the row.
                            group["timestamps_out_of_order"] += 1
                    if ordered:
                        durations[previous_state] += int(
                            (stamp - previous_stamp).total_seconds()
                        )
                if usable:
                    previous_stamp = stamp
                    # The AWARE instant feeds durations and reconciliation
                    # downstream (_session_metrics compares it against aware
                    # coverage stamps); the original text is provenance.
                    last_state_at = stamp.isoformat(timespec="seconds")
                    last_state_at_recorded = stamp_text
                else:
                    # An unresolvable or malformed stamp is an unknown clock
                    # boundary: elapsed time across it would be invented, so
                    # the duration chain restarts at the next trusted stamp.
                    previous_stamp = None
                previous_state = state
            group["state_observations"] = dict(state_counts[key])
            group["state_transitions"] = dict(transitions)
            group["state_duration_seconds_observed"] = dict(durations)
            group["last_state"] = previous_state
            group["last_state_at"] = last_state_at
            group["last_state_at_recorded"] = last_state_at_recorded
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
        # File-level anomaly visibility (per-group detail lives on each group).
        "timestamps_legacy_naive": sum(
            int(g.get("timestamps_legacy_naive", 0)) for g in groups.values()
        ),
        "timestamps_unresolved": sum(
            int(g.get("timestamps_unresolved", 0)) for g in groups.values()
        ),
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
    active_has_rows = log_path.exists() and log_path.stat().st_size
    if active_has_rows:
        active_scan = scan_raw_archive(log_path, engine)
        for active_group in (active_scan.get("groups") or {}).values():
            active_summary = _summary_path(
                log_path,
                engine,
                str(active_group.get("session_date") or ""),
                str(active_group.get("config_hash") or ""),
            )
            if active_summary.exists():
                raise RuntimeError(
                    "active shadow log contains an already-finalized "
                    "session/configuration scope; refusing to overwrite its "
                    f"replay evidence: {active_summary}"
                )
    if not archive.exists() and active_has_rows:
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
        # A stamp with no trustworthy timezone means part of this session's
        # clock is unknown.  Rows normalized from RECORDED evidence
        # (timestamps_naive_normalized, with naive_timezone_source stating the
        # source) do not block eligibility; unresolved ones do - an eligible
        # session must not stand on invented time.
        if int(raw_stats.get("timestamps_unresolved", 0) or 0) > 0:
            reasons.append("unresolvable raw timestamps")
        if int(raw_stats.get("timestamps_malformed", 0) or 0) > 0:
            reasons.append("malformed raw timestamps")
        # Trusted stamps running backwards is an unexplained ordering anomaly;
        # an eligible session cannot stand on a clock that went backwards.
        if int(raw_stats.get("timestamps_out_of_order", 0) or 0) > 0:
            reasons.append("out-of-order raw timestamps")
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
