"""Read-only acceptance audit for Regime Infrastructure Phase 1.

The audit never reconstructs frozen evidence and never writes a policy. It
reports whether the live process actually appended the artifacts required for
the session, including explicit missed/data-gap markers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from diagnostics.artifact_io import read_jsonl
from market_session import get_market_session_window, normalize_market_local_datetime
from technical_integrity import (
    CAPTURE_MODE_BACKFILL,
    COLLECTION_CODE_VERSION,
    FOLLOWUP_HORIZONS_MINUTES,
    row_capture_mode,
    technical_integrity_events_path,
)
from vold_recorder import vold_ledger_path


AUDIT_SCHEMA = "regime_collection_audit_v1"
COLLECTION_EVENT_TYPES = {
    "post_resolution_tracking_started",
    "post_resolution_followup",
    "frozen_intraday_snapshot",
    "missed_snapshot",
    "opening_range_baseline",
    "missed_opening_range_baseline",
}
VOLD_EVENT_TYPES = {"contract_verified", "breadth_bar", "data_gap", "recorder_unavailable"}
REQUIRED_PROVENANCE_FIELDS = ("code_version", "as_of", "written_at")


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def _missing_provenance(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for row in rows:
        absent = [field for field in REQUIRED_PROVENANCE_FIELDS if not str(row.get(field) or "")]
        if absent:
            missing.append(
                {
                    "event_type": str(row.get("event_type") or ""),
                    "event_id": str(row.get("event_id") or row.get("bar_end") or ""),
                    "missing": absent,
                }
            )
    return missing


def _expected_completed_bars(now: datetime) -> int:
    session = get_market_session_window(now)
    if now <= session.open_local:
        return 0
    complete_through = min(now, session.close_local)
    return max(0, int((complete_through - session.open_local).total_seconds() // 300))


def audit_regime_collection(
    *,
    session_date: str | date,
    technical_events_path: Path | None = None,
    breadth_events_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    moment = normalize_market_local_datetime(now)
    session_text = (
        session_date.isoformat() if isinstance(session_date, date) else str(session_date)
    )
    technical_path = Path(technical_events_path or technical_integrity_events_path())
    breadth_path = Path(breadth_events_path or vold_ledger_path())
    technical = [
        row
        for row in read_jsonl(technical_path)
        if str(row.get("session_date") or "") == session_text
    ]
    breadth = [
        row
        for row in read_jsonl(breadth_path)
        if str(row.get("session_date") or "") == session_text
    ]

    resolutions = [
        row
        for row in technical
        if row.get("event_type") == "level_resolved"
        and row.get("followup_tracking_version") == COLLECTION_CODE_VERSION
    ]
    starts = [
        row for row in technical if row.get("event_type") == "post_resolution_tracking_started"
    ]
    followups = [
        row for row in technical if row.get("event_type") == "post_resolution_followup"
    ]
    start_by_source = {
        str(row.get("source_resolution_id") or ""): row for row in starts
    }
    backfilled_followups = sum(
        row_capture_mode(row) == CAPTURE_MODE_BACKFILL for row in followups
    )
    horizons_by_source: dict[str, set[int]] = {}
    for row in followups:
        source = str(row.get("source_resolution_id") or "")
        try:
            horizon = int(row.get("horizon_minutes"))
        except (TypeError, ValueError):
            continue
        horizons_by_source.setdefault(source, set()).add(horizon)

    session_window = get_market_session_window(
        datetime.fromisoformat(f"{session_text}T12:00:00").replace(
            tzinfo=moment.tzinfo
        )
    )
    current_market_date = get_market_session_window(moment).market_date
    audited_market_date = date.fromisoformat(session_text)
    after_close = (
        audited_market_date < current_market_date
        or (
            audited_market_date == current_market_date
            and moment >= session_window.close_local
        )
    )
    missing_starts = sorted(
        str(row.get("event_id") or "")
        for row in resolutions
        if str(row.get("event_id") or "") not in start_by_source
    )
    incomplete_followups = {
        source: sorted(set(FOLLOWUP_HORIZONS_MINUTES) - horizons)
        for source, horizons in sorted(horizons_by_source.items())
        if horizons != set(FOLLOWUP_HORIZONS_MINUTES)
    }
    for source in start_by_source:
        if source not in horizons_by_source:
            incomplete_followups[source] = list(FOLLOWUP_HORIZONS_MINUTES)

    snapshots = [
        row
        for row in technical
        if row.get("event_type") in {"frozen_intraday_snapshot", "missed_snapshot"}
    ]
    snapshot_by_label = {
        str(row.get("target_market_time") or ""): row for row in snapshots
    }
    opening_rows = [
        row
        for row in technical
        if row.get("event_type")
        in {"opening_range_baseline", "missed_opening_range_baseline"}
    ]
    required_snapshot_labels: list[str] = []
    for label, target in (
        ("10:30", session_window.open_local + timedelta(minutes=60)),
        ("12:00", session_window.open_local + timedelta(minutes=150)),
    ):
        if moment >= target + timedelta(minutes=5):
            required_snapshot_labels.append(label)
    missing_snapshots = [
        label for label in required_snapshot_labels if label not in snapshot_by_label
    ]
    opening_due = moment >= session_window.open_local + timedelta(minutes=65)

    breadth_bars = [row for row in breadth if row.get("event_type") == "breadth_bar"]
    breadth_gaps = [row for row in breadth if row.get("event_type") == "data_gap"]
    unavailable = [
        row for row in breadth if row.get("event_type") == "recorder_unavailable"
    ]
    contract_rows = [
        row for row in breadth if row.get("event_type") == "contract_verified"
    ]
    actual_contract = (
        dict(contract_rows[-1].get("contract") or {}) if contract_rows else {}
    )
    if audited_market_date < current_market_date:
        expected_breadth_bars = int(
            (session_window.close_local - session_window.open_local).total_seconds()
            // 300
        )
    elif audited_market_date == current_market_date:
        expected_breadth_bars = _expected_completed_bars(moment)
    else:
        expected_breadth_bars = 0
    explicit_missing_bars = sum(
        max(0, int(row.get("missing_bar_count") or 0)) for row in breadth_gaps
    )

    collection_rows = [
        row for row in technical if str(row.get("event_type") or "") in COLLECTION_EVENT_TYPES
    ]
    breadth_collection_rows = [
        row for row in breadth if str(row.get("event_type") or "") in VOLD_EVENT_TYPES
    ]
    provenance_missing = _missing_provenance(
        [*collection_rows, *breadth_collection_rows]
    )
    duplicate_followup_ids = _duplicates(
        str(row.get("event_id") or "") for row in followups
    )
    duplicate_breadth_bars = _duplicates(
        str(row.get("bar_end") or "") for row in breadth_bars
    )

    blockers: list[str] = []
    if missing_starts:
        blockers.append(f"{len(missing_starts)} new resolutions have no follow-up start")
    if after_close and incomplete_followups:
        blockers.append(f"{len(incomplete_followups)} follow-up chains are incomplete after close")
    if missing_snapshots:
        blockers.append(f"missing frozen/missed markers for {', '.join(missing_snapshots)}")
    if opening_due and not opening_rows:
        blockers.append("missing 10:30 opening-range baseline or missed marker")
    if expected_breadth_bars and not breadth_bars:
        blockers.append("breadth ledger has no completed-M5 rows")
    elif len(breadth_bars) + explicit_missing_bars < expected_breadth_bars:
        blockers.append(
            "breadth ledger is partial without enough explicit data-gap coverage"
        )
    if breadth_bars and not contract_rows:
        blockers.append("breadth bars have no contract-verification event")
    if unavailable:
        blockers.append("breadth recorder reported unavailable")
    if duplicate_followup_ids:
        blockers.append(f"{len(duplicate_followup_ids)} duplicate follow-up event IDs")
    if duplicate_breadth_bars:
        blockers.append(f"{len(duplicate_breadth_bars)} duplicate breadth bar ends")
    if provenance_missing:
        blockers.append(f"{len(provenance_missing)} collection events lack provenance fields")

    instrumented_sessions = {
        str(row.get("session_date") or "")
        for row in read_jsonl(technical_path)
        if row.get("event_type") in {"frozen_intraday_snapshot", "missed_snapshot"}
    }
    return {
        "schema": AUDIT_SCHEMA,
        "generated_at": moment.isoformat(timespec="seconds"),
        "session_date": session_text,
        "status": "HEALTHY" if not blockers else "UNHEALTHY",
        "promotion_status": "EXPLORATORY / NON-PROMOTABLE",
        "promotion_floor": {
            "instrumented_sessions": len(instrumented_sessions),
            "minimum_sessions": 40,
            "preferred_sessions": 60,
            "eligible": len(instrumented_sessions) >= 40,
            "note": "Session count is necessary, not sufficient; predictive lift still needs point-in-time validation.",
        },
        "technical_followups": {
            "new_resolution_count": len(resolutions),
            "tracking_start_count": len(starts),
            "followup_event_count": len(followups),
            "horizon_counts": {
                str(horizon): sum(
                    int(row.get("horizon_minutes") or 0) == horizon for row in followups
                )
                for horizon in FOLLOWUP_HORIZONS_MINUTES
            },
            "truncated_count": sum(bool(row.get("truncated")) for row in followups),
            "data_gap_count": sum(bool(row.get("data_gap")) for row in followups),
            # A HEALTHY-with-backfill session is not the same evidence as a
            # fully-live one. What the promotion study is willing to count
            # toward its 40-session floor is declared in the study, not here;
            # the audit's job is only to keep the two distinguishable.
            "backfilled_count": backfilled_followups,
            "live_count": len(followups) - backfilled_followups,
            "missing_tracking_starts": missing_starts,
            "incomplete_chains": incomplete_followups,
            "duplicate_event_ids": duplicate_followup_ids,
        },
        "frozen_snapshots": {
            "required_labels": required_snapshot_labels,
            "observed": {
                label: str(row.get("event_type") or "")
                for label, row in sorted(snapshot_by_label.items())
            },
            "missing_labels": missing_snapshots,
            "opening_range_event": (
                str(opening_rows[-1].get("event_type") or "") if opening_rows else ""
            ),
            "opening_range_data_gap": (
                bool(opening_rows[-1].get("data_gap")) if opening_rows else None
            ),
        },
        "breadth_recorder": {
            "target_metric": "$VOLD",
            "actual_contract": actual_contract,
            "completed_bar_count": len(breadth_bars),
            "expected_completed_bar_count": expected_breadth_bars,
            "explicit_missing_bar_count": explicit_missing_bars,
            "backfilled_bar_count": sum(
                row_capture_mode(row) == CAPTURE_MODE_BACKFILL for row in breadth_bars
            ),
            "data_gap_count": len(breadth_gaps),
            "unavailable_count": len(unavailable),
            "duplicate_bar_ends": duplicate_breadth_bars,
            "semantic_status": (
                "EXACT_VOLD"
                if actual_contract.get("is_exact_vold")
                else f"PROXY:{actual_contract.get('proxy_kind') or 'UNKNOWN'}"
            ),
        },
        "provenance_missing": provenance_missing,
        "blockers": blockers,
        "sources": {
            "technical_events": str(technical_path),
            "breadth_events": str(breadth_path),
        },
    }


def format_audit(report: Mapping[str, Any]) -> str:
    followups = report["technical_followups"]
    snapshots = report["frozen_snapshots"]
    breadth = report["breadth_recorder"]
    floor = report["promotion_floor"]
    lines = [
        f"Regime collection {report['session_date']}: {report['status']}",
        f"{report['promotion_status']} - {floor['instrumented_sessions']}/{floor['minimum_sessions']} session floor",
        (
            "Follow-ups: "
            f"{followups['tracking_start_count']} started, "
            f"{followups['followup_event_count']} windows, "
            f"horizons={followups['horizon_counts']}, "
            f"truncated={followups['truncated_count']}, gaps={followups['data_gap_count']}, "
            f"live={followups['live_count']}, backfilled={followups['backfilled_count']}"
        ),
        (
            "Snapshots: "
            f"observed={snapshots['observed']}, missing={snapshots['missing_labels']}, "
            f"opening={snapshots['opening_range_event'] or 'missing'}"
        ),
        (
            "Breadth: "
            f"{breadth['semantic_status']}, bars={breadth['completed_bar_count']}/"
            f"{breadth['expected_completed_bar_count']}, gaps={breadth['data_gap_count']}, "
            f"backfilled={breadth['backfilled_bar_count']}"
        ),
    ]
    if report["blockers"]:
        lines.append("Blockers:")
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", help="NYSE session date (YYYY-MM-DD)")
    parser.add_argument("--technical-events", type=Path)
    parser.add_argument("--breadth-events", type=Path)
    parser.add_argument("--json", action="store_true", help="Print full JSON")
    args = parser.parse_args()
    moment = normalize_market_local_datetime()
    session = args.session or get_market_session_window(moment).market_date.isoformat()
    report = audit_regime_collection(
        session_date=session,
        technical_events_path=args.technical_events,
        breadth_events_path=args.breadth_events,
        now=moment,
    )
    print(json.dumps(report, indent=2) if args.json else format_audit(report))
    return 0 if report["status"] == "HEALTHY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
