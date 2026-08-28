"""Setup Tracker -> warehouse occurrence adapter (BD-44).

The 1 GB tracker snapshot is deliberately not a source here.  The small
append-only transition ledger says when a setup existed and what state it was
in; the scenario CSV supplies the stop geometry that the ledger intentionally
does not duplicate.  Both are streamed, so the desk never holds a second copy
of the tracker in memory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # package import
    from . import exchange_calendar as xcal
    from .occurrences import OccurrenceReport, record_occurrences
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import exchange_calendar as xcal  # type: ignore
    from occurrences import OccurrenceReport, record_occurrences  # type: ignore

ADAPTER_VERSION = "setup_tracker_ledger_scenarios_v1"
ENTRY_CONTRACT = "next_session_first_completed_m5_close_v1"
TRACKER_EVENT_TYPES = frozenset({"initial", "transition", "reopened", "tombstone"})
STOP_SOURCE_ORDER = {
    "post_earnings_candle": 0,
    "post_earnings_anchor": 1,
    "current_anchor": 2,
    "ema": 3,
    "sma": 4,
}


@dataclass
class AdapterReport:
    status: str = "OK"
    scenario_rows: int = 0
    tracker_events: int = 0
    setups_seen: int = 0
    detections: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    occurrence_report: dict[str, Any] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def canonical_setup_id(family: Any) -> str:
    """The tracker family as a stable warehouse id, for every family."""
    raw = str(family or "").strip().lower()
    if raw == "avwape_to_1stdev":
        raw = "avwape_to_first_dev"
    try:
        from master_avwap_lib.setup_tagging import _FAMILY_TAGS

        return str(_FAMILY_TAGS.get(raw) or raw.upper())
    except ImportError:  # pragma: no cover - packaged layout still has fallback
        return raw.upper()


def _number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def _truth(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _moment(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def load_tracker_events(*, directory: Path | None = None, rows: Iterable[Mapping[str, Any]] | None = None) -> tuple[dict[str, dict], int, int]:
    """Latest state per setup id, plus row and unreadable counts."""
    if rows is None:
        from evidence_ledger import EvidenceLedger
        from setup_tracker_ledger import SCHEMA_SETUP_TRACKER_EVENT, STREAM

        read = EvidenceLedger(
            stream=STREAM,
            schema=SCHEMA_SETUP_TRACKER_EVENT,
            directory=directory,
        ).read(event_types=TRACKER_EVENT_TYPES)
        source_rows: Iterable[Mapping[str, Any]] = read.rows
        unreadable = int(read.unreadable)
    else:
        source_rows = rows
        unreadable = 0

    latest: dict[str, dict] = {}
    count = 0
    for raw in source_rows:
        event = dict(raw or {})
        setup_id = str(event.get("setup_id") or "").strip()
        kind = str(event.get("event_type") or "")
        if not setup_id or kind not in TRACKER_EVENT_TYPES:
            continue
        count += 1
        if kind == "tombstone":
            latest.pop(setup_id, None)
            continue
        previous = latest.get(setup_id)
        if previous is None or str(event.get("event_at") or "") >= str(previous.get("event_at") or ""):
            latest[setup_id] = event
    return latest, count, unreadable


def _scenario_source(path: Path | None, rows: Iterable[Mapping[str, Any]] | None):
    if rows is not None:
        yield from rows
        return
    if path is None:
        from project_paths import MASTER_AVWAP_SETUP_SCENARIOS_FILE

        path = Path(MASTER_AVWAP_SETUP_SCENARIOS_FILE)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _candidate(row: Mapping[str, Any]) -> dict[str, Any] | None:
    entry = _number(row.get("entry_price"))
    risk = _number(row.get("initial_risk_per_share"))
    side = str(row.get("side") or "").upper()
    source = str(row.get("stop_source_type") or "").strip()
    label = str(row.get("stop_reference_label") or "").strip()
    if entry is None or risk is None or risk <= 0 or side not in {"LONG", "SHORT"} or not source:
        return None
    level = entry - risk if side == "LONG" else entry + risk
    failures = int(_number(row.get("close_failure_limit")) or 1)
    return {
        "source_type": source,
        "label": label,
        "level": round(level, 8),
        "close_failure_limit": max(1, failures),
    }


def _scenario_groups(
    *, path: Path | None = None, rows: Iterable[Mapping[str, Any]] | None = None
) -> tuple[dict[str, dict], int]:
    groups: dict[str, dict] = {}
    count = 0
    for raw in _scenario_source(path, rows):
        row = dict(raw or {})
        count += 1
        if _truth(row.get("experimental")) or not _truth(row.get("tradeable", True)):
            continue
        setup_id = str(row.get("setup_id") or "").strip()
        if not setup_id:
            continue
        group = groups.setdefault(
            setup_id,
            {
                "setup_id": setup_id,
                "scan_date": str(row.get("scan_date") or ""),
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "side": str(row.get("side") or "").strip().upper(),
                "priority_bucket": str(row.get("priority_bucket") or ""),
                "setup_family": str(row.get("setup_family") or "").strip().lower(),
                "entry_price": _number(row.get("entry_price")),
                "anchor_date": str(row.get("anchor_date") or ""),
                "candidates": {},
            },
        )
        candidate = _candidate(row)
        if candidate is not None:
            key = (
                candidate["source_type"],
                candidate["label"],
                candidate["level"],
                candidate["close_failure_limit"],
            )
            group["candidates"][key] = candidate
    for group in groups.values():
        group["candidates"] = sorted(
            group["candidates"].values(),
            key=lambda item: (
                STOP_SOURCE_ORDER.get(str(item.get("source_type")), 99),
                str(item.get("label")),
                float(item.get("level") or 0),
            ),
        )
    return groups, count


def detections_from_tracker(
    *,
    scenario_path: Path | None = None,
    scenario_rows: Iterable[Mapping[str, Any]] | None = None,
    event_directory: Path | None = None,
    event_rows: Iterable[Mapping[str, Any]] | None = None,
    report: AdapterReport | None = None,
) -> list[dict]:
    """Build the documented occurrence inputs without opening the tracker JSON."""
    audit = report or AdapterReport()
    states, audit.tracker_events, unreadable = load_tracker_events(directory=event_directory, rows=event_rows)
    if unreadable:
        audit.skipped["UNREADABLE_TRACKER_EVENT"] = unreadable
    groups, audit.scenario_rows = _scenario_groups(path=scenario_path, rows=scenario_rows)
    audit.setups_seen = len(groups)
    theses: dict[tuple[str, str, str, str], list[tuple[date, dict, dict]]] = {}
    for setup_id, group in groups.items():
        state = states.get(setup_id)
        if state is None:
            audit.skip("NO_TRACKER_EVENT")
            continue
        scan_day = _day(group.get("scan_date") or state.get("scan_date"))
        if scan_day is None:
            audit.skip("NO_TRIGGER_SESSION")
            continue
        family = canonical_setup_id(group.get("setup_family") or state.get("state_setup_family"))
        candidates = list(group.get("candidates") or [])
        if not family:
            audit.skip("NO_SETUP_FAMILY")
            continue
        if not candidates:
            audit.skip("NO_STOP_GEOMETRY")
            continue
        side = str(group.get("side") or state.get("state_side") or state.get("side") or "").upper()
        if side not in {"LONG", "SHORT"}:
            audit.skip("NO_SIDE")
            continue
        symbol = str(group.get("symbol") or "")
        anchor = str(group.get("anchor_date") or "")
        # Daily tracker rows for one live thesis are rescans, not independent
        # samples.  Family stays in occurrence identity; this thesis key does
        # not, so simultaneous family variants also share a dependency cluster.
        key = (symbol, side, family, anchor or scan_day.isoformat())
        theses.setdefault(key, []).append((scan_day, group, state))

    detections: list[dict] = []
    for (symbol, side, family, anchor_or_start), members in theses.items():
        members.sort(key=lambda item: item[0])
        first_day, origin, first_state = members[0]
        _last_day, _latest, latest_state = members[-1]
        session = xcal.trading_session(first_day)
        if session is None:
            audit.skip("NO_TRIGGER_SESSION")
            continue
        candidates = list(origin.get("candidates") or [])
        episode = "|".join([symbol, side, anchor_or_start])
        setup_ids = [str(group.get("setup_id") or "") for _day, group, _state in members]
        tags = {
            "schema": "tracker_stop_geometry_v1",
            "tracker_setup_id": setup_ids[0],
            "tracker_setup_ids": setup_ids,
            "latest_tracker_setup_id": setup_ids[-1],
            "rescan_count": len(setup_ids),
            "priority_bucket": origin.get("priority_bucket"),
            "anchor_date": origin.get("anchor_date"),
            "entry_contract": ENTRY_CONTRACT,
            # Point-in-time: stop geometry comes from the FIRST scan, never a
            # later rescan whose moving AVWAP/SMA level was not known at entry.
            "stop_candidates": candidates,
        }
        detections.append(
            {
                "symbol": symbol,
                "canonical_setup_id": family,
                "side": side,
                "structural_timeframe": "D1",
                "trigger_timeframe": "D1",
                "episode_start": episode,
                "status": str(latest_state.get("state_setup_status") or ""),
                "trigger_at": session.rth_close_at,
                "entry_price_ref": origin.get("entry_price"),
                "stop_price_ref": candidates[0].get("level"),
                "detector_version": ADAPTER_VERSION,
                "run_id": str(latest_state.get("run_id") or first_state.get("run_id") or ""),
                "tags": json.dumps(tags, sort_keys=True, separators=(",", ":")),
                "event_at": session.rth_close_at,
                "observed_at": _moment(first_state.get("event_at")) or session.rth_close_at,
            }
        )
    audit.detections = len(detections)
    return detections


def record_tracker_occurrences(
    store,
    *,
    scenario_path: Path | None = None,
    event_directory: Path | None = None,
    run_id: str = "",
    now: datetime | None = None,
) -> AdapterReport:
    report = AdapterReport()
    if store is None:
        report.status = "DISABLED"
        return report
    try:
        detections = detections_from_tracker(
            scenario_path=scenario_path,
            event_directory=event_directory,
            report=report,
        )
    except OSError as exc:
        report.status = "MISSING_SOURCE"
        report.skip(type(exc).__name__)
        return report
    occurrence_report: OccurrenceReport = record_occurrences(
        store, detections, run_id=run_id, job_id="tracker_occurrence_adapter", now=now
    )
    report.occurrence_report = vars(occurrence_report)
    if occurrence_report.status not in {"OK", "NOTHING_TO_RECORD"}:
        report.status = occurrence_report.status
    elif not detections:
        report.status = "NOTHING_TO_RECORD"
    return report


__all__ = [
    "ADAPTER_VERSION",
    "ENTRY_CONTRACT",
    "AdapterReport",
    "canonical_setup_id",
    "detections_from_tracker",
    "load_tracker_events",
    "record_tracker_occurrences",
]
