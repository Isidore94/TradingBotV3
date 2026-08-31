"""Streaming integrity scan of the RAW shadow evidence (plan.md sec 12 item 2).

Why this module exists
----------------------
``operations_audit`` graded the two shadow engines by reading the sidecars the
writers maintain about themselves - ``spy_state_shadow_status.json`` and
``greatness_candidates.json``. Nothing ever opened ``spy_state_shadow.jsonl`` or
``greatness_shadow.jsonl``. That is a self-report: the writing process is the
only witness to its own output, so a truncated last line after a crash, a
half-written row, or a schema-drifted record all pass GREEN BY CONSTRUCTION.

The rest of the repo is deliberately forgiving with these files, and correctly
so - :func:`diagnostics.artifact_io.read_jsonl` skips bad lines,
:func:`diagnostics.run_manifest.load_recent_manifests` continues past a
``JSONDecodeError`` and :meth:`job_ledger.JobLedger._replay` continues past a
malformed line, because one damaged row must never cost a trader a session. The
cost of that tolerance is that damage is *invisible*: a corrupt ledger reports
as "No jobs recorded". This module is the counterpart that counts what those
readers skip.

Design constraints
------------------
- **Streaming, never slurped.** ``greatness_shadow.jsonl`` is already ~14.5 MB
  with no retention policy, and the Health page refreshes on a timer. The file
  is iterated line by line in binary; nothing accumulates except bounded
  counters and one compact "latest record" summary.
- **A truncated FINAL line is a distinct finding.** An unterminated,
  unparseable last line is the crash/kill signature (the process died between
  ``write`` and the newline). A malformed line in the *interior* means
  something else entirely - interleaved appends from two processes, or a disk
  error - so the two are counted separately and never merged.
- **Multiple row schemas in one file are normal.** SPY episode rows
  (``spy_episode_shadow_v1``) live in the same log as state rows
  (``spy_state_shadow_v4``, with v2/v3 rows retained), so the validator accepts
  a *set* of schemas per file and reports the mix rather than flagging it.
- **Missing data is uncertainty, never silent confirmation** (plan.md sec 5):
  an absent log is UNKNOWN, an unreadable or damaged one is UNHEALTHY, and
  neither is ever HEALTHY.

Strictly observational: this module reads. It repairs nothing, prunes nothing,
promotes nothing, and no champion decision, detector or scoring behavior
depends on it.
"""

from __future__ import annotations

import copy
import json
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from diagnostics.shadow_session_rollup import audit_session_summaries

__all__ = [
    "SHADOW_LOG_AUDIT_SCHEMA",
    "GREATNESS_PROFILE",
    "SPY_PROFILE",
    "ShadowLogProfile",
    "audit_shadow_log",
    "audit_shadow_session_progress",
    "parse_timestamp",
    "scan_shadow_log",
]

SHADOW_LOG_AUDIT_SCHEMA = "shadow_log_audit_v1"

#: Statuses, mirrored from ``operations_audit`` so this module stays importable
#: on its own (the audit is the consumer, not the other way round).
STATUS_HEALTHY = "healthy"
STATUS_UNKNOWN = "unknown"
STATUS_DEGRADED = "degraded"
STATUS_UNHEALTHY = "unhealthy"
_STATUS_ORDER = {STATUS_HEALTHY: 0, STATUS_UNKNOWN: 1, STATUS_DEGRADED: 2, STATUS_UNHEALTHY: 3}

#: A record stamped further ahead than this is a clock/timezone fault, not a
#: rounding artifact. Writers stamp to the second, and the audit's ``now`` and
#: the writers' clock are the same machine's clock in the normal case.
FUTURE_TOLERANCE_SECONDS = 120.0
#: Appends are ordered by evaluation time, so any backwards step is a finding.
#: One second of slack absorbs `isoformat(timespec="seconds")` truncation
#: against a sub-second comparison; it does not absorb a real reordering.
ORDER_TOLERANCE_SECONDS = 1.0

#: Bounded reporting: distinct values per dimension and worked examples kept.
#: A damaged log could otherwise carry unbounded distinct "schema" strings and
#: turn the audit into the memory problem it exists to detect.
MAX_DISTINCT_VALUES = 64
MAX_EXAMPLES = 5
_OTHER = "(other)"
_MISSING = "(missing)"

#: Keys copied onto the "latest valid record" summary. The whole row is never
#: retained: a Greatness row carries a provenance block and an event list, and
#: this summary is rendered in a UI table.
_LATEST_RECORD_KEYS = (
    "schema",
    "ts",
    "evaluated_at",
    "session_date",
    "machine",
    "timezone",
    "engine_version",
    "config_hash",
    "state",
    "symbol",
    "side",
    "event",
    "step",
    "stage",
    "outcome",
    "candidate_id",
    "episode_id",
    "observation_id",
    "bar_ts",
    "complete_bar_ts",
    "usable",
    "stale",
    "stale_reason",
)


# ---------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------
def parse_timestamp(value: Any, local_tz=None) -> datetime | None:
    """ISO-8601 text -> aware datetime in ``local_tz``; ``None`` when unusable.

    Shadow logs legitimately mix naive and aware stamps (the SPY log's first
    rows predate the timezone stamp), so a naive value is read as local wall
    clock rather than discarded - otherwise the ordering check would report
    every pre-timezone row as an unparseable timestamp.
    """
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if local_tz is None:
        return parsed
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_tz)
    return parsed.astimezone(local_tz)


def _worst(statuses) -> str:
    known = [str(item) for item in statuses if str(item) in _STATUS_ORDER]
    if not known:
        return STATUS_UNKNOWN
    return max(known, key=lambda item: _STATUS_ORDER[item])


def _bump(counter: Counter, key: str) -> None:
    """Count ``key``, collapsing anything past the cap into ``(other)``."""
    if key not in counter and len(counter) >= MAX_DISTINCT_VALUES:
        counter[_OTHER] += 1
        return
    counter[key] += 1


# ---------------------------------------------------------------------------
# per-engine row semantics
# ---------------------------------------------------------------------------
def _spy_completed_bar(row: dict) -> bool | None:
    """Completed-bar evidence for a SPY shadow row (``None`` = not stated).

    v4 rows carry ``complete_bar_ts``/``incomplete_bar`` explicitly; retained v2
    rows only carry the merged ``stale`` flag; episode rows only ever advance on
    bars the engine accepted and say so with ``derived_from_completed_bars``.
    """
    if row.get("derived_from_completed_bars") is not None:
        return bool(row.get("derived_from_completed_bars"))
    if "complete_bar_ts" in row:
        return bool(str(row.get("complete_bar_ts") or "").strip())
    if "incomplete_bar" in row:
        return not bool(row.get("incomplete_bar"))
    if "stale" in row:
        return not bool(row.get("stale"))
    return None


def _greatness_completed_bar(row: dict) -> bool | None:
    bar = row.get("bar")
    if isinstance(bar, dict) and "complete" in bar:
        return bool(bar.get("complete"))
    return None


@dataclass(frozen=True)
class ShadowLogProfile:
    """What one shadow log is allowed to contain, and how to read a row."""

    name: str
    label: str
    #: Every row schema this file may legitimately hold, INCLUDING retained
    #: older versions and the episode rows that share the file.
    accepted_schemas: frozenset[str]
    #: The engine's evaluation rows (what a sidecar's ``rows_written`` counts).
    primary_schemas: frozenset[str]
    #: Rows with their own identity and cadence, counted separately.
    episode_schemas: frozenset[str] = frozenset()
    #: Ordering is judged on evaluation time; the first present field wins.
    order_fields: tuple[str, ...] = ("evaluated_at", "ts")
    completed_bar_evidence: Callable[[dict], bool | None] = field(
        default=lambda row: None, repr=False, compare=False
    )

    def kind(self, schema: str) -> str:
        if schema in self.episode_schemas:
            return "episode"
        if schema in self.primary_schemas:
            return "primary"
        return "other"


def _spy_schemas() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Schema sets from the writer itself, with literal fallbacks.

    Imported from :mod:`market_state_bridge` so the validator can never drift
    away from the writer it audits; the literals only matter if that import is
    unavailable (a partially installed tree), and a test pins them equal.
    """
    accepted = {"spy_state_shadow_v2", "spy_state_shadow_v3", "spy_state_shadow_v4"}
    episodes = {"spy_episode_shadow_v1"}
    try:  # pragma: no cover - exercised implicitly by the equality test
        import market_state_bridge

        accepted = set(market_state_bridge.COMPATIBLE_SHADOW_SCHEMAS)
        episodes = {market_state_bridge.EPISODE_SCHEMA}
    except Exception:
        pass
    primary = frozenset(accepted - episodes)
    return frozenset(accepted | episodes), primary, frozenset(episodes)


def _greatness_schemas() -> tuple[frozenset[str], frozenset[str]]:
    accepted = {"greatness_shadow_v2", "greatness_shadow_v3", "greatness_shadow_v4"}
    try:  # pragma: no cover - exercised implicitly by the equality test
        import greatness_shadow

        accepted = set(greatness_shadow.COMPATIBLE_SHADOW_SCHEMAS)
    except Exception:
        pass
    return frozenset(accepted), frozenset(accepted)


_SPY_ACCEPTED, _SPY_PRIMARY, _SPY_EPISODE = _spy_schemas()
_GREAT_ACCEPTED, _GREAT_PRIMARY = _greatness_schemas()

SPY_PROFILE = ShadowLogProfile(
    name="spy_state_shadow",
    label="SPY state shadow log",
    accepted_schemas=_SPY_ACCEPTED,
    primary_schemas=_SPY_PRIMARY,
    episode_schemas=_SPY_EPISODE,
    order_fields=("evaluated_at", "ts"),
    completed_bar_evidence=_spy_completed_bar,
)

GREATNESS_PROFILE = ShadowLogProfile(
    name="greatness_shadow",
    label="Greatness shadow log",
    accepted_schemas=_GREAT_ACCEPTED,
    primary_schemas=_GREAT_PRIMARY,
    # Greatness rows carry the bar timestamp in `ts` and interleave candidates,
    # so only `evaluated_at` is an append-order signal.
    order_fields=("evaluated_at",),
    completed_bar_evidence=_greatness_completed_bar,
)


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------
def _latest_summary(row: dict, line_number: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"line": line_number}
    for key in _LATEST_RECORD_KEYS:
        if key in row:
            value = row[key]
            if isinstance(value, (dict, list)):
                continue
            summary[key] = value
    return summary


#: (profile, path) -> ((st_mtime_ns, st_size, local_tz, market_date,
#: reconcile_session_date), scan result). The logs are append-only, so a stamp
#: that has not moved cannot describe different rows; both stamps are needed
#: because an append inside one filesystem timestamp tick still moves the byte
#: count (same template as review_events.load_review_events, and the same
#: (st_mtime_ns, st_size) key shadow_session_rollup._cached_scan already uses
#: for the archived sessions). ``now`` is deliberately NOT part of the key:
#: its only reach into the scan is the future-timestamp tolerance, so a
#: future-stamped row in an unchanged file stays counted until the file next
#: moves - a clock defect frozen for a few minutes, in exchange for not
#: re-streaming a multi-megabyte log every 15 s Health pass.
_scan_cache: dict[tuple[str, str], tuple[tuple, dict[str, Any]]] = {}
_scan_cache_lock = threading.Lock()


def scan_shadow_log(
    path: Path | str,
    profile: ShadowLogProfile,
    *,
    now: datetime,
    local_tz=None,
    market_date: str = "",
    reconcile_session_date: str = "",
) -> dict[str, Any]:
    """Stream ``path`` once and report what is actually in it.

    Never raises on a damaged file: unreadable bytes, undecodable text and
    unparseable JSON are all *results*, because "the audit crashed" is the one
    outcome that would put the operator back where they started.

    Streamed at most once per change to the log: an unchanged
    (st_mtime_ns, st_size) stamp returns the cached scan byte-identically
    (see ``_scan_cache``).
    """
    path = Path(path)
    result: dict[str, Any] = {
        "schema": SHADOW_LOG_AUDIT_SCHEMA,
        "profile": profile.name,
        "path": str(path),
        "exists": False,
        "readable": None,
        "read_error": "",
        "bytes": 0,
        "modified_at": "",
        "lines": 0,
        "blank_lines": 0,
        "valid_rows": 0,
        "primary_rows": 0,
        "episode_rows": 0,
        "other_rows": 0,
        # Interior damage and a truncated tail are different failures and are
        # never merged into one "bad lines" number.
        "malformed_lines": 0,
        "malformed_examples": [],
        "truncated_final_line": False,
        "truncated_final_line_detail": "",
        "final_line_unterminated": False,
        "non_object_rows": 0,
        "undecodable_lines": 0,
        "schemas": {},
        "unknown_schemas": {},
        "unknown_schema_rows": 0,
        "engine_versions": {},
        "config_hashes": {},
        "machines": {},
        "timezones": {},
        "session_dates": {},
        "session_date_count": 0,
        "rows_missing_session_date": 0,
        "rows_missing_timestamp": 0,
        "rows_with_unparsable_timestamp": 0,
        "out_of_order_rows": 0,
        "out_of_order_examples": [],
        "future_rows": 0,
        "future_examples": [],
        "first_timestamp": "",
        "last_timestamp": "",
        "completed_bar_rows": 0,
        "incomplete_bar_rows": 0,
        "rows_without_bar_evidence": 0,
        "market_date": market_date,
        "rows_for_market_date": 0,
        "primary_rows_for_market_date": 0,
        "episode_rows_for_market_date": 0,
        "completed_bar_rows_for_market_date": 0,
        "reconcile_session_date": reconcile_session_date or market_date,
        "rows_for_reconcile_date": 0,
        "primary_rows_for_reconcile_date": 0,
        "episode_rows_for_reconcile_date": 0,
        "latest_valid_record": {},
        "accepted_schemas": sorted(profile.accepted_schemas),
    }
    if not path.exists():
        return result
    result["exists"] = True
    slot = (profile.name, str(path))
    cache_key: tuple | None = None
    try:
        stat = path.stat()
        result["bytes"] = int(stat.st_size)
        result["modified_at"] = datetime.fromtimestamp(
            stat.st_mtime, tz=local_tz
        ).isoformat(timespec="seconds")
        cache_key = (
            stat.st_mtime_ns,
            stat.st_size,
            str(local_tz),
            market_date,
            reconcile_session_date,
        )
    except OSError as exc:
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    if cache_key is not None:
        with _scan_cache_lock:
            cached = _scan_cache.get(slot)
        if cached is not None and cached[0] == cache_key:
            # Copied out: audit_shadow_log embeds the scan in its payload and
            # consumers annotate what they are handed.
            return copy.deepcopy(cached[1])

    schemas: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    engines: Counter[str] = Counter()
    hashes: Counter[str] = Counter()
    machines: Counter[str] = Counter()
    timezones: Counter[str] = Counter()
    sessions: Counter[str] = Counter()

    reconcile_date = str(result["reconcile_session_date"] or "")
    future_cutoff = now + timedelta(seconds=FUTURE_TOLERANCE_SECONDS)
    order_slack = timedelta(seconds=ORDER_TOLERANCE_SECONDS)
    previous_ts: datetime | None = None
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    # State of the line most recently handled; only after the loop ends do we
    # know it was the FINAL line, which is what makes truncation identifiable.
    last_line_number = 0
    last_line_terminated = True
    last_line_ok = True
    last_line_error = ""
    last_line_bytes = 0

    try:
        handle = path.open("rb")
    except OSError as exc:
        result["readable"] = False
        result["read_error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        result["readable"] = True
        with handle:
            for line_number, raw in enumerate(handle, start=1):  # streams; never slurps
                terminated = raw.endswith(b"\n")
                last_line_number = line_number
                last_line_terminated = terminated
                last_line_bytes = len(raw)
                last_line_ok = True
                last_line_error = ""
                result["lines"] = line_number
                try:
                    text = raw.decode("utf-8").strip()
                except UnicodeDecodeError as exc:
                    result["undecodable_lines"] += 1
                    last_line_ok = False
                    last_line_error = f"undecodable bytes: {exc}"
                    if len(result["malformed_examples"]) < MAX_EXAMPLES:
                        result["malformed_examples"].append(
                            {"line": line_number, "error": last_line_error}
                        )
                    result["malformed_lines"] += 1
                    continue
                if not text:
                    result["blank_lines"] += 1
                    continue
                try:
                    row = json.loads(text)
                except (ValueError, RecursionError) as exc:
                    last_line_ok = False
                    last_line_error = str(exc)
                    result["malformed_lines"] += 1
                    if len(result["malformed_examples"]) < MAX_EXAMPLES:
                        result["malformed_examples"].append(
                            {"line": line_number, "error": str(exc)[:200], "bytes": len(raw)}
                        )
                    continue
                if not isinstance(row, dict):
                    last_line_ok = False
                    last_line_error = f"row is a {type(row).__name__}, not a JSON object"
                    result["non_object_rows"] += 1
                    result["malformed_lines"] += 1
                    if len(result["malformed_examples"]) < MAX_EXAMPLES:
                        result["malformed_examples"].append(
                            {"line": line_number, "error": last_line_error}
                        )
                    continue

                result["valid_rows"] += 1
                schema = str(row.get("schema") or _MISSING)
                _bump(schemas, schema)
                kind = profile.kind(schema)
                if kind == "episode":
                    result["episode_rows"] += 1
                elif kind == "primary":
                    result["primary_rows"] += 1
                else:
                    result["other_rows"] += 1
                if schema not in profile.accepted_schemas:
                    result["unknown_schema_rows"] += 1
                    _bump(unknown, schema)
                _bump(engines, str(row.get("engine_version") or _MISSING))
                _bump(hashes, str(row.get("config_hash") or _MISSING))
                _bump(machines, str(row.get("machine") or _MISSING))
                _bump(timezones, str(row.get("timezone") or _MISSING))

                session_date = str(row.get("session_date") or "").strip()
                if session_date:
                    _bump(sessions, session_date)
                else:
                    result["rows_missing_session_date"] += 1

                stamp_text = ""
                for name in profile.order_fields:
                    if str(row.get(name) or "").strip():
                        stamp_text = str(row[name]).strip()
                        break
                if not stamp_text:
                    result["rows_missing_timestamp"] += 1
                    parsed = None
                else:
                    parsed = parse_timestamp(stamp_text, local_tz)
                    if parsed is None:
                        result["rows_with_unparsable_timestamp"] += 1
                        if len(result["malformed_examples"]) < MAX_EXAMPLES:
                            result["malformed_examples"].append(
                                {"line": line_number, "error": f"unparsable timestamp {stamp_text!r}"}
                            )
                if parsed is not None:
                    if previous_ts is not None and parsed < previous_ts - order_slack:
                        result["out_of_order_rows"] += 1
                        if len(result["out_of_order_examples"]) < MAX_EXAMPLES:
                            result["out_of_order_examples"].append(
                                {
                                    "line": line_number,
                                    "timestamp": stamp_text,
                                    "previous": previous_ts.isoformat(timespec="seconds"),
                                }
                            )
                    else:
                        previous_ts = parsed
                    if parsed > future_cutoff:
                        result["future_rows"] += 1
                        if len(result["future_examples"]) < MAX_EXAMPLES:
                            result["future_examples"].append(
                                {"line": line_number, "timestamp": stamp_text}
                            )
                    if first_ts is None or parsed < first_ts:
                        first_ts = parsed
                    if last_ts is None or parsed > last_ts:
                        last_ts = parsed

                complete = profile.completed_bar_evidence(row)
                if complete is None:
                    result["rows_without_bar_evidence"] += 1
                elif complete:
                    result["completed_bar_rows"] += 1
                else:
                    result["incomplete_bar_rows"] += 1

                if market_date and session_date == market_date:
                    result["rows_for_market_date"] += 1
                    if kind == "episode":
                        result["episode_rows_for_market_date"] += 1
                    elif kind == "primary":
                        result["primary_rows_for_market_date"] += 1
                    if complete:
                        result["completed_bar_rows_for_market_date"] += 1
                if reconcile_date and session_date == reconcile_date:
                    result["rows_for_reconcile_date"] += 1
                    if kind == "episode":
                        result["episode_rows_for_reconcile_date"] += 1
                    elif kind == "primary":
                        result["primary_rows_for_reconcile_date"] += 1

                result["latest_valid_record"] = _latest_summary(row, line_number)
    except OSError as exc:
        # A read failure partway through is still evidence: everything counted
        # before it stands, and the error says the tail is unaccounted for.
        result["readable"] = False
        result["read_error"] = f"{type(exc).__name__}: {exc}"

    # The crash signature: the last line has no newline AND does not parse.
    if last_line_number and not last_line_terminated:
        if last_line_ok:
            result["final_line_unterminated"] = True
        else:
            result["truncated_final_line"] = True
            result["malformed_lines"] = max(0, result["malformed_lines"] - 1)
            result["malformed_examples"] = [
                item for item in result["malformed_examples"] if item.get("line") != last_line_number
            ]
            result["truncated_final_line_detail"] = (
                f"line {last_line_number} is {last_line_bytes} byte(s), has no line terminator "
                f"and does not parse ({last_line_error[:160]}). A writer died mid-record."
            )

    result["schemas"] = dict(schemas)
    result["unknown_schemas"] = dict(unknown)
    result["engine_versions"] = dict(engines)
    result["config_hashes"] = dict(hashes)
    result["machines"] = dict(machines)
    result["timezones"] = dict(timezones)
    result["session_dates"] = dict(sessions)
    result["session_date_count"] = len(sessions)
    result["first_timestamp"] = first_ts.isoformat(timespec="seconds") if first_ts else ""
    result["last_timestamp"] = last_ts.isoformat(timespec="seconds") if last_ts else ""
    # A scan that hit a read error partway through is transient evidence and is
    # never cached - the next pass must re-measure it.
    if cache_key is not None and result["readable"] is True and not result["read_error"]:
        with _scan_cache_lock:
            if len(_scan_cache) > 32:
                _scan_cache.clear()
            _scan_cache[slot] = (cache_key, copy.deepcopy(result))
    return result


# ---------------------------------------------------------------------------
# sidecar reconciliation
# ---------------------------------------------------------------------------
#: ``equal``  - the sidecar counts exactly the rows it appended, so a mismatch
#:              in either direction is a discrepancy (over-claim is the serious
#:              one: it means rows the writer says it wrote are not in the log).
#: ``floor``  - the sidecar counts a SUBSET of what is appended (the Greatness
#:              board writes audit rows that never bump ``events_emitted``), so
#:              only a claim ABOVE the observed row count is a finding.
RELATION_EQUAL = "equal"
RELATION_FLOOR = "floor"


def reconcile_claims(claims, scan: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare each sidecar self-report against what the log actually holds."""
    rows: list[dict[str, Any]] = []
    for claim in claims:
        name = str(claim["claim"])
        observed_key = str(claim["observed_key"])
        relation = str(claim.get("relation") or RELATION_EQUAL)
        claimed = claim.get("claimed")
        if claimed is None:
            rows.append(
                {
                    "claim": name,
                    "claimed": None,
                    "observed": int(scan.get(observed_key, 0) or 0),
                    "observed_key": observed_key,
                    "relation": relation,
                    "state": "unclaimed",
                    "detail": f"The sidecar does not report {name}.",
                }
            )
            continue
        claimed_value = int(claimed)
        observed = int(scan.get(observed_key, 0) or 0)
        if claimed_value > observed:
            state = "over_claimed"
            detail = (
                f"The sidecar claims {claimed_value} {name} but the log holds {observed} "
                "matching row(s): the writer's self-report is not supported by its own output."
            )
        elif relation == RELATION_EQUAL and claimed_value < observed:
            state = "under_claimed"
            detail = (
                f"The sidecar claims {claimed_value} {name} but the log holds {observed} "
                "matching row(s); the counter was probably reset (session/config change) "
                "while the log kept accumulating."
            )
        else:
            state = "reconciled"
            detail = f"{claimed_value} claimed, {observed} row(s) in the log."
        rows.append(
            {
                "claim": name,
                "claimed": claimed_value,
                "observed": observed,
                "observed_key": observed_key,
                "relation": relation,
                "state": state,
                "delta": observed - claimed_value,
                "detail": detail,
            }
        )
    return rows


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def spy_claims(sidecar: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Reconcilable counters from ``spy_state_shadow_status.json``."""
    payload = sidecar or {}
    return [
        {
            "claim": "rows_written",
            "claimed": _int_or_none(payload.get("rows_written")),
            "observed_key": "primary_rows_for_reconcile_date",
            "relation": RELATION_EQUAL,
        },
        {
            "claim": "episode_rows_written",
            "claimed": _int_or_none(payload.get("episode_rows_written")),
            "observed_key": "episode_rows_for_reconcile_date",
            "relation": RELATION_EQUAL,
        },
    ]


def greatness_claims(coverage: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Reconcilable counters from ``greatness_candidates.json``'s coverage block.

    ``events_emitted`` counts engine transitions only; the board also appends
    stage/plan-revision/session-summary rows that never bump it, so the claim is
    a FLOOR on the log's row count, not an equality.
    """
    payload = coverage or {}
    return [
        {
            "claim": "events_emitted",
            "claimed": _int_or_none(payload.get("events_emitted")),
            "observed_key": "rows_for_reconcile_date",
            "relation": RELATION_FLOOR,
        }
    ]


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def audit_shadow_log(
    path: Path | str,
    profile: ShadowLogProfile,
    *,
    now: datetime,
    local_tz=None,
    market_date: str = "",
    reconcile_session_date: str = "",
    claims=(),
) -> dict[str, Any]:
    """Scan the log, reconcile it against the sidecar, and grade the evidence.

    The returned payload carries an explicit ``promotable`` verdict: plan.md
    sec 7's evidence floors are counted in rows of these logs, so a damaged log
    must make those floors unclaimable rather than merely paint a tile yellow.
    """
    scan = scan_shadow_log(
        path,
        profile,
        now=now,
        local_tz=local_tz,
        market_date=market_date,
        reconcile_session_date=reconcile_session_date,
    )
    reconciliation = reconcile_claims(claims, scan)

    # `reasons` are integrity defects and BLOCK promotability; `notes` are
    # observations that must be reported but are not damage. Keeping them apart
    # matters: "no rows yet today" is true every pre-market, and folding it into
    # the blocking list would make NOT PROMOTABLE the permanent default and
    # therefore meaningless on the morning a log really is broken.
    reasons: list[str] = []
    notes: list[str] = []
    statuses: list[str] = []
    headline: list[str] = []

    if not scan["exists"]:
        statuses.append(STATUS_UNKNOWN)
        reasons.append(f"{Path(path).name} does not exist; there is no raw evidence to audit.")
        headline.append("log absent")
    elif scan["readable"] is False:
        statuses.append(STATUS_UNHEALTHY)
        reasons.append(f"{Path(path).name} could not be read: {scan['read_error']}")
        headline.append("log unreadable")
    else:
        statuses.append(STATUS_HEALTHY)
        headline.append(
            f"{scan['valid_rows']} valid row(s) over {scan['session_date_count']} session(s)"
        )
        if scan["truncated_final_line"]:
            statuses.append(STATUS_UNHEALTHY)
            reasons.append(
                "The final line is truncated (unterminated and unparseable): a writer died "
                "mid-record, so the newest evidence is incomplete."
            )
            headline.append("TRUNCATED FINAL LINE")
        if scan["malformed_lines"]:
            statuses.append(STATUS_UNHEALTHY)
            reasons.append(
                f"{scan['malformed_lines']} malformed interior line(s) that every runtime reader "
                "silently skips."
            )
            headline.append(f"{scan['malformed_lines']} malformed line(s)")
        if scan["undecodable_lines"]:
            reasons.append(f"{scan['undecodable_lines']} line(s) are not valid UTF-8.")
        if scan["non_object_rows"]:
            reasons.append(f"{scan['non_object_rows']} line(s) parse to something other than a record.")
        if scan["unknown_schema_rows"]:
            statuses.append(STATUS_DEGRADED)
            names = ", ".join(sorted(scan["unknown_schemas"])) or "unnamed"
            reasons.append(
                f"{scan['unknown_schema_rows']} row(s) carry a schema this validator does not "
                f"accept ({names}); the evidence has drifted from the writer."
            )
            headline.append("unknown schema rows")
        if scan["rows_with_unparsable_timestamp"]:
            statuses.append(STATUS_DEGRADED)
            reasons.append(
                f"{scan['rows_with_unparsable_timestamp']} row(s) carry an unparsable timestamp."
            )
        if scan["rows_missing_timestamp"]:
            statuses.append(STATUS_DEGRADED)
            reasons.append(f"{scan['rows_missing_timestamp']} row(s) carry no timestamp at all.")
        if scan["out_of_order_rows"]:
            statuses.append(STATUS_DEGRADED)
            reasons.append(
                f"{scan['out_of_order_rows']} row(s) are stamped earlier than the row before "
                "them; append order and evaluation order disagree."
            )
            headline.append("out-of-order rows")
        if scan["future_rows"]:
            statuses.append(STATUS_DEGRADED)
            reasons.append(
                f"{scan['future_rows']} row(s) are stamped in the future relative to this audit; "
                "a clock or timezone is wrong."
            )
            headline.append("future-stamped rows")
        if scan["valid_rows"] == 0:
            statuses.append(STATUS_UNKNOWN)
            reasons.append(f"{Path(path).name} holds no valid records; coverage is unknown.")
        elif market_date and scan["rows_for_market_date"] == 0:
            notes.append(
                f"No rows for {market_date}: this session contributes nothing to the "
                "evidence base (normal before the first evaluation of the day)."
            )
        elif market_date and scan["completed_bar_rows_for_market_date"] == 0:
            statuses.append(STATUS_DEGRADED)
            reasons.append(
                f"None of the {scan['rows_for_market_date']} row(s) for {market_date} carry "
                "completed-bar evidence (plan.md sec 5: completed bars only)."
            )
        if len(scan["machines"]) > 1:
            statuses.append(STATUS_DEGRADED)
            names = ", ".join(sorted(name for name in scan["machines"] if name != _MISSING))
            reasons.append(f"{len(scan['machines'])} machines appended to one log: {names}.")
            headline.append("multi-machine writers")

    over = [row for row in reconciliation if row["state"] == "over_claimed"]
    under = [row for row in reconciliation if row["state"] == "under_claimed"]
    if over:
        statuses.append(STATUS_UNHEALTHY)
        for row in over:
            reasons.append(row["detail"])
        headline.append("sidecar over-claims the log")
    if under:
        statuses.append(STATUS_DEGRADED)
        for row in under:
            reasons.append(row["detail"])

    status = _worst(statuses)
    promotable = not reasons and status == STATUS_HEALTHY
    return {
        "schema": SHADOW_LOG_AUDIT_SCHEMA,
        "status": status,
        "promotable": promotable,
        "non_promotable_reasons": reasons,
        "notes": notes,
        "summary": (
            f"Log: {'; '.join(headline)}."
            if headline
            else "Log: no evidence."
        ),
        "scan": scan,
        "reconciliation": reconciliation,
        "promotion_note": (
            "plan.md sec 7 evidence floors are counted in these rows; they may not be "
            "claimed while this log is damaged, drifted, or contradicted by its sidecar."
        ),
    }


def audit_shadow_session_progress(
    log_path: Path | str,
    profile: ShadowLogProfile,
) -> dict[str, Any]:
    """Section 7 counters from finalized sessions; never a promotion verdict."""

    return audit_session_summaries(log_path, profile.name)
