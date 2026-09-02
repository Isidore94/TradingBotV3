"""Append-only log of every Alert Center review decision, for learning.

The learning triple is (features -> trader action -> outcome). Outcomes are
already tracked (intraday bounce outcomes CSV, human-focus forward returns)
and every bounce alert already carries its feature snapshot plus an
``event_id`` joining it to the 41-column candidates CSV. The missing middle -
what was SHOWN in the review pane and what the trader DID about it - is what
this module captures. Current builds write one JSONL shard per stable
machine-local installation under ``alert_review_events/``; readers merge those
shards with the read-only legacy ``alert_review_events.jsonl``.

Actions (see the Alert Center panel for the emit sites):
    shown           an alert became the active visual review (the impression -
                    the denominator for P(take | shown))
    skip            "Skip for now" - looked at the chart and passed
    remove_today    "Remove for today" / the ✕ dislike's removal
    restore_today   a removed symbol returned to processing
    add_focus       the review pane's type-matched focus add (advances queue)
    toggle_d1_focus / toggle_m5_focus   cross-focus toggles (detail.on)
    favorite        the feed item ★ (detail.on)
    dislike         the feed item ✕ (detail.reason)
    arm_watch / disarm_watch            one-shot session watches (detail.kind)
    watch_fired / watch_expired         how an armed watch actually ended
    arm_level / disarm_level            persistent price-level alerts
                    (detail.direction/level/fill_source: which quick-fill
                    button - vwap, upper_1, hod, ... - produced the price)
    level_fired     a persistent level alert triggered

Every row snapshots the alert's decision-relevant context as structured
fields (tier, bounce types, RRS numbers, session rvol, market environment),
so the log is analyzable standalone; ``event_id`` joins back to the full
candidate row when deeper features are needed. ``dwell_ms`` separates
"considered and passed" from "flushed the queue".

This module must stay import-light (no Qt, no pandas): the GUI calls it on
every click and the offline analysis job imports it headless.
"""

from __future__ import annotations

import json
import threading
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path
from project_paths import (
    ALERT_REVIEW_EVENTS_DIR,
    ALERT_REVIEW_EVENTS_FILE,
    LOCAL_SETTINGS_DIR,
)

REVIEW_EVENTS_SCHEMA = "review_events_v2"
LEGACY_REVIEW_EVENTS_SCHEMA = "review_events_v1"
SUPPORTED_REVIEW_EVENTS_SCHEMAS = frozenset(
    {LEGACY_REVIEW_EVENTS_SCHEMA, REVIEW_EVENTS_SCHEMA}
)
REVIEW_INSTALLATION_ID_FILE = LOCAL_SETTINGS_DIR / "review_installation_id"
_INSTALLATION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHARD_NAME_RE = re.compile(r"^review-events-([0-9a-f]{32})\.jsonl$")

_TIER_RE = re.compile(r"\[([SABCD])-TIER\]", re.IGNORECASE)
_PROVEN_RE = re.compile(r"\bPROVEN\b")

# context_json keys worth inlining on every row: the numbers the trader is
# implicitly weighing when they act on a chart. Everything else stays behind
# the event_id join.
_CONTEXT_FIELDS = (
    "rrs_spy",
    "rrs_sector",
    "rrs_industry",
    "session_rvol",
    "market_environment",
    "internals_tape",
    "internals_breadth_spread",
    "sector",
    "industry",
)


def _trade_date_text() -> str:
    try:
        from market_session import get_market_session_window

        return get_market_session_window().market_date.isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def _machine_name() -> str:
    """Which machine wrote the row.

    The log lives in the shared home and syncs across machines, so an episode
    count is only trustworthy if it can be attributed to one writer. Two
    machines appending to the same trade date is the concurrent-writer hazard
    the roadmap treats as an immediate rollback trigger; recording the name is
    what makes it detectable (review_capture_audit.py reports it).
    """
    try:
        import socket

        return socket.gethostname()
    except Exception:
        return ""


def _valid_installation_id(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if _INSTALLATION_ID_RE.fullmatch(text) else ""


def get_review_installation_id(
    path: Path = REVIEW_INSTALLATION_ID_FILE,
) -> str:
    """Return this installation's durable, machine-local identity.

    The identity file is deliberately outside the Drive-synchronized home.
    Exclusive creation makes concurrent first-start processes converge on the
    same value.  An unreadable, empty, or malformed existing identity fails
    closed: silently replacing it would make one physical installation look
    like two writers and strand evidence in a second shard.
    """

    target = Path(path)
    try:
        existing = _valid_installation_id(target.read_text(encoding="ascii"))
    except FileNotFoundError:
        existing = ""
    except OSError:
        return ""
    if existing:
        return existing
    if target.exists():
        return ""

    candidate = uuid.uuid4().hex
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="ascii") as handle:
            handle.write(candidate + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return candidate
    except FileExistsError:
        try:
            return _valid_installation_id(target.read_text(encoding="ascii"))
        except OSError:
            return ""
    except OSError:
        return ""


def review_event_shard_path(
    installation_id: str,
    *,
    shards_dir: Path = ALERT_REVIEW_EVENTS_DIR,
) -> Path | None:
    """The one shared-home file owned by ``installation_id``."""

    identity = _valid_installation_id(installation_id)
    if not identity:
        return None
    return Path(shards_dir) / f"review-events-{identity}.jsonl"


def review_event_shard_installation_id(path: Path) -> str:
    """Installation identity encoded by a canonical shard filename."""

    match = _SHARD_NAME_RE.fullmatch(Path(path).name)
    return match.group(1) if match else ""


def review_event_sources(
    path: Path = ALERT_REVIEW_EVENTS_FILE,
    *,
    shards_dir: Path | None = None,
    include_shards: bool | None = None,
) -> list[Path]:
    """Readable legacy + partitioned sources in deterministic file order."""

    legacy = Path(path)
    canonical = legacy == Path(ALERT_REVIEW_EVENTS_FILE)
    partitioned = (
        canonical or shards_dir is not None
        if include_shards is None
        else bool(include_shards)
    )
    sources = [legacy] if legacy.exists() else []
    if not partitioned:
        return sources
    directory = Path(shards_dir) if shards_dir is not None else (
        Path(ALERT_REVIEW_EVENTS_DIR)
        if canonical
        else legacy.with_name(legacy.stem)
    )
    try:
        shards = sorted(
            candidate
            for candidate in directory.glob("review-events-*.jsonl")
            if candidate.is_file()
        )
    except OSError:
        shards = []
    sources.extend(shards)
    return sources


def review_event_store_mtime(
    path: Path = ALERT_REVIEW_EVENTS_FILE,
    *,
    shards_dir: Path | None = None,
    include_shards: bool | None = None,
) -> float | None:
    """Latest source mtime, including shards; None when nothing is readable."""

    values: list[float] = []
    for source in review_event_sources(
        path, shards_dir=shards_dir, include_shards=include_shards
    ):
        try:
            values.append(source.stat().st_mtime)
        except OSError:
            continue
    return max(values) if values else None


def _as_float(value) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved == resolved else None  # drop NaN


def alert_context_fields(alert) -> dict[str, Any]:
    """Structured decision context from a BounceAlert-shaped object.

    Duck-typed so tests and the offline job can pass plain stand-ins; every
    field degrades to ""/None rather than raising, because a malformed alert
    must never break the click that is being logged.
    """
    fields: dict[str, Any] = {}
    if alert is None:
        return fields
    raw_text = str(getattr(alert, "raw_text", "") or "")
    match = _TIER_RE.search(raw_text)
    fields["tier"] = match.group(1).upper() if match else ""
    fields["proven"] = bool(_PROVEN_RE.search(raw_text))
    # RETIRED 2026-09-01 (trader: "We can probably remove this because idk what
    # it is"). The column stays, always False, so every reader of the 8,818
    # historical rows keeps working and the row shape does not move. Nothing
    # in the tree ever emitted the token - 0 of those rows carried True - so
    # writing the constant loses no information.
    fields["banger"] = False
    fields["tag"] = str(getattr(alert, "tag", "") or "")
    fields["timeframe"] = str(getattr(alert, "timeframe", "") or "")
    fields["is_d1"] = bool(getattr(alert, "is_d1", False))
    fields["trigger"] = str(getattr(alert, "trigger", "") or "")

    payload = getattr(alert, "payload", None)
    feedback = payload.get("feedback") if isinstance(payload, dict) else None
    feedback = feedback if isinstance(feedback, dict) else {}
    fields["event_id"] = str(feedback.get("event_id") or "")
    fields["bounce_types"] = str(feedback.get("bounce_types") or "")
    fields["entry_price"] = _as_float(feedback.get("entry_price"))
    fields["stop_price"] = _as_float(feedback.get("stop_price"))
    fields["risk_per_share"] = _as_float(feedback.get("risk_per_share"))
    fields["score"] = _as_float(feedback.get("score"))
    fields["is_focus_pick"] = bool(feedback.get("is_focus_pick"))

    context = feedback.get("context_json")
    if isinstance(context, str) and context.strip():
        try:
            context = json.loads(context)
        except (json.JSONDecodeError, ValueError):
            context = None
    if isinstance(context, dict):
        for key in _CONTEXT_FIELDS:
            if key in context:
                fields[key] = context.get(key)

    # Chart-watch hits carry their own payload shape instead of feedback.
    if isinstance(payload, dict) and payload.get("chart_watch_kind"):
        fields["chart_watch_kind"] = str(payload.get("chart_watch_kind") or "")
    return fields


def setup_context_fields(row) -> dict[str, Any]:
    """Structured decision context from a SetupRow-shaped object (the Master
    AVWAP setups table). The swing-side counterpart of ``alert_context_fields``
    - buckets, families, and tags are the dimensions the swing scoreboard
    aggregates on, and a setup row carries them natively.
    """
    fields: dict[str, Any] = {"surface": "setups", "is_d1": True, "timeframe": "D1"}
    if row is None:
        return fields
    fields["bucket"] = str(getattr(row, "bucket", "") or "")
    raw = getattr(row, "raw", None)
    raw = raw if isinstance(raw, dict) else {}
    fields["setup_family"] = str(
        raw.get("setup_family") or raw.get("master_avwap_setup_family") or ""
    )
    tags = getattr(row, "setup_tags", None) or []
    fields["setup_tags"] = ";".join(str(tag) for tag in tags if str(tag or "").strip())
    fields["score"] = _as_float(getattr(row, "score", None))
    fields["expected_r"] = _as_float(getattr(row, "expected_r", None))
    fields["days_to_earnings"] = getattr(row, "days_to_earnings", None)
    fields["d1_vs_sector"] = _as_float(getattr(row, "d1_vs_sector", None))
    fields["d1_vs_industry"] = _as_float(getattr(row, "d1_vs_industry", None))
    fields["sector"] = str(getattr(row, "sector", "") or "")
    fields["industry"] = str(getattr(row, "industry", "") or "")
    return fields


def record_review_event(
    action: str,
    *,
    alert=None,
    symbol: object = "",
    side: object = "",
    detail: dict[str, Any] | None = None,
    context_fields: dict[str, Any] | None = None,
    dwell_ms: int | None = None,
    queue_len: int | None = None,
    now: datetime | None = None,
    path: Path = ALERT_REVIEW_EVENTS_FILE,
    shards_dir: Path | None = None,
    installation_id_path: Path = REVIEW_INSTALLATION_ID_FILE,
    partitioned: bool | None = None,
) -> dict[str, Any] | None:
    """Append one decision row. Returns the row, or None when unusable.

    Best-effort like pick_feedback: a cloud-synced folder briefly locking the
    file must never surface as a GUI error, so OSError is swallowed.
    """
    action_text = str(action or "").strip().lower()
    sym = str(symbol or getattr(alert, "symbol", "") or "").strip().upper()
    if not action_text or not sym:
        return None
    installation_id = get_review_installation_id(installation_id_path)
    if not installation_id:
        return None
    side_text = str(side or getattr(alert, "side", "") or "").strip().upper()
    timestamp = now or datetime.now()
    row: dict[str, Any] = {
        "schema": REVIEW_EVENTS_SCHEMA,
        "review_record_id": uuid.uuid4().hex,
        "ts": timestamp.isoformat(timespec="microseconds"),
        "trade_date": _trade_date_text(),
        "installation_id": installation_id,
        "machine": _machine_name(),
        "pid": os.getpid(),
        "action": action_text,
        "symbol": sym,
        "side": side_text,
    }
    row.update(alert_context_fields(alert))
    if context_fields:
        # Surface-specific structured context (e.g. setup_context_fields for
        # the setups table). Applied after the alert extraction so a richer
        # explicit snapshot wins over duck-typed defaults.
        row.update({k: v for k, v in context_fields.items() if v not in (None, "")})
    if dwell_ms is not None:
        row["dwell_ms"] = max(0, int(dwell_ms))
    if queue_len is not None:
        row["queue_len"] = max(0, int(queue_len))
    if detail:
        row["detail"] = detail
    try:
        legacy = Path(path)
        canonical = legacy == Path(ALERT_REVIEW_EVENTS_FILE)
        use_partitioned = canonical if partitioned is None else bool(partitioned)
        if use_partitioned:
            directory = Path(shards_dir) if shards_dir is not None else (
                Path(ALERT_REVIEW_EVENTS_DIR)
                if canonical
                else legacy.with_name(legacy.stem)
            )
            target = review_event_shard_path(installation_id, shards_dir=directory)
            if target is None:
                return None
        else:
            target = legacy
        target.parent.mkdir(parents=True, exist_ok=True)
        # Partitioning removes cross-machine contention.  The local kernel
        # lock covers two GUI processes from the *same* installation, which
        # intentionally share one shard.
        with local_writer_lock(lock_key_for_path(target), timeout_seconds=1.0):
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    except (OSError, LocalLockUnavailable):
        return None
    return row


#: (sources, mtime, size) -> parsed rows for the last store read. The store is
#: append-only within a session, so a stamp that has not moved cannot describe
#: different rows.
_events_cache: tuple[tuple, float | None, int, list[dict[str, Any]]] | None = None
_events_cache_lock = threading.Lock()


def _store_stamp(sources: list[Path]) -> tuple[float | None, int]:
    """(latest mtime, total bytes) across the sources, cheap enough to poll."""
    latest: float | None = None
    total = 0
    for source in sources:
        try:
            stat = source.stat()
        except OSError:
            continue
        total += stat.st_size
        latest = stat.st_mtime if latest is None else max(latest, stat.st_mtime)
    return latest, total


def load_review_events(
    path: Path = ALERT_REVIEW_EVENTS_FILE,
    *,
    shards_dir: Path | None = None,
    include_shards: bool | None = None,
) -> list[dict[str, Any]]:
    """All legacy + shard rows, oldest first. Bad lines are skipped.

    Parsed at most once per change to the store. On 2026-08-21 this was four
    files, 5.8 MB and 8808 rows - 74 ms of JSON per call - and the GUI-thread
    stall log caught the main thread inside the parse loop 27 times, once for
    20.5 seconds. Nothing about the rows changes between two calls that see the
    same mtimes and the same byte count, so the second call reads no file.

    Size is part of the key, not just mtime: an append inside the same
    filesystem timestamp tick still moves the byte count.

    The cached list is copied out, because callers sort and filter what they
    get back and the rows are handed on to consumers that annotate them.
    """
    global _events_cache

    sources = review_event_sources(
        path, shards_dir=shards_dir, include_shards=include_shards
    )
    key = tuple(str(source) for source in sources)
    mtime, size = _store_stamp(sources)
    with _events_cache_lock:
        cached = _events_cache
        if cached is not None and cached[0] == key and cached[1] == mtime and cached[2] == size:
            return [dict(row) for row in cached[3]]
    collected: list[tuple[str, int, int, dict[str, Any]]] = []
    seen_record_ids: set[str] = set()
    for source_index, source in enumerate(sources):
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_index, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            record_id = str(row.get("review_record_id") or "").strip()
            if record_id and record_id in seen_record_ids:
                continue
            if record_id:
                seen_record_ids.add(record_id)
            collected.append(
                (
                    str(row.get("ts") or ""),
                    source_index,
                    line_index,
                    row,
                )
            )
    if len(sources) > 1:
        collected.sort(key=lambda item: item[:3])
    rows = [item[3] for item in collected]
    with _events_cache_lock:
        _events_cache = (key, mtime, size, rows)
    return [dict(row) for row in rows]
