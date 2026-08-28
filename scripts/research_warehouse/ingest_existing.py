"""Bronze wraps of the existing artifacts, plus the daily universe/geometry
snapshots (plan Phase 2; inventory in sec 19.0, migration rules in sec 19.5).

The principle is fixed: **reimplement nothing, re-own nothing**. The legacy
writer keeps writing exactly as today; the warehouse ingests beside it, hashes
what it read, and never touches the original file. Two dispositions exist here:

* **wrap-as-bronze** - the artifact's records are preserved verbatim in a
  ``bronze_<artifact>`` dataset, one row per source record, each carrying the
  source path, the source file's SHA-256, its offset, and a record hash;
* **reuse-as-is** - per-symbol D1 stores, HV level stores and ``d1_level_feed``
  are read through their own loaders (wrapped reads) and projected into the
  canonical silver datasets. No copy of those stores is made.

Idempotency (the Phase-2 exit criterion) comes from the manifest, not from a
side-car state file: every bronze publish records the source path, the source
file hash, and the highest offset ingested, so a re-run with an unchanged
source writes nothing at all. Append-only logs resume from their watermark;
whole-file snapshots version by content hash.

Every entry point is inert when the warehouse is disabled, and nothing in this
module can change a detector, score, ranking, or alert.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

try:  # package import
    from . import config
    from .manifest import ACTION_PUBLISH, utc_now
    from .schemas import (
        BRONZE_FORMAT_CSV_ROW,
        BRONZE_FORMAT_JSON,
        BRONZE_FORMAT_JSONL,
        SCHEMA_VERSION,
        bronze_dataset_name,
        level_id,
    )
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import config  # type: ignore
    from manifest import ACTION_PUBLISH, utc_now  # type: ignore
    from schemas import (  # type: ignore
        BRONZE_FORMAT_CSV_ROW,
        BRONZE_FORMAT_JSON,
        BRONZE_FORMAT_JSONL,
        SCHEMA_VERSION,
        bronze_dataset_name,
        level_id,
    )
    from store import ResearchStore  # type: ignore

# Wrapped legacy evidence is recorded after the fact: it is never LIVE capture,
# so every bronze row is BACKFILL and is excluded from AS_OBSERVED coverage,
# latency, live-shadow, and promotion claims (sec 9.3).
BRONZE_CAPTURE_MODE = "BACKFILL"
QUALITY_COMPLETE = "COMPLETE"
QUALITY_INVALID = "INVALID_DATA"

#: Above this size a SNAPSHOT payload is stored WITHOUT being `json.loads`-ed
#: (BD-73 in docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md). The payload is still
#: captured in full and its format is still declared; what is skipped is only
#: the in-memory parse, which for a gigabyte file costs several GB inside the
#: desk process. `master_avwap_setup_tracker.json` measured 1,026,057,028 bytes
#: on 2026-08-27; every other bronze snapshot on the desk is orders of
#: magnitude smaller, so 64 MB separates them with a wide margin.
SNAPSHOT_PARSE_MAX_BYTES = 64 * 1024 * 1024
#: Read the file in chunks rather than whole when only its hash is wanted.
_HASH_CHUNK_BYTES = 1024 * 1024

MODE_APPEND_LOG = "APPEND_LOG"
MODE_SNAPSHOT = "SNAPSHOT"
MODE_CSV_ROWS = "CSV_ROWS"

EXPLORATION_COHORT_FILE = Path(__file__).with_name("exploration_cohort.txt")

# Extra keys the bronze publish stamps on its manifest line; they are the
# ingest watermark, so no mutable state file is needed anywhere.
EXTRA_SOURCE_PATH = "bronze_source_path"
EXTRA_SOURCE_SHA = "bronze_source_sha256"
EXTRA_MAX_OFFSET = "bronze_max_offset"


def _paths():
    try:
        from scripts import project_paths
    except ImportError:  # pragma: no cover - scripts/ on sys.path
        import project_paths  # type: ignore
    return project_paths


@dataclass(frozen=True)
class BronzeArtifact:
    """One legacy artifact and how to wrap it without re-owning it."""

    artifact: str
    mode: str
    # Resolved lazily: paths depend on the trader's configured home folder.
    path_attr: str = ""
    path_factory: Callable[[], Path] | None = None
    event_keys: tuple[str, ...] = ()
    id_keys: tuple[str, ...] = ()
    # Class A = irreplaceable-small, mirrored to Drive + backup disk (sec 8.5).
    class_a: bool = False
    note: str = ""

    @property
    def dataset(self) -> str:
        return bronze_dataset_name(self.artifact)

    def resolve_path(self) -> Path | None:
        if self.path_factory is not None:
            try:
                return Path(self.path_factory())
            except Exception:  # pragma: no cover - a missing legacy helper
                return None
        if self.path_attr:
            value = getattr(_paths(), self.path_attr, None)
            return Path(value) if value else None
        return None


def _diagnostics(name: str):
    return lambda: _paths().get_diagnostics_dir() / name


def _technical_integrity_events_path():
    def resolve():
        _ensure_scripts_on_path()
        try:
            from technical_integrity import technical_integrity_events_path
        except ImportError:  # pragma: no cover - packaged import
            from scripts.technical_integrity import technical_integrity_events_path  # type: ignore
        return technical_integrity_events_path()

    return resolve


# The sec 19.0 inventory, wrap-as-bronze and daily-ingest rows only. Reuse-as-is
# rows (D1/H1 stores, HV level stores, d1_level_feed) are wrapped reads and
# appear further down, not here.
BRONZE_ARTIFACTS: tuple[BronzeArtifact, ...] = (
    # --- setup tracker + scenario CSVs -------------------------------------
    BronzeArtifact(
        "setup_tracker",
        MODE_SNAPSHOT,
        path_attr="MASTER_AVWAP_SETUP_TRACKER_FILE",
        note="legacy IDs and watermarks preserved; the tracker writer is untouched",
    ),
    BronzeArtifact("setup_scenarios", MODE_CSV_ROWS, path_attr="MASTER_AVWAP_SETUP_SCENARIOS_FILE"),
    BronzeArtifact("setup_daily", MODE_CSV_ROWS, path_attr="MASTER_AVWAP_SETUP_DAILY_FILE"),
    BronzeArtifact("tracker_scoring_snapshot", MODE_SNAPSHOT, path_attr="MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE"),
    # --- bounce ledgers / day-trade tracker outputs -------------------------
    BronzeArtifact("intraday_bounces", MODE_CSV_ROWS, path_attr="INTRADAY_BOUNCES_FILE"),
    BronzeArtifact("intraday_bounce_candidates", MODE_CSV_ROWS, path_attr="INTRADAY_BOUNCE_CANDIDATES_FILE"),
    BronzeArtifact("intraday_bounce_outcomes", MODE_CSV_ROWS, path_attr="INTRADAY_BOUNCE_OUTCOMES_FILE"),
    BronzeArtifact("intraday_bounce_feedback", MODE_CSV_ROWS, path_attr="INTRADAY_BOUNCE_FEEDBACK_FILE"),
    # --- review / preference evidence (Class A) ----------------------------
    BronzeArtifact(
        "alert_review_events",
        MODE_APPEND_LOG,
        path_attr="ALERT_REVIEW_EVENTS_FILE",
        event_keys=("ts", "timestamp", "event_at", "created_at"),
        id_keys=("event_id", "id", "alert_id"),
        class_a=True,
    ),
    BronzeArtifact("pick_feedback", MODE_APPEND_LOG, path_attr="PICK_FEEDBACK_FILE", class_a=True),
    BronzeArtifact(
        "market_environment_annotations",
        MODE_APPEND_LOG,
        path_attr="MARKET_ENVIRONMENT_ANNOTATIONS_FILE",
        class_a=True,
    ),
    # --- regime / RS / shadow artifacts ------------------------------------
    BronzeArtifact(
        "spy_state_shadow",
        MODE_APPEND_LOG,
        path_factory=_diagnostics("spy_state_shadow.jsonl"),
        event_keys=("ts", "timestamp", "as_of"),
    ),
    BronzeArtifact(
        "greatness_shadow",
        MODE_APPEND_LOG,
        path_factory=_diagnostics("greatness_shadow.jsonl"),
        event_keys=("ts", "timestamp", "as_of"),
    ),
    BronzeArtifact("regime_pause_observations", MODE_SNAPSHOT, path_attr="REGIME_PAUSE_OBSERVATIONS_FILE"),
    BronzeArtifact("industry_board_snapshot", MODE_SNAPSHOT, path_attr="INDUSTRY_BOARD_STATE_FILE"),
    BronzeArtifact("industry_intraday_rs_snapshot", MODE_SNAPSHOT, path_attr="INDUSTRY_INTRADAY_RS_STATE_FILE"),
    BronzeArtifact("rrs_strength_extremes", MODE_CSV_ROWS, path_attr="RRS_STRENGTH_LOG_FILE"),
    BronzeArtifact("rrs_group_strength_extremes", MODE_CSV_ROWS, path_attr="RRS_GROUP_STRENGTH_LOG_FILE"),
    # --- technical integrity (its retention cleanup unlocks after ingest) ---
    BronzeArtifact(
        "technical_integrity_events",
        MODE_APPEND_LOG,
        path_factory=_technical_integrity_events_path(),
        event_keys=("ts", "timestamp", "observed_at"),
    ),
    # --- run manifests / job ledger / heartbeat ----------------------------
    BronzeArtifact(
        "job_ledger",
        MODE_APPEND_LOG,
        path_factory=_diagnostics("job_ledger.jsonl"),
        event_keys=("ts", "timestamp", "scheduled_at"),
        id_keys=("key", "job_key"),
    ),
    BronzeArtifact("heartbeat", MODE_SNAPSHOT, path_factory=_diagnostics("heartbeat.json")),
    # --- anchors (feeds anchor_instance from Phase 5) ----------------------
    BronzeArtifact("earnings_avwap_anchors", MODE_CSV_ROWS, path_attr="EARNINGS_ANCHORS_FILE", class_a=True),
    BronzeArtifact("earnings_calendar_history", MODE_SNAPSHOT, path_attr="EARNINGS_CALENDAR_HISTORY_FILE", class_a=True),
    # --- trader geometry / watch JSONs, daily from creation time (Class A) --
    BronzeArtifact("d1_level_watches", MODE_SNAPSHOT, path_attr="D1_LEVEL_WATCHES_FILE", class_a=True),
    BronzeArtifact("d1_event_watches", MODE_SNAPSHOT, path_attr="D1_EVENT_WATCHES_FILE", class_a=True),
    BronzeArtifact("alert_chart_watches", MODE_SNAPSHOT, path_attr="ALERT_CHART_WATCHES_FILE", class_a=True),
    BronzeArtifact("price_alerts", MODE_SNAPSHOT, path_attr="PRICE_ALERTS_FILE", class_a=True),
)

RUN_MANIFEST_ARTIFACT = BronzeArtifact(
    "run_manifests",
    MODE_SNAPSHOT,
    path_factory=_diagnostics("run_manifests"),
    note="one snapshot row per manifest file in the directory",
)

CLASS_A_ARTIFACTS = tuple(artifact.artifact for artifact in BRONZE_ARTIFACTS if artifact.class_a)


@dataclass
class IngestReport:
    artifact: str
    dataset: str
    source_path: str = ""
    source_sha256: str = ""
    status: str = "OK"  # OK | UNCHANGED | MISSING_SOURCE | DISABLED
    rows_ingested: int = 0
    rows_quarantined: int = 0
    files_sealed: int = 0


@dataclass
class SnapshotReport:
    dataset: str
    session_date: str = ""
    status: str = "OK"  # OK | ALREADY_CAPTURED | NO_SOURCE | DISABLED
    rows: int = 0
    sources: list[str] = field(default_factory=list)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    """The same digest `_sha256_bytes(path.read_bytes())` gives, in chunks.

    Identical by construction - sha256 is a streaming hash - which matters
    because the watermarks already on disk were written by the whole-file
    spelling. A different digest here would make every artifact look changed
    forever and re-publish the lot.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_json(text: str) -> bool:
    """Cheap structural check for a payload too large to parse.

    Deliberately weak and deliberately honest: it establishes that the text
    OPENS and CLOSES like a JSON container, which is what lets the row claim
    `payload_format=json`. It cannot prove the middle parses, so it is used
    only above the size threshold, where the alternative is several GB of
    dicts that this artifact derives nothing from.
    """
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    return (stripped[0], stripped[-1]) in (("{", "}"), ("[", "]"))


def _record_hash(source_path: str, offset: int, payload: str, *, include_offset: bool) -> str:
    material = f"{source_path}|{offset}|{payload}" if include_offset else payload
    return _sha256_bytes(material.encode("utf-8"))


def _parse_event_at(payload, keys: tuple[str, ...]):
    if not keys or not isinstance(payload, dict):
        return None
    for key in keys:
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        text = str(raw).strip().replace("Z", "+00:00")
        for candidate in (text, text.replace(" ", "T", 1)):
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                continue
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _first_value(payload, keys: tuple[str, ...]) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _watermark(store: ResearchStore, dataset: str) -> dict:
    """Last publish state for a bronze dataset, read from the manifest."""
    latest = None
    max_offset = -1
    for entry in store.manifest.read_entries():
        if entry.dataset != dataset or entry.action != ACTION_PUBLISH:
            continue
        latest = entry
        try:
            max_offset = max(max_offset, int(entry.extra.get(EXTRA_MAX_OFFSET, -1)))
        except (TypeError, ValueError):
            pass
    return {
        "latest": latest,
        "max_offset": max_offset,
        "last_source_sha": str(latest.extra.get(EXTRA_SOURCE_SHA) or "") if latest else "",
    }


def _bronze_row(
    artifact: BronzeArtifact,
    *,
    source_path: str,
    source_sha: str,
    offset: int,
    payload_text: str,
    payload_format: str,
    parsed,
    observed_at: datetime,
    run_id: str,
    include_offset_in_hash: bool,
) -> dict:
    event_at = _parse_event_at(parsed, artifact.event_keys)
    return {
        "source_artifact": artifact.artifact,
        "source_path": source_path,
        "source_sha256": source_sha,
        "source_offset": offset,
        "record_hash": _record_hash(source_path, offset, payload_text, include_offset=include_offset_in_hash),
        "legacy_id": _first_value(parsed, artifact.id_keys),
        "payload": payload_text,
        "payload_format": payload_format,
        "quality": QUALITY_COMPLETE if parsed is not None else QUALITY_INVALID,
        "event_at": event_at,
        "observed_at": observed_at,
        "partition_ts": event_at or observed_at,
        "capture_mode": BRONZE_CAPTURE_MODE,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }


def ingest_artifact(
    store: ResearchStore | None,
    artifact: BronzeArtifact,
    *,
    run_id: str = "",
    job_id: str = "bronze_ingest",
    path: Path | None = None,
    now: datetime | None = None,
) -> IngestReport:
    """Wrap one legacy artifact into bronze. A re-run with no change is a no-op."""
    report = IngestReport(artifact=artifact.artifact, dataset=artifact.dataset)
    if store is None:
        report.status = "DISABLED"
        return report
    source = Path(path) if path is not None else artifact.resolve_path()
    if source is None or not source.exists():
        # A machine that has never produced this artifact is not an error.
        report.status = "MISSING_SOURCE"
        report.source_path = str(source or "")
        return report
    observed_at = now or utc_now()
    report.source_path = str(source)

    if artifact.mode == MODE_SNAPSHOT and source.is_dir():
        return _ingest_snapshot_dir(store, artifact, source, report, run_id, job_id, observed_at)

    # Hash the file WITHOUT holding it. The watermark comparison below only
    # ever needed the digest, but this read the whole file first - which for
    # `master_avwap_setup_tracker.json` (1.03 GB on 2026-08-27) meant every
    # ingest allocated a gigabyte inside the desk process, including the ones
    # that immediately concluded UNCHANGED.
    source_sha = _sha256_path(source)
    report.source_sha256 = source_sha
    state = _watermark(store, artifact.dataset)

    # Compared against the LAST version, not every version ever seen: a
    # document that reverts to earlier content is a real new version of a
    # trader-owned file, not a duplicate to swallow. Hoisted above the mode
    # split because both branches asked exactly this question.
    if source_sha == state["last_source_sha"]:
        report.status = "UNCHANGED"
        return report

    raw = source.read_bytes()
    if artifact.mode == MODE_SNAPSHOT:
        rows = [_snapshot_row(artifact, source, raw, source_sha, observed_at, run_id, offset=0)]
    else:
        rows = _log_rows(artifact, source, raw, source_sha, observed_at, run_id, after_offset=state["max_offset"])
        if not rows:
            report.status = "UNCHANGED"
            return report

    result = store.publish(
        artifact.dataset,
        rows,
        job_id=job_id,
        extra=_source_extras(artifact, str(source), source_sha, rows),
    )
    report.rows_ingested = result.rows_published
    report.rows_quarantined = result.rows_quarantined
    report.files_sealed = len(result.published)
    return report


def _source_extras(artifact: BronzeArtifact, source_path: str, source_sha: str, rows) -> dict:
    """The ingest watermark, carried on the seal line itself.

    The manifest is the only state this module keeps: source path, source file
    hash, and the highest offset read. A re-run compares the current file hash
    against the recorded one and stops there when nothing changed.
    """
    return {
        EXTRA_SOURCE_PATH: source_path,
        EXTRA_SOURCE_SHA: source_sha,
        EXTRA_MAX_OFFSET: max((int(row["source_offset"]) for row in rows), default=0),
        "bronze_artifact": artifact.artifact,
        "bronze_mode": artifact.mode,
        "class_a": artifact.class_a,
    }


def _snapshot_row(artifact, source: Path, raw: bytes, source_sha: str, observed_at, run_id, offset: int) -> dict:
    """One bronze row for a whole-file snapshot.

    Above `SNAPSHOT_PARSE_MAX_BYTES` the payload is stored in full but NOT
    parsed. What the parse feeds is narrow: `_parse_event_at` and
    `_first_value`, both keyed on the artifact's `event_keys`/`id_keys`, plus
    the `quality` flag. `setup_tracker` - the only artifact anywhere near the
    threshold - declares NEITHER key tuple, so for it the parse influences
    exactly one column and the skip costs nothing measurable; a regression test
    asserts the parsed and skipped rows come out identical.

    For a large artifact that DID declare those keys the skip would empty them,
    so this is a documented decision (BD-73) with that as its reopen trigger,
    not an invisible optimisation.
    """
    text = raw.decode("utf-8", errors="replace")
    if len(raw) > SNAPSHOT_PARSE_MAX_BYTES:
        # `parsed = {}` means "nothing derivable", which `_parse_event_at` and
        # `_first_value` both already treat as empty; the quality flag comes
        # from the structural check instead of from a parse we did not run.
        return _bronze_row(
            artifact,
            source_path=str(source),
            source_sha=source_sha,
            offset=offset,
            payload_text=text,
            payload_format=BRONZE_FORMAT_JSON,
            parsed={} if _looks_like_json(text) else None,
            observed_at=observed_at,
            run_id=run_id,
            include_offset_in_hash=False,
        )
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    return _bronze_row(
        artifact,
        source_path=str(source),
        source_sha=source_sha,
        offset=offset,
        payload_text=text,
        payload_format=BRONZE_FORMAT_JSON,
        parsed=parsed if isinstance(parsed, dict) else ({} if parsed is not None else None),
        observed_at=observed_at,
        run_id=run_id,
        include_offset_in_hash=False,
    )


def _log_rows(artifact, source: Path, raw: bytes, source_sha: str, observed_at, run_id, *, after_offset: int):
    text = raw.decode("utf-8", errors="replace")
    rows: list[dict] = []
    if artifact.mode == MODE_CSV_ROWS:
        reader = csv.reader(io.StringIO(text))
        try:
            header = next(reader)
        except StopIteration:
            return rows
        for offset, values in enumerate(reader, start=1):
            if offset <= after_offset:
                continue
            parsed = dict(zip(header, values))
            rows.append(
                _bronze_row(
                    artifact,
                    source_path=str(source),
                    source_sha=source_sha,
                    offset=offset,
                    payload_text=json.dumps(parsed, sort_keys=True),
                    payload_format=BRONZE_FORMAT_CSV_ROW,
                    parsed=parsed,
                    observed_at=observed_at,
                    run_id=run_id,
                    include_offset_in_hash=True,
                )
            )
        return rows

    for offset, line in enumerate(text.splitlines(), start=1):
        if offset <= after_offset or not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            parsed = None  # preserved verbatim, marked INVALID_DATA
        rows.append(
            _bronze_row(
                artifact,
                source_path=str(source),
                source_sha=source_sha,
                offset=offset,
                payload_text=line,
                payload_format=BRONZE_FORMAT_JSONL,
                parsed=parsed if isinstance(parsed, dict) else ({} if parsed is not None else None),
                observed_at=observed_at,
                run_id=run_id,
                include_offset_in_hash=True,
            )
        )
    return rows


def _ingest_snapshot_dir(store, artifact, source: Path, report, run_id, job_id, observed_at) -> IngestReport:
    """A directory of immutable snapshots (run manifests): one row per file.

    Each manifest file is written once and never edited, so the per-file
    content hash already in bronze is the whole idempotency rule.
    """
    seen = set(
        store.read_table(artifact.dataset, columns=["source_sha256"]).column("source_sha256").to_pylist()
    )
    rows: list[dict] = []
    digests: list[str] = []
    for offset, path in enumerate(sorted(source.rglob("*.json")), start=1):
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        digest = _sha256_bytes(raw)
        if digest in seen:
            continue
        seen.add(digest)
        digests.append(digest)
        rows.append(_snapshot_row(artifact, path, raw, digest, observed_at, run_id, offset=offset))
    if not rows:
        report.status = "UNCHANGED"
        return report
    directory_digest = _sha256_bytes("|".join(digests).encode("utf-8"))
    report.source_sha256 = directory_digest
    result = store.publish(
        artifact.dataset,
        rows,
        job_id=job_id,
        extra=_source_extras(artifact, str(source), directory_digest, rows),
    )
    report.rows_ingested = result.rows_published
    report.rows_quarantined = result.rows_quarantined
    report.files_sealed = len(result.published)
    return report


def run_bronze_ingest(
    store: ResearchStore | None,
    *,
    run_id: str = "",
    job_id: str = "bronze_ingest",
    artifacts=None,
    now: datetime | None = None,
) -> list[IngestReport]:
    """Wrap every registered artifact. Safe and cheap to run every night."""
    if store is None:
        return []
    selected = tuple(artifacts) if artifacts is not None else BRONZE_ARTIFACTS + (RUN_MANIFEST_ARTIFACT,)
    return [
        ingest_artifact(store, artifact, run_id=run_id, job_id=job_id, now=now)
        for artifact in selected
    ]


# ---------------------------------------------------------------------------
# Daily universe snapshot (silver) - LD-05: first capture wins, never backfilled
# ---------------------------------------------------------------------------
UNIVERSE_LISTS: tuple[tuple[str, str, str], ...] = (
    ("universe_all", "UNIVERSE_ALL_FILE", "universe_builder"),
    ("longs", "LONGS_FILE", "watchlist_file"),
    ("shorts", "SHORTS_FILE", "watchlist_file"),
    ("autolongs", "AUTO_LONGS_FILE", "auto_populate"),
    ("autoshorts", "AUTO_SHORTS_FILE", "auto_populate"),
    ("swinglongs", "SWING_LONGS_FILE", "watchlist_file"),
    ("shortswings", "SWING_SHORTS_FILE", "watchlist_file"),
    ("focus", "FOCUS_LONGS_FILE", "focus_pick"),
    ("focus", "FOCUS_SHORTS_FILE", "focus_pick"),
)


def _read_symbol_file(path: Path) -> list[str]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    symbols: list[str] = []
    for line in text.splitlines():
        symbol = line.split("#", 1)[0].strip().upper()
        if symbol:
            symbols.append(symbol)
    return symbols


def load_exploration_cohort(path: Path | None = None) -> list[str]:
    """The fixed exploration list (sec 19.3). Empty until Aaron confirms it."""
    return _read_symbol_file(path or EXPLORATION_COHORT_FILE)


def snapshot_universe_membership(
    store: ResearchStore | None,
    *,
    session_date: date | None = None,
    run_id: str = "",
    job_id: str = "universe_snapshot",
    now: datetime | None = None,
    lists=None,
    exploration_path: Path | None = None,
) -> SnapshotReport:
    """One immutable daily membership snapshot per list (LD-05).

    Cohort membership is point-in-time: the snapshot is taken at first capture
    and never rewritten from a later day's files, because a watchlist edited at
    noon must not retroactively change who was a member this morning.
    """
    day = session_date or utc_now().date()
    report = SnapshotReport(dataset="universe_membership_daily", session_date=day.isoformat())
    if store is None:
        report.status = "DISABLED"
        return report

    existing = store.read_table(
        "universe_membership_daily",
        f"year={day.year}",
        columns=["session_date", "list_name"],
    )
    already = {
        (str(session), str(name))
        for session, name in zip(
            existing.column("session_date").to_pylist(),
            existing.column("list_name").to_pylist(),
        )
    }
    stamp = now or utc_now()
    rows: list[dict] = []
    sources = list(lists) if lists is not None else list(UNIVERSE_LISTS)
    for list_name, attr, reason in sources:
        if (day.isoformat(), list_name) in already:
            continue
        path = getattr(_paths(), attr, None)
        if path is None:
            continue
        symbols = _read_symbol_file(Path(path))
        if not symbols:
            continue
        report.sources.append(Path(path).name)
        for rank, symbol in enumerate(symbols, start=1):
            rows.append(
                {
                    "session_date": day,
                    "list_name": list_name,
                    "symbol": symbol,
                    "rank_in_list": rank,
                    "inclusion_reason": reason,
                    "snapshot_at": stamp,
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                }
            )
    exploration = load_exploration_cohort(exploration_path)
    if exploration and (day.isoformat(), "exploration_fixed") not in already:
        report.sources.append(EXPLORATION_COHORT_FILE.name)
        for rank, symbol in enumerate(exploration, start=1):
            rows.append(
                {
                    "session_date": day,
                    "list_name": "exploration_fixed",
                    "symbol": symbol,
                    "rank_in_list": rank,
                    "inclusion_reason": "exploration_cohort_file",
                    "snapshot_at": stamp,
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                }
            )
    if not rows:
        report.status = "ALREADY_CAPTURED" if already else "NO_SOURCE"
        return report
    result = store.publish("universe_membership_daily", rows, job_id=job_id)
    report.rows = result.rows_published
    return report


# ---------------------------------------------------------------------------
# Daily geometry snapshot (silver) - wrapped reads of the existing level stores
# ---------------------------------------------------------------------------
def _ensure_scripts_on_path() -> None:
    """The champion modules import each other as top-level names.

    Entry points (``launch_gui.py``, ``scripts/smoke_check.py``) already put
    ``scripts/`` on ``sys.path``; a warehouse job imported as a package may not
    have, and a wrapped read must not depend on who imported it first.
    """
    import sys

    scripts_dir = str(Path(__file__).resolve().parents[1])
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def _hv_store_levels(symbol: str, levels_dir: Path):
    """Wrapped read of the HV level store through its own loader."""
    _ensure_scripts_on_path()
    try:
        from master_avwap_lib.levels import level_store_path, load_level_store
    except ImportError:  # pragma: no cover - packaged import
        from scripts.master_avwap_lib.levels import level_store_path, load_level_store  # type: ignore
    path = level_store_path(Path(levels_dir), symbol)
    if not path.exists():
        return []
    payload = load_level_store(path, symbol)
    return [level for level in (payload.get("levels") or []) if isinstance(level, dict)]


def _d1_feed_levels(symbols: set[str]):
    """Wrapped read of ``d1_level_feed``'s own AI-state loader (SMA + trendline)."""
    _ensure_scripts_on_path()
    try:
        import d1_level_feed
    except ImportError:  # pragma: no cover - packaged import
        from scripts import d1_level_feed  # type: ignore
    loader = getattr(d1_level_feed, "_load_ai_state_feed", None)
    state_file = getattr(d1_level_feed, "MASTER_AVWAP_AI_STATE_FILE", None)
    if loader is None or state_file is None:
        return {}
    try:
        feed = loader(Path(state_file))
    except Exception:  # pragma: no cover - a malformed champion artifact
        return {}
    return {symbol: entry for symbol, entry in feed.items() if not symbols or symbol in symbols}


def _watch_json(path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _level_row(
    *,
    symbol: str,
    day: date,
    family: str,
    source_store: str,
    price,
    subtype: str,
    known_at: datetime,
    run_id: str,
    strength=None,
    touch_count=None,
    zone_low=None,
    zone_high=None,
    definition_version: str = "v1",
    is_active: bool = True,
    extra=None,
) -> dict:
    return {
        "symbol": symbol,
        "session_date": day,
        "level_id": level_id(symbol, source_store, family, subtype, price, extra),
        "level_family": family,
        "level_price": None if price is None else float(price),
        "zone_low": zone_low,
        "zone_high": zone_high,
        "source_timeframe": "D1",
        "source_store": source_store,
        "strength_score": strength,
        "touch_count": touch_count,
        "is_active": is_active,
        "definition_version": definition_version,
        "known_at": known_at,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }


def _parse_armed_at(value, fallback: datetime) -> datetime:
    text = str(value or "").strip()
    if not text:
        return fallback
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return fallback
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def snapshot_level_geometry(
    store: ResearchStore | None,
    *,
    session_date: date | None = None,
    symbols=None,
    run_id: str = "",
    job_id: str = "geometry_snapshot",
    now: datetime | None = None,
    levels_dir: Path | None = None,
) -> SnapshotReport:
    """Daily ``level_state_daily`` rows from the existing geometry sources.

    Sources, all wrapped reads: the HV level stores, ``d1_level_feed``'s SMA and
    trendline state, and the trader's own watch/alert JSONs. Human geometry is a
    historical fact - it is snapshotted from its arm time onward and never
    invented, edited, or removed here (sec 6.5, LD-08).
    """
    day = session_date or utc_now().date()
    report = SnapshotReport(dataset="level_state_daily", session_date=day.isoformat())
    if store is None:
        report.status = "DISABLED"
        return report

    existing = store.read_table("level_state_daily", f"year={day.year}", columns=["session_date", "level_id"])
    already = {
        (str(session), str(level))
        for session, level in zip(
            existing.column("session_date").to_pylist(),
            existing.column("level_id").to_pylist(),
        )
    }
    stamp = now or utc_now()
    paths = _paths()
    wanted = {str(symbol).strip().upper() for symbol in (symbols or [])}
    rows: list[dict] = []

    # 1. HV horizontal level stores (~1,539 symbols x 5 yr), reuse-as-is.
    levels_root = Path(levels_dir or getattr(paths, "MASTER_AVWAP_LEVELS_DIR", ""))
    if wanted and levels_root:
        for symbol in sorted(wanted):
            store_levels = _hv_store_levels(symbol, levels_root)
            if store_levels:
                report.sources.append("hv_level_store")
            for level in store_levels:
                price = level.get("price")
                if price in (None, ""):
                    continue
                effective = level.get("effective_range")
                rows.append(
                    _level_row(
                        symbol=symbol,
                        day=day,
                        family="HORIZONTAL_STORE",
                        source_store="hv_level_store",
                        price=price,
                        subtype=str(level.get("kind") or ""),
                        strength=_as_float(level.get("strength")),
                        touch_count=_as_int(level.get("touch_count")),
                        known_at=stamp,
                        run_id=run_id,
                        extra=effective if isinstance(effective, list) else None,
                    )
                )

    # 2. d1_level_feed: D1 SMA levels and projected trendline values.
    for symbol, entry in sorted(_d1_feed_levels(wanted).items()):
        for label, value in sorted((entry.get("smas") or {}).items()):
            rows.append(
                _level_row(
                    symbol=symbol,
                    day=day,
                    family="MA_LEVEL",
                    source_store="d1_level_feed",
                    price=value,
                    subtype=label,
                    known_at=stamp,
                    run_id=run_id,
                )
            )
        for index, value in enumerate(entry.get("trendlines") or []):
            # TRENDLINE rows carry the projected value for the session; full
            # geometry arrives with the post-slice trendline datasets.
            rows.append(
                _level_row(
                    symbol=symbol,
                    day=day,
                    family="TRENDLINE",
                    source_store="d1_level_feed",
                    price=value,
                    subtype=f"projection_{index}",
                    known_at=stamp,
                    run_id=run_id,
                )
            )
        if entry.get("smas") or entry.get("trendlines"):
            report.sources.append("d1_level_feed")

    # 3. Trader geometry: armed D1 level watches, chart watches, price alerts.
    for watch in _watch_json(getattr(paths, "D1_LEVEL_WATCHES_FILE", "")).get("watches") or []:
        if not isinstance(watch, dict):
            continue
        symbol = str(watch.get("symbol") or "").strip().upper()
        price = _as_float(watch.get("level"))
        if not symbol or price is None:
            continue
        report.sources.append("d1_level_watches.json")
        rows.append(
            _level_row(
                symbol=symbol,
                day=day,
                family="WATCH_JSON",
                source_store="d1_level_watches.json",
                price=price,
                subtype=str(watch.get("direction") or ""),
                known_at=_parse_armed_at(watch.get("armed_at"), stamp),
                run_id=run_id,
                extra=str(watch.get("candle_date") or ""),
            )
        )
    for watch in _watch_json(getattr(paths, "ALERT_CHART_WATCHES_FILE", "")).get("watches") or []:
        if not isinstance(watch, dict):
            continue
        symbol = str(watch.get("symbol") or "").strip().upper()
        price = _as_float(watch.get("baseline"))
        if not symbol or price is None:
            continue
        report.sources.append("alert_chart_watches.json")
        rows.append(
            _level_row(
                symbol=symbol,
                day=day,
                family="WATCH_JSON",
                source_store="alert_chart_watches.json",
                price=price,
                subtype=str(watch.get("kind") or ""),
                known_at=_parse_armed_at(watch.get("armed_at"), stamp),
                run_id=run_id,
            )
        )
    for entry in _watch_json(getattr(paths, "PRICE_ALERTS_FILE", "")).get("entries") or []:
        if not isinstance(entry, dict):
            continue
        symbol = str(entry.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        for side in ("above", "below"):
            price = _as_float(entry.get(side))
            if price is None:
                continue
            report.sources.append("price_alerts.json")
            rows.append(
                _level_row(
                    symbol=symbol,
                    day=day,
                    family="WATCH_JSON",
                    source_store="price_alerts.json",
                    price=price,
                    subtype=side,
                    known_at=stamp,
                    run_id=run_id,
                    is_active=bool(entry.get(f"armed_{side}")),
                )
            )

    report.sources = sorted(set(report.sources))
    fresh = [row for row in rows if (day.isoformat(), row["level_id"]) not in already]
    if not fresh:
        report.status = "ALREADY_CAPTURED" if rows else "NO_SOURCE"
        return report
    result = store.publish("level_state_daily", fresh, job_id=job_id)
    report.rows = result.rows_published
    return report


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Reuse-as-is: the durable per-symbol D1 Parquet store -> bar_d1 (wrapped read)
# ---------------------------------------------------------------------------
# The scanner's own writer keeps owning that store; nothing is copied and
# nothing is re-fetched. Only the sanitizer that turns a symbol into its file
# name is mirrored here (identical rule to
# ``master_avwap_lib.levels.level_store_path``), because importing the legacy
# scanner core for one path helper would drag the whole GUI stack into a
# headless build job.
_WINDOWS_RESERVED_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def durable_daily_bar_file(symbol: str, bars_dir: Path | None = None) -> Path:
    import re

    text = str(symbol or "").strip().upper()
    cleaned = re.sub(r"[^A-Z0-9._-]+", "_", text) or "UNKNOWN"
    root, sep, suffix = cleaned.partition(".")
    if root in _WINDOWS_RESERVED_STEMS:
        cleaned = f"{root}_{sep}{suffix}" if sep else f"{root}_"
    root_dir = Path(bars_dir or getattr(_paths(), "MASTER_AVWAP_DAILY_BARS_DIR", ""))
    return root_dir / f"{cleaned}.parquet"


def read_durable_daily_bars(symbol: str, bars_dir: Path | None = None):
    """The durable D1 frame for one symbol, or None when there is none."""
    path = durable_daily_bar_file(symbol, bars_dir)
    if not path.exists():
        return None
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
    except Exception:  # pragma: no cover - a corrupt legacy file is not fatal
        return None
    if frame is None or frame.empty or "datetime" not in frame.columns:
        return None
    return frame


def ingest_daily_bars(
    store: ResearchStore | None,
    symbols,
    *,
    as_of: date | None = None,
    exchange_calendar: str = "XNYS",
    provider: str = "UNKNOWN",
    bars_dir: Path | None = None,
    run_id: str = "",
    job_id: str = "d1_wrapped_read",
    now: datetime | None = None,
) -> SnapshotReport:
    """Project the durable D1 store into ``bar_d1`` (completed sessions only).

    Two rules are load-bearing:

    * **Completed bars only.** The durable store can hold the current session's
      forming bar, so anything dated on or after ``as_of`` is skipped and picked
      up on a later run. A forming bar is preview, never evidence.
    * **Provider honesty.** That store does not persist which provider produced
      a row, so provider is recorded as UNKNOWN rather than assumed to be IBKR.
      Missing provenance is uncertainty (sec 2); Phase 3's tee records the real
      provider going forward.

    Re-runs are idempotent: a (symbol, session) already present is not rewritten.
    """
    report = SnapshotReport(dataset="bar_d1")
    if store is None:
        report.status = "DISABLED"
        return report
    today = as_of or utc_now().date()
    stamp = now or utc_now()
    wanted = [str(symbol).strip().upper() for symbol in symbols or [] if str(symbol).strip()]
    if not wanted:
        report.status = "NO_SOURCE"
        return report

    existing = store.read_table("bar_d1", columns=["symbol", "session_date"])
    already = {
        (str(symbol), str(session))
        for symbol, session in zip(
            existing.column("symbol").to_pylist(), existing.column("session_date").to_pylist()
        )
    }
    rows: list[dict] = []
    for symbol in wanted:
        frame = read_durable_daily_bars(symbol, bars_dir)
        if frame is None:
            continue
        report.sources.append(symbol)
        for record in frame.to_dict("records"):
            stamp_value = record.get("datetime")
            session_day = _as_session_date(stamp_value)
            if session_day is None or session_day >= today:
                continue  # forming or unknown session: not completed evidence
            if (symbol, session_day.isoformat()) in already:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "session_id": f"{exchange_calendar}-{session_day.isoformat()}",
                    "session_date": session_day,
                    "open": _as_float(record.get("open")),
                    "high": _as_float(record.get("high")),
                    "low": _as_float(record.get("low")),
                    "close": _as_float(record.get("close")),
                    "volume": _as_int(record.get("volume")),
                    "adjustment_version": None,
                    "corporate_action_id": None,
                    "provider": provider,
                    "quality": QUALITY_COMPLETE,
                    "is_complete": True,
                    "event_at": datetime(
                        session_day.year, session_day.month, session_day.day, tzinfo=timezone.utc
                    ),
                    "observed_at": stamp,
                    "capture_mode": BRONZE_CAPTURE_MODE,
                    "revision_id": "",
                    "supersedes_revision_id": "",
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                }
            )
    if not rows:
        report.status = "ALREADY_CAPTURED" if already else "NO_SOURCE"
        return report
    result = store.publish("bar_d1", rows, job_id=job_id)
    report.rows = result.rows_published
    return report


def _as_session_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def run_daily_snapshots(
    store: ResearchStore | None,
    *,
    session_date: date | None = None,
    symbols=None,
    run_id: str = "",
    now: datetime | None = None,
) -> list[SnapshotReport]:
    if store is None:
        return []
    return [
        snapshot_universe_membership(store, session_date=session_date, run_id=run_id, now=now),
        snapshot_level_geometry(store, session_date=session_date, symbols=symbols, run_id=run_id, now=now),
    ]


def ingest_everything(
    store: ResearchStore | None = None,
    *,
    session_date: date | None = None,
    symbols=None,
    run_id: str = "",
    now: datetime | None = None,
) -> dict:
    """Phase-2 entry point: wrap every artifact and take the daily snapshots."""
    target = store if store is not None else ResearchStore.open()
    if target is None:  # warehouse disabled: a total no-op
        return {"enabled": False, "bronze": [], "snapshots": []}
    return {
        "enabled": True,
        "bronze": run_bronze_ingest(target, run_id=run_id, now=now),
        "snapshots": run_daily_snapshots(target, session_date=session_date, symbols=symbols, run_id=run_id, now=now),
    }


__all__ = [
    "BRONZE_ARTIFACTS",
    "CLASS_A_ARTIFACTS",
    "BronzeArtifact",
    "IngestReport",
    "SnapshotReport",
    "config",
    "durable_daily_bar_file",
    "ingest_artifact",
    "ingest_daily_bars",
    "ingest_everything",
    "read_durable_daily_bars",
    "load_exploration_cohort",
    "run_bronze_ingest",
    "run_daily_snapshots",
    "snapshot_level_geometry",
    "snapshot_universe_membership",
]
