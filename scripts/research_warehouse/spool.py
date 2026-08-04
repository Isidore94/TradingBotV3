"""The machine-local capture spool (plan sec 8.4).

Live capture never writes the lake directly. The GUI owns one
:class:`ResearchSpoolWriter` that appends rows to a segment under
``%LOCALAPPDATA%\\TradingBotV3\\research_spool``; the post-scan/EOD CLI build
job seals **closed** segments into the lake through the 4-step protocol. That
split is the whole ownership contract:

* the writer only ever touches its ``.open`` segment;
* the sealer only ever touches ``.closed`` segments;
* a segment is deleted only after its manifest line has landed, so a crash
  mid-seal re-seals rather than loses.

It is also the DAS-unavailable answer: when the lake is unreachable the desk
keeps capturing into the spool, capped at 5 GB / 7 days. Over the cap, capture
sheds in a fixed order - M1 exploration extras first, then non-Focus M1, then
ETH bars - and **D1/M5 capture is never shed**, nor is anything operational.
Shedding is not silent deletion: every shed segment writes a shed-log record so
the CLI can turn it into an explicit ``collection_gap`` row when the lake
returns. Evidence that was dropped is recorded as dropped.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:  # package import
    from . import config
    from .manifest import utc_now
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import config  # type: ignore
    from manifest import utc_now  # type: ignore
    from store import ResearchStore  # type: ignore

OPEN_SUFFIX = ".open.jsonl"
CLOSED_SUFFIX = ".closed.jsonl"
SHED_LOG_NAME = "shed_log.jsonl"

# Locked in sec 8.4; not tuning knobs.
SPOOL_CAP_BYTES = 5 * 1024**3
SPOOL_MAX_AGE_DAYS = 7
SEGMENT_MAX_BYTES = 64 * 1024**2
SEGMENT_MAX_AGE_SECONDS = 3600

# Shed classes, most-droppable first. PROTECTED never sheds: D1/M5 capture and
# every operational champion artifact carry it.
SHED_M1_EXPLORATION = "M1_EXPLORATION"
SHED_M1_NON_FOCUS = "M1_NON_FOCUS"
SHED_ETH_BARS = "ETH_BARS"
SHED_PROTECTED = "PROTECTED"
SHED_ORDER = (SHED_M1_EXPLORATION, SHED_M1_NON_FOCUS, SHED_ETH_BARS)


@dataclass
class SpoolStats:
    segments: int = 0
    open_segments: int = 0
    closed_segments: int = 0
    bytes: int = 0
    oldest_age_seconds: float = 0.0
    shed_records: int = 0


@dataclass
class SealResult:
    segments_sealed: int = 0
    rows_published: int = 0
    rows_quarantined: int = 0
    gaps_recorded: int = 0
    segments_failed: list = field(default_factory=list)
    status: str = "OK"  # OK | DISABLED | NOTHING_TO_SEAL


def _json_default(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"spooled timestamps must be timezone-aware: {value!r}")
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class ResearchSpoolWriter:
    """The single live writer. One instance, GUI-owned (sec 8.4, LD-01)."""

    def __init__(
        self,
        spool_dir: Path | None = None,
        *,
        segment_max_bytes: int = SEGMENT_MAX_BYTES,
        segment_max_age_seconds: int = SEGMENT_MAX_AGE_SECONDS,
        cap_bytes: int = SPOOL_CAP_BYTES,
        max_age_days: int = SPOOL_MAX_AGE_DAYS,
    ):
        self.dir = Path(spool_dir) if spool_dir is not None else config.research_spool_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.segment_max_bytes = int(segment_max_bytes)
        self.segment_max_age_seconds = int(segment_max_age_seconds)
        self.cap_bytes = int(cap_bytes)
        self.max_age_days = int(max_age_days)
        self._active: Path | None = None
        self._opened_at: datetime | None = None
        # A previous writer that died leaves an .open segment. It belongs to
        # the writer role, never to the sealer, so this instance closes it -
        # that is what keeps writer and sealer off the same file.
        self.adopt_stale_segments()

    # -- segments -----------------------------------------------------------
    def adopt_stale_segments(self) -> list[Path]:
        adopted = []
        for path in sorted(self.dir.glob(f"*{OPEN_SUFFIX}")):
            adopted.append(self._close_segment(path))
        return adopted

    def _close_segment(self, path: Path) -> Path:
        closed = path.with_name(path.name[: -len(OPEN_SUFFIX)] + CLOSED_SUFFIX)
        os.replace(path, closed)
        return closed

    def _segment(self, now: datetime) -> Path:
        if self._active is not None and self._active.exists():
            too_big = self._active.stat().st_size >= self.segment_max_bytes
            too_old = (
                self._opened_at is not None
                and (now - self._opened_at).total_seconds() >= self.segment_max_age_seconds
            )
            if not (too_big or too_old):
                return self._active
            self.roll()
        self._active = self.dir / f"segment-{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}{OPEN_SUFFIX}"
        self._opened_at = now
        self._active.touch()
        return self._active

    def roll(self) -> Path | None:
        """Close the active segment so the CLI may seal it."""
        if self._active is None or not self._active.exists():
            self._active = None
            return None
        closed = self._close_segment(self._active)
        self._active = None
        self._opened_at = None
        return closed

    # -- writing ------------------------------------------------------------
    def write(
        self,
        dataset: str,
        rows,
        *,
        shed_class: str = SHED_PROTECTED,
        now: datetime | None = None,
    ) -> int:
        """Append rows for one dataset. Returns how many were spooled."""
        records = list(rows or [])
        if not records:
            return 0
        stamp = now or utc_now()
        self.enforce_cap(now=stamp)
        segment = self._segment(stamp)
        payload = "".join(
            json.dumps({"dataset": dataset, "shed_class": shed_class, "row": row}, default=_json_default) + "\n"
            for row in records
        )
        with open(segment, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return len(records)

    # -- cap, age, shedding -------------------------------------------------
    def stats(self, *, now: datetime | None = None) -> SpoolStats:
        stamp = now or utc_now()
        stats = SpoolStats()
        for path in self.dir.glob("segment-*.jsonl"):
            info = path.stat()
            stats.segments += 1
            stats.bytes += info.st_size
            if path.name.endswith(OPEN_SUFFIX):
                stats.open_segments += 1
            else:
                stats.closed_segments += 1
            age = stamp.timestamp() - info.st_mtime
            stats.oldest_age_seconds = max(stats.oldest_age_seconds, age)
        shed_log = self.dir / SHED_LOG_NAME
        if shed_log.exists():
            stats.shed_records = sum(1 for line in shed_log.read_text(encoding="utf-8").splitlines() if line.strip())
        return stats

    def enforce_cap(self, *, now: datetime | None = None) -> list[dict]:
        """Shed in the fixed order, and record every shed as evidence."""
        stamp = now or utc_now()
        shed: list[dict] = []
        cutoff = stamp - timedelta(days=self.max_age_days)
        for path in sorted(self.dir.glob(f"*{CLOSED_SUFFIX}")):
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < cutoff:
                shed.extend(self._shed_segment(path, reason="AGE_CAP", now=stamp))
        for shed_class in SHED_ORDER:
            if self.stats(now=stamp).bytes <= self.cap_bytes:
                break
            for path in sorted(self.dir.glob(f"*{CLOSED_SUFFIX}")):
                if self.stats(now=stamp).bytes <= self.cap_bytes:
                    break
                if self._segment_shed_class(path) == shed_class:
                    shed.extend(self._shed_segment(path, reason="SIZE_CAP", now=stamp))
        return shed

    def _segment_shed_class(self, path: Path) -> str:
        """A segment sheds only if EVERY record in it is that droppable."""
        classes = set()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        classes.add(str(json.loads(line).get("shed_class") or SHED_PROTECTED))
                    except ValueError:
                        classes.add(SHED_PROTECTED)
        except OSError:
            return SHED_PROTECTED
        if len(classes) == 1:
            return classes.pop()
        return SHED_PROTECTED

    def _shed_segment(self, path: Path, *, reason: str, now: datetime) -> list[dict]:
        """Drop one segment, leaving a record of exactly what was dropped."""
        summary: dict[tuple[str, str], dict] = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    row = record.get("row") or {}
                    key = (str(record.get("dataset") or ""), str(row.get("symbol") or ""))
                    entry = summary.setdefault(
                        key,
                        {
                            "dataset": key[0],
                            "symbol": key[1],
                            "timeframe": str(row.get("timeframe") or _timeframe_of(key[0])),
                            "rows": 0,
                            "first_ts": str(row.get("interval_start") or row.get("event_at") or ""),
                            "last_ts": "",
                            "shed_class": str(record.get("shed_class") or SHED_PROTECTED),
                            "reason": reason,
                            "shed_at": now.astimezone(timezone.utc).isoformat(),
                            "segment": path.name,
                        },
                    )
                    entry["rows"] += 1
                    entry["last_ts"] = str(row.get("interval_start") or row.get("event_at") or entry["last_ts"])
        except OSError:
            return []
        records = list(summary.values())
        with open(self.dir / SHED_LOG_NAME, "a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        path.unlink(missing_ok=True)
        return records


def _timeframe_of(dataset: str) -> str:
    if dataset.endswith("_m1"):
        return "M1"
    if dataset.endswith("_m5"):
        return "M5"
    if dataset.endswith("_d1"):
        return "D1"
    return ""


def closed_segments(spool_dir: Path) -> list[Path]:
    return sorted(Path(spool_dir).glob(f"*{CLOSED_SUFFIX}"))


def seal_spool(
    store: ResearchStore | None,
    spool_dir: Path | None = None,
    *,
    job_id: str = "spool_seal",
    delete_after_seal: bool = True,
) -> SealResult:
    """Seal every CLOSED spool segment into the lake, then drop it.

    The active ``.open`` segment is never touched: it belongs to the writer.
    A segment is unlinked only after its rows are sealed and their manifest
    lines exist, so an interrupted seal re-seals on the next run (publishes are
    idempotent per dataset only insofar as the caller made them so, which is
    why the tee de-duplicates before spooling).
    """
    result = SealResult()
    if store is None:
        result.status = "DISABLED"
        return result
    spool = Path(spool_dir) if spool_dir is not None else config.research_spool_dir()
    segments = closed_segments(spool) if spool.exists() else []
    shed_log = spool / SHED_LOG_NAME
    if not segments and not shed_log.exists():
        result.status = "NOTHING_TO_SEAL"
        return result

    for segment in segments:
        grouped: dict[str, list[dict]] = {}
        try:
            with open(segment, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue  # a torn tail line: the row never completed
                    dataset = str(record.get("dataset") or "")
                    row = record.get("row")
                    if dataset and isinstance(row, dict):
                        grouped.setdefault(dataset, []).append(row)
        except OSError:
            result.segments_failed.append(segment.name)
            continue
        sealed_ok = True
        for dataset, rows in sorted(grouped.items()):
            try:
                published = store.publish(dataset, rows, job_id=job_id, extra={"spool_segment": segment.name})
            except Exception:
                sealed_ok = False
                result.segments_failed.append(segment.name)
                break
            result.rows_published += published.rows_published
            result.rows_quarantined += published.rows_quarantined
        if sealed_ok:
            result.segments_sealed += 1
            if delete_after_seal:
                segment.unlink(missing_ok=True)

    result.gaps_recorded = _seal_shed_log(store, shed_log, job_id=job_id)
    return result


def _seal_shed_log(store: ResearchStore, shed_log: Path, *, job_id: str) -> int:
    """Turn shed records into explicit ``collection_gap`` rows."""
    if not shed_log.exists():
        return 0
    try:
        from .schemas import SCHEMA_VERSION
    except ImportError:  # pragma: no cover - scripts/ on sys.path
        from schemas import SCHEMA_VERSION  # type: ignore

    rows = []
    for line in shed_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        rows.append(
            {
                "symbol": str(record.get("symbol") or ""),
                "timeframe": str(record.get("timeframe") or ""),
                "gap_start": record.get("first_ts") or record.get("shed_at"),
                "gap_end": record.get("last_ts") or record.get("shed_at"),
                "expected_bars": int(record.get("rows") or 0),
                # The spool shed this evidence under its declared policy; that
                # is policy absence, not missing data.
                "reason": "NOT_COLLECTED_BY_POLICY",
                "detected_at": record.get("shed_at"),
                "resolved_at": None,
                "resolution": "POLICY",
                "schema_version": SCHEMA_VERSION,
                "run_id": job_id,
            }
        )
    if not rows:
        return 0
    published = store.publish("collection_gap", rows, job_id=job_id)
    shed_log.unlink(missing_ok=True)
    return published.rows_published


__all__ = [
    "CLOSED_SUFFIX",
    "OPEN_SUFFIX",
    "SEGMENT_MAX_AGE_SECONDS",
    "SEGMENT_MAX_BYTES",
    "SHED_ETH_BARS",
    "SHED_M1_EXPLORATION",
    "SHED_M1_NON_FOCUS",
    "SHED_ORDER",
    "SHED_PROTECTED",
    "SPOOL_CAP_BYTES",
    "SPOOL_MAX_AGE_DAYS",
    "ResearchSpoolWriter",
    "SealResult",
    "SpoolStats",
    "closed_segments",
    "seal_spool",
]
