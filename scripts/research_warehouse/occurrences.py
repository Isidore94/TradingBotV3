"""Detector-output ingestion and occurrence identity (plan Phase 6, sec 7.3).

**The warehouse never re-detects.** Everything here records what a champion
detector reported: its status, its trigger, its geometry, its version. If the
detector says a setup is ELIGIBLE, that is what the row says, and no code in
this module has an opinion about whether it should have.

The one piece of real logic is identity. The occurrence key is deterministic
(``schemas.occurrence_id``), so an hourly rescan of a live thesis recomputes
the *same* key and updates the snapshot instead of appending a second episode -
the tracker episode-dedup lesson, and risk R9. In an append-only lake "update"
means a new **revision** of the same ``occurrence_id``: the revision chain
records what changed and when it became known, while the episode count stays
one. A rescan that changes nothing writes nothing at all.

Identity rules this enforces (sec 7.3):

* long and short theses on one symbol are different occurrences;
* swing and intraday theses are different occurrences;
* two anchors on one symbol are different occurrences;
* variants of one underlying move share a ``dependency_cluster_id``, which is
  the episode unit that evidence floors count - not the row count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

try:  # package import
    from .manifest import utc_now
    from .schemas import SCHEMA_VERSION, SIDES, _identity_hash, occurrence_id
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    from manifest import utc_now  # type: ignore
    from schemas import SCHEMA_VERSION, SIDES, _identity_hash, occurrence_id  # type: ignore
    from store import ResearchStore  # type: ignore

#: The two slice setups, canonical IDs verbatim from setup_tagging.py.
SLICE_SETUPS = {
    "AVWAPE_TO_FIRST_DEV": "LONG",
    "POST_EARNINGS_CANDLE_BREAK": "SHORT",
}

#: Mutable snapshot columns: a change in any of them is a new revision.
SNAPSHOT_FIELDS = (
    "status",
    "trigger_at",
    "trigger_bar_interval_start",
    "entry_price_ref",
    "stop_price_ref",
    "detector_version",
    "trigger_timeframe",
    "tags",
    "dependency_cluster_id",
    "anchor_instance_id",
)


@dataclass
class OccurrenceReport:
    dataset: str = "setup_occurrence"
    status: str = "OK"  # OK | DISABLED | NOTHING_TO_RECORD
    created: int = 0
    revised: int = 0
    unchanged: int = 0
    rows: int = 0
    skipped: dict = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def dependency_cluster_id(symbol: str, side: str, structural_timeframe: str, episode_key) -> str:
    """The outcome-blind episode identity: one underlying move.

    Setup family is deliberately NOT an input - simultaneous AVWAP, band, MA
    and level variants on the same move are several hypotheses about one
    episode, and evidence floors count episodes. Side is an input, because a
    long thesis and a short thesis are not the same move.
    """
    return _identity_hash(str(symbol).upper(), str(side).upper(), str(structural_timeframe).upper(), episode_key)


def _episode_key(detected: dict):
    """Anchor instance when there is one, else the declared episode window."""
    return detected.get("anchor_instance_id") or detected.get("episode_start")


def _as_datetime(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def build_occurrence_row(detected: dict, *, run_id: str = "", now: datetime | None = None) -> dict | None:
    """Shape one detector report into a ``setup_occurrence`` row (revision 1)."""
    symbol = str(detected.get("symbol") or "").strip().upper()
    setup_id = str(detected.get("canonical_setup_id") or "").strip()
    side = str(detected.get("side") or "").strip().upper()
    structural = str(detected.get("structural_timeframe") or "").strip().upper()
    if not symbol or not setup_id or side not in SIDES or not structural:
        return None
    stamp = now or utc_now()
    episode = _episode_key(detected)
    if episode in (None, ""):
        # Neither an anchor instance nor a declared episode window. Hashing the
        # empty token would give a March thesis and a November thesis on the
        # same (symbol, setup, side, timeframe) one occurrence_id and one
        # episode, forever (review defect D16). The detector adapter (BD-44)
        # must supply one of the two; absence is rejected, never guessed.
        return None
    identity = occurrence_id(symbol, setup_id, side, structural, episode)
    return {
        "occurrence_id": identity,
        "symbol": symbol,
        "canonical_setup_id": setup_id,
        "side": side,
        "structural_timeframe": structural,
        "trigger_timeframe": str(detected.get("trigger_timeframe") or "").upper(),
        "anchor_instance_id": detected.get("anchor_instance_id"),
        "dependency_cluster_id": dependency_cluster_id(symbol, side, structural, episode),
        # Detector lifecycle state as reported - never re-derived here.
        "status": str(detected.get("status") or ""),
        "trigger_at": _as_datetime(detected.get("trigger_at")),
        "trigger_bar_interval_start": _as_datetime(detected.get("trigger_bar_interval_start")),
        "entry_price_ref": _float(detected.get("entry_price_ref")),
        "stop_price_ref": _float(detected.get("stop_price_ref")),
        "detector_version": str(detected.get("detector_version") or ""),
        "first_detected_run_id": str(detected.get("run_id") or run_id),
        "last_updated_run_id": str(detected.get("run_id") or run_id),
        "tags": str(detected.get("tags") or ""),
        "event_at": _as_datetime(detected.get("event_at")) or _as_datetime(detected.get("trigger_at")) or stamp,
        "observed_at": _as_datetime(detected.get("observed_at")) or stamp,
        "computed_at": stamp,
        "revision_id": "rev-1",
        "supersedes_revision_id": "",
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _revision_number(revision_id: str) -> int:
    text = str(revision_id or "")
    if text.startswith("rev-"):
        try:
            return int(text[4:])
        except ValueError:
            return 0
    return 0


def latest_occurrences(store: ResearchStore, year: int, occurrence_ids=None, *, span_years: int = 1) -> dict:
    """The current view: the highest revision of each occurrence.

    Reads ``year`` and the ``span_years`` partitions on either side of it. The
    dataset partitions on ``event_at``, so a December occurrence rescanned in
    January resolves against a *different* partition than the one holding its
    rev-1: looking at one year appended a second rev-1 and inflated the episode
    count at every year boundary (review defect D15).
    """
    wanted = set(occurrence_ids or [])
    latest: dict[str, dict] = {}
    years = range(int(year) - int(span_years), int(year) + int(span_years) + 1)
    for row in _rows_across_years(store, years):
        identity = str(row.get("occurrence_id") or "")
        if wanted and identity not in wanted:
            continue
        current = latest.get(identity)
        if current is None or _revision_number(row.get("revision_id")) > _revision_number(current.get("revision_id")):
            latest[identity] = row
    return latest


def _rows_across_years(store: ResearchStore, years):
    for value in years:
        yield from store.read_table("setup_occurrence", f"year={int(value)}").to_pylist()


def record_occurrences(
    store: ResearchStore | None,
    detected_rows,
    *,
    year: int | None = None,
    run_id: str = "",
    job_id: str = "setup_occurrence",
    now: datetime | None = None,
) -> OccurrenceReport:
    """Record detector output. A rescan updates; it never appends an episode.

    Returns counts of created / revised / unchanged occurrences, which is the
    evidence that repeated scans cannot inflate the denominator.
    """
    report = OccurrenceReport()
    if store is None:
        report.status = "DISABLED"
        return report
    stamp = now or utc_now()
    candidates = []
    for detected in detected_rows or []:
        row = build_occurrence_row(detected, run_id=run_id, now=stamp)
        if row is None:
            report.skip("INCOMPLETE_DETECTION")
            continue
        candidates.append(row)
    if not candidates:
        report.status = "NOTHING_TO_RECORD"
        return report

    target_year = year or candidates[0]["event_at"].year
    existing = latest_occurrences(store, target_year, {row["occurrence_id"] for row in candidates})

    rows = []
    for row in candidates:
        previous = existing.get(row["occurrence_id"])
        if previous is None:
            rows.append(row)
            report.created += 1
            continue
        if all(_same(previous.get(field_name), row.get(field_name)) for field_name in SNAPSHOT_FIELDS):
            # An hourly rescan that found nothing new writes nothing at all.
            report.unchanged += 1
            continue
        revision = _revision_number(previous.get("revision_id")) + 1
        row["revision_id"] = f"rev-{revision}"
        row["supersedes_revision_id"] = str(previous.get("revision_id") or "")
        # The first sighting is a historical fact and is carried forward.
        row["first_detected_run_id"] = str(previous.get("first_detected_run_id") or row["first_detected_run_id"])
        row["event_at"] = previous.get("event_at") or row["event_at"]
        rows.append(row)
        report.revised += 1

    if not rows:
        report.status = "NOTHING_TO_RECORD"
        return report
    report.rows = store.publish("setup_occurrence", rows, job_id=job_id).rows_published
    return report


def _same(left, right) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right) < 1e-12
    if isinstance(left, datetime) and isinstance(right, datetime):
        return left.astimezone(timezone.utc) == right.astimezone(timezone.utc)
    if left in (None, "") and right in (None, ""):
        return True
    return left == right


def episode_counts(store: ResearchStore, year: int) -> dict:
    """Occurrences and independent episodes - never the raw row count.

    Evidence floors count ``dependency_cluster_id`` values. Reporting rows
    instead is exactly how rescans and variants inflate a sample.
    """
    latest = latest_occurrences(store, year)
    return {
        "rows": store.read_table("setup_occurrence", f"year={year}").num_rows,
        "occurrences": len(latest),
        "episodes": len({row.get("dependency_cluster_id") for row in latest.values()}),
        "symbols": len({row.get("symbol") for row in latest.values()}),
    }


def link_bounce_events(occurrence_rows, bounce_events, *, window_minutes: int = 60) -> dict:
    """Join occurrences to BounceBot M5 bounce events (sec 19.3 mapping).

    ``intraday_bounce_v1`` is evaluated **only** where such a link exists: the
    bounce event supplies the bounce bar and bounce type. Where no linked event
    exists, no intraday row is produced - the warehouse never re-detects a
    bounce to manufacture one.
    """
    by_key: dict[tuple[str, date], list[dict]] = {}
    for event in bounce_events or []:
        symbol = str(event.get("symbol") or "").strip().upper()
        stamp = _as_datetime(event.get("bounce_at") or event.get("interval_start"))
        if not symbol or stamp is None:
            continue
        by_key.setdefault((symbol, stamp.date()), []).append({**event, "_at": stamp})

    linked: dict[str, dict] = {}
    for row in occurrence_rows or []:
        trigger = _as_datetime(row.get("trigger_at"))
        symbol = str(row.get("symbol") or "").upper()
        if trigger is None:
            continue
        for event in by_key.get((symbol, trigger.date()), []):
            delta = abs((event["_at"] - trigger).total_seconds()) / 60.0
            if delta <= window_minutes:
                current = linked.get(row["occurrence_id"])
                if current is None or delta < current["_delta"]:
                    linked[row["occurrence_id"]] = {**event, "_delta": delta}
    return linked


__all__ = [
    "SLICE_SETUPS",
    "SNAPSHOT_FIELDS",
    "OccurrenceReport",
    "build_occurrence_row",
    "dependency_cluster_id",
    "episode_counts",
    "latest_occurrences",
    "link_bounce_events",
    "record_occurrences",
]
