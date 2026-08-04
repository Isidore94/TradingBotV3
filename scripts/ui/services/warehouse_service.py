"""Warehouse Health tiles and job registration (plan sec 18, Phase 8).

System Health gains exactly **six tiles**, no more (the six-tile cap is the
plan's own guard against dashboard sprawl):

1. DAS mount / free GB / 30-day growth;
2. backup age + last spot-restore date;
3. expected-vs-observed bar coverage per (resolution, cohort), worst-5 symbols;
4. inbox/spool backlog + oldest age + quarantine count;
5. last seal/import result;
6. manifest integrity - live files not in ``manifest_log.jsonl`` must be 0.

Every tile is computed from the ledger and cheap filesystem stats, never by
reading bar data, and the whole service degrades to a single "not configured"
tile when ``research_store_dir`` is unset. A green tile means the mechanics
worked; it never means a setup is predictive.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_RED = "RED"
STATUS_OFF = "OFF"

BACKUP_STALE_HOURS = 36
RESTORE_STALE_DAYS = 183  # semiannual spot restore (sec 8.5)


@dataclass
class HealthTile:
    key: str
    label: str
    value: str = "-"
    status: str = STATUS_OFF
    detail: str = ""
    metrics: dict = field(default_factory=dict)


def _warehouse():
    import sys

    scripts_dir = str(Path(__file__).resolve().parents[2])
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from research_warehouse import config, spool, store as store_module

    return config, spool, store_module


def _age_hours(stamp: datetime | None, now: datetime) -> float | None:
    if stamp is None:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (now - stamp).total_seconds() / 3600.0


def warehouse_health_tiles(
    store=None,
    *,
    now: datetime | None = None,
    backup_root: Path | None = None,
    last_restore_at: datetime | None = None,
) -> list[HealthTile]:
    """The six tiles. Cheap enough to refresh on the Health page's own cadence."""
    config, spool_module, store_module = _warehouse()
    moment = now or datetime.now(timezone.utc)
    target = store if store is not None else store_module.ResearchStore.open()
    if target is None:
        return [
            HealthTile(
                key="warehouse",
                label="Research warehouse",
                value="not configured",
                status=STATUS_OFF,
                detail="Set a research store directory to enable capture and these tiles.",
            )
        ]

    health = target.health_counts()
    tiles: list[HealthTile] = []

    # 1. DAS mount / free GB / growth
    try:
        usage = shutil.disk_usage(target.root)
        free_gb = usage.free / 1024**3
        lake_bytes = sum(path.stat().st_size for path in target.root.rglob("*.parquet"))
        tiles.append(
            HealthTile(
                key="das_mount",
                label="DAS mount / free",
                value=f"{free_gb:,.0f} GB free",
                status=STATUS_OK if free_gb > 50 else (STATUS_WARN if free_gb > 10 else STATUS_RED),
                detail=f"lake {lake_bytes / 1024**3:.2f} GB at {target.root}",
                metrics={"free_gb": round(free_gb, 2), "lake_gb": round(lake_bytes / 1024**3, 3)},
            )
        )
    except OSError as exc:
        tiles.append(
            HealthTile(
                key="das_mount",
                label="DAS mount / free",
                value="unreachable",
                status=STATUS_RED,
                detail=f"{target.root}: {exc}",
            )
        )

    # 2. Backup age + last spot restore
    backup_age = None
    if backup_root is not None and Path(backup_root).exists():
        stamps = [path.stat().st_mtime for path in Path(backup_root).rglob("*") if path.is_file()]
        if stamps:
            backup_age = _age_hours(datetime.fromtimestamp(max(stamps), tz=timezone.utc), moment)
    restore_age_days = None
    if last_restore_at is not None:
        restore_age_days = (_age_hours(last_restore_at, moment) or 0) / 24.0
    backup_status = STATUS_RED if backup_age is None else (STATUS_OK if backup_age <= BACKUP_STALE_HOURS else STATUS_WARN)
    if restore_age_days is not None and restore_age_days > RESTORE_STALE_DAYS:
        backup_status = STATUS_WARN
    tiles.append(
        HealthTile(
            key="backup",
            label="Backup age / last restore",
            value="never" if backup_age is None else f"{backup_age:.1f} h ago",
            status=backup_status,
            detail=(
                "no spot restore logged"
                if restore_age_days is None
                else f"last restore check {restore_age_days:.0f} days ago"
            ),
            metrics={"backup_age_hours": backup_age, "restore_age_days": restore_age_days},
        )
    )

    # 3. Coverage: expected vs observed, worst 5 symbols
    coverage = _coverage_tile(target, moment)
    tiles.append(coverage)

    # 4. Spool backlog + quarantine count
    try:
        writer_dir = Path(config.research_spool_dir())
        segments = sorted(writer_dir.glob("segment-*.jsonl")) if writer_dir.exists() else []
        oldest = None
        if segments:
            oldest = _age_hours(
                datetime.fromtimestamp(min(path.stat().st_mtime for path in segments), tz=timezone.utc), moment
            )
        backlog_bytes = sum(path.stat().st_size for path in segments)
        over_cap = backlog_bytes > spool_module.SPOOL_CAP_BYTES
        quarantine = int(health.get("quarantine_files") or 0)
        tiles.append(
            HealthTile(
                key="spool",
                label="Spool backlog / quarantine",
                value=f"{len(segments)} segment(s), {quarantine} quarantined",
                status=STATUS_RED if over_cap else (STATUS_WARN if quarantine or len(segments) > 20 else STATUS_OK),
                detail=(
                    f"{backlog_bytes / 1024**2:.1f} MB"
                    + (f", oldest {oldest:.1f} h" if oldest is not None else "")
                    + f", {health.get('quarantine_rows', 0)} quarantined row(s)"
                ),
                metrics={
                    "segments": len(segments),
                    "backlog_mb": round(backlog_bytes / 1024**2, 2),
                    "oldest_hours": oldest,
                    "quarantine_files": quarantine,
                },
            )
        )
    except OSError as exc:
        tiles.append(HealthTile(key="spool", label="Spool backlog / quarantine", value="unreadable", status=STATUS_RED, detail=str(exc)))

    # 5. Last seal / import
    seal_at = health.get("last_seal_at") or ""
    seal_age = None
    if seal_at:
        try:
            seal_age = _age_hours(datetime.fromisoformat(seal_at), moment)
        except ValueError:
            seal_age = None
    tiles.append(
        HealthTile(
            key="last_seal",
            label="Last seal / import",
            value=health.get("last_seal_dataset") or "never",
            status=STATUS_OK if seal_at else STATUS_WARN,
            detail=(f"{seal_age:.1f} h ago" if seal_age is not None else "no seal recorded yet")
            + f"; {health.get('live_files', 0)} live file(s), {health.get('live_rows', 0)} row(s)",
            metrics={"seal_age_hours": seal_age, "live_files": health.get("live_files")},
        )
    )

    # 6. Manifest integrity: live files not in the ledger must be zero
    unmanifested = int(health.get("unmanifested_live_files") or 0)
    missing = int(health.get("missing_live_files") or 0)
    tiles.append(
        HealthTile(
            key="manifest_integrity",
            label="Manifest integrity",
            value=f"{unmanifested} unmanifested, {missing} missing",
            status=STATUS_OK if unmanifested == 0 and missing == 0 else STATUS_RED,
            detail="_retired/ is expected GC lag and is not counted here.",
            metrics={"unmanifested": unmanifested, "missing": missing, "retired_pending": health.get("retired_pending")},
        )
    )
    return tiles


def _coverage_tile(store, moment: datetime) -> HealthTile:
    """Expected vs observed bars for the current month, worst five symbols."""
    partition = f"month={moment:%Y-%m}"
    try:
        import sys

        scripts_dir = str(Path(__file__).resolve().parents[2])
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from research_warehouse.backfill import open_gap_keys

        # Only gaps still open: a gap a later backfill filled is closed by a
        # superseding row (BD-60), and counting both would double it.
        gaps = list(open_gap_keys(store, [partition]).values())
    except Exception as exc:
        return HealthTile(key="coverage", label="Bar coverage", value="unreadable", status=STATUS_WARN, detail=str(exc))
    # Counted in *short sessions*, not bars: since D18, ``expected_bars`` holds
    # the count expected across the gap interval rather than the shortfall, so
    # summing it here would report the session size as if it were the loss.
    short_sessions: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for row in gaps:
        reason = str(row.get("reason") or "")
        by_reason[reason] = by_reason.get(reason, 0) + 1
        # Policy absence is not a coverage defect - it is a declared decision.
        if reason == "NOT_COLLECTED_BY_POLICY":
            continue
        symbol = str(row.get("symbol"))
        short_sessions[symbol] = short_sessions.get(symbol, 0) + 1
    worst = sorted(short_sessions.items(), key=lambda item: -item[1])[:5]
    return HealthTile(
        key="coverage",
        label="Bar coverage (expected vs observed)",
        value=f"{len(short_sessions)} symbol(s) short" if short_sessions else "complete",
        status=STATUS_OK if not short_sessions else (STATUS_WARN if len(short_sessions) <= 5 else STATUS_RED),
        detail=(
            "worst: " + ", ".join(f"{symbol} {count} session(s)" for symbol, count in worst)
        )
        if worst
        else "no open gaps recorded",
        metrics={"by_reason": by_reason, "worst_5": worst},
    )


class WarehouseTeeCapture:
    """The GUI-owned M5 tee (BD-20 wiring). Spool-only, zero provider cost.

    The contract this class exists to keep, from the 2026-08-04 review's design
    ruling and sec 8.4/LD-01:

    * **Nothing in ``bounce_bot_lib`` knows it exists.** The champion populates
      ``latest_bars`` for its own reasons; this reads that dict and nothing
      else. No request is issued, so the tee cannot fail or delay a champion
      fetch, and no champion module imports the warehouse.
    * **The snapshot happens on the owning thread.** ``capture()`` is called
      from a GUI-thread slot and takes ``dict(bot.latest_bars)`` immediately;
      ``extract_tee_bars`` iterating a dict another thread is resizing would
      raise, which is why the copy is here and not deeper.
    * **Spool-only.** ``store`` is deliberately ``None``, so ``capture_m5_tee``
      does no lake I/O at all on the GUI thread - not even the read that
      normally seeds its de-duplication. This object's own ``seen`` set does
      that instead, which is what ``capture_m5_tee(seen=...)`` is for, and the
      EOD build job seals the segments.
    * **Never fatal.** A capture failure is logged once per session and the
      desk carries on. Research evidence must never be able to break trading.
    """

    def __init__(self, *, spool=None, provider: str = "IBKR"):
        self._spool = spool
        self._provider = provider
        self._seen: set = set()
        self._seen_date = None
        self.last_report = None
        self.last_error = ""
        self.captures = 0
        self.rows_spooled = 0

    @property
    def spool(self):
        """The writer, created on first use so an unset lake stays a no-op."""
        if self._spool is None:
            _config, spool_module, _store_module = _warehouse()
            self._spool = spool_module.ResearchSpoolWriter()
        return self._spool

    def _session_seen(self, moment: datetime) -> set:
        """De-dup keys, reset per session so the set cannot grow unbounded."""
        day = moment.date()
        if day != self._seen_date:
            self._seen = set()
            self._seen_date = day
        return self._seen

    def capture(self, bot, *, now: datetime | None = None):
        """Snapshot the champion's M5 cache and spool whatever is new."""
        if bot is None:
            return None
        config, _spool_module, _store_module = _warehouse()
        if not config.warehouse_enabled():
            return None
        cache = getattr(bot, "latest_bars", None)
        if not cache:
            return None
        # The copy, on this (GUI) thread, before anything iterates it.
        snapshot = dict(cache)
        moment = now or datetime.now(timezone.utc)

        import sys

        scripts_dir = str(Path(__file__).resolve().parents[2])
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from research_warehouse.bar_archive import capture_m5_tee

        try:
            report = capture_m5_tee(
                None,  # spool-only: no lake I/O on the GUI thread
                snapshot,
                now=moment,
                provider=self._provider,
                spool=self.spool,
                seen=self._session_seen(moment),
            )
        except Exception as exc:  # research capture never breaks the desk
            if str(exc) != self.last_error:
                self.last_error = str(exc)
                logging.exception("Warehouse M5 tee capture failed; the desk is unaffected.")
            return None
        self.last_error = ""
        self.last_report = report
        self.captures += 1
        self.rows_spooled += int(getattr(report, "rows_published", 0) or 0)
        return report


def register_build_job(scheduler=None) -> dict:
    """Declare the build job for the existing scheduler/job ledger.

    Returns the registration descriptor rather than starting anything: the
    warehouse adds no process, daemon, or timer of its own (sec 8.4).
    """
    descriptor = {
        "job_type": "research_warehouse_build",
        "entry_point": "python -m scripts.research_warehouse.cli build",
        "cadence": "post_scan_and_eod",
        "single_flight": True,
        "owner": "main_desktop",
    }
    register = getattr(scheduler, "register_job", None)
    if callable(register):
        register(descriptor)
    return descriptor


__all__ = [
    "BACKUP_STALE_HOURS",
    "RESTORE_STALE_DAYS",
    "STATUS_OFF",
    "STATUS_OK",
    "STATUS_RED",
    "STATUS_WARN",
    "HealthTile",
    "WarehouseTeeCapture",
    "register_build_job",
    "warehouse_health_tiles",
]
