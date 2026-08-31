#!/usr/bin/env python3
"""Dated snapshot backup of the evidence the cold push deliberately excludes.

plan.md Phase 0.7 / R10.A, and ground rule 4's backup-and-restore contract for
every store R10 creates.

**Two jobs, two scopes. Do not merge them.**

* ``push_cold_to_das.ps1`` mirrors the COLD, append-only subtrees
  (``data\\daily_bars``, ``data\\intraday_bars``, ``output``, ``logs``,
  ``away_report_archive``) to the DAS hourly, incrementally, forever. Nothing
  there is ever deleted and nothing is dated.
* **This** takes a DATED SNAPSHOT of the HOT state the cold push excludes on
  purpose - ``data\\runtime`` (3.5 GB: the 960 MB setup tracker, the 203 MB
  outcome CSV, the journal SQLite, every outcome / cohort / focus store), the
  home-root evidence files, ``_tools``, and the machine-local diagnostics tree
  (529 MB of run manifests, shadow JSONL and stall evidence). Those are
  rewritten constantly, so decision 0015 stands and they stay on the local SSD:
  this **copies**, it never moves.

Trader, 2026-08-22: *"Any and all very important files that we use occasionally
should go to the server with the massive HDD."*

Local staging first, DAS second, exactly like the cold push: a share that is
unreachable exits 0 and leaves the staged snapshot on disk, which is the
intended fallback and not an error.

Nothing here reads or writes a detector, score, alert, watchlist or Focus store.
It only copies files and records what it copied.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

MANIFEST_SCHEMA = "evidence_snapshot_manifest_v1"

# A file at or above this size is gzipped into the snapshot. The two tracker
# JSONs are ~960 MB and ~939 MB and the technical-integrity log is ~466 MB;
# copying them raw is ~2 GB a night, or 60 GB a month, for files whose daily
# diff R10.D will carry anyway.
COMPRESS_MIN_BYTES = 64 * 1024 * 1024
# A file at or above this size must hold a steady size and mtime across the
# stability window before it is copied. A ~1 GB atomic replace caught mid-write
# produces a snapshot that restores to nothing, and a torn 960 MB JSON is
# indistinguishable from a good one until the day you need it.
STABILITY_MIN_BYTES = 256 * 1024 * 1024
STABILITY_WINDOW_SECONDS = 60.0
STABILITY_POLL_SECONDS = 5.0

# Retention. `evidence_frozen/` is never touched by pruning.
KEEP_DAILY = 7
KEEP_WEEKLY = 4
KEEP_MONTHLY = 12
FROZEN_DIR_NAME = "evidence_frozen"

# Files excluded by an explicit rule rather than a silent skip (trader,
# 2026-08-22). The setup tracker rotates its `.bak` on every save, so once this
# runs nightly, day N's main IS day N+1's `.bak` - 133 MB a night of the same
# bytes under a different name. The on-disk `.bak` is NEVER deleted: the tracker
# reads it back when the main payload is corrupt.
ROTATED_DUPLICATE_SUFFIXES = (".bak",)
EXCLUDED_ROTATED_REASON = "excluded_rotated_duplicate"


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScopeItem:
    """One thing to snapshot, and where it lands inside the snapshot."""

    label: str
    source: Path
    #: "tree" copies recursively; "files" copies only the immediate files, which
    #: is how the home root is taken without dragging its subdirectories in
    #: (they are the cold push's scope, not this one).
    mode: str = "tree"


def default_scope() -> list[ScopeItem]:
    from project_paths import PERSISTENT_DATA_DIR, get_diagnostics_dir

    home = Path(PERSISTENT_DATA_DIR)
    return [
        ScopeItem("data-runtime", home / "data" / "runtime", "tree"),
        ScopeItem("home-root", home, "files"),
        ScopeItem("tools", home / "_tools", "tree"),
        ScopeItem("diagnostics", Path(get_diagnostics_dir()), "tree"),
    ]


@dataclass
class FileRecord:
    label: str
    relative: str
    source_bytes: int
    stored_bytes: int = 0
    #: SHA-256 of the STORED bytes - proves the archive is intact. `verify()`
    #: checks this one, because it is cheap: no decompression.
    sha256: str = ""
    #: SHA-256 of the SOURCE bytes - the only thing that proves the CONTENT
    #: survived compression, and the only hash a restored file can be compared
    #: against. Both are recorded; neither substitutes for the other.
    source_sha256: str = ""
    compressed: bool = False
    method: str = "copy"
    skipped: str = ""

    def as_row(self) -> dict:
        row = {
            "label": self.label,
            "path": self.relative,
            "source_bytes": self.source_bytes,
            "stored_bytes": self.stored_bytes,
            "sha256": self.sha256,
            "source_sha256": self.source_sha256,
            "compressed": self.compressed,
            "method": self.method,
        }
        if self.skipped:
            row["skipped"] = self.skipped
        return row


@dataclass
class SnapshotResult:
    snapshot_date: str
    staging: Path
    records: list[FileRecord] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    @property
    def copied(self) -> list[FileRecord]:
        return [r for r in self.records if not r.skipped]

    @property
    def skipped(self) -> list[FileRecord]:
        return [r for r in self.records if r.skipped]

    def manifest(self) -> dict:
        skipped_by_reason: dict[str, int] = {}
        for record in self.skipped:
            skipped_by_reason[record.skipped] = skipped_by_reason.get(record.skipped, 0) + 1
        return {
            "schema": MANIFEST_SCHEMA,
            "snapshot_date": self.snapshot_date,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "files": len(self.copied),
            "source_bytes": sum(r.source_bytes for r in self.copied),
            "stored_bytes": sum(r.stored_bytes for r in self.copied),
            # Skipped files are counted and named, never dropped: a snapshot
            # that quietly omits the 960 MB tracker looks identical to one that
            # captured it (plan.md sec 5 - missing data is uncertainty).
            "skipped": len(self.skipped),
            "skipped_by_reason": skipped_by_reason,
            "entries": [r.as_row() for r in self.records],
        }


# ---------------------------------------------------------------------------
# copy-while-hot
# ---------------------------------------------------------------------------
def is_stable(path: Path, *, window: float = STABILITY_WINDOW_SECONDS,
              poll: float = STABILITY_POLL_SECONDS, sleep=time.sleep) -> bool:
    """Has ``path`` held one size and mtime across the stability window?

    Only asked of large files. ``sleep`` is injected so a test does not wait a
    real minute to prove the rule.
    """
    try:
        first = path.stat()
    except OSError:
        return False
    waited = 0.0
    while waited < window:
        sleep(poll)
        waited += poll
        try:
            now = path.stat()
        except OSError:
            return False
        if (now.st_size, now.st_mtime_ns) != (first.st_size, first.st_mtime_ns):
            return False
        first = now
    return True


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_sqlite(source: Path, target: Path) -> None:
    """Copy a live SQLite database through the backup API.

    A byte copy of an open database can catch it mid-transaction. The backup API
    is the documented way to take a consistent copy of a database someone else
    is using, and the journal is written to every night by the AI runner.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _copy_one(source: Path, target: Path, *, compress: bool) -> tuple[int, str, str]:
    """Copy (optionally gzipped) and return (stored_bytes, stored_sha, source_sha)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        with source.open("rb") as raw, gzip.open(target, "wb", compresslevel=6) as out:
            shutil.copyfileobj(raw, out, length=1 << 20)
    else:
        shutil.copyfile(source, target)
    return target.stat().st_size, _sha256_of(target), _sha256_of(source)


def _iter_sources(item: ScopeItem):
    if not item.source.exists():
        return
    if item.mode == "files":
        for child in sorted(item.source.iterdir()):
            if child.is_file():
                yield child, child.name
        return
    for child in sorted(item.source.rglob("*")):
        if child.is_file():
            yield child, child.relative_to(item.source).as_posix()


def build_snapshot(
    staging_root: Path,
    *,
    scope: list[ScopeItem] | None = None,
    snapshot_date: str | None = None,
    compress_min_bytes: int = COMPRESS_MIN_BYTES,
    stability_min_bytes: int = STABILITY_MIN_BYTES,
    exclude_rotated: bool = True,
    sleep=time.sleep,
) -> SnapshotResult:
    """Stage one dated snapshot locally. Never raises on a single bad file."""
    stamp = snapshot_date or date.today().isoformat()
    staging = Path(staging_root) / stamp
    result = SnapshotResult(snapshot_date=stamp, staging=staging)
    result.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    staging.mkdir(parents=True, exist_ok=True)

    for item in scope if scope is not None else default_scope():
        for source, relative in _iter_sources(item):
            try:
                size = source.stat().st_size
            except OSError as exc:
                result.records.append(
                    FileRecord(item.label, relative, 0, skipped=f"unreadable: {type(exc).__name__}")
                )
                continue
            record = FileRecord(item.label, relative, size)
            if exclude_rotated and source.suffix.lower() in ROTATED_DUPLICATE_SUFFIXES:
                # Counted with a reason, never a silent omission. §0's frozen
                # pair is the one-time exception and passes exclude_rotated=False.
                record.skipped = EXCLUDED_ROTATED_REASON
                result.records.append(record)
                continue
            if size >= stability_min_bytes and not is_stable(source, sleep=sleep):
                # Not an error - it is being written right now. Recorded so the
                # gap is visible tonight rather than discovered during a restore.
                record.skipped = "unstable_during_snapshot"
                result.records.append(record)
                continue
            target = staging / item.label / relative
            try:
                if source.suffix.lower() in {".sqlite3", ".sqlite", ".db"}:
                    _copy_sqlite(source, target)
                    record.method = "sqlite_backup_api"
                    record.stored_bytes = target.stat().st_size
                    record.sha256 = _sha256_of(target)
                    # The backup API rewrites page layout, so the source hash is
                    # NOT the copy's hash and comparing them would be wrong.
                    record.source_sha256 = _sha256_of(source)
                else:
                    compress = size >= compress_min_bytes
                    if compress:
                        target = target.with_name(target.name + ".gz")
                        record.compressed = True
                        record.method = "gzip"
                    (record.stored_bytes, record.sha256,
                     record.source_sha256) = _copy_one(source, target, compress=compress)
            except OSError as exc:
                record.skipped = f"copy_failed: {type(exc).__name__}"
            result.records.append(record)

    result.finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    (staging / "manifest.json").write_text(
        json.dumps(result.manifest(), indent=1) + "\n", encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------
def snapshots_to_prune(
    dates: list[str],
    *,
    keep_daily: int = KEEP_DAILY,
    keep_weekly: int = KEEP_WEEKLY,
    keep_monthly: int = KEEP_MONTHLY,
) -> list[str]:
    """Which dated snapshots may be deleted: 7 daily, 4 weekly, 12 monthly.

    Weekly and monthly keep the NEWEST snapshot in each period, so pruning is a
    pure function of the date list and never depends on when it is run.
    """
    parsed = sorted({d for d in dates if _is_date(d)}, reverse=True)
    keep = set(parsed[:keep_daily])
    for bucket, limit in (
        (lambda d: date.fromisoformat(d).isocalendar()[:2], keep_weekly),
        (lambda d: date.fromisoformat(d).strftime("%Y-%m"), keep_monthly),
    ):
        seen: dict = {}
        for day in parsed:
            seen.setdefault(bucket(day), day)
        keep.update(list(seen.values())[:limit])
    return sorted(set(parsed) - keep, reverse=True)


def _is_date(text: str) -> bool:
    try:
        date.fromisoformat(str(text))
        return True
    except ValueError:
        return False


def prune(root: Path, **kwargs) -> list[str]:
    """Delete prunable snapshot directories under ``root``. Never touches frozen."""
    root = Path(root)
    if not root.is_dir():
        return []
    dates = [p.name for p in root.iterdir() if p.is_dir() and _is_date(p.name)]
    removed = []
    for stamp in snapshots_to_prune(dates, **kwargs):
        target = root / stamp
        if target.name == FROZEN_DIR_NAME:
            continue
        try:
            shutil.rmtree(target)
            removed.append(stamp)
        except OSError:
            logging.exception("could not prune snapshot %s", stamp)
    return removed


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------
def verify(snapshot_dir: Path) -> dict:
    """Re-hash every stored file against the manifest. Read-only."""
    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    checked = missing = mismatched = 0
    problems: list[str] = []
    for entry in manifest.get("entries", []):
        if entry.get("skipped"):
            continue
        name = entry["path"] + (".gz" if entry.get("compressed") else "")
        stored = snapshot_dir / entry["label"] / name
        if not stored.exists():
            missing += 1
            problems.append(f"missing: {entry['label']}/{name}")
            continue
        checked += 1
        if _sha256_of(stored) != entry["sha256"]:
            mismatched += 1
            problems.append(f"sha256 mismatch: {entry['label']}/{name}")
    return {
        "snapshot_date": manifest.get("snapshot_date"),
        "checked": checked,
        "missing": missing,
        "mismatched": mismatched,
        "ok": missing == 0 and mismatched == 0,
        "problems": problems[:20],
    }


def restore(snapshot_dir: Path, target_dir: Path, *, dry_run: bool = True) -> dict:
    """Restore a snapshot into a SCRATCH directory. Never into the live store.

    ``target_dir`` must not be the home folder or the diagnostics tree: a
    restore that overwrites live state during a drill is how a drill becomes an
    incident. The caller picks a scratch path and this refuses anything else.
    """
    from project_paths import PERSISTENT_DATA_DIR, get_diagnostics_dir

    snapshot_dir = Path(snapshot_dir)
    target = Path(target_dir).resolve()
    for forbidden in (Path(PERSISTENT_DATA_DIR).resolve(), Path(get_diagnostics_dir()).resolve()):
        if target == forbidden or forbidden in target.parents or target in forbidden.parents:
            raise ValueError(
                f"refusing to restore into {target}: it is inside the live store. "
                "Restore into a scratch directory and compare."
            )
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    planned = [e for e in manifest.get("entries", []) if not e.get("skipped")]
    if dry_run:
        return {"dry_run": True, "would_restore": len(planned), "target": str(target)}

    restored = 0
    for entry in planned:
        name = entry["path"] + (".gz" if entry.get("compressed") else "")
        stored = snapshot_dir / entry["label"] / name
        out = target / entry["label"] / entry["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if entry.get("compressed"):
            with gzip.open(stored, "rb") as raw, out.open("wb") as handle:
                shutil.copyfileobj(raw, handle, length=1 << 20)
        else:
            shutil.copyfile(stored, out)
        restored += 1
    return {"dry_run": False, "restored": restored, "target": str(target)}


# ---------------------------------------------------------------------------
def record_restore_test(staging_root: Path, *, snapshot_date: str, restored: int) -> None:
    """Note that a real restore actually happened. Best effort."""
    try:
        marker = Path(staging_root) / "last_restore_test.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "snapshot_date": snapshot_date,
                    "files_restored": int(restored),
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        logging.exception("restore-test marker not written (the restore itself succeeded).")


def health(staging_root: Path, das_root: Path | None = None) -> dict:
    """What the System Health tile shows. Never raises."""
    staging_root = Path(staging_root)
    out: dict = {
        "last_snapshot_date": "",
        "last_snapshot_at": "",
        "files": 0,
        "source_bytes": 0,
        "stored_bytes": 0,
        "skipped": 0,
        "das_reachable": False,
        "last_restore_test": "",
    }
    try:
        dates = sorted(p.name for p in staging_root.iterdir() if p.is_dir() and _is_date(p.name))
    except OSError:
        return out
    if dates:
        latest = staging_root / dates[-1] / "manifest.json"
        try:
            m = json.loads(latest.read_text(encoding="utf-8"))
            out.update(
                last_snapshot_date=m.get("snapshot_date", ""),
                last_snapshot_at=m.get("finished_at", ""),
                files=int(m.get("files", 0)),
                source_bytes=int(m.get("source_bytes", 0)),
                stored_bytes=int(m.get("stored_bytes", 0)),
                skipped=int(m.get("skipped", 0)),
            )
        except (OSError, ValueError):
            pass
    if das_root is not None:
        try:
            out["das_reachable"] = Path(das_root).is_dir()
        except OSError:
            out["das_reachable"] = False
    marker = staging_root / "last_restore_test.json"
    try:
        out["last_restore_test"] = json.loads(marker.read_text(encoding="utf-8")).get("at", "")
    except (OSError, ValueError):
        pass
    return out


def main() -> int:
    from project_paths import CACHE_DIR

    parser = argparse.ArgumentParser(description="Dated evidence snapshot (R10.A)")
    parser.add_argument("--staging", type=Path, default=Path(CACHE_DIR) / "evidence_snapshots")
    parser.add_argument("--date", default=None)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--verify", type=Path, default=None, help="verify a snapshot directory")
    parser.add_argument("--restore", type=Path, default=None, help="snapshot directory to restore")
    parser.add_argument("--into", type=Path, default=None, help="scratch directory to restore into")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.verify:
        print(json.dumps(verify(args.verify), indent=1))
        return 0
    if args.restore:
        if not args.into:
            print("--restore needs --into <scratch directory>")
            return 2
        outcome = restore(args.restore, args.into, dry_run=args.dry_run)
        if not args.dry_run and outcome.get("restored"):
            # A backup nobody has restored is a hypothesis, so the drill records
            # itself and the health tile reports when it last happened. A dry
            # run deliberately does not count - it proved nothing about the bytes.
            record_restore_test(
                args.staging,
                snapshot_date=Path(args.restore).name,
                restored=int(outcome["restored"]),
            )
        print(json.dumps(outcome, indent=1))
        return 0

    result = build_snapshot(args.staging, snapshot_date=args.date)
    m = result.manifest()
    print(
        f"snapshot {m['snapshot_date']}: {m['files']} files, "
        f"{m['source_bytes'] / 1e6:.0f} MB source -> {m['stored_bytes'] / 1e6:.0f} MB stored, "
        f"{m['skipped']} skipped {m['skipped_by_reason'] or ''}"
    )
    if args.prune:
        removed = prune(args.staging)
        print(f"pruned {len(removed)} snapshot(s): {', '.join(removed) or 'none'}")

    # R10.V step 6: the daily-bar unit measurement rides the nightly job rather
    # than the health tile, because it takes ~7 s over 1,958 files and a tile a
    # human waits on is a tile nobody opens. A failure here must not fail the
    # snapshot - the backup is the job, this is a reading taken beside it.
    try:
        from datetime import datetime, timezone

        from ops.daily_bar_cliff import HEALTH_FILENAME, write_store_health
        from project_paths import MASTER_AVWAP_DAILY_BARS_DIR, get_diagnostics_dir

        health = write_store_health(
            MASTER_AVWAP_DAILY_BARS_DIR,
            Path(get_diagnostics_dir()) / HEALTH_FILENAME,
            measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        print(
            f"daily-bar units: {health['rows_by_volume_unit'].get('shares', 0):,} shares of "
            f"{health['rows']:,} rows, {health['cliff']['cliffed']} file(s) over "
            f"{health['cliff']['threshold']:g}x"
        )
    except Exception:
        logging.exception("daily-bar unit health measurement skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
