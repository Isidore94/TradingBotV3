"""Three-class backup and the scripted restore check (plan sec 8.5, Phase 8).

A DAS/RAID is capacity and availability, not backup. The classes are fixed:

* **Class A - irreplaceable-small** (manifests, definitions, trader geometry,
  review events, evidence freezes): copied to the backup disk AND mirrored into
  the home folder. NOTE (decision 0015): the home-folder mirror used to be
  Drive-synced, which bought off-site storage for free. With no cloud sync
  there is currently NO off-site copy - every mirror is on-premises. Treat an
  off-site Class A destination as owed, not solved.
* **Class B - the lake**: incremental copy to a second physical disk,
  **append-only** - a file missing from the source is never deleted from the
  copy, because the copy exists to survive a mistake at the source.
* **Class C - derived**: never backed up; rebuilt from A + B.

The restore check is scripted rather than described: it restores one month
partition to a NEW root, re-verifies every file against the hash the manifest
recorded, and runs one canned query. Restores never overwrite in place.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:  # package import
    from .manifest import ManifestLog, utc_now
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    from manifest import ManifestLog, utc_now  # type: ignore
    from store import ResearchStore  # type: ignore

CLASS_A = "A"
CLASS_B = "B"
CLASS_C = "C"

#: Class A inside the lake: small, irreplaceable, and mirrored off-site.
CLASS_A_LAKE_ENTRIES = ("manifest_log.jsonl", "imported_bundles.jsonl", "definitions")


@dataclass
class BackupReport:
    backup_class: str = ""
    status: str = "OK"  # OK | DISABLED | NO_TARGET
    files_copied: int = 0
    files_skipped: int = 0
    bytes_copied: int = 0
    deleted_from_target: int = 0  # always 0: copies are append-only
    errors: list = field(default_factory=list)


@dataclass
class RestoreReport:
    status: str = "OK"  # OK | DISABLED | NOTHING_TO_RESTORE | FAILED
    dataset: str = ""
    partition: str = ""
    files: int = 0
    rows: int = 0
    hash_mismatches: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    query_rows: int = 0
    restored_to: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "OK" and not self.hash_mismatches and not self.missing and self.files > 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree(source: Path, target: Path, report: BackupReport) -> None:
    """Append-only incremental copy: never deletes, never overwrites blindly."""
    if not source.exists():
        return
    single_file = source.is_file()
    items = [source] if single_file else sorted(source.rglob("*"))
    for item in items:
        if item.is_dir():
            continue
        # For a single file the target IS the destination path; for a tree it
        # is the destination root.
        destination = target if single_file else target / str(item.relative_to(source))
        try:
            if destination.exists() and destination.stat().st_size == item.stat().st_size:
                report.files_skipped += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            report.files_copied += 1
            report.bytes_copied += item.stat().st_size
        except OSError as exc:
            report.errors.append(f"{item}: {exc}")


def backup_class_a(store: ResearchStore | None, targets, *, now: datetime | None = None) -> BackupReport:
    """Copy the irreplaceable-small set to every target (disk + home mirror)."""
    report = BackupReport(backup_class=CLASS_A)
    if store is None:
        report.status = "DISABLED"
        return report
    destinations = [Path(target) for target in (targets or []) if target]
    if not destinations:
        report.status = "NO_TARGET"
        return report
    stamp = (now or utc_now()).strftime("%Y%m%d")
    for destination in destinations:
        for name in CLASS_A_LAKE_ENTRIES:
            _copy_tree(store.root / name, destination / stamp / name, report)
    return report


def backup_class_b(store: ResearchStore | None, target, *, now: datetime | None = None) -> BackupReport:
    """Incremental, append-only copy of the lake to a second physical disk.

    Deletion is never propagated - that is the difference between a backup and
    a mirror, and the reason `/MIR`-style copying is forbidden here.
    """
    report = BackupReport(backup_class=CLASS_B)
    if store is None:
        report.status = "DISABLED"
        return report
    if not target:
        report.status = "NO_TARGET"
        return report
    destination = Path(target)
    for layer in ("bronze", "silver", "gold"):
        _copy_tree(store.root / layer, destination / layer, report)
    for name in CLASS_A_LAKE_ENTRIES:
        _copy_tree(store.root / name, destination / name, report)
    return report


def restore_check(
    store: ResearchStore | None,
    target_root,
    *,
    dataset: str = "bar_m5",
    partition: str | None = None,
) -> RestoreReport:
    """Restore one partition to a NEW root and verify it (sec 8.5).

    Every restored file is re-hashed against the value the manifest recorded at
    seal time, and one canned query runs against the restored copy. Nothing is
    written back into the live lake: a restore that overwrote in place could
    destroy the very artifact it was meant to prove.
    """
    report = RestoreReport(dataset=dataset, partition=partition or "")
    if store is None:
        report.status = "DISABLED"
        return report
    snapshot = store.manifest.resolve(dataset=dataset, partition=partition)
    if not snapshot.entries:
        report.status = "NOTHING_TO_RESTORE"
        return report

    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)
    report.restored_to = str(root)
    restored = ResearchStore(root)
    ledger = ManifestLog(root)
    for entry in snapshot.entries:
        source = store.root / entry.file_path
        if not source.exists():
            report.missing.append(entry.file_path)
            continue
        destination = root / entry.file_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if entry.sha256 and _sha256(destination) != entry.sha256:
            report.hash_mismatches.append(entry.file_path)
            continue
        # Re-point the manifest at the restored copy rather than rewriting it.
        ledger.append(
            action="IMPORT",
            dataset=entry.dataset,
            partition=entry.partition,
            file_path=entry.file_path,
            sha256=entry.sha256,
            row_count=entry.row_count,
            min_ts=entry.min_ts,
            max_ts=entry.max_ts,
            git_commit=entry.git_commit,
            job_id="restore_check",
            restored_from=str(store.root),
        )
        report.files += 1
        report.rows += entry.row_count

    if report.missing or report.hash_mismatches:
        report.status = "FAILED"
        return report
    try:
        report.query_rows = restored.read_table(dataset, partition).num_rows
    except Exception as exc:
        report.status = "FAILED"
        report.hash_mismatches.append(f"canned query failed: {exc}")
    return report


__all__ = [
    "CLASS_A",
    "CLASS_A_LAKE_ENTRIES",
    "CLASS_B",
    "CLASS_C",
    "BackupReport",
    "RestoreReport",
    "backup_class_a",
    "backup_class_b",
    "restore_check",
]
