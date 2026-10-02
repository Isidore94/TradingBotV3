"""Lossless archive of the per-bar detail that setup-tracker compaction strips.

Compaction (``_compact_tracker_setup_record`` in ``master_avwap_lib.legacy``)
empties a sealed record's ``daily_marks`` and every scenario's ``events`` to keep
the tracker JSON bounded. Before this module, that detail was simply gone. Now
the tracker save archives it here first and strips only the records whose archive
write verified; a record that did not verify stays whole and is retried next save.

Store: one small SQLite file (``SETUP_TRACKER_DETAIL_ARCHIVE_DB``), separate from
the tracker's SQLite mirror. One row per (namespace, setup key, content hash),
holding the zlib-compressed JSON text of exactly what compaction removes, encoded
the way the tracker save encodes it (``ensure_ascii=False``, compact separators,
the saver's ``default``), so re-inserting the decoded values reproduces the
pre-compaction record's JSON text.

Different content for a key already archived is KEPT BESIDE the old row, never
written over it: the archive's job is to lose nothing, and refusing would leave
that record unstripped on every save forever. ``load_detail`` returns the newest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import sqlite3
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

FORMAT = "tracker_detail_v1"
RECORD_NAMESPACES = ("setups", "control_setups", "study_setups")
_COMPRESS_LEVEL = 6
_BUSY_TIMEOUT_SECONDS = 10.0


def default_archive_path() -> Path:
    from project_paths import SETUP_TRACKER_DETAIL_ARCHIVE_DB

    return Path(SETUP_TRACKER_DETAIL_ARCHIVE_DB)


# -- pure: what compaction removes, and putting it back ------------------------
def extract_detail(setup: dict) -> dict | None:
    """Exactly what ``_compact_tracker_setup_record`` would remove or change on
    ``setup``; None when it would change nothing. Values are the record's own
    objects (not copies): encode them before the record is compacted."""
    if not isinstance(setup, dict):
        return None
    detail: dict[str, Any] = {}
    if setup.get("daily_marks"):
        detail["daily_marks"] = setup["daily_marks"]
        # Stripping marks may (re)write `short_horizon`; keep its prior state too.
        detail["short_horizon"] = {"present": "short_horizon" in setup, "value": setup.get("short_horizon")}
    scenarios = setup.get("scenarios")
    if isinstance(scenarios, dict):
        events = {
            str(name): scenario["events"]
            for name, scenario in scenarios.items()
            if isinstance(scenario, dict) and scenario.get("events")
        }
        if events:
            detail["events"] = events
    return detail or None


def restore_record(record: dict, detail: dict | None) -> dict:
    """A new record: ``record`` (compacted) with the archived ``detail`` put back,
    which reproduces the record as it was before compaction. Pure."""
    restored = copy.deepcopy(record)
    if not detail:
        return restored
    if "daily_marks" in detail:
        restored["daily_marks"] = copy.deepcopy(detail["daily_marks"])
        prior = detail.get("short_horizon")
        if isinstance(prior, dict):
            if prior.get("present"):
                restored["short_horizon"] = copy.deepcopy(prior.get("value"))
            else:
                restored.pop("short_horizon", None)
    scenarios = restored.get("scenarios")
    for name, events in (detail.get("events") or {}).items():
        if isinstance(scenarios, dict) and isinstance(scenarios.get(name), dict):
            scenarios[name]["events"] = copy.deepcopy(events)
    return restored


def encode_detail(detail: dict, json_default=None) -> str:
    """The tracker save's own encoding (``save_json``): insertion order, compact."""
    return json.dumps(detail, ensure_ascii=False, separators=(",", ":"), default=json_default or str)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _compress(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8", "surrogatepass"), _COMPRESS_LEVEL)


def _decompress(blob: bytes) -> str:
    return zlib.decompress(blob).decode("utf-8", "surrogatepass")


# -- the store -----------------------------------------------------------------
@dataclass
class ArchiveResult:
    """One archive pass. ``safe`` holds the (namespace, key) pairs that may be
    compacted: nothing to archive, or archived and verified by read-back."""

    safe: set = field(default_factory=set)
    written: int = 0
    already_archived: int = 0
    failed: list = field(default_factory=list)
    raw_bytes: int = 0
    stored_bytes: int = 0
    error: str = ""


class DetailArchive:
    """SQLite file of compressed detail blobs keyed by (namespace, key, sha256)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _connect(self, *, create: bool) -> sqlite3.Connection:
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=_BUSY_TIMEOUT_SECONDS)
        else:
            conn = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, timeout=_BUSY_TIMEOUT_SECONDS)
        if create:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS detail ("
                " namespace TEXT NOT NULL, setup_key TEXT NOT NULL, sha256 TEXT NOT NULL,"
                " format TEXT NOT NULL, raw_bytes INTEGER NOT NULL, blob BLOB NOT NULL,"
                " archived_at TEXT NOT NULL, PRIMARY KEY (namespace, setup_key, sha256))"
            )
        return conn

    def archive(self, items: Iterable[tuple[str, str, dict]], *, json_default=None) -> ArchiveResult:
        """Archive each item's strippable detail, then verify every row through a
        fresh read-only connection. Never raises; the result says what is safe."""
        items = list(items)
        result = ArchiveResult()
        pending: list[tuple[str, str, str]] = []  # (namespace, key, sha256) written or already present
        try:
            conn = self._connect(create=True)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.failed = [(ns, key) for ns, key, setup in items if extract_detail(setup)]
            return result
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with conn:
                for namespace, key, setup in items:
                    ident = (str(namespace), str(key))
                    detail = extract_detail(setup)
                    if detail is None:
                        result.safe.add(ident)
                        continue
                    try:
                        text = encode_detail(detail, json_default)
                        digest = _hash(text)
                        blob = _compress(text)
                        cursor = conn.execute(
                            "INSERT OR IGNORE INTO detail VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (ident[0], ident[1], digest, FORMAT, len(text), blob, stamp),
                        )
                    except Exception as exc:
                        logging.warning("Tracker detail archive: could not write %s/%s: %s", *ident, exc)
                        result.failed.append(ident)
                        continue
                    if cursor.rowcount:
                        result.written += 1
                        result.raw_bytes += len(text)
                        result.stored_bytes += len(blob)
                    else:
                        result.already_archived += 1
                    pending.append((ident[0], ident[1], digest))
        except Exception as exc:
            # The transaction rolled back: nothing from this pass is trusted.
            result.error = f"{type(exc).__name__}: {exc}"
            result.failed.extend((ns, key) for ns, key, _ in pending)
            result.written = result.already_archived = result.raw_bytes = result.stored_bytes = 0
            pending = []
        finally:
            conn.close()
        if pending:
            self._verify_pending(pending, result)
        return result

    def _verify_pending(self, pending: list[tuple[str, str, str]], result: ArchiveResult) -> None:
        try:
            reader = self._connect(create=False)
        except Exception as exc:
            result.error = f"read-back unavailable: {type(exc).__name__}: {exc}"
            result.failed.extend((ns, key) for ns, key, _ in pending)
            return
        try:
            for namespace, key, digest in pending:
                row = reader.execute(
                    "SELECT blob FROM detail WHERE namespace=? AND setup_key=? AND sha256=?",
                    (namespace, key, digest),
                ).fetchone()
                if row is not None and _blob_ok(row[0], digest):
                    result.safe.add((namespace, key))
                else:
                    result.failed.append((namespace, key))
        except Exception as exc:
            result.error = f"read-back failed: {type(exc).__name__}: {exc}"
            result.failed.extend((ns, key) for ns, key, _ in pending if (ns, key) not in result.safe)
        finally:
            reader.close()

    # -- reading --------------------------------------------------------------
    def load_versions(self, namespace: str, key: str) -> list[dict]:
        """Every archived detail for one record, oldest first."""
        if not self.path.exists():
            return []
        conn = self._connect(create=False)
        try:
            rows = conn.execute(
                "SELECT sha256, blob FROM detail WHERE namespace=? AND setup_key=? ORDER BY rowid",
                (str(namespace), str(key)),
            ).fetchall()
        finally:
            conn.close()
        versions = []
        for digest, blob in rows:
            text = _decompress(blob)
            if _hash(text) != digest:
                raise ValueError(f"archived detail for {namespace}/{key} fails its hash")
            versions.append(json.loads(text))
        return versions

    def load_detail(self, namespace: str, key: str) -> dict | None:
        """The newest archived detail for one record, or None."""
        versions = self.load_versions(namespace, key)
        return versions[-1] if versions else None

    def status(self) -> dict:
        if not self.path.exists():
            return {"path": str(self.path), "exists": False, "rows": 0, "records": 0}
        conn = self._connect(create=False)
        try:
            rows, records, raw, stored, last = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT namespace || char(0) || setup_key),"
                " COALESCE(SUM(raw_bytes), 0), COALESCE(SUM(LENGTH(blob)), 0), MAX(archived_at) FROM detail"
            ).fetchone()
            by_namespace = dict(conn.execute("SELECT namespace, COUNT(*) FROM detail GROUP BY namespace").fetchall())
        finally:
            conn.close()
        return {
            "path": str(self.path),
            "exists": True,
            "rows": int(rows),
            "records": int(records),
            "rows_by_namespace": by_namespace,
            "raw_bytes": int(raw),
            "stored_bytes": int(stored),
            "file_bytes": self.path.stat().st_size,
            "last_write": last or "",
        }

    def verify(self) -> dict:
        """Decode every blob and check its hash. Streams one row at a time."""
        report = {"path": str(self.path), "rows": 0, "bad": []}
        if not self.path.exists():
            report["ok"] = True
            return report
        conn = self._connect(create=False)
        try:
            for namespace, key, digest, blob in conn.execute(
                "SELECT namespace, setup_key, sha256, blob FROM detail ORDER BY rowid"
            ):
                report["rows"] += 1
                if not _blob_ok(blob, digest):
                    report["bad"].append(f"{namespace}/{key}/{digest[:12]}")
        finally:
            conn.close()
        report["ok"] = not report["bad"]
        return report


def _blob_ok(blob: bytes, digest: str) -> bool:
    try:
        text = _decompress(blob)
        json.loads(text)
    except Exception:
        return False
    return _hash(text) == digest


# -- the tracker save's hook ---------------------------------------------------
def archive_before_compaction(
    items: Iterable[tuple[str, str, dict]], *, json_default=None, path: Path | str | None = None
) -> set:
    """Archive the detail of each (namespace, key, record) about to be compacted.
    Returns the (namespace, key) pairs that are safe to compact. Never raises; a
    failure is logged as an error with its count and costs only disk space."""
    items = list(items)
    try:
        archive = DetailArchive(path or default_archive_path())
        result = archive.archive(items, json_default=json_default)
    except Exception as exc:
        result = ArchiveResult(error=f"{type(exc).__name__}: {exc}")
        result.failed = [(ns, key) for ns, key, setup in items if extract_detail(setup)]
    if result.written or result.already_archived:
        logging.info(
            "Setup tracker detail archive: %d record(s) archived (%d already), %d -> %d bytes, verified.",
            result.written, result.already_archived, result.raw_bytes, result.stored_bytes,
        )
    if result.failed or result.error:
        logging.error(
            "Setup tracker detail archive FAILED for %d sealed record(s) (%s); they are NOT compacted "
            "this save, stay full size, and are retried next save.",
            len(result.failed), result.error or "read-back verify failed",
        )
    return result.safe


# -- CLI -----------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Setup tracker detail archive: status, verify, show.")
    parser.add_argument("--db", default="", help="archive path (default: the desk's)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="rows, records, bytes, last write")
    sub.add_parser("verify", help="decode and hash-check every blob")
    show = sub.add_parser("show", help="print one record's archived detail (newest version)")
    show.add_argument("namespace", choices=RECORD_NAMESPACES)
    show.add_argument("key")
    show.add_argument("--all-versions", action="store_true")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    archive = DetailArchive(Path(args.db) if args.db else default_archive_path())
    if args.command == "status":
        print(json.dumps(archive.status(), indent=2))
        return 0
    if args.command == "verify":
        report = archive.verify()
        report["bad"] = report["bad"][:50]
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    versions = archive.load_versions(args.namespace, args.key)
    if not versions:
        print(f"no archived detail for {args.namespace}/{args.key}")
        return 1
    print(json.dumps(versions if args.all_versions else versions[-1], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
