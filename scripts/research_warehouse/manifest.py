"""``manifest_log.jsonl`` - the lake's read authority (plan sec 7.1, 8.3).

The directory tree is *not* the read authority. Every supported read resolves
its file list from this append-only ledger and hands that explicit list to the
reader, which is what makes a query concurrent with a compaction return either
the pre- or the post-compaction row set and never a double count: files are
immutable, a compaction publishes its replacement and retires its inputs in
**one** appended line (the atomic switch), and physical moves into
``_retired/<yyyymmdd>/`` are garbage collection that may lag by up to 30 days.

Corruption policy (sec 8.3): a torn final line is the ordinary crash artifact -
the writer died mid-append - so it is ignored on read and truncated before the
next append. A malformed line anywhere *earlier* is real manifest corruption
and raises :class:`ManifestCorruptionError`; that is the one condition that
vetoes a publish wholesale. A bounded dirty tail of *data* never does - it goes
to ``_quarantine/`` and the clean remainder publishes (tracker incident, week
of 2026-07-13).

This is a few dozen lines of ledger reading, deliberately not a catalog system.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest_log.jsonl"
IMPORTED_BUNDLES_NAME = "imported_bundles.jsonl"

ACTION_PUBLISH = "PUBLISH"
ACTION_COMPACT = "COMPACT"
ACTION_RETIRE = "RETIRE"
ACTION_IMPORT = "IMPORT"
ACTION_QUARANTINE = "QUARANTINE"
ACTIONS = (ACTION_PUBLISH, ACTION_COMPACT, ACTION_RETIRE, ACTION_IMPORT, ACTION_QUARANTINE)

# Actions that make a file part of the live dataset file set.
_ADDING_ACTIONS = {ACTION_PUBLISH, ACTION_COMPACT, ACTION_IMPORT}

_APPEND_LOCKS: dict[str, threading.Lock] = {}
_APPEND_LOCKS_GUARD = threading.Lock()


class ManifestCorruptionError(RuntimeError):
    """The ledger itself is unreadable - the only wholesale-veto condition."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:  # never record a naive clock (sec 2)
            raise ValueError(f"Manifest timestamps must be timezone-aware: {value!r}")
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def lake_relative(root: Path, path: Path) -> str:
    """Lake-relative POSIX path - the manifest's portable file identity."""
    resolved = Path(path)
    try:
        rel = resolved.relative_to(Path(root))
    except ValueError:
        rel = resolved
    return rel.as_posix()


@dataclass(frozen=True)
class ManifestEntry:
    manifest_seq: int
    action: str
    dataset: str
    partition: str
    file_path: str
    sha256: str = ""
    row_count: int = 0
    min_ts: str = ""
    max_ts: str = ""
    supersedes: tuple[str, ...] = ()
    git_commit: str = ""
    job_id: str = ""
    written_at: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "ManifestEntry":
        known = {
            "manifest_seq",
            "action",
            "dataset",
            "partition",
            "file_path",
            "sha256",
            "row_count",
            "min_ts",
            "max_ts",
            "supersedes",
            "git_commit",
            "job_id",
            "written_at",
        }
        supersedes = payload.get("supersedes") or ()
        return cls(
            manifest_seq=int(payload.get("manifest_seq", 0)),
            action=str(payload.get("action", "")),
            dataset=str(payload.get("dataset", "")),
            partition=str(payload.get("partition", "")),
            file_path=str(payload.get("file_path", "")),
            sha256=str(payload.get("sha256", "")),
            row_count=int(payload.get("row_count", 0) or 0),
            min_ts=str(payload.get("min_ts", "") or ""),
            max_ts=str(payload.get("max_ts", "") or ""),
            supersedes=tuple(str(item) for item in supersedes),
            git_commit=str(payload.get("git_commit", "") or ""),
            job_id=str(payload.get("job_id", "") or ""),
            written_at=str(payload.get("written_at", "") or ""),
            extra={key: value for key, value in payload.items() if key not in known},
        )


@dataclass(frozen=True)
class ManifestSnapshot:
    """One query's consistent view: the live file list at a manifest position."""

    manifest_seq: int
    entries: tuple[ManifestEntry, ...]

    @property
    def file_paths(self) -> tuple[str, ...]:
        return tuple(entry.file_path for entry in self.entries)

    @property
    def row_count(self) -> int:
        return sum(entry.row_count for entry in self.entries)


def _append_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _APPEND_LOCKS_GUARD:
        lock = _APPEND_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _APPEND_LOCKS[key] = lock
        return lock


class ManifestLog:
    """Append/resolve over one lake's ``manifest_log.jsonl``."""

    def __init__(self, root: Path, name: str = MANIFEST_NAME):
        self.root = Path(root)
        self.path = self.root / name

    # -- reading ------------------------------------------------------------
    def read_entries(self) -> list[ManifestEntry]:
        """Every ledger line in order; a torn *final* line is tolerated."""
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if not raw:
            return []
        text = raw.decode("utf-8", errors="replace")
        lines = text.split("\n")
        trailing_newline = text.endswith("\n")
        if trailing_newline:
            lines = lines[:-1]
        entries: list[ManifestEntry] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except (ValueError, UnicodeDecodeError) as exc:
                is_last = index == len(lines) - 1
                if is_last and not trailing_newline:
                    # Crash mid-append: the record never completed, so it never
                    # happened. It is dropped on the next append.
                    break
                raise ManifestCorruptionError(
                    f"{self.path} line {index + 1} is not valid JSON: {exc}. "
                    "Manifest corruption is the one wholesale-veto condition; "
                    "restore the ledger from backup before writing again."
                ) from exc
            if not isinstance(payload, dict):
                raise ManifestCorruptionError(f"{self.path} line {index + 1} is not a JSON object.")
            entries.append(ManifestEntry.from_dict(payload))
        return entries

    def next_seq(self) -> int:
        entries = self.read_entries()
        return (entries[-1].manifest_seq + 1) if entries else 1

    def resolve(self, dataset: str | None = None, partition: str | None = None) -> ManifestSnapshot:
        """The live file set for a dataset/partition, as of the current end.

        Retirement is applied logically: a file superseded by a COMPACT line, or
        named by a RETIRE line, leaves the live set the moment that line lands,
        regardless of whether the physical GC move has happened yet.
        """
        live: dict[str, ManifestEntry] = {}
        position = 0
        for entry in self.read_entries():
            position = max(position, entry.manifest_seq)
            if entry.action in _ADDING_ACTIONS:
                for retired in entry.supersedes:
                    live.pop(retired, None)
                live[entry.file_path] = entry
            elif entry.action == ACTION_RETIRE:
                live.pop(entry.file_path, None)
                for retired in entry.supersedes:
                    live.pop(retired, None)
            # QUARANTINE files are evidence, never part of a dataset read.
        selected = [
            entry
            for entry in live.values()
            if (dataset is None or entry.dataset == dataset)
            and (partition is None or entry.partition == partition)
        ]
        selected.sort(key=lambda entry: entry.manifest_seq)
        return ManifestSnapshot(manifest_seq=position, entries=tuple(selected))

    def quarantine_entries(self) -> list[ManifestEntry]:
        return [entry for entry in self.read_entries() if entry.action == ACTION_QUARANTINE]

    def retired_entries(self) -> list[ManifestEntry]:
        """(retired file path, the line that retired it) for the GC pass."""
        retired: list[ManifestEntry] = []
        for entry in self.read_entries():
            if entry.action in {ACTION_COMPACT, ACTION_RETIRE} and entry.supersedes:
                retired.append(entry)
            elif entry.action == ACTION_RETIRE and entry.file_path:
                retired.append(entry)
        return retired

    # -- writing ------------------------------------------------------------
    def append(
        self,
        *,
        action: str,
        dataset: str,
        partition: str,
        file_path: str,
        sha256: str = "",
        row_count: int = 0,
        min_ts=None,
        max_ts=None,
        supersedes=None,
        git_commit: str = "",
        job_id: str = "",
        **extra,
    ) -> ManifestEntry:
        """Append exactly one ledger line, durably. Step 4 of the seal."""
        if action not in ACTIONS:
            raise ValueError(f"Unknown manifest action {action!r}; allowed: {', '.join(ACTIONS)}")
        with _append_lock(self.path):
            self.repair_torn_tail()
            payload = {
                "manifest_seq": self.next_seq(),
                "action": action,
                "dataset": dataset,
                "partition": partition,
                "file_path": file_path,
                "sha256": sha256,
                "row_count": int(row_count),
                "min_ts": _iso(min_ts),
                "max_ts": _iso(max_ts),
                "supersedes": [str(item) for item in (supersedes or [])],
                "git_commit": git_commit,
                "job_id": job_id,
                "written_at": _iso(utc_now()),
            }
            payload.update({key: value for key, value in extra.items() if key not in payload})
            line = json.dumps(payload, sort_keys=False, default=str) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            return ManifestEntry.from_dict(payload)

    def repair_torn_tail(self) -> bool:
        """Drop a half-written final line so the next append starts clean.

        Returns True when a torn tail was actually removed (startup
        reconciliation reports it).
        """
        if not self.path.exists():
            return False
        size = self.path.stat().st_size
        if size == 0:
            return False
        with open(self.path, "rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return False
            handle.seek(0)
            data = handle.read()
            cut = data.rfind(b"\n")
            handle.seek(0)
            handle.truncate(cut + 1 if cut >= 0 else 0)
            handle.flush()
            os.fsync(handle.fileno())
        return True


_GIT_COMMIT_CACHE: dict[str, str] = {}


def definitions_git_commit(repo_root: Path | None = None) -> str:
    """Current HEAD commit, read from ``.git`` without spawning a process.

    Every manifest line records it so an old evidence freeze can name the exact
    definitions that produced it (sec 8.2). An unreadable repo yields "" rather
    than failing a publish - provenance is evidence, not a gate.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    key = str(root)
    if key in _GIT_COMMIT_CACHE:
        return _GIT_COMMIT_CACHE[key]
    commit = ""
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            ref_path = root / ".git" / ref
            if ref_path.exists():
                commit = ref_path.read_text(encoding="utf-8").strip()
            else:  # packed refs
                packed = root / ".git" / "packed-refs"
                if packed.exists():
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line.endswith(f" {ref}"):
                            commit = line.split(" ", 1)[0].strip()
                            break
        else:
            commit = head
    except OSError:
        commit = ""
    _GIT_COMMIT_CACHE[key] = commit
    return commit
