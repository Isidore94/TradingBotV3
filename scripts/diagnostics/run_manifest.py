"""Structured per-run manifests (plan.md Phase 1 / sec 6.1).

Every production scan writes one manifest - success or failure - so speed and
unattended failures are measurable before they are optimized. Manifests are
local operational data (not Drive-synced): small JSON files, newest last,
pruned to a bounded count.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "run_manifest_v1"
DEFAULT_KEEP = 90


def default_manifest_dir() -> Path:
    try:
        from project_paths import CACHE_DIR

        return Path(CACHE_DIR).parent / "diagnostics" / "run_manifests"
    except Exception:
        return Path.home() / ".tradingbotv3" / "run_manifests"


@dataclass
class ManifestRecorder:
    job_type: str
    trigger: str = "unspecified"
    run_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    status: str = "running"
    error: str = ""
    phases: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.run_id = f"{self.job_type}-{stamp}-{uuid.uuid4().hex[:6]}"
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    def record_phase(self, label: str, seconds: float) -> None:
        self.phases.append({"label": str(label), "seconds": round(float(seconds), 3)})

    def set_counter(self, name: str, value) -> None:
        self.counters[str(name)] = value

    def incr(self, name: str, amount: int = 1) -> None:
        self.counters[str(name)] = int(self.counters.get(str(name), 0)) + amount

    def finalize(self, status: str, error: str = "") -> None:
        self.status = str(status)
        self.error = str(error or "")
        self.ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "job_type": self.job_type,
            "trigger": self.trigger,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "error": self.error,
            "total_seconds": round(sum(p["seconds"] for p in self.phases), 3),
            "phases": self.phases,
            "counters": self.counters,
            "outputs": self.outputs,
        }

    def save(self, directory: Path | str | None = None, keep: int = DEFAULT_KEEP) -> Path:
        directory = Path(directory) if directory is not None else default_manifest_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        fd, tmp_name = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=1)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
        prune_manifests(directory, keep=keep)
        return path


def prune_manifests(directory: Path | str, keep: int = DEFAULT_KEEP) -> int:
    directory = Path(directory)
    if not directory.exists():
        return 0
    files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
    removed = 0
    for path in files[: max(0, len(files) - keep)]:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def load_recent_manifests(directory: Path | str | None = None, limit: int = 30) -> list[dict]:
    directory = Path(directory) if directory is not None else default_manifest_dir()
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict] = []
    for path in files[: max(0, int(limit))]:
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


# ---------------------------------------------------------------------------
# active-recorder registry (per thread; scans run one per worker/subprocess)
# ---------------------------------------------------------------------------
_active = threading.local()


def set_active_recorder(recorder: ManifestRecorder | None) -> None:
    _active.recorder = recorder


def get_active_recorder() -> ManifestRecorder | None:
    return getattr(_active, "recorder", None)


def clear_active_recorder() -> None:
    _active.recorder = None
