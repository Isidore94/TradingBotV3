from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from industry_scanner import (
    INDUSTRY_BOARD_CSV_FILE,
    SECTOR_BOARD_CSV_FILE,
    run_industry_scan,
)
from project_paths import INDUSTRY_BOARD_STATE_FILE


INDUSTRY_REFRESH_INTERVAL_SECONDS = 60 * 60
INDUSTRY_REFRESH_CHECK_MS = 60_000
def inspect_industry_snapshot(
    *,
    sector_path: Path = SECTOR_BOARD_CSV_FILE,
    industry_path: Path = INDUSTRY_BOARD_CSV_FILE,
    now: datetime | None = None,
    fresh_for_seconds: int = INDUSTRY_REFRESH_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Return one freshness identity for the two files consumed as a board."""
    moment = now or datetime.now()
    paths = (Path(sector_path), Path(industry_path))
    existing = [path for path in paths if path.exists()]
    if not existing:
        return {
            "state": "missing",
            "fresh": False,
            "as_of": None,
            "as_of_iso": "",
            "age_seconds": None,
            "snapshot_id": "",
            "sector_path": str(paths[0]),
            "industry_path": str(paths[1]),
        }

    mtimes = [path.stat().st_mtime for path in existing]
    # The pair is only as current as its older member.
    as_of = datetime.fromtimestamp(min(mtimes))
    age_seconds = max(0.0, (moment - as_of).total_seconds())
    complete = len(existing) == len(paths)
    fresh = complete and age_seconds <= max(1, int(fresh_for_seconds))
    state = "fresh" if fresh else ("stale" if complete else "partial")
    identity = "|".join(
        f"{path.name}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
        for path in existing
    )
    snapshot_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return {
        "state": state,
        "fresh": fresh,
        "as_of": as_of,
        "as_of_iso": as_of.isoformat(timespec="seconds"),
        "age_seconds": age_seconds,
        "snapshot_id": snapshot_id,
        "sector_path": str(paths[0]),
        "industry_path": str(paths[1]),
    }


def industry_refresh_due(snapshot: dict[str, Any]) -> bool:
    return not bool((snapshot or {}).get("fresh"))


def _read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


class IndustryBoardService(QObject):
    """Single owner for startup/manual/hourly Industry Board refreshes."""

    refreshStarted = Signal()
    refreshFinished = Signal(object)
    snapshotChanged = Signal(object)
    _workerFinished = Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        scan_runner: Callable[..., dict] = run_industry_scan,
        sector_path: Path = SECTOR_BOARD_CSV_FILE,
        industry_path: Path = INDUSTRY_BOARD_CSV_FILE,
        state_path: Path = INDUSTRY_BOARD_STATE_FILE,
        refresh_interval_seconds: int = INDUSTRY_REFRESH_INTERVAL_SECONDS,
        startup_delay_ms: int = 5_000,
    ) -> None:
        super().__init__(parent)
        self._scan_runner = scan_runner
        self._sector_path = Path(sector_path)
        self._industry_path = Path(industry_path)
        self._state_path = Path(state_path)
        self._refresh_interval_seconds = max(60, int(refresh_interval_seconds))
        self._startup_delay_ms = max(0, int(startup_delay_ms))
        self._refresh_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False

        self._timer = QTimer(self)
        self._timer.setInterval(INDUSTRY_REFRESH_CHECK_MS)
        self._timer.timeout.connect(self.refresh_if_due)
        self._startup_timer = QTimer(self)
        self._startup_timer.setSingleShot(True)
        self._startup_timer.timeout.connect(self.refresh_if_due)
        self._workerFinished.connect(self._on_worker_finished)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
        self._startup_timer.start(self._startup_delay_ms)
        self.snapshotChanged.emit(self.snapshot())

    def shutdown(self) -> None:
        self._timer.stop()
        self._startup_timer.stop()

    def snapshot(self) -> dict[str, Any]:
        return inspect_industry_snapshot(
            sector_path=self._sector_path,
            industry_path=self._industry_path,
            fresh_for_seconds=self._refresh_interval_seconds,
        )

    @Slot()
    def refresh_if_due(self) -> bool:
        snapshot = self.snapshot()
        self.snapshotChanged.emit(snapshot)
        if not industry_refresh_due(snapshot):
            return False
        return self.request_refresh(force=False)

    def request_refresh(self, *, force: bool = True) -> bool:
        if not force and not industry_refresh_due(self.snapshot()):
            return False
        with self._lock:
            if self._running:
                return False
            self._running = True
        self.refreshStarted.emit()
        self._refresh_thread = threading.Thread(
            target=self._refresh_worker,
            name="industry-board-refresh",
            daemon=True,
        )
        self._refresh_thread.start()
        return True

    def _refresh_worker(self) -> None:
        attempted_at = datetime.now().isoformat(timespec="seconds")
        try:
            result = self._scan_runner(write_outputs=True)
            sector_count = len(result.get("sector_rows") or [])
            industry_count = len(result.get("industry_rows") or [])
            if sector_count <= 0 or industry_count <= 0:
                raise RuntimeError(
                    f"incomplete provider result ({sector_count} sectors, {industry_count} industries)"
                )
            payload = {
                "ok": True,
                "attempted_at": attempted_at,
                "sector_count": sector_count,
                "industry_count": industry_count,
                "symbol_count": int(result.get("symbol_count") or 0),
                "error": "",
            }
        except Exception as exc:
            payload = {
                "ok": False,
                "attempted_at": attempted_at,
                "sector_count": 0,
                "industry_count": 0,
                "symbol_count": 0,
                "error": str(exc),
            }
        self._workerFinished.emit(payload)

    @Slot(object)
    def _on_worker_finished(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._running = False
        snapshot = self.snapshot()
        previous = _read_state(self._state_path)
        state = {
            "schema": "industry_board_snapshot_v1",
            "last_attempt_at": payload.get("attempted_at") or "",
            "last_success_at": (
                snapshot.get("as_of_iso")
                if payload.get("ok")
                else previous.get("last_success_at", "")
            ),
            "status": "ok" if payload.get("ok") else "failed",
            "error": payload.get("error") or "",
            "snapshot_id": snapshot.get("snapshot_id") or previous.get("snapshot_id", ""),
            "sector_count": (
                payload.get("sector_count")
                if payload.get("ok")
                else previous.get("sector_count", 0)
            ),
            "industry_count": (
                payload.get("industry_count")
                if payload.get("ok")
                else previous.get("industry_count", 0)
            ),
            "symbol_count": (
                payload.get("symbol_count")
                if payload.get("ok")
                else previous.get("symbol_count", 0)
            ),
            "sector_path": str(self._sector_path),
            "industry_path": str(self._industry_path),
        }
        try:
            _write_state(self._state_path, state)
        except OSError:
            pass
        result = {**payload, "snapshot": snapshot, "state": state}
        self.snapshotChanged.emit(snapshot)
        self.refreshFinished.emit(result)
