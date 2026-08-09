from __future__ import annotations

"""Main-thread stall watchdog: proof of where the UI thread actually blocks.

Part C rule C1 - measure before optimizing. A QTimer on the GUI thread stamps
a heartbeat; a daemon sampler thread watches that stamp, and when it stops
moving for longer than the threshold the main thread is, by definition,
inside something that is not the event loop. The sampler grabs that thread's
Python stack via ``sys._current_frames()`` and writes one JSONL record per
stall to the machine-local diagnostics dir.

Why a separate thread has to do the looking: a stalled event loop cannot run
the code that would notice it is stalled, so any main-thread-only detector
reports the stall after it is over, without a stack.

Off by default. Enable per machine with the ``ui_stall_watchdog`` local
setting, or per run with TRADINGBOTV3_UI_STALL_WATCHDOG=1 - a diagnostic that
samples stacks 100x/second is not something a trading session should carry
unasked.

Read the log back with::

    .venv\\Scripts\\python.exe scripts/ui/stall_watchdog.py

which prints the top offenders by total blocked time.
"""

import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer

#: Local-settings keys (``%LOCALAPPDATA%/.../local_settings.json``).
SETTING_ENABLED = "ui_stall_watchdog"
SETTING_THRESHOLD_MS = "ui_stall_watchdog_threshold_ms"
#: Per-run environment overrides; either one wins over the saved setting.
ENV_ENABLED = "TRADINGBOTV3_UI_STALL_WATCHDOG"
ENV_THRESHOLD_MS = "TRADINGBOTV3_UI_STALL_THRESHOLD_MS"

#: C1's bar: anything holding the GUI thread longer than this is a defect.
DEFAULT_THRESHOLD_MS = 50.0
#: Heartbeat cadence. Also the sampler's poll interval and therefore the
#: measurement's resolution: a stall is seen within one tick of starting.
HEARTBEAT_MS = 10
#: A runaway stall loop must not fill the disk during a session.
MAX_RECORDS_PER_SESSION = 2000
LOG_NAME = "ui_stalls.jsonl"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _flag(raw: str) -> bool:
    return str(raw or "").strip().lower() not in ("", "0", "false", "no", "off")


def is_enabled() -> bool:
    """Whether this machine/run wants the watchdog. Env beats saved setting."""
    raw = str(os.environ.get(ENV_ENABLED) or "").strip()
    if raw:
        return _flag(raw)
    try:
        from project_paths import get_local_setting

        return bool(get_local_setting(SETTING_ENABLED, False))
    except Exception:
        # A watchdog that cannot read its own setting must stay off, never
        # take the desk down on the way up.
        return False


def threshold_ms() -> float:
    raw = str(os.environ.get(ENV_THRESHOLD_MS) or "").strip()
    if not raw:
        try:
            from project_paths import get_local_setting

            raw = get_local_setting(SETTING_THRESHOLD_MS, "")
        except Exception:
            raw = ""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD_MS
    return value if value > 0 else DEFAULT_THRESHOLD_MS


def log_path() -> Path:
    from project_paths import get_diagnostics_dir

    return Path(get_diagnostics_dir()) / LOG_NAME


def _culprit(stack: list[str]) -> str:
    """Deepest frame inside this repo - the line to go read first.

    Site-packages frames (Qt, pandas, pyarrow) are where the time is *spent*,
    but the repo frame calling into them is what a fix can change.
    """
    for entry in reversed(stack):
        location = entry.split(" ", 1)[0]
        path = location.rsplit(":", 1)[0]
        try:
            candidate = Path(path).resolve()
        except (OSError, ValueError):
            continue
        if _REPO_ROOT in candidate.parents and "site-packages" not in path:
            try:
                return f"{candidate.relative_to(_REPO_ROOT).as_posix()}:{location.rsplit(':', 1)[1]}"
            except (ValueError, IndexError):
                return location
    return stack[-1].split(" ", 1)[0] if stack else "unknown"


class StallWatchdog(QObject):
    """Heartbeat on the GUI thread + sampler thread that logs the gaps."""

    def __init__(
        self,
        *,
        threshold_ms: float = DEFAULT_THRESHOLD_MS,
        log_path: Path | str | None = None,
        heartbeat_ms: int = HEARTBEAT_MS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._threshold_s = max(0.001, float(threshold_ms) / 1000.0)
        self._heartbeat_s = max(0.001, float(heartbeat_ms) / 1000.0)
        self._log_path = Path(log_path) if log_path is not None else None
        # Constructed on the GUI thread, so this is the thread to sample.
        self._main_thread_id = threading.get_ident()
        self._beat = time.perf_counter()
        # A counter, not the timestamp, decides "did it move": two beats can
        # land on the same float, and a wrapped compare would miss the resume.
        self._beat_seq = 0
        self._records = 0
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(int(heartbeat_ms))
        self._timer.timeout.connect(self._on_beat)

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._sampler is not None:
            return
        self._beat = time.perf_counter()
        self._timer.start()
        self._sampler = threading.Thread(
            target=self._sample_loop, name="ui-stall-watchdog", daemon=True
        )
        self._sampler.start()

    def stop(self) -> None:
        self._stop.set()
        self._timer.stop()
        sampler, self._sampler = self._sampler, None
        if sampler is not None:
            sampler.join(timeout=1.0)

    @property
    def records_written(self) -> int:
        return self._records

    # -- heartbeat -----------------------------------------------------
    def _on_beat(self) -> None:
        self._beat = time.perf_counter()
        self._beat_seq += 1

    # -- sampler thread ------------------------------------------------
    def _sample_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_s):
            seq = self._beat_seq
            started = self._beat
            # The heartbeat interval itself sits inside every gap; only the
            # excess over it is time the main thread was actually held.
            if (time.perf_counter() - started) - self._heartbeat_s < self._threshold_s:
                continue
            # Still blocked right now: this is the one moment the offending
            # frame is on the stack to be read.
            stack = self._capture_main_stack()
            samples = 1
            while not self._stop.wait(self._heartbeat_s):
                if self._beat_seq != seq:
                    break
                samples += 1
            if self._beat_seq != seq:
                gap_s = self._beat - started  # resumed: measured end to end
            else:
                gap_s = time.perf_counter() - started  # shutting down mid-stall
            self._write(gap_s, stack, samples)

    def _capture_main_stack(self) -> list[str]:
        frame = sys._current_frames().get(self._main_thread_id)
        if frame is None:
            return []
        try:
            return [
                f"{item.filename}:{item.lineno} {item.name}"
                for item in traceback.extract_stack(frame)
            ]
        except Exception:
            return []

    def _write(self, gap_s: float, stack: list[str], samples: int) -> None:
        if self._records >= MAX_RECORDS_PER_SESSION:
            return
        path = self._log_path if self._log_path is not None else log_path()
        gap_ms = gap_s * 1000.0
        record: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "gap_ms": round(gap_ms, 1),
            # What the main thread was actually held for, net of the cadence.
            "blocked_ms": round(max(0.0, gap_ms - self._heartbeat_s * 1000.0), 1),
            "threshold_ms": round(self._threshold_s * 1000.0, 1),
            "samples": samples,
            "culprit": _culprit(stack),
            "stack": stack,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            return  # diagnostics must never break the session
        self._records += 1


def install(parent: QObject | None = None) -> StallWatchdog | None:
    """Start the watchdog when this machine asked for it, else do nothing."""
    if not is_enabled():
        return None
    try:
        watchdog = StallWatchdog(
            threshold_ms=threshold_ms(), log_path=log_path(), parent=parent
        )
        watchdog.start()
        return watchdog
    except Exception:
        return None


# ---------------------------------------------------------------------
# Reading the log back
# ---------------------------------------------------------------------
def load_stalls(path: Path | str | None = None) -> list[dict]:
    target = Path(path) if path is not None else log_path()
    if not target.exists():
        return []
    records = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def summarize_stalls(path: Path | str | None = None, *, top: int = 12) -> list[dict]:
    """Top offenders by total time the GUI thread spent blocked in them."""
    grouped: dict[str, list[float]] = {}
    for record in load_stalls(path):
        culprit = str(record.get("culprit") or "unknown")
        try:
            blocked = float(record.get("blocked_ms") or 0.0)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(culprit, []).append(blocked)
    rows = [
        {
            "culprit": culprit,
            "count": len(values),
            "total_ms": round(sum(values), 1),
            "worst_ms": round(max(values), 1),
            "median_ms": round(sorted(values)[len(values) // 2], 1),
        }
        for culprit, values in grouped.items()
    ]
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return rows[:top]


def _main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else log_path()
    rows = summarize_stalls(target)
    if not rows:
        print(f"No stalls recorded in {target}")
        print(
            "Enable with: TRADINGBOTV3_UI_STALL_WATCHDOG=1, or set "
            f'"{SETTING_ENABLED}": true in local_settings.json'
        )
        return 0
    total = len(load_stalls(target))
    print(f"{total} stalls over {threshold_ms():.0f}ms in {target}\n")
    print(f"{'total':>9}  {'worst':>8}  {'median':>8}  {'n':>5}  culprit")
    for row in rows:
        print(
            f"{row['total_ms']:9.1f}  {row['worst_ms']:8.1f}  "
            f"{row['median_ms']:8.1f}  {row['count']:5d}  {row['culprit']}"
        )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    raise SystemExit(_main())
