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
from collections import Counter
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
#: Stacks captured per stall. A five-minute freeze must not spend itself
#: formatting tracebacks; 240 samples is four minutes of heartbeats.
MAX_SAMPLES_PER_STALL = 240
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
        # "<frozen importlib._bootstrap>" and friends are not paths; resolving
        # them yields a CWD-relative name that looks repo-local and would be
        # reported as the culprit for every lazy import.
        if not path or path.startswith("<"):
            continue
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
            # Still blocked right now. Sample for as long as it lasts, not
            # once at the start: the first stack says where a stall BEGAN,
            # which is a different question from where its time went. On
            # 2026-08-21 that difference left 56% of stalls attributed to
            # `app.exec()` - true, and useless.
            stack = self._capture_main_stack()
            culprits: Counter[str] = Counter()
            # One representative stack per culprit - the first time that frame
            # was seen. The record then carries the stack that BELONGS to the
            # frame it names, rather than whichever sample happened to be last.
            stacks: dict[str, list[str]] = {}
            first = _culprit(stack)
            culprits[first] += 1
            stacks[first] = stack
            samples = 1
            while not self._stop.wait(self._heartbeat_s):
                if self._beat_seq != seq:
                    break
                samples += 1
                # Sampling costs one sys._current_frames() per heartbeat, and
                # only while the main thread is already stuck - it competes
                # with nothing. Capped so a five-minute freeze cannot spend
                # itself formatting tracebacks.
                if samples <= MAX_SAMPLES_PER_STALL:
                    later = self._capture_main_stack()
                    if later:
                        name = _culprit(later)
                        culprits[name] += 1
                        stacks.setdefault(name, later)
            if self._beat_seq != seq:
                gap_s = self._beat - started  # resumed: measured end to end
            else:
                gap_s = time.perf_counter() - started  # shutting down mid-stall
            self._write(gap_s, stack, samples, culprits, stacks)

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

    def _write(
        self,
        gap_s: float,
        stack: list[str],
        samples: int,
        culprits: "Counter[str] | None" = None,
        stacks: dict[str, list[str]] | None = None,
    ) -> None:
        if self._records >= MAX_RECORDS_PER_SESSION:
            return
        path = self._log_path if self._log_path is not None else log_path()
        gap_ms = gap_s * 1000.0
        counted = culprits or Counter([_culprit(stack)])
        # The frame seen in the MOST samples, not the first one seen. Ties go
        # to the frame sampled first, which keeps a two-sample stall reading
        # the way it always did.
        modal = counted.most_common(1)[0][0] if counted else _culprit(stack)
        # The stack shown is the one that produced the frame being named.
        stack = (stacks or {}).get(modal, stack)
        record: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "gap_ms": round(gap_ms, 1),
            # What the main thread was actually held for, net of the cadence.
            "blocked_ms": round(max(0.0, gap_ms - self._heartbeat_s * 1000.0), 1),
            "threshold_ms": round(self._threshold_s * 1000.0, 1),
            "samples": samples,
            "culprit": modal,
            # Where the time actually went, as {frame: samples}. A single-frame
            # entry reads exactly like the old one-sample record.
            "culprit_samples": dict(counted.most_common(8)),
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


def session_summary(path: Path | str | None = None) -> dict:
    """The headline numbers for one session's log.

    These are what a fluidity pass is judged on, and they are the shape the
    2026-08-21 baseline was recorded in: how many hitches, how long the typical
    one was, how bad the tail got, and how much of the session the trader spent
    waiting. See docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md.
    """
    records = load_stalls(path)
    if not records:
        return {"stalls": 0}
    blocked = sorted(
        float(record.get("blocked_ms") or 0.0) for record in records
    )
    stamps = sorted(str(record.get("ts") or "") for record in records if record.get("ts"))
    count = len(blocked)

    def _at(fraction: float) -> float:
        return blocked[min(count - 1, max(0, int(fraction * count) - 1))]

    return {
        "stalls": count,
        "median_ms": round(blocked[count // 2], 1),
        "p90_ms": round(_at(0.90), 1),
        "p99_ms": round(_at(0.99), 1),
        "worst_ms": round(blocked[-1], 1),
        "total_blocked_s": round(sum(blocked) / 1000.0, 1),
        "over_1s": sum(1 for value in blocked if value >= 1000.0),
        "over_5s": sum(1 for value in blocked if value >= 5000.0),
        "first_ts": stamps[0] if stamps else "",
        "last_ts": stamps[-1] if stamps else "",
    }


def _print_summary(label: str, summary: dict) -> None:
    if not summary.get("stalls"):
        print(f"{label}: no stalls recorded")
        return
    print(
        f"{label}: {summary['stalls']} stalls  "
        f"median {summary['median_ms']:.0f}ms  p90 {summary['p90_ms']:.0f}ms  "
        f"worst {summary['worst_ms']:.0f}ms  "
        f"total {summary['total_blocked_s']:.0f}s blocked  "
        f"(>=1s: {summary['over_1s']}, >=5s: {summary['over_5s']})"
    )
    if summary.get("first_ts"):
        print(f"{'':>{len(label)}}  window {summary['first_ts'][11:19]} -> {summary['last_ts'][11:19]}")


def _print_histograms(path, limit: int = 5) -> None:
    """Where the time went INSIDE the worst stalls.

    Only records written after 2026-08-21 carry ``culprit_samples``; older ones
    were a single stack captured at detection, and say where a stall began
    rather than where it spent itself.
    """
    records = [
        record
        for record in load_stalls(path)
        if record.get("culprit_samples")
    ]
    if not records:
        return
    records.sort(key=lambda record: float(record.get("blocked_ms") or 0.0), reverse=True)
    print("\nworst stalls, by where their samples actually landed:")
    for record in records[:limit]:
        blocked = float(record.get("blocked_ms") or 0.0)
        print(f"  {blocked:9.0f}ms  {record.get('ts', '')[11:19]}")
        for frame, hits in (record.get("culprit_samples") or {}).items():
            print(f"{'':>14}{hits:5d} samples  {frame}")


def _main() -> int:
    # `stall_watchdog.py [LOG] [--compare BASELINE]`. Hand-parsed rather than
    # argparse-d because this module is imported by the GUI at startup and
    # stays deliberately import-light.
    positional: list[str] = []
    baseline: Path | None = None
    remaining = list(sys.argv[1:])
    while remaining:
        value = remaining.pop(0)
        if value == "--compare":
            if remaining:
                baseline = Path(remaining.pop(0))
            continue
        if value.startswith("--"):
            continue
        positional.append(value)
    target = Path(positional[0]) if positional else log_path()
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
    _print_summary("this session", session_summary(target))
    if baseline is not None:
        _print_summary("baseline    ", session_summary(baseline))
    print()
    print(f"{'total':>9}  {'worst':>8}  {'median':>8}  {'n':>5}  culprit")
    for row in rows:
        print(
            f"{row['total_ms']:9.1f}  {row['worst_ms']:8.1f}  "
            f"{row['median_ms']:8.1f}  {row['count']:5d}  {row['culprit']}"
        )
    _print_histograms(target)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    raise SystemExit(_main())
