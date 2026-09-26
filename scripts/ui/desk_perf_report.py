"""One trading day's desk performance, read from the two live diagnostics logs.

Reads `ui_stalls.jsonl` (the stall watchdog) and `thread_cpu.jsonl` (the thread
CPU gauge) READ-ONLY and prints one compact table for a time window of one day:
GUI stalls, GC sweeps, Qt-thread CPU, memory, and the top Python culprits by
attributed blocked seconds. `--compare DAY2` prints two days side by side.

    python scripts/ui/desk_perf_report.py --day 2026-09-24
    python scripts/ui/desk_perf_report.py --day 2026-09-24 --compare 2026-09-25

Times are Pacific (the desk's clock). It measures only; nothing here writes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo

DESK_TZ = ZoneInfo("America/Los_Angeles")
DEFAULT_FROM = "06:30"
DEFAULT_TO = "13:05"
TOP_CULPRITS = 10

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TS_RE = re.compile(r'"ts":\s*"([^"]+)"')


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile; None for an empty sample."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    rank = int(-(-len(ordered) * float(fraction) // 1))
    rank = max(1, min(len(ordered), rank))
    return ordered[rank - 1]


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def parse_ts(text: str) -> datetime | None:
    """ISO timestamp -> aware datetime in desk time; a naive stamp is taken as desk time."""
    try:
        stamp = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=DESK_TZ)
    return stamp.astimezone(DESK_TZ)


def iter_window(
    path: Path | str, day: date, start: dt_time, end: dt_time
) -> Iterator[dict[str, Any]]:
    """Records of one JSONL log whose `ts` falls on `day` between `start` and `end` (desk time).

    The timestamp is read with a regex first so the ~100 MB stall log is not
    JSON-decoded line by line; only lines inside the window are parsed.
    """
    target = Path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _TS_RE.search(line, 0, 200)
            if match is None:
                continue
            stamp = parse_ts(match.group(1))
            if stamp is None or stamp.date() != day:
                continue
            clock = stamp.time()
            if clock < start or clock > end:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def app_exec_lines(app_source: Path | None = None) -> set[int]:
    """Line numbers of `return app.exec()` in scripts/ui/app.py, read at runtime."""
    source = app_source or (_REPO_ROOT / "scripts" / "ui" / "app.py")
    try:
        text = source.read_text(encoding="utf-8")
    except OSError:
        return set()
    return {
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip() == "return app.exec()"
    }


def _frame_parts(entry: str) -> tuple[str, str, str]:
    """`C:\\...\\app.py:2570 main` -> (`C:/.../app.py`, `2570`, `main`)."""
    location, _, function = str(entry or "").partition(" ")
    path, _, line = location.replace("\\", "/").rpartition(":")
    return path, line, function.strip()


def event_loop_lines_from_stacks(records: Iterable[dict[str, Any]]) -> set[int]:
    """app.py lines the desk that WROTE the log was idling on.

    A desk started from an older commit has `app.exec()` on a different line,
    so the current source alone misses it. A stack made only of launch_gui.py
    frames and then app.py's `main` is the bare event loop; its last line is
    that desk's exec line.
    """
    lines: set[int] = set()
    for record in records:
        stack = record.get("stack")
        if not isinstance(stack, list) or not stack:
            continue
        path, line, function = _frame_parts(stack[-1])
        if not path.endswith("ui/app.py") or function != "main":
            continue
        if all(_frame_parts(entry)[0].endswith("launch_gui.py") for entry in stack[:-1]):
            try:
                lines.add(int(line))
            except ValueError:
                continue
    return lines


def is_event_loop_frame(culprit: str, exec_lines: Iterable[int]) -> bool:
    """True for the bare event-loop frames: anything in launch_gui.py, or app.py's `app.exec()`."""
    text = str(culprit or "").replace("\\", "/")
    path, _, line = text.rpartition(":")
    if not path:
        path, line = text, ""
    if path.endswith("launch_gui.py"):
        return True
    if path.endswith("ui/app.py"):
        try:
            return int(line) in set(exec_lines)
        except ValueError:
            return False
    return False


def stall_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    blocked = []
    for record in records:
        try:
            blocked.append(float(record.get("blocked_ms") or 0.0))
        except (TypeError, ValueError):
            continue
    return {
        "stalls": len(blocked),
        "blocked_s": sum(blocked) / 1000.0,
        "p50_ms": percentile(blocked, 0.50),
        "p90_ms": percentile(blocked, 0.90),
        "max_ms": max(blocked) if blocked else None,
        "over_1s": sum(1 for value in blocked if value >= 1000.0),
    }


def culprit_seconds(
    records: Sequence[dict[str, Any]], exec_lines: Iterable[int]
) -> tuple[dict[str, float], dict[str, int], float]:
    """Blocked seconds per culprit frame, split across `culprit_samples` by sample share.

    Returns `(seconds_by_frame, stalls_by_frame, event_loop_seconds)`; the
    event-loop frames are left out of the first two and summed in the third.
    """
    lines = set(exec_lines)
    seconds: dict[str, float] = {}
    stalls: dict[str, int] = {}
    event_loop = 0.0
    for record in records:
        try:
            blocked_s = float(record.get("blocked_ms") or 0.0) / 1000.0
        except (TypeError, ValueError):
            continue
        samples = record.get("culprit_samples")
        shares: dict[str, float] = {}
        if isinstance(samples, dict):
            for frame, count in samples.items():
                try:
                    shares[str(frame)] = float(count)
                except (TypeError, ValueError):
                    continue
        total = sum(value for value in shares.values() if value > 0)
        if total <= 0:
            shares = {str(record.get("culprit") or "unknown"): 1.0}
            total = 1.0
        for frame, count in shares.items():
            if count <= 0:
                continue
            part = blocked_s * count / total
            if is_event_loop_frame(frame, lines):
                event_loop += part
                continue
            seconds[frame] = seconds.get(frame, 0.0) + part
            stalls[frame] = stalls.get(frame, 0) + 1
    return seconds, stalls, event_loop


def _gc_part(record: dict[str, Any], kind: str) -> dict[str, Any]:
    part = (record.get("gc") or {}).get(kind)
    return part if isinstance(part, dict) else {}


def gauge_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """GC sweeps/min and ms/min, Qt-thread core fraction and RSS over the gauge records."""
    out: dict[str, Any] = {"gauge_minutes": 0.0}
    minutes_total = 0.0
    per_kind: dict[str, dict[str, list[float]]] = {
        kind: {"sweeps": [], "ms": [], "rate": [], "max": []} for kind in ("full", "young")
    }
    gui_fraction: list[float] = []
    rss: list[float] = []
    for record in records:
        try:
            minutes = max(1e-6, float(record.get("interval_s") or 60.0) / 60.0)
        except (TypeError, ValueError):
            minutes = 1.0
        if isinstance(record.get("gc"), dict):
            minutes_total += minutes
            for kind, bucket in per_kind.items():
                part = _gc_part(record, kind)
                sweeps = float(part.get("sweeps") or 0)
                total_ms = float(part.get("total_ms") or 0.0)
                bucket["sweeps"].append(sweeps)
                bucket["ms"].append(total_ms)
                bucket["rate"].append(total_ms / minutes)
                if sweeps:
                    bucket["max"].append(float(part.get("max_ms") or 0.0))
        for row in record.get("top") or ():
            if isinstance(row, dict) and row.get("gui"):
                try:
                    gui_fraction.append(float(row.get("core_fraction") or 0.0))
                except (TypeError, ValueError):
                    pass
                break
        memory = record.get("memory")
        if isinstance(memory, dict) and memory.get("rss_mb") is not None:
            try:
                rss.append(float(memory["rss_mb"]))
            except (TypeError, ValueError):
                pass
    out["gauge_minutes"] = minutes_total
    for kind, bucket in per_kind.items():
        have = minutes_total > 0
        out[f"{kind}_sweeps_min"] = (sum(bucket["sweeps"]) / minutes_total) if have else None
        out[f"{kind}_ms_min"] = (sum(bucket["ms"]) / minutes_total) if have else None
        out[f"{kind}_ms_min_p90"] = percentile(bucket["rate"], 0.90)
        out[f"{kind}_max_sweep_ms"] = max(bucket["max"]) if bucket["max"] else None
    out["gui_core_mean"] = _mean(gui_fraction)
    out["gui_core_p90"] = percentile(gui_fraction, 0.90)
    out["rss_mean_mb"] = _mean(rss)
    out["rss_max_mb"] = max(rss) if rss else None
    return out


def build_report(
    day: date,
    start: dt_time,
    end: dt_time,
    *,
    stalls_path: Path | str,
    gauge_path: Path | str,
    exec_lines: Iterable[int] | None = None,
) -> dict[str, Any]:
    lines = set(app_exec_lines() if exec_lines is None else exec_lines)
    stall_records = list(iter_window(stalls_path, day, start, end))
    lines |= event_loop_lines_from_stacks(stall_records)
    gauge_records = list(iter_window(gauge_path, day, start, end))
    seconds, stalls, event_loop = culprit_seconds(stall_records, lines)
    ranked = sorted(seconds.items(), key=lambda item: item[1], reverse=True)
    report: dict[str, Any] = {
        "day": day.isoformat(),
        "from": start.strftime("%H:%M"),
        "to": end.strftime("%H:%M"),
        "gauge_records": len(gauge_records),
        **stall_stats(stall_records),
        **gauge_stats(gauge_records),
        "event_loop_s": event_loop,
        "event_loop_lines": sorted(lines),
        "culprits": [
            {"culprit": frame, "seconds": value, "stalls": stalls.get(frame, 0)}
            for frame, value in ranked
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
#: (label, key, decimals) - the metric rows, in print order.
METRICS: tuple[tuple[str, str, int], ...] = (
    ("stalls", "stalls", 0),
    ("GUI blocked s", "blocked_s", 1),
    ("stall p50 ms", "p50_ms", 0),
    ("stall p90 ms", "p90_ms", 0),
    ("stall max ms", "max_ms", 0),
    ("stalls over 1 s", "over_1s", 0),
    ("full gc sweeps/min", "full_sweeps_min", 2),
    ("full gc ms/min mean", "full_ms_min", 1),
    ("full gc ms/min p90", "full_ms_min_p90", 1),
    ("full gc max sweep ms", "full_max_sweep_ms", 1),
    ("young gc sweeps/min", "young_sweeps_min", 1),
    ("young gc ms/min mean", "young_ms_min", 1),
    ("young gc ms/min p90", "young_ms_min_p90", 1),
    ("young gc max sweep ms", "young_max_sweep_ms", 1),
    ("Qt thread core mean", "gui_core_mean", 3),
    ("Qt thread core p90", "gui_core_p90", 3),
    ("rss MB mean", "rss_mean_mb", 0),
    ("rss MB max", "rss_max_mb", 0),
    ("gauge minutes", "gauge_minutes", 0),
)


def _fmt(value: Any, decimals: int) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.{decimals}f}"


def _delta(now: Any, was: Any, decimals: int) -> str:
    if now is None or was is None:
        return "-"
    return f"{float(now) - float(was):+,.{decimals}f}"


def format_report(report: dict[str, Any], *, top: int = TOP_CULPRITS) -> str:
    lines = [f"Desk performance {report['day']} {report['from']}-{report['to']} PT", ""]
    for label, key, decimals in METRICS:
        lines.append(f"{label:<24} {_fmt(report.get(key), decimals):>12}")
    lines.append("")
    lines.append(f"Top {top} Python culprits (blocked s, split by sample share)")
    for row in report["culprits"][:top]:
        lines.append(f"{row['seconds']:>9,.1f}  {row['stalls']:>6}  {row['culprit']}")
    exec_text = ",".join(str(n) for n in report.get("event_loop_lines", ())) or "-"
    lines.append(
        f"event loop frames (excluded): {report['event_loop_s']:,.1f} s "
        f"(launch_gui.py, ui/app.py:{exec_text})"
    )
    return "\n".join(lines)


def format_compare(first: dict[str, Any], second: dict[str, Any], *, top: int = TOP_CULPRITS) -> str:
    a, b = first["day"], second["day"]
    lines = [f"Desk performance {first['from']}-{first['to']} PT: {a} vs {b}", ""]
    lines.append(f"{'metric':<24} {a:>12} {b:>12} {'delta':>12}")
    for label, key, decimals in METRICS:
        lines.append(
            f"{label:<24} {_fmt(first.get(key), decimals):>12} "
            f"{_fmt(second.get(key), decimals):>12} "
            f"{_delta(second.get(key), first.get(key), decimals):>12}"
        )
    lines.append("")
    lines.append(f"Top {top} Python culprits (blocked s)")
    lines.append(f"{a:>10} {b:>10} {'delta':>9}  frame")
    was = {row["culprit"]: row["seconds"] for row in first["culprits"]}
    now = {row["culprit"]: row["seconds"] for row in second["culprits"]}
    ranked = sorted(set(was) | set(now), key=lambda f: max(was.get(f, 0.0), now.get(f, 0.0)), reverse=True)
    for frame in ranked[:top]:
        lines.append(
            f"{was.get(frame, 0.0):>10,.1f} {now.get(frame, 0.0):>10,.1f} "
            f"{now.get(frame, 0.0) - was.get(frame, 0.0):>+9,.1f}  {frame}"
        )
    lines.append(
        f"event loop frames (excluded): {first['event_loop_s']:,.1f} s vs {second['event_loop_s']:,.1f} s "
        f"(ui/app.py lines {first.get('event_loop_lines')} / {second.get('event_loop_lines')})"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _clock(text: str) -> dt_time:
    return datetime.strptime(text, "%H:%M").time()


def _clock_end(text: str) -> dt_time:
    """`--to 13:05` includes the whole 13:05 minute."""
    return _clock(text).replace(second=59, microsecond=999999)


def _day(text: str) -> date:
    return date.fromisoformat(text)


def _default_log(module: str) -> Path:
    scripts_dir = str(_REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    if module == "stalls":
        from ui import stall_watchdog

        return stall_watchdog.log_path()
    from ui import thread_cpu_gauge

    return thread_cpu_gauge.log_path()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="desk_perf_report", description=__doc__.splitlines()[0])
    parser.add_argument("--day", required=True, type=_day, help="YYYY-MM-DD")
    parser.add_argument("--from", dest="start", default=DEFAULT_FROM, type=_clock, help="HH:MM PT")
    parser.add_argument("--to", dest="end", default=DEFAULT_TO, type=_clock_end, help="HH:MM PT, inclusive")
    parser.add_argument("--compare", type=_day, default=None, help="a second day, printed side by side")
    parser.add_argument("--stalls", default=None, help="ui_stalls.jsonl (default: the live log)")
    parser.add_argument("--gauge", default=None, help="thread_cpu.jsonl (default: the live log)")
    parser.add_argument("--top", type=int, default=TOP_CULPRITS)
    return parser


def main(argv: Sequence[str] | None = None, *, stream=None) -> int:
    stream = stream or sys.stdout
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    stalls_path = Path(args.stalls) if args.stalls else _default_log("stalls")
    gauge_path = Path(args.gauge) if args.gauge else _default_log("gauge")
    exec_lines = app_exec_lines()
    first = build_report(
        args.day, args.start, args.end,
        stalls_path=stalls_path, gauge_path=gauge_path, exec_lines=exec_lines,
    )
    if args.compare is None:
        print(format_report(first, top=args.top), file=stream)
        return 0
    second = build_report(
        args.compare, args.start, args.end,
        stalls_path=stalls_path, gauge_path=gauge_path, exec_lines=exec_lines,
    )
    print(format_compare(first, second, top=args.top), file=stream)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
