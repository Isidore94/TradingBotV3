"""desk_perf_report: one day's stalls, GC, Qt-thread CPU and culprits from the two logs."""

from __future__ import annotations

import ast
import io
import json
import sys
from datetime import date, time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ui import desk_perf_report as rep  # noqa: E402

DAY = "2026-09-24"
EXEC_LINE = sorted(rep.app_exec_lines())[0]
LOOP_STACK = [
    r"C:\repo\launch_gui.py:136 <module>",
    r"C:\repo\launch_gui.py:127 main",
    r"C:\repo\scripts\ui\app.py:2385 main",
]


def _stall(clock: str, blocked: float, samples: dict[str, int], stack=None, day=DAY) -> dict:
    return {
        "ts": f"{day}T{clock}.000-07:00",
        "blocked_ms": blocked,
        "culprit": max(samples, key=samples.get),
        "culprit_samples": samples,
        "stack": stack or ["C:\\repo\\scripts\\x.py:1 f"],
    }


def _gauge(clock: str, *, full=(0, 0.0, 0.0), young=(0, 0.0, 0.0), gui=0.1, rss=1000.0, day=DAY) -> dict:
    return {
        "ts": f"{day}T{clock}-07:00",
        "interval_s": 60.0,
        "top": [
            {"thread": "worker", "core_fraction": 0.9, "gui": False},
            {"thread": "MainThread", "core_fraction": gui, "gui": True},
        ],
        "memory": {"rss_mb": rss, "commit_mb": rss + 100, "gc_objects": 5},
        "gc": {
            "young": {"sweeps": young[0], "total_ms": young[1], "max_ms": young[2]},
            "full": {"sweeps": full[0], "total_ms": full[1], "max_ms": full[2]},
        },
    }


def _write(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + "not json\n", encoding="utf-8")
    return path


@pytest.fixture()
def logs(tmp_path):
    stalls = _write(
        tmp_path / "ui_stalls.jsonl",
        [
            _stall("06:29:59", 5000.0, {"scripts/a.py:1": 1}),  # before the window
            _stall("07:00:00", 100.0, {"scripts/a.py:1": 1}),
            _stall("08:00:00", 300.0, {"scripts/a.py:1": 1, "scripts/b.py:2": 2}),
            _stall("09:00:00", 1200.0, {f"scripts/ui/app.py:{EXEC_LINE}": 3, "scripts/b.py:2": 1}),
            # an older desk's exec line, learned from a bare event-loop stack
            _stall("10:00:00", 400.0, {"scripts/ui/app.py:2385": 3, "launch_gui.py:127": 1}, stack=LOOP_STACK),
            _stall("13:05:30", 200.0, {"scripts/ui/app.py:999": 1}),  # inside the 13:05 minute
            _stall("13:06:00", 9000.0, {"scripts/a.py:1": 1}),  # after the window
            _stall("09:00:00", 7000.0, {"scripts/a.py:1": 1}, day="2026-09-25"),
        ],
    )
    gauge = _write(
        tmp_path / "thread_cpu.jsonl",
        [
            _gauge("06:00:00", full=(9, 9000.0, 9000.0)),  # before the window
            _gauge("07:00:00", full=(1, 300.0, 300.0), young=(20, 100.0, 20.0), gui=0.1, rss=1000.0),
            _gauge("07:01:00", full=(0, 0.0, 0.0), young=(30, 200.0, 40.0), gui=0.2, rss=2000.0),
            _gauge("07:02:00", full=(2, 900.0, 800.0), young=(40, 300.0, 30.0), gui=0.3, rss=3000.0),
            _gauge("09:00:00", full=(1, 50.0, 50.0), day="2026-09-25"),
        ],
    )
    return stalls, gauge


def _report(logs, day=DAY):
    stalls, gauge = logs
    return rep.build_report(
        date.fromisoformat(day), time(6, 30), time(13, 5, 59, 999999),
        stalls_path=stalls, gauge_path=gauge,
    )


def test_stall_numbers_cover_the_window_only(logs):
    report = _report(logs)
    assert report["stalls"] == 5
    assert report["blocked_s"] == pytest.approx(2.2)
    assert report["p50_ms"] == 300.0
    assert report["p90_ms"] == 1200.0
    assert report["max_ms"] == 1200.0
    assert report["over_1s"] == 1


def test_culprits_are_split_by_sample_share_and_skip_the_event_loop(logs):
    report = _report(logs)
    seconds = {row["culprit"]: row["seconds"] for row in report["culprits"]}
    # a: 0.1 + 0.1 ; b: 0.2 + 0.3 ; app.py:999 is a real frame, kept
    assert seconds == pytest.approx({"scripts/a.py:1": 0.2, "scripts/b.py:2": 0.5, "scripts/ui/app.py:999": 0.2})
    assert [row["culprit"] for row in report["culprits"]][0] == "scripts/b.py:2"
    # 0.9 s of the current exec line + 0.4 s of the old line and launch_gui.py
    assert report["event_loop_s"] == pytest.approx(1.3)
    assert EXEC_LINE in report["event_loop_lines"] and 2385 in report["event_loop_lines"]


def test_exec_line_is_read_from_app_py_at_runtime():
    source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8").splitlines()
    assert rep.app_exec_lines()
    for number in rep.app_exec_lines():
        assert source[number - 1].strip() == "return app.exec()"
    assert rep.is_event_loop_frame("launch_gui.py:127", set())
    assert not rep.is_event_loop_frame("scripts/ui/app.py:12", {EXEC_LINE})


def test_gauge_numbers(logs):
    report = _report(logs)
    assert report["gauge_minutes"] == pytest.approx(3.0)
    assert report["full_sweeps_min"] == pytest.approx(1.0)
    assert report["full_ms_min"] == pytest.approx(400.0)
    assert report["full_ms_min_p90"] == pytest.approx(900.0)
    assert report["full_max_sweep_ms"] == pytest.approx(800.0)
    assert report["young_sweeps_min"] == pytest.approx(30.0)
    assert report["young_ms_min"] == pytest.approx(200.0)
    assert report["young_max_sweep_ms"] == pytest.approx(40.0)
    assert report["gui_core_mean"] == pytest.approx(0.2)
    assert report["gui_core_p90"] == pytest.approx(0.3)
    assert report["rss_mean_mb"] == pytest.approx(2000.0)
    assert report["rss_max_mb"] == pytest.approx(3000.0)


def test_cli_prints_one_day_and_counts_the_whole_to_minute(logs):
    stalls, gauge = logs
    out = io.StringIO()
    before = (stalls.read_bytes(), gauge.read_bytes())
    code = rep.main(["--day", DAY, "--stalls", str(stalls), "--gauge", str(gauge)], stream=out)
    text = out.getvalue()
    assert code == 0
    assert "stalls over 1 s" in text and "full gc ms/min mean" in text
    assert "scripts/b.py:2" in text
    lines = {line[:24].strip(): line[24:].strip() for line in text.splitlines()}
    assert lines["stalls"] == "5"
    assert (stalls.read_bytes(), gauge.read_bytes()) == before  # read-only


def test_compare_prints_both_days_and_the_delta(logs):
    stalls, gauge = logs
    out = io.StringIO()
    rep.main(
        ["--day", DAY, "--compare", "2026-09-25", "--stalls", str(stalls), "--gauge", str(gauge)],
        stream=out,
    )
    row = next(line for line in out.getvalue().splitlines() if line.startswith("stalls "))
    assert row.split()[1:] == ["5", "1", "-4"]
    assert "2026-09-24" in out.getvalue() and "2026-09-25" in out.getvalue()


def test_module_imports_nothing_from_scripts_at_module_top():
    tree = ast.parse((SCRIPTS_DIR / "ui" / "desk_perf_report.py").read_text(encoding="utf-8"))
    offenders = []
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            head = name.split(".")[0]
            if (SCRIPTS_DIR / f"{head}.py").exists() or (SCRIPTS_DIR / head).is_dir():
                offenders.append(name)
    assert offenders == []
