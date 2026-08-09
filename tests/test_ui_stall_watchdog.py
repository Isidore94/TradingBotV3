"""The main-thread stall watchdog must catch a real block, with the stack.

Part C rule C1 turns on measurement before optimization, so the measurement
itself needs a test: a watchdog that silently records nothing would make an
unoptimized chart path look clean.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _pump(app, seconds: float, *, until=None) -> bool:
    """Spin the event loop for ``seconds``, stopping early once ``until`` holds."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.005)
    return until() if until is not None else False


def _block_the_gui_thread(seconds: float) -> None:
    """A deliberate synchronous hold, standing in for an inline file read."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass


def test_watchdog_records_a_blocking_call_with_its_stack(tmp_path):
    app = _qt_app()
    if app is None:
        pytest.skip("PySide6 unavailable")
    from ui.stall_watchdog import StallWatchdog, load_stalls

    log = tmp_path / "ui_stalls.jsonl"
    watchdog = StallWatchdog(threshold_ms=30, log_path=log, heartbeat_ms=5)
    watchdog.start()
    try:
        # Let the heartbeat establish itself before the deliberate stall, so
        # the record under test is the block and not the startup gap.
        _pump(app, 0.3)
        _block_the_gui_thread(0.25)
        found = _pump(app, 3.0, until=lambda: bool(load_stalls(log)))
    finally:
        watchdog.stop()

    assert found, "a 250ms main-thread block produced no stall record"
    records = load_stalls(log)
    # Pick the record for OUR deliberate block rather than whichever stall
    # happens to be longest. Run inside the full suite the machine is busy
    # enough to produce other stalls (a GC pause, a loaded scheduler), and
    # keying on the maximum makes this flaky instead of meaningful.
    blocks = [
        record
        for record in records
        if any("_block_the_gui_thread" in frame for frame in record["stack"])
    ]
    assert blocks, (
        "no stall was attributed to the deliberate block; recorded culprits: "
        f"{[record['culprit'] for record in records]}"
    )
    worst = max(blocks, key=lambda record: record["blocked_ms"])
    # The measured hold must resemble the real one. The lower bound is the
    # real assertion; the upper is loose on purpose, because the gap is
    # measured between heartbeats and a loaded machine can be slow to run
    # the first one after the block ends.
    assert 150.0 <= worst["blocked_ms"] <= 2000.0
    assert worst["threshold_ms"] == 30.0
    assert worst["samples"] >= 1
    assert worst["ts"].strip()
    # A stall with no stack is not actionable; the culprit must name the
    # repo frame that was holding the thread.
    assert worst["stack"], "no stack captured for the stall"
    assert "test_ui_stall_watchdog.py" in worst["culprit"], worst["culprit"]
    assert any("_block_the_gui_thread" in frame for frame in worst["stack"])


def test_idle_loop_records_nothing(tmp_path):
    app = _qt_app()
    if app is None:
        pytest.skip("PySide6 unavailable")
    from ui.stall_watchdog import StallWatchdog, load_stalls

    log = tmp_path / "ui_stalls.jsonl"
    watchdog = StallWatchdog(threshold_ms=50, log_path=log, heartbeat_ms=5)
    watchdog.start()
    try:
        _pump(app, 0.6)
    finally:
        watchdog.stop()
    # A responsive loop must stay silent, or every real stall drowns in noise.
    assert load_stalls(log) == []


def test_summary_ranks_offenders_by_total_blocked_time(tmp_path):
    import json

    from ui.stall_watchdog import summarize_stalls

    log = tmp_path / "ui_stalls.jsonl"
    rows = [
        {"culprit": "a.py:1", "blocked_ms": 60.0},
        {"culprit": "b.py:2", "blocked_ms": 200.0},
        {"culprit": "a.py:1", "blocked_ms": 500.0},
        {"culprit": "a.py:1", "blocked_ms": 100.0},
    ]
    log.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    summary = summarize_stalls(log)
    assert [row["culprit"] for row in summary] == ["a.py:1", "b.py:2"]
    assert summary[0]["total_ms"] == 660.0
    assert summary[0]["worst_ms"] == 500.0
    assert summary[0]["count"] == 3


def test_watchdog_is_off_unless_asked_for(monkeypatch):
    from ui import stall_watchdog

    monkeypatch.delenv(stall_watchdog.ENV_ENABLED, raising=False)
    monkeypatch.setattr(
        stall_watchdog, "is_enabled", lambda: False
    )
    assert stall_watchdog.install() is None

    monkeypatch.setenv(stall_watchdog.ENV_ENABLED, "1")
    # Re-read the real predicate now that the env says yes.
    from importlib import reload

    reloaded = reload(stall_watchdog)
    assert reloaded.is_enabled() is True
    monkeypatch.setenv(reloaded.ENV_ENABLED, "0")
    assert reloaded.is_enabled() is False


def test_threshold_defaults_and_overrides(monkeypatch):
    from ui import stall_watchdog

    monkeypatch.setenv(stall_watchdog.ENV_THRESHOLD_MS, "120")
    assert stall_watchdog.threshold_ms() == 120.0
    monkeypatch.setenv(stall_watchdog.ENV_THRESHOLD_MS, "not-a-number")
    assert stall_watchdog.threshold_ms() == stall_watchdog.DEFAULT_THRESHOLD_MS
