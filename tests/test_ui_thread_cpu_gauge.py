"""The per-thread CPU gauge names the thread starving the GUI.

The stall watchdog samples the GUI thread's own stack and so cannot name a
stall another thread caused by holding the interpreter lock; on 2026-09-03 the
research tee thread ran at 91% of GIL samples for eight hours and nothing on
the desk said so. This gauge is the one-line answer, and it has to be right
about WHICH thread and never blame the GUI thread for being starved.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ui.thread_cpu_gauge import ThreadCpuGauge, load_records, summarize, supported  # noqa: E402


def test_tick_names_the_hot_thread_and_never_the_gui_thread(tmp_path, caplog):
    log = tmp_path / "thread_cpu.jsonl"
    gauge = ThreadCpuGauge(
        interval_seconds=60, hot_fraction=0.5, log_path=log, logger=logging.getLogger("gauge-test")
    )
    main_id = threading.main_thread().native_id
    before = {main_id: ("MainThread", 1.0), 7: ("warehouse-m5-tee", 10.0), 8: ("idle", 3.0)}
    after = {main_id: ("MainThread", 1.1), 7: ("warehouse-m5-tee", 65.0), 8: ("idle", 3.0), 9: ("newborn", 5.0)}

    with caplog.at_level(logging.WARNING, logger="gauge-test"):
        record = gauge.tick(before, after, 60.0)

    assert record["hot"] == ["warehouse-m5-tee"]
    top = record["top"][0]
    assert top["thread"] == "warehouse-m5-tee" and abs(top["core_fraction"] - 55 / 60) < 0.01
    assert all(row["thread"] != "newborn" for row in record["top"]), "a thread born mid-interval has no baseline"
    assert gauge.records_written == 1 and load_records(log)[0]["hot"] == ["warehouse-m5-tee"]
    assert any("warehouse-m5-tee" in message for message in caplog.messages)

    # The GUI thread is the one being starved; it is never reported as hot.
    starved = gauge.tick({main_id: ("MainThread", 0.0)}, {main_id: ("MainThread", 60.0)}, 60.0)
    assert starved["hot"] == [] and starved["top"][0]["gui"] is True


@pytest.mark.skipif(not supported(), reason="thread CPU times are read on Windows and Linux only")
def test_a_spinning_thread_is_measured_from_the_os(tmp_path):
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            pass

    worker = threading.Thread(target=spin, name="spinner", daemon=True)
    worker.start()
    gauge = ThreadCpuGauge(interval_seconds=0.4, hot_fraction=0.3, log_path=tmp_path / "t.jsonl")
    gauge.start()
    try:
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and not gauge.hot_seen:
            time.sleep(0.05)
    finally:
        stop.set()
        gauge.stop()
        worker.join(1.0)
    assert any(row["thread"] == "spinner" for row in gauge.hot_seen), gauge.last_record


def test_the_summary_reads_the_log_back(tmp_path):
    log = tmp_path / "thread_cpu.jsonl"
    gauge = ThreadCpuGauge(log_path=log)
    main_id = threading.main_thread().native_id
    gauge.tick({main_id: ("MainThread", 0.0), 5: ("tee", 0.0)}, {main_id: ("MainThread", 1.0), 5: ("tee", 50.0)}, 60.0)
    gauge.tick({main_id: ("MainThread", 1.0), 5: ("tee", 50.0)}, {main_id: ("MainThread", 2.0), 5: ("tee", 90.0)}, 60.0)
    rows = summarize(log)
    assert rows[0] == {"thread": "tee", "cpu_s": 90.0, "hot_ticks": 2}
    assert rows[1]["thread"] == "MainThread" and rows[1]["hot_ticks"] == 0


# ---------------------------------------------------------------------
# Memory and collector fields (2026-09-23: 20-44 s of GUI sweeps per 10 min
# with no record of heap size or sweep cost to explain them).
# ---------------------------------------------------------------------
def test_each_record_carries_process_memory_and_the_object_count(tmp_path):
    from ui import thread_cpu_gauge

    gauge = ThreadCpuGauge(interval_seconds=60, log_path=tmp_path / "t.jsonl")
    record = gauge.tick({}, {}, 60.0)

    memory = record["memory"]
    assert len(memory["gc_counts"]) == 3
    assert memory["gc_objects"] > 0 and memory["gc_objects_ms"] >= 0.0
    if sys.platform.startswith("win"):
        assert memory["rss_mb"] > 1.0 and memory["commit_mb"] > 1.0
        assert memory["peak_commit_mb"] >= memory["commit_mb"]
    assert load_records(tmp_path / "t.jsonl")[0]["memory"]["gc_objects"] == memory["gc_objects"]
    assert thread_cpu_gauge.memory_curve(tmp_path / "t.jsonl")[0]["gc_objects"] == memory["gc_objects"]


def test_the_object_count_switches_itself_off_when_it_costs_too_much(tmp_path, monkeypatch):
    from ui import thread_cpu_gauge

    monkeypatch.setattr(thread_cpu_gauge, "OBJECT_COUNT_BUDGET_MS", -1.0)
    gauge = ThreadCpuGauge(interval_seconds=60, log_path=tmp_path / "t.jsonl")
    first = gauge.tick({}, {}, 60.0)
    second = gauge.tick({}, {}, 60.0)

    assert "gc_objects" in first["memory"], "measured once, so the log says what it cost"
    assert "gc_objects" not in second["memory"], "then never again this session"
    assert "gc_counts" in second["memory"]


def test_gc_timer_times_young_and_full_sweeps_and_drains_per_tick(tmp_path):
    from ui.thread_cpu_gauge import GcTimer

    now = [0.0]
    timer = GcTimer(clock=lambda: now[0])

    def sweep(generation, seconds, collected):
        timer._callback("start", {"generation": generation})
        now[0] += seconds
        timer._callback("stop", {"generation": generation, "collected": collected, "uncollectable": 0})

    sweep(0, 0.010, 5)
    sweep(0, 1.500, 80)
    sweep(2, 0.700, 1000)

    gauge = ThreadCpuGauge(interval_seconds=60, log_path=tmp_path / "t.jsonl")
    gauge.gc_timer = timer
    record = gauge.tick({}, {}, 60.0)

    young, full = record["gc"]["young"], record["gc"]["full"]
    assert young["sweeps"] == 2 and young["total_ms"] == 1510.0 and young["max_ms"] == 1500.0
    assert young["freed"] == 85
    assert full["sweeps"] == 1 and full["max_ms"] == 700.0 and full["freed"] == 1000
    assert record["gc"]["last_full"]["ms"] == 700.0

    # Drained: the next minute starts from zero but remembers the last full sweep.
    again = gauge.tick({}, {}, 60.0)
    assert again["gc"]["young"]["sweeps"] == 0 and again["gc"]["full"]["sweeps"] == 0
    assert again["gc"]["last_full"]["ms"] == 700.0


def test_gc_timer_sees_a_real_collection_and_uninstalls_cleanly():
    import gc

    from ui.thread_cpu_gauge import GcTimer

    timer = GcTimer()
    timer.install()
    timer.install()  # idempotent
    try:
        assert gc.callbacks.count(timer._callback) == 1
        gc.collect(0)
        gc.collect(2)
    finally:
        timer.uninstall()
    assert timer._callback not in gc.callbacks
    drained = timer.drain()
    assert drained["young"]["sweeps"] >= 1 and drained["full"]["sweeps"] >= 1
