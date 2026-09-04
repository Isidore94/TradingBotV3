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
