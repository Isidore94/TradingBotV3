"""Closing the desk must never wait forever on a background read.

Found in live use 2026-08-26: the trader closed the window, it "froze for a few
seconds", and the process outlived the window - the console it was launched
from stayed open. Three shutdown paths joined their reader with a bare
`worker.wait()`, which has no upper bound.

The worst of them is the warehouse readout, whose reader is on the DAS - the
one read in the whole desk that can block for minutes when the share is unwell,
which is exactly the condition under which a trader gives up and closes the
app.

This is the same lesson `_GuiGcController` paid for on 2026-08-21: a wait with
no bound is a hang waiting for a slow day. Every wait here now carries one.

The tradeoff is stated rather than hidden. On timeout the reader is disowned
and parked in a module-level list instead of being dropped, because dropping
the last reference to a running QThread destroys its C++ half mid-run - a crash
rather than a leak. These are reads: no writes, no side effects, and the
process is leaving anyway.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the readers are QThreads")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _clear_abandoned():
    from ui import read_worker

    read_worker._abandoned.clear()
    yield
    read_worker._abandoned.clear()


def test_a_finished_reader_joins_immediately():
    from ui.read_worker import ReadWorker, join_worker

    worker = ReadWorker(lambda: 42)
    worker.start()
    assert join_worker(worker) is True


def test_a_stuck_reader_does_not_hold_shutdown_open():
    """The defect itself: a bare `wait()` here is unbounded."""
    from ui.read_worker import ReadWorker, join_worker

    release = threading.Event()
    started = threading.Event()

    def stuck():
        started.set()
        release.wait(30)
        return None

    worker = ReadWorker(stuck)
    worker.start()
    assert started.wait(10), "the reader never started"

    began = time.perf_counter()
    finished = join_worker(worker, timeout_ms=300)
    elapsed = time.perf_counter() - began

    try:
        assert finished is False, "a stuck reader reported that it finished"
        assert elapsed < 5.0, (
            f"shutdown waited {elapsed:.1f}s on a reader that was never going "
            "to finish"
        )
    finally:
        release.set()
        worker.wait(10000)


def test_an_abandoned_reader_is_kept_referenced_not_dropped():
    """Dropping a running QThread destroys its C++ half - a crash, not a leak."""
    from ui import read_worker
    from ui.read_worker import ReadWorker, join_worker

    release = threading.Event()
    started = threading.Event()

    def stuck():
        started.set()
        release.wait(30)
        return None

    worker = ReadWorker(stuck)
    worker.start()
    assert started.wait(10)
    try:
        join_worker(worker, timeout_ms=200)
        assert worker in read_worker._abandoned
        assert worker.parent() is None, "an abandoned reader kept its parent"
    finally:
        release.set()
        worker.wait(10000)


def test_join_tolerates_none_and_never_raises():
    from ui.read_worker import join_worker

    assert join_worker(None) is True


def test_no_shutdown_path_still_uses_an_unbounded_wait():
    """Source-level, because this is a defect of OMISSION.

    A new panel that grows a reader and writes `worker.wait()` in its shutdown
    reintroduces exactly the hang this file exists to prevent, and no
    behavioural test would catch it until a trader could not close the desk.
    """
    offenders: list[str] = []
    for path in sorted((SCRIPTS_DIR / "ui").rglob("*.py")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped.endswith(".wait()") or "join_worker" in stripped:
                continue
            # A subprocess handle is a different animal: `proc.wait()` in
            # `scan_service` is reached only once `poll()` has already reported
            # the child gone, so it returns at once and is not a shutdown join.
            if stripped.startswith(("proc.wait", "returncode = proc.wait")):
                continue
            offenders.append(f"{path.relative_to(SCRIPTS_DIR)}:{number} {stripped}")
    assert not offenders, (
        "unbounded wait in a UI path - use `join_worker`: " + "; ".join(offenders)
    )
