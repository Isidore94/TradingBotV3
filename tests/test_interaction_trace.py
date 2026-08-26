"""An interaction id, so a stall log entry says which click it belongs to.

P1 item 3. The stall watchdog names the frame that held the GUI thread, which
is the wrong question for a large share of the 2026-08-25 samples: a stall
whose modal frame is inside Qt's own event dispatch names no application code,
so the record says the desk froze and nothing about why.

The rule this instrument lives under is `ScanCycleClock`'s, and the last test
here is the same test that class carries: it measures and formats and decides
NOTHING. A timing helper that could defer, skip or schedule work would be a
scheduling change wearing a diagnostic label.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

TRACE_SOURCE = (SCRIPTS_DIR / "ui" / "interaction_trace.py").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_trace():
    from ui import interaction_trace

    interaction_trace.reset()
    yield
    interaction_trace.reset()


# ==========================================================================
# the span itself
# ==========================================================================
def test_an_interaction_reports_the_stage_it_has_reached():
    from ui import interaction_trace as trace

    assert trace.current() is None, "an idle desk has no open interaction"

    interaction_id = trace.begin("page_select", "Weekend Prep")
    assert interaction_id
    trace.mark("model_apply")

    live = trace.current()
    assert live is not None
    assert live["interaction_id"] == interaction_id
    assert live["kind"] == "page_select"
    assert live["detail"] == "Weekend Prep"
    assert live["stage"] == "model_apply"
    assert live["elapsed_ms"] >= 0.0

    trace.end()
    assert trace.current() is None


def test_ids_are_distinct_across_interactions():
    from ui import interaction_trace as trace

    first = trace.begin("page_select", "A")
    trace.end()
    second = trace.begin("page_select", "B")
    trace.end()
    assert first != second


def test_a_second_click_supersedes_the_first_and_the_first_is_still_recorded():
    """A trader who clicks again has abandoned the first click.

    The stalls from that moment belong to the new one - but the abandoned span
    is exactly the kind of thing worth seeing in a slow session, so it is
    recorded rather than dropped.
    """
    from ui import interaction_trace as trace

    trace.begin("page_select", "Weekend Prep")
    trace.mark("layout")
    second = trace.begin("page_select", "Focus")

    live = trace.current()
    assert live is not None and live["interaction_id"] == second

    spans = trace.recent_spans()
    assert len(spans) == 1
    assert spans[0]["detail"] == "Weekend Prep"
    assert spans[0]["outcome"] == "superseded"
    assert spans[0]["last_stage"] == "layout"


def test_marking_with_nothing_open_is_a_no_op_not_an_error():
    from ui import interaction_trace as trace

    trace.mark("layout")
    trace.end()
    assert trace.current() is None


def test_the_span_ring_is_bounded():
    from ui import interaction_trace as trace

    for index in range(trace.MAX_SPANS + 25):
        trace.begin("page_select", f"page-{index}")
        trace.end()
    assert len(trace.recent_spans()) == trace.MAX_SPANS


def test_current_hands_back_a_copy_not_the_live_state():
    from ui import interaction_trace as trace

    trace.begin("page_select", "Weekend Prep")
    live = trace.current()
    live["stage"] = "tampered"
    assert trace.current()["stage"] != "tampered"


# ==========================================================================
# the stamp on the stall record - the point of the whole thing
# ==========================================================================
def test_a_stall_record_carries_the_interaction_that_was_open(tmp_path):
    pytest.importorskip("PySide6", reason="the watchdog is a Qt object")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from ui import interaction_trace as trace
    from ui.stall_watchdog import StallWatchdog

    log = tmp_path / "ui_stalls.jsonl"
    watchdog = StallWatchdog(threshold_ms=1.0, log_path=log)

    trace.begin("page_select", "Weekend Prep")
    trace.mark("model_apply")
    watchdog._write(0.5, ["frame.py:1 f"], 2)

    import json

    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["interaction_id"].startswith("page_select-")
    assert record["interaction_kind"] == "page_select"
    assert record["interaction_detail"] == "Weekend Prep"
    assert record["interaction_stage"] == "model_apply"


def test_a_stall_with_no_interaction_says_so_rather_than_guessing(tmp_path):
    """An idle-desk stall has no click behind it. That is a fact, not a gap."""
    pytest.importorskip("PySide6", reason="the watchdog is a Qt object")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from ui.stall_watchdog import StallWatchdog

    log = tmp_path / "ui_stalls.jsonl"
    watchdog = StallWatchdog(threshold_ms=1.0, log_path=log)
    watchdog._write(0.5, ["frame.py:1 f"], 2)

    import json

    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["interaction_id"] == ""
    assert record["interaction_stage"] == ""


def test_a_broken_trace_never_costs_a_stall_record(tmp_path, monkeypatch):
    """The instrument is subordinate to the measurement it annotates."""
    pytest.importorskip("PySide6", reason="the watchdog is a Qt object")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from ui import interaction_trace as trace
    from ui.stall_watchdog import StallWatchdog

    def boom():
        raise RuntimeError("trace is broken")

    monkeypatch.setattr(trace, "current", boom)

    log = tmp_path / "ui_stalls.jsonl"
    watchdog = StallWatchdog(threshold_ms=1.0, log_path=log)
    watchdog._write(0.5, ["frame.py:1 f"], 2)

    import json

    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["gap_ms"] == 500.0
    assert record["interaction_id"] == ""


# ==========================================================================
# it measures, and it decides NOTHING (the ScanCycleClock rule)
# ==========================================================================
FORBIDDEN_CALLS = {"sleep", "wait", "start", "join", "Thread", "Timer", "QTimer"}


def test_the_trace_can_never_defer_skip_or_schedule_anything():
    """Parsed, not grepped: a comment mentioning `sleep` is not a call to it."""
    tree = ast.parse(TRACE_SOURCE)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name in FORBIDDEN_CALLS:
            offenders.append(f"{name} at line {node.lineno}")
    assert not offenders, (
        "interaction_trace must never defer, skip or schedule work: " f"{offenders}"
    )


def test_the_trace_imports_no_threading_or_qt():
    """It runs on whatever thread called it and owns nothing."""
    tree = ast.parse(TRACE_SOURCE)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "threading" not in imported, "a trace that owns a lock can stall its subject"
    assert not {name for name in imported if name.startswith("PySide")}, imported


def test_the_trace_holds_no_lock():
    """Deliberate: the watchdog reads this from another thread.

    A lock here would let a diagnostic block the GUI thread it exists to
    measure. Correctness rests on replacing one tuple whole instead.
    """
    assert "Lock(" not in TRACE_SOURCE
    assert "acquire(" not in TRACE_SOURCE


# ==========================================================================
# wired to real clicks, not only to itself
# ==========================================================================
@pytest.mark.qt
def test_selecting_a_page_records_a_span_naming_that_page():
    pytest.importorskip("PySide6", reason="the desk needs PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from ui import interaction_trace as trace
    from ui.app import PAGE_SPECS, MainWindow
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        trace.reset()
        target = [spec.title for spec in PAGE_SPECS].index("Weekend Prep")
        window._select_page(target)

        spans = trace.recent_spans()
        assert spans, "selecting a page recorded no interaction"
        span = spans[-1]
        assert span["kind"] == "page_select"
        assert span["detail"] == "Weekend Prep"
        assert span["last_stage"] in ("model_apply", "layout", "chart_request")
        # Closed, so the next idle stall is not blamed on this click.
        assert trace.current() is None
    finally:
        try:
            window.close()
        except Exception:
            pass
