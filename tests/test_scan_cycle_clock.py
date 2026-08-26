"""The scan-cycle timing instrument (trader-authorized 2026-08-25).

The 2026-08-25 investigation could narrow a 92-minute silence in `run_strategy`
no further than "somewhere in the preamble", because every call in that stretch
is silent on the normal path. These tests pin the instrument that answers it -
and pin that it is an INSTRUMENT: it measures and formats, and decides nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bounce_bot_lib.legacy import ScanCycleClock  # noqa: E402


class _Clock:
    """A hand-cranked clock, so elapsed times are exact rather than flaky."""

    def __init__(self, *ticks):
        self._ticks = list(ticks)
        self._at = 0.0

    def __call__(self):
        if self._ticks:
            self._at += self._ticks.pop(0)
        return self._at


def test_each_stage_records_its_own_elapsed_time():
    clock = ScanCycleClock(now=_Clock(0.0, 2.0, 88.0, 1.0))
    assert clock.mark("atr") == pytest.approx(2.0)
    assert clock.mark("rrs") == pytest.approx(88.0)
    assert clock.mark("m5_engines") == pytest.approx(1.0)
    assert clock.total() == pytest.approx(91.0)


def test_the_summary_names_the_slowest_stages_first():
    """The question is always "what took the time"; declaration order buries it."""
    clock = ScanCycleClock(now=_Clock(0.0, 2.0, 88.0, 1.0))
    for stage in ("atr", "rrs", "m5_engines"):
        clock.mark(stage)

    summary = clock.summary()

    assert summary.startswith("91.0s total: rrs 88.1s") or summary.startswith("91.0s total: rrs 88.0s")
    assert summary.index("rrs") < summary.index("atr") < summary.index("m5_engines")


def test_stages_beyond_the_named_few_are_counted_never_dropped():
    """A breakdown that silently omits stages reads as a complete account of the
    time and is not one."""
    clock = ScanCycleClock(now=_Clock(0.0, *([1.0] * 8)))
    for index in range(8):
        clock.mark(f"stage{index}")

    summary = clock.summary(limit=3)

    assert "+5 other 5.0s" in summary
    assert summary.startswith("8.0s total:")


def test_a_cycle_that_marked_nothing_still_reports_its_total():
    clock = ScanCycleClock(now=_Clock(0.0))
    assert clock.summary() == "0.0s total"


def test_a_clock_that_runs_backwards_never_reports_negative_time():
    """Windows clock adjustments happen. A negative stage would read as an
    instrument fault rather than as the measurement it is meant to be."""
    clock = ScanCycleClock(now=_Clock(0.0, -5.0))
    assert clock.mark("atr") == 0.0
    assert clock.total() >= 0.0


def test_the_clock_decides_nothing():
    """It measures and formats. A timing helper that could skip, defer or wait
    would be a scheduling change wearing an instrument's name - and scheduling
    is explicitly not what the trader authorized."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(ScanCycleClock)))
    called = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    forbidden = {"time.sleep", "self._stop_event.wait", "threading.Thread"}
    assert not (called & forbidden), f"the clock must not call {called & forbidden}"
    for name in called:
        # `", ".join(...)` is string formatting, which is half this class's job.
        assert not name.endswith((".sleep", ".wait", ".start")), name


def test_the_loop_logs_one_cycle_line_with_the_breakdown():
    """The instrument has to be WIRED, not merely present - the defect it
    replaces was a function that existed and was never called."""
    source = (SCRIPTS_DIR / "bounce_bot_lib" / "legacy.py").read_text(encoding="utf-8")
    assert "ScanCycleClock(" in source, "run_strategy must build one per cycle"
    assert "cycle_clock.mark(" in source, "the preamble stages must be marked"
    assert "scan cycle" in source.lower()
