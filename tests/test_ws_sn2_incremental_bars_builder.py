"""WS-SN2, builder-added tests: what a FORMING tail does to the kept window.

These sit beside `tests/test_ws_sn2_incremental_bars.py` (the tester's red
tests) and exist because one assertion in that file names a MECHANISM the code
refutes.

The tester's fake IB serves completed bars only unless a test asks for a
forming one, so its
`test_a_forming_last_bar_forces_a_whole_window_refetch_and_the_cache_recovers`
treats a forming tail as an anomaly ("a forming bar that slipped in").  The
real provider does not behave that way: `bounce_bot_lib/legacy.py`
`_rows_after_bounce_entry_for_session` records that "the frame arrives from a
request with an empty `endDateTime`, so its last row is the bar still forming",
and `_m5_bar_completed` exists because "IB's historical cache has no
complete/forming marker".  A rule that refetches the whole window whenever the
served tail is forming therefore refetches it on EVERY cycle of the session,
and SN2 saves nothing.

So the shipped policy keeps the completed rows only - the forming bar never
enters the window a later delta is merged onto, which is the invariant
plan.md sec 5 actually states - and the next delta re-reads that bar once it
has closed.  The first test below proves the frame is still byte-identical to a
fresh fetch through a forming tail and across the two cycles after it.  The
second proves the tester's mechanism is still THERE and still correct, one
constant away (`BounceBot.SN2_FORMING_TAIL_FORCES_REFETCH`), so the trader's
call is a one-line change rather than a rewrite.

Everything here reuses the tester's harness, its fake IB and its golden
fixture; nothing new is recorded.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

TESTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TESTS_DIR.parent / "scripts"
for _path in (TESTS_DIR, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import bounce_bot  # noqa: E402
from test_ws_sn2_incremental_bars import (  # noqa: E402
    CYCLE_3_NOW,
    CYCLE_4_NOW,
    FORMING_NOW,
    SYMBOL,
    Harness,
    golden,
    is_whole_window_request,
)


def _forming_cycle(monkeypatch):
    """Cycle 0 served with the 09:20 bar still forming at 09:22."""
    harness = Harness(monkeypatch)
    harness.ib.serve_forming_bar = True
    harness.cycle(FORMING_NOW)
    harness.ib.serve_forming_bar = False
    assert harness.bot.latest_bars[SYMBOL][-1].dt == datetime(2026, 6, 5, 9, 20), (
        "the fake must actually have served the forming bar, or this proves nothing"
    )
    return harness


def test_a_forming_bar_never_enters_the_kept_window(monkeypatch):
    """The invariant, stated as the code keeps it: the window a delta is
    merged onto holds only bars that had CLOSED when they were served."""
    harness = _forming_cycle(monkeypatch)

    kept = harness.bot.cached_bounce_frame(SYMBOL)
    assert kept, "cycle 0 kept no window at all"
    assert kept[-1]["time"] == "20260605  09:15:00", (
        "the 09:20 bar was still forming at 09:22 and must not be in the kept "
        "window; the window ends at %s" % (kept[-1]["time"],)
    )


def test_the_frame_after_a_forming_tail_is_still_a_fresh_fetch(monkeypatch):
    """The thing that actually matters: through a forming tail, and for the
    two cycles after it, the detectors are handed the fresh five-day frame."""
    harness = _forming_cycle(monkeypatch)
    harness.cycle(CYCLE_3_NOW)
    harness.cycle(CYCLE_4_NOW)

    assert_frame_equal(harness.frame_for(SYMBOL, 1), golden("intraday_cycle_3"))
    assert_frame_equal(harness.frame_for(SYMBOL, 2), golden("intraday_cycle_4"))

    for cycle_index in (1, 2):
        requests = harness.ib.requests_in_cycle(cycle_index)
        assert len(requests) == 1, (
            "cycle %s cost %s requests (%s); a forming tail must not cost a "
            "second one" % (cycle_index, len(requests), [r.duration for r in requests])
        )
        assert harness.ib.bars_served_in_cycle(cycle_index) <= 12, (
            "cycle %s pulled %s bars - a forming tail must not cost the whole "
            "window" % (cycle_index, harness.ib.bars_served_in_cycle(cycle_index))
        )


def test_the_forming_tail_refetch_policy_is_one_constant_away(monkeypatch):
    """The tester's mechanism, proven to work when it is switched on.

    With `SN2_FORMING_TAIL_FORCES_REFETCH` True the cycle after a forming tail
    refetches the whole window and the cycle after THAT is a delta again -
    exactly what the packet asked for - at the cost of doing it every cycle of
    a live session.
    """
    monkeypatch.setattr(
        bounce_bot.BounceBot, "SN2_FORMING_TAIL_FORCES_REFETCH", True, raising=True
    )
    harness = _forming_cycle(monkeypatch)
    harness.cycle(CYCLE_3_NOW)
    harness.cycle(CYCLE_4_NOW)

    assert any(is_whole_window_request(req) for req in harness.ib.requests_in_cycle(1)), (
        "with the switch on, a forming tail must refetch the window; cycle 1 "
        "asked for %s" % ([req.duration for req in harness.ib.requests_in_cycle(1)],)
    )
    assert_frame_equal(harness.frame_for(SYMBOL, 1), golden("intraday_cycle_3"))

    third = harness.ib.requests_in_cycle(2)
    assert third and not any(is_whole_window_request(req) for req in third), (
        "after a clean refetch the next cycle must be a delta again; it asked "
        "for %s" % ([req.duration for req in third],)
    )
    assert_frame_equal(harness.frame_for(SYMBOL, 2), golden("intraday_cycle_4"))


@pytest.mark.parametrize("kept_symbols", [(), ("ZZZZ",)])
def test_a_window_is_never_kept_for_a_symbol_the_cycle_no_longer_scans(
    monkeypatch, kept_symbols
):
    """The bound on the cache, from the cycle-start hook's own signature."""
    harness = _forming_cycle(monkeypatch)
    assert harness.bot.cached_bounce_frame(SYMBOL) is not None

    harness.bot._prune_latest_bars_for_cycle(
        True, set(), scanned_symbols=set(kept_symbols)
    )
    assert harness.bot.cached_bounce_frame(SYMBOL) is None, (
        "a background-refresh cycle that names its scanned set must still free "
        "the window of a symbol that is not in it"
    )
