"""R10.A / D3 - finalization that does not depend on being scanned again.

The backlog: **576 pending outcomes, 94 older than 2026-08-18, 17 from June,
the oldest 2026-06-22.** Finalization only ever happened inside
`_update_pending_bounce_outcomes`, which runs for a symbol the scan is looking at
*right now* - so a name that stopped being scanned was never finalized at all.

**D4 is the same gap, not an IB outage.** On 2026-08-21 the store has 409
`registered`, 399 `1_bar`, 398 `3_bar`, 397 `6_bar`, 394 `12_bar` and **0
`final`**: the milestones ran all day and only the EOD pass is missing.

The sweep needs no bars and no IB. It finalizes from what each trade already
measured, expires a trade that never measured anything after three completed
sessions, and is **idempotent** - a second pass cannot write a second final.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Friday 2026-08-21, well after the close.
AFTER_CLOSE = datetime(2026, 8, 21, 14, 30)
# The following Thursday: three completed sessions later.
MUCH_LATER = datetime(2026, 8, 27, 14, 30)


class _Sweeper:
    """A minimal host for the sweep and the one row writer it calls."""


def _host(tmp_path):
    from bounce_bot_lib.legacy import BounceBot

    host = _Sweeper.__new__(_Sweeper)
    host.rows = []
    host.coverage = []
    host.saves = 0
    host.pending_bounce_outcomes = {}
    host._finalized_outcome_memory = {}
    host.PENDING_EXPIRY_SESSIONS = BounceBot.PENDING_EXPIRY_SESSIONS
    host.FINALIZED_MEMORY = BounceBot.FINALIZED_MEMORY
    host._append_learning_row = lambda path, fieldnames, row: host.rows.append(dict(row))
    host._mirror_outcome_row_to_ledger = lambda row, state: None
    host._save_pending_bounce_outcomes = lambda: setattr(host, "saves", host.saves + 1)
    host._write_outcome_coverage = lambda counts: host.coverage.append(counts)
    for name in (
        "_parse_bar_time", "_json_for_learning", "_context_with_finalization",
        "_append_bounce_outcome_row", "_is_eod_finalization_due", "_sessions_since",
        "_finalized_outcome_ids", "_remember_finalized_outcome",
        "sweep_pending_bounce_outcomes",
    ):
        setattr(host, name, getattr(BounceBot, name).__get__(host, _Sweeper))
    return host


def _state(event_id="AAPL_long_20260821_06_30_00_h1_ema10_bounce", **kwargs):
    state = {
        "event_id": event_id,
        "symbol": "AAPL",
        "direction": "long",
        "trade_date": "2026-08-21",
        "entry_time": "2026-08-21T07:00:00",
        "entry_price": 100.0,
        "stop_price": 99.0,
        "risk_per_share": 1.0,
        "target_1r": 101.0,
        "target_2r": 102.0,
        "milestones_logged": [],
        "outcome_mode": "eod_hold",
        "context": {},
    }
    state.update(kwargs)
    return state


def _context(row) -> dict:
    return json.loads(row["context_json"])


# ---------------------------------------------------------------------------
# it finalizes what the scan loop cannot reach
# ---------------------------------------------------------------------------
def test_a_pending_trade_whose_session_is_over_is_finalized(tmp_path):
    host = _host(tmp_path)
    host.pending_bounce_outcomes = {"a": _state(event_id="a")}
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert counts["finalized"] == 1
    assert counts["pending_after"] == 0
    assert host.rows[-1]["event_type"] == "final"


def test_a_trade_whose_session_is_still_running_is_left_alone(tmp_path):
    host = _host(tmp_path)
    host.pending_bounce_outcomes = {"a": _state(event_id="a")}
    counts = host.sweep_pending_bounce_outcomes(now=datetime(2026, 8, 21, 8, 0))
    assert counts["still_open"] == 1
    assert counts["finalized"] == 0
    assert host.pending_bounce_outcomes


def test_it_finalizes_from_what_the_trade_already_measured(tmp_path):
    host = _host(tmp_path)
    state = _state(event_id="a")
    state["last_measured"] = {
        "bars": 6, "close_r": 0.5, "mfe_r": 0.8, "mae_r": -0.2,
        "best_price": 100.8, "worst_price": 99.8, "last_close": 100.5,
        "stop_hit": False, "target_1r_hit": False, "target_2r_hit": False,
        "minutes_elapsed": 30, "at": "2026-08-21T08:00:00",
    }
    host.pending_bounce_outcomes = {"a": state}
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]
    assert row["eod_close"] == 100.5
    assert row["close_r"] == pytest.approx(0.5)
    assert _context(row)["finalization"]["basis"] == "last_measured_bar"


def test_a_measured_stop_out_finalizes_at_its_stop(tmp_path):
    host = _host(tmp_path)
    state = _state(event_id="a")
    state["last_measured"] = {"bars": 3, "stop_hit": True, "best_price": 100.4,
                              "worst_price": 98.5, "mfe_r": 0.4, "mae_r": -1.5,
                              "last_close": 98.8}
    host.pending_bounce_outcomes = {"a": state}
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]
    assert row["close_r"] == -1.0 and row["eod_close"] == 99.0


def test_a_trade_that_measured_nothing_finalizes_unresolved(tmp_path):
    host = _host(tmp_path)
    host.pending_bounce_outcomes = {"a": _state(event_id="a")}
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]
    assert row["status"] == "unresolved"
    assert row["close_r"] == ""
    assert _context(row)["finalization"]["reason"] == "no_bars_after_entry"


# ---------------------------------------------------------------------------
# expiry
# ---------------------------------------------------------------------------
def test_a_trade_with_no_data_expires_after_three_sessions(tmp_path):
    host = _host(tmp_path)
    host.pending_bounce_outcomes = {"a": _state(event_id="a")}
    counts = host.sweep_pending_bounce_outcomes(now=MUCH_LATER)
    assert counts["expired"] == 1
    assert _context(host.rows[-1])["finalization"]["reason"] == "expired_no_data"


def test_expiry_counts_sessions_not_days(tmp_path):
    """A long weekend must not expire a trade that is two sessions old."""
    host = _host(tmp_path)
    host.pending_bounce_outcomes = {"a": _state(event_id="a")}
    # Monday 2026-08-24: one completed session after Friday's entry.
    counts = host.sweep_pending_bounce_outcomes(now=datetime(2026, 8, 24, 14, 30))
    assert counts["expired"] == 0
    assert _context(host.rows[-1])["finalization"]["reason"] == "no_bars_after_entry"


def test_a_trade_that_measured_something_is_never_expired(tmp_path):
    """It has evidence, so it finalizes on that evidence however old it is."""
    host = _host(tmp_path)
    state = _state(event_id="a")
    state["last_measured"] = {"bars": 3, "stop_hit": False, "last_close": 100.5,
                              "close_r": 0.5, "mfe_r": 0.8, "mae_r": -0.2}
    host.pending_bounce_outcomes = {"a": state}
    counts = host.sweep_pending_bounce_outcomes(now=MUCH_LATER)
    assert counts["expired"] == 0 and counts["finalized"] == 1


def test_the_expiry_window_is_three_sessions():
    from bounce_bot_lib.legacy import BounceBot

    assert BounceBot.PENDING_EXPIRY_SESSIONS == 3


# ---------------------------------------------------------------------------
# idempotence
# ---------------------------------------------------------------------------
def test_a_second_sweep_writes_no_second_final(tmp_path):
    host = _host(tmp_path)
    host.pending_bounce_outcomes = {"a": _state(event_id="a")}
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    written = len(host.rows)
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert len(host.rows) == written


def test_a_reappearing_id_is_recognised_and_not_re_finalized(tmp_path):
    """A restart that reloads a stale checkpoint must not double-write."""
    host = _host(tmp_path)
    host.pending_bounce_outcomes = {"a": _state(event_id="a")}
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    host.pending_bounce_outcomes = {"a": _state(event_id="a")}
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert counts["already_finalized"] == 1
    assert counts["finalized"] == 0
    assert not host.pending_bounce_outcomes


def test_the_finalized_memory_is_bounded(tmp_path):
    host = _host(tmp_path)
    host.FINALIZED_MEMORY = 10
    for index in range(25):
        host._remember_finalized_outcome(f"e{index}", "2026-08-21")
    assert len(host._finalized_outcome_ids()) == 10


def test_the_memory_rides_in_the_same_checkpoint_as_the_pending_dict():
    """A restart has to know what the last run finalized."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    save = inspect.getsource(BounceBot._save_pending_bounce_outcomes)
    load = inspect.getsource(BounceBot._load_pending_bounce_outcomes)
    assert '"finalized": self._finalized_outcome_ids()' in save
    assert "_finalized_outcome_memory" in load


# ---------------------------------------------------------------------------
# it reports
# ---------------------------------------------------------------------------
def test_an_unparseable_entry_is_counted_not_silently_dropped(tmp_path):
    host = _host(tmp_path)
    host.pending_bounce_outcomes = {"a": _state(event_id="a", entry_time="not a time")}
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert counts["unparseable"] == 1
    assert not host.pending_bounce_outcomes


def test_the_sweep_files_its_own_coverage(tmp_path):
    host = _host(tmp_path)
    host.pending_bounce_outcomes = {"a": _state(event_id="a"), "b": _state(event_id="b")}
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert host.coverage and host.coverage[-1] is counts
    assert counts["pending_before"] == 2
    assert counts["by_reason"]["no_bars_after_entry"] == 2
    assert counts["swept_at"].startswith("2026-08-21")


def test_a_writer_failure_leaves_the_trade_pending_rather_than_losing_it(tmp_path):
    host = _host(tmp_path)

    def angry(*args, **kwargs):
        raise RuntimeError("disk full")

    host._append_bounce_outcome_row = angry
    host.pending_bounce_outcomes = {"a": _state(event_id="a")}
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert counts["finalized"] == 0
    assert "a" in host.pending_bounce_outcomes, "an unwritten final must stay pending"


def test_the_after_close_worker_sweeps_before_the_learning_refresh():
    """The refresh reads the rows the sweep writes."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._maybe_refresh_learning_after_close)
    sweep_at = source.index("sweep_pending_bounce_outcomes()")
    refresh_at = source.index("refresh_bounce_learning_state")
    assert sweep_at < refresh_at
    assert "Outcome sweep failed" in source, "a sweep failure must not cost the refresh"


def test_the_sweep_and_its_tests_share_one_clock():
    """`_is_eod_finalization_due` reads the live clock unless given one.

    Without the injection the sweep asked the real clock whether a 2026-08-21
    session was over while the test was pretending it was 08:00 that morning -
    so "still open" could not be tested at all, and any bug in that branch would
    have been invisible.
    """
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    signature = inspect.signature(BounceBot._is_eod_finalization_due)
    assert "now" in signature.parameters
    assert signature.parameters["now"].default is None, "existing callers are unchanged"
    sweep = inspect.getsource(BounceBot.sweep_pending_bounce_outcomes)
    assert "_is_eod_finalization_due(entry_dt, moment)" in sweep
