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

import threading

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


@pytest.fixture(autouse=True)
def _isolate_stores():
    """Never leave the module pointed at another test's files - or the desk's."""
    import bounce_bot_lib.legacy as legacy

    checkpoint = legacy.INTRADAY_BOUNCE_OUTCOME_STATE_JSON
    csv_path = legacy.INTRADAY_BOUNCE_OUTCOMES_CSV
    try:
        yield
    finally:
        legacy.INTRADAY_BOUNCE_OUTCOME_STATE_JSON = checkpoint
        legacy.INTRADAY_BOUNCE_OUTCOMES_CSV = csv_path


class _Sweeper:
    """A minimal host for the sweep and the one row writer it calls."""


def _seed(host, pending: dict) -> None:
    """Put the trades in memory AND on disk.

    The finalization transaction re-reads the checkpoint from disk inside its
    lock - in-memory state is not authoritative, which is the whole point of
    Sol's two-process finding - so a test that only sets the dict is testing a
    machine whose disk says the trade was never registered.
    """
    host.pending_bounce_outcomes = dict(pending)
    host._save_pending_bounce_outcomes()


def _host(tmp_path):
    import bounce_bot_lib.legacy as legacy
    from bounce_bot_lib.legacy import BounceBot

    # Each test gets its own stores. Never the desk's.
    legacy.INTRADAY_BOUNCE_OUTCOME_STATE_JSON = tmp_path / "state.json"
    legacy.INTRADAY_BOUNCE_OUTCOMES_CSV = tmp_path / "outcomes.csv"

    host = _Sweeper.__new__(_Sweeper)
    host.rows = []
    host.coverage = []
    host.saves = 0
    host.pending_bounce_outcomes = {}
    host._finalized_outcome_memory = {}
    host._finalizing_outcome_marks = {}
    host.OUTCOME_LOCK_TIMEOUT_SECONDS = BounceBot.OUTCOME_LOCK_TIMEOUT_SECONDS
    host.PENDING_EXPIRY_SESSIONS = BounceBot.PENDING_EXPIRY_SESSIONS
    host.FINALIZED_MEMORY = BounceBot.FINALIZED_MEMORY
    host._append_learning_row = lambda path, fieldnames, row: host.rows.append(dict(row))
    host._mirror_outcome_row_to_ledger = lambda row, state: None
    real_save = BounceBot._save_pending_bounce_outcomes.__get__(host, _Sweeper)

    def counted_save():
        host.saves += 1
        return real_save()

    host._save_pending_bounce_outcomes = counted_save
    host._save_pending_bounce_outcomes_locked = (
        BounceBot._save_pending_bounce_outcomes_locked.__get__(host, _Sweeper)
    )
    host._load_pending_bounce_outcomes = (
        BounceBot._load_pending_bounce_outcomes.__get__(host, _Sweeper)
    )
    host._write_outcome_coverage = lambda counts: host.coverage.append(counts)
    host.SWEEP_AFTER_SCAN_WINDOW_MINUTES = BounceBot.SWEEP_AFTER_SCAN_WINDOW_MINUTES
    host.OUTCOME_BAR_MINUTES = BounceBot.OUTCOME_BAR_MINUTES
    # A staticmethod: binding it through __get__ would hand it `self`.
    host._naive_market_local = BounceBot._naive_market_local
    host.RECOVERABLE_EVENT_TYPES = BounceBot.RECOVERABLE_EVENT_TYPES
    host._pending_outcome_lock = threading.RLock()
    # The property is on the class, so it is fetched off BounceBot and bound.
    type(host)._pending_lock = BounceBot.__dict__["_pending_lock"]
    for name in (
        "_parse_bar_time", "_json_for_learning", "_context_with_finalization",
        "_append_bounce_outcome_row", "_is_eod_finalization_due", "_sessions_since",
        "_finalized_outcome_ids", "_remember_finalized_outcome",
        "_sweep_window_is_open", "_recover_measurements_from_csv", "_exit_facts",
        "_completed_session_rows", "_rows_after_bounce_entry_for_session",
        "actual_session_close", "_final_event_ids_in_csv", "_read_checkpoint_from_disk",
        "_commit_checkpoint", "_outcome_transaction", "finalize_outcome_once",
        "_finalizing_outcome_ids", "resolve_unfinished_finalizations",
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
    _seed(host, {"a": _state(event_id="a")})
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert counts["finalized"] == 1
    assert counts["pending_after"] == 0
    assert host.rows[-1]["event_type"] == "final"


def test_a_trade_whose_session_is_still_running_is_left_alone(tmp_path):
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    counts = host.sweep_pending_bounce_outcomes(
        now=datetime(2026, 8, 21, 8, 0), wait_for_scan_window=False
    )
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
    _seed(host, {"a": state})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]
    # MAJOR-3: `close_r` is the eod_hold number and there are no bars through
    # the close, so it stays blank. What was measured is kept beside it.
    assert row["status"] == "unresolved"
    assert row["close_r"] == "" and row["eod_close"] == ""
    context = _context(row)
    assert context["finalization"]["basis"] == "last_measured_bar"
    assert context["finalization"]["reason"] == "no_eod_close"
    assert context["exit"]["last_measured_close"] == 100.5


def test_a_measured_stop_out_finalizes_at_its_stop(tmp_path):
    host = _host(tmp_path)
    state = _state(event_id="a")
    state["last_measured"] = {"bars": 3, "stop_hit": True, "best_price": 100.4,
                              "worst_price": 98.5, "mfe_r": 0.4, "mae_r": -1.5,
                              "last_close": 98.8}
    _seed(host, {"a": state})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]
    assert row["status"] == "unresolved", "no bars through the close"
    assert row["close_r"] == "" and row["eod_close"] == ""
    assert _context(row)["exit"]["stop_exit_r"] == -1.0
    assert row["stop_hit"] is True


def test_a_trade_that_measured_nothing_finalizes_unresolved(tmp_path):
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]
    assert row["status"] == "unresolved"
    assert row["close_r"] == ""
    # Distinguishable from "the session produced no bars": this trade predates
    # the measurement field and the CSV had nothing to recover either.
    assert _context(row)["finalization"]["reason"] == "no_measurement_in_checkpoint"


# ---------------------------------------------------------------------------
# expiry
# ---------------------------------------------------------------------------
def test_a_trade_with_no_data_expires_after_three_sessions(tmp_path):
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    counts = host.sweep_pending_bounce_outcomes(now=MUCH_LATER)
    assert counts["expired"] == 1
    assert _context(host.rows[-1])["finalization"]["reason"] == "expired_no_data"


def test_expiry_counts_sessions_not_days(tmp_path):
    """A long weekend must not expire a trade that is two sessions old."""
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    # Monday 2026-08-24: one completed session after Friday's entry.
    counts = host.sweep_pending_bounce_outcomes(now=datetime(2026, 8, 24, 14, 30))
    assert counts["expired"] == 0
    assert _context(host.rows[-1])["finalization"]["reason"] == "no_measurement_in_checkpoint"


def test_a_trade_that_measured_something_is_never_expired(tmp_path):
    """It has evidence, so it finalizes on that evidence however old it is."""
    host = _host(tmp_path)
    state = _state(event_id="a")
    state["last_measured"] = {"bars": 3, "stop_hit": False, "last_close": 100.5,
                              "close_r": 0.5, "mfe_r": 0.8, "mae_r": -0.2}
    _seed(host, {"a": state})
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
    _seed(host, {"a": _state(event_id="a")})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    written = len(host.rows)
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert len(host.rows) == written


def test_a_reappearing_id_is_recognised_and_not_re_finalized(tmp_path):
    """A restart that reloads a stale checkpoint must not double-write."""
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    _seed(host, {"a": _state(event_id="a")})
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

    save = inspect.getsource(BounceBot._save_pending_bounce_outcomes_locked)
    load = inspect.getsource(BounceBot._load_pending_bounce_outcomes)
    assert '"finalized": self._finalized_outcome_ids()' in save
    assert "_finalized_outcome_memory" in load


# ---------------------------------------------------------------------------
# it reports
# ---------------------------------------------------------------------------
def test_an_unparseable_entry_is_counted_not_silently_dropped(tmp_path):
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a", entry_time="not a time")})
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert counts["unparseable"] == 1
    assert not host.pending_bounce_outcomes


def test_the_sweep_files_its_own_coverage(tmp_path):
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a"), "b": _state(event_id="b")})
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert host.coverage and host.coverage[-1] is counts
    assert counts["pending_before"] == 2
    assert counts["by_reason"]["no_measurement_in_checkpoint"] == 2
    assert counts["swept_at"].startswith("2026-08-21")


def test_a_writer_failure_leaves_the_trade_pending_rather_than_losing_it(tmp_path):
    host = _host(tmp_path)

    def angry(*args, **kwargs):
        raise RuntimeError("disk full")

    host._append_bounce_outcome_row = angry
    _seed(host, {"a": _state(event_id="a")})
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert counts["finalized"] == 0
    assert "a" in host.pending_bounce_outcomes, "an unwritten final must stay pending"


def test_the_after_close_worker_sweeps_before_the_learning_refresh():
    """The refresh reads the rows the sweep writes."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._maybe_refresh_learning_after_close)
    sweep_at = source.index("sweep_pending_bounce_outcomes()")
    refresh_at = source.index("refresh_learning_state_with_shadow()")
    assert sweep_at < refresh_at, "the refresh reads the rows the sweep writes"
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


# ---------------------------------------------------------------------------
# BLOCKER-1: two finalizers, one lock
# ---------------------------------------------------------------------------
def test_the_sweep_defers_while_the_scan_window_is_still_open(tmp_path):
    """The scan thread finalizes through close+30; this is the catch-up.

    Close+10 to close+30 is twenty minutes in which both could finalize the same
    trade. The lock makes an overlap correct; deferring makes it rare.
    """
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    counts = host.sweep_pending_bounce_outcomes(now=datetime(2026, 8, 21, 13, 15))
    assert counts["deferred"] == "scan_window_open"
    assert counts["finalized"] == 0
    assert host.pending_bounce_outcomes, "nothing is touched while it defers"


def test_the_sweep_runs_once_the_scan_window_has_closed(tmp_path):
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    counts = host.sweep_pending_bounce_outcomes(now=datetime(2026, 8, 21, 13, 40))
    assert "deferred" not in counts
    assert counts["finalized"] == 1


def test_two_threads_finalizing_the_same_trade_write_exactly_one_final(tmp_path):
    """BLOCKER-1, reproduced: the scan thread and the sweep at the same instant."""
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    start = threading.Barrier(2)

    def per_symbol():
        """What the per-symbol finalizer does, in miniature."""
        start.wait()
        with host._pending_lock:
            state = host.pending_bounce_outcomes.get("a")
            if state is None or "a" in host._finalized_outcome_ids():
                return
            host._append_bounce_outcome_row(
                state, "final", bars_elapsed=0, milestone_bar=None,
                rows_after_entry=pd.DataFrame(), finalize_eod=True,
            )
            host._remember_finalized_outcome("a", state.get("trade_date"))
            host.pending_bounce_outcomes.pop("a", None)

    def sweeper():
        start.wait()
        host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)

    threads = [threading.Thread(target=per_symbol), threading.Thread(target=sweeper)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    finals = [row for row in host.rows if row["event_type"] == "final"]
    assert len(finals) == 1, f"expected one final, got {len(finals)}"
    assert not host.pending_bounce_outcomes


def test_the_sweep_re_reads_each_entry_under_the_lock():
    """The id list is stale by construction; the state must be re-read."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot.sweep_pending_bounce_outcomes)
    assert "for event_id in list(self.pending_bounce_outcomes):" in source
    assert "with self._pending_lock:" in source
    assert "state = self.pending_bounce_outcomes.get(event_id)" in source


# ---------------------------------------------------------------------------
# MAJOR-2: the backlog's own measurements are recovered
# ---------------------------------------------------------------------------
def _csv(tmp_path: Path, rows) -> Path:
    import csv as _csv

    path = tmp_path / "outcomes.csv"          # where `_host` points the module
    fields = ["event_id", "event_type", "close_r", "mfe_r", "mae_r", "best_price",
              "worst_price", "stop_hit", "target_1r_hit", "target_2r_hit",
              "bars_elapsed", "minutes_elapsed", "logged_at"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return path


def test_a_backlog_stop_out_is_recovered_from_its_own_csv_rows(tmp_path, monkeypatch):
    """The 563 stop-outs at 0R are only recovered this way.

    `last_measured` landed on 2026-08-23 and no trade already in the checkpoint
    carries it, so without this the whole backlog finalizes as having seen no
    bars - including trades whose milestone rows are sitting in the CSV.

    **Expectations changed 2026-08-25 by trader Decision B.1.** Both rows here
    record the stop, and the exit now comes from the EARLIEST of them rather
    than the furthest milestone: R10.0's stop-first decision says the trade was
    over at bar 3, so bar 12's wider excursion describes price action after an
    exit that had already happened. The recovery itself is unchanged.
    """
    path = _csv(tmp_path, [
        {"event_id": "a", "event_type": "3_bar", "close_r": "-0.8", "mfe_r": "0.4",
         "mae_r": "-1.5", "best_price": "100.4", "worst_price": "98.5",
         "stop_hit": "True", "bars_elapsed": "3", "logged_at": "2026-08-21T08:00:00"},
        {"event_id": "a", "event_type": "12_bar", "close_r": "-1.2", "mfe_r": "0.4",
         "mae_r": "-1.8", "best_price": "100.4", "worst_price": "98.2",
         "stop_hit": "True", "bars_elapsed": "12", "logged_at": "2026-08-21T09:00:00"},
    ])

    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)

    assert counts["recovered_from_csv"] == 1
    row = host.rows[-1]
    # It has evidence, and that evidence is recorded - but there is still no
    # close-through-the-session, so `close_r` stays blank (MAJOR-3).
    assert row["status"] == "unresolved"
    assert row["stop_hit"] is True and row["mae_r"] == pytest.approx(-1.5)
    assert _context(row)["exit"]["stop_exit_r"] == -1.0
    finalization = _context(row)["finalization"]
    assert finalization["basis"] == "stop_hit_from_prior_measurement"
    assert finalization["measurement_source"] == "legacy_csv_milestones"
    assert finalization["recovered_from"] == "3_bar", "the earliest stop-hit milestone wins"
    assert counts["by_reason"]["stop_hit:legacy_csv_milestones"] == 1


def test_a_recovered_measurement_reconstructs_the_close_from_stored_numbers(tmp_path, monkeypatch):
    """`last_close` is arithmetic on the row's own close_r, entry and risk."""
    path = _csv(tmp_path, [
        {"event_id": "a", "event_type": "12_bar", "close_r": "0.5", "mfe_r": "0.8",
         "mae_r": "-0.2", "best_price": "100.8", "worst_price": "99.8",
         "stop_hit": "False", "bars_elapsed": "12", "logged_at": "2026-08-21T09:00:00"},
    ])
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    # entry 100.0 + 0.5R * risk 1.0 - kept as the measured close, not as an
    # EOD close it never was.
    assert _context(host.rows[-1])["exit"]["last_measured_close"] == pytest.approx(100.5)


def test_a_short_recovers_its_close_on_the_other_side(tmp_path, monkeypatch):
    path = _csv(tmp_path, [
        {"event_id": "a", "event_type": "12_bar", "close_r": "0.5", "stop_hit": "False",
         "bars_elapsed": "12", "logged_at": "2026-08-21T09:00:00"},
    ])
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a", direction="short",
                             stop_price=101.0, target_1r=99.0)})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert _context(host.rows[-1])["exit"]["last_measured_close"] == pytest.approx(99.5)


def test_an_empty_csv_row_is_not_recovered_as_a_measurement(tmp_path, monkeypatch):
    """Recovering nothing would only relabel an absence as a measurement."""
    path = _csv(tmp_path, [
        {"event_id": "a", "event_type": "1_bar", "close_r": "", "stop_hit": "False",
         "bars_elapsed": "1", "logged_at": "2026-08-21T08:00:00"},
    ])
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert counts["recovered_from_csv"] == 0
    assert _context(host.rows[-1])["finalization"]["reason"] == "no_measurement_in_checkpoint"


def test_a_missing_csv_does_not_stop_the_sweep(tmp_path):
    host = _host(tmp_path)          # `_host` points at a CSV that does not exist yet
    _seed(host, {"a": _state(event_id="a")})
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert counts["finalized"] == 1 and counts["recovered_from_csv"] == 0


# ---------------------------------------------------------------------------
# the autorun switch (trader decision, 2026-08-23)
# ---------------------------------------------------------------------------
def test_the_sweep_does_not_fire_itself_by_default():
    from unittest import mock

    from bounce_bot_lib.legacy import BounceBot

    host = object.__new__(BounceBot)
    for value in ("", "off", "nonsense", None):
        host._sweep_autorun_announced = False
        with mock.patch("project_paths.get_local_setting", return_value=value):
            assert BounceBot._sweep_autorun_enabled(host) is False


def test_the_switch_turns_it_on():
    from unittest import mock

    from bounce_bot_lib.legacy import BounceBot

    host = object.__new__(BounceBot)
    for value in ("on", "1", "true", "YES"):
        host._sweep_autorun_announced = False
        with mock.patch("project_paths.get_local_setting", return_value=value):
            assert BounceBot._sweep_autorun_enabled(host) is True


def test_the_after_close_scheduler_asks_the_switch_first():
    """The switch is consulted where the decision is made, not in the worker."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    assert "_sweep_autorun_enabled()" in inspect.getsource(BounceBot._after_close_jobs_due)
    worker = inspect.getsource(BounceBot._maybe_refresh_learning_after_close)
    assert "_after_close_jobs_due(now)" in worker


# ---------------------------------------------------------------------------
# the checkpoint itself
# ---------------------------------------------------------------------------
def test_the_checkpoint_is_written_temp_then_replaced():
    """A bare write_text leaves a torn file if the process dies mid-write."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._save_pending_bounce_outcomes_locked)
    assert "os.replace(" in source
    assert ".tmp" in source


def test_an_unreadable_checkpoint_is_quarantined_and_loud_not_silently_empty(tmp_path, monkeypatch, caplog):
    """It used to return {} - discarding 576 pending trades without a word."""
    import logging

    import bounce_bot_lib.legacy as legacy
    from bounce_bot_lib.legacy import BounceBot

    path = tmp_path / "state.json"
    path.write_text("{ truncated", encoding="utf-8")
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOME_STATE_JSON", path)

    host = object.__new__(BounceBot)
    with caplog.at_level(logging.ERROR):
        assert BounceBot._load_pending_bounce_outcomes(host) == {}
    assert any("Quarantining" in record.getMessage() for record in caplog.records)
    assert not path.exists(), "the unreadable file is moved aside, not left to be overwritten"
    assert list(tmp_path.glob("state.corrupt-*.json")), "and it is kept, so the rows are recoverable"


# ---------------------------------------------------------------------------
# Decision B.1 (2026-08-25): milestone recovery must not erase a recorded stop
# ---------------------------------------------------------------------------
def test_a_later_milestone_cannot_erase_an_earlier_recorded_stop(tmp_path, monkeypatch):
    """Sol C4, reproduced verbatim.

    The first row at the FURTHEST milestone won outright, so a 12-bar row
    saying `stop_hit=False` erased the 3-bar row that had already recorded the
    stop, and the trade finalized `last_measured_bar` at +0.5R. `stop_hit` is
    now `any()` across the trade's recoverable rows, and the exit comes from the
    EARLIEST stop-hit row - R10.0's stop-first decision, applied here: once the
    stop is reached the trade is over, and later rows describe price action
    after an exit that already happened.
    """
    _csv(tmp_path, [
        {"event_id": "a", "event_type": "3_bar", "close_r": "-1.0", "mae_r": "-1.2",
         "stop_hit": "True", "bars_elapsed": "3", "logged_at": "2026-08-21T08:00:00"},
        {"event_id": "a", "event_type": "12_bar", "close_r": "0.5", "mae_r": "-0.2",
         "stop_hit": "False", "bars_elapsed": "12", "logged_at": "2026-08-21T09:00:00"},
    ])
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})

    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]

    assert row["stop_hit"] is True, "a later milestone erased an earlier recorded stop hit"
    assert _context(row)["exit"]["stop_exit_r"] == -1.0
    assert _context(row)["finalization"]["basis"] == "stop_hit_from_prior_measurement"
    assert _context(row)["finalization"]["recovered_from"] == "3_bar"
    # The exit numbers are the earliest stop-hit row's, not the furthest row's.
    assert row["mae_r"] == pytest.approx(-1.2)
    assert counts["by_reason"]["stop_hit:legacy_csv_milestones"] == 1


def test_the_earliest_stop_hit_row_wins_even_when_a_later_one_also_stopped(tmp_path, monkeypatch):
    """Two rows both record the stop. The trade was over at the first one."""
    _csv(tmp_path, [
        {"event_id": "a", "event_type": "12_bar", "close_r": "-1.2", "mae_r": "-1.8",
         "stop_hit": "True", "bars_elapsed": "12", "logged_at": "2026-08-21T09:00:00"},
        {"event_id": "a", "event_type": "3_bar", "close_r": "-0.8", "mae_r": "-1.5",
         "stop_hit": "True", "bars_elapsed": "3", "logged_at": "2026-08-21T08:00:00"},
    ])
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]

    assert row["stop_hit"] is True
    assert _context(row)["finalization"]["recovered_from"] == "3_bar"
    assert row["mae_r"] == pytest.approx(-1.5)


def test_with_no_stop_anywhere_the_best_rank_row_still_wins(tmp_path, monkeypatch):
    """Unchanged where nothing stopped: the furthest milestone is still the
    most complete measurement of a trade that was never cut short."""
    _csv(tmp_path, [
        {"event_id": "a", "event_type": "3_bar", "close_r": "0.2", "mae_r": "-0.1",
         "stop_hit": "False", "bars_elapsed": "3", "logged_at": "2026-08-21T08:00:00"},
        {"event_id": "a", "event_type": "12_bar", "close_r": "0.5", "mae_r": "-0.2",
         "stop_hit": "False", "bars_elapsed": "12", "logged_at": "2026-08-21T09:00:00"},
    ])
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]

    assert row["stop_hit"] is False
    assert _context(row)["finalization"]["recovered_from"] == "12_bar"
    assert row["mae_r"] == pytest.approx(-0.2)


def test_an_unreadable_bar_count_does_not_make_a_stop_row_look_earliest(tmp_path, monkeypatch):
    """A blank `bars_elapsed` cannot be ordered, so it sorts LAST among the
    stop rows rather than winning by accident."""
    _csv(tmp_path, [
        {"event_id": "a", "event_type": "update", "close_r": "-1.0", "mae_r": "-9.9",
         "stop_hit": "True", "bars_elapsed": "", "logged_at": "2026-08-21T07:00:00"},
        {"event_id": "a", "event_type": "3_bar", "close_r": "-1.0", "mae_r": "-1.2",
         "stop_hit": "True", "bars_elapsed": "3", "logged_at": "2026-08-21T08:00:00"},
    ])
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)

    assert _context(host.rows[-1])["finalization"]["recovered_from"] == "3_bar"
    assert host.rows[-1]["mae_r"] == pytest.approx(-1.2)
