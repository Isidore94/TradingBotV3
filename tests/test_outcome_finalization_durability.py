"""R10.A - finalization survives a crash at any point, and a second process.

Sol reproduced two ways to write the same trade twice.

**Blocker 2.** The sweep appended every final row and committed ONE checkpoint at
the end of the batch, so a failure in `os.replace` left the disk still holding
the trade as pending with no finalized memory - and the restart wrote a second
final. On the 576-row backlog a crash near the end would have duplicated most of
the batch.

**Blocker 3.** `_pending_lock` is an in-process `RLock`. Sol started two real
Python processes; both loaded the same pending entry before either committed,
both reported `finalized=1`, and the CSV got two final rows.

The fix is one transaction with a write-ahead intent:

1. take the machine-wide lock (`local_writer_lock`: named mutex AND byte-range
   file lock, both, failing closed);
2. re-read the checkpoint from **disk** - in-memory state predates whatever
   another process just committed;
3. record the intent and commit it **before** appending, so a crash between the
   append and the commit is resolvable rather than a guess;
4. append - unless the CSV already has that final, which is what an interrupted
   attempt leaves behind;
5. record the finalization and commit. **A failed commit is not a finalization**
   and is not reported as one.

Every test below drives real files in `tmp_path`. Nothing touches the live store.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

AFTER_CLOSE = datetime(2026, 8, 21, 14, 30)
EVENT_ID = "AAPL_long_20260821_06_30_00_h1_ema10_bounce"


def _state():
    return {
        "event_id": EVENT_ID,
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


def _install_stores(monkeypatch, tmp_path: Path, *, key: str = ""):
    """Point the module's two stores at temp files and return them."""
    import bounce_bot_lib.legacy as legacy

    checkpoint = tmp_path / "state.json"
    csv_path = tmp_path / "outcomes.csv"
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOME_STATE_JSON", checkpoint)
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOMES_CSV", csv_path)
    checkpoint.write_text(
        json.dumps({"pending": {EVENT_ID: _state()}, "finalized": {}, "finalizing": {}}),
        encoding="utf-8",
    )
    return checkpoint, csv_path


class _Bot:
    """A BounceBot with only the finalization surface bound to it."""


def _bot(tmp_path: Path):
    from bounce_bot_lib.legacy import BounceBot

    bot = _Bot.__new__(_Bot)
    bot.pending_bounce_outcomes = {}
    bot._finalized_outcome_memory = {}
    bot._finalizing_outcome_marks = {}
    bot._outcome_ledger_obj = None
    bot._outcome_ledger_failed = True          # no ledger in these tests
    bot.PENDING_EXPIRY_SESSIONS = BounceBot.PENDING_EXPIRY_SESSIONS
    bot.FINALIZED_MEMORY = BounceBot.FINALIZED_MEMORY
    bot.RECOVERABLE_EVENT_TYPES = BounceBot.RECOVERABLE_EVENT_TYPES
    bot.SWEEP_AFTER_SCAN_WINDOW_MINUTES = BounceBot.SWEEP_AFTER_SCAN_WINDOW_MINUTES
    bot.OUTCOME_LOCK_TIMEOUT_SECONDS = BounceBot.OUTCOME_LOCK_TIMEOUT_SECONDS
    bot.OUTCOME_BAR_MINUTES = BounceBot.OUTCOME_BAR_MINUTES
    bot._naive_market_local = BounceBot._naive_market_local
    type(bot)._pending_lock = BounceBot.__dict__["_pending_lock"]
    for name in (
        "_parse_bar_time", "_json_for_learning", "_context_with_finalization",
        "_append_bounce_outcome_row", "_append_learning_row", "_learning_csv_header",
        "_is_eod_finalization_due", "_sessions_since", "_finalized_outcome_ids",
        "_remember_finalized_outcome", "_finalizing_outcome_ids", "_exit_facts",
        "_sweep_window_is_open", "actual_session_close", "_recover_measurements_from_csv",
        "_final_event_ids_in_csv", "_read_checkpoint_from_disk", "_commit_checkpoint",
        "_outcome_transaction", "finalize_outcome_once", "resolve_unfinished_finalizations",
        "_load_pending_bounce_outcomes", "_save_pending_bounce_outcomes",
        "_save_pending_bounce_outcomes_locked", "_write_outcome_coverage",
        "_mirror_outcome_row_to_ledger", "_outcome_ledger", "_ledger_canary_enabled",
        "sweep_pending_bounce_outcomes", "_completed_session_rows",
    ):
        setattr(bot, name, getattr(BounceBot, name).__get__(bot, _Bot))
    bot._write_outcome_coverage = lambda counts: None
    bot._mirror_outcome_row_to_ledger = lambda row, state: None
    bot.pending_bounce_outcomes = bot._load_pending_bounce_outcomes()
    return bot


def _finals(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    frame = pd.read_csv(csv_path)
    if "event_type" not in frame.columns:
        return []
    return frame[frame["event_type"] == "final"].to_dict("records")


def _restart(tmp_path: Path):
    """A fresh process's worth of state: reload from disk and resolve."""
    bot = _bot(tmp_path)
    bot.resolve_unfinished_finalizations()
    return bot


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------
def test_one_finalization_writes_one_final_and_commits_it(monkeypatch, tmp_path):
    checkpoint, csv_path = _install_stores(monkeypatch, tmp_path)
    bot = _bot(tmp_path)
    assert bot.finalize_outcome_once(EVENT_ID) == "finalized"

    assert len(_finals(csv_path)) == 1
    disk = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert EVENT_ID not in disk["pending"]
    assert EVENT_ID in disk["finalized"]
    assert disk["finalizing"] == {}, "the intent is cleared once it is durable"


def test_finalizing_the_same_trade_twice_writes_one_final(monkeypatch, tmp_path):
    _, csv_path = _install_stores(monkeypatch, tmp_path)
    bot = _bot(tmp_path)
    assert bot.finalize_outcome_once(EVENT_ID) == "finalized"
    assert bot.finalize_outcome_once(EVENT_ID) == "skipped"
    assert len(_finals(csv_path)) == 1


# ---------------------------------------------------------------------------
# blocker 2: a crash at every point converges to exactly one final
# ---------------------------------------------------------------------------
def _crash_after_n_commits(bot, n: int):
    """Let `n` checkpoint commits succeed, then die like a killed process."""
    real = bot._commit_checkpoint
    state = {"count": 0}

    def crashing(*args, **kwargs):
        if state["count"] >= n:
            raise OSError("simulated crash during commit")
        state["count"] += 1
        return real(*args, **kwargs)

    bot._commit_checkpoint = crashing


@pytest.mark.parametrize("commits_allowed", [0, 1])
def test_a_crash_during_a_commit_converges_to_one_final(monkeypatch, tmp_path, commits_allowed):
    """0 = the intent commit dies; 1 = the finalizing commit dies after the append."""
    checkpoint, csv_path = _install_stores(monkeypatch, tmp_path)
    bot = _bot(tmp_path)
    _crash_after_n_commits(bot, commits_allowed)
    result = bot.finalize_outcome_once(EVENT_ID)
    assert result == "commit_failed", "a failed commit is never reported as finalized"

    before = len(_finals(csv_path))
    assert before <= 1

    restarted = _restart(tmp_path)
    restarted.finalize_outcome_once(EVENT_ID)
    assert len(_finals(csv_path)) == 1, "exactly one durable final after the restart"
    disk = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert EVENT_ID in disk["finalized"] and EVENT_ID not in disk["pending"]


def test_a_crash_between_the_append_and_the_pop_converges_to_one_final(monkeypatch, tmp_path):
    """The gap Sol's os.replace injection opened."""
    checkpoint, csv_path = _install_stores(monkeypatch, tmp_path)
    bot = _bot(tmp_path)
    real_append = bot._append_bounce_outcome_row

    def append_then_die(*args, **kwargs):
        real_append(*args, **kwargs)
        raise OSError("simulated crash after the row reached the CSV")

    bot._append_bounce_outcome_row = append_then_die
    with pytest.raises(OSError):
        bot.finalize_outcome_once(EVENT_ID)
    assert len(_finals(csv_path)) == 1, "the row landed"

    restarted = _restart(tmp_path)
    assert len(_finals(csv_path)) == 1, "the restart recognises it and does NOT append again"
    disk = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert EVENT_ID in disk["finalized"]
    assert restarted.finalize_outcome_once(EVENT_ID) == "skipped"
    assert len(_finals(csv_path)) == 1


def test_a_crash_during_the_temp_write_leaves_the_previous_checkpoint_intact(monkeypatch, tmp_path):
    checkpoint, csv_path = _install_stores(monkeypatch, tmp_path)
    before = checkpoint.read_bytes()
    bot = _bot(tmp_path)
    real_dump = json.dump

    def dying_dump(*args, **kwargs):
        raise OSError("simulated crash mid temp write")

    monkeypatch.setattr(json, "dump", dying_dump)
    assert bot.finalize_outcome_once(EVENT_ID) == "commit_failed"
    monkeypatch.setattr(json, "dump", real_dump)

    assert checkpoint.read_bytes() == before, "a torn temp file never replaces the good one"
    assert _finals(csv_path) == [], "nothing was appended either"

    restarted = _restart(tmp_path)
    restarted.finalize_outcome_once(EVENT_ID)
    assert len(_finals(csv_path)) == 1


def test_a_crash_during_os_replace_converges_to_one_final(monkeypatch, tmp_path):
    """Sol's exact injection."""
    import bounce_bot_lib.legacy as legacy

    checkpoint, csv_path = _install_stores(monkeypatch, tmp_path)
    bot = _bot(tmp_path)
    real_replace = legacy.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:                       # let the intent commit land
            raise OSError("simulated crash during os.replace")
        return real_replace(src, dst)

    monkeypatch.setattr(legacy.os, "replace", flaky_replace)
    result = bot.finalize_outcome_once(EVENT_ID)
    monkeypatch.setattr(legacy.os, "replace", real_replace)
    assert result == "commit_failed"

    disk = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert EVENT_ID in disk["finalizing"], "the intent is on disk, so this is resolvable"

    restarted = _restart(tmp_path)
    assert len(_finals(csv_path)) == 1
    assert restarted.finalize_outcome_once(EVENT_ID) == "skipped"
    assert len(_finals(csv_path)) == 1


def test_a_crash_immediately_after_the_commit_changes_nothing(monkeypatch, tmp_path):
    checkpoint, csv_path = _install_stores(monkeypatch, tmp_path)
    bot = _bot(tmp_path)
    assert bot.finalize_outcome_once(EVENT_ID) == "finalized"
    # "crash" = throw the object away without another word
    del bot
    restarted = _restart(tmp_path)
    assert len(_finals(csv_path)) == 1
    assert restarted.finalize_outcome_once(EVENT_ID) == "skipped"
    assert len(_finals(csv_path)) == 1


def test_a_sweep_that_crashes_mid_batch_does_not_duplicate_the_finished_ones(monkeypatch, tmp_path):
    """The 576-row shape: a crash near the end used to duplicate most of it."""
    import bounce_bot_lib.legacy as legacy

    checkpoint = tmp_path / "state.json"
    csv_path = tmp_path / "outcomes.csv"
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOME_STATE_JSON", checkpoint)
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOMES_CSV", csv_path)
    pending = {}
    for index in range(6):
        state = _state()
        state["event_id"] = f"SYM{index}_long_20260821_06_30_00_h1_ema10_bounce"
        pending[state["event_id"]] = state
    checkpoint.write_text(json.dumps({"pending": pending, "finalized": {}, "finalizing": {}}), encoding="utf-8")

    bot = _bot(tmp_path)
    real = bot._commit_checkpoint
    state_counter = {"n": 0}

    def crash_after_seven(*args, **kwargs):
        state_counter["n"] += 1
        if state_counter["n"] > 7:                # three trades in, mid-transaction
            raise OSError("simulated crash mid batch")
        return real(*args, **kwargs)

    bot._commit_checkpoint = crash_after_seven
    bot.sweep_pending_bounce_outcomes(now=AFTER_CLOSE, wait_for_scan_window=False)

    restarted = _restart(tmp_path)
    restarted.sweep_pending_bounce_outcomes(now=AFTER_CLOSE, wait_for_scan_window=False)

    finals = _finals(csv_path)
    ids = [row["event_id"] for row in finals]
    assert len(ids) == len(set(ids)) == 6, f"one final per trade, got {len(ids)} rows for {len(set(ids))} ids"


def test_a_commit_failure_is_not_counted_as_a_finalization(monkeypatch, tmp_path):
    checkpoint, csv_path = _install_stores(monkeypatch, tmp_path)
    bot = _bot(tmp_path)
    _crash_after_n_commits(bot, 1)
    counts = bot.sweep_pending_bounce_outcomes(now=AFTER_CLOSE, wait_for_scan_window=False)
    assert counts["finalized"] == 0
    assert counts["commit_failed"] == 1
    assert "finalized" not in json.loads(checkpoint.read_text(encoding="utf-8")).get("finalized", {})


def test_the_bookkeeping_save_reports_failure_instead_of_swallowing_it(monkeypatch, tmp_path):
    """`_save_pending_bounce_outcomes` is best-effort, but it says so."""
    import bounce_bot_lib.legacy as legacy

    _install_stores(monkeypatch, tmp_path)
    bot = _bot(tmp_path)
    assert bot._save_pending_bounce_outcomes() is True
    monkeypatch.setattr(legacy.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("no")))
    assert bot._save_pending_bounce_outcomes() is False


# ---------------------------------------------------------------------------
# blocker 3: two real processes
# ---------------------------------------------------------------------------
WORKER = textwrap.dedent(
    '''
    import json, sys
    sys.path.insert(0, r"{scripts}")
    import bounce_bot_lib.legacy as legacy
    from pathlib import Path

    checkpoint = Path(r"{checkpoint}")
    csv_path = Path(r"{csv}")
    legacy.INTRADAY_BOUNCE_OUTCOME_STATE_JSON = checkpoint
    legacy.INTRADAY_BOUNCE_OUTCOMES_CSV = csv_path

    from bounce_bot_lib.legacy import BounceBot

    class Bot: pass
    bot = Bot.__new__(Bot)
    bot.pending_bounce_outcomes = {{}}
    bot._finalized_outcome_memory = {{}}
    bot._finalizing_outcome_marks = {{}}
    bot._outcome_ledger_obj = None
    bot._outcome_ledger_failed = True
    for attribute in ("PENDING_EXPIRY_SESSIONS", "FINALIZED_MEMORY", "RECOVERABLE_EVENT_TYPES",
                      "SWEEP_AFTER_SCAN_WINDOW_MINUTES", "OUTCOME_LOCK_TIMEOUT_SECONDS",
                      "OUTCOME_BAR_MINUTES"):
        setattr(bot, attribute, getattr(BounceBot, attribute))
    bot._naive_market_local = BounceBot._naive_market_local
    type(bot)._pending_lock = BounceBot.__dict__["_pending_lock"]
    for name in ("_parse_bar_time", "_json_for_learning", "_context_with_finalization",
                 "_append_bounce_outcome_row", "_append_learning_row", "_learning_csv_header",
                 "_finalized_outcome_ids", "_remember_finalized_outcome", "_exit_facts",
                 "_finalizing_outcome_ids", "_final_event_ids_in_csv", "_read_checkpoint_from_disk",
                 "_commit_checkpoint", "_outcome_transaction", "finalize_outcome_once",
                 "_load_pending_bounce_outcomes", "_save_pending_bounce_outcomes",
                 "_save_pending_bounce_outcomes_locked"):
        setattr(bot, name, getattr(BounceBot, name).__get__(bot, Bot))
    bot._mirror_outcome_row_to_ledger = lambda row, state: None

    # BOTH processes load the same checkpoint before either commits - that is
    # the race, and it is arranged deliberately.
    bot.pending_bounce_outcomes = bot._load_pending_bounce_outcomes()
    ready = Path(r"{ready}") / (sys.argv[1] + ".loaded")
    ready.write_text("loaded", encoding="utf-8")
    while len(list(Path(r"{ready}").glob("*.loaded"))) < 2:
        pass

    print(bot.finalize_outcome_once("{event}"))
    '''
)


def test_two_real_processes_write_exactly_one_final(tmp_path):
    """Sol's reproduction. Both load the same entry, then finalize in staggered order."""
    checkpoint = tmp_path / "state.json"
    csv_path = tmp_path / "outcomes.csv"
    ready = tmp_path / "ready"
    ready.mkdir()
    checkpoint.write_text(
        json.dumps({"pending": {EVENT_ID: _state()}, "finalized": {}, "finalizing": {}}),
        encoding="utf-8",
    )
    script = tmp_path / "worker.py"
    script.write_text(
        WORKER.format(scripts=SCRIPTS_DIR, checkpoint=checkpoint, csv=csv_path,
                      ready=ready, event=EVENT_ID),
        encoding="utf-8",
    )

    processes = [
        subprocess.Popen([sys.executable, str(script), name],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for name in ("a", "b")
    ]
    results = []
    for process in processes:
        out, err = process.communicate(timeout=180)
        assert process.returncode == 0, err
        results.append(out.strip().splitlines()[-1])

    finals = _finals(csv_path)
    assert len(finals) == 1, f"expected one final row, got {len(finals)}: {results}"
    assert sorted(results) == ["finalized", "skipped"], results
    disk = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert EVENT_ID in disk["finalized"] and EVENT_ID not in disk["pending"]
    assert disk["finalizing"] == {}


def test_the_transaction_takes_the_cross_process_lock():
    """An in-process RLock cannot see another process; this must not regress."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._outcome_transaction)
    assert "local_writer_lock" in source
    assert "lock_key_for_path" in source
    finalize = inspect.getsource(BounceBot.finalize_outcome_once)
    assert "_outcome_transaction()" in finalize
    assert "_read_checkpoint_from_disk()" in finalize, "disk is authoritative, not memory"


def test_the_disk_is_re_read_inside_the_lock(monkeypatch, tmp_path):
    """A stale in-memory copy must not decide to append."""
    checkpoint, csv_path = _install_stores(monkeypatch, tmp_path)
    bot = _bot(tmp_path)
    # Another process finalizes it and commits, while our copy still says pending.
    other = _bot(tmp_path)
    other.finalize_outcome_once(EVENT_ID)
    assert EVENT_ID in bot.pending_bounce_outcomes, "our copy is deliberately stale"

    assert bot.finalize_outcome_once(EVENT_ID) == "skipped"
    assert len(_finals(csv_path)) == 1
