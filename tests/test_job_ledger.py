"""Phase 2.3-2.5 (plan.md): durable job ledger with bounded retries."""

import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from job_ledger import JobLedger, job_key  # noqa: E402

NOW = datetime(2026, 7, 13, 9, 0)
DATE = "2026-07-13"


def test_full_lifecycle_and_idempotency(tmp_path):
    ledger = JobLedger(tmp_path / "ledger.jsonl")
    key = job_key(DATE, "swing_scan", "07:30")
    ledger.schedule(DATE, "swing_scan", "07:30", now=NOW)
    assert ledger.is_active(key) and not ledger.is_done(key)
    ledger.start(key, run_id="run-1", now=NOW)
    ledger.complete(key, run_id="run-1", now=NOW)
    assert ledger.is_done(key)
    assert not ledger.should_retry(key), "completed jobs never retry"
    job = ledger.get(key)
    assert job.attempt == 1 and job.run_id == "run-1"


def test_restart_replays_state_and_marks_running_as_stale(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = JobLedger(path)
    done = job_key(DATE, "swing_scan", "09:00")
    ledger.schedule(DATE, "swing_scan", "09:00", now=NOW)
    ledger.start(done, now=NOW)
    ledger.complete(done, now=NOW)
    crashed = job_key(DATE, "swing_scan", "10:00")
    ledger.schedule(DATE, "swing_scan", "10:00", now=NOW)
    ledger.start(crashed, now=NOW)
    # app dies here; a new process replays the ledger
    reborn = JobLedger(path)
    assert reborn.is_done(done)
    assert reborn.get(crashed).state == "RUNNING"
    stale = reborn.mark_stale_running(now=NOW)
    assert [j.key for j in stale] == [crashed]
    assert reborn.get(crashed).state == "STALE"
    assert reborn.should_retry(crashed), "a stale job is retryable"


def test_retry_budget_is_bounded_by_error_class(tmp_path):
    ledger = JobLedger(tmp_path / "ledger.jsonl")
    key = job_key(DATE, "swing_scan", "11:00")
    ledger.schedule(DATE, "swing_scan", "11:00", now=NOW)
    for _ in range(3):
        ledger.start(key, now=NOW)
        ledger.fail(key, error_class="unexpected", error="boom", now=NOW)
    assert ledger.get(key).attempt == 3
    assert not ledger.should_retry(key), "unexpected failures cap at 2 retries"

    ib = job_key(DATE, "swing_scan", "12:00")
    ledger.schedule(DATE, "swing_scan", "12:00", now=NOW)
    for _ in range(3):
        ledger.start(ib, now=NOW)
        ledger.fail(ib, error_class="ib_disconnected", error="socket", now=NOW)
    assert ledger.should_retry(ib), "IB disconnects get a larger budget"

    closed = job_key(DATE, "swing_scan", "13:00")
    ledger.schedule(DATE, "swing_scan", "13:00", now=NOW)
    ledger.start(closed, now=NOW)
    ledger.fail(closed, error_class="no_market_session", now=NOW)
    assert not ledger.should_retry(closed), "no-session failures never retry"


def test_skip_is_terminal_and_dated_queries_work(tmp_path):
    ledger = JobLedger(tmp_path / "ledger.jsonl")
    key = job_key(DATE, "swing_scan", "07:30")
    ledger.schedule(DATE, "swing_scan", "07:30", now=NOW)
    ledger.skip(key, reason="holiday", now=NOW)
    assert ledger.is_done(key)
    other_day = job_key("2026-07-14", "swing_scan", "07:30")
    ledger.schedule("2026-07-14", "swing_scan", "07:30", now=NOW)
    assert [j.key for j in ledger.jobs_for_date(DATE)] == [key]
    assert [j.key for j in ledger.jobs_for_date("2026-07-14")] == [other_day]
