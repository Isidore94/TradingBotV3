"""Overnight job runner: ledger, idempotency, and the skip rules.

The behaviours that matter when nobody is watching: every outcome leaves a
ledger row, a completed job is never redone when the task fires again, a job
that cannot finish before the window closes is skipped rather than started,
a failure never takes the rest of the night down with it, and an unreachable
store means nothing runs at all rather than something half-runs.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")
#: 02:00 ET on a Wednesday: inside the trader's 01:00-09:00 window.
OVERNIGHT = datetime(2026, 8, 12, 2, 0, tzinfo=ET)


def _store_ok(tmp_path):
    from ai_jobs import store

    return mock.patch.object(store, "store_available", return_value=(True, "ready"))


def _window_open():
    from ai_jobs import window

    return mock.patch.object(window, "launch_allowed", return_value=(True, "window open"))


def _no_session_block():
    from ai_jobs import window

    return mock.patch.object(window, "market_session_block", return_value="")


def _slot(name, fn, reserve=15.0, enabled=True):
    from ai_jobs.runner import JobSlot

    return JobSlot(name=name, run=fn, reserve_minutes=reserve, enabled=enabled)


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_session_date_attributes_an_overnight_run_to_the_prior_session():
    from ai_jobs import runner

    # 02:00 ET Wednesday is processing Tuesday's evidence.
    assert runner.session_date_for(datetime(2026, 8, 12, 2, 0, tzinfo=ET)) == "2026-08-11"
    # An evening run belongs to the day that just closed.
    assert runner.session_date_for(datetime(2026, 8, 11, 19, 0, tzinfo=ET)) == "2026-08-11"


def test_successful_job_writes_an_ok_row_with_its_outputs(tmp_path):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    calls = []

    def job(*, session_date, now):
        calls.append(session_date)
        return {"model": "gemma3:12b", "outputs": ["a.md"], "reason": "did the thing"}

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        report = runner.run_slots([_slot("ai_summary", job)], now=OVERNIGHT, ledger_path=led)

    assert report.ran == 1 and report.failed == 0 and report.skipped == 0
    row = _rows(led)[0]
    assert row["job"] == "ai_summary"
    assert row["status"] == "ok"
    assert row["session_date"] == "2026-08-11"
    assert row["model"] == "gemma3:12b"
    assert row["outputs"] == ["a.md"]
    assert calls == ["2026-08-11"]


def test_a_completed_job_is_not_redone_when_the_task_fires_again(tmp_path):
    """Task Scheduler fires every 30 min through the window; that must be safe."""
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    runs = []

    def job(*, session_date, now):
        runs.append(session_date)
        return {}

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        runner.run_slots([_slot("ai_summary", job)], now=OVERNIGHT, ledger_path=led)
        runner.run_slots([_slot("ai_summary", job)], now=OVERNIGHT, ledger_path=led)

    assert runs == ["2026-08-11"], "second launch must not redo completed work"
    assert len(_rows(led)) == 1


def test_a_failed_job_is_retried_by_the_next_launch(tmp_path):
    """The inverse of the above: failure is exactly what re-firing is for."""
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    attempts = []

    def flaky(*, session_date, now):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("endpoint asleep")
        return {}

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        first = runner.run_slots([_slot("ai_summary", flaky)], now=OVERNIGHT, ledger_path=led)
        second = runner.run_slots([_slot("ai_summary", flaky)], now=OVERNIGHT, ledger_path=led)

    assert first.failed == 1
    assert second.ran == 1
    statuses = [row["status"] for row in _rows(led)]
    assert statuses == ["failed", "ok"]
    assert "endpoint asleep" in _rows(led)[0]["error"]


def test_one_failure_does_not_take_down_the_rest_of_the_night(tmp_path):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    ran_second = []

    def boom(*, session_date, now):
        raise RuntimeError("model load failed")

    def fine(*, session_date, now):
        ran_second.append(True)
        return {}

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        report = runner.run_slots(
            [_slot("first", boom), _slot("second", fine)], now=OVERNIGHT, ledger_path=led
        )

    assert ran_second == [True]
    assert report.failed == 1 and report.ran == 1


def test_a_job_that_cannot_finish_is_skipped_with_a_reason(tmp_path):
    from ai_jobs import runner, window

    led = tmp_path / "ledger.jsonl"
    ran = []

    with _store_ok(tmp_path), _no_session_block(), mock.patch.object(
        window, "launch_allowed", return_value=(False, "only 5 min left in the window")
    ):
        report = runner.run_slots(
            [_slot("ai_summary", lambda **k: ran.append(True))], now=OVERNIGHT, ledger_path=led
        )

    assert ran == []
    assert report.skipped == 1
    row = _rows(led)[0]
    assert row["status"] == "skipped"
    assert "5 min left" in row["reason"]


def test_unreachable_store_means_nothing_runs_at_all(tmp_path):
    from ai_jobs import runner, store

    ran = []
    with mock.patch.object(
        store, "store_available", return_value=(False, "AI store is unreachable: NAS asleep")
    ):
        report = runner.run_slots(
            [_slot("ai_summary", lambda **k: ran.append(True))], now=OVERNIGHT
        )

    assert ran == []
    assert report.store_ok is False
    assert "unreachable" in report.store_reason
    # No ledger row either -- the ledger lives in the store we cannot reach.
    assert report.results == []


def test_reaching_market_hours_stops_the_remaining_jobs(tmp_path):
    """Sec 6.1: finish the current call, then stop gracefully."""
    from ai_jobs import runner, window

    led = tmp_path / "ledger.jsonl"
    ran = []

    def job_one(*, session_date, now):
        ran.append("one")
        return {}

    def job_two(*, session_date, now):
        ran.append("two")
        return {}

    # The open arrives *while* job one runs: clear on the pre-launch check,
    # blocking on the post-job re-read.
    with _store_ok(tmp_path), _window_open(), mock.patch.object(
        window,
        "market_session_block",
        side_effect=["", "market session is live", "market session is live"],
    ):
        runner.run_slots(
            [_slot("one", job_one), _slot("two", job_two)], now=OVERNIGHT, ledger_path=led
        )

    assert ran == ["one"], "the second job must not start once the session is live"


def test_disabled_slots_stay_dormant(tmp_path):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    ran = []
    with _store_ok(tmp_path), _window_open(), _no_session_block():
        report = runner.run_slots(
            [_slot("staged", lambda **k: ran.append(True), enabled=False)],
            now=OVERNIGHT,
            ledger_path=led,
        )

    assert ran == []
    assert report.results == []


def test_only_runs_the_named_slot(tmp_path):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    ran = []
    with _store_ok(tmp_path), _window_open(), _no_session_block():
        runner.run_slots(
            [_slot("a", lambda **k: ran.append("a")), _slot("b", lambda **k: ran.append("b"))],
            now=OVERNIGHT,
            only="b",
            ledger_path=led,
        )

    assert ran == ["b"]


def test_force_overrides_the_window_and_the_completed_check(tmp_path):
    from ai_jobs import runner, window

    led = tmp_path / "ledger.jsonl"
    ran = []

    with _store_ok(tmp_path), _no_session_block(), mock.patch.object(
        window, "launch_allowed", return_value=(False, "outside the window")
    ):
        runner.run_slots([_slot("a", lambda **k: ran.append(1))], now=OVERNIGHT,
                         force=True, ledger_path=led)
        runner.run_slots([_slot("a", lambda **k: ran.append(1))], now=OVERNIGHT,
                         force=True, ledger_path=led)

    assert len(ran) == 2, "--force is the manual override for exactly this"


def test_default_slate_registers_both_phase_1_jobs():
    from ai_jobs import runner

    slots = runner.default_slots()
    assert [slot.name for slot in slots] == ["ai_summary", "ticker_briefs"]
    assert all(slot.enabled for slot in slots)


def test_entry_point_reports_store_failure_as_exit_2():
    import run_ai_jobs
    from ai_jobs import store

    with mock.patch.object(store, "store_available", return_value=(False, "NAS asleep")):
        assert run_ai_jobs.main([]) == 2


def test_entry_point_reports_job_failure_as_exit_1(tmp_path):
    import run_ai_jobs
    from ai_jobs import runner, window

    def boom(*, session_date, now):
        raise RuntimeError("nope")

    with _store_ok(tmp_path), _window_open(), _no_session_block(), mock.patch.object(
        runner, "default_slots", return_value=[_slot("ai_summary", boom)]
    ), mock.patch.object(runner.ledger, "ledger_path", return_value=tmp_path / "l.jsonl"):
        assert run_ai_jobs.main([]) == 1


def test_entry_point_success_is_exit_0(tmp_path):
    import run_ai_jobs
    from ai_jobs import runner

    with _store_ok(tmp_path), _window_open(), _no_session_block(), mock.patch.object(
        runner, "default_slots", return_value=[_slot("ai_summary", lambda **k: {})]
    ), mock.patch.object(runner.ledger, "ledger_path", return_value=tmp_path / "l.jsonl"):
        assert run_ai_jobs.main([]) == 0


# ---------------------------------------------------------------------------
# --force is a window convenience, never a hard-rule override
# (checkpoint review 2026-08-08 second review)
# ---------------------------------------------------------------------------
def test_force_skips_the_window_checks(tmp_path):
    from ai_jobs import runner, window

    led = tmp_path / "ledger.jsonl"
    ran = []

    def job(*, session_date, now):
        ran.append(session_date)
        return {}

    with _store_ok(tmp_path), _no_session_block(), mock.patch.object(
        window, "launch_allowed", return_value=(False, "outside the off-hours window")
    ):
        report = runner.run_slots(
            [_slot("ai_summary", job)], now=OVERNIGHT, force=True, ledger_path=led
        )

    assert ran, "--force must still get past a shut window"
    # ...but it publishes as a manual test, never as session coverage.
    assert report.manual == 1
    assert report.ran == 0
    assert _rows(led)[0]["status"] == "manual_test"
    from ai_jobs import ledger

    assert ledger.completed_jobs("2026-08-11", path=led) == set()


def test_force_does_not_get_past_the_market_session_block(tmp_path):
    """Plan sec 2 is a hard rule; a CLI flag that switches it off is not one.

    --force used to short-circuit straight to (True, "forced"), so an operator
    running the job at 11:00 on a Tuesday would load a 14GB model onto the desk
    mid-session, competing with the trading complement it is forbidden to
    compete with.
    """
    from ai_jobs import runner, window

    led = tmp_path / "ledger.jsonl"
    ran = []

    def job(*, session_date, now):  # pragma: no cover - must never run
        ran.append(session_date)
        return {}

    with _store_ok(tmp_path), _window_open(), mock.patch.object(
        window, "market_session_block", return_value="market session is live (09:30-16:00 ET)"
    ):
        report = runner.run_slots(
            [_slot("ai_summary", job)], now=OVERNIGHT, force=True, ledger_path=led
        )

    assert ran == [], "--force must not run a job during market hours"
    assert report.skipped == 1
    row = _rows(led)[0]
    assert row["status"] == "skipped"
    assert "market session is live" in row["reason"]


def test_force_does_not_get_past_the_post_job_session_break(tmp_path):
    # The open arriving mid-run is exactly when stopping matters most, so
    # --force does not exempt the between-jobs re-read either.
    from ai_jobs import runner, window

    led = tmp_path / "ledger.jsonl"
    ran = []

    def job_one(*, session_date, now):
        ran.append("one")
        return {}

    def job_two(*, session_date, now):  # pragma: no cover - must never run
        ran.append("two")
        return {}

    with _store_ok(tmp_path), _window_open(), mock.patch.object(
        window,
        "market_session_block",
        side_effect=["", "market session is live", "market session is live"],
    ):
        runner.run_slots(
            [_slot("one", job_one), _slot("two", job_two)],
            now=OVERNIGHT,
            force=True,
            ledger_path=led,
        )

    assert ran == ["one"]


# ---------------------------------------------------------------------------
# session identity (Sol 5.6 verification review, item 2)
# ---------------------------------------------------------------------------
def test_a_weekend_run_is_attributed_to_fridays_session():
    """The defect, directly: a Saturday run filed its work under Saturday."""
    from ai_jobs import runner

    saturday_evening = datetime(2026, 8, 8, 21, 0, tzinfo=ET)
    assert saturday_evening.weekday() == 5
    assert runner.session_date_for(saturday_evening) == "2026-08-07"
    assert runner.is_session_day(saturday_evening) is False


def test_a_weekday_overnight_run_is_attributed_to_the_prior_session():
    from ai_jobs import runner

    assert runner.session_date_for(datetime(2026, 8, 12, 2, 0, tzinfo=ET)) == "2026-08-11"
    assert runner.is_session_day(datetime(2026, 8, 12, 2, 0, tzinfo=ET)) is True


def test_a_holiday_evening_run_walks_back_past_the_holiday():
    from ai_jobs import runner

    thanksgiving_evening = datetime(2026, 11, 26, 21, 0, tzinfo=ET)
    assert runner.session_date_for(thanksgiving_evening) == "2026-11-25"
    assert runner.is_session_day(thanksgiving_evening) is False


def test_an_unanswerable_calendar_stops_the_run_rather_than_guessing(tmp_path):
    from market_calendar import SessionCalendarError

    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    ran = []

    with mock.patch.object(
        runner, "session_date_for", side_effect=SessionCalendarError("no calendar")
    ):
        report = runner.run_slots(
            [_slot("ai_summary", lambda **k: ran.append(True))],
            now=OVERNIGHT,
            ledger_path=led,
        )

    assert ran == []
    assert report.store_ok is False
    assert "session calendar cannot answer" in report.store_reason
    assert report.session_date == ""
    assert not led.exists(), "no row may be keyed to a session we could not resolve"


def test_a_weekend_firing_produces_the_missing_friday_brief(tmp_path):
    # Friday's session has no canonical artifact, so the Saturday firing does
    # the work -- keyed to Friday.
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    seen = []

    def job(*, session_date, now):
        seen.append(session_date)
        return {}

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        report = runner.run_slots(
            [_slot("ai_summary", job)],
            now=datetime(2026, 8, 8, 21, 0, tzinfo=ET),
            ledger_path=led,
        )

    assert seen == ["2026-08-07"]
    assert report.ran == 1
    assert _rows(led)[0]["session_date"] == "2026-08-07"


def test_a_weekend_firing_over_a_covered_session_records_no_session_once(tmp_path):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    runs = []

    def job(*, session_date, now):
        runs.append(session_date)
        return {}

    saturday = datetime(2026, 8, 8, 21, 0, tzinfo=ET)
    with _store_ok(tmp_path), _window_open(), _no_session_block():
        runner.run_slots([_slot("ai_summary", job)], now=saturday, ledger_path=led)
        # Two more firings of the every-30-minutes task.
        second = runner.run_slots([_slot("ai_summary", job)], now=saturday, ledger_path=led)
        runner.run_slots([_slot("ai_summary", job)], now=saturday, ledger_path=led)

    assert runs == ["2026-08-07"], "the covered session is not redone"
    statuses = [row["status"] for row in _rows(led)]
    assert statuses == ["ok", "skipped"], "one no-session row, not one per repeat"
    skip = _rows(led)[1]
    assert skip["session_date"] == "2026-08-07"
    assert "no session" in skip["reason"]
    assert "weekend" in skip["reason"]
    assert skip["no_session"] is True
    assert second.skipped == 1


def test_a_weekday_repeat_over_a_covered_session_stays_silent(tmp_path):
    # The ~27 firings of a healthy weeknight must not each leave a row.
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    with _store_ok(tmp_path), _window_open(), _no_session_block():
        runner.run_slots([_slot("ai_summary", lambda **k: {})], now=OVERNIGHT, ledger_path=led)
        runner.run_slots([_slot("ai_summary", lambda **k: {})], now=OVERNIGHT, ledger_path=led)

    assert [row["status"] for row in _rows(led)] == ["ok"]


def test_a_manual_run_never_satisfies_the_canonical_completion_check(tmp_path):
    from ai_jobs import ledger, runner

    led = tmp_path / "ledger.jsonl"
    runs = []

    def job(*, session_date, now):
        runs.append(session_date)
        return {}

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        runner.run_slots([_slot("ai_summary", job)], now=OVERNIGHT, force=True, ledger_path=led)
        # The scheduled run afterwards still has work to do.
        runner.run_slots([_slot("ai_summary", job)], now=OVERNIGHT, ledger_path=led)

    assert runs == ["2026-08-11", "2026-08-11"]
    assert [row["status"] for row in _rows(led)] == ["manual_test", "ok"]
    assert ledger.completed_jobs("2026-08-11", path=led) == {"ai_summary"}


def test_a_correction_retracts_a_coverage_claim_without_rewriting_it(tmp_path):
    from ai_jobs import ledger

    led = tmp_path / "ledger.jsonl"
    original = ledger.record(
        job="ai_summary", status=ledger.STATUS_OK, session_date="2026-08-08",
        reason="summary for 2026-08-08", path=led,
    )
    assert ledger.completed_jobs("2026-08-08", path=led) == {"ai_summary"}

    ledger.mark_noncanonical(
        job="ai_summary",
        session_date="2026-08-08",
        reason="2026-08-08 was a Saturday; the exchange never opened",
        corrects=[original["finished_at"]],
        path=led,
    )

    assert ledger.completed_jobs("2026-08-08", path=led) == set()
    rows = _rows(led)
    assert len(rows) == 2, "the ledger is append-only: the original row stays"
    assert rows[0] == original
    assert rows[1]["status"] == "correction"
    assert rows[1]["noncanonical"] is True
    assert rows[1]["corrects"] == [original["finished_at"]]

    # A genuine run afterwards re-establishes the claim.
    ledger.record(job="ai_summary", status=ledger.STATUS_OK, session_date="2026-08-08", path=led)
    assert ledger.completed_jobs("2026-08-08", path=led) == {"ai_summary"}
