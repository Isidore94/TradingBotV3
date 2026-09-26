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


def test_real_journal_slot_normalizes_its_uppercase_status_into_the_ledger(tmp_path, monkeypatch):
    """Exercise the production default-slot wrapper, not a lookalike slot.

    The journal runner returns ``OK``/``FAILED`` for its CLI/UI callers.  The
    overnight ledger speaks lowercase, and retry accounting happens only after
    that exact return value crosses this seam.
    """
    from ai_jobs import runner
    import journal_runner

    led = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(
        journal_runner,
        "run_nightly_journal_import",
        lambda *, trigger: {"status": "OK", "messages": ["quiet night"]},
    )
    journal_slot = runner.default_slots()[0]

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        report = runner.run_slots([journal_slot], now=OVERNIGHT, ledger_path=led)

    assert report.ran == 1 and report.failed == 0
    assert _rows(led)[0]["status"] == "ok"


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


#: The nightly slate, pinned. **Decision 0018**
#: (`docs/decisions/0018-deterministic-stage-before-narration.md`, 2026-09-04):
#: three stages - every deterministic slot, then narration, then the
#: model-gated slots. A later phase appends INSIDE its stage and never reorders
#: across stages, so this tuple is the one place the order is written down in a
#: test and the only place a future packet edits.
EXPECTED_SLOT_ORDER = (
    # stage 1 - deterministic, no model, minutes not hours
    "journal_import",
    "journal_auto_tag",
    "veto_cohort_grading",
    "like_cohort_grading",
    "sidecar_completion",
    "pass_cohort_grading",
    "rejection_cohort_grading",
    "preference_trade_outcomes",
    "outcome_sweep",
    "evidence_report",
    "daily_digest",
    # WS-TH (2026-09-12): appended at the END of the deterministic stage. It
    # reads `theta_picks.jsonl` and the daily bars and feeds nothing above it,
    # so it sits after the digest and stays ahead of `ai_summary`.
    "theta_pick_grading",
    # TJ-15 (2026-09-19): what the misses had in common. Deterministic, no
    # model. It is INSIDE stage 1 and deliberately ahead of `day_review_facts`:
    # `_STAGE_ONE_LAST_SLOT` is that name and `_deterministic_stage` walks up to
    # and including it, so a slot appended after it would leave the Sunday
    # slate. Nothing here reads the measured report and nothing there reads this
    # pack, so only the position is a choice - and only one of the two runs on a
    # Sunday.
    # TJ-10 (2026-09-20): close the market reads whose horizon has matured.
    # Deterministic, no model. Beside the cohort graders in kind - a decision
    # measured after the fact - and INSIDE stage 1 for the same reason
    # `miss_contrast` is: `_STAGE_ONE_LAST_SLOT` is `day_review_facts` and a slot
    # appended after that name leaves the Sunday slate. Nothing below it reads
    # the read ledger and it reads nothing above it.
    "read_grades_mature",
    "miss_contrast",
    # TJ-16 (2026-09-20): what leads to a good call - the trader's right reads
    # against their wrong ones, through the same `evidence_contrast.contrast`.
    # Deterministic, no model. DIRECTLY after `miss_contrast` (the same question
    # asked of the other half of the record) and after `read_grades_mature`,
    # which closes the reads it counts - and still ahead of `day_review_facts`
    # for the Sunday-slate reason above it.
    "prediction_contrast",
    # S11 (2026-09-26): exit-window truth from the M5 outcome log. Deterministic,
    # no model; after `outcome_sweep` finalizes the day's outcomes, and ahead of
    # `day_review_facts` so the Sunday slate keeps it. Nothing above reads its file.
    "exit_windows",
    # WS-10D (2026-09-12): the Market Journal's weekly/monthly/quarterly rollups.
    # Deterministic, no model; it reads the daily stories and the exchange calendar
    # and feeds nothing above it, so it CLOSES the deterministic stage.
    "market_story_rollups",
    # WS-RP (2026-09-13): one measured report for the session - the five
    # WISHLIST 10K answers with their populations. Deterministic, no model; it
    # reads what the slots above it wrote and feeds nothing above it, so it
    # CLOSES the deterministic stage.
    "measured_report",
    # TJ-17: current and recent saved facts before any story.
    "day_review_facts",
    # S12 (2026-09-26): the SP4 family evidence. Deterministic, no model; the
    # brief puts it directly after `day_review_facts`, so it now CLOSES stage 1
    # (`_STAGE_ONE_LAST_SLOT`) and keeps its Sunday slot. Nothing above reads it.
    "family_side_evidence",
    # stage 2 - the original pair moved here by decision 0018; Phase 0.31
    # appends the bounded market-story narration inside the same stage.
    "ai_summary",
    # TJ-4 (2026-09-20): the overnight day story and the rolling D1 view.
    # Appended INSIDE stage 2 and deliberately AHEAD of `ticker_briefs`: gate
    # #158 reads the ledger for a day story finished before 23:30 Pacific, and
    # the briefs reserve 120 minutes in front of it. It cannot go further
    # forward either - `ai_summary` sits directly after `family_side_evidence` and
    # two other pins say so.
    "day_review_narration",
    # R1 (2026-09-26): the Day Review Show reads that night's verified story, so
    # it sits DIRECTLY after it, still inside stage 2 (decision 0018 unchanged).
    "day_review_show",
    # TJ-16 item 4 (2026-09-20): grounded codes for the trader's own words.
    # A local MEDIUM model slot, so it is in stage 2 - after `ai_summary`
    # because AI-R3 pins `day_review_facts` directly before that name, and
    # before `ticker_briefs`, whose two hours of reserve it must not queue
    # behind for seconds of work.
    "observation_tags",
    # TJ-5 (2026-09-20): the week story, `plan.md` §12.4 TJ-13 item 9's Stage 2
    # position - directly after the word tagger and directly before the briefs,
    # whose 120 minutes of reserve it must not queue behind. SATURDAY ONLY: it
    # is in `runner.WEEKEND_ONLY_SLOTS`, so `slots_for("weeknight")` never
    # offers it and Sunday picks it up only when Saturday left it unfinished.
    "week_review_narration",
    # TJ-9E (2026-09-21): the night's reading of the trader's own exit note.
    # A local MEDIUM model slot, so it is in stage 2 - after the week story
    # because TJ-5 pins that name DIRECTLY after `observation_tags`, and before
    # `ticker_briefs`, whose two hours of reserve it must not queue behind.
    "exit_note_fields",
    "ticker_briefs",
    # Econ morning brief (2026-09-24): the next session's "what to watch" from
    # the newest pasted brief. Stage 2, directly after the briefs (the slots
    # before them are pinned closed, `week_questions` is pinned after the
    # market story); a local model words it and the fixed parser owns every time.
    "econ_brief",
    "market_story_narration",
    # Day Recap coach (2026-09-23): answer the trader's Week Review questions
    # from the day/week records, cited. End of stage 2: after `day_review_facts`
    # rebuilt the records, and not directly after it (`ai_summary` is pinned there).
    "week_questions",
    # stage 3 - the model-gated slots, unchanged
    "journal_enrichment",
    "review_policy_draft",
    # P1-7 7b (2026-09-25): challenges to the trader's plan. Stage 3, before
    # `setup_research` because `improvement_ideas` is pinned last.
    "plan_review",
    # P1-4 4d (2026-09-25): Saturday-only setup-keys narration, inside stage 3,
    # directly before `setup_research` (only `improvement_ideas` may follow it).
    "setup_keys_narration",
    "setup_research",
    # TJ-6 (2026-09-20), appended LAST inside stage 3: it reads what the rest of
    # the night wrote and feeds nothing.
    "improvement_ideas",
)


def test_default_slate_runs_the_deterministic_stage_before_narration():
    """R7 §9 step 10 put `journal_import` at the front, and it belongs there.

    This asserted exactly ["ai_summary", "ticker_briefs"] until then, then
    "later phases append; they never reorder these" with `journal_import` and
    `journal_auto_tag` as its two sanctioned exceptions
    (`docs/LOCAL_AI_AUTOMATION_PLAN.md` §6.4c, promoted into R7 §6).

    **Decision 0018 replaced that rule**: the two narration slots held up to
    two and a half hours of reserve ahead of every deterministic slot, and on
    2026-09-01 the night ran six hours - so the cheap deterministic work that
    nothing narrated feeds was the work most likely to be skipped for want of
    window. The order is now three stages, and this tuple is where it is
    written down.
    """
    from ai_jobs import runner

    slots = runner.default_slots()
    names = tuple(slot.name for slot in slots)
    assert names == EXPECTED_SLOT_ORDER
    assert all(slot.enabled for slot in slots)
    by_name = {slot.name: slot for slot in slots}
    # Both long slots are capped.
    assert by_name["ticker_briefs"].max_attempts == 3
    # UPDATED by TJ-13A's review round (2026-09-19). This line asserted 0 with
    # the comment "the cheap one keeps retrying all window", and that comment
    # had it backwards by then: `ai_summary` was the EXPENSIVE slot, and it was
    # uncapped precisely because spending hours per attempt meant the 30-minute
    # firings could never repeat it inside one night. TJ-13A item 3 made a dead
    # endpoint give up on its first call, which turned the missing cap into a
    # loop - a degrading summary ran 10 times over one weekend night's firings,
    # writing 10 ledger rows and 10 export sets for a single session. plan.md
    # §12.3: every slot sets `max_attempts`, never 0.
    assert by_name["ai_summary"].max_attempts == 3
    # Seconds of work, so it reserves almost nothing - and it is capped, because
    # a broker that is down stays down and should not spend the whole window.
    # P1-3 3d (2026-09-24): plus the ~7 min of IBKR Flex not-ready waits.
    assert by_name["journal_import"].reserve_minutes == 12.0
    assert by_name["journal_import"].max_attempts == 3


# ---------------------------------------------------------------------------
# TB-4: a session's attempts are finite
# ---------------------------------------------------------------------------
def _capped_slot(fn, attempts=3):
    from ai_jobs.runner import JobSlot

    return JobSlot(name="ticker_briefs", run=fn, max_attempts=attempts)


def test_a_deterministically_failing_job_stops_after_its_attempt_cap(tmp_path):
    """11 consecutive failures and ~111 minutes of inference, once."""
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    runs = []

    def failing(*, session_date, now):
        runs.append(session_date)
        raise RuntimeError(f"endpoint down (call {len(runs)})")

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        for _ in range(6):
            runner.run_slots([_capped_slot(failing)], now=OVERNIGHT, ledger_path=led)

    assert len(runs) == 3, "the cap is spent, then the job is finished for the night"
    rows = _rows(led)
    terminal = [row for row in rows if row.get("terminal")]
    assert len(terminal) == 1, "the marker is recorded once, not once per firing"
    assert terminal[0]["status"] == "skipped"
    assert "cap is 3" in terminal[0]["reason"]
    # Every later firing costs a ledger read and nothing else.
    assert len(rows) == 4


def test_two_identical_failures_end_the_night_before_the_cap(tmp_path):
    """Same error twice is deterministic; a third try is a third wasted hour."""
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    runs = []

    def failing(*, session_date, now):
        runs.append(1)
        raise RuntimeError("local AI endpoint at http://127.0.0.1:11434 is unreachable")

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        for _ in range(4):
            runner.run_slots([_capped_slot(failing)], now=OVERNIGHT, ledger_path=led)

    assert len(runs) == 2
    terminal = [row for row in _rows(led) if row.get("terminal")]
    assert len(terminal) == 1
    assert "failed identically" in terminal[0]["reason"]


def test_transient_failures_still_self_heal_within_the_cap(tmp_path):
    """The retry ladder is the point; only the grind is being ended."""
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    calls = []

    def flaky(*, session_date, now):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("NAS asleep")
        return {"status": "ok", "reason": "briefed"}

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        for _ in range(3):
            runner.run_slots([_capped_slot(flaky)], now=OVERNIGHT, ledger_path=led)

    assert len(calls) == 2, "recovered, then skipped as already completed"
    assert not [row for row in _rows(led) if row.get("terminal")]


def test_a_cheap_skip_never_spends_an_attempt(tmp_path):
    """An unmounted Drive must not burn the night's allowance in seconds."""
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    calls = []

    def refusing(*, session_date, now):
        calls.append(1)
        if len(calls) <= 4:
            return {"status": "skipped", "reason": "watchlists unreadable"}
        return {"status": "ok", "reason": "briefed"}

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        for _ in range(5):
            runner.run_slots([_capped_slot(refusing)], now=OVERNIGHT, ledger_path=led)

    assert len(calls) == 5
    assert not [row for row in _rows(led) if row.get("terminal")]


def test_force_overrides_the_terminal_marker(tmp_path):
    """The cap protects an unattended night, not an operator at the desk."""
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    runs = []

    def failing(*, session_date, now):
        runs.append(1)
        raise RuntimeError(f"still broken {len(runs)}")

    with _store_ok(tmp_path), _window_open(), _no_session_block():
        for _ in range(5):
            runner.run_slots([_capped_slot(failing)], now=OVERNIGHT, ledger_path=led)
        assert len(runs) == 3
        runner.run_slots([_capped_slot(failing)], now=OVERNIGHT, force=True, ledger_path=led)

    assert len(runs) == 4


def test_the_cap_is_per_session(tmp_path):
    from ai_jobs import ledger

    led = tmp_path / "ledger.jsonl"
    for _ in range(3):
        ledger.record(
            job="ticker_briefs", status=ledger.STATUS_FAILED, session_date="2026-08-11",
            error="boom", path=led,
        )
    assert ledger.attempt_cap_reason(
        "ticker_briefs", "2026-08-11", max_attempts=3, path=led
    )
    assert not ledger.attempt_cap_reason(
        "ticker_briefs", "2026-08-12", max_attempts=3, path=led
    )
    assert not ledger.has_terminal_marker("ticker_briefs", "2026-08-11", path=led)


def test_entry_point_reports_store_failure_as_exit_2():
    import run_ai_jobs
    from ai_jobs import store

    with mock.patch.object(store, "store_available", return_value=(False, "NAS asleep")):
        assert run_ai_jobs.main([]) == 2


#: TJ-13A item 2 made the slate depend on WHICH NIGHT it is, so the two
#: entry-point tests below pin the night rather than inheriting the day the
#: suite happens to run on. Without the pin a Saturday-evening suite run would
#: build the Saturday slate, whose `weekly_synthesis` comes from
#: `optional_slots()` and is therefore NOT covered by the `default_slots` patch
#: - the test would call the real job. The slot is named `journal_import`
#: because `ai_summary` is not on a weeknight slate any more; nothing else
#: about either assertion changed.
def _weeknight(runner):
    return mock.patch.object(runner, "night_kind", return_value="weeknight")


def test_entry_point_reports_job_failure_as_exit_1(tmp_path):
    import run_ai_jobs
    from ai_jobs import runner

    def boom(*, session_date, now):
        raise RuntimeError("nope")

    with _store_ok(tmp_path), _window_open(), _no_session_block(), _weeknight(
        runner
    ), mock.patch.object(
        runner, "default_slots", return_value=[_slot("journal_import", boom)]
    ), mock.patch.object(runner.ledger, "ledger_path", return_value=tmp_path / "l.jsonl"):
        assert run_ai_jobs.main([]) == 1


def test_entry_point_success_is_exit_0(tmp_path):
    import run_ai_jobs
    from ai_jobs import runner

    ran = []
    with _store_ok(tmp_path), _window_open(), _no_session_block(), _weeknight(
        runner
    ), mock.patch.object(
        runner,
        "default_slots",
        return_value=[_slot("journal_import", lambda **k: ran.append(1) or {})],
    ), mock.patch.object(runner.ledger, "ledger_path", return_value=tmp_path / "l.jsonl"):
        assert run_ai_jobs.main([]) == 0
    assert ran, "the slate must actually have held the slot it was given"


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


# ---------------------------------------------------------------------------
# R10.0: a failure with no explanation is not an observable failure.
#
# `journal_import` failed on 20 nightly runs with `error=""` AND `reason=""` in
# the ai_store ledger, so all that survived was "something went wrong". The
# explanation was never missing: `run_nightly_journal_import` returns it in
# `messages`, and the runner's normal (non-exception) path reads only `reason`,
# so the diagnostic was produced and then dropped at the seam.
# ---------------------------------------------------------------------------
def test_a_failing_job_records_the_messages_it_returned(tmp_path):
    """The job's own explanation must survive into the ledger row."""
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    def job(**k):
        return {
            "status": "FAILED",
            "ok": False,
            "messages": [
                "journal database requires trader-present preparation in the GUI; "
                "nightly import refused without migrating it"
            ],
        }
    with _store_ok(tmp_path), _window_open(), _no_session_block():
        runner.run_slots([_slot("journal_import", job)], now=OVERNIGHT, ledger_path=led)
    row = _rows(led)[-1]
    assert row["status"] == "failed"
    assert row["reason"], "a failure with a blank reason is not observable"
    assert "trader-present preparation" in row["reason"]


def test_a_failing_job_with_nothing_to_say_still_says_so(tmp_path):
    """Silence is filled with a named placeholder, never left blank.

    A blank reason and "the job declined to explain itself" look identical in a
    file and are completely different to debug.
    """
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    with _store_ok(tmp_path), _window_open(), _no_session_block():
        runner.run_slots(
            [_slot("journal_import", lambda **k: {"status": "FAILED"})],
            now=OVERNIGHT,
            ledger_path=led,
        )
    row = _rows(led)[-1]
    assert row["status"] == "failed"
    assert row["reason"], "a failure must never record an empty reason"
    assert "journal_import" in row["reason"]


def test_a_successful_job_is_not_given_a_manufactured_reason(tmp_path):
    """The floor applies to failures only; an ok row stays quiet."""
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    with _store_ok(tmp_path), _window_open(), _no_session_block():
        runner.run_slots(
            [_slot("ai_summary", lambda **k: {"status": "ok"})],
            now=OVERNIGHT,
            ledger_path=led,
        )
    assert _rows(led)[-1]["reason"] == ""
