"""Plan to 8/10, packets P3 and P4 part 2: night AI runner plumbing."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# Budget order: plan_review and improvement_ideas rank after enrichment
# ---------------------------------------------------------------------------


def test_budget_priority_puts_plan_review_and_ideas_between_enrichment_and_tags():
    from ai_jobs import runner

    assert runner.MODEL_SLOT_PRIORITY == (
        "daily_digest",
        "day_review_narration",
        "market_story_narration",
        "setup_research",
        "journal_enrichment",
        "plan_review",
        "improvement_ideas",
        "observation_tags",
    )
    order = sorted(
        ["ticker_briefs", "observation_tags", "improvement_ideas", "econ_brief",
         "plan_review", "journal_enrichment"],
        key=runner.model_slot_priority,
    )
    assert order == [
        "journal_enrichment", "plan_review", "improvement_ideas", "observation_tags",
        "econ_brief", "ticker_briefs",
    ]


# ---------------------------------------------------------------------------
# note_vocabulary_audit is gone: its report had no reader
# ---------------------------------------------------------------------------


def test_note_vocabulary_audit_slot_and_module_are_gone():
    import importlib.util

    from ai_jobs import runner

    names = [slot.name for slot in runner.default_slots() + runner.optional_slots()]
    assert "note_vocabulary_audit" not in names
    assert importlib.util.find_spec("ai_jobs.note_vocabulary_audit") is None


# ---------------------------------------------------------------------------
# Goal map: every slot names the goal it serves; its ledger rows carry it
# ---------------------------------------------------------------------------


LEAD_GOAL_MAP = {
    "journal_import": "journal",
    "journal_auto_tag": "journal",
    "journal_enrichment": "journal",
    "preference_trade_outcomes": "journal",
    "setup_keys_narration": "permutations",
    "read_grades_mature": "market_read",
    "prediction_contrast": "market_read",
    "market_story_rollups": "market_read",
    "market_story_narration": "market_read",
    "econ_brief": "market_read",
    "daily_digest": "market_read",
    "day_review_facts": "coaching",
    "day_review_narration": "coaching",
    "week_review_narration": "coaching",
    "week_questions": "coaching",
    "exit_note_fields": "coaching",
    "observation_tags": "coaching",
    "plan_review": "coaching",
    "improvement_ideas": "coaching",
    "ticker_briefs": "coaching",
    "outcome_sweep": "ops",
    "evidence_report": "ops",
    "measured_report": "ops",
    "sidecar_completion": "ops",
    "review_policy_draft": "ops",
    "ai_summary": "ops",
}
SETUP_GOALS = {"setup_quality", "permutations"}
SETUP_SLOTS = (
    "veto_cohort_grading", "like_cohort_grading", "pass_cohort_grading",
    "rejection_cohort_grading", "setup_research", "theta_pick_grading", "miss_contrast",
)


def test_every_slot_declares_a_goal_from_the_fixed_set():
    from ai_jobs import runner

    assert runner.SLOT_GOALS == (
        "trade_identification", "setup_quality", "permutations", "coaching",
        "market_read", "journal", "ops",
    )
    slots = runner.default_slots() + runner.optional_slots()
    for slot in slots:
        assert slot.goal in runner.SLOT_GOALS, slot.name
    goals = {slot.name: slot.goal for slot in slots}
    for name, goal in LEAD_GOAL_MAP.items():
        assert goals[name] == goal, name
    for name in SETUP_SLOTS:
        assert goals[name] in SETUP_GOALS, name


def test_a_slot_ledger_row_carries_its_goal(tmp_path, monkeypatch):
    from ai_jobs import runner, store, window

    monkeypatch.setattr(store, "store_available", lambda: (True, "ready"))
    monkeypatch.setattr(window, "launch_allowed", lambda *a, **k: (True, "window open"))
    monkeypatch.setattr(window, "market_session_block", lambda *a, **k: "")
    led = tmp_path / "ledger.jsonl"
    slots = [
        runner.JobSlot(name="a", run=lambda **k: {"status": "ok"}, goal="journal"),
        runner.JobSlot(name="b", run=lambda **k: {"status": "failed"}, goal="coaching"),
    ]
    runner.run_slots(slots, now=datetime(2026, 8, 12, 2, 0, tzinfo=ET), ledger_path=led)
    assert [(row["job"], row.get("goal")) for row in _rows(led)] == [
        ("a", "journal"), ("b", "coaching"),
    ]


#: 07:00 PT Thursday 2026-08-13 = 10:00 ET: the morning after Wednesday's night.
MORNING = datetime(2026, 8, 13, 10, 0, tzinfo=ET)
#: The session that night processed (the last closed session at 10:00 ET).
NIGHT_SESSION = "2026-08-12"


def _led(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "ledger.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def morning(monkeypatch):
    """Store up; the machine lock is free (patched, so parallel tests cannot hold it)."""
    import contextlib

    import local_writer_lock
    from ai_jobs import store

    monkeypatch.setattr(store, "store_available", lambda: (True, "ready"))
    monkeypatch.setattr(
        local_writer_lock, "local_writer_lock", lambda key, **k: contextlib.nullcontext()
    )


def _import_spy(calls: list, outcome: dict):
    def run(**kwargs):
        calls.append(kwargs)
        return outcome

    return run


def test_a_failed_night_import_is_retried_once_in_the_morning(tmp_path, morning):
    from ai_jobs import runner

    led = _led(tmp_path, [
        {"job": "journal_import", "status": "failed", "session_date": NIGHT_SESSION,
         "reason": "failed: IBKR Flex: Statement could not be generated"},
    ])
    calls: list = []
    outcome = {"status": "OK", "reason": "imported 4 execution(s)"}
    result = runner.retry_journal_import(
        now=MORNING, ledger_path=led, run=_import_spy(calls, outcome)
    )
    assert result["status"] == "ok"
    assert calls == [{"trigger": "morning_retry"}]
    row = _rows(led)[-1]
    assert row["job"] == "journal_import"
    assert row["session_date"] == NIGHT_SESSION
    assert row["status"] == "ok"
    assert row["reason"].startswith("morning retry")
    assert "imported 4 execution(s)" in row["reason"]
    assert row["morning_retry"] is True
    assert row["goal"] == "journal"

    # A second firing the same morning does nothing: one retry per session.
    again = runner.retry_journal_import(
        now=MORNING, ledger_path=led, run=_import_spy(calls, outcome)
    )
    assert again["status"] == "skipped"
    assert len(calls) == 1
    assert len(_rows(led)) == 2


def test_a_failed_retry_is_still_only_one_retry(tmp_path, morning):
    from ai_jobs import runner

    led = _led(tmp_path, [
        {"job": "journal_import", "status": "failed", "session_date": NIGHT_SESSION},
        {"job": "journal_import", "status": "skipped", "session_date": NIGHT_SESSION,
         "terminal": True, "reason": "3 attempts spent"},
    ])
    calls: list = []
    result = runner.retry_journal_import(
        now=MORNING, ledger_path=led,
        run=_import_spy(calls, {"status": "FAILED", "reason": "failed: Questrade: timeout"}),
    )
    assert result["status"] == "failed"
    row = _rows(led)[-1]
    assert row["status"] == "failed"
    assert row["reason"] == "morning retry: failed: Questrade: timeout"
    runner.retry_journal_import(
        now=MORNING, ledger_path=led, run=_import_spy(calls, {"status": "OK"})
    )
    assert len(calls) == 1


def test_a_crashing_retry_writes_a_failed_morning_row(tmp_path, morning):
    from ai_jobs import runner

    led = _led(tmp_path, [
        {"job": "journal_import", "status": "failed", "session_date": NIGHT_SESSION},
    ])

    def boom(**kwargs):
        raise RuntimeError("flex down")

    result = runner.retry_journal_import(now=MORNING, ledger_path=led, run=boom)
    assert result["status"] == "failed"
    row = _rows(led)[-1]
    assert row["status"] == "failed" and row["morning_retry"] is True
    assert row["reason"].startswith("morning retry")
    assert "flex down" in row["error"]


@pytest.mark.parametrize(
    "rows",
    [
        [],  # the night never ran the import
        [{"job": "journal_import", "status": "ok", "session_date": NIGHT_SESSION}],
        [{"job": "journal_import", "status": "failed", "session_date": NIGHT_SESSION},
         {"job": "journal_import", "status": "ok", "session_date": NIGHT_SESSION}],
        # an older session's failure is not this morning's business
        [{"job": "journal_import", "status": "failed", "session_date": "2026-08-11"}],
    ],
)
def test_no_retry_unless_the_night_import_failed(tmp_path, morning, rows):
    from ai_jobs import runner

    led = _led(tmp_path, rows)
    calls: list = []
    result = runner.retry_journal_import(
        now=MORNING, ledger_path=led, run=_import_spy(calls, {"status": "OK"})
    )
    assert result["status"] == "skipped"
    assert calls == []
    assert len(_rows(led)) == len(rows)


def test_the_retry_never_runs_while_the_night_run_holds_the_lock(tmp_path, monkeypatch):
    import local_writer_lock
    from ai_jobs import runner, store

    monkeypatch.setattr(store, "store_available", lambda: (True, "ready"))
    seen: list = []

    def busy(key, **kwargs):
        seen.append((key, kwargs))
        raise local_writer_lock.LocalLockUnavailable("someone else holds it")

    monkeypatch.setattr(local_writer_lock, "local_writer_lock", busy)
    led = _led(tmp_path, [
        {"job": "journal_import", "status": "failed", "session_date": NIGHT_SESSION},
    ])
    calls: list = []
    result = runner.retry_journal_import(
        now=MORNING, ledger_path=led, run=_import_spy(calls, {"status": "OK"})
    )
    assert result["status"] == "skipped"
    assert "in progress" in result["reason"]
    assert calls == []
    assert seen == [(runner.RUNNER_LOCK_KEY, {"timeout_seconds": 0.0})]
    assert len(_rows(led)) == 1


def test_the_cli_flag_runs_only_the_retry(monkeypatch):
    import run_ai_jobs
    from ai_jobs import runner

    called: list = []
    monkeypatch.setattr(
        runner, "retry_journal_import", lambda **k: called.append(k) or {"status": "failed"}
    )
    monkeypatch.setattr(runner, "run_slots", lambda *a, **k: pytest.fail("slate ran"))
    assert run_ai_jobs.main(["--retry-journal-import"]) == 1
    assert called == [{}]
    monkeypatch.setattr(runner, "retry_journal_import", lambda **k: {"status": "skipped"})
    assert run_ai_jobs.main(["--retry-journal-import"]) == 0


def test_the_morning_retry_task_is_registered_at_0700_with_the_flag():
    source = (SCRIPTS_DIR / "register_ai_jobs_task.ps1").read_text(encoding="utf-8")
    assert '"07:00"' in source
    assert "--retry-journal-import" in source
    assert "TradingBotV3 AI Jobs Morning Retry" in source


def test_the_ollama_probe_row_is_an_ops_row(tmp_path):
    from ai_jobs import ollama_probe

    led = tmp_path / "ledger.jsonl"
    ollama_probe.record_probe(True, "ok", session_date="2026-08-11", path=led)
    assert _rows(led)[0]["goal"] == "ops"
