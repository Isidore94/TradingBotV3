"""P8b B1: the night is measurable - tokens per slot, goal coverage, Health rows."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")
NIGHT = datetime(2026, 8, 12, 2, 0, tzinfo=ET)
SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
    "additionalProperties": False,
}


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _Reply:
    status_code = 200
    text = ""

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _fake_post(usages: list[dict]):
    """A local chat endpoint answering valid JSON with the given usage blocks in turn."""
    queue = list(usages)

    def post(url, **kwargs):
        usage = queue.pop(0)
        body = {
            "id": "r1",
            "choices": [{"message": {"content": json.dumps({"a": "x"})}, "finish_reason": "stop"}],
        }
        if usage:
            body["usage"] = usage
        return _Reply(body)

    return post


def _ask(post):
    import ai_summary

    return ai_summary.request_ai_summary(
        provider="local",
        model="m",
        api_key="",
        evidence={"package_id": "p", "evidence_hash": "h", "sources": []},
        post=post,
        schema=SCHEMA,
        schema_name="t",
    )


def _night(monkeypatch):
    import ai_summary
    from ai_jobs import store, window

    monkeypatch.setattr(store, "store_available", lambda: (True, "ready"))
    monkeypatch.setattr(window, "launch_allowed", lambda *a, **k: (True, "window open"))
    monkeypatch.setattr(window, "market_session_block", lambda *a, **k: "")
    monkeypatch.setattr(ai_summary, "local_endpoint_url", lambda: "http://127.0.0.1:11434/v1")


# ---------------------------------------------------------------------------
# 1. Tokens per slot
# ---------------------------------------------------------------------------


def test_every_model_calling_slot_row_carries_its_summed_tokens(tmp_path, monkeypatch):
    from ai_jobs import runner

    _night(monkeypatch)
    post = _fake_post([
        {"prompt_tokens": 100, "completion_tokens": 10},
        {"prompt_tokens": 50, "completion_tokens": 5},
        {"prompt_tokens": 7, "completion_tokens": 1},
    ])

    def two_calls(**kwargs):
        _ask(post)
        _ask(post)
        return {"status": "ok"}

    def one_call_then_crash(**kwargs):
        _ask(post)
        raise RuntimeError("boom")

    led = tmp_path / "ledger.jsonl"
    slots = [
        runner.JobSlot(name="a", run=two_calls, goal="coaching"),
        runner.JobSlot(name="b", run=lambda **k: {"status": "ok"}, goal="ops"),
        runner.JobSlot(name="c", run=one_call_then_crash, goal="journal"),
    ]
    runner.run_slots(slots, now=NIGHT, ledger_path=led)
    tokens = {row["job"]: row["tokens"] for row in _rows(led)}
    assert tokens["a"] == {"prompt_tokens": 150, "completion_tokens": 15, "calls": 2}
    assert tokens["b"] == {}
    assert tokens["c"] == {"prompt_tokens": 7, "completion_tokens": 1, "calls": 1}


def test_a_slots_own_richer_tokens_dict_is_merged_not_clobbered(tmp_path, monkeypatch):
    from ai_jobs import runner

    _night(monkeypatch)
    post = _fake_post([
        {"prompt_tokens": 40, "completion_tokens": 4},
        {},
    ])

    def briefs(**kwargs):
        _ask(post)
        _ask(post)
        return {"status": "ok", "tokens": {"ticker_calls": 2, "prompt_tokens": 40}}

    led = tmp_path / "ledger.jsonl"
    runner.run_slots(
        [runner.JobSlot(name="ticker_briefs", run=briefs, goal="market_read")],
        now=NIGHT,
        ledger_path=led,
    )
    assert _rows(led)[0]["tokens"] == {
        "ticker_calls": 2,
        "prompt_tokens": 40,
        "completion_tokens": 4,
        "calls": 2,
    }


# ---------------------------------------------------------------------------
# 2. Goal coverage and tokens in the digest facts (never shown to the narrator)
# ---------------------------------------------------------------------------

DIGEST_DAY = "2026-08-24"
DIGEST_NOW = datetime(2026, 8, 25, 2, 0, tzinfo=ET)


def _ledger_rows() -> list[dict]:
    day = DIGEST_DAY
    return [
        {"job": "journal_import", "goal": "journal", "status": "failed", "session_date": day},
        {"job": "journal_import", "goal": "journal", "status": "ok", "session_date": day},
        {"job": "day_review_narration", "goal": "coaching", "status": "degraded_no_narrative",
         "session_date": day,
         "tokens": {"prompt_tokens": 900, "completion_tokens": 90, "calls": 2}},
        {"job": "econ_brief", "goal": "market_read", "status": "failed", "session_date": day,
         "tokens": {"prompt_tokens": 300, "completion_tokens": 30, "calls": 1}},
        {"job": "econ_brief", "goal": "market_read", "status": "failed", "session_date": day,
         "tokens": {"prompt_tokens": 300, "completion_tokens": 30, "calls": 1}},
        {"job": "econ_brief", "goal": "market_read", "status": "skipped", "session_date": day,
         "terminal": True, "tokens": {}},
        {"job": "ai_summary", "goal": "coaching", "status": "ok", "session_date": day,
         "tokens": {"duration_seconds": 12.5, "prompt_tokens": 5000, "completion_tokens": 400}},
        {"job": "ticker_briefs", "goal": "market_read", "status": "skipped",
         "session_date": day, "tokens": {}},
        {"job": "setup_research", "goal": "setup_quality", "status": "ok",
         "session_date": day, "tokens": {"prompt_tokens": 100, "completion_tokens": 10,
                                          "calls": 1}},
        # Another night: never counted.
        {"job": "ai_summary", "goal": "coaching", "status": "failed", "session_date": "2026-08-21",
         "tokens": {"prompt_tokens": 99999, "completion_tokens": 1, "calls": 1}},
    ]


#: The digest's own session: its night is under way, so the lines name DIGEST_DAY.
NEXT_DAY = "2026-08-25"


def test_the_telemetry_lines_are_deterministic_facts_from_the_last_complete_night():
    from ai_jobs import digest

    goal_line, token_line = digest.night_telemetry_lines(_ledger_rows(), NEXT_DAY)[:2]
    assert goal_line == (
        "slots per goal (night of 2026-08-24): "
        "trade_identification: ok 0 / degraded 0 / failed 0 / skipped 0; "
        "setup_quality: ok 1 / degraded 0 / failed 0 / skipped 0; "
        "permutations: ok 0 / degraded 0 / failed 0 / skipped 0; "
        "coaching: ok 1 / degraded 1 / failed 0 / skipped 0; "
        "market_read: ok 0 / degraded 0 / failed 1 / skipped 1; "
        "journal: ok 1 / degraded 0 / failed 0 / skipped 0; "
        "ops: ok 0 / degraded 0 / failed 0 / skipped 0"
    )
    assert token_line == (
        "tokens (night of 2026-08-24): 6600/560 over 5 calls; top 3 slots by prompt tokens: "
        "ai_summary 5000, day_review_narration 900, econ_brief 600"
    )


def _at(stamp: str) -> str:
    return f"2026-09-{stamp}:00-07:00"


def test_a_half_run_night_reports_the_previous_complete_night():
    from ai_jobs import digest

    rows = [
        # 2026-09-22: complete, but older than the newest complete night.
        {"job": "journal_import", "goal": "journal", "status": "failed",
         "session_date": "2026-09-22", "started_at": _at("22T22:00")},
        # 2026-09-23: complete - it reached stage 3.
        {"job": "journal_import", "goal": "journal", "status": "ok",
         "session_date": "2026-09-23", "started_at": _at("23T22:00"),
         "tokens": {}},
        {"job": "day_review_narration", "goal": "coaching", "status": "ok",
         "session_date": "2026-09-23", "started_at": _at("23T22:10"),
         "tokens": {"prompt_tokens": 800, "completion_tokens": 80, "calls": 1}},
        {"job": "improvement_ideas", "goal": "setup_quality", "status": "ok",
         "session_date": "2026-09-23", "started_at": _at("24T01:00"),
         "tokens": {"prompt_tokens": 200, "completion_tokens": 20, "calls": 1}},
        # 2026-09-24: tonight, half-run when the digest writes.
        {"job": "journal_import", "goal": "journal", "status": "failed",
         "session_date": "2026-09-24", "started_at": _at("24T22:00")},
        {"job": "day_review_narration", "goal": "coaching", "status": "failed",
         "session_date": "2026-09-24", "started_at": _at("24T22:05"),
         "tokens": {"prompt_tokens": 5, "completion_tokens": 0, "calls": 1}},
        # A redo of the 23rd, run tonight: the 23rd still counts (stage-3 row).
        {"job": "day_review_narration", "goal": "coaching", "status": "ok",
         "session_date": "2026-09-23", "started_at": _at("24T22:07"),
         "tokens": {"prompt_tokens": 100, "completion_tokens": 10, "calls": 1}},
    ]
    now = datetime(2026, 9, 25, 1, 10, tzinfo=ET)
    goal_line, token_line = digest.night_telemetry_lines(rows, "2026-09-24", now=now)
    assert goal_line.startswith(
        "slots per goal (night of 2026-09-23): trade_identification: ok 0 / degraded 0 / "
        "failed 0 / skipped 0; setup_quality: ok 1 / degraded 0 / failed 0 / skipped 0; "
    )
    assert "coaching: ok 1 / degraded 0 / failed 0 / skipped 0" in goal_line
    assert "journal: ok 1 / degraded 0 / failed 0 / skipped 0" in goal_line
    assert token_line == (
        "tokens (night of 2026-09-23): 1100/110 over 3 calls; top 3 slots by prompt tokens: "
        "day_review_narration 900, improvement_ideas 200"
    )


def test_a_previous_night_without_stage_three_is_complete_only_if_it_ended_before_tonight():
    from ai_jobs import digest

    rows = [
        {"job": "journal_import", "goal": "journal", "status": "ok",
         "session_date": "2026-09-22", "started_at": _at("22T22:00")},
        # The 23rd has a row that started after tonight began and no stage-3 row.
        {"job": "journal_import", "goal": "journal", "status": "ok",
         "session_date": "2026-09-23", "started_at": _at("24T22:30")},
        {"job": "journal_import", "goal": "journal", "status": "failed",
         "session_date": "2026-09-24", "started_at": _at("24T22:00")},
    ]
    lines = digest.night_telemetry_lines(rows, "2026-09-24")
    assert lines[0].startswith("slots per goal (night of 2026-09-22): ")
    assert digest.night_telemetry_lines([], "2026-09-24") == [
        "slots per goal: unknown (no complete night in the ledger yet)",
        "tokens: unknown (no complete night in the ledger yet)",
    ]


def test_the_digest_file_carries_the_lines_and_the_narrator_never_sees_them(
    tmp_path, monkeypatch
):
    from ai_jobs import digest

    monkeypatch.setattr(digest, "_read_job_rows", _ledger_rows)
    handed: list[dict] = []

    def narrator(*, pack, now=None):
        handed.append(digest.narration_evidence_package(pack))
        return {"model": "m", "narration": {}}

    monkeypatch.setattr(digest, "_narrate", narrator)
    digest.run_daily_digest(
        session_date=NEXT_DAY, now=DIGEST_NOW, root=tmp_path, is_session=False,
    )
    written = json.loads(digest.facts_path(tmp_path, NEXT_DAY).read_text(encoding="utf-8"))
    lines = written[digest.NIGHT_TELEMETRY_KEY]["lines"]
    assert lines[0].startswith("slots per goal (night of 2026-08-24): trade_identification: ok 0")
    assert lines[1].startswith("tokens (night of 2026-08-24): 6600/560 over 5 calls")
    assert handed, "the narrator was not asked"
    seen = json.dumps(handed[0])
    assert "slots per goal" not in seen and "tokens (night of" not in seen
    assert digest.NIGHT_TELEMETRY_KEY not in seen


# ---------------------------------------------------------------------------
# 3. Health rows: Night AI timing and Broker import
# ---------------------------------------------------------------------------

FLEX = "failed: IBKR Flex: IBKR Flex request failed: Statement could not be generated at this time."
MAX_RETRIES = (
    "failed: IBKR Flex: HTTPSConnectionPool(host='gdcdyn.interactivebrokers.com', "
    "port=443): Max retries exceeded"
)


def _fixture_ledger(tmp_path: Path) -> Path:
    rows = [
        {"job": "ollama_probe", "status": "ok", "session_date": "2026-09-23",
         "started_at": "2026-09-23T22:06:10-07:00",
         "reason": "gemma3:12b-tbv3ctx-64k answered one token in 21.4 s"},
        {"job": "day_review_narration", "status": "degraded_no_narrative",
         "session_date": "2026-09-24", "started_at": "2026-09-24T22:21:42-07:00",
         "duration_seconds": 729.056},
        {"job": "ollama_probe", "status": "ok", "session_date": "2026-09-24",
         "started_at": "2026-09-24T23:30:19-07:00",
         "reason": "gemma3:12b-tbv3ctx-64k answered one token in 16.3 s"},
        {"job": "day_review_narration", "status": "ok", "session_date": "2026-09-24",
         "started_at": "2026-09-24T23:30:19-07:00", "duration_seconds": 400.576},
        # A later window skip has no run time and does not replace the run.
        {"job": "day_review_narration", "status": "skipped", "session_date": "2026-09-24",
         "duration_seconds": 0.0},
    ]
    # Sixteen import nights: the two oldest fall out of the 14-night window.
    for day in range(1, 17):
        session = f"2026-09-{day:02d}"
        if day in (1, 2, 5, 9):
            rows.append({"job": "journal_import", "status": "failed",
                         "session_date": session, "reason": FLEX})
            rows.append({"job": "journal_import", "status": "failed",
                         "session_date": session, "reason": MAX_RETRIES})
            rows.append({"job": "journal_import", "status": "skipped", "terminal": True,
                         "session_date": session, "reason": "3 attempt(s) already made"})
        elif day == 12:
            rows.append({"job": "journal_import", "status": "failed",
                         "session_date": session, "reason": FLEX})
            # The 07:00 morning retry keeps the previous session_date and saves it.
            rows.append({"job": "journal_import", "status": "ok", "session_date": session,
                         "morning_retry": True, "reason": "morning retry: imported 3"})
        else:
            rows.append({"job": "journal_import", "status": "ok",
                         "session_date": session, "reason": "imported 400"})
    path = tmp_path / "ai_job_ledger.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_health_night_ai_and_broker_import_lines_from_a_fixture_ledger(tmp_path):
    import operations_audit

    lines = operations_audit.ai_telemetry_lines(_fixture_ledger(tmp_path))
    assert lines == [
        "Night AI: first token 16.3 s (probe 2026-09-24 23:30); day story 6:41 (ok)",
        "Broker import: ok 12 of last 14 nights; last failure 2026-09-09: "
        + MAX_RETRIES[:57] + "...",
    ]


def test_health_night_ai_line_says_failed_probe_and_degraded_story(tmp_path):
    import operations_audit

    rows = [
        {"job": "day_review_narration", "status": "degraded_no_narrative",
         "duration_seconds": 729.056},
        {"job": "ollama_probe", "status": "failed", "started_at": "2026-09-24T22:00:05-07:00",
         "error": "did not answer within 30 s"},
    ]
    assert operations_audit.night_ai_timing_line(rows) == (
        "Night AI: first token none, probe failed (probe 2026-09-24 22:00); "
        "day story 12:09 (degraded)"
    )
    assert operations_audit.ai_telemetry_lines(tmp_path / "missing.jsonl") == []


def test_the_health_worker_payload_carries_the_telemetry_lines(tmp_path, monkeypatch):
    import operations_audit
    from ui.panels import health_panel

    store = tmp_path / "store"
    (store / "logs").mkdir(parents=True)
    _fixture_ledger(tmp_path).replace(store / "logs" / operations_audit.AI_JOB_LEDGER_NAME)
    monkeypatch.setenv(operations_audit.AI_STORE_DIR_ENV, str(store))
    lines = health_panel._with_ai_night_lines({})["ai_night_lines"]
    assert any(line.startswith("Night AI: first token 16.3 s") for line in lines)
    assert any(line.startswith("Broker import: ok 12 of last 14 nights") for line in lines)


# ---------------------------------------------------------------------------
# 4. Unread-output stamp
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path, monkeypatch):
    import project_paths
    import slot_output_reads

    path = tmp_path / "slot_output_reads.json"
    monkeypatch.setattr(project_paths, "SLOT_OUTPUT_READS_FILE", path)
    monkeypatch.setattr(slot_output_reads, "_NOTED_TODAY", {})
    return path


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_registry_stamps_reads_and_counts_days(registry):
    import slot_output_reads as reads

    assert reads.days_since_read("econ_brief") is None
    assert reads.unread_line() == "unread 14+ days: unknown (no reads recorded yet)"
    reads.note_slot_output_read("econ_brief", today=date(2026, 9, 1), path=registry)
    reads.note_slot_output_read("week_questions", today=date(2026, 9, 20), path=registry)
    assert reads.days_since_read("econ_brief", today=date(2026, 9, 25)) == 24
    assert reads.days_since_read("week_questions", today=date(2026, 9, 25)) == 5
    # Never read since stamping began 24 days ago: unread too.
    assert reads.unread_line(today=date(2026, 9, 25)) == (
        "unread 14+ days: day_review_narration, econ_brief, improvement_ideas"
    )
    for slot in reads.TRACKED_SLOTS:
        reads.note_slot_output_read(slot, today=date(2026, 9, 25), path=registry)
    assert reads.unread_line(today=date(2026, 9, 25)) == "unread 14+ days: none"


def test_the_digest_facts_carry_the_unread_line(tmp_path, monkeypatch, registry):
    from ai_jobs import digest

    monkeypatch.setattr(digest, "_read_job_rows", _ledger_rows)
    digest.run_daily_digest(
        session_date=DIGEST_DAY, now=DIGEST_NOW, root=tmp_path, is_session=False, narrate=False,
    )
    written = json.loads(digest.facts_path(tmp_path, DIGEST_DAY).read_text(encoding="utf-8"))
    assert written[digest.NIGHT_TELEMETRY_KEY]["lines"][2] == (
        "unread 14+ days: unknown (no reads recorded yet)"
    )


def test_the_panels_stamp_a_read_when_they_render_a_fresh_output(qapp, registry):
    import slot_output_reads as reads
    from ui.panels.day_review_panel import DayReviewPanel
    from ui.widgets.econ_brief_block import EconBriefBlock
    from ui.widgets.ideas_card import IdeasCard
    from ui.widgets.week_coach_card import WeekCoachCard

    # Day story: the night's narration for the session on screen.
    fake = SimpleNamespace(
        story_body=MagicMock(), story_note=MagicMock(), story_warning=MagicMock(),
        _story_attempt_state=lambda card: "", _verdict_text=lambda verdict: verdict,
    )
    DayReviewPanel._render_day_story(
        fake, {"session_date": "2026-09-24", "narration": {"headline": "A day"}},
        "2026-09-24", {},
    )
    assert reads.days_since_read("day_review_narration") == 0

    block = EconBriefBlock()
    try:
        block.set_view({"session": "2026-09-25", "origin": "last_brief", "summary_lines": []})
        assert reads.days_since_read("econ_brief") is None  # not the night's words
        block.set_view({"session": "2026-09-25", "origin": "night", "summary_lines": ["x"]})
        assert reads.days_since_read("econ_brief") == 0
    finally:
        block.deleteLater()

    ideas = IdeasCard(writer=lambda *_a: None)
    try:
        ideas.show_ideas([])
        assert reads.days_since_read("improvement_ideas") is None
        ideas.show_ideas([{"idea_id": "a", "kind": "process", "text": "First", "status": ""}])
        assert reads.days_since_read("improvement_ideas") == 0
    finally:
        ideas.deleteLater()

    card = WeekCoachCard(read=lambda *_a, **_k: {})
    try:
        card.render({"questions": [{"question": "why?", "status": "pending"}]})
        assert reads.days_since_read("week_questions") is None
        card.render({"questions": [{"question": "why?", "status": "answered",
                                    "answer": {"claims": [{"text": "because"}]}}]})
        assert reads.days_since_read("week_questions") == 0
    finally:
        card.deleteLater()
