"""WISHLIST P1-3: the night AI chain - Ollama probe, night budget, ticker
briefs, summary slice cap and the journal import retry."""

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
#: 02:00 ET on a Wednesday: inside the night window, processing Tuesday.
OVERNIGHT = datetime(2026, 8, 12, 2, 0, tzinfo=ET)


@pytest.fixture
def night(monkeypatch):
    """Store up, window open, no session block, no machine lock contention."""
    from ai_jobs import store, window

    monkeypatch.setattr(store, "store_available", lambda: (True, "ready"))
    monkeypatch.setattr(window, "launch_allowed", lambda *a, **k: (True, "window open"))
    monkeypatch.setattr(window, "market_session_block", lambda *a, **k: "")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _slot(name, fn, *, uses_model=False, model_free_kwargs=None, reserve=5.0, max_attempts=3):
    from ai_jobs.runner import JobSlot

    return JobSlot(
        name=name,
        run=fn,
        reserve_minutes=reserve,
        uses_model=uses_model,
        model_free_kwargs=model_free_kwargs,
        max_attempts=max_attempts,
    )


# ---------------------------------------------------------------------------
# 3e: the Ollama probe
# ---------------------------------------------------------------------------


def test_a_failed_probe_makes_the_night_deterministic_only(tmp_path, night):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    calls: list[str] = []
    probes: list[int] = []

    def det(**kwargs):
        calls.append("det")
        return {"reason": "facts"}

    def model(**kwargs):
        calls.append("model")
        return {"reason": "story"}

    def probe():
        probes.append(1)
        return False, "gemma3:12b did not answer within 30 s (ReadTimeout)"

    slots = [
        _slot("det_job", det),
        _slot("story_job", model, uses_model=True),
        _slot("other_story", model, uses_model=True),
    ]
    runner.run_slots(slots, now=OVERNIGHT, ledger_path=led, probe=probe)
    # a second firing the same night writes no second skip row per slot
    runner.run_slots(slots, now=OVERNIGHT, ledger_path=led, probe=probe)

    assert calls == ["det"], "no model slot may run when the probe failed"
    rows = _rows(led)
    probe_rows = [r for r in rows if r["job"] == "ollama_probe"]
    assert [r["status"] for r in probe_rows] == ["failed", "failed"]
    assert "deterministic work only" in probe_rows[0]["reason"]
    skips = [r for r in rows if r["job"] in ("story_job", "other_story")]
    assert [(r["job"], r["status"]) for r in skips] == [
        ("story_job", "skipped"),
        ("other_story", "skipped"),
    ]
    assert all("Ollama probe failed" in r["reason"] for r in skips)
    assert len(probes) == 2, "one probe per firing, not one per slot"


def test_a_failed_probe_runs_the_digest_facts_only_as_degraded(tmp_path, night):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    seen: list[dict] = []

    def digest(**kwargs):
        seen.append(kwargs)
        return {"reason": "fact pack written"}

    slot = _slot("daily_digest", digest, uses_model=True, model_free_kwargs={"narrate": False})
    runner.run_slots([slot], now=OVERNIGHT, ledger_path=led, probe=lambda: (False, "down"))

    assert seen and seen[0].get("narrate") is False
    row = [r for r in _rows(led) if r["job"] == "daily_digest"][0]
    assert row["status"] == "degraded_no_narrative"
    assert "Ollama probe failed" in row["reason"]


def test_a_live_model_is_probed_once_and_the_slots_run(tmp_path, night):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    probes: list[int] = []
    calls: list[str] = []

    def probe():
        probes.append(1)
        return True, "gemma3:12b answered one token in 2.1 s"

    slots = [
        _slot("det_job", lambda **k: calls.append("det") or {}),
        _slot("a", lambda **k: calls.append("a") or {}, uses_model=True),
        _slot("b", lambda **k: calls.append("b") or {}, uses_model=True),
    ]
    runner.run_slots(slots, now=OVERNIGHT, ledger_path=led, probe=probe)

    assert calls == ["det", "a", "b"]
    assert len(probes) == 1
    assert [r["status"] for r in _rows(led) if r["job"] == "ollama_probe"] == ["ok"]


def test_no_model_slot_due_means_no_probe(tmp_path, night):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    probes: list[int] = []
    runner.run_slots(
        [_slot("det_job", lambda **k: {})],
        now=OVERNIGHT,
        ledger_path=led,
        probe=lambda: probes.append(1) or (True, "ok"),
    )
    assert probes == []
    assert not [r for r in _rows(led) if r["job"] == "ollama_probe"]


def test_probe_local_model_answers_in_one_plain_sentence(monkeypatch):
    import ai_summary
    from ai_jobs import ollama_probe

    monkeypatch.setattr(ai_summary, "local_endpoint_url", lambda: "")
    ok, detail = ollama_probe.probe_local_model(post=lambda *a, **k: None)
    assert not ok and "no local model endpoint" in detail

    monkeypatch.setattr(ai_summary, "local_endpoint_url", lambda: "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(ai_summary, "local_model", lambda tier="medium": "gemma3:12b")
    seen: dict = {}

    def timeout_post(url, **kwargs):
        seen.update(kwargs, url=url)
        raise TimeoutError("read timed out")

    ok, detail = ollama_probe.probe_local_model(post=timeout_post)
    assert not ok and "did not answer within 30 s" in detail
    assert seen["timeout"] == 30.0
    assert seen["json"]["max_tokens"] == 1
    assert seen["url"].endswith("/chat/completions")

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "OK"}}]}

    ok, detail = ollama_probe.probe_local_model(post=lambda url, **k: _Response())
    assert ok and "answered one token" in detail


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_health_and_phone_digest_say_when_the_probe_failed(tmp_path):
    import autopilot_core
    import operations_audit

    led = _write_rows(
        tmp_path / "ai_job_ledger.jsonl",
        [
            {"job": "ollama_probe", "status": "ok", "started_at": "2026-09-22T22:00:05-07:00"},
            {
                "job": "ollama_probe",
                "status": "failed",
                "started_at": "2026-09-23T22:00:05-07:00",
                "error": "gemma3:12b did not answer within 30 s (ReadTimeout)",
            },
        ],
    )
    lines = operations_audit.ai_night_lines(led)
    assert lines[0] == (
        "Ollama: DOWN at 2026-09-23 22:00 (gemma3:12b did not answer within 30 s "
        "(ReadTimeout)) - the night ran deterministic work only"
    )
    now = datetime.fromisoformat("2026-09-24T06:30:00-07:00")
    line = operations_audit.ai_night_digest_line(led, now=now)
    assert line.startswith("Night AI: Ollama: DOWN")
    # a stale failure (two days old) is not news in the morning digest
    later = datetime.fromisoformat("2026-09-26T06:30:00-07:00")
    assert operations_audit.ai_night_digest_line(led, now=later) == ""
    report = autopilot_core.render_away_report({"ai_night_line": line})
    operations = report.split("== OPERATIONS ==", 1)[1]
    assert "Night AI: Ollama: DOWN" in operations


# ---------------------------------------------------------------------------
# 3a: the night budget
# ---------------------------------------------------------------------------


def _ran(job, minutes, day="2026-08-05"):
    return {
        "job": job,
        "status": "ok",
        "session_date": day,
        "started_at": f"{day}T23:00:00-04:00",
        "duration_seconds": minutes * 60.0,
    }


def test_the_budget_holds_room_for_higher_priority_slots_that_run_later(tmp_path, night):
    """`ticker_briefs` runs before `market_story_narration` and `setup_research`
    in the slate, but it is last in priority, so it is the one skipped."""
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    calls: list[str] = []

    def job(name):
        return lambda **k: calls.append(name) or {}

    slots = [
        _slot("journal_import", job("journal_import")),
        _slot("ticker_briefs", job("ticker_briefs"), uses_model=True, reserve=120.0),
        _slot("market_story_narration", job("market_story_narration"), uses_model=True, reserve=15.0),
        _slot("setup_research", job("setup_research"), uses_model=True, reserve=20.0),
    ]
    runner.run_slots(slots, now=OVERNIGHT, ledger_path=led, budget_minutes=150.0)

    assert calls == ["journal_import", "market_story_narration", "setup_research"]
    skip = [r for r in _rows(led) if r["job"] == "ticker_briefs"][0]
    assert skip["status"] == "skipped" and skip["night_budget"] is True
    assert skip["reason"].startswith("night budget: 150 of 150 min left")
    assert "35 min held for market_story_narration, setup_research" in skip["reason"]


def test_a_spent_night_skips_model_slots_but_not_deterministic_work(tmp_path, night):
    from ai_jobs import runner

    led = _write_rows(
        tmp_path / "ledger.jsonl",
        # this night's first row: 22:00 ET, four hours before OVERNIGHT
        [{"job": "journal_import", "status": "ok", "session_date": "2026-08-11",
          "started_at": "2026-08-11T22:00:00-04:00", "duration_seconds": 40.0}],
    )
    calls: list[str] = []
    slots = [
        _slot("evidence_report", lambda **k: calls.append("det") or {}),
        _slot("day_review_narration", lambda **k: calls.append("story") or {},
              uses_model=True, reserve=10.0),
    ]
    runner.run_slots(slots, now=OVERNIGHT, ledger_path=led, budget_minutes=150.0)
    runner.run_slots(slots, now=OVERNIGHT, ledger_path=led, budget_minutes=150.0)

    assert calls == ["det"], "deterministic work runs; the model slot never does"
    skips = [r for r in _rows(led) if r["job"] == "day_review_narration"]
    assert len(skips) == 1, "one budget row per slot per session, not one per firing"
    assert skips[0]["reason"].startswith("night budget: 0 of 150 min left")


def test_the_estimate_comes_from_measured_runs_not_the_reserve(tmp_path, night):
    from ai_jobs import runner

    led = _write_rows(
        tmp_path / "ledger.jsonl",
        [_ran("ticker_briefs", 4.0), _ran("ticker_briefs", 6.0), _ran("ticker_briefs", 5.0)],
    )
    calls: list[str] = []
    slot = _slot("ticker_briefs", lambda **k: calls.append("briefs") or {},
                 uses_model=True, reserve=120.0)
    runner.run_slots([slot], now=OVERNIGHT, ledger_path=led, budget_minutes=30.0)
    assert calls == ["briefs"], "a 5-minute measured job fits a 30-minute budget"


def test_measured_slot_minutes_is_the_median_of_recent_real_runs(tmp_path):
    from ai_jobs import model_probe

    rows = [
        _ran("a", 100.0), _ran("a", 1.0), _ran("a", 2.0), _ran("a", 3.0),
        _ran("a", 4.0), _ran("a", 5.0),
        {"job": "a", "status": "skipped", "duration_seconds": 9999.0},
        _ran("b", 7.0),
    ]
    measured = model_probe.measured_slot_minutes(rows=rows, sample=5)
    assert measured == {"a": 3.0, "b": 7.0}


def test_the_budget_is_a_weeknight_rule_and_a_setting(monkeypatch):
    from ai_jobs import runner, store

    class _Paths:
        settings: dict = {}

        @classmethod
        def get_local_setting(cls, key, default=None):
            return cls.settings.get(key, default)

    monkeypatch.setattr(store, "_paths", lambda: _Paths)
    assert runner.night_budget_for("weeknight") == 150.0
    assert runner.night_budget_for("saturday") == 0.0
    assert runner.night_budget_for("sunday") == 0.0
    _Paths.settings = {"ai_night_budget_minutes": 90}
    assert runner.night_budget_for("weeknight") == 90.0
    _Paths.settings = {"ai_night_budget_minutes": 0}
    assert runner.night_budget_for("weeknight") == 0.0


def test_priority_order_is_the_trader_list_with_ticker_briefs_last():
    from ai_jobs import runner

    order = sorted(
        [
            "ticker_briefs", "observation_tags", "econ_brief", "journal_enrichment",
            "setup_research", "market_story_narration", "day_review_narration", "daily_digest",
        ],
        key=runner.model_slot_priority,
    )
    assert order == [
        "daily_digest", "day_review_narration", "market_story_narration", "setup_research",
        "journal_enrichment", "observation_tags", "econ_brief", "ticker_briefs",
    ]


def test_health_names_the_slots_the_budget_skipped(tmp_path):
    import operations_audit

    led = _write_rows(
        tmp_path / "ai_job_ledger.jsonl",
        [
            {"job": "ticker_briefs", "status": "skipped", "session_date": "2026-09-22",
             "night_budget": True, "reason": "night budget: ...", "started_at": "2026-09-23T01:00:00-07:00"},
            {"job": "setup_research", "status": "skipped", "session_date": "2026-09-23",
             "night_budget": True, "reason": "night budget: ...", "started_at": "2026-09-24T01:00:00-07:00"},
            {"job": "observation_tags", "status": "skipped", "session_date": "2026-09-23",
             "night_budget": True, "reason": "night budget: ...", "started_at": "2026-09-24T01:30:00-07:00"},
        ],
    )
    assert "night budget 2026-09-23: skipped setup_research, observation_tags" in (
        operations_audit.ai_night_lines(led)
    )


def test_a_healthy_probe_is_not_in_the_phone_digest(tmp_path):
    import operations_audit

    led = _write_rows(
        tmp_path / "ai_job_ledger.jsonl",
        [{"job": "ollama_probe", "status": "ok", "started_at": "2026-09-23T22:00:05-07:00"}],
    )
    assert operations_audit.ai_night_lines(led) == ["Ollama: ok at 2026-09-23 22:00"]
    assert operations_audit.ai_night_digest_line(led) == ""
