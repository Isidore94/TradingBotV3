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


def test_a_healthy_probe_is_not_in_the_phone_digest(tmp_path):
    import operations_audit

    led = _write_rows(
        tmp_path / "ai_job_ledger.jsonl",
        [{"job": "ollama_probe", "status": "ok", "started_at": "2026-09-23T22:00:05-07:00"}],
    )
    assert operations_audit.ai_night_lines(led) == ["Ollama: ok at 2026-09-23 22:00"]
    assert operations_audit.ai_night_digest_line(led) == ""
