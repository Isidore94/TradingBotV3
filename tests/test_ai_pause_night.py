"""Pause AI and the night runner: model slots record SKIPPED with `ai_paused`, the
deterministic halves still run (once), no probe and no model call is made."""

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

import ai_pause  # noqa: E402
import project_paths  # noqa: E402

ET = ZoneInfo("America/New_York")
#: 02:00 ET on a Wednesday: inside the night window, processing Tuesday.
OVERNIGHT = datetime(2026, 8, 12, 2, 0, tzinfo=ET)


@pytest.fixture
def night(tmp_path, monkeypatch):
    """Store up, window open, no session block; a scratch settings file."""
    from ai_jobs import store, window

    monkeypatch.setattr(store, "store_available", lambda: (True, "ready"))
    monkeypatch.setattr(window, "launch_allowed", lambda *a, **k: (True, "window open"))
    monkeypatch.setattr(window, "market_session_block", lambda *a, **k: "")
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_FILE", tmp_path / "local_settings.json")
    project_paths.invalidate_local_settings_cache()
    yield
    project_paths.invalidate_local_settings_cache()


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _slot(name, fn, *, uses_model=False, model_free_kwargs=None):
    from ai_jobs.runner import JobSlot

    return JobSlot(name=name, run=fn, reserve_minutes=5.0, uses_model=uses_model,
                   model_free_kwargs=model_free_kwargs, max_attempts=3)


def _slots(calls):
    def det(**kwargs):
        calls.append("det")
        return {"reason": "facts"}

    def story(**kwargs):
        calls.append("story")
        return {"reason": "story"}

    def digest(**kwargs):
        calls.append(("digest", kwargs.get("narrate", True)))
        return {"reason": "fact pack written"}

    return [
        _slot("det_job", det),
        _slot("story_job", story, uses_model=True),
        _slot("daily_digest", digest, uses_model=True, model_free_kwargs={"narrate": False}),
    ]


def test_paused_model_slots_skip_with_ai_paused_and_facts_still_run(tmp_path, night):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    calls: list = []
    probes: list = []
    ai_pause.pause_for("until_resumed", OVERNIGHT)

    slots = _slots(calls)
    runner.run_slots(slots, now=OVERNIGHT, ledger_path=led, probe=lambda: probes.append(1) or (True, "ok"))
    # A second firing the same night: no second skip row, no second facts run.
    runner.run_slots(slots, now=OVERNIGHT, ledger_path=led, probe=lambda: probes.append(1) or (True, "ok"))

    assert calls == ["det", ("digest", False)], "no model call while paused; the facts half runs once"
    assert probes == [], "no Ollama probe while paused"
    rows = {r["job"]: r for r in _rows(led)}
    assert len([r for r in _rows(led) if r["job"] == "story_job"]) == 1
    story = rows["story_job"]
    assert story["status"] == "skipped"
    assert story.get(runner.AI_PAUSED_FLAG) is True
    assert story["reason"].startswith("AI paused until you resume")
    digest = rows["daily_digest"]
    assert digest["status"] == "degraded_no_narrative"
    assert digest.get(runner.AI_PAUSED_FLAG) is True
    assert "deterministic facts only" in digest["reason"]
    assert rows["det_job"]["status"] == "ok"


def test_after_the_pause_the_model_slots_run_again(tmp_path, night):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    calls: list = []
    ai_pause.pause_for("until_resumed", OVERNIGHT)
    runner.run_slots(_slots(calls), now=OVERNIGHT, ledger_path=led, probe=lambda: (True, "ok"))
    ai_pause.resume()
    calls.clear()
    runner.run_slots(_slots(calls), now=OVERNIGHT, ledger_path=led, probe=lambda: (True, "ok"))

    assert "story" in calls and ("digest", True) in calls


def test_an_expired_pause_changes_nothing(tmp_path, night):
    from ai_jobs import runner

    led = tmp_path / "ledger.jsonl"
    calls: list = []
    ai_pause.pause_for("2h", datetime(2026, 8, 11, 12, 0, tzinfo=ET))
    runner.run_slots(_slots(calls), now=OVERNIGHT, ledger_path=led, probe=lambda: (True, "ok"))

    assert calls == ["det", "story", ("digest", True)]
    assert not any(r.get(runner.AI_PAUSED_FLAG) for r in _rows(led))


def test_the_model_probe_cli_refuses_while_paused(night, capsys):
    import run_ai_jobs

    ai_pause.pause_for("4h")
    assert run_ai_jobs.main(["--probe-model", "large"]) == 1
    assert "AI paused" in capsys.readouterr().out
