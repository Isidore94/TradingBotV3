from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _packs(root: Path) -> None:
    weekly = {
        "kind": "weekly",
        "period_id": "2026-W37",
        "inputs_hash": "week-hash",
        "sessions_covered": ["2026-09-14"],
        "sessions_missing": [],
        "sessions": [
            {
                "session_date": "2026-09-14",
                "entries": [{"entry_id": "note-1", "text": "SPY holds 5400"}],
                "measured": [{"symbol": "SPY", "status": "measured", "change_pct": 1.0}],
            }
        ],
    }
    monthly = {"kind": "monthly", "period_id": "2026-09", "inputs_hash": "month-hash"}
    quarterly = {"kind": "quarterly", "period_id": "2026-Q3", "inputs_hash": "q-hash"}
    for kind, payload in (("weekly", weekly), ("monthly", monthly), ("quarterly", quarterly)):
        path = root / kind / f"{payload['period_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_grounded_story_narration_writes_only_after_valid_sources(tmp_path):
    from ai_jobs.market_story_narration import run_market_story_narration

    rollups = tmp_path / "rollups"
    output = tmp_path / "narration"
    _packs(rollups)

    def request(**kwargs):
        assert kwargs["provider"] == "local"
        assert kwargs["evidence"]["allowed_source_ids"] == [
            "journal:note-1",
            "rollup:monthly:2026-09",
            "rollup:quarterly:2026-Q3",
            "rollup:weekly:2026-W37",
        ]
        return {
            "model": "local-test",
            "summary": {
                "summary": "The trader expected SPY to hold 5400.",
                "changes": ["SPY strengthened."],
                "open_questions": ["Will 5400 hold?"],
                "mentor_question": "What would make the 5400 thesis wrong today?",
                "sources": ["journal:note-1", "rollup:weekly:2026-W37"],
            },
        }

    result = run_market_story_narration(
        session_date="2026-09-14",
        now=datetime(2026, 9, 15, tzinfo=timezone.utc),
        rollups_dir=rollups,
        out_dir=output,
        request=request,
    )
    assert result["status"] == "ok"
    saved = json.loads((output / "2026-09-14.json").read_text(encoding="utf-8"))
    assert saved["narration"]["mentor_question"].startswith("What would")
    assert saved["periods"]["weekly"] == "2026-W37"


def test_bad_story_citation_keeps_the_prior_verified_file(tmp_path):
    from ai_jobs.market_story_narration import run_market_story_narration

    rollups = tmp_path / "rollups"
    output = tmp_path / "narration"
    _packs(rollups)
    target = output / "2026-09-14.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"verified":"old"}\n', encoding="utf-8")
    before = target.read_bytes()

    def request(**_kwargs):
        return {
            "model": "local-test",
            "summary": {
                "summary": "Invented.",
                "changes": [],
                "open_questions": [],
                "mentor_question": "",
                "sources": ["not-in-the-pack"],
            },
        }

    result = run_market_story_narration(
        session_date="2026-09-14",
        rollups_dir=rollups,
        out_dir=output,
        request=request,
    )
    assert result["status"] == "degraded_no_narrative"
    assert target.read_bytes() == before
