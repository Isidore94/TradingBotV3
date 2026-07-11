"""Plan.md 23.8: verified atomic away-report publication."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autopilot_core as core  # noqa: E402


def _payload(**overrides):
    payload = {
        "generated_at": "2026-07-10 13:00:00",
        "enabled": True,
        "ib_status": "connected",
        "regime": "bullish_weak",
        "longs": ["NVDA"],
        "shorts": ["WOLF"],
        "swing_picks": [{"symbol": "AAPL", "side": "LONG", "bucket": "A", "expected_r": 0.4}],
        "alerts": [],
        "slots_done": ["09:00"],
        "next_slot": "10:00",
        "log_lines": [],
        "auto_longs": [],
        "auto_shorts": [],
    }
    payload.update(overrides)
    return payload


def test_publish_is_verified_and_archived(tmp_path):
    target = tmp_path / "autopilot_today.txt"
    result = core.publish_away_report(_payload(), target)
    assert result["ok"] is True and result["verified"] is True
    text = target.read_text(encoding="utf-8")
    assert "TRADINGBOT AUTO PILOT - TODAY" in text
    assert "Freshness: next planned update 10:00" in text
    archive = list((tmp_path / "away_report_archive").glob("autopilot_today_*.txt"))
    assert len(archive) == 1


def test_render_failure_never_clears_previous_valid_report(tmp_path, monkeypatch):
    target = tmp_path / "autopilot_today.txt"
    assert core.publish_away_report(_payload(), target)["ok"]
    before = target.read_text(encoding="utf-8")

    def broken_render(payload):
        raise ValueError("renderer exploded")

    monkeypatch.setattr(core, "render_away_report", broken_render)
    result = core.publish_away_report(_payload(), target)
    assert result["ok"] is False
    assert "render failed" in result["error"]
    assert target.read_text(encoding="utf-8") == before, "previous valid report must survive"


def test_archive_history_is_bounded(tmp_path):
    target = tmp_path / "autopilot_today.txt"
    for i in range(5):
        core.publish_away_report(
            _payload(generated_at=f"2026-07-10 13:0{i}:00"), target, archive_keep=3
        )
    archive = list((tmp_path / "away_report_archive").glob("autopilot_today_*.txt"))
    assert len(archive) <= 3


def test_write_away_report_compat_wrapper_still_returns_path(tmp_path):
    target = tmp_path / "autopilot_today.txt"
    path = core.write_away_report(_payload(), target)
    assert path == target
    assert target.exists()
