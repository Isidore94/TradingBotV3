"""Plan.md 23.8: verified atomic away-report publication."""

import hashlib
import json
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
    metadata = json.loads((tmp_path / "autopilot_today.txt.meta.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == "away_report_publish_v1"
    assert metadata["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def test_readback_mismatch_restores_previous_verified_report(tmp_path, monkeypatch):
    target = tmp_path / "autopilot_today.txt"
    assert core.publish_away_report(_payload(generated_at="first"), target)["ok"]
    before = target.read_bytes()
    real_replace = core.os.replace
    target_replacements = 0

    def corrupt_first_target_replace(source, destination):
        nonlocal target_replacements
        real_replace(source, destination)
        if Path(destination) == target and target_replacements == 0:
            target_replacements += 1
            target.write_text("corrupted after replace", encoding="utf-8")

    monkeypatch.setattr(core.os, "replace", corrupt_first_target_replace)
    result = core.publish_away_report(_payload(generated_at="second"), target)

    assert result["ok"] is False
    assert result["restored_previous"] is True
    assert target.read_bytes() == before


def test_metadata_write_failure_restores_whole_previous_publication(tmp_path, monkeypatch):
    target = tmp_path / "autopilot_today.txt"
    metadata_path = tmp_path / "autopilot_today.txt.meta.json"
    assert core.publish_away_report(_payload(generated_at="first"), target)["ok"]
    before_report = target.read_bytes()
    before_metadata = metadata_path.read_bytes()
    real_replace = core.os.replace

    def fail_new_metadata_replace(source, destination):
        if Path(destination) == metadata_path:
            raise OSError("simulated metadata replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(core.os, "replace", fail_new_metadata_replace)
    result = core.publish_away_report(_payload(generated_at="second"), target)

    assert result["ok"] is False
    assert result["restored_previous"] is True
    assert target.read_bytes() == before_report
    assert metadata_path.read_bytes() == before_metadata


def test_phone_digest_includes_operational_and_tracker_truth():
    text = core.render_away_report(
        _payload(
            runtime_line="Runtime: DESKTOP pid=123",
            operations_line="Health: DEGRADED",
            last_scan_line="Last scan: 13:00 ok in 18.5m",
            swing_data_line="Swing data: awaiting today's first completed scan.",
            tracker_line="Tracker: WRITE SKIPPED - non-IBKR daily data",
        )
    )

    assert "Health: DEGRADED" in text
    assert "Last scan: 13:00 ok in 18.5m" in text
    assert "awaiting today's first completed scan" in text
    assert "Tracker: WRITE SKIPPED" in text


def test_phone_digest_prioritizes_swings_before_intraday_candidates():
    text = core.render_away_report(
        _payload(
            swing_data_current=True,
            swing_picks=[
                {
                    "symbol": "NEAR",
                    "side": "LONG",
                    "bucket": "Near Favorite Zone",
                    "expected_r": 0.8,
                },
                {"symbol": "FAVE", "side": "SHORT", "bucket": "Favorite", "expected_r": 1.0},
                {
                    "symbol": "BEST",
                    "side": "LONG",
                    "bucket": "High Conviction",
                    "expected_r": 1.4,
                },
            ],
            auto_longs=["AUTO"],
            auto_shorts=["BOTSHORT"],
        )
    )

    swing_at = text.index("== SWING OPPORTUNITIES ==")
    day_longs_at = text.index("== DAY TRADE LONGS")
    day_shorts_at = text.index("== DAY TRADE SHORTS")
    bot_longs_at = text.index("== BOT PICKS - LONGS")

    assert text.index("Updated:") < swing_at
    assert swing_at < day_longs_at < day_shorts_at < bot_longs_at
    assert text.index("BEST (LONG) | High Conviction | 1.40R") < text.index(
        "FAVE (SHORT) | Favorite | 1.00R"
    )
    assert text.index("FAVE (SHORT) | Favorite | 1.00R") < text.index(
        "NEAR (LONG) | Near Favorite Zone | 0.80R"
    )


def test_phone_digest_caps_near_favorite_rows():
    """2026-07-17 week review: near_favorite_zone measured -0.18R vs favorite
    +1.01R - favorites always show, the near bucket is capped with a note."""
    near = [
        {"symbol": f"NEAR{i}", "side": "LONG", "bucket": "Near", "expected_r": -0.1}
        for i in range(6)
    ]
    favorites = [
        {"symbol": f"FAVE{i}", "side": "LONG", "bucket": "Favorite", "expected_r": 0.5}
        for i in range(4)
    ]
    text = core.render_away_report(
        _payload(swing_data_current=True, swing_picks=[*near, *favorites])
    )

    for i in range(4):
        assert f"FAVE{i}" in text  # every favorite survives
    shown_near = [f"NEAR{i}" for i in range(6) if f"NEAR{i} (LONG)" in text]
    assert len(shown_near) == core.AWAY_REPORT_MAX_NEAR_ROWS
    assert "+3 more near-favorite rows hidden" in text
    # Favorites still lead the section.
    assert text.index("FAVE0 (LONG)") < text.index("NEAR0 (LONG)")


def test_phone_digest_distinguishes_no_current_swings_from_unscanned_data():
    current = core.render_away_report(
        _payload(swing_picks=[], swing_data_current=True)
    )
    awaiting = core.render_away_report(
        _payload(
            swing_picks=[],
            swing_data_current=False,
            swing_data_line="Swing data: awaiting today's first completed scan.",
        )
    )

    assert "No qualified current-session swing opportunity." in current
    assert "Awaiting today's first completed swing scan." in awaiting
    assert awaiting.index("== SWING OPPORTUNITIES ==") < awaiting.index(
        "== DAY TRADE LONGS"
    )


def test_operations_audit_is_condensed_for_phone_report():
    lines = core.build_away_operations_lines(
        {
            "status": "degraded",
            "summary": {"healthy": 5, "degraded": 1, "unhealthy": 0},
            "latest_manifest": {
                "trigger": "Auto Pilot swing scan (13:00, WITH setup-tracker write)",
                "status": "ok",
                "total_seconds": 1110,
                "counters": {
                    "update_setup_tracker": True,
                    "setup_tracker_updated": False,
                },
                "outputs": {"setup_tracker_skip_reason": "IBKR daily bars unavailable"},
            },
        }
    )

    assert lines["operations_line"].startswith("Health: DEGRADED")
    assert "18.5m" in lines["last_scan_line"]
    assert lines["tracker_line"] == "Tracker: WRITE SKIPPED - IBKR daily bars unavailable"
