"""Auto EVENING mode: sleep-in semantics, strength persistence, briefing."""

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import evening_mode  # noqa: E402
import autopilot_core as core  # noqa: E402


# ---------------------------------------------------------------------------
# Mode plumbing
# ---------------------------------------------------------------------------
def test_auto_mode_truth_table_includes_evening():
    from ui.services.autopilot_service import (
        AUTO_MODE_OFF,
        AUTO_PROFILE_EVENING,
        AutopilotService,
    )

    service = AutopilotService.__new__(AutopilotService)
    service._enabled = True
    service._profile = AUTO_PROFILE_EVENING
    assert service.auto_mode == "EVENING"
    service._enabled = False
    assert service.auto_mode == AUTO_MODE_OFF


def test_read_auto_pilot_mode_accepts_evening(tmp_path):
    state = tmp_path / "autopilot_state.json"
    state.write_text('{"enabled": true, "profile": "EVENING"}', encoding="utf-8")
    assert core.read_auto_pilot_mode(state) == "EVENING"
    state.write_text('{"enabled": true, "profile": "NONSENSE"}', encoding="utf-8")
    assert core.read_auto_pilot_mode(state) == "DESK"
    state.write_text('{"enabled": false, "profile": "EVENING"}', encoding="utf-8")
    assert core.read_auto_pilot_mode(state) == "OFF"


def test_evening_stages_picks_like_desk():
    # The BounceBot gate reads the mode file and stages (never self-applies)
    # for both DESK and EVENING - the "no recommendations while asleep" rule.
    source = (SCRIPTS_DIR / "bounce_bot_lib" / "legacy.py").read_text(encoding="utf-8")
    assert 'read_auto_pilot_mode() in ("DESK", "EVENING")' in source


def test_evening_hourly_report_publishes_like_away():
    from ui.services.autopilot_service import AUTO_PROFILE_EVENING, AutopilotService

    writes = []
    service = AutopilotService.__new__(AutopilotService)
    service._enabled = True
    service._profile = AUTO_PROFILE_EVENING
    service._state = {"hourly_report_slot": None}
    service._write_report = lambda: writes.append("write") or {"ok": True}
    service._save_state = lambda: None
    service._log = lambda _message: None

    moment = datetime(2026, 8, 3, 7, 0)  # a Monday
    service._maybe_hourly_away_report(moment)
    service._maybe_hourly_away_report(moment)
    assert writes == ["write"]


def test_early_swing_slot_only_in_evening():
    reference = datetime(2026, 8, 3, 6, 0)  # Monday, normal 06:30 session
    normal = core.get_autopilot_swing_slots(reference, "America/Los_Angeles")
    early = core.get_autopilot_swing_slots(
        reference, "America/Los_Angeles", include_early_slot=True
    )
    assert normal[0] == "07:30"
    assert early[0] == "07:00" and early[1:] == normal


def test_report_header_renders_evening_mode():
    base = {
        "generated_at": "2026-08-03 07:00:00",
        "ib_status": "connected",
        "regime": "bullish_weak",
        "longs": [],
        "shorts": [],
        "swing_picks": [],
        "alerts": [],
        "slots_done": [],
        "next_slot": "08:00",
        "log_lines": [],
        "auto_longs": [],
        "auto_shorts": [],
    }
    report = core.render_away_report({**base, "enabled": True, "auto_mode": "EVENING"})
    assert "Mode: AUTO - EVENING" in report
    assert "hourly from 07:00 local" in report
    assert "picks stage for chart approval only" in report

    with_briefing = core.render_away_report(
        {
            **base,
            "enabled": True,
            "auto_mode": "EVENING",
            "evening_briefing_lines": ["Environment: bullish_weak", "Best D1 longs: NVDA"],
        }
    )
    assert "== MORNING BRIEFING (EVENING MODE) ==" in with_briefing
    assert "Best D1 longs: NVDA" in with_briefing
    assert "== MORNING BRIEFING" not in report


# ---------------------------------------------------------------------------
# Strength checks + persistence verdicts
# ---------------------------------------------------------------------------
def test_due_strength_check_takes_latest_and_never_replays():
    slots = ("07:00", "07:15", "07:30")
    assert evening_mode.due_strength_check(datetime(2026, 8, 3, 6, 45), [], slots) is None
    assert evening_mode.due_strength_check(datetime(2026, 8, 3, 7, 1), [], slots) == "07:00"
    # Started late: the 07:15 look runs immediately, 07:00 is never replayed.
    assert evening_mode.due_strength_check(datetime(2026, 8, 3, 7, 20), [], slots) == "07:15"
    assert (
        evening_mode.due_strength_check(datetime(2026, 8, 3, 7, 20), ["07:15"], slots) is None
    )
    # A recorded later slot retires the earlier ones too.
    assert (
        evening_mode.due_strength_check(datetime(2026, 8, 3, 8, 0), ["07:30"], slots) is None
    )
    assert evening_mode.due_strength_check(datetime(2026, 8, 3, 7, 40), ["07:15"], slots) == "07:30"


def test_persistence_held_and_faded_verdicts():
    staged = [
        {"symbol": "AAA", "side": "long", "score": 2.0, "reason": "PDH break"},
        {"symbol": "BBB", "side": "long", "score": 1.5, "reason": "RS leader"},
        {"symbol": "CCC", "side": "short", "score": 1.8, "reason": "PDL break"},
    ]
    state = {"date": "2026-08-03"}
    # 07:00: everything pressing its extreme.
    evening_mode.record_strength_check(
        state,
        "07:00",
        staged,
        {
            "AAA": {"last": 99.8, "day_high": 100.0, "day_low": 95.0},
            "BBB": {"last": 49.9, "day_high": 50.0, "day_low": 48.0},
            "CCC": {"last": 20.05, "day_high": 22.0, "day_low": 20.0},
        },
        datetime(2026, 8, 3, 7, 0),
    )
    # 07:30: AAA still pressing, BBB slipped off its high, CCC still weak.
    evening_mode.record_strength_check(
        state,
        "07:30",
        staged,
        {
            "AAA": {"last": 100.4, "day_high": 100.6, "day_low": 95.0},
            "BBB": {"last": 48.9, "day_high": 50.2, "day_low": 48.0},
            "CCC": {"last": 20.02, "day_high": 22.0, "day_low": 19.95},
        },
        datetime(2026, 8, 3, 7, 30),
    )
    verdicts = evening_mode.assess_pick_persistence(state)
    assert verdicts["AAA"]["verdict"] == "held"
    assert verdicts["CCC"]["verdict"] == "held"
    assert verdicts["BBB"]["verdict"] == "faded"
    assert verdicts["BBB"]["checks"] == 2


def test_persistence_price_drift_against_side_fades_even_near_extreme():
    # Short that bounced 1% off its low but is "near LOD" because the low
    # moved: price direction still vetoes it.
    state = {"date": "2026-08-03"}
    staged = [{"symbol": "DDD", "side": "short", "score": 1.0, "reason": ""}]
    evening_mode.record_strength_check(
        state, "07:00", staged, {"DDD": {"last": 20.0, "day_high": 22.0, "day_low": 19.99}},
        datetime(2026, 8, 3, 7, 0),
    )
    evening_mode.record_strength_check(
        state, "07:30", staged, {"DDD": {"last": 20.3, "day_high": 22.0, "day_low": 20.25}},
        datetime(2026, 8, 3, 7, 30),
    )
    assert evening_mode.assess_pick_persistence(state)["DDD"]["verdict"] == "faded"


def test_staged_picks_flatten_from_pending_payload():
    payload = {
        "date": "2026-08-03",
        "pending": {
            "long": {"AAA": {"score": 2.5, "reason": "ADR breakout"}},
            "short": {"ZZZ": {"score": 1.1, "reason": "relative weakness"}},
        },
        "decided": {"long": {}, "short": {}},
    }
    picks = evening_mode.staged_picks_from_pending(payload)
    assert {(p["symbol"], p["side"]) for p in picks} == {("AAA", "long"), ("ZZZ", "short")}


# ---------------------------------------------------------------------------
# Briefing build + render
# ---------------------------------------------------------------------------
def _sample_briefing():
    swing_rows = [
        {"symbol": "NVDA", "side": "long", "expected_r": 2.4, "bucket_label": "high conviction"},
        {"symbol": "AMD", "side": "long", "expected_r": 1.1, "bucket_label": "favorite"},
        {"symbol": "XOM", "side": "short", "expected_r": 1.9, "bucket_label": "favorite"},
        {"symbol": "JUNK", "side": "long", "expected_r": None, "bucket_label": ""},
    ]
    persistence = {
        "AAA": {"side": "long", "score": 2.0, "reason": "PDH break", "verdict": "held", "detail": "0.20% off HOD at 07:30", "checks": 3, "last": 100.4},
        "BBB": {"side": "long", "score": 1.5, "reason": "RS leader", "verdict": "faded", "detail": "faded to 2.59% off HOD by 07:30", "checks": 3, "last": 48.9},
    }
    triggers = [
        {"date": "2026-08-03", "at": "05:12:00", "symbol": "SPY", "side": "below", "level": "555.0", "last": "554.8", "note": ""},
    ]
    return evening_mode.build_evening_briefing(
        now=datetime(2026, 8, 3, 7, 31),
        regime="bullish_strong",
        swing_rows=swing_rows,
        swing_data_current=True,
        persistence=persistence,
        overnight_triggers=triggers,
        checks_done=["07:00", "07:15", "07:30"],
    )


def test_briefing_ranks_d1s_and_separates_held_from_faded():
    payload = _sample_briefing()
    assert [row["symbol"] for row in payload["best_d1"]["long"]] == ["NVDA", "AMD", "JUNK"]
    assert [row["symbol"] for row in payload["best_d1"]["short"]] == ["XOM"]
    assert [item["symbol"] for item in payload["held_picks"]] == ["AAA"]
    assert [item["symbol"] for item in payload["faded_picks"]] == ["BBB"]

    text = evening_mode.render_evening_briefing(payload)
    assert "EVENING MODE - MORNING BRIEFING" in text
    assert "Market environment: bullish_strong" in text
    assert "NVDA | high conviction | 2.40R" in text
    assert "FADED SINCE THE FIRST CHECK - NOT RECOMMENDED" in text
    assert "BBB (LONG)" in text
    assert "SPY 554.8 crossed BELOW 555.0" in text
    assert "Flip Auto mode off EVENING" in text


def test_briefing_summary_lines_are_phone_compact():
    lines = evening_mode.briefing_summary_lines(_sample_briefing())
    joined = "\n".join(lines)
    assert "Best D1 longs: NVDA, AMD, JUNK" in joined
    assert "Held strong: AAA" in joined
    assert "Faded (skip): BBB" in joined
    assert "Overnight price alerts: 1 fired" in joined


def test_briefing_flags_stale_swing_data():
    payload = evening_mode.build_evening_briefing(
        now=datetime(2026, 8, 3, 7, 5),
        regime="neutral",
        swing_rows=[{"symbol": "NVDA", "side": "long", "expected_r": 2.0, "bucket_label": ""}],
        swing_data_current=False,
        persistence={},
        overnight_triggers=[],
        checks_done=["07:00"],
    )
    text = evening_mode.render_evening_briefing(payload)
    assert "PREVIOUS session" in text


def test_evening_state_day_scopes(tmp_path):
    path = tmp_path / "evening_state.json"
    state = evening_mode.load_evening_state(datetime(2026, 8, 3, 7, 0), path)
    state["checks"]["07:00"] = {"at": "07:00:05", "symbols": {}}
    evening_mode.save_evening_state(state, path)
    same_day = evening_mode.load_evening_state(datetime(2026, 8, 3, 9, 0), path)
    assert "07:00" in same_day["checks"]
    next_day = evening_mode.load_evening_state(datetime(2026, 8, 4, 7, 0), path)
    assert next_day["checks"] == {}
