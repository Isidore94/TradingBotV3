"""The tracker adapter, M5-close recipes, five bias views, and night report."""

from __future__ import annotations

import ast
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ai_jobs import setup_research  # noqa: E402
from research_warehouse import exchange_calendar as xcal  # noqa: E402
from research_warehouse import market_bias_context, outcomes, tracker_adapter  # noqa: E402

UTC = timezone.utc


def _scenario(family="post_earnings_candle_break", **overrides):
    row = {
        "setup_id": "2026-08-03:AAA:LONG:2026-07-20:favorite_setup",
        "scan_date": "2026-08-03",
        "symbol": "AAA",
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "entry_price": "100",
        "anchor_date": "2026-07-20",
        "stop_reference_label": "POST_EARNINGS_CANDLE_LOW",
        "stop_source_type": "post_earnings_candle",
        "close_failure_limit": "1",
        "experimental": "False",
        "tradeable": "True",
        "initial_risk_per_share": "2",
    }
    row.update(overrides)
    return row


def _event(family="post_earnings_candle_break"):
    return {
        "event_type": "initial",
        "event_at": "2026-08-04T01:00:00+00:00",
        "setup_id": _scenario()["setup_id"],
        "scan_date": "2026-08-03",
        "symbol": "AAA",
        "side": "LONG",
        "state_side": "LONG",
        "state_setup_family": family,
        "state_setup_status": "OPEN",
    }


def _m5(session, symbol="AAA", base=100.0, count=78, step=0.0):
    rows = []
    for index in range(count):
        close = base + index * step
        rows.append(
            {
                "symbol": symbol,
                "interval_start": session.rth_open_at + timedelta(minutes=5 * index),
                "interval_end": session.rth_open_at + timedelta(minutes=5 * (index + 1)),
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1000,
                "capture_mode": "BACKFILL",
                "is_complete": True,
            }
        )
    return rows


def test_adapter_covers_every_family_and_never_needs_the_giant_tracker():
    families = ["post_earnings_candle_break", "avwap_breakout", "top_pattern"]
    scenarios = []
    events = []
    for index, family in enumerate(families):
        setup_id = f"2026-08-03:S{index}:LONG:2026-07-20:favorite_setup"
        scenarios.append(_scenario(family, setup_id=setup_id, symbol=f"S{index}"))
        events.append({**_event(family), "setup_id": setup_id, "symbol": f"S{index}"})

    detections = tracker_adapter.detections_from_tracker(
        scenario_rows=scenarios, event_rows=events
    )

    assert {row["canonical_setup_id"] for row in detections} == {
        "POST_EARNINGS_CANDLE_BREAK", "AVWAP_BREAKOUT", "TOP_PATTERN_ENTRY"
    }
    assert all(row["trigger_at"] == xcal.trading_session(date(2026, 8, 3)).rth_close_at for row in detections)
    tags = json.loads(detections[0]["tags"])
    assert tags["entry_contract"] == tracker_adapter.ENTRY_CONTRACT
    assert tags["stop_candidates"][0]["level"] == 98.0
    assert "MASTER_AVWAP_SETUP_TRACKER_FILE" not in Path(tracker_adapter.__file__).read_text(encoding="utf-8")


def test_daily_rescans_of_one_anchor_stay_one_occurrence():
    second_id = "2026-08-04:AAA:LONG:2026-07-20:favorite_setup"
    rows = tracker_adapter.detections_from_tracker(
        scenario_rows=[
            _scenario(),
            _scenario(setup_id=second_id, scan_date="2026-08-04", entry_price="105"),
        ],
        event_rows=[
            _event(),
            {**_event(), "setup_id": second_id, "scan_date": "2026-08-04"},
        ],
    )

    assert len(rows) == 1
    assert rows[0]["trigger_at"] == xcal.trading_session(date(2026, 8, 3)).rth_close_at
    tags = json.loads(rows[0]["tags"])
    assert tags["rescan_count"] == 2
    assert tags["stop_candidates"][0]["level"] == 98.0  # first scan; no look-ahead


def test_d1_setup_enters_on_next_sessions_first_completed_m5_close():
    detection = tracker_adapter.detections_from_tracker(
        scenario_rows=[_scenario(stop_source_type="current_anchor", stop_reference_label="AVWAPE")],
        event_rows=[_event()],
    )[0]
    from research_warehouse.occurrences import build_occurrence_row

    occurrence = build_occurrence_row(detection, now=datetime(2026, 8, 5, tzinfo=UTC))
    monday = xcal.trading_session(date(2026, 8, 3))
    tuesday = xcal.trading_session(date(2026, 8, 4))
    bars = _m5(monday, base=99.0, count=78)
    bars += _m5(tuesday, base=101.0, count=4)
    bars[-3] = dict(bars[-3], high=104.2, close=102.0)
    recipe = next(row for row in outcomes.M5_CLOSE_RECIPES if row.recipe_id == "m5close_current_anchor1_1r_v1")

    at_entry = outcomes.simulate_m5_close_opportunity(
        occurrence,
        bars,
        recipe,
        as_of=tuesday.rth_open_at + timedelta(minutes=5),
    )
    result = outcomes.simulate_m5_close_opportunity(
        occurrence, bars, recipe, as_of=datetime(2026, 9, 1, tzinfo=UTC)
    )

    assert at_entry["result_state"] == "OPEN", "bars after as_of must not leak into the path"
    assert result["entry_at"] == tuesday.rth_open_at + timedelta(minutes=5)
    assert result["entry_price"] == 101.0
    assert result["stop_price"] == 98.0
    assert result["result_state"] == "TARGETED"
    assert result["gross_r"] == 1.0


def test_five_timeframe_bias_uses_only_completed_spy_bars():
    spy_m5 = []
    spy_d1 = []
    day = date(2026, 6, 22)
    built = 0
    while built < 26:
        session = xcal.trading_session(day)
        if session:
            base = 500.0 + built
            spy_m5.extend(_m5(session, symbol="SPY", base=base, step=0.01))
            spy_d1.append(
                {
                    "symbol": "SPY", "session_date": day, "open": base,
                    "high": base + 2, "low": base - 1, "close": base + 1,
                    "volume": 10_000_000, "capture_mode": "BACKFILL",
                }
            )
            built += 1
        day += timedelta(days=1)
    entry_session = xcal.trading_session(day)
    while entry_session is None:
        day += timedelta(days=1)
        entry_session = xcal.trading_session(day)
    spy_m5.extend(_m5(entry_session, symbol="SPY", base=530.0, count=1))
    entry_at = entry_session.rth_open_at + timedelta(minutes=5)

    reads = market_bias_context.context_at(entry_at, spy_m5=spy_m5, spy_d1=spy_d1)

    assert tuple(reads) == market_bias_context.TIMEFRAMES
    assert reads["M5"]["env_key"] == "bullish_strong"  # exact live early-session fallback
    assert reads["M5"]["source"] == "champion_day_pct_regime_window"
    assert reads["D1"]["env_key"] != "unknown"
    assert reads["H4"]["env_key"] != "unknown"


def test_night_fact_pack_gates_ai_on_real_sample_depth():
    occurrence_map = {}
    result_rows = []
    contexts = {}
    start = datetime(2026, 7, 1, 13, 35, tzinfo=UTC)
    for index in range(30):
        occurrence_id = f"occ-{index}"
        occurrence_map[occurrence_id] = {
            "occurrence_id": occurrence_id,
            "canonical_setup_id": "POST_EARNINGS_CANDLE_BREAK",
            "side": "LONG",
            "symbol": f"S{index % 5}",
        }
        result_rows.append(
            {
                "occurrence_id": occurrence_id,
                "recipe_id": "m5close_post_earnings_candle1_2r_v1",
                "entry_at": start + timedelta(days=index % 5),
                "net_r": 2.0 if index % 3 else -1.0,
                "first_hit": "TARGET" if index % 3 else "STOP",
                "result_state": "TARGETED" if index % 3 else "STOPPED",
            }
        )
        contexts[occurrence_id] = {"M5": "bullish_weak", "D1": "bullish_strong"}

    pack = setup_research.build_fact_pack(result_rows, occurrence_map, contexts)

    assert pack["gate"]["met"] is True
    assert pack["policies"][0]["stats"]["eligible"] is True
    assert pack["policies"][0]["stats"]["win_rate"] == pytest.approx(2 / 3, abs=0.0001)
    assert pack["data_contract"] == {
        "planned_stop_or_risk_required": False,
        "m1_used": False,
        "bid_ask_used": False,
        "earnings_fundamentals_used": False,
        "same_bar_rule": "STOP_FIRST",
        "numbers_written_by": "deterministic code",
    }


def test_setup_research_is_appended_to_the_nightly_slate():
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]
    assert names[-1] == "setup_research"


def test_night_report_cannot_import_live_control_paths_and_skips_ai_below_gate(tmp_path, monkeypatch):
    tree = ast.parse((SCRIPTS / "ai_jobs" / "setup_research.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        name.startswith(("bounce_bot", "master_avwap", "ui", "autopilot"))
        for name in imported
    )

    monkeypatch.setattr(
        setup_research,
        "_narrate",
        lambda *_args, **_kwargs: pytest.fail("model must not run below the evidence floor"),
    )
    result = setup_research.run_setup_research(
        root=tmp_path,
        inputs=([], {}, {}, {"outcomes": 0}),
    )
    assert result["status"] == "ok"
    assert result["model"] == ""
    assert "no model called" in result["reason"]
