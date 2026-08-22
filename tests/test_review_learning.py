"""Phase 1 review learning: episodes, take rates, outcome joins, callouts."""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import review_learning
from review_learning import (
    aggregate_dimensions,
    attach_bounce_outcomes,
    attach_forward_returns,
    build_episodes,
    build_review_learning_state,
    find_callouts,
    load_review_learning_state,
    refresh_review_learning_if_stale,
    render_report,
    save_review_learning_state,
    watch_conversion,
)


def _row(action, symbol="NVDA", trade_date="2026-07-27", **extra):
    row = {
        "schema": "review_events_v1",
        "ts": f"{trade_date}T10:15:00",
        "trade_date": trade_date,
        "action": action,
        "symbol": symbol,
        "side": "LONG",
        "tier": "A",
        "bounce_types": "dynamic_vwap_upper_band",
        "market_environment": "BULLISH_WEAK",
        "session_rvol": 2.1,
        "rrs_spy": 1.4,
        "event_id": f"evt-{symbol}-{trade_date}",
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
def test_build_episodes_resolution_priority_and_context_merge():
    rows = [
        _row("shown", dwell_ms=None),
        _row("skip", dwell_ms=3000),
        # A later positive action outranks the earlier skip.
        _row("arm_watch", dwell_ms=9000, detail={"kind": "band_bounce"}),
        # Different symbol, explicit rejection only.
        _row("shown", symbol="AMD"),
        _row("remove_today", symbol="AMD", dwell_ms=1200),
        # Shown then nothing.
        _row("shown", symbol="TSLA"),
        # Toggle OFF is not a take.
        _row("shown", symbol="WMT"),
        _row("toggle_d1_focus", symbol="WMT", detail={"on": False}),
    ]
    episodes = {e.symbol: e for e in build_episodes(rows)}
    assert episodes["NVDA"].resolution == "take"
    assert episodes["NVDA"].dwell_ms == 9000
    assert episodes["NVDA"].tier == "A"
    assert episodes["NVDA"].shown is True
    assert episodes["AMD"].resolution == "reject"
    assert episodes["TSLA"].resolution == "shown_only"
    assert episodes["WMT"].resolution == "shown_only"

    # toggle ON without an impression still builds a (non-shown) take episode.
    only_toggle = build_episodes([_row("toggle_m5_focus", symbol="GME", detail={"on": True})])
    assert only_toggle[0].resolution == "take"
    assert only_toggle[0].shown is False


def test_aggregate_shrinks_thin_segments_toward_overall():
    rows = []
    # 12 S-tier shown, 8 taken; 12 B-tier shown, none taken.
    for index in range(12):
        rows.append(_row("shown", symbol=f"S{index}", tier="S"))
        if index < 8:
            rows.append(_row("add_focus", symbol=f"S{index}", tier="S"))
        rows.append(_row("shown", symbol=f"B{index}", tier="B"))
        rows.append(_row("skip", symbol=f"B{index}", tier="B", dwell_ms=2000))
    aggregate = aggregate_dimensions(build_episodes(rows))
    assert aggregate["shown"] == 24
    assert aggregate["overall_take_rate"] == round(8 / 24, 3)
    tiers = aggregate["dimensions"]["tier"]
    assert tiers["S"]["take"] == 8 and tiers["S"]["shown"] == 12
    # Shrinkage: (take + k*overall) / (n + k), k=10.
    overall = 8 / 24
    assert tiers["B"]["take_rate_shrunk"] == round((0 + 10 * overall) / 22, 3)
    assert tiers["S"]["take_rate_shrunk"] == round((8 + 10 * overall) / 22, 3)
    assert tiers["B"]["median_pass_dwell_ms"] == 2000


# ---------------------------------------------------------------------------
# Outcome joins
# ---------------------------------------------------------------------------
def test_attach_bounce_outcomes_streams_and_prefers_eod(tmp_path):
    outcomes = tmp_path / "outcomes.csv"
    with outcomes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "close_r", "outcome_mode"])
        writer.writeheader()
        writer.writerow({"event_id": "evt-NVDA-2026-07-27", "close_r": "0.4", "outcome_mode": "milestone"})
        writer.writerow({"event_id": "evt-NVDA-2026-07-27", "close_r": "1.2", "outcome_mode": "eod"})
        # A later non-eod row must NOT displace the eod value.
        writer.writerow({"event_id": "evt-NVDA-2026-07-27", "close_r": "0.9", "outcome_mode": "milestone"})
        writer.writerow({"event_id": "evt-OTHER", "close_r": "-2.0", "outcome_mode": "eod"})
    episodes = build_episodes([_row("shown"), _row("skip", dwell_ms=100)])
    matched = attach_bounce_outcomes(episodes, outcomes)
    assert matched == 1
    assert episodes[0].close_r == 1.2


def test_attach_forward_returns_grades_d1_names_side_adjusted():
    closes = [
        ("2026-07-01", 100.0),
        ("2026-07-02", 102.0),
        ("2026-07-03", 104.0),
        ("2026-07-06", 106.0),
        ("2026-07-07", 108.0),
        ("2026-07-08", 110.0),
    ]
    rows = [
        _row("shown", symbol="LNG", trade_date="2026-07-01", is_d1=True, event_id=""),
        _row(
            "shown",
            symbol="XPEV",
            trade_date="2026-07-01",
            is_d1=True,
            side="SHORT",
            event_id="",
        ),
    ]
    episodes = build_episodes(rows)
    matched = attach_forward_returns(episodes, load_frame=lambda _s: closes)
    assert matched == 2
    by_symbol = {e.symbol: e for e in episodes}
    assert by_symbol["LNG"].forward_pct[3] == 6.0  # 100 -> 106
    assert by_symbol["LNG"].forward_pct[5] == 10.0
    assert by_symbol["XPEV"].forward_pct[3] == -6.0  # short side-adjusted

    # Immature names grade only the horizons that have played out.
    young = build_episodes(
        [_row("shown", symbol="NEW", trade_date="2026-07-01", is_d1=True, event_id="")]
    )
    attach_forward_returns(young, load_frame=lambda _s: closes[:4])
    assert 3 in young[0].forward_pct and 5 not in young[0].forward_pct

    # M5 episodes with an intraday outcome are not forward-graded.
    m5 = build_episodes([_row("shown")])
    m5[0].close_r = 0.5
    assert attach_forward_returns(m5, load_frame=lambda _s: closes) == 0


# ---------------------------------------------------------------------------
# Callouts
# ---------------------------------------------------------------------------
def test_swing_setups_actions_feed_bucket_family_and_tag_dimensions():
    closes = [
        ("2026-07-27", 100.0),
        ("2026-07-28", 103.0),
        ("2026-07-29", 106.0),
        ("2026-07-30", 109.0),
    ]
    swing_extra = {
        "surface": "setups",
        "is_d1": True,
        "timeframe": "D1",
        "bucket": "favorite_setup",
        "setup_family": "avwap_breakout",
        "setup_tags": "AVWAP_BREAKOUT;D1_RS",
        "expected_r": 0.85,
        "event_id": "",
        "tier": "",
        "bounce_types": "",
    }
    rows = [
        # A ★ from the setups table (no impression) and a ✕ on another name.
        _row("favorite", symbol="LNG", detail={"on": True, "origin": "setups"}, **swing_extra),
        _row("dislike", symbol="WMT", detail={"reason": "meh", "origin": "setups"}, **swing_extra),
    ]
    episodes = build_episodes(rows)
    assert {e.resolution for e in episodes} == {"take", "reject"}
    assert all(e.bucket == "favorite_setup" for e in episodes)

    matched = attach_forward_returns(episodes, load_frame=lambda _s: closes)
    assert matched == 2  # both graded despite never being "shown"

    aggregate = aggregate_dimensions(episodes)
    bucket = aggregate["dimensions"]["bucket"]["favorite_setup"]
    assert bucket["n"] == 2 and bucket["shown"] == 0
    # The ★ lands in taken, the table-✕ counts as an active pass.
    assert bucket["taken"]["fwd_n"] == 1
    assert bucket["passed"]["fwd_n"] == 1
    assert bucket["taken"]["fwd_avg_pct"] == 9.0  # 100 -> 109 over 3 sessions
    family = aggregate["dimensions"]["setup_family"]["avwap_breakout"]
    assert family["n"] == 2
    assert aggregate["dimensions"]["setup_tag"]["AVWAP_BREAKOUT"]["n"] == 2
    assert aggregate["dimensions"]["expected_r_band"]["decent(0.5-1)"]["n"] == 2


def test_structured_setup_dislikes_are_counted_by_reason_code():
    swing_extra = {
        "surface": "setups",
        "is_d1": True,
        "timeframe": "D1",
        "bucket": "favorite_setup",
        "setup_family": "avwap_breakout",
        "event_id": "",
    }
    rows = [
        _row(
            "dislike",
            symbol="NVDA",
            detail={
                "origin": "setups",
                "reason": "late after a vertical move",
                "reason_code": "too_extended_from_base",
                "reason_codes": ["too_extended_from_base"],
                "vocab_version": 1,
            },
            **swing_extra,
        ),
        _row(
            "dislike",
            symbol="AMD",
            detail={
                "origin": "setups",
                "reason_codes": ["incoming_trendline", "overhead_horizontal"],
                "vocab_version": 1,
            },
            **swing_extra,
        ),
    ]
    episodes = build_episodes(rows)
    aggregate = aggregate_dimensions(episodes)
    reasons = aggregate["dimensions"]["dislike_reason"]
    assert reasons["too_extended_from_base"]["reject"] == 1
    assert reasons["incoming_trendline"]["n"] == 1
    assert reasons["overhead_horizontal"]["n"] == 1
    report = render_report({**aggregate, "generated_at": "now", "window_days": 90})
    assert "DISLIKE REASON" in report
    assert "too_extended_from_base" in report


def test_find_callouts_flags_blind_spots_and_leaks():
    rows = []
    for index in range(12):
        # S tier: taken often, measured badly -> leak.
        rows.append(_row("shown", symbol=f"S{index}", tier="S"))
        if index < 8:
            rows.append(_row("add_focus", symbol=f"S{index}", tier="S"))
        # B tier: passed always, measured well -> blind spot.
        rows.append(_row("shown", symbol=f"B{index}", tier="B"))
        rows.append(_row("skip", symbol=f"B{index}", tier="B", dwell_ms=500))
    episodes = build_episodes(rows)
    for episode in episodes:
        if episode.tier == "S" and episode.resolution == "take":
            episode.close_r = -0.5
        if episode.tier == "B":
            episode.close_r = 0.6
    aggregate = aggregate_dimensions(episodes)
    blind_spots, leaks = find_callouts(aggregate)
    assert any(e["dimension"] == "tier" and e["segment"] == "B" for e in blind_spots)
    assert any(e["dimension"] == "tier" and e["segment"] == "S" for e in leaks)
    # Thin segments never make a callout.
    assert all(e["shown"] >= review_learning.MIN_CALLOUT_EPISODES for e in blind_spots + leaks)


def test_watch_conversion_counts_endings_and_fill_sources():
    rows = [
        _row("arm_watch", detail={"kind": "band_bounce"}),
        _row("arm_watch", symbol="AMD", detail={"kind": "band_bounce"}),
        _row("watch_fired", detail={"kind": "band_bounce", "message": "hit"}),
        _row("watch_expired", symbol="AMD", detail={"kind": "band_bounce"}),
        _row("arm_watch", symbol="TSLA", detail={"kind": "new_hod"}),
        _row("disarm_watch", symbol="TSLA", detail={"kind": "new_hod"}),
        _row("arm_level", detail={"direction": "above", "level": 181.5, "fill_source": "upper_1"}),
        _row("arm_level", symbol="AMD", detail={"direction": "below", "level": 150.0, "fill_source": "vwap"}),
        _row("level_fired", detail={"direction": "above", "level": 181.5}),
    ]
    conversion = watch_conversion(rows)
    band = conversion["kinds"]["band_bounce"]
    assert band == {"armed": 2, "fired": 1, "expired": 1}
    assert conversion["kinds"]["new_hod"] == {"armed": 1, "disarmed": 1}
    assert conversion["kinds"]["d1_level"] == {"armed": 2, "fired": 1}
    assert conversion["level_fill_sources"] == {"upper_1": 1, "vwap": 1}


# ---------------------------------------------------------------------------
# End to end: state build, report, staleness refresh
# ---------------------------------------------------------------------------
def _write_events(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_state_end_to_end_with_window_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(review_learning, "_time_bucket", lambda _ts: "late_morning")
    events = tmp_path / "events.jsonl"
    outcomes = tmp_path / "outcomes.csv"
    rows = [
        _row("shown"),
        _row("add_focus", detail={"category": "m5", "added": True}),
        _row("shown", symbol="AMD"),
        _row("skip", symbol="AMD", dwell_ms=1500),
        # Outside the window: must be excluded.
        _row("shown", symbol="OLD", trade_date="2020-01-02"),
    ]
    _write_events(events, rows)
    with outcomes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "close_r", "outcome_mode"])
        writer.writeheader()
        writer.writerow({"event_id": "evt-NVDA-2026-07-27", "close_r": "0.8", "outcome_mode": "eod"})
        writer.writerow({"event_id": "evt-AMD-2026-07-27", "close_r": "0.3", "outcome_mode": "eod"})

    state = build_review_learning_state(
        events_path=events,
        outcomes_path=outcomes,
        window_days=90,
        now=datetime(2026, 7, 28, 18, 0),
    )
    assert state["schema"] == review_learning.REVIEW_LEARNING_SCHEMA
    assert state["shown"] == 2  # OLD filtered out
    assert state["overall_take_rate"] == 0.5
    assert state["outcome_matches"] == 2
    tier = state["dimensions"]["tier"]["A"]
    assert tier["taken"]["r_avg"] == 0.8
    assert tier["passed"]["r_avg"] == 0.3
    assert state["dimensions"]["time_bucket"]["late_morning"]["shown"] == 2

    report = render_report(state)
    assert "REVIEW PREFERENCE SCOREBOARD" in report
    assert "BLIND SPOTS" in report and "LEAKS" in report
    assert "TIME BUCKET" in report

    state_path = tmp_path / "state.json"
    save_review_learning_state(state, state_path)
    assert load_review_learning_state(state_path) == json.loads(
        state_path.read_text(encoding="utf-8")
    )


def test_refresh_if_stale_gates_on_mtimes(tmp_path):
    events = tmp_path / "events.jsonl"
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.txt"
    outcomes = tmp_path / "outcomes.csv"  # absent is fine

    # No events log yet -> nothing to build.
    assert not refresh_review_learning_if_stale(
        events_path=events, outcomes_path=outcomes, state_path=state_path, report_path=report_path
    )

    _write_events(events, [_row("shown"), _row("skip", dwell_ms=100)])
    assert refresh_review_learning_if_stale(
        events_path=events, outcomes_path=outcomes, state_path=state_path, report_path=report_path
    )
    assert state_path.exists() and report_path.exists()

    # Fresh state, no new events -> skip.
    assert not refresh_review_learning_if_stale(
        events_path=events, outcomes_path=outcomes, state_path=state_path, report_path=report_path
    )

    # New decisions after the state was built -> rebuild.
    time.sleep(0.05)
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row("shown", symbol="AMD")) + "\n")
    later = state_path.stat().st_mtime + 5
    import os

    os.utime(events, (later, later))
    assert refresh_review_learning_if_stale(
        events_path=events, outcomes_path=outcomes, state_path=state_path, report_path=report_path
    )


def test_refresh_reads_partitioned_shards_when_legacy_file_does_not_exist(
    tmp_path, monkeypatch
):
    import review_events

    legacy = tmp_path / "alert_review_events.jsonl"
    shards = tmp_path / "alert_review_events"
    shards.mkdir()
    identity = "a" * 32
    _write_events(
        shards / f"review-events-{identity}.jsonl",
        [
            _row(
                "shown",
                schema="review_events_v2",
                installation_id=identity,
                review_record_id="row-1",
            )
        ],
    )
    monkeypatch.setattr(review_events, "ALERT_REVIEW_EVENTS_FILE", legacy)
    monkeypatch.setattr(review_events, "ALERT_REVIEW_EVENTS_DIR", shards)
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.txt"

    assert refresh_review_learning_if_stale(
        events_path=legacy,
        outcomes_path=tmp_path / "outcomes.csv",
        state_path=state_path,
        report_path=report_path,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["event_rows"] == 1


def test_a_like_advance_is_a_positive_episode_not_a_rejection():
    """R9.2 (2026-08-22). The capture rail's LIKE used to route through the
    "Not today" verb, so `remove_today` was written and `REJECT_ACTIONS`
    counted every like the trader ever filed as a dismissal. Across the
    2026-07-24..08-21 window that is 40 of 52 like_claim rows scored backwards:
    the strongest positive signal in the store, read as its opposite.
    """
    assert "like_advance" in review_learning.TAKE_ACTIONS
    assert "like_advance" not in review_learning.REJECT_ACTIONS

    episodes = build_episodes([_row("shown"), _row("like_advance")])
    assert [e.resolution for e in episodes] == ["take"]


def test_a_like_advance_outranks_a_later_queue_clear():
    """Resolution priority is "any positive engagement outranks an explicit
    rejection", and a like is positive engagement. A remove_today arriving on
    the same symbol later in the day must not overturn it."""
    episodes = build_episodes(
        [_row("shown"), _row("like_advance"), _row("remove_today")]
    )
    assert [e.resolution for e in episodes] == ["take"]
