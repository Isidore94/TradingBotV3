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
    blind_spots, leaks, _r_gaps = find_callouts(aggregate)
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


# ---------------------------------------------------------------------------
# The seven explicit decisions that used to score as silence (P1 #4, 2026-09-01)
# ---------------------------------------------------------------------------
def test_every_explicit_decision_resolves_as_one():
    """About 640 decisions the trader made on a chart were scored
    `shown_only` - i.e. as if they had never answered. Live counts across the
    store: auto_pick_pass 254, arm_d1_event 160, focus_review_remove 88,
    focus_review_keep 71, auto_pick_approve 63, arm_any_bounce 22,
    veto_day_trade 4.

    Each classification is the writer's BEHAVIOUR, read in
    alert_center_panel.py: approve writes the pick to a watchlist, pass leaves
    watchlists untouched, keep leaves the pick in Focus, remove calls
    `remove_everywhere`, the two arms are the same gesture as `arm_watch`, and
    veto_day_trade vetoes the D1 chart that was shown.

    Fail-before-fix: on the un-fixed sets all seven resolve to "shown_only".
    """
    expected = {
        "auto_pick_approve": "take",
        "focus_review_keep": "take",
        "arm_d1_event": "take",
        "arm_any_bounce": "take",
        "auto_pick_pass": "reject",
        "focus_review_remove": "reject",
        "veto_day_trade": "reject",
    }
    for index, (action, resolution) in enumerate(expected.items()):
        symbol = f"SYM{index}"
        episodes = build_episodes([_row("shown", symbol=symbol), _row(action, symbol=symbol)])
        assert len(episodes) == 1
        assert episodes[0].resolution == resolution, f"{action} must resolve as {resolution}"
        assert episodes[0].shown is True


def test_a_veto_day_trade_is_a_reject_of_the_chart_that_was_shown():
    """"Veto D1 - but M5 today" adds the name to M5 Focus, but the
    episode being graded is the D1 chart, and the trader vetoed it. Scoring it
    a take would say the D1 setup was taken, which is the opposite of what
    happened; the M5 interest is a different claim on a different timeframe and
    the annotation store carries it."""
    assert "veto_day_trade" in review_learning.REJECT_ACTIONS
    assert "veto_day_trade" not in review_learning.TAKE_ACTIONS


def test_machine_events_and_disarms_are_still_not_decisions():
    """Nothing automatic, and no later change of mind about an arm, may resolve
    an episode: only what the trader answered about the chart in front of
    them."""
    for action in (
        "focus_d1_flag",
        "auto_pick_auto_focus",
        "regime_pause_auto_focus",
        "watch_fired",
        "level_fired",
        "d1_event_fired",
        "armed_alert_expired",
        "hold_expired",
        "watch_expired",
        "disarm_watch",
        "disarm_d1_event",
        "disarm_any_bounce",
    ):
        assert action not in review_learning.TAKE_ACTIONS
        assert action not in review_learning.TOGGLE_TAKE_ACTIONS
        assert action not in review_learning.REJECT_ACTIONS
        episodes = build_episodes([_row("shown", symbol="ZZZ"), _row(action, symbol="ZZZ")])
        assert episodes[0].resolution == "shown_only", action


# ---------------------------------------------------------------------------
# The r_gap callout class
# ---------------------------------------------------------------------------
def _segment(taken_r, taken_n, passed_r, passed_n, *, shown=32, take=8):
    return {
        "dimensions": {
            "bounce_type": {
                "lrsi_cross_20": {
                    "n": shown,
                    "shown": shown,
                    "take": take,
                    "take_total": take,
                    "skip": 0,
                    "reject": 0,
                    "take_rate": round(take / shown, 3),
                    "take_rate_shrunk": 0.25,
                    "median_take_dwell_ms": None,
                    "median_pass_dwell_ms": None,
                    "taken": {"r_n": taken_n, "r_avg": taken_r, "fwd_n": 0, "fwd_avg_pct": None},
                    "passed": {"r_n": passed_n, "r_avg": passed_r, "fwd_n": 0, "fwd_avg_pct": None},
                }
            }
        },
        "overall_take_rate": 0.25,
    }


def test_the_r_gap_callout_catches_the_live_lrsi_case():
    """The case this class exists for, with the numbers it was measured
    at on 2026-09-01: `bounce_type=lrsi_cross_20`, taken -0.376R over 8
    measured takes against passed +0.962R over 24. Its take rate sat at the
    trader's overall rate, so neither the blind-spot nor the leak test - both
    of which start FROM the take rate - could ever see it.

    Fail-before-fix: `find_callouts` returned two lists and had no such class.
    """
    aggregate = _segment(-0.376, 8, 0.962, 24)
    blind_spots, leaks, r_gaps = find_callouts(aggregate)

    assert blind_spots == [] and leaks == [], "the take-rate classes cannot see this"
    assert len(r_gaps) == 1
    entry = r_gaps[0]
    assert entry["dimension"] == "bounce_type"
    assert entry["segment"] == "lrsi_cross_20"
    assert entry["taken_r_avg"] == -0.376
    assert entry["passed_r_avg"] == 0.962
    assert entry["r_difference"] == -1.338


def test_the_r_gap_needs_measured_R_on_BOTH_sides():
    """One side with a thin sample is not a disagreement, it is an absence."""
    thin = review_learning.MIN_OUTCOME_SAMPLES - 1
    assert find_callouts(_segment(-0.5, thin, 1.0, 24))[2] == []
    assert find_callouts(_segment(-0.5, 24, 1.0, thin))[2] == []
    assert find_callouts(_segment(None, 24, 1.0, 24))[2] == []


def test_a_small_r_gap_is_not_a_callout():
    below = review_learning.R_GAP_MIN_DIFFERENCE - 0.01
    assert find_callouts(_segment(0.0, 12, below, 12))[2] == []
    assert len(find_callouts(_segment(0.0, 12, review_learning.R_GAP_MIN_DIFFERENCE, 12))[2]) == 1


def test_r_gaps_are_ordered_by_the_widest_disagreement_either_way():
    aggregate = _segment(-0.376, 8, 0.962, 24)
    aggregate["dimensions"]["tier"] = {
        "B": {
            **aggregate["dimensions"]["bounce_type"]["lrsi_cross_20"],
            "taken": {"r_n": 10, "r_avg": 2.0, "fwd_n": 0, "fwd_avg_pct": None},
            "passed": {"r_n": 10, "r_avg": -1.0, "fwd_n": 0, "fwd_avg_pct": None},
        }
    }
    r_gaps = find_callouts(aggregate)[2]
    assert [entry["segment"] for entry in r_gaps] == ["B", "lrsi_cross_20"]


def test_the_report_renders_the_r_gap_section():
    aggregate = _segment(-0.376, 8, 0.962, 24)
    blind_spots, leaks, r_gaps = find_callouts(aggregate)
    report = render_report(
        {
            **aggregate,
            "generated_at": "now",
            "window_days": 90,
            "blind_spots": blind_spots,
            "leaks": leaks,
            "r_gaps": r_gaps,
        }
    )
    assert "R GAPS" in report
    assert "lrsi_cross_20" in report


# ---------------------------------------------------------------------------
# Chart Review veto codes reach the dislike_reason dimension
# ---------------------------------------------------------------------------
def _annotation(path, symbol, side, code, session_date="2026-07-27"):
    import json as _json

    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(
            _json.dumps(
                {
                    "schema_version": 1,
                    "event_id": "a" * 32,
                    "event_type": "veto",
                    "symbol": symbol,
                    "side": side,
                    "timeframe": "D1",
                    "source": "chart_review",
                    "session_date": session_date,
                    "created_at": f"{session_date}T10:00:00+00:00",
                    "reason_code": code,
                    "vocab_version": 1,
                }
            )
            + "\n"
        )


def test_coded_vetoes_join_into_the_dislike_reason_dimension(tmp_path):
    """212 coded vetoes carry the most specific thing the trader ever
    says about a chart, and the `dislike_reason` dimension was fed only by the
    33 `dislike` review events. Measured before this was built: 202 of 212 join
    to an existing episode, 198 of those to a SHOWN one, side matching on 202
    of 202 with zero mismatches.

    Fail-before-fix: the function does not exist and the codes never appear.
    """
    log = tmp_path / "trader_annotations.jsonl"
    _annotation(log, "NVDA", "LONG", "too_extended_from_base")
    _annotation(log, "NVDA", "LONG", "compressed")  # two codes on one name
    _annotation(log, "GHOST", "LONG", "volume_dry")  # no episode: never invented

    episodes = build_episodes([_row("shown"), _row("skip", dwell_ms=800)])
    from review_learning import attach_annotation_veto_reasons

    matched = attach_annotation_veto_reasons(episodes, log)

    assert matched == 1
    assert episodes[0].dislike_reasons == "compressed;too_extended_from_base"
    segments = review_learning.DIMENSIONS["dislike_reason"](episodes[0])
    assert sorted(segments) == ["compressed", "too_extended_from_base"]


def test_a_coded_veto_never_changes_what_the_trader_did(tmp_path):
    """The annotation stream is analysis-only: it may add a segment label and
    must never re-resolve an episode. A name the trader ARMED and also vetoed
    on the chart stays a take; the code just labels it."""
    log = tmp_path / "trader_annotations.jsonl"
    _annotation(log, "NVDA", "LONG", "too_extended_from_base")

    from review_learning import attach_annotation_veto_reasons

    episodes = build_episodes([_row("shown"), _row("arm_watch")])
    assert episodes[0].resolution == "take"
    attach_annotation_veto_reasons(episodes, log)
    assert episodes[0].resolution == "take"
    assert "too_extended_from_base" in episodes[0].dislike_reasons


def test_a_veto_on_the_other_side_is_skipped_rather_than_guessed(tmp_path):
    """Two directional claims on one name in one day. Nothing here can say
    which chart the veto was about, so it is not attached to either."""
    log = tmp_path / "trader_annotations.jsonl"
    _annotation(log, "NVDA", "SHORT", "too_extended_from_base")

    from review_learning import attach_annotation_veto_reasons

    episodes = build_episodes([_row("shown"), _row("skip")])  # side LONG
    assert attach_annotation_veto_reasons(episodes, log) == 0
    assert episodes[0].dislike_reasons == ""


def test_an_unreadable_annotation_log_is_a_quieter_board_not_a_failure(tmp_path):
    from review_learning import attach_annotation_veto_reasons

    episodes = build_episodes([_row("shown")])
    assert attach_annotation_veto_reasons(episodes, tmp_path / "missing.jsonl") == 0
    assert episodes[0].dislike_reasons == ""


def test_the_r_gap_class_annotates_and_never_reaches_review_policy():
    """Callouts annotate only, and this one does not even do that: it is a
    field on the scoreboard state and a section in the report. It is
    deliberately NOT wired into `draft_policy_from_state`, `review_guidance`
    or the AI evidence package, because those write priority deltas into
    `review_policy.json` and P1 may not reach that file.
    """
    from review_policy import draft_policy_from_state

    aggregate = _segment(-0.376, 8, 0.962, 24)
    blind_spots, leaks, r_gaps = find_callouts(aggregate)
    assert r_gaps, "the fixture must produce one, or this proves nothing"

    state = {"blind_spots": blind_spots, "leaks": leaks, "r_gaps": r_gaps}
    assert draft_policy_from_state(state) == []

    import inspect

    import review_guidance
    import review_policy
    from ai_jobs import policy_draft

    for module in (review_policy, review_guidance, policy_draft):
        assert "r_gaps" not in inspect.getsource(module), (
            f"{module.__name__} must not read the r_gap callouts"
        )
