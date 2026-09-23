"""Day Recap coach: find it next time, and the session's environment (trader, 2026-09-23).

Pure-function tests on fixture rows: unknown handling, small-sample flags, the
point-in-time rule (a trait or label recorded after the pick is never used) and
the environment-label splits.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import recap_findability as rf  # noqa: E402

SESSION = "2026-09-22"


def _et(clock: str, day: str = SESSION) -> str:
    return f"{day}T{clock}-04:00"


def _d1_row(session, label, written_at, **extra):
    return {"session": session, "benchmark": "SPY", "label": label, "rule_version": "d1_environment_v1",
            "written_at": written_at, "range_atr": 3.5, "slope_atr": -0.6, "reason": "", **extra}


def _shift(to, at, frm="bullish_strong", session=SESSION, source="auto"):
    return {"session_date": session, "event_at": at, "to_regime": to, "from_regime": frm, "source": source,
            "detail": rf.INTRADAY_ENV_WORDS.get(to, ""), "spy_day_pct": None}


def _tracker(symbol, side, family, bucket, at, event_type="initial", status="OPEN"):
    return {"symbol": symbol, "side": side, "event_type": event_type, "event_at": at,
            "setup_id": f"2026-09-21:{symbol}:{side}:x:{bucket}", "scan_date": "2026-09-21",
            "state_setup_family": family, "state_priority_bucket": bucket, "state_setup_status": status}


def _inputs(**extra):
    base = {"session": SESSION, "local_tz": "America/New_York", "window_sessions": [SESSION]}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# point in time
# ---------------------------------------------------------------------------
def test_d1_label_written_after_the_pick_is_never_used():
    rows = [
        _d1_row("2026-09-18", "trending_down", _et("11:05", "2026-09-21")),
        _d1_row("2026-09-21", "mixed", _et("11:13")),
    ]
    before = rf.d1_env_at(rows, rf.moment(_et("10:00")))
    after = rf.d1_env_at(rows, rf.moment(_et("11:30")))
    assert before["label"] == "trending_down" and before["session"] == "2026-09-18"
    assert after["label"] == "mixed"
    assert rf.d1_env_at(rows, rf.moment(_et("10:00", "2026-09-20")))["label"] == rf.UNKNOWN


def test_intraday_regime_uses_only_shifts_recorded_by_the_pick():
    shifts = [
        _shift("bullish_weak", "2026-09-22T13:57:26+00:00"),   # 09:57 ET
        _shift("bearish_strong", "2026-09-22T15:00:28+00:00", frm="bullish_weak"),  # 11:00 ET
        _shift("bearish_weak", "2026-09-21T15:00:00+00:00", session="2026-09-21"),
    ]
    at = rf.moment(_et("10:30"))
    assert rf.intraday_env_at(shifts, SESSION, at)["label"] == "bullish_weak"
    assert rf.intraday_env_at(shifts, SESSION, rf.moment(_et("11:30")))["label"] == "bearish_strong"
    # before the first shift: its from_regime was recorded later, so unknown
    early = rf.intraday_env_at(shifts, SESSION, rf.moment(_et("09:40")))
    assert early["label"] == rf.UNKNOWN


def test_opening_environment_counts_only_when_recorded_before_the_pick():
    opening = {"date": SESSION, "env": "bearish_weak", "recorded_at": "2026-09-22T07:07:18"}
    tz = rf.ZoneInfo("America/Los_Angeles")
    got = rf.intraday_env_at([], SESSION, rf.moment(_et("10:30")), opening=opening, local_tz=tz)
    assert got["label"] == "bearish_weak" and got["basis"] == "opening_environment"
    # 07:07 PT is 10:07 ET: a 09:45 ET pick came before it
    assert rf.intraday_env_at([], SESSION, rf.moment(_et("09:45")), opening=opening, local_tz=tz)["label"] == rf.UNKNOWN
    other_day = dict(opening, date="2026-09-21")
    assert rf.intraday_env_at([], SESSION, rf.moment(_et("12:00")), opening=other_day)["label"] == rf.UNKNOWN


def test_traits_recorded_after_the_pick_are_unknown():
    pick = {"symbol": "ABC", "side": "LONG", "timeframe": "D1", "pick_at": _et("10:00"), "category": rf.REAL_MISS}
    inputs = _inputs(
        tracker_events=[_tracker("ABC", "LONG", "avwape_to_1stdev", "near_favorite_zone", "2026-09-22T14:30:00+00:00")],
        m5_alerts=[{"trade_date": SESSION, "time_local": "10:15:00", "symbol": "ABC", "direction": "long",
                    "bounce_types": "vwap", "tier": "A"}],
        d1_environment=[_d1_row("2026-09-21", "mixed", _et("11:13"))],
    )
    traits = rf.traits_for(pick, inputs)
    assert traits["family"] == rf.UNKNOWN  # tracker row at 10:30 ET
    assert traits["alert_kind"] == rf.UNKNOWN  # alert at 10:15 ET
    assert traits["env_d1"]["label"] == rf.UNKNOWN  # label written 11:13 ET
    later = rf.traits_for(dict(pick, pick_at=_et("11:30")), inputs)
    assert later["family"] == "avwape_to_1stdev" and later["bucket"] == "near_favorite_zone"
    assert later["alert_kind"] == "vwap" and later["alert_tier"] == "A"
    assert later["env_d1"]["label"] == "mixed"


def test_missing_traits_read_unknown_never_blank_or_zero():
    traits = rf.traits_for({"symbol": "ZZZ", "side": "SHORT", "timeframe": "M5", "pick_at": ""}, _inputs())
    for key in ("family", "bucket", "grade", "alert_kind", "alert_tier", "sector", "rs_signal", "time_of_day"):
        assert traits[key] == rf.UNKNOWN, key
    assert traits["rs_vs_spy"] is None
    assert traits["env_d1"]["label"] == rf.UNKNOWN and traits["env_intraday"]["label"] == rf.UNKNOWN


def test_swing_grade_counts_only_from_a_grades_file_built_before_the_session():
    grades = {"as_of": SESSION, "swing": [{"key": "LONG|near_favorite_zone|avwape_to_1stdev", "grade": "A"}]}
    same_day = rf.swing_grade_at(grades, "LONG", "near_favorite_zone", "avwape_to_1stdev", SESSION)
    assert same_day["grade"] == rf.UNKNOWN and same_day["grade_now"] == "A"
    earlier = rf.swing_grade_at(dict(grades, as_of="2026-09-21"), "LONG", "near_favorite_zone",
                                "avwape_to_1stdev", SESSION)
    assert earlier["grade"] == "A"
    assert rf.swing_grade_at(grades, "LONG", rf.UNKNOWN, rf.UNKNOWN, SESSION)["grade_now"] == rf.UNKNOWN


def _events(n_win, n_loss, day, kind="vwap", side="LONG", env="bullish_weak"):
    out = []
    for i in range(n_win + n_loss):
        out.append({"event_id": f"X{i}_{day}", "trade_date": day, "symbol": f"S{i}", "side": side,
                    "bounce_type": kind, "entry_at": f"{day}T10:{i % 60:02d}:00", "env_intraday": env,
                    "sector": "Technology", "rs_signal": "RS", "rrs_spy": 1.0,
                    "result": "win" if i < n_win else "loss", "close_r": 1.0 if i < n_win else -1.0})
    return out


def test_m5_grade_uses_only_outcomes_from_before_the_session():
    earlier = []
    for day in [f"2026-09-{d:02d}" for d in range(1, 12)]:
        earlier += _events(3, 0, day)
    today = _events(0, 40, SESSION)
    before = rf.m5_grade_at(earlier, "vwap", "LONG", SESSION)
    with_today = rf.m5_grade_at(earlier + today, "vwap", "LONG", SESSION)
    assert before["grade"] == with_today["grade"]
    assert before["grade"] != rf.UNKNOWN
    assert rf.m5_grade_at(today, "vwap", "LONG", SESSION)["grade"] == rf.UNKNOWN


# ---------------------------------------------------------------------------
# cohorts
# ---------------------------------------------------------------------------
def test_small_cohort_says_too_few_to_tell():
    obs = rf.m5_observations(_events(5, 4, SESSION), _inputs())
    stats = rf.cohort(obs)
    assert stats["n"] == 9 and stats["too_few"] is True
    assert stats["words"].startswith("too few to tell")
    big = rf.cohort(rf.m5_observations(_events(7, 3, SESSION), _inputs()))
    assert big["too_few"] is False and big["hit_rate"] == 0.7 and big["median"] == 1.0
    assert big["words"] == "7 of 10 worked (70%), median +1.00R"
    assert rf.cohort([])["words"] == "no measured rows lately"


def test_cohorts_split_by_environment_label():
    rows = _events(8, 2, SESSION, env="bullish_weak") + _events(1, 4, SESSION, env="bearish_strong")
    for i, row in enumerate(rows):
        row["event_id"] = f"E{i}"
    stats = rf.cohort(rf.m5_observations(rows, _inputs()))
    split = stats["by_env_intraday"]
    assert split["bullish_weak"]["n"] == 10 and split["bullish_weak"]["hits"] == 8
    assert split["bearish_strong"]["n"] == 5 and split["bearish_strong"]["too_few"] is True


def test_undecided_and_out_of_window_m5_rows_are_not_counted():
    rows = _events(2, 0, SESSION) + _events(2, 0, "2026-08-01")
    rows[0]["result"] = ""
    obs = rf.m5_observations(rows, _inputs())
    assert len(obs) == 1


def test_d1_observations_keep_measured_horizon_rows_and_label_them_point_in_time():
    rows = [
        {"observation_id": "a", "scan_date": "2026-09-15", "symbol": "AAA", "side": "LONG", "horizon_sessions": "5",
         "measured": "True", "favorable": "True", "side_return_pct": "2.5", "setup_family": "fam",
         "priority_bucket": "favorite_setup"},
        {"observation_id": "b", "scan_date": "2026-09-15", "symbol": "BBB", "side": "LONG", "horizon_sessions": "1",
         "measured": "True", "favorable": "True", "side_return_pct": "1"},
        {"observation_id": "c", "scan_date": "2026-09-15", "symbol": "CCC", "side": "LONG", "horizon_sessions": "5",
         "measured": "False", "favorable": "", "side_return_pct": ""},
    ]
    env = [_d1_row("2026-09-14", "compressed", _et("10:58", "2026-09-15")),
           _d1_row("2026-09-15", "trending_up", _et("10:26", "2026-09-16"))]
    obs = rf.d1_observations(rows, _inputs(window_sessions=["2026-09-15"], d1_environment=env,
                                           sectors={"AAA": "Technology"}))
    assert len(obs) == 1
    assert obs[0]["env_d1"] == "compressed"  # the 09-15 label was written the next day
    assert obs[0]["sector"] == "Technology" and obs[0]["hit"] is True and obs[0]["value"] == 2.5


# ---------------------------------------------------------------------------
# find it next time
# ---------------------------------------------------------------------------
def _session_inputs():
    events = []
    for day in [f"2026-09-{d:02d}" for d in range(8, 23)]:
        events += _events(2, 1, day)
    for i, row in enumerate(events):
        row["event_id"] = f"E{i}"
    return _inputs(
        window_sessions=[f"2026-09-{d:02d}" for d in range(8, 23)],
        m5_events=events,
        m5_alerts=[{"trade_date": SESSION, "time_local": "10:05:00", "symbol": "WIN", "direction": "long",
                    "bounce_types": "vwap", "tier": "A"}],
        regime_shifts=[_shift("bullish_weak", "2026-09-22T13:57:26+00:00"),
                       _shift("bearish_weak", "2026-09-23T03:39:22+00:00")],
        d1_environment=[_d1_row("2026-09-18", "trending_down", _et("11:05", "2026-09-21")),
                        _d1_row(SESSION, "mixed", _et("11:00", "2026-09-23"))],
        focus_events=[{"symbol": "WIN", "side": "long", "event_type": "joined", "category": "m5",
                       "event_at": "2026-09-22T13:40:00+00:00"}],
        review_events=[{"symbol": "WIN", "trade_date": SESSION, "action": "shown", "ts": "2026-09-22T10:06:00"}],
        sectors={"WIN": "Technology"},
        notable=[
            {"symbol": "WIN", "side": "LONG", "category": rf.PASS_RAN, "timeframe": "M5",
             "pick_at": _et("10:10"), "verdict": "run", "ran_after_pct": 4.2, "source": "rejected",
             "what_you_did": "not today"},
            {"symbol": "MEH", "side": "SHORT", "category": rf.GOOD_PASS, "timeframe": "D1",
             "pick_at": _et("10:10"), "verdict": "no_run", "source": "rejected"},
        ],
    )


def test_a_pass_that_ran_gets_a_recipe_stats_and_a_place_to_look():
    out = rf.findability_for_session(SESSION, _session_inputs())
    win = next(p for p in out["picks"] if p["symbol"] == "WIN")
    # grade B: 28 of 42 earlier-session brackets won (Wilson low bound 0.52)
    assert win["recipe"] == "M5 vwap · long · grade B · trend-down market · Bullish Weak tape · first hour · Technology"
    core = win["recipe_lately"]["core"]
    assert core["traits"] == {"alert_kind": "vwap", "side": "LONG"}
    assert core["n"] == 45 and core["hits"] == 30
    assert win["recipe_lately"]["core_in_this_environment"]["environment"] == "bullish_weak"
    assert any("Min tier = 'A tier and above'" in line for line in win["where_to_look"])
    assert win["surfaced"]["first_at"] == "2026-09-22T09:40:00-04:00"  # Focus before the alert
    assert win["surfaced"]["before_pick"] is True
    assert win["saw_it"]["saw"] == "yes" and win["saw_it"]["review_actions"] == ["shown"]
    meh = next(p for p in out["picks"] if p["symbol"] == "MEH")
    assert "recipe" not in meh
    assert meh["surfaced"]["words"] == "the desk has no record of surfacing it"


def test_surfaced_prefers_the_focus_add_that_came_before_the_pick():
    pick = {"symbol": "OLD", "side": "LONG", "timeframe": "D1", "pick_at": _et("11:00")}
    joins = [
        {"symbol": "OLD", "side": "long", "event_type": "joined", "category": "swing",
         "event_at": "2026-09-18T20:30:00+00:00"},
        {"symbol": "OLD", "side": "long", "event_type": "joined", "category": "swing",
         "event_at": "2026-09-22T17:00:00+00:00"},
    ]
    got = rf.surfaced(pick, {}, _inputs(focus_events=joins))
    assert got["first_at"] == "2026-09-18T16:30:00-04:00"
    assert got["before_pick"] is True and got["before_open"] is True
    only_later = rf.surfaced(pick, {}, _inputs(focus_events=joins[1:]))
    assert only_later["first_at"] == "2026-09-22T13:00:00-04:00" and only_later["before_pick"] is False


def test_d1_where_to_look_names_the_bucket_filters_own_label():
    lines = rf.where_to_look({"timeframe": "D1", "side": "SHORT", "bucket": "near_favorite_zone",
                              "sector": "Energy", "grade_now": "A"}, {"surfaces": []})
    assert lines[0] == ("Master AVWAP page (Setup Tracker scan): sort by the Grade badge (best first), "
                        "Side = SHORT, Bucket = Near, type 'Energy' in 'Filter symbol, tag, level'")
    assert "Today this family grades A on that page." in lines


def test_recipe_leaves_out_unknown_traits():
    parts = rf.recipe_parts({"timeframe": "D1", "family": rf.UNKNOWN, "claim_setup": "trendline_break",
                             "side": "SHORT", "grade": rf.UNKNOWN, "env_d1": {"label": rf.UNKNOWN},
                             "env_intraday": {"label": rf.UNKNOWN}, "time_of_day": rf.UNKNOWN,
                             "sector": rf.UNKNOWN, "rs_signal": rf.UNKNOWN})
    assert " · ".join(w for _t, _v, w in parts) == "D1 trendline break · short"


def test_outputs_are_json_serializable():
    inputs = _session_inputs()
    json.dumps(rf.findability_for_session(SESSION, inputs))
    json.dumps(rf.environment_summary(SESSION, inputs))


# ---------------------------------------------------------------------------
# environment clarity
# ---------------------------------------------------------------------------
def test_environment_summary_lists_every_label_in_time_order():
    out = rf.environment_summary(SESSION, _session_inputs())
    kinds = [(t["kind"], t["label"]) for t in out["timeline"]]
    assert kinds == [("d1", "trending_down"), ("intraday", "bullish_weak"),
                     ("intraday", "bearish_weak"), ("d1_after", "mixed")]
    assert out["timeline"][2]["after_close"] is True
    assert out["d1_known_at_open"]["session"] == "2026-09-18"
    assert out["main_intraday_label"] == "bullish_weak"
    assert out["intraday_minutes_by_label"] == {"bullish_weak": 363}
    m5 = out["lately_in_this_environment"]["m5"]
    assert m5["environment"] == "bullish_weak" and m5["by_alert_kind"][0]["value"] == "vwap"
    d1 = out["lately_in_this_environment"]["d1"]
    assert d1["by_family"] == [] and "no measured rows" in d1["why"]


def test_environment_summary_with_nothing_recorded_says_unknown():
    out = rf.environment_summary(SESSION, _inputs())
    assert out["timeline"] == [] and out["main_intraday_label"] == rf.UNKNOWN
    assert out["d1_for_the_session"]["label"] == rf.UNKNOWN
    assert out["lately_in_this_environment"]["m5"]["why"].startswith("the environment is unknown")


def test_vocabulary_matches_the_desks_own_labels():
    from indicators.d1_environment import LABELS

    vocab = rf.environment_vocabulary()
    assert {row["label"] for row in vocab["d1"]} == set(LABELS)
    from bounce_bot_lib.config import MARKET_ENVIRONMENTS

    assert {row["label"] for row in vocab["intraday"]} == set(MARKET_ENVIRONMENTS) | {rf.UNKNOWN}


# ---------------------------------------------------------------------------
# readers of already-read rows
# ---------------------------------------------------------------------------
def _row(symbol, side, verdict, tf="D1", category="chart_review", clock="10:00", traded="no"):
    return SimpleNamespace(
        symbol=symbol, side=side, real_miss=verdict, traded=traded, category=category,
        decision_id=(SESSION, symbol, side, "chart_review", "like", tf, _et(clock)),
        time=datetime.fromisoformat(_et(clock)), ran_after_pct=3.0, state="measured",
        what_you_did="x", reason="",
    )


def test_notable_names_are_sorted_into_the_five_categories():
    walkaway = SimpleNamespace(
        liked_not_traded=(_row("LIK", "LONG", "run"),),
        claimed_d1=(_row("CLM", "SHORT", "unmeasured:horizon_open", category="trendline_break"),),
        rejected=(_row("PRN", "LONG", "run", tf="M5"), _row("GPF", "LONG", "no_run"),
                  _row("UNM", "LONG", "unmeasured:atr_unreadable")),
    )
    payload = {"session_date": SESSION, "walkaway": walkaway, "trades": [
        {"symbol": "TRD", "direction": "LONG", "net_pnl_usd": 50.0, "opened_at": _et("09:45"),
         "closed_at": _et("11:00"), "auto_tag_summary": "day_trade"},
        {"symbol": "OLD", "direction": "LONG", "net_pnl_usd": 10.0, "opened_at": _et("09:45", "2026-08-01"),
         "closed_at": _et("11:00"), "auto_tag_summary": "swing"},
        {"symbol": "LOS", "direction": "LONG", "net_pnl_usd": -5.0, "opened_at": _et("09:45"),
         "closed_at": _et("11:00")},
    ]}
    got = {row["symbol"]: row for row in rf.notable_from_payload(payload)}
    assert {s: r["category"] for s, r in got.items()} == {
        "TRD": rf.GREAT_TRADE, "OLD": rf.GREAT_TRADE, "LIK": rf.REAL_MISS, "CLM": rf.LIKED,
        "PRN": rf.PASS_RAN, "GPF": rf.GOOD_PASS,
    }
    assert got["TRD"]["timeframe"] == "M5" and got["PRN"]["timeframe"] == "M5"
    assert got["CLM"]["family"] == "trendline_break"


def test_m5_event_summary_reads_context_and_bracket_result():
    ctx = json.dumps({"market_environment": "bearish_weak", "sector": "Energy", "rrs_spy": -2.5,
                      "rrs_spy_signal": "RW"})
    rows = [
        {"event_id": "XOM_short_20260922_10_00_00_vwap", "event_type": "registered", "trade_date": SESSION,
         "symbol": "XOM", "direction": "short", "entry_time": "2026-09-22T10:00:00", "bars_elapsed": "0",
         "target_1r_hit": "False", "stop_hit": "False", "context_json": ctx},
        {"event_id": "XOM_short_20260922_10_00_00_vwap", "event_type": "final", "trade_date": SESSION,
         "symbol": "XOM", "direction": "short", "entry_time": "2026-09-22T10:00:00", "bars_elapsed": "12",
         "target_1r_hit": "True", "stop_hit": "False", "close_r": "1.8", "context_json": ""},
    ]
    (event,) = rf.m5_event_summaries(rows)
    assert event["side"] == "SHORT" and event["bounce_type"] == "vwap" and event["result"] == "win"
    assert event["env_intraday"] == "bearish_weak" and event["rs_signal"] == "RW" and event["close_r"] == 1.8
    assert event["sector"] == "Energy"


def test_time_of_day_buckets_are_new_york_time():
    assert rf.tod_bucket(rf.moment("2026-09-22T06:45:00-07:00")) == "first_hour"
    assert rf.tod_bucket(rf.moment(_et("15:59"))) == "last_hour"
    assert rf.tod_bucket(rf.moment(_et("16:00"))) == "after_close"
    assert rf.tod_bucket(None) == rf.UNKNOWN
    assert rf.moment("2026-09-22T10:00:00", rf.ZoneInfo("America/Los_Angeles")).utcoffset().total_seconds() == -7 * 3600
    assert rf.moment("nonsense") is None
    assert datetime(2026, 9, 22, tzinfo=timezone.utc)


def test_swing_grade_at_the_pick_comes_from_the_grade_history_written_before_it():
    now_file = {"as_of": SESSION, "swing": [{"key": "LONG|near_favorite_zone|avwape_to_1stdev", "grade": "A"}]}
    history = [
        (datetime(2026, 9, 22, 13, 0, tzinfo=timezone.utc),
         {"swing": [{"key": "LONG|near_favorite_zone|avwape_to_1stdev", "grade": "C"}], "written_at": "09:00"}),
        (datetime(2026, 9, 22, 15, 0, tzinfo=timezone.utc),
         {"swing": [{"key": "LONG|near_favorite_zone|avwape_to_1stdev", "grade": "B"}], "written_at": "11:00"}),
    ]

    def as_of(when):
        written = [payload for stamp, payload in history if stamp <= when]
        return written[-1] if written else None

    pick = {"symbol": "ABC", "side": "LONG", "timeframe": "D1", "pick_at": _et("10:00"), "category": rf.REAL_MISS}
    inputs = _inputs(
        tracker_events=[_tracker("ABC", "LONG", "avwape_to_1stdev", "near_favorite_zone", _et("08:00"))],
        grades_now=now_file, grades_as_of=as_of,
    )
    traits = rf.traits_for(pick, inputs)
    assert traits["grade"] == "C" and traits["grade_now"] == "A"  # the 11:00 snapshot came after the pick
    early = rf.traits_for(dict(pick, pick_at=_et("08:30")), inputs)
    assert early["grade"] == rf.UNKNOWN  # nothing written yet at 08:30
