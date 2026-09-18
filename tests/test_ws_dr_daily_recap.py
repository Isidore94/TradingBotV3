"""Packet WS-DR - the Daily Recap reads the day from the STORES, not the process.

Written RED by the tester before any fix (2026-09-13). A builder may ADD to this
file; it may not weaken, skip or delete anything in it.

What this file pins, and why each shape is the real one:

* `scripts/daily_recap_reader.py` does not exist yet, so every test here fails on
  the import. That is the point: the contract below is what makes them pass.
* Every fixture is written in the LIVE column order and the LIVE row shape, read
  off the desk's stores on 2026-09-13 and reproduced by hand here:
  `intraday_bounce_outcomes.csv` names its side column `direction` (not `side`)
  and already stores `mfe_pct` / `eod_move_pct` SIDE-ADJUSTED; `pick_feedback`
  and `alert_review_events` rows carry a NAIVE `ts` (0 of 1,377 pick-feedback
  rows on the desk are offset-aware, so the packet's "`ts` aware" premise is
  wrong and the reader has to attach a zone); a pass sidecar's bar `dt` is naive
  on 621 of the desk's 622 annotation rows; an old outcome column is PRESENT AND
  EMPTY, never absent.
* Numbers, not shapes. A short whose close is 5% BELOW its entry is a +5.00
  favorable move and a long with the same close is -5.00, so a reader that
  recomputed the move without the side sorts them into the same place and fails.
* A decision's credit starts at its own timestamp. The SHW pass at 12:30 is
  measured from the 12:30 bar (+2.00), never from the 06:35 bar's 323.00 low
  (+5.00), and the MU like at 12:30 - which has no bar series behind it at all -
  is `unavailable`, never MU's 8.00 day MFE and never 0.0.

The reader's seam. `read_session` is pure and takes its inputs as PATHS
(`RecapSources`), because a session read has to be reproducible from files alone
- that is the whole difference from today's `_feed_away_recap`, which hands the
recap `center._alerts`, a process-scoped list capped at 250 / 100 items. The
durability test below re-reads the same fixture in a SEPARATE INTERPRETER and
requires the same answer.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from dataclasses import fields as dataclass_fields
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# The sessions the fixture is built on. Real NYSE sessions, resolved through the
# real calendar so a holiday can never be assumed away: 2026-09-07 is Labor Day,
# which is exactly why the third prior session of 2026-09-10 is 09-04.
# ---------------------------------------------------------------------------

SESSION = "2026-09-10"
PRIOR_1 = "2026-09-09"
PRIOR_2 = "2026-09-08"
PRIOR_3 = "2026-09-04"
#: `now` for every read: Friday morning, mid-session. The latest COMPLETED
#: session is therefore 2026-09-10 and "today" (2026-09-11) is provisional.
NOW = datetime(2026, 9, 11, 7, 30)
TODAY = "2026-09-11"

#: The four views, in the packet's order.
VIEW_NAMES = ("worked_today", "recent_swings", "my_decisions", "rejected_that_worked")

#: Every durable store the packet names. The reader must declare one coverage
#: entry per source; it may declare MORE, never fewer.
REQUIRED_SOURCES = (
    "intraday_outcomes",
    "tier_outcomes",
    "session_horizon_outcomes",
    "annotations",
    "pick_feedback",
    "swing_favorites",
    "human_focus_outcomes",
    "review_events",
    "preference_report",
    "staged_picks",
    "environment_labels",
    "working_lately",
)

#: A sort key that would be a RANKING BY RESULT rather than a declared measure.
#: The same refusal gate #43 puts on the narration view: a recap orders by what
#: it measured, never by a score it made up.
FORBIDDEN_SORT_KEYS = (
    "win_rate",
    "mean_r",
    "profit_factor",
    "expectancy",
    "score",
    "points",
    "priority_score",
    "held_run_score",
    "rank",
)


def test_the_calendar_still_says_what_this_fixture_assumes():
    """Guard, not a packet item: the fixture's session arithmetic is real."""
    import market_calendar

    day = date.fromisoformat(SESSION)
    assert market_calendar.is_session(day)
    assert market_calendar.previous_session(day).isoformat() == PRIOR_1
    assert market_calendar.previous_session(date.fromisoformat(PRIOR_1)).isoformat() == PRIOR_2
    assert market_calendar.previous_session(date.fromisoformat(PRIOR_2)).isoformat() == PRIOR_3
    assert market_calendar.last_completed_session(NOW).isoformat() == SESSION


# ---------------------------------------------------------------------------
# The fixture stores, in their live column order
# ---------------------------------------------------------------------------

def _csv_text(header: str, rows) -> str:
    """Write with the csv module, so an embedded `context_json` comma is quoted
    exactly the way the live writer quotes it."""
    columns = header.split(",")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        assert len(row) == len(columns), (len(row), len(columns))
        writer.writerow(row)
    return buffer.getvalue()


INTRADAY_HEADER = (
    "schema_version,event_id,event_type,logged_at,trade_date,symbol,direction,"
    "entry_time,entry_price,stop_price,risk_per_share,bars_elapsed,minutes_elapsed,"
    "close_r,mfe_r,mae_r,best_price,worst_price,target_1r_hit,target_2r_hit,stop_hit,"
    "status,milestone_bar,context_json,outcome_mode,eod_close,eod_move_pct,mfe_pct,mae_pct"
)

#: (event_id, trade_date, symbol, direction, entry_time, entry_price, status,
#:  eod_close, eod_move_pct, mfe_pct, mae_pct)
INTRADAY_ROWS = (
    ("NVDA_long_20260910_09_40_00_ema_15", SESSION, "NVDA", "long", "09:40:00",
     "100.00", "closed", "95.00", "-5.00", "1.20", "-6.00"),
    ("TSLA_short_20260910_09_45_00_ema_15", SESSION, "TSLA", "short", "09:45:00",
     "100.00", "closed", "95.00", "5.00", "6.00", "-0.80"),
    # An OPEN row: the measured columns are present and EMPTY, which is what an
    # unfinalized row looks like on disk. Unmeasured is counted and shown, never
    # assumed to be zero.
    ("AMD_long_20260910_10_05_00_ema_15", SESSION, "AMD", "long", "10:05:00",
     "50.00", "open", "", "", "", ""),
    # The prior session. It must never leak into "what worked today".
    ("NFLX_long_20260909_09_40_00_ema_15", PRIOR_1, "NFLX", "long", "09:40:00",
     "700.00", "closed", "707.00", "1.00", "2.00", "-1.00"),
    ("SHW_short_20260910_06_35_00_ema_15", SESSION, "SHW", "short", "06:35:00",
     "340.00", "closed", "336.60", "1.00", "5.00", "-0.30"),
    ("MU_long_20260910_09_40_00_ema_15", SESSION, "MU", "long", "09:40:00",
     "200.00", "closed", "201.00", "0.50", "8.00", "-1.00"),
    ("KBR_short_20260910_09_35_00_ema_15", SESSION, "KBR", "short", "09:35:00",
     "80.00", "closed", "77.60", "3.00", "4.00", "-1.50"),
    ("STT_long_20260910_09_50_00_ema_15", SESSION, "STT", "long", "09:50:00",
     "90.00", "closed", "88.20", "-2.00", "0.00", "-3.00"),
    ("AAPL_long_20260910_07_00_00_ema_15", SESSION, "AAPL", "long", "07:00:00",
     "220.00", "closed", "224.40", "2.00", "3.50", "-0.50"),
)


def _intraday_csv() -> str:
    context = json.dumps(
        {"market_environment": "bullish_strong", "sector": "Technology"}, sort_keys=True
    )
    rows = []
    for (event_id, trade_date, symbol, direction, entry_time, entry_price,
         status, eod_close, eod_move, mfe_pct, mae_pct) in INTRADAY_ROWS:
        rows.append(
            [
                "1", event_id, "registered", f"{trade_date}T13:10:00-07:00", trade_date,
                symbol, direction, f"{trade_date}T{entry_time}", entry_price, "", "",
                "", "", "", "", "", "", "", "False", "False", "False", status, "",
                context, "", eod_close, eod_move, mfe_pct, mae_pct,
            ]
        )
    return _csv_text(INTRADAY_HEADER, rows)


SESSION_HORIZON_HEADER = (
    "observation_id,scan_row_id,symbol,side,scan_date,target_session,horizon_sessions,"
    "sessions_spanned,entry_close,entry_close_source,target_close,side_return_pct,"
    "favorable,measured,maturity,unmeasured_reason,outcome_kind,knowledge_basis,tier,"
    "tier_source,priority_bucket,setup_family,favorite_zone,collapsed_same_session"
)

#: (symbol, side, scan_date, target_session, horizon, side_return_pct,
#:  favorable, measured, maturity)
SESSION_HORIZON_ROWS = (
    ("AAPL", "LONG", PRIOR_1, SESSION, "1", "2.00", "True", "True", "mature"),
    ("AAPL", "LONG", PRIOR_1, "2026-09-14", "3", "6.00", "True", "True", "mature"),
    ("ORCL", "SHORT", PRIOR_2, PRIOR_1, "1", "1.25", "True", "True", "mature"),
    ("ORCL", "SHORT", PRIOR_2, "2026-09-11", "3", "3.00", "True", "True", "mature"),
    ("MSFT", "SHORT", PRIOR_2, PRIOR_1, "1", "1.50", "True", "True", "mature"),
    # The selected 3-session end has NOT arrived. `measured` is present and
    # empty, `side_return_pct` is present and empty: pending, never zero.
    ("MSFT", "SHORT", PRIOR_2, "2026-09-11", "3", "", "", "", "immature"),
    # Outside the three-session lookback entirely.
    ("GOOG", "LONG", "2026-09-02", "2026-09-03", "1", "4.00", "True", "True", "mature"),
    ("GOOG", "LONG", "2026-09-02", "2026-09-08", "3", "9.00", "True", "True", "mature"),
)


def _session_horizon_csv() -> str:
    rows = []
    for (symbol, side, scan_date, target, horizon, ret, favorable,
         measured, maturity) in SESSION_HORIZON_ROWS:
        scan_row_id = f"{symbol}:{scan_date}:{scan_date}-130129"
        rows.append(
            [
                f"{scan_row_id}:{horizon}", scan_row_id, symbol, side, scan_date, target,
                horizon, horizon, "100.00", "session_bar", "", ret, favorable, measured,
                maturity, "" if measured else "horizon_not_reached",
                "favorable_direction_session_v2",
                "entry_session_close_to_target_session_close", "S",
                "derived_from_bucket", "favorite_setup", "avwap_band_bounce", "", "1",
            ]
        )
    return _csv_text(SESSION_HORIZON_HEADER, rows)


TIER_HEADER = (
    "observation_id,scan_row_id,run_id,run_timestamp,run_date,watchlist_label,scan_date,"
    "future_scan_date,horizon_sessions,tier,tier_source,symbol,side,priority_bucket,"
    "priority_score,setup_family,favorite_zone,entry_close,future_close,raw_return_pct,"
    "side_return_pct,win,spy_forward_return_pct,spy_relative_side_return_pct,"
    "sessions_spanned,stale_horizon,positive_scan_factor_match_count,"
    "positive_scan_factor_matches,outcome_kind"
)

TIER_ROWS = (("AAPL", "LONG", PRIOR_1), ("ORCL", "SHORT", PRIOR_2))


def _tier_csv() -> str:
    rows = []
    for symbol, side, scan_date in TIER_ROWS:
        scan_row_id = f"{symbol}:{scan_date}:{scan_date}-130129"
        rows.append(
            [
                f"{scan_row_id}:5", scan_row_id, f"{scan_date}-130129",
                f"{scan_date}T13:01:29", scan_date, "home folder watchlists", scan_date,
                "2026-09-17", "5", "S", "derived_from_bucket", symbol, side,
                "favorite_setup", "65.0", "avwap_breakout", "", "100.00", "104.00",
                "4.0", "4.0", "True", "0.5", "3.5", "5", "False", "0", "",
                "favorable_direction_scanrow_v1",
            ]
        )
    return _csv_text(TIER_HEADER, rows)


HUMAN_FOCUS_HEADER = (
    "trade_date,symbol,side,source,entry_date,entry_close,h1_date,h1_return,h3_date,"
    "h3_return,h5_date,h5_return,h10_date,h10_return,matured_horizons,fully_matured,"
    "updated_at"
)


def _human_focus_csv() -> str:
    rows = [
        [SESSION, symbol, side, source, SESSION, "220.0000", "2026-09-11", "0.016000",
         "", "", "", "", "", "", "1", "0", f"{TODAY}T07:35:18"]
        for symbol, side, source in (("AAPL", "LONG", "focus_swing_vetted"),
                                     ("FTNT", "LONG", "focus_swing_vetted"))
    ]
    return _csv_text(HUMAN_FOCUS_HEADER, rows)


def _annotation_rows(pass_created_at: str) -> tuple[dict, ...]:
    """The trader's own decisions, in the live `trader_annotations.jsonl` shape.

    `created_at` carries an explicit offset (as every real row does); the PASS
    row's stamp is built through `pass_bars.attach_desk_zone`, so the partition
    of its naive sidecar bars into before/after the decision is the same on any
    machine that runs this suite.
    """
    return (
        # The 12:30 claimed like, and the SAME opportunity clicked again at
        # 13:05. Two clicks, one decision.
        {
            "created_at": f"{SESSION}T12:30:00-07:00",
            "event_id": "c0de0000000000000000000000000001",
            "event_type": "like_claim", "like_mode": "claimed",
            "note": "held the 8ema all morning", "schema_version": 1,
            "session_date": SESSION, "side": "LONG", "source": "chart_review",
            "surface": "chart_review", "symbol": "MU", "timeframe": "D1",
        },
        {
            "created_at": f"{SESSION}T13:05:00-07:00",
            "event_id": "c0de0000000000000000000000000002",
            "event_type": "like_claim", "like_mode": "claimed", "note": "",
            "schema_version": 1, "session_date": SESSION, "side": "LONG",
            "source": "chart_review", "surface": "chart_review", "symbol": "MU",
            "timeframe": "D1",
        },
        # Alt+L. The quick like never prompts and names no setup.
        {
            "created_at": f"{SESSION}T11:00:00-07:00",
            "event_id": "c0de0000000000000000000000000003",
            "event_type": "like_claim", "like_mode": "quick", "schema_version": 1,
            "session_date": SESSION, "side": "LONG", "source": "chart_review",
            "surface": "chart_review", "symbol": "NVDA", "timeframe": "D1",
        },
        # A veto whose later path went the trader's way: it did not "work", so
        # it stays out of view 4.
        {
            "created_at": f"{SESSION}T09:50:00-07:00",
            "event_id": "be590000000000000000000000000001",
            "event_type": "veto", "reason_code": "compressed", "schema_version": 1,
            "session_date": SESSION, "side": "LONG", "source": "chart_review",
            "surface": "chart_review", "symbol": "STT", "timeframe": "D1",
            "vocab_version": 3,
        },
        # A veto whose later path WAS favorable. The reason travels with it, and
        # so does the adverse movement - a later rise alone does not prove a
        # timing veto wrong.
        {
            "created_at": f"{SESSION}T09:35:00-07:00",
            "event_id": "be590000000000000000000000000002",
            "event_type": "veto", "reason_code": "timing", "schema_version": 1,
            "session_date": SESSION, "side": "SHORT", "source": "chart_review",
            "surface": "chart_review", "symbol": "KBR", "timeframe": "D1",
            "vocab_version": 3,
        },
        # One day-trade PASS carrying TWO reason codes. The cohorts overlap and
        # are never summed: this is ONE decision.
        {
            "created_at": pass_created_at,
            "event_id": "aaaa0000000000000000000000000001",
            "event_type": "pass", "m5_bar_count": 5,
            "m5_bars_ref": "trader_annotation_bars/aaaa0000000000000000000000000001.json",
            "m5_first_bar": f"{SESSION}T06:35:00", "m5_last_bar": f"{SESSION}T12:55:00",
            "reason_codes": ["poor_market_conditions", "choppy_tape"],
            "schema_version": 1, "session_date": SESSION, "side": "SHORT",
            "source": "chart_review", "symbol": "SHW", "timeframe": "D1",
            "vocab_version": 1, "vocabulary_id": "pass_reasons",
        },
        # A note is a fact of its own and is neither a like nor a rejection.
        {
            "created_at": f"{SESSION}T10:40:00-07:00",
            "event_id": "0dde0000000000000000000000000001",
            "event_type": "note", "note": "watching the open range",
            "schema_version": 1, "session_date": SESSION, "side": "LONG",
            "source": "chart_review", "surface": "chart_review", "symbol": "NVDA",
            "timeframe": "D1",
        },
        # Another session entirely.
        {
            "created_at": f"{PRIOR_1}T10:00:00-07:00",
            "event_id": "be590000000000000000000000000003",
            "event_type": "veto", "reason_code": "extended", "schema_version": 1,
            "session_date": PRIOR_1, "side": "LONG", "source": "chart_review",
            "surface": "chart_review", "symbol": "NFLX", "timeframe": "D1",
            "vocab_version": 3,
        },
    )


#: The bars behind the SHW pass, in the live sidecar shape: `dt` is NAIVE, the
#: way 621 of the desk's 622 annotation sidecars are.
PASS_SIDECAR_BARS = (
    {"dt": f"{SESSION}T06:35:00", "open": 340.00, "high": 341.00, "low": 323.00,
     "close": 339.00, "volume": 1200.0},
    {"dt": f"{SESSION}T12:25:00", "open": 340.00, "high": 340.00, "low": 339.50,
     "close": 340.00, "volume": 300.0},
    {"dt": f"{SESSION}T12:30:00", "open": 340.00, "high": 340.50, "low": 336.00,
     "close": 337.00, "volume": 410.0},
    {"dt": f"{SESSION}T12:35:00", "open": 337.00, "high": 337.50, "low": 333.20,
     "close": 334.00, "volume": 520.0},
    {"dt": f"{SESSION}T12:55:00", "open": 334.00, "high": 336.00, "low": 334.00,
     "close": 336.60, "volume": 260.0},
)

#: The whole day's favorable excursion for the SHW short, from its 340.00 entry
#: to the 06:35 bar's 323.00 low. This is the number the trader must NOT be
#: credited with for a decision taken at 12:30.
SHW_DAY_MFE_PCT = 5.00
#: From the last completed bar at the decision (12:25, close 340.00 - which is
#: also the 12:30 open) to the post-decision low of 333.20.
SHW_AFTER_DECISION_MFE_PCT = 2.00

#: MU's whole-day favorable excursion. The 12:30 like has no bar series behind
#: it, so its credited number is `unavailable` - never this, never 0.0.
MU_DAY_MFE_PCT = 8.00


PICK_FEEDBACK_ROWS = (
    # `ts` is NAIVE, which is what every row on the desk looks like.
    {"category": "m5", "context": "", "origin": "auto_pick",
     "reason": "triple-VWAP invalidation", "side": "SHORT", "symbol": "KBR",
     "trade_date": SESSION, "ts": f"{SESSION}T12:51:40", "verdict": "not_today"},
    # Never graded, never a rejection.
    {"category": "swing", "context": "", "origin": "manual", "reason": "",
     "side": "LONG", "symbol": "MU", "trade_date": SESSION,
     "ts": f"{SESSION}T13:20:00", "verdict": "unfavorite"},
    {"category": "swing", "context": "", "origin": "vetted", "reason": "",
     "side": "LONG", "symbol": "AAPL", "trade_date": SESSION,
     "ts": f"{SESSION}T10:15:00", "verdict": "like"},
    {"category": "swing", "context": "", "origin": "manual",
     "reason": "too extended from the anchor", "side": "LONG", "symbol": "STT",
     "trade_date": SESSION, "ts": f"{SESSION}T09:55:00", "verdict": "dislike"},
    {"category": "m5", "context": "", "origin": "auto_pick", "reason": "",
     "side": "LONG", "symbol": "NFLX", "trade_date": PRIOR_1,
     "ts": f"{PRIOR_1}T11:00:00", "verdict": "not_today"},
)


SWING_FAVORITE_ROWS = (
    {"action": "add", "event_at": f"{SESSION}T08:13:02-07:00", "origin": "trader",
     "schema": "swing_favorite_v1", "session_date": SESSION, "side": "long",
     "symbol": "FTNT"},
    {"action": "add", "event_at": f"{SESSION}T12:27:31-07:00", "origin": "trader",
     "schema": "swing_favorite_v1", "session_date": SESSION, "side": "short",
     "symbol": "BXWT"},
    # The retraction. A removed favorite is not a favorite.
    {"action": "remove", "event_at": f"{SESSION}T14:02:10-07:00", "origin": "trader",
     "schema": "swing_favorite_v1", "session_date": SESSION, "side": "short",
     "symbol": "BXWT"},
    {"action": "add", "event_at": f"{PRIOR_1}T09:05:00-07:00", "origin": "trader",
     "schema": "swing_favorite_v1", "session_date": PRIOR_1, "side": "long",
     "symbol": "NFLX"},
)


REVIEW_EVENT_ROWS = (
    # A click away from an M5 alert IS a pass (trader, 2026-09-01). `ts` naive,
    # `detail` a nested dict, `event_id` at the outcome store's own grain.
    {"action": "skip", "banger": False, "bounce_types": "ema_15",
     "detail": {"reason": "clicked_away_from_m5_alert"}, "dwell_ms": 5195,
     "entry_price": 100.0, "event_id": "TSLA_short_20260910_09_45_00_ema_15",
     "is_d1": False, "is_focus_pick": True, "machine": "NucBox_K8_Plus",
     "proven": False, "queue_len": 4, "risk_per_share": 1.0,
     "schema": "review_events_v2", "score": 12.48, "side": "SHORT",
     "stop_price": 101.0, "symbol": "TSLA", "tag": "red", "tier": "A",
     "timeframe": "M5", "trade_date": SESSION, "trigger": "M5 move -1.44%",
     "ts": f"{SESSION}T09:50:11"},
    # An impression, not a decision.
    {"action": "shown", "banger": False, "bounce_types": "", "event_id": "",
     "is_d1": True, "is_focus_pick": False, "machine": "MainPC", "proven": False,
     "queue_len": 0, "schema": "review_events_v1", "score": None, "side": "LONG",
     "symbol": "CLMT", "tag": "d1_flag_long", "tier": "", "timeframe": "D1",
     "trade_date": SESSION, "trigger": "CLMT (long) zone1 1st-dev break",
     "ts": f"{SESSION}T10:23:28"},
)


ENVIRONMENT_ROWS = (
    {"session": SESSION, "benchmark": "SPY", "label": "bullish_strong",
     "rule_version": "d1_environment_v1", "range_atr": 1.2, "slope_atr": 0.4,
     "sma20": 640.0, "atr14": 6.0, "bars_used": 60, "reason": "",
     "bars_through": SESSION, "written_at": f"{SESSION}T13:05:00-07:00",
     "source": "scan"},
    {"session": PRIOR_1, "benchmark": "SPY", "label": "neutral",
     "rule_version": "d1_environment_v1", "range_atr": 0.6, "slope_atr": 0.0,
     "sma20": 638.0, "atr14": 6.0, "bars_used": 60, "reason": "",
     "bars_through": PRIOR_1, "written_at": f"{PRIOR_1}T13:05:00-07:00",
     "source": "scan"},
    # 2026-09-08 is deliberately UNLABELLED: `unknown` is its own cell.
)


def _preference_report_csv() -> str:
    import preference_trade_outcomes as pto

    matched = {
        "schema": pto.SCHEMA, "generated_at": f"{TODAY}T06:00:00-07:00",
        "session_date": SESSION, "symbol": "AAPL", "side": "LONG",
        "channel": "pick_feedback:like", "statement": "liked",
        "statement_detail": "vetted", "statement_id": "pf-aapl-1", "traded": "yes",
        "trade_id": "T-1", "trade_opened_at": f"{SESSION}T10:20:00-07:00",
        "match_confidence": "high", "match_basis": "same_session_symbol_side",
        "journal_r": "1.4", "journal_net_pnl": "312.50",
        "paper_forward_return_h3": "", "paper_forward_return_h5": "",
        "paper_cohort": "", "like_mode": "", "verdict_family": pto.FAMILY_ENDORSE,
        "match_state": pto.MATCH_STATE_MATCHED,
    }
    missed = {
        "schema": pto.SCHEMA, "generated_at": f"{TODAY}T06:00:00-07:00",
        "session_date": SESSION, "symbol": "KBR", "side": "SHORT",
        "channel": "pick_feedback:not_today", "statement": "not today",
        "statement_detail": "triple-VWAP invalidation", "statement_id": "pf-kbr-1",
        "traded": "no", "trade_id": "", "trade_opened_at": "",
        "match_confidence": "", "match_basis": "", "journal_r": "",
        "journal_net_pnl": "", "paper_forward_return_h3": "",
        "paper_forward_return_h5": "", "paper_cohort": "", "like_mode": "",
        "verdict_family": pto.FAMILY_REJECT,
        "match_state": pto.MATCH_STATE_NO_MATCH_AFTER_WINDOW,
    }
    return _csv_text(
        ",".join(pto.COLUMNS),
        [[str(row.get(column, "")) for column in pto.COLUMNS] for row in (matched, missed)],
    )


STAGED_PICKS = {
    "date": TODAY,
    "pending": {"long": {"AAA": {"staged_at": f"{TODAY}T06:35:00-07:00"}},
                "short": {"BBB": {"staged_at": f"{TODAY}T06:40:00-07:00"}}},
    "decided": {"long": {}, "short": {}},
}


WORKING_LATELY_SNAPSHOT = {
    "snapshot_id": "0000000000000000000000000000000000000001",
    "as_of": f"{SESSION}T13:05:00-07:00",
    "cells": [
        {"kind": "m5", "side": "LONG", "family": "ema_15", "n": 12},
        {"kind": "swing", "side": "SHORT", "family": "avwap_band_bounce", "n": 9},
    ],
}


# ---------------------------------------------------------------------------
# Laying the stores down, and the one seam that reads them
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _write_stores(root: Path) -> dict[str, str]:
    """Write every fixture store under `root`. Returns {source key: path}."""
    import ui.annotations.pass_bars as pass_bars

    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "intraday_outcomes": root / "intraday_bounce_outcomes.csv",
        "tier_outcomes": root / "master_avwap_tier_outcomes.csv",
        "session_horizon_outcomes": root / "master_avwap_session_horizon_outcomes.csv",
        "annotations": root / "trader_annotations.jsonl",
        "pick_feedback": root / "pick_feedback.jsonl",
        "swing_favorites": root / "swing_favorites.jsonl",
        "human_focus_outcomes": root / "human_focus_outcomes.csv",
        "review_events": root / "alert_review_events.jsonl",
        "preference_report": root / "preference_trade_outcomes.csv",
        "staged_picks": root / "auto_populate_pending.json",
        "environment_labels": root / "d1_environment.jsonl",
        "working_lately": root / "snapshot_latest.json",
    }

    paths["intraday_outcomes"].write_text(_intraday_csv(), encoding="utf-8")
    paths["tier_outcomes"].write_text(_tier_csv(), encoding="utf-8")
    paths["session_horizon_outcomes"].write_text(_session_horizon_csv(), encoding="utf-8")
    paths["human_focus_outcomes"].write_text(_human_focus_csv(), encoding="utf-8")
    paths["preference_report"].write_text(_preference_report_csv(), encoding="utf-8")

    pass_created_at = pass_bars.attach_desk_zone(datetime(2026, 9, 10, 12, 30)).isoformat()
    _write_jsonl(paths["annotations"], _annotation_rows(pass_created_at))
    _write_jsonl(paths["pick_feedback"], PICK_FEEDBACK_ROWS)
    _write_jsonl(paths["swing_favorites"], SWING_FAVORITE_ROWS)
    _write_jsonl(paths["review_events"], REVIEW_EVENT_ROWS)
    _write_jsonl(paths["environment_labels"], ENVIRONMENT_ROWS)

    paths["staged_picks"].write_text(json.dumps(STAGED_PICKS, indent=2), encoding="utf-8")
    paths["working_lately"].write_text(
        json.dumps(WORKING_LATELY_SNAPSHOT, indent=2), encoding="utf-8"
    )

    sidecar_dir = root / "trader_annotation_bars"
    sidecar_dir.mkdir(exist_ok=True)
    (sidecar_dir / "aaaa0000000000000000000000000001.json").write_text(
        json.dumps(
            {
                "sidecar_schema_version": 1,
                "event_id": "aaaa0000000000000000000000000001",
                "symbol": "SHW", "side": "SHORT", "interval": "M5",
                "created_at": pass_created_at,
                "bar_count": len(PASS_SIDECAR_BARS),
                "bars": [dict(bar) for bar in PASS_SIDECAR_BARS],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {key: str(value) for key, value in paths.items()}


def _clear_store_caches() -> None:
    """Every mtime-keyed cache the source stores hold, so a re-read is a re-read."""
    try:
        import pick_feedback

        pick_feedback.clear_reviewed_today_cache()
    except Exception:
        pass
    try:
        import d1_environment_store

        d1_environment_store._LABEL_CACHE.clear()
    except Exception:
        pass
    try:
        import review_events

        review_events._events_cache = None
    except Exception:
        pass


@pytest.fixture
def stores(tmp_path):
    root = tmp_path / "stores"
    mapping = _write_stores(root)
    _clear_store_caches()
    yield mapping
    _clear_store_caches()


def _sources(mapping: dict[str, str]):
    from daily_recap_reader import RecapSources

    return RecapSources(**{key: Path(value) for key, value in mapping.items()})


def _read(mapping: dict[str, str], *, session_date: str = SESSION, lookback: int = 3):
    import daily_recap_reader

    return daily_recap_reader.read_session(
        session_date, lookback_sessions=lookback, now=NOW, sources=_sources(mapping)
    )


def _symbols(view) -> tuple[str, ...]:
    return tuple(row.symbol for row in view.rows)


def _by_symbol(view, symbol: str, verdict: str | None = None):
    matches = [row for row in view.rows if row.symbol == symbol]
    if verdict is not None:
        matches = [row for row in matches if row.detail.get("verdict") == verdict]
    assert len(matches) == 1, (
        f"expected exactly one {symbol} row"
        + (f" with verdict {verdict!r}" if verdict else "")
        + f", got {[(r.symbol, r.detail.get('verdict')) for r in view.rows]}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# 1 - The read is over the STORES, and a restart does not change it
# ---------------------------------------------------------------------------

_RESTART_SCRIPT = r'''
import json, os, sys
root, mapping_json, session, lookback = sys.argv[1:5]
os.environ["TRADINGBOTV3_DATA_DIR"] = os.path.join(root, "_scratch_data")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.environ["WS_DR_REPO"], "scripts"))
import project_paths
assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
from pathlib import Path
from datetime import datetime
import daily_recap_reader
mapping = json.loads(mapping_json)
sources = daily_recap_reader.RecapSources(**{k: Path(v) for k, v in mapping.items()})
read = daily_recap_reader.read_session(
    session, lookback_sessions=int(lookback),
    now=datetime(2026, 9, 11, 7, 30), sources=sources,
)
sys.stdout.write(repr(read))
'''


def test_a_session_read_survives_a_restart_unchanged(stores, tmp_path):
    """The record of the day lives in the files, not in this process.

    Today the AWAY Recap is handed `center._alerts` + `center._d1_alerts`
    (`scripts/ui/app.py` `_feed_away_recap`): a process-scoped list capped at
    250 / 100 items. A desk restarted mid-session, or left running across
    midnight, reports what the PROCESS saw. This is the property that has to
    replace it, proved the only way it can be: the same fixture read in a
    SEPARATE INTERPRETER must produce the identical session.
    """
    first = _read(stores)

    script = tmp_path / "restart_read.py"
    script.write_text(_RESTART_SCRIPT, encoding="utf-8")
    environment = dict(os.environ)
    environment["WS_DR_REPO"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path), json.dumps(stores), SESSION, "3"],
        capture_output=True, text=True, timeout=300, env=environment,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout == repr(first), (
        "the session read in a fresh interpreter differs from the one read here"
    )


def test_read_session_takes_no_process_scoped_feed(stores):
    """Its whole input is a session, a lookback, a clock and a set of PATHS.

    TJ-1 item 4 adds ONE more: `index`, a stored payload of the rows this reader
    would itself have read off those same paths. It is DATA, not a feed - it
    carries no process state, it defaults to `None`, and an index that is not of
    this session and this window is ignored and the stores are streamed
    (`tests/test_tj1_day_review_index.py`).
    """
    import inspect

    import daily_recap_reader

    parameters = inspect.signature(daily_recap_reader.read_session).parameters
    assert set(parameters) == {
        "session_date", "lookback_sessions", "now", "sources", "index",
    }
    assert parameters["lookback_sessions"].default == 3
    assert parameters["index"].default is None

    source_fields = {field.name for field in dataclass_fields(daily_recap_reader.RecapSources)}
    missing = [name for name in REQUIRED_SOURCES if name not in source_fields]
    assert not missing, f"RecapSources names no path for {missing}"


def test_the_session_declares_coverage_for_every_source(stores):
    """`rows`, `oldest`, `newest`, `unavailable_reason` - per source, per read."""
    session = _read(stores)

    missing = [name for name in REQUIRED_SOURCES if name not in session.coverage]
    assert not missing, f"no coverage entry for {missing}"

    for name in REQUIRED_SOURCES:
        entry = session.coverage[name]
        for attribute in ("rows", "oldest", "newest", "unavailable_reason"):
            assert hasattr(entry, attribute), f"{name} coverage has no {attribute}"
        assert entry.unavailable_reason == "", (
            f"{name} is present on disk but reported unavailable: "
            f"{entry.unavailable_reason}"
        )


def test_source_row_counts_reconcile_to_the_fixtures(stores):
    """Counting, not sampling: every row on disk is accounted for."""
    session = _read(stores)
    expected = {
        "intraday_outcomes": len(INTRADAY_ROWS),
        "tier_outcomes": len(TIER_ROWS),
        "session_horizon_outcomes": len(SESSION_HORIZON_ROWS),
        "annotations": len(_annotation_rows("x")),
        "pick_feedback": len(PICK_FEEDBACK_ROWS),
        "swing_favorites": len(SWING_FAVORITE_ROWS),
        "human_focus_outcomes": 2,
        "review_events": len(REVIEW_EVENT_ROWS),
        "preference_report": 2,
    }
    actual = {name: session.coverage[name].rows for name in expected}
    assert actual == expected


def test_a_missing_source_is_unavailable_with_its_reason_and_the_rest_still_reads(stores):
    """Missing data is uncertainty. It is named, and it costs no other view."""
    Path(stores["intraday_outcomes"]).unlink()
    _clear_store_caches()

    session = _read(stores)

    entry = session.coverage["intraday_outcomes"]
    assert entry.rows == 0
    assert entry.unavailable_reason.strip(), "a missing source must say why"
    assert "intraday_bounce_outcomes" in entry.unavailable_reason, (
        f"the reason must name the file: {entry.unavailable_reason!r}"
    )
    # The decisions the trader made are in other files and are still there.
    assert session.my_decisions.n > 0
    assert session.coverage["annotations"].rows == len(_annotation_rows("x"))


def test_an_old_session_reads_from_todays_files(stores):
    """A session the desk has long since moved past is still readable.

    The fixture's prior session holds exactly one intraday outcome, one veto,
    one not-today and one swing favorite; none of the selected session's rows
    may appear, and none of the prior session's rows may leak into the later
    read.
    """
    session = _read(stores, session_date=PRIOR_1)

    assert _symbols(session.worked_today) == ("NFLX",)
    decisions = {(row.symbol, row.detail.get("verdict")) for row in session.my_decisions.rows}
    assert decisions == {("NFLX", "veto"), ("NFLX", "not_today"), ("NFLX", "swing_favorite")}

    later = _read(stores)
    assert "NFLX" not in _symbols(later.worked_today)


def test_the_default_session_is_the_last_completed_one():
    """`now` is 07:30 on 2026-09-11 - that session is open, so 09-10 is the day."""
    import daily_recap_reader

    assert daily_recap_reader.default_session(now=NOW) == SESSION


def test_a_provisional_session_says_so(stores):
    completed = _read(stores)
    assert completed.provisional is False

    today = _read(stores, session_date=TODAY)
    assert today.provisional is True, (
        "an unfinished session is explicitly provisional (WISHLIST 10F)"
    )


# ---------------------------------------------------------------------------
# 2 - View 1: what worked today
# ---------------------------------------------------------------------------

def test_worked_today_holds_the_sessions_m5_rows_and_nothing_else(stores):
    session = _read(stores)
    view = session.worked_today

    assert view.window == (SESSION, SESSION)
    assert view.n == 8, _symbols(view)
    assert "NFLX" not in _symbols(view)
    assert set(_symbols(view)) == {"NVDA", "TSLA", "AMD", "SHW", "MU", "KBR", "STT", "AAPL"}


def test_worked_today_is_side_adjusted_so_a_short_that_fell_five_percent_leads(stores):
    """The number, not the shape.

    TSLA (short) and NVDA (long) both closed at 95.00 from a 100.00 entry. For
    the short that is +5.00% held at the close; for the long it is -5.00%. A
    reader that recomputed `(eod_close - entry) / entry` puts them in the same
    place and this fails.
    """
    session = _read(stores)
    view = session.worked_today

    assert _by_symbol(view, "TSLA").measures["eod_move_pct"] == pytest.approx(5.00)
    assert _by_symbol(view, "NVDA").measures["eod_move_pct"] == pytest.approx(-5.00)
    assert _by_symbol(view, "TSLA").side == "SHORT"
    assert _by_symbol(view, "NVDA").side == "LONG"

    order = tuple(row.symbol for row in view.sorted_by("eod_move_pct"))
    assert order == ("TSLA", "KBR", "AAPL", "SHW", "MU", "STT", "NVDA", "AMD")


def test_worked_today_defaults_to_the_biggest_favorable_move(stores):
    session = _read(stores)
    view = session.worked_today

    assert view.sort_key == "mfe_pct"
    assert _symbols(view) == ("MU", "TSLA", "SHW", "KBR", "AAPL", "NVDA", "STT", "AMD")
    assert _by_symbol(view, "MU").measures["mfe_pct"] == pytest.approx(8.00)


def test_an_unfinalized_row_is_unmeasured_not_zero(stores):
    """AMD's columns are PRESENT AND EMPTY on disk. That is not a 0.00 day."""
    session = _read(stores)
    row = _by_symbol(session.worked_today, "AMD")

    assert row.measures["mfe_pct"] is None
    assert row.measures["eod_move_pct"] is None
    assert row.unavailable.get("mfe_pct", "").strip(), "an unmeasured cell names its reason"
    assert row.unavailable.get("eod_move_pct", "").strip()
    # It is still shown and still counted.
    assert row in session.worked_today.rows


def test_every_worked_row_keeps_the_capture_id_it_came_from(stores):
    session = _read(stores)
    ids = {row.capture_id for row in session.worked_today.rows}
    assert "TSLA_short_20260910_09_45_00_ema_15" in ids
    assert "SHW_short_20260910_06_35_00_ema_15" in ids


# ---------------------------------------------------------------------------
# 3 - View 2: recent swing picks that followed through
# ---------------------------------------------------------------------------

def test_recent_swings_reads_the_three_prior_sessions_by_default(stores):
    """The lookback window is the 3 sessions BEFORE the selected one.

    Labor Day 2026-09-07 is not a session, so the third prior session of
    2026-09-10 is 09-04. A window counted in calendar days would say 09-07 and
    would pull GOOG's 09-02 rows in with it.
    """
    session = _read(stores)
    view = session.recent_swings

    assert view.window == (PRIOR_3, PRIOR_1)
    assert set(_symbols(view)) == {"AAPL", "ORCL"}
    assert "GOOG" not in _symbols(view)
    assert view.n == 2


def test_a_swing_row_carries_the_instant_follow_through_and_the_selected_end(stores):
    session = _read(stores)
    view = session.recent_swings

    aapl = _by_symbol(view, "AAPL")
    # The next session's close, from the horizon-1 row.
    assert aapl.measures["next_close_pct"] == pytest.approx(2.00)
    # The selected 1-3-session end, from the horizon-3 row - a SEPARATE column.
    assert aapl.measures["selected_end_pct"] == pytest.approx(6.00)
    # The first next-session favorable move: AAPL's own M5 outcome row on the
    # target session, 2026-09-10.
    assert aapl.measures["first_favorable_pct"] == pytest.approx(3.50)

    orcl = _by_symbol(view, "ORCL")
    assert orcl.measures["next_close_pct"] == pytest.approx(1.25)
    assert orcl.measures["selected_end_pct"] == pytest.approx(3.00)
    # Nothing measured ORCL's next-session excursion. Unavailable, never zero.
    assert orcl.measures["first_favorable_pct"] is None
    assert orcl.unavailable.get("first_favorable_pct", "").strip()


def test_the_lookback_selects_the_window_and_the_horizon_together(stores):
    """1/2/3 is one control: how far back to look, and which end to report."""
    session = _read(stores, lookback=1)
    view = session.recent_swings

    assert session.lookback_sessions == 1
    assert view.window == (PRIOR_1, PRIOR_1)
    assert _symbols(view) == ("AAPL",)
    assert _by_symbol(view, "AAPL").measures["selected_end_pct"] == pytest.approx(2.00)


def test_an_incomplete_horizon_is_pending_and_never_a_zero(stores):
    """MSFT's 3-session end has not arrived. `measured` is present and empty."""
    session = _read(stores)
    view = session.recent_swings

    assert "MSFT" not in _symbols(view)
    pending = [row for row in view.pending if row.symbol == "MSFT"]
    assert len(pending) == 1, [row.symbol for row in view.pending]
    assert pending[0].pending is True
    assert pending[0].measures["selected_end_pct"] is None
    assert pending[0].unavailable.get("selected_end_pct", "").strip()


# ---------------------------------------------------------------------------
# 4 - View 3: my decisions
# ---------------------------------------------------------------------------

#: Every decision the fixture's selected session contains, as
#: (symbol, side, verdict). `unfavorite` is absent because taking a name out of
#: Focus is not a verdict on it; the removed swing favorite is absent because a
#: retraction removes it; the note is absent because a note is neither a like
#: nor a rejection and the packet's view 3 lists neither.
EXPECTED_DECISIONS = {
    ("MU", "LONG", "like"),
    ("NVDA", "LONG", "like"),
    ("AAPL", "LONG", "like"),
    ("FTNT", "LONG", "swing_favorite"),
    ("SHW", "SHORT", "pass"),
    ("STT", "LONG", "veto"),
    ("KBR", "SHORT", "veto"),
    ("KBR", "SHORT", "not_today"),
    ("STT", "LONG", "dislike"),
    ("TSLA", "SHORT", "m5_click_away"),
}


def test_every_verdict_is_its_own_fact_and_none_are_combined(stores):
    """Ten decisions, ten rows.

    KBR was vetoed AND thrown back for the day; STT was vetoed AND disliked.
    Neither pair may be merged: they are two statements about one name, and a
    reader that pooled them would count one decision twice.
    """
    session = _read(stores)
    view = session.my_decisions

    actual = {(row.symbol, row.side, row.detail.get("verdict")) for row in view.rows}
    assert actual == EXPECTED_DECISIONS
    assert view.n == len(EXPECTED_DECISIONS)
    assert view.window == (SESSION, SESSION)


def test_unfavorite_is_excluded_everywhere(stores):
    """`unfavorite` is never graded (CLAUDE.md, P5)."""
    session = _read(stores)
    verdicts = {row.detail.get("verdict") for row in session.my_decisions.rows}
    assert "unfavorite" not in verdicts
    assert "unfavorite" not in {
        row.detail.get("verdict") for row in session.rejected_that_worked.rows
    }


def test_a_retraction_removes_the_swing_favorite(stores):
    """BXWT was added at 12:27 and taken back at 14:02. It is not a favorite."""
    session = _read(stores)
    favorites = {
        row.symbol for row in session.my_decisions.rows
        if row.detail.get("verdict") == "swing_favorite"
    }
    assert favorites == {"FTNT"}


def test_repeated_likes_on_one_opportunity_link_and_n_stays_one(stores):
    """Two clicks at 12:30 and 13:05 are one decision with two occurrences."""
    session = _read(stores)
    view = session.my_decisions

    row = _by_symbol(view, "MU", verdict="like")
    assert row.occurrences == 2
    assert len([r for r in view.rows if r.symbol == "MU"]) == 1
    # Credit starts at the FIRST click, which is still after the morning high.
    assert (row.observed_at.hour, row.observed_at.minute) == (12, 30)


def test_the_two_like_modes_stay_apart(stores):
    """Alt+L names no setup; Alt+K does. An absent mode reads `claimed`."""
    session = _read(stores)
    view = session.my_decisions

    assert _by_symbol(view, "MU", verdict="like").detail["like_mode"] == "claimed"
    assert _by_symbol(view, "NVDA", verdict="like").detail["like_mode"] == "quick"


def test_overlapping_pass_cohorts_count_once(stores):
    """One pass, two reason codes. The cohorts overlap and are never summed."""
    session = _read(stores)
    view = session.my_decisions

    row = _by_symbol(view, "SHW", verdict="pass")
    assert row.occurrences == 1
    assert len([r for r in view.rows if r.detail.get("verdict") == "pass"]) == 1
    assert tuple(row.detail["reason_codes"]) == ("poor_market_conditions", "choppy_tape")


def test_a_like_at_1230_gets_no_credit_for_the_1000_high(stores):
    """A decision's credit starts at its own timestamp.

    MU ran 8.00% favorable across the session. The like was clicked at 12:30 and
    nothing in the stores says WHEN that 8.00% happened, so the credited number
    is `unavailable` - not 8.00, and not 0.00 either.
    """
    session = _read(stores)
    row = _by_symbol(session.my_decisions, "MU", verdict="like")

    credited = row.measures["mfe_pct_after_decision"]
    assert credited is None, (
        f"the 12:30 like was credited with {credited}; MU's whole-day MFE is "
        f"{MU_DAY_MFE_PCT}"
    )
    assert row.unavailable.get("mfe_pct_after_decision", "").strip()


def test_a_pass_with_bars_is_measured_from_the_moment_it_was_taken(stores):
    """The sidecar makes the timing knowable, so the number is measurable.

    SHW's whole-day favorable excursion for a short is 5.00% (the 06:35 bar's
    323.00 low against the 340.00 entry). The pass was taken at 12:30, and from
    the last completed bar there - 12:25, close 340.00 - the post-decision low
    is 333.20, which is 2.00%. A reader that handed the pass the day's MFE
    reports 5.00 and fails.
    """
    session = _read(stores)
    row = _by_symbol(session.my_decisions, "SHW", verdict="pass")

    assert row.measures["mfe_pct_after_decision"] == pytest.approx(
        SHW_AFTER_DECISION_MFE_PCT
    )
    assert row.measures["mfe_pct_after_decision"] != pytest.approx(SHW_DAY_MFE_PCT)


def test_the_journal_pnl_arrives_from_the_preference_report_with_its_match_state(stores):
    """WS-5B's symmetric report is the join. MFE, EOD and P&L stay separate."""
    session = _read(stores)
    view = session.my_decisions

    liked = _by_symbol(view, "AAPL", verdict="like")
    assert liked.measures["journal_r"] == pytest.approx(1.4)
    assert liked.detail["journal_net_pnl"] == pytest.approx(312.50)
    assert liked.detail["match_state"] == "matched"

    rejected = _by_symbol(view, "KBR", verdict="not_today")
    assert rejected.measures["journal_r"] is None
    assert rejected.detail["match_state"] == "no_match_after_window"


def test_a_decision_is_identified_at_the_opportunity_grain(stores):
    """`(trade_date, symbol, side, category slot)` per `_pick_key`.

    Four parts, and the category is one of them: without it a name on both the
    swing and the M5 list collapses into a single row.
    """
    session = _read(stores)

    for row in session.my_decisions.rows:
        assert len(row.pick_key) == 4, row.pick_key
        assert row.pick_key[:3] == (SESSION, row.symbol, row.side), row.pick_key
        assert str(row.pick_key[3]).strip(), f"{row.symbol} has no category slot"


# ---------------------------------------------------------------------------
# 5 - View 4: my rejected picks that worked
# ---------------------------------------------------------------------------

def test_only_the_rejections_whose_later_path_was_favorable_appear(stores):
    """An unmeasured D1 refusal is unavailable, not an M5 fallback."""
    session = _read(stores)
    view = session.rejected_that_worked

    actual = {(row.symbol, row.detail.get("verdict")) for row in view.rows}
    assert actual == {("KBR", "not_today"), ("TSLA", "m5_click_away")}
    assert "STT" not in _symbols(view)
    assert "SHW" not in _symbols(view)


def test_a_d1_rejection_without_a_matching_horizon_is_unavailable(stores):
    """A D1 veto never borrows a same-name M5 path."""
    session = _read(stores)
    row = _by_symbol(session.my_decisions, "KBR", verdict="veto")
    assert row.detail["timeframe"] == "D1"
    assert row.detail["result_state"] == "unmeasured"
    assert row.measures["d1_result_pct"] is None


def test_a_d1_pass_keeps_only_the_reachable_post_decision_m5_measure(stores):
    """The D1 result remains unavailable while its bar-sidecar measure remains honest."""
    session = _read(stores)
    shw = _by_symbol(session.my_decisions, "SHW", verdict="pass")
    assert shw.measures["d1_result_pct"] is None
    assert shw.measures["mfe_pct_after_decision"] == pytest.approx(
        SHW_AFTER_DECISION_MFE_PCT
    )


# ---------------------------------------------------------------------------
# 6 - Cross-cutting: sort keys, timestamps, environment, staged picks
# ---------------------------------------------------------------------------

def test_every_view_declares_its_sort_keys_and_refuses_anything_else(stores):
    """A sort control limited to the declared measures.

    The same refusal gate #43 puts on the narration view: no result statistic
    may become an ordering key.
    """
    session = _read(stores)

    for name in VIEW_NAMES:
        view = getattr(session, name)
        assert view.sort_keys, f"{name} declares no sort key"
        assert view.sort_key in view.sort_keys
        forbidden = [key for key in view.sort_keys if key in FORBIDDEN_SORT_KEYS]
        assert not forbidden, f"{name} would sort by {forbidden}"
        for row in view.rows:
            assert set(row.measures) == set(view.sort_keys), (
                f"{name}'s row {row.symbol} measures {sorted(row.measures)} but the "
                f"view declares {sorted(view.sort_keys)}"
            )
        with pytest.raises(ValueError):
            view.sorted_by("priority_score")


def test_every_timestamp_is_aware_and_the_offset_is_the_one_on_disk(stores):
    """Attach market-local to the naive side; never strip the aware side (N1)."""
    session = _read(stores)

    for name in VIEW_NAMES:
        view = getattr(session, name)
        for row in tuple(view.rows) + tuple(view.pending):
            if row.observed_at is None:
                continue
            assert row.observed_at.tzinfo is not None, (
                f"{name}/{row.symbol} carries a naive timestamp"
            )
            assert row.observed_at.utcoffset() is not None

    # The MU like's `created_at` says -07:00 on disk. It stays -07:00.
    like = _by_symbol(session.my_decisions, "MU", verdict="like")
    assert like.observed_at.utcoffset() == timedelta(hours=-7)

    # `pick_feedback` rows are NAIVE on disk (0 of 1,377 on the desk are aware),
    # so this one had to have a zone attached rather than being dropped.
    not_today = _by_symbol(session.my_decisions, "KBR", verdict="not_today")
    assert not_today.observed_at.tzinfo is not None
    assert (not_today.observed_at.hour, not_today.observed_at.minute) == (12, 51)


def test_every_row_carries_the_environment_of_the_session_it_was_decided_in(stores):
    """WS-ENV, point in time: the tape it was DECIDED in, never the exit's."""
    session = _read(stores)

    assert _by_symbol(session.worked_today, "TSLA").d1_environment == "bullish_strong"
    assert _by_symbol(session.my_decisions, "MU", verdict="like").d1_environment == (
        "bullish_strong"
    )
    # Scanned on 2026-09-09, which the store labels `neutral`.
    assert _by_symbol(session.recent_swings, "AAPL").d1_environment == "neutral"
    # Scanned on 2026-09-08, which nobody labelled. `unknown` is its own cell.
    assert _by_symbol(session.recent_swings, "ORCL").d1_environment == "unknown"


def test_the_staged_picks_travel_with_the_session(stores):
    """AWAY's staged-pick block moves onto this page unchanged."""
    session = _read(stores)
    assert {key: tuple(value) for key, value in dict(session.staged_picks).items()} == {
        "long": ("AAA",),
        "short": ("BBB",),
    }


def test_nothing_here_loads_the_tracker_json(stores, monkeypatch):
    """The tracker is 1.1 GB. A recap that opened it would freeze the desk."""
    import project_paths

    opened: list[str] = []
    real_open = io.open

    def watched(file, *args, **kwargs):
        try:
            opened.append(str(file))
        except Exception:
            pass
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", watched)
    _read(stores)
    monkeypatch.undo()

    tracker = str(project_paths.MASTER_AVWAP_SETUP_TRACKER_FILE)
    assert tracker not in opened
    assert not [name for name in opened if "master_avwap_setup_tracker" in name]


# ---------------------------------------------------------------------------
# 7 - The page
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _process(qapp, rounds: int = 4) -> None:
    for _ in range(rounds):
        qapp.processEvents()


@pytest.fixture
def make_page(qapp):
    built = []

    def _build(width: int = 1640, height: int = 980):
        from ui.panels.daily_recap_panel import DailyRecapPanel

        page = DailyRecapPanel()
        built.append(page)
        page.resize(width, height)
        page.show()
        _process(qapp)
        return page

    yield _build

    for page in built:
        try:
            page.shutdown()
        except Exception:
            pass
        page.hide()
        page.deleteLater()
    _process(qapp)


TABLE_ATTRS = {
    "worked_today": "worked_today_table",
    "recent_swings": "recent_swings_table",
    "my_decisions": "my_decisions_table",
    "rejected_that_worked": "rejected_that_worked_table",
}
NOTE_ATTRS = {name: attr.replace("_table", "_note") for name, attr in TABLE_ATTRS.items()}


def _row_of(table, symbol: str) -> int:
    for index in range(table.rowCount()):
        for column in range(table.columnCount()):
            item = table.item(index, column)
            if item is not None and item.text().strip() == symbol:
                return index
    raise AssertionError(f"{symbol} is not in the table")


@pytest.mark.qt
def test_the_nav_shows_daily_recap_where_away_recap_was():
    from ui.app import PAGE_SPECS

    titles = [spec.title for spec in PAGE_SPECS]
    assert "AWAY Recap" not in titles
    assert "Daily Recap" in titles
    spec = PAGE_SPECS[titles.index("Daily Recap")]
    assert spec.attribute == "daily_recap_panel"
    assert spec.icon.strip()
    # It takes the retired page's place rather than being appended at the end.
    assert titles.index("Daily Recap") == titles.index("Market Journal") + 1


@pytest.mark.qt
def test_the_page_opens_on_the_last_completed_session_with_today_offered(make_page):
    import market_calendar

    page = make_page()

    now = datetime.now()
    completed = market_calendar.last_completed_session(now).isoformat()
    assert page.session_date() == completed
    entries = [
        page.session_picker.itemText(index) for index in range(page.session_picker.count())
    ]
    provisional = [text for text in entries if "Today" in text and "provisional" in text]
    if now.date().isoformat() == completed:
        # Run after the close on a session day: today IS the completed head
        # and must not also be offered as a provisional entry (the pre-2026-09-14
        # form of this test only passed before 13:00 Pacific).
        assert not provisional, f"a closed session offered as provisional in {entries}"
        assert page.session_picker.itemData(0) == completed
    else:
        assert provisional, f"no explicitly provisional Today entry in {entries}"
    assert page.lookback_sessions() == 3


@pytest.mark.qt
def test_the_page_renders_before_it_has_any_data(make_page):
    """The read is on a worker, so the page must be usable while it loads."""
    page = make_page()

    for name, attr in TABLE_ATTRS.items():
        table = getattr(page, attr)
        assert table.rowCount() == 0, name
        assert table.columnCount() > 0, name
        assert getattr(page, NOTE_ATTRS[name]).text().strip(), (
            f"{name} says nothing at all before its data arrives"
        )


@pytest.mark.qt
def test_the_page_draws_every_row_the_reader_produced(stores, make_page, qapp):
    page = make_page(3456, 2160)
    session = _read(stores)
    page.render_session(session)
    _process(qapp)

    assert page.worked_today_table.rowCount() == session.worked_today.n == 8
    assert page.recent_swings_table.rowCount() == (
        session.recent_swings.n + len(session.recent_swings.pending)
    ) == 3
    assert page.my_decisions_table.rowCount() == session.my_decisions.n == 10
    assert page.rejected_that_worked_table.rowCount() == session.rejected_that_worked.n == 2


@pytest.mark.qt
def test_every_table_carries_its_population_sentence(stores, make_page, qapp):
    """Cohort, window, n, pending, coverage - said out loud, per table."""
    page = make_page(3456, 2160)
    session = _read(stores)
    page.render_session(session)
    _process(qapp)

    for name in VIEW_NAMES:
        view = getattr(session, name)
        text = getattr(page, NOTE_ATTRS[name]).text()
        assert str(view.n) in text, f"{name}: {text!r} does not say n={view.n}"
        assert view.window[0] in text and view.window[1] in text, (
            f"{name}: {text!r} does not say its window {view.window}"
        )
        assert "pending" in text.lower(), f"{name}: {text!r} does not report pending"


@pytest.mark.qt
def test_a_row_click_asks_for_a_board_chart_and_carries_the_decision_time(
    stores, make_page, qapp
):
    """One chart surface, the board door, and the moment the decision was made."""
    page = make_page(3456, 2160)
    session = _read(stores)
    page.render_session(session)
    _process(qapp)

    seen: list[tuple] = []
    page.chartRequested.connect(lambda *args: seen.append(args))

    table = page.rejected_that_worked_table
    index = _row_of(table, "TSLA")
    table.setCurrentCell(index, 0)
    table.itemDoubleClicked.emit(table.item(index, 0))
    _process(qapp)

    assert seen, "a double click on a recap row charted nothing"
    assert seen[0][0] == "TSLA"
    assert str(seen[0][1]).upper().startswith("SHORT")

    # The decision time is in front of the trader, in a cell or on the tooltip.
    texts = " ".join(
        table.item(index, column).text() + " " + table.item(index, column).toolTip()
        for column in range(table.columnCount())
        if table.item(index, column) is not None
    )
    assert "09:50" in texts, (
        f"the click-away's 09:50 decision time is nowhere on its row: {texts!r}"
    )


@pytest.mark.qt
def test_the_away_staged_pick_block_still_works(stores, make_page, qapp):
    page = make_page()
    session = _read(stores)
    page.render_session(session)
    _process(qapp)

    assert page.staged.rowCount() == 2
    asked: list[tuple] = []
    page.focusAddRequested.connect(lambda *args: asked.append(args))
    page.staged.setCurrentCell(0, 0)
    page.add_button.click()
    _process(qapp)

    assert asked, "the staged-pick add button asked nobody for a Focus add"
    assert asked[0][0] in {"AAA", "BBB"}


@pytest.mark.qt
@pytest.mark.parametrize("width,height", [(3456, 2160), (1640, 980)])
def test_the_page_fits_the_screen_it_is_given(stores, make_page, qapp, width, height):
    """No overflow at the trader's 4K desk or in the desk's windowed default."""
    from PySide6.QtCore import QPoint

    page = make_page(width, height)
    session = _read(stores)
    page.render_session(session)
    _process(qapp)

    assert page.minimumSizeHint().height() <= height, (
        f"the Daily Recap insists on {page.minimumSizeHint().height()} px of height "
        f"at {width}x{height}"
    )
    assert page.minimumSizeHint().width() <= width, (
        f"the Daily Recap insists on {page.minimumSizeHint().width()} px of width "
        f"at {width}x{height}"
    )

    bottom = page.height()
    for name, attr in TABLE_ATTRS.items():
        table = getattr(page, attr)
        if not table.isVisible():
            continue
        top_left = table.mapTo(page, QPoint(0, 0))
        assert top_left.y() + table.height() <= bottom, (
            f"{name} runs {top_left.y() + table.height() - bottom} px past the bottom "
            f"of the page at {width}x{height}"
        )


@pytest.mark.qt
def test_the_desk_charts_a_recap_row_through_the_board_door_and_requeues_nothing():
    """`show_board_symbol`, never `_enqueue_review_alert`.

    A board chart holds no place in the waiting list. The recap is a board on
    another page, so it uses the named door - the same one the AWAY Recap used -
    and never the door for things the SCANNER said.
    """
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])

    from ui.app import MainWindow
    from ui.state import UiState

    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        center = window.trading_panel.alert_center
        charted: list[tuple] = []
        queued: list[tuple] = []
        center.show_board_symbol = lambda *args, **kwargs: charted.append((args, kwargs))
        # Lead fix (2026-09-13): the scanner's own symbol-less status row (`Scanning
        # paused.`, side WATCH) reaches this door from the bot thread on the first
        # processEvents() and the real door discards it on its first line; the
        # recorder therefore keeps only symbol-bearing alerts, which is what a
        # re-queue would be. The waiting-list length is measured by the next test.
        center._enqueue_review_alert = lambda alert, *args, **kwargs: (
            queued.append((alert, args)) if getattr(alert, "symbol", "") else None
        )

        window.daily_recap_panel.chartRequested.emit("TSLA", "SHORT")
        application.processEvents()

        assert charted, "the Daily Recap is not wired to the board chart door"
        assert charted[0][0][0] == "TSLA"
        assert not queued, "charting a recap row re-queued a review alert"
    finally:
        try:
            window.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 8 - The phone digest is NOT what changed
# ---------------------------------------------------------------------------

def test_the_phone_text_digest_still_builds_the_way_it_did():
    """`away_recap.build_recap` is untouched: the phone still gets its text.

    Characterization, pinned on the code as it stands BEFORE this packet (the
    module is explicitly out of scope for it). If a builder deletes or reshapes
    the digest while replacing the page, this is what says so.
    """
    import away_recap

    recap = away_recap.build_recap(
        session_date=SESSION,
        alerts=[
            {"symbol": "TSLA", "side": "SHORT", "tier": "A", "trigger": "M5 move -1.44%",
             "time_text": "09:50", "is_d1": False},
        ],
        staged_picks={"long": ["AAA"], "short": ["BBB"]},
        digest_swings=[],
        focus_picks={"long": [], "short": []},
        unavailable={},
    )
    assert recap["session_date"] == SESSION
    assert len(recap["classified_alerts"]) == 1
    assert [row["symbol"] for row in recap["staged_picks"]] == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# 9 - ADDED BY THE BUILDER (2026-09-13). Nothing above is weakened, skipped or
# rewritten; this section only adds.
# ---------------------------------------------------------------------------


@pytest.mark.qt
def test_the_recap_chart_request_adds_nothing_to_the_waiting_list():
    """The same property as the test above, measured on the QUEUE itself.

    `test_the_desk_charts_a_recap_row_through_the_board_door_and_requeues_nothing`
    replaces `_enqueue_review_alert` with a recorder and asserts it is never
    CALLED. On a real `MainWindow` that recorder also catches something the
    recap had nothing to do with: the scanner's own `Scanning paused.` status
    row (`symbol=''`, `side='WATCH'`), delivered from the bot thread on the
    first `processEvents()`. The product is right about it - the real
    `_enqueue_review_alert` drops a symbol-less alert on its FIRST line, so
    nothing is queued - but the recorder cannot tell a call from a queue entry.

    Measured here the other way round: the door is left alone and the WAITING
    LIST is counted before and after. A board chart holds no place in it.
    """
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])

    from ui.app import MainWindow
    from ui.state import UiState

    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        center = window.trading_panel.alert_center
        application.processEvents()
        charted: list[tuple] = []
        center.show_board_symbol = lambda *args, **kwargs: charted.append((args, kwargs))
        before = len(center._review_queue)

        window.daily_recap_panel.chartRequested.emit("TSLA", "SHORT")
        application.processEvents()

        assert charted and charted[0][0][0] == "TSLA"
        assert len(center._review_queue) == before, (
            "charting a recap row put something in the waiting list"
        )
    finally:
        try:
            window.close()
        except Exception:
            pass
