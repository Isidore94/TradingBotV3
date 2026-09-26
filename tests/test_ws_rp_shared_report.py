"""Packet WS-RP - one shared MEASURED report, one report id, two views.

Written RED by the tester before any fix (2026-09-13). A builder may ADD to
this file; it may not weaken, skip or delete anything in it.

WISHLIST 10K asks for one numerical contract with five separate answers -
total profit, biggest opportunity, quickest result, end of day, last day or
two - frozen before the results are read, published once, and readable in two
places that can never disagree. This file is that contract as numbers.

What every fixture here is, and why it is that shape
----------------------------------------------------

* `scripts/measured_report.py` does not exist yet, so every test below fails
  on the import. That is the point.
* Every store is written in its LIVE column order. `intraday_bounce_outcomes.csv`
  names its side column `direction` and already stores `mfe_pct` /
  `eod_move_pct` SIDE-ADJUSTED (`bounce_bot_lib/legacy.py:5230-5252`), so a
  short whose close fell 5% is `+5.00` and a long that fell 5% is `-5.00`; a
  reader that recomputed the move without the side turns a mean of `+1.00`
  into `-2.33` and fails here. `session_horizon_outcomes.csv` carries the 24
  columns of `SESSION_HORIZON_OUTCOME_COLUMNS`. An unfinished row has its
  measured columns PRESENT AND EMPTY, never absent.
* The warehouse outcome rows are the real row `research_warehouse/outcomes.py`
  builds (`:913-963`): `entry_at`, `stop_price`, `stop_distance`, `mfe_r`,
  `time_to_mfe_min`, `first_hit`, `first_hit_at`, `r_at_eod`, `result_state`.
  A row with no stop has `stop_distance = None`, therefore `mfe_r = None` -
  which is exactly the "missing risk" case: R is unknown, the percentage is
  still measured, and neither is a zero.
* Money is the Journal's, once per `trade_id`. The expected total is computed
  in the test by calling the REAL `preference_trade_outcomes.trade_level_summary`
  on the same fixture rows, so this file cannot drift from that contract.
* The bias split is `journal_exposure`'s: a LONG option is never a bullish
  setup. The fixture's bought PUT is -40.00 of BEARISH money even though its
  `direction` is LONG - a reader that bucketed on `direction` puts it under
  bullish and fails.

Numbers, not shapes. Every cell below is a number a wrong formula gets wrong.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import socket
import subprocess
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# The sessions. Real NYSE sessions through the real calendar: 2026-09-07 is
# Labor Day and 2026-09-05/06 is a weekend, so the FOURTH session back from
# 2026-09-10 is 2026-09-04. A window counted in calendar days lands on
# 2026-09-07 and fails every endpoint assertion below.
# ---------------------------------------------------------------------------

SESSION = "2026-09-10"          # Thursday
PRIOR_1 = "2026-09-09"          # Wednesday
PRIOR_2 = "2026-09-08"          # Tuesday
PRIOR_3 = "2026-09-04"          # Friday - 09-07 is Labor Day, 09-05/06 a weekend
OUTSIDE = "2026-09-02"          # outside a four-session selection window

#: Two clocks inside the same session-after. The report id must not move
#: between them: an `as_of` that is a wall clock makes a timer tick look like
#: new evidence (ST6's rule for `working_lately.snapshot_id`, same reason).
NOW_A = datetime(2026, 9, 11, 7, 30)
NOW_B = datetime(2026, 9, 11, 8, 0)

#: The six sections WISHLIST 10K names, in its order.
SECTION_NAMES = (
    "market_thoughts",
    "measured_context",
    "opportunity_results",
    "preference_decisions",
    "actual_trades",
    "missing_evidence",
)

#: The three states a cell may be in. `unavailable` is the SENTENCE beside a
#: cell that is not `measured`; it is never an empty string there, because
#: "nobody looked" and "0.00" are different answers.
STATE_MEASURED = "measured"
STATE_PENDING = "pending"
STATE_UNKNOWN = "unknown"

#: Every cell id this packet's five answers are read through. These ids ARE
#: the contract: the narration cites them, the export carries them and the
#: Review tab prints them, so they cannot be renamed without renaming them
#: here first.
CELL_TOTAL_PROFIT_ALL = "total_profit.all"
CELL_TOTAL_PROFIT_BULLISH = "total_profit.bias.bullish"
CELL_TOTAL_PROFIT_BEARISH = "total_profit.bias.bearish"
CELL_TOTAL_PROFIT_BULL_NEUTRAL = "total_profit.bias.bullish_or_neutral"
CELL_TOTAL_PROFIT_STOCK = "total_profit.instrument.stock"
CELL_TOTAL_PROFIT_OPTION = "total_profit.instrument.option"

CELL_DAY_MFE_PCT = "biggest_opportunity.day.mfe_pct"
CELL_DAY_MAE_PCT = "biggest_opportunity.day.mae_pct"
CELL_DAY_MFE_R = "biggest_opportunity.day.mfe_r"
CELL_SWING_MFE_R = "biggest_opportunity.swing.mfe_r"
CELL_SWING_MAE_R = "biggest_opportunity.swing.mae_r"
CELL_SWING_TIME_TO_MFE = "biggest_opportunity.swing.time_to_mfe_min"

CELL_QUICK_HIT_RATE = "quickest_result.swing.hit_rate"
CELL_QUICK_MEDIAN = "quickest_result.swing.median_trading_minutes"
CELL_QUICK_HITS = "quickest_result.swing.hits"
CELL_QUICK_UNHIT = "quickest_result.swing.unhit"
CELL_QUICK_PENDING = "quickest_result.swing.pending"
CELL_QUICK_UNKNOWN = "quickest_result.swing.unknown"

CELL_EOD_SWING = "end_of_day.swing.r_at_eod"
CELL_EOD_SWING_AT_CLOSE = "end_of_day.swing.entry_at_close.r_at_eod"
CELL_EOD_DAY = "end_of_day.day.eod_move_pct"

CELL_LAST_OBSERVATIONS = "last_sessions.selection.observations"
CELL_LAST_FOLLOW_THROUGH = "last_sessions.follow_through.side_return_pct"
CELL_LAST_FOLLOW_PENDING = "last_sessions.follow_through.pending"

#: Every swing answer the warehouse owns. When the warehouse cannot be read,
#: each of these is `unknown` with a reason and the report is still built.
SWING_CELL_IDS = (
    CELL_SWING_MFE_R,
    CELL_SWING_MAE_R,
    CELL_SWING_TIME_TO_MFE,
    CELL_QUICK_HIT_RATE,
    CELL_QUICK_MEDIAN,
    CELL_EOD_SWING,
    CELL_EOD_SWING_AT_CLOSE,
)


# ---------------------------------------------------------------------------
# fixture writers - the live headers, quoted the way the live writers quote
# ---------------------------------------------------------------------------


def _csv_text(header: str, rows: Sequence[Sequence[str]]) -> str:
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

#: (event_id, session, symbol, direction, entry_time, entry_price, status,
#:  eod_close, eod_move_pct, mfe_pct, mae_pct)
#:
#: Every one of these rows has an EMPTY `risk_per_share` and an EMPTY `mfe_r`,
#: which is what the live file looks like for a bounce the trader never sized:
#: the percentage is measured and the R is not knowable. The side-adjustment
#: is load-bearing - NVDA (long) fell 5% and reads -5.00; TSLA (short) fell 5%
#: and reads +5.00.
INTRADAY_ROWS = (
    ("NVDA_long_20260910_09_40_00_ema_15", SESSION, "NVDA", "long", "09:40:00",
     "100.00", "closed", "95.00", "-5.00", "1.20", "-6.00"),
    ("TSLA_short_20260910_09_45_00_ema_15", SESSION, "TSLA", "short", "09:45:00",
     "100.00", "closed", "95.00", "5.00", "6.00", "-0.80"),
    ("KBR_short_20260910_09_35_00_ema_15", SESSION, "KBR", "short", "09:35:00",
     "80.00", "closed", "77.60", "3.00", "4.00", "-1.50"),
    # PRESENT AND EMPTY: an unfinalized row. Counted as unmeasured, never zero.
    ("AMD_long_20260910_10_05_00_ema_15", SESSION, "AMD", "long", "10:05:00",
     "50.00", "open", "", "", "", ""),
    # The prior session. It must never leak into this session's day cells.
    ("NFLX_long_20260909_09_40_00_ema_15", PRIOR_1, "NFLX", "long", "09:40:00",
     "700.00", "closed", "707.00", "1.00", "40.00", "-1.00"),
)

#: The row a rerun MATURES: AMD finishes. Appending it changes exactly one
#: measured number, so the report id must change and a new version must be
#: written beside the old one rather than over it.
INTRADAY_MATURED_AMD = (
    "AMD_long_20260910_10_05_00_ema_15", SESSION, "AMD", "long", "10:05:00",
    "50.00", "closed", "51.00", "2.00", "2.50", "-0.40",
)

#: The three day answers over the SESSION rows above, by hand:
#:   max mfe_pct   = max(1.20, 6.00, 4.00)      = 6.00   (AMD empty, NFLX prior)
#:   min mae_pct   = min(-6.00, -0.80, -1.50)   = -6.00
#:   mean eod_move = (-5.00 + 5.00 + 3.00) / 3  =  1.00
DAY_MAX_MFE_PCT = 6.00
DAY_WORST_MAE_PCT = -6.00
DAY_MEAN_EOD_PCT = 1.00
#: What a reader that recomputed the move from prices and forgot the side gets.
DAY_MEAN_EOD_PCT_IF_SIDE_IGNORED = (-5.00 + -5.00 + -3.00) / 3


def _intraday_csv(rows: Sequence[Sequence[str]]) -> str:
    context = json.dumps({"market_environment": "bullish_strong"}, sort_keys=True)
    out = []
    for (event_id, trade_date, symbol, direction, entry_time, entry_price,
         status, eod_close, eod_move, mfe_pct, mae_pct) in rows:
        out.append([
            "1", event_id, "registered", f"{trade_date}T13:10:00-07:00", trade_date,
            symbol, direction, f"{trade_date}T{entry_time}", entry_price, "", "",
            "", "", "", "", "", "", "", "False", "False", "False", status, "",
            context, "", eod_close, eod_move, mfe_pct, mae_pct,
        ])
    return _csv_text(INTRADAY_HEADER, out)


SESSION_HORIZON_HEADER = (
    "observation_id,scan_row_id,symbol,side,scan_date,target_session,horizon_sessions,"
    "sessions_spanned,entry_close,entry_close_source,target_close,side_return_pct,"
    "favorable,measured,maturity,unmeasured_reason,outcome_kind,knowledge_basis,tier,"
    "tier_source,priority_bucket,setup_family,favorite_zone,collapsed_same_session"
)

#: (symbol, side, scan_date, target_session, horizon, side_return_pct, measured)
#:
#: Note what is NOT here: a horizon of 2. `SCAN_FACTOR_HORIZONS` is
#: `(1, 3, 5, 10)` (`master_avwap_lib/legacy.py:11079`) and
#: `_write_session_horizon_outcomes` never overrides it, so a two-session
#: FOLLOW-THROUGH has no row on this desk at all. That is uncertainty and the
#: report must say so; it is never a zero and never a silent fallback to 1 or 3.
SESSION_HORIZON_ROWS = (
    ("AAPL", "LONG", PRIOR_3, PRIOR_2, "1", "2.00", True),
    ("AAPL", "LONG", PRIOR_3, PRIOR_1, "3", "6.00", True),
    ("ORCL", "SHORT", PRIOR_2, PRIOR_1, "1", "1.00", True),
    ("ORCL", "SHORT", PRIOR_2, SESSION, "3", "3.00", True),
    ("MSFT", "SHORT", PRIOR_1, SESSION, "1", "3.00", True),
    ("MSFT", "SHORT", PRIOR_1, "2026-09-15", "3", "", False),
    ("TSLA", "LONG", SESSION, "2026-09-11", "1", "6.00", True),
    # OUTSIDE a four-session selection window. A window in calendar days (or
    # one that walks through Labor Day) swallows this +40.00 and wrecks
    # every mean below.
    ("GOOG", "LONG", OUTSIDE, "2026-09-03", "1", "40.00", True),
)

#: selection = 4 sessions ending 2026-09-10 -> {09-04, 09-08, 09-09, 09-10}
#:   follow-through 1: mean(2.00, 1.00, 3.00, 6.00) = 3.00 over n = 4
#:   follow-through 3: mean(6.00, 3.00)             = 4.50 over n = 2, 1 pending
#: selection = 2 sessions ending 2026-09-10 -> {09-09, 09-10}
#:   follow-through 1: mean(3.00, 6.00)             = 4.50 over n = 2
SELECT_4_FT_1_MEAN, SELECT_4_FT_1_N = 3.00, 4
SELECT_4_FT_3_MEAN, SELECT_4_FT_3_N, SELECT_4_FT_3_PENDING = 4.50, 2, 1
SELECT_2_FT_1_MEAN, SELECT_2_FT_1_N = 4.50, 2
#: What a reader that let GOOG in gets for the first cell.
SELECT_4_FT_1_MEAN_IF_CALENDAR_DAYS = (2.00 + 1.00 + 3.00 + 6.00 + 40.00) / 5


def _session_horizon_csv() -> str:
    rows = []
    for symbol, side, scan_date, target, horizon, ret, measured in SESSION_HORIZON_ROWS:
        scan_row_id = f"{symbol}:{scan_date}:{scan_date}-130129"
        rows.append([
            f"{scan_row_id}:{horizon}", scan_row_id, symbol, side, scan_date, target,
            horizon, horizon, "100.00", "session_bar", "", ret,
            "True" if measured else "", "True" if measured else "",
            "mature" if measured else "immature",
            "" if measured else "horizon_not_reached",
            "favorable_direction_session_v2",
            "entry_session_close_to_target_session_close", "S",
            "derived_from_bucket", "favorite_setup", "avwap_band_bounce", "", "1",
        ])
    return _csv_text(SESSION_HORIZON_HEADER, rows)


# ---------------------------------------------------------------------------
# The money. Rows in the LIVE `preference_trade_outcomes.csv` column order, and
# four journal trades in the LIVE `trades` table shape.
# ---------------------------------------------------------------------------

#: (statement_id, symbol, side, channel, statement, trade_id, journal_net_pnl)
#: `t-long-nvda` carries TWO statements - a quick like and a claimed like about
#: the same trade, which is what a partial close reviewed twice looks like in
#: this file. Its +300.00 is counted ONCE.
PREFERENCE_ROWS = (
    ("s1", "NVDA", "LONG", "like", "like", "t-long-nvda", "300.00"),
    ("s2", "NVDA", "LONG", "note", "note", "t-long-nvda", "300.00"),
    ("s3", "TSLA", "SHORT", "like", "like", "t-short-tsla", "-120.00"),
    ("s4", "DRAM", "SHORT", "like", "like", "t-opt-dram", "80.00"),
    ("s5", "SPY", "LONG", "like", "like", "t-opt-spyput", "-40.00"),
)

#: The Journal's own rows. `security_type` / `direction` / the OCC symbol are
#: what `journal_exposure.classify_all` reads; nothing here is invented.
JOURNAL_TRADES = (
    {"trade_id": "t-long-nvda", "symbol": "NVDA", "security_type": "STK",
     "direction": "LONG", "status": "closed", "net_pnl_cad": 300.00,
     "trade_date": SESSION, "opened_at": f"{SESSION}T09:40:00",
     "closed_at": f"{SESSION}T15:50:00"},
    {"trade_id": "t-short-tsla", "symbol": "TSLA", "security_type": "STK",
     "direction": "SHORT", "status": "closed", "net_pnl_cad": -120.00,
     "trade_date": SESSION, "opened_at": f"{SESSION}T09:45:00",
     "closed_at": f"{SESSION}T15:55:00"},
    # A SOLD put: the wheel. Bullish-or-neutral, never pooled with bullish.
    {"trade_id": "t-opt-dram", "symbol": "DRAM  261218P00055000",
     "security_type": "OPT", "direction": "SHORT", "status": "closed",
     "net_pnl_cad": 80.00, "trade_date": SESSION,
     "opened_at": f"{SESSION}T10:00:00", "closed_at": f"{SESSION}T15:00:00"},
    # A BOUGHT put. `direction` is LONG and the bias is BEARISH: a long option
    # is never a bullish setup.
    {"trade_id": "t-opt-spyput", "symbol": "SPY   261218P00600000",
     "security_type": "OPT", "direction": "LONG", "status": "closed",
     "net_pnl_cad": -40.00, "trade_date": SESSION,
     "opened_at": f"{SESSION}T10:30:00", "closed_at": f"{SESSION}T15:30:00"},
)

TOTAL_PROFIT_ALL = 220.00                   # 300 - 120 + 80 - 40
TOTAL_PROFIT_BULLISH = 300.00               # the stock long only
TOTAL_PROFIT_BEARISH = -160.00              # -120 stock short AND -40 long put
TOTAL_PROFIT_BULLISH_OR_NEUTRAL = 80.00     # the sold put
TOTAL_PROFIT_STOCK = 180.00                 # 300 - 120
TOTAL_PROFIT_OPTION = 40.00                 # 80 - 40
#: What a statement-grain sum gets: NVDA's 300 counted twice.
TOTAL_PROFIT_IF_PER_STATEMENT = 520.00
#: What a reader that bucketed the long put on `direction` gets.
BEARISH_IF_BUCKETED_ON_DIRECTION = -120.00


def _preference_csv() -> str:
    import preference_trade_outcomes

    columns = list(preference_trade_outcomes.COLUMNS)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in _preference_row_dicts():
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def _preference_row_dicts() -> list[dict[str, str]]:
    rows = []
    for statement_id, symbol, side, channel, statement, trade_id, net in PREFERENCE_ROWS:
        rows.append({
            "schema": "preference_trade_outcomes_v1",
            "generated_at": f"{SESSION}T21:00:00",
            "session_date": SESSION,
            "symbol": symbol,
            "side": side,
            "channel": channel,
            "statement": statement,
            "statement_detail": "",
            "statement_id": statement_id,
            "traded": "yes",
            "trade_id": trade_id,
            "trade_opened_at": f"{SESSION}T09:40:00",
            "match_confidence": "0.90",
            "match_basis": "symbol_side_session",
            "journal_r": "",
            "journal_net_pnl": net,
            "paper_forward_return_h3": "",
            "paper_forward_return_h5": "",
            "paper_cohort": "",
            "like_mode": "claimed",
            "verdict_family": "endorse",
            "match_state": "matched",
        })
    return rows


# ---------------------------------------------------------------------------
# The warehouse. Real `outcome_path` rows; a reader object with a DECLARED
# source path, because a citation the report prints has to be a file that
# exists (an occurrence the report cannot point at is a fabricated citation).
# ---------------------------------------------------------------------------

ET = "-04:00"  # America/New_York in September


def _et(day: str, hhmm: str) -> str:
    return f"{day}T{hhmm}:00{ET}"


#: `occ-late` enters at 15:50 and first hits its target at 09:40 the NEXT
#: session: 10 trading minutes to the 16:00 close plus 10 from the 09:30 open
#: is 20 TRADING minutes. A wall clock says 1,070.
WAREHOUSE_ROWS: tuple[dict[str, Any], ...] = (
    {
        "occurrence_id": "occ-late", "recipe_id": "swing_house_v1",
        "outcome_definition_id": "house_default_v1", "analysis_unit": "swing",
        "entry_at": _et(SESSION, "15:50"), "entry_price": 100.0,
        "stop_price": 98.0, "stop_distance": 2.0,
        "mfe_r": 3.00, "mae_r": -0.50, "time_to_mfe_min": 300,
        "first_hit": "target_1", "first_hit_at": _et("2026-09-11", "09:40"),
        "r_at_eod": 1.00, "result_state": "closed", "path_kind": "plain_target",
        "maturity_at": _et("2026-09-11", "16:00"),
    },
    {
        "occurrence_id": "occ-fast", "recipe_id": "swing_house_v1",
        "outcome_definition_id": "house_default_v1", "analysis_unit": "swing",
        "entry_at": _et(SESSION, "09:40"), "entry_price": 50.0,
        "stop_price": 49.0, "stop_distance": 1.0,
        "mfe_r": 1.00, "mae_r": -1.50, "time_to_mfe_min": 60,
        "first_hit": "target_1", "first_hit_at": _et(SESSION, "10:10"),
        "r_at_eod": 2.00, "result_state": "closed", "path_kind": "plain_target",
        "maturity_at": _et(SESSION, "16:00"),
    },
    # Closed without ever reaching the target: UNHIT. Counted, not dropped.
    {
        "occurrence_id": "occ-unhit", "recipe_id": "swing_house_v1",
        "outcome_definition_id": "house_default_v1", "analysis_unit": "swing",
        "entry_at": _et(SESSION, "10:00"), "entry_price": 20.0,
        "stop_price": 19.0, "stop_distance": 1.0,
        "mfe_r": 0.50, "mae_r": -1.00, "time_to_mfe_min": 120,
        "first_hit": None, "first_hit_at": None,
        "r_at_eod": None, "result_state": "closed", "path_kind": "plain_target",
        "maturity_at": _et(SESSION, "16:00"),
    },
    # Still running: PENDING.
    {
        "occurrence_id": "occ-open", "recipe_id": "swing_house_v1",
        "outcome_definition_id": "house_default_v1", "analysis_unit": "swing",
        "entry_at": _et(SESSION, "11:00"), "entry_price": 30.0,
        "stop_price": 29.0, "stop_distance": 1.0,
        "mfe_r": None, "mae_r": None, "time_to_mfe_min": None,
        "first_hit": None, "first_hit_at": None,
        "r_at_eod": None, "result_state": "open", "path_kind": "plain_target",
        "maturity_at": None,
    },
    # The bars ran out under it: TRUNCATED. Missing bars are UNKNOWN, and an
    # unknown is never folded into "did not hit".
    {
        "occurrence_id": "occ-truncated", "recipe_id": "swing_house_v1",
        "outcome_definition_id": "house_default_v1", "analysis_unit": "swing",
        "entry_at": _et(SESSION, "12:00"), "entry_price": 40.0,
        "stop_price": 39.0, "stop_distance": 1.0,
        "mfe_r": None, "mae_r": None, "time_to_mfe_min": None,
        "first_hit": None, "first_hit_at": None,
        "r_at_eod": None, "result_state": "truncated", "path_kind": "plain_target",
        "maturity_at": _et(SESSION, "16:00"),
    },
    # ENTERED AT THE CLOSE. There is no same-session bar after 16:00, so
    # `r_at_eod` is null and the answer is "unavailable", never 0.00.
    {
        "occurrence_id": "occ-at-close", "recipe_id": "swing_house_v1",
        "outcome_definition_id": "house_default_v1", "analysis_unit": "swing",
        "entry_at": _et(SESSION, "16:00"), "entry_price": 10.0,
        "stop_price": 9.5, "stop_distance": 0.5,
        "mfe_r": None, "mae_r": None, "time_to_mfe_min": None,
        "first_hit": None, "first_hit_at": None,
        "r_at_eod": None, "result_state": "open", "path_kind": "plain_target",
        "maturity_at": None,
    },
    # A row with NO STOP. `stop_distance` is null, so `mfe_r` is null: R is
    # UNKNOWN here and nothing may fill it in.
    {
        "occurrence_id": "occ-no-risk", "recipe_id": "swing_house_v1",
        "outcome_definition_id": "house_default_v1", "analysis_unit": "swing",
        "entry_at": _et(SESSION, "13:00"), "entry_price": 60.0,
        "stop_price": None, "stop_distance": None,
        "mfe_r": None, "mae_r": None, "time_to_mfe_min": None,
        "first_hit": None, "first_hit_at": None,
        "r_at_eod": None, "result_state": "closed", "path_kind": "plain_no_target",
        "maturity_at": _et(SESSION, "16:00"),
    },
)

#: Quickest result over the rows above:
#:   hits    = occ-late, occ-fast                       -> 2
#:   unhit   = occ-unhit, occ-no-risk                   -> 2  (closed, no hit)
#:   pending = occ-open, occ-at-close                   -> 2
#:   unknown = occ-truncated                            -> 1
#:   hit rate = hits / MEASURED (hits + unhit) = 2 / 4  -> 0.50
#:   median trading minutes among hits = median(20, 30) -> 25.0
QUICK_HITS, QUICK_UNHIT, QUICK_PENDING, QUICK_UNKNOWN = 2, 2, 2, 1
QUICK_HIT_RATE = 0.50
QUICK_MEDIAN_TRADING_MINUTES = 25.0
#: What a wall-clock elapsed-minutes reader gets: median(1070, 30).
QUICK_MEDIAN_IF_WALL_CLOCK = 550.0
#: The hindsight fact, and it is a DIFFERENT number: median(300, 60).
SWING_TIME_TO_MFE_MEDIAN = 180.0

SWING_MAX_MFE_R = 3.00          # max(3.00, 1.00, 0.50)
SWING_WORST_MAE_R = -1.50       # min(-0.50, -1.50, -1.00)
SWING_MEAN_R_AT_EOD = 1.50      # mean(1.00, 2.00); occ-at-close excluded
SWING_R_AT_EOD_N = 2


class _FakeWarehouse:
    """The warehouse seam, with the two things the report needs from it.

    `source_paths` is not decoration: a cell that quotes the warehouse has to
    name the file its numbers came from, and the no-fabricated-citation test
    below opens every path the report prints.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], source_path: Path) -> None:
        self._rows = tuple(dict(row) for row in rows)
        self._source = source_path
        self.calls = 0

    @property
    def source_paths(self) -> tuple[str, ...]:
        return (str(self._source),)

    def read_outcomes(self, session_date: str, *, now: datetime | None = None):
        self.calls += 1
        return self._rows


class _UnreachableWarehouse:
    """`research_store_dir` unset, the DAS offline, a partition half-written."""

    source_paths: tuple[str, ...] = ()

    def read_outcomes(self, session_date: str, *, now: datetime | None = None):
        raise OSError("research_store_dir is not configured")


# ---------------------------------------------------------------------------
# the fixture tree
# ---------------------------------------------------------------------------


@pytest.fixture()
def tree(tmp_path: Path) -> dict[str, Path]:
    """Every store the report reads, on disk, in its live shape."""
    root = tmp_path / "stores"
    root.mkdir()
    paths = {
        "intraday_outcomes": root / "intraday_bounce_outcomes.csv",
        "session_horizon_outcomes": root / "master_avwap_session_horizon_outcomes.csv",
        "preference_report": root / "preference_trade_outcomes.csv",
        "journal_trades": root / "journal_trades.jsonl",
        "working_lately": root / "snapshot_latest.json",
        "market_theses": root / "market_theses.jsonl",
        "warehouse_partition": root / "outcome_path_2026.parquet",
        "digests": tmp_path / "digests",
        "export": tmp_path / "export",
    }
    paths["digests"].mkdir()
    paths["export"].mkdir()
    paths["intraday_outcomes"].write_text(_intraday_csv(INTRADAY_ROWS), encoding="utf-8")
    paths["session_horizon_outcomes"].write_text(_session_horizon_csv(), encoding="utf-8")
    paths["preference_report"].write_text(_preference_csv(), encoding="utf-8")
    paths["journal_trades"].write_text(
        "".join(json.dumps(trade, sort_keys=True) + "\n" for trade in JOURNAL_TRADES),
        encoding="utf-8",
    )
    paths["working_lately"].write_text(
        json.dumps({
            "schema": "working_lately_snapshot_v1",
            "snapshot_id": "9f2a1c4d6e8b0a35719d2c4e6f80a1b3c5d7e9f1",
            "as_of": SESSION,
            "cells": [],
        }),
        encoding="utf-8",
    )
    paths["market_theses"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in (
            {"thesis_id": "th-1", "kind": "thesis", "status": "open",
             "session_date": SESSION, "claim": "breadth is narrowing",
             "stance": "cautious", "invalidation": "a new high on volume"},
            {"thesis_id": "th-2", "kind": "thesis", "status": "closed",
             "session_date": PRIOR_2, "claim": "semis lead", "stance": "bullish",
             "invalidation": ""},
        )),
        encoding="utf-8",
    )
    # The partition the fake warehouse cites. A real file, because the report
    # prints its path and the citation test opens it.
    paths["warehouse_partition"].write_bytes(b"PAR1")
    return paths


def _sources(paths: Mapping[str, Path]):
    import measured_report

    return measured_report.ReportSources(
        intraday_outcomes=paths["intraday_outcomes"],
        session_horizon_outcomes=paths["session_horizon_outcomes"],
        preference_report=paths["preference_report"],
        journal_trades=paths["journal_trades"],
        working_lately=paths["working_lately"],
        market_theses=paths["market_theses"],
    )


def _build(paths: Mapping[str, Path], **kwargs):
    import measured_report

    kwargs.setdefault("now", NOW_A)
    kwargs.setdefault("sources", _sources(paths))
    kwargs.setdefault(
        "warehouse", _FakeWarehouse(WAREHOUSE_ROWS, paths["warehouse_partition"])
    )
    return measured_report.build_report(SESSION, **kwargs)


# ---------------------------------------------------------------------------
# 1. total profit - money once per trade, split the way the LEGS say
# ---------------------------------------------------------------------------


def test_total_profit_counts_a_trades_money_once_however_often_it_was_discussed(tree):
    """Two statements about one trade are two statements and one P&L.

    The expected number is not a literal typed twice: it is computed here by
    the REAL `trade_level_summary`, so this test and that contract cannot
    drift apart.
    """
    import preference_trade_outcomes

    expected = preference_trade_outcomes.trade_level_summary(_preference_row_dicts())
    assert expected["net_pnl"] == pytest.approx(TOTAL_PROFIT_ALL)
    assert expected["duplicate_statement_rows"] == 1

    report = _build(tree)
    cell = report.cell(CELL_TOTAL_PROFIT_ALL)
    assert cell.state == STATE_MEASURED
    assert cell.value == pytest.approx(TOTAL_PROFIT_ALL)
    assert cell.value != pytest.approx(TOTAL_PROFIT_IF_PER_STATEMENT)
    # `n` is at this cell's own grain: TRADES, not statements.
    assert cell.n == 4
    assert cell.unit.lower() in {"cad", "cad_dollars", "dollars_cad"}


def test_a_bought_put_is_bearish_money_even_though_its_direction_is_long(tree):
    """The bias split is the LEGS', not the `direction` column's."""
    report = _build(tree)
    bullish = report.cell(CELL_TOTAL_PROFIT_BULLISH)
    bearish = report.cell(CELL_TOTAL_PROFIT_BEARISH)
    bull_neutral = report.cell(CELL_TOTAL_PROFIT_BULL_NEUTRAL)

    assert bullish.value == pytest.approx(TOTAL_PROFIT_BULLISH)
    assert bearish.value == pytest.approx(TOTAL_PROFIT_BEARISH)
    assert bearish.value != pytest.approx(BEARISH_IF_BUCKETED_ON_DIRECTION)
    assert bull_neutral.value == pytest.approx(TOTAL_PROFIT_BULLISH_OR_NEUTRAL)
    assert bearish.n == 2 and bullish.n == 1 and bull_neutral.n == 1


def test_option_exposure_is_its_own_bucket_and_the_buckets_add_up_to_the_total(tree):
    report = _build(tree)
    stock = report.cell(CELL_TOTAL_PROFIT_STOCK)
    option = report.cell(CELL_TOTAL_PROFIT_OPTION)
    assert stock.value == pytest.approx(TOTAL_PROFIT_STOCK)
    assert option.value == pytest.approx(TOTAL_PROFIT_OPTION)
    assert stock.value + option.value == pytest.approx(TOTAL_PROFIT_ALL)

    by_bias = sum(
        report.cell(cell_id).value
        for cell_id in (
            CELL_TOTAL_PROFIT_BULLISH,
            CELL_TOTAL_PROFIT_BEARISH,
            CELL_TOTAL_PROFIT_BULL_NEUTRAL,
        )
    )
    assert by_bias == pytest.approx(TOTAL_PROFIT_ALL)


# ---------------------------------------------------------------------------
# 2. biggest opportunity - the percentage is measured, the R is not knowable
# ---------------------------------------------------------------------------


def test_a_day_row_with_no_recorded_risk_leaves_mfe_r_unknown_and_the_percent_measured(tree):
    report = _build(tree)
    pct = report.cell(CELL_DAY_MFE_PCT)
    r_cell = report.cell(CELL_DAY_MFE_R)

    assert pct.state == STATE_MEASURED
    assert pct.value == pytest.approx(DAY_MAX_MFE_PCT)
    assert pct.n == 3                       # AMD is empty, NFLX is the prior session
    assert pct.distinct_symbols == 3
    assert pct.distinct_sessions == 1

    assert r_cell.state == STATE_UNKNOWN
    assert r_cell.value is None
    assert r_cell.n == 0
    assert r_cell.unavailable.strip(), "an unknown cell names its reason"
    assert "risk" in r_cell.unavailable.lower()


def test_the_adverse_side_is_reported_beside_the_favorable_one(tree):
    report = _build(tree)
    mae = report.cell(CELL_DAY_MAE_PCT)
    assert mae.state == STATE_MEASURED
    assert mae.value == pytest.approx(DAY_WORST_MAE_PCT)
    assert mae.n == 3


def test_the_swing_opportunity_reads_r_only_where_a_stop_was_known(tree):
    report = _build(tree)
    mfe = report.cell(CELL_SWING_MFE_R)
    mae = report.cell(CELL_SWING_MAE_R)
    assert mfe.state == STATE_MEASURED
    assert mfe.value == pytest.approx(SWING_MAX_MFE_R)
    assert mae.value == pytest.approx(SWING_WORST_MAE_R)
    # occ-no-risk, occ-open, occ-at-close and occ-truncated carry no `mfe_r`.
    assert mfe.n == 3


def test_time_to_mfe_is_a_separate_hindsight_fact_and_not_the_speed_answer(tree):
    report = _build(tree)
    hindsight = report.cell(CELL_SWING_TIME_TO_MFE)
    speed = report.cell(CELL_QUICK_MEDIAN)
    assert hindsight.value == pytest.approx(SWING_TIME_TO_MFE_MEDIAN)
    assert speed.value == pytest.approx(QUICK_MEDIAN_TRADING_MINUTES)
    assert hindsight.value != pytest.approx(speed.value)


# ---------------------------------------------------------------------------
# 3. quickest result - trading minutes, and everything counted
# ---------------------------------------------------------------------------


def test_time_to_the_first_target_is_counted_in_trading_minutes_across_a_close(tree):
    """15:50 to 09:40 the next session is 20 trading minutes, not 1,070."""
    report = _build(tree)
    median = report.cell(CELL_QUICK_MEDIAN)
    assert median.state == STATE_MEASURED
    assert median.value == pytest.approx(QUICK_MEDIAN_TRADING_MINUTES)
    assert median.value != pytest.approx(QUICK_MEDIAN_IF_WALL_CLOCK)
    assert median.n == QUICK_HITS
    assert median.unit == "trading_minutes"


def test_unhit_targets_pending_rows_and_missing_bars_are_counted_never_dropped(tree):
    report = _build(tree)
    assert report.cell(CELL_QUICK_HITS).value == QUICK_HITS
    assert report.cell(CELL_QUICK_UNHIT).value == QUICK_UNHIT
    assert report.cell(CELL_QUICK_PENDING).value == QUICK_PENDING
    assert report.cell(CELL_QUICK_UNKNOWN).value == QUICK_UNKNOWN
    # Every observation is somewhere. Nothing was silently discarded.
    total = sum(
        report.cell(cell_id).value
        for cell_id in (CELL_QUICK_HITS, CELL_QUICK_UNHIT,
                        CELL_QUICK_PENDING, CELL_QUICK_UNKNOWN)
    )
    assert total == len(WAREHOUSE_ROWS)


def test_the_hit_rate_denominator_is_what_was_measured_not_what_was_observed(tree):
    """2 of the 4 MEASURED rows hit. Pending and unknown are shown, not assumed."""
    report = _build(tree)
    rate = report.cell(CELL_QUICK_HIT_RATE)
    assert rate.state == STATE_MEASURED
    assert rate.value == pytest.approx(QUICK_HIT_RATE)
    assert rate.n == QUICK_HITS + QUICK_UNHIT
    assert rate.value != pytest.approx(QUICK_HITS / len(WAREHOUSE_ROWS))


# ---------------------------------------------------------------------------
# 4. end of day - the stated close, its clock, and the entry that has none
# ---------------------------------------------------------------------------


def test_an_entry_at_the_close_has_no_same_session_end_of_day_measure(tree):
    import measured_report

    report = _build(tree)
    at_close = report.cell(CELL_EOD_SWING_AT_CLOSE)
    assert at_close.state == STATE_UNKNOWN
    assert at_close.value is None
    assert at_close.n == 1
    assert at_close.unavailable == measured_report.UNAVAILABLE_ENTRY_AT_CLOSE
    assert "close" in measured_report.UNAVAILABLE_ENTRY_AT_CLOSE.lower()

    # ... and it is excluded from the measured cell rather than zero-filled.
    measured = report.cell(CELL_EOD_SWING)
    assert measured.state == STATE_MEASURED
    assert measured.value == pytest.approx(SWING_MEAN_R_AT_EOD)
    assert measured.n == SWING_R_AT_EOD_N


def test_the_day_end_of_day_move_is_the_stored_side_adjusted_one(tree):
    report = _build(tree)
    cell = report.cell(CELL_EOD_DAY)
    assert cell.state == STATE_MEASURED
    assert cell.value == pytest.approx(DAY_MEAN_EOD_PCT)
    assert cell.value != pytest.approx(DAY_MEAN_EOD_PCT_IF_SIDE_IGNORED)
    assert cell.n == 3


def test_every_end_of_day_cell_names_the_exchange_clock_it_was_measured_on(tree):
    report = _build(tree)
    for cell_id in (CELL_EOD_SWING, CELL_EOD_DAY, CELL_EOD_SWING_AT_CLOSE):
        clock = report.cell(cell_id).reference_clock
        assert "New_York" in clock, (cell_id, clock)


def test_every_cell_declares_its_exit_policy_and_its_version(tree):
    report = _build(tree)
    for cell in report.cells():
        assert cell.population.strip(), cell.cell_id
        assert cell.version.strip(), cell.cell_id
        assert cell.state in {STATE_MEASURED, STATE_PENDING, STATE_UNKNOWN}, cell.cell_id
        if cell.state != STATE_MEASURED:
            assert cell.unavailable.strip(), cell.cell_id
    for cell_id in (CELL_EOD_SWING, CELL_EOD_DAY):
        assert report.cell(cell_id).exit_policy.strip(), cell_id


# ---------------------------------------------------------------------------
# 5. last day or two - two INDEPENDENT controls with exchange-session endpoints
# ---------------------------------------------------------------------------


def test_which_sessions_were_selected_and_how_long_they_were_followed_are_two_controls(tree):
    """Hold one, move the other, and only that one's answer changes."""
    wide_short = _build(tree, selection_sessions=4, follow_through_sessions=1)
    wide_long = _build(tree, selection_sessions=4, follow_through_sessions=3)
    narrow_short = _build(tree, selection_sessions=2, follow_through_sessions=1)

    a = wide_short.cell(CELL_LAST_FOLLOW_THROUGH)
    b = wide_long.cell(CELL_LAST_FOLLOW_THROUGH)
    c = narrow_short.cell(CELL_LAST_FOLLOW_THROUGH)

    assert (a.value, a.n) == (pytest.approx(SELECT_4_FT_1_MEAN), SELECT_4_FT_1_N)
    assert (b.value, b.n) == (pytest.approx(SELECT_4_FT_3_MEAN), SELECT_4_FT_3_N)
    assert (c.value, c.n) == (pytest.approx(SELECT_2_FT_1_MEAN), SELECT_2_FT_1_N)

    # The selection control changes the OBSERVATION count; the follow-through
    # control does not touch it.
    assert wide_short.cell(CELL_LAST_OBSERVATIONS).value == 4
    assert wide_long.cell(CELL_LAST_OBSERVATIONS).value == 4
    assert narrow_short.cell(CELL_LAST_OBSERVATIONS).value == 2

    assert wide_long.cell(CELL_LAST_FOLLOW_PENDING).value == SELECT_4_FT_3_PENDING


def test_the_selection_window_endpoints_are_exchange_sessions_never_calendar_days(tree):
    import market_calendar

    wide = _build(tree, selection_sessions=4, follow_through_sessions=1)
    narrow = _build(tree, selection_sessions=2, follow_through_sessions=1)

    # The literal, so a broken calendar walk is caught ...
    assert wide.cell(CELL_LAST_OBSERVATIONS).window == (PRIOR_3, SESSION)
    assert narrow.cell(CELL_LAST_OBSERVATIONS).window == (PRIOR_1, SESSION)
    # ... and the calendar, so a hard-coded literal is caught.
    cursor = date.fromisoformat(SESSION)
    for _ in range(3):
        cursor = market_calendar.previous_session(cursor)
    assert cursor.isoformat() == PRIOR_3

    # The 40.00 observation two sessions earlier never entered the mean.
    assert wide.cell(CELL_LAST_FOLLOW_THROUGH).value != pytest.approx(
        SELECT_4_FT_1_MEAN_IF_CALENDAR_DAYS
    )


def test_a_follow_through_horizon_the_desk_never_publishes_is_unknown_not_zero(tree):
    """`SCAN_FACTOR_HORIZONS` is (1, 3, 5, 10): no 2-session row is ever built."""
    report = _build(tree, selection_sessions=4, follow_through_sessions=2)
    cell = report.cell(CELL_LAST_FOLLOW_THROUGH)
    assert cell.state == STATE_UNKNOWN
    assert cell.value is None
    assert cell.n == 0
    assert cell.unavailable.strip()
    # The SELECTION control still answered - one missing horizon is not a
    # missing report.
    assert report.cell(CELL_LAST_OBSERVATIONS).value == 4


# ---------------------------------------------------------------------------
# 6. the warehouse is allowed to be missing
# ---------------------------------------------------------------------------


def test_an_unreachable_warehouse_leaves_the_swing_cells_unknown_and_still_builds(tree):
    report = _build(tree, warehouse=_UnreachableWarehouse())
    for cell_id in SWING_CELL_IDS:
        cell = report.cell(cell_id)
        assert cell.state == STATE_UNKNOWN, cell_id
        assert cell.value is None, cell_id
        assert cell.unavailable.strip(), cell_id
    # The reason is the real one, not a generic shrug.
    assert "research_store_dir" in report.cell(CELL_SWING_MFE_R).unavailable

    # Everything that does not need the warehouse is still measured.
    assert report.cell(CELL_DAY_MFE_PCT).value == pytest.approx(DAY_MAX_MFE_PCT)
    assert report.cell(CELL_TOTAL_PROFIT_ALL).value == pytest.approx(TOTAL_PROFIT_ALL)
    assert report.report_id


def test_with_no_warehouse_configured_at_all_the_report_is_still_published(tree):
    """The default path on a desk where `research_store_dir` was never set."""
    import measured_report

    report = measured_report.build_report(
        SESSION, now=NOW_A, sources=_sources(tree)
    )
    assert report.report_id
    for cell_id in SWING_CELL_IDS:
        assert report.cell(cell_id).state == STATE_UNKNOWN, cell_id
    assert report.cell(CELL_EOD_DAY).state == STATE_MEASURED


def test_missing_evidence_is_a_section_that_names_what_was_not_measured(tree):
    report = _build(tree, warehouse=_UnreachableWarehouse())
    assert tuple(report.sections) == SECTION_NAMES
    missing = report.sections["missing_evidence"]
    named = {cell.cell_id for cell in missing}
    for cell_id in SWING_CELL_IDS:
        assert cell_id in named, cell_id


def test_an_all_pending_real_shaped_warehouse_cohort_stays_pending_not_unknown(tree):
    """Open occurrence rows are evidence still maturing, not unavailable evidence."""
    pending_rows = tuple(
        row for row in WAREHOUSE_ROWS if row["result_state"] == "open"
    )
    report = _build(
        tree,
        warehouse=_FakeWarehouse(pending_rows, tree["warehouse_partition"]),
    )

    hit_rate = report.cell(CELL_QUICK_HIT_RATE)
    speed = report.cell(CELL_QUICK_MEDIAN)
    assert hit_rate.state == STATE_PENDING
    assert speed.state == STATE_PENDING
    assert hit_rate.value is None
    assert speed.value is None
    assert not hit_rate.unavailable
    assert not speed.unavailable


# ---------------------------------------------------------------------------
# 7. one report id - stable under a clock, moved by evidence
# ---------------------------------------------------------------------------


def test_the_report_id_does_not_move_when_only_the_clock_moves(tree):
    first = _build(tree, now=NOW_A)
    second = _build(tree, now=NOW_B)
    assert first.as_of == second.as_of
    assert first.report_id == second.report_id


def test_a_cell_that_matured_changes_the_report_id(tree):
    before = _build(tree, now=NOW_A)
    rows = tuple(row for row in INTRADAY_ROWS if not row[0].startswith("AMD_"))
    tree["intraday_outcomes"].write_text(
        _intraday_csv(rows + (INTRADAY_MATURED_AMD,)), encoding="utf-8"
    )
    after = _build(tree, now=NOW_A)
    assert after.cell(CELL_DAY_MFE_PCT).n == 4
    assert after.report_id != before.report_id


def test_the_report_id_is_a_sha1_over_the_cells_and_the_as_of_alone(tree):
    report = _build(tree)
    digest = hashlib.sha1()
    for line in sorted(f"{cell.cell_id}={cell.value!r}={cell.state}"
                       for cell in report.cells()):
        digest.update(line.encode("utf-8"))
        digest.update(b"\x1e")
    digest.update(report.as_of.encode("utf-8"))
    assert report.report_id == digest.hexdigest()


# ---------------------------------------------------------------------------
# 8. publishing - the slot, the version files, the entry index
# ---------------------------------------------------------------------------


def _run_slot(tree, **kwargs):
    from ai_jobs import measured_report_publish

    kwargs.setdefault("session_date", SESSION)
    kwargs.setdefault("now", NOW_A)
    kwargs.setdefault("root", tree["digests"])
    kwargs.setdefault("sources", _sources(tree))
    kwargs.setdefault(
        "warehouse", _FakeWarehouse(WAREHOUSE_ROWS, tree["warehouse_partition"])
    )
    return measured_report_publish.run_measured_report(**kwargs)


def test_the_slot_writes_the_report_and_its_markdown_sibling_under_the_digest_root(tree):
    outcome = _run_slot(tree)
    json_path = tree["digests"] / f"measured_report_{SESSION}.json"
    md_path = tree["digests"] / f"measured_report_{SESSION}.md"
    assert json_path.exists() and md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["report_id"] == outcome["report_id"]
    assert payload["session_date"] == SESSION
    assert {cell["cell_id"] for cell in payload["cells"]} >= {
        CELL_TOTAL_PROFIT_ALL, CELL_DAY_MFE_PCT, CELL_QUICK_MEDIAN, CELL_EOD_DAY,
        CELL_LAST_FOLLOW_THROUGH,
    }


def test_a_rerun_on_matured_data_writes_a_new_version_and_never_rewrites_the_old(tree):
    _run_slot(tree)
    first = tree["digests"] / f"measured_report_{SESSION}.json"
    original = first.read_bytes()

    # Same evidence: no new version, no rewrite.
    _run_slot(tree)
    assert sorted(p.name for p in tree["digests"].glob("measured_report_*.json")) == [
        f"measured_report_{SESSION}.json"
    ]
    assert first.read_bytes() == original

    # Matured evidence: a NEW file beside the old one.
    rows = tuple(row for row in INTRADAY_ROWS if not row[0].startswith("AMD_"))
    tree["intraday_outcomes"].write_text(
        _intraday_csv(rows + (INTRADAY_MATURED_AMD,)), encoding="utf-8"
    )
    _run_slot(tree)
    second = tree["digests"] / f"measured_report_{SESSION}_v2.json"
    assert second.exists()
    assert (tree["digests"] / f"measured_report_{SESSION}_v2.md").exists()
    assert first.read_bytes() == original
    assert json.loads(second.read_text(encoding="utf-8"))["report_id"] != \
        json.loads(original.decode("utf-8"))["report_id"]


def test_a_failed_markdown_sibling_removes_its_json_and_a_rerun_can_repair(tree, monkeypatch):
    """A report is published as a JSON/Markdown pair, or not at all."""
    from ai_jobs import digest

    original_publish = digest._publish

    def _fail_markdown(path, content):
        if Path(path).suffix == ".md":
            raise OSError("simulated markdown sibling failure")
        return original_publish(path, content)

    monkeypatch.setattr(digest, "_publish", _fail_markdown)
    failed = _run_slot(tree)
    json_path = tree["digests"] / f"measured_report_{SESSION}.json"
    markdown_path = tree["digests"] / f"measured_report_{SESSION}.md"
    assert failed["status"] != "ok"
    assert not json_path.exists()
    assert not markdown_path.exists()

    monkeypatch.setattr(digest, "_publish", original_publish)
    repaired = _run_slot(tree)
    assert repaired["status"] == "ok"
    assert json_path.exists()
    assert markdown_path.exists()


def test_the_entry_index_names_the_newest_report_without_touching_its_four_sections(tree):
    from ai_jobs import digest

    _run_slot(tree)
    rows = tuple(row for row in INTRADAY_ROWS if not row[0].startswith("AMD_"))
    tree["intraday_outcomes"].write_text(
        _intraday_csv(rows + (INTRADAY_MATURED_AMD,)), encoding="utf-8"
    )
    outcome = _run_slot(tree)

    index = digest.build_entry_index(tree["digests"], as_of=SESSION)
    # The four sections are a published contract and are NOT extended here.
    assert digest.ENTRY_INDEX_SECTIONS == (
        "intraday_held_run", "swing_win_rates", "preference_observations",
        "journal_execution",
    )
    section = index["measured_report"]
    assert section["report_id"] == outcome["report_id"]
    assert section["as_of"]
    assert Path(section["path"]).name == f"measured_report_{SESSION}_v2.json"
    assert Path(section["path"]).exists()
    assert Path(section["markdown_path"]).name == f"measured_report_{SESSION}_v2.md"


def test_a_failed_report_never_fails_the_night(tree, monkeypatch):
    import measured_report

    def _boom(*args, **kwargs):
        raise RuntimeError("the preference report was half-written")

    monkeypatch.setattr(measured_report, "build_report", _boom)
    outcome = _run_slot(tree)
    assert isinstance(outcome, dict)
    assert outcome.get("status") not in (None, "ok")
    assert str(outcome.get("reason") or "").strip()
    assert not list(tree["digests"].glob("measured_report_*.json"))


def test_the_slot_sits_at_the_end_of_the_deterministic_stage(tree):
    """Decision 0018's stage order, with this packet appended INSIDE stage 1."""
    from ai_jobs import runner

    spec = importlib.util.spec_from_file_location(
        "_ws_rp_runner_pin", ROOT / "tests" / "test_ai_jobs_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = tuple(module.EXPECTED_SLOT_ORDER)

    assert "measured_report" in expected
    assert expected.index("measured_report") == expected.index("market_story_rollups") + 1
    assert expected.index("measured_report") == expected.index("day_review_facts") - 1
    # S12 (2026-09-26): the SP4 family evidence closes stage 1 after the facts.
    assert expected.index("day_review_facts") == expected.index("family_side_evidence") - 1
    # S15 (2026-09-26): swing path facts follow it and now close stage 1.
    assert expected.index("family_side_evidence") == expected.index("swing_path_facts") - 1
    assert expected.index("swing_path_facts") == expected.index("ai_summary") - 1
    # The existing pin is not weakened: the real slate still equals it exactly.
    assert tuple(slot.name for slot in runner.default_slots()) == expected


def test_publishing_the_report_never_writes_a_frontier_handoff(tree):
    """Export is the trader's click. The night publishes the report, nothing else."""
    _run_slot(tree)
    assert not list(tree["digests"].rglob("frontier_handoff_*"))
    assert not list(tree["digests"].rglob("manifest.json"))


# ---------------------------------------------------------------------------
# 9. the frontier handoff
# ---------------------------------------------------------------------------


def _handoff(tree, **kwargs):
    import measured_report

    report = kwargs.pop("report", None) or _build(tree)
    kwargs.setdefault("sources", _sources(tree))
    return report, measured_report.build_handoff(report, **kwargs)


def test_the_brief_is_capped_at_32_kib_of_utf8(tree):
    import measured_report

    assert measured_report.HANDOFF_MARKDOWN_CAP_BYTES == 32 * 1024
    _report, handoff = _handoff(tree)
    assert len(handoff.markdown.encode("utf-8")) <= measured_report.HANDOFF_MARKDOWN_CAP_BYTES


def test_a_brief_that_had_to_drop_cells_says_how_many_and_out_of_how_many(tree):
    report, handoff = _handoff(tree, cap_bytes=1500)
    encoded = handoff.markdown.encode("utf-8")
    assert len(encoded) <= 1500

    total = len(report.cells())
    present = sum(1 for cell in report.cells() if cell.cell_id in handoff.markdown)
    assert 0 < present < total, "the cap has to actually bite for this to mean anything"

    match = re.search(r"omitted\s+(\d+)\s+of\s+(\d+)\s+cell", handoff.markdown)
    assert match, handoff.markdown[-400:]
    assert int(match.group(1)) == total - present
    assert int(match.group(2)) == total
    assert handoff.manifest["omitted_cells"] == total - present


def test_the_manifest_measures_the_brief_with_the_ratio_the_ai_layer_already_uses(tree):
    """There is no tokenizer on this desk - `ai_summary._ESTIMATED_CHARS_PER_TOKEN`
    (2.5, measured 2026-08-28) is the whole of its token arithmetic, and the
    manifest uses THAT rather than inventing a second one."""
    import ai_summary

    _report, handoff = _handoff(tree)
    manifest = handoff.manifest
    assert manifest["chars_per_token"] == ai_summary._ESTIMATED_CHARS_PER_TOKEN
    assert manifest["markdown_bytes"] == len(handoff.markdown.encode("utf-8"))
    expected = len(handoff.markdown) / ai_summary._ESTIMATED_CHARS_PER_TOKEN
    assert abs(manifest["markdown_tokens_estimated"] - expected) <= 1


def test_the_manifest_carries_the_working_lately_snapshot_id_verbatim(tree):
    _report, handoff = _handoff(tree)
    expected = json.loads(tree["working_lately"].read_text(encoding="utf-8"))["snapshot_id"]
    assert handoff.manifest["tracker_snapshot_id"] == expected


def test_the_manifest_carries_the_open_theses_and_leaves_the_closed_ones_out(tree):
    _report, handoff = _handoff(tree)
    theses = handoff.manifest["open_theses"]
    ids = {str(row.get("thesis_id")) for row in theses}
    assert ids == {"th-1"}


def test_no_citation_in_the_report_points_at_a_file_that_does_not_exist(tree):
    report = _build(tree)
    cited = {source for cell in report.cells() for source in cell.sources}
    assert cited, "a measured report that cites nothing is not evidence"
    for source in cited:
        assert Path(source).exists(), source
    # The warehouse-backed cells name the partition they were read from.
    assert str(tree["warehouse_partition"]) in report.cell(CELL_SWING_MFE_R).sources
    for cell in report.cells():
        if cell.state == STATE_MEASURED:
            assert cell.sources, cell.cell_id


def test_the_manifest_lists_only_source_paths_that_exist(tree):
    _report, handoff = _handoff(tree)
    for source in handoff.manifest["source_paths"]:
        assert Path(source).exists(), source


def test_a_best_and_worst_table_carries_its_full_denominator_and_says_it_is_selected(tree):
    _report, handoff = _handoff(tree)
    tables = handoff.payload["example_tables"]
    assert tables, "10K asks for best/worst examples"
    for table in tables:
        assert table["label"] == "retrospective, result-selected"
        assert int(table["denominator"]) >= len(table["rows"])
        assert int(table["denominator"]) > 0


# ---------------------------------------------------------------------------
# 10. two views, one report id
# ---------------------------------------------------------------------------


@pytest.fixture()
def panel(qapp_or_skip):
    from ui.panels.daily_recap_panel import DailyRecapPanel

    widget = DailyRecapPanel()
    yield widget
    widget.deleteLater()


@pytest.fixture(scope="session")
def qapp_or_skip():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


def test_the_daily_recap_gains_a_review_tab_after_its_four_views(panel):
    from ui.panels.daily_recap_panel import REVIEW_TAB_TITLE, VIEW_ORDER

    # The four views are untouched: the Review tab is added, not swapped in.
    assert len(VIEW_ORDER) == 4
    assert panel.tabs.count() == 5
    assert panel.tabs.tabText(4) == REVIEW_TAB_TITLE


def test_the_review_tab_prints_the_same_report_id_the_export_carries(tree, panel):
    report, handoff = _handoff(tree)
    panel.render_report(report)
    assert report.report_id in panel.report_id_label.text()
    assert handoff.payload["report_id"] == report.report_id
    assert handoff.manifest["report_id"] == report.report_id


def test_what_the_tab_shows_and_what_the_export_carries_are_the_same_cells(tree, panel):
    report, handoff = _handoff(tree)
    panel.render_report(report)
    shown = tuple(panel.review_cells())
    exported = tuple((cell["cell_id"], cell["value"]) for cell in handoff.payload["cells"])
    assert shown == exported
    assert len(shown) == len(report.cells())


def test_the_review_area_says_so_when_no_local_review_has_been_written_yet(tree, panel):
    from ui.panels.daily_recap_panel import NO_REVIEW_YET

    report = _build(tree)
    panel.render_report(report, narration=None)
    assert panel.review_note.text().strip() == NO_REVIEW_YET

    panel.render_report(report, narration=f"{CELL_QUICK_MEDIAN} is the slow half.")
    assert CELL_QUICK_MEDIAN in panel.review_note.text()
    # The narration did not become the numbers.
    assert tuple(panel.review_cells()) == tuple(
        (cell["cell_id"], cell["value"])
        for cell in _handoff(tree, report=report)[1].payload["cells"]
    )


def test_export_handoff_writes_the_brief_the_payload_and_the_manifest(tree, panel):
    report = _build(tree)
    panel.render_report(report)
    written = panel.export_handoff(tree["export"])

    names = sorted(p.name for p in Path(tree["export"]).iterdir())
    assert names == sorted([
        f"frontier_handoff_{SESSION}.json",
        f"frontier_handoff_{SESSION}.md",
        "manifest.json",
    ])
    assert set(written) == {"markdown", "payload", "manifest"}
    manifest = json.loads((Path(tree["export"]) / "manifest.json").read_text("utf-8"))
    assert manifest["report_id"] == report.report_id


def test_copy_handoff_puts_the_brief_on_the_clipboard_and_nothing_on_disk(tree, panel, qapp_or_skip):
    report, handoff = _handoff(tree)
    panel.render_report(report)
    panel.copy_handoff_button.click()
    assert qapp_or_skip.clipboard().text() == handoff.markdown
    assert not list(Path(tree["export"]).iterdir())


def test_nothing_in_the_report_or_the_handoff_opens_a_socket(tree, panel, monkeypatch):
    """User-initiated means no model call and no upload - on any path."""

    def _no_network(*args, **kwargs):
        raise AssertionError("the measured report reached the network")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)

    report = _build(tree)
    panel.render_report(report)
    panel.export_handoff(tree["export"])
    panel.copy_handoff_button.click()
    _run_slot(tree)


# ---------------------------------------------------------------------------
# 11. N3's narration selection is not touched by any of this
# ---------------------------------------------------------------------------

#: The commit this branch was cut from. N3's selection key is a refusal
#: (gate #43): no R statistic may enter it, and this packet has no business
#: anywhere near it.
N3_BASE_COMMIT = "68b1fd7b"

N3_PINNED_FILES = (
    "tests/test_n3_narration_bounded.py",
    "tests/test_n3_narration_bounded_edges.py",
)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def test_n3s_narration_tests_are_byte_identical_to_the_branch_point():
    for relative in N3_PINNED_FILES:
        base = _git("show", f"{N3_BASE_COMMIT}:{relative}")
        now = (ROOT / relative).read_text(encoding="utf-8")
        assert now == base, relative


def test_the_bounded_narration_view_itself_is_unchanged():
    import ast
    import inspect

    from ai_jobs import setup_research

    base_source = _git("show", f"{N3_BASE_COMMIT}:scripts/ai_jobs/setup_research.py")
    tree_ = ast.parse(base_source)
    base_lines = base_source.splitlines()
    wanted = None
    for node in ast.walk(tree_):
        if isinstance(node, ast.FunctionDef) and node.name == "_bounded_narration_view":
            wanted = "\n".join(base_lines[node.lineno - 1:node.end_lineno])
            break
    assert wanted is not None, "the base commit has no `_bounded_narration_view`"

    current = inspect.getsource(setup_research._bounded_narration_view)
    assert current.strip() == wanted.strip()
