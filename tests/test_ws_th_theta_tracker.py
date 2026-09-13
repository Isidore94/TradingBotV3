"""Packet WS-TH - a Setup Tracker for the theta plays (WISHLIST item 6).

Written RED, before `scripts/theta_pick_tracker.py` exists. Every row a test
records here comes out of `legacy.evaluate_theta_put_candidate` /
`evaluate_theta_pcs_candidate` and through `_apply_best_option_to_theta_row` -
the two builders the scan itself calls - so the recorder is exercised against
the fields the scan really produces rather than a hand-written dict that agrees
with the recorder by construction.

**What the real rows taught this file, and why each matters.**

* `SMA_20` is BUILT as a theta support and then DROPPED:
  `_is_valid_theta_support_entry` (legacy.py:21082) refuses it. A fixture that
  lists SMA_20/50/100/200 and expects four supports is describing a scan that
  does not exist, and a three-support name built only from SMA_20/50/100 fails
  the `THETA_MIN_SUPPORT_LEVELS` floor and is not a pick at all.
* A support may sit ABOVE the close - `_theta_support_entry` keeps a level up to
  `THETA_SUPPORT_ABOVE_TOL_ATR` (0.05 ATR) overhead - and it CLAMPS that
  negative distance to `distance_atr: 0.0`. So `held` cannot be read off
  `distance_atr`: a level 0.05 above price and a level exactly at price both
  record 0.0, and only `level <= close` separates them.
* A sold-put row's option carries `strike`; a PCS row's carries `short_strike`.
  A recorder that reads `strike` on both writes `None` for every credit spread.
* `_apply_best_option_to_theta_row` REPLACES `score` with the option's
  `rank_score` and keeps the support score as `base_score`. The number the
  report ranks on - and so the number a grade line must grade - is the former.
* IB expirations arrive as `YYYYMMDD` (`_format_option_expiration`), never ISO.

**The calendar is the exchange calendar, and the test says the dates out loud.**
A scan on Monday 2026-06-01 has its 20th session on **2026-06-30**, because
Juneteenth (Friday 2026-06-19) is not a session. Twenty business days lands on
2026-06-29. The fixtures put a BROKEN close on 06-29 and a HELD close on 06-30,
so a grader that counts weekdays fails on the number.

Shadow only: nothing here reaches the theta scan, the theta score or the theta
report.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import market_calendar  # noqa: E402
from evidence_stats import MIN_REPORTABLE_N  # noqa: E402
from swing_headline import wilson_lower_bound  # noqa: E402


SCAN_DATE = date(2026, 6, 1)
#: The exact exchange-session endpoints for a 2026-06-01 scan. Juneteenth
#: (2026-06-19) is skipped, which is the whole reason 20 sessions is 06-30 and
#: not the 06-29 a business-day walk would return.
SESSION_5 = date(2026, 6, 8)
SESSION_10 = date(2026, 6, 15)
SESSION_19 = date(2026, 6, 29)
SESSION_20 = date(2026, 6, 30)


# ---------------------------------------------------------------------------
# Real theta rows, from the scan's own builders
# ---------------------------------------------------------------------------


def _daily_frame(bars: int = 12, close: float = 100.0) -> pd.DataFrame:
    """A frame too short for a trendline pivot, so the fixture is deterministic."""
    rows = []
    for index in range(bars):
        rows.append(
            {
                "datetime": pd.Timestamp(2026, 5, 1) + pd.Timedelta(days=index),
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000,
            }
        )
    return pd.DataFrame(rows)


def _put_row(
    symbol: str,
    *,
    current_anchor=None,
    previous_anchor=None,
    indicators=None,
    option=None,
    unavailable_reason: str = "",
) -> dict:
    from master_avwap_lib import legacy

    row = legacy.evaluate_theta_put_candidate(
        symbol=symbol,
        side="LONG",
        df=_daily_frame(),
        last_trade_date=SCAN_DATE,
        last_close=100.0,
        atr20=2.0,
        current_anchor_meta=current_anchor,
        previous_anchor_meta=previous_anchor,
        indicator_row=indicators
        or {"sma_20": 99.0, "sma_50": 97.0, "sma_100": 95.0, "sma_200": 94.0},
        compression_summary=None,
        recent_earnings_dates=[],
        upcoming_earnings_dates=[],
    )
    assert row is not None, f"{symbol} did not survive the real theta gate"
    legacy._apply_best_option_to_theta_row(
        row, [option] if option else [], unavailable_reason
    )
    return row


def _pcs_row(symbol: str, *, option=None) -> dict:
    from master_avwap_lib import legacy

    row = legacy.evaluate_theta_pcs_candidate(
        symbol=symbol,
        side="LONG",
        df=_daily_frame(),
        last_trade_date=SCAN_DATE,
        last_close=100.0,
        atr20=2.0,
        current_anchor_meta=None,
        previous_anchor_meta=None,
        indicator_row={"sma_20": 99.0, "sma_50": 97.0, "sma_100": 95.0, "sma_200": 94.0},
        compression_summary=None,
        recent_earnings_dates=[],
        upcoming_earnings_dates=[],
    )
    assert row is not None
    legacy._apply_best_option_to_theta_row(row, [option] if option else [])
    return row


SOLD_PUT_OPTION = {
    "strike": 95.0,
    "expiration": "20260717",
    "credit": 1.25,
    "status": "recommended",
    "rank_score": 87.4,
}
PCS_OPTION = {
    "short_strike": 94.0,
    "long_strike": 89.0,
    "expiration": "20260717",
    "credit": 1.10,
    "status": "recommended",
    "rank_score": 70.0,
}


def _stacked_row() -> dict:
    """AAA: nine supports, one of them 0.05 ABOVE the close.

    `CURRENT_UPPER_1` at 100.05 against a 100.00 close is inside the 0.05-ATR
    overhead tolerance, so the scan keeps it - with `distance_atr` clamped to
    0.0, exactly like a level sitting on price.
    """
    return _put_row(
        "AAA",
        current_anchor={"vwap": 98.0, "bands": {"LOWER_1": 96.0, "UPPER_1": 100.05}},
        previous_anchor={"vwap": 96.5, "bands": {"LOWER_1": 95.5, "UPPER_1": 98.5}},
        option=SOLD_PUT_OPTION,
    )


def _sma_row(symbol: str = "BBB", *, option=SOLD_PUT_OPTION) -> dict:
    """BBB: SMA_50 97.00, SMA_100 95.00, SMA_200 94.00. SMA_20 is dropped."""
    return _put_row(symbol, option=option)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


NOW = datetime(2026, 6, 1, 16, 30, 0)


# ---------------------------------------------------------------------------
# Item 1 - the recorder
# ---------------------------------------------------------------------------


def test_the_store_and_the_outcome_file_sit_where_the_packet_puts_them():
    import project_paths
    import theta_pick_tracker

    assert theta_pick_tracker.THETA_PICKS_FILE.name == "theta_picks.jsonl"
    assert (
        project_paths.MASTER_AVWAP_THETA_OUTCOMES_FILE.name
        == "master_avwap_theta_outcomes.csv"
    )
    # "beside master_avwap_tier_outcomes.csv" - the packet's word, so the same
    # directory and not a second runtime root.
    assert (
        project_paths.MASTER_AVWAP_THETA_OUTCOMES_FILE.parent
        == project_paths.MASTER_AVWAP_TIER_OUTCOMES_FILE.parent
    )


def test_a_scans_theta_rows_become_one_store_row_each_with_the_scans_own_numbers(tmp_path):
    """One row per (symbol, scan_date, play_type), off the real scan rows."""
    import theta_pick_tracker

    store = tmp_path / "theta_picks.jsonl"
    theta_pick_tracker.record_theta_picks(
        [_stacked_row(), _sma_row()],
        [_pcs_row("CCC", option=PCS_OPTION)],
        SCAN_DATE,
        NOW,
        path=store,
    )
    rows = _read_jsonl(store)
    assert len(rows) == 3
    by_symbol = {row["symbol"]: row for row in rows}
    assert set(by_symbol) == {"AAA", "BBB", "CCC"}

    bbb = by_symbol["BBB"]
    assert bbb["scan_date"] == "2026-06-01"
    assert bbb["play_type"] == "sold_put"
    # Report order, 1-based, per play_type list: AAA then BBB.
    assert by_symbol["AAA"]["rank"] == 1
    assert bbb["rank"] == 2
    assert by_symbol["CCC"]["rank"] == 1
    # `_apply_best_option_to_theta_row` REPLACES the support score with the
    # option rank score (87.4 -> 87); 35 is the support score it keeps as
    # `base_score`. The report ranks on 87, so the store records 87.
    assert bbb["score"] == 87
    assert bbb["atr"] == pytest.approx(2.0)
    assert bbb["close"] == pytest.approx(100.0)
    assert bbb["strike"] == pytest.approx(95.0)
    assert bbb["premium"] == pytest.approx(1.25)
    # IB hands back YYYYMMDD; the store keeps an ISO session date so a grader
    # can ask the calendar about it without re-parsing a broker format.
    assert bbb["expiry"] == "2026-07-17"
    assert bbb["first_seen_scan_date"] == "2026-06-01"

    # THE SUPPORT SET, as the scan built it. Lexicographic, not numeric: the
    # combo is a KEY, and a key sorted by level would collide across names.
    assert bbb["support_combo"] == "SMA_100+SMA_200+SMA_50"
    supports = {entry["label"]: entry for entry in bbb["supports"]}
    assert set(supports) == {"SMA_50", "SMA_100", "SMA_200"}
    assert supports["SMA_50"]["source"] == "sma"
    assert supports["SMA_50"]["distance_atr"] == pytest.approx(1.5)
    assert supports["SMA_200"]["distance_atr"] == pytest.approx(3.0)
    assert supports["SMA_50"]["held"] is True

    # A PCS row's option is a SPREAD: the sold leg is `short_strike`, and a
    # recorder that reads `strike` writes None for every credit spread.
    ccc = by_symbol["CCC"]
    assert ccc["play_type"] == "pcs"
    assert ccc["strike"] == pytest.approx(94.0)
    assert ccc["premium"] == pytest.approx(1.10)


def test_a_support_above_the_close_is_recorded_as_not_held(tmp_path):
    """`distance_atr` is CLAMPED to 0.0 overhead, so `held` must read the level.

    `CURRENT_UPPER_1` sits at 100.05 against a 100.00 close - inside the
    0.05-ATR tolerance `_theta_support_entry` allows - and comes out of the scan
    with `distance_atr: 0.0`, the same value a level exactly on price gets.
    Deriving `held` from the distance marks it held.
    """
    import theta_pick_tracker

    store = tmp_path / "theta_picks.jsonl"
    theta_pick_tracker.record_theta_picks([_stacked_row()], [], SCAN_DATE, NOW, path=store)
    supports = {entry["label"]: entry for entry in _read_jsonl(store)[0]["supports"]}
    assert supports["CURRENT_UPPER_1"]["distance_atr"] == pytest.approx(0.0)
    assert supports["CURRENT_UPPER_1"]["held"] is False
    assert supports["CURRENT_AVWAPE"]["held"] is True


def test_recording_the_same_scan_twice_writes_nothing_the_second_time(tmp_path):
    """A scan that reruns - or a deferred option pass - must not double the n."""
    import theta_pick_tracker

    store = tmp_path / "theta_picks.jsonl"
    for _ in range(2):
        theta_pick_tracker.record_theta_picks(
            [_stacked_row(), _sma_row()], [], SCAN_DATE, NOW, path=store
        )
    rows = _read_jsonl(store)
    assert len(rows) == 2
    assert sorted(row["symbol"] for row in rows) == ["AAA", "BBB"]


def test_a_repeat_appearance_is_its_own_row_and_keeps_the_first_seen_date(tmp_path):
    """Every appearance is an observation; the cohort grain is the FIRST one."""
    import theta_pick_tracker

    store = tmp_path / "theta_picks.jsonl"
    theta_pick_tracker.record_theta_picks([_sma_row()], [], SCAN_DATE, NOW, path=store)
    second = date(2026, 6, 2)
    theta_pick_tracker.record_theta_picks(
        [_sma_row()], [], second, datetime(2026, 6, 2, 16, 30), path=store
    )
    rows = _read_jsonl(store)
    assert [row["scan_date"] for row in rows] == ["2026-06-01", "2026-06-02"]
    assert [row["first_seen_scan_date"] for row in rows] == [
        "2026-06-01",
        "2026-06-01",
    ]


def test_a_recorder_that_cannot_write_never_raises_into_the_scan(tmp_path):
    """Evidence stores are never allowed to cost the thing they record.

    The store path is an existing DIRECTORY, so every open() for append fails.
    The scan's next statement must still run.
    """
    import theta_pick_tracker

    blocked = tmp_path / "theta_picks.jsonl"
    blocked.mkdir()
    theta_pick_tracker.record_theta_picks([_sma_row()], [], SCAN_DATE, NOW, path=blocked)


def test_a_malformed_row_in_the_list_never_raises_into_the_scan(tmp_path):
    import theta_pick_tracker

    store = tmp_path / "theta_picks.jsonl"
    theta_pick_tracker.record_theta_picks(
        [None, "not a row", _sma_row()], [], SCAN_DATE, NOW, path=store
    )
    rows = _read_jsonl(store)
    assert [row["symbol"] for row in rows] == ["BBB"]


def test_the_scan_records_the_picks_right_after_it_writes_the_theta_report():
    """The hook is in the RUNNER, after `write_theta_put_report` (ruling (c)).

    A wiring pin, read off the parsed module rather than its text: the recorder
    lives in the scan's own output pass, so a theta report written without a
    store row is a defect this catches. `legacy.py` is untouched.
    """
    source = (ROOT / "scripts" / "master_avwap_lib" / "runner.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    impl = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_master_impl"
    )
    calls: list[tuple[int, str]] = []
    for node in ast.walk(impl):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name:
                calls.append((node.lineno, name))
    calls.sort()
    names = [name for _lineno, name in calls]
    assert "record_theta_picks" in names, "the scan never records its theta picks"
    assert names.index("record_theta_picks") > names.index("write_theta_put_report")


# ---------------------------------------------------------------------------
# Item 2 - the grader
# ---------------------------------------------------------------------------


def _sessions_after(start: date, count: int) -> list[date]:
    out: list[date] = []
    cursor = start
    while len(out) < count:
        cursor = cursor + timedelta(days=1)
        if market_calendar.is_session(cursor):
            out.append(cursor)
    return out


def _bars(
    overrides: dict[date, dict] | None = None,
    *,
    default_close: float = 101.0,
    default_low: float = 100.5,
    count: int = 30,
) -> dict[date, dict]:
    """Completed daily bars for the sessions after the scan date.

    A bar's four prices keep their invariant: a session given a close below the
    default low gets that close as its low, so no fixture bar is one the desk
    would draw dashed.
    """
    overrides = overrides or {}
    bars: dict[date, dict] = {}
    for day in _sessions_after(SCAN_DATE, count):
        override = overrides.get(day, {})
        close = float(override.get("close", default_close))
        low = float(override.get("low", min(default_low, close)))
        bars[day] = {"close": close, "low": low, "high": max(close, low) + 0.5}
    return bars


def _closes_for(bars: dict[date, dict]):
    return lambda symbol: bars


def _record_and_read(tmp_path, rows, pcs=(), scan_date=SCAN_DATE) -> list[dict]:
    import theta_pick_tracker

    store = tmp_path / "theta_picks.jsonl"
    theta_pick_tracker.record_theta_picks(
        list(rows), list(pcs), scan_date, NOW, path=store
    )
    return _read_jsonl(store)


def test_the_marks_land_on_the_exact_5_10_and_20_session_endpoints(tmp_path):
    """The holiday is the test. 20 BUSINESS days is 06-29, which is broken;
    the 20th SESSION is 06-30, which held."""
    import theta_pick_tracker

    picks = _record_and_read(tmp_path, [_sma_row()])
    bars = _bars(
        {
            SESSION_5: {"close": 96.0},  # above the 95.00 strike
            SESSION_10: {"close": 94.0},  # below it
            SESSION_19: {"close": 94.0},  # the business-day answer: broken
            SESSION_20: {"close": 96.5},  # the session answer: held
        }
    )
    graded = theta_pick_tracker.grade_theta_picks(
        picks,
        closes_for=_closes_for(bars),
        calendar=market_calendar,
        as_of=SESSION_20,
        path=tmp_path / "theta_outcomes.csv",
    )
    assert len(graded) == 1
    row = graded[0]
    assert row["symbol"] == "BBB"
    assert row["held_5"] is True
    assert row["held_10"] is False
    assert row["held_20"] is True, "20 sessions is 2026-06-30; Juneteenth is skipped"
    assert row["status"] == "measured"


def test_a_mark_the_calendar_has_not_reached_is_pending_with_a_reason(tmp_path):
    import theta_pick_tracker
    from master_avwap_lib.session_horizon_outcomes import REASON_IMMATURE

    picks = _record_and_read(tmp_path, [_sma_row()])
    bars = _bars({SESSION_5: {"close": 96.0}, SESSION_10: {"close": 94.0}})
    graded = theta_pick_tracker.grade_theta_picks(
        picks,
        closes_for=_closes_for(bars),
        calendar=market_calendar,
        as_of=SESSION_10,
        path=tmp_path / "theta_outcomes.csv",
    )
    row = graded[0]
    assert row["held_5"] is True
    assert row["held_10"] is False
    assert row["held_20"] is None, "a session that has not happened is not a break"
    assert REASON_IMMATURE in str(row["unmeasured_reason"])
    assert row["status"] == "pending"


def test_mae_is_measured_in_atr_and_names_the_support_that_broke_first(tmp_path):
    """The crafted path: -4.00 on session 2, -12.00 on session 7.

    BBB's supports are SMA_50 97.00, SMA_100 95.00, SMA_200 94.00 against a
    100.00 close and a 2.00 ATR. The session-2 low of 96.00 is under SMA_50 and
    above SMA_100, so exactly one support breaks and there is no tie to resolve.
    The session-7 low of 88.00 is the excursion: (100.00 - 88.00) / 2.00 = 6.00 ATR.
    """
    import theta_pick_tracker

    picks = _record_and_read(tmp_path, [_sma_row()])
    sessions = _sessions_after(SCAN_DATE, 20)
    bars = _bars({sessions[1]: {"low": 96.0}, sessions[6]: {"low": 88.0}})
    graded = theta_pick_tracker.grade_theta_picks(
        picks,
        closes_for=_closes_for(bars),
        calendar=market_calendar,
        as_of=SESSION_20,
        path=tmp_path / "theta_outcomes.csv",
    )
    row = graded[0]
    assert row["mae_atr"] == pytest.approx(6.0)
    assert row["first_support_broken"] == "SMA_50"


def test_a_path_that_never_reaches_a_support_breaks_nothing(tmp_path):
    """`none`, not a blank and not the nearest support. 97.50 never crosses 97.00."""
    import theta_pick_tracker

    picks = _record_and_read(tmp_path, [_sma_row()])
    bars = _bars(default_low=97.5)
    graded = theta_pick_tracker.grade_theta_picks(
        picks,
        closes_for=_closes_for(bars),
        calendar=market_calendar,
        as_of=SESSION_20,
        path=tmp_path / "theta_outcomes.csv",
    )
    row = graded[0]
    assert row["first_support_broken"] == "none"
    assert row["mae_atr"] == pytest.approx(1.25)


def test_the_expiry_grade_waits_for_the_expiry_session_to_complete(tmp_path):
    """2026-07-17 is the sold put's expiry. Before it: no verdict, not a loss."""
    import theta_pick_tracker

    picks = _record_and_read(tmp_path, [_sma_row()])
    early_bars = _bars(count=40)
    early = theta_pick_tracker.grade_theta_picks(
        picks,
        closes_for=_closes_for(early_bars),
        calendar=market_calendar,
        as_of=SESSION_20,
        path=tmp_path / "early.csv",
    )
    assert early[0]["held_at_expiry"] is None

    expiry = date(2026, 7, 17)
    assert market_calendar.is_session(expiry)
    late_bars = dict(early_bars)
    late_bars[expiry] = {"close": 93.0, "low": 92.0, "high": 94.0}
    late = theta_pick_tracker.grade_theta_picks(
        picks,
        closes_for=_closes_for(late_bars),
        calendar=market_calendar,
        as_of=expiry,
        path=tmp_path / "late.csv",
    )
    # 93.00 is under the 95.00 strike on the expiry session: assigned.
    assert late[0]["held_at_expiry"] is False


def test_a_pick_with_no_option_quote_is_unmeasured_rather_than_broken(tmp_path):
    """`option_status: no_weekly_options` leaves `best_option` EMPTY.

    There is no strike, so "did price hold above the strike" has no answer.
    Grading it False would score a play the trader was never offered.
    """
    import theta_pick_tracker

    picks = _record_and_read(
        tmp_path, [_put_row("DDD", option=None, unavailable_reason="no_weekly_options")]
    )
    assert picks[0]["strike"] is None
    graded = theta_pick_tracker.grade_theta_picks(
        picks,
        closes_for=_closes_for(_bars()),
        calendar=market_calendar,
        as_of=SESSION_20,
        path=tmp_path / "theta_outcomes.csv",
    )
    row = graded[0]
    assert row["status"] == "unmeasured"
    assert row["held_20"] is None
    assert str(row["unmeasured_reason"]).strip() != ""


def test_the_outcome_file_is_written_with_the_columns_the_readout_joins_on(tmp_path):
    import theta_pick_tracker

    picks = _record_and_read(tmp_path, [_sma_row()])
    out = tmp_path / "master_avwap_theta_outcomes.csv"
    theta_pick_tracker.grade_theta_picks(
        picks,
        closes_for=_closes_for(_bars()),
        calendar=market_calendar,
        as_of=SESSION_20,
        path=out,
    )
    assert out.exists()
    with out.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    for column in (
        "symbol",
        "scan_date",
        "first_seen_scan_date",
        "play_type",
        "support_combo",
        "score",
        "strike",
        "expiry",
        "held_5",
        "held_10",
        "held_20",
        "held_at_expiry",
        "mae_atr",
        "first_support_broken",
        "status",
        "unmeasured_reason",
    ):
        assert column in header, column
    assert tuple(header) == tuple(theta_pick_tracker.THETA_OUTCOME_COLUMNS)


def test_the_rs_cut_is_recorded_as_not_measured_and_never_invented(tmp_path):
    """Ruling (d): today's theta scoring has no RS term, so the column says so."""
    import theta_pick_tracker

    picks = _record_and_read(tmp_path, [_sma_row()])
    graded = theta_pick_tracker.grade_theta_picks(
        picks,
        closes_for=_closes_for(_bars()),
        calendar=market_calendar,
        as_of=SESSION_20,
        path=tmp_path / "theta_outcomes.csv",
    )
    assert graded[0]["rs_flag"] == "not_measured"


# ---------------------------------------------------------------------------
# Item 3 - the readout
# ---------------------------------------------------------------------------


def _outcome(
    symbol: str,
    combo: str,
    *,
    scan_date: str = "2026-06-01",
    first_seen: str | None = None,
    held_20=None,
    score: int = 50,
    play_type: str = "sold_put",
    held_at_expiry=None,
    status: str = "measured",
) -> dict:
    return {
        "symbol": symbol,
        "scan_date": scan_date,
        "first_seen_scan_date": first_seen or scan_date,
        "play_type": play_type,
        "support_combo": combo,
        "score": score,
        "strike": 95.0,
        "expiry": "2026-07-17",
        "held_5": None,
        "held_10": None,
        "held_20": held_20,
        "held_at_expiry": held_at_expiry,
        "mae_atr": 1.0,
        "first_support_broken": "none",
        "status": status,
        "unmeasured_reason": "",
        "rs_flag": "not_measured",
    }


def _cohort(combo: str, *, n: int, held: int, start: int = 0) -> list[dict]:
    rows = []
    for index in range(n):
        rows.append(
            _outcome(f"{combo}{start + index}", combo, held_20=index < held)
        )
    return rows


def test_the_cells_lead_with_the_hold_rate_and_sort_by_the_wilson_bound():
    """A 2-of-2 perfect combo does not outrank a 24-of-40 one."""
    import theta_pick_tracker

    rows = _cohort("FAT", n=40, held=24) + _cohort("THIN", n=2, held=2)
    rows.append(_outcome("UNGRADED0", "NEW", held_20=None, status="pending"))
    readout = theta_pick_tracker.theta_readout(rows)
    combos = [cell["support_combo"] for cell in readout.cells]
    assert combos[0] == "FAT", "sorted by the raw rate, not by the bound"
    assert combos[1] == "THIN"
    assert combos[-1] == "NEW", "a cell with no bound has nothing to rank on"

    fat = readout.cells[0]
    assert fat["n"] == 40
    assert fat["hold_rate"] == pytest.approx(0.60)
    assert fat["hold_rate_lb"] == pytest.approx(0.44595893660346186)
    assert fat["meets_floor"] is True
    thin = readout.cells[1]
    assert thin["n"] == 2
    assert thin["hold_rate"] == pytest.approx(1.0)
    assert thin["hold_rate_lb"] == pytest.approx(0.3423802275066531)
    assert thin["meets_floor"] is False
    assert MIN_REPORTABLE_N == 30


def test_repeat_days_are_counted_and_never_summed_into_n():
    """n is FIRST appearances. A name the scan finds eight days running is one
    observation for the rate and eight days of interest beside it."""
    import theta_pick_tracker

    rows = _cohort("FAT", n=40, held=24)
    for index in range(5):
        rows.append(
            _outcome(
                f"FAT{index}",
                "FAT",
                scan_date="2026-06-02",
                first_seen="2026-06-01",
                held_20=True,
            )
        )
    readout = theta_pick_tracker.theta_readout(rows)
    cell = readout.cells[0]
    assert cell["n"] == 40, "a repeat appearance was counted as a new observation"
    assert cell["repeat_days"] == 5
    assert cell["hold_rate"] == pytest.approx(0.60)


def test_the_cells_are_cut_by_play_type_as_well_as_by_support_combo():
    import theta_pick_tracker

    rows = _cohort("FAT", n=40, held=24)
    for index in range(6):
        rows.append(
            _outcome(f"P{index}", "FAT", held_20=index < 2, play_type="pcs")
        )
    readout = theta_pick_tracker.theta_readout(rows)
    keys = {(cell["support_combo"], cell["play_type"]) for cell in readout.cells}
    assert ("FAT", "sold_put") in keys
    assert ("FAT", "pcs") in keys
    pcs = next(cell for cell in readout.cells if cell["play_type"] == "pcs")
    assert pcs["n"] == 6
    assert pcs["hold_rate"] == pytest.approx(2 / 6)


def test_the_grade_line_refuses_a_verdict_under_thirty_per_third():
    import theta_pick_tracker

    rows = [
        _outcome(f"S{index}", "FAT", score=index, held_20=index > 4)
        for index in range(9)
    ]
    sentence = theta_pick_tracker.theta_readout(rows).grade_sentence()
    assert "not enough per third yet" in sentence


def test_the_grade_line_states_the_lift_once_each_third_is_deep_enough():
    """90 graded picks, 30 a third: the top third held 27, the bottom third 9."""
    import theta_pick_tracker

    rows = []
    for score in range(1, 91):
        if score >= 61:
            held = score >= 64  # 27 of 30
        elif score >= 31:
            held = score >= 46  # 15 of 30
        else:
            held = score >= 22  # 9 of 30
        rows.append(_outcome(f"S{score}", "FAT", score=score, held_20=held))
    sentence = theta_pick_tracker.theta_readout(rows).grade_sentence()
    assert "not enough per third yet" not in sentence
    assert "90% of 30" in sentence
    assert "30% of 30" in sentence
    assert "+60" in sentence
    assert wilson_lower_bound(27, 30) == pytest.approx(0.7437891742081592)


def test_the_population_sentence_counts_appearances_picks_and_scan_dates():
    import theta_pick_tracker

    rows = [
        _outcome("AAA", "FAT", held_20=True, held_at_expiry=True),
        _outcome("BBB", "FAT", held_20=True),
        _outcome("CCC", "FAT", held_20=False),
        _outcome("AAA", "FAT", scan_date="2026-06-02", first_seen="2026-06-01", held_20=True),
        _outcome("BBB", "FAT", scan_date="2026-06-02", first_seen="2026-06-01", held_20=True),
    ]
    sentence = theta_pick_tracker.theta_readout(rows).population_sentence()
    assert (
        "3 first appearances of 5 theta picks over 2 scan dates; "
        "graded at 20 sessions; expiry grades: 1" in sentence
    )


def test_the_readout_reads_a_csv_round_trip_the_same_as_native_rows(tmp_path):
    """The export is a CSV: `True` is a string and an unmeasured cell is empty.

    A readout that tests `row["held_20"] is True` counts every CSV row a break,
    and one that tests truthiness counts the empty string as a hold.
    """
    import theta_pick_tracker

    rows = _cohort("FAT", n=40, held=24) + [
        _outcome("PEND0", "FAT", held_20=None, status="pending")
    ]
    native = theta_pick_tracker.theta_readout(rows).cells

    path = tmp_path / "theta_outcomes.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})
    with path.open(newline="", encoding="utf-8") as handle:
        from_csv = list(csv.DictReader(handle))
    assert from_csv[0]["held_20"] == "True"
    assert from_csv[-1]["held_20"] == ""

    round_tripped = theta_pick_tracker.theta_readout(from_csv).cells
    assert [cell["n"] for cell in round_tripped] == [cell["n"] for cell in native]
    assert round_tripped[0]["hold_rate"] == pytest.approx(native[0]["hold_rate"])
    assert round_tripped[0]["hold_rate_lb"] == pytest.approx(native[0]["hold_rate_lb"])


# ---------------------------------------------------------------------------
# Item 4 - the Theta tab
# ---------------------------------------------------------------------------


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - GUI extra not installed
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def theta_panel(monkeypatch, tmp_path):
    if _qt_app() is None:
        pytest.skip("PySide6 is not installed")
    from ui.panels import setup_tracker_panel

    setup_tracker_panel.clear_setup_tracker_csv_cache()
    monkeypatch.setattr(
        setup_tracker_panel,
        "MASTER_AVWAP_THETA_OUTCOMES_FILE",
        tmp_path / "master_avwap_theta_outcomes.csv",
        raising=False,
    )
    return setup_tracker_panel


def _write_theta_outcomes(path: Path, rows: list[dict]) -> None:
    import theta_pick_tracker

    fields = list(theta_pick_tracker.THETA_OUTCOME_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: ("" if row.get(key) is None else row.get(key, "")) for key in fields}
            )


def _load(panel) -> None:
    from tests.conftest import refresh_setup_tracker

    refresh_setup_tracker(panel)


def test_the_theta_tab_is_on_the_setup_tracker(theta_panel, tmp_path):
    _write_theta_outcomes(
        tmp_path / "master_avwap_theta_outcomes.csv",
        _cohort("FAT", n=40, held=24) + _cohort("THIN", n=2, held=2),
    )
    panel = theta_panel.SetupTrackerPanel()
    try:
        titles = [panel.tabs.tabText(index) for index in range(panel.tabs.count())]
        assert "Theta" in titles
    finally:
        panel.deleteLater()


def test_the_theta_tab_renders_its_cells_and_says_what_they_are(theta_panel, tmp_path):
    rows = _cohort("FAT", n=40, held=24) + _cohort("THIN", n=2, held=2)
    _write_theta_outcomes(tmp_path / "master_avwap_theta_outcomes.csv", rows)
    panel = theta_panel.SetupTrackerPanel()
    _load(panel)
    try:
        assert panel.theta_model.rowCount() == 2
        assert [cell["support_combo"] for cell in panel.theta_rows] == ["FAT", "THIN"]
        sentence = panel.theta_status_label.text()
        # The singular/plural of "scan date" is the builder's; the counts are not.
        assert "42 first appearances of 42 theta picks" in sentence
        assert "graded at 20 sessions" in sentence
        assert "not enough per third yet" in panel.theta_grade_label.text()
    finally:
        panel.deleteLater()


def test_the_theta_tab_says_no_export_yet_before_the_first_grade(theta_panel):
    panel = theta_panel.SetupTrackerPanel()
    _load(panel)
    try:
        assert panel.theta_model.rowCount() == 0
        assert "no export yet" in panel.theta_status_label.text().lower()
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# Item 2 - the overnight slot
# ---------------------------------------------------------------------------


def test_the_grader_runs_at_the_end_of_the_deterministic_stage():
    """Decision 0018's stage order: a later phase APPENDS inside its stage.

    `daily_digest` closes the deterministic block, so the theta grade goes
    directly after it and stays ahead of `ai_summary`. Nothing reorders.
    """
    from ai_jobs import runner

    names = tuple(slot.name for slot in runner.default_slots())
    assert "theta_pick_grading" in names
    assert names.index("theta_pick_grading") == names.index("daily_digest") + 1
    assert names.index("theta_pick_grading") < names.index("ai_summary")
    by_name = {slot.name: slot for slot in runner.default_slots()}
    assert by_name["theta_pick_grading"].enabled
