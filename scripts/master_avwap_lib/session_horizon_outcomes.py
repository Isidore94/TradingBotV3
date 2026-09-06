"""Exact exchange-session horizons from completed bars - packet ST1 item 2.

The trader, 2026-09-06: *"Define exact exchange-session horizons from the entry
session and completed-bar data, independent of later scan membership. Missing
target-session data remains unmeasured with a reason. Do not forward-fill to the
next scan and call it the original horizon."*

**What v1 does, and why this exists beside it.**
`legacy.build_scan_factor_observation_rows` picks the future row as
`future_idx = idx + horizon` into the SYMBOL'S OWN scan rows. For a name that
appears on a watchlist every session that is five sessions; for one that appears
irregularly it is wherever its next six appearances happen to be. Measured on the
live file (2026-09-04, 19,558 rows): of the 2,989 horizon-5 rows in the last 20
scan dates, `sessions_spanned` was 5 on 1,128, 6 on 699, 7 on 626, 8 on 409 and
9-18 on 127 - so a "5-session" cell is mostly 5 to 8 sessions, and the
`stale_horizon` flag (span > 2x the declared horizon) catches 2.4% of it.

**This module is VERSIONED AND BESIDE v1, never a replacement.** v1 keeps every
row, every column and every value: re-selecting its future row would silently
restate every number the tracker has produced, which is a scoring change the
trader's 2026-09-06 prompt explicitly does not approve. The v2 rows go to their
own file, `outcome_kind` says which is which, and `observation_id` is computed
the SAME way for the same `(scan_row_id, horizon)`, so a v1 row joins its v2 row
**one to one**: both keep the session's LAST scan row, so both name it the same.

**It never fetches.** `closes_for(symbol)` is supplied by the caller and returns
the COMPLETED daily closes already in hand - on the desk, the frames the scan
already walked. A symbol with no frame produces `no_bar_for_target_session`
rows, never a network call inside an export (ground rule 8).

**Completed bars only.** A bar dated after `last_completed_session` is not read
at all, and a target session after it is `immature` - unmeasured with a reason,
never the next available bar.

**ONE ROW PER SESSION, and it says how many scans stand behind it.** The
measurement for a `(symbol, side, scan_date, horizon)` is the same number for
every scan the desk ran that day - it is the entry session's close against the
target session's close, and neither moves because the desk looked again at
11:15. So the build keeps the session's LAST scan row (v1's choice, off the same
sort) and each row carries `collapsed_same_session`, the number of other scan
rows it speaks for. That is NOT a duplicate count: `dropped_duplicates` stays
the true `(scan_row_id, horizon)` repeat count and the two are reported
separately, because "the desk scanned 15 times that day" and "the input recorded
one scan twice" are different facts.

**It is a ROLLING WINDOW, and it says so.** The build covers scan dates within
`BUILD_WINDOW_SESSIONS` (30 exchange sessions, 1.5x the widest window any
surface reads) of `last_completed_session`, because a settled row's target close
does not move and rewriting three years of them on every scan buys nothing.
Measured through the export path on a copy of the live history (146,367 scan
rows, 2026-09-04, `last_completed_session` 2026-09-03): **91,116 rows, 5.3 s,
25.6 MB** at the shipped settings, against 110,308 rows / 5.7 s / 30.9 MB with
the same collapse and no window, and 458,336 rows / 13.4 s / 127.5 MB before the
collapse at a 60-session window. The collapse does most of the work (91,880 scan
rows folded, 300 true duplicates); the window takes the last 17% and keeps the
file from growing without bound. Rows older than the window are COUNTED
(`excluded['outside_build_window']`), never silently skipped, and the exported
file is a window rather than an archive.

Shadow only: nothing here reaches a detector, a score, a rank that gates, an
alert, a watchlist, Focus, the review queue or `review_policy.json`.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Mapping

import pandas as pd

import market_calendar
from swing_evidence import OUTCOME_KIND_SESSION_V2

#: What was compared with what. One string, on every row.
SESSION_HORIZON_KNOWLEDGE_BASIS = "entry_session_close_to_target_session_close"

#: The header, in order. `observation_id` first so the file joins to the v1
#: observations and tier outcomes on sight.
SESSION_HORIZON_OUTCOME_COLUMNS = [
    "observation_id",
    "scan_row_id",
    "symbol",
    "side",
    "scan_date",
    "target_session",
    "horizon_sessions",
    "sessions_spanned",
    "entry_close",
    "entry_close_source",
    "target_close",
    "side_return_pct",
    "favorable",
    "measured",
    "maturity",
    "unmeasured_reason",
    "outcome_kind",
    "knowledge_basis",
    "tier",
    "tier_source",
    "priority_bucket",
    "setup_family",
    "favorite_zone",
    # How many OTHER scan rows of the same (symbol, side, session) this row
    # speaks for. The measurement does not change when the desk scans again, so
    # the session has ONE row and it says how many looks stand behind it.
    "collapsed_same_session",
]

#: Why a row could not be measured. Every unmeasured row carries exactly one.
REASON_NO_TARGET_BAR = "no_bar_for_target_session"
REASON_IMMATURE = "target_session_not_complete"
REASON_TARGET_OUT_OF_RANGE = "target_session_outside_calendar_range"

#: Why a scan row produced no rows at all. Counted, never silently skipped.
EXCLUDED_ENTRY_NOT_A_SESSION = "entry_not_a_session"
EXCLUDED_ENTRY_OUT_OF_RANGE = "entry_outside_calendar_range"

#: How far back the build reaches, in EXCHANGE SESSIONS ending at
#: `last_completed_session`. 1.5x `LATELY_SESSIONS` (20) - half a window wider
#: than the widest any surface reads - so every reader sees its whole window with
#: room to spare, and no scan pays to rewrite three years of settled rows whose
#: target closes cannot move. The trader's lead set this on 2026-09-06, after 60
#: sessions without the same-session collapse measured 127.5 MB per scan for a
#: file no reader opens yet.
BUILD_WINDOW_SESSIONS = 30


@dataclass(frozen=True)
class SessionHorizonBuild:
    """The rows, and everything that did not become one."""

    rows: list[dict] = field(default_factory=list)
    #: TRUE REPEATS: `(scan_row_id, horizon)` pairs the input carried twice - the
    #: same scan row recorded twice, which is a defect in the input. Counted
    #: because `_prepare_scan_factor_history_frame` de-duplicates SILENTLY, and a
    #: row nobody can reconcile is a row nobody can check.
    dropped_duplicates: int = 0
    #: SCAN ROWS folded into a session's representative, because the measurement
    #: is the same number for every scan that day. NOT a duplicate and never
    #: counted as one. In scan rows, so it equals the sum of the row column over
    #: any ONE horizon rather than over the whole file.
    collapsed_same_session: int = 0
    #: Scan rows that produced no row at all, by reason.
    excluded: Counter = field(default_factory=Counter)


def closes_from_daily_frames(
    frames_by_symbol: Mapping[str, Any] | None,
) -> Callable[[str], dict[date, float] | None]:
    """A `closes_for` over the daily frames a scan already holds.

    Lazy and memoised per symbol: the desk's scan holds ~1,100 frames and the
    export needs the ones its history mentions. Returns None for a symbol with
    no frame in hand - "we did not look" is not "there was no bar".
    """
    frames = frames_by_symbol or {}
    cache: dict[str, dict[date, float] | None] = {}

    def closes_for(symbol: str) -> dict[date, float] | None:
        key = str(symbol or "").strip().upper()
        if key in cache:
            return cache[key]
        frame = frames.get(key)
        closes: dict[date, float] | None = None
        try:
            if frame is not None and getattr(frame, "empty", True) is False:
                stamps = frame["datetime"] if "datetime" in frame.columns else frame[frame.columns[0]]
                closes = {}
                for stamp, close in zip(stamps, frame["close"]):
                    day = pd.to_datetime(stamp)
                    value = float(close)
                    if value > 0 and not pd.isna(day):
                        closes[day.date()] = value
        except Exception:  # noqa: BLE001 - a bad frame is "no bars", never a failed export
            logging.debug("Session-horizon closes unreadable for %s", symbol, exc_info=True)
            closes = None
        cache[key] = closes
        return closes

    return closes_for


def _is_session(day: date, calendar) -> bool | None:
    """True / False, or None when the calendar refuses to answer for that day."""
    try:
        return bool(calendar.is_session(day))
    except Exception:  # noqa: BLE001 - outside the validated range is "cannot say"
        return None


def _nth_session_after(day: date, horizon: int, calendar) -> date | None:
    """The `horizon`-th session strictly after `day`, holidays skipped."""
    cursor = day
    remaining = int(horizon)
    # A generous bound so a calendar bug cannot become an infinite loop: a
    # ten-session horizon is at most a fortnight of calendar days plus holidays.
    for _ in range(remaining * 5 + 30):
        cursor = cursor + timedelta(days=1)
        answer = _is_session(cursor, calendar)
        if answer is None:
            return None
        if answer:
            remaining -= 1
            if remaining <= 0:
                return cursor
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def build_session_horizon_observation_rows(
    history_df: pd.DataFrame | None,
    closes_for: Callable[[str], Mapping[date, float] | None],
    *,
    horizons: tuple[int, ...] | None = None,
    last_completed_session: date,
    calendar=market_calendar,
    window_sessions: int | None = BUILD_WINDOW_SESSIONS,
) -> SessionHorizonBuild:
    """One row per `(scan row, horizon)`, on the exchange calendar.

    * entry session = the scan date, which must BE a session;
    * target session = the N-th session after it, holidays skipped;
    * `entry_close` = the bar close on the entry session, falling back to the
      scan row's own recorded close and SAYING so (`entry_close_source`);
    * `target_close` = the bar close ON the target session exactly. Missing is
      `measured: False` with a reason - never the next bar, never a later scan.

    **A REPEAT and a COLLAPSE are different things, and each is counted under
    its own name** (trader's lead, 2026-09-06). `_scan_factor_row_id` is
    `symbol:scan_date:run_id`, so two scans of one symbol on one day are two
    scan rows, NOT a duplicate - the desk ran 15 scans on 2026-08-31 and a
    `(symbol, scan_date)` key called 14 of each of those a duplicate.

    * `dropped_duplicates` is the true repeat count: the same `scan_row_id`
      present twice, at `(scan_row_id, horizon)` grain. That is an input defect.
    * `collapsed_same_session` is the honest name for the rest. THE MEASUREMENT
      IS THE SAME NUMBER FOR EVERY SCAN THAT DAY - it is the entry session's
      close against the target session's close, and neither moves because the
      desk looked again at 11:15 - so the build keeps ONE row per
      `(symbol, side, scan_date, horizon)`, the session's LAST scan row, exactly
      as `_prepare_scan_factor_history_frame` chooses v1's. Every row says how
      many scan rows it represents, and the builder totals it in SCAN ROWS.

    Keeping the last row is also what restores the 1:1 join: v1's
    `observation_id` for a session is that same row's.

    **`window_sessions` bounds the build to a ROLLING WINDOW** of scan dates
    ending at `last_completed_session` - by default `BUILD_WINDOW_SESSIONS`
    (30 exchange sessions), 1.5x the widest window any surface reads. Rewriting
    three years of settled observations on every scan is a cost the trader pays
    for nothing: a settled row's target close does not move. Rows older than the
    window are counted in `excluded['outside_build_window']`, never silently
    skipped. Pass `None` to build everything.
    """
    from .legacy import (  # local: `legacy` is large and this module is imported lazily
        SCAN_FACTOR_HORIZONS,
        _scan_factor_text,
        normalize_side,
        tier_for_tracker_row,
    )

    if horizons is None:
        horizons = SCAN_FACTOR_HORIZONS
    normalized_horizons = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    if not normalized_horizons:
        return SessionHorizonBuild()

    last_complete = last_completed_session
    if isinstance(last_complete, datetime):
        last_complete = last_complete.date()

    frame = _prepare_session_horizon_frame(history_df)
    if frame.empty:
        return SessionHorizonBuild()

    excluded: Counter = Counter()
    window_start = _build_window_start(last_complete, window_sessions, calendar)
    if window_start is not None:
        inside = frame["_scan_date_dt"] >= pd.Timestamp(window_start)
        dropped = int((~inside).sum())
        if dropped:
            excluded["outside_build_window"] += dropped * len(normalized_horizons)
        frame = frame[inside]
        if frame.empty:
            return SessionHorizonBuild(excluded=excluded)

    # THE TRUE REPEAT COUNT, taken before the collapse: the same `scan_row_id`
    # present twice is the input recording one observation twice. It is a subset
    # of what the collapse folds, and it is the only part of it that is a defect.
    dropped_duplicates = (
        len(frame) - int(frame["_scan_row_id"].nunique())
    ) * len(normalized_horizons)

    # THE COLLAPSE: one row per (symbol, side, scan date), the session's LAST -
    # the same choice `_prepare_scan_factor_history_frame` makes for v1, off the
    # same sort, so v1's `observation_id` is this row's.
    group_keys = ["_symbol", "_side", "_scan_date_text"]
    group_sizes = frame.groupby(group_keys, sort=False)["_scan_row_id"].transform("size")
    frame = frame.assign(_collapsed_same_session=group_sizes.astype(int) - 1)
    representatives = frame.drop_duplicates(group_keys, keep="last")
    collapsed_same_session = len(frame) - len(representatives)
    frame = representatives

    built: dict[tuple[str, int], dict] = {}
    entry_is_session: dict[date, bool | None] = {}
    target_cache: dict[tuple[date, int], date | None] = {}
    closes_cache: dict[str, Mapping[date, float] | None] = {}

    for entry in frame.to_dict("records"):
        symbol = _text(entry.get("_symbol"))
        scan_date_text = _text(entry.get("_scan_date_text"))
        if not symbol or not scan_date_text:
            continue
        try:
            entry_day = date.fromisoformat(scan_date_text[:10])
        except ValueError:
            excluded[EXCLUDED_ENTRY_NOT_A_SESSION] += 1
            continue
        if entry_day not in entry_is_session:
            entry_is_session[entry_day] = _is_session(entry_day, calendar)
        answer = entry_is_session[entry_day]
        if answer is None:
            excluded[EXCLUDED_ENTRY_OUT_OF_RANGE] += len(normalized_horizons)
            continue
        if not answer:
            excluded[EXCLUDED_ENTRY_NOT_A_SESSION] += len(normalized_horizons)
            continue

        if symbol not in closes_cache:
            try:
                closes_cache[symbol] = closes_for(symbol)
            except Exception:  # noqa: BLE001 - a caller's lookup never fails an export
                logging.debug("closes_for(%s) raised", symbol, exc_info=True)
                closes_cache[symbol] = None
        closes = closes_cache[symbol] or {}

        side = normalize_side(entry.get("_side") or entry.get("side") or "")
        side_multiplier = -1.0 if side == "SHORT" else 1.0
        scan_row_id = _text(entry.get("_scan_row_id"))
        tier, tier_source = tier_for_tracker_row(entry)

        # COMPLETED BARS ONLY: a bar dated after the last completed session is a
        # forming bar and is not read, on either leg.
        bar_entry_close = closes.get(entry_day) if entry_day <= last_complete else None
        entry_close = _number(bar_entry_close)
        entry_close_source = "session_bar"
        if entry_close is None or entry_close <= 0:
            entry_close = _number(entry.get("_entry_close"))
            entry_close_source = "scan_row"
        if entry_close is None or entry_close <= 0:
            excluded["entry_close_unreadable"] += len(normalized_horizons)
            continue

        for horizon in normalized_horizons:
            key = (entry_day, horizon)
            if key not in target_cache:
                target_cache[key] = _nth_session_after(entry_day, horizon, calendar)
            target_day = target_cache[key]
            row = {
                "observation_id": f"{scan_row_id}:{horizon}",
                "scan_row_id": scan_row_id,
                "symbol": symbol,
                "side": side,
                "scan_date": scan_date_text,
                "target_session": target_day.isoformat() if target_day else "",
                "horizon_sessions": int(horizon),
                "sessions_spanned": int(horizon),
                # How many OTHER scan rows of this session this row speaks for.
                # 0 on a day the desk scanned once; 14 on 2026-08-31.
                "collapsed_same_session": int(entry.get("_collapsed_same_session") or 0),
                "entry_close": float(entry_close),
                "entry_close_source": entry_close_source,
                "target_close": "",
                "side_return_pct": "",
                "favorable": "",
                "measured": False,
                "maturity": "mature",
                "unmeasured_reason": "",
                "outcome_kind": OUTCOME_KIND_SESSION_V2,
                "knowledge_basis": SESSION_HORIZON_KNOWLEDGE_BASIS,
                "tier": tier,
                "tier_source": tier_source,
                "priority_bucket": _scan_factor_text(entry.get("priority_bucket")),
                "setup_family": _scan_factor_text(entry.get("setup_family")),
                "favorite_zone": _scan_factor_text(entry.get("favorite_zone")),
            }
            if target_day is None:
                row["unmeasured_reason"] = REASON_TARGET_OUT_OF_RANGE
            elif target_day > last_complete:
                row["maturity"] = "immature"
                row["unmeasured_reason"] = REASON_IMMATURE
            else:
                target_close = _number(closes.get(target_day))
                if target_close is None or target_close <= 0:
                    row["unmeasured_reason"] = REASON_NO_TARGET_BAR
                else:
                    raw_return_pct = ((target_close / float(entry_close)) - 1.0) * 100.0
                    side_return_pct = raw_return_pct * side_multiplier
                    row["target_close"] = float(target_close)
                    row["side_return_pct"] = side_return_pct
                    row["favorable"] = bool(side_return_pct > 0)
                    row["measured"] = True
            # THE FILE's identity is `(scan_row_id, horizon)` - that is what
            # `observation_id` is, and two rows may never share one. After the
            # collapse the only way to collide is one run recording a symbol on
            # both sides; it is counted rather than silently overwritten.
            identity = (scan_row_id, int(horizon))
            if identity in built:
                excluded["observation_id_collision"] += 1
            built[identity] = row

    rows = list(built.values())
    rows.sort(
        key=lambda item: (
            str(item.get("scan_date") or ""),
            str(item.get("symbol") or ""),
            int(item.get("horizon_sessions", 0) or 0),
        )
    )
    return SessionHorizonBuild(
        rows=rows,
        dropped_duplicates=dropped_duplicates,
        collapsed_same_session=collapsed_same_session,
        excluded=excluded,
    )


def _build_window_start(last_complete: date, window_sessions, calendar) -> date | None:
    """The first scan date the build covers, or None for "everything".

    Walked on the exchange calendar, not in calendar days, for the same reason
    every other window on this desk is: a holiday week would quietly shorten it.
    A calendar that refuses to answer gives None - an unbounded build is slow,
    never wrong, and refusing to build at all would be worse.
    """
    if window_sessions is None:
        return None
    try:
        sessions = int(window_sessions)
    except (TypeError, ValueError):
        return None
    if sessions <= 0:
        return None
    cursor = last_complete
    try:
        for _ in range(sessions - 1):
            cursor = calendar.previous_session(cursor)
    except Exception:  # noqa: BLE001 - outside the validated range: build everything
        logging.debug("Session-horizon build window unavailable.", exc_info=True)
        return None
    return cursor


def _prepare_session_horizon_frame(history_df) -> pd.DataFrame:
    """`_prepare_scan_factor_history_frame`, MINUS the `(symbol, scan date)` collapse.

    The v1 frame keeps one row per symbol per scan date because that is the
    grain its file is written at. v2's grain is the SCAN ROW: the desk ran 15
    scans on 2026-08-31 and each one recorded what it saw at the time, under its
    own `run_id`. Collapsing them here reported 475,492 "duplicates" against
    109,584 rows on the live history when the truly repeated `scan_row_id`s
    numbered 75. Everything else - the validity filter, the sort order, the row
    id - is the v1 rule, so the two files still join on `observation_id`.
    """
    from .legacy import _scan_factor_row_id, _scan_factor_text, normalize_side

    if history_df is None or not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return pd.DataFrame()
    if not {"symbol", "last_close"}.issubset(set(history_df.columns)):
        return pd.DataFrame()
    if "last_trade_date" not in history_df.columns and "run_date" not in history_df.columns:
        return pd.DataFrame()

    frame = history_df.copy()
    frame["_input_order"] = range(len(frame))
    frame["_symbol"] = frame["symbol"].apply(lambda value: _scan_factor_text(value).upper())
    date_source = (
        frame["last_trade_date"] if "last_trade_date" in frame.columns else frame["run_date"]
    )
    frame["_scan_date_dt"] = pd.to_datetime(date_source, errors="coerce")
    frame["_scan_date_text"] = frame["_scan_date_dt"].dt.strftime("%Y-%m-%d")
    frame["_entry_close"] = pd.to_numeric(frame["last_close"], errors="coerce")
    if "side" in frame.columns:
        frame["_side"] = frame["side"].apply(normalize_side)
    else:
        frame["_side"] = "LONG"
    frame["_run_id_text"] = (
        frame["run_id"].fillna("").astype(str) if "run_id" in frame.columns else ""
    )
    frame["_run_timestamp_text"] = (
        frame["run_timestamp"].fillna("").astype(str)
        if "run_timestamp" in frame.columns
        else ""
    )
    frame = frame[
        (frame["_symbol"] != "")
        & frame["_scan_date_dt"].notna()
        & frame["_entry_close"].notna()
        & (frame["_entry_close"] > 0)
    ].copy()
    if frame.empty:
        return frame
    frame.sort_values(
        ["_symbol", "_scan_date_dt", "_run_timestamp_text", "_run_id_text", "_input_order"],
        inplace=True,
    )
    frame["_scan_row_id"] = frame.apply(_scan_factor_row_id, axis=1)
    return frame
