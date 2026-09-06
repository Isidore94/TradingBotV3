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
the SAME way for the same `(scan_row_id, horizon)` so the two files join 1:1.

**It never fetches.** `closes_for(symbol)` is supplied by the caller and returns
the COMPLETED daily closes already in hand - on the desk, the frames the scan
already walked. A symbol with no frame produces `no_bar_for_target_session`
rows, never a network call inside an export (ground rule 8).

**Completed bars only.** A bar dated after `last_completed_session` is not read
at all, and a target session after it is `immature` - unmeasured with a reason,
never the next available bar.

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
]

#: Why a row could not be measured. Every unmeasured row carries exactly one.
REASON_NO_TARGET_BAR = "no_bar_for_target_session"
REASON_IMMATURE = "target_session_not_complete"
REASON_TARGET_OUT_OF_RANGE = "target_session_outside_calendar_range"

#: Why a scan row produced no rows at all. Counted, never silently skipped.
EXCLUDED_ENTRY_NOT_A_SESSION = "entry_not_a_session"
EXCLUDED_ENTRY_OUT_OF_RANGE = "entry_outside_calendar_range"


@dataclass(frozen=True)
class SessionHorizonBuild:
    """The rows, and everything that did not become one."""

    rows: list[dict] = field(default_factory=list)
    #: `(scan_row_id, horizon)` pairs the input carried twice. Counted here
    #: because `_prepare_scan_factor_history_frame` de-duplicates SILENTLY, and
    #: a row nobody can reconcile is a row nobody can check.
    dropped_duplicates: int = 0
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
) -> SessionHorizonBuild:
    """One row per `(scan row, horizon)`, on the exchange calendar.

    * entry session = the scan date, which must BE a session;
    * target session = the N-th session after it, holidays skipped;
    * `entry_close` = the bar close on the entry session, falling back to the
      scan row's own recorded close and SAYING so (`entry_close_source`);
    * `target_close` = the bar close ON the target session exactly. Missing is
      `measured: False` with a reason - never the next bar, never a later scan.
    """
    from .legacy import (  # local: `legacy` is large and this module is imported lazily
        SCAN_FACTOR_HORIZONS,
        _prepare_scan_factor_history_frame,
        _scan_factor_text,
        normalize_side,
        tier_for_tracker_row,
    )

    if horizons is None:
        horizons = SCAN_FACTOR_HORIZONS
    normalized_horizons = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    if not normalized_horizons:
        return SessionHorizonBuild()

    dropped_duplicates = _count_dropped_duplicates(
        history_df, len(normalized_horizons), _scan_factor_text
    )

    frame = _prepare_scan_factor_history_frame(history_df)
    if frame.empty:
        return SessionHorizonBuild(dropped_duplicates=dropped_duplicates)

    last_complete = last_completed_session
    if isinstance(last_complete, datetime):
        last_complete = last_complete.date()

    rows: list[dict] = []
    excluded: Counter = Counter()
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
            rows.append(row)

    rows.sort(
        key=lambda item: (
            str(item.get("scan_date") or ""),
            str(item.get("symbol") or ""),
            int(item.get("horizon_sessions", 0) or 0),
        )
    )
    return SessionHorizonBuild(
        rows=rows, dropped_duplicates=dropped_duplicates, excluded=excluded
    )


def _count_dropped_duplicates(history_df, horizon_count: int, text_of) -> int:
    """How many `(scan_row_id, horizon)` pairs the input carried twice.

    `_prepare_scan_factor_history_frame` keeps ONE row per `(symbol, scan date)`
    and says nothing about what it dropped. v2 keeps that identity - the two
    files must join 1:1 - so the count is taken here, from the same validity
    rules, before the de-duplication happens.
    """
    if history_df is None or not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return 0
    if not {"symbol", "last_close"}.issubset(set(history_df.columns)):
        return 0
    if "last_trade_date" not in history_df.columns and "run_date" not in history_df.columns:
        return 0
    try:
        symbols = history_df["symbol"].apply(lambda value: text_of(value).upper())
        source = (
            history_df["last_trade_date"]
            if "last_trade_date" in history_df.columns
            else history_df["run_date"]
        )
        stamps = pd.to_datetime(source, errors="coerce")
        closes = pd.to_numeric(history_df["last_close"], errors="coerce")
        valid = (symbols != "") & stamps.notna() & closes.notna() & (closes > 0)
        if not bool(valid.any()):
            return 0
        keys = pd.DataFrame(
            {"symbol": symbols[valid], "scan_date": stamps[valid].dt.strftime("%Y-%m-%d")}
        )
        return int((len(keys) - len(keys.drop_duplicates())) * max(1, int(horizon_count)))
    except Exception:  # noqa: BLE001 - a count that cannot be taken is 0, never a failed build
        logging.debug("Session-horizon duplicate count failed.", exc_info=True)
        return 0
