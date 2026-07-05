#!/usr/bin/env python3
"""Auto Pilot core: the unattended "mini PC mode" logic, UI- and socket-free.

Runs the trading day without the trader at the desk:

- Swing scans on the away schedule: first at open+60m (07:30 for a 06:30
  open), then every top of the hour from the first full hour onward through
  the close slot. Slots in the session's final hour (and the close slot)
  write the setup tracker, matching the main bot's behavior.
- Self-built BounceBot watchlists ~30-60 minutes after the open: every gap-up
  plus the names showing real relative strength vs SPY in the first 30-60
  minutes become longs; gap-downs plus relative weakness become shorts. The
  broad cut runs on yfinance batch data on purpose - the IBKR API budget
  stays reserved for BounceBot's live M5 scanning at the open.
- Near-HOD adds on bullish pauses: when the tape is bullish but SPY prints
  red, the swing scanner's long rows are checked for names holding near their
  high of day and folded into longs.txt (inverted for bearish/LOD).
- A phone-digestible report written to the shared Google Drive home folder.

Everything here is deliberately testable: scheduling, ranking, filtering and
rendering are pure; the yfinance fetchers accept an injectable downloader.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from market_session import get_market_session_window
from project_paths import (
    AUTOPILOT_REPORT_FILE,
    LONGS_FILE,
    SHORTS_FILE,
    UNIVERSE_ALL_FILE,
    UNIVERSE_LONGS_FILE,
    UNIVERSE_SHORTS_FILE,
)
from watchlist_utils import read_watchlist_symbols

# Scheduling (minutes are relative to the session open, local clock).
AUTOPILOT_FIRST_SCAN_AFTER_OPEN_MINUTES = 60
AUTOPILOT_WATCHLIST_BUILD_AFTER_OPEN_MINUTES = 30
AUTOPILOT_WATCHLIST_BUILD_DEADLINE_MINUTES = 120

# Watchlist self-build thresholds.
AUTOPILOT_GAP_MIN_PCT = 2.0  # open vs yesterday's close
AUTOPILOT_RS_EXCESS_MIN_PCT = 1.5  # early move minus SPY's early move
AUTOPILOT_WATCHLIST_CAP = 25  # per side; protects BounceBot's IB pacing
AUTOPILOT_OPEN_SCAN_MAX_SYMBOLS = 1200
AUTOPILOT_OPEN_SCAN_CHUNK_SIZE = 150

# Near-HOD/LOD adds during regime pauses.
AUTOPILOT_HOD_PROXIMITY_PCT = 1.0
AUTOPILOT_HOD_TOP_ROWS = 30
AUTOPILOT_HOD_CHECK_COOLDOWN_MINUTES = 30

# Universe freshness: Auto Pilot is used sporadically, so freshness is checked
# on every activation/tick instead of trusting a nightly job to have run.
AUTOPILOT_UNIVERSE_RETRY_MINUTES = 60


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
def get_autopilot_swing_slots(
    reference: datetime | None = None,
    local_timezone_name: str | None = None,
) -> list[str]:
    """The away-day swing scan slots as local HH:MM labels.

    First slot at open+60m; the API's first hour belongs to the intraday
    strong/weak discovery. Hourly top-of-hour slots resume from the first
    full hour after that (09:00 for a 06:30 open - deliberately skipping
    08:00) through the close slot.
    """
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    open_naive = session.open_local.replace(tzinfo=None)
    close_naive = session.close_local.replace(tzinfo=None)

    first = open_naive + timedelta(minutes=AUTOPILOT_FIRST_SCAN_AFTER_OPEN_MINUTES)
    slots = [first]
    cursor = first + timedelta(minutes=60)
    if cursor.minute or cursor.second or cursor.microsecond:
        cursor = cursor.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while cursor <= close_naive:
        slots.append(cursor)
        cursor += timedelta(hours=1)
    return [slot.strftime("%H:%M") for slot in slots]


def slot_writes_setup_tracker(
    slot: str,
    reference: datetime | None = None,
    local_timezone_name: str | None = None,
) -> bool:
    """Tracker-writing slots: anything in the session's final hour or later.

    For a 06:30-13:00 session that is the 12:00 and 13:00 runs - same window
    the main bot uses for its setup-tracker refresh.
    """
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    last_hour_label = str(getattr(session, "last_hour_start_label", "") or "")
    if not last_hour_label:
        return False
    try:
        slot_minutes = _label_to_minutes(slot)
        refresh_minutes = _label_to_minutes(last_hour_label)
    except ValueError:
        return False
    return slot_minutes >= refresh_minutes


def _label_to_minutes(label: str) -> int:
    hours, minutes = str(label).strip().split(":", 1)
    return int(hours) * 60 + int(minutes)


def minutes_since_open(
    now: datetime,
    local_timezone_name: str | None = None,
) -> float:
    session = get_market_session_window(reference=now, local_timezone_name=local_timezone_name)
    open_naive = session.open_local.replace(tzinfo=None)
    return (now - open_naive).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Universe freshness (sporadic activation must self-heal a stale universe)
# ---------------------------------------------------------------------------
def last_completed_session_close(
    now: datetime,
    local_timezone_name: str | None = None,
) -> datetime | None:
    """Local-naive close of the most recent finished session.

    Walks back weekday by weekday; holidays are not modeled, so the worst case
    on a holiday-adjacent day is one harmless extra universe rebuild.
    """
    for days_back in range(0, 10):
        probe_date = now.date() - timedelta(days=days_back)
        if probe_date.weekday() >= 5:
            continue
        session = get_market_session_window(reference=probe_date, local_timezone_name=local_timezone_name)
        close_naive = session.close_local.replace(tzinfo=None)
        if days_back == 0 and now < close_naive:
            continue
        return close_naive
    return None


def universe_built_at(paths: Iterable[Path] | None = None) -> datetime | None:
    """Universe build stamp: newest mtime across the universe files."""
    stamps = []
    for path in paths or (UNIVERSE_ALL_FILE, UNIVERSE_LONGS_FILE, UNIVERSE_SHORTS_FILE):
        try:
            stamps.append(datetime.fromtimestamp(Path(path).stat().st_mtime))
        except OSError:
            continue
    return max(stamps) if stamps else None


_UNSET = object()


def universe_is_stale(
    now: datetime,
    built_at: datetime | None | object = _UNSET,
    local_timezone_name: str | None = None,
) -> bool:
    """Stale = built before the most recent completed session's close.

    A build from yesterday afternoon stays fresh all of today's session; the
    moment today's close passes, it goes stale (the after-close wrap-up
    rebuilds it with today's data).
    """
    if built_at is _UNSET:
        built_at = universe_built_at()
    if built_at is None:
        return True
    reference_close = last_completed_session_close(now, local_timezone_name)
    if reference_close is None:
        return False
    return built_at < reference_close


# One rebuild at a time, no matter who asks (GUI-launch self-heal, Auto Pilot
# tick, manual button) - they all funnel through this lock.
_UNIVERSE_REBUILD_LOCK = threading.Lock()


def rebuild_universe_if_stale(
    now: datetime | None = None,
    *,
    force: bool = False,
    log: Callable[[str], None] | None = None,
    builder: Callable[..., dict] | None = None,
    built_at: datetime | None | object = _UNSET,
) -> str:
    """Blocking rebuild (call from a worker thread). Returns one of
    "fresh" | "rebuilt" | "busy" | "failed"."""
    now = now or datetime.now()
    if not force and not universe_is_stale(now, built_at):
        return "fresh"
    if not _UNIVERSE_REBUILD_LOCK.acquire(blocking=False):
        return "busy"
    started = datetime.now()
    try:
        if builder is None:
            from universe_builder import DEFAULT_OPTIONS_FILTER, build_universe

            result = build_universe(options_filter=DEFAULT_OPTIONS_FILTER)
        else:
            result = builder()
        if log:
            elapsed = (datetime.now() - started).total_seconds()
            log(
                f"Universe rebuilt in {elapsed:.0f}s: {len(result.get('all', []))} total / "
                f"{len(result.get('longs', []))} longs / {len(result.get('shorts', []))} shorts."
            )
        return "rebuilt"
    except Exception as exc:
        logging.exception("Universe rebuild failed")
        if log:
            log(f"Universe rebuild failed: {exc}")
        return "failed"
    finally:
        _UNIVERSE_REBUILD_LOCK.release()


def after_close_wrapup_due(
    now: datetime,
    slots_done: Iterable[str],
    wrapup_done: bool,
    scan_running: bool,
    local_timezone_name: str | None = None,
) -> bool:
    """The wrap-up runs once, after every swing slot (incl. failures) is done."""
    if wrapup_done or scan_running or now.weekday() >= 5:
        return False
    slots = get_autopilot_swing_slots(now, local_timezone_name)
    if not slots:
        return False
    done = {str(slot) for slot in slots_done or []}
    return all(slot in done for slot in slots)


def merge_autopilot_watchlist(
    auto_picks: Iterable[str],
    current_symbols: Iterable[str],
    last_autopilot: Iterable[str],
) -> dict[str, list[str]]:
    """Fresh auto picks + the user's hand-added names.

    Anything in the file that Auto Pilot did NOT write last time is treated as
    the trader's and survives; yesterday's auto picks get replaced. (File
    mtimes can't make this distinction - the bot itself rewrites the files
    intraday - hence the explicit last-written state.)
    """
    last = {str(item or "").strip().upper() for item in last_autopilot or []}
    merged: list[str] = []
    seen: set[str] = set()
    for symbol in auto_picks or []:
        symbol = str(symbol or "").strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            merged.append(symbol)
    manual_kept: list[str] = []
    for symbol in current_symbols or []:
        symbol = str(symbol or "").strip().upper()
        if symbol and symbol not in seen and symbol not in last:
            seen.add(symbol)
            merged.append(symbol)
            manual_kept.append(symbol)
    return {"symbols": merged, "manual_kept": manual_kept}


def score_autopilot_picks(
    picks: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
    outcome_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Did the self-built watchlist produce anything? Join picks vs the
    day-trade candidate/outcome logs (confirmed events on the same date,
    symbol and direction; outcomes joined by event_id, last row wins)."""
    picks = [dict(pick) for pick in picks or []]
    pick_keys = {
        (
            str(pick.get("date") or "").strip(),
            str(pick.get("symbol") or "").strip().upper(),
            str(pick.get("side") or "").strip().lower(),
        )
        for pick in picks
    }

    latest_outcome: dict[str, Mapping[str, Any]] = {}
    for row in outcome_rows or []:
        event_id = str(row.get("event_id") or "").strip()
        if event_id:
            latest_outcome[event_id] = row

    alerted_symbols: set[str] = set()
    close_values: list[float] = []
    mfe_values: list[float] = []
    for row in candidate_rows or []:
        if str(row.get("event_type") or "").strip().lower() != "confirmed":
            continue
        key = (
            str(row.get("trade_date") or "").strip(),
            str(row.get("symbol") or "").strip().upper(),
            str(row.get("direction") or "").strip().lower(),
        )
        if key not in pick_keys:
            continue
        alerted_symbols.add(key[1])
        outcome = latest_outcome.get(str(row.get("event_id") or "").strip())
        if not outcome:
            continue
        for field, bucket in (("close_r", close_values), ("mfe_r", mfe_values)):
            try:
                bucket.append(float(outcome.get(field)))
            except (TypeError, ValueError):
                continue

    sides = [str(pick.get("side") or "").strip().lower() for pick in picks]
    return {
        "picks": len(picks),
        "longs": sides.count("long"),
        "shorts": sides.count("short"),
        "alerted": len(alerted_symbols),
        "alerted_symbols": sorted(alerted_symbols),
        "avg_close_r": (sum(close_values) / len(close_values)) if close_values else None,
        "avg_mfe_r": (sum(mfe_values) / len(mfe_values)) if mfe_values else None,
    }


def format_scorecard_line(scorecard: Mapping[str, Any], label: str = "Auto picks today") -> str:
    if not scorecard or not scorecard.get("picks"):
        return f"{label}: none logged."
    close_r = scorecard.get("avg_close_r")
    mfe_r = scorecard.get("avg_mfe_r")
    r_text = ""
    if close_r is not None or mfe_r is not None:
        close_part = f"avg close {close_r:+.2f}R" if close_r is not None else "avg close n/a"
        mfe_part = f"MFE {mfe_r:.2f}R" if mfe_r is not None else "MFE n/a"
        r_text = f", {close_part} / {mfe_part}"
    return (
        f"{label}: {scorecard.get('longs', 0)} longs + {scorecard.get('shorts', 0)} shorts "
        f"-> {scorecard.get('alerted', 0)} alerted{r_text}."
    )


# Pick provenance: the scorecard compares how the bot's own picks, its
# not-acted-on suggestions, and the trader's hand-picked names each performed.
PICK_SOURCE_GROUPS = {
    "open_scan": "auto",
    "hod_add": "auto",
    "suggestion": "suggested",
    "manual": "manual",
}
PICK_GROUP_LABELS = {
    "auto": "Bot picks",
    "suggested": "Bot suggestions (not acted on)",
    "manual": "Your picks",
}


def group_picks_by_source(picks: Iterable[Mapping[str, Any]]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for pick in picks or []:
        source = str(pick.get("source") or "").strip().lower()
        group = PICK_SOURCE_GROUPS.get(source, "auto")
        groups.setdefault(group, []).append(dict(pick))
    return groups


def format_suggestion_message(built: Mapping[str, Any], limit: int = 8) -> str:
    """One alert line with the open scan's suggested adds (manual-mode days)."""

    def _side(side_key: str, reasons_key: str) -> str:
        symbols = list(built.get(side_key) or [])[:limit]
        reasons = built.get(reasons_key) or {}
        return ", ".join(
            f"{symbol} ({reasons[symbol]})" if reasons.get(symbol) else symbol for symbol in symbols
        )

    parts = []
    if built.get("longs"):
        parts.append(f"longs: {_side('longs', 'long_reasons')}")
    if built.get("shorts"):
        parts.append(f"shorts: {_side('shorts', 'short_reasons')}")
    if not parts:
        return ""
    return "AUTO PILOT SUGGESTS - " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Open scan: gaps + early RS/RW vs SPY -> longs.txt / shorts.txt
# ---------------------------------------------------------------------------
def summarize_open_move(prev_close: float | None, today_bars: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    """{gap_pct, early_move_pct, last_price} from one symbol's session bars."""
    if not today_bars:
        return None
    try:
        today_open = float(today_bars[0]["open"])
        last_price = float(today_bars[-1]["close"])
    except (KeyError, TypeError, ValueError):
        return None
    if not today_open:
        return None
    gap_pct = None
    if prev_close:
        gap_pct = (today_open - float(prev_close)) / float(prev_close) * 100.0
    early_move_pct = (last_price - today_open) / today_open * 100.0
    session_date = None
    last_dt = today_bars[-1].get("dt")
    if last_dt is not None and hasattr(last_dt, "date"):
        session_date = last_dt.date()
    return {
        "gap_pct": gap_pct,
        "early_move_pct": early_move_pct,
        "last_price": last_price,
        "session_date": session_date,
    }


def build_watchlists_from_moves(
    moves: Mapping[str, Mapping[str, Any]],
    spy_move: Mapping[str, Any] | None,
    *,
    gap_min_pct: float = AUTOPILOT_GAP_MIN_PCT,
    rs_excess_min_pct: float = AUTOPILOT_RS_EXCESS_MIN_PCT,
    cap: int = AUTOPILOT_WATCHLIST_CAP,
) -> dict[str, Any]:
    """Rank the open-scan moves into capped longs/shorts lists.

    Longs: gap up >= gap_min_pct OR early move beating SPY by rs_excess_min_pct.
    Shorts inverted. A symbol qualifying both ways keeps its stronger side.
    """
    spy_early = 0.0
    if spy_move and spy_move.get("early_move_pct") is not None:
        spy_early = float(spy_move["early_move_pct"])

    longs: list[tuple[float, str, str]] = []
    shorts: list[tuple[float, str, str]] = []
    for symbol, move in moves.items():
        symbol = str(symbol or "").strip().upper()
        if not symbol or symbol == "SPY" or not isinstance(move, Mapping):
            continue
        early = move.get("early_move_pct")
        if early is None:
            continue
        early = float(early)
        gap = move.get("gap_pct")
        gap = float(gap) if gap is not None else 0.0
        excess = early - spy_early

        long_reasons = []
        if gap >= gap_min_pct:
            long_reasons.append(f"gap {gap:+.1f}%")
        if excess >= rs_excess_min_pct:
            long_reasons.append(f"RS {excess:+.1f}% vs SPY")
        short_reasons = []
        if gap <= -gap_min_pct:
            short_reasons.append(f"gap {gap:+.1f}%")
        if excess <= -rs_excess_min_pct:
            short_reasons.append(f"RW {excess:+.1f}% vs SPY")

        long_score = max(gap, 0.0) + max(excess, 0.0)
        short_score = max(-gap, 0.0) + max(-excess, 0.0)
        if long_reasons and (not short_reasons or long_score >= short_score):
            longs.append((long_score, symbol, ", ".join(long_reasons)))
        elif short_reasons:
            shorts.append((short_score, symbol, ", ".join(short_reasons)))

    longs.sort(key=lambda item: (-item[0], item[1]))
    shorts.sort(key=lambda item: (-item[0], item[1]))
    cap = max(1, int(cap))
    return {
        "longs": [symbol for _score, symbol, _why in longs[:cap]],
        "shorts": [symbol for _score, symbol, _why in shorts[:cap]],
        "long_reasons": {symbol: why for _score, symbol, why in longs[:cap]},
        "short_reasons": {symbol: why for _score, symbol, why in shorts[:cap]},
        "scanned": len(moves),
    }


def load_universe_pool(max_symbols: int = AUTOPILOT_OPEN_SCAN_MAX_SYMBOLS) -> list[str]:
    """Candidate pool for the open scan: the self-built universe lists."""
    symbols: list[str] = []
    seen: set[str] = set()
    for path in (UNIVERSE_LONGS_FILE, UNIVERSE_SHORTS_FILE, UNIVERSE_ALL_FILE):
        try:
            for symbol in read_watchlist_symbols(Path(path)):
                symbol = str(symbol or "").strip().upper()
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    symbols.append(symbol)
        except Exception:
            continue
        if len(symbols) >= max_symbols:
            break
    return symbols[:max_symbols]


def _default_downloader(symbols: list[str], *, period: str, interval: str):
    import yfinance as yf

    return yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )


def _frame_rows(frame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame is None:
        return rows
    try:
        empty = frame.empty
    except AttributeError:
        return rows
    if empty:
        return rows
    for stamp, row in frame.iterrows():
        try:
            open_val = float(row["Open"])
            close_val = float(row["Close"])
            high_val = float(row["High"])
            low_val = float(row["Low"])
        except (KeyError, TypeError, ValueError):
            continue
        if open_val != open_val or close_val != close_val:  # NaN guard
            continue
        rows.append(
            {
                "dt": stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
            }
        )
    return rows


def _split_last_session(rows: list[dict[str, Any]]) -> tuple[float | None, list[dict[str, Any]]]:
    """(previous session close, latest session bars) from mixed-day rows."""
    if not rows:
        return None, []
    last_date = rows[-1]["dt"].date()
    today = [row for row in rows if row["dt"].date() == last_date]
    prior = [row for row in rows if row["dt"].date() < last_date]
    prev_close = prior[-1]["close"] if prior else None
    return prev_close, today


def fetch_open_scan_moves(
    symbols: Iterable[str],
    *,
    downloader: Callable[..., Any] | None = None,
    chunk_size: int = AUTOPILOT_OPEN_SCAN_CHUNK_SIZE,
    log: Callable[[str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Batch yfinance 5m data (2 sessions) -> per-symbol open-scan moves.

    Includes SPY automatically so the caller can compute excess moves.
    """
    downloader = downloader or _default_downloader
    pool = [str(symbol or "").strip().upper() for symbol in symbols]
    pool = [symbol for symbol in pool if symbol]
    if "SPY" not in pool:
        pool.insert(0, "SPY")

    moves: dict[str, dict[str, Any]] = {}
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(pool), chunk_size):
        chunk = pool[start : start + chunk_size]
        try:
            data = downloader(chunk, period="2d", interval="5m")
        except Exception as exc:
            if log:
                log(f"Open scan chunk failed ({chunk[0]}..{chunk[-1]}): {exc}")
            continue
        for symbol in chunk:
            frame = None
            try:
                frame = data[symbol] if len(chunk) > 1 else data
            except Exception:
                frame = None
            prev_close, today = _split_last_session(_frame_rows(frame))
            summary = summarize_open_move(prev_close, today)
            if summary is not None:
                moves[symbol] = summary
    return moves


def fetch_day_snapshot(
    symbols: Iterable[str],
    *,
    downloader: Callable[..., Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, dict[str, float]]:
    """{symbol: {last, day_high, day_low}} from today's 5m bars."""
    downloader = downloader or _default_downloader
    pool = sorted({str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()})
    if not pool:
        return {}
    try:
        data = downloader(pool, period="1d", interval="5m")
    except Exception as exc:
        if log:
            log(f"Day snapshot fetch failed: {exc}")
        return {}
    snapshot: dict[str, dict[str, float]] = {}
    for symbol in pool:
        try:
            frame = data[symbol] if len(pool) > 1 else data
        except Exception:
            frame = None
        rows = _frame_rows(frame)
        if not rows:
            continue
        snapshot[symbol] = {
            "last": rows[-1]["close"],
            "day_high": max(row["high"] for row in rows),
            "day_low": min(row["low"] for row in rows),
        }
    return snapshot


def near_extreme_candidates(
    snapshot: Mapping[str, Mapping[str, float]],
    side: str,
    proximity_pct: float = AUTOPILOT_HOD_PROXIMITY_PCT,
) -> list[str]:
    """Symbols trading within proximity_pct of their HOD (longs) / LOD (shorts)."""
    matches: list[str] = []
    for symbol, quote in snapshot.items():
        try:
            last = float(quote["last"])
            day_high = float(quote["day_high"])
            day_low = float(quote["day_low"])
        except (KeyError, TypeError, ValueError):
            continue
        if side == "long":
            if day_high <= 0:
                continue
            distance = (day_high - last) / day_high * 100.0
        else:
            if last <= 0:
                continue
            distance = (last - day_low) / last * 100.0
        if 0 <= distance <= proximity_pct:
            matches.append(str(symbol).strip().upper())
    return sorted(matches)


# ---------------------------------------------------------------------------
# Watchlist file IO
# ---------------------------------------------------------------------------
def write_watchlist_file(path: Path, symbols: Iterable[str]) -> None:
    cleaned = []
    seen = set()
    for symbol in symbols:
        symbol = str(symbol or "").strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            cleaned.append(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(cleaned) + ("\n" if cleaned else ""), encoding="utf-8")


def append_watchlist_symbols(path: Path, symbols: Iterable[str]) -> list[str]:
    """Append new symbols to a watchlist file; returns what was added."""
    existing = list(read_watchlist_symbols(Path(path))) if Path(path).exists() else []
    existing_set = {str(item).strip().upper() for item in existing}
    added = []
    for symbol in symbols:
        symbol = str(symbol or "").strip().upper()
        if symbol and symbol not in existing_set:
            existing.append(symbol)
            existing_set.add(symbol)
            added.append(symbol)
    if added:
        write_watchlist_file(Path(path), existing)
    return added


def write_bouncebot_watchlists(longs: Iterable[str], shorts: Iterable[str]) -> None:
    write_watchlist_file(Path(LONGS_FILE), longs)
    write_watchlist_file(Path(SHORTS_FILE), shorts)


# ---------------------------------------------------------------------------
# Away report (shared Google Drive digest)
# ---------------------------------------------------------------------------
def render_away_report(payload: Mapping[str, Any]) -> str:
    """Phone-first digest: tickers up top, operations detail below."""

    def _lines(items: Iterable[str]) -> str:
        items = [str(item) for item in items if str(item).strip()]
        return "\n".join(items) if items else "(none)"

    def _tickers(items: Iterable[str]) -> str:
        items = [str(item).strip().upper() for item in items if str(item).strip()]
        return ", ".join(items) if items else "(none)"

    picks_lines = []
    for pick in payload.get("swing_picks", []) or []:
        if not isinstance(pick, Mapping):
            continue
        symbol = str(pick.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        side = str(pick.get("side") or "").strip().upper() or "?"
        bucket = str(pick.get("bucket") or "").strip()
        expected_r = pick.get("expected_r")
        expected_text = f" | {float(expected_r):.2f}R" if expected_r is not None else ""
        bucket_text = f" | {bucket}" if bucket else ""
        picks_lines.append(f"{symbol} ({side}){bucket_text}{expected_text}")

    def _tv_line(items: Iterable[str]) -> str:
        items = [str(item).strip().upper() for item in items if str(item).strip()]
        return ",".join(items) if items else "(none)"

    header_bits = [
        f"Mode: {'ON' if payload.get('enabled') else 'OFF'} | IB: {payload.get('ib_status', 'unknown')} | Regime: {payload.get('regime', 'unknown')}",
    ]
    if payload.get("universe_line"):
        header_bits.append(str(payload["universe_line"]))
    if payload.get("scorecard_line"):
        header_bits.append(str(payload["scorecard_line"]))

    sections = [
        "TRADINGBOT AUTO PILOT - TODAY",
        f"Updated: {payload.get('generated_at', '')}",
        *header_bits,
        "",
        "== DAY TRADE LONGS (longs.txt) ==",
        _tickers(payload.get("longs", [])),
        f"TV paste: {_tv_line(payload.get('longs', []))}",
        "",
        "== DAY TRADE SHORTS (shorts.txt) ==",
        _tickers(payload.get("shorts", [])),
        f"TV paste: {_tv_line(payload.get('shorts', []))}",
        "",
        "== TOP SWING PICKS ==",
        _lines(picks_lines),
        "",
        "== TODAY'S ALERTS (latest first) ==",
        _lines(payload.get("alerts", [])),
        "",
        "== SCHEDULE ==",
        f"Swing slots done: {_tickers(payload.get('slots_done', []))}",
        f"Next swing slot: {payload.get('next_slot') or '(none left today)'}",
        "",
        "== ACTIVITY LOG (latest first) ==",
        _lines(payload.get("log_lines", [])),
        "",
    ]
    return "\n".join(sections)


def write_away_report(payload: Mapping[str, Any], path: Path | None = None) -> Path:
    target = Path(path) if path is not None else Path(AUTOPILOT_REPORT_FILE)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_away_report(payload), encoding="utf-8")
    except Exception:
        logging.exception("Failed writing the Auto Pilot away report to %s", target)
    return target
