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
- Aggressive universe discovery: repeated completed-M5 HODs on bearish days
  seed counter-trend longs, while LOD holders during a legacy-detected SPY
  rebound seed shorts. Bullish days mirror both rules exactly.
- A phone-digestible report written to the shared home folder.

Everything here is deliberately testable: scheduling, ranking, filtering and
rendering are pure; the yfinance fetchers accept an injectable downloader.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping

import focus_adoption_gate
import prev_day_gate
from market_session import get_market_session_window, normalize_market_local_datetime
from project_paths import (
    AUTO_LONGS_FILE,
    AUTO_POPULATE_MEMBERSHIP_FILE,
    AUTO_POPULATE_PENDING_FILE,
    AUTO_SHORTS_FILE,
    AUTOPILOT_REPORT_FILE,
    AUTOPILOT_STATE_FILE,
    LONGS_FILE,
    MASTER_AVWAP_DAILY_BARS_DIR,
    RUNTIME_DATA_DIR,
    SHORTS_FILE,
    UNIVERSE_ALL_FILE,
    UNIVERSE_LONGS_FILE,
    UNIVERSE_SHORTS_FILE,
)
from watchlist_utils import read_watchlist_symbols

# Scheduling (minutes are relative to the session open, local clock).
AUTOPILOT_FIRST_SCAN_AFTER_OPEN_MINUTES = 60
# Evening mode's early Master AVWAP run: open+30 = 07:00 on a normal session,
# so the briefing's best-D1 ranking is ready before the trader sits down.
AUTOPILOT_EVENING_EARLY_SCAN_AFTER_OPEN_MINUTES = 30
AUTOPILOT_WATCHLIST_BUILD_AFTER_OPEN_MINUTES = 30
AUTOPILOT_WATCHLIST_BUILD_DEADLINE_MINUTES = 120

# Watchlist self-build thresholds.
AUTOPILOT_GAP_MIN_PCT = 2.0  # open vs yesterday's close
AUTOPILOT_RS_EXCESS_MIN_PCT = 1.5  # early move minus SPY's early move
AUTOPILOT_WATCHLIST_CAP = 40  # per side; protects BounceBot's IB pacing
# (25 -> 40 on 2026-07-23: the open scan hit the 25 cap on both sides nearly
# every session since 07-10 while pick quality held up, so the cap - not the
# gap/RS gates - was the binding constraint. Raise further only after a live
# session confirms the pacing governor stays quiet at this size.)
AUTOPILOT_OPEN_SCAN_MAX_SYMBOLS = 1200
AUTOPILOT_OPEN_SCAN_CHUNK_SIZE = 150

# Near-HOD/LOD adds during regime pauses.
AUTOPILOT_HOD_PROXIMITY_PCT = 1.0
AUTOPILOT_HOD_TOP_ROWS = 30
AUTOPILOT_HOD_CHECK_COOLDOWN_MINUTES = 30

# Universe freshness: Auto Pilot is used sporadically, so freshness is checked
# on every activation/tick instead of trusting a nightly job to have run.
AUTOPILOT_UNIVERSE_RETRY_MINUTES = 60
_CANDIDATE_REGISTRY_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
def get_autopilot_swing_slots(
    reference: datetime | None = None,
    local_timezone_name: str | None = None,
    *,
    include_early_slot: bool = False,
) -> list[str]:
    """The away-day swing scan slots as local HH:MM labels.

    First slot at open+60m; the API's first hour belongs to the intraday
    strong/weak discovery. Hourly top-of-hour slots resume from the first
    full hour after that (09:00 for a 06:30 open - deliberately skipping
    08:00) through the close slot.

    ``include_early_slot`` (Evening mode) prepends an open+30 slot - 07:00 on
    a normal 06:30 session - so the Master AVWAP setups are already scanned
    when the trader arrives at 07:00-07:30, and the morning briefing can rank
    the best D1s off a current run instead of yesterday's.
    """
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    open_naive = session.open_local.replace(tzinfo=None)
    close_naive = session.close_local.replace(tzinfo=None)

    first = open_naive + timedelta(minutes=AUTOPILOT_FIRST_SCAN_AFTER_OPEN_MINUTES)
    slots = [first]
    if include_early_slot:
        early = open_naive + timedelta(minutes=AUTOPILOT_EVENING_EARLY_SCAN_AFTER_OPEN_MINUTES)
        if early < first:
            slots.insert(0, early)
    cursor = first + timedelta(minutes=60)
    if cursor.minute or cursor.second or cursor.microsecond:
        cursor = cursor.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while cursor <= close_naive:
        slots.append(cursor)
        cursor += timedelta(hours=1)
    return [slot.strftime("%H:%M") for slot in slots]


def autopilot_evening_early_slot(
    reference: datetime | None = None,
    local_timezone_name: str | None = None,
) -> str:
    """Local HH:MM label of Evening mode's open+30 early swing slot.

    Evening runs this one slot and then stops for the day (trader rule
    2026-08-14), so the label has to be nameable on its own rather than only
    as the first entry of a full slot list.
    """
    session = get_market_session_window(
        reference=reference, local_timezone_name=local_timezone_name
    )
    open_naive = session.open_local.replace(tzinfo=None)
    early = open_naive + timedelta(
        minutes=AUTOPILOT_EVENING_EARLY_SCAN_AFTER_OPEN_MINUTES
    )
    return early.strftime("%H:%M")


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


# BounceBot's intraday sweep only earns its IB traffic while the tape moves.
# Auto Pilot used to re-enable scanning on every 30-second tick with no clock
# check at all, so a desk left running swept ~95-150 names about eight times an
# hour straight through the night against prices that had not changed since the
# close (trader report 2026-08-10; measured in trading_bot.log). The window
# below is the session plus a warm-up and a wind-down. Outside it the sweep
# pauses; the bot object and its IB connection stay up, so the open costs no
# reconnect and BounceBot's paused branch keeps its regime read live.
BOUNCEBOT_SCAN_PREOPEN_MINUTES = 30
BOUNCEBOT_SCAN_POSTCLOSE_MINUTES = 30
BOUNCEBOT_SCAN_SESSION_ONLY_SETTING = "qt_bouncebot_scan_session_only"
BOUNCEBOT_SCAN_PREOPEN_SETTING = "qt_bouncebot_scan_preopen_minutes"
BOUNCEBOT_SCAN_POSTCLOSE_SETTING = "qt_bouncebot_scan_postclose_minutes"


def _scan_margin_minutes(setting_key: str, default: int) -> int:
    """A configured margin in minutes, clamped to something sane.

    A negative margin would eat into the session itself, so it is refused
    rather than honored; an unreadable or non-numeric value falls back to the
    default instead of silently collapsing the window to nothing.
    """
    try:
        from project_paths import get_local_setting

        raw = get_local_setting(setting_key, default)
    except Exception:
        return default
    try:
        minutes = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(0, min(minutes, 12 * 60))


def bouncebot_scan_window(
    reference: datetime | None = None,
    local_timezone_name: str | None = None,
) -> tuple[datetime, datetime]:
    """Naive local start/end of the window BounceBot's sweep may run in.

    The session bounds come from `market_session`, the one source of truth for
    them, so a timezone change moves this window with everything else instead
    of drifting away from it. Early closes are NOT modelled - the session
    helper hardcodes regular hours - so a half-day still reports the regular
    close. Fail-open: the window is too long, never too short.
    """
    session = get_market_session_window(
        reference=reference, local_timezone_name=local_timezone_name
    )
    start = session.open_local.replace(tzinfo=None) - timedelta(
        minutes=_scan_margin_minutes(
            BOUNCEBOT_SCAN_PREOPEN_SETTING, BOUNCEBOT_SCAN_PREOPEN_MINUTES
        )
    )
    end = session.close_local.replace(tzinfo=None) + timedelta(
        minutes=_scan_margin_minutes(
            BOUNCEBOT_SCAN_POSTCLOSE_SETTING, BOUNCEBOT_SCAN_POSTCLOSE_MINUTES
        )
    )
    return start, end


def bouncebot_scanning_due(
    now: datetime,
    *,
    session_only: bool | None = None,
    local_timezone_name: str | None = None,
) -> tuple[bool, str]:
    """Whether BounceBot's intraday sweep belongs on the wire right now.

    Returns the verdict and a short reason fit for the Auto Pilot log. Weekends
    are closed outright - there is no session to sweep, and the tick loop's own
    weekend short-circuit means nothing else would ever pause a sweep that ran
    into Friday evening.
    """
    if session_only is None:
        try:
            from project_paths import get_local_setting

            session_only = bool(get_local_setting(BOUNCEBOT_SCAN_SESSION_ONLY_SETTING, True))
        except Exception:
            session_only = True
    if not session_only:
        return True, "session-only scanning is disabled; scanning around the clock"
    if now.weekday() >= 5:
        return False, "weekend - no session to scan"
    start, end = bouncebot_scan_window(
        reference=now, local_timezone_name=local_timezone_name
    )
    label = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
    if start <= now <= end:
        return True, f"inside the {label} scan window"
    return False, f"outside the {label} scan window"


# ---------------------------------------------------------------------------
# Quiet hours (automatic work is confined to the session and its shoulders)
# ---------------------------------------------------------------------------
# Trader rule 2026-08-14 (packet R1): automatic work runs on weekdays from the
# session open through one hour after the close, and at no other time. An app
# boot outside the window starts nothing - no universe rebuild, no bar
# fetching, no BounceBot IB connect from saved Auto state, no daily self-arm -
# until either the window opens or the trader acts by hand.
#
# MANUAL WORK IS NEVER GATED. Only automatic starters consult this; every
# trader-initiated button (manual scan, manual rebuild, manual resume) runs at
# any hour, the same way the manual master-scan buttons already bypass the
# Auto Pilot scheduler.
#
# The pre-open margin defaults to BounceBot's own warm-up rather than to zero.
# This window has to be a SUPERSET of every window it gates or the gates
# contradict one another: at 06:10 a zero-margin quiet-hours gate would refuse
# the IB connect while `bouncebot_scanning_due` said the sweep could run,
# stranding the sweep with no connection to run on.
AUTO_QUIET_HOURS_PREOPEN_MINUTES = BOUNCEBOT_SCAN_PREOPEN_MINUTES
AUTO_QUIET_HOURS_POSTCLOSE_MINUTES = 60
AUTO_QUIET_HOURS_SETTING = "qt_auto_quiet_hours"
AUTO_QUIET_HOURS_PREOPEN_SETTING = "qt_auto_quiet_hours_preopen_minutes"
AUTO_QUIET_HOURS_POSTCLOSE_SETTING = "qt_auto_quiet_hours_postclose_minutes"


def auto_scanning_window(
    reference: datetime | None = None,
    local_timezone_name: str | None = None,
) -> tuple[datetime, datetime]:
    """Naive local start/end of the window automatic work may run in.

    Like `bouncebot_scan_window`, the bounds come from `market_session`, so a
    timezone change moves this with everything else instead of drifting away
    from it. (Early closes are NOT modelled anywhere yet - the session helper
    hardcodes regular hours - so a half-day still reports the regular close.
    Pre-existing, and fail-open: the window is too long, never too short.)

    The result is widened to CONTAIN the BounceBot sweep window. The containment
    is what makes the two gates coherent - quiet hours decide whether BounceBot
    may connect, and the sweep window decides whether it may sweep, so a quiet
    window that opened later would strand the sweep with no connection. Both
    windows have independent settings keys and could otherwise be configured
    into that contradiction; widening here means they cannot be.
    """
    session = get_market_session_window(
        reference=reference, local_timezone_name=local_timezone_name
    )
    start = session.open_local.replace(tzinfo=None) - timedelta(
        minutes=_scan_margin_minutes(
            AUTO_QUIET_HOURS_PREOPEN_SETTING, AUTO_QUIET_HOURS_PREOPEN_MINUTES
        )
    )
    end = session.close_local.replace(tzinfo=None) + timedelta(
        minutes=_scan_margin_minutes(
            AUTO_QUIET_HOURS_POSTCLOSE_SETTING, AUTO_QUIET_HOURS_POSTCLOSE_MINUTES
        )
    )
    try:
        sweep_start, sweep_end = bouncebot_scan_window(
            reference=reference, local_timezone_name=local_timezone_name
        )
    except Exception:
        logging.exception("Sweep-window lookup failed; quiet hours use their own bounds.")
        return start, end
    return min(start, sweep_start), max(end, sweep_end)


def auto_scanning_due(
    now: datetime,
    *,
    quiet_hours: bool | None = None,
    local_timezone_name: str | None = None,
) -> tuple[bool, str]:
    """Whether automatic work may start right now, plus a reason for the log.

    Fails OPEN. A session lookup this cannot answer must never be the reason
    the desk sits out a trading day: an extra overnight rebuild is waste, a
    silent daytime refusal is a missed session.
    """
    if quiet_hours is None:
        try:
            from project_paths import get_local_setting

            quiet_hours = bool(get_local_setting(AUTO_QUIET_HOURS_SETTING, True))
        except Exception:
            quiet_hours = True
    if not quiet_hours:
        return True, "quiet hours are disabled; automatic work runs around the clock"
    if now.weekday() >= 5:
        return False, "weekend - quiet hours until the next session"
    try:
        start, end = auto_scanning_window(
            reference=now, local_timezone_name=local_timezone_name
        )
    except Exception:
        logging.exception("Quiet-hours window lookup failed; allowing automatic work.")
        return True, "session window unavailable; allowing automatic work"
    label = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
    if start <= now <= end:
        return True, f"inside the {label} automatic-work window"
    return False, f"quiet hours - outside the {label} automatic-work window"


# ---------------------------------------------------------------------------
# Evening's SPY wake alarm
# ---------------------------------------------------------------------------
# Trader rule 2026-08-14: Evening is armed after a late shift, when the trader
# sleeps through the open. Its second job - after having the day's trades ready
# on waking - is to wake the trader if the market actually moves. This is the
# SECOND deliberate exception to the AWAY-only phone-push rule; the first is
# the always-on Research/Focus price alerts.
EVENING_SPY_ALARM_PCT = 1.0
EVENING_SPY_ALARM_REPEAT_SECONDS = 300
EVENING_SPY_ALARM_SETTING = "push_evening_spy_alarm"
EVENING_SPY_ALARM_PCT_SETTING = "push_evening_spy_alarm_pct"


def spy_move_alarm_due(
    day_pct: float | None,
    last_sent_at: datetime | None,
    now: datetime,
    *,
    threshold_pct: float = EVENING_SPY_ALARM_PCT,
    repeat_seconds: int = EVENING_SPY_ALARM_REPEAT_SECONDS,
) -> bool:
    """Whether Evening's SPY alarm should phone right now.

    Missing data is uncertainty, never confirmation (plan.md sec 5): an absent,
    unreadable or NaN day-percent is silence, not an alarm. NaN is checked
    explicitly because `nan < threshold` is False, so a NaN would otherwise
    read as "past the threshold" and phone the trader awake over no data at
    all.
    """
    if day_pct is None:
        return False
    try:
        move = abs(float(day_pct))
    except (TypeError, ValueError):
        return False
    if move != move:  # NaN
        return False
    try:
        threshold = abs(float(threshold_pct))
    except (TypeError, ValueError):
        threshold = EVENING_SPY_ALARM_PCT
    if threshold != threshold:  # NaN threshold would make every move qualify
        threshold = EVENING_SPY_ALARM_PCT
    if move < threshold:
        return False
    if last_sent_at is None:
        return True
    return (now - last_sent_at).total_seconds() >= float(repeat_seconds)


# Hands-off default (2026-07-09, user rule "everything automatic - all I do is
# fill longs/shorts.txt"): Auto Pilot arms itself once per weekday at/after
# this local hour. Arming once per day means a manual OFF sticks for the rest
# of that day - the trader's hand always wins.
AUTOPILOT_AUTO_ARM_HOUR = 7
AUTOPILOT_AWAY_REPORT_START_HOUR = 7


def hourly_away_report_slot_due(
    now: datetime,
    *,
    last_completed_slot: str | None = None,
    start_hour: int = AUTOPILOT_AWAY_REPORT_START_HOUR,
    local_timezone_name: str | None = None,
) -> str | None:
    """Return the current hourly report slot once, from 07:00 through close.

    Starting the app partway through an hour performs one catch-up publication
    for that hour. The persisted date in the slot key prevents yesterday's
    completion marker from suppressing today's report.
    """
    if now.weekday() >= 5 or now.hour < int(start_hour):
        return None
    session = get_market_session_window(
        reference=now,
        local_timezone_name=local_timezone_name,
    )
    close_naive = session.close_local.replace(tzinfo=None)
    if now.hour > close_naive.hour:
        return None
    slot = f"{now.date().isoformat()}|{now.hour:02d}:00"
    return None if str(last_completed_slot or "") == slot else slot


def autopilot_auto_arm_due(
    now: datetime,
    *,
    enabled: bool,
    armed_date: str | None,
    auto_arm_enabled: bool = True,
    arm_hour: int = AUTOPILOT_AUTO_ARM_HOUR,
    local_timezone_name: str | None = None,
    quiet_hours: bool | None = None,
) -> bool:
    """True when the daily 07:00 self-arm should flip Auto Pilot ON.

    The arm hour is a floor, quiet hours are the ceiling: before packet R1 this
    had no upper bound at all, so launching the desk at 21:00 on a weekday
    self-armed Auto Pilot and set the whole automatic apparatus going against a
    closed tape.
    """
    if enabled or not auto_arm_enabled:
        return False
    if now.weekday() >= 5:
        return False
    if now.hour < int(arm_hour):
        return False
    allowed, _ = auto_scanning_due(
        now, quiet_hours=quiet_hours, local_timezone_name=local_timezone_name
    )
    if not allowed:
        return False
    return str(armed_date or "") != now.date().isoformat()


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
    """Universe build stamp: the OLDEST mtime across all required files.

    A multi-file generation is only as fresh as its stalest member; using the
    newest mtime let one fresh file hide stale or missing companions
    (plan.md 23.5). A missing required file means there is no valid
    generation at all, so the caller treats the universe as stale/absent.
    """
    stamps = []
    for path in paths or (UNIVERSE_ALL_FILE, UNIVERSE_LONGS_FILE, UNIVERSE_SHORTS_FILE):
        try:
            stamps.append(datetime.fromtimestamp(Path(path).stat().st_mtime))
        except OSError:
            return None  # incomplete generation: never report it as built
    return min(stamps) if stamps else None


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
    trend_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank the open-scan moves into capped longs/shorts lists.

    Longs: gap up >= gap_min_pct OR early move beating SPY by rs_excess_min_pct.
    Shorts inverted. A symbol qualifying both ways keeps its stronger side.

    When ``trend_context`` is provided (a :func:`load_daily_context` map), the
    daily-trend gate is applied BEFORE the cap so quality names fill the list:
    longs must sit above the daily 15EMA and 200SMA, shorts below the 15EMA and
    50SMA. It fails open when the context is empty (no daily store available).
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
            if _daily_trend_allows("long", symbol, trend_context):
                longs.append((long_score, symbol, ", ".join(long_reasons)))
        elif short_reasons:
            if _daily_trend_allows("short", symbol, trend_context):
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
        try:
            volume_val = float(row["Volume"])
        except (KeyError, TypeError, ValueError):
            volume_val = 0.0
        if volume_val != volume_val or volume_val < 0:  # NaN guard
            volume_val = 0.0
        rows.append(
            {
                "dt": stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "volume": volume_val,
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
# Universe auto-populate (2026-07-08): keep longs.txt/shorts.txt stocked from
# the universe all session. Longs = above the previous day's high with a real
# move relative to ADR; shorts = below the previous day's low, inverted. Time
# spent at HOD/LOD sweetens the rank. Regime decides how many of each side.
# The engine only ever rotates/removes names IT added (membership file) - the
# trader's own entries are untouchable. BounceBot's existing triple-VWAP
# removal (check_removal_conditions) handles intraday cuts; cut names land in
# a day-scoped blacklist here so a refresh doesn't re-add them.
# ---------------------------------------------------------------------------
AUTO_POPULATE_REFRESH_MINUTES = 30
AUTO_POPULATE_ADR_SESSIONS = 14
# "A good move relative to ADR": at least half an average daily range from
# yesterday's close, on top of the PDH/PDL break itself.
AUTO_POPULATE_MIN_ADR_MOVE = 0.5
# A 5m bar counts as "at the extreme" when its high (low) is within this % of
# the day's final high (low); the fraction of such bars sweetens the score.
AUTO_POPULATE_EXTREME_PROXIMITY_PCT = 0.3
# Aggressive completed-M5 discovery. Three real extreme breaks in six bars is
# deliberately more permissive than the PDH + 0.5 ADR rule, but avoids treating
# one wick or sub-tick drift as persistent pressure. Pullback holders must be
# tighter to HOD/LOD than the older 1% swing-row alert.
AGGRESSIVE_EXTREME_FEATURE_VERSION = "aggressive_extremes_v1"
AGGRESSIVE_EXTREME_WINDOW_BARS = 6
AGGRESSIVE_MIN_NEW_EXTREMES = 3
AGGRESSIVE_MIN_EXTREME_BREAK_PCT = 0.05
AGGRESSIVE_NEAR_EXTREME_PCT = 0.35
AGGRESSIVE_MAX_DATA_AGE_MINUTES = 15
AGGRESSIVE_SPY_PULLBACK_MIN_MOVE_PCT = 0.03
# Relative weakness/strength vs SPY (2026-07-17 study: SNPS/AA/SBLK/PSKY were
# the day's clean shorts; all four showed session moves lagging SPY by >=2%
# while pressing their lows on >=1.0 same-time-of-day relative volume, and the
# first three flagged during SPY's opening bounce - exactly when a weak name
# shows its hand by failing to lift).
RELATIVE_WEAKNESS_FEATURE_VERSION = "relative_weakness_v1"
RW_MIN_EXCESS_PCT = 2.0  # session move must lag (lead) SPY's by this much
RW_MIN_SESSION_MOVE_PCT = 1.0  # and the name itself must be truly moving
RW_NEAR_EXTREME_PCT = 1.0  # pressing: close within 1% of completed LOD/HOD
RW_MIN_SESSION_RVOL = 1.0  # trader's TC2000 bar: over 1.00 = real participation
# Quality bar (2026-07-16, trader directive): the auto slice takes what meets
# the criteria well - it does not fill toward the cap. Score = |ADR move| +
# fraction of the day spent at the extreme, so 1.25 means e.g. a 1.25-ADR
# breaker, or a 0.8-ADR mover that spent ~half the session pressing HOD/LOD.
# On the 2026-07-16 bearish tape this bar floats the short side to ~49 names
# instead of a cap-pegged 150 whose tail ran -0.6 ADR barely-breakers.
AUTO_POPULATE_MIN_SCORE = 1.25
# Ceiling only - the quality bar governs the count, symmetric both sides.
AUTO_POPULATE_MAX_PER_SIDE = 100
# Auto-populate never runs before this local-clock hour (trader directive
# 2026-07-31: picks start at 07:00, same clock as the Auto Pilot auto-arm).
AUTO_POPULATE_START_HOUR = 7

_AUTO_POPULATE_LOCK = threading.Lock()
_DAILY_CONTEXT_CACHE: dict[str, Any] = {"date": None, "contexts": {}}

# Daily-trend quality gate (2026-07-18, trader directive: "too many trash stocks
# that are just 1-day wonders"). A long is only auto-added when its last COMPLETED
# daily close sits above BOTH the daily 15EMA and 200SMA; a short only when the
# close sits below BOTH the daily 15EMA and 50SMA. Evaluated on completed daily
# bars so an intraday pop can't sneak a structurally broken name onto the list. A
# name without enough history for the required average is treated as failing (it
# cannot be "above its 200SMA" without 200 sessions). The gate fails OPEN only when
# no daily store is available at all, so a missing store never empties the lists.
AUTO_POPULATE_TREND_EMA = 15
AUTO_POPULATE_LONG_TREND_SMA = 200
AUTO_POPULATE_SHORT_TREND_SMA = 50


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def passes_daily_trend_gate(side: str, ctx: Mapping[str, Any] | None) -> bool:
    """Daily-trend quality gate for one auto-populate candidate.

    Long: close >= 15EMA and close >= 200SMA. Short: close <= 15EMA and close
    <= 50SMA. Missing close/EMA/SMA -> fails (cannot verify the structure).
    ``ctx`` is a daily-context entry from :func:`load_daily_context`.
    """
    if not isinstance(ctx, Mapping):
        return False
    close = _finite_float(ctx.get("prev_close"))
    ema = _finite_float(ctx.get("ema15"))
    if close is None or ema is None:
        return False
    if str(side or "").strip().lower().startswith("short"):
        sma = _finite_float(ctx.get("sma50"))
        if sma is None:
            return False
        return close <= ema and close <= sma
    sma = _finite_float(ctx.get("sma200"))
    if sma is None:
        return False
    return close >= ema and close >= sma


def _daily_trend_allows(side: str, symbol: str, trend_context: Mapping[str, Any] | None) -> bool:
    """Per-symbol gate wrapper. Fails OPEN when no gating is requested
    (``trend_context is None``) or no daily store is available (empty context),
    so the pipeline never empties when the durable daily bars are missing."""
    if trend_context is None or not trend_context:
        return True
    return passes_daily_trend_gate(side, trend_context.get(str(symbol or "").strip().upper()))


def filter_candidates_by_daily_trend(
    candidates: Mapping[str, list[dict[str, Any]]],
    trend_context: Mapping[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Drop auto-populate candidate rows that fail the daily-trend gate."""
    result: dict[str, list[dict[str, Any]]] = {}
    for side_key, side in (("longs", "long"), ("shorts", "short")):
        rows = candidates.get(side_key) or []
        result[side_key] = [
            row
            for row in rows
            if _daily_trend_allows(side, str(row.get("symbol") or ""), trend_context)
        ]
    for key, value in candidates.items():
        result.setdefault(key, value)
    return result


def passes_prev_day_extreme_gate(
    side: str,
    last: Any,
    ctx: Mapping[str, Any] | None,
) -> bool:
    """PDH/PDL break gate for one auto pick (trader rule 2026-07-31): a long
    must trade ABOVE the previous day's high, a short BELOW the previous
    day's low. Missing price or level -> fails (cannot verify the break).

    The break itself lives in `prev_day_gate` so the Alert Center's Focus
    gate and this one can never drift apart; this wrapper only unpacks the
    daily-context mapping.
    """
    if not isinstance(ctx, Mapping):
        return False
    return prev_day_gate.passes_prev_day_extreme_gate(
        side, last, ctx.get("prev_high"), ctx.get("prev_low")
    )


def candidate_focus_gate_verdict(
    side: str,
    profile: Mapping[str, Any] | None,
    ctx: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """(passes, reason) for one candidate against the combined R2 gate.

    Reads the last COMPLETED bar, not `last`. `last` is the forming bar, and a
    pick admitted on a break that bar then closes back inside is exactly the
    noise the trader asked this gate to remove (plan.md sec 5: completed bars
    only for state transitions).
    """
    if not isinstance(ctx, Mapping):
        return False, "no prior session to measure against"
    price = profile.get("last_complete") if isinstance(profile, Mapping) else None
    vwap = profile.get("completed_session_vwap") if isinstance(profile, Mapping) else None
    return focus_adoption_gate.passes_focus_adoption_gate(
        side, price, ctx.get("prev_high"), ctx.get("prev_low"), vwap
    )


def filter_candidates_by_prev_day_extremes(
    candidates: Mapping[str, list[dict[str, Any]]],
    profiles: Mapping[str, Mapping[str, Any]],
    daily_context: Mapping[str, Any] | None,
    *,
    log: Callable[[str], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Drop auto-populate candidate rows that fail the M5 Focus adoption gate.

    Trader rule 2026-08-14 (packet R2) extends the 2026-07-31 previous-day rule
    with its session-VWAP half: a long must be above yesterday's high AND above
    session VWAP, a short below both. The ADR-breakout family requires the
    extreme break by construction; this makes the whole rule universal across
    the aggressive-extremes and RW/RS families too.

    Like the daily-trend gate it fails OPEN only when no daily store is
    available at all, so missing data never empties the lists wholesale; a
    symbol whose own completed price, prior-day level or session VWAP is
    unknown fails individually, because none of those can be verified.

    The name is kept for its callers; the gate it applies is now the combined
    one in `focus_adoption_gate`.
    """
    if daily_context is None or not daily_context:
        return {key: list(value) for key, value in candidates.items()}
    result: dict[str, list[dict[str, Any]]] = {}
    for side_key, side in (("longs", "long"), ("shorts", "short")):
        rows = candidates.get(side_key) or []
        kept = []
        refused: list[str] = []
        for row in rows:
            sym = str(row.get("symbol") or "").strip().upper()
            profile = profiles.get(sym) if isinstance(profiles, Mapping) else None
            passes, reason = candidate_focus_gate_verdict(
                side, profile, daily_context.get(sym)
            )
            if passes:
                kept.append(row)
            else:
                refused.append(f"{sym} ({reason})")
        result[side_key] = kept
        if refused and log:
            log(
                f"Focus gate refused {len(refused)} {side} candidate(s): "
                f"{', '.join(refused[:8])}{'...' if len(refused) > 8 else ''}"
            )
    for key, value in candidates.items():
        result.setdefault(key, value)
    return result


def filter_symbols_by_daily_trend(
    symbols: Iterable[str],
    side: str,
    trend_context: Mapping[str, Any] | None,
) -> list[str]:
    """Keep only symbols whose daily structure passes the trend gate for ``side``."""
    return [
        str(symbol).strip().upper()
        for symbol in symbols or []
        if _daily_trend_allows(side, symbol, trend_context)
    ]


def auto_populate_caps(env_key) -> tuple[int, int]:
    """(long_cap, short_cap): a flat per-side ceiling, regime-independent.

    Regime-scaled caps (150/50 style) are gone - they pressured the list
    toward a count. The regime still shapes WHAT qualifies (the aggressive
    discovery rules are side-specific); the cap only stops a runaway tape
    from flooding the scan set.
    """
    return (AUTO_POPULATE_MAX_PER_SIDE, AUTO_POPULATE_MAX_PER_SIDE)


def load_daily_context(
    symbols: Iterable[str],
    *,
    daily_bars_dir: Path = MASTER_AVWAP_DAILY_BARS_DIR,
    adr_sessions: int = AUTO_POPULATE_ADR_SESSIONS,
    reference_date=None,
) -> dict[str, dict[str, float]]:
    """{symbol: {prev_high, prev_low, prev_close, adr}} from the durable daily store.

    Only completed sessions strictly before ``reference_date`` (today) count,
    so an intraday call never treats today's partial bar as "yesterday".
    Cached per day - the values are static intraday.
    """
    from human_focus_tracking import _load_durable_daily_frame

    today = reference_date or datetime.now().date()
    with _AUTO_POPULATE_LOCK:
        if _DAILY_CONTEXT_CACHE["date"] == today:
            cached = _DAILY_CONTEXT_CACHE["contexts"]
            missing = [s for s in symbols if str(s or "").strip().upper() not in cached]
            if not missing:
                return dict(cached)

    contexts: dict[str, dict[str, float]] = {}
    for symbol in symbols:
        sym = str(symbol or "").strip().upper()
        if not sym or sym in contexts:
            continue
        try:
            frame = _load_durable_daily_frame(sym, daily_bars_dir)
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            frame.columns = [str(col).strip().lower() for col in frame.columns]
            if "datetime" not in frame.columns:
                continue
            import pandas as pd

            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
            frame = frame.dropna(subset=["datetime"])
            frame = frame[frame["datetime"].dt.date < today]
            if frame.empty or not {"high", "low", "close"} <= set(frame.columns):
                continue
            # Daily-trend MAs are computed from the full completed history (before
            # tailing to the ADR window) so the 200SMA has enough bars.
            closes = frame["close"].astype(float)
            ema15 = (
                float(closes.ewm(span=AUTO_POPULATE_TREND_EMA, adjust=False).mean().iloc[-1])
                if len(closes) >= AUTO_POPULATE_TREND_EMA
                else None
            )
            sma50 = (
                float(closes.tail(AUTO_POPULATE_SHORT_TREND_SMA).mean())
                if len(closes) >= AUTO_POPULATE_SHORT_TREND_SMA
                else None
            )
            sma200 = (
                float(closes.tail(AUTO_POPULATE_LONG_TREND_SMA).mean())
                if len(closes) >= AUTO_POPULATE_LONG_TREND_SMA
                else None
            )
            recent = frame.tail(adr_sessions)
            ranges = (recent["high"] - recent["low"]).astype(float)
            adr = float(ranges.mean())
            last_row = recent.iloc[-1]
            if adr <= 0:
                continue
            contexts[sym] = {
                "prev_high": float(last_row["high"]),
                "prev_low": float(last_row["low"]),
                "prev_close": float(last_row["close"]),
                "adr": adr,
                "ema15": ema15,
                "sma50": sma50,
                "sma200": sma200,
            }
        except Exception:
            continue
    with _AUTO_POPULATE_LOCK:
        _DAILY_CONTEXT_CACHE["date"] = today
        _DAILY_CONTEXT_CACHE["contexts"].update(contexts)
        return dict(_DAILY_CONTEXT_CACHE["contexts"])


def fetch_intraday_profiles(
    symbols: Iterable[str],
    *,
    downloader: Callable[..., Any] | None = None,
    chunk_size: int = AUTOPILOT_OPEN_SCAN_CHUNK_SIZE,
    proximity_pct: float = AUTO_POPULATE_EXTREME_PROXIMITY_PCT,
    now: datetime | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build intraday profiles used by both legacy and aggressive discovery.

    The original aggregate fields intentionally keep their prior semantics.
    ``completed_*`` fields are a separate feature family and exclude the
    forming M5 bar so they are safe for aggressive state transitions.
    """
    downloader = downloader or _default_downloader
    pool = [str(s or "").strip().upper() for s in symbols]
    pool = [s for s in pool if s]
    profiles: dict[str, dict[str, float]] = {}
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(pool), chunk_size):
        chunk = pool[start : start + chunk_size]
        try:
            data = downloader(chunk, period="1d", interval="5m")
        except Exception as exc:
            if log:
                log(f"Intraday profile chunk failed ({chunk[0]}..{chunk[-1]}): {exc}")
            continue
        for symbol in chunk:
            try:
                frame = data[symbol] if len(chunk) > 1 else data
            except Exception:
                frame = None
            rows = _frame_rows(frame)
            if len(rows) < 2:
                continue
            day_high = max(row["high"] for row in rows)
            day_low = min(row["low"] for row in rows)
            at_high = at_low = 0
            for row in rows:
                if day_high > 0 and (day_high - row["high"]) / day_high * 100.0 <= proximity_pct:
                    at_high += 1
                if row["low"] > 0 and (row["low"] - day_low) / row["low"] * 100.0 <= proximity_pct:
                    at_low += 1
            profiles[symbol] = {
                "last": rows[-1]["close"],
                "day_high": day_high,
                "day_low": day_low,
                "time_at_high_frac": at_high / len(rows),
                "time_at_low_frac": at_low / len(rows),
                **_intraday_extreme_metrics(
                    rows,
                    now=now,
                    proximity_pct=proximity_pct,
                ),
            }
    return profiles


def _intraday_extreme_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    bar_minutes: int = 5,
    recent_window_bars: int = AGGRESSIVE_EXTREME_WINDOW_BARS,
    minimum_break_pct: float = AGGRESSIVE_MIN_EXTREME_BREAK_PCT,
    proximity_pct: float = AUTO_POPULATE_EXTREME_PROXIMITY_PCT,
) -> dict[str, Any]:
    """Completed-bar HOD/LOD pressure metrics for one intraday series."""

    def empty_metrics(health: str) -> dict[str, Any]:
        return {
            "feature_version": AGGRESSIVE_EXTREME_FEATURE_VERSION,
            "last_complete": None,
            "completed_session_vwap": None,
            "completed_day_high": None,
            "completed_day_low": None,
            "completed_session_open": None,
            "completed_move_pct": None,
            "completed_time_at_high_frac": 0.0,
            "completed_time_at_low_frac": 0.0,
            "recent_new_highs": 0,
            "recent_new_lows": 0,
            "recent_window_bars": 0,
            "completed_bar_count": 0,
            "as_of": "",
            "data_age_minutes": None,
            "data_health": health,
        }

    moment = normalize_market_local_datetime(now)
    completed: list[tuple[Mapping[str, Any], datetime, datetime]] = []
    for row in rows:
        raw_stamp = row.get("dt")
        if not isinstance(raw_stamp, datetime):
            continue
        local_start = normalize_market_local_datetime(raw_stamp)
        local_end = local_start + timedelta(minutes=max(1, int(bar_minutes)))
        if local_end > moment:
            continue
        explicit_start = raw_stamp if raw_stamp.tzinfo is not None else local_start
        explicit_end = explicit_start + timedelta(minutes=max(1, int(bar_minutes)))
        completed.append((row, local_start, explicit_end))

    if not completed:
        return empty_metrics("missing_completed_bars")

    latest_date = completed[-1][1].date()
    session = [item for item in completed if item[1].date() == latest_date]
    session_rows = [item[0] for item in session]
    try:
        numeric_rows = [
            {
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            for row in session_rows
        ]
        session_open = float(session_rows[0].get("open", session_rows[0]["close"]))
    except (KeyError, TypeError, ValueError):
        return empty_metrics("invalid_completed_bars")
    if not math.isfinite(session_open) or session_open <= 0:
        return empty_metrics("invalid_completed_bars")
    if any(
        not all(math.isfinite(value) for value in row.values()) or row["low"] > row["high"]
        for row in numeric_rows
    ):
        return empty_metrics("invalid_completed_bars")
    day_high = max(row["high"] for row in numeric_rows)
    day_low = min(row["low"] for row in numeric_rows)
    last_complete = numeric_rows[-1]["close"]
    if not day_low <= last_complete <= day_high:
        return empty_metrics("invalid_completed_bars")

    at_high = sum(
        1
        for row in numeric_rows
        if day_high > 0
        and (day_high - row["high"]) / day_high * 100.0 <= proximity_pct
    )
    at_low = sum(
        1
        for row in numeric_rows
        if row["low"] > 0
        and (row["low"] - day_low) / row["low"] * 100.0 <= proximity_pct
    )

    new_high_flags = [False] * len(numeric_rows)
    new_low_flags = [False] * len(numeric_rows)
    if numeric_rows:
        running_high = numeric_rows[0]["high"]
        running_low = numeric_rows[0]["low"]
        break_fraction = max(0.0, float(minimum_break_pct)) / 100.0
        for index, row in enumerate(numeric_rows[1:], start=1):
            high = row["high"]
            low = row["low"]
            if running_high > 0 and high >= running_high * (1.0 + break_fraction):
                new_high_flags[index] = True
            if running_low > 0 and low <= running_low * (1.0 - break_fraction):
                new_low_flags[index] = True
            running_high = max(running_high, high)
            running_low = min(running_low, low)

    window = min(max(1, int(recent_window_bars)), len(numeric_rows))
    last_end_local = session[-1][1] + timedelta(minutes=max(1, int(bar_minutes)))
    data_age_minutes = max(0.0, (moment - last_end_local).total_seconds() / 60.0)
    if len(numeric_rows) < int(recent_window_bars):
        health = "insufficient_completed_bars"
    elif data_age_minutes > AGGRESSIVE_MAX_DATA_AGE_MINUTES:
        health = "stale"
    else:
        health = "ok"
    # Session VWAP over exactly these completed session bars, for the M5 Focus
    # adoption gate (packet R2). `session_vwap_series` is the running-deviation
    # variant every band consumer is calibrated to and it restarts on each date
    # change, so the last value is this session's VWAP at the last completed
    # bar - the same bar `last_complete` came from. Never BounceBot's dynamic
    # or EOD VWAP: those blend prior sessions and answer a different question.
    session_vwap: float | None = None
    try:
        from chart_snapshot import session_vwap_series

        vwap_values = session_vwap_series(session_rows).get("vwap") or []
        if vwap_values:
            candidate_vwap = vwap_values[-1]
            if candidate_vwap is not None and math.isfinite(float(candidate_vwap)):
                session_vwap = float(candidate_vwap)
    except Exception:
        logging.debug("Session VWAP unavailable for this profile.", exc_info=True)

    return {
        "feature_version": AGGRESSIVE_EXTREME_FEATURE_VERSION,
        "last_complete": last_complete,
        "completed_session_vwap": session_vwap,
        "completed_day_high": day_high,
        "completed_day_low": day_low,
        "completed_session_open": session_open,
        "completed_move_pct": (last_complete - session_open) / session_open * 100.0,
        "completed_time_at_high_frac": at_high / len(numeric_rows),
        "completed_time_at_low_frac": at_low / len(numeric_rows),
        "recent_new_highs": sum(new_high_flags[-window:]),
        "recent_new_lows": sum(new_low_flags[-window:]),
        "recent_window_bars": window,
        "completed_bar_count": len(numeric_rows),
        "as_of": session[-1][2].isoformat(timespec="seconds"),
        "data_age_minutes": data_age_minutes,
        "data_health": health,
    }


def build_adr_breakout_candidates(
    profiles: Mapping[str, Mapping[str, float]],
    daily_context: Mapping[str, Mapping[str, float]],
    *,
    min_adr_move: float = AUTO_POPULATE_MIN_ADR_MOVE,
    min_score: float = AUTO_POPULATE_MIN_SCORE,
) -> dict[str, list[dict[str, Any]]]:
    """Rank PDH-break longs / PDL-break shorts by ADR-relative move + HOD/LOD time.

    Only names clearing ``min_score`` qualify - the list is however long the
    tape deserves, not a fill toward the watchlist cap.
    """
    longs: list[dict[str, Any]] = []
    shorts: list[dict[str, Any]] = []
    for symbol, profile in profiles.items():
        sym = str(symbol or "").strip().upper()
        if not sym or sym == "SPY":
            continue
        ctx = daily_context.get(sym)
        if not ctx:
            continue
        try:
            last = float(profile["last"])
            prev_high = float(ctx["prev_high"])
            prev_low = float(ctx["prev_low"])
            prev_close = float(ctx["prev_close"])
            adr = float(ctx["adr"])
        except (KeyError, TypeError, ValueError):
            continue
        if adr <= 0 or last <= 0:
            continue
        adr_move = (last - prev_close) / adr
        at_high = float(profile.get("time_at_high_frac") or 0.0)
        at_low = float(profile.get("time_at_low_frac") or 0.0)
        if last > prev_high and adr_move >= min_adr_move:
            score = adr_move + at_high
            if score < min_score:
                continue
            longs.append(
                {
                    "symbol": sym,
                    "score": score,
                    "adr_move": adr_move,
                    "time_at_extreme": at_high,
                    "reason": f"PDH break, {adr_move:+.1f} ADR, {at_high:.0%} of day at HOD",
                }
            )
        elif last < prev_low and adr_move <= -min_adr_move:
            score = -adr_move + at_low
            if score < min_score:
                continue
            shorts.append(
                {
                    "symbol": sym,
                    "score": score,
                    "adr_move": adr_move,
                    "time_at_extreme": at_low,
                    "reason": f"PDL break, {adr_move:+.1f} ADR, {at_low:.0%} of day at LOD",
                }
            )
    longs.sort(key=lambda row: (-row["score"], row["symbol"]))
    shorts.sort(key=lambda row: (-row["score"], row["symbol"]))
    return {"longs": longs, "shorts": shorts}


def build_aggressive_regime_candidates(
    profiles: Mapping[str, Mapping[str, Any]],
    env_key: str,
    *,
    spy_pullback_active: bool = False,
    minimum_new_extremes: int = AGGRESSIVE_MIN_NEW_EXTREMES,
    recent_window_bars: int = AGGRESSIVE_EXTREME_WINDOW_BARS,
    near_extreme_pct: float = AGGRESSIVE_NEAR_EXTREME_PCT,
) -> dict[str, list[dict[str, Any]]]:
    """Build side-symmetric aggressive candidates from completed M5 bars.

    Bearish: repeated HOD pressure -> long; LOD hold during a legacy SPY
    rebound -> short. Bullish mirrors both rules. Neutral tape deliberately
    produces nothing.
    """
    env = str(env_key or "").strip().lower()
    if not (env.startswith("bearish") or env.startswith("bullish")):
        return {"longs": [], "shorts": []}

    longs: list[dict[str, Any]] = []
    shorts: list[dict[str, Any]] = []
    required_window = max(1, int(recent_window_bars))
    required_extremes = max(1, int(minimum_new_extremes))
    proximity = max(0.0, float(near_extreme_pct))

    def candidate_row(
        symbol: str,
        profile: Mapping[str, Any],
        *,
        rule: str,
        score: float,
        reason: str,
    ) -> dict[str, Any]:
        as_of = str(profile.get("as_of") or "")
        persisted_reason = (
            f"{reason}; {AGGRESSIVE_EXTREME_FEATURE_VERSION}; "
            f"{required_window} completed M5 bars; as of {as_of}"
        )
        return {
            "symbol": symbol,
            "score": score,
            "reason": persisted_reason,
            "source_rule": rule,
            "feature_version": AGGRESSIVE_EXTREME_FEATURE_VERSION,
            "calculation_horizon": f"{required_window}xcompleted_M5",
            "as_of": as_of,
            "data_age_minutes": profile.get("data_age_minutes"),
            "data_health": "ok",
        }

    for symbol, profile in profiles.items():
        sym = str(symbol or "").strip().upper()
        if not sym or sym == "SPY" or str(profile.get("data_health") or "") != "ok":
            continue
        try:
            last = float(profile["last_complete"])
            day_high = float(profile["completed_day_high"])
            day_low = float(profile["completed_day_low"])
            new_highs = int(profile.get("recent_new_highs") or 0)
            new_lows = int(profile.get("recent_new_lows") or 0)
            actual_window = int(profile.get("recent_window_bars") or 0)
            completed_bars = int(profile.get("completed_bar_count") or 0)
            at_high = float(profile.get("completed_time_at_high_frac") or 0.0)
            at_low = float(profile.get("completed_time_at_low_frac") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not all(
                math.isfinite(value)
                for value in (last, day_high, day_low, at_high, at_low)
            )
            or last <= 0
            or day_high <= 0
            or day_low <= 0
            or not day_low <= last <= day_high
            or not 0.0 <= at_high <= 1.0
            or not 0.0 <= at_low <= 1.0
            or actual_window < required_window
            or completed_bars < required_window
            or not profile.get("as_of")
        ):
            continue

        high_distance = max(0.0, (day_high - last) / day_high * 100.0)
        low_distance = max(0.0, (last - day_low) / last * 100.0)
        near_high = high_distance <= proximity
        near_low = low_distance <= proximity

        if env.startswith("bearish"):
            if near_high and new_highs >= required_extremes:
                score = 2.0 + new_highs / required_window + at_high
                longs.append(
                    candidate_row(
                        sym,
                        profile,
                        rule="repeated_hod",
                        score=score,
                        reason=(
                            f"Bearish-day strength: {new_highs} new HODs in {required_window} "
                            f"completed M5 bars, close {high_distance:.2f}% off HOD"
                        ),
                    )
                )
            if spy_pullback_active and near_low:
                score = 2.0 + max(0.0, 1.0 - low_distance / max(proximity, 0.000001)) + at_low
                shorts.append(
                    candidate_row(
                        sym,
                        profile,
                        rule="bearish_pullback_lod",
                        score=score,
                        reason=f"Bearish-day SPY rebound: close {low_distance:.2f}% off LOD",
                    )
                )
        else:
            if near_low and new_lows >= required_extremes:
                score = 2.0 + new_lows / required_window + at_low
                shorts.append(
                    candidate_row(
                        sym,
                        profile,
                        rule="repeated_lod",
                        score=score,
                        reason=(
                            f"Bullish-day weakness: {new_lows} new LODs in {required_window} "
                            f"completed M5 bars, close {low_distance:.2f}% off LOD"
                        ),
                    )
                )
            if spy_pullback_active and near_high:
                score = 2.0 + max(0.0, 1.0 - high_distance / max(proximity, 0.000001)) + at_high
                longs.append(
                    candidate_row(
                        sym,
                        profile,
                        rule="bullish_pullback_hod",
                        score=score,
                        reason=f"Bullish-day SPY pullback: close {high_distance:.2f}% off HOD",
                    )
                )

    longs.sort(key=lambda row: (-row["score"], row["symbol"]))
    shorts.sort(key=lambda row: (-row["score"], row["symbol"]))
    return {"longs": longs, "shorts": shorts}


def _is_directional_env(env_key: Any) -> bool:
    return str(env_key or "").strip().lower().startswith(("bearish", "bullish"))


def resolve_discovery_env(current_env: str, opening_env: str | None) -> str:
    """The env label discovery should hunt with: current while directional,
    else the day's opening read.

    2026-07-17: the tape opened bearish_strong and decayed to bearish_weak
    then neutral by mid-morning - and the decayed label switched aggressive
    discovery off exactly while SNPS/AA/SBLK/PSKY kept bleeding. A day that
    OPENED directional keeps that bias for discovery purposes; a genuine
    reversal still wins because the current read is directional again.
    """
    current = str(current_env or "").strip().lower()
    if _is_directional_env(current):
        return current
    opening = str(opening_env or "").strip().lower()
    if _is_directional_env(opening):
        return opening
    return current


def record_opening_environment(
    env_key: str,
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> str:
    """Persist the day's first directional regime read; returns today's anchor.

    First directional write wins for the day: once the open is classified
    bearish/bullish, later decay to neutral cannot erase it. Returns "" until
    a directional read arrives.
    """
    if path is None:
        from project_paths import AUTO_OPENING_ENV_FILE

        path = AUTO_OPENING_ENV_FILE
    today_iso = (now or datetime.now()).date().isoformat()
    env = str(env_key or "").strip().lower()
    with _AUTO_POPULATE_LOCK:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
        except Exception:
            payload = {}
        if isinstance(payload, dict) and payload.get("date") == today_iso and _is_directional_env(payload.get("env")):
            return str(payload["env"])
        if not _is_directional_env(env):
            return ""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(
                json.dumps(
                    {
                        "date": today_iso,
                        "env": env,
                        "recorded_at": (now or datetime.now()).isoformat(timespec="seconds"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        return env


def load_opening_environment(
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> str:
    """Today's recorded opening env, or "" when none was recorded."""
    if path is None:
        from project_paths import AUTO_OPENING_ENV_FILE

        path = AUTO_OPENING_ENV_FILE
    today_iso = (now or datetime.now()).date().isoformat()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    except Exception:
        return ""
    if isinstance(payload, dict) and payload.get("date") == today_iso and _is_directional_env(payload.get("env")):
        return str(payload["env"])
    return ""


def fetch_session_rvol(
    symbols: Iterable[str],
    *,
    downloader: Callable[..., Any] | None = None,
    chunk_size: int = AUTOPILOT_OPEN_SCAN_CHUNK_SIZE,
    log: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """{symbol: session rvol} on the trader's TC2000 same-time-of-day basis.

    Needs ~15 prior sessions of 5m volume, so it is fetched only for the
    handful of names that already passed the RW/RS price gates - never for
    the whole universe pool.
    """
    from rvol import session_rvol, split_sessions

    downloader = downloader or _default_downloader
    pool: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        symbol = str(symbol or "").strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            pool.append(symbol)
    readings: dict[str, float] = {}
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(pool), chunk_size):
        chunk = pool[start : start + chunk_size]
        try:
            data = downloader(chunk, period="1mo", interval="5m")
        except Exception as exc:
            if log:
                log(f"RVOL chunk failed ({chunk[0]}..{chunk[-1]}): {exc}")
            continue
        for symbol in chunk:
            try:
                frame = data[symbol] if len(chunk) > 1 else data
            except Exception:
                frame = None
            rows = _frame_rows(frame)
            sessions = split_sessions(
                (row["dt"].date(), row.get("volume", 0.0))
                for row in rows
                if isinstance(row.get("dt"), datetime)
            )
            if len(sessions) < 2:
                continue
            value = session_rvol(sessions[-1], sessions[:-1])
            if value is not None:
                readings[symbol] = value
    return readings


def fetch_rvol_baselines(
    symbols: Iterable[str],
    *,
    downloader: Callable[..., Any] | None = None,
    chunk_size: int = AUTOPILOT_OPEN_SCAN_CHUNK_SIZE,
    reference_date=None,
    log: Callable[[str], None] | None = None,
) -> dict[str, list]:
    """{symbol: per-slot baseline volumes} - the static TC2000 denominator.

    Prior sessions only (today's bars are excluded), so one fetch per symbol
    per day is enough; live scanning divides fresh IB volume into it.
    """
    from rvol import slot_baselines, split_sessions

    downloader = downloader or _default_downloader
    pool: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        symbol = str(symbol or "").strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            pool.append(symbol)
    today = reference_date or datetime.now().date()
    baselines: dict[str, list] = {}
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(pool), chunk_size):
        chunk = pool[start : start + chunk_size]
        try:
            data = downloader(chunk, period="1mo", interval="5m")
        except Exception as exc:
            if log:
                log(f"RVOL baseline chunk failed ({chunk[0]}..{chunk[-1]}): {exc}")
            continue
        for symbol in chunk:
            try:
                frame = data[symbol] if len(chunk) > 1 else data
            except Exception:
                frame = None
            rows = _frame_rows(frame)
            sessions = split_sessions(
                (row["dt"].date(), row.get("volume", 0.0))
                for row in rows
                if isinstance(row.get("dt"), datetime) and row["dt"].date() < today
            )
            if not sessions:
                continue
            slots = slot_baselines(sessions)
            if any(value is not None for value in slots):
                baselines[symbol] = slots
    return baselines


def build_relative_weakness_candidates(
    profiles: Mapping[str, Mapping[str, Any]],
    anchor_env: str,
    *,
    rvol_by_symbol: Mapping[str, float] | None = None,
    min_excess_pct: float = RW_MIN_EXCESS_PCT,
    min_session_move_pct: float = RW_MIN_SESSION_MOVE_PCT,
    near_extreme_pct: float = RW_NEAR_EXTREME_PCT,
    min_session_rvol: float = RW_MIN_SESSION_RVOL,
) -> dict[str, list[dict[str, Any]]]:
    """RS/RW discovery keyed to the day's directional anchor (2026-07-17).

    Bearish anchor -> shorts: names moving down harder than SPY on the
    session, still pressing their lows (near completed LOD or printing fresh
    ones), on elevated same-time-of-day volume. A weak name that cannot lift
    while SPY bounces is exactly this signature - the excess-vs-SPY gate
    fires hardest during SPY strength. Bullish anchor mirrors to longs.

    Two-phase by design: ``rvol_by_symbol=None`` returns the price-qualified
    pre-candidates (the list worth spending an RVOL fetch on); passing the
    fetched readings applies the trader's >=1.00 participation gate, and a
    name without a computable reading does not qualify.
    """
    env = str(anchor_env or "").strip().lower()
    if not _is_directional_env(env):
        return {"longs": [], "shorts": []}
    spy = profiles.get("SPY") or {}
    spy_move = spy.get("completed_move_pct")
    if str(spy.get("data_health") or "") != "ok" or spy_move is None:
        return {"longs": [], "shorts": []}
    spy_move = float(spy_move)
    bearish = env.startswith("bearish")

    rows: list[dict[str, Any]] = []
    for symbol, profile in profiles.items():
        sym = str(symbol or "").strip().upper()
        if not sym or sym == "SPY" or str(profile.get("data_health") or "") != "ok":
            continue
        move = profile.get("completed_move_pct")
        if move is None:
            continue
        try:
            move = float(move)
            last = float(profile["last_complete"])
            day_high = float(profile["completed_day_high"])
            day_low = float(profile["completed_day_low"])
            new_highs = int(profile.get("recent_new_highs") or 0)
            new_lows = int(profile.get("recent_new_lows") or 0)
            at_high = float(profile.get("completed_time_at_high_frac") or 0.0)
            at_low = float(profile.get("completed_time_at_low_frac") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not all(math.isfinite(value) for value in (move, last, day_high, day_low))
            or last <= 0
            or day_high <= 0
            or day_low <= 0
            or not day_low <= last <= day_high
            or not profile.get("as_of")
        ):
            continue
        excess = move - spy_move
        if bearish:
            pressing = (
                max(0.0, (last - day_low) / last * 100.0) <= near_extreme_pct or new_lows >= 1
            )
            qualifies = (
                move <= -float(min_session_move_pct)
                and excess <= -float(min_excess_pct)
                and pressing
            )
            extreme_frac = at_low
            rule = "relative_weakness"
            label = "RW"
        else:
            pressing = (
                max(0.0, (day_high - last) / day_high * 100.0) <= near_extreme_pct or new_highs >= 1
            )
            qualifies = (
                move >= float(min_session_move_pct)
                and excess >= float(min_excess_pct)
                and pressing
            )
            extreme_frac = at_high
            rule = "relative_strength"
            label = "RS"
        if not qualifies:
            continue
        rvol_text = ""
        rvol_bonus = 0.0
        if rvol_by_symbol is not None:
            rvol = rvol_by_symbol.get(sym)
            if rvol is None or float(rvol) < float(min_session_rvol):
                continue
            rvol = float(rvol)
            rvol_bonus = min(max(rvol - 1.0, 0.0), 1.5)
            rvol_text = f", rvol {rvol:.2f}"
        score = 2.0 + min(abs(excess), 6.0) / 2.0 + extreme_frac + rvol_bonus
        as_of = str(profile.get("as_of") or "")
        rows.append(
            {
                "symbol": sym,
                "score": score,
                "reason": (
                    f"{label} vs SPY: {excess:+.1f}% excess "
                    f"({move:+.1f}% session vs SPY {spy_move:+.1f}%){rvol_text}; "
                    f"{RELATIVE_WEAKNESS_FEATURE_VERSION}; as of {as_of}"
                ),
                "source_rule": rule,
                "feature_version": RELATIVE_WEAKNESS_FEATURE_VERSION,
                "excess_pct": excess,
                "as_of": as_of,
                "data_age_minutes": profile.get("data_age_minutes"),
                "data_health": "ok",
            }
        )

    rows.sort(key=lambda row: (-row["score"], row["symbol"]))
    if bearish:
        return {"longs": [], "shorts": rows}
    return {"longs": rows, "shorts": []}


def merge_auto_populate_candidates(
    *candidate_sets: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Merge candidate families without duplicate symbols or unstable order."""
    merged: dict[str, list[dict[str, Any]]] = {"longs": [], "shorts": []}
    for side in ("longs", "shorts"):
        by_symbol: dict[str, dict[str, Any]] = {}
        for candidate_set in candidate_sets:
            for raw_row in candidate_set.get(side) or []:
                row = dict(raw_row)
                symbol = str(row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                row["symbol"] = symbol
                prior = by_symbol.get(symbol)
                if prior is None or float(row.get("score") or 0.0) > float(prior.get("score") or 0.0):
                    by_symbol[symbol] = row
        merged[side] = sorted(
            by_symbol.values(),
            key=lambda row: (-float(row.get("score") or 0.0), row["symbol"]),
        )
    return merged


def _load_auto_populate_membership(path: Path, today_iso: str) -> dict[str, Any]:
    import json

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or payload.get("date") != today_iso:
        payload = {"date": today_iso, "long": {}, "short": {}, "cut": {"long": [], "short": []}}
    payload.setdefault("long", {})
    payload.setdefault("short", {})
    payload.setdefault("cut", {"long": [], "short": []})
    return payload


def _save_auto_populate_membership(path: Path, payload: Mapping[str, Any]) -> None:
    import json

    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


#: Requests from BounceBot's triple-VWAP invalidation that a Focus entry be
#: reconciled. BounceBot cuts the watchlist line on a worker thread (and can run
#: headless), but `FocusPickStore` is the GUI's mutable store and plan.md sec 5
#: gives every mutable store exactly one owner. So the bot files a request here
#: and the Alert Center's poll performs the removal - one owner, and the record
#: simply waits if no GUI is running.
FOCUS_DESYNC_REQUEST_FILE = RUNTIME_DATA_DIR / "focus_desync_requests.json"


def record_focus_desync(
    symbol: str,
    side: str,
    *,
    reason: str = "",
    path: Path = FOCUS_DESYNC_REQUEST_FILE,
    now: datetime | None = None,
) -> None:
    """Note that a watchlist line was cut out from under a possible Focus pick.

    Day-scoped, like the blacklist it rides beside: a request that nothing
    drained by the end of the day describes a watchlist state that no longer
    exists.
    """
    sym = str(symbol or "").strip().upper()
    side = "short" if str(side or "").lower().startswith("short") else "long"
    if not sym:
        return
    moment = now or datetime.now()
    today_iso = moment.date().isoformat()
    with _AUTO_POPULATE_LOCK:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if not isinstance(payload, dict) or payload.get("date") != today_iso:
            payload = {"date": today_iso, "requests": []}
        requests = payload.setdefault("requests", [])
        if not any(
            isinstance(row, Mapping) and row.get("symbol") == sym and row.get("side") == side
            for row in requests
        ):
            requests.append(
                {
                    "symbol": sym,
                    "side": side,
                    "reason": str(reason or "triple-VWAP invalidation"),
                    "cut_at": moment.isoformat(timespec="seconds"),
                }
            )
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            pass


def take_focus_desync_requests(
    path: Path = FOCUS_DESYNC_REQUEST_FILE,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Read and CLEAR today's requests. Yesterday's read as empty.

    Draining is destructive on purpose: a request describes one cut, and
    re-applying it after the trader has re-added the name by hand would be the
    automatic removal this whole design exists to prevent.
    """
    today_iso = (now or datetime.now()).date().isoformat()
    with _AUTO_POPULATE_LOCK:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(payload, dict) or payload.get("date") != today_iso:
            return []
        requests = [row for row in (payload.get("requests") or []) if isinstance(row, Mapping)]
        if requests:
            try:
                Path(path).write_text(
                    json.dumps({"date": today_iso, "requests": []}, indent=2), encoding="utf-8"
                )
            except OSError:
                pass
        return [dict(row) for row in requests]


def record_auto_watchlist_cut(
    symbol: str,
    side: str,
    *,
    membership_path: Path = AUTO_POPULATE_MEMBERSHIP_FILE,
    now: datetime | None = None,
) -> None:
    """Day-scoped blacklist: a VWAP-cut name must not be re-added today."""
    sym = str(symbol or "").strip().upper()
    side = "short" if str(side or "").lower().startswith("short") else "long"
    if not sym:
        return
    today_iso = (now or datetime.now()).date().isoformat()
    with _AUTO_POPULATE_LOCK:
        payload = _load_auto_populate_membership(membership_path, today_iso)
        cut = payload["cut"].setdefault(side, [])
        if sym not in cut:
            cut.append(sym)
        # Ownership stays: if the name lingers in the file, the next rotation
        # sweeps it out (the blacklist only prevents re-adding today).
        _save_auto_populate_membership(membership_path, payload)
    _remove_candidate_registry_membership(sym, side, "auto_populate")


def apply_auto_populated_watchlists(
    candidates: Mapping[str, list[dict[str, Any]]],
    env_key: str,
    *,
    longs_path: Path = LONGS_FILE,
    shorts_path: Path = SHORTS_FILE,
    membership_path: Path = AUTO_POPULATE_MEMBERSHIP_FILE,
    now: datetime | None = None,
    preserve_existing_auto: bool = False,
) -> dict[str, Any]:
    """Rotate the auto-owned slice of longs.txt/shorts.txt to the new top-N.

    Trader-added names (anything not in the membership file) are never touched.
    Day-cut names are skipped. A symbol can only hold one side at a time.
    Event-triggered pullback sweeps preserve the existing auto slice so they
    can add promptly without changing the ordinary rotation/removal cadence.
    """
    today_iso = (now or datetime.now()).date().isoformat()
    long_cap, short_cap = auto_populate_caps(env_key)
    with _AUTO_POPULATE_LOCK:
        membership = _load_auto_populate_membership(membership_path, today_iso)
        summary: dict[str, Any] = {
            "env": env_key,
            "caps": (long_cap, short_cap),
            "preserved_existing_auto": bool(preserve_existing_auto),
        }
        taken: set[str] = set()
        for side, cap, path in (("long", long_cap, longs_path), ("short", short_cap, shorts_path)):
            rows = candidates.get(f"{side}s") or []
            cut = {str(s).strip().upper() for s in membership["cut"].get(side, [])}
            existing = [str(s).strip().upper() for s in read_watchlist_symbols(Path(path))]
            owned = {str(s).strip().upper() for s in membership.get(side, {})}
            trader_names = [s for s in existing if s not in owned]
            trader_set = set(trader_names)
            picked: dict[str, str] = (
                {
                    sym: str(membership.get(side, {}).get(sym) or "")
                    for sym in existing
                    if sym in owned and sym not in cut and sym not in taken
                }
                if preserve_existing_auto
                else {}
            )
            for row in rows:
                if len(picked) >= cap:
                    break
                sym = str(row.get("symbol") or "").strip().upper()
                if not sym or sym in cut or sym in taken or sym in trader_set or sym in picked:
                    continue
                picked[sym] = str(row.get("reason") or "")
            taken.update(trader_set)
            taken.update(picked)
            write_watchlist_file(Path(path), [*trader_names, *picked])
            summary[side] = {
                "added": sorted(set(picked) - owned),
                "rotated_out": sorted(owned - set(picked)),
                "kept": len(set(picked) & owned),
                "total_auto": len(picked),
                "trader_names": len(trader_names),
            }
            membership[side] = picked
        _save_auto_populate_membership(membership_path, membership)
        registry_longs = list(membership.get("long", {}))
        registry_shorts = list(membership.get("short", {}))
    _sync_candidate_registry_source(
        "auto_populate",
        registry_longs,
        registry_shorts,
        lease_minutes=90,
    )
    return summary


def read_auto_pilot_mode(state_path: Path = AUTOPILOT_STATE_FILE) -> str:
    """The GUI's Auto mode as OFF/DESK/AWAY/EVENING, read from the machine-local
    Auto Pilot state file. The BounceBot scan loop runs off-GUI-thread (and on
    the mini-PC, off-GUI-process), so the file is the one shared truth. An
    unreadable or absent file reads as OFF - the historical auto-apply path."""
    try:
        payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "OFF"
    if not isinstance(payload, dict) or not payload.get("enabled"):
        return "OFF"
    profile = str(payload.get("profile") or "DESK").strip().upper()
    return profile if profile in ("DESK", "AWAY", "EVENING") else "DESK"


def auto_populate_clock_open(now: datetime | None = None) -> bool:
    """Picks start at 07:00 local - before that the engine stays quiet."""
    return (now or datetime.now()).hour >= AUTO_POPULATE_START_HOUR


def _load_auto_populate_pending(path: Path, today_iso: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or payload.get("date") != today_iso:
        payload = {"date": today_iso}
    for key in ("pending", "decided"):
        bucket = payload.get(key)
        if not isinstance(bucket, dict):
            bucket = {}
        bucket.setdefault("long", {})
        bucket.setdefault("short", {})
        payload[key] = bucket
    return payload


def _save_auto_populate_pending(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


#: How long a stored gate verdict stays usable at adoption time. The staging
#: refresh runs every AUTO_POPULATE_REFRESH_MINUTES, so 1.5x that tolerates one
#: missed refresh and refuses anything older. The Alert Center adopts on the GUI
#: thread and a staged pick is on no watchlist yet, so BounceBot holds no bars
#: for it - the stored verdict is how adoption re-checks without its own fetch.
FOCUS_GATE_VERDICT_MAX_AGE_MINUTES = int(AUTO_POPULATE_REFRESH_MINUTES * 1.5)


def _restamp_or_evict_pending_picks(
    pending: MutableMapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]] | None,
    daily_context: Mapping[str, Any] | None,
    moment: datetime,
    *,
    log: Callable[[str], None] | None = None,
) -> dict[str, list[str]]:
    """Re-run the adoption gate over the queue: evict failures, stamp survivors.

    Trader rule 2026-08-14: "while a pick sits in the pending queue, if it falls
    back through VWAP or the previous-day extreme it is removed from the queue."
    Eviction is silent to the trader by their own wording; the reason goes to the
    log, because a pick that vanishes has to be explainable afterwards.

    An evicted pick may re-propose later the same day if it re-qualifies (trader
    decision 2026-08-15). The queue is meant to say what qualifies NOW, not what
    once did - a name that pulled back at 10:00 and broke out cleanly at 13:00 is
    the setup, not the noise.

    Survivors carry the fresh verdict so adoption can re-check it without its own
    market-data fetch. Fails open on missing evidence: with no profiles or no
    daily store there is nothing to measure against, and an unmeasurable queue is
    not an empty one - the verdict simply ages and adoption refuses on staleness.
    """
    evicted: dict[str, list[str]] = {"long": [], "short": []}
    if not profiles or not daily_context:
        return evicted
    stamp = moment.isoformat(timespec="seconds")
    for side in ("long", "short"):
        side_pending = pending["pending"].get(side) or {}
        for sym in list(side_pending):
            ctx = daily_context.get(sym)
            profile = profiles.get(sym)
            if not isinstance(ctx, Mapping) or not isinstance(profile, Mapping):
                continue  # nothing measured this pass; the stamp ages instead
            entry = side_pending.get(sym)
            if not isinstance(entry, dict):
                continue
            passes, reason = candidate_focus_gate_verdict(side, profile, ctx)
            if passes:
                entry["gate_state"] = focus_adoption_gate.OPEN
                entry["gate_reason"] = reason
                entry["gate_checked_at"] = stamp
                continue
            side_pending.pop(sym, None)
            evicted[side].append(f"{sym} ({reason})")
    if log:
        for side in ("long", "short"):
            names = evicted[side]
            if names:
                log(
                    f"Focus gate evicted {len(names)} staged {side} pick(s): "
                    f"{', '.join(names[:8])}{'...' if len(names) > 8 else ''}"
                )
    return evicted


def pending_pick_gate_ok(
    entry: Mapping[str, Any] | None,
    now: datetime | None = None,
    *,
    max_age_minutes: int = FOCUS_GATE_VERDICT_MAX_AGE_MINUTES,
) -> tuple[bool, str]:
    """(ok, reason) for adopting one staged pick, from its stored verdict.

    Refuses a failing verdict, a missing one, and a stale one. Missing is
    refused for the same reason UNKNOWN fails everywhere else in this gate: a
    pick nothing has measured is not a pick something has approved.
    """
    if not isinstance(entry, Mapping):
        return False, "no staged record"
    state = str(entry.get("gate_state") or "")
    if state != focus_adoption_gate.OPEN:
        return False, str(entry.get("gate_reason") or "never passed the Focus gate")
    raw = str(entry.get("gate_checked_at") or "")
    if not raw:
        return False, "no gate check recorded"
    try:
        checked = datetime.fromisoformat(raw)
    except ValueError:
        return False, "unreadable gate timestamp"
    age = ((now or datetime.now()) - checked).total_seconds() / 60.0
    if age > float(max_age_minutes) or age < 0:
        return False, f"gate check is {age:.0f} min old (limit {max_age_minutes})"
    return True, str(entry.get("gate_reason") or "passed the Focus gate")


def stage_auto_populate_candidates(
    candidates: Mapping[str, list[dict[str, Any]]],
    env_key: str,
    *,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
    daily_context: Mapping[str, Any] | None = None,
    pending_path: Path = AUTO_POPULATE_PENDING_FILE,
    membership_path: Path = AUTO_POPULATE_MEMBERSHIP_FILE,
    longs_path: Path = LONGS_FILE,
    shorts_path: Path = SHORTS_FILE,
    now: datetime | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Queue the picks for adoption instead of writing them to the watchlists.

    Used by every mode the trader can be in - DESK adopts from the queue almost
    immediately, AWAY and EVENING hold it until the flip back (packet R1).

    Nothing touches longs.txt/shorts.txt here. A symbol proposes at most once
    per day while it stays queued or decided, day-cut names stay out, and
    anything already on either watchlist is never proposed.

    Each pass first re-runs the adoption gate over everything already queued
    (packet R2): picks that have fallen back through VWAP or the previous-day
    extreme are evicted, and survivors are re-stamped with a fresh verdict that
    adoption reads instead of fetching its own bars.
    """
    moment = now or datetime.now()
    today_iso = moment.date().isoformat()
    long_cap, short_cap = auto_populate_caps(env_key)
    with _AUTO_POPULATE_LOCK:
        membership = _load_auto_populate_membership(membership_path, today_iso)
        pending = _load_auto_populate_pending(pending_path, today_iso)
        # Before anything is added: re-verify what is already queued, so `queued`
        # below reflects the post-eviction truth and an evicted name is free to
        # re-propose in this same pass if it now qualifies again.
        evicted = _restamp_or_evict_pending_picks(
            pending, profiles, daily_context, moment, log=log
        )
        listed = {
            str(s).strip().upper()
            for path in (longs_path, shorts_path)
            for s in read_watchlist_symbols(Path(path))
        }
        decided = {
            sym
            for side in ("long", "short")
            for sym in pending["decided"].get(side, {})
        }
        queued = {
            sym
            for side in ("long", "short")
            for sym in pending["pending"].get(side, {})
        }
        summary: dict[str, Any] = {
            "env": env_key,
            "mode": "staged",
            "caps": (long_cap, short_cap),
            "evicted": evicted,
        }
        stamp = moment.isoformat(timespec="seconds")
        for side, cap in (("long", long_cap), ("short", short_cap)):
            rows = candidates.get(f"{side}s") or []
            cut = {str(s).strip().upper() for s in membership["cut"].get(side, [])}
            staged: list[str] = []
            side_pending = pending["pending"][side]
            for row in rows:
                if len(side_pending) >= cap:
                    break
                sym = str(row.get("symbol") or "").strip().upper()
                if not sym or sym in cut or sym in listed or sym in decided or sym in queued:
                    continue
                # Everything reaching here already cleared the gate at candidate
                # build in this same pass, so the verdict is stamped now rather
                # than left for the next refresh - otherwise a pick staged at
                # 09:00 could not be adopted until 09:30.
                side_pending[sym] = {
                    "reason": str(row.get("reason") or ""),
                    "score": float(row.get("score") or 0.0),
                    "staged_at": moment.strftime("%H:%M:%S"),
                    "env": env_key,
                    "gate_state": focus_adoption_gate.OPEN,
                    "gate_reason": "passed the Focus gate at candidate build",
                    "gate_checked_at": stamp,
                }
                queued.add(sym)
                staged.append(sym)
            summary[side] = {"staged": staged, "pending": len(side_pending)}
        _save_auto_populate_pending(pending_path, pending)
    return summary


def load_auto_populate_pending_picks(
    pending_path: Path = AUTO_POPULATE_PENDING_FILE,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Today's approval queue (day-scoped; yesterday's leftovers read empty)."""
    today_iso = (now or datetime.now()).date().isoformat()
    with _AUTO_POPULATE_LOCK:
        return _load_auto_populate_pending(pending_path, today_iso)


def resolve_auto_populate_pick(
    symbol: str,
    side: str,
    approved: bool,
    *,
    decision_label: str = "",
    write_watchlist: bool = True,
    pending_path: Path = AUTO_POPULATE_PENDING_FILE,
    membership_path: Path = AUTO_POPULATE_MEMBERSHIP_FILE,
    longs_path: Path = LONGS_FILE,
    shorts_path: Path = SHORTS_FILE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record the trader's verdict on one staged pick.

    Approve appends the symbol to its side's watchlist and files it under
    auto-populate membership (so the ordinary VWAP-cut path still owns it);
    Pass just retires the proposal for the day. Either way the symbol never
    re-proposes today.

    ``write_watchlist=False`` accepts the pick without claiming the watchlist
    entry - used when Focus Picks takes the symbol instead, since two owners
    for one watchlist line means one of them deletes what the other thinks it
    holds. ``decision_label`` names that outcome in the ledger rather than
    filing a machine action under the trader's "approved"/"passed".
    """
    sym = str(symbol or "").strip().upper()
    side = "short" if str(side or "").lower().startswith("short") else "long"
    moment = now or datetime.now()
    today_iso = moment.date().isoformat()
    result: dict[str, Any] = {
        "symbol": sym,
        "side": side,
        "approved": bool(approved),
        "written": False,
        "already_listed": False,
    }
    if not sym:
        return result
    registry_sync: tuple[list[str], list[str]] | None = None
    with _AUTO_POPULATE_LOCK:
        pending = _load_auto_populate_pending(pending_path, today_iso)
        entry = pending["pending"][side].pop(sym, None)
        result["was_pending"] = entry is not None
        pending["decided"][side][sym] = {
            "decision": decision_label or ("approved" if approved else "passed"),
            "at": moment.strftime("%H:%M:%S"),
        }
        _save_auto_populate_pending(pending_path, pending)
        if approved and write_watchlist:
            target = Path(longs_path if side == "long" else shorts_path)
            listed = {
                str(s).strip().upper()
                for path in (longs_path, shorts_path)
                for s in read_watchlist_symbols(Path(path))
            }
            if sym in listed:
                result["already_listed"] = True
            else:
                existing = list(read_watchlist_symbols(target))
                if write_watchlist_file(target, [*existing, sym]):
                    result["written"] = True
                    membership = _load_auto_populate_membership(membership_path, today_iso)
                    reason = str((entry or {}).get("reason") or "trader-approved auto pick")
                    membership[side][sym] = reason
                    _save_auto_populate_membership(membership_path, membership)
                    registry_sync = (
                        list(membership.get("long", {})),
                        list(membership.get("short", {})),
                    )
    if registry_sync is not None:
        _sync_candidate_registry_source(
            "auto_populate",
            registry_sync[0],
            registry_sync[1],
            lease_minutes=90,
        )
    return result


def refresh_auto_populated_watchlists(
    env_key: str,
    *,
    opening_env_key: str | None = None,
    downloader: Callable[..., Any] | None = None,
    spy_pullback_active: bool = False,
    preserve_existing_auto: bool = False,
    stage_only: bool = False,
    now: datetime | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """One full auto-populate pass: universe -> criteria -> regime-capped lists."""
    pool = load_universe_pool()
    if not pool:
        if log:
            log("Auto-populate skipped: universe pool is empty.")
        return None
    moment = now or datetime.now()
    daily_context = load_daily_context(pool, reference_date=moment.date())
    # SPY rides along for the RW/RS excess baseline (builders never emit it).
    profile_pool = pool if "SPY" in pool else ["SPY", *pool]
    profiles = fetch_intraday_profiles(profile_pool, downloader=downloader, now=moment, log=log)
    if not profiles:
        if log:
            log("Auto-populate skipped: no intraday profiles fetched.")
        return None
    discovery_env = resolve_discovery_env(env_key, opening_env_key)
    aggressive = build_aggressive_regime_candidates(
        profiles,
        discovery_env,
        spy_pullback_active=spy_pullback_active,
    )
    rw_pre = build_relative_weakness_candidates(profiles, discovery_env)
    relative = {"longs": [], "shorts": []}
    pre_symbols = [row["symbol"] for side in ("longs", "shorts") for row in rw_pre[side]]
    if pre_symbols:
        rvol_readings = fetch_session_rvol(pre_symbols, downloader=downloader, log=log)
        relative = build_relative_weakness_candidates(
            profiles,
            discovery_env,
            rvol_by_symbol=rvol_readings,
        )
    candidates = merge_auto_populate_candidates(
        build_adr_breakout_candidates(profiles, daily_context),
        aggressive,
        relative,
    )
    # M5 Focus adoption gate (trader directives 2026-07-31 and 2026-08-14):
    # every auto pick must be beyond yesterday's range AND on the right side of
    # session VWAP - longs above the previous day's high and above VWAP, shorts
    # below the previous day's low and below VWAP - regardless of which
    # discovery family produced it. Measured on completed bars.
    candidates = filter_candidates_by_prev_day_extremes(
        candidates, profiles, daily_context, log=log
    )
    # Daily-trend quality gate: no 1-day wonders (longs must hold above the daily
    # 15EMA/200SMA, shorts below the 15EMA/50SMA). Fails open if the daily store
    # is unavailable so the lists never empty on missing data.
    candidates = filter_candidates_by_daily_trend(candidates, daily_context)
    if stage_only:
        # DESK mode: the trader is at the desk - picks queue for chart
        # approval instead of landing on the watchlists unseen.
        summary = stage_auto_populate_candidates(
            candidates,
            env_key,
            profiles=profiles,
            daily_context=daily_context,
            now=moment,
            log=log,
        )
    else:
        summary = apply_auto_populated_watchlists(
            candidates,
            env_key,
            now=moment,
            preserve_existing_auto=preserve_existing_auto,
        )
        summary["mode"] = "applied"
    summary["scanned"] = len(profiles)
    summary["discovery_env"] = discovery_env
    summary["candidates"] = {"longs": len(candidates["longs"]), "shorts": len(candidates["shorts"])}
    summary["aggressive_candidates"] = {
        "longs": len(aggressive["longs"]),
        "shorts": len(aggressive["shorts"]),
        "spy_pullback_active": bool(spy_pullback_active),
        "feature_version": AGGRESSIVE_EXTREME_FEATURE_VERSION,
    }
    summary["relative_candidates"] = {
        "longs": len(relative["longs"]),
        "shorts": len(relative["shorts"]),
        "pre_rvol": len(pre_symbols),
        "feature_version": RELATIVE_WEAKNESS_FEATURE_VERSION,
    }
    return summary


# ---------------------------------------------------------------------------
# Watchlist file IO
# ---------------------------------------------------------------------------
def _is_shared_mutable_path(path: Path) -> bool:
    """Does ``path`` live in the shared folder a second machine could mount?

    Scoped by directory rather than by filename so a test target, a cache copy
    or a machine-local export is never gated, and so a shared file added later
    is covered without anybody remembering to add it to a list.
    """
    try:
        from project_paths import SHARED_HOME_DIR

        shared = Path(SHARED_HOME_DIR).resolve()
    except Exception:
        return False
    try:
        candidate = Path(path).resolve()
    except OSError:
        return False
    return candidate == shared or shared in candidate.parents


def shared_write_refusal(path: Path) -> str:
    """Why this machine must not write ``path``, or ``""`` when it may.

    The publication gate protected ``autopilot_today.txt`` and nothing else,
    while ``longs.txt`` / ``shorts.txt`` / ``autolongs.txt`` / ``autoshorts.txt``
    sit in the *same* shared folder and were rewritten by whichever machine
    happened to run a scheduled build. That put the plan.md sec 5 invariant
    "user-entered watchlist names are never auto-removed" at cross-machine risk:
    the merge basis is the local machine's own record of what it wrote, so a
    secondary cannot know which names the desktop's trader added by hand.

    Layer 1 is the authority here too - the same configured designated writer,
    read from machine-local settings.
    """
    if not _is_shared_mutable_path(path):
        return ""
    try:
        from writer_role import resolve_writer_role

        role = resolve_writer_role()
    except Exception as exc:  # fail closed: an unreadable role is not a licence
        return (
            "the designated-writer configuration could not be read, so this machine "
            f"refuses to rewrite shared file {Path(path).name}: {exc!r}"
        )
    if role.may_publish:
        return ""
    return (
        f"{Path(path).name} is shared state and this machine may not write it: "
        f"{role.reason}"
    )


def write_watchlist_file(path: Path, symbols: Iterable[str]) -> bool:
    """Atomically replace a watchlist file. ``False`` when the role refused.

    Returns ``True`` when the file was written. A read-only secondary gets
    ``False`` and an error in the log; the previous contents - including any
    names the designated writer's trader typed - are left exactly as they were.
    """
    refusal = shared_write_refusal(Path(path))
    if refusal:
        logging.error("Shared watchlist write refused: %s", refusal)
        return False
    cleaned = []
    seen = set()
    for symbol in symbols:
        symbol = str(symbol or "").strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            cleaned.append(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(cleaned) + ("\n" if cleaned else "")
    fd, staged_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged_name, path)
    finally:
        if os.path.exists(staged_name):
            try:
                os.remove(staged_name)
            except OSError:
                pass
    return True


def append_watchlist_symbols(path: Path, symbols: Iterable[str]) -> list[str]:
    """Append new symbols to a watchlist file; returns what was added.

    A machine that may not write this file adds nothing and says so, rather than
    reporting names it did not manage to persist.
    """
    refusal = shared_write_refusal(Path(path))
    if refusal:
        logging.error("Shared watchlist append refused: %s", refusal)
        return []
    existing = list(read_watchlist_symbols(Path(path))) if Path(path).exists() else []
    existing_set = {str(item).strip().upper() for item in existing}
    added = []
    for symbol in symbols:
        symbol = str(symbol or "").strip().upper()
        if symbol and symbol not in existing_set:
            existing.append(symbol)
            existing_set.add(symbol)
            added.append(symbol)
    if added and not write_watchlist_file(Path(path), existing):
        return []
    return added


def write_bouncebot_watchlists(longs: Iterable[str], shorts: Iterable[str]) -> bool:
    """Replace the shared BounceBot watchlists; ``False`` when the role refused."""
    wrote_longs = write_watchlist_file(Path(LONGS_FILE), longs)
    wrote_shorts = write_watchlist_file(Path(SHORTS_FILE), shorts)
    return bool(wrote_longs and wrote_shorts)


def write_auto_watchlists(longs: Iterable[str], shorts: Iterable[str]) -> bool:
    """The bot's own morning picks - written every day in both modes so the
    picks accumulate a clean, separately-attributable outcome history.

    Gated on the configured designated writer, because these files live in the
    same shared folder as the report. ``False`` means nothing shared was touched;
    the registry mirror is skipped too, so machine-local provenance never claims
    a rotation that did not happen.
    """
    longs = [str(s).strip().upper() for s in longs if str(s).strip()]
    shorts = [str(s).strip().upper() for s in shorts if str(s).strip()]
    wrote_longs = write_watchlist_file(Path(AUTO_LONGS_FILE), longs)
    wrote_shorts = write_watchlist_file(Path(AUTO_SHORTS_FILE), shorts)
    if not (wrote_longs and wrote_shorts):
        return False
    _mirror_auto_picks_into_registry(longs, shorts)
    return True


def candidate_registry_path() -> Path:
    from project_paths import CACHE_DIR

    return Path(CACHE_DIR).parent / "candidate_registry.json"


def _mutate_candidate_registry(
    operation: Callable[[Any], None],
    *,
    description: str,
) -> None:
    """Best-effort shadow mutation with one stale-writer merge retry.

    Text watchlists remain authoritative during the migration, so registry
    failures are visible but never allowed to break the established writer.
    """

    try:
        from candidate_registry import CandidateRegistry, StaleWriterError

        path = candidate_registry_path()
        with _CANDIDATE_REGISTRY_LOCK:
            for attempt in range(2):
                registry = CandidateRegistry.load(path)
                operation(registry)
                try:
                    registry.save(path)
                    return
                except StaleWriterError:
                    if attempt == 0:
                        continue
                    raise
    except Exception:
        logging.exception("Candidate-registry %s failed (text watchlists unaffected).", description)


def _sync_candidate_registry_source(
    source: str,
    longs: Iterable[str],
    shorts: Iterable[str],
    *,
    lease_minutes: int | None,
) -> None:
    long_symbols = [str(symbol).strip().upper() for symbol in longs if str(symbol).strip()]
    short_symbols = [str(symbol).strip().upper() for symbol in shorts if str(symbol).strip()]

    def sync(registry) -> None:
        registry.sync_source(
            source,
            {"LONG": long_symbols, "SHORT": short_symbols},
            lease_minutes=lease_minutes,
        )

    _mutate_candidate_registry(sync, description=f"{source} sync")


def _remove_candidate_registry_membership(symbol: str, side: str, source: str) -> None:
    side_name = "SHORT" if str(side).lower().startswith("short") else "LONG"

    def remove(registry) -> None:
        registry.remove_source(symbol, side_name, source)

    _mutate_candidate_registry(remove, description=f"{source} removal")


def add_candidate_registry_memberships(
    source: str,
    side: str,
    symbols: Iterable[str],
    *,
    lease_minutes: int | None,
) -> None:
    """Dual-write additive candidates without replacing earlier live leases."""

    side_name = "SHORT" if str(side).lower().startswith("short") else "LONG"
    cleaned = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]

    def add_all(registry) -> None:
        for symbol in cleaned:
            registry.add(
                symbol,
                side_name,
                source,
                lease_minutes=lease_minutes,
            )

    _mutate_candidate_registry(add_all, description=f"{source} additions")


def _mirror_auto_picks_into_registry(longs: list[str], shorts: list[str]) -> None:
    """Shadow adoption (plan.md Packet D step 2): the registry records the
    same picks with provenance/leases while the text files stay the
    authoritative export. Never allowed to break the write path."""
    _sync_candidate_registry_source(
        "open_scan",
        longs,
        shorts,
        lease_minutes=24 * 60,
    )


# ---------------------------------------------------------------------------
# Away report (shared home-folder digest)
# ---------------------------------------------------------------------------
def format_industry_snapshot_line(
    board_state: Mapping[str, Any] | None,
    intraday_state: Mapping[str, Any] | None,
) -> str:
    """One truthful identity shared by Auto Pilot and its published report."""
    board = board_state if isinstance(board_state, Mapping) else {}
    intraday = intraday_state if isinstance(intraday_state, Mapping) else {}
    board_id = str(board.get("snapshot_id") or "")
    board_status = str(board.get("status") or "missing").upper()
    if board_id:
        parts = [f"Industry Board: {board_status} snapshot {board_id}"]
    else:
        parts = ["Industry Board: unavailable"]

    intraday_id = str(intraday.get("snapshot_id") or "")
    if intraday_id:
        source_id = str(intraday.get("source_board_snapshot_id") or "")
        qualified = int(intraday.get("qualified_industry_count") or 0)
        total = int(intraday.get("industry_count") or 0)
        parts.append(
            f"M5 advisory {intraday_id} ({qualified}/{total} qualified; source board "
            f"{source_id or 'unknown'})"
        )
        if board_id and source_id and source_id != board_id:
            parts.append("SOURCE MISMATCH - refresh RS/RW")
    else:
        parts.append("M5 advisory not built yet")
    return " | ".join(parts)


def build_away_operations_lines(audit: Mapping[str, Any] | None) -> dict[str, str]:
    """Condense the local operations audit into phone-sized truthful lines."""
    payload = audit if isinstance(audit, Mapping) else {}
    status = str(payload.get("status") or "unknown").upper()
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    operations_line = (
        f"Health: {status} "
        f"({int(summary.get('healthy', 0) or 0)} healthy, "
        f"{int(summary.get('degraded', 0) or 0)} degraded, "
        f"{int(summary.get('unhealthy', 0) or 0)} unhealthy)"
    )

    manifest = payload.get("latest_manifest") if isinstance(payload.get("latest_manifest"), Mapping) else {}
    if manifest:
        trigger = str(manifest.get("trigger") or manifest.get("job_type") or "scan")
        manifest_status = str(manifest.get("status") or "unknown")
        minutes = float(manifest.get("total_seconds") or 0.0) / 60.0
        last_scan_line = f"Last scan: {trigger} | {manifest_status} | {minutes:.1f}m"
    else:
        last_scan_line = "Last scan: UNKNOWN - no manifest"

    counters = manifest.get("counters") if isinstance(manifest.get("counters"), Mapping) else {}
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), Mapping) else {}
    if counters.get("update_setup_tracker") is True:
        if counters.get("setup_tracker_updated") is True:
            tracker_line = "Tracker: updated and verified by the latest requested scan"
        else:
            reason = str(outputs.get("setup_tracker_skip_reason") or "reason not recorded")
            tracker_line = f"Tracker: WRITE SKIPPED - {reason}"
    else:
        tracker_line = "Tracker: latest scan was not a scheduled tracker-write slot"
    return {
        "operations_line": operations_line,
        "last_scan_line": last_scan_line,
        "tracker_line": tracker_line,
    }


# Near-favorite demotion (2026-07-17 week review): favorite_setup scenario
# closes ran +1.01R avg (n=1,940) while near_favorite_zone ran -0.18R at 5x
# the volume (n=9,164). Favorites lead the page; the near bucket is capped so
# the measured-losing bucket cannot crowd the report.
AWAY_REPORT_MAX_NEAR_ROWS = 3


# The swing PUSH starts later than the report it rides on. The digest keeps
# publishing hourly from AUTOPILOT_AWAY_REPORT_START_HOUR (07:00); the phone
# just stays quiet until the setups behind it are worth reading. Trader call
# 2026-08-10: Master AVWAP setups are not meaningfully formed in the first
# couple of hours of the session, so a 07:00/08:00 push is noise, and noise on
# this channel costs the credibility that the urgent alerts depend on.
AUTOPILOT_SWING_PUSH_START_HOUR = 9


def swing_push_due(
    now: datetime, *, start_hour: int = AUTOPILOT_SWING_PUSH_START_HOUR
) -> bool:
    """Whether the swing picks may be phoned at ``now`` (desk-local clock).

    Suppresses the push only. The 07:00 and 08:00 publishes still write the
    digest, so nothing stops the trader from opening it early - the picks are
    simply not pushed at them.
    """
    return now.hour >= max(0, int(start_hour))


# The phone push carries the FULL favorite / high-conviction roster under the
# ranked picks (trader ask 2026-08-11). The ranked five answer "what do I
# trade now"; the roster answers "which names are live at all", which is the
# question that otherwise costs opening the digest. Near-favorite stays out on
# the same evidence that keeps it capped in the report: it measured -0.18R
# against favorites' +1.01R, and a push has less room than the report, not more.
PUSH_ROSTER_GROUPS = (("high_conviction", "HC"), ("favorite", "FAV"))

# ntfy's default per-message ceiling is 4 KB. Stay under it with room to spare,
# and when a roster genuinely does not fit, SAY a line was dropped - a silently
# shortened list reads as a complete one, which is the failure this channel
# cannot have (plan.md sec 5: missing data is uncertainty, never confirmation).
PUSH_MESSAGE_MAX_CHARS = 3500


def normalize_bucket_key(bucket: Any) -> str:
    """Collapse a raw bucket key or its display label to one comparable token.

    The same bucket reaches this module as ``high_conviction`` from the scan
    CSV and as ``High Conviction`` from ``SetupRow.bucket_label``; both must
    land on one key or the roster silently splits in two.
    """
    text = str(bucket or "").strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    if not text:
        return ""
    if "near" in text:
        return "near"
    if text.startswith("high conviction"):
        return "high_conviction"
    if text.startswith("favorite") or text.startswith("favourite"):
        return "favorite"
    return text.replace(" ", "_")


def build_bucket_roster(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Every favorite / high-conviction name in the current feed, side-split.

    Order is the feed's own (already ranked), duplicates are dropped, and a
    symbol that appears both long and short keeps both entries - that
    disagreement is information, not noise.
    """
    roster: dict[str, dict[str, list[str]]] = {
        key: {"LONG": [], "SHORT": []} for key, _short in PUSH_ROSTER_GROUPS
    }
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        group = roster.get(normalize_bucket_key(row.get("bucket")))
        if group is None:
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        if not symbol or side not in group:
            continue
        if symbol not in group[side]:
            group[side].append(symbol)
    return roster


def format_roster_lines(roster: Mapping[str, Any] | None) -> list[str]:
    """Compact push lines for the roster, one per non-empty bucket/side."""
    lines: list[str] = []
    for key, short in PUSH_ROSTER_GROUPS:
        sides = roster.get(key) if isinstance(roster, Mapping) else None
        if not isinstance(sides, Mapping):
            continue
        for side, side_short in (("LONG", "L"), ("SHORT", "S")):
            symbols = [
                str(symbol).strip().upper()
                for symbol in (sides.get(side) or [])
                if str(symbol).strip()
            ]
            if symbols:
                lines.append(f"{short} {side_short} ({len(symbols)}): " + ",".join(symbols))
    return lines


def fit_push_message(lines: Iterable[str], *, limit: int = PUSH_MESSAGE_MAX_CHARS) -> str:
    """Join push lines, dropping trailing ones only when they cannot fit.

    A trimmed message always ends by saying how many lines it dropped, so a
    short list on the phone is never mistaken for a short list in the data.
    """
    lines = [str(line) for line in lines]
    message = "\n".join(lines)
    if len(message) <= limit:
        return message
    kept: list[str] = []
    for index, line in enumerate(lines):
        # Reserve room for the widest marker this line could still need.
        marker = f"! {len(lines) - index} more line(s) did not fit - see the digest"
        if len("\n".join([*kept, line, marker])) > limit:
            break
        kept.append(line)
    dropped = len(lines) - len(kept)
    if not dropped:
        return "\n".join(kept)
    return "\n".join([*kept, f"! {dropped} more line(s) did not fit - see the digest"])


def build_swing_push(payload: Mapping[str, Any], *, limit: int = 5) -> tuple[str, str] | None:
    """The best Master AVWAP swing trades, compact enough for a phone push.

    Returns ``(title, message)``, or ``None`` when there is nothing worth
    sending. Silence is deliberate: an hourly "no setups" push seven times a
    session teaches the trader to swipe this channel away, and the digest in
    ``autopilot_today.txt`` already says so for anyone who looks.

    Deliberately NOT the report's own swing block. A notification shows a few
    lines before the OS truncates it, so this carries symbol, side, expected R
    and the key level, and drops the bucket/family detail the report has room
    for. "Near" rows are dropped outright - the report already caps them
    because that bucket measured -0.18R against favorites' +1.01R, and a push
    has less room than the report, not more.

    Staleness is never hidden: picks from anything but the current session say
    so, because a stale pick presented as current is the one failure this must
    not have (plan.md sec 5 - missing data is uncertainty, never confirmation).
    """
    picks = payload.get("swing_picks")
    if not isinstance(picks, (list, tuple)):
        picks = ()
    lines: list[str] = []
    symbols: list[str] = []
    for pick in picks:
        if not isinstance(pick, Mapping):
            continue
        symbol = str(pick.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if "near" in str(pick.get("bucket") or "").lower():
            continue
        side = str(pick.get("side") or "").strip().upper() or "?"
        expected_r = pick.get("expected_r")
        try:
            r_text = f" {float(expected_r):.1f}R" if expected_r is not None else ""
        except (TypeError, ValueError):
            r_text = ""
        level = str(pick.get("key_level") or "").strip()
        symbols.append(symbol)
        lines.append(
            f"{len(symbols)}. {symbol} {side}{r_text}" + (f" @ {level}" if level else "")
        )
        if len(symbols) >= max(1, int(limit)):
            break
    roster_lines = format_roster_lines(payload.get("bucket_roster"))
    if not lines and not roster_lines:
        return None
    if payload.get("swing_data_current") is not True:
        lines.append("! not from the current session - check the digest")
    if symbols:
        lines.append("TV: " + ",".join(symbols))
    lines.extend(roster_lines)
    title = (
        f"Best swings ({len(symbols)})"
        if symbols
        else "Favorites / high conviction"
    )
    return title, fit_push_message(lines)


# D1 level events on the phone (trader ask 2026-08-11): a second hourly push
# naming every stock that fired a D1 level or event alert since the previous
# one. NEW-since-last-push on purpose - a cumulative list re-sent each hour
# trains the trader to swipe the channel away, and the channel's whole value is
# that it is worth reading.
D1_EVENT_PUSH_MAX_SYMBOLS = 25


def build_d1_events_push(
    events: Iterable[Mapping[str, Any]], *, limit: int = D1_EVENT_PUSH_MAX_SYMBOLS
) -> tuple[str, str] | None:
    """One line per symbol that fired D1 level/event alerts, newest label last.

    Returns ``None`` when nothing fired: silence is the correct hourly output
    for an hour with no D1 events, exactly as it is for the swing push.
    """
    ordered: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        symbol = str(event.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        label = str(event.get("label") or "").strip()
        entry = ordered.setdefault(symbol, {"labels": [], "time_text": ""})
        if label and label not in entry["labels"]:
            entry["labels"].append(label)
        time_text = str(event.get("time_text") or "").strip()
        if time_text:
            entry["time_text"] = time_text
    if not ordered:
        return None
    total = len(ordered)
    lines: list[str] = []
    for index, (symbol, entry) in enumerate(list(ordered.items())[: max(1, int(limit))], start=1):
        labels = ", ".join(entry["labels"])
        stamp = f" ({entry['time_text'][:5]})" if entry["time_text"] else ""
        lines.append(f"{index}. {symbol}" + (f" {labels}" if labels else "") + stamp)
    shown = min(total, max(1, int(limit)))
    if total > shown:
        lines.append(f"! {total - shown} more symbol(s) not shown - see the desk")
    lines.append("TV: " + ",".join(list(ordered)[:shown]))
    return f"D1 events ({total})", fit_push_message(lines)


def render_away_report(payload: Mapping[str, Any]) -> str:
    """Phone-first digest: ONE central shared file, best swing trades on top.

    The ranked swing list is the deliverable; day-trade watchlists and
    alerts follow, and all operational/diagnostic truth is condensed into
    a single OPERATIONS section at the tail so the phone view stays mostly
    trades. The freshness warning stays in the header - a stale report must
    say so before it says anything else.
    """

    def _lines(items: Iterable[str]) -> str:
        items = [str(item) for item in items if str(item).strip()]
        return "\n".join(items) if items else "(none)"

    def _tickers(items: Iterable[str]) -> str:
        items = [str(item).strip().upper() for item in items if str(item).strip()]
        return ", ".join(items) if items else "(none)"

    def _swing_bucket_priority(item: Mapping[str, Any]) -> int:
        bucket = str(item.get("bucket") or "").strip().lower()
        bucket = " ".join(bucket.replace("_", " ").replace("-", " ").split())
        if bucket == "high conviction":
            return 0
        if bucket in {"favorite", "favorite setup"}:
            return 1
        return 2

    indexed_picks = [
        (index, pick)
        for index, pick in enumerate(payload.get("swing_picks", []) or [])
        if isinstance(pick, Mapping)
    ]
    indexed_picks.sort(key=lambda item: (_swing_bucket_priority(item[1]), item[0]))

    picks_lines = []
    picks_symbols: list[str] = []
    near_rows_shown = 0
    near_rows_suppressed = 0
    for _index, pick in indexed_picks:
        symbol = str(pick.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        side = str(pick.get("side") or "").strip().upper() or "?"
        bucket = str(pick.get("bucket") or "").strip()
        is_near = "near" in bucket.lower()
        if is_near:
            if near_rows_shown >= AWAY_REPORT_MAX_NEAR_ROWS:
                near_rows_suppressed += 1
                continue
            near_rows_shown += 1
        expected_r = pick.get("expected_r")
        expected_text = f" | {float(expected_r):.2f}R" if expected_r is not None else ""
        bucket_text = f" | {bucket}" if bucket else ""
        family = str(pick.get("family") or "").strip().replace("_", " ")
        family_text = f" | {family}" if family else ""
        key_level = str(pick.get("key_level") or "").strip()
        level_text = f" @ {key_level}" if key_level else ""
        picks_symbols.append(symbol)
        picks_lines.append(
            f"{len(picks_symbols)}. {symbol} ({side}){bucket_text}{expected_text}{family_text}{level_text}"
        )
    if near_rows_suppressed:
        picks_lines.append(
            f"(+{near_rows_suppressed} more near-favorite rows hidden - bucket measured "
            "-0.18R avg vs favorites +1.01R, week of 2026-07-13)"
        )
    if picks_symbols:
        picks_lines.append("TV paste: " + ",".join(picks_symbols))

    swing_data_line = str(payload.get("swing_data_line") or "")
    if picks_lines:
        swing_lines = picks_lines
    elif payload.get("swing_data_current") is False or "awaiting" in swing_data_line.lower():
        swing_lines = ["Awaiting today's first completed swing scan."]
    else:
        swing_lines = ["No qualified current-session swing opportunity."]
    if swing_data_line:
        swing_lines = [*swing_lines, swing_data_line]

    def _tv_line(items: Iterable[str]) -> str:
        items = [str(item).strip().upper() for item in items if str(item).strip()]
        return ",".join(items) if items else "(none)"

    mode_text = str(payload.get("auto_mode") or ("ON" if payload.get("enabled") else "OFF"))
    if mode_text in ("DESK", "AWAY", "EVENING"):
        mode_text = f"AUTO - {mode_text}"
    header_bits = [
        f"Mode: {mode_text} | IB: {payload.get('ib_status', 'unknown')} | Regime: {payload.get('regime', 'unknown')}",
    ]
    if mode_text in ("AUTO - AWAY", "AUTO - EVENING"):
        header_bits.append(
            "Report schedule: hourly from 07:00 local through market close; scan completions may add updates."
        )
    if mode_text == "AUTO - EVENING":
        header_bits.append(
            "Evening mode: picks stage for chart approval only (no self-applied trades); "
            "morning briefing finalizes after the 07:30 strength check."
        )

    # Everything operational lands in one tail section so the top of the
    # phone view is trades, not diagnostics. Order preserves the old
    # header's reading order; absent lines simply drop out.
    operations_lines = [
        str(payload[key])
        for key in (
            "universe_line",
            "scorecard_line",
            "runtime_line",
            "operations_line",
            "last_scan_line",
            "industry_line",
            "tracker_line",
        )
        if payload.get(key)
    ]

    briefing_sections: list[str] = []
    briefing_lines = [str(line) for line in (payload.get("evening_briefing_lines") or []) if str(line).strip()]
    if briefing_lines:
        briefing_sections = ["== MORNING BRIEFING (EVENING MODE) ==", _lines(briefing_lines), ""]

    sections = [
        "TRADINGBOT AUTO PILOT - TODAY",
        f"Updated: {payload.get('generated_at', '')}",
        f"Freshness: next planned update {payload.get('next_slot') or '(none left today)'} - "
        "if Updated is hours old, automation is NOT running; do not trade this as current.",
        *header_bits,
        "",
        *briefing_sections,
        "== BEST SWING TRADES ==",
        _lines(swing_lines),
        "",
        "== DAY TRADE LONGS (longs.txt) ==",
        _tickers(payload.get("longs", [])),
        f"TV paste: {_tv_line(payload.get('longs', []))}",
        "",
        "== DAY TRADE SHORTS (shorts.txt) ==",
        _tickers(payload.get("shorts", [])),
        f"TV paste: {_tv_line(payload.get('shorts', []))}",
        "",
        "== BOT PICKS - LONGS (autolongs.txt) ==",
        _tickers(payload.get("auto_longs", [])),
        f"TV paste: {_tv_line(payload.get('auto_longs', []))}",
        "",
        "== BOT PICKS - SHORTS (autoshorts.txt) ==",
        _tickers(payload.get("auto_shorts", [])),
        f"TV paste: {_tv_line(payload.get('auto_shorts', []))}",
        "",
        "== TODAY'S ALERTS (latest first) ==",
        _lines(payload.get("alerts", [])),
        "",
        "== SCHEDULE ==",
        f"Swing slots done: {_tickers(payload.get('slots_done', []))}",
        f"Next swing slot: {payload.get('next_slot') or '(none left today)'}",
        "",
        "== OPERATIONS ==",
        _lines(operations_lines),
        "",
        "== ACTIVITY LOG (latest first) ==",
        _lines(payload.get("log_lines", [])),
        "",
    ]
    return "\n".join(sections)


def _writer_health_snapshot(
    *,
    role=None,
    lease: Mapping[str, Any] | None = None,
    lock_info: Mapping[str, Any] | None = None,
    override=None,
    lease_path: Path | None = None,
    failure: tuple[str, str] | None = None,
    publication: Mapping[str, Any] | None = None,
    clock_offset_seconds: float | None = None,
    pair_disagreement: str = "",
) -> None:
    """Write the Layer 5 health artifact for this publish attempt.

    Called on every path - success, configuration refusal, ownership refusal -
    because a machine that stopped publishing must be visible as such. Telemetry
    failures are swallowed here on purpose: health reporting must never be the
    thing that fails (or half-completes) a publication.
    """
    try:
        import writer_lease as wl
        from writer_health import write_writer_health

        moment = datetime.now().isoformat(timespec="seconds")
        state: dict[str, Any] = {
            "machine": wl.machine_name(),
            "pid": os.getpid(),
            "instance_id": wl.process_instance_id(),
            "holder_identity": wl.machine_holder_id(),
            "lease_path": str(lease_path or ""),
        }
        if role is not None:
            state.update(role.as_dict())
        if lock_info:
            state["local_lock"] = dict(lock_info)
        if override is not None:
            state["emergency_override"] = override.as_dict()
        if lease:
            state.update(
                {
                    "lease_holder": str(lease.get("holder") or ""),
                    "lease_instance_id": str(lease.get("instance_id") or ""),
                    "lease_acquired_at": str(lease.get("acquired_at") or ""),
                    "lease_expires_at": str(lease.get("expires_at") or ""),
                    # Only a real renewal fills this in. It used to be an alias
                    # for the acquisition time, so a Health panel showed a
                    # renewal timestamp for a lease that had never been renewed.
                    "last_renewal_at": str(lease.get("renewed_at") or ""),
                    "fencing_generation": lease.get("generation"),
                }
            )
        if clock_offset_seconds is not None:
            state["observed_clock_offset_seconds"] = clock_offset_seconds
        if pair_disagreement:
            state["previous_pair_disagreement"] = str(pair_disagreement)
        if failure is not None:
            kind, message = failure
            state["last_failure"] = {"at": moment, "kind": kind, "message": str(message)}
            state["last_blocked_at"] = moment
            state["status"] = f"blocked: {kind}"
            state["healthy"] = False
        if publication is not None:
            state["last_verified_publication"] = dict(publication)
            state["status"] = "published"
            state["healthy"] = True
        write_writer_health(state)
    except Exception:
        logging.debug("Writer health telemetry write failed.", exc_info=True)


def publish_away_report(
    payload: Mapping[str, Any],
    path: Path | None = None,
    *,
    archive: bool = True,
    archive_keep: int = 30,
) -> dict[str, Any]:
    """Verified atomic publish (plan.md 23.8) behind the full publication gate.

    The gate order is fixed and each step is a precondition for the next:

    1. **configured writer role** - an unconfigured machine, or one configured
       as a read-only secondary, touches nothing at all: not the report, not the
       metadata, not the lease. There is no "first machine wins" fallback.
    2. **machine-local cross-process lock** - real exclusion between two
       processes on this PC, held around the lease acquisition *and* the whole
       publication transaction.
    3. **hardened writer lease / fencing validation** - cross-machine writer
       protection, failing closed on every state it cannot verify.
    4. render / stage locally.
    5. **re-verify ownership and fencing generation immediately before
       replacement** - a writer that was fenced off while rendering aborts. The
       check is made before the report replace *and* again before the metadata
       replace, because both write shared output.
    6. replace report and metadata (a failure of either - including a
       ``BaseException`` such as a hard interrupt between the two - rolls both
       back, so the pair on disk never describes two different publications).
    7. verified readback.

    A failure never clears the previous valid report, and the caller gets an
    honest result instead of a path that may not have been written.

    RESIDUAL WINDOW, STATED PLAINLY
    -------------------------------
    Re-verifying immediately before ``os.replace`` narrows the ownership window;
    it does not close it. A writer fenced off in the sub-millisecond gap between
    the check and the replace still completes that replace and reports success.
    Closing it would need an atomic compare-and-swap across machines, which a
    plain shared file does not provide. This is cross-machine
    writer protection - it makes ambiguous states fail closed and makes the
    common races visible; it is not proof that clobbering cannot happen.

    A report/metadata pair left disagreeing by a kill in an earlier cycle is
    detected here and reported in ``previous_pair_disagreement`` and in Health
    telemetry, rather than being silently overwritten.
    """
    import writer_lease as wl
    from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path
    from writer_role import resolve_emergency_override, resolve_writer_role

    target = Path(path) if path is not None else Path(AUTOPILOT_REPORT_FILE)
    metadata_path = target.with_suffix(target.suffix + ".meta.json")
    lease_path = target.with_suffix(target.suffix + ".lease")
    result: dict[str, Any] = {
        "ok": False,
        "verified": False,
        "path": target,
        "error": "",
        "holder": "",
        "lease_expires_at": "",
        "sha256": "",
        "restored_previous": False,
        "role": "",
        "designated_writer": "",
        "generation": None,
        "takeover": False,
        "previous_holder": "",
    }

    # --- gate 1: configured writer role (machine-local, never shared) ---
    try:
        role = resolve_writer_role()
    except Exception as exc:
        result["error"] = (
            "the designated-writer configuration could not be read, so this machine "
            f"refuses to publish shared output: {exc!r}"
        )
        logging.exception("Away report publish blocked: writer role configuration unreadable.")
        _writer_health_snapshot(
            lease_path=lease_path, failure=("configuration", result["error"])
        )
        return result

    result["role"] = role.role
    result["designated_writer"] = role.designated_writer
    if not role.may_publish:
        result["error"] = role.reason
        logging.error("Away report publish refused: %s", role.reason)
        _writer_health_snapshot(
            role=role, lease_path=lease_path, failure=("configuration", role.reason)
        )
        return result

    override = resolve_emergency_override()

    # --- gate 2: machine-local cross-process exclusion, held for the whole
    # publication transaction (not just the lease read/replace) ---
    try:
        with local_writer_lock(lock_key_for_path(lease_path)) as lock_state:
            lock_info = lock_state.as_dict()
            return _publish_under_local_lock(
                payload,
                target=target,
                metadata_path=metadata_path,
                lease_path=lease_path,
                result=result,
                role=role,
                override=override,
                lock_info=lock_info,
                archive=archive,
                archive_keep=archive_keep,
                wl=wl,
            )
    except LocalLockUnavailable as exc:
        result["error"] = (
            f"another process on this machine is publishing the shared report: {exc}"
        )
        logging.error("Away report publish skipped: %s", result["error"])
        _writer_health_snapshot(
            role=role,
            override=override,
            lease_path=lease_path,
            failure=("local_lock", result["error"]),
        )
        return result


def _previous_pair_disagreement(target: Path, metadata_path: Path) -> str:
    """Does the report on disk match the metadata beside it? ``""`` when it does.

    Report and metadata are replaced by two separate ``os.replace`` calls. A
    hard kill between them leaves the pair describing two different
    publications, and nothing used to notice: the next publish overwrote both
    and the discrepancy was never recorded. This is the detector - it never
    modifies either file, because guessing which half is right is not a repair.
    """
    try:
        if not target.exists() or not metadata_path.is_file():
            return ""
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return f"{metadata_path.name} is not a publication metadata object"
        recorded = str(metadata.get("sha256") or "")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if recorded and recorded != actual:
            return (
                f"{metadata_path.name} describes sha256 {recorded[:12]} (holder "
                f"{metadata.get('holder')!r}, generation {metadata.get('generation')!r}) but "
                f"{target.name} on disk hashes to {actual[:12]}; the pair was left describing "
                "two different publications, most likely by a kill between the two replaces"
            )
    except (OSError, ValueError):
        # Unreadable metadata is a separate, already-reported condition; it is
        # not evidence of a torn pair.
        return ""
    return ""


def _observed_clock_offset(lease_path: Path, wl) -> float | None:
    """Seconds by which the lease's writer appears to be ahead of our clock.

    Recorded for the plan.md sec 6.2 clock-comparison drill. Only the "their
    clock is ahead of ours" direction is observable from a lease; our own clock
    running fast looks exactly like an old lease, which is why this is a
    reported observation rather than a correction.
    """
    try:
        state = wl.inspect_lease(lease_path)
    except Exception:
        return None
    if state.get("kind") == "missing":
        return None
    try:
        return wl.observed_clock_offset_seconds(state)
    except Exception:
        return None


def _acquire_with_bounded_override(wl, lease_path: Path, override):
    """Acquire the lease, using the emergency override only if it is needed.

    An active override used to be passed as ``takeover=True`` on *every* publish,
    so an emergency configured once broke whatever live lease the other machine
    held, once per hour, for as long as the configuration stayed in place. The
    override is a recovery action: try the normal path first, and break a live
    lease only when the normal path is actually blocked.
    """
    try:
        return wl.acquire(lease_path)
    except wl.LeaseUnavailable:
        if not bool(getattr(override, "active", False)):
            raise
        logging.warning(
            "Away report lease blocked; using the configured emergency takeover "
            "(expires %s, reason %r).",
            getattr(override, "expires_at", ""),
            getattr(override, "reason", ""),
        )
        return wl.acquire(
            lease_path, takeover=True, reason=getattr(override, "reason", "")
        )


def _restore_previous_publication(
    result: dict[str, Any],
    *,
    target: Path,
    metadata_path: Path,
    previous_bytes: bytes | None,
    previous_metadata_bytes: bytes | None,
    replaced: bool,
    metadata_replaced: bool,
) -> bool:
    """Put the last verified report/metadata pair back. ``True`` if it ran.

    A failed publish must never destroy the last verified report, and the pair
    must never be left describing two different publications - so this restores
    whichever halves were already replaced, together.
    """
    if result.get("ok") or not (replaced or metadata_replaced):
        return False
    try:
        restore_items = []
        if replaced:
            restore_items.append((target, previous_bytes))
        if metadata_replaced:
            restore_items.append((metadata_path, previous_metadata_bytes))
        for restore_target, restore_bytes in restore_items:
            if restore_bytes is None:
                restore_target.unlink(missing_ok=True)
                continue
            fd, restore_tmp = tempfile.mkstemp(
                dir=str(restore_target.parent), suffix=".restore.tmp"
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(restore_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(restore_tmp, restore_target)
            finally:
                if os.path.exists(restore_tmp):
                    os.remove(restore_tmp)
        result["restored_previous"] = True
    except Exception as restore_exc:
        result["error"] += f"; previous report restore failed: {restore_exc!r}"
        logging.exception("Failed restoring the previous verified Away report.")
    return True


def _publish_under_local_lock(
    payload: Mapping[str, Any],
    *,
    target: Path,
    metadata_path: Path,
    lease_path: Path,
    result: dict[str, Any],
    role,
    override,
    lock_info: dict[str, Any],
    archive: bool,
    archive_keep: int,
    wl,
) -> dict[str, Any]:
    """Steps 3-7 of the publication gate, with the local lock already held."""

    disagreement = _previous_pair_disagreement(target, metadata_path)
    if disagreement:
        # Detection only - nothing is repaired here, because the previous
        # verified pair is the thing being protected and a guess about which
        # half is correct is not a repair. The next successful publish replaces
        # both. Reported so a crash between the two replaces cannot pass
        # unnoticed until somebody compares hashes by hand.
        result["previous_pair_disagreement"] = disagreement
        logging.error("Away report and its metadata disagree on disk: %s", disagreement)

    # --- gate 3: hardened writer lease + fencing validation ---
    clock_offset = _observed_clock_offset(lease_path, wl)

    def snapshot(**kwargs) -> None:
        kwargs.setdefault("role", role)
        kwargs.setdefault("override", override)
        kwargs.setdefault("lock_info", lock_info)
        kwargs.setdefault("lease_path", lease_path)
        kwargs.setdefault("clock_offset_seconds", clock_offset)
        kwargs.setdefault("pair_disagreement", disagreement)
        _writer_health_snapshot(**kwargs)

    try:
        lease = _acquire_with_bounded_override(wl, lease_path, override)
        result["holder"] = str(lease.get("holder") or "")
        result["lease_expires_at"] = str(lease.get("expires_at") or "")
        result["generation"] = lease.get("generation")
        result["takeover"] = bool(lease.get("takeover"))
        result["previous_holder"] = str(lease.get("previous_holder") or "")
    except wl.LeaseUnavailable as exc:
        result["error"] = f"another machine is the active writer: {exc}"
        logging.info("Away report publish skipped: %s", result["error"])
        snapshot(failure=("lease", result["error"]))
        return result
    except Exception as exc:
        result["error"] = f"writer lease check failed: {exc!r}"
        logging.exception(
            "Away report publish blocked because writer ownership could not be verified."
        )
        snapshot(failure=("lease", result["error"]))
        return result

    # --- gate 4: render locally (nothing shared is touched yet) ---
    try:
        text = render_away_report(payload)
    except Exception as exc:
        result["error"] = f"render failed: {exc!r}"
        logging.exception("Away report render failed; previous report left untouched.")
        snapshot(lease=lease, failure=("render", result["error"]))
        return result

    # --- gate 5a: ownership can have changed while we rendered ---
    try:
        wl.assert_still_owned(
            lease_path, holder=result["holder"], generation=result["generation"]
        )
    except wl.LeaseUnavailable as exc:
        result["error"] = f"writer ownership was lost before publication: {exc}"
        logging.error("Away report publish aborted: %s", result["error"])
        snapshot(lease=lease, failure=("ownership_lost", result["error"]))
        return result

    previous_bytes = None
    previous_metadata_bytes = None
    try:
        if target.exists():
            previous_bytes = target.read_bytes()
        if metadata_path.exists():
            previous_metadata_bytes = metadata_path.read_bytes()
    except OSError as exc:
        result["error"] = f"could not preserve previous publication: {exc!r}"
        logging.exception("Away report publish blocked because the previous publication was unreadable.")
        snapshot(lease=lease, failure=("previous_publication", result["error"]))
        return result

    replaced = False
    metadata_replaced = False
    aborting: BaseException | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            report_bytes = text.encode("utf-8")
            with os.fdopen(fd, "wb") as handle:
                handle.write(report_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            candidate = Path(tmp_name).read_bytes()
            expected = hashlib.sha256(report_bytes).hexdigest()
            candidate_hash = hashlib.sha256(candidate).hexdigest()
            if candidate_hash != expected:
                raise ValueError("staged report hash mismatch")
            # --- gate 5b: last check, immediately before the replacement ---
            wl.assert_still_owned(
                lease_path, holder=result["holder"], generation=result["generation"]
            )
            os.replace(tmp_name, target)
            replaced = True
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
        readback = target.read_bytes()
        actual = hashlib.sha256(readback).hexdigest()
        if expected != actual:
            raise ValueError("readback hash mismatch")
        result["sha256"] = actual

        metadata = {
            "schema": "away_report_publish_v1",
            "verified_at": datetime.now().isoformat(timespec="seconds"),
            "report_generated_at": str(payload.get("generated_at") or ""),
            "sha256": actual,
            "bytes": len(readback),
            "holder": result["holder"],
            "lease_expires_at": result["lease_expires_at"],
            "generation": lease.get("generation"),
            "instance_id": str(lease.get("instance_id") or ""),
            "machine": str(lease.get("machine") or ""),
            "designated_writer": role.designated_writer,
            "takeover": bool(lease.get("takeover")),
            "previous_holder": str(lease.get("previous_holder") or ""),
        }
        metadata_bytes = (json.dumps(metadata, indent=1) + "\n").encode("utf-8")
        fd, metadata_tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".meta.tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(metadata_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            # --- gate 5c: the metadata replace writes shared output too,
            # and the last ownership check was a whole readback + hash + stage
            # ago. Re-verify here as well, so the second half of the pair is not
            # published by a writer that has since been fenced off.
            wl.assert_still_owned(
                lease_path, holder=result["holder"], generation=result["generation"]
            )
            os.replace(metadata_tmp, metadata_path)
            metadata_replaced = True
        finally:
            if os.path.exists(metadata_tmp):
                os.remove(metadata_tmp)
        metadata_readback = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata_readback.get("schema") != "away_report_publish_v1"
            or metadata_readback.get("sha256") != actual
            or metadata_readback.get("holder") != result["holder"]
            or metadata_readback.get("generation") != lease.get("generation")
        ):
            raise ValueError("publication metadata readback mismatch")
        result["metadata_path"] = metadata_path
        result["verified"] = True
        result["ok"] = True
    except wl.LeaseUnavailable as exc:
        result["ok"] = False
        result["verified"] = False
        result["error"] = f"writer ownership was lost before publication: {exc}"
        logging.error("Away report publish aborted: %s", result["error"])
    except Exception as exc:
        result["ok"] = False
        result["verified"] = False
        result["error"] = str(exc) if isinstance(exc, ValueError) else f"publish failed: {exc!r}"
        logging.exception("Failed writing the Auto Pilot away report to %s", target)
    except BaseException as exc:
        # KeyboardInterrupt, SystemExit, a Qt/thread abort - anything that is
        # NOT an Exception used to skip both handlers and therefore skip the
        # rollback below, leaving the report replaced and the metadata still
        # describing the previous publication. Layer 4 forbids exactly that, and
        # this repo has a documented native-crash history, so the window is not
        # hypothetical. Roll back first, then let the interrupt continue.
        result["ok"] = False
        result["verified"] = False
        result["error"] = (
            f"publish interrupted by {type(exc).__name__}; the previous verified "
            "publication is being restored"
        )
        logging.error("Away report publish interrupted by %s.", type(exc).__name__)
        aborting = exc
    rolled_back = _restore_previous_publication(
        result,
        target=target,
        metadata_path=metadata_path,
        previous_bytes=previous_bytes,
        previous_metadata_bytes=previous_metadata_bytes,
        replaced=replaced,
        metadata_replaced=metadata_replaced,
    )
    if result["ok"]:
        snapshot(
            lease=lease,
            publication={
                "at": datetime.now().isoformat(timespec="seconds"),
                "path": str(target),
                "holder": result["holder"],
                "generation": lease.get("generation"),
                "sha256": result["sha256"],
                "takeover": bool(lease.get("takeover")),
                "previous_holder": str(lease.get("previous_holder") or ""),
            },
        )
    else:
        snapshot(lease=lease, failure=("publish", result["error"]))
    if aborting is not None:
        raise aborting
    if rolled_back:
        return result
    if result["ok"] and archive:
        try:
            archive_dir = target.parent / "away_report_archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            (archive_dir / f"{target.stem}_{stamp}{target.suffix}").write_text(
                text, encoding="utf-8"
            )
            history = sorted(archive_dir.glob(f"{target.stem}_*{target.suffix}"))
            for stale in history[: max(0, len(history) - int(archive_keep))]:
                try:
                    stale.unlink()
                except OSError:
                    pass
        except Exception:
            logging.exception("Away report archive write failed (latest report is fine).")
    return result


def release_away_report_lease(path: Path | None = None) -> bool:
    """Hand the away-report writer lease back on a clean shutdown.

    Called from ``AutopilotService.shutdown`` and ``MainWindow.closeEvent``.
    Without it the lease sat on disk until its TTL expired, so the other machine
    (or this one, restarted) was locked out of the writer slot for up to a full
    TTL after a perfectly clean exit - and ``release`` was dead code that only
    the tests ever called.

    Safety is unchanged either way:

    * it releases only a lease this exact process instance can prove is its own,
      so it can never drop the other machine's claim, nor another process's on
      this machine;
    * a **hard kill**, where this never runs, is covered by expiry. The stale
      lease is recovered through the normal acquisition path once it ages out -
      never by a later process claiming the dead one's lease was already its
      own. A killed writer therefore degrades to a bounded wait, not a wedge.
    """
    try:
        import writer_lease as wl

        target = Path(path) if path is not None else Path(AUTOPILOT_REPORT_FILE)
        lease_path = target.with_suffix(target.suffix + ".lease")
        released = wl.release(lease_path)
        if released:
            logging.info("Away report writer lease released on shutdown (%s).", lease_path.name)
        return bool(released)
    except Exception:
        logging.debug("Away report writer lease release failed.", exc_info=True)
        return False


def write_heartbeat(
    *,
    current_job: str = "",
    next_job: str = "",
    last_success: str = "",
    path: Path | None = None,
) -> Path | None:
    """Atomic heartbeat (plan.md Phase 2.8): proves the runtime is alive and
    says what it is doing, machine-locally (never in the shared folder)."""
    import socket
    import tempfile

    try:
        from project_paths import get_diagnostics_dir

        target = Path(path) if path is not None else get_diagnostics_dir() / "heartbeat.json"
        payload = {
            "schema": "heartbeat_v1",
            "machine": socket.gethostname(),
            "pid": __import__("os").getpid(),
            "ts": datetime.now().isoformat(timespec="seconds"),
            "current_job": str(current_job or ""),
            "next_job": str(next_job or ""),
            "last_success": str(last_success or ""),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        import os as _os

        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        with _os.fdopen(fd, "w", encoding="utf-8") as handle:
            _json.dump(payload, handle)
        _os.replace(tmp, target)
        return target
    except Exception:
        logging.debug("Heartbeat write failed.", exc_info=True)
        return None


def write_away_report(payload: Mapping[str, Any], path: Path | None = None) -> Path:
    """Compatibility wrapper; prefer :func:`publish_away_report`.

    Raises ``RuntimeError`` when the publish was refused. It used to return the
    *target* path either way - and ``result["path"]`` is pre-populated before any
    gate runs - so a read-only secondary, or a machine with no designated writer
    configured, handed the caller a path to a file it had not written and had no
    intention of writing. This is the one function whose name promises a write.
    """
    result = publish_away_report(payload, path)
    if not result.get("ok"):
        raise RuntimeError(
            f"the away report was not written: {result.get('error') or 'refused'}"
        )
    return Path(result["path"])
