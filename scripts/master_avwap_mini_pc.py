#!/usr/bin/env python3
"""GUI-first Master AVWAP scheduler for an always-on Windows mini PC."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, time as dt_time, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception:
    tk = None
    ttk = None
    messagebox = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from market_session import (
    get_default_hourly_scan_schedule,
    get_default_setup_tracker_refresh_slot,
    get_default_stop_time_label,
    get_market_session_window,
)
from master_avwap import (
    MasterAvwapGUI,
    connect_daily_data_client,
    disconnect_daily_data_client,
    fetch_daily_bars,
    load_tickers,
    run_master,
)
from project_paths import (
    APP_LOG_BACKUP_COUNT,
    LOG_DIR,
    LONGS_FILE,
    MASTER_AVWAP_EVENT_TICKERS_FILE,
    MASTER_AVWAP_PRIORITY_SETUPS_FILE,
    MASTER_AVWAP_REPORT_FILE,
    MASTER_AVWAP_SETUP_TRACKER_FILE,
    MASTER_AVWAP_TRADINGVIEW_REPORT_FILE,
    PERSISTENT_RUNTIME_DATA_DIR,
    SHORTS_FILE,
    get_tracker_storage_details,
)

STATUS_FILE = LONGS_FILE.parent / "master_avwap_mini_pc_status.txt"
STATE_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_mini_pc_state.json"
LOCK_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_mini_pc.lock"
SCHEDULER_LOG_FILE = LOG_DIR / "master_avwap_mini_pc.log"
APP_LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s]: %(message)s"
STATUS_PREVIEW_RUNS = 8
SLEEP_POLL_SECONDS = 30
STALE_LOCK_MAX_AGE = timedelta(hours=12)
GUI_TICK_MS = 15_000
WATCHLIST_FILTER_CLIENT_ID = 1005
WATCHLIST_FILTER_DAYS = 5
WATCHLIST_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]+$")

_LOCK_ACQUIRED = False


def get_current_default_schedule(reference: datetime | None = None) -> list[str]:
    return list(get_default_hourly_scan_schedule(reference=reference))


def get_current_default_stop_at(reference: datetime | None = None) -> str:
    return get_default_stop_time_label(reference=reference)


def parse_clock(value: str) -> dt_time:
    try:
        return datetime.strptime(str(value).strip(), "%H:%M").time()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid time '{value}'. Use HH:MM in 24-hour format.") from exc


def normalize_schedule(raw_schedule: str) -> list[str]:
    parsed = []
    seen = set()
    for item in str(raw_schedule or "").split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        slot = parse_clock(cleaned).strftime("%H:%M")
        if slot in seen:
            continue
        seen.add(slot)
        parsed.append(slot)
    parsed.sort()
    if not parsed:
        raise argparse.ArgumentTypeError("Schedule cannot be empty.")
    return parsed


def default_state(schedule: list[str]) -> dict[str, Any]:
    now_iso = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": 1,
        "date": datetime.now().date().isoformat(),
        "schedule": list(schedule),
        "slots": {slot: {"status": "pending"} for slot in schedule},
        "runs": [],
        "last_status": "idle",
        "last_error": "",
        "last_success_at": None,
        "last_filter_summary": None,
        "updated_at": now_iso,
    }


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _pid_is_running(pid_value: Any) -> bool:
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        return False

    if pid <= 0:
        return False
    if pid == os.getpid():
        return True

    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return True

        output = (completed.stdout or "").strip()
        if not output or output.startswith("INFO:"):
            return False
        return f'"{pid}"' in output

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _stale_lock_reason(existing: Any) -> str | None:
    if not isinstance(existing, dict):
        return "lock payload is invalid"

    started_at = existing.get("started_at")
    try:
        started_dt = datetime.fromisoformat(str(started_at))
    except (TypeError, ValueError):
        started_dt = None
    if started_dt and (datetime.now() - started_dt) > STALE_LOCK_MAX_AGE:
        return f"lock is older than {int(STALE_LOCK_MAX_AGE.total_seconds() // 3600)} hours"

    pid = existing.get("pid")
    if pid is not None and not _pid_is_running(pid):
        return f"recorded PID {pid} is not running"

    script_value = existing.get("script")
    if script_value:
        try:
            script_path = Path(str(script_value)).expanduser()
        except Exception:
            script_path = None
        if script_path is not None and not script_path.exists():
            return f"recorded script path is missing: {script_path}"

    return None


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_state(schedule: list[str]) -> dict[str, Any]:
    loaded = load_json(STATE_FILE, default={})
    today_iso = datetime.now().date().isoformat()
    if not isinstance(loaded, dict):
        return default_state(schedule)
    if loaded.get("date") != today_iso:
        return default_state(schedule)
    if loaded.get("schedule") != list(schedule):
        return default_state(schedule)

    state = default_state(schedule)
    state.update(loaded)

    slots = state.get("slots")
    if not isinstance(slots, dict):
        slots = {}
    normalized_slots = {}
    for slot in schedule:
        existing = slots.get(slot, {})
        normalized_slots[slot] = existing if isinstance(existing, dict) else {"status": "pending"}
    state["slots"] = normalized_slots

    runs = state.get("runs")
    state["runs"] = runs if isinstance(runs, list) else []
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return state


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(STATE_FILE, state)


def slot_datetime(slot: str, base_date: datetime | None = None) -> datetime:
    target = base_date or datetime.now()
    return datetime.combine(target.date(), parse_clock(slot))


def get_due_pending_slots(state: dict[str, Any], schedule: list[str], now: datetime | None = None) -> list[str]:
    now = now or datetime.now()
    due = []
    slots = state.get("slots", {})
    for slot in schedule:
        slot_state = slots.get(slot, {})
        if slot_state.get("status") != "pending":
            continue
        if slot_datetime(slot, now) <= now:
            due.append(slot)
    return due


def get_next_pending_slot(state: dict[str, Any], schedule: list[str], now: datetime | None = None) -> str | None:
    now = now or datetime.now()
    slots = state.get("slots", {})
    for slot in schedule:
        slot_state = slots.get(slot, {})
        if slot_state.get("status") != "pending":
            continue
        if slot_datetime(slot, now) > now:
            return slot
    return None


def count_watchlist_symbols(path: Path) -> int:
    if not path.exists():
        return 0
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            symbol = str(line).strip().upper()
            if not symbol or symbol.startswith("SYMBOLS FROM TC2000"):
                continue
            seen.add(symbol)
    return len(seen)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    cleaned = []
    seen = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)
    return cleaned


def _format_symbol_preview(symbols: list[str], max_items: int = 12) -> str:
    cleaned = _dedupe_symbols(symbols)
    if not cleaned:
        return "None"
    preview = ", ".join(cleaned[:max_items])
    if len(cleaned) > max_items:
        preview += f" (+{len(cleaned) - max_items} more)"
    return preview


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _coerce_bar_date(value: Any):
    if value is None:
        return None
    if hasattr(value, "date"):
        try:
            return value.date()
        except Exception:
            pass
    try:
        return datetime.fromisoformat(str(value)).date()
    except (TypeError, ValueError):
        return None


def _get_previous_day_range_break(df) -> dict[str, Any] | None:
    if df is None or getattr(df, "empty", True):
        return None
    if len(df.index) < 2:
        return None

    work = df.sort_values("datetime").reset_index(drop=True)
    current_row = work.iloc[-1]
    previous_row = work.iloc[-2]

    current_date = _coerce_bar_date(current_row.get("datetime"))
    previous_date = _coerce_bar_date(previous_row.get("datetime"))
    if current_date is None or previous_date is None:
        return None
    if current_date != datetime.now().date():
        return None

    current_high = _coerce_float(current_row.get("high"))
    current_low = _coerce_float(current_row.get("low"))
    previous_high = _coerce_float(previous_row.get("high"))
    previous_low = _coerce_float(previous_row.get("low"))
    if None in (current_high, current_low, previous_high, previous_low):
        return None

    return {
        "current_date": current_date.isoformat(),
        "previous_date": previous_date.isoformat(),
        "current_high": current_high,
        "current_low": current_low,
        "previous_high": previous_high,
        "previous_low": previous_low,
    }


def _rewrite_watchlist_symbols(path: Path, keep_symbols: list[str]) -> None:
    keep_ordered = _dedupe_symbols(keep_symbols)
    keep_set = set(keep_ordered)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(keep_ordered)
        path.write_text(f"{content}\n" if content else "", encoding="utf-8")
        return

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    rendered_lines = []
    emitted = set()

    for raw_line in raw_lines:
        stripped = raw_line.strip()
        upper = stripped.upper()
        if not stripped:
            rendered_lines.append("")
            continue
        if upper.startswith("SYMBOLS FROM TC2000"):
            rendered_lines.append(raw_line.rstrip())
            continue
        if WATCHLIST_SYMBOL_RE.fullmatch(upper):
            if upper in keep_set and upper not in emitted:
                rendered_lines.append(upper)
                emitted.add(upper)
            continue
        rendered_lines.append(raw_line.rstrip())

    for symbol in keep_ordered:
        if symbol not in emitted:
            rendered_lines.append(symbol)

    content = "\n".join(rendered_lines).rstrip()
    path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def format_watchlist_filter_summary(summary: Any) -> str:
    if not isinstance(summary, dict):
        return "Not run yet."

    message = str(summary.get("message") or "").strip()
    if message:
        return message

    status = str(summary.get("status") or "").strip().lower()
    if status == "error":
        error_text = str(summary.get("error") or "").strip()
        return f"Filter error: {error_text or 'Unknown error'}"
    if status == "no_symbols":
        return "No symbols were available to filter."
    if status == "waiting_for_session":
        return "No live session bar was available yet."
    return "No watchlist filter summary available."


def filter_watchlists_by_previous_day_levels() -> dict[str, Any]:
    started_at = datetime.now()
    longs = _dedupe_symbols(load_tickers(LONGS_FILE))
    shorts = _dedupe_symbols(load_tickers(SHORTS_FILE))
    summary = {
        "ran_at": started_at.isoformat(timespec="seconds"),
        "status": "ok",
        "longs_before": len(longs),
        "shorts_before": len(shorts),
        "longs_after": len(longs),
        "shorts_after": len(shorts),
        "symbols_considered": 0,
        "symbols_with_live_session_bar": 0,
        "symbols_skipped_no_session_bar": 0,
        "removed_longs": [],
        "removed_shorts": [],
        "message": "",
    }

    if not longs and not shorts:
        summary["status"] = "no_symbols"
        summary["message"] = "No symbols were available to filter."
        logging.info(summary["message"])
        return summary

    long_set = set(longs)
    short_set = set(shorts)
    symbol_list = sorted(long_set | short_set)
    summary["symbols_considered"] = len(symbol_list)

    removed_longs = []
    removed_shorts = []
    skipped_no_session_bar = 0
    ib = None
    try:
        ib = connect_daily_data_client(client_id=WATCHLIST_FILTER_CLIENT_ID, startup_wait=1.0)
        for symbol in symbol_list:
            df = fetch_daily_bars(ib, symbol, WATCHLIST_FILTER_DAYS)
            day_range = _get_previous_day_range_break(df)
            if not day_range:
                skipped_no_session_bar += 1
                continue

            summary["symbols_with_live_session_bar"] += 1
            if symbol in long_set and day_range["current_low"] < day_range["previous_low"]:
                removed_longs.append(symbol)
            if symbol in short_set and day_range["current_high"] > day_range["previous_high"]:
                removed_shorts.append(symbol)
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = str(exc)
        summary["message"] = f"Filter error: {exc}"
        logging.exception("Previous-day watchlist filter failed.")
        return summary
    finally:
        disconnect_daily_data_client(ib)

    summary["symbols_skipped_no_session_bar"] = skipped_no_session_bar

    removed_long_set = set(removed_longs)
    removed_short_set = set(removed_shorts)
    kept_longs = [symbol for symbol in longs if symbol not in removed_long_set]
    kept_shorts = [symbol for symbol in shorts if symbol not in removed_short_set]

    if removed_long_set:
        _rewrite_watchlist_symbols(LONGS_FILE, kept_longs)
    if removed_short_set:
        _rewrite_watchlist_symbols(SHORTS_FILE, kept_shorts)

    summary["removed_longs"] = sorted(removed_long_set)
    summary["removed_shorts"] = sorted(removed_short_set)
    summary["longs_after"] = len(kept_longs)
    summary["shorts_after"] = len(kept_shorts)

    if summary["symbols_with_live_session_bar"] == 0:
        summary["status"] = "waiting_for_session"
        summary["message"] = (
            "No live session bar was available yet; watchlists were left unchanged."
        )
    else:
        parts = [
            f"checked {summary['symbols_with_live_session_bar']}/{summary['symbols_considered']} symbol(s)",
            f"removed {len(summary['removed_longs'])} long(s)",
            f"removed {len(summary['removed_shorts'])} short(s)",
        ]
        if summary["removed_longs"]:
            parts.append(f"longs: {_format_symbol_preview(summary['removed_longs'])}")
        if summary["removed_shorts"]:
            parts.append(f"shorts: {_format_symbol_preview(summary['removed_shorts'])}")
        if skipped_no_session_bar:
            parts.append(f"skipped {skipped_no_session_bar} without a live session bar")
        summary["message"] = "; ".join(parts)

    logging.info("Watchlist range filter: %s", summary["message"])
    return summary


def run_master_with_watchlist_filter(update_setup_tracker: bool = True) -> dict[str, Any]:
    filter_summary = filter_watchlists_by_previous_day_levels()
    try:
        run_master(
            use_shared_watchlists=True,
            update_setup_tracker=update_setup_tracker,
        )
    except Exception as exc:
        setattr(exc, "watchlist_filter_summary", filter_summary)
        raise
    return filter_summary


def should_update_setup_tracker_for_slot(
    schedule: list[str],
    scheduled_slot: str | None,
) -> bool:
    if not scheduled_slot:
        return False
    if not schedule:
        return False
    scheduled_dt = slot_datetime(scheduled_slot)
    default_refresh_slot = get_default_setup_tracker_refresh_slot(reference=scheduled_dt)
    refresh_slot = (
        default_refresh_slot
        if default_refresh_slot in schedule
        else schedule[-1]
    )
    return scheduled_slot == refresh_slot


def _extract_symbols_from_text(text: str) -> list[str]:
    symbols = []
    seen = set()
    for raw_value in str(text or "").split(","):
        symbol = raw_value.strip().upper()
        if not symbol or symbol == "NONE":
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _format_symbol_group(symbols: list[str]) -> str:
    cleaned = []
    seen = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)
    return ", ".join(cleaned) if cleaned else "None"


def load_tradingview_groups() -> dict[str, dict[str, list[str]]]:
    text = read_text(MASTER_AVWAP_TRADINGVIEW_REPORT_FILE)
    groups = {
        "favorites": {"LONG": [], "SHORT": []},
        "near_favorite_zones": {"LONG": [], "SHORT": []},
    }
    if not text:
        return groups

    section_lookup = {
        "Best current favorite setups": "favorites",
        "Near favorite zones": "near_favorite_zones",
    }
    current_section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            current_section = None
            continue
        if line in section_lookup:
            current_section = section_lookup[line]
            continue
        if line.startswith("-"):
            continue
        if current_section not in groups or ":" not in line:
            continue

        side_label, values = line.split(":", 1)
        side = side_label.strip().upper()
        if side not in ("LONG", "SHORT"):
            continue
        groups[current_section][side] = _extract_symbols_from_text(values)

    return groups


def write_status_file(
    state: dict[str, Any],
    schedule: list[str],
    stop_at: str,
    phase: str,
    note: str = "",
    active_slot: str = "",
) -> None:
    storage = get_tracker_storage_details()
    now = datetime.now()
    next_slot = get_next_pending_slot(state, schedule, now=now)
    longs_count = count_watchlist_symbols(LONGS_FILE)
    shorts_count = count_watchlist_symbols(SHORTS_FILE)
    main_report = read_text(MASTER_AVWAP_REPORT_FILE)
    filter_summary = state.get("last_filter_summary")
    tradingview_groups = load_tradingview_groups()
    favorite_symbols = (
        tradingview_groups["favorites"]["LONG"]
        + tradingview_groups["favorites"]["SHORT"]
    )
    near_favorite_symbols = (
        tradingview_groups["near_favorite_zones"]["LONG"]
        + tradingview_groups["near_favorite_zones"]["SHORT"]
    )
    favorite_focus_symbols = favorite_symbols + near_favorite_symbols

    lines = [
        "Master AVWAP Mini PC Status",
        f"TV Paste: {_format_symbol_group(favorite_focus_symbols)}",
        f"Generated at: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Phase: {phase}",
        f"Today's schedule: {', '.join(schedule)}",
        f"Stop time: {stop_at}",
        f"Active slot: {active_slot or 'None'}",
        f"Next pending slot: {next_slot or 'None'}",
        f"Last status: {state.get('last_status', 'idle')}",
        f"Last success at: {state.get('last_success_at') or 'None'}",
        f"Last error: {state.get('last_error') or 'None'}",
        f"Shared folder: {storage['data_dir']}",
        f"Long watchlist count: {longs_count} ({LONGS_FILE})",
        f"Short watchlist count: {shorts_count} ({SHORTS_FILE})",
        f"Last watchlist filter: {format_watchlist_filter_summary(filter_summary)}",
        f"Setup tracker: {MASTER_AVWAP_SETUP_TRACKER_FILE}",
        f"Priority setups report: {MASTER_AVWAP_PRIORITY_SETUPS_FILE}",
        f"Ticker buckets report: {MASTER_AVWAP_EVENT_TICKERS_FILE}",
        f"TradingView report: {MASTER_AVWAP_TRADINGVIEW_REPORT_FILE}",
        f"Main AVWAP report: {MASTER_AVWAP_REPORT_FILE}",
    ]

    if note:
        lines.extend(["", f"Note: {note}"])

    lines.extend(
        [
            "",
            "Mobile TradingView paste",
            f"Favorite setups: {_format_symbol_group(favorite_symbols)}",
            f"Favorite zones: {_format_symbol_group(near_favorite_symbols)}",
            f"Favorite focus combined: {_format_symbol_group(favorite_focus_symbols)}",
        ]
    )

    lines.extend(["", "Slot coverage:"])
    slots = state.get("slots", {})
    for slot in schedule:
        slot_state = slots.get(slot, {})
        status = slot_state.get("status", "pending")
        detail_parts = [status]
        if slot_state.get("run_id"):
            detail_parts.append(f"run={slot_state['run_id']}")
        if slot_state.get("covered_at"):
            detail_parts.append(f"covered={slot_state['covered_at']}")
        if slot_state.get("note"):
            detail_parts.append(str(slot_state["note"]))
        lines.append(f"- {slot}: {' | '.join(detail_parts)}")

    lines.extend(["", "Recent runs:"])
    runs = state.get("runs", [])
    if runs:
        for run in runs[-STATUS_PREVIEW_RUNS:]:
            duration = format_duration(run.get("duration_seconds"))
            lines.append(
                f"- {run.get('slot', 'manual')} {run.get('status', 'unknown')} "
                f"start={run.get('started_at', 'n/a')} end={run.get('finished_at', 'n/a')} "
                f"duration={duration}"
            )
            if run.get("error"):
                lines.append(f"  error={run['error']}")
    else:
        lines.append("- No runs recorded today.")

    lines.extend(["", "Latest main AVWAP output:", "-" * 80])
    if main_report:
        lines.extend(main_report.splitlines())
    else:
        lines.append("(No main AVWAP report has been written yet.)")

    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def attach_scheduler_log_handler() -> None:
    root_logger = logging.getLogger()
    target = str(SCHEDULER_LOG_FILE.resolve())
    for handler in root_logger.handlers:
        handler_path = getattr(handler, "baseFilename", "")
        if handler_path and str(Path(handler_path).resolve()) == target:
            return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        SCHEDULER_LOG_FILE,
        maxBytes=2_000_000,
        backupCount=APP_LOG_BACKUP_COUNT,
    )
    handler.setFormatter(logging.Formatter(APP_LOG_FORMAT))
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def release_lock() -> None:
    global _LOCK_ACQUIRED
    if not _LOCK_ACQUIRED:
        return
    try:
        LOCK_FILE.unlink(missing_ok=True)
    finally:
        _LOCK_ACQUIRED = False


def acquire_lock(force_reset: bool = False) -> None:
    global _LOCK_ACQUIRED

    PERSISTENT_RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force_reset:
        LOCK_FILE.unlink(missing_ok=True)

    if LOCK_FILE.exists():
        existing = load_json(LOCK_FILE, default={})
        if _stale_lock_reason(existing):
            LOCK_FILE.unlink(missing_ok=True)

    payload = {
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
    }
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another mini-PC runner appears to be active: {LOCK_FILE}") from exc

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    _LOCK_ACQUIRED = True
    atexit.register(release_lock)


def mark_slots_skipped_before_trigger(
    state: dict[str, Any],
    schedule: list[str],
    trigger_slot: str,
    started_at: datetime,
) -> None:
    for slot in schedule:
        if slot == trigger_slot:
            break
        slot_state = state["slots"].get(slot, {})
        if slot_state.get("status") != "pending":
            continue
        state["slots"][slot] = {
            "status": "covered_by_late_start",
            "covered_at": started_at.isoformat(timespec="seconds"),
            "note": f"Superseded by {trigger_slot} run",
        }


def cover_slots_after_success(
    state: dict[str, Any],
    schedule: list[str],
    trigger_slot: str,
    run_id: str,
    finished_at: datetime,
    dry_run: bool = False,
) -> None:
    finished_iso = finished_at.isoformat(timespec="seconds")
    trigger_status = "dry_run" if dry_run else "success"
    covered_status = "covered_by_dry_run" if dry_run else "covered_by_running_scan"
    for slot in schedule:
        slot_dt = slot_datetime(slot, finished_at)
        if slot_dt > finished_at:
            break
        slot_state = state["slots"].get(slot, {})
        if slot == trigger_slot:
            state["slots"][slot] = {
                "status": trigger_status,
                "run_id": run_id,
                "covered_at": finished_iso,
            }
            continue
        if slot_state.get("status") == "pending":
            state["slots"][slot] = {
                "status": covered_status,
                "run_id": run_id,
                "covered_at": finished_iso,
            }


def execute_scan(
    state: dict[str, Any],
    schedule: list[str],
    stop_at: str,
    trigger_slot: str,
    dry_run: bool = False,
) -> bool:
    started_at = datetime.now()
    run_id = started_at.strftime("%Y%m%d-%H%M%S")
    update_setup_tracker = should_update_setup_tracker_for_slot(schedule, trigger_slot)
    mark_slots_skipped_before_trigger(state, schedule, trigger_slot, started_at)
    save_state(state)
    write_status_file(
        state,
        schedule,
        stop_at,
        phase="running",
        note=f"Running slot {trigger_slot}.",
        active_slot=trigger_slot,
    )

    logging.info(
        "Starting mini-PC Master AVWAP run for slot %s. update_setup_tracker=%s",
        trigger_slot,
        update_setup_tracker,
    )
    error_text = ""
    status = "success"
    filter_summary = None
    try:
        if dry_run:
            logging.info("Dry-run enabled; skipping live Master AVWAP scan for slot %s.", trigger_slot)
            filter_summary = {
                "ran_at": datetime.now().isoformat(timespec="seconds"),
                "status": "dry_run",
                "message": "Dry-run mode skipped the watchlist filter.",
            }
        else:
            filter_summary = run_master_with_watchlist_filter(
                update_setup_tracker=update_setup_tracker,
            )
    except Exception as exc:
        status = "failed"
        error_text = str(exc)
        filter_summary = getattr(exc, "watchlist_filter_summary", filter_summary)
        logging.exception("Mini-PC Master AVWAP run failed for slot %s.", trigger_slot)

    finished_at = datetime.now()
    duration_seconds = round((finished_at - started_at).total_seconds(), 2)
    run_record = {
        "run_id": run_id,
        "slot": trigger_slot,
        "status": status if not dry_run else "dry_run",
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": duration_seconds,
    }
    if error_text:
        run_record["error"] = error_text
    state.setdefault("runs", []).append(run_record)
    state["last_filter_summary"] = filter_summary

    if status == "failed":
        state["slots"][trigger_slot] = {
            "status": "failed",
            "run_id": run_id,
            "covered_at": finished_at.isoformat(timespec="seconds"),
            "note": error_text,
        }
        state["last_status"] = "failed"
        state["last_error"] = error_text
        save_state(state)
        write_status_file(
            state,
            schedule,
            stop_at,
            phase="waiting",
            note=f"Slot {trigger_slot} failed: {error_text}",
        )
        return False

    cover_slots_after_success(state, schedule, trigger_slot, run_id, finished_at, dry_run=dry_run)
    state["last_status"] = "success" if not dry_run else "dry_run"
    state["last_error"] = ""
    state["last_success_at"] = finished_at.isoformat(timespec="seconds")
    save_state(state)
    write_status_file(
        state,
        schedule,
        stop_at,
        phase="waiting",
        note=f"Slot {trigger_slot} completed in {format_duration(duration_seconds)}.",
    )
    logging.info("Finished mini-PC Master AVWAP run for slot %s in %s.", trigger_slot, format_duration(duration_seconds))
    return True


def sleep_until(target_dt: datetime, state: dict[str, Any], schedule: list[str], stop_at: str) -> None:
    while True:
        now = datetime.now()
        remaining = (target_dt - now).total_seconds()
        if remaining <= 0:
            return
        write_status_file(
            state,
            schedule,
            stop_at,
            phase="waiting",
            note=f"Sleeping until {target_dt.strftime('%H:%M:%S')}.",
        )
        time.sleep(min(SLEEP_POLL_SECONDS, max(1, int(remaining))))


def maybe_shutdown_windows() -> None:
    if sys.platform != "win32":
        logging.warning("Auto-shutdown was requested, but this platform is not Windows. Skipping shutdown.")
        return
    logging.info("Issuing Windows shutdown command.")
    subprocess.run(["shutdown", "/s", "/t", "0"], check=False)


def run_once(schedule: list[str], stop_at: str, dry_run: bool = False) -> int:
    state = load_state(schedule)
    update_setup_tracker = False
    write_status_file(
        state,
        schedule,
        stop_at,
        phase="running",
        note="Running one immediate mini-PC scan.",
        active_slot="manual",
    )
    logging.info(
        "Starting one immediate mini-PC Master AVWAP scan. update_setup_tracker=%s",
        update_setup_tracker,
    )
    started_at = datetime.now()
    error_text = ""
    filter_summary = None
    try:
        if dry_run:
            logging.info("Dry-run enabled; skipping live Master AVWAP scan.")
            filter_summary = {
                "ran_at": datetime.now().isoformat(timespec="seconds"),
                "status": "dry_run",
                "message": "Dry-run mode skipped the watchlist filter.",
            }
        else:
            filter_summary = run_master_with_watchlist_filter(
                update_setup_tracker=update_setup_tracker,
            )
        status = "success" if not dry_run else "dry_run"
    except Exception as exc:
        status = "failed"
        error_text = str(exc)
        filter_summary = getattr(exc, "watchlist_filter_summary", filter_summary)
        logging.exception("Immediate mini-PC Master AVWAP scan failed.")

    finished_at = datetime.now()
    duration_seconds = round((finished_at - started_at).total_seconds(), 2)
    state.setdefault("runs", []).append(
        {
            "run_id": started_at.strftime("%Y%m%d-%H%M%S"),
            "slot": "manual",
            "status": status,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": duration_seconds,
            "error": error_text,
        }
    )
    state["last_status"] = status
    state["last_error"] = error_text
    state["last_filter_summary"] = filter_summary
    if status != "failed":
        state["last_success_at"] = finished_at.isoformat(timespec="seconds")
    save_state(state)
    note = (
        f"Immediate run completed in {format_duration(duration_seconds)}."
        if status != "failed"
        else f"Immediate run failed: {error_text}"
    )
    write_status_file(state, schedule, stop_at, phase="idle", note=note)
    return 0 if status != "failed" else 1


def run_schedule(schedule: list[str], stop_at: str, dry_run: bool = False, shutdown_at_end: bool = False) -> int:
    state = load_state(schedule)
    stop_dt = slot_datetime(stop_at)
    write_status_file(
        state,
        schedule,
        stop_at,
        phase="waiting",
        note="Mini-PC scheduler is active.",
    )

    while True:
        now = datetime.now()
        if now >= stop_dt:
            break

        due_slots = get_due_pending_slots(state, schedule, now=now)
        if due_slots:
            trigger_slot = due_slots[-1]
            execute_scan(state, schedule, stop_at, trigger_slot, dry_run=dry_run)
            state = load_state(schedule)
            continue

        next_slot = get_next_pending_slot(state, schedule, now=now)
        if not next_slot:
            break

        next_slot_dt = slot_datetime(next_slot, now)
        sleep_until(min(next_slot_dt, stop_dt), state, schedule, stop_at)
        state = load_state(schedule)

    state = load_state(schedule)
    state["last_status"] = state.get("last_status", "idle") or "idle"
    save_state(state)
    write_status_file(
        state,
        schedule,
        stop_at,
        phase="completed",
        note=f"Schedule window ended at {stop_at}.",
    )
    logging.info("Mini-PC schedule finished for %s.", datetime.now().date().isoformat())
    if shutdown_at_end:
        maybe_shutdown_windows()
    return 0


class MiniPCMasterAvwapGUI(MasterAvwapGUI):
    def __init__(
        self,
        root,
        schedule: list[str],
        stop_at: str,
        dry_run: bool = False,
        shutdown_at_end: bool = False,
        auto_start: bool = True,
        dynamic_schedule: bool = False,
        dynamic_stop_at: bool = False,
    ):
        self.schedule = list(schedule)
        self.stop_at = stop_at
        self.dry_run = dry_run
        self.shutdown_at_end = shutdown_at_end
        self.dynamic_schedule = bool(dynamic_schedule)
        self.dynamic_stop_at = bool(dynamic_stop_at)
        self.dynamic_defaults_date = None
        self.scheduler_enabled = bool(auto_start)
        self.background_task_active = False
        self.current_background_label = ""
        self.scheduler_scan_active = False
        self.scheduler_active_slot = ""
        self.scheduler_phase = "starting"
        self.scheduler_note = "Launching mini-PC GUI scheduler."
        self.completed_window_date = None
        self.shutdown_issued_date = None
        self.scheduler_state = load_state(self.schedule)

        self.scheduler_summary_var = tk.StringVar(master=root, value="")
        self.scheduler_toggle_text = tk.StringVar(master=root, value="Pause Scheduler" if auto_start else "Resume Scheduler")

        super().__init__(root, standalone=True)
        self.root.title("Master AVWAP Mini PC Scheduler")
        self.root.geometry("1380x860")
        self.refresh_tracker_storage_summary()
        self.refresh_mini_pc_status_view()
        self._refresh_scheduler_panel(phase="waiting" if self.scheduler_enabled else "paused", note="Mini-PC GUI ready.")
        self.notebook.select(self.mini_pc_tab)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(750, self._scheduler_tick)

    def _refresh_dynamic_market_defaults(self, force: bool = False):
        today_iso = datetime.now().date().isoformat()
        if not force and self.dynamic_defaults_date == today_iso:
            return
        now = datetime.now()
        if self.dynamic_schedule:
            self.schedule = get_current_default_schedule(reference=now)
        if self.dynamic_stop_at:
            self.stop_at = get_current_default_stop_at(reference=now)
        self.dynamic_defaults_date = today_iso

    def _build_layout(self):
        super()._build_layout()

        mini_pc_tab = ttk.Frame(self.notebook)
        self.mini_pc_tab = mini_pc_tab
        self.notebook.insert(0, mini_pc_tab, text="Mini PC")

        toolbar = ttk.Frame(mini_pc_tab)
        toolbar.pack(fill="x", padx=10, pady=(8, 8))

        self.scheduler_toggle_button = ttk.Button(
            toolbar,
            textvariable=self.scheduler_toggle_text,
            command=self.toggle_scheduler,
        )
        self.scheduler_toggle_button.pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Run Home Folder Scan Now", command=self.run_shared_watchlist_scan_once).pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Refresh Status Preview", command=self.refresh_mini_pc_status_view).pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Change Home Folder", command=self.choose_tracker_storage_dir).pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Open Home Folder", command=self.open_tracker_storage_dir).pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Open Status Folder", command=self.open_status_folder).pack(side="left")

        ttk.Label(
            mini_pc_tab,
            textvariable=self.scheduler_summary_var,
            justify="left",
            wraplength=1220,
        ).pack(fill="x", padx=10, pady=(0, 8))

        ttk.Label(
            mini_pc_tab,
            text=(
                "This mini-PC view auto-runs the shared-folder AVWAP scan on the configured schedule. "
                "If you change the home folder, restart this app before the next scan so the new location is used."
            ),
            justify="left",
            wraplength=1220,
        ).pack(fill="x", padx=10, pady=(0, 8))

        preview_frame = ttk.LabelFrame(mini_pc_tab, text="Phone Status File Preview")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.mini_pc_status_text = tk.Text(preview_frame, wrap="word", font=("Courier New", 10))
        self.mini_pc_status_text.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        status_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.mini_pc_status_text.yview)
        self.mini_pc_status_text.configure(yscrollcommand=status_scroll.set)
        status_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self._style_mini_pc_text_widget()

    def _style_mini_pc_text_widget(self):
        self.mini_pc_status_text.configure(
            bg="#1F1F1F",
            fg="#F0F0F0",
            insertbackground="#F0F0F0",
            selectbackground="#4A4A4A",
            selectforeground="#F0F0F0",
            highlightbackground="#202020",
            highlightcolor="#3A3A3A",
        )

    def _refresh_active_tab(self):
        selected_tab = self.notebook.select()
        if selected_tab == str(self.mini_pc_tab):
            self.refresh_mini_pc_status_view()
            self._refresh_scheduler_panel()
            return
        super()._refresh_active_tab()

    def refresh_tracker_storage_summary(self):
        super().refresh_tracker_storage_summary()
        if hasattr(self, "scheduler_summary_var"):
            self._refresh_scheduler_panel()

    def refresh_mini_pc_status_view(self):
        if not hasattr(self, "mini_pc_status_text"):
            return
        text = read_text(STATUS_FILE)
        if not text:
            text = (
                "The status file does not exist yet.\n\n"
                "It will be created automatically after the scheduler starts waiting or after the first scan runs.\n\n"
                f"Expected path:\n{STATUS_FILE}"
            )
        self._set_text_widget_contents(self.mini_pc_status_text, text)

    def open_status_folder(self):
        self._open_folder_in_explorer(STATUS_FILE.parent)

    def toggle_scheduler(self):
        self.scheduler_enabled = not self.scheduler_enabled
        phase = "waiting" if self.scheduler_enabled else "paused"
        note = "Scheduler resumed from GUI." if self.scheduler_enabled else "Scheduler paused from GUI."
        self._refresh_scheduler_panel(phase=phase, note=note)
        write_status_file(load_state(self.schedule), self.schedule, self.stop_at, phase=phase, note=note, active_slot=self.scheduler_active_slot)
        self.refresh_mini_pc_status_view()
        self.status_var.set(note)

    def _refresh_scheduler_panel(self, phase: str | None = None, note: str | None = None):
        if phase is not None:
            self.scheduler_phase = phase
        if note is not None:
            self.scheduler_note = note

        self._refresh_dynamic_market_defaults()
        self.scheduler_state = load_state(self.schedule)
        now = datetime.now()
        next_slot = get_next_pending_slot(self.scheduler_state, self.schedule, now=now)
        self.scheduler_toggle_text.set("Pause Scheduler" if self.scheduler_enabled else "Resume Scheduler")
        market_session = get_market_session_window(reference=now)

        active_task = self.scheduler_active_slot or self.current_background_label or "None"
        shutdown_text = "enabled" if self.shutdown_at_end else "disabled"
        lines = [
            f"Scheduler phase: {self.scheduler_phase} | Auto scheduler: {'running' if self.scheduler_enabled else 'paused'} | Auto shutdown: {shutdown_text}",
            f"Local market session today: {market_session.session_label} | Today's schedule: {', '.join(self.schedule)} | Stop at: {self.stop_at}",
            f"Next pending slot: {next_slot or 'None'} | Active task: {active_task}",
            f"Last status: {self.scheduler_state.get('last_status') or 'idle'} | Last success: {self.scheduler_state.get('last_success_at') or 'None'}",
            f"Shared folder root: {LONGS_FILE.parent}",
            f"Status file: {STATUS_FILE}",
            f"Scheduler log: {SCHEDULER_LOG_FILE}",
        ]
        if self.scheduler_note:
            lines.append(f"Note: {self.scheduler_note}")
        self.scheduler_summary_var.set("\n".join(lines))

    def _write_scheduler_status(self, phase: str, note: str, active_slot: str = ""):
        self.scheduler_state = load_state(self.schedule)
        write_status_file(
            self.scheduler_state,
            self.schedule,
            self.stop_at,
            phase=phase,
            note=note,
            active_slot=active_slot,
        )
        self.refresh_mini_pc_status_view()
        self._refresh_scheduler_panel(phase=phase, note=note)

    def _run_background(self, target, running_msg, done_msg, done_callback=None):
        if self.background_task_active:
            self.status_var.set("Another background task is already running. Please wait for it to finish.")
            return

        self.background_task_active = True
        self.current_background_label = running_msg
        self.status_var.set(running_msg)
        self._write_scheduler_status("busy", running_msg)

        def _task():
            error_message = ""
            try:
                target()
            except Exception as exc:
                logging.exception("GUI background task failed")
                error_message = str(exc)

            def _finish():
                self.background_task_active = False
                self.current_background_label = ""
                if error_message:
                    final_note = f"Error: {error_message}"
                    self.status_var.set(final_note)
                else:
                    final_note = done_msg
                    self.status_var.set(done_msg)
                    self.refresh_table()
                    if done_callback:
                        done_callback()
                phase = "waiting" if self.scheduler_enabled else "paused"
                self._write_scheduler_status(phase, final_note)
                self._maybe_complete_schedule_window()

            self.root.after(0, _finish)

        threading.Thread(target=_task, daemon=True).start()

    def run_shared_watchlist_scan_once(self):
        self.notebook.select(self.avwap_tab)
        self._start_shared_scan(run_label="manual", scheduled_slot=None)

    def _start_shared_scan(self, run_label: str, scheduled_slot: str | None):
        if self.background_task_active:
            self.status_var.set("A scan is already running. Please wait for it to finish.")
            return

        started_at = datetime.now()
        run_id = started_at.strftime("%Y%m%d-%H%M%S")
        state = load_state(self.schedule)
        active_slot = scheduled_slot or "manual"
        update_setup_tracker = should_update_setup_tracker_for_slot(self.schedule, scheduled_slot)

        if scheduled_slot:
            mark_slots_skipped_before_trigger(state, self.schedule, scheduled_slot, started_at)
            save_state(state)
            running_note = f"Running scheduled slot {scheduled_slot} from the shared home folder."
            phase = "running"
            self.scheduler_scan_active = True
            self.scheduler_active_slot = scheduled_slot
        else:
            running_note = "Running a manual shared home-folder scan from the GUI."
            phase = "manual_run"
            self.scheduler_scan_active = False
            self.scheduler_active_slot = ""

        self.background_task_active = True
        self.current_background_label = running_note
        self.status_var.set(running_note)
        self._write_scheduler_status(phase, running_note, active_slot=active_slot)
        logging.info(
            "Starting GUI mini-PC scan. label=%s scheduled_slot=%s dry_run=%s update_setup_tracker=%s",
            run_label,
            scheduled_slot,
            self.dry_run,
            update_setup_tracker,
        )

        def _task():
            error_text = ""
            run_status = "dry_run" if self.dry_run else "success"
            filter_summary = None
            try:
                if self.dry_run:
                    logging.info("Dry-run enabled; skipping live Master AVWAP GUI mini-PC scan.")
                    filter_summary = {
                        "ran_at": datetime.now().isoformat(timespec="seconds"),
                        "status": "dry_run",
                        "message": "Dry-run mode skipped the watchlist filter.",
                    }
                else:
                    filter_summary = run_master_with_watchlist_filter(
                        update_setup_tracker=update_setup_tracker,
                    )
            except Exception as exc:
                run_status = "failed"
                error_text = str(exc)
                filter_summary = getattr(exc, "watchlist_filter_summary", filter_summary)
                logging.exception("GUI mini-PC scan failed. label=%s scheduled_slot=%s", run_label, scheduled_slot)

            finished_at = datetime.now()
            duration_seconds = round((finished_at - started_at).total_seconds(), 2)

            def _finish():
                state = load_state(self.schedule)
                run_record = {
                    "run_id": run_id,
                    "slot": scheduled_slot or "manual",
                    "status": run_status,
                    "started_at": started_at.isoformat(timespec="seconds"),
                    "finished_at": finished_at.isoformat(timespec="seconds"),
                    "duration_seconds": duration_seconds,
                }
                if error_text:
                    run_record["error"] = error_text
                state.setdefault("runs", []).append(run_record)
                state["last_filter_summary"] = filter_summary

                if scheduled_slot:
                    if run_status == "failed":
                        state["slots"][scheduled_slot] = {
                            "status": "failed",
                            "run_id": run_id,
                            "covered_at": finished_at.isoformat(timespec="seconds"),
                            "note": error_text,
                        }
                    else:
                        cover_slots_after_success(
                            state,
                            self.schedule,
                            scheduled_slot,
                            run_id,
                            finished_at,
                            dry_run=self.dry_run,
                        )

                if run_status == "failed":
                    state["last_status"] = "failed" if scheduled_slot else "manual_failed"
                    state["last_error"] = error_text
                    done_note = (
                        f"Scheduled slot {scheduled_slot} failed: {error_text}"
                        if scheduled_slot
                        else f"Manual shared-folder scan failed: {error_text}"
                    )
                else:
                    state["last_status"] = run_status if scheduled_slot else f"manual_{run_status}"
                    state["last_error"] = ""
                    state["last_success_at"] = finished_at.isoformat(timespec="seconds")
                    done_note = (
                        f"Scheduled slot {scheduled_slot} completed in {format_duration(duration_seconds)}."
                        if scheduled_slot
                        else f"Manual shared-folder scan completed in {format_duration(duration_seconds)}."
                    )

                save_state(state)
                self.scheduler_state = state
                self.background_task_active = False
                self.current_background_label = ""
                self.scheduler_scan_active = False
                self.scheduler_active_slot = ""

                self.refresh_table()
                self.refresh_avwap_output_view()
                self.refresh_anchor_output_view()
                self.refresh_setup_tracker_view()
                self.refresh_tracker_storage_summary()
                self.status_var.set(done_note)

                phase_after = "waiting" if self.scheduler_enabled else "paused"
                if not scheduled_slot:
                    phase_after = "idle" if not self.scheduler_enabled else "waiting"
                self._write_scheduler_status(phase_after, done_note)
                self._maybe_complete_schedule_window()

            self.root.after(0, _finish)

        threading.Thread(target=_task, daemon=True).start()

    def _maybe_complete_schedule_window(self):
        today_iso = datetime.now().date().isoformat()
        stop_dt = slot_datetime(self.stop_at)
        if self.background_task_active or datetime.now() < stop_dt:
            return
        if self.completed_window_date == today_iso:
            return

        state = load_state(self.schedule)
        state["last_status"] = state.get("last_status", "idle") or "idle"
        save_state(state)
        self.completed_window_date = today_iso
        self._write_scheduler_status("completed", f"Schedule window ended at {self.stop_at}.")
        logging.info("Mini-PC GUI schedule finished for %s.", today_iso)

        if self.shutdown_at_end and self.shutdown_issued_date != today_iso:
            self.shutdown_issued_date = today_iso
            self.status_var.set("Schedule window ended. Shutting down Windows...")
            self.root.after(1500, self._shutdown_host)

    def _shutdown_host(self):
        maybe_shutdown_windows()

    def _scheduler_tick(self):
        self._refresh_dynamic_market_defaults()
        self.scheduler_state = load_state(self.schedule)
        today_iso = datetime.now().date().isoformat()
        if self.completed_window_date and self.completed_window_date != today_iso:
            self.completed_window_date = None
        if self.shutdown_issued_date and self.shutdown_issued_date != today_iso:
            self.shutdown_issued_date = None

        if self.background_task_active:
            phase = "running" if self.scheduler_scan_active else "busy"
            note = self.current_background_label or "Background task is still running."
            self._write_scheduler_status(phase, note, active_slot=self.scheduler_active_slot)
            self.root.after(GUI_TICK_MS, self._scheduler_tick)
            return

        if not self.scheduler_enabled:
            self._write_scheduler_status("paused", "Scheduler is paused in the GUI.")
            self.root.after(GUI_TICK_MS, self._scheduler_tick)
            return

        stop_dt = slot_datetime(self.stop_at)
        if datetime.now() >= stop_dt:
            self._maybe_complete_schedule_window()
            self.root.after(GUI_TICK_MS, self._scheduler_tick)
            return

        due_slots = get_due_pending_slots(self.scheduler_state, self.schedule)
        if due_slots:
            self._start_shared_scan(run_label=due_slots[-1], scheduled_slot=due_slots[-1])
            self.root.after(GUI_TICK_MS, self._scheduler_tick)
            return

        next_slot = get_next_pending_slot(self.scheduler_state, self.schedule)
        waiting_note = (
            f"Waiting for next slot {next_slot}."
            if next_slot
            else "No pending slots remain for today."
        )
        self._write_scheduler_status("waiting", waiting_note)
        self.root.after(GUI_TICK_MS, self._scheduler_tick)

    def _on_close(self):
        if self.background_task_active:
            if messagebox and not messagebox.askyesno(
                "Exit Mini-PC Scheduler",
                "A background task is still running. Closing now will stop the scheduler window. Exit anyway?",
            ):
                return
        elif self.scheduler_enabled and messagebox and datetime.now() < slot_datetime(self.stop_at):
            if not messagebox.askyesno(
                "Exit Mini-PC Scheduler",
                "The auto scheduler is still active for today. Closing now will stop future scheduled scans. Exit anyway?",
            ):
                return
        self.root.destroy()


def launch_gui_app(
    schedule: list[str],
    stop_at: str,
    dry_run: bool = False,
    shutdown_at_end: bool = False,
    auto_start: bool = True,
    dynamic_schedule: bool = False,
    dynamic_stop_at: bool = False,
) -> int:
    if tk is None or ttk is None:
        logging.error("tkinter is unavailable in this Python environment; cannot launch the mini-PC GUI.")
        return 1

    root = tk.Tk()
    MiniPCMasterAvwapGUI(
        root,
        schedule=schedule,
        stop_at=stop_at,
        dry_run=dry_run,
        shutdown_at_end=shutdown_at_end,
        auto_start=auto_start,
        dynamic_schedule=dynamic_schedule,
        dynamic_stop_at=dynamic_stop_at,
    )
    root.mainloop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    current_session = get_market_session_window()
    current_schedule = ",".join(get_current_default_schedule())
    current_stop_at = get_current_default_stop_at()
    parser = argparse.ArgumentParser(description="Run the Master AVWAP mini-PC scheduler.")
    parser.add_argument(
        "--schedule",
        default=None,
        help=(
            "Comma-separated HH:MM times to run the scan. "
            "Default: auto-generated from today's local NYSE session "
            f"({current_session.session_label}), currently {current_schedule}."
        ),
    )
    parser.add_argument(
        "--stop-at",
        default=None,
        help=(
            "Time to stop the daily scheduler loop. "
            f"Default: 30 minutes after today's local close, currently {current_stop_at}."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one immediate shared-folder scan and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the scheduler/status flow without running the live scan.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the scheduler loop without launching the GUI.",
    )
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Launch the GUI without immediately starting the auto scheduler.",
    )
    parser.add_argument(
        "--shutdown-at-end",
        action="store_true",
        help="After the schedule window ends, issue a Windows shutdown command.",
    )
    parser.add_argument(
        "--reset-lock",
        action="store_true",
        help="Clear a stale mini-PC lock file before starting.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dynamic_schedule = not args.schedule
    dynamic_stop_at = not args.stop_at
    schedule = normalize_schedule(args.schedule) if args.schedule else get_current_default_schedule()
    stop_at = parse_clock(args.stop_at).strftime("%H:%M") if args.stop_at else get_current_default_stop_at()

    attach_scheduler_log_handler()
    try:
        acquire_lock(force_reset=args.reset_lock)
    except RuntimeError as exc:
        logging.error(str(exc))
        if not args.headless and tk is not None and messagebox is not None:
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Mini-PC Scheduler", str(exc))
                root.destroy()
            except Exception:
                pass
        return 1

    logging.info(
        "Mini-PC runner starting. once=%s dry_run=%s headless=%s schedule=%s stop_at=%s shared_root=%s",
        args.once,
        args.dry_run,
        args.headless,
        ",".join(schedule),
        stop_at,
        LONGS_FILE.parent,
    )

    if args.once:
        return run_once(schedule, stop_at, dry_run=args.dry_run)

    if args.headless:
        return run_schedule(
            schedule,
            stop_at,
            dry_run=args.dry_run,
            shutdown_at_end=args.shutdown_at_end,
        )

    return launch_gui_app(
        schedule,
        stop_at,
        dry_run=args.dry_run,
        shutdown_at_end=args.shutdown_at_end,
        auto_start=not args.no_autostart,
        dynamic_schedule=dynamic_schedule,
        dynamic_stop_at=dynamic_stop_at,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        release_lock()
