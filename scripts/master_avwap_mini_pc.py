#!/usr/bin/env python3
"""Headless Master AVWAP scheduler for an always-on Windows mini PC."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, time as dt_time, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_avwap import run_master
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

DEFAULT_SCHEDULE = ("07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00")
DEFAULT_STOP_TIME = "13:30"
STATUS_FILE = LONGS_FILE.parent / "master_avwap_mini_pc_status.txt"
STATE_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_mini_pc_state.json"
LOCK_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_mini_pc.lock"
SCHEDULER_LOG_FILE = LOG_DIR / "master_avwap_mini_pc.log"
APP_LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s]: %(message)s"
STATUS_PREVIEW_RUNS = 8
SLEEP_POLL_SECONDS = 30
STALE_LOCK_MAX_AGE = timedelta(hours=12)

_LOCK_ACQUIRED = False


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

    lines = [
        "Master AVWAP Mini PC Status",
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
        f"Setup tracker: {MASTER_AVWAP_SETUP_TRACKER_FILE}",
        f"Priority setups report: {MASTER_AVWAP_PRIORITY_SETUPS_FILE}",
        f"Ticker buckets report: {MASTER_AVWAP_EVENT_TICKERS_FILE}",
        f"TradingView report: {MASTER_AVWAP_TRADINGVIEW_REPORT_FILE}",
        f"Main AVWAP report: {MASTER_AVWAP_REPORT_FILE}",
    ]

    if note:
        lines.extend(["", f"Note: {note}"])

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
        started_at = existing.get("started_at") if isinstance(existing, dict) else None
        try:
            started_dt = datetime.fromisoformat(str(started_at))
        except (TypeError, ValueError):
            started_dt = None
        if started_dt and (datetime.now() - started_dt) > STALE_LOCK_MAX_AGE:
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

    logging.info("Starting mini-PC Master AVWAP run for slot %s.", trigger_slot)
    error_text = ""
    status = "success"
    try:
        if dry_run:
            logging.info("Dry-run enabled; skipping live Master AVWAP scan for slot %s.", trigger_slot)
        else:
            run_master(use_shared_watchlists=True)
    except Exception as exc:
        status = "failed"
        error_text = str(exc)
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
    write_status_file(
        state,
        schedule,
        stop_at,
        phase="running",
        note="Running one immediate mini-PC scan.",
        active_slot="manual",
    )
    logging.info("Starting one immediate mini-PC Master AVWAP scan.")
    started_at = datetime.now()
    error_text = ""
    try:
        if dry_run:
            logging.info("Dry-run enabled; skipping live Master AVWAP scan.")
        else:
            run_master(use_shared_watchlists=True)
        status = "success" if not dry_run else "dry_run"
    except Exception as exc:
        status = "failed"
        error_text = str(exc)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the headless Master AVWAP mini-PC scheduler.")
    parser.add_argument(
        "--schedule",
        default=",".join(DEFAULT_SCHEDULE),
        help="Comma-separated HH:MM times to run the scan. Default: 07:00 through 13:00 hourly.",
    )
    parser.add_argument(
        "--stop-at",
        default=DEFAULT_STOP_TIME,
        help="Time to stop the daily scheduler loop. Default: 13:30.",
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

    schedule = normalize_schedule(args.schedule)
    stop_at = parse_clock(args.stop_at).strftime("%H:%M")

    attach_scheduler_log_handler()
    acquire_lock(force_reset=args.reset_lock)

    logging.info(
        "Mini-PC runner starting. once=%s dry_run=%s schedule=%s stop_at=%s shared_root=%s",
        args.once,
        args.dry_run,
        ",".join(schedule),
        stop_at,
        LONGS_FILE.parent,
    )

    if args.once:
        return run_once(schedule, stop_at, dry_run=args.dry_run)

    return run_schedule(
        schedule,
        stop_at,
        dry_run=args.dry_run,
        shutdown_at_end=args.shutdown_at_end,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        release_lock()
