#!/usr/bin/env python3
"""Shared Master AVWAP output helpers for BounceBot and Mini PC flows."""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from project_paths import (
    AVWAP_SIGNALS_FILE,
    MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE,
    MASTER_AVWAP_D1_WATCHLIST_FILE,
    MASTER_AVWAP_FOCUS_FILE,
    MASTER_AVWAP_TRADINGVIEW_REPORT_FILE,
)

VALID_SIDES = ("LONG", "SHORT")


def _parse_iso_date_safe(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        text = str(value).strip()
        return datetime.fromisoformat(text).date() if text else None
    except Exception:
        return None


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logging.warning("Failed reading JSON from %s: %s", path, exc)
        return default


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logging.warning("Failed reading text from %s: %s", path, exc)
        return ""


def _extract_symbols_from_text(text: str) -> list[str]:
    symbols = []
    seen = set()
    for raw_value in str(text or "").split(","):
        symbol = raw_value.strip().upper()
        if not symbol or symbol == "NONE" or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_side(value: Any) -> str:
    side = str(value or "").strip().upper()
    return side if side in VALID_SIDES else ""


def _direction_from_side(side: Any) -> str:
    normalized = _normalize_side(side)
    if normalized == "LONG":
        return "long"
    if normalized == "SHORT":
        return "short"
    return ""


def _clean_string_list(values: Any, *, uppercase: bool = False) -> list[str]:
    if isinstance(values, str):
        raw_values = re.split(r"[;,]", values)
    elif isinstance(values, (list, tuple, set)):
        raw_values = values
    else:
        raw_values = []

    cleaned = []
    seen = set()
    for value in raw_values:
        text = str(value or "").strip()
        if not text:
            continue
        if uppercase:
            text = text.upper()
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _normalize_d1_trigger_levels(values: Any, side: Any = "") -> list[dict[str, Any]]:
    if isinstance(values, dict):
        raw_values = values.values()
    elif isinstance(values, (list, tuple)):
        raw_values = values
    else:
        raw_values = []

    normalized_side = _normalize_side(side)
    default_action = "break_below" if normalized_side == "SHORT" else "break_above"
    trigger_levels: list[dict[str, Any]] = []
    seen = set()
    for raw_entry in raw_values:
        if not isinstance(raw_entry, dict):
            continue
        level = _coerce_float(raw_entry.get("level") or raw_entry.get("price"))
        if level is None:
            continue
        label = str(raw_entry.get("label") or raw_entry.get("level_label") or "").strip().upper()
        if not label:
            label = "LEVEL"
        action = str(raw_entry.get("action") or raw_entry.get("trigger_action") or default_action).strip().lower()
        if action not in {"break_above", "break_below"}:
            action = default_action
        event_type = str(raw_entry.get("event_type") or "preloaded_level_break").strip().lower()
        trigger_id = str(raw_entry.get("trigger_id") or "").strip()
        if not trigger_id:
            trigger_id = f"{event_type}:{label}:{round(float(level), 4):.4f}"
        key = (trigger_id, label, round(float(level), 4), action)
        if key in seen:
            continue
        seen.add(key)
        trigger_levels.append(
            {
                "schema_version": raw_entry.get("schema_version"),
                "trigger_id": trigger_id,
                "side": _normalize_side(raw_entry.get("side") or normalized_side),
                "action": action,
                "event_type": event_type,
                "label": label,
                "alert_label": str(raw_entry.get("alert_label") or label).strip(),
                "level": float(level),
                "reason": str(raw_entry.get("reason") or "").strip(),
                "source": str(raw_entry.get("source") or "").strip(),
                "armed_at": str(raw_entry.get("armed_at") or "").strip(),
                "armed_price": _coerce_float(raw_entry.get("armed_price")),
                "anchor_type": str(raw_entry.get("anchor_type") or "").strip(),
                "anchor_date": str(raw_entry.get("anchor_date") or "").strip(),
                "priority_bucket": str(raw_entry.get("priority_bucket") or "").strip(),
                "setup_family": str(raw_entry.get("setup_family") or "").strip(),
                "target_tier": str(raw_entry.get("target_tier") or "").strip(),
                "upgrade_only": bool(raw_entry.get("upgrade_only")),
            }
        )
    return trigger_levels


def _normalize_bucket_upgrade_events(values: Any, side: Any = "") -> list[dict[str, Any]]:
    if isinstance(values, dict):
        raw_values = values.values()
    elif isinstance(values, (list, tuple)):
        raw_values = values
    else:
        raw_values = []

    normalized_side = _normalize_side(side)
    events: list[dict[str, Any]] = []
    seen = set()
    for raw_entry in raw_values:
        if not isinstance(raw_entry, dict):
            continue
        symbol = str(raw_entry.get("symbol") or "").strip().upper()
        event_side = _normalize_side(raw_entry.get("side") or normalized_side)
        if not event_side:
            continue
        current_bucket = str(raw_entry.get("priority_bucket") or raw_entry.get("bucket") or "").strip().lower()
        previous_bucket = str(raw_entry.get("previous_bucket") or "").strip().lower()
        event_type = str(raw_entry.get("event_type") or "bucket_upgrade").strip().lower()
        key = (symbol, event_side, event_type, previous_bucket, current_bucket)
        if key in seen:
            continue
        seen.add(key)
        label = str(raw_entry.get("label") or "").strip()
        if not label:
            bucket_label = current_bucket.replace("_", " ").strip().title() or "Bucket"
            label = f"{bucket_label} upgrade"
        reason = str(raw_entry.get("reason") or "").strip()
        if not reason:
            reason = f"{previous_bucket or 'missing'} -> {current_bucket or 'unbucketed'}"
        events.append(
            {
                "symbol": symbol,
                "side": event_side,
                "event_type": event_type,
                "label": label,
                "reason": reason,
                "source": str(raw_entry.get("source") or "bucket_upgrade").strip() or "bucket_upgrade",
                "previous_bucket": previous_bucket,
                "priority_bucket": current_bucket,
                "bucket": current_bucket,
                "priority_score": _coerce_float(raw_entry.get("priority_score")),
                "setup_family": str(raw_entry.get("setup_family") or "").strip(),
                "trade_date": str(raw_entry.get("trade_date") or raw_entry.get("last_trade_date") or "").strip(),
                "bucket_upgrade": True,
            }
        )
    return events


def _empty_focus_groups(
    source: str = "none",
    source_label: str = "No focus output yet",
) -> dict[str, Any]:
    return {
        "favorites": {"LONG": [], "SHORT": []},
        "near_favorite_zones": {"LONG": [], "SHORT": []},
        "source": source,
        "source_label": source_label,
    }


def _append_focus_symbol(groups: dict[str, Any], section: str, side: str, symbol: str) -> None:
    section_groups = groups.get(section)
    if not isinstance(section_groups, dict):
        return
    target = section_groups.get(side)
    if not isinstance(target, list):
        return

    cleaned_symbol = str(symbol or "").strip().upper()
    if not cleaned_symbol or cleaned_symbol in target:
        return
    target.append(cleaned_symbol)


def normalize_master_avwap_event_row(row: dict[str, Any]) -> dict[str, Any] | None:
    signal_type = str(row.get("signal_type") or "").strip().upper()
    if not signal_type or "BETWEEN" in signal_type:
        return None
    if not (signal_type.startswith("CROSS") or signal_type.startswith("BOUNCE")):
        return None

    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol:
        return None

    trade_date = _parse_iso_date_safe(row.get("trade_date"))
    if trade_date is None:
        return None

    level = signal_type
    for prefix in ("CROSS_UP_", "CROSS_DOWN_", "BOUNCE_"):
        if signal_type.startswith(prefix) and len(signal_type) > len(prefix):
            level = signal_type[len(prefix):]
            break

    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "signal_type": signal_type,
        "anchor_type": str(row.get("anchor_type") or "").strip().upper() or "UNKNOWN",
        "anchor_date": str(row.get("anchor_date") or "").strip(),
        "side": str(row.get("side") or "").strip().upper(),
        "level": level,
        "priority_bucket": str(row.get("priority_bucket") or "").strip().lower(),
        "favorite_zone": str(row.get("favorite_zone") or "").strip(),
        "favorite_signals": [
            value.strip().upper()
            for value in str(row.get("favorite_signals") or "").split(";")
            if value.strip()
        ],
        "is_favorite_setup": str(row.get("is_favorite_setup") or "").strip().lower() in ("1", "true", "yes"),
        "is_near_favorite_zone": str(row.get("is_near_favorite_zone") or "").strip().lower() in ("1", "true", "yes"),
    }


def load_master_avwap_events_for_date(
    trade_date: date | None = None,
    signals_path: Path = AVWAP_SIGNALS_FILE,
) -> dict[str, list[dict[str, Any]]]:
    if not signals_path.exists():
        return {}

    try:
        with signals_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        logging.warning("Failed reading Master AVWAP signals file %s: %s", signals_path, exc)
        return {}

    target_date = trade_date or datetime.now().date()
    events_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        normalized = normalize_master_avwap_event_row(row)
        if not normalized or normalized["trade_date"] != target_date:
            continue
        events_by_symbol.setdefault(normalized["symbol"], []).append(normalized)
    return events_by_symbol


def build_master_avwap_active_level_map(
    events_by_symbol: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    active_levels = {}
    for symbol, events in events_by_symbol.items():
        levels = {
            event.get("level") or event.get("signal_type")
            for event in events
            if str(event.get("signal_type") or "").startswith(("CROSS", "BOUNCE"))
        }
        cleaned_levels = sorted(str(level) for level in levels if level)
        if cleaned_levels:
            active_levels[symbol] = cleaned_levels
    return active_levels


def build_master_avwap_second_stdev_cross_map(
    events_by_symbol: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    cross_map = {}
    for symbol, events in events_by_symbol.items():
        matching_events = [
            event
            for event in events
            if event.get("anchor_type") == "CURRENT"
            and event.get("signal_type") in {"CROSS_UP_UPPER_2", "CROSS_DOWN_LOWER_2"}
        ]
        if not matching_events:
            continue

        latest_event = sorted(
            matching_events,
            key=lambda event: (
                str(event.get("trade_date") or ""),
                str(event.get("anchor_date") or ""),
                str(event.get("signal_type") or ""),
            ),
        )[-1]
        cross_map[symbol] = {
            "symbol": symbol,
            "side": str(latest_event.get("side") or "").strip().upper(),
            "signal_type": str(latest_event.get("signal_type") or "").strip().upper(),
            "level": str(latest_event.get("level") or "").strip().upper(),
            "anchor_type": str(latest_event.get("anchor_type") or "").strip().upper(),
            "anchor_date": str(latest_event.get("anchor_date") or "").strip(),
        }
    return cross_map


def load_master_avwap_focus_map(
    focus_path: Path = MASTER_AVWAP_FOCUS_FILE,
) -> dict[str, dict[str, Any]]:
    payload = _read_json(focus_path, default={})
    if not isinstance(payload, dict):
        return {}

    raw_symbols = payload.get("symbols", {})
    if not isinstance(raw_symbols, dict):
        return {}

    focus_map = {}
    for raw_symbol, raw_entry in raw_symbols.items():
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        symbol = str(entry.get("symbol") or raw_symbol or "").strip().upper()
        if not symbol:
            continue
        focus_map[symbol] = {
            "symbol": symbol,
            "side": _normalize_side(entry.get("side")),
            "priority_bucket": str(entry.get("priority_bucket") or "").strip().lower(),
            "setup_family": str(entry.get("setup_family") or "").strip(),
            "priority_rank": entry.get("priority_rank"),
            "priority_score": _coerce_float(entry.get("priority_score")),
            "preferred_swing_focus": bool(entry.get("preferred_swing_focus")),
            "favorite_zone": str(entry.get("favorite_zone") or "").strip(),
            "favorite_signals": _clean_string_list(entry.get("favorite_signals"), uppercase=True),
            "favorite_context_signals": _clean_string_list(
                entry.get("favorite_context_signals"),
                uppercase=True,
            ),
            "breakout_5d": bool(entry.get("breakout_5d")),
            "retest_followthrough": bool(entry.get("retest_followthrough")),
            "retest_reference_level": str(entry.get("retest_reference_level") or "").strip(),
            "retest_note": str(entry.get("retest_note") or "").strip(),
            "previous_day_date": str(entry.get("previous_day_date") or "").strip(),
            "previous_day_high": _coerce_float(entry.get("previous_day_high")),
            "previous_day_low": _coerce_float(entry.get("previous_day_low")),
            "previous_day_range_break": bool(entry.get("previous_day_range_break")),
            "previous_day_range_note": str(entry.get("previous_day_range_note") or "").strip(),
            "extreme_move_watch": bool(entry.get("extreme_move_watch")),
            "extreme_move_favorite_ready": bool(entry.get("extreme_move_favorite_ready")),
            "extreme_move_note": str(entry.get("extreme_move_note") or "").strip(),
            "post_earnings_active": bool(entry.get("post_earnings_active")),
            "post_earnings_break_intraday": bool(entry.get("post_earnings_break_intraday")),
            "post_earnings_break_close": bool(entry.get("post_earnings_break_close")),
            "mid_earnings_ema15_trigger": bool(entry.get("mid_earnings_ema15_trigger")),
            "mid_earnings_ema21_trigger": bool(entry.get("mid_earnings_ema21_trigger")),
            "mid_earnings_first_dev_trigger": bool(entry.get("mid_earnings_first_dev_trigger")),
            "mid_earnings_note": str(entry.get("mid_earnings_note") or "").strip(),
            "trendline_break_recent": bool(entry.get("trendline_break_recent")),
            "trendline_break_note": str(entry.get("trendline_break_note") or "").strip(),
            "trendline_note": str(entry.get("trendline_note") or "").strip(),
            "last_trade_date": str(entry.get("last_trade_date") or "").strip(),
            "mid_earnings_watch": bool(entry.get("mid_earnings_watch")),
            "mid_earnings_active_second_stdev_hold": bool(
                entry.get("mid_earnings_active_second_stdev_hold")
            ),
            "mid_earnings_sessions_since_gap": entry.get("mid_earnings_sessions_since_gap"),
            "mid_earnings_zone_streak_days": entry.get("mid_earnings_zone_streak_days"),
            "sma_breakout_watch": bool(entry.get("sma_breakout_watch")),
            "sma_breakout_confirmed": bool(entry.get("sma_breakout_confirmed")),
            "sma_breakout_higher_high_confirmed": bool(entry.get("sma_breakout_higher_high_confirmed")),
            "sma_breakout_previous_day_high_break": bool(entry.get("sma_breakout_previous_day_high_break")),
            "sma_breakout_confirmation_trigger": str(entry.get("sma_breakout_confirmation_trigger") or "").strip(),
            "sma_breakout_sma_label": str(entry.get("sma_breakout_sma_label") or "").strip(),
            "sma_breakout_retest_level": str(entry.get("sma_breakout_retest_level") or "").strip(),
            "top_pattern_watch": bool(entry.get("top_pattern_watch")),
            "top_pattern_entry": bool(entry.get("top_pattern_entry")),
            "top_pattern_entry_trigger": str(entry.get("top_pattern_entry_trigger") or "").strip(),
            "top_pattern_weekly_return_13w_pct": entry.get("top_pattern_weekly_return_13w_pct"),
            "top_pattern_weekly_return_26w_pct": entry.get("top_pattern_weekly_return_26w_pct"),
            "top_pattern_note": str(entry.get("top_pattern_note") or "").strip(),
        }
    return focus_map


def describe_master_avwap_focus(focus_entry: dict[str, Any] | None) -> str:
    bucket = str((focus_entry or {}).get("priority_bucket") or "").strip().lower()
    setup_family = str((focus_entry or {}).get("setup_family") or "").strip().lower()
    if setup_family in {"top_pattern", "top_pattern_tracking"} or (focus_entry or {}).get("top_pattern_watch"):
        if (focus_entry or {}).get("top_pattern_entry"):
            trigger = str((focus_entry or {}).get("top_pattern_entry_trigger") or "").replace("_", " ").strip()
            return f"TOP long entry {trigger}".strip()
        return "TOP long leader, H1 15EMA entry watch"
    if setup_family == "sma_breakout":
        if (focus_entry or {}).get("sma_breakout_previous_day_high_break"):
            return "SMA breakout setup, prior-day high break"
        return "SMA breakout setup"
    if setup_family == "post_earnings_52w_break":
        return "post-earnings 52w high break"
    if setup_family == "post_earnings_avwap_bounce":
        return "post-earnings AVWAPE bounce"
    if setup_family == "avwap_retest_followthrough":
        return "AVWAPE pullback bounce"
    if setup_family == "mid_earnings_ema15_retest":
        return "2nd stdev runner 15EMA retest"
    if setup_family == "mid_earnings_1stdev_retest":
        return "2nd stdev runner 1st-dev retest"
    if setup_family == "extreme_move_retest":
        return "2nd stdev runner 15EMA/1st-dev retest"
    if bucket == "sma_breakout_tracking" or setup_family == "sma_breakout_retest_tracking":
        sma_label = str((focus_entry or {}).get("sma_breakout_sma_label") or "").strip()
        retest_label = str((focus_entry or {}).get("sma_breakout_retest_level") or "").strip()
        detail = " ".join(part for part in (sma_label, retest_label) if part)
        return f"SMA breakout retest tracker {detail}".strip()
    if bucket == "favorite_setup":
        return "best current favorite setup"
    if bucket == "near_favorite_zone":
        return "near favorite zone"
    if bucket == "stdev_retest_tracking":
        if setup_family == "mid_earnings_above_2nd_stdev" or (focus_entry or {}).get(
            "mid_earnings_active_second_stdev_hold"
        ):
            return "mid-earnings 2nd stdev H1 bounce tracker"
        return "2nd/3rd stdev retest tracker"
    return "master avwap focus"


def describe_master_avwap_second_stdev_cross(cross_entry: dict[str, Any] | None) -> str:
    signal_type = str((cross_entry or {}).get("signal_type") or "").strip().upper()
    if signal_type == "CROSS_UP_UPPER_2":
        return "current UPPER_2 cross"
    if signal_type == "CROSS_DOWN_LOWER_2":
        return "current LOWER_2 cross"
    return "current 2nd stdev cross"


def load_master_avwap_d1_watchlist(
    watchlist_path: Path = MASTER_AVWAP_D1_WATCHLIST_FILE,
) -> dict[str, dict[str, Any]]:
    payload = _read_json(watchlist_path, default={})
    if not isinstance(payload, dict):
        return {}

    raw_symbols = payload.get("symbols", {})
    if not isinstance(raw_symbols, dict):
        return {}

    generated_at = str(payload.get("generated_at") or "").strip()
    run_date = str(payload.get("run_date") or "").strip()
    watchlist = {}
    for raw_symbol, raw_entry in raw_symbols.items():
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        symbol = str(entry.get("symbol") or raw_symbol or "").strip().upper()
        if not symbol:
            continue

        theta = entry.get("theta")
        theta = theta if isinstance(theta, dict) else {}
        watchlist[symbol] = {
            "symbol": symbol,
            "side": _normalize_side(entry.get("side")),
            "direction": _direction_from_side(entry.get("side")),
            "first_seen": str(entry.get("first_seen") or "").strip(),
            "last_seen": str(entry.get("last_seen") or "").strip(),
            "active_current_scan": bool(entry.get("active_current_scan")),
            "priority_bucket": str(entry.get("priority_bucket") or "").strip().lower(),
            "priority_score": _coerce_float(entry.get("priority_score")),
            "setup_family": str(entry.get("setup_family") or "").strip(),
            "favorite_zone": str(entry.get("favorite_zone") or "").strip(),
            "watch_reasons": _clean_string_list(entry.get("watch_reasons")),
            "reason_summary": str(entry.get("reason_summary") or "").strip(),
            "last_close": _coerce_float(entry.get("last_close")),
            "atr20": _coerce_float(entry.get("atr20")),
            "ema15": _coerce_float(entry.get("ema15") or entry.get("ema_15")),
            "ema15_distance_atr": _coerce_float(entry.get("ema15_distance_atr")),
            "previous_day_high": _coerce_float(entry.get("previous_day_high")),
            "previous_day_low": _coerce_float(entry.get("previous_day_low")),
            "retest_reference_level": str(entry.get("retest_reference_level") or "").strip(),
            "trendline_break_recent": bool(entry.get("trendline_break_recent")),
            "trendline_break_note": str(entry.get("trendline_break_note") or "").strip(),
            "post_earnings_active": bool(entry.get("post_earnings_active")),
            "post_earnings_monitor_level": _coerce_float(entry.get("post_earnings_monitor_level")),
            "post_earnings_monitor_level_label": str(entry.get("post_earnings_monitor_level_label") or "").strip(),
            "post_earnings_break_intraday": bool(entry.get("post_earnings_break_intraday")),
            "post_earnings_note": str(entry.get("post_earnings_note") or "").strip(),
            "trigger_levels": _normalize_d1_trigger_levels(entry.get("trigger_levels"), entry.get("side")),
            "upgrade_targets": _normalize_d1_trigger_levels(
                entry.get("upgrade_targets") or entry.get("trigger_levels"),
                entry.get("side"),
            ),
            "upgrade_summary": str(entry.get("upgrade_summary") or "").strip(),
            "trigger_summary": str(entry.get("trigger_summary") or "").strip(),
            "watchlist_generated_at": generated_at,
            "watchlist_run_date": run_date,
            "theta": {
                "play_type": str(theta.get("play_type") or "").strip(),
                "status": str(theta.get("status") or "").strip().lower(),
                "credit": _coerce_float(theta.get("credit")),
                "strike": _coerce_float(theta.get("strike")),
                "short_strike": _coerce_float(theta.get("short_strike")),
                "long_strike": _coerce_float(theta.get("long_strike")),
                "expiration": str(theta.get("expiration") or "").strip(),
                "credit_width_ratio": _coerce_float(theta.get("credit_width_ratio")),
                "contracts_needed": theta.get("contracts_needed"),
                "support_summary": str(theta.get("support_summary") or "").strip(),
            },
        }
    return watchlist


def load_master_avwap_d1_upgrade_alerts(
    alerts_path: Path = MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE,
) -> dict[str, dict[str, Any]]:
    payload = _read_json(alerts_path, default={})
    if not isinstance(payload, dict):
        return {}

    raw_symbols = payload.get("symbols", {})
    if not isinstance(raw_symbols, dict):
        return {}

    generated_at = str(payload.get("generated_at") or "").strip()
    run_date = str(payload.get("run_date") or "").strip()
    alerts: dict[str, dict[str, Any]] = {}
    for raw_symbol, raw_entry in raw_symbols.items():
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        symbol = str(entry.get("symbol") or raw_symbol or "").strip().upper()
        if not symbol:
            continue
        upgrade_targets = _normalize_d1_trigger_levels(
            entry.get("upgrade_targets"),
            entry.get("side"),
        )
        bucket_upgrade_events = _normalize_bucket_upgrade_events(
            entry.get("bucket_upgrade_events") or entry.get("upgrade_events"),
            entry.get("side"),
        )
        for event in bucket_upgrade_events:
            event["symbol"] = event.get("symbol") or symbol
            if not event.get("trade_date"):
                event["trade_date"] = str(entry.get("last_trade_date") or run_date).strip()
            if event.get("priority_score") is None:
                event["priority_score"] = _coerce_float(entry.get("priority_score"))
            if not event.get("setup_family"):
                event["setup_family"] = str(entry.get("setup_family") or "").strip()
        if not upgrade_targets and not bucket_upgrade_events:
            continue
        reason_summary = "D1 bucket upgrade" if bucket_upgrade_events else "A/S upgrade target"
        alerts[symbol] = {
            "symbol": symbol,
            "side": _normalize_side(entry.get("side")),
            "direction": _direction_from_side(entry.get("side")),
            "first_seen": run_date,
            "last_seen": run_date,
            "last_trade_date": str(entry.get("last_trade_date") or run_date).strip(),
            "active_current_scan": True,
            "priority_bucket": str(entry.get("priority_bucket") or "").strip().lower(),
            "priority_score": _coerce_float(entry.get("priority_score")),
            "setup_family": str(entry.get("setup_family") or "").strip(),
            "favorite_zone": str(entry.get("favorite_zone") or "").strip(),
            "current_band_zone": str(entry.get("current_band_zone") or "").strip(),
            "watch_reasons": ["bucket_upgrade"] if bucket_upgrade_events else ["a_s_upgrade_target"],
            "reason_summary": reason_summary,
            "upgrade_targets": upgrade_targets,
            "trigger_levels": [dict(item) for item in upgrade_targets],
            "bucket_upgrade_events": bucket_upgrade_events,
            "upgrade_summary": str(entry.get("upgrade_summary") or "").strip(),
            "watchlist_generated_at": generated_at,
            "watchlist_run_date": run_date,
        }
    return alerts


def _event_side_from_signal(event: dict[str, Any]) -> str:
    side = _normalize_side(event.get("side"))
    if side:
        return side
    signal_type = str(event.get("signal_type") or "").strip().upper()
    if signal_type.startswith("CROSS_UP") or "UPPER" in signal_type:
        return "LONG"
    if signal_type.startswith("CROSS_DOWN") or "LOWER" in signal_type:
        return "SHORT"
    return ""


def _signal_flag_details(event: dict[str, Any]) -> tuple[str, str, int] | None:
    signal_type = str(event.get("signal_type") or "").strip().upper()
    level = str(event.get("level") or "").strip().upper()
    is_cross = signal_type.startswith("CROSS")
    is_bounce = signal_type.startswith("BOUNCE")
    if not (is_cross or is_bounce):
        return None

    if level == "VWAP":
        label = "AVWAPE breakthrough" if is_cross else "AVWAPE D1 bounce"
        return ("avwap_breakthrough" if is_cross else "avwap_bounce", label, 10 if is_cross else 16)
    if level in {"UPPER_1", "LOWER_1"}:
        label = "1st-dev breakthrough" if is_cross else "1st-dev D1 bounce"
        return ("first_dev_breakthrough" if is_cross else "first_dev_bounce", label, 11 if is_cross else 15)
    if level in {"UPPER_2", "LOWER_2"}:
        label = "2nd-dev breakthrough" if is_cross else "2nd-dev D1 bounce"
        return ("second_dev_breakthrough" if is_cross else "second_dev_bounce", label, 18 if is_cross else 22)
    if level:
        label = f"{level.replace('_', ' ')} {'breakthrough' if is_cross else 'D1 bounce'}"
        return ("avwap_band_breakthrough" if is_cross else "avwap_band_bounce", label, 25)
    return None


def _add_d1_flag_event(
    events: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
    *,
    symbol: str,
    side: str,
    event_type: str,
    label: str,
    reason: str,
    sort_rank: int,
    source: str,
    priority_score: float | None = None,
    trade_date: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    cleaned_symbol = str(symbol or "").strip().upper()
    cleaned_side = _normalize_side(side)
    if not cleaned_symbol or cleaned_side not in VALID_SIDES:
        return
    reason_text = str(reason or "").strip()
    key = (
        cleaned_symbol,
        cleaned_side,
        str(event_type or "").strip().lower(),
        str(label or "").strip().lower(),
        reason_text.lower(),
    )
    if key in seen:
        return
    seen.add(key)

    parsed_trade_date = _parse_iso_date_safe(trade_date)
    event_payload = {
        "symbol": cleaned_symbol,
        "side": cleaned_side,
        "direction": _direction_from_side(cleaned_side),
        "event_type": str(event_type or "").strip(),
        "label": str(label or "").strip(),
        "reason": reason_text,
        "sort_rank": int(sort_rank or 50),
        "source": str(source or "").strip(),
        "priority_score": _coerce_float(priority_score),
        "trade_date": parsed_trade_date.isoformat() if parsed_trade_date else str(trade_date or "").strip(),
    }
    if extra:
        event_payload.update(extra)
    events.append(event_payload)


def _add_bucket_upgrade_d1_flags(
    events: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
    symbol: str,
    entry: dict[str, Any],
) -> None:
    fallback_side = entry.get("side")
    fallback_score = _coerce_float(entry.get("priority_score"))
    fallback_trade_date = entry.get("last_trade_date") or entry.get("last_seen")
    for upgrade_event in entry.get("bucket_upgrade_events") or []:
        if not isinstance(upgrade_event, dict):
            continue
        previous_bucket = str(upgrade_event.get("previous_bucket") or entry.get("previous_bucket") or "").strip().lower()
        current_bucket = str(
            upgrade_event.get("priority_bucket")
            or upgrade_event.get("bucket")
            or entry.get("priority_bucket")
            or ""
        ).strip().lower()
        reason = str(upgrade_event.get("reason") or "").strip()
        if not reason:
            reason = f"{previous_bucket or 'missing'} -> {current_bucket or 'unbucketed'}"
        _add_d1_flag_event(
            events,
            seen,
            symbol=symbol,
            side=upgrade_event.get("side") or fallback_side,
            event_type=str(upgrade_event.get("event_type") or "bucket_upgrade").strip(),
            label=str(upgrade_event.get("label") or "D1 bucket upgrade").strip(),
            reason=reason,
            sort_rank=5,
            source="bucket_upgrade",
            priority_score=(
                _coerce_float(upgrade_event.get("priority_score"))
                if upgrade_event.get("priority_score") is not None
                else fallback_score
            ),
            trade_date=upgrade_event.get("trade_date") or fallback_trade_date,
            extra={
                "bucket_upgrade": True,
                "previous_bucket": previous_bucket,
                "priority_bucket": current_bucket,
            },
        )


def _add_focus_like_d1_flags(
    events: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
    symbol: str,
    entry: dict[str, Any],
    *,
    source: str,
) -> None:
    side = entry.get("side")
    priority_score = _coerce_float(entry.get("priority_score"))
    trade_date = entry.get("last_trade_date") or entry.get("last_seen")

    include_upgrade_targets = source != "watchlist" or bool(entry.get("active_current_scan"))
    if include_upgrade_targets:
        for target in entry.get("upgrade_targets") or []:
            if not isinstance(target, dict):
                continue
            level = _coerce_float(target.get("level"))
            if level is None:
                continue
            target_side = _normalize_side(target.get("side") or side)
            alert_label = str(target.get("alert_label") or target.get("label") or "D1 upgrade target").strip()
            tier = str(target.get("target_tier") or "A/S").strip() or "A/S"
            reason = str(target.get("reason") or entry.get("upgrade_summary") or "D1 upgrade target").strip()
            _add_d1_flag_event(
                events,
                seen,
                symbol=symbol,
                side=target_side,
                event_type=str(target.get("event_type") or "a_s_upgrade_target").strip(),
                label=f"{tier} upgrade: {alert_label}",
                reason=reason,
                sort_rank=9,
                source="watchlist_upgrade_target",
                priority_score=priority_score,
                trade_date=target.get("armed_at") or trade_date,
                extra={
                    "trigger_id": str(target.get("trigger_id") or "").strip(),
                    "trigger_action": str(target.get("action") or "").strip().lower(),
                    "level_label": str(target.get("label") or "").strip().upper(),
                    "level": level,
                    "anchor_type": str(target.get("anchor_type") or "").strip(),
                    "anchor_date": str(target.get("anchor_date") or "").strip(),
                    "trigger_source": str(target.get("source") or "").strip(),
                    "target_tier": tier,
                },
            )

    if entry.get("retest_followthrough"):
        level = str(entry.get("retest_reference_level") or "").strip()
        note = str(entry.get("retest_note") or "").strip()
        reason = f"{level} retest follow-through".strip()
        if note:
            reason = f"{reason}: {note}" if reason else note
        _add_d1_flag_event(
            events,
            seen,
            symbol=symbol,
            side=side,
            event_type="retest_followthrough",
            label="D1 retest follow-through",
            reason=reason,
            sort_rank=14,
            source=source,
            priority_score=priority_score,
            trade_date=trade_date,
        )

    if entry.get("mid_earnings_ema15_trigger"):
        _add_d1_flag_event(
            events,
            seen,
            symbol=symbol,
            side=side,
            event_type="ema15_bounce",
            label="15EMA D1 bounce",
            reason=str(entry.get("mid_earnings_note") or "mid-earnings EMA15 retest"),
            sort_rank=12,
            source=source,
            priority_score=priority_score,
            trade_date=trade_date,
        )

    if entry.get("mid_earnings_first_dev_trigger"):
        _add_d1_flag_event(
            events,
            seen,
            symbol=symbol,
            side=side,
            event_type="first_dev_retest",
            label="1st-dev D1 bounce",
            reason=str(entry.get("mid_earnings_note") or "mid-earnings 1st-dev retest"),
            sort_rank=13,
            source=source,
            priority_score=priority_score,
            trade_date=trade_date,
        )

    if entry.get("trendline_break_recent"):
        _add_d1_flag_event(
            events,
            seen,
            symbol=symbol,
            side=side,
            event_type="trendline_break",
            label="Trendline breakthrough",
            reason=str(entry.get("trendline_break_note") or entry.get("trendline_note") or "recent D1 trendline break"),
            sort_rank=12,
            source=source,
            priority_score=priority_score,
            trade_date=trade_date,
        )

    if entry.get("previous_day_range_break"):
        label = "Previous-day high break" if _normalize_side(side) == "LONG" else "Previous-day low break"
        _add_d1_flag_event(
            events,
            seen,
            symbol=symbol,
            side=side,
            event_type="previous_day_range_break",
            label=label,
            reason=str(entry.get("previous_day_range_note") or "D1 previous-day range break"),
            sort_rank=18,
            source=source,
            priority_score=priority_score,
            trade_date=trade_date,
        )

    if entry.get("breakout_5d"):
        _add_d1_flag_event(
            events,
            seen,
            symbol=symbol,
            side=side,
            event_type="five_day_breakout",
            label="5D breakout",
            reason="D1 breakout over recent range",
            sort_rank=24,
            source=source,
            priority_score=priority_score,
            trade_date=trade_date,
        )

    theta = entry.get("theta") if isinstance(entry.get("theta"), dict) else {}
    theta_status = str(theta.get("status") or "").strip().lower()
    if theta_status in {"recommended", "cusp"}:
        play_type = str(theta.get("play_type") or "sold_put").strip()
        strike = theta.get("strike") or theta.get("short_strike")
        credit = theta.get("credit")
        expiration = theta.get("expiration")
        label = "Put premium viable" if theta_status == "recommended" else "Put premium cusp"
        reason_parts = []
        if play_type:
            reason_parts.append(play_type.replace("_", " "))
        if strike:
            reason_parts.append(f"strike {float(strike):.2f}")
        if credit:
            reason_parts.append(f"credit {float(credit):.2f}")
        if expiration:
            reason_parts.append(f"exp {expiration}")
        _add_d1_flag_event(
            events,
            seen,
            symbol=symbol,
            side=side,
            event_type=f"theta_{theta_status}",
            label=label,
            reason=", ".join(reason_parts) or "premium now close enough to monitor",
            sort_rank=28 if theta_status == "recommended" else 70,
            source=source,
            priority_score=priority_score,
            trade_date=trade_date,
            extra={"theta": theta},
        )


def build_master_avwap_d1_flag_events(
    focus_map: dict[str, dict[str, Any]],
    events_by_symbol: dict[str, list[dict[str, Any]]],
    d1_watchlist: dict[str, dict[str, Any]] | None = None,
    trade_date: date | None = None,
    d1_upgrade_alerts: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    d1_events: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for symbol, events_for_symbol in (events_by_symbol or {}).items():
        for event in events_for_symbol or []:
            details = _signal_flag_details(event)
            if details is None:
                continue
            event_type, label, sort_rank = details
            signal_type = str(event.get("signal_type") or "").strip().upper()
            anchor_type = str(event.get("anchor_type") or "").strip().upper()
            anchor_date = str(event.get("anchor_date") or "").strip()
            reason = signal_type
            if anchor_type:
                reason = f"{reason} {anchor_type}"
            if anchor_date:
                reason = f"{reason} anchor={anchor_date}"
            _add_d1_flag_event(
                d1_events,
                seen,
                symbol=symbol,
                side=_event_side_from_signal(event),
                event_type=event_type,
                label=label,
                reason=reason,
                sort_rank=sort_rank,
                source="signals",
                priority_score=(focus_map or {}).get(symbol, {}).get("priority_score"),
                trade_date=event.get("trade_date") or trade_date,
                extra={
                    "signal_type": signal_type,
                    "level": event.get("level"),
                    "anchor_type": anchor_type,
                    "anchor_date": anchor_date,
                },
            )

    for symbol, entry in (focus_map or {}).items():
        _add_focus_like_d1_flags(d1_events, seen, symbol, entry, source="focus")

    for symbol, entry in (d1_upgrade_alerts or {}).items():
        if isinstance(entry, dict) and entry.get("bucket_upgrade_events"):
            _add_bucket_upgrade_d1_flags(d1_events, seen, symbol, entry)
        else:
            _add_focus_like_d1_flags(d1_events, seen, symbol, entry, source="upgrade_alerts")

    for symbol, entry in (d1_watchlist or {}).items():
        _add_focus_like_d1_flags(d1_events, seen, symbol, entry, source="watchlist")

    return sorted(
        d1_events,
        key=lambda event: (
            int(event.get("sort_rank", 50) or 50),
            -float(event.get("priority_score") or 0.0),
            str(event.get("symbol") or ""),
            str(event.get("label") or ""),
        ),
    )


def _load_focus_groups_from_feed(
    focus_path: Path = MASTER_AVWAP_FOCUS_FILE,
) -> dict[str, Any]:
    payload = _read_json(focus_path, default={})
    if not isinstance(payload, dict):
        return _empty_focus_groups()

    groups = _empty_focus_groups(source="focus_feed", source_label="Focus feed JSON")
    symbol_map = payload.get("symbols")
    symbol_map = symbol_map if isinstance(symbol_map, dict) else {}

    for section, payload_key in (
        ("favorites", "favorites"),
        ("near_favorite_zones", "near_favorite_zones"),
    ):
        entries = payload.get(payload_key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            symbol_state = symbol_map.get(symbol)
            symbol_state = symbol_state if isinstance(symbol_state, dict) else {}
            side = str(entry.get("side") or symbol_state.get("side") or "").strip().upper()
            if side not in VALID_SIDES:
                continue
            _append_focus_symbol(groups, section, side, symbol)

    has_symbols = any(
        groups[section][side]
        for section in ("favorites", "near_favorite_zones")
        for side in VALID_SIDES
    )
    return groups if has_symbols else _empty_focus_groups()


def load_tradingview_groups(
    focus_path: Path = MASTER_AVWAP_FOCUS_FILE,
    tradingview_path: Path = MASTER_AVWAP_TRADINGVIEW_REPORT_FILE,
) -> dict[str, Any]:
    text = _read_text(tradingview_path)
    groups = _empty_focus_groups(source="tradingview_report", source_label="TradingView report")
    if text:
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
            if line.startswith("-") or current_section not in groups or ":" not in line:
                continue

            side_label, values = line.split(":", 1)
            side = side_label.strip().upper()
            if side not in VALID_SIDES:
                continue
            groups[current_section][side] = _extract_symbols_from_text(values)

        has_report_symbols = any(
            groups[section][side]
            for section in ("favorites", "near_favorite_zones")
            for side in VALID_SIDES
        )
        if has_report_symbols:
            return groups

    focus_groups = _load_focus_groups_from_feed(focus_path=focus_path)
    if focus_groups.get("source") != "none":
        return focus_groups
    return groups


__all__ = [
    "build_master_avwap_active_level_map",
    "build_master_avwap_d1_flag_events",
    "build_master_avwap_second_stdev_cross_map",
    "describe_master_avwap_focus",
    "describe_master_avwap_second_stdev_cross",
    "load_master_avwap_d1_upgrade_alerts",
    "load_master_avwap_d1_watchlist",
    "load_master_avwap_events_for_date",
    "load_master_avwap_focus_map",
    "load_tradingview_groups",
    "normalize_master_avwap_event_row",
]
