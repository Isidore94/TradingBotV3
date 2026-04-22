#!/usr/bin/env python3
"""Shared Master AVWAP output helpers for BounceBot and Mini PC flows."""

from __future__ import annotations

import csv
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from project_paths import (
    AVWAP_SIGNALS_FILE,
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
            "side": str(entry.get("side") or "").strip().upper(),
            "priority_bucket": str(entry.get("priority_bucket") or "").strip().lower(),
            "priority_rank": entry.get("priority_rank"),
            "priority_score": entry.get("priority_score"),
            "favorite_zone": str(entry.get("favorite_zone") or "").strip(),
            "favorite_signals": [
                str(value).strip().upper()
                for value in (entry.get("favorite_signals") or [])
                if str(value).strip()
            ],
            "favorite_context_signals": [
                str(value).strip().upper()
                for value in (entry.get("favorite_context_signals") or [])
                if str(value).strip()
            ],
            "breakout_5d": bool(entry.get("breakout_5d")),
            "retest_followthrough": bool(entry.get("retest_followthrough")),
        }
    return focus_map


def describe_master_avwap_focus(focus_entry: dict[str, Any] | None) -> str:
    bucket = str((focus_entry or {}).get("priority_bucket") or "").strip().lower()
    if bucket == "favorite_setup":
        return "best current favorite setup"
    if bucket == "near_favorite_zone":
        return "near favorite zone"
    return "master avwap focus"


def describe_master_avwap_second_stdev_cross(cross_entry: dict[str, Any] | None) -> str:
    signal_type = str((cross_entry or {}).get("signal_type") or "").strip().upper()
    if signal_type == "CROSS_UP_UPPER_2":
        return "current UPPER_2 cross"
    if signal_type == "CROSS_DOWN_LOWER_2":
        return "current LOWER_2 cross"
    return "current 2nd stdev cross"


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
    focus_groups = _load_focus_groups_from_feed(focus_path=focus_path)
    if focus_groups.get("source") != "none":
        return focus_groups

    text = _read_text(tradingview_path)
    groups = _empty_focus_groups(source="tradingview_report", source_label="TradingView report")
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
        if line.startswith("-") or current_section not in groups or ":" not in line:
            continue

        side_label, values = line.split(":", 1)
        side = side_label.strip().upper()
        if side not in VALID_SIDES:
            continue
        groups[current_section][side] = _extract_symbols_from_text(values)

    return groups


__all__ = [
    "build_master_avwap_active_level_map",
    "build_master_avwap_second_stdev_cross_map",
    "describe_master_avwap_focus",
    "describe_master_avwap_second_stdev_cross",
    "load_master_avwap_events_for_date",
    "load_master_avwap_focus_map",
    "load_tradingview_groups",
    "normalize_master_avwap_event_row",
]
