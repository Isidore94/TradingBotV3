"""Consolidated GUI snapshot assembly helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from project_paths import (
    LONGS_FILE,
    MASTER_AVWAP_FOCUS_FILE,
    MASTER_AVWAP_MARKET_PREP_FILE,
    MASTER_AVWAP_MARKET_PREP_REPORT_FILE,
    MASTER_AVWAP_TRADINGVIEW_REPORT_FILE,
    SHORTS_FILE,
    get_tracker_storage_details,
)
from master_avwap import (
    EVENT_TICKERS_FILE,
    PRIORITY_SETUPS_FILE,
    STDEV_RANGE_FILE,
    THETA_PUTS_FILE,
    USER_FAVORITES_FILE,
    build_combined_avwap_output_text,
    build_master_avwap_focus_setup_type_text,
    build_master_avwap_focus_side_groups,
    extract_theta_symbols_from_report,
    format_market_prep_payload_report,
)
from master_avwap_shared import load_tradingview_groups
from watchlist_utils import count_watchlist_symbols

MAIN_GUI_OUTPUT_FILE = LONGS_FILE.parent / "consolidated_gui_output.txt"
MAIN_GUI_OUTPUT_DEBOUNCE_MS = 750
MAIN_GUI_OUTPUT_REFRESH_MS = 30_000
MAIN_GUI_BOUNCE_ALERT_LINES = 120


def _read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_widget_text(widget: Any) -> str:
    if widget is None:
        return ""
    try:
        return widget.get("1.0", "end").strip()
    except Exception:
        return ""


def _format_symbol_group(symbols: list[str]) -> str:
    ordered = []
    seen = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol or symbol == "NONE" or symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return ", ".join(ordered) if ordered else "None"


def _normalize_copy_list_text(text: str) -> str:
    cleaned = str(text or "").strip()
    return "" if cleaned.upper() == "NONE" else cleaned


def _tail_lines(text: str, max_lines: int) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines()]
    if len(lines) <= max_lines:
        return "\n".join(lines).strip()
    trimmed = lines[-max_lines:]
    return "\n".join([f"... showing last {max_lines} lines ...", *trimmed]).strip()


def _build_master_avwap_copy_lists(avwap_gui: Any) -> dict[str, str]:
    copy_lists = {
        "favorites": _normalize_copy_list_text(_read_widget_text(getattr(avwap_gui, "favorite_symbols_text", None))),
        "near_favorites": _normalize_copy_list_text(
            _read_widget_text(getattr(avwap_gui, "near_favorite_symbols_text", None))
        ),
        "long_focus": _normalize_copy_list_text(_read_widget_text(getattr(avwap_gui, "long_focus_symbols_text", None))),
        "short_focus": _normalize_copy_list_text(
            _read_widget_text(getattr(avwap_gui, "short_focus_symbols_text", None))
        ),
        "setup_types": _normalize_copy_list_text(_read_widget_text(getattr(avwap_gui, "setup_type_symbols_text", None))),
        "theta": _normalize_copy_list_text(_read_widget_text(getattr(avwap_gui, "theta_symbols_text", None))),
    }

    focus_payload = _read_json_file(MASTER_AVWAP_FOCUS_FILE, default={})
    if not isinstance(focus_payload, dict):
        focus_payload = {}

    tradingview_groups = None

    if not copy_lists["favorites"] or not copy_lists["near_favorites"]:
        tradingview_groups = load_tradingview_groups(
            focus_path=MASTER_AVWAP_FOCUS_FILE,
            tradingview_path=MASTER_AVWAP_TRADINGVIEW_REPORT_FILE,
        )
        if not copy_lists["favorites"]:
            copy_lists["favorites"] = _format_symbol_group(
                tradingview_groups["favorites"]["LONG"] + tradingview_groups["favorites"]["SHORT"]
            )
        if not copy_lists["near_favorites"]:
            copy_lists["near_favorites"] = _format_symbol_group(
                tradingview_groups["near_favorite_zones"]["LONG"]
                + tradingview_groups["near_favorite_zones"]["SHORT"]
            )

    if not copy_lists["long_focus"] or not copy_lists["short_focus"]:
        side_groups = build_master_avwap_focus_side_groups(focus_payload)
        if not side_groups["LONG"] and not side_groups["SHORT"]:
            if tradingview_groups is None:
                tradingview_groups = load_tradingview_groups(
                    focus_path=MASTER_AVWAP_FOCUS_FILE,
                    tradingview_path=MASTER_AVWAP_TRADINGVIEW_REPORT_FILE,
                )
            side_groups = {
                "LONG": tradingview_groups["favorites"]["LONG"] + tradingview_groups["near_favorite_zones"]["LONG"],
                "SHORT": tradingview_groups["favorites"]["SHORT"] + tradingview_groups["near_favorite_zones"]["SHORT"],
            }
        if not copy_lists["long_focus"]:
            copy_lists["long_focus"] = _format_symbol_group(side_groups.get("LONG", []))
        if not copy_lists["short_focus"]:
            copy_lists["short_focus"] = _format_symbol_group(side_groups.get("SHORT", []))

    if not copy_lists["setup_types"]:
        copy_lists["setup_types"] = build_master_avwap_focus_setup_type_text(focus_payload)

    if not copy_lists["theta"]:
        copy_lists["theta"] = _format_symbol_group(
            extract_theta_symbols_from_report(_read_text_file(THETA_PUTS_FILE))
        )

    for key, value in list(copy_lists.items()):
        copy_lists[key] = str(value or "").strip() or "None"

    return copy_lists


def build_consolidated_gui_output(
    mode: str,
    bounce_panel: Any,
    avwap_gui: Any,
) -> str:
    storage = get_tracker_storage_details()
    bounce_controller = getattr(bounce_panel, "controller", None)
    bounce_status = (
        str(bounce_controller.status_var.get()).strip()
        if bounce_controller and hasattr(bounce_controller, "status_var")
        else "Unavailable"
    )
    bounce_connection = (
        str(bounce_controller.connection_var.get()).strip()
        if bounce_controller and hasattr(bounce_controller, "connection_var")
        else "Unavailable"
    )
    bounce_active = (
        str(bounce_controller.active_bounce_var.get()).strip()
        if bounce_controller and hasattr(bounce_controller, "active_bounce_var")
        else "Unavailable"
    )
    bounce_alerts = _tail_lines(
        _read_widget_text(getattr(bounce_panel, "alert_text", None)),
        MAIN_GUI_BOUNCE_ALERT_LINES,
    )
    d1_alerts = _tail_lines(
        _read_widget_text(getattr(bounce_panel, "d1_alert_text", None)),
        MAIN_GUI_BOUNCE_ALERT_LINES,
    )
    avwap_status = (
        str(avwap_gui.status_var.get()).strip()
        if avwap_gui and hasattr(avwap_gui, "status_var")
        else "Unavailable"
    )
    avwap_output = _read_widget_text(getattr(avwap_gui, "avwap_text", None)) or (
        build_combined_avwap_output_text(
            _read_text_file(PRIORITY_SETUPS_FILE),
            _read_text_file(THETA_PUTS_FILE),
            _read_text_file(EVENT_TICKERS_FILE),
            _read_text_file(STDEV_RANGE_FILE),
        )
    )
    market_prep_output = _read_widget_text(getattr(avwap_gui, "market_prep_report_text", None))
    if not market_prep_output:
        market_prep_output = _read_text_file(MASTER_AVWAP_MARKET_PREP_REPORT_FILE)
    if not market_prep_output:
        market_prep_output = format_market_prep_payload_report(
            _read_json_file(MASTER_AVWAP_MARKET_PREP_FILE, default={})
        )
    copy_lists = _build_master_avwap_copy_lists(avwap_gui)

    lines = [
        "Consolidated Trading GUI Snapshot",
        "=" * 80,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"GUI mode: {mode}",
        f"Home folder: {storage['data_dir']}",
        f"Local machine cache: {storage['local_cache_dir']}",
        f"Reports folder: {storage['output_dir']}",
        f"Logs folder: {storage['logs_dir']}",
        f"Snapshot file: {MAIN_GUI_OUTPUT_FILE}",
        "",
        "Watchlists",
        "-" * 80,
        f"Longs: {count_watchlist_symbols(LONGS_FILE)} | Shorts: {count_watchlist_symbols(SHORTS_FILE)}",
        f"longs.txt: {LONGS_FILE}",
        f"shorts.txt: {SHORTS_FILE}",
        "",
        "BounceBot",
        "-" * 80,
        f"Connection: {bounce_connection}",
        f"Status: {bounce_status}",
        f"Active bounces: {bounce_active}",
        "",
        "Master AVWAP",
        "-" * 80,
        f"Status: {avwap_status}",
        f"Priority setups report: {PRIORITY_SETUPS_FILE}",
        f"Theta plays report: {THETA_PUTS_FILE}",
        f"Event tickers report: {EVENT_TICKERS_FILE}",
        f"Stdev report: {STDEV_RANGE_FILE}",
        f"TradingView copy lists report: {MASTER_AVWAP_TRADINGVIEW_REPORT_FILE}",
        f"Market prep report: {MASTER_AVWAP_MARKET_PREP_REPORT_FILE}",
        f"Focus feed: {MASTER_AVWAP_FOCUS_FILE}",
        f"User favorites log: {USER_FAVORITES_FILE}",
        "",
        "Latest AVWAP Results",
        "-" * 80,
        avwap_output or "No AVWAP output yet.",
        "",
        "AVWAP Copy/Paste Lists",
        "-" * 80,
        "Favorite Setups",
        copy_lists["favorites"],
        "",
        "Near Favorite Zones",
        copy_lists["near_favorites"],
        "",
        "Theta Plays",
        copy_lists["theta"],
        "",
        "Directional Longs",
        copy_lists["long_focus"],
        "",
        "Directional Shorts",
        copy_lists["short_focus"],
        "",
        "Score-Ranked Setups",
        copy_lists["setup_types"],
        "",
        "Market Prep",
        "-" * 80,
        market_prep_output or "No market prep output yet.",
        "",
        "Recent BounceBot Alerts",
        "-" * 80,
        bounce_alerts or "No BounceBot alerts yet.",
        "",
        "Recent D1 Master AVWAP Events",
        "-" * 80,
        d1_alerts or "No D1 Master AVWAP events yet.",
    ]
    return "\n".join(lines).rstrip() + "\n"
