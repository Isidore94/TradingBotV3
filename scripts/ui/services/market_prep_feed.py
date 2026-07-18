from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from market_session import get_market_session_window
from project_paths import (
    HUMAN_FOCUS_DAILY_PICKS_FILE,
    MASTER_AVWAP_MARKET_PREP_FILE,
    MASTER_AVWAP_MARKET_PREP_REPORT_FILE,
)


MARKET_PREP_SECTION_DEFINITIONS = [
    {
        "id": "strongest_stocks_top_decile",
        "title": "Strongest Stocks: Top 10% D1 vs SPY",
        "empty_message": "None",
    },
    {
        "id": "weakest_stocks_bottom_decile",
        "title": "Weakest Stocks: Bottom 10% D1 vs SPY",
        "empty_message": "None",
    },
    {
        "id": "recently_strong_now_pulling_back",
        "title": "Recently Strong, Now Pulling Back",
        "empty_message": "None",
    },
    {
        "id": "strongest_industries",
        "title": "Strongest Industries: D1 vs SPY",
        "empty_message": "None",
    },
    {
        "id": "post_earnings_potential_plays",
        "title": "Post-Earnings Potential Plays",
        "empty_message": "None",
    },
    {
        "id": "earnings_last_night_or_today",
        "title": "Earnings Last Night / Today",
        "empty_message": "None",
    },
    {
        "id": "long_avwape_to_1stdev",
        "title": "Longs: AVWAPE to 1st Dev",
        "empty_message": "None",
    },
    {
        "id": "short_avwape_to_1stdev",
        "title": "Shorts: AVWAPE to 1st Dev",
        "empty_message": "None",
    },
    {
        "id": "long_1st_to_2nd_stdev",
        "title": "Longs: 1st to 2nd Dev",
        "empty_message": "None",
    },
    {
        "id": "short_1st_to_2nd_stdev",
        "title": "Shorts: 1st to 2nd Dev",
        "empty_message": "None",
    },
    {
        "id": "long_2nd_to_3rd_stdev_2_sessions",
        "title": "Longs: 2nd to 3rd Dev, 2+ Sessions",
        "empty_message": "None",
    },
    {
        "id": "short_2nd_to_3rd_stdev_2_sessions",
        "title": "Shorts: 2nd to 3rd Dev, 2+ Sessions",
        "empty_message": "None",
    },
]


def load_market_prep_payload(path: Path = MASTER_AVWAP_MARKET_PREP_FILE) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_market_prep_report() -> str:
    if MASTER_AVWAP_MARKET_PREP_REPORT_FILE.exists():
        try:
            text = MASTER_AVWAP_MARKET_PREP_REPORT_FILE.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                return text
        except OSError:
            pass
    payload = load_market_prep_payload()
    if not payload:
        return "No market prep output yet."
    try:
        from master_avwap_lib.outputs.market_prep import format_market_prep_payload_report

        return format_market_prep_payload_report(payload)
    except Exception:
        return json.dumps(payload, indent=2)


def market_prep_sections(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sections = payload.get("sections", []) if isinstance(payload, dict) else []
    if not isinstance(sections, list):
        sections = []
    return {
        str(section.get("id") or ""): section
        for section in sections
        if isinstance(section, dict)
    }


def section_copy_text(section: dict[str, Any], definition: dict[str, str]) -> str:
    return str(section.get("copy_text") or definition.get("empty_message") or "None").strip() or "None"


def section_symbol_count(section: dict[str, Any], text: str) -> int:
    symbols = section.get("symbols") if isinstance(section, dict) else []
    if isinstance(symbols, list):
        return len([symbol for symbol in symbols if str(symbol).strip()])
    clean = str(text or "").strip()
    if not clean or clean == "None":
        return 0
    return len([part for part in clean.replace(",", " ").split() if part.strip()])


def load_human_focus_daily_picks(
    *,
    trade_date: Any = None,
    path: Path = HUMAN_FOCUS_DAILY_PICKS_FILE,
) -> list[dict[str, str]]:
    target_date = str(trade_date or get_market_session_window().market_date.isoformat()).strip()[:10]
    if not Path(path).exists():
        return []
    try:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []
    return [
        row
        for row in rows
        if str(row.get("trade_date") or "").strip() == target_date
    ]


def human_focus_pick_text(rows: list[dict[str, Any]]) -> str:
    grouped = {"LONG": [], "SHORT": []}
    for row in rows or []:
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        if not symbol:
            continue
        side = "SHORT" if side.startswith("SHORT") else "LONG"
        if symbol not in grouped[side]:
            grouped[side].append(symbol)
    if not grouped["LONG"] and not grouped["SHORT"]:
        return "No human focus picks snapshotted for this market date."
    lines = []
    if grouped["LONG"]:
        lines.append(f"LONG: {', '.join(grouped['LONG'])}")
    if grouped["SHORT"]:
        lines.append(f"SHORT: {', '.join(grouped['SHORT'])}")
    return "\n".join(lines)


def human_focus_pick_count(rows: list[dict[str, Any]]) -> int:
    return len(
        {
            (str(row.get("symbol") or "").strip().upper(), str(row.get("side") or "").strip().upper())
            for row in rows or []
            if str(row.get("symbol") or "").strip()
        }
    )
