"""Replay for the favourite-zone long gut golden: the scan's post-scoring steps and a points grid."""

import copy
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import legacy  # noqa: E402

SCAN_DATE = "2026-09-25"
SUMMARY_SIGNALS = {
    "LONG": ([], ["BOUNCE_VWAP"], ["CROSS_UP_UPPER_1"], ["POST_EARNINGS_52W_BREAK"]),
    "SHORT": ([], ["BOUNCE_VWAP"], ["CROSS_DOWN_LOWER_1"], ["POST_EARNINGS_52W_BREAK"]),
}
ZONES = {"LONG": "AVWAPE to UPPER_1", "SHORT": "LOWER_1 to AVWAPE"}


def _jsonable(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def priority_row(raw: dict) -> dict:
    """The scan's priority row, rebuilt from its ai_state entry (`priority_<key>` -> `<key>`)."""
    row = copy.deepcopy(raw)
    for key, value in list(row.items()):
        if key.startswith("priority_") and key[len("priority_"):] not in row:
            row[key[len("priority_"):]] = value
    row["score"] = raw.get("priority_score")
    row["has_favorite_signal"] = bool(
        raw.get("favorite_signals") or raw.get("extreme_move_favorite_ready")
        or raw.get("sma_breakout_confirmed") or raw.get("top_pattern_entry")
        or raw.get("first_dev_break_bonus")
    )
    return row


def observe_scan(raw_rows: list[dict], tmp_dir: Path) -> dict:
    """Replay the scan's post-scoring steps on the rows; one record per symbol."""
    rows = [priority_row(raw) for raw in raw_rows]
    ai_state = {"symbols": {raw["symbol"]: copy.deepcopy(raw) for raw in raw_rows}}
    legacy.apply_final_priority_buckets(rows, ai_state, [], {})

    favorites = [row for row in rows if row.get("priority_bucket") == "favorite_setup"]
    watchlist = [row for row in rows if row.get("priority_bucket") == "near_favorite_zone"]
    post_earnings = [row for row in rows if legacy._is_post_earnings_play_ready(row)]
    actionable = legacy._priority_unique_rows_by_symbol(favorites + watchlist + post_earnings)
    report_rows = legacy._priority_unique_rows_by_symbol(
        actionable
        + legacy._priority_top_pattern_tracking_rows(rows)
        + legacy._priority_sma_breakout_tracking_rows(rows)
        + legacy._priority_stdev_tracking_rows(rows)
    )
    high_conviction = legacy._priority_high_conviction_rows(actionable)
    best_swing = legacy._priority_best_swing_trade_rows(actionable)
    legacy._priority_partition_tier_rows(
        actionable_rows=actionable,
        report_rows=report_rows,
        high_conviction_rows=high_conviction,
        best_swing_rows=best_swing,
    )
    hc_symbols = {row["symbol"] for row in high_conviction}
    best_symbols = {row["symbol"] for row in best_swing}

    tracked = [
        row for row in rows
        if row.get("priority_bucket") in {"favorite_setup", "near_favorite_zone"} and not row.get("ranking_blocked")
    ]
    tracked_symbols = {row["symbol"] for row in tracked}
    legacy.select_tracker_control_rows(rows, tracked, scan_date=SCAN_DATE)

    feed_path = tmp_dir / "focus.json"
    legacy.write_master_avwap_focus_feed(feed_path, rows, ai_state)
    feed = json.loads(feed_path.read_text(encoding="utf-8"))
    sections: dict[str, list[str]] = {}
    for key in ("high_conviction", "favorites", "near_favorite_zones", "post_earnings_plays"):
        for entry in feed.get(key) or []:
            sections.setdefault(entry["symbol"], []).append(key)

    out = {}
    for row in rows:
        symbol = row["symbol"]
        reasons = legacy._d1_watchlist_priority_reasons(row)
        entry_row = dict(row)
        entry_row["_symbol_state"] = ai_state["symbols"][symbol]
        triggers = legacy._build_d1_watchlist_trigger_levels(entry_row, entry_row["_symbol_state"], today_iso=SCAN_DATE)
        if not reasons and not legacy._is_priority_recommendation_blocked(row) and triggers:
            reasons = ["a_s_upgrade_target"]
        out[symbol] = _jsonable({
            "side": row.get("side"),
            "bucket": row.get("priority_bucket") or "",
            "score": row.get("score"),
            "setup_family": row.get("setup_family") or "",
            "tier": row.get(legacy.ASSIGNED_TIER_FIELD) or "",
            "high_conviction": symbol in hc_symbols,
            "best_swing": symbol in best_symbols,
            "tracked": symbol in tracked_symbols,
            "control": row.get("control_reason") or "" if row.get("is_control") else "",
            "focus_sections": sections.get(symbol, []),
            "d1_reasons": reasons,
            "d1_triggers": [
                {key: item.get(key) for key in ("trigger_id", "action", "event_type", "source", "label")}
                for item in triggers
            ],
        })
    return out


def observe_points() -> list[dict]:
    """`build_priority_setup_summary` over a grid: side x zone x retest x trend x signal."""
    out = []
    for side in ("LONG", "SHORT"):
        for zone in (None, ZONES[side]):
            for retest in ("", "AVWAPE", "UPPER_1" if side == "LONG" else "LOWER_1"):
                for trend in ("UP", "DOWN", "SIDEWAYS"):
                    for signals in SUMMARY_SIGNALS[side]:
                        summary = legacy.build_priority_setup_summary(
                            "TEST", side, list(signals), list(signals), trend, zone,
                            retest_followthrough=bool(retest), retest_reference_level=retest,
                        )
                        out.append(_jsonable({
                            "inputs": {"side": side, "zone": zone, "retest": retest, "trend": trend,
                                       "signals": list(signals)},
                            "score": summary["score"],
                            "setup_family": summary["setup_family"],
                            "favorite_zone": summary["favorite_zone"],
                            "has_favorite_signal": summary["has_favorite_signal"],
                        }))
    return out
