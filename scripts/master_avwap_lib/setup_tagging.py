"""Evidence-backed setup tags shared by the scanner, tracker, and Qt desk.

The legacy scanner's ``setup_tags`` field historically contained only the raw
same-day AVWAP events.  That made many valid setups look unclassified and made
different setup families look identical.  This module keeps those raw events
separately and derives a compact semantic tag set from the full setup row.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any


SETUP_TAGS_SCHEMA = "setup_tags_v2"
DEFAULT_MAX_SETUP_TAGS = 6

_FAMILY_TAGS = {
    "avwape_to_first_dev": "AVWAPE_TO_FIRST_DEV",
    "avwap_retest_followthrough": "AVWAP_RETEST",
    "previous_avwape_bounce": "PREVIOUS_AVWAPE_BOUNCE",
    "extreme_move_retest": "EXTREME_MOVE_RETEST",
    "favorite_zone_watch": "FAVORITE_ZONE_WATCH",
    "mid_earnings_above_2nd_stdev": "MID_EARNINGS_SECOND_DEV_HOLD",
    "mid_earnings_ema15_retest": "MID_EARNINGS_EMA15_RETEST",
    "mid_earnings_ema21_retest": "MID_EARNINGS_EMA21_RETEST",
    "mid_earnings_1stdev_retest": "MID_EARNINGS_FIRST_DEV_RETEST",
    "post_earnings_52w_break": "POST_EARNINGS_52W_BREAK",
    "post_earnings_avwap_bounce": "POST_EARNINGS_AVWAP_BOUNCE",
    "sma_breakout": "SMA_BREAKOUT_CONFIRMED",
    "sma_breakout_retest_tracking": "SMA_BREAKOUT_WATCH",
    "top_pattern": "TOP_PATTERN_ENTRY",
    "top_pattern_tracking": "TOP_PATTERN_WATCH",
    "avwap_band_bounce": "AVWAP_BAND_BOUNCE",
    "avwap_breakout": "AVWAP_BREAKOUT",
}

_RAW_SIGNAL_RE = re.compile(
    r"^(?:PREV_)?(?:CROSS_(?:UP|DOWN)|BOUNCE)_(?:AVWAPE|VWAP|UPPER_[123]|LOWER_[123])$"
)
_DEV_NAMES = {"1": "FIRST", "2": "SECOND", "3": "THIRD"}
_TAG_ALIASES = {
    "POST_EARNINGS_AVWAPE_BOUNCE": "POST_EARNINGS_AVWAP_BOUNCE",
}


def _listish(value: Any) -> list[str]:
    if isinstance(value, str):
        delimiter = ";" if ";" in value else ","
        values = value.split(delimiter)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = []
    normalized: list[str] = []
    for value in values:
        tag = str(value or "").strip().upper().replace(" ", "_")
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").strip().upper()).strip("_")


def _side(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in {"LONG", "SHORT"} else ""


def _side_aligned_score(value: Any, side: str) -> bool:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return False
    return (side == "LONG" and score > 0.0) or (side == "SHORT" and score < 0.0)


def _signal_tag(signal: str, side: str) -> tuple[str, str]:
    """Return (semantic tag, warning) for one raw AVWAP event."""
    signal = str(signal or "").strip().upper()
    if signal in {"PREV_BOUNCE_VWAP", "PREV_BOUNCE_AVWAPE"}:
        return "PREVIOUS_AVWAPE_BOUNCE", ""
    if signal in {"BOUNCE_VWAP", "BOUNCE_AVWAPE"}:
        return "VWAP_BOUNCE", ""

    bounce = re.fullmatch(r"BOUNCE_(?:UPPER|LOWER)_([123])", signal)
    if bounce:
        expected_level = "UPPER" if side == "LONG" else "LOWER" if side == "SHORT" else ""
        if expected_level and f"BOUNCE_{expected_level}_" not in signal:
            return "", f"{signal} conflicts with side={side}"
        return f"{_DEV_NAMES[bounce.group(1)]}_DEV_BOUNCE", ""

    cross = re.fullmatch(r"CROSS_(UP|DOWN)_(VWAP|AVWAPE|UPPER_([123])|LOWER_([123]))", signal)
    if cross:
        direction, level = cross.group(1), cross.group(2)
        expected_direction = "UP" if side == "LONG" else "DOWN" if side == "SHORT" else ""
        if expected_direction and direction != expected_direction:
            return "", f"{signal} conflicts with side={side}"
        if level in {"VWAP", "AVWAPE"}:
            return ("VWAP_RECLAIM" if direction == "UP" else "VWAP_BREAKDOWN"), ""
        dev_number = cross.group(3) or cross.group(4) or ""
        move = "BREAKOUT" if direction == "UP" else "BREAKDOWN"
        return f"{_DEV_NAMES[dev_number]}_DEV_{move}", ""

    return "", ""


def _family_tag(family: str, signal_tags: list[str]) -> str:
    if family == "avwap_band_bounce":
        return next(
            (tag for tag in signal_tags if tag in {"VWAP_BOUNCE", "FIRST_DEV_BOUNCE", "SECOND_DEV_BOUNCE", "THIRD_DEV_BOUNCE"}),
            _FAMILY_TAGS[family],
        )
    if family == "avwap_breakout":
        return next(
            (tag for tag in signal_tags if tag.endswith(("BREAKOUT", "BREAKDOWN")) or tag in {"VWAP_RECLAIM", "VWAP_BREAKDOWN"}),
            _FAMILY_TAGS[family],
        )
    if family in _FAMILY_TAGS:
        return _FAMILY_TAGS[family]
    if family and family != "general":
        return _slug(family)
    return ""


def derive_setup_tag_payload(
    row: Mapping[str, Any] | None,
    *,
    max_tags: int = DEFAULT_MAX_SETUP_TAGS,
) -> dict[str, Any]:
    """Derive compact semantic tags plus provenance from a complete setup row."""
    row = row if isinstance(row, Mapping) else {}
    side = _side(row.get("side"))
    family = str(row.get("setup_family") or row.get("tracker_setup_family") or "").strip().lower()

    existing_tags = _listish(row.get("setup_tags"))
    source_tags = _listish(row.get("setup_source_tags"))
    explicit_signal_tags = _listish(row.get("setup_signal_tags"))
    favorite_signals = _listish(row.get("favorite_signals"))
    if explicit_signal_tags:
        raw_signals = explicit_signal_tags
    else:
        raw_signals = list(dict.fromkeys([*favorite_signals, *[tag for tag in existing_tags if _RAW_SIGNAL_RE.match(tag)]]))

    signal_tags: list[str] = []
    warnings: list[str] = []
    signal_evidence: dict[str, list[str]] = {}
    for signal in raw_signals:
        tag, warning = _signal_tag(signal, side)
        if warning and warning not in warnings:
            warnings.append(warning)
        if tag:
            if tag not in signal_tags:
                signal_tags.append(tag)
            signal_evidence.setdefault(tag, []).append(f"favorite_signals={signal}")

    tags: list[str] = []
    roles: dict[str, str] = {}
    evidence: dict[str, list[str]] = {}

    def add(tag: str, role: str, reason: str) -> None:
        tag = _slug(tag)
        tag = _TAG_ALIASES.get(tag, tag)
        if not tag:
            return
        if tag not in tags:
            tags.append(tag)
            roles[tag] = role
        evidence.setdefault(tag, [])
        if reason and reason not in evidence[tag]:
            evidence[tag].append(reason)

    primary = _family_tag(family, signal_tags)
    add(primary, "primary", f"setup_family={family}" if family else "")
    for tag in signal_tags:
        add(tag, "trigger", "; ".join(signal_evidence.get(tag, [])))

    # Preserve explicit semantic study/custom tags, but never leak raw AVWAP
    # event names back into the canonical display field.
    if row.get("setup_tags_schema") != SETUP_TAGS_SCHEMA:
        source_tags = [tag for tag in existing_tags if not _RAW_SIGNAL_RE.match(tag)]
    for tag in source_tags:
        add(tag, "trigger", f"legacy_setup_tags={tag}")

    # Confirmations are deliberately side-aware where the underlying metric is
    # directional.  This prevents a strong-looking but backwards RS label.
    if bool(row.get("top_pattern_entry")):
        add("TOP_PATTERN_ENTRY", "confirmation", "top_pattern_entry=true")
    if bool(row.get("previous_day_range_break")):
        add("RANGE_BREAK_CONFIRMED", "confirmation", "previous_day_range_break=true")
    if bool(row.get("breakout_5d")):
        add("FIVE_DAY_BREAKOUT", "confirmation", "breakout_5d=true")
    trend = str(row.get("trend_20d") or "").strip().upper()
    if (side == "LONG" and trend == "UP") or (side == "SHORT" and trend == "DOWN"):
        add("D1_TREND_ALIGNED", "confirmation", f"trend_20d={trend}")
    if bool(row.get("trend_ma_alignment")):
        add("MA_TREND_ALIGNED", "confirmation", "trend_ma_alignment=true")
    if bool(row.get("htf_trend_aligned")):
        add("HTF_TREND_ALIGNED", "confirmation", "htf_trend_aligned=true")
    if bool(row.get("htf_retest_confirmed")):
        sma = str(row.get("htf_retest_sma") or "").strip().upper()
        add("HTF_RETEST", "confirmation", f"htf_retest_confirmed=true{sma and f'; htf_retest_sma={sma}'}")
    if bool(row.get("vwap_range_confirmation")):
        add("VWAP_RANGE_CONFIRMED", "confirmation", "vwap_range_confirmation=true")
    if bool(row.get("trendline_break_recent")):
        add("TRENDLINE_BREAK", "confirmation", "trendline_break_recent=true")
    if bool(row.get("sma_breakout_confirmed")):
        label = str(row.get("sma_breakout_sma_label") or "").strip().upper()
        add("SMA_BREAKOUT_CONFIRMED", "confirmation", f"sma_breakout_confirmed=true{label and f'; sma={label}'}")
    if _side_aligned_score(row.get("daily_relative_strength_score"), side):
        add("D1_RS" if side == "LONG" else "D1_RW", "context", "daily_relative_strength_score is side-aligned")
    if _side_aligned_score(row.get("industry_relative_strength_score"), side):
        add("INDUSTRY_RS" if side == "LONG" else "INDUSTRY_RW", "context", "industry_relative_strength_score is side-aligned")

    if not tags:
        add("UNCLASSIFIED_SETUP", "diagnostic", "no setup-family, trigger, or confirmation evidence")
        warnings.append("No evidence-backed setup classification was available")

    max_tags = max(1, int(max_tags or DEFAULT_MAX_SETUP_TAGS))
    visible_tags = tags[:max_tags]
    return {
        "setup_tags_schema": SETUP_TAGS_SCHEMA,
        "setup_tags": visible_tags,
        "setup_signal_tags": raw_signals,
        "setup_source_tags": source_tags,
        "setup_tag_roles": {tag: roles[tag] for tag in visible_tags},
        "setup_tag_evidence": {tag: evidence[tag] for tag in visible_tags},
        "setup_tag_warnings": warnings,
    }


def apply_setup_tag_payload(
    row: MutableMapping[str, Any],
    *,
    max_tags: int = DEFAULT_MAX_SETUP_TAGS,
) -> dict[str, Any]:
    """Derive and apply the v2 payload to a mutable setup mapping."""
    payload = derive_setup_tag_payload(row, max_tags=max_tags)
    row.update(payload)
    return payload


def canonicalize_priority_setup_tags(
    priority_rows: list[dict[str, Any]],
    ai_state: MutableMapping[str, Any] | None = None,
    feature_rows_by_symbol: MutableMapping[str, dict[str, Any]] | None = None,
) -> int:
    """Apply v2 tags and keep scanner persistence surfaces in sync."""
    ai_symbols = ai_state.get("symbols", {}) if isinstance(ai_state, Mapping) else {}
    features = feature_rows_by_symbol if isinstance(feature_rows_by_symbol, MutableMapping) else {}
    updated = 0
    for row in priority_rows or []:
        if not isinstance(row, MutableMapping):
            continue
        payload = apply_setup_tag_payload(row)
        symbol = str(row.get("symbol") or "").strip().upper()
        symbol_entry = ai_symbols.get(symbol) if isinstance(ai_symbols, Mapping) else None
        if isinstance(symbol_entry, MutableMapping):
            symbol_entry.update(payload)
        feature_row = features.get(symbol)
        if isinstance(feature_row, MutableMapping):
            feature_row["setup_tags"] = ";".join(payload["setup_tags"])
            feature_row["setup_signal_tags"] = ";".join(payload["setup_signal_tags"])
            feature_row["setup_source_tags"] = ";".join(payload["setup_source_tags"])
            feature_row["setup_tags_schema"] = SETUP_TAGS_SCHEMA
            feature_row["setup_tag_evidence"] = payload["setup_tag_evidence"]
            feature_row["setup_tag_warnings"] = payload["setup_tag_warnings"]
        updated += 1
    return updated
