#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from master_avwap import (
    export_setup_tracker_views,
    load_priority_scoring_config,
    load_setup_tracker_payload,
    save_priority_scoring_config,
)
from project_paths import (
    MASTER_AVWAP_SCORING_CONFIG_FILE,
    MASTER_AVWAP_SCORING_RECOMMENDATIONS_FILE,
    MASTER_AVWAP_SCORING_TUNER_REPORT_FILE,
    MASTER_AVWAP_SETUP_ATTRIBUTES_FILE,
    MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE,
)

SIGNAL_ATTRIBUTE_MAP = {
    "signals.favorite_signals": "current",
    "signals.context_signals": "context",
}

RULE_WHITELIST = {
    "bouncebot.bullish_weak_long_seen_today",
    "bouncebot.bearish_weak_short_seen_today",
    "bouncebot.relevant_focus_hit_today",
    "pattern.breakout_5d",
    "pattern.retest_followthrough",
    "pattern.retest_reference_level",
    "pattern.extreme_move_watch",
    "pattern.extreme_move_favorite_ready",
    "pattern.extreme_move_retest_level",
    "setup.favorite_zone",
    "trend.trend_20d",
    "trend.trendline_break_recent",
    "structure.compression_flag",
    "structure.previous_avwape_near_0_5atr",
    "structure.ema21_consolidation",
    "levels.current_active_level",
    "levels.previous_active_level",
    "levels.current_band_zone",
    "levels.previous_band_zone",
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _coerce_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _select_outcome_avg_r(row) -> float | None:
    closed_count = _coerce_float(
        row.get("closed_tradeable_setup_count", row.get("closed_tradeable_scenario_count", 0))
    )
    avg_closed_r = _coerce_float(row.get("avg_closed_r"))
    if closed_count and closed_count > 0 and avg_closed_r is not None:
        return avg_closed_r
    return _coerce_float(row.get("avg_total_r"))


def _baseline_by_side(attributes_df: pd.DataFrame) -> dict[str, dict]:
    if attributes_df.empty or "setup_id" not in attributes_df.columns:
        return {}

    setup_df = attributes_df.drop_duplicates(subset=["setup_id"]).copy()
    if "closed_tradeable_scenario_count" in setup_df.columns:
        closed = setup_df[
            pd.to_numeric(setup_df["closed_tradeable_scenario_count"], errors="coerce").fillna(0) > 0
        ].copy()
        if not closed.empty:
            setup_df = closed
    elif "tradeable_scenario_count" in setup_df.columns:
        tradeable = setup_df[pd.to_numeric(setup_df["tradeable_scenario_count"], errors="coerce").fillna(0) > 0].copy()
        if not tradeable.empty:
            setup_df = tradeable

    baselines = {}
    for side, group in setup_df.groupby("side", dropna=False):
        side_key = str(side or "").strip().upper()
        if not side_key or group.empty:
            continue
        avg_total_series = pd.to_numeric(group.apply(_select_outcome_avg_r, axis=1), errors="coerce").dropna()
        target_hit_series = group.get("any_target_hit")
        stop_series = group.get("any_stopped")
        baselines[side_key] = {
            "setup_count": int(group["setup_id"].nunique()),
            "avg_total_r": float(avg_total_series.mean()) if not avg_total_series.empty else 0.0,
            "target_hit_rate": float(target_hit_series.fillna(False).astype(bool).mean()) if target_hit_series is not None else 0.0,
            "stop_rate": float(stop_series.fillna(False).astype(bool).mean()) if stop_series is not None else 0.0,
        }
    return baselines


def _derive_score_delta(row: pd.Series, baseline: dict, *, min_setups: int, max_abs: int) -> int:
    setup_count = int(
        row.get("closed_tradeable_setup_count", 0)
        or row.get("closed_tradeable_scenario_count", 0)
        or row.get("tradeable_setup_count", 0)
        or row.get("tradeable_scenario_count", 0)
        or row.get("setup_count", 0)
        or 0
    )
    if setup_count < min_setups:
        return 0

    avg_total_r = _select_outcome_avg_r(row)
    target_hit_rate = _coerce_float(row.get("target_hit_rate"))
    stop_rate = _coerce_float(row.get("stop_rate"))
    if avg_total_r is None:
        return 0

    edge_r = avg_total_r - float(baseline.get("avg_total_r", 0.0) or 0.0)
    edge_target = (target_hit_rate or 0.0) - float(baseline.get("target_hit_rate", 0.0) or 0.0)
    edge_stop = float(baseline.get("stop_rate", 0.0) or 0.0) - (stop_rate or 0.0)

    confidence = min(1.0, math.sqrt(max(setup_count, 1) / float(max(min_setups, 1))) / 2.0)
    raw_score = (edge_r * 10.0 + edge_target * 4.0 + edge_stop * 4.0) * confidence
    delta = int(round(max(-max_abs, min(max_abs, raw_score))))
    return delta if abs(delta) >= 2 else 0


def _coerce_rule_value(row: pd.Series):
    value_kind = str(row.get("value_kind") or "").strip().lower()
    value_label = row.get("value_label")
    if value_kind == "bool":
        return str(value_label).strip().lower() == "true"
    return str(value_label).strip()


def _recommend_signal_changes(
    leaderboard_df: pd.DataFrame,
    baselines: dict[str, dict],
    current_config: dict,
    *,
    min_setups: int,
) -> list[dict]:
    recommendations = []
    if leaderboard_df.empty:
        return recommendations

    signal_rows = leaderboard_df[
        leaderboard_df["attribute_key"].isin(SIGNAL_ATTRIBUTE_MAP.keys())
        & (leaderboard_df["value_kind"] == "list_item")
    ].copy()
    if signal_rows.empty:
        return recommendations

    for _, row in signal_rows.iterrows():
        side = str(row.get("side") or "").strip().upper()
        baseline = baselines.get(side)
        if not baseline:
            continue
        bucket = SIGNAL_ATTRIBUTE_MAP.get(str(row.get("attribute_key") or ""))
        score_delta = _derive_score_delta(
            row,
            baseline,
            min_setups=min_setups,
            max_abs=12 if bucket == "current" else 8,
        )
        if score_delta == 0:
            continue

        signal_name = str(row.get("value_label") or "").strip()
        if not signal_name:
            continue

        current_weight = int(
            current_config.get("signal_weights", {})
            .get(bucket, {})
            .get(side, {})
            .get(signal_name, 0)
            or 0
        )
        new_weight = max(0, current_weight + score_delta)
        if new_weight == current_weight:
            continue

        recommendations.append(
            {
                "section": "signal_weight",
                "bucket": bucket,
                "side": side,
                "signal": signal_name,
                "attribute_key": row.get("attribute_key"),
                "setup_count": int(row.get("setup_count", 0) or 0),
                "avg_total_r": _select_outcome_avg_r(row),
                "baseline_avg_total_r": float(baseline.get("avg_total_r", 0.0) or 0.0),
                "target_hit_rate": _coerce_float(row.get("target_hit_rate")),
                "baseline_target_hit_rate": float(baseline.get("target_hit_rate", 0.0) or 0.0),
                "stop_rate": _coerce_float(row.get("stop_rate")),
                "baseline_stop_rate": float(baseline.get("stop_rate", 0.0) or 0.0),
                "weight_delta": int(score_delta),
                "old_weight": int(current_weight),
                "new_weight": int(new_weight),
            }
        )

    recommendations.sort(
        key=lambda item: (
            -abs(int(item.get("weight_delta", 0) or 0)),
            -int(item.get("setup_count", 0) or 0),
            str(item.get("signal") or ""),
        )
    )
    return recommendations


def _recommend_attribute_rules(
    leaderboard_df: pd.DataFrame,
    baselines: dict[str, dict],
    *,
    min_setups: int,
) -> list[dict]:
    rules = []
    if leaderboard_df.empty:
        return rules

    candidate_rows = leaderboard_df[
        leaderboard_df["attribute_key"].isin(RULE_WHITELIST)
        & leaderboard_df["value_kind"].isin(["bool", "text"])
    ].copy()
    if candidate_rows.empty:
        return rules

    grouped_choices: dict[tuple[str, str], list[dict]] = {}
    for _, row in candidate_rows.iterrows():
        side = str(row.get("side") or "").strip().upper()
        baseline = baselines.get(side)
        if not baseline:
            continue
        delta = _derive_score_delta(row, baseline, min_setups=min_setups, max_abs=10)
        if delta == 0:
            continue
        grouped_choices.setdefault((side, str(row.get("attribute_key") or "")), []).append(
            {
                "row": row,
                "delta": delta,
                "baseline": baseline,
            }
        )

    for (side, attribute_key), items in grouped_choices.items():
        positives = [item for item in items if item["delta"] > 0]
        negatives = [item for item in items if item["delta"] < 0]
        selected = []
        if positives:
            selected.append(sorted(positives, key=lambda item: (-item["delta"], -int(item["row"].get("setup_count", 0) or 0)))[0])
        if negatives:
            selected.append(sorted(negatives, key=lambda item: (item["delta"], -int(item["row"].get("setup_count", 0) or 0)))[0])

        for item in selected:
            row = item["row"]
            value = _coerce_rule_value(row)
            label = f"{side} {row.get('attribute_label')} = {row.get('value_label')}"
            rules.append(
                {
                    "enabled": True,
                    "source": "auto_tuner",
                    "side": side,
                    "attribute_key": attribute_key,
                    "operator": "equals",
                    "value": value,
                    "score_delta": int(item["delta"]),
                    "label": label,
                    "setup_count": int(row.get("setup_count", 0) or 0),
                    "avg_total_r": _select_outcome_avg_r(row),
                    "baseline_avg_total_r": float(item["baseline"].get("avg_total_r", 0.0) or 0.0),
                    "target_hit_rate": _coerce_float(row.get("target_hit_rate")),
                    "baseline_target_hit_rate": float(item["baseline"].get("target_hit_rate", 0.0) or 0.0),
                    "stop_rate": _coerce_float(row.get("stop_rate")),
                    "baseline_stop_rate": float(item["baseline"].get("stop_rate", 0.0) or 0.0),
                }
            )

    rules.sort(
        key=lambda item: (
            -abs(int(item.get("score_delta", 0) or 0)),
            -int(item.get("setup_count", 0) or 0),
            str(item.get("attribute_key") or ""),
        )
    )
    return rules


def _apply_recommendations_to_config(current_config: dict, signal_changes: list[dict], attribute_rules: list[dict]) -> dict:
    updated = json.loads(json.dumps(current_config))
    updated.setdefault("signal_weights", {}).setdefault("current", {})
    updated.setdefault("signal_weights", {}).setdefault("context", {})

    for change in signal_changes:
        bucket = str(change.get("bucket") or "current")
        side = str(change.get("side") or "").strip().upper()
        signal = str(change.get("signal") or "").strip()
        if not side or not signal:
            continue
        updated["signal_weights"].setdefault(bucket, {}).setdefault(side, {})
        updated["signal_weights"][bucket][side][signal] = int(change.get("new_weight", 0) or 0)

    preserved_rules = [
        rule
        for rule in updated.get("attribute_adjustments", [])
        if isinstance(rule, dict) and str(rule.get("source") or "") != "auto_tuner"
    ]
    updated["attribute_adjustments"] = preserved_rules + [dict(rule) for rule in attribute_rules]
    return updated


def _build_report_text(
    baselines: dict[str, dict],
    signal_changes: list[dict],
    attribute_rules: list[dict],
    *,
    apply_path: Path | None,
) -> str:
    lines = []
    lines.append("Master AVWAP scoring tuner")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Baselines")
    lines.append("-" * 80)
    if not baselines:
        lines.append("No baseline data found.")
    else:
        for side in sorted(baselines.keys()):
            baseline = baselines[side]
            lines.append(
                f"{side}: setups={baseline['setup_count']} avg_total_r={baseline['avg_total_r']:.2f} "
                f"target_hit={baseline['target_hit_rate'] * 100:.0f}% stop_rate={baseline['stop_rate'] * 100:.0f}%"
            )

    lines.append("")
    lines.append("Signal weight changes")
    lines.append("-" * 80)
    if not signal_changes:
        lines.append("No signal weight changes recommended.")
    else:
        for change in signal_changes[:24]:
            lines.append(
                f"{change['side']} {change['bucket']} {change['signal']}: "
                f"{change['old_weight']} -> {change['new_weight']} ({change['weight_delta']:+d}) "
                f"setups={change['setup_count']} avg_total_r={change['avg_total_r']:.2f}"
            )

    lines.append("")
    lines.append("Attribute rules")
    lines.append("-" * 80)
    if not attribute_rules:
        lines.append("No attribute rules recommended.")
    else:
        for rule in attribute_rules[:24]:
            lines.append(
                f"{rule['label']}: {int(rule['score_delta']):+d} "
                f"setups={rule['setup_count']} avg_total_r={float(rule['avg_total_r'] or 0.0):.2f}"
            )

    lines.append("")
    lines.append("Outputs")
    lines.append("-" * 80)
    lines.append(f"Recommendations JSON: {MASTER_AVWAP_SCORING_RECOMMENDATIONS_FILE}")
    lines.append(f"Tuner report: {MASTER_AVWAP_SCORING_TUNER_REPORT_FILE}")
    lines.append(f"Scoring config: {MASTER_AVWAP_SCORING_CONFIG_FILE}")
    if apply_path:
        lines.append(f"Applied config written to: {apply_path}")
    else:
        lines.append("Config not applied. Re-run with --apply to write the recommendations.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Master AVWAP tracker results and recommend scoring adjustments.")
    parser.add_argument("--min-setups", type=int, default=8, help="Minimum setup count before a signal or attribute can influence scoring.")
    parser.add_argument("--apply", action="store_true", help="Write the generated recommendations into the live scoring config JSON.")
    args = parser.parse_args()

    tracker_payload = load_setup_tracker_payload()
    if isinstance(tracker_payload, dict) and tracker_payload.get("setups"):
        export_setup_tracker_views(tracker_payload)

    attributes_df = _load_csv(MASTER_AVWAP_SETUP_ATTRIBUTES_FILE)
    leaderboard_df = _load_csv(MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE)
    if attributes_df.empty:
        message = (
            f"No setup attribute data found at {MASTER_AVWAP_SETUP_ATTRIBUTES_FILE}\n"
            "Run the Master AVWAP scanner with setup-tracker updates enabled, or use the tracker backfill first."
        )
        MASTER_AVWAP_SCORING_TUNER_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        MASTER_AVWAP_SCORING_TUNER_REPORT_FILE.write_text(message, encoding="utf-8")
        print(message)
        return 0
    if leaderboard_df.empty:
        message = (
            f"No setup attribute leaderboard data found at {MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE}\n"
            "Run a scan or backfill first so the setup tracker can export attribute stats."
        )
        MASTER_AVWAP_SCORING_TUNER_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        MASTER_AVWAP_SCORING_TUNER_REPORT_FILE.write_text(message, encoding="utf-8")
        print(message)
        return 0

    baselines = _baseline_by_side(attributes_df)
    current_config = load_priority_scoring_config()
    signal_changes = _recommend_signal_changes(leaderboard_df, baselines, current_config, min_setups=max(1, int(args.min_setups)))
    attribute_rules = _recommend_attribute_rules(leaderboard_df, baselines, min_setups=max(1, int(args.min_setups)))

    recommendation_payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "min_setups": max(1, int(args.min_setups)),
        "baselines": baselines,
        "signal_changes": signal_changes,
        "attribute_rules": attribute_rules,
        "attribute_source": str(MASTER_AVWAP_SETUP_ATTRIBUTES_FILE),
        "leaderboard_source": str(MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE),
    }
    MASTER_AVWAP_SCORING_RECOMMENDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MASTER_AVWAP_SCORING_RECOMMENDATIONS_FILE.write_text(json.dumps(recommendation_payload, indent=2), encoding="utf-8")

    applied_path = None
    if args.apply:
        next_config = _apply_recommendations_to_config(current_config, signal_changes, attribute_rules)
        save_priority_scoring_config(next_config)
        applied_path = MASTER_AVWAP_SCORING_CONFIG_FILE

    report_text = _build_report_text(baselines, signal_changes, attribute_rules, apply_path=applied_path)
    MASTER_AVWAP_SCORING_TUNER_REPORT_FILE.write_text(report_text, encoding="utf-8")
    print(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
