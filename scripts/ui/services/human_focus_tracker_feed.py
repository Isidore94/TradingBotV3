from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from project_paths import (
    HUMAN_FOCUS_OUTCOMES_FILE,
    HUMAN_FOCUS_PERFORMANCE_FILE,
    MASTER_AVWAP_TIER_PERFORMANCE_FILE,
)


HEADLINE_HORIZON_ORDER = {5: 0, 10: 1, 1: 2, 3: 3}


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_human_focus_performance_rows(path: Path = HUMAN_FOCUS_PERFORMANCE_FILE) -> list[dict[str, Any]]:
    return load_csv_rows(path)


def load_human_focus_outcome_rows(path: Path = HUMAN_FOCUS_OUTCOMES_FILE) -> list[dict[str, Any]]:
    return load_csv_rows(path)


def build_human_focus_comparison_rows(
    human_performance_rows: list[dict[str, Any]],
    tier_performance_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baselines = _bot_sa_baselines(tier_performance_rows)
    comparison_rows: list[dict[str, Any]] = []
    for row in human_performance_rows or []:
        horizon = _int(row.get("horizon_sessions"))
        side = _side(row.get("side"))
        sample_count = _int(row.get("sample_count"))
        if horizon <= 0 or sample_count <= 0:
            continue
        human_avg_pct = _float(row.get("avg_side_return"), 0.0) * 100.0
        bot = baselines.get((side, horizon), {})
        bot_avg_pct = bot.get("avg_side_return_pct")
        delta = "" if bot_avg_pct is None else human_avg_pct - float(bot_avg_pct)
        comparison_rows.append(
            {
                "cohort": "Human Focus",
                "side": side,
                "horizon_sessions": str(horizon),
                "sample_count": str(sample_count),
                "win_rate": row.get("win_rate", ""),
                "avg_side_return_pct": f"{human_avg_pct:.4f}",
                "profit_factor": row.get("profit_factor", ""),
                "bot_sa_sample_count": str(bot.get("sample_count", "")),
                "bot_sa_win_rate": "" if bot.get("win_rate") is None else f"{float(bot['win_rate']):.4f}",
                "bot_sa_avg_side_return_pct": "" if bot_avg_pct is None else f"{float(bot_avg_pct):.4f}",
                "avg_side_return_delta_pct": "" if delta == "" else f"{float(delta):.4f}",
            }
        )
    return sorted(
        comparison_rows,
        key=lambda item: (
            HEADLINE_HORIZON_ORDER.get(_int(item.get("horizon_sessions")), 99),
            _side_rank(item.get("side")),
        ),
    )


def _bot_sa_baselines(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows or []:
        if str(row.get("tier") or "").strip().upper() not in {"S", "A"}:
            continue
        horizon = _int(row.get("horizon_sessions"))
        side = _side(row.get("side"))
        sample_count = _int(row.get("observation_count") or row.get("sample_count"))
        if horizon <= 0 or sample_count <= 0:
            continue
        buckets.setdefault((side, horizon), []).append(row)
        buckets.setdefault(("ALL", horizon), []).append(row)

    baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for key, grouped_rows in buckets.items():
        weights = [_int(row.get("observation_count") or row.get("sample_count")) for row in grouped_rows]
        total = sum(weights)
        if total <= 0:
            continue
        baselines[key] = {
            "sample_count": total,
            "win_rate": _weighted_average(grouped_rows, weights, "win_rate"),
            "avg_side_return_pct": _weighted_average(grouped_rows, weights, "avg_side_return_pct"),
        }
    return baselines


def _weighted_average(rows: list[dict[str, Any]], weights: list[int], key: str) -> float | None:
    total_weight = 0
    total_value = 0.0
    for row, weight in zip(rows, weights):
        value = _float(row.get(key))
        if value is None or weight <= 0:
            continue
        total_weight += weight
        total_value += value * weight
    if total_weight <= 0:
        return None
    return total_value / total_weight


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("LONG"):
        return "LONG"
    if text.startswith("SHORT"):
        return "SHORT"
    return "ALL"


def _side_rank(value: Any) -> int:
    return {"ALL": 0, "LONG": 1, "SHORT": 2}.get(_side(value), 9)


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0

