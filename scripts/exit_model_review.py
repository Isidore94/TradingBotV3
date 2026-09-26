"""S13: three exit models side by side, per setup family x side. Display only.

F17 (2026-09-26): the tracker's target/stop exits book about -0.04R on
`avwap_retest_followthrough` SHORT while those names run +1.2 ATR in 10 sessions.
This measures, on the SAME session-horizon outcomes, what three exits would book:

* ``current``   - the tracker's own R (target then stop), the setup's
  `_scoring_outcome_summary.avg_total_r` from the scoring snapshot, joined on
  ``(SYMBOL, SIDE, scan_date)``. Open setups are marked to market, as the
  tracker's own leaderboard does.
* ``stop_only`` - stop 1 ATR against, no target, hold 10 sessions.
* ``trail``     - stop 1 ATR behind the best close so far, hold 10 sessions.

The two ATR models read the session-horizon closes at sessions 1, 3, 5 and 10
(the only closes that file carries), so a stop is checked on those closes and
filled at that close. 1R = 1 x atr20 as the scan saw it on the scan date
(`d1_features_history.csv`, the day's last scan row): point-in-time only.

A row counts only when all four closes are measured and its ATR is known; the
rest are counted as unknown, never as zero. Nothing here changes an exit rule,
a score, a rank that gates, an alert, a watchlist or `review_policy.json`.
Pure: rows in, plain dicts out, no file access, no clock.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

CHECKPOINTS = (1, 3, 5, 10)
STOP_ATR = 1.0
TRAIL_ATR = 1.0
OUTCOME_KIND = "favorable_direction_session_v2"

UNKNOWN_NO_ATR = "no_atr"
UNKNOWN_NOT_MEASURED = "not_measured"

NO_DATA_SENTENCE = "Exit models: no session-horizon outcomes to measure yet."


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # NaN is unknown


def _flag(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _key(symbol: Any, side: Any, scan_date: Any) -> tuple[str, str, str]:
    return (
        str(symbol or "").strip().upper(),
        str(side or "").strip().upper(),
        str(scan_date or "").strip()[:10],
    )


def atr_index(feature_rows: Iterable[Mapping[str, Any]] | None) -> dict[tuple[str, str], float]:
    """``{(SYMBOL, run_date): atr20}`` - the LAST scan row of each day wins."""
    out: dict[tuple[str, str], float] = {}
    for row in feature_rows or ():
        atr = _float(row.get("atr20"))
        if atr is None or atr <= 0:
            continue
        day = str(row.get("run_date") or "").strip()[:10]
        symbol = str(row.get("symbol") or "").strip().upper()
        if day and symbol:
            out[(symbol, day)] = atr
    return out


def current_r_index(setups: Mapping[str, Any] | None) -> dict[tuple[str, str, str], float]:
    """``{(SYMBOL, SIDE, scan_date): mean avg_total_r}`` from the scoring snapshot."""
    sums: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for setup in (setups or {}).values():
        if not isinstance(setup, Mapping):
            continue
        summary = setup.get("_scoring_outcome_summary") or {}
        r = _float(summary.get("avg_total_r")) if isinstance(summary, Mapping) else None
        if r is None:
            continue
        sums[_key(setup.get("symbol"), setup.get("side"), setup.get("scan_date"))].append(r)
    return {key: sum(values) / len(values) for key, values in sums.items()}


def exit_model_r(moves_atr: Mapping[int, float | None]) -> dict[str, float] | None:
    """R under stop-only and trail from favorable moves in ATR at each checkpoint.

    ``moves_atr[h]`` is the side-adjusted close move after ``h`` sessions, in
    ATR. None when any checkpoint is missing.
    """
    path = [moves_atr.get(h) for h in CHECKPOINTS]
    if any(move is None for move in path):
        return None
    stop_only = path[-1]
    for move in path:
        if move <= -STOP_ATR:
            stop_only = move
            break
    trail = path[-1]
    best = 0.0
    for move in path:
        if move <= best - TRAIL_ATR:
            trail = move
            break
        best = max(best, move)
    return {"stop_only": float(stop_only), "trail": float(trail)}


def review(
    horizon_rows: Iterable[Mapping[str, Any]] | None,
    atr_by_symbol_day: Mapping[tuple[str, str], float] | None,
    current_r: Mapping[tuple[str, str, str], float] | None,
) -> dict[str, Any]:
    """Per family x side: the three models' mean R with n, and the unknowns."""
    entries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in horizon_rows or ():
        if str(row.get("outcome_kind") or "").strip() != OUTCOME_KIND:
            continue
        try:
            horizon = int(float(row.get("horizon_sessions")))
        except (TypeError, ValueError):
            continue
        if horizon not in CHECKPOINTS:
            continue
        key = _key(row.get("symbol"), row.get("side"), row.get("scan_date"))
        entry = entries.setdefault(
            key,
            {
                "family": str(row.get("setup_family") or "").strip() or "unknown",
                "entry_close": _float(row.get("entry_close")),
                "returns": {},
            },
        )
        if _flag(row.get("measured")):
            entry["returns"][horizon] = _float(row.get("side_return_pct"))

    atrs = atr_by_symbol_day or {}
    currents = current_r or {}
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for (symbol, side, scan_day), entry in entries.items():
        cell = cells.setdefault(
            (entry["family"], side),
            {"family": entry["family"], "side": side, "stop_only": [], "trail": [],
             "current": [], UNKNOWN_NO_ATR: 0, UNKNOWN_NOT_MEASURED: 0},
        )
        returns = entry["returns"]
        if any(returns.get(h) is None for h in CHECKPOINTS):
            cell[UNKNOWN_NOT_MEASURED] += 1
            continue
        atr = atrs.get((symbol, scan_day))
        entry_close = entry["entry_close"]
        if atr is None or not entry_close or entry_close <= 0:
            cell[UNKNOWN_NO_ATR] += 1
            continue
        moves = {h: returns[h] / 100.0 * entry_close / atr for h in CHECKPOINTS}
        result = exit_model_r(moves)
        if result is None:
            cell[UNKNOWN_NOT_MEASURED] += 1
            continue
        cell["stop_only"].append(result["stop_only"])
        cell["trail"].append(result["trail"])
        tracker_r = currents.get((symbol, side, scan_day))
        if tracker_r is not None:
            cell["current"].append(tracker_r)

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    out_cells = []
    for cell in cells.values():
        out_cells.append(
            {
                "family": cell["family"],
                "side": cell["side"],
                "n": len(cell["stop_only"]),
                "current_r": mean(cell["current"]),
                "current_n": len(cell["current"]),
                "stop_only_r": mean(cell["stop_only"]),
                "trail_r": mean(cell["trail"]),
                "unknown_no_atr": cell[UNKNOWN_NO_ATR],
                "unknown_not_measured": cell[UNKNOWN_NOT_MEASURED],
            }
        )
    out_cells.sort(key=lambda c: (c["side"], -c["n"], c["family"]))
    scan_days = sorted({key[2] for key in entries})
    return {
        "cells": out_cells,
        "first": scan_days[0] if scan_days else "",
        "last": scan_days[-1] if scan_days else "",
    }


def review_sentence(summary: Mapping[str, Any] | None) -> str:
    """The tab's one status line. Formatting only."""
    summary = summary or {}
    cells = summary.get("cells") or []
    if not cells:
        return NO_DATA_SENTENCE
    measured = sum(int(c.get("n") or 0) for c in cells)
    no_atr = sum(int(c.get("unknown_no_atr") or 0) for c in cells)
    pending = sum(int(c.get("unknown_not_measured") or 0) for c in cells)
    return (
        f"Scan dates {summary.get('first')} to {summary.get('last')}: {measured} entries "
        f"measured; {pending} not yet 10 sessions (or a close missing), "
        f"{no_atr} with no ATR - unknown, never zero."
    )
