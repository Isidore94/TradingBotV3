#!/usr/bin/env python3
"""Packet ST3.3 - the immutable old-policy / repaired-policy comparison.

Replays each tradeable Setup Tracker record TWICE on COPIED inputs, once under
the shipped convention ``(literal_level_v1, same_session_v1)`` and once under
the repaired one ``(gap_aware_v2, prior_session_v2)``, and writes a stamped
``comparison_<stamp>.json`` + ``.csv`` pair naming, per setup, the
representative R under each policy, whether it changed and the fill bases; and,
per ``(side, setup_family, priority_bucket)``, n, changed n, expectancy, win
rate with the ONE Wilson lower bound (``swing_headline.WILSON_Z``), the tail
(min R, 5th-percentile R, count of R < -2) and the RANK IMPACT - the family
order by Wilson bound under each policy with the moves spelled out.

**The output of this tool is evidence for a decision the trader has not taken.**
It authorizes nothing. It promotes nothing. It never writes into the tracker,
the SQLite mirror, any export or any live store: the only directory it creates
a file in is ``--out``, and it REFUSES to run at all if ``--out`` or any input
path sits under ``C:\\TradingBotData`` (or if ``project_paths.DATA_DIR`` itself
resolves there, which is the scratch-script rule of 2026-09-05: run this with
``TRADINGBOTV3_DATA_DIR`` pointed at a scratch tree). A stamp is never
overwritten - a second run in the same second gets ``_2``.

Usage (from ``scripts/``)::

    python tracker_execution_compare.py \\
        --tracker <COPY of the tracker JSON, or a COPY of the .sqlite mirror> \\
        --bars <dir of <SYMBOL>.csv daily bars in the _daily_bar_cache_file shape> \\
        --out <scratch dir> [--limit N] [--sample-seed S] [--symbols A,B,C]

``--limit`` takes the first N tradeable setups in tracker order; adding
``--sample-seed`` makes it a deterministic random sample instead, so a bounded
run is reproducible rather than "whatever came first".
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

#: The live home folder. Nothing this tool touches may live under it. Kept as a
#: module constant so a test can point it at a scratch tree and exercise the
#: refusal without going anywhere near the real store.
PROTECTED_DATA_ROOT = Path(r"C:\TradingBotData")

#: The two policy pairs compared. Left is what ships and scores today.
POLICY_OLD = "old"
POLICY_NEW = "new"

#: R worse than this counts into the tail column. Named, not inlined.
TAIL_R_THRESHOLD = -2.0

DAILY_BAR_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def _stamped_paths(out_dir: Path, now: datetime) -> tuple[Path, Path]:
    """A ``comparison_<stamp>`` pair that does not exist yet.

    The stamp is second-resolution, so two runs inside one second would collide;
    the suffix loop is what makes "never overwrites" true rather than likely.
    """
    stamp = now.strftime("%Y%m%dT%H%M%S")
    suffix = 1
    while True:
        name = stamp if suffix == 1 else f"{stamp}_{suffix}"
        json_path = out_dir / f"comparison_{name}.json"
        csv_path = out_dir / f"comparison_{name}.csv"
        if not json_path.exists() and not csv_path.exists():
            return json_path, csv_path
        suffix += 1


def _load_tracker_setups(path: Path) -> dict[str, dict]:
    """``{setup_id: setup}`` from a tracker JSON copy or a mirror copy.

    The SQLite branch opens the file ``mode=ro&immutable=1``: a plain connect
    would create a WAL beside a copy and, worse, would be a writable handle on
    something the caller may have pointed at the real mirror by accident.
    """
    if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        try:
            rows = conn.execute(
                "SELECT key, payload FROM records WHERE section = 'setups' ORDER BY rowid"
            ).fetchall()
        finally:
            conn.close()
        return {str(key): json.loads(text) for key, text in rows}

    payload = json.loads(path.read_text(encoding="utf-8"))
    setups = payload.get("setups") if isinstance(payload, dict) else None
    if not isinstance(setups, dict):
        raise ValueError(f"{path} has no 'setups' mapping")
    return {str(key): value for key, value in setups.items() if isinstance(value, dict)}


def _load_bars(bars_dir: Path, symbol: str, frame_cache: dict[str, Any]):
    import pandas as pd

    key = str(symbol or "").strip().upper()
    if key in frame_cache:
        return frame_cache[key]
    path = bars_dir / f"{key}.csv"
    frame = None
    if path.exists():
        try:
            frame = pd.read_csv(path, parse_dates=["datetime"])
        except Exception:
            frame = None
    if frame is not None:
        missing = [column for column in DAILY_BAR_COLUMNS if column not in frame.columns]
        if missing:
            frame = None
    if frame is not None:
        frame = frame[DAILY_BAR_COLUMNS].sort_values("datetime").reset_index(drop=True)
        if frame.empty:
            frame = None
    frame_cache[key] = frame
    return frame


def _representative_r(record: dict) -> float | None:
    import master_avwap as m

    probe = dict(record)
    # `_summarize_tracker_setup_outcome` short-circuits on a cached summary the
    # tracker wrote under the OLD policy; reading it would compare a policy
    # against itself.
    probe.pop("_scoring_outcome_summary", None)
    summary = m._summarize_tracker_setup_outcome(probe)
    value = summary.get("representative_total_r")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fill_bases(record: dict) -> list[str]:
    bases: list[str] = []
    for scenario in (record.get("scenarios") or {}).values():
        if not isinstance(scenario, dict):
            continue
        for event in scenario.get("events") or []:
            if isinstance(event, dict) and event.get("fill_basis"):
                bases.append(str(event["fill_basis"]))
    return sorted(set(bases))


def _event_reasons(record: dict) -> list[str]:
    reasons: list[str] = []
    for scenario in (record.get("scenarios") or {}).values():
        if not isinstance(scenario, dict):
            continue
        for event in scenario.get("events") or []:
            if isinstance(event, dict) and event.get("reason"):
                reasons.append(str(event["reason"]))
    return reasons


def _skip_reasons(record: dict) -> dict[str, int]:
    totals: dict[str, int] = {}
    for scenario in (record.get("scenarios") or {}).values():
        if not isinstance(scenario, dict):
            continue
        skips = scenario.get("intrabar_skip_reasons")
        if isinstance(skips, dict):
            for reason, count in skips.items():
                totals[str(reason)] = totals.get(str(reason), 0) + int(count or 0)
    return totals


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _cell_stats(values: list[float]) -> dict[str, Any]:
    """One policy's numbers for one group. `n` and the bound lead, per V3."""
    from swing_headline import WILSON_Z, wilson_lower_bound

    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "wins": 0,
            "win_rate": None,
            "win_rate_wilson_lower": None,
            "wilson_z": WILSON_Z,
            "expectancy_r": None,
            "min_r": None,
            "p5_r": None,
            "n_r_below_minus_2": 0,
        }
    wins = sum(1 for value in values if value > 0.0)
    return {
        "n": n,
        "wins": wins,
        "win_rate": wins / n,
        "win_rate_wilson_lower": wilson_lower_bound(wins, n),
        "wilson_z": WILSON_Z,
        "expectancy_r": statistics.fmean(values),
        "min_r": min(values),
        "p5_r": _percentile(values, 0.05),
        "n_r_below_minus_2": sum(1 for value in values if value < TAIL_R_THRESHOLD),
    }


def _rank_by_bound(groups: list[dict], policy: str) -> dict[str, int]:
    """Family order by the Wilson lower bound; an ungraded cell ranks last."""

    def sort_key(group: dict):
        cell = group[policy]
        bound = cell.get("win_rate_wilson_lower")
        graded = bound is not None
        return (
            0 if graded else 1,
            -(bound if graded else 0.0),
            -(cell.get("expectancy_r") or 0.0),
            group["group_id"],
        )

    ordered = sorted(groups, key=sort_key)
    return {group["group_id"]: index + 1 for index, group in enumerate(ordered)}


def _select_setups(
    setups: dict[str, dict],
    *,
    limit: int | None,
    seed: int | None,
    symbols: Iterable[str] | None,
) -> list[tuple[str, dict]]:
    wanted = {str(s).strip().upper() for s in (symbols or []) if str(s).strip()}
    rows = [
        (setup_id, setup)
        for setup_id, setup in setups.items()
        if isinstance(setup, dict)
        and (not wanted or str(setup.get("symbol") or "").strip().upper() in wanted)
    ]
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def _replay(setup: dict, frame, convention: str, knowledge: str) -> dict:
    import master_avwap as m

    return m.recompute_tracker_setup_record(
        copy.deepcopy(setup),
        frame,
        execution_convention=convention,
        level_knowledge=knowledge,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracker_execution_compare",
        description=(
            "Replay copied Setup Tracker records under the shipped and the "
            "repaired execution conventions and write a stamped comparison. "
            "Evidence only - it authorizes nothing."
        ),
    )
    parser.add_argument("--tracker", required=True, help="COPY of the tracker JSON or the .sqlite mirror")
    parser.add_argument("--bars", required=True, help="directory of <SYMBOL>.csv daily bars")
    parser.add_argument("--out", required=True, help="output directory (never under the live store)")
    parser.add_argument("--limit", type=int, default=None, help="replay at most N tradeable setups")
    parser.add_argument("--sample-seed", type=int, default=None, help="deterministic random sample")
    parser.add_argument("--symbols", default="", help="comma-separated symbol filter")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import project_paths

    print(f"project_paths.DATA_DIR = {project_paths.DATA_DIR}")

    tracker_path = Path(args.tracker).expanduser()
    bars_dir = Path(args.bars).expanduser()
    out_dir = Path(args.out).expanduser()

    # The scratch-script rule (2026-09-05), enforced rather than remembered.
    if _is_under(Path(project_paths.DATA_DIR), PROTECTED_DATA_ROOT):
        print(
            f"REFUSED: project_paths.DATA_DIR resolves under {PROTECTED_DATA_ROOT}. "
            "Set TRADINGBOTV3_DATA_DIR to a scratch directory before running this.",
            file=sys.stderr,
        )
        return 2
    for label, path in (("--tracker", tracker_path), ("--bars", bars_dir), ("--out", out_dir)):
        if _is_under(path, PROTECTED_DATA_ROOT):
            print(
                f"REFUSED: {label} {path} is under the live store {PROTECTED_DATA_ROOT}. "
                "Copy the input and write the output to a scratch directory.",
                file=sys.stderr,
            )
            return 2

    if not tracker_path.exists():
        print(f"REFUSED: --tracker {tracker_path} does not exist", file=sys.stderr)
        return 2
    if not bars_dir.is_dir():
        print(f"REFUSED: --bars {bars_dir} is not a directory", file=sys.stderr)
        return 2

    import master_avwap as m
    from master_avwap_lib import execution_convention as ec

    setups = _load_tracker_setups(tracker_path)
    selected = _select_setups(
        setups,
        limit=args.limit,
        seed=args.sample_seed,
        symbols=[part for part in str(args.symbols or "").split(",") if part.strip()],
    )
    print(f"tracker holds {len(setups)} setup(s); replaying {len(selected)}")

    frame_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    skipped_no_bars = 0
    skipped_untradeable = 0
    for setup_id, setup in selected:
        symbol = str(setup.get("symbol") or "").strip().upper()
        scenarios = setup.get("scenarios") or {}
        if not any(bool(s.get("tradeable")) for s in scenarios.values() if isinstance(s, dict)):
            skipped_untradeable += 1
            continue
        frame = _load_bars(bars_dir, symbol, frame_cache)
        if frame is None:
            skipped_no_bars += 1
            continue
        try:
            old_record = _replay(setup, frame, ec.EXECUTION_LITERAL_LEVEL_V1, ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1)
            new_record = _replay(setup, frame, ec.EXECUTION_GAP_AWARE_V2, ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2)
        except Exception as exc:  # a bad record never stops the comparison
            print(f"  {setup_id}: replay failed ({exc})", file=sys.stderr)
            continue
        old_r = _representative_r(old_record)
        new_r = _representative_r(new_record)
        changed = (old_r is None) != (new_r is None) or (
            old_r is not None and new_r is not None and abs(old_r - new_r) > 1e-9
        )
        skips = _skip_reasons(new_record)
        rows.append(
            {
                "setup_id": setup_id,
                "symbol": symbol,
                "side": str(m.normalize_side(setup.get("side")) or ""),
                "setup_family": str(setup.get("setup_family") or ""),
                "priority_bucket": str(setup.get("priority_bucket") or ""),
                "entry_trade_date": str(setup.get("entry_trade_date") or setup.get("scan_date") or ""),
                "r_old": old_r,
                "r_new": new_r,
                "r_delta": (new_r - old_r) if (old_r is not None and new_r is not None) else None,
                "changed": changed,
                "status_old": str(old_record.get("setup_status") or ""),
                "status_new": str(new_record.get("setup_status") or ""),
                "reasons_old": "|".join(_event_reasons(old_record)),
                "reasons_new": "|".join(_event_reasons(new_record)),
                "fill_bases_new": "|".join(_fill_bases(new_record)),
                "no_prior_session_level": int(skips.get(ec.NO_PRIOR_SESSION_LEVEL, 0)),
            }
        )

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["side"], row["setup_family"], row["priority_bucket"]), []).append(row)

    group_rows: list[dict[str, Any]] = []
    for (side, family, bucket), members in sorted(groups.items()):
        old_values = [row["r_old"] for row in members if row["r_old"] is not None]
        new_values = [row["r_new"] for row in members if row["r_new"] is not None]
        group_rows.append(
            {
                "group_id": f"{side}|{family}|{bucket}",
                "side": side,
                "setup_family": family,
                "priority_bucket": bucket,
                "n_setups": len(members),
                "n_changed": sum(1 for row in members if row["changed"]),
                POLICY_OLD: _cell_stats(old_values),
                POLICY_NEW: _cell_stats(new_values),
            }
        )

    old_ranks = _rank_by_bound(group_rows, POLICY_OLD)
    new_ranks = _rank_by_bound(group_rows, POLICY_NEW)
    for group in group_rows:
        group["rank_old"] = old_ranks[group["group_id"]]
        group["rank_new"] = new_ranks[group["group_id"]]
        group["rank_move"] = group["rank_old"] - group["rank_new"]

    all_old = [row["r_old"] for row in rows if row["r_old"] is not None]
    all_new = [row["r_new"] for row in rows if row["r_new"] is not None]
    headline = {
        "n_setups": len(rows),
        "n_changed": sum(1 for row in rows if row["changed"]),
        "n_skipped_no_bars": skipped_no_bars,
        "n_skipped_untradeable": skipped_untradeable,
        "n_groups": len(group_rows),
        "n_groups_rank_moved": sum(1 for group in group_rows if group["rank_move"] != 0),
        "max_rank_move": max((abs(group["rank_move"]) for group in group_rows), default=0),
        POLICY_OLD: _cell_stats(all_old),
        POLICY_NEW: _cell_stats(all_new),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = _stamped_paths(out_dir, datetime.now())
    payload = {
        "schema": "tracker_execution_compare/1",
        "written_at": datetime.now().astimezone().isoformat(),
        "authorization": (
            "EVIDENCE ONLY. This comparison is not authorization to overwrite live "
            "historical results, to restate the tracker, or to promote gap_aware_v2 / "
            "prior_session_v2 into scoring. The trader's decision is separate."
        ),
        "policies": {
            POLICY_OLD: {
                "execution_convention": ec.EXECUTION_LITERAL_LEVEL_V1,
                "level_knowledge": ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
                "note": "what ships and scores today",
            },
            POLICY_NEW: {
                "execution_convention": ec.EXECUTION_GAP_AWARE_V2,
                "level_knowledge": ec.LEVEL_KNOWLEDGE_PRIOR_SESSION_V2,
                "note": "the repair, shadow only",
            },
        },
        "inputs": {
            "tracker": str(tracker_path),
            "bars_dir": str(bars_dir),
            "limit": args.limit,
            "sample_seed": args.sample_seed,
            "symbols": str(args.symbols or ""),
            "data_dir": str(project_paths.DATA_DIR),
        },
        "headline": headline,
        "groups": group_rows,
        "setups": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    field_names = list(rows[0].keys()) if rows else [
        "setup_id", "symbol", "side", "setup_family", "priority_bucket",
        "entry_trade_date", "r_old", "r_new", "r_delta", "changed",
        "status_old", "status_new", "reasons_old", "reasons_new",
        "fill_bases_new", "no_prior_session_level",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(
        f"changed {headline['n_changed']} of {headline['n_setups']}; "
        f"expectancy {headline[POLICY_OLD]['expectancy_r']} -> {headline[POLICY_NEW]['expectancy_r']}; "
        f"min R {headline[POLICY_OLD]['min_r']} -> {headline[POLICY_NEW]['min_r']}; "
        f"R < -2 {headline[POLICY_OLD]['n_r_below_minus_2']} -> {headline[POLICY_NEW]['n_r_below_minus_2']}; "
        f"{headline['n_groups_rank_moved']} of {headline['n_groups']} groups moved rank"
    )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
