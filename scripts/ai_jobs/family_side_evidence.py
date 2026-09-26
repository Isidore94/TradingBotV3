"""The `family_side_evidence` night slot (S12): deterministic, no model.

Reads the session-horizon outcomes, SPY's daily bars, `d1_features_history.csv`
(atr20 and the live priority score), the attribute leaderboard and the tracker's
scoring snapshot; builds `points_challenger` evidence per (setup family, side)
over the trailing 40 sessions plus the SP4 shadow trial; replaces
`FAMILY_SIDE_EVIDENCE_FILE`. Fewer than 15 sessions writes nothing; a failed read
or build returns `failed` and the last good file stays.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

_log = logging.getLogger(__name__)

HORIZON_COLUMNS = (
    "symbol", "side", "scan_date", "target_session", "horizon_sessions", "side_return_pct",
    "entry_close", "measured", "maturity", "setup_family", "outcome_kind",
)
FEATURE_COLUMNS = ("run_date", "run_timestamp", "last_trade_date", "symbol", "side", "atr20", "priority_score")
FEATURE_CHUNK_ROWS = 200_000


def read_horizon_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return [{name: row.get(name) or "" for name in HORIZON_COLUMNS} for row in csv.DictReader(handle)]


def read_feature_facts(path: Path, *, first_day: str) -> tuple[dict, dict]:
    """``({(SYMBOL, day): atr20}, {(SYMBOL, SIDE, day): priority_score})``, last run of each day wins."""
    import pandas as pd

    atr: dict[tuple[str, str], float] = {}
    scores: dict[tuple[str, str, str], float] = {}
    reader = pd.read_csv(path, usecols=list(FEATURE_COLUMNS), dtype=str, chunksize=FEATURE_CHUNK_ROWS)
    for chunk in reader:
        day = chunk["last_trade_date"].fillna(chunk["run_date"]).fillna("").str[:10]
        chunk = chunk.assign(_day=day)
        chunk = chunk[chunk["_day"] >= first_day].sort_values("run_timestamp", kind="stable")
        for symbol, side, key_day, atr20, score in zip(
            chunk["symbol"], chunk["side"], chunk["_day"], chunk["atr20"], chunk["priority_score"], strict=False
        ):
            sym = str(symbol or "").strip().upper()
            try:
                value = float(atr20)
                if value == value and value > 0:
                    atr[(sym, key_day)] = value
            except (TypeError, ValueError):
                pass
            try:
                value = float(score)
                if value == value:
                    scores[(sym, str(side or "").strip().upper(), key_day)] = value
            except (TypeError, ValueError):
                pass
    return atr, scores


def read_leaderboard_family_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    try:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if (row.get("attribute_key") or "").strip() == "setup.setup_family":
                    rows.append(row)
    except OSError:
        return []
    return rows


def read_tracker_pick_r(path: Path) -> dict[tuple[str, str, str], float | None]:
    """``{(SYMBOL, SIDE, scan day): representative closed R or None}``; {} when unreadable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    setups = payload.get("setups") if isinstance(payload, dict) else None
    if not isinstance(setups, dict):
        return {}
    import looking_back

    return {
        (str(pick.get("symbol") or "").upper(), str(pick.get("side") or "").upper(), str(pick.get("session") or "")[:10]):
            pick.get("r")
        for pick in looking_back.swing_pick_results(setups)
    }


def read_payload(path: Path | None = None) -> dict[str, Any]:
    """The last published evidence, `{}` when absent or unreadable. Never on the Qt thread."""
    import points_challenger

    if path is None:
        from project_paths import FAMILY_SIDE_EVIDENCE_FILE

        path = FAMILY_SIDE_EVIDENCE_FILE
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) and payload.get("schema") == points_challenger.SCHEMA else {}


def write_payload(payload: Mapping[str, Any], path: Path) -> Path:
    """Temp file then replace, so a half-written file never replaces the last good one."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temp, target)
    return target


def build_payload(
    horizon_rows: list[Mapping[str, Any]],
    spy_closes: Mapping[str, float],
    atr_by_day: Mapping[tuple[str, str], float],
    scores: Mapping[tuple[str, str, str], float],
    leaderboard_rows: Iterable[Mapping[str, Any]],
    tracker_pick_r: Mapping[tuple[str, str, str], float | None],
    prior: Mapping[str, Any] | None,
    *,
    as_of: str,
    generated_at: str = "",
) -> dict[str, Any] | None:
    """The whole file, or None when the window has fewer than MIN_WINDOW_SESSIONS sessions."""
    import points_challenger as pc

    cells, days = pc.build_families(
        list(horizon_rows), spy_closes, atr_by_day, pc.tracker_r_by_family(leaderboard_rows), as_of=as_of
    )
    if len(days) < pc.MIN_WINDOW_SESSIONS:
        return None
    history = dict((prior or {}).get("adjust_history") or {})
    history[as_of] = {key: cell["adjust"] for key, cell in cells.items()}
    history = {day: history[day] for day in sorted(history)[-pc.ADJUST_HISTORY_KEEP:]}
    trial = pc.trial_summary(list(horizon_rows), spy_closes, scores, tracker_pick_r, history, as_of=as_of)
    return {
        "schema": pc.SCHEMA,
        "as_of": as_of,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": {"first": days[0], "last": days[-1], "sessions": len(days)},
        "rule": {
            "adjust": "60 x (beat_low_h5 - 0.50) + 20 x mean_move_atr_h10, clamped to [-40, +40]",
            "min_n": pc.MIN_N, "min_sessions": pc.MIN_SESSIONS,
            "trial_entry_sessions": pc.TRIAL_ENTRY_SESSIONS, "mature_sessions": pc.TRIAL_MATURE_SESSIONS,
        },
        "families": cells,
        "adjust_history": history,
        "trial": trial,
        "saturday_lines": pc.saturday_lines(trial),
    }


def run_family_side_evidence(
    *,
    session_date: str = "",
    horizon_path: Any = None,
    features_path: Any = None,
    leaderboard_path: Any = None,
    snapshot_path: Any = None,
    spy_path: Any = None,
    out_path: Any = None,
    **_ignored: Any,
) -> dict[str, Any]:
    import points_challenger as pc
    import project_paths as pp
    from setup_permutation_backfill import read_spy_closes

    horizon_path = Path(horizon_path or pp.MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE)
    features_path = Path(features_path or pp.D1_FEATURES_HISTORY_FILE)
    leaderboard_path = Path(leaderboard_path or pp.MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE)
    snapshot_path = Path(snapshot_path or pp.MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE)
    spy_path = Path(spy_path or Path(pp.MASTER_AVWAP_DAILY_BARS_DIR) / "SPY.parquet")
    out_path = Path(out_path or pp.FAMILY_SIDE_EVIDENCE_FILE)
    try:
        horizon_rows = read_horizon_rows(horizon_path)
        as_of = str(session_date or "")[:10] or max(
            (str(r.get("scan_date") or "")[:10] for r in horizon_rows), default=""
        )
        days = pc.window_days(horizon_rows, as_of=as_of)
        if len(days) < pc.MIN_WINDOW_SESSIONS:
            return {
                "status": "ok", "model": "",
                "reason": f"only {len(days)} session(s) of outcomes (< {pc.MIN_WINDOW_SESSIONS}); nothing written",
                "outputs": [],
            }
        atr, scores = read_feature_facts(features_path, first_day=days[0])
        payload = build_payload(
            horizon_rows, read_spy_closes(spy_path), atr, scores,
            read_leaderboard_family_rows(leaderboard_path), read_tracker_pick_r(snapshot_path),
            read_payload(out_path), as_of=as_of,
        )
    except Exception as exc:  # noqa: BLE001 - the night goes on; the last file stays
        _log.exception("family_side_evidence: build failed")
        return {
            "status": "failed", "model": "",
            "reason": f"family evidence not built ({type(exc).__name__}: {exc}); last file kept",
            "outputs": [],
        }
    if payload is None:
        return {
            "status": "ok", "model": "",
            "reason": f"fewer than {pc.MIN_WINDOW_SESSIONS} sessions of outcomes; nothing written",
            "outputs": [],
        }
    try:
        written = write_payload(payload, out_path)
    except OSError as exc:
        return {
            "status": "failed", "model": "",
            "reason": f"family evidence not written ({exc}); last file kept",
            "outputs": [],
        }
    moved = sum(1 for cell in payload["families"].values() if cell["adjust"])
    return {
        "status": "ok", "model": "",
        "reason": (
            f"{len(payload['families'])} family/side cells over {payload['window']['sessions']} sessions, "
            f"{moved} with an SP4 adjust; trial {len(payload['trial']['entry_sessions'])} entry session(s)"
        ),
        "outputs": [str(written)],
    }
