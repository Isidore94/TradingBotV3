"""Daily snapshot and forward-return tracking for trader-curated Focus Picks.

Plain Python so both the headless Master AVWAP scan and GUI helpers can use the
same files. This keeps the human-pick cohort separate from bot setup-tracker
aggregates.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from focus_picks import load_focus_map
from market_session import get_market_session_window
from project_paths import (
    HUMAN_FOCUS_DAILY_PICKS_FILE,
    HUMAN_FOCUS_OUTCOMES_FILE,
    HUMAN_FOCUS_PERFORMANCE_FILE,
    HUMAN_FOCUS_SNAPSHOT_STATE_FILE,
    MASTER_AVWAP_DAILY_BARS_DIR,
)


HORIZONS = (1, 3, 5, 10)
HUMAN_FOCUS_DAILY_PICK_COLUMNS = [
    "trade_date",
    "symbol",
    "side",
    "source",
    "snapshotted_at",
    "active_at_snapshot",
]
HUMAN_FOCUS_OUTCOME_COLUMNS = [
    "trade_date",
    "symbol",
    "side",
    "source",
    "entry_date",
    "entry_close",
    "h1_date",
    "h1_return",
    "h3_date",
    "h3_return",
    "h5_date",
    "h5_return",
    "h10_date",
    "h10_return",
    "matured_horizons",
    "fully_matured",
    "updated_at",
]
HUMAN_FOCUS_PERFORMANCE_COLUMNS = [
    "cohort",
    "side",
    "horizon_sessions",
    "sample_count",
    "win_rate",
    "avg_side_return",
    "profit_factor",
    "updated_at",
]


def _market_date(value: Any = None) -> date:
    if value is None:
        return get_market_session_window().market_date
    parsed = _parse_date(value)
    return parsed or get_market_session_window().market_date


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def _now_text(now: datetime | None = None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return data if data is not None else default


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def _write_csv_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
        os.replace(tmp, path)
    except OSError:
        pass


def _side_label(side: Any) -> str:
    text = str(side or "").strip().upper()
    return "SHORT" if text.startswith("SHORT") else "LONG"


def _pick_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("trade_date") or "").strip(),
        str(row.get("symbol") or "").strip().upper(),
        _side_label(row.get("side")),
    )


def load_human_focus_daily_picks(
    *,
    trade_date: Any = None,
    path: Path = HUMAN_FOCUS_DAILY_PICKS_FILE,
) -> list[dict[str, str]]:
    rows = _read_csv_rows(Path(path))
    target_date = _parse_date(trade_date) if trade_date is not None else None
    if target_date is None:
        return rows
    target_text = target_date.isoformat()
    return [row for row in rows if str(row.get("trade_date") or "").strip() == target_text]


def load_human_focus_map_for_date(
    trade_date: Any,
    *,
    path: Path = HUMAN_FOCUS_DAILY_PICKS_FILE,
) -> dict[str, set[str]]:
    rows = load_human_focus_daily_picks(trade_date=trade_date, path=path)
    focus = {"long": set(), "short": set()}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        side = _side_label(row.get("side"))
        focus["short" if side == "SHORT" else "long"].add(symbol)
    return focus


def mark_human_focus_rows(
    priority_rows: list[dict[str, Any]],
    feature_rows_by_symbol: dict[str, dict[str, Any]] | None = None,
    *,
    focus_map: dict[str, set[str] | list[str]] | None = None,
    trade_date: Any = None,
) -> int:
    focus = focus_map if focus_map is not None else load_human_focus_map_for_date(_market_date(trade_date))
    long_symbols = {str(symbol or "").strip().upper() for symbol in (focus.get("long") or []) if str(symbol or "").strip()}
    short_symbols = {str(symbol or "").strip().upper() for symbol in (focus.get("short") or []) if str(symbol or "").strip()}
    marked = 0
    for row in priority_rows or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        side = _side_label(row.get("side"))
        is_focus = bool((side == "LONG" and symbol in long_symbols) or (side == "SHORT" and symbol in short_symbols))
        row["human_focus_pick"] = is_focus
        row["human_focus_side"] = side if is_focus else ""
        if is_focus:
            marked += 1
            if isinstance(feature_rows_by_symbol, dict):
                feature_row = feature_rows_by_symbol.get(symbol)
                if isinstance(feature_row, dict):
                    feature_row["human_focus_pick"] = True
                    feature_row["human_focus_side"] = side
    return marked


def snapshot_human_focus_picks(
    *,
    market_date: Any = None,
    focus_map: dict[str, set[str] | list[str]] | None = None,
    force: bool = False,
    now: datetime | None = None,
    snapshot_state_path: Path = HUMAN_FOCUS_SNAPSHOT_STATE_FILE,
    daily_picks_path: Path = HUMAN_FOCUS_DAILY_PICKS_FILE,
) -> dict[str, Any]:
    """Snapshot focus_longs/focus_shorts into the dated human-pick cohort.

    With force=False this runs once per market date. With force=True it merges
    current focus names for the same date without duplicating existing rows.
    """
    trade_date = _market_date(market_date)
    trade_date_text = trade_date.isoformat()
    state = _read_json(Path(snapshot_state_path), default={})
    if not isinstance(state, dict):
        state = {}
    already_snapshotted = str(state.get("last_snapshot_market_date") or "") == trade_date_text
    if already_snapshotted and not force:
        return {
            "snapshotted": False,
            "reason": "already_snapshotted",
            "trade_date": trade_date_text,
            "added": 0,
            "total_for_date": len(load_human_focus_daily_picks(trade_date=trade_date, path=daily_picks_path)),
        }

    focus = focus_map if focus_map is not None else load_focus_map()
    longs = sorted({str(symbol or "").strip().upper() for symbol in (focus.get("long") or []) if str(symbol or "").strip()})
    shorts = sorted({str(symbol or "").strip().upper() for symbol in (focus.get("short") or []) if str(symbol or "").strip()})
    existing_rows = _read_csv_rows(Path(daily_picks_path))
    rows_by_key = {_pick_key(row): dict(row) for row in existing_rows if _pick_key(row)[0] and _pick_key(row)[1]}
    timestamp = _now_text(now)
    added = 0
    for side, symbols in (("LONG", longs), ("SHORT", shorts)):
        for symbol in symbols:
            key = (trade_date_text, symbol, side)
            if key in rows_by_key:
                continue
            rows_by_key[key] = {
                "trade_date": trade_date_text,
                "symbol": symbol,
                "side": side,
                "source": "focus_pick",
                "snapshotted_at": timestamp,
                "active_at_snapshot": "1",
            }
            added += 1

    rows = sorted(rows_by_key.values(), key=lambda row: (_pick_key(row)[0], _pick_key(row)[2], _pick_key(row)[1]))
    _write_csv_rows(Path(daily_picks_path), HUMAN_FOCUS_DAILY_PICK_COLUMNS, rows)
    state.update(
        {
            "last_snapshot_market_date": trade_date_text,
            "last_snapshot_at": timestamp,
            "last_snapshot_count": len([row for row in rows if _pick_key(row)[0] == trade_date_text]),
        }
    )
    _write_json_atomic(Path(snapshot_state_path), state)
    return {
        "snapshotted": True,
        "reason": "forced" if force and already_snapshotted else "new_market_date",
        "trade_date": trade_date_text,
        "added": added,
        "total_for_date": state["last_snapshot_count"],
    }


def _sanitize_symbol_for_filename(symbol: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {".", "-"} else "_" for ch in str(symbol or "").strip().upper())
    return cleaned or "UNKNOWN"


def _load_durable_daily_frame(symbol: str, daily_bars_dir: Path = MASTER_AVWAP_DAILY_BARS_DIR) -> pd.DataFrame:
    path = Path(daily_bars_dir) / f"{_sanitize_symbol_for_filename(symbol)}.parquet"
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _normalize_daily_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame(columns=["datetime", "close"])
    work = frame.copy()
    work.rename(columns={column: str(column).strip().lower() for column in work.columns}, inplace=True)
    if "datetime" not in work.columns:
        for candidate in ("date", "time", "timestamp"):
            if candidate in work.columns:
                work["datetime"] = work[candidate]
                break
    if "datetime" not in work.columns or "close" not in work.columns:
        return pd.DataFrame(columns=["datetime", "close"])
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce").dt.tz_localize(None)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna(subset=["datetime", "close"]).sort_values("datetime")
    work = work.drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    return work[["datetime", "close"]]


def _frame_for_symbol(
    symbol: str,
    daily_frames_by_symbol: dict[str, pd.DataFrame] | None,
    daily_bars_dir: Path,
) -> pd.DataFrame:
    symbol = str(symbol or "").strip().upper()
    if isinstance(daily_frames_by_symbol, dict):
        frame = daily_frames_by_symbol.get(symbol)
        if frame is None:
            frame = daily_frames_by_symbol.get(symbol.lower())
        if frame is not None:
            return _normalize_daily_frame(frame)
    return _normalize_daily_frame(_load_durable_daily_frame(symbol, daily_bars_dir))


def _side_adjusted_return(side: str, entry_close: float, horizon_close: float) -> float:
    if entry_close <= 0:
        return 0.0
    if _side_label(side) == "SHORT":
        return (entry_close - horizon_close) / entry_close
    return (horizon_close - entry_close) / entry_close


def _compute_pick_outcome(
    pick: dict[str, Any],
    frame: pd.DataFrame,
    *,
    updated_at: str,
) -> dict[str, Any] | None:
    trade_date = _parse_date(pick.get("trade_date"))
    symbol = str(pick.get("symbol") or "").strip().upper()
    side = _side_label(pick.get("side"))
    if trade_date is None or not symbol or frame is None or getattr(frame, "empty", True):
        return None
    # Accept an already-normalized frame (with a cached ``trade_day`` column) to avoid
    # re-normalizing the same symbol once per pick; fall back to normalizing raw frames.
    work = frame
    if "trade_day" not in work.columns:
        work = _normalize_daily_frame(work)
        if work.empty:
            return None
        work = work.assign(trade_day=work["datetime"].dt.date)
    candidates = work[work["trade_day"] >= trade_date].reset_index(drop=True)
    if candidates.empty:
        return None
    entry_close = float(candidates.loc[0, "close"])
    entry_date = candidates.loc[0, "trade_day"]
    row: dict[str, Any] = {
        "trade_date": trade_date.isoformat(),
        "symbol": symbol,
        "side": side,
        "source": str(pick.get("source") or "focus_pick").strip() or "focus_pick",
        "entry_date": entry_date.isoformat(),
        "entry_close": f"{entry_close:.4f}",
        "updated_at": updated_at,
    }
    matured: list[str] = []
    for horizon in HORIZONS:
        date_column = f"h{horizon}_date"
        return_column = f"h{horizon}_return"
        if len(candidates) <= horizon:
            row[date_column] = ""
            row[return_column] = ""
            continue
        horizon_close = float(candidates.loc[horizon, "close"])
        horizon_date = candidates.loc[horizon, "trade_day"]
        row[date_column] = horizon_date.isoformat()
        row[return_column] = f"{_side_adjusted_return(side, entry_close, horizon_close):.6f}"
        matured.append(str(horizon))
    row["matured_horizons"] = ",".join(matured)
    row["fully_matured"] = "1" if "10" in matured else "0"
    return row


def update_human_focus_outcomes(
    *,
    reference_date: Any = None,
    daily_frames_by_symbol: dict[str, pd.DataFrame] | None = None,
    daily_picks_path: Path = HUMAN_FOCUS_DAILY_PICKS_FILE,
    outcomes_path: Path = HUMAN_FOCUS_OUTCOMES_FILE,
    performance_path: Path = HUMAN_FOCUS_PERFORMANCE_FILE,
    daily_bars_dir: Path = MASTER_AVWAP_DAILY_BARS_DIR,
    recent_calendar_days: int = 45,
    now: datetime | None = None,
) -> dict[str, Any]:
    reference = _market_date(reference_date)
    daily_bars_dir = Path(daily_bars_dir)
    picks = _read_csv_rows(Path(daily_picks_path))
    existing = {_pick_key(row): dict(row) for row in _read_csv_rows(Path(outcomes_path))}
    updated_at = _now_text(now)
    updated_count = 0
    stale_before = reference - timedelta(days=recent_calendar_days)
    # One normalized frame per symbol per scan: the same focus name is snapshotted
    # across many trade dates, so resolving/normalizing once avoids redundant parquet
    # reads and DataFrame work for every pick row sharing that symbol.
    normalized_frames: dict[str, pd.DataFrame] = {}

    def _resolve_frame(symbol: str) -> pd.DataFrame:
        cached = normalized_frames.get(symbol)
        if cached is None:
            frame = _frame_for_symbol(symbol, daily_frames_by_symbol, daily_bars_dir)
            if not frame.empty:
                frame = frame.assign(trade_day=frame["datetime"].dt.date)
            normalized_frames[symbol] = frame
            cached = frame
        return cached

    for pick in picks:
        pick_date = _parse_date(pick.get("trade_date"))
        key = _pick_key(pick)
        if pick_date is None or not key[1]:
            continue
        existing_row = existing.get(key, {})
        # A fully matured pick (all 10 forward sessions recorded) never changes, so keep
        # its existing row instead of re-reading bars and recomputing it every scan.
        if str(existing_row.get("fully_matured") or "") in {"1", "true", "True"}:
            continue
        if pick_date < stale_before:
            continue
        outcome = _compute_pick_outcome(pick, _resolve_frame(key[1]), updated_at=updated_at)
        if outcome is None:
            continue
        existing[key] = outcome
        updated_count += 1

    outcome_rows = sorted(existing.values(), key=lambda row: (_pick_key(row)[0], _pick_key(row)[2], _pick_key(row)[1]))
    _write_csv_rows(Path(outcomes_path), HUMAN_FOCUS_OUTCOME_COLUMNS, outcome_rows)
    performance_rows = build_human_focus_performance_rows(outcome_rows, updated_at=updated_at)
    _write_csv_rows(Path(performance_path), HUMAN_FOCUS_PERFORMANCE_COLUMNS, performance_rows)
    return {
        "outcome_rows": len(outcome_rows),
        "updated_outcomes": updated_count,
        "performance_rows": len(performance_rows),
    }


def _coerce_return(value: Any) -> float | None:
    try:
        text = str(value or "").strip()
        return float(text) if text else None
    except ValueError:
        return None


def _profit_factor(values: list[float]) -> str:
    gains = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value < 0))
    if losses <= 0:
        return "" if gains <= 0 else "inf"
    return f"{gains / losses:.4f}"


def build_human_focus_performance_rows(
    outcome_rows: list[dict[str, Any]],
    *,
    updated_at: str | None = None,
) -> list[dict[str, Any]]:
    timestamp = updated_at or _now_text()
    rows: list[dict[str, Any]] = []
    for side_filter in ("ALL", "LONG", "SHORT"):
        filtered = [
            row
            for row in outcome_rows
            if side_filter == "ALL" or _side_label(row.get("side")) == side_filter
        ]
        for horizon in HORIZONS:
            values = [
                value
                for value in (_coerce_return(row.get(f"h{horizon}_return")) for row in filtered)
                if value is not None
            ]
            if not values:
                continue
            sample_count = len(values)
            wins = len([value for value in values if value > 0])
            rows.append(
                {
                    "cohort": "human_focus_pick",
                    "side": side_filter,
                    "horizon_sessions": str(horizon),
                    "sample_count": str(sample_count),
                    "win_rate": f"{wins / sample_count:.4f}",
                    "avg_side_return": f"{sum(values) / sample_count:.6f}",
                    "profit_factor": _profit_factor(values),
                    "updated_at": timestamp,
                }
            )
    return rows


def update_human_focus_tracking(
    *,
    market_date: Any = None,
    daily_frames_by_symbol: dict[str, pd.DataFrame] | None = None,
    force_snapshot: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    trade_date = _market_date(market_date)
    snapshot = snapshot_human_focus_picks(market_date=trade_date, force=force_snapshot, now=now)
    outcomes = update_human_focus_outcomes(
        reference_date=trade_date,
        daily_frames_by_symbol=daily_frames_by_symbol,
        now=now,
    )
    return {
        "snapshot": snapshot,
        "outcomes": outcomes,
    }
