"""S15 items 3 and 9: swing path facts and fill-model columns. Shadow research sidecar.

Reads the session-horizon outcomes file (`master_avwap_session_horizon_outcomes.csv`,
which is close-to-close only), the daily bars and the scan history's ATR, and writes
one row per ``(scan_row_id, horizon)`` for horizons 1, 3, 5, 10 and 20. It joins the
horizons file on ``observation_id``; it never edits that file or the scan.

Per row (all side-adjusted, 1 ATR = the scan row's ``atr20`` on the scan date):

* ``mfe_atr`` / ``mae_atr`` - best / worst excursion from the entry close over the
  sessions after entry up to the target session, from each bar's high and low;
  ``mfe_session`` / ``mae_session`` say which session (1-based) it came on;
  ``first_1atr`` - which of +1 / -1 ATR was touched first (``same_session`` when
  one bar touched both, ``neither`` when none did).
* ``next_open_side_return_pct`` - F18's fill: bought at session 1's open instead of
  the scan day's close, held to the same target close.
* ``pullback_*`` - the intraday-pullback entry for the leader-pullback study key
  (LONG, ``pct_from_current_vwap`` in [-10, -3], ``top_pattern_tracking`` or sector
  Technology): a limit at the entry close - 0.25 ATR resting through session 1,
  filled by `research_warehouse.retest_entry.limit_fill` (a gap fills at the open).
  No fill is ``no_fill`` - no trade, never a zero.

Point in time: only bars dated strictly after the scan date and no later than the
last completed session are read. Any missing bar in the path, a target session not
complete yet, or no ATR is "unknown" with a reason, never 0. Nothing here reaches a
detector, a score, a gate, an alert or `review_policy.json`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_warehouse.retest_entry import RETEST_ATR_FRACTION, limit_fill  # noqa: E402

PATH_HORIZONS = (1, 3, 5, 10, 20)
SOURCE_OUTCOME_KIND = "favorable_direction_session_v2"
KNOWLEDGE_BASIS = "bars_after_scan_date_atr20_at_scan"
ONE_ATR = 1.0

#: The leader-pullback study key (TODO S14). A study tag, not a scored family.
LEADER_PULLBACK_VWAP_RANGE = (-10.0, -3.0)
LEADER_PULLBACK_FAMILY = "top_pattern_tracking"
LEADER_PULLBACK_SECTOR = "Technology"

REASON_IMMATURE = "target_session_not_complete"
REASON_OUT_OF_RANGE = "target_session_outside_calendar_range"
REASON_MISSING_BAR = "missing_bar_in_path"
REASON_NO_BARS = "no_bars_for_symbol"

PULLBACK_NOT_IN_KEY = "not_in_key"
PULLBACK_KEY_UNKNOWN = "key_unknown"
PULLBACK_NO_ATR = "no_atr"
PULLBACK_UNMEASURED = "unmeasured"
PULLBACK_NO_FILL = "no_fill"
PULLBACK_FILLED = "filled"

COLUMNS = [
    "observation_id",
    "scan_row_id",
    "symbol",
    "side",
    "scan_date",
    "setup_family",
    "horizon_sessions",
    "target_session",
    "entry_close",
    "atr20",
    "measured",
    "maturity",
    "unmeasured_reason",
    "target_close",
    "side_return_pct",
    "mfe_atr",
    "mae_atr",
    "mfe_session",
    "mae_session",
    "first_1atr",
    "next_open",
    "next_open_side_return_pct",
    "leader_pullback",
    "pullback_limit",
    "pullback_status",
    "pullback_fill",
    "pullback_side_return_pct",
    "knowledge_basis",
]

Bar = tuple[float, float, float, float]  # open, high, low, close
BarsFor = Callable[[str], Mapping[date, Bar] | None]


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def scan_row_id(row: Mapping[str, Any]) -> str:
    """`legacy._scan_factor_row_id`'s shape: SYMBOL:scan_date:run_id (or run_timestamp)."""
    scan_date = (_text(row.get("last_trade_date")) or _text(row.get("run_date")))[:10]
    suffix = _text(row.get("run_id")) or _text(row.get("run_timestamp"))
    return f"{_text(row.get('symbol')).upper()}:{scan_date}:{suffix}".rstrip(":")


@dataclass(frozen=True)
class ScanFacts:
    """What the scan row knew on the scan date: ATR and the leader-pullback inputs."""

    atr20: float | None = None
    pct_from_current_vwap: float | None = None
    sector: str = ""


def leader_pullback(side: str, family: str, facts: ScanFacts | None) -> bool | None:
    """The S14 key on one scan row; None when an input it needs is unknown."""
    if side != "LONG":
        return False
    if facts is None or facts.pct_from_current_vwap is None:
        return None
    low, high = LEADER_PULLBACK_VWAP_RANGE
    if not (low <= facts.pct_from_current_vwap <= high):
        return False
    if family == LEADER_PULLBACK_FAMILY:
        return True
    if not facts.sector:
        return None
    return facts.sector == LEADER_PULLBACK_SECTOR


def path_excursions(entry: float, atr: float, side: str, bars: list[Bar]) -> dict[str, Any]:
    """MFE / MAE in ATR over ``bars`` (sessions 1..h after entry), side-adjusted."""
    short = side == "SHORT"
    mfe = mae = None
    mfe_at = mae_at = 0
    first = "neither"
    for index, (_open, high, low, _close) in enumerate(bars, start=1):
        favourable = ((entry - low) if short else (high - entry)) / atr
        adverse = ((entry - high) if short else (low - entry)) / atr
        if mfe is None or favourable > mfe:
            mfe, mfe_at = favourable, index
        if mae is None or adverse < mae:
            mae, mae_at = adverse, index
        if first == "neither":
            up, down = favourable >= ONE_ATR, adverse <= -ONE_ATR
            if up and down:
                first = "same_session"
            elif up:
                first = "favorable"
            elif down:
                first = "adverse"
    return {"mfe_atr": mfe, "mae_atr": mae, "mfe_session": mfe_at,
            "mae_session": mae_at, "first_1atr": first}


def _side_return_pct(entry: float, exit_: float, side: str) -> float:
    raw = (exit_ / entry - 1.0) * 100.0
    return -raw if side == "SHORT" else raw


def _sessions_after(day: date, count: int, calendar) -> list[date] | None:
    """The ``count`` sessions strictly after ``day``; None outside the calendar."""
    out: list[date] = []
    cursor = day
    try:
        for _ in range(count):
            cursor = calendar.next_session(cursor)
            out.append(cursor)
    except Exception:  # noqa: BLE001 - outside the validated calendar is unknown
        return None
    return out


@dataclass
class PathFactsBuild:
    rows: list[dict] = field(default_factory=list)
    entries: int = 0
    scan_facts_missing: int = 0


def build_path_fact_rows(
    horizon_rows: Iterable[Mapping[str, Any]],
    bars_for: BarsFor,
    scan_facts: Mapping[str, ScanFacts],
    *,
    last_completed_session: date,
    horizons: tuple[int, ...] = PATH_HORIZONS,
    calendar=None,
) -> PathFactsBuild:
    """One row per (scan row, horizon) for every v2 scan row in ``horizon_rows``."""
    if calendar is None:
        import market_calendar as calendar
    horizons = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    widest = max(horizons) if horizons else 0
    entries: dict[str, dict[str, Any]] = {}
    for row in horizon_rows:
        if _text(row.get("outcome_kind")) != SOURCE_OUTCOME_KIND:
            continue
        row_id = _text(row.get("scan_row_id"))
        if row_id and row_id not in entries:
            entries[row_id] = dict(row)

    build = PathFactsBuild(entries=len(entries))
    sessions_cache: dict[date, list[date] | None] = {}
    bars_cache: dict[str, Mapping[date, Bar] | None] = {}
    for row_id, entry in entries.items():
        symbol = _text(entry.get("symbol")).upper()
        side = _text(entry.get("side")).upper() or "LONG"
        family = _text(entry.get("setup_family"))
        scan_text = _text(entry.get("scan_date"))[:10]
        entry_close = _num(entry.get("entry_close"))
        try:
            scan_day = date.fromisoformat(scan_text)
        except ValueError:
            continue
        if entry_close is None or entry_close <= 0:
            continue
        if scan_day not in sessions_cache:
            sessions_cache[scan_day] = _sessions_after(scan_day, widest, calendar)
        sessions = sessions_cache[scan_day]
        if symbol not in bars_cache:
            try:
                bars_cache[symbol] = bars_for(symbol)
            except Exception:  # noqa: BLE001 - unreadable bars are unknown, never a failed build
                logging.debug("bars_for(%s) raised", symbol, exc_info=True)
                bars_cache[symbol] = None
        bars = bars_cache[symbol]
        facts = scan_facts.get(row_id)
        if facts is None:
            build.scan_facts_missing += 1
        atr = facts.atr20 if facts is not None and facts.atr20 and facts.atr20 > 0 else None
        key = leader_pullback(side, family, facts)
        # Completed bars only: the path stops at the last completed session.
        path: list[Bar | None] = []
        for day in sessions or ():
            bar = bars.get(day) if bars and day <= last_completed_session else None
            path.append(bar)

        for horizon in horizons:
            row = {column: "" for column in COLUMNS}
            row.update(
                observation_id=f"{row_id}:{horizon}", scan_row_id=row_id, symbol=symbol,
                side=side, scan_date=scan_text, setup_family=family, horizon_sessions=horizon,
                entry_close=entry_close, atr20="" if atr is None else atr, measured=False,
                maturity="mature", knowledge_basis=KNOWLEDGE_BASIS,
                leader_pullback="" if key is None else key,
            )
            build.rows.append(row)
            if sessions is None:
                row["unmeasured_reason"] = REASON_OUT_OF_RANGE
                row["pullback_status"] = _pullback_status_unmeasured(key)
                continue
            target = sessions[horizon - 1]
            row["target_session"] = target.isoformat()
            if target > last_completed_session:
                row["maturity"] = "immature"
                row["unmeasured_reason"] = REASON_IMMATURE
                row["pullback_status"] = _pullback_status_unmeasured(key)
                continue
            window = path[:horizon]
            if not bars:
                row["unmeasured_reason"] = REASON_NO_BARS
                row["pullback_status"] = _pullback_status_unmeasured(key)
                continue
            if any(bar is None for bar in window):
                row["unmeasured_reason"] = REASON_MISSING_BAR
                row["pullback_status"] = _pullback_status_unmeasured(key)
                continue
            target_close = window[-1][3]
            next_open = window[0][0]
            row.update(
                measured=True, target_close=target_close,
                side_return_pct=_side_return_pct(entry_close, target_close, side),
                next_open=next_open,
                next_open_side_return_pct=_side_return_pct(next_open, target_close, side),
            )
            if atr is not None:
                row.update(path_excursions(entry_close, atr, side, window))
            row.update(_pullback(key, entry_close, atr, window[0], target_close))
    return build


def _pullback_status_unmeasured(key: bool | None) -> str:
    if key is False:
        return PULLBACK_NOT_IN_KEY
    return PULLBACK_KEY_UNKNOWN if key is None else PULLBACK_UNMEASURED


def _pullback(key: bool | None, entry: float, atr: float | None, session_one: Bar,
              target_close: float) -> dict[str, Any]:
    """The leader-pullback limit on session 1: filled, no fill, or why it is unknown."""
    if key is False:
        return {"pullback_status": PULLBACK_NOT_IN_KEY}
    if key is None:
        return {"pullback_status": PULLBACK_KEY_UNKNOWN}
    if atr is None:
        return {"pullback_status": PULLBACK_NO_ATR}
    limit = entry - RETEST_ATR_FRACTION * atr
    open_, high, low, close = session_one
    fill = limit_fill({"open": open_, "high": high, "low": low}, limit, True)
    if fill is None:
        return {"pullback_limit": limit, "pullback_status": PULLBACK_NO_FILL}
    return {"pullback_limit": limit, "pullback_status": PULLBACK_FILLED, "pullback_fill": fill,
            "pullback_side_return_pct": _side_return_pct(fill, target_close, "LONG")}


# ---------------------------------------------------------------- loaders
def daily_bars_from_dir(bars_dir: Path) -> BarsFor:
    """``bars_for`` over ``<dir>/<SYMBOL>.parquet`` (or ``.csv``); None when absent."""
    import pandas as pd

    def read(symbol: str) -> dict[date, Bar] | None:
        base = Path(bars_dir) / symbol
        parquet, text = base.with_suffix(".parquet"), base.with_suffix(".csv")
        if parquet.is_file():
            frame = pd.read_parquet(parquet, columns=["datetime", "open", "high", "low", "close"])
        elif text.is_file():
            frame = pd.read_csv(text, usecols=["datetime", "open", "high", "low", "close"])
        else:
            return None
        out: dict[date, Bar] = {}
        stamps = pd.to_datetime(frame["datetime"], errors="coerce")
        for stamp, o, h, low, c in zip(stamps, frame["open"], frame["high"], frame["low"],
                                       frame["close"], strict=True):
            values = [_num(v) for v in (o, h, low, c)]
            if pd.isna(stamp) or any(v is None or v <= 0 for v in values):
                continue
            out[stamp.date()] = tuple(values)  # type: ignore[assignment]
        return out

    return read


FEATURE_COLUMNS = ["run_id", "run_timestamp", "run_date", "last_trade_date", "symbol",
                   "atr20", "pct_from_current_vwap", "sector"]


def scan_facts_from_features(path: Path, wanted: set[str], *, chunk_rows: int = 250_000
                             ) -> dict[str, ScanFacts]:
    """``{scan_row_id: ScanFacts}`` for the wanted rows of ``d1_features_history.csv``."""
    import pandas as pd

    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in FEATURE_COLUMNS if c in header]
    out: dict[str, ScanFacts] = {}
    for chunk in pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False,
                             chunksize=chunk_rows):
        for row in chunk.to_dict("records"):
            row_id = scan_row_id(row)
            if row_id not in wanted:
                continue
            out[row_id] = ScanFacts(
                atr20=_num(row.get("atr20")),
                pct_from_current_vwap=_num(row.get("pct_from_current_vwap")),
                sector=_text(row.get("sector")),
            )
    return out


def write_rows(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizons", type=Path, required=True,
                        help="copy of master_avwap_session_horizon_outcomes.csv")
    parser.add_argument("--daily-bars", type=Path, required=True, help="copy of the daily_bars dir")
    parser.add_argument("--features", type=Path, required=True, help="copy of d1_features_history.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--last-completed", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)
    from setup_permutation_backfill import refuse_live

    refuse_live([args.horizons, args.daily_bars, args.features, args.out])
    if args.last_completed is None:
        import market_calendar

        args.last_completed = market_calendar.last_completed_session(datetime.now())
    started = _time.perf_counter()
    with args.horizons.open("r", encoding="utf-8-sig", newline="") as handle:
        horizon_rows = list(csv.DictReader(handle))
    wanted = {_text(r.get("scan_row_id")) for r in horizon_rows}
    facts = scan_facts_from_features(args.features, wanted)
    loaded = _time.perf_counter()
    build = build_path_fact_rows(horizon_rows, daily_bars_from_dir(args.daily_bars), facts,
                                 last_completed_session=args.last_completed)
    write_rows(build.rows, args.out)
    done = _time.perf_counter()
    measured = sum(1 for r in build.rows if r["measured"] is True)
    print(f"{build.entries} scan rows -> {len(build.rows)} rows ({measured} measured), "
          f"{build.scan_facts_missing} with no scan facts; load {loaded - started:.1f}s, "
          f"build+write {done - loaded:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
