"""Read-only replay: how many past review charts would the wall gate have hidden?

Run on COPIES of the live stores only:

    python scripts/wall_gate_replay.py --scratch <dir> \
        --events <copy of alert_review_events dir or .jsonl> \
        --daily-bars <copy of data/daily_bars> \
        [--outcomes <copy of master_avwap_session_horizon_outcomes.csv>] [--json]

Inputs:
- the review decision log: every chart the trader was SHOWN (``action ==
  "shown"``), one per trade date, symbol, side and timeframe;
- the daily bar store (one parquet per symbol): SMAs and ATR20 from bars
  dated BEFORE the chart's trade date only (point in time);
- optionally the D1 session-horizon outcome store, joined on symbol, side
  and scan date, for the tracked 1/3/5-session side return.

Price is the ``px=`` in the chart's trigger text, else its entry price, else
the last completed close (labelled). SMA walls only: the scan's trendline
records are not kept point in time, so the trendline leg cannot be replayed.
The cap and the watch life are simulated: an arm lives until a later
completed close crosses its SMA level or 10 sessions pass.

Scratch-script rule (AGENTS.md): TRADINGBOTV3_DATA_DIR and LOCALAPPDATA are
pointed at ``--scratch`` before any project import, the run aborts if a
project_paths root or an input path is a live store, and nothing is written
anywhere but stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from swallowed import note_swallowed

LIVE_ROOTS = (Path(r"C:\TradingBotData"), Path(r"\\MINI-PC\Trading Bot Data"))
CAP = 20
ARM_LIFE_SESSIONS = 10
HORIZONS = (1, 3, 5)
_PX_RE = re.compile(r"px=([0-9]+(?:\.[0-9]+)?)")


class LiveStoreRefused(RuntimeError):
    """A root or an input is a live store."""


def _norm(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path))).rstrip("\\/")


def _is_under(path: Path | str, root: Path | str) -> bool:
    child, parent = _norm(path), _norm(root)
    return child == parent or child.startswith(parent + os.sep)


def live_roots(original_localappdata: str | None) -> list[Path]:
    roots = list(LIVE_ROOTS)
    if original_localappdata:
        roots.append(Path(original_localappdata) / "TradingBotV3")
    return roots


def prepare_scratch(scratch: Path, inputs: Iterable[Path]) -> None:
    """Point both roots at ``scratch`` and refuse any live path. Call first."""
    original = os.environ.get("LOCALAPPDATA")
    roots = live_roots(original)
    scratch = Path(scratch)
    for root in roots:
        if _is_under(scratch, root):
            raise LiveStoreRefused(f"scratch {scratch} is inside live store {root}")
    for item in inputs:
        for root in roots:
            if item is not None and _is_under(item, root):
                raise LiveStoreRefused(
                    f"input {item} is inside live store {root}; copy it to scratch first"
                )
    (scratch / "data").mkdir(parents=True, exist_ok=True)
    (scratch / "localappdata").mkdir(parents=True, exist_ok=True)
    os.environ["TRADINGBOTV3_DATA_DIR"] = str(scratch / "data")
    os.environ["LOCALAPPDATA"] = str(scratch / "localappdata")
    if "project_paths" in sys.modules:
        raise LiveStoreRefused("project_paths was imported before the scratch roots were set")
    import project_paths

    for name in ("PERSISTENT_DATA_DIR", "LOCAL_SETTINGS_DIR", "CACHE_DIR", "DATA_DIR"):
        value = getattr(project_paths, name)
        for root in roots:
            if _is_under(value, root):
                raise LiveStoreRefused(f"project_paths.{name} = {value} is a live store")


# ---------------------------------------------------------------- inputs
def _event_files(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    return [path]


def load_shown(path: Path) -> list[dict[str, Any]]:
    """One "shown" row per (trade date, symbol, side, timeframe), earliest first."""
    seen: dict[tuple, dict[str, Any]] = {}
    for file in _event_files(path):
        with open(file, "r", encoding="utf-8") as handle:
            for line in handle:
                if '"shown"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("action") != "shown":
                    continue
                symbol = str(row.get("symbol") or "").strip().upper()
                side = str(row.get("side") or "").strip().upper()
                trade_date = str(row.get("trade_date") or "")[:10]
                if not symbol or side not in ("LONG", "SHORT") or not trade_date:
                    continue
                timeframe = "D1" if row.get("is_d1") else str(row.get("timeframe") or "M5")
                key = (trade_date, symbol, side, timeframe)
                ts = str(row.get("ts") or "")
                if key not in seen or ts < str(seen[key].get("ts") or ""):
                    seen[key] = dict(row, symbol=symbol, side=side, trade_date=trade_date,
                                     timeframe=timeframe)
    return sorted(seen.values(), key=lambda row: (row["trade_date"], str(row.get("ts") or "")))


def parquet_loader(bars_dir: Path) -> Callable[[str], list[dict[str, Any]]]:
    cache: dict[str, list[dict[str, Any]]] = {}

    def load(symbol: str) -> list[dict[str, Any]]:
        if symbol in cache:
            return cache[symbol]
        bars: list[dict[str, Any]] = []
        path = Path(bars_dir) / f"{symbol}.parquet"
        if path.exists():
            import pandas as pd

            frame = pd.read_parquet(path)
            stamp = "datetime" if "datetime" in frame.columns else frame.columns[0]
            for record in frame.to_dict("records"):
                moment = record.get(stamp)
                if hasattr(moment, "to_pydatetime"):
                    moment = moment.to_pydatetime()
                if not isinstance(moment, datetime):
                    continue
                bars.append({
                    "dt": moment,
                    "open": record.get("open"),
                    "high": record.get("high"),
                    "low": record.get("low"),
                    "close": record.get("close"),
                })
            bars.sort(key=lambda bar: bar["dt"])
        cache[symbol] = bars
        return bars

    return load


def load_outcomes(path: Path | None) -> dict[tuple[str, str, str], dict[int, float]]:
    out: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    if path is None:
        return out
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                horizon = int(row.get("horizon_sessions") or 0)
            except ValueError:
                continue
            if horizon not in HORIZONS or str(row.get("measured")).lower() != "true":
                continue
            try:
                value = float(row.get("side_return_pct") or "")
            except ValueError:
                continue
            key = (
                str(row.get("symbol") or "").upper(),
                str(row.get("side") or "").upper(),
                str(row.get("scan_date") or "")[:10],
            )
            out[key].setdefault(horizon, value)
    return out


def price_for(row: dict[str, Any], completed: list[dict[str, Any]]) -> tuple[float | None, str]:
    match = _PX_RE.search(str(row.get("trigger") or ""))
    if match:
        return float(match.group(1)), "trigger_px"
    try:
        entry = float(row.get("entry_price"))
        if entry > 0:
            return entry, "entry_price"
    except (TypeError, ValueError) as exc:
        note_swallowed("row entry price unparseable; trying the completed price", exc, quiet=True)
    if completed:
        return float(completed[-1]["close"]), "prior_close"
    return None, "none"


# ---------------------------------------------------------------- replay
@dataclass
class Arm:
    symbol: str
    side: str
    level: float
    armed_on: date


@dataclass
class DaySummary:
    day: str
    alerts: int = 0
    hidden: int = 0
    shown_at_wall_cap: int = 0
    unknown: int = 0
    peak_armed: int = 0
    walls: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _arm_released(arm: Arm, bars: list[dict[str, Any]], today: date) -> bool:
    # The arm day's own close counts: the alert came before it.
    later = [bar for bar in bars if arm.armed_on <= bar["dt"].date() < today]
    if len(later) >= ARM_LIFE_SESSIONS:
        return True
    for bar in later:
        close = float(bar["close"])
        crossed = close > arm.level if arm.side == "LONG" else close < arm.level
        if crossed:
            return True
    return False


def replay(
    shown: list[dict[str, Any]],
    load_bars: Callable[[str], list[dict[str, Any]]],
    outcomes: dict[tuple[str, str, str], dict[int, float]] | None = None,
    *,
    cap: int = CAP,
) -> dict[str, Any]:
    import wall_gate

    outcomes = outcomes or {}
    days: dict[str, DaySummary] = {}
    rows: list[dict[str, Any]] = []
    arms: dict[str, Arm] = {}
    for row in shown:
        day_text = row["trade_date"]
        today = date.fromisoformat(day_text)
        summary = days.setdefault(day_text, DaySummary(day=day_text))
        if summary.alerts == 0:  # first chart of the day: release fired/expired arms
            for symbol in list(arms):
                if _arm_released(arms[symbol], load_bars(symbol), today):
                    arms.pop(symbol)
            summary.peak_armed = len(arms)
        summary.alerts += 1
        bars = load_bars(row["symbol"])
        completed = wall_gate.completed_daily_bars(bars, today=today)
        price, price_source = price_for(row, completed)
        verdict = wall_gate.wall_state(row["side"], price, completed, today=today)
        result = "shown"
        if verdict.state == wall_gate.UNKNOWN:
            summary.unknown += 1
        elif verdict.state == wall_gate.CLOSED:
            if row["symbol"] in arms:
                result = "hidden"
            elif len(arms) >= cap:
                result = "shown_at_wall_cap"
            else:
                arms[row["symbol"]] = Arm(row["symbol"], row["side"], float(verdict.level), today)
                result = "hidden"
            if result == "hidden":
                summary.hidden += 1
                summary.walls[verdict.wall] += 1
            else:
                summary.shown_at_wall_cap += 1
        summary.peak_armed = max(summary.peak_armed, len(arms))
        outcome = outcomes.get((row["symbol"], row["side"], day_text), {})
        rows.append({
            "trade_date": day_text,
            "symbol": row["symbol"],
            "side": row["side"],
            "timeframe": row["timeframe"],
            "price": price,
            "price_source": price_source,
            "state": verdict.state,
            "result": result,
            "wall": verdict.wall,
            "distance_atr": verdict.distance_atr,
            "outcome": {str(h): outcome.get(h) for h in HORIZONS},
        })
    return {"days": [_day_dict(days[key]) for key in sorted(days)], "rows": rows,
            "outcomes": _outcome_summary(rows)}


def _day_dict(summary: DaySummary) -> dict[str, Any]:
    return {
        "day": summary.day,
        "alerts": summary.alerts,
        "hidden": summary.hidden,
        "shown_at_wall_cap": summary.shown_at_wall_cap,
        "unknown": summary.unknown,
        "peak_armed": summary.peak_armed,
        "walls": dict(summary.walls),
    }


def _outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for label in ("hidden", "shown"):
        picked = [row for row in rows if (row["result"] == "hidden") == (label == "hidden")]
        stats: dict[str, Any] = {"count": len(picked)}
        for horizon in HORIZONS:
            values = [row["outcome"][str(horizon)] for row in picked
                      if row["outcome"][str(horizon)] is not None]
            stats[f"h{horizon}_n"] = len(values)
            stats[f"h{horizon}_mean_pct"] = (
                round(sum(values) / len(values), 3) if values else None
            )
            stats[f"h{horizon}_win_rate"] = (
                round(sum(1 for value in values if value > 0) / len(values), 3)
                if values else None
            )
        groups[label] = stats
    return groups


def format_report(report: dict[str, Any]) -> str:
    lines = ["day         alerts hidden cap_shown unknown peak_armed walls"]
    for day in report["days"]:
        walls = ", ".join(f"{k}:{v}" for k, v in sorted(day["walls"].items())) or "-"
        lines.append(
            f"{day['day']}  {day['alerts']:6d} {day['hidden']:6d} "
            f"{day['shown_at_wall_cap']:9d} {day['unknown']:7d} {day['peak_armed']:10d} {walls}"
        )
    days = report["days"]
    total_alerts = sum(day["alerts"] for day in days)
    total_hidden = sum(day["hidden"] for day in days)
    lines.append(
        f"{len(days)} days, {total_alerts} charts, {total_hidden} hidden, "
        f"peak armed {max((day['peak_armed'] for day in days), default=0)}"
    )
    for label, stats in report["outcomes"].items():
        parts = [f"{label}: n={stats['count']}"]
        for horizon in HORIZONS:
            parts.append(
                f"h{horizon} n={stats[f'h{horizon}_n']} mean={stats[f'h{horizon}_mean_pct']}% "
                f"win={stats[f'h{horizon}_win_rate']}"
            )
        lines.append("  ".join(parts))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--daily-bars", required=True, type=Path)
    parser.add_argument("--outcomes", type=Path, default=None)
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        prepare_scratch(args.scratch, [args.events, args.daily_bars, args.outcomes])
    except LiveStoreRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    report = replay(
        load_shown(args.events),
        parquet_loader(args.daily_bars),
        load_outcomes(args.outcomes),
        cap=args.cap,
    )
    if args.json:
        print(json.dumps({"days": report["days"], "outcomes": report["outcomes"]}, indent=1))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
