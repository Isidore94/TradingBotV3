"""Best-right-now log and its grader (B6, goal 9). Evidence only.

Log (`BEST_NOW_LOG_FILE`): one `kind: best_now` row the first time a
(symbol, side) appears on the strip in a trade date - ts (aware), trade_date,
symbol, side, grade (the M5 alert's setup grade; "" for a dip-strong-only
row), rank (1 = top) and entry_price_ref (the strip's entry). The strip's
worker thread writes it; a failed write loses rows, never the strip.

Grader (`summarize`, CLI `python scripts/best_now_outcomes.py --summary`):
each row is measured from the close of the last completed M5 bar at its ts
(the name and SPY alike) to +15/+30/+60 minutes; excess = name - SPY (SPY -
name for a short). A hit is excess > 0. The bars are the durable Day Review
session tape (`day_review_bars.read_session_bars`), never IB; a session with
no stored tape yet is `pending`, a missing bar is unmeasured. The headline
hit rate is +30 minutes (the Movers log's primary horizon; lead decision
2026-09-25, the trader can overrule).
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

KIND = "best_now"
HORIZON_MINUTES = (15, 30, 60)
HEADLINE_MINUTES = 30
BAR_MINUTES = 5
BENCHMARK = "SPY"
NO_DATA = "no data yet"

_log = logging.getLogger(__name__)


def _default_path() -> Path:
    from project_paths import BEST_NOW_LOG_FILE

    return Path(BEST_NOW_LOG_FILE)


def _trade_date_text() -> str:
    try:
        from market_session import get_market_session_window

        return get_market_session_window().market_date.isoformat()
    except Exception:  # noqa: BLE001 - the calendar never costs the log
        return datetime.now().date().isoformat()


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    return {"L": "LONG", "BUY": "LONG", "S": "SHORT", "SELL": "SHORT"}.get(text, text)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


# ---------------------------------------------------------------- the log
def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Every parseable row, oldest first; a missing file is []."""
    target = Path(path) if path is not None else _default_path()
    out: list[dict[str, Any]] = []
    try:
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except OSError:
        return []
    return out


def log_rows(
    entries: Sequence[Any],
    results: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime,
    trade_date: str,
    seen: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Rows for entries not yet in `seen` (which this updates). Pure."""
    grades: dict[tuple[str, str], str] = {}
    for row in results or ():
        key = (str(row.get("symbol") or "").strip().upper(), _side(row.get("side")))
        if key[0] and key not in grades:
            grades[key] = str(row.get("grade") or "")
    stamp = now if now.tzinfo is not None else now.astimezone()
    out: list[dict[str, Any]] = []
    for rank, entry in enumerate(entries or (), start=1):
        symbol = str(getattr(entry, "symbol", "") or "").strip().upper()
        side = _side(getattr(entry, "side", ""))
        key = (trade_date, symbol, side)
        if not symbol or key in seen:
            continue
        seen.add(key)
        out.append({
            "kind": KIND,
            "ts": stamp.isoformat(timespec="seconds"),
            "trade_date": trade_date,
            "symbol": symbol,
            "side": side,
            "grade": grades.get((symbol, side), ""),
            "rank": rank,
            "entry_price_ref": _number(getattr(entry, "entry", None)),
        })
    return out


class BestNowLog:
    """First appearances per trade date; today's already-logged keys are read once."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        trade_date: Callable[[], str] | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._trade_date = trade_date or _trade_date_text
        self._lock = threading.Lock()
        self._seen: set[tuple[str, str, str]] = set()
        self._loaded: set[str] = set()

    @property
    def path(self) -> Path:
        return self._path if self._path is not None else _default_path()

    def record(self, entries: Sequence[Any], results: Iterable[Mapping[str, Any]] | None) -> list[dict]:
        """Append the new first appearances. Never raises; a failed write loses the rows."""
        try:
            with self._lock:
                day = self._trade_date()
                if day not in self._loaded:
                    for row in load_records(self.path):
                        if row.get("kind") == KIND and str(row.get("trade_date") or "") == day:
                            self._seen.add((day, str(row.get("symbol") or ""), _side(row.get("side"))))
                    self._loaded.add(day)
                rows = log_rows(entries, results, now=self._clock(), trade_date=day, seen=self._seen)
                if rows:
                    target = self.path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("a", encoding="utf-8") as handle:
                        for row in rows:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                return rows
        except Exception as exc:  # noqa: BLE001 - evidence never costs the strip
            _log.warning("Best-right-now log write failed: %s", exc)
            return []


def logged_symbols(session: str, path: Path | None = None) -> set[str]:
    """Names logged on `session` (the post-close bar download adds them)."""
    return {
        str(row.get("symbol") or "").upper()
        for row in load_records(path)
        if row.get("kind") == KIND and str(row.get("trade_date") or "") == str(session)[:10]
    } - {""}


# ---------------------------------------------------------------- the grader
def _aware(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value or ""))
        except ValueError:
            return None
    return stamp if stamp.tzinfo is not None else None


def _timed(bars) -> list[Mapping[str, Any]]:
    """Bars with an aware start time, oldest first."""
    rows = [bar for bar in bars or () if _aware(bar.get("dt")) is not None]
    return sorted(rows, key=lambda bar: _aware(bar.get("dt")))


def _base_index(bars: Sequence[Mapping[str, Any]], ts: datetime) -> int:
    """The last bar completed at `ts` (its start + 5 min <= ts), or -1."""
    found = -1
    for index, bar in enumerate(bars):
        start = _aware(bar.get("dt"))
        if start is None:
            continue
        if start + timedelta(minutes=BAR_MINUTES) <= ts:
            found = index
        else:
            break
    return found


def _forward(bars, index: int, minutes: int) -> float | None:
    target = index + minutes // BAR_MINUTES
    if index < 0 or target >= len(bars):
        return None
    base = _number(bars[index].get("close"))
    close = _number(bars[target].get("close"))
    if base in (None, 0) or close is None:
        return None
    return (close / base - 1.0) * 100.0


def grade_row(row: Mapping[str, Any], bars: Mapping[str, Sequence[Mapping[str, Any]]] | None) -> dict:
    """One logged row measured vs SPY. `bars` None = the session tape is not stored yet."""
    out = {key: row.get(key) for key in ("trade_date", "symbol", "side", "grade", "rank", "ts")}
    out["status"] = "pending" if bars is None else "measured"
    ts = _aware(row.get("ts"))
    name_bars = _timed((bars or {}).get(str(row.get("symbol") or "").upper()))
    spy_bars = _timed((bars or {}).get(BENCHMARK))
    index = _base_index(name_bars, ts) if ts is not None else -1
    spy_index = -1
    if index >= 0:
        base_dt = _aware(name_bars[index].get("dt"))
        spy_index = next((i for i, b in enumerate(spy_bars) if _aware(b.get("dt")) == base_dt), -1)
    out["base_bar"] = str(name_bars[index].get("dt")) if index >= 0 else ""
    short = _side(row.get("side")) == "SHORT"
    for minutes in HORIZON_MINUTES:
        ret = _forward(name_bars, index, minutes)
        spy_ret = _forward(spy_bars, spy_index, minutes)
        excess = None
        if ret is not None and spy_ret is not None:
            excess = (spy_ret - ret) if short else (ret - spy_ret)
        out[f"ret{minutes}_pct"] = ret
        out[f"spy_ret{minutes}_pct"] = spy_ret
        out[f"excess{minutes}_pct"] = excess
    return out


def _read_bars(session: str):
    import day_review_bars

    return day_review_bars.read_session_bars(session)


def summarize(
    records: Iterable[Mapping[str, Any]],
    *,
    start: date | None = None,
    end: date | None = None,
    bars_reader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Per horizon: logged, graded, hit rate (beat SPY by side), overall and by side."""
    reader = bars_reader or _read_bars

    def in_range(row) -> bool:
        try:
            day = date.fromisoformat(str(row.get("trade_date") or "")[:10])
        except ValueError:
            return False
        return (start is None or day >= start) and (end is None or day <= end)

    rows = [r for r in records if r.get("kind") == KIND and in_range(r)]
    tapes: dict[str, Any] = {}
    graded_rows = []
    for row in rows:
        session = str(row.get("trade_date") or "")[:10]
        if session not in tapes:
            try:
                tapes[session] = reader(session)
            except Exception:  # noqa: BLE001 - an unreadable tape is pending, never a miss
                tapes[session] = None
        graded_rows.append(grade_row(row, tapes[session]))

    def block(group, minutes) -> dict[str, Any]:
        values = [float(r[f"excess{minutes}_pct"]) for r in group
                  if r.get(f"excess{minutes}_pct") is not None]
        return {
            "logged": len(group),
            "graded": len(values),
            "hit_rate": (sum(1 for v in values if v > 0) / len(values)) if values else None,
            "avg_excess_pct": (sum(values) / len(values)) if values else None,
        }

    by_horizon = {}
    for minutes in HORIZON_MINUTES:
        entry = block(graded_rows, minutes)
        entry["long"] = block([r for r in graded_rows if _side(r.get("side")) == "LONG"], minutes)
        entry["short"] = block([r for r in graded_rows if _side(r.get("side")) == "SHORT"], minutes)
        by_horizon[str(minutes)] = entry
    headline = by_horizon[str(HEADLINE_MINUTES)]
    return {
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "measure": "from the close of the last completed M5 bar at first appearance, name vs SPY",
        "logged": len(rows),
        "pending": sum(1 for r in graded_rows if r["status"] == "pending"),
        "sessions": sorted(tapes),
        "headline_minutes": HEADLINE_MINUTES,
        "hit_rate": headline["hit_rate"],
        "graded": headline["graded"],
        "by_horizon": by_horizon,
        "rows": graded_rows,
    }


def _pct(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value * 100:.0f}%"


def summary_lines(summary: Mapping[str, Any]) -> list[str]:
    """Plain lines for the CLI; "no data yet" when nothing was logged."""
    if not summary.get("logged"):
        return [NO_DATA]
    lines = [f"Best right now: {summary['logged']} logged, {summary.get('pending', 0)} pending "
             f"(no stored session tape yet)."]
    for minutes, entry in (summary.get("by_horizon") or {}).items():
        lines.append(
            f"+{minutes} min: hit rate {_pct(entry['hit_rate'])} ({entry['graded']} graded); "
            f"long {_pct(entry['long']['hit_rate'])} ({entry['long']['graded']}), "
            f"short {_pct(entry['short']['hit_rate'])} ({entry['short']['graded']})"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Best-right-now log: hit rate vs SPY")
    parser.add_argument("--summary", action="store_true", help="print the summary")
    parser.add_argument("--start", help="first trade date (YYYY-MM-DD)")
    parser.add_argument("--end", help="last trade date (YYYY-MM-DD)")
    parser.add_argument("--path", help="log (default: project_paths.BEST_NOW_LOG_FILE)")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)
    if not args.summary:
        parser.print_help()
        return 2
    path = Path(args.path) if args.path else _default_path()
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    summary = summarize(load_records(path), start=start, end=end)
    summary["path"] = str(path)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print("\n".join(summary_lines(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
