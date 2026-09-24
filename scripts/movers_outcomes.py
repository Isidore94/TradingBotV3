"""Dip-strong outcome log: what the Movers board flagged in a SPY pullback, and what happened.

Evidence only. A pullback (or bounce) episode is keyed by SPY's start bar and
side. While it is ON, every name Dip-strong lists is recorded once, point in
time (`kind: flag`). The episode ends when SPY closes back beyond the
pre-pullback extreme, 6 bars after it opened, or at the session's last bar.
Each flagged name then gets one `kind: outcome` row: its move from its own
pullback low (high for a short) to +3 and +6 bars, next to SPY's move over
the same bars. A bar that is not there yet is waited for; at the session's
last bar what is missing is recorded as None (unknown).

`summarize` and the CLI (`python scripts/movers_outcomes.py --summary`) read
the file for the night AI. Pure except `append_records` and `load_records`.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: An episode ends this many bars after it opened.
EPISODE_MAX_BARS = 6
#: Outcome horizons, in bars after the name's own pullback extreme.
HORIZONS = (3, 6)
#: The session's last regular M5 bar starts here (New York).
LAST_BAR_START = time(15, 55)


def _iso(value: Any) -> str:
    return value.isoformat(timespec="seconds") if isinstance(value, datetime) else str(value or "")


def _pct(value: float | None, base: float | None) -> float | None:
    if value is None or base in (None, 0):
        return None
    return (value / base - 1.0) * 100.0


class DipOutcomeTracker:
    """In-memory episode state. `observe` returns the rows to append."""

    def __init__(self) -> None:
        self.episodes: dict[tuple[str, str], dict[str, Any]] = {}

    def observe(
        self,
        board: Mapping[str, Any],
        series: Mapping[str, Sequence[Mapping[str, Any]]],
        spy: Sequence[Mapping[str, Any]],
        *,
        now: datetime,
    ) -> list[dict[str, Any]]:
        """One tick. `series`/`spy` are normalised completed bars (aware NY `dt`)."""
        out: list[dict[str, Any]] = []
        state = dict(board.get("state") or {})
        spy_today = _today(spy)
        if not spy_today:
            return out
        session = spy_today[-1]["dt"].date()
        for key in [k for k, ep in self.episodes.items() if ep["session"] != session]:
            self.episodes.pop(key)  # a new session: yesterday's leftovers are gone
        side = "long" if state.get("pullback") else "short" if state.get("bounce") else ""
        start = state.get("start_dt")
        start_text = _iso(start)
        if side and start_text:
            key = (start_text, side)
            episode = self.episodes.get(key)
            if episode is None:
                episode = {
                    "session": session, "side": side, "start": start_text,
                    "extreme": state.get("extreme_price"),
                    "opened_at": spy_today[-1]["dt"], "end": None, "end_reason": "",
                    "flagged": {}, "resolved": set(),
                }
                self.episodes[key] = episode
            if episode["end"] is None:
                rows = ((board.get("dip") or {}).get(side)) or []
                for rank, row in enumerate(rows, start=1):
                    symbol = str(row.get("symbol") or "").upper()
                    if not symbol or symbol in episode["flagged"]:
                        continue
                    record = {
                        "kind": "flag", "session": session.isoformat(), "episode": start_text,
                        "side": side, "symbol": symbol, "rank": rank,
                        "flagged_bar": _iso(spy_today[-1]["dt"]), "recorded_at": _iso(now),
                        "spy_start_extreme": episode["extreme"],
                        "spy_from_extreme_pct": state.get("spy_from_extreme_pct"),
                    }
                    for field in ("dip_score", "since_start_pct", "rvol", "day_pct",
                                  "move15_pct", "from_vwap_atr", "last"):
                        record[field] = row.get(field)
                    episode["flagged"][symbol] = record
                    out.append(record)
        for key, episode in list(self.episodes.items()):
            if episode["end"] is None:
                reason = self._end_reason(episode, spy_today, key[0], start_text)
                if reason:
                    episode["end"] = spy_today[-1]["dt"]
                    episode["end_reason"] = reason
            if episode["end"] is not None:
                closed = spy_today[-1]["dt"].time() >= LAST_BAR_START
                out.extend(self._resolve(episode, series, spy_today, now, closed))
                if len(episode["resolved"]) >= len(episode["flagged"]):
                    self.episodes.pop(key)
        return out

    @staticmethod
    def _end_reason(episode, spy_today, episode_start: str, live_start: str) -> str:
        last = spy_today[-1]
        extreme = episode["extreme"]
        if extreme is not None:
            if episode["side"] == "long" and last["close"] > extreme:
                return "spy_reclaimed_high"
            if episode["side"] == "short" and last["close"] < extreme:
                return "spy_lost_low"
        after = [bar for bar in spy_today if bar["dt"] > episode["opened_at"]]
        if len(after) >= EPISODE_MAX_BARS:
            return "six_bars"
        if last["dt"].time() >= LAST_BAR_START:
            return "session_close"
        if live_start != episode_start:
            return "episode_over"
        return ""

    @staticmethod
    def _resolve(episode, series, spy_today, now, closed) -> list[dict[str, Any]]:
        out = []
        side = episode["side"]
        start = datetime.fromisoformat(episode["start"])
        spy_close = {bar["dt"]: bar["close"] for bar in spy_today}
        for symbol, flag in episode["flagged"].items():
            if symbol in episode["resolved"]:
                continue
            bars = _today(series.get(symbol) or ())
            window = [bar for bar in bars if start <= bar["dt"] <= episode["end"]]
            if not window:
                if closed:
                    episode["resolved"].add(symbol)
                continue
            pick = min if side == "long" else max
            anchor = pick(window, key=lambda b: (b["low"] if side == "long" else b["high"]))
            index = bars.index(anchor)
            if index + max(HORIZONS) >= len(bars) and not closed:
                continue  # wait for +6 bars
            price = anchor["low"] if side == "long" else anchor["high"]
            record = {
                "kind": "outcome", "session": flag["session"], "episode": flag["episode"],
                "side": side, "symbol": symbol, "rank": flag["rank"],
                "end_reason": episode["end_reason"], "episode_end_bar": _iso(episode["end"]),
                "anchor_bar": _iso(anchor["dt"]), "anchor_price": price,
                "recorded_at": _iso(now),
            }
            for horizon in HORIZONS:
                target = bars[index + horizon] if index + horizon < len(bars) else None
                ret = _pct(target["close"], price) if target else None
                spy_ret = None
                if target and anchor["dt"] in spy_close and target["dt"] in spy_close:
                    spy_ret = _pct(spy_close[target["dt"]], spy_close[anchor["dt"]])
                excess = None
                if ret is not None and spy_ret is not None:
                    excess = ret - spy_ret if side == "long" else spy_ret - ret
                record[f"ret{horizon}_pct"] = ret
                record[f"spy_ret{horizon}_pct"] = spy_ret
                record[f"excess{horizon}_pct"] = excess
            episode["resolved"].add(symbol)
            out.append(record)
        return out


def _today(bars: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not bars:
        return []
    day = bars[-1]["dt"].date()
    return [bar for bar in bars if bar["dt"].date() == day]


# ---------------------------------------------------------------- I/O
def append_records(path: Path, records: Iterable[Mapping[str, Any]]) -> bool:
    """Append rows. A failed write loses these rows and logs a warning; never raises."""
    rows = list(records)
    if not rows:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return True
    except Exception as exc:
        logging.warning("Movers outcome log write failed (%s rows lost): %s", len(rows), exc)
        return False


def load_records(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except FileNotFoundError:
        return []
    return out


# ---------------------------------------------------------------- summary
def summarize(
    records: Iterable[Mapping[str, Any]],
    *,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, Any]:
    """Hit rate (+6 excess over SPY > 0, else +3) and mean excess, overall and by side."""
    def in_range(row) -> bool:
        try:
            day = date.fromisoformat(str(row.get("session") or "")[:10])
        except ValueError:
            return False
        return (start is None or day >= start) and (end is None or day <= end)

    outcomes = [r for r in records if r.get("kind") == "outcome" and in_range(r)]

    def block(rows) -> dict[str, Any]:
        graded = []
        for row in rows:
            value = row.get("excess6_pct")
            if value is None:
                value = row.get("excess3_pct")
            if value is not None:
                graded.append(float(value))
        ex3 = [float(r["excess3_pct"]) for r in rows if r.get("excess3_pct") is not None]
        ex6 = [float(r["excess6_pct"]) for r in rows if r.get("excess6_pct") is not None]
        return {
            "outcomes": len(rows),
            "graded": len(graded),
            "hit_rate": (sum(1 for v in graded if v > 0) / len(graded)) if graded else None,
            "avg_excess3_pct": (sum(ex3) / len(ex3)) if ex3 else None,
            "avg_excess6_pct": (sum(ex6) / len(ex6)) if ex6 else None,
            "episodes": len({(r.get("session"), r.get("episode"), r.get("side")) for r in rows}),
        }

    return {
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "all": block(outcomes),
        "long": block([r for r in outcomes if r.get("side") == "long"]),
        "short": block([r for r in outcomes if r.get("side") == "short"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Movers Dip-strong outcome log")
    parser.add_argument("--summary", action="store_true", help="print the summary as JSON")
    parser.add_argument("--start", help="first session (YYYY-MM-DD)")
    parser.add_argument("--end", help="last session (YYYY-MM-DD)")
    parser.add_argument("--path", help="log file (default: project_paths.MOVERS_DIP_OUTCOMES_FILE)")
    args = parser.parse_args(argv)
    if not args.summary:
        parser.print_help()
        return 2
    if args.path:
        path = Path(args.path)
    else:
        from project_paths import MOVERS_DIP_OUTCOMES_FILE

        path = MOVERS_DIP_OUTCOMES_FILE
    summary = summarize(
        load_records(path),
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
    )
    summary["path"] = str(path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
