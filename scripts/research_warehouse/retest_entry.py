"""Retest-entry study (TODO S8, finding F8). Shadow research, read-only.

For every M5 alert and every Movers Dip-log flag it compares two entries on the
same stop, both graded by the 1:1 bracket (+1R before -1R; a bar that touches
both is a loss; no hit by the session's last bar = close R):

* ``flag``   - enter at the alert (flag) bar's close;
* ``retest`` - a limit at the level +/- 0.25 M5 ATR, live for the 6 bars after
  the alert. It fills only if price comes back to it; otherwise no trade.

Point in time: the ATR and the Movers level use bars up to and including the
alert bar; fills and outcomes use only bars after it, in the same session.
Missing bars are "unknown" and are counted as skipped, never as a result.

Levels and the one shared stop:
* M5 alert: the level of its primary (first) bounce type from ``levels_json``.
* Movers Dip flag: the name's own low (long) / high (short) from the episode
  start to the flag bar.
* Stop, both entries, both sources: the level -/+ ``STOP_ATR`` ATR. The logged
  M5 stop is not used: it often sits inside the retest zone (or a few cents
  from the level), which would make the retest's R measure noise.

Nothing here feeds a live score, alert, tier or gate. The desk only displays
the report, computed on a worker when the trader presses Run.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

SCHEMA = "retest_entry_study_v1"
RETEST_ATR_FRACTION = 0.25
RETEST_WINDOW_BARS = 6
ATR_BARS = 14
MIN_ATR_BARS = 5
STOP_ATR = 0.5
DEFAULT_SINCE_DAYS = 45
CSV_CHUNK_ROWS = 250_000
EXCHANGE_ZONE = ZoneInfo("America/New_York")
M5_ALERT_COLUMNS = (
    "event_id",
    "event_type",
    "trade_date",
    "symbol",
    "direction",
    "bounce_types",
    "entry_price",
    "levels_json",
    "candle_json",
)

BarsFor = Callable[[date, Sequence[str]], Mapping[str, Sequence[Mapping[str, Any]]]]


# ---------------------------------------------------------------- pure core
def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def m5_atr(bars: Sequence[Mapping[str, Any]], end_index: int) -> float | None:
    """Mean true range of up to ATR_BARS bars ending at ``end_index`` (inclusive)."""
    start = max(0, end_index - ATR_BARS + 1)
    ranges = []
    for i in range(start, end_index + 1):
        high, low = _num(bars[i].get("high")), _num(bars[i].get("low"))
        if high is None or low is None:
            return None
        prev = _num(bars[i - 1].get("close")) if i > 0 else None
        ranges.append(max(high, prev) - min(low, prev) if prev is not None else high - low)
    if len(ranges) < MIN_ATR_BARS:
        return None
    atr = sum(ranges) / len(ranges)
    return atr if atr > 0 else None


def _bracket_r(entry: float, stop: float, side: str, bars: Sequence[Mapping[str, Any]],
               *, fill_bar_stop_only: bool = False) -> tuple[float, str]:
    """1:1 bracket from ``entry``: (R, how). Both touched in one bar = stop first."""
    risk = abs(entry - stop)
    long = side == "long"
    target = entry + risk if long else entry - risk
    for i, bar in enumerate(bars):
        high, low = float(bar["high"]), float(bar["low"])
        stopped = low <= stop if long else high >= stop
        if stopped:
            return -1.0, "stop"
        if fill_bar_stop_only and i == 0:
            continue  # the fill bar's high may have come before the fill
        if (high >= target) if long else (low <= target):
            return 1.0, "target"
    last = float(bars[-1]["close"])
    move = (last - entry) if long else (entry - last)
    return move / risk, "close"


def simulate(alert: Mapping[str, Any], bars: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Both entries for one alert over one session's completed bars.

    ``alert``: family, side, alert_bar (aware datetime = bar start), entry (flag
    close, or None to use the alert bar's close), level (None to use the name's
    extreme from ``episode_start`` to the alert bar), episode_start (Movers only). ``bars``: dicts with dt/open/high/low/close.
    """
    side = str(alert.get("side") or "").lower()
    out: dict[str, Any] = {"family": alert.get("family"), "side": side,
                           "symbol": alert.get("symbol"), "status": "ok"}
    if side not in ("long", "short"):
        return {**out, "status": "bad_side"}
    ordered = sorted(bars, key=lambda bar: bar["dt"])
    at = alert.get("alert_bar")
    index = next((i for i, bar in enumerate(ordered) if bar["dt"] == at), None)
    if index is None:
        return {**out, "status": "no_alert_bar"}
    after = ordered[index + 1:]
    if not after:
        return {**out, "status": "no_bars_after"}
    atr = m5_atr(ordered, index)
    if atr is None:
        return {**out, "status": "no_atr"}
    long = side == "long"
    level = _num(alert.get("level"))
    if level is None and alert.get("episode_start") is not None:
        window = [bar for bar in ordered[: index + 1] if bar["dt"] >= alert["episode_start"]]
        if window:
            level = (min(float(bar["low"]) for bar in window) if long
                     else max(float(bar["high"]) for bar in window))
    if level is None:
        return {**out, "status": "no_level"}
    entry = _num(alert.get("entry"))
    if entry is None:
        entry = float(ordered[index]["close"])
    limit = level + RETEST_ATR_FRACTION * atr if long else level - RETEST_ATR_FRACTION * atr
    stop = level - STOP_ATR * atr if long else level + STOP_ATR * atr
    if (entry <= stop) if long else (entry >= stop):
        return {**out, "status": "bad_geometry"}
    flag_r, flag_how = _bracket_r(entry, stop, side, after)
    out.update(atr=atr, level=level, stop=stop,
               flag_entry=entry, flag_r=flag_r, flag_exit=flag_how, retest_limit=limit)
    for offset, bar in enumerate(after[:RETEST_WINDOW_BARS]):
        open_ = float(bar["open"])
        touched = float(bar["low"]) <= limit if long else float(bar["high"]) >= limit
        if not touched:
            continue
        fill = min(open_, limit) if long else max(open_, limit)
        if (fill <= stop) if long else (fill >= stop):
            # Opened through the stop: the limit would be stopped at once.
            out.update(retest_filled=True, retest_fill=fill, retest_bar=offset + 1,
                       retest_r=-1.0, retest_exit="gap_through_stop")
            return out
        r, how = _bracket_r(fill, stop, side, after[offset:], fill_bar_stop_only=True)
        out.update(retest_filled=True, retest_fill=fill, retest_bar=offset + 1,
                   retest_r=r, retest_exit=how)
        return out
    out.update(retest_filled=False, retest_r=None, retest_exit="no_fill")
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per (source, family, side): n, flag EV, retest EV per fill and per alert, no-fill share."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    for row in results:
        if row.get("status") != "ok":
            skipped[str(row.get("status"))] += 1
            continue
        key = (str(row.get("source") or ""), str(row.get("family") or ""), str(row.get("side") or ""))
        groups[key].append(row)
    families = []
    for (source, family, side), rows in groups.items():
        filled = [float(r["retest_r"]) for r in rows if r.get("retest_filled")]
        n = len(rows)
        flag = [float(r["flag_r"]) for r in rows]
        families.append({
            "source": source, "family": family, "side": side, "n": n,
            "flag_ev_r": _mean(flag),
            "flag_win_share": sum(1 for v in flag if v > 0) / n,
            "retest_n_filled": len(filled),
            "retest_no_fill_share": (n - len(filled)) / n,
            "retest_ev_r_per_fill": _mean(filled),
            # A no-fill is no trade: 0R, so this is comparable to the flag EV per alert.
            "retest_ev_r_per_alert": sum(filled) / n,
        })
    families.sort(key=lambda row: (row["source"], -row["n"], row["family"], row["side"]))
    return {"families": families, "skipped": dict(sorted(skipped.items()))}


# ---------------------------------------------------------------- loaders
def _parse_json(text) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _alert_bar_from_candle(candle_json, local_zone) -> datetime | None:
    bounce = _parse_json(candle_json).get("bounce") or {}
    raw = " ".join(str(bounce.get("time") or "").split())
    try:
        naive = datetime.strptime(raw, "%Y%m%d %H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=local_zone)


def load_m5_alerts(path: Path, *, since: date, local_zone) -> list[dict[str, Any]]:
    """Confirmed M5 alerts since ``since``, one per event_id. Chunked, usecols only."""
    import pandas as pd

    alerts: dict[str, dict[str, Any]] = {}
    if not Path(path).is_file():
        return []
    reader = pd.read_csv(path, usecols=list(M5_ALERT_COLUMNS), chunksize=CSV_CHUNK_ROWS,
                         dtype=str, keep_default_na=False)
    since_text = since.isoformat()
    for chunk in reader:
        chunk = chunk[(chunk["event_type"] == "confirmed") & (chunk["trade_date"] >= since_text)]
        for row in chunk.itertuples(index=False):
            event_id = str(row.event_id)
            if not event_id or event_id in alerts:
                continue
            types = [part for part in str(row.bounce_types).split(";") if part]
            if not types:
                continue
            levels = _parse_json(row.levels_json)
            level = _num(levels.get(types[0]))
            alert_bar = _alert_bar_from_candle(row.candle_json, local_zone)
            try:
                session = date.fromisoformat(str(row.trade_date)[:10])
            except ValueError:
                continue
            alerts[event_id] = {
                "source": "m5_alert", "family": types[0], "side": str(row.direction).lower(),
                "symbol": str(row.symbol).upper(), "session": session, "alert_bar": alert_bar,
                "entry": _num(row.entry_price), "level": level,
                "_no_level": level is None,
            }
    return list(alerts.values())


def load_movers_dip(path: Path, *, since: date) -> list[dict[str, Any]]:
    """Dip-log flag rows since ``since``. The level is computed from bars later."""
    alerts: list[dict[str, Any]] = []
    try:
        handle = Path(path).open("r", encoding="utf-8")
    except OSError:
        return []
    with handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("kind") != "flag":
                continue
            try:
                session = date.fromisoformat(str(row.get("session"))[:10])
                flagged = datetime.fromisoformat(str(row.get("flagged_bar")))
                start = datetime.fromisoformat(str(row.get("episode")))
            except ValueError:
                continue
            if session < since or flagged.tzinfo is None or start.tzinfo is None:
                continue
            side = str(row.get("side") or "").lower()
            state = str(row.get("state") or ("pullback" if side == "long" else "bounce"))
            alerts.append({
                "source": "movers_dip", "family": f"movers_{state}", "side": side,
                "symbol": str(row.get("symbol") or "").upper(), "session": session,
                "alert_bar": flagged, "episode_start": start, "entry": None,
                "level": None,
            })
    return alerts


def lake_bars_for(store) -> BarsFor:
    """Reads one session's completed RTH M5 bars for the named symbols from ``bar_m5``."""

    def read(session: date, symbols: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        start = datetime.combine(session, time(0, 0), EXCHANGE_ZONE).astimezone(timezone.utc)
        rows = store.read_rows(
            "bar_m5", f"month={session:%Y-%m}", symbols=sorted(set(symbols)),
            columns=["symbol", "interval_start", "session_phase", "is_complete",
                     "open", "high", "low", "close"],
            interval_start_range=(start, start + timedelta(days=1)),
        )
        by_symbol: dict[str, dict[datetime, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            if row.get("session_phase") != "RTH" or row.get("is_complete") is False:
                continue
            dt = row.get("interval_start")
            if not isinstance(dt, datetime) or dt.tzinfo is None:
                continue
            values = [_num(row.get(key)) for key in ("open", "high", "low", "close")]
            if any(value is None for value in values):
                continue
            # Duplicate revisions of one bar: the last read wins, one row per bar.
            by_symbol[str(row.get("symbol"))][dt] = dict(
                zip(("open", "high", "low", "close"), values, strict=True), dt=dt)
        return {symbol: sorted(bars.values(), key=lambda bar: bar["dt"])
                for symbol, bars in by_symbol.items()}

    return read


# ---------------------------------------------------------------- runner
def run_study(
    alerts: Iterable[Mapping[str, Any]],
    bars_for: BarsFor,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Simulate every alert against its session's bars; returns the report dict."""
    by_session: dict[date, list[Mapping[str, Any]]] = defaultdict(list)
    for alert in alerts:
        by_session[alert["session"]].append(alert)
    results: list[dict[str, Any]] = []
    for session in sorted(by_session):
        batch = by_session[session]
        try:
            bars = bars_for(session, [str(a.get("symbol")) for a in batch]) or {}
        except Exception as exc:  # noqa: BLE001 - one unreadable session is unknown, not fatal
            bars = {}
            results.append({"status": f"bars_unreadable: {type(exc).__name__}"})
        for alert in batch:
            base = {"source": alert.get("source"), "family": alert.get("family"),
                    "side": alert.get("side"), "symbol": alert.get("symbol")}
            if alert.get("alert_bar") is None:
                results.append({**base, "status": "no_alert_time"})
                continue
            if alert.get("_no_level"):
                results.append({**base, "status": "no_level"})
                continue
            series = bars.get(str(alert.get("symbol")))
            if not series:
                results.append({**base, "status": "no_bars"})
                continue
            results.append({**base, **simulate(alert, series), "source": alert.get("source")})
    report = summarize(results)
    report.update(
        schema=SCHEMA,
        generated_at=(now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        sessions=len(by_session),
        first_session=min(by_session).isoformat() if by_session else None,
        last_session=max(by_session).isoformat() if by_session else None,
        params={
            "retest_atr_fraction": RETEST_ATR_FRACTION, "retest_window_bars": RETEST_WINDOW_BARS,
            "atr_bars": ATR_BARS, "stop_atr": STOP_ATR, "exit": "1:1 bracket",
        },
    )
    return report


def study_from_live_inputs(store, *, since_days: int = DEFAULT_SINCE_DAYS,
                           candidates_path: Path | None = None,
                           movers_path: Path | None = None,
                           today: date | None = None) -> dict[str, Any]:
    """Loads both alert sources (read-only) and runs the study against the lake."""
    import project_paths
    from market_session import get_market_local_timezone

    since = (today or date.today()) - timedelta(days=int(since_days))
    local_zone, _name = get_market_local_timezone()
    alerts = load_m5_alerts(candidates_path or project_paths.INTRADAY_BOUNCE_CANDIDATES_FILE,
                            since=since, local_zone=local_zone)
    alerts += load_movers_dip(movers_path or project_paths.MOVERS_DIP_OUTCOMES_FILE, since=since)
    return run_study(alerts, lake_bars_for(store))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since-days", type=int, default=DEFAULT_SINCE_DAYS)
    parser.add_argument("--lake", type=Path, default=None, help="lake root (default: configured)")
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--movers", type=Path, default=None)
    args = parser.parse_args(argv)
    from research_warehouse.store import ResearchStore

    store = ResearchStore(args.lake) if args.lake else ResearchStore.open()
    if store is None:
        print("research warehouse is not configured")
        return 2
    report = study_from_live_inputs(store, since_days=args.since_days,
                                    candidates_path=args.candidates, movers_path=args.movers)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
