"""Bars pack: where a name is trading now, from the bounce bot's cached M5 bars on disk. Read-only.

The bot keeps its M5 bars in memory; the only on-disk copy is the research spool's M5 tee
(``bar_m5`` rows in ``research_spool/segment-*.jsonl``), which archives completed bars
only, after the champion is done with them. This pack reads those files and nothing else:
never IB, never the bot process. A name the bot does not watch is not cached; the pack
says so. Naive bar times are market-local (``market_session.get_market_local_timezone``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "bars_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Where one stock is trading now from the bot's cached 5-minute bars: the last completed bars "
            "(time ET, OHLC, volume), today's open/high/low, the last price and its age, and an approximate "
            "session VWAP when the cache covers the whole session. Only names the bot watches are cached."
        ),
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "the ticker"},
            "n": {"type": "integer", "description": "how many recent completed bars (default 12)"},
        }, "required": ["symbol"]},
    },
}

ET = ZoneInfo("America/New_York")
BAR = timedelta(minutes=5)
STALE_MINUTES = 15
MAX_BARS = 78
RTH_OPEN, RTH_CLOSE = time(9, 30), time(16, 0)


@dataclass(frozen=True)
class Sources:
    """``bars(symbol)`` returns raw cached bar dicts (``interval_start``/``dt``, OHLCV, optional ``vwap``)."""

    bars: Callable[[str], Iterable[Mapping[str, Any]]]
    market_tz: Callable[[], Any]


def _spool_files() -> list[Path]:
    from research_warehouse import config

    root = Path(config.research_spool_dir())
    return sorted(root.glob("segment-*.jsonl")) if root.exists() else []


def read_spool_bars(symbol: str, files: Iterable[Path]) -> list[dict[str, Any]]:
    """``bar_m5`` rows for one symbol from spool segments; a torn last line is skipped."""
    needle = f'"symbol": "{symbol}",'
    out: list[dict[str, Any]] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if needle not in line or '"bar_m5"' not in line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if record.get("dataset") == "bar_m5" and isinstance(record.get("row"), dict):
                        out.append(record["row"])
        except OSError:
            continue
    return out


def _live_market_tz() -> Any:
    import market_session

    return market_session.get_market_local_timezone()[0]


#: P18: the desk publisher's day files read for a symbol (today's and the one before).
PUBLISHER_DAYS = 2
SOURCE_LABELS = {"desk_publisher": "the desk's M5 publisher", "spool": "the research spool tee"}


def _publisher_files(directory: Path | None = None) -> list[Path]:
    if directory is None:
        from project_paths import M5_BARS_DIR

        directory = Path(M5_BARS_DIR)
    try:
        return sorted(Path(directory).glob("????-??-??.jsonl"))[-PUBLISHER_DAYS:]
    except OSError:
        return []


def read_publisher_bars(symbol: str, files: Iterable[Path]) -> list[dict[str, Any]]:
    """One symbol's rows from the desk publisher's ``<date>.jsonl`` files (``source`` = desk_publisher)."""
    needle = f'"symbol": "{symbol}"'
    out: list[dict[str, Any]] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if needle not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict) and row.get("symbol") == symbol:
                        out.append({**row, "interval_start": row.get("start"), "source": "desk_publisher"})
        except OSError:
            continue
    return out


def preferred_bars(symbol: str, publisher_files: Iterable[Path], spool_files: Callable[[], Iterable[Path]]
                   ) -> list[dict[str, Any]]:
    """The publisher's bars when it has the name, else the spool tee's (each row labelled with its source)."""
    rows = read_publisher_bars(symbol, publisher_files)
    if rows:
        return rows
    return [{**row, "source": "spool"} for row in read_spool_bars(symbol, spool_files())]


def live_sources() -> Sources:
    return Sources(bars=lambda symbol: preferred_bars(symbol, _publisher_files(), _spool_files),
                   market_tz=_live_market_tz)


def _float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def bar_start(raw: Mapping[str, Any], market_tz: Any) -> datetime | None:
    """The bar's start, tz-aware. A naive time is market-local (the bot's bars are)."""
    value = raw.get("interval_start") or raw.get("dt") or raw.get("datetime")
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value or "").strip())
        except ValueError:
            return None
    return moment if moment.tzinfo else moment.replace(tzinfo=market_tz)


def age_text(minutes: float) -> str:
    """Minutes under 120, hours under 48, else days."""
    if minutes < 120:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} days"


def _clean(symbol: Any) -> str:
    return str(symbol or "").strip().upper().lstrip("$")


def build(symbol: str = "", n: int = 12, *, now: datetime | None = None, sources: Sources | None = None) -> Pack:
    """Build the bars pack. File reads only: call it on a worker."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    sym = _clean(symbol)
    if not sym:
        return make_pack(NAME, (), empty_text="bars_pack needs a symbol")
    n = max(1, min(MAX_BARS, int(n or 12)))
    src = sources or live_sources()
    market_tz = src.market_tz()
    bars: dict[datetime, dict[str, Any]] = {}
    for raw in src.bars(sym) or ():
        start = bar_start(raw, market_tz)
        values = [_float(raw.get(key)) for key in ("open", "high", "low", "close")]
        if start is None or None in values or raw.get("is_complete") is False:
            continue
        # Completed bars only: a bar whose 5 minutes have not closed is never shown.
        if start + BAR > moment:
            continue
        bars[start] = {"start": start, "open": values[0], "high": values[1], "low": values[2], "close": values[3],
                       "volume": _float(raw.get("volume")) or 0.0, "vwap": _float(raw.get("vwap")),
                       "source": str(raw.get("source") or "")}
    if not bars:
        return make_pack(NAME, [{"id": f"bars:{sym}:none", "kind": "none",
                                 "text": (f"{sym}: no cached 5-minute bars. The bot only caches names it is "
                                          "watching; price now unknown")}])
    ordered = [bars[key] for key in sorted(bars)]
    last = ordered[-1]
    last_end = last["start"] + BAR
    age_min = (moment - last_end).total_seconds() / 60.0
    stale = age_min > STALE_MINUTES
    rows: list[dict[str, Any]] = [{
        "id": f"bars:{sym}:asof", "kind": "asof", "symbol": sym, "stale": stale,
        "at_utc": last_end.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "text": (f"{sym} cached M5 bars: last completed bar closed {last_end.astimezone(ET):%Y-%m-%d %H:%M} ET, "
                 f"{age_text(age_min)} ago" + (f" (STALE > {STALE_MINUTES} min)" if stale else "")
                 + (f"; from {SOURCE_LABELS.get(last['source'], last['source'])}" if last["source"] else "")),
    }]
    if last["source"]:
        rows[0]["source"] = last["source"]
    shown = ordered[-n:]
    for i, bar in enumerate(shown, 1):
        rows.append({
            "id": f"bars:{sym}:bar:{i}", "kind": "bar",
            "at_utc": bar["start"].astimezone(timezone.utc).isoformat(timespec="seconds"),
            "text": (f"{bar['start'].astimezone(ET):%H:%M} ET O {bar['open']:.2f} H {bar['high']:.2f} "
                     f"L {bar['low']:.2f} C {bar['close']:.2f} V {bar['volume']:,.0f}"),
        })
    day = last["start"].astimezone(ET).date()
    session = [bar for bar in ordered if bar["start"].astimezone(ET).date() == day
               and RTH_OPEN <= bar["start"].astimezone(ET).time() < RTH_CLOSE]
    if session:
        high = max(bar["high"] for bar in session)
        low = min(bar["low"] for bar in session)
        vwap, vwap_text = session_vwap(session)
        first = session[0]["start"].astimezone(ET)
        if first.time() == RTH_OPEN:
            span = f"regular session so far: open {session[0]['open']:.2f}, high {high:.2f}, low {low:.2f}"
        else:
            # The cache starts mid-session: its high/low are the cached bars', and the open is unknown.
            span = (f"cached from {first:%H:%M} ET only (today's open not cached): high {high:.2f}, "
                    f"low {low:.2f} of the cached bars")
        rows.append({"id": f"bars:{sym}:day", "kind": "day",
                     "text": (f"{sym} {day} {span}, {len(session)} bars; {vwap_text}; last "
                              f"{last['close']:.2f} is {vwap_relation(last['close'], vwap)}")})
    else:
        vwap = None
        rows.append({"id": f"bars:{sym}:day", "kind": "day",
                     "text": f"{sym} {day}: no regular-session bars cached (pre/after-hours only)"})
    rows.append({
        "id": f"bars:{sym}:last", "kind": "last", "symbol": sym, "price": last["close"],
        "text": (f"{sym} last {last['close']:.2f} at {last_end.astimezone(ET):%H:%M} ET (close of the last completed "
                 f"bar), {vwap_relation(last['close'], vwap)}" + (f"; stale, {age_text(age_min)} old" if stale else "")),
    })
    if vwap is not None:
        rows[-1]["vwap"] = round(vwap, 4)
    return make_pack(NAME, rows)


def session_vwap(session: list[dict[str, Any]]) -> tuple[float | None, str]:
    """(value, text): the cache's own VWAP, else typical-price VWAP when the bars cover 09:30 on with no gap."""
    first = session[0]["start"].astimezone(ET)
    complete = first.time() == RTH_OPEN and all(
        later["start"] - earlier["start"] == BAR for earlier, later in zip(session, session[1:], strict=False))
    volume = sum(bar["volume"] for bar in session)
    if not complete or volume <= 0:
        return None, "VWAP not in cache (the cached bars do not cover the whole session)"
    if all(bar["vwap"] is not None for bar in session):
        value = sum(bar["vwap"] * bar["volume"] for bar in session) / volume
        return value, f"session VWAP {value:.2f} (from the bars' own VWAP)"
    value = sum((bar["high"] + bar["low"] + bar["close"]) / 3.0 * bar["volume"] for bar in session) / volume
    return value, f"approx session VWAP {value:.2f} (typical price x volume of the cached bars; VWAP not in cache)"


def vwap_relation(price: float, vwap: float | None) -> str:
    """``below session VWAP 279.00 by 0.12 (0.04%)``, computed here so the model never guesses the side."""
    if vwap is None or vwap <= 0:
        return "above/below session VWAP unknown (VWAP not in cache)"
    gap = price - vwap
    if abs(gap) < 0.005:
        return f"at session VWAP {vwap:.2f}"
    side = "above" if gap > 0 else "below"
    return f"{side} session VWAP {vwap:.2f} by {abs(gap):.2f} ({abs(gap) / vwap * 100:.2f}%)"


FIXTURE_NOW = datetime(2026, 9, 30, 14, 3, tzinfo=timezone.utc)  # 10:03 ET


def fixture_rows() -> list[dict[str, Any]]:
    """Seven 5-minute bars 09:30-10:00 ET in the spool's shape; the 10:00 bar is still forming at 10:03."""
    out = []
    for i in range(7):
        start = datetime(2026, 9, 30, 13, 30, tzinfo=timezone.utc) + BAR * i
        base = 100.0 + i
        out.append({"symbol": "ALL", "interval_start": start.isoformat(), "open": base, "high": base + 1.0,
                    "low": base - 0.5, "close": base + 0.5, "volume": 1000 * (i + 1), "vwap": None,
                    "is_complete": True})
    return out


def fixture_sources(rows: list[dict[str, Any]] | None = None) -> Sources:
    data = fixture_rows() if rows is None else rows
    return Sources(bars=lambda symbol: [row for row in data if row.get("symbol") == symbol],
                   market_tz=lambda: ZoneInfo("America/Los_Angeles"))


def fixture() -> Pack:
    return build("ALL", n=3, now=FIXTURE_NOW, sources=fixture_sources())
