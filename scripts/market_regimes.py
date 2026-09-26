"""Multi-timeframe auto regimes (S17 half 1): the champion env_key on M5..W.

One definition of the champion Auto Market Bias reads over stored bars, shared
by `journal_regime_fill` (the journal's `regimes` rows) and the night
`market_regime_table` slot (an append-only JSONL, one row per session and
symbol). Same classifier, same thresholds: every read goes through
`research_warehouse.market_bias_context` (`_champion_read` / `context_at`).

* M5, M30, H1, H4: `context_at` over the symbol's M5 bars completed by the
  snapshot (10:00 ET, 12:00 ET and the close); H1/H4 are aggregated from M5
  exactly as M30 is.
* D1: the 20 sessions completed before the session.
* W: the 20 finished weeks before the session, built from those daily bars.

Point in time: nothing reads the session's own D1 bar or a bar after the
snapshot. Missing bars give `unknown`. The table is never re-labelled: a
(session, symbol) already in the file is skipped.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")
RULE = "market_regime_table_v1"
UNKNOWN = "unknown"
PRIMARY = "SPY"
INDEXES = ("SPY", "QQQ", "IWM")
INTRADAY_TIMEFRAMES = ("M5", "M30", "H1", "H4")
TIMEFRAMES = INTRADAY_TIMEFRAMES + ("D1", "W")
#: (label, ET clock time); None is the session's own close (half days included).
SNAPSHOTS = (("10:00", time(10, 0)), ("12:00", time(12, 0)), ("close", None))
D1_WINDOW = 20
W_WINDOW = 20
#: M5 history handed to the intraday reads: 20 completed H4 bars need ~11 sessions.
INTRADAY_LOOKBACK = timedelta(days=21)


def sector_etfs() -> tuple[str, ...]:
    """The sector ETFs the desk's group tape already tracks."""
    from group_rrs import SECTOR_ETFS

    return tuple(SECTOR_ETFS.values())


def table_symbols() -> tuple[str, ...]:
    return INDEXES + tuple(symbol for symbol in sector_etfs() if symbol not in INDEXES)


# ---------------------------------------------------------------- reads --


def session_day(row: Mapping[str, Any]) -> date | None:
    value = row.get("session_date")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def completed_before(rows: Iterable[Mapping[str, Any]], day: date) -> list[Mapping[str, Any]]:
    kept = [row for row in rows or () if (session_day(row) or day) < day]
    return sorted(kept, key=lambda row: session_day(row))


def d1_env_key(completed: Sequence[Mapping[str, Any]], window: int) -> str:
    """Champion env_key over the last `window` completed bars, or `unknown`."""
    if len(completed) <= window:
        return UNKNOWN
    from research_warehouse import market_bias_context as bias

    try:
        reference = float(completed[-window - 1].get("close"))
    except (TypeError, ValueError):
        return UNKNOWN
    reading = bias._champion_read([dict(row) for row in completed[-window:]], reference)
    return str(reading.get("env_key") or UNKNOWN)


def weekly_env_key(completed_d1: Sequence[Mapping[str, Any]], day: date) -> str:
    """Champion env_key over the last `W_WINDOW` weeks finished before `day`."""
    from market_structure import completed_weekly_bars

    return d1_env_key(completed_weekly_bars(completed_d1, day), W_WINDOW)


def intraday_env_keys(
    moment: datetime,
    m5_rows: Sequence[Mapping[str, Any]],
    d1_rows: Sequence[Mapping[str, Any]],
    *,
    lookback: timedelta = INTRADAY_LOOKBACK,
) -> tuple[dict[str, str], str]:
    """`({M5, M30, H1, H4: env_key}, note)` from bars completed by `moment`; note is "" when read."""
    from research_warehouse import market_bias_context as bias

    unknown = {timeframe: UNKNOWN for timeframe in INTRADAY_TIMEFRAMES}
    d1 = [{**dict(row), "session_date": session_day(row)} for row in d1_rows]
    # Bound the tape so a year of bars is not re-aggregated per read.
    floor = moment - lookback
    window = [
        row for row in m5_rows
        if isinstance(row.get("interval_start"), datetime) and floor <= row["interval_start"] < moment
    ]
    if not window:
        return unknown, "no M5 bars"
    readings = bias.context_at(moment, spy_m5=window, spy_d1=d1)
    return {timeframe: str(readings.get(timeframe, {}).get("env_key") or UNKNOWN) for timeframe in INTRADAY_TIMEFRAMES}, ""


def snapshot_moment(day: date, clock: time | None) -> datetime | None:
    """The instant a snapshot describes, capped at the session close; None off-session."""
    from research_warehouse import exchange_calendar as xcal

    session = xcal.trading_session(day)
    if session is None:
        return None
    if clock is None:
        return session.rth_close_at
    return min(datetime.combine(day, clock, tzinfo=MARKET_TZ), session.rth_close_at)


def session_row(
    symbol: str,
    day: date,
    *,
    d1_rows: Sequence[Mapping[str, Any]],
    m5_rows: Sequence[Mapping[str, Any]],
    computed_at: datetime,
) -> dict[str, Any]:
    """One table row: six timeframes at the close, three intraday snapshots, structure facts."""
    from market_structure import structure_facts

    completed = completed_before(d1_rows, day)
    snapshots: dict[str, dict[str, str]] = {}
    for label, clock in SNAPSHOTS:
        moment = snapshot_moment(day, clock)
        if moment is None:
            snapshots[label] = {"as_of": "", **{tf: UNKNOWN for tf in INTRADAY_TIMEFRAMES}}
            continue
        keys, _note = intraday_env_keys(moment, m5_rows, completed)
        snapshots[label] = {"as_of": moment.astimezone(MARKET_TZ).isoformat(), **keys}
    close = snapshots["close"]
    session_m5 = sum(
        1 for row in m5_rows
        if isinstance(row.get("interval_start"), datetime)
        and row["interval_start"].astimezone(MARKET_TZ).date() == day
    )
    return {
        "rule": RULE,
        "session_date": day.isoformat(),
        "symbol": symbol,
        "timeframes": {
            **{timeframe: close[timeframe] for timeframe in INTRADAY_TIMEFRAMES},
            "D1": d1_env_key(completed, D1_WINDOW),
            "W": weekly_env_key(completed, day),
        },
        "snapshots": snapshots,
        "structure": structure_facts(completed, day),
        "d1_bars": len(completed),
        "session_m5_bars": session_m5,
        "computed_at": computed_at.astimezone(MARKET_TZ).isoformat(),
    }


# ----------------------------------------------------------- desk strip --


def live_m5_rows(symbol: str, chart_bars: Iterable[Mapping[str, Any]], *, now: datetime, tz=None) -> list[dict]:
    """The bot's cached M5 chart dicts as bar rows, completed bars only.

    `dt` is naive market-local time (`market_session.get_market_local_timezone`);
    a bar still forming at `now` is dropped.
    """
    if tz is None:
        from market_session import get_market_local_timezone

        tz = get_market_local_timezone()[0]
    rows = []
    for bar in chart_bars or ():
        stamp = bar.get("dt")
        if not isinstance(stamp, datetime):
            continue
        start = stamp if stamp.tzinfo else stamp.replace(tzinfo=tz)
        end = start + timedelta(minutes=5)
        if end > now:
            continue
        rows.append(
            {
                "symbol": symbol, "interval_start": start, "interval_end": end,
                **{key: bar.get(key) for key in ("open", "high", "low", "close", "volume")},
                "is_complete": True,
            }
        )
    return rows


def merge_m5(history: Sequence[Mapping[str, Any]], live: Sequence[Mapping[str, Any]]) -> list:
    """Stored history up to the first live bar, then the live bars."""
    if not live:
        return list(history)
    first = min(row["interval_start"] for row in live)
    return [row for row in history if isinstance(row.get("interval_start"), datetime) and row["interval_start"] < first] + list(live)


def strip_moment(now: datetime) -> datetime | None:
    """`now` inside a session; otherwise the last session's close."""
    from research_warehouse import exchange_calendar as xcal

    session = xcal.session_for(now)
    if session is not None and now >= session.rth_open_at:
        return min(now, session.rth_close_at)
    day = now.astimezone(MARKET_TZ).date()
    for back in range(1, 15):
        earlier = xcal.trading_session(day - timedelta(days=back))
        if earlier is not None:
            return earlier.rth_close_at
    return None


def strip_readings(
    now: datetime,
    d1_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    m5_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    symbols: Sequence[str] = INDEXES,
) -> dict[str, Any]:
    """`{"as_of", "symbols": {symbol: {M5..W: env_key}}}` for the desk strip, at `now`."""
    moment = strip_moment(now)
    if moment is None:
        return {"as_of": "", "symbols": {}}
    day = moment.astimezone(MARKET_TZ).date()
    readings: dict[str, dict[str, str]] = {}
    for symbol in symbols:
        completed = completed_before(d1_by_symbol.get(symbol) or (), day)
        keys, _note = intraday_env_keys(moment, m5_by_symbol.get(symbol) or (), completed)
        readings[symbol] = {**keys, "D1": d1_env_key(completed, D1_WINDOW), "W": weekly_env_key(completed, day)}
    return {"as_of": moment.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M ET"), "symbols": readings}


# ---------------------------------------------------------------- table --


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("session_date") or ""), str(row.get("symbol") or "").upper()


def read_table(path: Path | str) -> list[dict[str, Any]]:
    """Every readable row, file order; a torn or foreign line is skipped."""
    rows: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return rows
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_rows(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> int:
    """Append rows whose (session, symbol) is not in the file yet. Returns rows written."""
    target = Path(path)
    seen = {_key(row) for row in read_table(target)}
    fresh = []
    for row in rows:
        key = _key(row)
        if key in seen or not all(key):
            continue
        seen.add(key)
        fresh.append(row)
    if not fresh:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if target.exists() and target.stat().st_size:
        with target.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                prefix = "\n"  # close a torn last line so it cannot swallow the next row
    payload = prefix + "".join(json.dumps(row, sort_keys=True) + "\n" for row in fresh)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return len(fresh)


# ---------------------------------------------------------- regime read --


def _reads_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    from project_paths import REGIME_READS_DIR

    return Path(REGIME_READS_DIR)


def regime_read_path(session_date: str, *, root: Path | str | None = None) -> Path:
    """`REGIME_READS_DIR/<date>.json`; a path only, nothing is opened."""
    return _reads_root(root) / f"{str(session_date or '').strip()[:10]}.json"


def latest_regime_read(on: Any = None, *, root: Path | str | None = None) -> dict[str, Any] | None:
    """The newest verified regime read dated on or before `on` (any date when None).

    Only the night slot writes these files, and only after its checks pass.
    """
    limit = str(on or "")[:10]
    try:
        paths = sorted(_reads_root(root).glob("*.json"), reverse=True)
    except OSError:
        return None
    for path in paths:
        if limit and path.stem > limit:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        read = payload.get("read") if isinstance(payload, dict) else None
        if isinstance(read, dict) and str(read.get("paragraph") or "").strip():
            return payload
    return None


# ----------------------------------------------------------------- bars --


def _d1_rows_from_frame(symbol: str, frame) -> list[dict[str, Any]]:
    rows = []
    for record in frame.to_dict("records"):
        stamp = record.get("datetime")
        day = stamp.date() if hasattr(stamp, "date") else None
        if day is None:
            continue
        values = {key: record.get(key) for key in ("open", "high", "low", "close", "volume")}
        rows.append({**values, "symbol": symbol, "session_date": day})
    return rows


def load_bars(
    d1_symbols: Sequence[str],
    m5_symbols: Sequence[str],
    *,
    m5_since: date | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], str]:
    """Read-only: D1 (lake, then the durable D1 store) and M5 (lake) per symbol.

    Returns `(d1_by_symbol, m5_by_symbol, source text)`. `m5_since` skips lake
    months wholly before it. Never writes; an unreadable source contributes nothing.
    """
    d1_by_symbol: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in d1_symbols}
    m5_by_symbol: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in m5_symbols}
    sources: list[str] = []
    try:
        from research_warehouse.config import get_research_store_dir
        from research_warehouse.store import ResearchStore

        root = get_research_store_dir()
        if root is not None and Path(root).exists():
            lake = ResearchStore(Path(root))
            for row in lake.read_rows("bar_d1", symbols=list(d1_symbols)):
                symbol = str(row.get("symbol") or "").upper()
                if symbol in d1_by_symbol:
                    d1_by_symbol[symbol].append(row)
            partitions = sorted(
                {entry.partition for entry in lake.manifest.resolve(dataset="bar_m5").entries}
            )
            floor = f"month={m5_since:%Y-%m}" if m5_since else ""
            for partition in partitions:
                if floor and str(partition).startswith("month=") and str(partition) < floor:
                    continue
                for row in lake.read_rows("bar_m5", partition, symbols=list(m5_symbols)):
                    symbol = str(row.get("symbol") or "").upper()
                    if symbol in m5_by_symbol:
                        m5_by_symbol[symbol].append(row)
            sources.append("lake")
    except Exception:  # noqa: BLE001 - an unreadable lake is unknown, never a failure
        logging.debug("Research lake unreadable for the regime reads.", exc_info=True)
    try:
        from research_warehouse.ingest_existing import read_durable_daily_bars

        for symbol in d1_symbols:
            frame = read_durable_daily_bars(symbol)
            if frame is None:
                continue
            known = {session_day(row) for row in d1_by_symbol[symbol]}
            extra = [row for row in _d1_rows_from_frame(symbol, frame) if row["session_date"] not in known]
            if extra:
                d1_by_symbol[symbol].extend(extra)
                if "durable_d1" not in sources:
                    sources.append("durable_d1")
    except Exception:  # noqa: BLE001
        logging.debug("Durable D1 store unreadable for the regime reads.", exc_info=True)
    return d1_by_symbol, m5_by_symbol, "+".join(sources) or "none"


M5_SOURCE_LAKE = "lake"
M5_SOURCE_DAY_REVIEW = "day_review_yahoo"
M5_SOURCE_NONE = "none"


def _day_review_rows(symbol: str, bars: Iterable[Mapping[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Day Review store bars (`dt` = zone-aware bar start) as bar rows; completed by `now` only."""
    rows = []
    for bar in bars or ():
        stamp = bar.get("dt")
        if not isinstance(stamp, datetime) or stamp.tzinfo is None:
            continue
        start = stamp.astimezone(MARKET_TZ)
        end = start + timedelta(minutes=5)
        if end > now:
            continue
        rows.append(
            {
                "symbol": symbol, "interval_start": start, "interval_end": end,
                **{key: bar.get(key) for key in ("open", "high", "low", "close", "volume")},
                "is_complete": True,
            }
        )
    return rows


def fill_m5_from_day_review(
    m5_by_symbol: dict[str, list[dict[str, Any]]],
    symbols: Sequence[str],
    days: Iterable[date],
    *,
    root: Path | str | None = None,
    now: datetime | None = None,
) -> dict[tuple[str, str], str]:
    """Fill (symbol, session) pairs the lake lacks from the Day Review M5 store, in place.

    Cached files only (`day_review_bars.read_session_bars`), never a download; the
    lake wins whenever it has any bar that session. Returns `{(session iso, symbol):
    "lake" | "day_review_yahoo" | "none"}` for every pair asked about.
    """
    moment = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    wanted = [str(symbol).upper() for symbol in symbols]
    lake_days = {
        symbol: {
            row["interval_start"].astimezone(MARKET_TZ).date()
            for row in m5_by_symbol.get(symbol) or ()
            if isinstance(row.get("interval_start"), datetime)
        }
        for symbol in wanted
    }
    sources: dict[tuple[str, str], str] = {}
    filled: set[str] = set()
    for day in sorted(set(days)):
        need = [symbol for symbol in wanted if day not in lake_days[symbol]]
        for symbol in wanted:
            sources[(day.isoformat(), symbol)] = M5_SOURCE_NONE if symbol in need else M5_SOURCE_LAKE
        if not need:
            continue
        try:
            from day_review_bars import read_session_bars

            stored = read_session_bars(day.isoformat(), root=Path(root) if root is not None else None) or {}
        except Exception:  # noqa: BLE001 - an unreadable file is unknown, never a failure
            logging.debug("Day Review bars unreadable for the regime reads.", exc_info=True)
            continue
        for symbol in need:
            rows = _day_review_rows(symbol, stored.get(symbol) or (), moment)
            if rows:
                m5_by_symbol.setdefault(symbol, []).extend(rows)
                sources[(day.isoformat(), symbol)] = M5_SOURCE_DAY_REVIEW
                filled.add(symbol)
    for symbol in filled:
        m5_by_symbol[symbol].sort(key=lambda row: row["interval_start"])
    return sources


__all__ = [
    "INDEXES",
    "RULE",
    "SNAPSHOTS",
    "TIMEFRAMES",
    "append_rows",
    "completed_before",
    "d1_env_key",
    "fill_m5_from_day_review",
    "intraday_env_keys",
    "latest_regime_read",
    "load_bars",
    "read_table",
    "regime_read_path",
    "session_day",
    "session_row",
    "snapshot_moment",
    "table_symbols",
    "weekly_env_key",
]
