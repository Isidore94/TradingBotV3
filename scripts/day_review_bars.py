"""Durable completed M5 bars for a closed Day Review session (TJ-2A).

This module has no Qt dependency.  Network access is confined to
``fetch_session_bars`` and tests pass its downloader explicitly.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import yfinance as yf

import daily_recap_reader
from completed_bars import completed_m5_bars
from project_paths import DAY_REVIEW_DIR
from ui.services.market_journal_service import shared_journal_service

_log = logging.getLogger(__name__)
BENCHMARKS = ("SPY", "QQQ", "IWM", "VXX")
CHUNK_SIZE = 50
MARKET_ZONE = ZoneInfo("America/Los_Angeles")


def _session_text(session: str | date) -> str:
    return str(session or "")[:10]


def session_is_closed(session: str | date, *, now: datetime | None = None) -> bool:
    """Whether ``session`` is complete according to the exchange calendar."""
    try:
        import market_calendar

        return date.fromisoformat(_session_text(session)) <= market_calendar.last_completed_session(
            now or datetime.now()
        )
    except Exception:
        return False


def session_is_backfillable(session: str | date, *, now: datetime | None = None) -> bool:
    """Yahoo's M5 window is short; old absent files are not retried forever."""
    try:
        moment = now or datetime.now(MARKET_ZONE)
        return session_is_closed(session, now=moment) and date.fromisoformat(
            _session_text(session)
        ) >= moment.date() - timedelta(days=60)
    except ValueError:
        return False


def decided_symbols(session: str, sources: Any) -> set[str]:
    """Names with a trader decision or trade, plus the four fixed benchmarks."""
    names = set(BENCHMARKS)
    try:
        if all(hasattr(sources, name) for name in ("annotations", "pick_feedback", "swing_favorites", "review_events")):
            annotations = daily_recap_reader._read_jsonl("annotations", sources.annotations, "created_at")
            feedback = daily_recap_reader._read_jsonl("pick_feedback", sources.pick_feedback, "ts")
            favorites = daily_recap_reader._read_jsonl("swing_favorites", sources.swing_favorites, "event_at")
            events = daily_recap_reader._read_jsonl("review_events", sources.review_events, "ts")
            decisions = daily_recap_reader._decisions(
                _session_text(session), annotations, feedback, favorites, events
            )
        else:  # narrow test seam; production always has RecapSources paths
            decisions = daily_recap_reader._decisions(_session_text(session), sources=sources)
        for row in decisions:
            symbol = str(row.get("symbol") if isinstance(row, Mapping) else getattr(row, "symbol", "") or "").strip().upper()
            if symbol:
                names.add(symbol)
    except Exception:
        _log.info("Day Review decisions were unavailable while selecting session bars.", exc_info=True)
    try:
        journal = shared_journal_service()
        reader = getattr(journal, "list_trades", None)
        if callable(reader):
            trades = reader(trade_date=_session_text(session))
        else:
            from ui.services.journal_feed import trades_on

            trades = trades_on(_session_text(session))
        for row in trades:
            symbol = str(row.get("symbol") or "").strip().upper()
            if symbol:
                names.add(symbol)
    except Exception:
        _log.info("Day Review trades were unavailable while selecting session bars.", exc_info=True)
    return names


def _chunks(symbols: Iterable[str]) -> Iterable[tuple[str, ...]]:
    unique = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    for start in range(0, len(unique), CHUNK_SIZE):
        yield tuple(unique[start : start + CHUNK_SIZE])


def _rows_from_download(download: Any, symbols: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    """Accept injected dict fixtures and yfinance's single/multi ticker frames."""
    if isinstance(download, Mapping):
        return {str(key).upper(): list(value or ()) for key, value in download.items() if value}
    result: dict[str, list[dict[str, Any]]] = {}
    if download is None or getattr(download, "empty", True):
        return result
    columns = getattr(download, "columns", ())
    multi = getattr(columns, "nlevels", 1) > 1
    for symbol in symbols:
        try:
            frame = download.xs(symbol, axis=1, level=1) if multi else download
            if multi and frame.empty:
                continue
            rows = []
            for stamp, row in frame.iterrows():
                rows.append({
                    "dt": stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp,
                    "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]), "close": float(row["Close"]),
                    "volume": int(row.get("Volume", 0) or 0),
                })
            if rows:
                result[symbol] = rows
        except Exception:
            _log.info("Day Review bars for %s were unavailable.", symbol, exc_info=True)
    return result


def _normalise(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    now = datetime.now(MARKET_ZONE)
    answer: list[dict[str, Any]] = []
    for row in completed_m5_bars(list(rows), now=now):
        try:
            stamp = row["dt"]
            if not isinstance(stamp, datetime):
                stamp = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
            stamp = stamp.replace(tzinfo=MARKET_ZONE) if stamp.tzinfo is None else stamp.astimezone(MARKET_ZONE)
            if not (time(9, 30) <= stamp.timetz().replace(tzinfo=None) < time(16, 0)):
                continue
            answer.append({"dt": stamp, "open": float(row["open"]), "high": float(row["high"]),
                           "low": float(row["low"]), "close": float(row["close"]),
                           "volume": int(row.get("volume", 0) or 0)})
        except (KeyError, TypeError, ValueError):
            continue
    return answer


def fetch_session_bars(symbols: Iterable[str], session: str, *, downloader=None) -> dict[str, list[dict[str, Any]]]:
    """Download regular-hours completed M5 bars, batching symbols at fifty."""
    day = date.fromisoformat(_session_text(session))
    download = downloader or yf.download
    answer: dict[str, list[dict[str, Any]]] = {}
    for tickers in _chunks(symbols):
        try:
            payload = download(tickers, start=day.isoformat(), end=(day + timedelta(days=1)).isoformat(),
                               interval="5m", auto_adjust=False, prepost=False, progress=False, group_by="column")
        except Exception:
            _log.info("Day Review M5 download failed for %s.", ",".join(tickers), exc_info=True)
            continue
        for symbol, rows in _rows_from_download(payload, tickers).items():
            normal = _normalise(rows)
            if normal:
                answer[symbol] = normal
    return answer


def bars_path(session: str | date) -> Path:
    return Path(DAY_REVIEW_DIR) / "bars" / f"{_session_text(session)}.parquet"


def write_session_bars(session: str, bars: Mapping[str, Iterable[Mapping[str, Any]]]) -> Path:
    """Atomically write one durable parquet file; an open session is refused."""
    if not session_is_closed(session):
        raise ValueError("Day Review bars require a closed session")
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for symbol, values in bars.items():
        for value in values:
            stamp = value.get("dt")
            if isinstance(stamp, str):
                stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if not isinstance(stamp, datetime):
                continue
            stamp = stamp.replace(tzinfo=MARKET_ZONE) if stamp.tzinfo is None else stamp.astimezone(MARKET_ZONE)
            rows.append({"symbol": str(symbol).upper(), "dt": stamp, "open": float(value["open"]),
                         "high": float(value["high"]), "low": float(value["low"]), "close": float(value["close"]),
                         "volume": int(value.get("volume", 0) or 0)})
    path = bars_path(session)
    temp = path.with_suffix(".parquet.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        schema = pa.schema([
            ("symbol", pa.string()), ("dt", pa.timestamp("us", tz="America/Los_Angeles")),
            ("open", pa.float64()), ("high", pa.float64()), ("low", pa.float64()),
            ("close", pa.float64()), ("volume", pa.int64()),
        ])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temp)
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
    return path


def read_session_bars(session: str) -> dict[str, list[dict[str, Any]]] | None:
    """Read bars grouped by symbol, or ``None`` when no durable file exists."""
    path = bars_path(session)
    if not path.is_file():
        return None
    try:
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    except Exception:
        _log.info("Day Review bars %s were unreadable.", path, exc_info=True)
        return None
    answer: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stamp = row.get("dt")
        if isinstance(stamp, datetime):
            stamp = stamp.replace(tzinfo=MARKET_ZONE) if stamp.tzinfo is None else stamp.astimezone(MARKET_ZONE)
        answer.setdefault(str(row.get("symbol") or "").upper(), []).append({
            "dt": stamp, "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]),
            "close": float(row["close"]), "volume": int(row.get("volume", 0) or 0)})
    return answer


__all__ = ["BENCHMARKS", "bars_path", "decided_symbols", "fetch_session_bars", "read_session_bars", "session_is_closed", "write_session_bars"]
