"""What the charts looked like when the trader wrote it down - R10.H follow-on.

R10.H stored the words. A note that says "wicks off the HOD, this is not a
strong market" is only re-readable months later if the tape it was written
against is stored beside it, and re-deriving that tape from a bar archive is
not the same thing: the archive says what the market did, this says what the
trader was *looking at*.

**Bars, not pictures.** A PNG is unreadable by the nightly AI layer and cannot
be re-scaled, re-ranged or measured. What is stored is the bar window itself -
the symbol's M5 and D1, and the benchmark's (SPY) M5 and D1 - so the Market
Journal page redraws the real chart and the model reads real numbers.

**Two stores, deliberately.**

* A **sidecar** JSON per capture holds the bars. It is large-ish (tens of KB)
  and only the page ever reads it.
* A **ledger row** per capture holds a short text ``digest`` - where price sat
  against its session range, VWAP, the prior session's extremes, the daily
  averages and RVOL. That is what the AI grant reads. The raw bar window would
  starve every other source in the packet; the digest costs a few hundred
  characters and says the same thing in words.

**The entry row is never touched.** ``market_journal_entry_v1`` keeps its
meaning; a capture is joined to an entry by ``entry_id`` from the outside. That
is what lets the capture be written *after* the entry, on a worker, without the
note ever waiting on a chart - and a capture that fails leaves an entry that is
honestly chartless rather than an entry that never got saved.

Nothing here decides anything. It measures, formats and stores.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: Schema NAME, never a bare number (ground rule 5).
SCHEMA_MARKET_JOURNAL_CHART = "market_journal_chart_v1"
STREAM_CHARTS = "market_journal_charts"

#: The index every note is implicitly written against. One benchmark, named,
#: so a reader never has to guess which "the market" meant.
BENCHMARK_SYMBOL = "SPY"

#: How much tape a capture keeps. Two sessions of M5 (78 bars each) is enough
#: to see today against yesterday; 120 daily bars is enough for the 20/50 and a
#: useful stretch of the 200 without storing five years per note.
M5_BAR_LIMIT = 160
D1_BAR_LIMIT = 120

#: Why a capture happened. Free text is not enough here - the page groups by it.
REASON_ENTRY = "journal_entry"
REASON_MODE_FLIP = "auto_mode_flip"

DIR_NAME = "market_journal_charts"

_log = logging.getLogger(__name__)


# -- bar normalization ----------------------------------------------------
def _stamp_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")


def _parse_stamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def trim_bars(bars: Iterable[Mapping[str, Any]] | None, limit: int) -> list[dict[str, Any]]:
    """The last ``limit`` bars as plain JSON-safe dicts.

    A bar missing any of its four prices is dropped and not counted: a capture
    is a picture of a chart, and a row the chart could not draw was not in the
    picture. Volume is optional and defaults to 0.0 - a bar with no volume is
    still a bar.
    """
    rows: list[dict[str, Any]] = []
    for bar in list(bars or [])[-max(0, int(limit)) :] if limit else []:
        prices = [_number(bar.get(name)) for name in ("open", "high", "low", "close")]
        if any(price is None for price in prices):
            continue
        rows.append(
            {
                "dt": _stamp_text(bar.get("dt")),
                "open": prices[0],
                "high": prices[1],
                "low": prices[2],
                "close": prices[3],
                "volume": _number(bar.get("volume")) or 0.0,
            }
        )
    return rows


def revive_bars(rows: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Stored bars back in the shape the chart widgets expect (``dt`` a datetime).

    A row whose stamp cannot be parsed is DROPPED and counted by
    :func:`unreadable_bar_count`, not carried through with its text: the axis
    formats every stamp with ``strftime``, so one string would take the chart
    down rather than degrade it. ``trim_bars`` always writes an ISO stamp, so
    this can only happen to a corrupted sidecar - a gap the page says out loud.
    """
    bars: list[dict[str, Any]] = []
    for row in rows or ():
        bar = dict(row)
        stamp = _parse_stamp(bar.get("dt"))
        if stamp is None:
            continue
        bar["dt"] = stamp
        bars.append(bar)
    return bars


def unreadable_bar_count(rows: Iterable[Mapping[str, Any]] | None) -> int:
    """How many stored bars had no readable stamp. A gap, never an absence."""
    rows = list(rows or ())
    return len(rows) - len(revive_bars(rows))


def _bar_date(bar: Mapping[str, Any]):
    stamp = _parse_stamp(bar.get("dt"))
    return stamp.date() if stamp is not None else None


def _sessions(bars: Sequence[Mapping[str, Any]]) -> list[tuple[Any, list[Mapping[str, Any]]]]:
    """Bars grouped by calendar date, in order. Undated bars group under None."""
    grouped: list[tuple[Any, list[Mapping[str, Any]]]] = []
    for bar in bars:
        key = _bar_date(bar)
        if grouped and grouped[-1][0] == key:
            grouped[-1][1].append(bar)
        else:
            grouped.append((key, [bar]))
    return grouped


def _extremes(bars: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    highs = [_number(bar.get("high")) for bar in bars]
    lows = [_number(bar.get("low")) for bar in bars]
    highs = [value for value in highs if value is not None]
    lows = [value for value in lows if value is not None]
    return (max(highs) if highs else None, min(lows) if lows else None)


def _side_word(price: float | None, level: float | None) -> str:
    if price is None or level is None:
        return "unmeasured"
    if price > level:
        return "above"
    if price < level:
        return "below"
    return "at"


# -- the digest -----------------------------------------------------------
def describe_m5(label: str, bars: Sequence[Mapping[str, Any]]) -> str:
    """One line describing an intraday series the way a trader reads it.

    Every clause is dropped rather than guessed when its input is missing.
    "No bars" is stated, because a capture with an empty series must never read
    as a flat or quiet chart.
    """
    if not bars:
        return f"{label} M5: no bars were cached, so this chart was not captured."
    sessions = _sessions(bars)
    today_key, today = sessions[-1]
    prior = sessions[-2][1] if len(sessions) > 1 else []
    last = bars[-1]
    close = _number(last.get("close"))
    open_price = _number(today[0].get("open")) if today else None
    high, low = _extremes(today)
    parts = [
        f"{label} M5: {len(bars)} bars, last {_stamp_text(last.get('dt'))[:16]}"
    ]
    if close is not None:
        parts.append(f"last {close:.2f}")
    if close is not None and open_price:
        parts.append(f"session open {open_price:.2f} ({(close / open_price - 1) * 100:+.2f}%)")
    if high is not None and low is not None:
        parts.append(f"session H {high:.2f} L {low:.2f}")
        span = high - low
        if close is not None and span > 0:
            parts.append(f"close sits {(close - low) / span * 100:.0f}% up the session range")
    vwap = _session_vwap_last(bars)
    if vwap is not None:
        parts.append(f"session VWAP {vwap:.2f} ({_side_word(close, vwap)})")
    if prior:
        prior_high, prior_low = _extremes(prior)
        if prior_high is not None and prior_low is not None:
            parts.append(
                f"prior session H {prior_high:.2f} ({_side_word(close, prior_high)}) "
                f"L {prior_low:.2f} ({_side_word(close, prior_low)})"
            )
    if today_key is None:
        parts.append("stamps unreadable, so the session split is not trustworthy")
    return ", ".join(parts) + "."


def _session_vwap_last(bars: Sequence[Mapping[str, Any]]) -> float | None:
    """Session VWAP at the last bar, through the one implementation that owns it."""
    try:
        import chart_snapshot

        series = chart_snapshot.session_vwap_series(revive_bars(bars))
    except Exception:  # noqa: BLE001 - a digest must never cost the capture
        _log.debug("Session VWAP unavailable for a journal capture.", exc_info=True)
        return None
    values = [value for value in (series.get("vwap") or []) if value is not None]
    return float(values[-1]) if values else None


def describe_d1(label: str, bars: Sequence[Mapping[str, Any]]) -> str:
    """One line describing a daily series: trend averages and relative volume."""
    if not bars:
        return f"{label} D1: no daily bars were cached, so this chart was not captured."
    last = bars[-1]
    close = _number(last.get("close"))
    parts = [f"{label} D1: {len(bars)} bars, last {_stamp_text(last.get('dt'))[:10]}"]
    if close is not None:
        parts.append(f"last {close:.2f}")
    prior_close = _number(bars[-2].get("close")) if len(bars) > 1 else None
    if close is not None and prior_close:
        parts.append(f"prior close {prior_close:.2f} ({(close / prior_close - 1) * 100:+.2f}%)")
    closes = [_number(bar.get("close")) for bar in bars]
    for period in (20, 50, 200):
        average = _sma_last(closes, period)
        if average is not None:
            parts.append(f"{period}d SMA {average:.2f} ({_side_word(close, average)})")
    volume = _number(last.get("volume"))
    prior_volumes = [
        value
        for value in (_number(bar.get("volume")) for bar in bars[-21:-1])
        if value is not None
    ]
    if volume is not None and len(prior_volumes) >= 5:
        average_volume = sum(prior_volumes) / len(prior_volumes)
        if average_volume > 0:
            parts.append(
                f"volume {volume:,.0f} vs {len(prior_volumes)}d avg "
                f"{average_volume:,.0f} (RVOL {volume / average_volume:.2f})"
            )
    return ", ".join(parts) + "."


def _sma_last(closes: Sequence[float | None], period: int) -> float | None:
    window = [value for value in closes[-period:] if value is not None]
    if len(window) < period:
        # An average over fewer bars than it names is a different number
        # wearing the same label. Absent is the honest answer.
        return None
    return sum(window) / len(window)


def capture_digest(capture: Mapping[str, Any]) -> str:
    """The whole capture in a few lines - what the AI grant carries.

    A pane with no stored bars is left out rather than described as empty: the
    per-series "no bars were cached" sentence exists for a chart that was
    EXPECTED and missing, and printing it for a pane that was never asked for
    (an auto-mode flip captures SPY and nothing else) would read as a failure.
    """
    symbol = str(capture.get("symbol") or "").strip().upper()
    benchmark = str(capture.get("benchmark") or BENCHMARK_SYMBOL).strip().upper()
    series = capture.get("series") or {}
    lines: list[str] = []
    if symbol:
        lines.append(describe_m5(symbol, series.get("symbol_m5") or ()))
        lines.append(describe_d1(symbol, series.get("symbol_d1") or ()))
    if benchmark and benchmark != symbol:
        lines.append(describe_m5(benchmark, series.get("benchmark_m5") or ()))
        lines.append(describe_d1(benchmark, series.get("benchmark_d1") or ()))
    return "\n".join(lines)


# -- building -------------------------------------------------------------
def build_capture(
    *,
    entry_id: str,
    symbol: str = "",
    reason: str = REASON_ENTRY,
    m5_bars: Iterable[Mapping[str, Any]] | None = None,
    d1_bars: Iterable[Mapping[str, Any]] | None = None,
    benchmark_m5: Iterable[Mapping[str, Any]] | None = None,
    benchmark_d1: Iterable[Mapping[str, Any]] | None = None,
    benchmark: str = BENCHMARK_SYMBOL,
    note: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """One capture. Pure: the caller supplies the bars, this shapes them."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    capture = {
        "entry_id": str(entry_id or "").strip(),
        "symbol": str(symbol or "").strip().upper(),
        "benchmark": str(benchmark or BENCHMARK_SYMBOL).strip().upper(),
        "reason": str(reason or REASON_ENTRY),
        "note": str(note or ""),
        "captured_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "series": {
            "symbol_m5": trim_bars(m5_bars, M5_BAR_LIMIT),
            "symbol_d1": trim_bars(d1_bars, D1_BAR_LIMIT),
            "benchmark_m5": trim_bars(benchmark_m5, M5_BAR_LIMIT),
            "benchmark_d1": trim_bars(benchmark_d1, D1_BAR_LIMIT),
        },
    }
    capture["digest"] = capture_digest(capture)
    return capture


def has_any_bars(capture: Mapping[str, Any]) -> bool:
    """Did anything at all get captured? An empty capture is not written."""
    return any(bool(rows) for rows in (capture.get("series") or {}).values())


# -- storage --------------------------------------------------------------
def charts_dir() -> Path:
    from project_paths import RUNTIME_DATA_DIR

    return Path(RUNTIME_DATA_DIR) / DIR_NAME


def _safe_name(entry_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(entry_id))


def capture_path(entry_id: str, *, captured_at: str = "") -> Path:
    """Where a capture's bars live. Month subdirectories, like the ledgers.

    The month comes from the capture stamp when there is one and from the
    entry id otherwise (`mj-YYYY-MM-DD-...`), so a capture and the entry it
    belongs to always land in the same folder.
    """
    stamp = _parse_stamp(captured_at)
    if stamp is not None:
        month = stamp.strftime("%Y%m")
    else:
        parts = str(entry_id or "").split("-")
        month = f"{parts[1]}{parts[2]}" if len(parts) >= 3 and parts[1].isdigit() else "unfiled"
    return charts_dir() / month / f"{_safe_name(entry_id)}.json"


def save_capture(capture: Mapping[str, Any]) -> Path:
    """Write the bars sidecar. Temp file + replace, so a torn write is never read."""
    path = capture_path(
        str(capture.get("entry_id") or ""), captured_at=str(capture.get("captured_at") or "")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(capture, handle, default=str, separators=(",", ":"), sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return path


def load_capture(entry_id: str) -> dict[str, Any] | None:
    """The stored capture for an entry, or None. A missing one is not an error."""
    entry_id = str(entry_id or "").strip()
    if not entry_id:
        return None
    direct = capture_path(entry_id)
    candidates = [direct]
    if not direct.exists():
        try:
            candidates = sorted(charts_dir().glob(f"*/{_safe_name(entry_id)}.json"))
        except OSError:
            candidates = []
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def chart_ledger():
    from evidence_ledger import EvidenceLedger

    return EvidenceLedger(stream=STREAM_CHARTS, schema=SCHEMA_MARKET_JOURNAL_CHART)


def ledger_row(capture: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """The small, AI-readable half of a capture: counts and the digest."""
    series = capture.get("series") or {}
    return {
        "event_type": "chart_capture",
        "entry_id": str(capture.get("entry_id") or ""),
        "symbol": str(capture.get("symbol") or ""),
        "benchmark": str(capture.get("benchmark") or BENCHMARK_SYMBOL),
        "reason": str(capture.get("reason") or ""),
        "note": str(capture.get("note") or ""),
        "captured_at": str(capture.get("captured_at") or ""),
        "bar_counts": {name: len(rows or ()) for name, rows in series.items()},
        "digest": str(capture.get("digest") or ""),
        "bars_file": str(path) if path is not None else "",
    }


def record_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Sidecar first, then the digest row. Returns what happened, never raises.

    Order matters: the ledger row names a file, so the file exists before the
    row that points at it. A failed sidecar means no row, because a digest
    promising bars that are not there is worse than no capture at all.
    """
    if not has_any_bars(capture):
        return {"ok": False, "reason": "no bars were cached, so nothing was captured"}
    try:
        path = save_capture(capture)
    except OSError as exc:
        _log.warning("Journal chart capture not saved: %s", exc)
        return {"ok": False, "reason": str(exc)}
    try:
        row = chart_ledger().append(ledger_row(capture, path=path))
    except Exception as exc:  # noqa: BLE001
        _log.warning("Journal chart digest row not written: %s", exc)
        return {"ok": False, "reason": str(exc), "bars_file": str(path)}
    return {"ok": True, "row": row, "bars_file": str(path)}


def digests_by_entry(*, limit: int = 0) -> dict[str, dict[str, Any]]:
    """The latest digest row per entry id. A missing store is an empty map."""
    try:
        rows = list(chart_ledger().read().rows)
    except Exception:  # noqa: BLE001
        _log.debug("Journal chart digests unreadable.", exc_info=True)
        return {}
    rows.sort(key=lambda row: str(row.get("captured_at") or row.get("event_at") or ""))
    if limit:
        rows = rows[-int(limit) :]
    return {str(row.get("entry_id") or ""): dict(row) for row in rows if row.get("entry_id")}
