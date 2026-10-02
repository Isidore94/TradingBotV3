"""AVWAPE quick test: a swing setup for the Setup Tracker, testing only. Pure, no I/O.

The trader, 2026-10-02: "a stock tests the LOWER_1 stdev level for longs (invert for shorts,
so UPPER_1) then hammers through AVWAPE ... a quick test, not a prolonged period on the
LOWER_1 level". Their answers: the test is a wick touch of LOWER_1 with at most ONE close
under it; the reclaim close above AVWAPE comes within 3 sessions of the touch; the bands are
the EARNINGS AVWAP (`long_setups.earnings_anchor_index` + `long_setups.avwap_bands`); rows
show in the Setup Tracker only, both sides, graded like the Long leaders.

No points, no alert, no Focus, no phone line, no promotion, no Setups-table chip. Nothing in
a detector, a score or `review_policy.json` reads it.

Completed daily bars only, oldest first, as ``{date, open, high, low, close, volume}``.
Point in time: nothing after the last bar is read. Missing data is no setup, never a guess.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from long_setups import (
    AVWAPE_SESSIONS,  # noqa: F401 - the anchor window, re-exported for readers of this rule
    ENTRY_ATR_BELOW,
    GRADE_SESSIONS,
    GRADE_SPY_UP_MIN_PCT,  # noqa: F401 - the grading window, read by `setup_grades`
    _atr,
    _clean_bars,
    _num,
    _text,
    earnings_anchor_index,
    meets_liquidity_floor,
)
from research_warehouse.retest_entry import limit_fill

# --- the rule's numbers, in ONE place (trader, 2026-10-02). Recalibrate here and nowhere else.
SETUP = "avwape_quick_test"
LABEL = "AVWAPE quick test"
#: The touch is within the last this-many completed bars, the trigger bar included.
TEST_WINDOW_SESSIONS = 3
#: The "quick" check looks back this many completed bars, the trigger bar included.
LOOKBACK_SESSIONS = 10
#: At most this many closes beyond the band in the lookback = quick, not prolonged.
MAX_CLOSES_BEYOND_BAND = 1
STATUS_TESTING = "testing"
SIDES = ("LONG", "SHORT")
#: The exit: the long lab's time stop (the grade reads the return at this session's close).
HOLD_SESSIONS = GRADE_SESSIONS


def band_series(bars: Sequence[Mapping[str, Any]], start: int) -> list[tuple[float, float] | None]:
    """Per bar k: `long_setups.avwap_bands(bars[:k+1], start)` in one pass; None before ``start``
    or before the first bar with volume."""
    out: list[tuple[float, float] | None] = [None] * len(bars)
    if not 0 <= start < len(bars):
        return out
    cum_volume = cum_vp = cum_sd = 0.0
    for index in range(start, len(bars)):
        bar = bars[index]
        volume = _num(bar.get("volume"))
        if volume is not None and volume > 0:
            price = (bar["open"] + bar["high"] + bar["low"] + bar["close"]) / 4.0
            cum_volume += volume
            cum_vp += price * volume
            deviation = price - cum_vp / cum_volume
            cum_sd += deviation * deviation * volume
        if cum_volume > 0:
            out[index] = (cum_vp / cum_volume, (cum_sd / cum_volume) ** 0.5)
    return out


def detect(bars: Any, *, side: str, earnings_dates: Iterable[Any] | None, atr: Any = None) -> dict[str, Any] | None:
    """The quick-test row for one name and side on its completed bars, or None."""
    long = side == "LONG"
    if side not in SIDES:
        return None
    bars = _clean_bars(bars)
    if not bars:
        return None
    last = len(bars) - 1
    anchor = earnings_anchor_index(bars, earnings_dates=earnings_dates)
    if anchor is None or last - anchor < TEST_WINDOW_SESSIONS:
        return None
    series = band_series(bars, anchor)
    first = max(last - LOOKBACK_SESSIONS + 1, anchor)
    if any(series[k] is None for k in range(first, last + 1)):
        return None
    avwap = [None if value is None else value[0] for value in series]
    sign = -1.0 if long else 1.0
    band = [None if value is None else value[0] + sign * value[1] for value in series]
    window = range(last - TEST_WINDOW_SESSIONS + 1, last + 1)
    test = next((k for k in window if (bars[k]["low"] <= band[k] if long else bars[k]["high"] >= band[k])), None)
    if test is None:
        return None
    beyond = sum(1 for j in range(first, last + 1)
                 if (bars[j]["close"] < band[j] if long else bars[j]["close"] > band[j]))
    if beyond > MAX_CLOSES_BEYOND_BAND:
        return None
    close = bars[last]["close"]
    if not (close > avwap[last] if long else close < avwap[last]):
        return None
    # T is the FIRST reclaim after the touch, so a name fires once.
    if any((bars[j]["close"] > avwap[j] if long else bars[j]["close"] < avwap[j]) for j in range(test, last)):
        return None
    sigma = series[last][1]
    if not sigma > 0:
        return None
    atr_value = _atr(bars, atr)
    if atr_value is None:
        return None
    level, word, over = ("LOWER_1", "under", "above") if long else ("UPPER_1", "over", "below")
    extreme = bars[test]["low"] if long else bars[test]["high"]
    stop = (min(bars[k]["low"] for k in window) if long else max(bars[k]["high"] for k in window))
    entry = close - ENTRY_ATR_BELOW * atr_value if long else close + ENTRY_ATR_BELOW * atr_value
    closes_text = f"{beyond} close{'' if beyond == 1 else 's'} {word}"
    return {
        "setup": SETUP,
        "side": side,
        "as_of": bars[last]["date"],
        "close": round(close, 4),
        "avwape": round(avwap[last], 4),
        "sigma": round(sigma, 4),
        "band": round(band[last], 4),
        "avwape_z": round((close - avwap[last]) / sigma, 4),
        "test_date": bars[test]["date"],
        "test_extreme": round(extreme, 4),
        "closes_beyond": beyond,
        "sessions_since_test": last - test,
        "atr": round(atr_value, 4),
        "entry_limit": round(entry, 4),
        "stop": round(stop, 4),
        "exit": f"hold up to {HOLD_SESSIONS} sessions, stop {word} {stop:.2f}",
        "status": STATUS_TESTING,
        "reasons": [f"tested {level} on {bars[test]['date']} ({closes_text})", f"closed back {over} AVWAPE"],
    }


def build_rows(
    *,
    bars_by_symbol: Mapping[str, Any],
    spy_bars: Any = None,
    earnings_dates_by_symbol: Mapping[str, Iterable[Any]] | None = None,
    atr_by_symbol: Mapping[str, Any] | None = None,
    market_cap_by_symbol: Mapping[str, Any] | None = None,
    feature_rows: Iterable[Mapping[str, Any]] = (),
    as_of: Any = None,
) -> dict[str, Any]:
    """Every quick-test row of one scan, both sides: ``{as_of, rows}`` sorted by (side, symbol).

    A name whose last bar is not ``as_of`` is stale and skipped; SPY is skipped; a name under the
    trader's liquidity floor (cap from ``market_cap_by_symbol``, else the scan row's
    ``perm_market_cap_m``) gives no row. No market gate, no promotion, no ranking.
    """
    del spy_bars  # the rule reads no SPY; grading does (`settle`)
    as_of_text = _text(as_of)[:10]
    if not as_of_text:
        return {"as_of": "", "rows": []}
    caps_from_rows: dict[str, float] = {}
    for row in feature_rows or ():
        symbol = _text(row.get("symbol")).upper()
        cap = _num(row.get("perm_market_cap_m"))
        if symbol and cap is not None and symbol not in caps_from_rows:
            caps_from_rows[symbol] = cap
    out = []
    for raw_symbol, raw in (bars_by_symbol or {}).items():
        symbol = _text(raw_symbol).upper()
        bars = _clean_bars(raw)
        if not symbol or symbol == "SPY" or not bars or bars[-1]["date"] != as_of_text:
            continue
        cap = _num((market_cap_by_symbol or {}).get(symbol))
        if cap is None:
            cap = caps_from_rows.get(symbol)
        if not meets_liquidity_floor(bars, cap):
            continue
        dates = (earnings_dates_by_symbol or {}).get(symbol)
        atr = (atr_by_symbol or {}).get(symbol)
        for side in SIDES:
            row = detect(bars, side=side, earnings_dates=dates, atr=atr)
            if row is not None:
                out.append({"symbol": symbol, **row})
    return {"as_of": as_of_text, "rows": sorted(out, key=lambda row: (row["side"], row["symbol"]))}


# --- grading (the scan settles; the Setup Tracker grades)

def settle(history: Iterable[Mapping[str, Any]], bars_by_symbol: Mapping[str, Any],
           spy_bars: Any) -> list[dict[str, Any]]:
    """`long_setups.settle`, side-aware: the limit rests through session 1 (a gap fills at the
    open); no touch is ``no_fill``. ``return_pct`` is positive in the trade's favour (a short
    gains when the close is under the fill); ``spy_return_pct`` is SPY's raw return."""
    spy_closes = {bar["date"]: bar["close"] for bar in (_clean_bars(spy_bars) or [])}
    cleaned: dict[str, list[dict[str, Any]] | None] = {}
    out = []
    for raw in history or ():
        row = dict(raw)
        out.append(row)
        if _text(row.get("outcome")):
            continue
        symbol = _text(row.get("symbol")).upper()
        side = _text(row.get("side")).upper()
        if side not in SIDES:
            continue
        if symbol not in cleaned:
            cleaned[symbol] = _clean_bars((bars_by_symbol or {}).get(symbol))
        bars = cleaned[symbol] or []
        index = next((i for i, bar in enumerate(bars) if bar["date"] == _text(row.get("as_of"))[:10]), None)
        entry = _num(row.get("entry_limit"))
        if index is None or entry is None or index + GRADE_SESSIONS >= len(bars):
            continue
        target = bars[index + GRADE_SESSIONS]
        row["target_session"] = target["date"]
        fill = limit_fill(bars[index + 1], entry, side == "LONG")
        if fill is None:
            row["outcome"] = "no_fill"
            continue
        spy_start, spy_end = spy_closes.get(bars[index]["date"]), spy_closes.get(target["date"])
        row["outcome"] = "filled"
        row["fill"] = round(fill, 4)
        change = target["close"] / fill - 1.0 if side == "LONG" else fill / target["close"] - 1.0
        row["return_pct"] = round(change * 100.0, 4)
        row["spy_return_pct"] = (round((spy_end / spy_start - 1.0) * 100.0, 4)
                                 if spy_start and spy_end else None)
    return out


HISTORY_KEYS = ("symbol", "as_of", "side", "setup", "close", "avwape", "sigma", "test_date",
                "closes_beyond", "atr", "entry_limit", "stop")


def upsert_history(history: Iterable[Mapping[str, Any]], rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The history plus this scan's NEW rows: the first write of (session, symbol, side) wins."""
    def key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return _text(row.get("as_of")), _text(row.get("symbol")), _text(row.get("side"))

    kept = [dict(row) for row in history or ()]
    seen = {key(row) for row in kept}
    added = []
    for row in rows or ():
        if key(row) not in seen:
            seen.add(key(row))
            added.append({name: row.get(name) for name in HISTORY_KEYS})
    return kept + added


# --- the words on the Setup Tracker

HEAD = f"{LABEL} (testing, both sides)"


def row_line(row: Mapping[str, Any]) -> str:
    """``NVDA LONG | buy limit 120.10 | hold up to 10 sessions, stop under 115.20 | tested ...``."""
    long = row.get("side") == "LONG"
    level, word, verb = ("LOWER_1", "under", "reclaimed") if long else ("UPPER_1", "over", "lost")
    beyond = int(_num(row.get("closes_beyond")) or 0)
    exit_text = row.get("exit") or f"stop {word} {_num(row.get('stop')) or 0:.2f}"
    return (f"{row.get('symbol')} {row.get('side')} | {'buy' if long else 'sell'} limit "
            f"{_num(row.get('entry_limit')) or 0:.2f} | {exit_text} | tested {level} {row.get('test_date')} "
            f"({beyond} close{'' if beyond == 1 else 's'} {word}), {verb} AVWAPE")


def tracker_lines(payload: Mapping[str, Any] | None, *, limit: int = 12) -> list[str]:
    """The Setup Tracker's AVWAPE quick test section: a head line, then one line per row."""
    if not payload:
        return [f"{HEAD}: no scan yet."]
    rows = list(payload.get("rows") or ())
    longs = sum(1 for row in rows if row.get("side") == "LONG")
    head = f"{HEAD}: {longs} long, {len(rows) - longs} short"
    if not rows:
        return [head, "none this scan"]
    lines = [head, *(row_line(row) for row in rows[:limit])]
    if len(rows) > limit:
        lines.append(f"(+{len(rows) - limit} more)")
    return lines
