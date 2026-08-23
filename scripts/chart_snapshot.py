from __future__ import annotations

"""Pure D1/M5 snapshot-chart data for the symbol quick-look popup.

Turns the data the app already holds (the master-AVWAP daily parquet store
and BounceBot's cached M5 bars) into plot-ready payloads: candle bars plus
aligned overlay series. No Qt, no network, no IB - everything here is local
reads and arithmetic, so the GUI can call it synchronously on click.

Overlay contract: each overlay is {"label", "values", "color", "width",
"dash"} where ``values`` aligns 1:1 with the bar list (None = undefined at
that bar; the chart breaks the line there). ``color`` is a ui.theme role
name resolved by the widget, keeping this module theme-agnostic.

The M5 VWAP/sigma series mirrors ``calc_anchored_vwap_bands`` exactly
(typical price = OHLC/4, sigma accumulated from each bar's deviation from
the RUNNING vwap, volume-weighted) - the running-deviation variant every
band consumer is calibrated to. Do not "fix" it toward a distribution
stdev; see plan.md section 5.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

D1_DEFAULT_SESSIONS = 90
_daily_bars_cache: dict[str, tuple[tuple[str, int], list[dict[str, Any]]]] = {}
# (mtime, {symbol: [iso dates...]}) for the earnings-dates cache file.
_earnings_dates_cache: list = [None, {}]

# Fixed color assignments (user-specified 2026-07-29): the trader reads these
# lines by color first. SMAs dotted, EMAs solid.
D1_OVERLAY_SPECS = (
    ("sma", 50, "SMA50", "chart_light_blue", 1.6, "dot"),
    ("sma", 100, "SMA100", "chart_pink", 1.6, "dot"),
    ("sma", 200, "SMA200", "chart_purple", 1.6, "dot"),
    ("ema", 8, "EMA8", "chart_grey", 1.1, False),
    ("ema", 15, "EMA15", "chart_pink", 1.1, False),
    ("ema", 21, "EMA21", "chart_yellow", 1.1, False),
)

# AVWAPE band colors by σ multiple (upper and lower share the color).
AVWAPE_BAND_COLORS = {1: "chart_blue", 2: "chart_green", 3: "chart_light_blue"}

M5_EMA_SPECS = (
    ("ema", 15, "EMA15", "chart_pink", 1.1, True),
    ("ema", 21, "EMA21", "chart_yellow", 1.1, True),
)


def sma_series(values: Iterable[float], period: int) -> list[float | None]:
    """Simple moving average; None until a full window exists."""
    values = [float(value) for value in values]
    period = max(1, int(period))
    result: list[float | None] = [None] * len(values)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index >= period - 1:
            result[index] = running / period
    return result


def ema_series(values: Iterable[float], span: int) -> list[float | None]:
    """Exponential moving average, pandas ``ewm(span, adjust=False)`` recursion.

    Matches how the rest of the codebase computes EMAs (e.g. the daily-trend
    gate), so the popup's EMA8/15/21 agree with the values the bot acts on.
    """
    values = [float(value) for value in values]
    if not values:
        return []
    span = max(1, int(span))
    alpha = 2.0 / (span + 1.0)
    result: list[float | None] = [values[0]]
    for value in values[1:]:
        previous = result[-1]
        result.append(alpha * value + (1.0 - alpha) * previous)
    return result


def session_vwap_series(bars: list[Mapping[str, Any]]) -> dict[str, list[float | None]]:
    """Per-session running VWAP and +/-1 sigma bands, aligned to ``bars``.

    Accumulation restarts on every date change (session VWAP). Bar math is
    the calc_anchored_vwap_bands running-deviation variant: at each bar the
    plotted value is exactly what that function would return anchored at the
    session open and ended at that bar. Zero-volume bars carry the previous
    values forward so the drawn line has no artificial breaks.
    """
    vwap: list[float | None] = []
    upper: list[float | None] = []
    lower: list[float | None] = []
    cum_vol = cum_vp = cum_sd = 0.0
    current_date = None
    for bar in bars or []:
        stamp = bar.get("dt")
        bar_date = stamp.date() if hasattr(stamp, "date") else None
        if bar_date != current_date:
            current_date = bar_date
            cum_vol = cum_vp = cum_sd = 0.0
        try:
            volume = float(bar.get("volume") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        if volume > 0:
            tp = (
                float(bar["open"]) + float(bar["high"]) + float(bar["low"]) + float(bar["close"])
            ) / 4.0
            cum_vol += volume
            cum_vp += tp * volume
            running = cum_vp / cum_vol
            cum_sd += (tp - running) * (tp - running) * volume
        if cum_vol > 0:
            value = cum_vp / cum_vol
            stdev = (cum_sd / cum_vol) ** 0.5
            vwap.append(value)
            upper.append(value + stdev)
            lower.append(value - stdev)
        else:
            vwap.append(None)
            upper.append(None)
            lower.append(None)
    return {"vwap": vwap, "upper_1": upper, "lower_1": lower}


def anchored_vwap_band_series(
    bars: list[Mapping[str, Any]], anchor_index: int
) -> dict[str, list[float | None]]:
    """Anchored VWAP + ±1/2/3σ bands as aligned per-bar series.

    Exactly ``calc_anchored_vwap_bands``'s running-deviation accumulation
    (typical price = OHLC/4, σ from each bar's deviation vs the RUNNING
    anchored VWAP, volume-weighted, zero-volume bars skipped): the value at
    bar i is what that function returns for a frame ending at i. Bars before
    the anchor are None. Do not swap toward a distribution σ - plan.md sec 5.
    """
    count = len(bars or [])
    series: dict[str, list[float | None]] = {
        key: [None] * count
        for key in (
            "avwap",
            "upper_1",
            "lower_1",
            "upper_2",
            "lower_2",
            "upper_3",
            "lower_3",
        )
    }
    if not 0 <= anchor_index < count:
        return series
    cum_vol = cum_vp = cum_sd = 0.0
    for index in range(anchor_index, count):
        bar = bars[index]
        try:
            volume = float(bar.get("volume") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        if volume > 0:
            tp = (
                float(bar["open"]) + float(bar["high"]) + float(bar["low"]) + float(bar["close"])
            ) / 4.0
            cum_vol += volume
            cum_vp += tp * volume
            running = cum_vp / cum_vol
            cum_sd += (tp - running) * (tp - running) * volume
        if cum_vol > 0:
            value = cum_vp / cum_vol
            stdev = (cum_sd / cum_vol) ** 0.5
            series["avwap"][index] = value
            for k in (1, 2, 3):
                series[f"upper_{k}"][index] = value + k * stdev
                series[f"lower_{k}"][index] = value - k * stdev
    return series


def _earnings_dates_map() -> dict[str, list[str]]:
    """{symbol: [iso earnings dates]} from the cache file, mtime-cached."""
    import json

    from project_paths import EARNINGS_DATES_CACHE_FILE

    path = Path(EARNINGS_DATES_CACHE_FILE)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return {}
    if _earnings_dates_cache[0] == mtime_ns:
        return _earnings_dates_cache[1]
    symbols: dict[str, list[str]] = {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("symbols") or payload.get("data") or {}
        for raw_symbol, entry in raw.items():
            symbol = str(raw_symbol or "").strip().upper()
            dates = entry.get("dates") if isinstance(entry, dict) else entry
            if symbol and isinstance(dates, list):
                symbols[symbol] = [str(value) for value in dates if value]
    except (OSError, ValueError, TypeError, AttributeError):
        symbols = {}
    _earnings_dates_cache[0] = mtime_ns
    _earnings_dates_cache[1] = symbols
    return symbols


def symbol_earnings_dates(symbol: str) -> list[str]:
    """Every known earnings date for a symbol, as ISO strings.

    Read-only accessor over the same mtime-cached map the AVWAPE anchors use,
    so the chart's earnings ribbon costs no extra file read. The cache is
    HISTORY only - it carries no future dates for any symbol - which is why
    the next report has to be projected (``earnings_projection``).
    """
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return []
    dates_map = _earnings_dates_map()
    dates = dates_map.get(symbol)
    if not dates and "." in symbol:
        dates = dates_map.get(symbol.replace(".", "-"))
    return list(dates or [])


def earnings_anchor_dates(
    symbol: str, *, today: date | None = None
) -> tuple[date | None, date | None]:
    """(current, previous) AVWAPE anchors the master system would use.

    Same selection as the scanner (``pick_current/previous_earnings_anchor_
    for_reference_date``): the most recent earnings date, except a very fresh
    one (< RECENT_DAYS) defers to the prior anchor while its own accumulation
    is still thin; "previous" is the anchor before that. (None, None) when
    the earnings cache has nothing for the symbol.
    """
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return (None, None)
    dates_map = _earnings_dates_map()
    dates = dates_map.get(symbol)
    if not dates and "." in symbol:
        dates = dates_map.get(symbol.replace(".", "-"))
    if not dates:
        return (None, None)
    try:
        from master_avwap_lib.legacy import (
            pick_current_earnings_anchor_for_reference_date,
            pick_previous_earnings_anchor_for_reference_date,
        )

        reference = today or date.today()
        return (
            pick_current_earnings_anchor_for_reference_date(dates, reference),
            pick_previous_earnings_anchor_for_reference_date(dates, reference),
        )
    except Exception:
        return (None, None)


def earnings_anchor_date(symbol: str, *, today: date | None = None) -> date | None:
    """The CURRENT AVWAPE anchor alone (watch evaluation uses only this)."""
    return earnings_anchor_dates(symbol, today=today)[0]


def _overlay(label: str, values: list[float | None], color: str, width: float, dash: bool) -> dict:
    return {"label": label, "values": values, "color": color, "width": width, "dash": dash}


def _tail(values: list, count: int) -> list:
    return values[-count:] if count and len(values) > count else values


def _daily_store_candidates(symbol: str) -> list[tuple[str, Path]]:
    """Loader stems/paths in preference order for the durable daily store."""
    from master_avwap_lib.legacy import (
        MASTER_AVWAP_DAILY_BARS_DIR,
        _sanitize_symbol_for_filename,
    )

    symbol = str(symbol or "").strip().upper()
    aliases = [symbol]
    # The universe/store use Yahoo's dashed class-share form (BF-B), while a
    # user-entered watchlist can legitimately contain the exchange form BF.B.
    if "." in symbol:
        aliases.append(symbol.replace(".", "-"))
    candidates = []
    seen = set()
    for alias in aliases:
        stem = _sanitize_symbol_for_filename(alias)
        if stem in seen:
            continue
        seen.add(stem)
        candidates.append(
            (stem, Path(MASTER_AVWAP_DAILY_BARS_DIR) / f"{stem}.parquet")
        )
    return candidates


def _volume_or_zero(value: Any) -> float:
    """A missing volume draws nothing; it never draws a NaN-shaped hole."""
    try:
        volume = float(value)
    except (TypeError, ValueError):
        return 0.0
    return volume if volume == volume and volume > 0 else 0.0


def load_d1_bars(symbol: str) -> list[dict[str, Any]]:
    """Full daily history from the durable parquet store as chart bars.

    Results are cached by the resolved file's mtime. Each click still stats
    the tiny per-symbol file path, but unchanged parquet is not re-read.
    """
    from setup_playbook_study import _load_daily_frame

    symbol = str(symbol or "").strip().upper()
    candidates = _daily_store_candidates(symbol)
    resolved_stem, resolved_path, mtime_ns = "", None, 0
    for stem, path in candidates:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            continue
        resolved_stem, resolved_path = stem, path
        break
    if resolved_path is None and candidates:
        resolved_stem, resolved_path = candidates[0]

    cache_key = (str(resolved_path or ""), mtime_ns)
    cached = _daily_bars_cache.get(symbol)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    frame = _load_daily_frame(resolved_stem) if resolved_stem else None
    if frame is None:
        _daily_bars_cache[symbol] = (cache_key, [])
        return []
    bars: list[dict[str, Any]] = []
    has_volume = "volume" in frame.columns
    for row in frame.itertuples(index=False):
        bars.append(
            {
                "dt": row.datetime.to_pydatetime() if hasattr(row.datetime, "to_pydatetime") else row.datetime,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                # R10.V step 3: the store may hold a bar with NO volume (an
                # IB-sourced row keeps its prices and drops its round-lot
                # number rather than being rescaled). NaN must not reach the
                # paint path, and 0.0 is what this loader already emits for a
                # file with no volume column at all - one "no data" value, not
                # two.
                "volume": _volume_or_zero(getattr(row, "volume", 0.0)) if has_volume else 0.0,
            }
        )
    _daily_bars_cache[symbol] = (cache_key, bars)
    return bars


def latest_completed_session_date(now: datetime | None = None) -> date:
    """The most recent session whose daily bar should exist by ``now``.

    Weekdays before that day's close -> the previous weekday; after the close
    -> today; weekends -> Friday. Holidays are not modelled (no exchange
    calendar in the repo): on a holiday this names a session that never
    traded, which callers must treat as "possibly stale" - never as proof.
    """
    from market_session import get_market_session_window, normalize_market_local_datetime

    moment = normalize_market_local_datetime(now)
    candidate = moment.date()
    try:
        window = get_market_session_window(candidate)
        before_close = moment < window.close_local
    except Exception:
        before_close = moment.hour < 13  # market-local fallback
    if before_close:
        candidate = candidate - timedelta(days=1)
    while candidate.weekday() >= 5:  # Sat/Sun -> back to Friday
        candidate = candidate - timedelta(days=1)
    return candidate


def d1_store_is_stale(
    d1_bars: list[Mapping[str, Any]], now: datetime | None = None
) -> bool:
    """True when the newest STORED bar predates the last completed session.

    The trader's complaint this answers: "sometimes the latest D1 bar isn't
    loaded in." The chart deliberately performs no provider call during paint
    (plan.md Milestone 8), so a symbol outside the current scan set - nothing
    refreshing its durable store, no cached M5 bars to preview from - could
    show a tail days old with nothing ever noticing. This predicate is the
    noticing: hosts use it to trigger an OFF-paint background backfill.
    Preview candles do not count as stored evidence. An EMPTY store is not
    "stale" (that is the out-of-universe case with its own note).
    """
    stored = [bar for bar in d1_bars or [] if not bar.get("preview")]
    if not stored:
        return False
    stamp = (stored[-1] or {}).get("dt")
    last = stamp.date() if hasattr(stamp, "date") else None
    if last is None:
        return True
    return last < latest_completed_session_date(now)


def session_has_opened(now: datetime | None = None) -> bool:
    """True once today's regular session has started (market-local clock).

    Weekends answer False. Holidays are not modelled (no exchange calendar in
    the repo), so a holiday answers True and callers merely look for a
    forming candle that will not arrive - never a claim that one traded.
    """
    from market_session import get_market_session_window, normalize_market_local_datetime

    moment = normalize_market_local_datetime(now)
    if moment.weekday() >= 5:
        return False
    try:
        window = get_market_session_window(moment.date())
        return moment >= window.open_local
    except Exception:
        return moment.hour >= 6  # market-local fallback (06:30 open)


def forming_d1_bar(
    d1_bars: list[Mapping[str, Any]],
    intraday_bars: list[Mapping[str, Any]],
    *,
    session_complete_through: date | None = None,
) -> dict[str, Any] | None:
    """Synthesize today's forming daily candle from cached intraday bars.

    The durable daily store only gains a session's bar after the close, so
    during the day a store-fed D1 chart always ends at the previous session.
    Aggregating the bot's cached M5 bars for their newest session (open =
    first open, high = max, low = min, close = last close, volume = sum)
    yields a preview candle, marked ``"preview": True`` so the chart can draw
    it distinctly. Display-only by design: nothing here feeds a detector, so
    the completed-bars-only invariant (plan.md sec 5) is untouched.

    Returns None when there are no intraday bars, or when the store already
    holds that session AND the session is over. ``session_complete_through``
    (the last session whose daily bar is final) is what makes that second
    test honest: a scan running at 11:00 writes today's PARTIAL daily bar
    into the durable store, and treating that frozen bar as authoritative is
    what made the chart's "latest" candle hours old. During the session the
    live aggregate wins; after the close the real bar does. Omit the argument
    for the historical behavior (the store always wins).
    """
    if not intraday_bars:
        return None
    last_stamp = (intraday_bars[-1] or {}).get("dt")
    session = last_stamp.date() if hasattr(last_stamp, "date") else None
    if session is None:
        return None
    if d1_bars:
        stored_stamp = (d1_bars[-1] or {}).get("dt")
        stored_date = stored_stamp.date() if hasattr(stored_stamp, "date") else None
        if stored_date is not None and stored_date >= session:
            if session_complete_through is None or session <= session_complete_through:
                return None
    session_bars = [
        bar
        for bar in intraday_bars
        if hasattr(bar.get("dt"), "date") and bar["dt"].date() == session
    ]
    if not session_bars:
        return None
    try:
        return {
            "dt": datetime(session.year, session.month, session.day),
            "open": float(session_bars[0]["open"]),
            "high": max(float(bar["high"]) for bar in session_bars),
            "low": min(float(bar["low"]) for bar in session_bars),
            "close": float(session_bars[-1]["close"]),
            "volume": sum(float(bar.get("volume") or 0.0) for bar in session_bars),
            "preview": True,
        }
    except (TypeError, ValueError, KeyError):
        return None


def build_d1_snapshot(
    symbol: str,
    *,
    sessions: int = D1_DEFAULT_SESSIONS,
    loader: Callable[[str], list[dict[str, Any]]] | None = None,
    intraday_bars: list[Mapping[str, Any]] | None = None,
    anchor_resolver: Callable[[str], Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Daily candles + SMA50/100/200 + EMA8/15/21 + AVWAPE ±1/2/3σ bands,
    indicators computed on the full history so the displayed tail carries
    correct long-lookback values.

    The chart ALWAYS ends at the current session while one is running, and
    that last candle is always a preview (trader rule 2026-08-05: "I want to
    always see the latest D1 candle as it's forming intraday"). Two things
    can supply it, freshest first: ``intraday_bars`` (the bot's cached M5
    series, or any today-dated bar a host fetched for this purpose)
    aggregated into a candle, else a partial daily bar the store itself
    picked up from a scan earlier today.

    Everything the indicators touch is COMPLETED sessions only - today's
    partial bar is excluded from the SMA/EMA/AVWAPE math and each overlay
    gets a trailing None, so the lines honestly stop at the last completed
    session instead of hanging off half a day. ``anchor_resolver`` overrides
    the earnings-anchor lookup (tests); it may return either a (current,
    previous) tuple or a bare current anchor date.
    """
    stored = (loader or load_d1_bars)(symbol)
    if not stored:
        return {"symbol": symbol, "timeframe": "D1", "bars": [], "overlays": [], "note": "no daily store"}
    # A scan running mid-session writes today's PARTIAL daily bar to the
    # durable store. It is a preview candle wearing a stored bar's clothes:
    # never let it into the indicator math, and prefer a live aggregate over
    # it for display.
    complete_through = latest_completed_session_date(now)
    # Exactly one bar can be the forming one: today's, and only while today's
    # session is still running. Anything else - including a future-dated bar
    # from a synthetic/repaired store - is left alone as stored history.
    from market_session import normalize_market_local_datetime

    today = normalize_market_local_datetime(now).date()
    forming_session = None if complete_through >= today else today

    def _is_partial(bar: Mapping[str, Any]) -> bool:
        stamp = bar.get("dt")
        return (
            forming_session is not None
            and hasattr(stamp, "date")
            and stamp.date() == forming_session
        )

    bars = [bar for bar in stored if not _is_partial(bar)]
    partial = [bar for bar in stored if _is_partial(bar)]
    if not bars:  # a brand-new symbol whose only bar is today's partial
        bars, partial = stored, []
    closes = [bar["close"] for bar in bars]
    preview = forming_d1_bar(
        bars, intraday_bars or [], session_complete_through=complete_through
    )
    if preview is None and partial:
        preview = dict(partial[-1]) | {"preview": True}
    shown = _tail(bars, sessions)
    if preview is not None:
        shown = shown + [preview]

    def tail_values(series: list[float | None]) -> list[float | None]:
        values = _tail(series, sessions)
        if preview is not None:
            # Indicators stay computed on completed sessions only; the line
            # honestly stops at the last stored bar instead of previewing.
            values = values + [None]
        return values

    overlays = []
    for kind, period, label, color, width, dash in D1_OVERLAY_SPECS:
        series = sma_series(closes, period) if kind == "sma" else ema_series(closes, period)
        overlays.append(_overlay(label, tail_values(series), color, width, dash))

    # AVWAPE + its σ bands, anchored where the master scanner anchors (the
    # current earnings AVWAP), plus the PREVIOUS earnings AVWAP as a single
    # yellow line. Upper/lower of each σ multiple share a label so the legend
    # gains one entry per band pair, not one per line. Anchor dates ride in
    # the snapshot for hosts (legend stamp, watch evaluation).
    resolved = (anchor_resolver or earnings_anchor_dates)(symbol)
    if isinstance(resolved, tuple):
        anchor, prev_anchor = resolved
    else:  # a bare current-anchor resolver (older tests/callers)
        anchor, prev_anchor = resolved, None

    def _anchor_index_for(target: date | None) -> int | None:
        if target is None:
            return None
        for index, bar in enumerate(bars):
            stamp = bar.get("dt")
            if hasattr(stamp, "date") and stamp.date() == target:
                return index
        return None

    anchor_index = _anchor_index_for(anchor)
    prev_index = _anchor_index_for(prev_anchor)
    if anchor_index is not None:
        bands = anchored_vwap_band_series(bars, anchor_index)
        overlays.append(
            _overlay("AVWAPE", tail_values(bands["avwap"]), "chart_white", 1.6, False)
        )
        for k, width in ((1, 1.1), (2, 1.0), (3, 0.9)):
            color = AVWAPE_BAND_COLORS[k]
            for side_key in (f"upper_{k}", f"lower_{k}"):
                overlays.append(
                    _overlay(f"±{k}σ", tail_values(bands[side_key]), color, width, True)
                )
    if prev_index is not None:
        prev_bands = anchored_vwap_band_series(bars, prev_index)
        overlays.append(
            _overlay(
                "AVWAPE prev", tail_values(prev_bands["avwap"]), "chart_yellow", 1.2, False
            )
        )
    return {
        "symbol": symbol,
        "timeframe": "D1",
        "bars": shown,
        "overlays": overlays,
        "note": "",
        "avwape_anchor": anchor.isoformat() if anchor_index is not None else "",
        "avwape_prev_anchor": prev_anchor.isoformat() if prev_index is not None else "",
    }


def build_m5_snapshot(symbol: str, bars: list[Mapping[str, Any]]) -> dict[str, Any]:
    """M5 candles + session VWAP with +/-1 sigma bands + EMA15/21."""
    bars = [dict(bar) for bar in bars or []]
    if not bars:
        return {"symbol": symbol, "timeframe": "M5", "bars": [], "overlays": [], "note": "no cached M5 bars"}
    closes = [bar["close"] for bar in bars]
    vwap = session_vwap_series(bars)
    overlays = [
        _overlay("VWAP", vwap["vwap"], "accent", 1.6, False),
        _overlay("+1σ", vwap["upper_1"], "accent", 1.0, True),
        _overlay("-1σ", vwap["lower_1"], "accent", 1.0, True),
    ]
    for kind, span, label, color, width, dash in M5_EMA_SPECS:
        overlays.append(_overlay(label, ema_series(closes, span), color, width, dash))
    return {"symbol": symbol, "timeframe": "M5", "bars": bars, "overlays": overlays, "note": ""}
