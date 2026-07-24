from __future__ import annotations

"""User-armed one-shot M5 chart watches for the visual alert review surface.

The trader arms a watch ("New HOD", "New LOD", "VWAP bounce") while looking
at a symbol's M5 chart in the Alert Center's visual review pane. Each watch
is session-scoped and one-shot: the first COMPLETED M5 bar that meets the
condition produces a trigger (the hosting panel turns it into a red Alert
Center alert) and the watch is retired. A forming bar is preview only and
never triggers - plan.md section 5.

Pure module: plain datetimes and bar dicts ({"dt", "open", "high", "low",
"close", "volume"} as returned by ``BounceBot.m5_chart_bars``), no Qt, no
network. VWAP comes from ``chart_snapshot.session_vwap_series`` so the
bounce condition uses the exact running-deviation band math the desk is
calibrated to.
"""

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from chart_snapshot import session_vwap_series

M5_BAR_SPAN = timedelta(minutes=5)

# kind -> button/badge label. Ordered as the buttons appear in the review pane.
WATCH_KINDS = {
    "new_hod": "New HOD",
    "new_lod": "New LOD",
    "vwap_bounce": "VWAP bounce",
    "band_bounce": "σ-band bounce",
}

# The σ-band button mirrors the day-trade tracker's measured M5 winners:
# long = dynamic_vwap_upper_band (ride above +1σ, dip-tag it, reclaim),
# short = dynamic_vwap_lower_band (the mirror below -1σ). The tracker's
# 2026-07-24 read puts the family's prime production in the late-morning and
# afternoon buckets; trigger alerts annotate accordingly.
BAND_BOUNCE_TRACKER_TYPES = {
    "long": "dynamic_vwap_upper_band",
    "short": "dynamic_vwap_lower_band",
}
BAND_BOUNCE_PRIME_BUCKETS = ("late_morning", "afternoon")

# Persistent D1 candle-level alerts (armed by clicking a D1 candle). Not in
# WATCH_KINDS: they are level breaks, not session-scoped chart watches.
D1_LEVEL_KINDS = {
    "d1_level_above": "D1 break above",
    "d1_level_below": "D1 break below",
}


@dataclass(frozen=True)
class ChartWatch:
    symbol: str
    kind: str
    armed_at: datetime
    side: str = "WATCH"
    baseline: float | None = None
    source_text: str = ""


@dataclass(frozen=True)
class ChartWatchTrigger:
    watch: ChartWatch
    price: float
    bar_dt: datetime
    message: str
    # The direction the trigger actually fired as ("long"/"short") for the
    # bounce kinds a WATCH-side watch can hit either way; "" when the watch's
    # own side already says it.
    resolved_side: str = ""


def _naive(moment: datetime) -> datetime:
    # IB serves this desk's bars on the local clock (sometimes tz-stamped);
    # arm times come from the same clock, so comparisons drop tzinfo rather
    # than convert across zones.
    return moment.replace(tzinfo=None) if moment.tzinfo is not None else moment


def _session_bars(bars: Iterable[Mapping[str, Any]] | None, moment: datetime) -> list[dict[str, Any]]:
    session = _naive(moment).date()
    kept = []
    for bar in bars or []:
        stamp = bar.get("dt")
        if isinstance(stamp, datetime) and _naive(stamp).date() == session:
            kept.append(dict(bar))
    kept.sort(key=lambda bar: _naive(bar["dt"]))
    return kept


def _bar_end(bar: Mapping[str, Any]) -> datetime:
    return _naive(bar["dt"]) + M5_BAR_SPAN


def arm_chart_watch(
    kind: str,
    symbol: str,
    side: str,
    bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
    source_text: str = "",
) -> ChartWatch:
    """Arm a watch against what the trader sees on the chart right now.

    The HOD/LOD baseline is today's extreme across ALL cached bars including
    a forming one - that is exactly the day high/low drawn on the chart at
    the moment the button is clicked. Triggering later still requires a
    completed bar.
    """
    if kind not in WATCH_KINDS:
        raise ValueError(f"unknown chart watch kind: {kind!r}")
    moment = _naive(now or datetime.now())
    session = _session_bars(bars, moment)
    baseline: float | None = None
    if kind == "new_hod" and session:
        baseline = max(float(bar["high"]) for bar in session)
    elif kind == "new_lod" and session:
        baseline = min(float(bar["low"]) for bar in session)
    return ChartWatch(
        symbol=str(symbol or "").strip().upper(),
        kind=kind,
        armed_at=moment,
        side=side if side in ("LONG", "SHORT") else "WATCH",
        baseline=baseline,
        source_text=str(source_text or ""),
    )


def watch_is_stale(watch: ChartWatch, *, now: datetime | None = None) -> bool:
    """A watch never survives into the next session."""
    moment = _naive(now or datetime.now())
    return _naive(watch.armed_at).date() != moment.date()


def evaluate_chart_watch(
    watch: ChartWatch,
    bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
) -> ChartWatchTrigger | None:
    """First completed post-arm bar meeting the condition, or None."""
    moment = _naive(now or datetime.now())
    completed = [
        bar for bar in _session_bars(bars, moment) if _bar_end(bar) <= moment
    ]
    if not completed:
        return None
    if watch.kind in ("new_hod", "new_lod"):
        return _evaluate_extreme(watch, completed)
    if watch.kind == "vwap_bounce":
        return _evaluate_vwap_bounce(watch, completed)
    if watch.kind == "band_bounce":
        return _evaluate_band_bounce(watch, completed)
    return None


def _evaluate_extreme(
    watch: ChartWatch, completed: list[dict[str, Any]]
) -> ChartWatchTrigger | None:
    is_high = watch.kind == "new_hod"
    armed_at = _naive(watch.armed_at)
    baseline = watch.baseline
    for bar in completed:
        value = float(bar["high"] if is_high else bar["low"])
        if _bar_end(bar) <= armed_at:
            # Pre-arm bar: it can only tighten the reference level (covers a
            # watch armed before the bot had cached this symbol's bars).
            if baseline is None:
                baseline = value
            else:
                baseline = max(baseline, value) if is_high else min(baseline, value)
            continue
        if baseline is None:
            # No reference yet: the first tracked bar defines the day's
            # extreme instead of trivially "breaking" nothing.
            baseline = value
            continue
        if (is_high and value > baseline) or (not is_high and value < baseline):
            stamp = _naive(bar["dt"])
            if is_high:
                message = (
                    f"New HOD {value:.2f} > armed day high {baseline:.2f} "
                    f"(bar {stamp:%H:%M})"
                )
            else:
                message = (
                    f"New LOD {value:.2f} < armed day low {baseline:.2f} "
                    f"(bar {stamp:%H:%M})"
                )
            return ChartWatchTrigger(watch=watch, price=value, bar_dt=stamp, message=message)
    return None


def _evaluate_vwap_bounce(
    watch: ChartWatch, completed: list[dict[str, Any]]
) -> ChartWatchTrigger | None:
    """Touch-and-reclaim off session VWAP on a completed bar.

    Long: the bar trades down to VWAP (low <= vwap) and closes back above.
    Short: the bar trades up to VWAP (high >= vwap) and closes back below.
    A WATCH-side watch accepts either direction.
    """
    armed_at = _naive(watch.armed_at)
    vwap_values = session_vwap_series(completed)["vwap"]
    want_long = watch.side in ("LONG", "WATCH")
    want_short = watch.side in ("SHORT", "WATCH")
    for index, bar in enumerate(completed):
        if _bar_end(bar) <= armed_at:
            continue
        vwap = vwap_values[index]
        if vwap is None:
            continue
        low = float(bar["low"])
        high = float(bar["high"])
        close = float(bar["close"])
        stamp = _naive(bar["dt"])
        if want_long and low <= vwap and close > vwap:
            return ChartWatchTrigger(
                watch=watch,
                price=close,
                bar_dt=stamp,
                message=(
                    f"VWAP bounce (long): tagged VWAP {vwap:.2f}, closed back "
                    f"above at {close:.2f} (bar {stamp:%H:%M})"
                ),
                resolved_side="long",
            )
        if want_short and high >= vwap and close < vwap:
            return ChartWatchTrigger(
                watch=watch,
                price=close,
                bar_dt=stamp,
                message=(
                    f"VWAP bounce (short): tagged VWAP {vwap:.2f}, closed back "
                    f"below at {close:.2f} (bar {stamp:%H:%M})"
                ),
                resolved_side="short",
            )
    return None


def _evaluate_band_bounce(
    watch: ChartWatch, completed: list[dict[str, Any]]
) -> ChartWatchTrigger | None:
    """Touch-and-reclaim off the session VWAP ±1σ band on a completed bar.

    Long: the bar tags the UPPER band from above (low <= +1σ) and closes back
    over it - the tracker's dynamic_vwap_upper_band continuation long.
    Short: the bar tags the LOWER band from below (high >= -1σ) and closes
    back under it. A WATCH-side watch accepts either direction.
    """
    armed_at = _naive(watch.armed_at)
    series = session_vwap_series(completed)
    upper_values = series["upper_1"]
    lower_values = series["lower_1"]
    want_long = watch.side in ("LONG", "WATCH")
    want_short = watch.side in ("SHORT", "WATCH")
    for index, bar in enumerate(completed):
        if _bar_end(bar) <= armed_at:
            continue
        upper = upper_values[index]
        lower = lower_values[index]
        low = float(bar["low"])
        high = float(bar["high"])
        close = float(bar["close"])
        stamp = _naive(bar["dt"])
        if want_long and upper is not None and low <= upper and close > upper:
            return ChartWatchTrigger(
                watch=watch,
                price=close,
                bar_dt=stamp,
                message=(
                    f"σ-band bounce (long): tagged +1σ {upper:.2f}, closed back "
                    f"above at {close:.2f} (bar {stamp:%H:%M})"
                ),
                resolved_side="long",
            )
        if want_short and lower is not None and high >= lower and close < lower:
            return ChartWatchTrigger(
                watch=watch,
                price=close,
                bar_dt=stamp,
                message=(
                    f"σ-band bounce (short): tagged -1σ {lower:.2f}, closed back "
                    f"below at {close:.2f} (bar {stamp:%H:%M})"
                ),
                resolved_side="short",
            )
    return None


# ---------------------------------------------------------------------------
# Persistence: intraday watches are trading-day scoped (a GUI restart keeps
# them armed; a new session drops them), mirroring alert_review_state.py.
# ---------------------------------------------------------------------------
def _market_date_text(value: date | str | None) -> str:
    return value.isoformat() if isinstance(value, date) else str(value or date.today().isoformat())


def _atomic_write_json(payload: dict, path: Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".tmp")
    try:
        staged.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, target)
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass


def chart_watch_to_dict(watch: ChartWatch) -> dict:
    return {
        "symbol": watch.symbol,
        "kind": watch.kind,
        "armed_at": _naive(watch.armed_at).isoformat(),
        "side": watch.side,
        "baseline": watch.baseline,
        "source_text": watch.source_text,
    }


def chart_watch_from_dict(payload: Mapping[str, Any]) -> ChartWatch | None:
    try:
        armed_at = datetime.fromisoformat(str(payload["armed_at"]))
        kind = str(payload["kind"])
        symbol = str(payload["symbol"] or "").strip().upper()
    except (KeyError, TypeError, ValueError):
        return None
    if not symbol or kind not in WATCH_KINDS:
        return None
    baseline = payload.get("baseline")
    try:
        baseline = float(baseline) if baseline is not None else None
    except (TypeError, ValueError):
        baseline = None
    side = str(payload.get("side") or "WATCH")
    return ChartWatch(
        symbol=symbol,
        kind=kind,
        armed_at=armed_at,
        side=side if side in ("LONG", "SHORT") else "WATCH",
        baseline=baseline,
        source_text=str(payload.get("source_text") or ""),
    )


def save_chart_watches(
    watches: Iterable[ChartWatch],
    path: Path,
    *,
    market_date: date | str | None = None,
) -> None:
    _atomic_write_json(
        {
            "market_date": _market_date_text(market_date),
            "watches": [chart_watch_to_dict(watch) for watch in watches],
        },
        path,
    )


def load_chart_watches(
    path: Path,
    *,
    market_date: date | str | None = None,
) -> list[ChartWatch]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError:
        return []
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    if str(payload.get("market_date") or "") != _market_date_text(market_date):
        return []  # armed watches never survive into a new session
    watches = []
    for item in payload.get("watches") or []:
        if isinstance(item, Mapping):
            watch = chart_watch_from_dict(item)
            if watch is not None:
                watches.append(watch)
    return watches


# ---------------------------------------------------------------------------
# Persistent D1 candle-level alerts: armed from a clicked D1 candle, kept
# ACROSS sessions until the level flags. The symbol need not be in any scan -
# evaluation uses whatever evidence exists (cached M5 bars while scanned, the
# durable daily store otherwise) and simply waits when there is none.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class D1LevelWatch:
    symbol: str
    direction: str  # "above" | "below"
    level: float
    armed_at: datetime
    candle_date: str = ""  # ISO date of the clicked candle

    @property
    def kind(self) -> str:
        return "d1_level_above" if self.direction == "above" else "d1_level_below"


def d1_level_watch_to_dict(watch: D1LevelWatch) -> dict:
    return {
        "symbol": watch.symbol,
        "direction": watch.direction,
        "level": watch.level,
        "armed_at": _naive(watch.armed_at).isoformat(),
        "candle_date": watch.candle_date,
    }


def d1_level_watch_from_dict(payload: Mapping[str, Any]) -> D1LevelWatch | None:
    try:
        symbol = str(payload["symbol"] or "").strip().upper()
        direction = str(payload["direction"])
        level = float(payload["level"])
        armed_at = datetime.fromisoformat(str(payload["armed_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not symbol or direction not in ("above", "below") or not level > 0:
        return None
    return D1LevelWatch(
        symbol=symbol,
        direction=direction,
        level=level,
        armed_at=armed_at,
        candle_date=str(payload.get("candle_date") or ""),
    )


def save_d1_level_watches(watches: Iterable[D1LevelWatch], path: Path) -> None:
    _atomic_write_json(
        {"watches": [d1_level_watch_to_dict(watch) for watch in watches]},
        path,
    )


def load_d1_level_watches(path: Path) -> list[D1LevelWatch]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError:
        return []
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    watches = []
    for item in payload.get("watches") or []:
        if isinstance(item, Mapping):
            watch = d1_level_watch_from_dict(item)
            if watch is not None:
                watches.append(watch)
    return watches


def evaluate_d1_level_watch(
    watch: D1LevelWatch,
    m5_bars: Iterable[Mapping[str, Any]] | None,
    d1_bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
) -> ChartWatchTrigger | None:
    """First post-arm evidence bar crossing the level, or None.

    Evidence: completed M5 bars ending after the arm (covers the armed day
    while the symbol is scanned), and COMPLETED daily bars from sessions
    strictly after the arm date (covers unscanned symbols; the armed day's
    own daily bar is excluded because it also contains pre-arm prices).
    """
    moment = _naive(now or datetime.now())
    armed_at = _naive(watch.armed_at)
    is_above = watch.direction == "above"
    level = float(watch.level)

    def _hit(value: float) -> bool:
        return value >= level if is_above else value <= level

    for bar in _session_bars(m5_bars, moment):
        if _bar_end(bar) <= armed_at or _bar_end(bar) > moment:
            continue
        value = float(bar["high"] if is_above else bar["low"])
        if _hit(value):
            stamp = _naive(bar["dt"])
            word = "above" if is_above else "below"
            return ChartWatchTrigger(
                watch=watch,  # type: ignore[arg-type] (duck-typed carrier)
                price=value,
                bar_dt=stamp,
                message=(
                    f"D1 level break {word} {level:.2f}: reached {value:.2f} "
                    f"(M5 bar {stamp:%m/%d %H:%M})"
                ),
                resolved_side="long" if is_above else "short",
            )

    for bar in d1_bars or []:
        stamp = bar.get("dt")
        if not isinstance(stamp, datetime):
            continue
        bar_date = _naive(stamp).date()
        # Completed sessions only, strictly after the arm date.
        if bar_date <= armed_at.date() or bar_date >= moment.date():
            continue
        value = float(bar["high"] if is_above else bar["low"])
        if _hit(value):
            word = "above" if is_above else "below"
            return ChartWatchTrigger(
                watch=watch,  # type: ignore[arg-type]
                price=value,
                bar_dt=_naive(stamp),
                message=(
                    f"D1 level break {word} {level:.2f}: reached {value:.2f} "
                    f"(D1 bar {bar_date:%m/%d})"
                ),
                resolved_side="long" if is_above else "short",
            )
    return None
