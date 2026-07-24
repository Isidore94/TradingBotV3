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

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping

from chart_snapshot import session_vwap_series

M5_BAR_SPAN = timedelta(minutes=5)

# kind -> button/badge label. Ordered as the buttons appear in the review pane.
WATCH_KINDS = {
    "new_hod": "New HOD",
    "new_lod": "New LOD",
    "vwap_bounce": "VWAP bounce",
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
            )
    return None
