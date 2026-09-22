"""The Pullback alert's rule sheet: `pullback_sma_reclaim_v1` (PCT-1).

The trader's words (2026-09-15), which this module is a transcription of:

    "we then monitor them for a pullback on a M15 or M30 basis. on the M15 we
    use the 150 moving average on the M30 we use the 75. we wait for the stock
    to go BELOW these levels then break back up. we can test entries on the
    breakup if they are accompanied with an LRSI reversal on the same time
    frame in the past 3 bars or so. we can also test on an M30 basis a breakup
    then waiting for an M30 or M15 LRSI reversal while staying above teh
    relevant SMA for an entry. we can also test for retests of the SMA."

Three triggers, one episode clock, one timeframe per call:

* ``sma_reclaim_lrsi`` - a completed close came BACK above the SMA after a
  completed close below it, and the LRSI crossed up through 80 on the reclaim
  bar or one of the two before it. The fire is stamped on the RECLAIM bar.
* ``reclaim_then_lrsi`` (M30 only, which is what the trader asked for) - the
  reclaim already happened, every completed close since has HELD the SMA, and
  the LRSI crosses up through 80 later - on the M30 series itself or on the
  M15 companion series the caller hands in. The fire is stamped on the CROSS
  bar and carries the timeframe of the SMA (``M30``), never the companion's;
  the message names which series crossed.
* ``sma_retest`` - after the reclaim, a completed bar AFTER the reclaim bar
  tags the SMA (its low within ``RETEST_TOLERANCE_ATR`` of the line, or clean
  through it), still CLOSES on the right side of it, and has an LRSI cross up
  through 80 on that bar or either of the two before it.

What this module holds, and why each line is here:

* **An episode is the unit, not a bar.** A new episode starts on every
  completed close back UNDER the SMA; each trigger fires at most once inside
  one episode, and again in the next one. The caller carries
  ``episode_state`` forward between polls; a state whose episode no longer
  matches the tape is simply ignored, so a restart re-reads the tape rather
  than replaying it.
* **80 is a parameter here, never a live level.** The oscillator is the
  champion's ``efficiency_lrsi.compute_efficiency_lrsi``; its
  ``CROSS_LEVELS (20, 50)`` are the M5 engines' and are not touched. A SHORT
  reads the NEGATED closes, which is the idiom
  ``m5_signal_engines.latest_lrsi_cross`` already uses, so "cross up through
  80" means the same thing on both sides.
* **"Ideally it was below 50 2-4 bars previously" is a LABEL, not a gate**
  (lead decision 2026-09-15, from the trader's "ideally"). It rides the fire
  as ``lrsi_from_below_50`` and is counted back from the CROSS bar.
* **Completed bars only** (plan.md sec 5), through the one rule
  ``completed_bars.is_completed_bar``; ``now`` is a parameter and this module
  never reads a clock, so the same bars answer the same way twice.
* **Missing history is NOT MEASURED, never a verdict.** Below the warm-up
  (``sma_length`` plus the oscillator's own) or on a tape whose last completed
  bar is more than ``STALE_AFTER`` behind ``now``, ``evaluate`` returns
  ``None`` and the caller waits.

Pure: plain bar dicts in (``dt, open, high, low, close``), a frozen result
out. No Qt, no network, no file, no clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

RULE_VERSION = "pullback_sma_reclaim_v1"

#: The three trigger names. They are written on every fire, every review row
#: and every feed line, so they are the vocabulary - never re-spelled.
TRIGGER_RECLAIM_LRSI = "sma_reclaim_lrsi"
TRIGGER_RECLAIM_THEN_LRSI = "reclaim_then_lrsi"
TRIGGER_SMA_RETEST = "sma_retest"
TRIGGERS = (TRIGGER_RECLAIM_LRSI, TRIGGER_RECLAIM_THEN_LRSI, TRIGGER_SMA_RETEST)

#: "Cross up through 80" - this rule's own level, passed to the champion's
#: oscillator. `efficiency_lrsi.CROSS_LEVELS` is the live M5 pair and stays
#: `(20, 50)`; nothing here reads or writes it.
LRSI_CROSS_LEVEL = 80.0
#: The quality clause's line and its window, counted back from the CROSS bar.
LRSI_BELOW_LEVEL = 50.0
LRSI_BELOW_LOOKBACK = (2, 3, 4)
#: "on the reclaim bar or the two before it".
CROSS_WINDOW_BARS = 2

#: Warm-up beyond the SMA's own length: the oscillator needs its EMA and its
#: 4-bar sums to mean anything. 150 + 10 = 160 M15 bars, 75 + 10 = 85 M30.
WARMUP_EXTRA_BARS = 10

#: The ATR the retest tolerance is measured in, and the tolerance itself.
ATR_LENGTH = 14
RETEST_TOLERANCE_ATR = 0.25

#: A tape whose last completed bar is older than this cannot answer.
STALE_AFTER = timedelta(hours=24)

#: `reclaim_then_lrsi` is the M30 leg the trader asked for by name.
RECLAIM_THEN_LRSI_MINUTES = 30

REASON_NO_EPISODE = "no_episode"
REASON_BELOW = "below"
REASON_RECLAIMED = "reclaimed"
REASON_RETESTED = "retested"


@dataclass(frozen=True)
class Fire:
    """One trigger, on one bar, with everything it measured."""

    trigger: str
    timeframe: str
    bar_dt: datetime
    sma: float
    close: float
    lrsi: float | None
    lrsi_from_below_50: bool
    atr: float | None
    message: str
    # The source event and the SMA leg are different for the M30/M15
    # companion path.  Keep both so a persisted alert is replayable.
    cross_timeframe: str | None = None
    cross_bar_dt: datetime | None = None
    cross_lrsi: float | None = None
    sma_bar_dt: datetime | None = None


@dataclass(frozen=True)
class EpisodeState:
    """What the caller carries between polls, so one event speaks once.

    ``episode_key`` is the START of the current below-the-SMA run, which is
    the only identity an episode has on the tape itself. A state whose key no
    longer matches what the bars say is a state about a finished episode and
    is discarded rather than trusted - uncertainty never suppresses.
    """

    episode_key: datetime | None = None
    phase: str = REASON_NO_EPISODE
    reclaim_bar_dt: datetime | None = None
    fired: tuple[str, ...] = ()

    def has_fired(self, trigger: str) -> bool:
        return trigger in self.fired


@dataclass(frozen=True)
class PullbackResult:
    rule_version: str
    timeframe: str
    side: str
    sma_length: int
    bars_read: int
    fired: tuple[Fire, ...]
    episode_state: EpisodeState
    reason: str
    sma: float | None = None
    close: float | None = None
    atr: float | None = None
    lrsi: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reading the bars
# ---------------------------------------------------------------------------
def timeframe_label(bar_minutes: int) -> str:
    return f"M{int(bar_minutes)}"


def warmup_bars(sma_length: int) -> int:
    return int(sma_length) + WARMUP_EXTRA_BARS


def _naive(moment: datetime) -> datetime:
    return moment.replace(tzinfo=None) if moment.tzinfo is not None else moment


def _completed(
    bars: Iterable[Mapping[str, Any]] | None,
    bar_minutes: int,
    *,
    now: datetime,
) -> list[dict[str, Any]]:
    """Completed, readable, oldest-first bar dicts. The one rule, once."""
    from completed_bars import bar_time, is_completed_bar

    kept: list[dict[str, Any]] = []
    for bar in bars or ():
        stamp = bar_time(bar)
        if stamp is None:
            continue
        if not is_completed_bar(bar, bar_minutes, now=now):
            continue
        try:
            row = {
                "dt": _naive(stamp),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if any(value != value for value in (row["open"], row["high"], row["low"], row["close"])):
            continue  # NaN is missing data, never a price
        kept.append(row)
    kept.sort(key=lambda row: row["dt"])
    return kept


def _simple_moving_averages(closes: Sequence[float], length: int) -> list[float | None]:
    """The simple mean of the last ``length`` closes, aligned 1:1 with them."""
    span = max(1, int(length))
    out: list[float | None] = []
    running = 0.0
    for index, close in enumerate(closes):
        running += close
        if index >= span:
            running -= closes[index - span]
        out.append(running / float(span) if index >= span - 1 else None)
    return out


def _lrsi_series(closes: Sequence[float], side: str):
    """The champion oscillator, with the short side's negated closes."""
    from indicators.efficiency_lrsi import compute_efficiency_lrsi

    sign = -1.0 if str(side or "").strip().lower() == "short" else 1.0
    return compute_efficiency_lrsi([sign * float(close) for close in closes])


def _is_long(side: str) -> bool:
    return str(side or "").strip().lower() != "short"


# ---------------------------------------------------------------------------
# The episode walk
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Episode:
    start_index: int
    reclaim_index: int | None


def _walk_episode(
    closes: Sequence[float],
    smas: Sequence[float | None],
    *,
    long_side: bool,
) -> _Episode | None:
    """The CURRENT episode: where the tape last went under, and its reclaim.

    One forward pass. A completed close on the wrong side of the SMA opens a
    new episode (and ends any reclaim that was standing); the first completed
    close back on the right side of it is that episode's reclaim.
    """
    start_index: int | None = None
    reclaim_index: int | None = None
    for index, close in enumerate(closes):
        sma = smas[index]
        if sma is None:
            continue
        beyond = close > sma if long_side else close < sma
        if not beyond:
            # Under the line: a fresh episode, unless we are still inside the
            # same unbroken run below it.
            if start_index is None or reclaim_index is not None:
                start_index = index
                reclaim_index = None
        elif start_index is not None and reclaim_index is None:
            reclaim_index = index
    if start_index is None:
        return None
    return _Episode(start_index=start_index, reclaim_index=reclaim_index)


def _crossed_from_below_fifty(values: Sequence[float | None], cross_index: int) -> bool:
    """Was the oscillator under 50 in the 2-4 bars before the CROSS bar?

    The trader's "ideally"; a label on the fire, never a gate (lead decision
    2026-09-15). An unmeasurable bar answers no, because a warm-up value is
    not evidence that the name was churning.
    """
    for back in LRSI_BELOW_LOOKBACK:
        index = cross_index - back
        if index < 0:
            continue
        value = values[index]
        if value is not None and value < LRSI_BELOW_LEVEL:
            return True
    return False


def _post_arm(bar_dt: datetime, bar_minutes: int, armed_at: datetime | None) -> bool:
    """The armed-watch convention: the event bar's END is after the arm.

    The same fence `chart_watch.h1_event_is_post_arm` holds for the H1 leg: a
    move that was over before the trader pressed the button is not this
    watch's event. A missing arm time fences nothing.
    """
    if armed_at is None:
        return True
    return bar_dt + timedelta(minutes=int(bar_minutes)) > _naive(armed_at)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------
def evaluate(
    bars: Iterable[Mapping[str, Any]] | None,
    *,
    side: str = "long",
    sma_length: int,
    bar_minutes: int,
    armed_at: datetime | None,
    now: datetime,
    episode_state: EpisodeState | None = None,
    companion_bars: Iterable[Mapping[str, Any]] | None = None,
    companion_minutes: int | None = None,
) -> PullbackResult | None:
    """The three triggers on one timeframe, or ``None`` when NOT MEASURED.

    ``companion_bars`` / ``companion_minutes`` are the OTHER series the
    trader named for the M30 leg ("waiting for an M30 or M15 LRSI reversal"):
    they are read for `reclaim_then_lrsi` only, they never move the SMA, and
    a fire they produce still carries this call's timeframe.
    """
    moment = _naive(now)
    completed = _completed(bars, bar_minutes, now=moment)
    needed = warmup_bars(sma_length)
    if len(completed) < needed:
        return None
    last = completed[-1]
    if moment - (last["dt"] + timedelta(minutes=int(bar_minutes))) > STALE_AFTER:
        return None

    closes = [row["close"] for row in completed]
    smas = _simple_moving_averages(closes, sma_length)
    lrsi = _lrsi_series(closes, side)
    values = list(lrsi.values)
    crosses = set(lrsi.cross_up_indices(LRSI_CROSS_LEVEL))
    long_side = _is_long(side)
    label = timeframe_label(bar_minutes)

    from indicators.atr import wilder_atr

    atr = wilder_atr(completed, ATR_LENGTH)

    episode = _walk_episode(closes, smas, long_side=long_side)
    state = episode_state if isinstance(episode_state, EpisodeState) else None
    episode_key = completed[episode.start_index]["dt"] if episode is not None else None
    already: tuple[str, ...] = ()
    if state is not None and state.episode_key == episode_key:
        already = tuple(state.fired)

    fired: list[Fire] = []

    def _fire(
        trigger: str,
        index: int,
        *,
        cross_index: int | None,
        message: str,
        cross_timeframe: str | None = None,
        cross_bar_dt: datetime | None = None,
        cross_lrsi: float | None = None,
        cross_from_below: bool | None = None,
        sma_index: int | None = None,
        event_bar_dt: datetime | None = None,
        event_minutes: int | None = None,
    ) -> None:
        if trigger in already or any(hit.trigger == trigger for hit in fired):
            return
        row = completed[index]
        event_dt = event_bar_dt or row["dt"]
        if not _post_arm(event_dt, event_minutes or bar_minutes, armed_at):
            return
        sma_at = smas[index]
        if sma_at is None:
            return
        from_below = (
            _crossed_from_below_fifty(values, cross_index)
            if cross_index is not None
            else False
        )
        if cross_from_below is not None:
            from_below = cross_from_below
        fired.append(
            Fire(
                trigger=trigger,
                timeframe=label,
                bar_dt=event_dt,
                sma=float(sma_at),
                close=float(row["close"]),
                lrsi=values[index] if index < len(values) else None,
                lrsi_from_below_50=bool(from_below),
                atr=atr,
                message=message,
                cross_timeframe=cross_timeframe or label,
                cross_bar_dt=(
                    cross_bar_dt
                    or (completed[cross_index]["dt"] if cross_index is not None else row["dt"])
                ),
                cross_lrsi=(
                    cross_lrsi
                    if cross_lrsi is not None
                    else (values[cross_index] if cross_index is not None else None)
                ),
                sma_bar_dt=completed[sma_index if sma_index is not None else index]["dt"],
            )
        )

    reclaim_index = episode.reclaim_index if episode is not None else None
    reason = REASON_NO_EPISODE if episode is None else REASON_BELOW

    if reclaim_index is not None:
        reason = REASON_RECLAIMED
        side_word = "reclaim" if long_side else "loss"

        # --- sma_reclaim_lrsi -------------------------------------------
        window = {
            reclaim_index - back for back in range(CROSS_WINDOW_BARS + 1)
        }
        in_window = sorted(index for index in crosses if index in window)
        if in_window:
            cross_index = in_window[-1]
            flag = (
                " (from below 50)"
                if _crossed_from_below_fifty(values, cross_index)
                else " (not under 50 first)"
            )
            _fire(
                TRIGGER_RECLAIM_LRSI,
                reclaim_index,
                cross_index=cross_index,
                message=(
                    f"{label} {int(sma_length)}-SMA {side_word} + "
                    f"LRSI {LRSI_CROSS_LEVEL:.0f} cross{flag}"
                ),
            )

        # --- reclaim_then_lrsi (the M30 leg) ----------------------------
        if int(bar_minutes) == RECLAIM_THEN_LRSI_MINUTES:
            # The earliest completed post-arm event wins.  A native M30 cross
            # must not hide an earlier M15 companion cross (or vice versa).
            candidates: list[
                tuple[datetime, datetime, str, int | None, float | None, bool]
            ] = []
            for cross_index in sorted(index for index in crosses if index > reclaim_index):
                stamp = completed[cross_index]["dt"]
                if _post_arm(stamp, bar_minutes, armed_at):
                    candidates.append((
                        stamp + timedelta(minutes=bar_minutes), stamp, label,
                        cross_index, values[cross_index],
                        _crossed_from_below_fifty(values, cross_index),
                    ))
            if companion_bars is not None and companion_minutes:
                companion = _companion_cross(
                    companion_bars,
                    companion_minutes,
                    side=side,
                    now=moment,
                    after=completed[reclaim_index]["dt"]
                    + timedelta(minutes=int(bar_minutes)),
                    armed_at=armed_at,
                )
                if companion is not None:
                    companion_dt, companion_label, companion_from_below, companion_lrsi = companion
                    candidates.append((
                        companion_dt + timedelta(minutes=companion_minutes),
                        companion_dt, companion_label, None, companion_lrsi,
                        companion_from_below,
                    ))
            if candidates:
                _cross_end, cross_dt, cross_label, native_index, cross_value, from_below = min(
                    candidates, key=lambda item: (item[0], item[1], item[2])
                )
                sma_index = native_index if native_index is not None else len(completed) - 1
                _fire(
                    TRIGGER_RECLAIM_THEN_LRSI,
                    native_index if native_index is not None else sma_index,
                    cross_index=native_index,
                    cross_timeframe=cross_label,
                    cross_bar_dt=cross_dt,
                    cross_lrsi=cross_value,
                    cross_from_below=from_below,
                    sma_index=sma_index,
                    event_bar_dt=cross_dt,
                    event_minutes=(bar_minutes if native_index is not None else companion_minutes),
                    message=(
                        f"{label} {int(sma_length)}-SMA held, then an {cross_label} "
                        f"LRSI {LRSI_CROSS_LEVEL:.0f} cross"
                    ),
                )

        # --- sma_retest --------------------------------------------------
        retest_index = _first_retest(
            completed,
            smas,
            after=reclaim_index,
            long_side=long_side,
            atr=atr,
        )
        if retest_index is not None:
            reason = REASON_RETESTED
            window = {
                retest_index - back for back in range(CROSS_WINDOW_BARS + 1)
            }
            in_window = sorted(index for index in crosses if index in window)
            if in_window:
                cross_index = in_window[-1]
                tolerance = ""
                if atr:
                    row = completed[retest_index]
                    sma_at = smas[retest_index] or 0.0
                    extreme = row["low"] if long_side else row["high"]
                    tolerance = f" (low {abs(extreme - sma_at) / atr:.2f} ATR off the line)"
                flag = (
                    " (from below 50)"
                    if _crossed_from_below_fifty(values, cross_index)
                    else " (not under 50 first)"
                )
                _fire(
                    TRIGGER_SMA_RETEST,
                    retest_index,
                    cross_index=cross_index,
                    message=(
                        f"{label} {int(sma_length)}-SMA retest held + "
                        f"LRSI {LRSI_CROSS_LEVEL:.0f} cross{flag}{tolerance}"
                    ),
                )

    new_fired = tuple(already) + tuple(
        hit.trigger for hit in fired if hit.trigger not in already
    )
    return PullbackResult(
        rule_version=RULE_VERSION,
        timeframe=label,
        side="SHORT" if not long_side else "LONG",
        sma_length=int(sma_length),
        bars_read=len(completed),
        fired=tuple(fired),
        episode_state=EpisodeState(
            episode_key=episode_key,
            phase=reason,
            reclaim_bar_dt=(
                completed[reclaim_index]["dt"] if reclaim_index is not None else None
            ),
            fired=new_fired,
        ),
        reason=reason,
        sma=smas[-1],
        close=closes[-1],
        atr=atr,
        lrsi=values[-1] if values else None,
    )


def _first_retest(
    completed: Sequence[Mapping[str, Any]],
    smas: Sequence[float | None],
    *,
    after: int,
    long_side: bool,
    atr: float | None,
) -> int | None:
    """The first bar AFTER the reclaim that tags the SMA and holds it.

    Lead ruling 2026-09-15: the reclaim bar is never its own retest. "Tags"
    is the bar's own extreme coming within ``RETEST_TOLERANCE_ATR`` of the
    line - or going clean through it - and "holds" is the CLOSE still on the
    right side. An unmeasurable ATR cannot answer, so it answers nothing.
    """
    if atr is None or atr <= 0:
        return None
    for index in range(after + 1, len(completed)):
        sma = smas[index]
        if sma is None:
            continue
        row = completed[index]
        close = float(row["close"])
        held = close > sma if long_side else close < sma
        if not held:
            return None  # the episode ended here; this is not a retest
        extreme = float(row["low"]) if long_side else float(row["high"])
        distance = (extreme - sma) if long_side else (sma - extreme)
        if distance <= RETEST_TOLERANCE_ATR * atr:
            return index
    return None


def _companion_cross(
    companion_bars: Iterable[Mapping[str, Any]] | None,
    companion_minutes: int,
    *,
    side: str,
    now: datetime,
    after: datetime,
    armed_at: datetime | None,
) -> tuple[datetime, str, bool, float | None] | None:
    """The first LRSI 80 cross on the OTHER series after the reclaim bar.

    The trader asked for the M30 hold to be answerable by an M15 reversal;
    this reads that series and nothing else from it. The fire it produces
    still belongs to the M30 evaluation, so the caller stamps the timeframe.
    """
    completed = _completed(companion_bars, companion_minutes, now=now)
    if len(completed) <= LRSI_BELOW_LOOKBACK[-1]:
        return None
    closes = [row["close"] for row in completed]
    result = _lrsi_series(closes, side)
    values = list(result.values)
    for index in result.cross_up_indices(LRSI_CROSS_LEVEL):
        stamp = completed[index]["dt"]
        if (
            stamp + timedelta(minutes=int(companion_minutes)) > after
            and _post_arm(stamp, companion_minutes, armed_at)
        ):
            return (
                stamp,
                timeframe_label(companion_minutes),
                _crossed_from_below_fifty(values, index),
                values[index],
            )
    return None
