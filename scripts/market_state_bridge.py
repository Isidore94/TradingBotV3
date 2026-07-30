"""Shadow bridge: run the new SPY market-state engine beside the legacy pause
detector (plan.md sec 16 / Phase 5.13 champion-challenger).

The legacy `_detect_spy_pause_start()` stays the champion - nothing here may
change live behavior. Each evaluation converts the bot's cached SPY 5-minute
bars, runs the pure MarketStateEngine, and appends a JSONL shadow record ONLY
when the engine state changes or its agreement with the legacy detector
flips. The log gives the promotion evidence the plan requires before the
engine replaces the one-red-candle pause rule.

Coverage accounting (plan.md sec 7.2 "stale/incomplete-bar counters"): an
evaluation only counts as *usable* when the engine's last snapshot was driven
by a bar that actually completed. A still-forming last bar, or a bar that
arrived after a data gap, is counted under its own distinct counter and can
never advance ``last_complete_bar_at``. The two staleness causes are tracked
separately here even though the engine gates on their union - this module adds
observability only and changes no engine decision.

Episode evidence (plan.md sec 7.2 "explicit episode IDs; impulse start/high/low,
counter-move start, depth, stabilization, resumption, and failure timestamps"):
the engine already builds a typed ``PullbackEpisode`` per counter-move. Those
episodes are emitted as their own JSONL rows (``EPISODE_SCHEMA``) beside the
state rows, so an ARMED -> ACTIVE -> STABILIZING -> RESUMED chain can be
grouped by ``episode_id`` and replayed.

EVALUATION CADENCE - READ THIS BEFORE DRAWING ANY CONCLUSION FROM THE LOG
------------------------------------------------------------------------
This hook fires once per **bounce-scan cycle**, NOT once per completed 5-minute
bar. The champion (``check_regime_pause_setups``) decides when it runs, so:

- ABSENCE OF AN EPISODE IN THIS LOG IS NOT EVIDENCE THAT NO EPISODE HAPPENED
  (plan.md sec 4: "Add evaluation-coverage records before using absence of an
  event as evidence"). If the scan loop was not running, if the champion
  returned early because the SPY bar cache was empty, or if the process was
  restarted and the cache lost older bars, real episodes are simply never
  observed and leave no trace at all.
- What the cadence does NOT cost is *discovery within a live session*: every
  evaluation replays the whole day's cached bars through a fresh engine, so an
  episode that opened and closed between two scan cycles is still found and
  logged - late. ``evaluation_lag_seconds`` (evaluation time minus last
  completed bar) and ``discovery_lag_seconds`` (evaluation time minus the
  episode's last event) measure exactly how late, per row.
- Recall must therefore be judged against ``evaluations`` /
  ``usable_evaluations`` in the status file, never against the row count alone.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from diagnostics.artifact_io import append_jsonl_rows, atomic_write_json
from market_state import (
    ENGINE_VERSION,
    M5Bar,
    MarketStateConfig,
    MarketStateEngine,
    MarketState,
    MarketStateSnapshot,
    PullbackEpisode,
)
from market_session import get_market_local_timezone, normalize_market_local_datetime

_BAR_MINUTES = 5
SHADOW_SCHEMA = "spy_state_shadow_v4"
# Episode rows live in the SAME log as the state rows, under their own schema.
# Justification for rows over an episode block on the state row:
#   * the state row is deduped on (state, legacy_paused, engine_paused), so an
#     embedded block would be suppressed exactly when the episode advanced
#     without a state change (a deepening counter-move) - the evidence would be
#     silently lossy;
#   * an episode has its own identity, lifecycle and dedupe key, so it wants its
#     own row cadence, and repeating the whole block on every state row would
#     bloat the log with duplicates;
#   * one file keeps the two streams in a single time order under a single
#     rotation/retention policy. Readers MUST branch on "schema" - a row that is
#     not SHADOW_SCHEMA has no "state" field.
EPISODE_SCHEMA = "spy_episode_shadow_v1"
# v3 and v4 only ADD fields to v2 (staleness cause, completed-bar stamp,
# episode pointers), so an existing v2/v3 log stays readable and must NOT be
# rotated out from under the evidence base; readers treat the added fields as
# optional and skip rows whose schema they do not know.
COMPATIBLE_SHADOW_SCHEMAS = frozenset(
    {"spy_state_shadow_v2", "spy_state_shadow_v3", SHADOW_SCHEMA, EPISODE_SCHEMA}
)
STATUS_SCHEMA = "spy_state_shadow_status_v3"

# Stamped on every row: this hook runs per bounce-scan cycle, not per completed
# bar (see the module docstring). Anything reasoning about recall must read it.
EVALUATION_CADENCE = "per_bounce_scan_cycle"
CADENCE_NOTE = (
    "The shadow hook fires once per bounce-scan cycle, not once per completed "
    "5-minute bar. Absence of an episode in this log is NOT evidence that no "
    "episode occurred (plan.md sec 4): a cycle that never ran, an empty SPY bar "
    "cache, or a process restart leaves no record at all. Each evaluation does "
    "replay the whole cached session, so within a live session an episode is "
    "still discovered late rather than lost; evaluation_lag_seconds and "
    "discovery_lag_seconds measure that lateness."
)

# States that mean "an impulse leg is in force" (the counter-move states are
# part of the same leg). Entering this family from outside it is exactly what
# MarketStateEngine._try_enter_impulse does, so the boundary marks the start of
# a new impulse leg.
_TREND_FAMILY = frozenset(
    {
        MarketState.BULL_IMPULSE,
        MarketState.BEAR_IMPULSE,
        MarketState.COUNTERMOVE_ARMED,
        MarketState.COUNTERMOVE_ACTIVE,
        MarketState.STABILIZING,
        MarketState.TREND_RESUMED,
    }
)

# Staleness causes, reported separately so an audit can tell "the 5-minute bar
# had not closed yet" apart from "the feed skipped bars".
STALE_INCOMPLETE_BAR = "incomplete_bar"
STALE_BAR_GAP = "bar_gap"

# Engine states the legacy detector would call "paused".
_PAUSE_LIKE_STATES = {
    MarketState.COUNTERMOVE_ARMED,
    MarketState.COUNTERMOVE_ACTIVE,
    MarketState.STABILIZING,
}

_lock = threading.Lock()
# Dedupe fingerprint of the last row written, SCOPED to a session + config: a
# process that lives across midnight (or a config change) must never suppress
# the first observation of the new session just because it happens to look like
# yesterday's last one.
_last_written: dict[str, str] = {}
# episode_uid -> progress fingerprint, under the same (session, config) scope.
_episodes_written: dict[str, str] = {}
_episode_scope = ""
_coverage: dict = {}


@dataclass(frozen=True)
class EpisodeRecord:
    """One :class:`PullbackEpisode` plus the leg context the engine computes
    but does not store on the episode itself.

    Everything here is observed from completed bars only - the engine refuses
    forming and gapped bars before any episode field can move (plan.md sec 5).
    """

    episode: PullbackEpisode
    impulse_start_ts: datetime | None
    impulse_start_price: float | None
    impulse_high: float | None
    impulse_low: float | None
    countermove_extreme: float | None

    @property
    def episode_id(self) -> str:
        return self.episode.episode_id

    @property
    def outcome(self) -> str:
        return self.episode.outcome

    @property
    def is_open(self) -> bool:
        return not self.episode.outcome

    @property
    def depth_price(self) -> float | None:
        """Counter-move depth in price terms, positive when adverse."""
        if self.countermove_extreme is None:
            return None
        return self.episode.side_sign * (self.episode.impulse_extreme - self.countermove_extreme)

    @property
    def max_depth_atr(self) -> float:
        return max((event.depth_atr for event in self.episode.events), default=0.0)

    def first_ts(self, state: MarketState) -> datetime | None:
        return next((e.ts for e in self.episode.events if e.state == state), None)

    def last_event_ts(self) -> datetime | None:
        return self.episode.events[-1].ts if self.episode.events else self.episode.armed_ts

    def progress_fingerprint(self) -> str:
        """Changes whenever the episode gained an event, an outcome or depth."""
        return "|".join(
            [
                str(len(self.episode.events)),
                self.episode.outcome or "OPEN",
                _stamp(self.last_event_ts()),
                f"{self.countermove_extreme:.4f}" if self.countermove_extreme is not None else "",
            ]
        )


class _EpisodeObserver:
    """Watches the engine bar by bar and records what the episodes omit.

    Strictly read-only: it calls no engine method, only reads public attributes
    and the two read-only accessors, so it cannot influence a transition. It
    exists because two facts plan.md sec 7.2 requires are computed by the engine
    but never stored on ``PullbackEpisode``:

    * the impulse leg's start (timestamp + price) and its intrabar high/low;
    * the counter-move extreme price, which the engine clears the moment the
      episode resolves.

    Re-deriving the counter-move extreme from the bars afterwards would drift
    from the engine: a failure raised out of STABILIZING is checked *before*
    ``_track_countermove_extreme`` runs, so that bar's adverse edge is not part
    of the engine's extreme even though it is part of the bar range. Sampling
    the engine's own value each bar cannot drift.
    """

    def __init__(self) -> None:
        self._extras: dict[str, dict] = {}
        self._prev_state: MarketState | None = None
        self._leg_start_ts: datetime | None = None
        self._leg_start_price: float | None = None
        self._leg_high: float | None = None
        self._leg_low: float | None = None

    def observe(self, engine: MarketStateEngine, bar: M5Bar, snapshot: MarketStateSnapshot) -> None:
        if snapshot.stale:
            # The engine rejected this bar outright: nothing happened, so
            # nothing may be attributed to it.
            return
        state = snapshot.state
        in_family = state in _TREND_FAMILY
        if not in_family:
            self._leg_start_ts = None
            self._leg_start_price = None
            self._leg_high = None
            self._leg_low = None
        elif self._prev_state not in _TREND_FAMILY or self._leg_start_ts is None:
            # Crossing into the trend family only happens via _try_enter_impulse.
            self._leg_start_ts = bar.ts
            self._leg_start_price = engine.impulse_start_price
            self._leg_high = bar.high
            self._leg_low = bar.low
        else:
            self._leg_high = max(self._leg_high, bar.high)
            self._leg_low = min(self._leg_low, bar.low)
        self._prev_state = state

        for episode in engine.episodes:
            if episode.episode_id not in self._extras:
                # Registered on the arming bar, so the leg context is the leg
                # that this counter-move interrupted.
                self._extras[episode.episode_id] = {
                    "impulse_start_ts": self._leg_start_ts,
                    "impulse_start_price": self._leg_start_price,
                    "impulse_high": self._leg_high,
                    "impulse_low": self._leg_low,
                    "countermove_extreme": None,
                }

        extreme = engine.countermove_extreme
        if extreme is not None and engine.episodes:
            episode = engine.episodes[-1]
            extras = self._extras[episode.episode_id]
            prior = extras["countermove_extreme"]
            if prior is None or episode.side_sign * (prior - extreme) > 0:
                extras["countermove_extreme"] = float(extreme)

    def records(self, engine: MarketStateEngine) -> tuple[EpisodeRecord, ...]:
        out = []
        for episode in engine.episodes:
            extras = self._extras.get(episode.episode_id, {})
            out.append(
                EpisodeRecord(
                    episode=episode,
                    impulse_start_ts=extras.get("impulse_start_ts"),
                    impulse_start_price=extras.get("impulse_start_price"),
                    impulse_high=extras.get("impulse_high"),
                    impulse_low=extras.get("impulse_low"),
                    countermove_extreme=extras.get("countermove_extreme"),
                )
            )
        return tuple(out)


@dataclass(frozen=True)
class ShadowEvaluation:
    """One engine pass plus the coverage facts the audit needs.

    ``snapshot`` is exactly what :class:`MarketStateEngine` returned - this
    wrapper adds no decision. It only separates the two reasons the engine
    refused to transition (``MarketStateSnapshot.stale`` merges them) and
    records the last bar the engine actually consumed, which is by definition a
    completed one.
    """

    snapshot: MarketStateSnapshot | None
    bars_converted: int = 0
    bars_consumed: int = 0
    incomplete_bar: bool = False
    gap_stale: bool = False
    last_complete_bar_ts: datetime | None = None
    # Every episode the engine built from this replay, oldest first. Episodes
    # are derived from completed bars only, so they stay valid even when the
    # evaluation itself is stale (a forming last bar).
    episodes: tuple[EpisodeRecord, ...] = ()

    @property
    def active_episode(self) -> EpisodeRecord | None:
        if self.episodes and self.episodes[-1].is_open:
            return self.episodes[-1]
        return None

    @property
    def stale(self) -> bool:
        return bool(self.snapshot is not None and self.snapshot.stale)

    @property
    def is_usable(self) -> bool:
        """True only when a genuinely completed, non-gapped bar drove the state."""
        return self.snapshot is not None and not self.snapshot.stale

    @property
    def stale_reason(self) -> str:
        reasons = []
        if self.incomplete_bar:
            reasons.append(STALE_INCOMPLETE_BAR)
        if self.gap_stale:
            reasons.append(STALE_BAR_GAP)
        return "+".join(reasons)


def _stamp(moment: datetime | None) -> str:
    return moment.isoformat(timespec="seconds") if moment is not None else ""


def _round(value: float | None, digits: int) -> float | None:
    return round(float(value), digits) if value is not None else None


def _lag_seconds(evaluated_at: datetime, moment: datetime | None) -> float | None:
    """Seconds between a bar/event and the evaluation that observed it.

    This is the honest measure of the per-scan-cycle cadence: it is how late the
    shadow saw something, not how late the engine reacted.
    """
    if moment is None:
        return None
    try:
        return round((evaluated_at - moment).total_seconds(), 1)
    except TypeError:  # naive/aware mismatch - never worth losing the row over
        return None


def shadow_log_path() -> Path:
    try:
        from project_paths import get_diagnostics_dir

        return get_diagnostics_dir() / "spy_state_shadow.jsonl"
    except Exception:
        return Path.home() / ".tradingbotv3" / "spy_state_shadow.jsonl"


def shadow_status_path() -> Path:
    return shadow_log_path().with_name("spy_state_shadow_status.json")


def _config_hash(config: MarketStateConfig) -> str:
    payload = {
        name: str(value) if isinstance(value, timedelta) else value
        for name, value in vars(config).items()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empty_coverage(session_date: str, config_hash: str) -> dict:
    return {
        "schema": STATUS_SCHEMA,
        "engine_version": ENGINE_VERSION,
        "config_hash": config_hash,
        "machine": socket.gethostname(),
        "session_date": session_date,
        # Cadence is part of the evidence, not a footnote (plan.md sec 4).
        "evaluation_cadence": EVALUATION_CADENCE,
        "cadence_note": CADENCE_NOTE,
        "evaluations": 0,
        # Evaluations whose snapshot came from a bar that really completed.
        "usable_evaluations": 0,
        # Union of the two causes below (an evaluation can hit both at once).
        "stale_evaluations": 0,
        "incomplete_bar_evaluations": 0,
        "gap_stale_evaluations": 0,
        "skipped_missing_input": 0,
        "rows_written": 0,
        "episode_rows_written": 0,
        # Daily episode summary. `episodes_today` and `episode_outcomes` are a
        # RECOMPUTATION, not a running total: every evaluation replays the whole
        # cached session, so the newest evaluation's episode list is the day's
        # list. They therefore step DOWN if the bar cache is trimmed.
        "episodes_today": 0,
        "episode_outcomes": {"RESUMED": 0, "FAILED": 0, "ABORTED": 0, "OPEN": 0},
        "last_episode_id": "",
        "errors": 0,
        "last_evaluation_at": "",
        # Only ever a completed bar's timestamp (plan.md sec 5).
        "last_complete_bar_at": "",
        # The bar the last snapshot was built on, complete or not.
        "last_snapshot_bar_at": "",
        "last_stale_reason": "",
        "last_error": "",
        "last_error_at": "",
    }


def _load_coverage(session_date: str, config_hash: str) -> dict:
    """Merge the on-disk status for this session over a fresh template.

    Tolerant of older status rows: unknown keys survive, missing counters start
    at zero, and the schema/identity stamps are always rewritten to the current
    values so a mid-session upgrade cannot leave a stale schema label behind.
    """
    path = shadow_status_path()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError, ValueError):
        loaded = {}
    coverage = _empty_coverage(session_date, config_hash)
    if (
        isinstance(loaded, dict)
        and loaded.get("session_date") == session_date
        and loaded.get("config_hash") == config_hash
    ):
        coverage.update(loaded)
        coverage["schema"] = STATUS_SCHEMA
        coverage["engine_version"] = ENGINE_VERSION
    return coverage


def _record_coverage(
    *,
    evaluated_at: datetime,
    config_hash: str,
    evaluation: "ShadowEvaluation | None" = None,
    row_written: bool = False,
    episode_rows_written: int = 0,
    missing_input: bool = False,
    error: Exception | None = None,
) -> None:
    global _coverage
    session_date = evaluated_at.date().isoformat()
    with _lock:
        if _coverage.get("session_date") != session_date or _coverage.get("config_hash") != config_hash:
            _coverage = _load_coverage(session_date, config_hash)
        _coverage["evaluations"] = int(_coverage.get("evaluations", 0)) + 1
        _coverage["last_evaluation_at"] = evaluated_at.isoformat(timespec="seconds")
        snapshot = evaluation.snapshot if evaluation is not None else None
        if snapshot is not None:
            # A forming or gapped bar is NOT a usable evaluation and must never
            # advance the completed-bar stamp (plan.md sec 5 / sec 7.2).
            if evaluation.is_usable:
                _coverage["usable_evaluations"] = int(_coverage.get("usable_evaluations", 0)) + 1
            else:
                _coverage["stale_evaluations"] = int(_coverage.get("stale_evaluations", 0)) + 1
            if evaluation.incomplete_bar:
                _coverage["incomplete_bar_evaluations"] = (
                    int(_coverage.get("incomplete_bar_evaluations", 0)) + 1
                )
            if evaluation.gap_stale:
                _coverage["gap_stale_evaluations"] = int(_coverage.get("gap_stale_evaluations", 0)) + 1
            _coverage["last_stale_reason"] = evaluation.stale_reason
            _coverage["last_snapshot_bar_at"] = _stamp(snapshot.ts)
            completed_at = _stamp(evaluation.last_complete_bar_ts)
            if completed_at:
                _coverage["last_complete_bar_at"] = completed_at
        if evaluation is not None and snapshot is not None:
            outcomes = {"RESUMED": 0, "FAILED": 0, "ABORTED": 0, "OPEN": 0}
            for record in evaluation.episodes:
                outcomes[record.outcome or "OPEN"] = outcomes.get(record.outcome or "OPEN", 0) + 1
            _coverage["episodes_today"] = len(evaluation.episodes)
            _coverage["episode_outcomes"] = outcomes
            if evaluation.episodes:
                _coverage["last_episode_id"] = evaluation.episodes[-1].episode_id
        if episode_rows_written:
            _coverage["episode_rows_written"] = (
                int(_coverage.get("episode_rows_written", 0)) + int(episode_rows_written)
            )
        if missing_input:
            _coverage["skipped_missing_input"] = int(_coverage.get("skipped_missing_input", 0)) + 1
        if row_written:
            _coverage["rows_written"] = int(_coverage.get("rows_written", 0)) + 1
        if error is not None:
            _coverage["errors"] = int(_coverage.get("errors", 0)) + 1
            _coverage["last_error"] = str(error)[:500]
            _coverage["last_error_at"] = evaluated_at.isoformat(timespec="seconds")
        payload = dict(_coverage)
        try:
            atomic_write_json(shadow_status_path(), payload, fsync=False)
        except Exception:
            # Never propagate: _record_coverage also runs on the error path of a
            # champion-invoked hook.
            logging.warning("SPY shadow coverage status write failed.", exc_info=True)


def _rotate_legacy_shadow_if_needed(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        first = next(
            (line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
            "",
        )
        payload = json.loads(first) if first else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    if payload.get("schema") in COMPATIBLE_SHADOW_SCHEMAS:
        return
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    archive = path.with_name(f"{path.stem}.legacy-{stamp}{path.suffix}")
    counter = 1
    while archive.exists():
        archive = path.with_name(f"{path.stem}.legacy-{stamp}-{counter}{path.suffix}")
        counter += 1
    os.replace(path, archive)
    logging.info("Archived legacy SPY shadow evidence to %s", archive)


def m5_bars_from_bot_bars(bot_bars, *, now: datetime | None = None) -> list[M5Bar]:
    """Convert the bot's cached SPY bars; the last bar is marked incomplete
    while it can still be forming so the engine never acts on a partial bar."""
    local_timezone, _ = get_market_local_timezone()
    moment = normalize_market_local_datetime(now, local_timezone=local_timezone)
    bars: list[M5Bar] = []
    total = len(bot_bars)
    for index, bar in enumerate(bot_bars):
        raw_start = getattr(bar, "dt", None)
        if raw_start is None:
            continue
        start = normalize_market_local_datetime(raw_start, local_timezone=local_timezone)
        is_last = index == total - 1
        complete = True
        if is_last:
            complete = (moment - start) >= timedelta(minutes=_BAR_MINUTES)
        bars.append(
            M5Bar(
                ts=start + timedelta(minutes=_BAR_MINUTES),
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(getattr(bar, "volume", 0.0) or 0.0),
                complete=complete,
            )
        )
    return bars


def evaluate_spy_shadow(
    bot_bars,
    prev_close,
    *,
    now: datetime | None = None,
    config: MarketStateConfig | None = None,
) -> ShadowEvaluation:
    """Fresh engine pass plus the coverage facts, without changing any decision.

    The engine merges "the last bar has not closed yet" and "the feed skipped
    bars" into one ``stale`` flag, and it keeps gating on that union. Here the
    two are measured independently:

    - ``incomplete_bar`` - the last converted bar was still forming;
    - ``gap_stale`` - the last converted bar sits more than ``max_bar_gap``
      after the last bar the engine consumed (measured against ``engine.bars``,
      which is exactly what ``MarketStateEngine._is_stale`` compares against, so
      the flag mirrors the engine instead of guessing).

    ``last_complete_bar_ts`` is the last bar the engine actually consumed - a
    completed, non-gapped bar by construction. It never points at a forming bar,
    which is what makes the ``last_complete_bar_at`` coverage stamp truthful.
    """
    if not bot_bars or not prev_close:
        return ShadowEvaluation(None)
    bars = m5_bars_from_bot_bars(bot_bars, now=now)
    if not bars:
        return ShadowEvaluation(None)
    active_config = config or MarketStateConfig()
    engine = MarketStateEngine(float(prev_close), config=active_config)
    observer = _EpisodeObserver()
    snapshot = None
    for bar in bars:
        snapshot = engine.on_bar(bar)
        observer.observe(engine, bar, snapshot)
    last = bars[-1]
    consumed = engine.bars
    gap_stale = bool(consumed) and (last.ts - consumed[-1].ts) > active_config.max_bar_gap
    return ShadowEvaluation(
        snapshot=snapshot,
        bars_converted=len(bars),
        bars_consumed=len(consumed),
        incomplete_bar=not last.complete,
        gap_stale=gap_stale,
        last_complete_bar_ts=consumed[-1].ts if consumed else None,
        episodes=observer.records(engine),
    )


def evaluate_spy_shadow_state(
    bot_bars,
    prev_close,
    *,
    now: datetime | None = None,
    config: MarketStateConfig | None = None,
) -> MarketStateSnapshot | None:
    """Fresh engine pass over today's cached SPY bars; None when unusable."""
    return evaluate_spy_shadow(bot_bars, prev_close, now=now, config=config).snapshot


def _episode_row(
    record: EpisodeRecord,
    *,
    evaluated_at: datetime,
    session_date: str,
    timezone_name: str,
    config_hash: str,
    evaluation: ShadowEvaluation,
    prev_close: float,
    input_bar_count: int,
) -> dict:
    """One replayable episode record (plan.md sec 7.2 required fields).

    The full event chain is carried on the row, so the newest row for an
    ``episode_id`` is a complete, self-contained account of the episode: no
    reader has to stitch earlier rows together to replay it.
    """
    episode = record.episode
    last_event = record.last_event_ts()
    return {
        "schema": EPISODE_SCHEMA,
        "ts": _stamp(evaluated_at),
        "evaluated_at": _stamp(evaluated_at),
        "session_date": session_date,
        "timezone": timezone_name,
        "machine": socket.gethostname(),
        "engine_version": episode.engine_version or ENGINE_VERSION,
        "config_hash": config_hash,
        "evaluation_cadence": EVALUATION_CADENCE,
        # --- identity -----------------------------------------------------
        "episode_id": episode.episode_id,
        "episode_uid": f"SPY|{session_date}|{episode.episode_id}",
        "symbol": "SPY",
        "side_sign": episode.side_sign,
        "direction": episode.direction,
        # --- impulse leg that the counter-move interrupted ----------------
        "impulse_start_ts": _stamp(record.impulse_start_ts),
        "impulse_start_price": _round(record.impulse_start_price, 4),
        # Intrabar range of the leg through the arming bar - not the same thing
        # as impulse_extreme, which is the engine's tracked favorable edge.
        "impulse_high": _round(record.impulse_high, 4),
        "impulse_low": _round(record.impulse_low, 4),
        "impulse_extreme": _round(episode.impulse_extreme, 4),
        # --- counter-move --------------------------------------------------
        "countermove_start_ts": _stamp(episode.armed_ts),
        "countermove_extreme": _round(record.countermove_extreme, 4),
        "depth_price": _round(record.depth_price, 4),
        "max_depth_atr": _round(record.max_depth_atr, 3),
        # --- lifecycle timestamps ------------------------------------------
        "armed_ts": _stamp(episode.armed_ts),
        "active_ts": _stamp(record.first_ts(MarketState.COUNTERMOVE_ACTIVE)),
        "stabilizing_ts": _stamp(record.first_ts(MarketState.STABILIZING)),
        "resumed_ts": _stamp(record.first_ts(MarketState.TREND_RESUMED)),
        "failed_ts": _stamp(record.first_ts(MarketState.REGIME_FAILED)),
        "aborted_ts": _stamp(last_event) if episode.outcome == "ABORTED" else "",
        "outcome": episode.outcome,
        "open": record.is_open,
        # --- replay ---------------------------------------------------------
        "events": [
            {
                "state": event.state.value,
                "ts": _stamp(event.ts),
                "price": _round(event.price, 4),
                "depth_atr": _round(event.depth_atr, 3),
            }
            for event in episode.events
        ],
        "event_count": len(episode.events),
        "last_event_ts": _stamp(last_event),
        # --- freshness / cadence -------------------------------------------
        # Episodes only ever move on bars the engine accepted, so this row is
        # completed-bar evidence even when the evaluation itself was stale.
        "derived_from_completed_bars": True,
        "complete_bar_ts": _stamp(evaluation.last_complete_bar_ts),
        "evaluation_lag_seconds": _lag_seconds(evaluated_at, evaluation.last_complete_bar_ts),
        # How late this episode's newest transition was observed - the direct
        # cost of the per-scan-cycle cadence.
        "discovery_lag_seconds": _lag_seconds(evaluated_at, last_event),
        "bars_consumed": evaluation.bars_consumed,
        "input_bar_count": input_bar_count,
        "prior_close": float(prev_close),
    }


def record_spy_shadow(
    bot_bars,
    prev_close,
    *,
    legacy_pause_start=None,
    side: str = "",
    now: datetime | None = None,
    config: MarketStateConfig | None = None,
) -> dict | None:
    """Champion/challenger observation; appends to the shadow log only on an
    engine state change or an agreement flip. Never raises."""
    moment = normalize_market_local_datetime(now)
    _, timezone_name = get_market_local_timezone()
    active_config = config or MarketStateConfig()
    config_hash = _config_hash(active_config)
    session_date = moment.date().isoformat()
    try:
        evaluation = evaluate_spy_shadow(
            bot_bars,
            prev_close,
            now=moment,
            config=active_config,
        )
        snapshot = evaluation.snapshot
        if snapshot is None:
            _record_coverage(
                evaluated_at=moment,
                config_hash=config_hash,
                missing_input=True,
            )
            return None
        engine_paused = snapshot.state in _PAUSE_LIKE_STATES
        legacy_paused = legacy_pause_start is not None
        row = {
            "schema": SHADOW_SCHEMA,
            "ts": moment.isoformat(timespec="seconds"),
            "evaluated_at": moment.isoformat(timespec="seconds"),
            "bar_ts": _stamp(snapshot.ts),
            "session_date": session_date,
            "timezone": timezone_name,
            "machine": socket.gethostname(),
            "engine_version": ENGINE_VERSION,
            "config_hash": config_hash,
            "evaluation_cadence": EVALUATION_CADENCE,
            "observation_id": (
                f"SPY|{session_date}|{snapshot.state.value}|"
                f"{_stamp(snapshot.ts) or 'none'}"
            ),
            "state": snapshot.state.value,
            "side_sign": snapshot.side_sign,
            "trend_score": round(snapshot.trend_score, 3),
            "day_return_pct": round(snapshot.day_return_pct, 4),
            "vwap": round(snapshot.vwap, 4) if snapshot.vwap is not None else None,
            "m5_atr": round(snapshot.m5_atr, 4) if snapshot.m5_atr is not None else None,
            "depth_atr": round(snapshot.countermove_depth_atr, 3),
            # `stale` is kept for v2 readers; the fields below split it into the
            # two causes the engine merges (plan.md sec 7.2).
            "stale": snapshot.stale,
            "stale_reason": evaluation.stale_reason,
            "incomplete_bar": evaluation.incomplete_bar,
            "gap_stale": evaluation.gap_stale,
            "usable": evaluation.is_usable,
            # The last bar the engine really consumed: never a forming bar.
            "complete_bar_ts": _stamp(evaluation.last_complete_bar_ts),
            "evaluation_lag_seconds": _lag_seconds(moment, evaluation.last_complete_bar_ts),
            "bars_consumed": evaluation.bars_consumed,
            # Join keys into the EPISODE_SCHEMA rows in this same log.
            "episode_count": len(evaluation.episodes),
            "active_episode_id": (
                evaluation.active_episode.episode_id if evaluation.active_episode else ""
            ),
            "last_episode_id": (
                evaluation.episodes[-1].episode_id if evaluation.episodes else ""
            ),
            "legacy_side": str(side or ""),
            "legacy_pause_start": (
                normalize_market_local_datetime(legacy_pause_start).isoformat(timespec="seconds")
                if hasattr(legacy_pause_start, "isoformat")
                else str(legacy_pause_start or "")
            ),
            "legacy_paused": legacy_paused,
            "engine_paused": engine_paused,
            "agree": engine_paused == legacy_paused,
            "input_bar_count": len(bot_bars or []),
            "prior_close": float(prev_close),
        }
        # Dedupe is scoped to (session, config): a process that survives a date
        # rollover must still write the new session's first observation, even
        # when it looks identical to yesterday's last one.
        scope = f"{session_date}|{config_hash}"
        fingerprint = f"{row['state']}|{row['legacy_paused']}|{row['engine_paused']}"
        global _episode_scope
        with _lock:
            duplicate = (
                _last_written.get("scope") == scope
                and _last_written.get("fingerprint") == fingerprint
            )
            if _episode_scope != scope:
                # A new session (or a config change) starts a fresh episode
                # ledger; an episode from another scope can never match.
                _episodes_written.clear()
                _episode_scope = scope
            episode_prints = dict(_episodes_written)
        # Episode rows have their own dedupe: an episode that gained an event,
        # deepened, or resolved must be re-emitted even when the state row is a
        # duplicate, and an unchanged episode must not be re-emitted on every
        # scan cycle (each evaluation replays the whole day).
        pending_episodes: list[tuple[str, str, dict]] = []
        for record in evaluation.episodes:
            uid = f"SPY|{session_date}|{record.episode_id}"
            progress = record.progress_fingerprint()
            if episode_prints.get(uid) == progress:
                continue
            pending_episodes.append(
                (
                    uid,
                    progress,
                    _episode_row(
                        record,
                        evaluated_at=moment,
                        session_date=session_date,
                        timezone_name=timezone_name,
                        config_hash=config_hash,
                        evaluation=evaluation,
                        prev_close=float(prev_close),
                        input_bar_count=len(bot_bars or []),
                    ),
                )
            )
        # State row first, so the log keeps its "state row then its episodes"
        # ordering and rows[0] of a fresh log is still a state row.
        pending = ([] if duplicate else [row]) + [item[2] for item in pending_episodes]
        if not pending:
            _record_coverage(
                evaluated_at=moment,
                config_hash=config_hash,
                evaluation=evaluation,
            )
            return row  # nothing new to persist
        path = shadow_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_legacy_shadow_if_needed(path)
        append_jsonl_rows(path, pending)
        # Committed only after the rows are really on disk: a failed append must
        # not silently suppress the retry.
        with _lock:
            if not duplicate:
                _last_written["scope"] = scope
                _last_written["fingerprint"] = fingerprint
            if _episode_scope == scope:
                for uid, progress, _ in pending_episodes:
                    _episodes_written[uid] = progress
        _record_coverage(
            evaluated_at=moment,
            config_hash=config_hash,
            evaluation=evaluation,
            row_written=not duplicate,
            episode_rows_written=len(pending_episodes),
        )
        return row
    except Exception as exc:
        _record_coverage(
            evaluated_at=moment,
            config_hash=config_hash,
            error=exc,
        )
        logging.warning("SPY shadow-state recording failed (live behavior unaffected).", exc_info=True)
        return None


def reset_shadow_dedupe() -> None:
    """Test hook: forget the last written fingerprint and cached coverage."""
    global _coverage
    with _lock:
        _last_written.clear()
        _coverage = {}
