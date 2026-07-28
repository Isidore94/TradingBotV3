#!/usr/bin/env python3
"""Phase 1 of the review-learning loop: the revealed-preference scoreboard.

Reads the Alert Center decision log (``alert_review_events.jsonl``, written by
review_events.py) and answers two questions per segment of alerts:

1. **What does the trader actually take?**  P(take | shown) per tier, side,
   bounce type, time bucket, market environment, rvol band, RRS alignment -
   with the same ``n/(n+k)`` shrinkage the bounce learning engine uses, so a
   segment seen three times cannot dominate the board.
2. **Does it pay?**  Taken vs passed outcomes: intraday alerts join to the
   bounce outcomes CSV through ``event_id`` (close R against the alert's own
   stop - tracked for every confirmed event whether the trader acted or not),
   and D1/swing alerts grade on side-adjusted forward returns from the durable
   daily parquet store.

The gold is where the two disagree:

- **Blind spots** - segments the trader habitually passes on that measurably
  outperform ("you skip 80% of closing-window sigma-band shorts; they averaged
  +0.6R").
- **Leaks** - segments the trader reliably takes that measurably bleed.

Also reports armed-watch conversion (armed -> fired / expired / disarmed per
watch kind) and which quick-fill sources the trader arms levels off.

Everything is counting + shrinkage in the house style - no fitted model. The
output feeds the Phase 2 UI (queue ordering / "you usually skip this" chips);
this phase only ranks and annotates, it never suppresses anything.

Run:
    .venv/Scripts/python.exe scripts/review_learning.py            # rebuild + print
    .venv/Scripts/python.exe scripts/review_learning.py --days 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import (  # noqa: E402
    ALERT_REVIEW_EVENTS_FILE,
    INTRADAY_BOUNCE_OUTCOMES_FILE,
    REVIEW_LEARNING_REPORT_FILE,
    REVIEW_PREFERENCE_STATE_FILE,
)
from review_events import load_review_events  # noqa: E402

REVIEW_LEARNING_SCHEMA = "review_learning_v1"

# Analysis window: behavior from months ago should age out of the board.
DEFAULT_WINDOW_DAYS = 90
# Empirical-Bayes shrinkage toward the overall take rate; matches the bounce
# learning engine's COMPOSITE_SHRINK_SAMPLES.
SHRINK_SAMPLES = 10
# Minimum shown-episodes before a segment may appear in a callout; matches the
# scoring tuner's min_setups gate.
MIN_CALLOUT_EPISODES = 8
# Callout thresholds. A blind spot must be taken well below the trader's
# overall rate AND measure positive when passed; a leak the reverse.
BLIND_SPOT_TAKE_RATIO = 0.6
BLIND_SPOT_MIN_PASSED_R = 0.30
BLIND_SPOT_MIN_PASSED_FWD_PCT = 1.5
LEAK_TAKE_RATIO = 1.4
LEAK_MAX_TAKEN_R = -0.15
LEAK_MAX_TAKEN_FWD_PCT = -1.0
# Forward-return horizons (trading sessions) for D1/swing-graded alerts.
FORWARD_HORIZONS = (3, 5)

# Episode resolution, strongest first: any positive engagement outranks an
# explicit rejection, which outranks a soft skip, which outranks silence.
TAKE_ACTIONS = {"add_focus", "arm_watch", "arm_level"}
TOGGLE_TAKE_ACTIONS = {"favorite", "toggle_d1_focus", "toggle_m5_focus"}
REJECT_ACTIONS = {"dislike", "remove_today"}


@dataclass
class Episode:
    """Everything that happened to one (trade_date, symbol) in the queue."""

    trade_date: str
    symbol: str
    side: str = ""
    resolution: str = "shown_only"  # take | reject | skip | shown_only
    shown: bool = False
    dwell_ms: float | None = None
    shown_ts: str = ""
    event_id: str = ""
    tier: str = ""
    tag: str = ""
    timeframe: str = ""
    is_d1: bool = False
    proven: bool = False
    banger: bool = False
    bounce_types: str = ""
    market_environment: str = ""
    session_rvol: float | None = None
    rrs_spy: float | None = None
    close_r: float | None = None
    forward_pct: dict[int, float] = field(default_factory=dict)


def _as_float(value) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved == resolved else None


def _detail_on(row: dict) -> bool:
    detail = row.get("detail")
    return bool(detail.get("on")) if isinstance(detail, dict) else False


def _is_take(row: dict) -> bool:
    action = str(row.get("action") or "")
    if action in TAKE_ACTIONS:
        return True
    return action in TOGGLE_TAKE_ACTIONS and _detail_on(row)


_CONTEXT_KEYS = (
    "side",
    "tier",
    "tag",
    "timeframe",
    "is_d1",
    "proven",
    "banger",
    "bounce_types",
    "market_environment",
    "event_id",
)


def build_episodes(rows: Iterable[dict]) -> list[Episode]:
    """Fold raw event rows into per-(trade_date, symbol) decision episodes."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        trade_date = str(row.get("trade_date") or "")
        symbol = str(row.get("symbol") or "").upper()
        if trade_date and symbol:
            grouped[(trade_date, symbol)].append(row)

    episodes = []
    for (trade_date, symbol), events in sorted(grouped.items()):
        episode = Episode(trade_date=trade_date, symbol=symbol)
        rank = 0  # 0 shown_only, 1 skip, 2 reject, 3 take
        for row in events:
            action = str(row.get("action") or "")
            if action == "shown":
                episode.shown = True
                if not episode.shown_ts:
                    episode.shown_ts = str(row.get("ts") or "")
            new_rank = (
                3
                if _is_take(row)
                else 2
                if action in REJECT_ACTIONS
                else 1
                if action == "skip"
                else 0
            )
            if new_rank >= rank and new_rank > 0:
                rank = new_rank
                episode.resolution = {3: "take", 2: "reject", 1: "skip"}[new_rank]
                dwell = _as_float(row.get("dwell_ms"))
                if dwell is not None:
                    episode.dwell_ms = dwell
            # Context: the richest row wins field by field, so a bare toggle
            # row cannot blank out what the shown impression already carried.
            for key in _CONTEXT_KEYS:
                value = row.get(key)
                if value not in (None, "", False) and not getattr(episode, key, None):
                    setattr(episode, key, value)
            for key in ("session_rvol", "rrs_spy"):
                value = _as_float(row.get(key))
                if value is not None and getattr(episode, key) is None:
                    setattr(episode, key, value)
        episodes.append(episode)
    return episodes


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def _alert_kind(episode: Episode) -> str:
    if episode.tag == "chart_watch":
        return "chart_watch"
    if episode.tag == "manual_chart":
        return "manual"
    if episode.is_d1 or str(episode.timeframe).upper() == "D1":
        return "d1"
    return "m5"


def _split_bounce_types(text: str) -> list[str]:
    parts = []
    for chunk in str(text or "").replace("+", ";").replace(",", ";").split(";"):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _rvol_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.8:
        return "quiet(<0.8)"
    if value < 1.5:
        return "normal(0.8-1.5)"
    if value < 2.5:
        return "elevated(1.5-2.5)"
    return "hot(>2.5)"


def _rrs_alignment(episode: Episode) -> str:
    value = episode.rrs_spy
    if value is None or episode.side not in ("LONG", "SHORT"):
        return "unknown"
    if abs(value) < 0.5:
        return "flat"
    aligned = value > 0 if episode.side == "LONG" else value < 0
    return "aligned" if aligned else "against"


def _time_bucket(ts_text: str) -> str:
    """Session bucket of the impression; separated for test injection."""
    try:
        from bounce_bot_lib.learning import time_bucket_for

        return time_bucket_for(datetime.fromisoformat(ts_text)) or "unknown"
    except Exception:
        return "unknown"


DIMENSIONS: dict[str, Callable[[Episode], list[str]]] = {
    "tier": lambda e: [e.tier or "untiered"],
    "side": lambda e: [e.side or "WATCH"],
    "alert_kind": lambda e: [_alert_kind(e)],
    "bounce_type": lambda e: _split_bounce_types(e.bounce_types),
    "time_bucket": lambda e: [_time_bucket(e.shown_ts)] if e.shown_ts else [],
    "market_environment": lambda e: [e.market_environment] if e.market_environment else [],
    "rvol_bucket": lambda e: [_rvol_bucket(e.session_rvol)],
    "rrs_alignment": lambda e: [_rrs_alignment(e)],
}


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def load_outcomes_for(
    event_ids: set[str], path: Path = INTRADAY_BOUNCE_OUTCOMES_FILE
) -> dict[str, float]:
    """{event_id: close_r} for the requested events, streamed (the outcomes
    CSV is ~100MB; only the rows we saw in review are worth decoding).

    Events log multiple milestone rows; the end-of-day row wins, else the
    last row in file order.
    """
    if not event_ids:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    results: dict[str, float] = {}
    is_eod: dict[str, bool] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                event_id = str(row.get("event_id") or "")
                if event_id not in event_ids:
                    continue
                close_r = _as_float(row.get("close_r"))
                if close_r is None:
                    continue
                eod = str(row.get("outcome_mode") or "").strip().lower() == "eod"
                if event_id not in results or eod or not is_eod.get(event_id):
                    results[event_id] = close_r
                    is_eod[event_id] = eod or is_eod.get(event_id, False)
    except OSError:
        return results
    return results


def attach_bounce_outcomes(
    episodes: list[Episode], path: Path = INTRADAY_BOUNCE_OUTCOMES_FILE
) -> int:
    wanted = {e.event_id for e in episodes if e.event_id}
    outcomes = load_outcomes_for(wanted, path)
    matched = 0
    for episode in episodes:
        if episode.event_id and episode.event_id in outcomes:
            episode.close_r = outcomes[episode.event_id]
            matched += 1
    return matched


def attach_forward_returns(
    episodes: list[Episode],
    *,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
    load_frame=None,
) -> int:
    """Side-adjusted forward returns for D1/swing-graded episodes.

    Uses the durable daily parquet store via human_focus_tracking's loader
    (import deferred: pandas). Entry = close of the first session on/after
    the trade date; immature picks simply skip the horizon.
    """
    candidates = [
        e
        for e in episodes
        if _alert_kind(e) == "d1" and e.side in ("LONG", "SHORT") and e.close_r is None
    ]
    if not candidates:
        return 0
    if load_frame is None:
        try:
            from human_focus_tracking import (
                _load_durable_daily_frame,
                _normalize_daily_frame,
            )

            def load_frame(symbol):  # noqa: F811 - deliberate late bind
                frame = _normalize_daily_frame(_load_durable_daily_frame(symbol))
                return list(
                    zip(
                        [d.date().isoformat() for d in frame["datetime"]],
                        [float(c) for c in frame["close"]],
                    )
                )
        except Exception:
            return 0

    closes_cache: dict[str, list[tuple[str, float]]] = {}
    matched = 0
    for episode in candidates:
        closes = closes_cache.get(episode.symbol)
        if closes is None:
            try:
                closes = list(load_frame(episode.symbol) or [])
            except Exception:
                closes = []
            closes_cache[episode.symbol] = closes
        if not closes:
            continue
        entry_index = next(
            (i for i, (day, _) in enumerate(closes) if day >= episode.trade_date), None
        )
        if entry_index is None:
            continue
        entry_close = closes[entry_index][1]
        if not entry_close or entry_close <= 0:
            continue
        sign = 1.0 if episode.side == "LONG" else -1.0
        got_any = False
        for horizon in horizons:
            exit_index = entry_index + horizon
            if exit_index >= len(closes):
                continue  # immature - grade only what has fully played out
            move = (closes[exit_index][1] - entry_close) / entry_close * 100.0
            episode.forward_pct[horizon] = sign * move
            got_any = True
        if got_any:
            matched += 1
    return matched


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _outcome_stats(episodes: list[Episode]) -> dict[str, Any]:
    rs = [e.close_r for e in episodes if e.close_r is not None]
    stats: dict[str, Any] = {
        "r_n": len(rs),
        "r_avg": round(sum(rs) / len(rs), 3) if rs else None,
    }
    horizon = FORWARD_HORIZONS[0]
    fwd = [e.forward_pct[horizon] for e in episodes if horizon in e.forward_pct]
    stats["fwd_n"] = len(fwd)
    stats["fwd_avg_pct"] = round(sum(fwd) / len(fwd), 2) if fwd else None
    return stats


def aggregate_dimensions(episodes: list[Episode]) -> dict[str, Any]:
    shown = [e for e in episodes if e.shown]
    takes = sum(1 for e in shown if e.resolution == "take")
    overall = takes / len(shown) if shown else 0.0

    dimensions: dict[str, dict[str, Any]] = {}
    for dim, key_fn in DIMENSIONS.items():
        segments: dict[str, list[Episode]] = defaultdict(list)
        for episode in shown:
            for segment in key_fn(episode):
                segments[segment].append(episode)
        table = {}
        for segment, members in segments.items():
            n = len(members)
            seg_takes = sum(1 for e in members if e.resolution == "take")
            taken = [e for e in members if e.resolution == "take"]
            passed = [e for e in members if e.resolution != "take"]
            take_dwells = [e.dwell_ms for e in taken if e.dwell_ms is not None]
            pass_dwells = [e.dwell_ms for e in passed if e.dwell_ms is not None]
            table[segment] = {
                "shown": n,
                "take": seg_takes,
                "skip": sum(1 for e in members if e.resolution == "skip"),
                "reject": sum(1 for e in members if e.resolution == "reject"),
                "take_rate": round(seg_takes / n, 3) if n else 0.0,
                # Shrunk toward the trader's overall rate: thin segments read
                # as "about average" until the evidence says otherwise.
                "take_rate_shrunk": round(
                    (seg_takes + SHRINK_SAMPLES * overall) / (n + SHRINK_SAMPLES), 3
                ),
                "median_take_dwell_ms": median(take_dwells) if take_dwells else None,
                "median_pass_dwell_ms": median(pass_dwells) if pass_dwells else None,
                "taken": _outcome_stats(taken),
                "passed": _outcome_stats(passed),
            }
        dimensions[dim] = table
    return {
        "episodes": len(episodes),
        "shown": len(shown),
        "takes": takes,
        "overall_take_rate": round(overall, 3),
        "dimensions": dimensions,
    }


def find_callouts(aggregate: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """(blind_spots, leaks): where revealed preference and measurement disagree."""
    overall = aggregate.get("overall_take_rate") or 0.0
    blind_spots, leaks = [], []
    for dim, table in (aggregate.get("dimensions") or {}).items():
        for segment, stats in table.items():
            if stats["shown"] < MIN_CALLOUT_EPISODES:
                continue
            entry = {
                "dimension": dim,
                "segment": segment,
                "shown": stats["shown"],
                "take_rate": stats["take_rate"],
                "take_rate_shrunk": stats["take_rate_shrunk"],
            }
            passed, taken = stats["passed"], stats["taken"]
            if overall > 0 and stats["take_rate_shrunk"] <= overall * BLIND_SPOT_TAKE_RATIO:
                if (
                    passed["r_n"] >= MIN_CALLOUT_EPISODES
                    and (passed["r_avg"] or 0) >= BLIND_SPOT_MIN_PASSED_R
                ):
                    blind_spots.append(
                        {**entry, "passed_r_avg": passed["r_avg"], "passed_r_n": passed["r_n"]}
                    )
                elif (
                    passed["fwd_n"] >= MIN_CALLOUT_EPISODES
                    and (passed["fwd_avg_pct"] or 0) >= BLIND_SPOT_MIN_PASSED_FWD_PCT
                ):
                    blind_spots.append(
                        {
                            **entry,
                            "passed_fwd_avg_pct": passed["fwd_avg_pct"],
                            "passed_fwd_n": passed["fwd_n"],
                        }
                    )
            if overall > 0 and stats["take_rate_shrunk"] >= overall * LEAK_TAKE_RATIO:
                if (
                    taken["r_n"] >= MIN_CALLOUT_EPISODES
                    and (taken["r_avg"] or 0) <= LEAK_MAX_TAKEN_R
                ):
                    leaks.append(
                        {**entry, "taken_r_avg": taken["r_avg"], "taken_r_n": taken["r_n"]}
                    )
                elif (
                    taken["fwd_n"] >= MIN_CALLOUT_EPISODES
                    and (taken["fwd_avg_pct"] or 0) <= LEAK_MAX_TAKEN_FWD_PCT
                ):
                    leaks.append(
                        {
                            **entry,
                            "taken_fwd_avg_pct": taken["fwd_avg_pct"],
                            "taken_fwd_n": taken["fwd_n"],
                        }
                    )
    blind_spots.sort(key=lambda e: e.get("passed_r_avg") or e.get("passed_fwd_avg_pct") or 0, reverse=True)
    leaks.sort(key=lambda e: e.get("taken_r_avg") or e.get("taken_fwd_avg_pct") or 0)
    return blind_spots, leaks


def watch_conversion(rows: Iterable[dict]) -> dict[str, Any]:
    """Per watch kind: armed -> fired / expired / disarmed, plus which
    quick-fill sources armed levels came from."""
    kinds: dict[str, Counter] = defaultdict(Counter)
    fill_sources: Counter = Counter()
    for row in rows:
        action = str(row.get("action") or "")
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        kind = str(detail.get("kind") or "")
        if action == "arm_watch" and kind:
            kinds[kind]["armed"] += 1
        elif action == "disarm_watch" and kind:
            kinds[kind]["disarmed"] += 1
        elif action == "watch_fired":
            kinds[kind or str(row.get("chart_watch_kind") or "unknown")]["fired"] += 1
        elif action == "watch_expired" and kind:
            kinds[kind]["expired"] += 1
        elif action == "arm_level":
            kinds["d1_level"]["armed"] += 1
            source = str(detail.get("fill_source") or "unspecified")
            fill_sources[source] += 1
        elif action == "disarm_level":
            kinds["d1_level"]["disarmed"] += 1
        elif action == "level_fired":
            kinds["d1_level"]["fired"] += 1
    return {
        "kinds": {kind: dict(counter) for kind, counter in sorted(kinds.items())},
        "level_fill_sources": dict(fill_sources),
    }


# ---------------------------------------------------------------------------
# State + report
# ---------------------------------------------------------------------------
def build_review_learning_state(
    *,
    events_path: Path = ALERT_REVIEW_EVENTS_FILE,
    outcomes_path: Path = INTRADAY_BOUNCE_OUTCOMES_FILE,
    window_days: int = DEFAULT_WINDOW_DAYS,
    load_frame=None,
    now: datetime | None = None,
) -> dict[str, Any]:
    moment = now or datetime.now()
    cutoff = (moment - timedelta(days=window_days)).date().isoformat()
    rows = [
        row
        for row in load_review_events(events_path)
        if str(row.get("trade_date") or "") >= cutoff
    ]
    episodes = build_episodes(rows)
    outcome_matches = attach_bounce_outcomes(episodes, outcomes_path)
    forward_matches = attach_forward_returns(episodes, load_frame=load_frame)
    aggregate = aggregate_dimensions(episodes)
    blind_spots, leaks = find_callouts(aggregate)
    return {
        "schema": REVIEW_LEARNING_SCHEMA,
        "generated_at": moment.isoformat(timespec="seconds"),
        "window_days": window_days,
        "event_rows": len(rows),
        "outcome_matches": outcome_matches,
        "forward_matches": forward_matches,
        **aggregate,
        "blind_spots": blind_spots,
        "leaks": leaks,
        "watch_conversion": watch_conversion(rows),
    }


def save_review_learning_state(
    state: dict[str, Any], path: Path = REVIEW_PREFERENCE_STATE_FILE
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=1, sort_keys=True, default=str)
        os.replace(temp_name, path)
    except OSError:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def load_review_learning_state(
    path: Path = REVIEW_PREFERENCE_STATE_FILE,
) -> dict[str, Any] | None:
    path = Path(path)
    try:
        if not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _fmt_rate(value) -> str:
    return f"{value * 100:.0f}%" if value is not None else "n/a"


def _fmt_r(value) -> str:
    return f"{value:+.2f}R" if value is not None else "n/a"


def _fmt_pct(value) -> str:
    return f"{value:+.1f}%" if value is not None else "n/a"


def render_report(state: dict[str, Any]) -> str:
    lines = [
        f"REVIEW PREFERENCE SCOREBOARD  (generated {state.get('generated_at', '')})",
        f"Window: last {state.get('window_days')} days · {state.get('shown', 0)} charts shown, "
        f"{state.get('takes', 0)} taken (overall take rate {_fmt_rate(state.get('overall_take_rate'))}) · "
        f"{state.get('outcome_matches', 0)} intraday outcomes joined, "
        f"{state.get('forward_matches', 0)} D1 names forward-graded.",
        "take% is shrunk toward your overall rate (k=10); R = vs the alert's own stop;",
        "fwd = side-adjusted % return after "
        f"{FORWARD_HORIZONS[0]} sessions for D1 names. Passed = skip/remove/no action.",
        "",
    ]

    blind_spots = state.get("blind_spots") or []
    leaks = state.get("leaks") or []
    lines.append("== BLIND SPOTS (you pass on these; they measure well) ==")
    if blind_spots:
        for entry in blind_spots:
            measure = (
                f"passed avg {_fmt_r(entry['passed_r_avg'])} (n={entry['passed_r_n']})"
                if "passed_r_avg" in entry
                else f"passed avg {_fmt_pct(entry['passed_fwd_avg_pct'])} (n={entry['passed_fwd_n']})"
            )
            lines.append(
                f"  {entry['dimension']}={entry['segment']}: take {_fmt_rate(entry['take_rate'])} "
                f"of {entry['shown']} shown; {measure}"
            )
    else:
        lines.append("  none at current sample sizes.")
    lines.append("")
    lines.append("== LEAKS (you take these; they measure poorly) ==")
    if leaks:
        for entry in leaks:
            measure = (
                f"taken avg {_fmt_r(entry['taken_r_avg'])} (n={entry['taken_r_n']})"
                if "taken_r_avg" in entry
                else f"taken avg {_fmt_pct(entry['taken_fwd_avg_pct'])} (n={entry['taken_fwd_n']})"
            )
            lines.append(
                f"  {entry['dimension']}={entry['segment']}: take {_fmt_rate(entry['take_rate'])} "
                f"of {entry['shown']} shown; {measure}"
            )
    else:
        lines.append("  none at current sample sizes.")
    lines.append("")

    for dim, table in (state.get("dimensions") or {}).items():
        if not table:
            continue
        lines.append(f"== {dim.upper().replace('_', ' ')} ==")
        lines.append(
            f"{'segment':<34}{'shown':>6}{'take%':>7}{'skip':>6}{'rej':>5}"
            f"{'takenR':>9}{'passedR':>9}{'taken fwd':>10}{'passed fwd':>11}"
        )
        ordered = sorted(table.items(), key=lambda kv: kv[1]["shown"], reverse=True)
        for segment, stats in ordered:
            taken, passed = stats["taken"], stats["passed"]
            lines.append(
                f"{segment[:33]:<34}{stats['shown']:>6}"
                f"{_fmt_rate(stats['take_rate_shrunk']):>7}"
                f"{stats['skip']:>6}{stats['reject']:>5}"
                f"{_fmt_r(taken['r_avg']):>9}{_fmt_r(passed['r_avg']):>9}"
                f"{_fmt_pct(taken['fwd_avg_pct']):>10}{_fmt_pct(passed['fwd_avg_pct']):>11}"
            )
        lines.append("")

    conversion = state.get("watch_conversion") or {}
    kinds = conversion.get("kinds") or {}
    if kinds:
        lines.append("== ARMED WATCH CONVERSION ==")
        lines.append(f"{'kind':<18}{'armed':>7}{'fired':>7}{'expired':>9}{'disarmed':>10}")
        for kind, counts in kinds.items():
            lines.append(
                f"{kind:<18}{counts.get('armed', 0):>7}{counts.get('fired', 0):>7}"
                f"{counts.get('expired', 0):>9}{counts.get('disarmed', 0):>10}"
            )
        sources = conversion.get("level_fill_sources") or {}
        if sources:
            ranked = sorted(sources.items(), key=lambda kv: kv[1], reverse=True)
            lines.append(
                "level fill sources: "
                + ", ".join(f"{source} x{count}" for source, count in ranked)
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def refresh_review_learning_if_stale(
    *,
    max_age_hours: float = 12.0,
    events_path: Path = ALERT_REVIEW_EVENTS_FILE,
    outcomes_path: Path = INTRADAY_BOUNCE_OUTCOMES_FILE,
    state_path: Path = REVIEW_PREFERENCE_STATE_FILE,
    report_path: Path = REVIEW_LEARNING_REPORT_FILE,
) -> bool:
    """Rebuild the scoreboard when it is old or the log has newer decisions.

    Called from GUI startup in a daemon thread (house pattern: the bounce
    learning refresh on bot startup). Returns True when a rebuild happened.
    """
    events_path = Path(events_path)
    state_path = Path(state_path)
    if not events_path.exists():
        return False
    try:
        if state_path.exists():
            state_mtime = state_path.stat().st_mtime
            fresh = (datetime.now().timestamp() - state_mtime) < max_age_hours * 3600
            if fresh and events_path.stat().st_mtime <= state_mtime:
                return False
    except OSError:
        pass
    state = build_review_learning_state(
        events_path=events_path, outcomes_path=outcomes_path
    )
    save_review_learning_state(state, state_path)
    try:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(state), encoding="utf-8")
    except OSError:
        pass
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the review preference scoreboard.")
    parser.add_argument("--days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--events", type=Path, default=ALERT_REVIEW_EVENTS_FILE)
    parser.add_argument("--outcomes", type=Path, default=INTRADAY_BOUNCE_OUTCOMES_FILE)
    parser.add_argument("--state", type=Path, default=REVIEW_PREFERENCE_STATE_FILE)
    parser.add_argument("--report", type=Path, default=REVIEW_LEARNING_REPORT_FILE)
    args = parser.parse_args(argv)

    state = build_review_learning_state(
        events_path=args.events, outcomes_path=args.outcomes, window_days=args.days
    )
    save_review_learning_state(state, args.state)
    report = render_report(state)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"state -> {args.state}")
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
