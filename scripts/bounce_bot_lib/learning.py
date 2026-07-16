"""BounceBot learning loop: measured outcomes -> live alert behavior.

The outcome tracker has been recording R-based results per bounce for months;
this module closes the loop. It distills the performance aggregation into a
compact ``learning state`` JSON (per-segment average R, sample counts, score
deltas, and mute flags), which the alert path consults on every confirmed
bounce to:

- assign an evidence-based tier (S/A/B/C/D) to the alert,
- MUTE segments that are proven losers (the candidate and its outcome are
  still recorded via the existing ``learning_only`` path, so evidence keeps
  accruing and a segment can earn its way back),
- attach the measured reasons so the alert explains itself.

Rules are deliberately conservative: a segment needs ``MIN_SAMPLES`` closed
episodes before it can mute or boost anything, deltas are clipped, and human
focus picks are never muted (the caller enforces that).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from market_session import get_market_session_window
from project_paths import INTRADAY_BOUNCE_OUTCOMES_FILE

BOUNCE_LEARNING_STATE_FILE = INTRADAY_BOUNCE_OUTCOMES_FILE.with_name("intraday_bounce_learning_state.json")

MIN_SAMPLES = 10
# A mute suppresses the live alert entirely, so it demands far more evidence
# than a tier: a real sample, spread across enough distinct sessions that one
# bad week (or one mis-modeled stop) cannot silence a whole setup family.
MUTE_ENTRY_R_THRESHOLD = -0.15
MUTE_MIN_SAMPLES = 30
MUTE_MIN_SESSIONS = 10
DELTA_R_TO_POINTS = 20.0
DELTA_CLIP_POINTS = 30
TIER_THRESHOLDS = (("S", 0.90), ("A", 0.45), ("B", 0.10), ("C", -0.15))
# Thin segments earn partial credit in the composite: n=10 counts half,
# n=30 counts 75%, n=90 counts ~90%.
COMPOSITE_SHRINK_SAMPLES = 10
LEARNING_STATE_STALE_DAYS = 1

# Only a setup's own identity may MUTE it (direction-specific, evidence-based).
# Context dimensions (time bucket, market environment) weigh on the tier but
# never veto: on 2026-07-16 the old context mutes had auto-D'd every long for
# the first 2.5 session hours and silenced every classic short bounce type.
MUTE_DIMENSIONS = ("bounce_type",)
# Dimensions blended into the tier composite, with weights. Setup identity
# carries the composite; context is a drag/boost, roughly half the say.
COMPOSITE_DIMENSIONS = (
    ("bounce_type", 1.0),
    ("time_bucket", 0.4),
    ("market_environment", 0.4),
    ("master_avwap_priority_bucket", 0.6),
    ("master_avwap_focus", 0.6),
)
# PROVEN segments (2026-07-09, user rule "see the best bounces live"): a
# segment with real sample size, strong average AND non-negative median R is a
# proven winner - a live bounce matching one gets stamped PROVEN, upgraded,
# and bypasses the Alert Center tier gate like a banger. Includes the
# dimensions the tier composite does NOT blend (combos, swing traits, setup
# family), because that is where the best measured results live
# (trendline_break_recent +1.93R n=31, dynamic_vwap_upper_band +0.88R n=59,
# eod_vwap+impulse_retest combo +0.53R with 88% 1R-hit).
PROVEN_DIMENSIONS = (
    "bounce_type",
    "bounce_combo",
    "master_avwap_setup_family",
    "master_avwap_swing_trait",
    "master_avwap_focus",
)
PROVEN_MIN_SAMPLES = 12
PROVEN_MIN_AVG_R = 0.45
PROVEN_MIN_MEDIAN_R = 0.0


def _seg_key(direction: str, segment: str) -> str:
    return f"{str(direction or '').strip().lower()}|{str(segment or '').strip()}"


def entry_quality_r(
    avg_close_r: float,
    target_1r_rate: float | None = None,
    stop_rate: float | None = None,
) -> tuple[float, float]:
    """Entry quality in R units, corrected for stop-first measurement bias.

    ``avg_close_r`` is recorded under a representative stop with conservative
    stop-first adjudication: when an episode touches BOTH the stop and +1R,
    the loss is booked. Right bias for outcome honesty, but a poor *entry
    ranking* stat exactly where it matters — the tracker's headline finding
    is avg MFE 2-3R against 0.3-0.7R closed.

    ``ambiguity`` (stop_rate + target_1r_rate - 1, clipped to [0, 1]) is the
    minimum fraction of episodes where adjudication, not the entry, decided
    the recorded close. Blend the recorded close-R with the 1R-harvest
    expectancy (take the +1R partial when it prints, -1R when it never does)
    in proportion to it: clean segments keep their close-R verbatim;
    both-touch-heavy segments rank by what the entry demonstrably reached.

    Returns ``(entry_r, ambiguity)``.
    """
    if target_1r_rate is None or stop_rate is None:
        return float(avg_close_r), 0.0
    ambiguity = max(0.0, min(1.0, float(stop_rate) + float(target_1r_rate) - 1.0))
    harvest_r = 2.0 * float(target_1r_rate) - 1.0
    return (1.0 - ambiguity) * float(avg_close_r) + ambiguity * harvest_r, ambiguity


def _segment_entry_r(entry: dict) -> float | None:
    """The ranking stat for a stored segment (entry_r, or close-R on old states)."""
    value = _float_or_none(entry.get("entry_r"))
    if value is not None:
        return value
    return _float_or_none(entry.get("avg_close_r"))


def _segment_is_muted(dimension: str, entry: dict) -> bool:
    """Mute policy, recomputed from segment stats (never trusts stale flags).

    Evaluated live so a pre-v2 state file on disk immediately follows the
    current policy instead of its baked-in context mutes.
    """
    if dimension not in MUTE_DIMENSIONS:
        return False
    entry_r = _segment_entry_r(entry)
    if entry_r is None or entry_r > MUTE_ENTRY_R_THRESHOLD:
        return False
    if int(entry.get("sample_count") or 0) < MUTE_MIN_SAMPLES:
        return False
    session_count = entry.get("session_count")
    if session_count is not None and int(session_count) < MUTE_MIN_SESSIONS:
        return False
    return True


def build_learning_state(perf_rows: list[dict], *, min_samples: int = MIN_SAMPLES) -> dict:
    """Distill performance rows into the alert-time learning state."""
    segments: dict[str, dict[str, dict]] = {}
    for row in perf_rows or []:
        dimension = str(row.get("dimension") or "").strip()
        direction = str(row.get("direction") or "").strip().lower()
        segment = str(row.get("segment") or "").strip()
        try:
            sample_count = int(float(row.get("sample_count") or 0))
            avg_close_r = float(row.get("avg_close_r"))
        except (TypeError, ValueError):
            continue
        if not dimension or not segment or segment.lower() in {"nan", "none", "unknown", ""}:
            continue
        if sample_count < min_samples:
            continue
        stop_rate = _float_or_none(row.get("stop_rate"))
        target_1r_rate = _float_or_none(row.get("target_1r_rate"))
        entry_r, ambiguity = entry_quality_r(avg_close_r, target_1r_rate, stop_rate)
        session_count = row.get("session_count")
        try:
            session_count = int(float(session_count)) if session_count not in (None, "") else None
        except (TypeError, ValueError):
            session_count = None
        delta = int(max(-DELTA_CLIP_POINTS, min(DELTA_CLIP_POINTS, round(entry_r * DELTA_R_TO_POINTS))))
        median_close_r = _float_or_none(row.get("median_close_r"))
        proven = bool(
            dimension in PROVEN_DIMENSIONS
            and sample_count >= PROVEN_MIN_SAMPLES
            and avg_close_r >= PROVEN_MIN_AVG_R
            and (median_close_r is None or median_close_r >= PROVEN_MIN_MEDIAN_R)
        )
        entry = {
            "avg_close_r": round(avg_close_r, 3),
            "entry_r": round(entry_r, 3),
            "ambiguity": round(ambiguity, 3),
            "sample_count": sample_count,
            "session_count": session_count,
            "stop_rate": stop_rate,
            "target_1r_rate": target_1r_rate,
            "avg_mfe_r": _float_or_none(row.get("avg_mfe_r")),
            "median_close_r": median_close_r,
            "score_delta": delta,
            "proven": proven,
        }
        entry["muted"] = _segment_is_muted(dimension, entry)
        segments.setdefault(dimension, {})[_seg_key(direction, segment)] = entry
    return {
        "schema_version": 2,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "min_samples": int(min_samples),
        "mute_threshold_entry_r": MUTE_ENTRY_R_THRESHOLD,
        "mute_min_samples": MUTE_MIN_SAMPLES,
        "mute_min_sessions": MUTE_MIN_SESSIONS,
        "segments": segments,
    }


def _float_or_none(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def time_bucket_for(when: datetime | None) -> str:
    """Classify a timestamp by elapsed time in its NYSE session.

    The former implementation compared local wall-clock hours with Eastern
    cutoffs. On a Pacific machine that mislabeled nearly the entire session.
    """
    if when is None:
        return "unknown"
    try:
        session = get_market_session_window(reference=when)
        moment = when
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=session.open_local.tzinfo)
        else:
            moment = moment.astimezone(session.open_local.tzinfo)
        minutes = (moment - session.open_local).total_seconds() / 60.0
    except Exception:
        return "unknown"
    if minutes < 60:
        return "opening_drive"
    if minutes < 150:
        return "late_morning"
    if minutes < 270:
        return "midday"
    if minutes < 360:
        return "afternoon"
    return "closing_window"


def evaluate_bounce_quality(
    state: dict | None,
    *,
    direction: str,
    bounce_types: list[str] | tuple = (),
    time_bucket: str = "",
    market_environment: str = "",
    priority_bucket: str = "",
    focus_label: str = "",
    bounce_combo: str = "",
    setup_family: str = "",
    swing_traits: list[str] | tuple = (),
) -> dict:
    """Tier + mute + PROVEN decision + human-readable reasons for one bounce.

    The composite is a weighted mean of the measured *entry quality* R (see
    ``entry_quality_r``) of every segment this bounce belongs to, each shrunk
    by sample size; unknown segments simply do not contribute, so a bounce
    with no history lands in the neutral B/C range instead of failing.
    A bounce matching any PROVEN segment (see PROVEN_* thresholds) is flagged
    so the alert path can stamp it and the Alert Center treats it like a
    banger - unless a mute fires (proven negatives keep the veto).
    """
    segments = (state or {}).get("segments") or {}
    direction = str(direction or "").strip().lower()

    def _lookup(dimension: str, segment: str) -> dict | None:
        if not segment:
            return None
        return (segments.get(dimension) or {}).get(_seg_key(direction, segment))

    def _shrunk(entry: dict) -> float:
        value = _segment_entry_r(entry) or 0.0
        n = max(0, int(entry.get("sample_count") or 0))
        return value * n / float(n + COMPOSITE_SHRINK_SAMPLES)

    reasons: list[str] = []
    mute_reasons: list[str] = []
    weighted_sum = 0.0
    weight_used = 0.0

    dim_values = {
        "bounce_type": list(bounce_types or []),
        "time_bucket": [time_bucket],
        "market_environment": [market_environment],
        "master_avwap_priority_bucket": [priority_bucket],
        "master_avwap_focus": [focus_label],
        "bounce_combo": [bounce_combo],
        "master_avwap_setup_family": [setup_family],
        "master_avwap_swing_trait": list(swing_traits or []),
    }
    for dimension, weight in COMPOSITE_DIMENSIONS:
        values = [v for v in dim_values.get(dimension, []) if str(v or "").strip()]
        entries = [(v, _lookup(dimension, v)) for v in values]
        entries = [(v, e) for v, e in entries if e]
        if not entries:
            continue
        dim_avg = sum(_shrunk(e) for _v, e in entries) / len(entries)
        weighted_sum += dim_avg * weight
        weight_used += weight
        for value, entry in entries:
            entry_r = _segment_entry_r(entry) or 0.0
            tag = (
                f"{value} {direction} entry {entry_r:+.2f}R "
                f"(close {entry['avg_close_r']:+.2f}R, n={entry['sample_count']})"
            )
            reasons.append(tag)
            if _segment_is_muted(dimension, entry):
                session_note = (
                    f" across {int(entry['session_count'])} sessions"
                    if entry.get("session_count") is not None
                    else ""
                )
                mute_reasons.append(
                    f"{dimension}={value}: entry {entry_r:+.2f}R over "
                    f"{entry['sample_count']}{session_note} — proven negative"
                )

    proven_reasons: list[str] = []
    best_proven_avg = None
    for dimension in PROVEN_DIMENSIONS:
        for value in dim_values.get(dimension, []):
            value = str(value or "").strip()
            entry = _lookup(dimension, value)
            if not entry or not entry.get("proven"):
                continue
            proven_reasons.append(
                f"{value}: {entry['avg_close_r']:+.2f}R (n={entry['sample_count']})"
            )
            if best_proven_avg is None or entry["avg_close_r"] > best_proven_avg:
                best_proven_avg = entry["avg_close_r"]

    composite = (weighted_sum / weight_used) if weight_used > 0 else None
    muted = bool(mute_reasons)
    proven = bool(proven_reasons) and not muted
    if muted:
        tier = "D"
    elif composite is None:
        tier = "B"  # no evidence either way: normal alert, keep collecting
    else:
        tier = "D"
        for label, threshold in TIER_THRESHOLDS:
            if composite >= threshold:
                tier = label
                break
    if proven:
        # Evidence-based floor: a proven match never rides below A, and a
        # segment that clears the S composite bar on its own carries the S.
        floor = "S" if (best_proven_avg or 0.0) >= TIER_THRESHOLDS[0][1] else "A"
        order = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
        if order.get(tier, 4) > order[floor]:
            tier = floor
    return {
        "tier": tier,
        "composite_r": round(composite, 3) if composite is not None else None,
        "muted": muted,
        "mute_reasons": mute_reasons,
        "reasons": reasons[:4],
        "proven": proven,
        "proven_reasons": proven_reasons[:3],
    }


# ---------------------------------------------------------------------------
# State IO + refresh
# ---------------------------------------------------------------------------
_state_cache: dict = {"mtime": None, "state": None}


def measured_exit_note(
    state: dict | None,
    *,
    direction: str,
    bounce_types: list[str] | tuple = (),
) -> str:
    """The tracker's own exit evidence for this bounce type, or "".

    Exits are the measured leak (MFE 2-3R vs 0.3-0.7R closed), so alerts show
    the numbers for the best-sampled matching bounce-type segment. Segments
    only exist in the state once they clear MIN_SAMPLES, so presence already
    implies enough evidence.
    """
    segments = (state or {}).get("segments") or {}
    bounce_segments = segments.get("bounce_type") or {}
    direction = str(direction or "").strip().lower()

    best = None
    best_type = ""
    for bounce_type in bounce_types or ():
        entry = bounce_segments.get(_seg_key(direction, str(bounce_type)))
        if not isinstance(entry, dict) or entry.get("avg_mfe_r") is None:
            continue
        if best is None or int(entry.get("sample_count") or 0) > int(best.get("sample_count") or 0):
            best = entry
            best_type = str(bounce_type)
    if best is None:
        return ""
    mfe = float(best["avg_mfe_r"])
    close_r = _float_or_none(best.get("avg_close_r"))
    close_text = f" vs {close_r:+.1f}R close" if close_r is not None else ""
    note = f"measured {best_type}: avg MFE {mfe:.1f}R{close_text} (n={int(best.get('sample_count') or 0)})"
    if mfe >= 1.2 and (close_r is None or mfe >= close_r + 0.5):
        note += " -> harvest the partial, trail the rest"
    return note


def load_bounce_learning_state(path: Path | None = None) -> dict | None:
    state_path = Path(path) if path else BOUNCE_LEARNING_STATE_FILE
    try:
        mtime = state_path.stat().st_mtime
    except OSError:
        return None
    if _state_cache["mtime"] == mtime and _state_cache["state"] is not None:
        return _state_cache["state"]
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logging.warning("Could not load bounce learning state: %s", exc)
        return _state_cache["state"]
    _state_cache["mtime"] = mtime
    _state_cache["state"] = state if isinstance(state, dict) else None
    return _state_cache["state"]


def refresh_bounce_learning_state(*, min_samples: int = MIN_SAMPLES, path: Path | None = None) -> dict:
    """Rebuild performance rows + report, then write the learning state JSON.

    Imports the aggregation lazily so this module stays import-light for the
    UI and for tests that only need the pure functions.
    """
    from bounce_bot_lib.legacy import (
        INTRADAY_BOUNCE_PERFORMANCE_CSV,
        build_intraday_bounce_performance_rows,
        write_intraday_bounce_performance_report,
    )
    import pandas as pd

    perf_rows = build_intraday_bounce_performance_rows(min_samples=min(5, min_samples))
    if perf_rows:
        INTRADAY_BOUNCE_PERFORMANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(perf_rows).to_csv(INTRADAY_BOUNCE_PERFORMANCE_CSV, index=False)
        write_intraday_bounce_performance_report(perf_rows)
    state = build_learning_state(perf_rows, min_samples=min_samples)
    state_path = Path(path) if path else BOUNCE_LEARNING_STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _state_cache["mtime"] = None  # force reload next read
    logging.info(
        "Bounce learning state refreshed: %s dimension(s), %s segment(s).",
        len(state["segments"]),
        sum(len(v) for v in state["segments"].values()),
    )
    return state


def refresh_bounce_learning_if_stale(*, max_age_days: int = LEARNING_STATE_STALE_DAYS) -> bool:
    try:
        age = datetime.now() - datetime.fromtimestamp(BOUNCE_LEARNING_STATE_FILE.stat().st_mtime)
        if age <= timedelta(days=max_age_days):
            return False
    except OSError:
        pass  # missing -> build the first one
    refresh_bounce_learning_state()
    return True


# ---------------------------------------------------------------------------
# Candidates CSV compaction (the 300MB JSON-blob bloat)
# ---------------------------------------------------------------------------
# near_miss rows are 97%+ of the file (~2.2KB each with the JSON blobs) but
# only matter for near-term "what almost fired" review; real learning joins
# only use confirmed rows, which are a rounding error by size.
NEAR_MISS_KEEP_DAYS = 30
DEFAULT_EVENT_KEEP_DAYS = 365


def compact_bounce_candidates_csv(
    path: Path,
    *,
    max_age_days: int = DEFAULT_EVENT_KEEP_DAYS,
    near_miss_keep_days: int = NEAR_MISS_KEEP_DAYS,
    min_bytes_to_bother: int = 50_000_000,
) -> dict:
    """Trim the candidates CSV with event-type-aware retention (streamed, atomic).

    ``near_miss`` rows older than ``near_miss_keep_days`` are dropped; all other
    event types (confirmed/detected/invalidated/expired) keep ``max_age_days``.
    Outcomes join by event_id against confirmed rows, so learning history is
    preserved. Returns a summary dict; no-ops when the file is small or missing.
    """
    csv_path = Path(path)
    try:
        size_before = csv_path.stat().st_size
    except OSError:
        return {"compacted": False, "reason": "missing"}
    if size_before < int(min_bytes_to_bother):
        return {"compacted": False, "reason": "below size threshold", "bytes": size_before}

    today = datetime.now().date()
    cutoff_default = (today - timedelta(days=int(max_age_days))).isoformat()
    cutoff_near_miss = (today - timedelta(days=int(near_miss_keep_days))).isoformat()
    kept = dropped = 0
    fd, temp_name = tempfile.mkstemp(prefix=csv_path.stem, suffix=".csv", dir=str(csv_path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as out_handle, open(
            csv_path, "r", newline="", encoding="utf-8-sig"
        ) as in_handle:
            reader = csv.reader(in_handle)
            writer = csv.writer(out_handle)
            header = next(reader, None)
            if header is None:
                return {"compacted": False, "reason": "empty"}
            writer.writerow(header)
            try:
                date_idx = header.index("trade_date")
                type_idx = header.index("event_type")
            except ValueError:
                return {"compacted": False, "reason": "missing trade_date/event_type column"}
            for row in reader:
                trade_date = row[date_idx] if date_idx < len(row) else ""
                event_type = (row[type_idx] if type_idx < len(row) else "").strip().lower()
                cutoff = cutoff_near_miss if event_type == "near_miss" else cutoff_default
                if trade_date and trade_date >= cutoff:
                    writer.writerow(row)
                    kept += 1
                else:
                    dropped += 1
        os.replace(temp_name, csv_path)
        temp_name = None
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)

    size_after = csv_path.stat().st_size
    logging.info(
        "Compacted %s: kept %s rows, dropped %s (near_miss < %s, others < %s); %.1fMB -> %.1fMB",
        csv_path.name,
        kept,
        dropped,
        cutoff_near_miss,
        cutoff_default,
        size_before / 1e6,
        size_after / 1e6,
    )
    return {
        "compacted": True,
        "kept": kept,
        "dropped": dropped,
        "cutoff": cutoff_default,
        "near_miss_cutoff": cutoff_near_miss,
        "bytes_before": size_before,
        "bytes_after": size_after,
    }
