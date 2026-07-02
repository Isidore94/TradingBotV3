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

from project_paths import INTRADAY_BOUNCE_OUTCOMES_FILE

BOUNCE_LEARNING_STATE_FILE = INTRADAY_BOUNCE_OUTCOMES_FILE.with_name("intraday_bounce_learning_state.json")

MIN_SAMPLES = 10
MUTE_AVG_R_THRESHOLD = -0.15
DELTA_R_TO_POINTS = 20.0
DELTA_CLIP_POINTS = 30
TIER_THRESHOLDS = (("S", 0.90), ("A", 0.45), ("B", 0.10), ("C", -0.15))
LEARNING_STATE_STALE_DAYS = 1

# Dimensions that may MUTE an alert outright (direction-specific, evidence-based).
MUTE_DIMENSIONS = ("bounce_type", "time_bucket", "market_environment")
# Dimensions blended into the tier composite, with weights.
COMPOSITE_DIMENSIONS = (
    ("bounce_type", 1.0),
    ("time_bucket", 0.8),
    ("market_environment", 0.6),
    ("master_avwap_priority_bucket", 0.8),
    ("master_avwap_focus", 0.6),
)


def _seg_key(direction: str, segment: str) -> str:
    return f"{str(direction or '').strip().lower()}|{str(segment or '').strip()}"


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
        delta = int(max(-DELTA_CLIP_POINTS, min(DELTA_CLIP_POINTS, round(avg_close_r * DELTA_R_TO_POINTS))))
        muted = bool(dimension in MUTE_DIMENSIONS and avg_close_r <= MUTE_AVG_R_THRESHOLD)
        segments.setdefault(dimension, {})[_seg_key(direction, segment)] = {
            "avg_close_r": round(avg_close_r, 3),
            "sample_count": sample_count,
            "stop_rate": _float_or_none(row.get("stop_rate")),
            "target_1r_rate": _float_or_none(row.get("target_1r_rate")),
            "score_delta": delta,
            "muted": muted,
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "min_samples": int(min_samples),
        "mute_threshold_avg_r": MUTE_AVG_R_THRESHOLD,
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
    """Same bucketing the performance aggregation uses (market-local time)."""
    if when is None:
        return "unknown"
    minutes = when.hour * 60 + when.minute
    if minutes < (10 * 60 + 30):
        return "opening_drive"
    if minutes < (12 * 60):
        return "late_morning"
    if minutes < (14 * 60):
        return "midday"
    if minutes < (15 * 60 + 30):
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
) -> dict:
    """Tier + mute decision + human-readable reasons for one confirmed bounce.

    The composite is a weighted mean of the measured avg R of every segment
    this bounce belongs to; unknown segments simply do not contribute, so a
    bounce with no history lands in the neutral B/C range instead of failing.
    """
    segments = (state or {}).get("segments") or {}
    direction = str(direction or "").strip().lower()

    def _lookup(dimension: str, segment: str) -> dict | None:
        if not segment:
            return None
        return (segments.get(dimension) or {}).get(_seg_key(direction, segment))

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
    }
    for dimension, weight in COMPOSITE_DIMENSIONS:
        values = [v for v in dim_values.get(dimension, []) if str(v or "").strip()]
        entries = [(v, _lookup(dimension, v)) for v in values]
        entries = [(v, e) for v, e in entries if e]
        if not entries:
            continue
        dim_avg = sum(e["avg_close_r"] for _v, e in entries) / len(entries)
        weighted_sum += dim_avg * weight
        weight_used += weight
        for value, entry in entries:
            tag = f"{value} {direction} {entry['avg_close_r']:+.2f}R (n={entry['sample_count']})"
            reasons.append(tag)
            if entry.get("muted") and dimension in MUTE_DIMENSIONS:
                mute_reasons.append(f"{dimension}={value}: {entry['avg_close_r']:+.2f}R over {entry['sample_count']} — proven negative")

    composite = (weighted_sum / weight_used) if weight_used > 0 else None
    muted = bool(mute_reasons)
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
    return {
        "tier": tier,
        "composite_r": round(composite, 3) if composite is not None else None,
        "muted": muted,
        "mute_reasons": mute_reasons,
        "reasons": reasons[:4],
    }


# ---------------------------------------------------------------------------
# State IO + refresh
# ---------------------------------------------------------------------------
_state_cache: dict = {"mtime": None, "state": None}


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
