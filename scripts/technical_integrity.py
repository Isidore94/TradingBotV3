"""Advisory Technical Integrity scoring and completed-M5 level-test monitor.

Technical Integrity answers a different question from relative strength:
"Are technical levels earning respect today, or are they easy to break?"

The engine is deliberately decision-support only. It observes every eligible
level test on the bot's scanned symbols, records the prediction made before the
test resolves, resolves it three completed M5 bars later, and publishes a
versioned market/sector/industry/stock hierarchy. Nothing in this module may
change watchlists, setup scores, alerts, or order state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from diagnostics.artifact_io import (
    CAPTURE_MODE_BACKFILL,
    CAPTURE_MODE_LIVE,
    # Re-exported, not used here: `regime_collection_audit` reads it as
    # `from technical_integrity import row_capture_mode`, which is why the
    # 2026-08-31 unused-import sweep put it back after removing it.
    row_capture_mode,  # noqa: F401
)
from durability_retry import fetch_with_bounded_retry
from market_session import get_market_session_window, normalize_market_local_datetime
from project_paths import get_diagnostics_dir, get_local_setting


FEATURE_VERSION = "technical_integrity_v1"
SNAPSHOT_SCHEMA = "technical_integrity_snapshot_v1"
EVENT_SCHEMA = "technical_integrity_event_v1"
STATE_SCHEMA = "technical_integrity_monitor_state_v1"
CALIBRATION_SCHEMA = "technical_integrity_calibration_v1"
COLLECTION_CODE_VERSION = "regime_infrastructure_phase1_v1"
FOLLOWUP_SCHEMA = "technical_integrity_followup_v1"
FROZEN_SNAPSHOT_SCHEMA = "technical_integrity_frozen_snapshot_v1"
OPENING_RANGE_SCHEMA = "technical_integrity_opening_range_v1"
FOLLOWUP_HORIZONS_MINUTES = (30, 60, 90)
FROZEN_SNAPSHOT_GRACE_MINUTES = 5

# Provenance for Tier B recovery (docs/DURABILITY_CATCHUP_PLAN.md sec 2.3).
# A follow-up window is a pure function of completed M5 bars, so it may be
# recomputed after an outage -- but research must be able to separate what the
# live process observed from what was reconstructed afterwards, forever. The
# ``capture_mode`` vocabulary is shared with every other evidence ledger, so it
# lives in artifact_io; these names are re-exported for this module's readers.
TI_CHAIN_BACKFILL_SETTING_KEY = "ti_chain_backfill"

#: Attempts one symbol is entitled to for one session, **across all sweeps**.
#: Spent a couple at a time so no single sweep holds the monitor lock through
#: a long backoff, and carried in the monitor's persisted state so a symbol
#: that ran out of luck at the close still has attempts left the next morning.
#: Six is three sweeps' worth: enough to outlast a provider having a bad
#: evening, few enough that a genuinely absent symbol is settled within a day.
FOLLOWUP_SYMBOL_ATTEMPT_ENTITLEMENT = 6

#: Extra attempts within a single sweep. One retry keeps the in-lock backoff
#: to roughly half a second per failing symbol; the rest of the entitlement is
#: spent by later sweeps.
FOLLOWUP_SWEEP_RETRIES = 1


#: Stamped by ``_append_event`` when a row is written, so they describe that
#: append and never the state a later event should inherit.
_APPEND_TIME_PROVENANCE_FIELDS = frozenset({"as_of", "written_at"})


def ti_chain_backfill_enabled() -> bool:
    """Default-on: a marked, append-only recovery path, not a behaviour change."""
    return bool(get_local_setting(TI_CHAIN_BACKFILL_SETTING_KEY, True))


# Measured share of DECISIVE level tests that end in respect (held/reclaimed
# weighted against broke). Levels mostly hold, so 0.5 was never the neutral
# point: scoring against it put a typical symbol at 6.55 and made "above the
# midpoint" meaningless. Re-measured 2026-07-22 over 1,293 decisive same-day
# resolutions. Reviewed against fresh sessions, not tuned per-session.
TECHNICAL_INTEGRITY_BASE_RESPECT = 0.736


@dataclass(frozen=True)
class TechnicalIntegrityConfig:
    prior_weight: float = 2.0
    prior_respect_probability: float = TECHNICAL_INTEGRITY_BASE_RESPECT
    held_value: float = 1.0
    reclaimed_value: float = 0.65
    chop_value: float = 0.5
    broke_value: float = 0.0
    # "chop" (price finishing inside the break buffer) was 69% of all
    # resolutions, and its value equalled the prior exactly - so it added
    # weight to the denominator and precisely zero information to the
    # numerator. With the prior that was ~75% dead weight, which is why every
    # symbol converged to ~6.15 and the cross-symbol spread measured as pure
    # sampling noise (permutation null 0.657 vs observed 0.639; split-half
    # r=0.09). Chop is now counted and reported, but never scored.
    count_chop_as_evidence: bool = False
    # Score midpoint: the probability that maps to 5.5/10.
    neutral_respect_probability: float = TECHNICAL_INTEGRITY_BASE_RESPECT
    # Below this much decisive (non-chop) evidence weight a score is not
    # measurement, it is the prior echoing back. Such entities report a null
    # score and BUILDING rather than a confident-looking number. Per-symbol
    # rows carry ~1 decisive D1 test, so in practice only the market and
    # sector rollups clear this bar - deliberately.
    min_decisive_weight_for_score: float = 8.0
    resolution_bars: int = 3
    touch_buffer_atr: float = 0.05
    break_buffer_atr: float = 0.10
    # D1 major levels (daily SMAs, D1 trendlines, horizontal S/R) resolve on a
    # longer window and tolerate a wider break buffer: they are coarser levels
    # tested against the same daily-ATR yardstick.
    d1_resolution_bars: int = 6
    d1_break_buffer_atr: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LEVEL_SPECS: tuple[tuple[str, str, float], ...] = (
    ("std_vwap", "vwap", 1.20),
    ("dynamic_vwap", "dynamic_vwap", 1.10),
    ("eod_vwap", "eod_vwap", 1.10),
    ("ema_8", "ema_8", 0.90),
    ("ema_15", "ema_15", 1.00),
    ("ema_21", "ema_21", 1.00),
    ("prev_high", "prev_day_high", 1.25),
    ("prev_low", "prev_day_low", 1.25),
    ("vwap_1stdev_upper", "vwap_upper_band", 0.80),
    ("vwap_1stdev_lower", "vwap_lower_band", 0.80),
    ("dynamic_vwap_1stdev_upper", "dynamic_vwap_upper_band", 0.75),
    ("dynamic_vwap_1stdev_lower", "dynamic_vwap_lower_band", 0.75),
    ("eod_vwap_1stdev_upper", "eod_vwap_upper_band", 0.75),
    ("eod_vwap_1stdev_lower", "eod_vwap_lower_band", 0.75),
)

# D1 major levels arrive through ``observe_symbol(extra_levels=...)`` with a
# family name carrying this prefix. They are fixed prices (unlike the drifting
# VWAP/EMA metrics), so several levels of one family may be under test at once.
D1_FAMILY_PREFIX = "d1_"


def family_timeframe(family: str) -> str:
    return "d1" if str(family or "").startswith(D1_FAMILY_PREFIX) else "intraday"


def _event_timeframe(event: Mapping[str, Any]) -> str:
    explicit = str(event.get("level_timeframe") or "").strip().lower()
    if explicit in {"d1", "intraday"}:
        return explicit
    return family_timeframe(str(event.get("level_family") or ""))


def _candidate_dedupe_key(family: str, level_value: float) -> str:
    # Drifting intraday metrics (VWAP/EMA) dedupe per family so a slowly moving
    # level cannot open a second concurrent test. D1 levels are fixed prices and
    # a symbol can legitimately test two different horizontals in one window, so
    # they dedupe per (family, price).
    if family_timeframe(family) == "d1":
        return f"{family}@{float(level_value):.4f}"
    return str(family)


def technical_integrity_events_path() -> Path:
    return get_diagnostics_dir() / "technical_integrity_events.jsonl"


#: Schema name for the resolved-events sidecar. By NAME, never by number.
RESOLVED_SIDECAR_SCHEMA = "technical_integrity_resolved_v1"


def technical_integrity_resolved_path(events_path: Path | None = None) -> Path:
    """The resolved-events sidecar, beside whatever events log it derives from.

    `technical_integrity_events.jsonl` measured **618 MB** on 2026-08-31, and the
    after-close wrap-up replayed it daily by streaming and `json.loads`-ing every
    line to keep the `level_resolved` rows - a small subset. That is an
    hour-class job inside the desk process, and Python's GIL means an hour of hot
    parsing steals GUI-thread time all evening.

    So the resolved rows are ALSO written here as they happen. The main log is
    untouched - same rows, same path, nothing removed - and this file is a
    derived convenience that can be deleted and rebuilt at any time.
    """
    source = Path(events_path or technical_integrity_events_path())
    return source.with_name(source.stem + "_resolved.jsonl")


def technical_integrity_state_path() -> Path:
    return get_diagnostics_dir() / "technical_integrity_state.json"


def technical_integrity_snapshot_path() -> Path:
    return get_diagnostics_dir() / "technical_integrity_snapshot.json"


def technical_integrity_calibration_path() -> Path:
    return get_diagnostics_dir() / "technical_integrity_calibration.json"


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for pattern in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def _records(rows: Any) -> list[Mapping[str, Any]]:
    if rows is None:
        return []
    if hasattr(rows, "to_dict"):
        try:
            records = rows.to_dict("records")
            if isinstance(records, list):
                return [row for row in records if isinstance(row, Mapping)]
        except Exception:
            return []
    return [row for row in rows if isinstance(row, Mapping)]


def completed_m5_bars(
    rows: Any,
    *,
    now: datetime | None = None,
    session_date: date | None = None,
) -> list[dict[str, Any]]:
    """Normalize valid bars and exclude any M5 candle that can still form.

    ``session_date`` selects a specific session instead of the newest one in
    ``rows``. Live callers leave it unset; the chain sweeper sets it so a
    multi-day fetch can complete *yesterday's* windows after a missed close.
    """
    moment = normalize_market_local_datetime(now)
    complete: list[dict[str, Any]] = []
    for row in _records(rows):
        raw_start = _parse_datetime(row.get("datetime") or row.get("dt") or row.get("time"))
        if raw_start is None:
            continue
        local_start = normalize_market_local_datetime(raw_start)
        local_end = local_start + timedelta(minutes=5)
        if local_end > moment:
            continue
        try:
            open_value = float(row["open"])
            high_value = float(row["high"])
            low_value = float(row["low"])
            close_value = float(row["close"])
            volume_value = float(row.get("volume") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        values = (open_value, high_value, low_value, close_value, volume_value)
        if not all(math.isfinite(value) for value in values) or low_value > high_value:
            continue
        if not low_value <= min(open_value, close_value) <= high_value:
            continue
        if not low_value <= max(open_value, close_value) <= high_value:
            continue
        explicit_start = raw_start if raw_start.tzinfo is not None else local_start
        explicit_end = explicit_start + timedelta(minutes=5)
        complete.append(
            {
                "_start_local": local_start,
                "bar_start": explicit_start.isoformat(timespec="seconds"),
                "bar_end": explicit_end.isoformat(timespec="seconds"),
                "open": open_value,
                "high": high_value,
                "low": low_value,
                "close": close_value,
                "volume": volume_value,
            }
        )
    complete.sort(key=lambda row: row["_start_local"])
    if not complete:
        return []
    target_date = session_date or complete[-1]["_start_local"].date()
    return [row for row in complete if row["_start_local"].date() == target_date]


def _score_from_probability(probability: float, neutral: float) -> float:
    """Map a respect probability onto the 1-10 scale, centred on ``neutral``.

    Piecewise-linear so the endpoints stay 1.0 and 10.0 while the measured
    base rate lands on 5.5. Reading a score is then unambiguous: below 5.5
    means levels are being respected LESS than they typically are, above
    means more. The previous straight ``1 + 9p`` mapping put the typical
    symbol at 6.55, so "6.1" looked mid-range while actually meaning
    below-average respect.
    """
    neutral = min(max(float(neutral), 0.05), 0.95)
    probability = min(max(float(probability), 0.0), 1.0)
    if probability <= neutral:
        return round(1.0 + 4.5 * (probability / neutral), 1)
    return round(5.5 + 4.5 * ((probability - neutral) / (1.0 - neutral)), 1)


def _score_state(score: float | None, test_count: int) -> str:
    if test_count <= 0 or score is None:
        return "BUILDING"
    if score <= 3.0:
        return "VERY WEAK"
    if score <= 4.5:
        return "WEAK"
    if score < 6.5:
        return "MIXED"
    if score < 8.0:
        return "FIRM"
    return "STRONG"


def _outcome_value(outcome: str, config: TechnicalIntegrityConfig) -> float | None:
    return {
        "held": config.held_value,
        "reclaimed": config.reclaimed_value,
        "chop": config.chop_value,
        "broke": config.broke_value,
    }.get(str(outcome or "").strip().lower())


def _integrity_probability(
    events: Iterable[Mapping[str, Any]],
    config: TechnicalIntegrityConfig,
) -> tuple[float, float, int, set[str]]:
    weighted_value = config.prior_weight * config.prior_respect_probability
    total_weight = float(config.prior_weight)
    resolved_weight = 0.0
    test_count = 0
    chop_count = 0
    symbols: set[str] = set()
    for event in events:
        outcome = str(event.get("outcome") or "").strip().lower()
        value = _outcome_value(outcome, config)
        if value is None:
            continue
        try:
            weight = max(0.0, float(event.get("event_weight") or 1.0))
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        symbol = str(event.get("symbol") or "").strip().upper()
        if outcome == "chop":
            # Counted for reporting, never scored: an inconclusive test is
            # not evidence of respect or of failure.
            chop_count += 1
            if symbol:
                symbols.add(symbol)
            if not config.count_chop_as_evidence:
                continue
        weighted_value += weight * value
        total_weight += weight
        resolved_weight += weight
        test_count += 1
        if symbol:
            symbols.add(symbol)
    probability = (
        weighted_value / total_weight
        if total_weight > 0
        else config.prior_respect_probability
    )
    return probability, resolved_weight, test_count, symbols, chop_count


def _pressure(events: Iterable[Mapping[str, Any]]) -> tuple[str, float, float]:
    up_weight = 0.0
    down_weight = 0.0
    for event in events:
        if str(event.get("outcome") or "").lower() != "broke":
            continue
        try:
            weight = max(0.0, float(event.get("event_weight") or 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        direction = str(event.get("break_direction") or "").lower()
        if direction == "up":
            up_weight += weight
        elif direction == "down":
            down_weight += weight
    if up_weight >= 1.0 and up_weight > down_weight * 1.25:
        return "BULLISH", up_weight, down_weight
    if down_weight >= 1.0 and down_weight > up_weight * 1.25:
        return "BEARISH", up_weight, down_weight
    return "BALANCED", up_weight, down_weight


def _confidence(resolved_weight: float, symbol_count: int) -> str:
    if resolved_weight >= 12.0 and symbol_count >= 5:
        return "HIGH"
    if resolved_weight >= 6.0 and symbol_count >= 3:
        return "MEDIUM"
    return "LOW"


def _side_score(
    events: list[Mapping[str, Any]],
    side: str,
    config: TechnicalIntegrityConfig,
) -> float | None:
    side_events = [event for event in events if str(event.get("approach_side") or "") == side]
    if not side_events:
        return None
    probability, weight, count, _symbols, _chop = _integrity_probability(side_events, config)
    if not count or weight < config.min_decisive_weight_for_score:
        return None
    return _score_from_probability(probability, config.neutral_respect_probability)


def _entity_row(
    events: list[Mapping[str, Any]],
    *,
    entity_type: str,
    entity_key: str,
    label: str,
    config: TechnicalIntegrityConfig,
) -> dict[str, Any]:
    probability, resolved_weight, test_count, symbols, chop_count = _integrity_probability(events, config)
    # Thin decisive evidence reports no score rather than the prior wearing a
    # decimal point - see min_decisive_weight_for_score.
    has_score = test_count > 0 and resolved_weight >= config.min_decisive_weight_for_score
    score = _score_from_probability(probability, config.neutral_respect_probability) if has_score else None
    pressure, break_up_weight, break_down_weight = _pressure(events)
    state = _score_state(score, test_count)
    confidence = _confidence(resolved_weight, len(symbols))
    row = {
        "entity_type": entity_type,
        "entity_key": entity_key,
        "label": label,
        "score": score,
        "respect_probability": round(probability, 4),
        "state": state,
        "pressure": pressure,
        "confidence": confidence,
        "test_count": test_count,
        "chop_count": chop_count,
        "resolved_weight": round(resolved_weight, 3),
        "symbol_count": len(symbols),
        "support_integrity": _side_score(events, "above", config),
        "resistance_integrity": _side_score(events, "below", config),
        "break_up_weight": round(break_up_weight, 3),
        "break_down_weight": round(break_down_weight, 3),
    }
    # D1 major levels (daily SMAs, D1 trendlines, horizontal S/R) are the
    # trader-priority read; intraday VWAP/EMA/band tests stay tracked but are
    # reported separately so the D1 verdict is never diluted by M5 noise.
    for timeframe, prefix in (("d1", "d1_"), ("intraday", "intraday_")):
        subset = [event for event in events if _event_timeframe(event) == timeframe]
        sub_probability, sub_weight, sub_count, sub_symbols, sub_chop = _integrity_probability(subset, config)
        sub_has_score = sub_count > 0 and sub_weight >= config.min_decisive_weight_for_score
        sub_score = (
            _score_from_probability(sub_probability, config.neutral_respect_probability)
            if sub_has_score
            else None
        )
        sub_pressure, _up, _down = _pressure(subset)
        row.update(
            {
                f"{prefix}score": sub_score,
                f"{prefix}state": _score_state(sub_score, sub_count),
                f"{prefix}pressure": sub_pressure,
                f"{prefix}confidence": _confidence(sub_weight, len(sub_symbols)),
                f"{prefix}test_count": sub_count,
                f"{prefix}chop_count": sub_chop,
                f"{prefix}symbol_count": len(sub_symbols),
            }
        )
        if timeframe == "d1":
            row["d1_support_integrity"] = _side_score(subset, "above", config)
            row["d1_resistance_integrity"] = _side_score(subset, "below", config)
    return row


def _latest_environment(events: list[Mapping[str, Any]]) -> str:
    ordered = sorted(
        events,
        key=lambda event: str(event.get("resolved_at") or event.get("as_of") or ""),
    )
    for event in reversed(ordered):
        value = str(event.get("market_environment") or "").strip()
        if value:
            return value
    return ""


def aggregate_technical_integrity(
    resolved_events: Iterable[Mapping[str, Any]],
    *,
    as_of: datetime | str | None = None,
    session_date: str = "",
    pending_count: int = 0,
    config: TechnicalIntegrityConfig | None = None,
) -> dict[str, Any]:
    """Build a stable stock -> industry -> sector -> market score hierarchy."""
    active_config = config or TechnicalIntegrityConfig()
    events = [
        dict(event)
        for event in resolved_events
        if str(event.get("event_type") or "") == "level_resolved"
        and (not session_date or str(event.get("session_date") or "") == session_date)
    ]
    if isinstance(as_of, datetime) or as_of is None:
        as_of_text = normalize_market_local_datetime(as_of).isoformat(timespec="seconds")
    else:
        as_of_text = str(as_of)
    if not session_date:
        session_date = str(events[-1].get("session_date") or "") if events else as_of_text[:10]

    entities: list[dict[str, Any]] = []
    market = _entity_row(
        events,
        entity_type="market",
        entity_key="MARKET",
        label="Scanned Market",
        config=active_config,
    )
    market["pending_count"] = int(pending_count)
    market["market_environment"] = _latest_environment(events)
    entities.append(market)

    group_specs = (
        ("sector", "sector_key", "sector"),
        ("industry", "industry_key", "industry"),
        ("stock", "symbol", "symbol"),
    )
    for entity_type, key_field, label_field in group_specs:
        grouped: dict[str, list[dict[str, Any]]] = {}
        labels: dict[str, str] = {}
        for event in events:
            raw_key = str(event.get(key_field) or "").strip()
            key = raw_key.upper() if entity_type == "stock" else raw_key.lower()
            if not key:
                continue
            grouped.setdefault(key, []).append(event)
            labels.setdefault(key, str(event.get(label_field) or raw_key).strip() or raw_key)
        for key in sorted(grouped):
            entities.append(
                _entity_row(
                    grouped[key],
                    entity_type=entity_type,
                    entity_key=key,
                    label=labels[key],
                    config=active_config,
                )
            )

    industries = [row for row in entities if row["entity_type"] == "industry"]
    sectors = [row for row in entities if row["entity_type"] == "sector"]
    # Only scored rows can be ranked; an unscored row has no measurement to
    # be strongest or weakest with.
    scored_industries = [row for row in industries if row["score"] is not None]
    scored_sectors = [row for row in sectors if row["score"] is not None]
    weakest_industries = sorted(scored_industries, key=lambda row: (row["score"], -row["test_count"], row["label"]))[:5]
    strongest_industries = sorted(scored_industries, key=lambda row: (-row["score"], -row["test_count"], row["label"]))[:5]
    weakest_sectors = sorted(scored_sectors, key=lambda row: (row["score"], -row["test_count"], row["label"]))[:5]
    strongest_sectors = sorted(scored_sectors, key=lambda row: (-row["score"], -row["test_count"], row["label"]))[:5]

    config_payload = active_config.to_dict()
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    identity = json.dumps(
        {
            "as_of": as_of_text,
            "session_date": session_date,
            "config_hash": config_hash,
            "event_ids": sorted(str(event.get("event_id") or "") for event in events),
            "pending_count": int(pending_count),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema": SNAPSHOT_SCHEMA,
        "feature_version": FEATURE_VERSION,
        "snapshot_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
        "as_of": as_of_text,
        "session_date": session_date,
        "scope": "BounceBot scanned symbols; advisory, not a full-market census",
        "config": config_payload,
        "config_hash": config_hash,
        "market": market,
        "entities": entities,
        "weakest_industries": weakest_industries,
        "strongest_industries": strongest_industries,
        "weakest_sectors": weakest_sectors,
        "strongest_sectors": strongest_sectors,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            try:
                os.remove(staged)
            except OSError:
                pass


#: The parsed snapshot, keyed by path, with the (mtime_ns, size) it came from.
#: Bounded to one entry per snapshot file - there is one.
_SNAPSHOT_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def clear_technical_integrity_snapshot_cache() -> None:
    """Forget the parsed snapshot. For tests and for a forced re-read."""
    _SNAPSHOT_CACHE.clear()


def load_technical_integrity_snapshot(path: Path | None = None) -> dict[str, Any]:
    """The advisory snapshot, parsed once per file version.

    The GUI polls this every 30 seconds and the file is ~453 KB that the scan
    rewrites about once an hour, so roughly 29 of every 30 parses were reading
    the same bytes to the same answer. Keyed on (mtime_ns, size); an
    unstampable file is not cached, so one that appears later is picked up.
    """
    target = Path(path or technical_integrity_snapshot_path())
    if not target.exists():
        return {}
    key = str(target)
    try:
        stat = target.stat()
        signature: tuple[int, int] | None = (int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        signature = None
    if signature is not None:
        cached = _SNAPSHOT_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    result = payload if isinstance(payload, dict) and payload.get("schema") == SNAPSHOT_SCHEMA else {}
    if signature is not None:
        _SNAPSHOT_CACHE[key] = (signature, result)
    return result


def _level_candidates(
    metrics: Mapping[str, Any],
    atr: float,
    extra_levels: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for metric_key, family, weight in LEVEL_SPECS:
        try:
            value = float(metrics.get(metric_key))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value <= 0:
            continue
        candidates.append(
            {
                "metric_key": metric_key,
                "level_family": family,
                "level_value": value,
                "event_weight": float(weight),
            }
        )
    for level in extra_levels or ():
        if not isinstance(level, Mapping):
            continue
        family = str(level.get("family") or "").strip()
        try:
            value = float(level.get("value"))
            weight = float(level.get("weight") or 1.0)
        except (TypeError, ValueError):
            continue
        if not family or not math.isfinite(value) or value <= 0 or weight <= 0:
            continue
        candidate = {
            "metric_key": str(level.get("metric_key") or family),
            "level_family": family,
            "level_value": value,
            "event_weight": weight,
        }
        detail = level.get("detail")
        if isinstance(detail, Mapping) and detail:
            candidate["level_detail"] = dict(detail)
        candidates.append(candidate)
    # Confluent levels should be one test, not three correlated votes.
    selected: list[dict[str, Any]] = []
    cluster_tolerance = max(0.0, float(atr)) * 0.05
    for candidate in sorted(candidates, key=lambda row: (-row["event_weight"], row["level_family"])):
        if any(abs(candidate["level_value"] - prior["level_value"]) <= cluster_tolerance for prior in selected):
            continue
        selected.append(candidate)
    return selected


def _test_id(symbol: str, bar_start: str, family: str, level: float) -> str:
    raw = f"{symbol}|{bar_start}|{family}|{level:.6f}|{FEATURE_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _new_level_tests(
    symbol: str,
    bars: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    atr: float,
    classification: Mapping[str, Any],
    market_environment: str,
    seen_ids: set[str],
    pending_keys: set[str],
    config: TechnicalIntegrityConfig,
    extra_levels: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if len(bars) < 2 or atr <= 0:
        return []
    previous = bars[-2]
    current = bars[-1]
    touch_buffer = config.touch_buffer_atr * atr
    tests: list[dict[str, Any]] = []
    for candidate in _level_candidates(metrics, atr, extra_levels):
        family = candidate["level_family"]
        level = candidate["level_value"]
        if _candidate_dedupe_key(family, level) in pending_keys:
            continue
        if current["low"] > level + touch_buffer or current["high"] < level - touch_buffer:
            continue
        approach_delta = previous["close"] - level
        if abs(approach_delta) <= touch_buffer * 0.5:
            approach_delta = current["open"] - level
        if abs(approach_delta) <= touch_buffer * 0.5:
            continue
        event_id = _test_id(symbol, current["bar_start"], family, level)
        if event_id in seen_ids:
            continue
        timeframe = family_timeframe(family)
        event = {
            "schema": EVENT_SCHEMA,
            "feature_version": FEATURE_VERSION,
            "event_type": "level_test_started",
            "event_id": event_id,
            "session_date": current["_start_local"].date().isoformat(),
            "started_at": current["bar_end"],
            "touch_bar_start": current["bar_start"],
            "touch_bar_start_local": current["_start_local"].isoformat(timespec="seconds"),
            "symbol": symbol,
            "sector_key": str(classification.get("sectorKey") or "").strip().lower(),
            "sector": str(classification.get("sector") or "").strip(),
            "industry_key": str(classification.get("industryKey") or "").strip().lower(),
            "industry": str(classification.get("industry") or "").strip(),
            "market_environment": str(market_environment or ""),
            "level_family": family,
            "level_timeframe": timeframe,
            "level_value": round(level, 6),
            "event_weight": candidate["event_weight"],
            "approach_side": "above" if approach_delta > 0 else "below",
            "atr": float(atr),
            "touch_buffer_atr": config.touch_buffer_atr,
            "break_buffer_atr": (
                config.d1_break_buffer_atr if timeframe == "d1" else config.break_buffer_atr
            ),
            "resolution_bars": (
                config.d1_resolution_bars if timeframe == "d1" else config.resolution_bars
            ),
            "data_health": "ok",
        }
        if candidate.get("level_detail"):
            event["level_detail"] = candidate["level_detail"]
        tests.append(event)
    return tests


def _resolve_pending(
    pending: Mapping[str, Any],
    bars: list[dict[str, Any]],
    config: TechnicalIntegrityConfig,
) -> dict[str, Any] | None:
    touch_start = _parse_datetime(pending.get("touch_bar_start_local"))
    if touch_start is None:
        return None
    touch_local = normalize_market_local_datetime(touch_start)
    subsequent = [bar for bar in bars if bar["_start_local"] > touch_local]
    resolution_bars = max(1, int(pending.get("resolution_bars") or config.resolution_bars))
    if len(subsequent) < resolution_bars:
        return None
    window = subsequent[:resolution_bars]
    try:
        level = float(pending["level_value"])
        atr = float(pending["atr"])
    except (KeyError, TypeError, ValueError):
        return None
    buffer_value = max(0.0, float(pending.get("break_buffer_atr") or config.break_buffer_atr)) * atr
    approach_side = str(pending.get("approach_side") or "")
    closes = [float(bar["close"]) for bar in window]
    final_close = closes[-1]
    if approach_side == "above":
        breached = any(close < level - buffer_value for close in closes)
        if final_close < level - buffer_value:
            outcome = "broke"
            break_direction = "down"
        elif breached:
            outcome = "reclaimed"
            break_direction = "down"
        elif final_close <= level + buffer_value:
            outcome = "chop"
            break_direction = ""
        else:
            outcome = "held"
            break_direction = ""
    elif approach_side == "below":
        breached = any(close > level + buffer_value for close in closes)
        if final_close > level + buffer_value:
            outcome = "broke"
            break_direction = "up"
        elif breached:
            outcome = "reclaimed"
            break_direction = "up"
        elif final_close >= level - buffer_value:
            outcome = "chop"
            break_direction = ""
        else:
            outcome = "held"
            break_direction = ""
    else:
        return None
    follow_through_atr = abs(final_close - level) / atr if atr > 0 else 0.0
    row = dict(pending)
    row.update(
        {
            "event_type": "level_resolved",
            "resolved_at": window[-1]["bar_end"],
            "resolution_bar_start": window[-1]["bar_start"],
            "resolution_close": final_close,
            "outcome": outcome,
            "break_direction": break_direction,
            "follow_through_atr": round(follow_through_atr, 4),
            "actual_intact": 1 if outcome in {"held", "reclaimed"} else (0 if outcome == "broke" else None),
        }
    )
    return row


def _resolution_direction(event: Mapping[str, Any]) -> tuple[str, int, str]:
    outcome = str(event.get("outcome") or "").lower()
    break_direction = str(event.get("break_direction") or "").lower()
    if outcome == "broke" and break_direction in {"up", "down"}:
        return break_direction, (1 if break_direction == "up" else -1), "clean_break"
    approach_side = str(event.get("approach_side") or "").lower()
    if approach_side == "above":
        return "up", 1, "held_or_reclaimed_side" if outcome != "chop" else "approach_side"
    if approach_side == "below":
        return "down", -1, "held_or_reclaimed_side" if outcome != "chop" else "approach_side"
    return "", 0, "unavailable"


def _followup_tracking_event(resolution: Mapping[str, Any]) -> dict[str, Any] | None:
    event_id = str(resolution.get("event_id") or "")
    resolved_at = str(resolution.get("resolved_at") or "")
    direction, direction_sign, direction_basis = _resolution_direction(resolution)
    if not event_id or not resolved_at or not direction_sign:
        return None
    return {
        "schema": FOLLOWUP_SCHEMA,
        "feature_version": FEATURE_VERSION,
        "code_version": COLLECTION_CODE_VERSION,
        "event_type": "post_resolution_tracking_started",
        "event_id": f"{event_id}|followup",
        "followup_id": f"{event_id}|followup",
        "source_resolution_id": event_id,
        "session_date": str(resolution.get("session_date") or ""),
        "symbol": str(resolution.get("symbol") or "").upper(),
        "level_family": str(resolution.get("level_family") or ""),
        "level_timeframe": _event_timeframe(resolution),
        "level_value": float(resolution.get("level_value") or 0.0),
        "atr": float(resolution.get("atr") or 0.0),
        "resolution_outcome": str(resolution.get("outcome") or ""),
        "resolution_direction": direction,
        "direction_sign": direction_sign,
        "direction_basis": direction_basis,
        "resolution_bar_close": resolved_at,
        "completed_horizons": [],
        "as_of": resolved_at,
    }


def _post_resolution_events(
    tracking: Mapping[str, Any],
    bars: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    force_data_gap_reason: str = "",
    capture_mode: str = CAPTURE_MODE_LIVE,
) -> list[dict[str, Any]]:
    resolution_raw = _parse_datetime(tracking.get("resolution_bar_close"))
    if resolution_raw is None:
        return []
    resolution_at = normalize_market_local_datetime(resolution_raw)
    moment = normalize_market_local_datetime(now)
    session = get_market_session_window(resolution_at)
    try:
        level = float(tracking.get("level_value"))
        atr = float(tracking.get("atr"))
        direction_sign = int(tracking.get("direction_sign") or 0)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(level) or not math.isfinite(atr) or atr <= 0 or direction_sign not in {-1, 1}:
        return []
    completed = {
        int(value)
        for value in tracking.get("completed_horizons") or []
        if str(value).isdigit()
    }
    available = [
        bar
        for bar in bars
        if bar["_start_local"] >= session.open_local
        and bar["_start_local"] < session.close_local
        and normalize_market_local_datetime(_parse_datetime(bar["bar_end"])) > resolution_at
    ]
    events: list[dict[str, Any]] = []
    for horizon in FOLLOWUP_HORIZONS_MINUTES:
        if horizon in completed:
            continue
        target_at = resolution_at + timedelta(minutes=horizon)
        truncated = target_at > session.close_local
        window_end = min(target_at, session.close_local)
        if moment < window_end:
            continue
        window = [
            bar
            for bar in available
            if normalize_market_local_datetime(_parse_datetime(bar["bar_end"])) <= window_end
        ]
        expected_count = max(
            0,
            int((window_end - resolution_at).total_seconds() // (5 * 60)),
        )
        actual_count = len(window)
        data_gap = bool(force_data_gap_reason) or actual_count < expected_count
        suffix = str(horizon)
        metrics: dict[str, float | None] = {
            f"displacement_atr_{suffix}": None,
            f"mfe_atr_{suffix}": None,
            f"mae_atr_{suffix}": None,
            f"range_atr_{suffix}": None,
        }
        if window:
            final_close = float(window[-1]["close"])
            highest = max(float(bar["high"]) for bar in window)
            lowest = min(float(bar["low"]) for bar in window)
            displacement = direction_sign * (final_close - level) / atr
            if direction_sign > 0:
                favorable = max(0.0, (highest - level) / atr)
                adverse = max(0.0, (level - lowest) / atr)
            else:
                favorable = max(0.0, (level - lowest) / atr)
                adverse = max(0.0, (highest - level) / atr)
            metrics = {
                f"displacement_atr_{suffix}": round(displacement, 6),
                f"mfe_atr_{suffix}": round(favorable, 6),
                f"mae_atr_{suffix}": round(adverse, 6),
                f"range_atr_{suffix}": round((highest - lowest) / atr, 6),
            }
        # Point-in-time: as_of is the moment this row's content became
        # knowable. With bars, that is the last one that closed. With *no*
        # bars, the absence itself is the content, and an absence is only
        # knowable once the window has run out -- it was stamped with
        # resolution_at, which is up to 90 minutes before anyone could have
        # known (checkpoint review 2026-08-08 second review). window_end is
        # that moment: the horizon target, clamped to the close for a
        # truncated horizon whose window genuinely ends there. It is also
        # exactly the gate this loop already waits for above.
        as_of = (
            window[-1]["bar_end"]
            if window
            else window_end.isoformat(timespec="seconds")
        )
        followup_id = str(tracking.get("followup_id") or tracking.get("event_id") or "")
        events.append(
            {
                "schema": FOLLOWUP_SCHEMA,
                "feature_version": FEATURE_VERSION,
                "code_version": COLLECTION_CODE_VERSION,
                "event_type": "post_resolution_followup",
                "event_id": f"{followup_id}|{horizon}",
                "followup_id": followup_id,
                "source_resolution_id": str(tracking.get("source_resolution_id") or ""),
                "session_date": str(tracking.get("session_date") or ""),
                "symbol": str(tracking.get("symbol") or "").upper(),
                "level_family": str(tracking.get("level_family") or ""),
                "level_timeframe": str(tracking.get("level_timeframe") or ""),
                "level_value": level,
                "atr": atr,
                "resolution_outcome": str(tracking.get("resolution_outcome") or ""),
                "resolution_direction": str(tracking.get("resolution_direction") or ""),
                "direction_basis": str(tracking.get("direction_basis") or ""),
                "resolution_bar_close": resolution_at.isoformat(timespec="seconds"),
                "horizon_minutes": horizon,
                "window_target_at": target_at.isoformat(timespec="seconds"),
                "actual_bar_count": actual_count,
                "expected_bar_count": expected_count,
                "truncated": truncated,
                "capture_mode": str(capture_mode or CAPTURE_MODE_LIVE),
                "data_gap": data_gap,
                "data_gap_reason": (
                    str(force_data_gap_reason)
                    if force_data_gap_reason
                    else ("missing_completed_m5_bars" if data_gap else "")
                ),
                **metrics,
                "as_of": as_of,
            }
        )
    return events


def build_opening_range_baseline(
    spy_rows: Any,
    *,
    spy_daily_atr: float | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Point-in-time first-hour SPY range baseline from completed M5 bars."""

    moment = normalize_market_local_datetime(now)
    session = get_market_session_window(moment)
    first_hour_end = session.open_local + timedelta(hours=1)
    bars = [
        bar
        for bar in completed_m5_bars(spy_rows, now=moment)
        if session.open_local <= bar["_start_local"] < first_hour_end
        and normalize_market_local_datetime(_parse_datetime(bar["bar_end"])) <= first_hour_end
    ]
    try:
        atr = float(spy_daily_atr)
    except (TypeError, ValueError):
        atr = 0.0
    data_gap = len(bars) != 12 or not math.isfinite(atr) or atr <= 0
    range_value = None
    if bars and math.isfinite(atr) and atr > 0:
        range_value = round(
            (max(float(bar["high"]) for bar in bars) - min(float(bar["low"]) for bar in bars))
            / atr,
            6,
        )
    return {
        "schema": OPENING_RANGE_SCHEMA,
        "feature_version": FEATURE_VERSION,
        "code_version": COLLECTION_CODE_VERSION,
        "event_type": "opening_range_baseline",
        "session_date": session.market_date.isoformat(),
        "target_metric": "SPY first-hour range / SPY daily ATR",
        "spy_first_hour_range_atr": range_value,
        "spy_daily_atr": atr if atr > 0 and math.isfinite(atr) else None,
        "actual_bar_count": len(bars),
        "expected_bar_count": 12,
        "data_gap": data_gap,
        "scanned_market_composite_range_atr": None,
        "composite_note": "Not collected in Phase 1; no canonical fixed-weight scanned-market price composite exists.",
        "as_of": (
            bars[-1]["bar_end"]
            if bars
            else session.open_local.isoformat(timespec="seconds")
        ),
    }


def _family_resolution_mix(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_timeframe = {"d1": 0, "m5": 0}
    by_family: dict[str, dict[str, Any]] = {}
    for event in events:
        timeframe = "d1" if _event_timeframe(event) == "d1" else "m5"
        by_timeframe[timeframe] += 1
        family = str(event.get("level_family") or "unknown")
        key = f"{timeframe}:{family}"
        row = by_family.setdefault(
            key,
            {
                "timeframe": timeframe,
                "level_family": family,
                "resolved_count": 0,
                "decisive_count": 0,
                "chop_count": 0,
            },
        )
        row["resolved_count"] += 1
        if str(event.get("outcome") or "") == "chop":
            row["chop_count"] += 1
        else:
            row["decisive_count"] += 1
    return {
        "by_timeframe": by_timeframe,
        "by_family": [by_family[key] for key in sorted(by_family)],
    }


def _prediction_for_test(snapshot: Mapping[str, Any], event: Mapping[str, Any]) -> tuple[float, str]:
    entities = snapshot.get("entities") if isinstance(snapshot, Mapping) else []
    entities = entities if isinstance(entities, list) else []
    candidates = (
        ("stock", str(event.get("symbol") or "").upper()),
        ("industry", str(event.get("industry_key") or "").lower()),
        ("sector", str(event.get("sector_key") or "").lower()),
        ("market", "MARKET"),
    )
    for entity_type, key in candidates:
        if not key:
            continue
        row = next(
            (
                item
                for item in entities
                if item.get("entity_type") == entity_type and item.get("entity_key") == key
            ),
            None,
        )
        if row and float(row.get("resolved_weight") or 0.0) >= 3.0:
            return float(row.get("respect_probability") or 0.5), f"{entity_type}:{key}"
    return 0.5, "prior"


def _config_hash(config: TechnicalIntegrityConfig) -> str:
    return hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class TechnicalIntegrityMonitor:
    """Stateful append-only monitor; safe to call repeatedly on the same bars."""

    def __init__(
        self,
        *,
        events_path: Path | None = None,
        state_path: Path | None = None,
        snapshot_path: Path | None = None,
        config: TechnicalIntegrityConfig | None = None,
        collector_started_at: datetime | None = None,
    ) -> None:
        self.events_path = Path(events_path or technical_integrity_events_path())
        self.state_path = Path(state_path or technical_integrity_state_path())
        self.snapshot_path = Path(snapshot_path or technical_integrity_snapshot_path())
        self.config = config or TechnicalIntegrityConfig()
        self.collector_started_at = normalize_market_local_datetime(collector_started_at)
        self._lock = threading.RLock()
        self.session_date = ""
        self.pending: dict[str, dict[str, Any]] = {}
        self.seen_test_ids: set[str] = set()
        self.resolved_events: list[dict[str, Any]] = []
        self.pending_followups: dict[str, dict[str, Any]] = {}
        self.followup_event_ids: set[str] = set()
        self.frozen_snapshot_markers: set[str] = set()
        self.followup_sweep_markers: set[str] = set()
        #: "{session}|{symbol}" -> attempts already spent this session.
        #: Persisted, because the entitlement spans sweeps: a symbol that
        #: ran out of luck at the close still has attempts the next morning.
        self.followup_symbol_attempts: dict[str, int] = {}
        self.latest_completed_bar_end = ""
        self._load_state()

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    @property
    def followup_symbols(self) -> set[str]:
        return {
            str(row.get("symbol") or "").upper()
            for row in self.pending_followups.values()
            if str(row.get("symbol") or "").strip()
        }

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("schema") != STATE_SCHEMA:
            return
        self.session_date = str(payload.get("session_date") or "")
        self.pending = {
            str(key): dict(value)
            for key, value in (payload.get("pending") or {}).items()
            if isinstance(value, dict)
        }
        self.seen_test_ids = {str(value) for value in payload.get("seen_test_ids") or [] if str(value)}
        self.pending_followups = {
            str(key): dict(value)
            for key, value in (payload.get("pending_followups") or {}).items()
            if isinstance(value, dict)
        }
        self.followup_event_ids = {
            str(value)
            for value in payload.get("followup_event_ids") or []
            if str(value)
        }
        self.frozen_snapshot_markers = {
            str(value)
            for value in payload.get("frozen_snapshot_markers") or []
            if str(value)
        }
        self.followup_sweep_markers = {
            str(value)
            for value in payload.get("followup_sweep_markers") or []
            if str(value)
        }
        self.followup_symbol_attempts = {
            str(key): int(value)
            for key, value in (payload.get("followup_symbol_attempts") or {}).items()
            if str(key) and str(value).lstrip("-").isdigit()
        }
        self.latest_completed_bar_end = str(payload.get("latest_completed_bar_end") or "")
        self._load_resolved_events()

    def _load_resolved_events(self) -> None:
        self.resolved_events = []
        if not self.session_date or not self.events_path.exists():
            return
        started: dict[str, dict[str, Any]] = {}
        resolved: dict[str, dict[str, Any]] = {}
        followup_started: dict[str, dict[str, Any]] = {}
        completed_horizons: dict[str, set[int]] = {}
        try:
            with self.events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict) or str(row.get("session_date") or "") != self.session_date:
                        continue
                    event_type = str(row.get("event_type") or "")
                    event_id = str(row.get("event_id") or "")
                    if event_type == "level_test_started":
                        if not event_id:
                            continue
                        self.seen_test_ids.add(event_id)
                        started[event_id] = row
                    elif event_type == "level_resolved":
                        if not event_id:
                            continue
                        self.seen_test_ids.add(event_id)
                        resolved[event_id] = row
                    elif event_type == "post_resolution_tracking_started":
                        followup_id = str(row.get("followup_id") or "")
                        if followup_id:
                            followup_started[followup_id] = row
                    elif event_type == "post_resolution_followup":
                        followup_id = str(row.get("followup_id") or "")
                        try:
                            horizon = int(row.get("horizon_minutes"))
                        except (TypeError, ValueError):
                            horizon = 0
                        if followup_id and horizon:
                            completed_horizons.setdefault(followup_id, set()).add(horizon)
                        if event_id:
                            self.followup_event_ids.add(event_id)
                    elif event_type in {
                        "frozen_intraday_snapshot",
                        "missed_snapshot",
                        "opening_range_baseline",
                        "missed_opening_range_baseline",
                    }:
                        marker = str(row.get("snapshot_key") or "")
                        if marker:
                            self.frozen_snapshot_markers.add(marker)
                    as_of = str(row.get("as_of") or "")
                    if as_of > self.latest_completed_bar_end:
                        self.latest_completed_bar_end = as_of
        except OSError:
            self.resolved_events = []
            return
        self.resolved_events = sorted(
            resolved.values(),
            key=lambda row: (str(row.get("resolved_at") or ""), str(row.get("event_id") or "")),
        )
        # The append-only ledger repairs a crash between event append and the
        # atomic state write. Resolved IDs suppress stale pending state.
        #
        # _append_event stamps as_of/written_at onto the row it writes, so a
        # ledger row carries the *started* event's provenance. _resolve_pending
        # copies the pending dict wholesale, so recovering it verbatim gave the
        # later resolution the touch time as its as_of - a restart between
        # touch and resolution produced a different row than an uninterrupted
        # run. Dropping the append-time stamps restores the in-memory shape and
        # lets the resolution stamp itself (pinned by the restart
        # characterization test in tests/test_ti_chain_backfill.py).
        recovered_pending = {
            event_id: {
                key: value
                for key, value in row.items()
                if key not in _APPEND_TIME_PROVENANCE_FIELDS
            }
            for event_id, row in started.items()
            if event_id not in resolved
        }
        recovered_pending.update(
            {
                event_id: row
                for event_id, row in self.pending.items()
                if event_id not in resolved and event_id not in recovered_pending
            }
        )
        self.pending = recovered_pending
        recovered_followups: dict[str, dict[str, Any]] = {}
        for followup_id, row in followup_started.items():
            candidate = dict(row)
            complete = completed_horizons.get(followup_id, set())
            candidate["completed_horizons"] = sorted(complete)
            if complete != set(FOLLOWUP_HORIZONS_MINUTES):
                recovered_followups[followup_id] = candidate
        for followup_id, row in self.pending_followups.items():
            if followup_id in recovered_followups:
                continue
            candidate = dict(row)
            complete = {
                int(value)
                for value in candidate.get("completed_horizons") or []
                if str(value).isdigit()
            }
            complete.update(completed_horizons.get(followup_id, set()))
            candidate["completed_horizons"] = sorted(complete)
            if complete != set(FOLLOWUP_HORIZONS_MINUTES):
                recovered_followups[followup_id] = candidate
        self.pending_followups = recovered_followups

    def _ensure_session(self, session_date: str) -> None:
        if self.session_date == session_date:
            return
        self.session_date = session_date
        self.pending = {}
        self.seen_test_ids = set()
        self.pending_followups = {}
        self.followup_event_ids = set()
        self.frozen_snapshot_markers = set()
        self.followup_sweep_markers = set()
        self.latest_completed_bar_end = ""
        self._load_resolved_events()

    def _append_event(self, row: Mapping[str, Any]) -> None:
        payload = dict(row)
        payload.setdefault("code_version", COLLECTION_CODE_VERSION)
        payload.setdefault(
            "as_of",
            str(
                payload.get("resolved_at")
                or payload.get("started_at")
                or payload.get("resolution_bar_close")
                or ""
            ),
        )
        payload.setdefault(
            "written_at",
            normalize_market_local_datetime().isoformat(timespec="seconds"),
        )
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            offset = handle.tell()
        # The resolved rows are mirrored to a sidecar as they happen, so the
        # daily replay never has to stream 618 MB to find them. The main log
        # above is the authority and is written FIRST; this is derived, its
        # failure is swallowed, and a missed line only costs a catch-up scan.
        if str(payload.get("event_type") or "") == "level_resolved":
            append_resolved_sidecar_row(
                technical_integrity_resolved_path(self.events_path), payload, offset
            )

    def _save_state(self) -> None:
        _atomic_write_json(
            self.state_path,
            {
                "schema": STATE_SCHEMA,
                "feature_version": FEATURE_VERSION,
                "session_date": self.session_date,
                "pending": self.pending,
                "seen_test_ids": sorted(self.seen_test_ids),
                "pending_followups": self.pending_followups,
                "followup_event_ids": sorted(self.followup_event_ids),
                "frozen_snapshot_markers": sorted(self.frozen_snapshot_markers),
                "followup_sweep_markers": sorted(self.followup_sweep_markers),
                "followup_symbol_attempts": dict(sorted(self.followup_symbol_attempts.items())),
                "latest_completed_bar_end": self.latest_completed_bar_end,
                "code_version": COLLECTION_CODE_VERSION,
                "updated_at": normalize_market_local_datetime().isoformat(timespec="seconds"),
            },
        )

    def _publish_snapshot(self, as_of: str, market_environment: str) -> dict[str, Any]:
        snapshot = aggregate_technical_integrity(
            self.resolved_events,
            as_of=as_of,
            session_date=self.session_date,
            pending_count=len(self.pending),
            config=self.config,
        )
        if market_environment and not snapshot["market"].get("market_environment"):
            snapshot["market"]["market_environment"] = market_environment
        _atomic_write_json(self.snapshot_path, snapshot)
        return snapshot

    def _start_followup(self, resolution: Mapping[str, Any]) -> bool:
        tracking = _followup_tracking_event(resolution)
        if tracking is None:
            return False
        followup_id = str(tracking["followup_id"])
        if followup_id in self.pending_followups:
            return False
        self._append_event(tracking)
        self.pending_followups[followup_id] = dict(tracking)
        return True

    def _process_followups(
        self,
        symbol: str,
        bars: list[dict[str, Any]],
        *,
        now: datetime | None = None,
        force_data_gap_reason: str = "",
        capture_mode: str = CAPTURE_MODE_LIVE,
    ) -> int:
        sym = str(symbol or "").upper()
        appended = 0
        for followup_id, tracking in list(self.pending_followups.items()):
            if str(tracking.get("symbol") or "").upper() != sym:
                continue
            events = _post_resolution_events(
                tracking,
                bars,
                now=now,
                force_data_gap_reason=force_data_gap_reason,
                capture_mode=capture_mode,
            )
            completed = {
                int(value)
                for value in tracking.get("completed_horizons") or []
                if str(value).isdigit()
            }
            for event in events:
                event_id = str(event.get("event_id") or "")
                horizon = int(event.get("horizon_minutes") or 0)
                if not event_id or event_id in self.followup_event_ids or not horizon:
                    continue
                self._append_event(event)
                self.followup_event_ids.add(event_id)
                completed.add(horizon)
                appended += 1
            if completed == set(FOLLOWUP_HORIZONS_MINUTES):
                self.pending_followups.pop(followup_id, None)
            else:
                updated = dict(tracking)
                updated["completed_horizons"] = sorted(completed)
                self.pending_followups[followup_id] = updated
        return appended

    def observe_followups(
        self,
        symbol: str,
        rows: Any,
        *,
        now: datetime | None = None,
    ) -> int:
        """Advance existing windows without discovering any new level tests."""

        sym = str(symbol or "").strip().upper()
        bars = completed_m5_bars(rows, now=now)
        if not sym or not bars:
            return 0
        with self._lock:
            self._ensure_session(bars[-1]["_start_local"].date().isoformat())
            self.latest_completed_bar_end = max(
                self.latest_completed_bar_end,
                str(bars[-1]["bar_end"]),
            )
            appended = self._process_followups(sym, bars, now=now)
            if appended:
                self._save_state()
            return appended

    def mark_followup_data_gap(
        self,
        symbol: str,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> int:
        """Finalize only due horizons with an explicit missing-data marker."""

        with self._lock:
            appended = self._process_followups(
                symbol,
                [],
                now=now,
                force_data_gap_reason=str(reason or "completed M5 data unavailable"),
            )
            if appended:
                self._save_state()
            return appended

    def followup_sweep_trigger(self, *, now: datetime | None = None) -> str:
        """Which sweep is due right now, or "" for none.

        Two moments matter, both from docs/DURABILITY_CATCHUP_PLAN.md sec 2.3:
        the close (windows that ran past it are now decidable) and startup on a
        later day (the close was missed entirely). Each fires at most once per
        session; the marker is persisted, so a restart does not re-spend IB
        requests on chains that already got their honest answer.

        Note the startup trigger only sees yesterday's chains while the monitor
        is still on yesterday's session -- once today's first completed bar
        arrives the monitor rolls over, as it always has. Running the sweep as
        the evidence clock's first action is what keeps that window open.
        """
        if not self.pending_followups or not self.session_date:
            return ""
        moment = normalize_market_local_datetime(now)
        window = get_market_session_window(moment)
        trigger = ""
        if self.session_date != window.market_date.isoformat():
            trigger = "startup_after_missed_close"
        elif moment >= window.close_local:
            trigger = "close_of_day"
        if not trigger:
            return ""
        if f"{self.session_date}|{trigger}" in self.followup_sweep_markers:
            return ""
        return trigger

    def _expected_followup_bars(self, symbol: str, *, now: datetime | None = None) -> int:
        """How many completed M5 bars this symbol's pending chains still need.

        This is the completeness check the sweep hands to the retry helper.
        Without it, a provider returning half a window looked exactly like a
        provider returning everything there was, and the short window was
        recorded as fact (Sol 5.6 verification review, item 6).

        Deliberately a *lower bound*: the longest matured window still owed for
        this symbol, clamped to the session close. Demanding more than a window
        can hold would make every fetch look incomplete and burn the symbol's
        entitlement on nothing.
        """
        target = str(symbol or "").upper()
        moment = normalize_market_local_datetime(now)
        needed = 0
        for tracking in self.pending_followups.values():
            if str(tracking.get("symbol") or "").upper() != target:
                continue
            resolution_raw = _parse_datetime(tracking.get("resolution_bar_close"))
            if resolution_raw is None:
                continue
            resolution_at = normalize_market_local_datetime(resolution_raw)
            session = get_market_session_window(resolution_at)
            completed = {
                int(value)
                for value in tracking.get("completed_horizons") or []
                if str(value).isdigit()
            }
            for horizon in FOLLOWUP_HORIZONS_MINUTES:
                if horizon in completed:
                    continue
                window_end = min(
                    resolution_at + timedelta(minutes=horizon), session.close_local
                )
                if moment < window_end:
                    continue  # not matured yet, so nothing is owed for it
                needed = max(
                    needed, int((window_end - resolution_at).total_seconds() // (5 * 60))
                )
        return max(0, needed)

    def sweep_incomplete_followups(
        self,
        fetch_bars: Callable[[str, str], Any],
        *,
        now: datetime | None = None,
        reason: str = "chain sweeper",
        trigger: str = "",
        retries: int = FOLLOWUP_SWEEP_RETRIES,
        entitlement: int = FOLLOWUP_SYMBOL_ATTEMPT_ENTITLEMENT,
        sleep: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """Complete every due follow-up horizon left hanging by an outage.

        A +30/60/90 window is a pure function of completed M5 bars, so it is
        recoverable (Tier B) -- unlike a frozen snapshot, whose value is what
        the live hierarchy said at a wall-clock moment and which stays missed
        (Tier C, unchanged). Rows written here carry
        ``capture_mode: "backfill"``; bars that genuinely cannot be fetched
        still produce the existing explicit ``data_gap`` rows, so the audit
        keeps counting them honestly instead of silently inventing evidence.

        ``fetch_bars(symbol, session_date)`` returns raw M5 rows and may return
        nothing; the sweeper never raises, because a failed sweep must leave
        the ledger exactly as it found it.

        A failed fetch is retried before its gap is written, and the retry
        entitlement is **per symbol and spans sweeps** (Sol 5.6 verification
        review, item 6). The earlier design spent a single wall-clock sleep
        budget shared across all symbols, so on a bad night the first few
        symbols consumed it and every symbol after them got one attempt and a
        permanent gap -- the shared budget rationed the wrong thing.

        Each symbol now has :data:`FOLLOWUP_SYMBOL_ATTEMPT_ENTITLEMENT`
        attempts for the session, spent a couple at a time per sweep and
        carried in the monitor's own state. A symbol whose entitlement is not
        yet spent is **deferred**: it stays pending, no gap row is written, and
        the next sweep tries again. Only an exhausted entitlement produces the
        permanent gap.

        Consequently the per-session sweep marker is written **only when no
        symbol is deferred**. The marker is what stops a session from ever
        being swept again, and writing it while work remains is what made a
        transient outage permanent in the first place.

        Partial responses count as failures too: the completeness check below
        compares bars received against bars the window should hold, so a
        provider under load returning half a window is retried rather than
        recorded as a short window nobody can distinguish from a real one.

        This is deliberately *not* ``observe_followups``: that call rolls the
        monitor onto the session of the bars it is handed, which on a next-day
        sweep would discard the very pending chains being recovered.
        """
        summary: dict[str, Any] = {
            "ran": False,
            "trigger": str(trigger or ""),
            "session_date": "",
            "symbols": [],
            "chains": 0,
            "events": 0,
            "data_gap_symbols": [],
            "deferred_symbols": [],
            "retry_attempts": {},
            "marker_written": False,
            "reason": "",
        }
        if not ti_chain_backfill_enabled():
            summary["reason"] = f"disabled by {TI_CHAIN_BACKFILL_SETTING_KEY} setting"
            return summary
        with self._lock:
            session_date = self.session_date
            if not self.pending_followups:
                summary["reason"] = "no incomplete follow-up chains"
                return summary
            summary["session_date"] = session_date
            summary["chains"] = len(self.pending_followups)
            symbols = sorted(self.followup_symbols)
            summary["symbols"] = symbols

            try:
                target_session = date.fromisoformat(session_date)
            except ValueError:
                target_session = None

            appended = 0
            data_gap_symbols: list[str] = []
            deferred_symbols: list[str] = []
            retry_attempts: dict[str, int] = {}
            retry_kwargs: dict[str, Any] = {}
            if sleep is not None:
                retry_kwargs["sleep"] = sleep
            for symbol in symbols:
                spent = int(self.followup_symbol_attempts.get(f"{session_date}|{symbol}", 0))
                remaining = max(0, entitlement - spent)
                if remaining <= 0:
                    # Entitlement spent across earlier sweeps: this is the
                    # point at which absence becomes a finding rather than a
                    # failure to look.
                    outcome = None
                    attempts = 0
                    bars: list[dict[str, Any]] = []
                else:
                    expected = self._expected_followup_bars(symbol, now=now)
                    outcome = fetch_with_bounded_retry(
                        lambda symbol=symbol: fetch_bars(symbol, session_date),
                        label=f"Follow-up chain sweep fetch for {symbol} on {session_date}",
                        retries=max(0, min(retries, remaining - 1)),
                        is_complete=(
                            (
                                lambda rows, expected=expected, symbol=symbol: len(
                                    completed_m5_bars(
                                        rows, now=now, session_date=target_session
                                    )
                                )
                                >= expected
                            )
                            if expected
                            else None
                        ),
                        **retry_kwargs,
                    )
                    attempts = outcome.attempts
                    self.followup_symbol_attempts[f"{session_date}|{symbol}"] = spent + attempts
                    remaining -= attempts
                    if attempts > 1:
                        retry_attempts[symbol] = attempts
                    bars = (
                        completed_m5_bars(outcome.value, now=now, session_date=target_session)
                        if outcome.value
                        else []
                    )

                if bars:
                    appended += self._process_followups(
                        symbol,
                        bars,
                        now=now,
                        capture_mode=CAPTURE_MODE_BACKFILL,
                    )
                elif remaining > 0:
                    # Deferred, not gapped: the chain stays pending and the
                    # next sweep spends the rest of this symbol's entitlement.
                    deferred_symbols.append(symbol)
                    logging.info(
                        "Follow-up chain sweep deferring %s on %s: %d attempt(s) of its "
                        "entitlement remain, so no permanent gap is written yet.",
                        symbol,
                        session_date,
                        remaining,
                    )
                else:
                    data_gap_symbols.append(symbol)
                    total_attempts = int(
                        self.followup_symbol_attempts.get(f"{session_date}|{symbol}", attempts)
                    )
                    appended += self._process_followups(
                        symbol,
                        [],
                        now=now,
                        force_data_gap_reason=(
                            f"{reason}: no completed M5 bars available for "
                            f"{symbol} on {session_date} after "
                            f"{total_attempts} attempt(s) across sweeps"
                        ),
                        capture_mode=CAPTURE_MODE_BACKFILL,
                    )

            # The marker is permanent, so it may only be written once there is
            # genuinely nothing left to come back for.
            marker_written = bool(trigger) and not deferred_symbols
            if marker_written:
                self.followup_sweep_markers.add(f"{session_date}|{trigger}")
            self._save_state()
            summary["ran"] = True
            summary["trigger"] = str(trigger or "")
            summary["events"] = appended
            summary["data_gap_symbols"] = data_gap_symbols
            summary["deferred_symbols"] = deferred_symbols
            summary["retry_attempts"] = retry_attempts
            summary["marker_written"] = marker_written
            retried = f"; retried {len(retry_attempts)} symbol(s)" if retry_attempts else ""
            deferred_note = (
                f"; deferred {len(deferred_symbols)} symbol(s) for a later sweep"
                if deferred_symbols
                else ""
            )
            summary["reason"] = (
                f"appended {appended} backfilled follow-up row(s) across "
                f"{len(symbols)} symbol(s) for {session_date}{retried}{deferred_note}"
            )
            return summary

    @staticmethod
    def _frozen_targets(now: datetime | None = None) -> tuple[tuple[str, str, datetime], ...]:
        session = get_market_session_window(normalize_market_local_datetime(now))
        return (
            ("10:30", "open_plus_60m", session.open_local + timedelta(minutes=60)),
            ("12:00", "open_plus_150m", session.open_local + timedelta(minutes=150)),
        )

    def needs_opening_range_baseline(self, *, now: datetime | None = None) -> bool:
        moment = normalize_market_local_datetime(now)
        session = get_market_session_window(moment)
        opening_key = f"{session.market_date.isoformat()}|opening_range"
        _label, _name, target = self._frozen_targets(moment)[0]
        return (
            target <= moment < target + timedelta(minutes=FROZEN_SNAPSHOT_GRACE_MINUTES)
            and opening_key not in self.frozen_snapshot_markers
        )

    def capture_frozen_snapshots(
        self,
        *,
        now: datetime | None = None,
        opening_range_baseline: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Append due point-in-time snapshots or explicit missed markers."""

        moment = normalize_market_local_datetime(now)
        session = get_market_session_window(moment)
        session_date = session.market_date.isoformat()
        written: list[dict[str, Any]] = []
        with self._lock:
            self._ensure_session(session_date)
            for market_label, target_name, target in self._frozen_targets(moment):
                snapshot_key = f"{session_date}|{market_label}"
                if moment < target:
                    continue
                grace_end = target + timedelta(minutes=FROZEN_SNAPSHOT_GRACE_MINUTES)
                collector_was_live = self.collector_started_at <= target
                if market_label == "10:30":
                    opening_key = f"{session_date}|opening_range"
                    if opening_key not in self.frozen_snapshot_markers:
                        baseline = dict(opening_range_baseline or {})
                        if not collector_was_live or moment >= grace_end or not baseline:
                            baseline = {
                                "schema": OPENING_RANGE_SCHEMA,
                                "feature_version": FEATURE_VERSION,
                                "code_version": COLLECTION_CODE_VERSION,
                                "event_type": (
                                    "missed_opening_range_baseline"
                                    if not collector_was_live or moment >= grace_end
                                    else "opening_range_baseline"
                                ),
                                "session_date": session_date,
                                "spy_first_hour_range_atr": None,
                                "actual_bar_count": 0,
                                "expected_bar_count": 12,
                                "data_gap": True,
                                "data_gap_reason": (
                                    "collector_process_was_not_live_at_the_snapshot_target"
                                    if not collector_was_live
                                    else "snapshot_clock_did_not_run_inside_the_five_minute_capture_window"
                                    if moment >= grace_end
                                    else "SPY bars or daily ATR unavailable at snapshot time"
                                ),
                                "as_of": self.latest_completed_bar_end,
                            }
                        baseline.update(
                            {
                                "event_id": f"snapshot|{opening_key}",
                                "snapshot_key": opening_key,
                                "snapshot_target": "open_plus_60m",
                                "target_market_time": "10:30",
                                "target_at": target.isoformat(timespec="seconds"),
                            }
                        )
                        self._append_event(baseline)
                        self.frozen_snapshot_markers.add(opening_key)
                        written.append(baseline)
                if snapshot_key in self.frozen_snapshot_markers:
                    continue
                if not collector_was_live or moment >= grace_end:
                    missed = {
                        "schema": FROZEN_SNAPSHOT_SCHEMA,
                        "feature_version": FEATURE_VERSION,
                        "code_version": COLLECTION_CODE_VERSION,
                        "event_type": "missed_snapshot",
                        "event_id": f"snapshot|{snapshot_key}|missed",
                        "snapshot_key": snapshot_key,
                        "snapshot_target": target_name,
                        "target_market_time": market_label,
                        "target_at": target.isoformat(timespec="seconds"),
                        "session_date": session_date,
                        "reason": (
                            "collector_process_was_not_live_at_the_snapshot_target"
                            if not collector_was_live
                            else "snapshot_clock_did_not_run_inside_the_five_minute_capture_window"
                        ),
                        "data_gap": True,
                        "as_of": self.latest_completed_bar_end,
                    }
                    self._append_event(missed)
                    self.frozen_snapshot_markers.add(snapshot_key)
                    written.append(missed)
                    continue

                aggregate = aggregate_technical_integrity(
                    self.resolved_events,
                    as_of=self.latest_completed_bar_end or target.isoformat(timespec="seconds"),
                    session_date=session_date,
                    pending_count=len(self.pending),
                    config=self.config,
                )
                market = aggregate["market"]
                d1_decisive = int(market.get("d1_test_count") or 0)
                d1_chop = int(market.get("d1_chop_count") or 0)
                m5_decisive = int(market.get("intraday_test_count") or 0)
                m5_chop = int(market.get("intraday_chop_count") or 0)
                unique_symbols = {
                    str(event.get("symbol") or "").upper()
                    for event in self.resolved_events
                    if str(event.get("symbol") or "").strip()
                }
                frozen = {
                    "schema": FROZEN_SNAPSHOT_SCHEMA,
                    "feature_version": FEATURE_VERSION,
                    "code_version": COLLECTION_CODE_VERSION,
                    "event_type": "frozen_intraday_snapshot",
                    "event_id": f"snapshot|{snapshot_key}",
                    "snapshot_key": snapshot_key,
                    "snapshot_target": target_name,
                    "target_market_time": market_label,
                    "target_at": target.isoformat(timespec="seconds"),
                    "session_date": session_date,
                    "d1_chop_rate": (
                        round(d1_chop / (d1_decisive + d1_chop), 6)
                        if d1_decisive + d1_chop
                        else None
                    ),
                    "m5_chop_rate": (
                        round(m5_chop / (m5_decisive + m5_chop), 6)
                        if m5_decisive + m5_chop
                        else None
                    ),
                    "decisive_count": d1_decisive + m5_decisive,
                    "chop_count": d1_chop + m5_chop,
                    "pending_count": len(self.pending),
                    "unique_symbol_count": len(unique_symbols),
                    "d1_decisive_count": d1_decisive,
                    "d1_chop_count": d1_chop,
                    "m5_decisive_count": m5_decisive,
                    "m5_chop_count": m5_chop,
                    "family_resolution_mix": _family_resolution_mix(self.resolved_events),
                    "break_pressure": {
                        "combined": str(market.get("pressure") or "BALANCED"),
                        "d1": str(market.get("d1_pressure") or "BALANCED"),
                        "m5": str(market.get("intraday_pressure") or "BALANCED"),
                    },
                    "as_of": self.latest_completed_bar_end
                    or target.isoformat(timespec="seconds"),
                }
                self._append_event(frozen)
                self.frozen_snapshot_markers.add(snapshot_key)
                written.append(frozen)
            if written:
                self._save_state()
        return written

    def observe_symbol(
        self,
        symbol: str,
        rows: Any,
        metrics: Mapping[str, Any],
        *,
        atr: float | None,
        classification: Mapping[str, Any] | None = None,
        market_environment: str = "",
        now: datetime | None = None,
        extra_levels: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        sym = str(symbol or "").strip().upper()
        try:
            atr_value = float(atr)
        except (TypeError, ValueError):
            atr_value = 0.0
        bars = completed_m5_bars(rows, now=now)
        if not sym or len(bars) < 2 or not math.isfinite(atr_value) or atr_value <= 0:
            return load_technical_integrity_snapshot(self.snapshot_path)
        classification = classification if isinstance(classification, Mapping) else {}
        session_date = bars[-1]["_start_local"].date().isoformat()

        with self._lock:
            self._ensure_session(session_date)
            self.latest_completed_bar_end = max(
                self.latest_completed_bar_end,
                str(bars[-1]["bar_end"]),
            )
            changed = False
            for event_id, pending in list(self.pending.items()):
                if str(pending.get("symbol") or "").upper() != sym:
                    continue
                resolved = _resolve_pending(pending, bars, self.config)
                if resolved is None:
                    continue
                resolved["followup_tracking_version"] = COLLECTION_CODE_VERSION
                self._append_event(resolved)
                self.resolved_events.append(resolved)
                self.pending.pop(event_id, None)
                self._start_followup(resolved)
                changed = True

            if self._process_followups(sym, bars, now=now):
                changed = True

            pre_snapshot = aggregate_technical_integrity(
                self.resolved_events,
                as_of=bars[-1]["bar_end"],
                session_date=self.session_date,
                pending_count=len(self.pending),
                config=self.config,
            )
            pending_keys = {
                _candidate_dedupe_key(
                    str(row.get("level_family") or ""),
                    float(row.get("level_value") or 0.0),
                )
                for row in self.pending.values()
                if str(row.get("symbol") or "").upper() == sym
            }
            new_tests = _new_level_tests(
                sym,
                bars,
                metrics,
                atr_value,
                classification,
                market_environment,
                self.seen_test_ids,
                pending_keys,
                self.config,
                extra_levels,
            )
            for event in new_tests:
                prediction, source = _prediction_for_test(pre_snapshot, event)
                event["predicted_hold_probability"] = round(prediction, 4)
                event["prediction_source"] = source
                event["score_config"] = self.config.to_dict()
                event["score_config_hash"] = _config_hash(self.config)
                self._append_event(event)
                self.pending[event["event_id"]] = dict(event)
                self.seen_test_ids.add(event["event_id"])
                changed = True

            if changed or not self.snapshot_path.exists():
                self._save_state()
                return self._publish_snapshot(bars[-1]["bar_end"], market_environment)
            return load_technical_integrity_snapshot(self.snapshot_path)


def _actual_intact(event: Mapping[str, Any]) -> int | None:
    outcome = str(event.get("outcome") or "").lower()
    if outcome in {"held", "reclaimed"}:
        return 1
    if outcome == "broke":
        return 0
    return None


def _calibration_bins(predictions: list[tuple[float, int]]) -> list[dict[str, Any]]:
    bins = []
    for low in (0.0, 0.2, 0.4, 0.6, 0.8):
        high = low + 0.2
        rows = [item for item in predictions if low <= item[0] < high or (high >= 1.0 and item[0] == 1.0)]
        if not rows:
            continue
        bins.append(
            {
                "range": f"{low:.1f}-{high:.1f}",
                "count": len(rows),
                "mean_prediction": round(sum(item[0] for item in rows) / len(rows), 4),
                "actual_hold_rate": round(sum(item[1] for item in rows) / len(rows), 4),
            }
        )
    return bins


def evaluate_scoring_config(
    resolved_events: Iterable[Mapping[str, Any]],
    config: TechnicalIntegrityConfig,
) -> dict[str, Any]:
    """Point-in-time replay: predict each event using only prior outcomes."""
    ordered = sorted(
        [dict(event) for event in resolved_events if event.get("event_type") == "level_resolved"],
        key=lambda event: str(event.get("resolved_at") or ""),
    )
    history: list[dict[str, Any]] = []
    active_session = ""
    predictions: list[tuple[float, int]] = []
    for event in ordered:
        session_date = str(event.get("session_date") or "")
        if session_date != active_session:
            history = []
            active_session = session_date
        actual = _actual_intact(event)
        if actual is None:
            history.append(event)
            continue
        snapshot = aggregate_technical_integrity(
            history,
            as_of=str(event.get("resolved_at") or ""),
            session_date=active_session,
            config=config,
        )
        probability, _source = _prediction_for_test(snapshot, event)
        predictions.append((probability, actual))
        history.append(event)
    if not predictions:
        return {
            "event_count": 0,
            "brier_score": None,
            "mean_prediction": None,
            "actual_hold_rate": None,
            "calibration_bins": [],
        }
    brier = sum((prediction - actual) ** 2 for prediction, actual in predictions) / len(predictions)
    return {
        "event_count": len(predictions),
        "brier_score": round(brier, 6),
        "mean_prediction": round(sum(item[0] for item in predictions) / len(predictions), 4),
        "actual_hold_rate": round(sum(item[1] for item in predictions) / len(predictions), 4),
        "calibration_bins": _calibration_bins(predictions),
    }


def evaluate_recorded_predictions(
    resolved_events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score probabilities recorded before resolution; no replay assumptions."""
    predictions: list[tuple[float, int]] = []
    for event in resolved_events:
        actual = _actual_intact(event)
        if actual is None:
            continue
        try:
            probability = float(event.get("predicted_hold_probability"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(probability) and 0.0 <= probability <= 1.0:
            predictions.append((probability, actual))
    if not predictions:
        return {
            "event_count": 0,
            "brier_score": None,
            "mean_prediction": None,
            "actual_hold_rate": None,
            "calibration_bins": [],
        }
    brier = sum((prediction - actual) ** 2 for prediction, actual in predictions) / len(predictions)
    return {
        "event_count": len(predictions),
        "brier_score": round(brier, 6),
        "mean_prediction": round(sum(item[0] for item in predictions) / len(predictions), 4),
        "actual_hold_rate": round(sum(item[1] for item in predictions) / len(predictions), 4),
        "calibration_bins": _calibration_bins(predictions),
    }


def compare_scoring_configs(
    resolved_events: Iterable[Mapping[str, Any]],
    configs: Mapping[str, TechnicalIntegrityConfig],
) -> dict[str, Any]:
    events = [dict(event) for event in resolved_events]
    rows = []
    for name, config in configs.items():
        result = evaluate_scoring_config(events, config)
        rows.append({"name": str(name), "config": config.to_dict(), **result})
    rows.sort(
        key=lambda row: (
            row.get("brier_score") is None,
            float(row.get("brier_score") or 0.0),
            row["name"],
        )
    )
    resolved = [event for event in events if event.get("event_type") == "level_resolved"]
    session_count = len({str(event.get("session_date") or "") for event in resolved})
    intact_count = sum(_actual_intact(event) == 1 for event in resolved)
    break_count = sum(_actual_intact(event) == 0 for event in resolved)
    review_eligible = (
        len(resolved) >= 100
        and session_count >= 5
        and intact_count >= 20
        and break_count >= 20
    )
    return {
        "schema": CALIBRATION_SCHEMA,
        "feature_version": FEATURE_VERSION,
        "generated_at": normalize_market_local_datetime().isoformat(timespec="seconds"),
        "method": "point-in-time, session-reset, stock/industry/sector/market hierarchy replay",
        "event_count": len(resolved),
        "session_count": session_count,
        "intact_count": intact_count,
        "break_count": break_count,
        "recorded_live_predictions": evaluate_recorded_predictions(resolved),
        "review_gate": {
            "eligible": review_eligible,
            "minimum_events": 100,
            "minimum_sessions": 5,
            "minimum_intact": 20,
            "minimum_breaks": 20,
            "note": "Evidence eligibility permits manual review only; this report never changes live configuration.",
        },
        "best_replay_config": rows[0]["name"] if rows and review_eligible else None,
        "configs": rows,
    }


def _sidecar_row_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """The event inside one sidecar line, or None for the header line."""
    if not isinstance(row, Mapping):
        return None
    payload = row.get("row")
    return dict(payload) if isinstance(payload, Mapping) else None


def _sidecar_watermark(sidecar_path: Path) -> int | None:
    """The source byte offset the sidecar is current to, or None.

    Every sidecar line carries the main log's size at the moment the row it
    mirrors was appended, so the LAST line is the watermark. Keeping it on the
    rows rather than in the header is what lets the sidecar stay append-only: a
    header watermark would have to be rewritten on every event.
    """
    last: int | None = None
    try:
        with Path(sidecar_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A torn tail is a doubt, and doubt means rebuild.
                    return None
                if not isinstance(row, dict):
                    return None
                offset = row.get("src_offset")
                if isinstance(offset, int):
                    last = offset
    except OSError:
        return None
    return last


def _stream_resolved_rows(events_path: Path, start_offset: int = 0):
    """(row, offset-after-this-line) for every `level_resolved` line from an offset."""
    try:
        with Path(events_path).open("rb") as handle:
            if start_offset:
                handle.seek(start_offset)
            for raw in handle:
                position = handle.tell()
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(row, dict) and row.get("event_type") == "level_resolved":
                    yield row, position
    except OSError:
        return


def append_resolved_sidecar_row(
    sidecar_path: Path, row: Mapping[str, Any], src_offset: int
) -> bool:
    """Mirror one resolved event. Returns False on failure and NEVER raises.

    Evidence rule: a failed append loses the event, never the thing it records.
    This sidecar is derived, so losing a line costs a catch-up scan later and
    nothing else - `_sidecar_watermark` will simply say the sidecar is behind.
    """
    try:
        target = Path(sidecar_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"src_offset": int(src_offset), "row": dict(row)}, separators=(",", ":")
                )
                + "\n"
            )
    except (OSError, TypeError, ValueError):
        return False
    return True


def rebuild_resolved_sidecar(
    events_path: Path | None = None, sidecar_path: Path | None = None
) -> dict[str, Any]:
    """Build the sidecar from the main log, completely. Idempotent.

    Written to a temporary file and moved into place, so a crash halfway
    through leaves the previous sidecar rather than a half one. Returns the
    verification counts: `rows` is what was written, `resolved` what the stream
    found, and they must agree.
    """
    source = Path(events_path or technical_integrity_events_path())
    target = Path(sidecar_path or technical_integrity_resolved_path(source))
    if not source.exists():
        return {"ok": False, "reason": "no events log", "rows": 0, "resolved": 0}
    try:
        stat = source.stat()
    except OSError as exc:
        return {"ok": False, "reason": str(exc), "rows": 0, "resolved": 0}
    header = {
        "schema": RESOLVED_SIDECAR_SCHEMA,
        "source": source.name,
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_size": int(stat.st_size),
        "built_at": normalize_market_local_datetime().isoformat(timespec="seconds"),
    }
    written = 0
    resolved = 0
    tmp = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(header, separators=(",", ":")) + "\n")
            for row, offset in _stream_resolved_rows(source):
                resolved += 1
                handle.write(
                    json.dumps(
                        {"src_offset": int(offset), "row": row}, separators=(",", ":")
                    )
                    + "\n"
                )
                written += 1
            # The last resolved row is rarely the last LINE, so the build ends
            # by recording how far it actually read. Without it every later
            # sync re-streams the tail after the final resolution.
            handle.write(
                json.dumps(
                    {"src_offset": int(stat.st_size), "row": {"event_type": "sidecar_watermark"}},
                    separators=(",", ":"),
                )
                + "\n"
            )
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return {"ok": False, "reason": str(exc), "rows": written, "resolved": resolved}
    return {"ok": written == resolved, "reason": "", "rows": written, "resolved": resolved}


def sync_resolved_sidecar(
    events_path: Path | None = None, sidecar_path: Path | None = None
) -> dict[str, Any]:
    """Bring the sidecar level with the main log. Cheap when it already is.

    Three outcomes: CURRENT (the watermark equals the log's size - nothing to
    do), CAUGHT UP (the log grew; only the tail past the watermark is streamed),
    or REBUILT (no sidecar, a torn one, or a watermark past the end of the log,
    which means the log was replaced under it).
    """
    source = Path(events_path or technical_integrity_events_path())
    target = Path(sidecar_path or technical_integrity_resolved_path(source))
    if not source.exists():
        return {"ok": False, "action": "no_source"}
    try:
        size = int(source.stat().st_size)
    except OSError:
        return {"ok": False, "action": "unreadable_source"}
    watermark = _sidecar_watermark(target) if target.exists() else None
    if watermark is None or watermark > size:
        result = rebuild_resolved_sidecar(source, target)
        return {"ok": bool(result.get("ok")), "action": "rebuilt", **result}
    if watermark == size:
        return {"ok": True, "action": "current"}
    appended = 0
    for row, offset in _stream_resolved_rows(source, watermark):
        if not append_resolved_sidecar_row(target, row, offset):
            # Could not keep up; a rebuild is the honest answer next time.
            return {"ok": False, "action": "append_failed", "appended": appended}
        appended += 1
    if appended == 0:
        # The tail held no resolved rows, but the sidecar must still record how
        # far it has read or every later call rescans the same tail.
        if not append_resolved_sidecar_row(target, {"event_type": "sidecar_watermark"}, size):
            return {"ok": False, "action": "append_failed", "appended": 0}
    return {"ok": True, "action": "caught_up", "appended": appended}


def load_resolved_technical_integrity_events(
    path: Path | None = None, *, use_sidecar: bool = True
) -> list[dict[str, Any]]:
    target = Path(path or technical_integrity_events_path())
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    if target.suffix.lower() == ".json":
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        candidates = payload.get("events") if isinstance(payload, dict) else payload
        if not isinstance(candidates, list):
            return []
        return [
            dict(row)
            for row in candidates
            if isinstance(row, Mapping) and row.get("event_type") == "level_resolved"
        ]
    if use_sidecar:
        sidecar_rows = _resolved_rows_from_sidecar(target)
        if sidecar_rows is not None:
            return sidecar_rows
    try:
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("event_type") == "level_resolved":
                    rows.append(row)
    except OSError:
        return []
    return rows


def _resolved_rows_from_sidecar(events_path: Path) -> list[dict[str, Any]] | None:
    """The sidecar's rows in log order, or None to make the caller stream.

    None on ANY doubt - no sidecar, a sync that could not finish, a torn line.
    The full stream is always available and always right, so the sidecar is
    allowed to be a pure optimisation.
    """
    sidecar = technical_integrity_resolved_path(events_path)
    try:
        state = sync_resolved_sidecar(events_path, sidecar)
    except Exception:  # noqa: BLE001 - a derived file must never break the replay
        return None
    if not state.get("ok"):
        return None
    rows: list[dict[str, Any]] = []
    # One ANSWER per source line, however many times it was mirrored. The
    # evidence clock appends to the main log first and mirrors second, and the
    # wrap-up's sync runs on another thread: a switch between those two steps
    # lets sync catch up the tail before the clock's own mirror lands, so the
    # same row reaches the sidecar twice (review round, 2026-08-31, reproduced).
    # Both copies carry the same source byte offset - the line's identity - so
    # the duplicate may sit on disk but never in what the replay is handed.
    seen_offsets: set[int] = set()
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    return None
                payload = _sidecar_row_payload(parsed)
                if payload is None:
                    continue
                if payload.get("event_type") != "level_resolved":
                    # The watermark placeholder rows carry no event.
                    continue
                offset = parsed.get("src_offset") if isinstance(parsed, dict) else None
                if isinstance(offset, int):
                    if offset in seen_offsets:
                        continue
                    seen_offsets.add(offset)
                rows.append(payload)
    except OSError:
        return None
    return rows


def default_calibration_configs() -> dict[str, TechnicalIntegrityConfig]:
    """Small, predeclared candidates; the report cannot mutate the active model."""
    return {
        "baseline_v1": TechnicalIntegrityConfig(),
        "faster_adaptation": TechnicalIntegrityConfig(prior_weight=1.0),
        "steadier_prior": TechnicalIntegrityConfig(prior_weight=4.0),
        "stricter_reclaims": TechnicalIntegrityConfig(reclaimed_value=0.50),
        "reclaim_friendly": TechnicalIntegrityConfig(reclaimed_value=0.80),
    }


def write_technical_integrity_calibration_report(
    *,
    events_path: Path | None = None,
    output_path: Path | None = None,
    configs: Mapping[str, TechnicalIntegrityConfig] | None = None,
) -> dict[str, Any]:
    events = load_resolved_technical_integrity_events(events_path)
    report = compare_scoring_configs(events, configs or default_calibration_configs())
    _atomic_write_json(Path(output_path or technical_integrity_calibration_path()), report)
    return report


def calibration_report_is_current(
    *,
    output_path: Path | None = None,
    now: datetime | None = None,
) -> bool:
    """True when today's calibration replay already completed.

    The report replays every stored outcome under five candidate configs; with
    a 100 MB+ append-only event log that is an hour-class, core-pegging job.
    The after-close wrap-up used to stamp only the END of its whole chain, so
    a crash or an impatient shutdown after this step re-burned the entire
    replay on every restart - measured live on 2026-07-30 as a full core
    pegged from launch, with the newest completed report a week old.  The
    report's own ``generated_at`` is the step-level completion stamp; missing,
    unreadable, corrupt or stale all honestly mean "not current" and the
    replay runs.
    """
    try:
        path = Path(output_path or technical_integrity_calibration_path())
        payload = json.loads(path.read_text(encoding="utf-8"))
        stamp = datetime.fromisoformat(str(payload.get("generated_at") or ""))
    except Exception:
        return False
    moment = normalize_market_local_datetime(now)
    return normalize_market_local_datetime(stamp).date() >= moment.date()


def format_technical_integrity_snapshot(
    payload: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> tuple[str, str, str]:
    """Return compact always-visible text, an explanatory tooltip, and color."""
    snapshot = payload if isinstance(payload, Mapping) else {}
    market = snapshot.get("market") if isinstance(snapshot.get("market"), Mapping) else {}
    if not market:
        return (
            "Technicals: building",
            "Technical Integrity appears after completed-M5 level tests resolve. It is advisory only.",
            "#8b8fa3",
        )
    session_date = str(snapshot.get("session_date") or "")
    current_date = normalize_market_local_datetime(now).date().isoformat()
    if session_date and session_date != current_date:
        return (
            "Technicals: building today",
            f"The latest Technical Integrity evidence is from {session_date}. "
            "Today's state appears after completed-M5 level tests resolve. Advisory only.",
            "#8b8fa3",
        )
    state = str(market.get("state") or "BUILDING")
    raw_score = market.get("score")
    score = float(raw_score) if raw_score is not None else None
    pressure = str(market.get("pressure") or "BALANCED")
    confidence = str(market.get("confidence") or "LOW")
    short = {"HIGH": "HIGH", "MEDIUM": "MED", "LOW": "LOW"}
    confidence_short = short.get(confidence, confidence)
    d1_test_count = int(market.get("d1_test_count") or 0)
    intraday_test_count = int(market.get("intraday_test_count") or 0)
    d1_raw_score = market.get("d1_score")
    # D1 major levels are the headline; intraday M5 levels stay visible but
    # secondary. A null score means the decisive evidence has not accumulated
    # yet - say "building" instead of printing the prior as a reading.
    if d1_test_count > 0 and d1_raw_score is not None:
        d1_score = float(d1_raw_score)
        d1_state = str(market.get("d1_state") or "BUILDING")
        d1_pressure = str(market.get("d1_pressure") or "BALANCED")
        d1_confidence = str(market.get("d1_confidence") or "LOW")
        chip = f"Technicals D1: {d1_state} {d1_score:.1f}/10 | {d1_pressure} | {short.get(d1_confidence, d1_confidence)}"
        if intraday_test_count > 0:
            intraday_score = market.get("intraday_score")
            if intraday_score is not None:
                chip += f" · M5 {float(intraday_score):.1f}/10"
        headline_pressure = d1_pressure
        headline_state = d1_state
    elif score is not None:
        chip = f"Technicals D1: building · M5 {state} {score:.1f}/10 | {pressure} | {confidence_short}"
        headline_pressure = pressure
        headline_state = state
    else:
        return (
            "Technicals: building",
            "Technical Integrity reports a score once enough level tests resolve "
            "decisively (held/reclaimed/broke). Inconclusive 'chop' tests are "
            "counted but never scored. Advisory only.",
            "#8b8fa3",
        )
    lines = [
        f"Scanned-market Technical Integrity: {score:.1f}/10 ({state})"
        if score is not None
        else "Scanned-market Technical Integrity: building (not enough decisive tests)",
    ]
    if d1_test_count > 0:
        d1_score_value = market.get("d1_score")
        lines.append(
            f"D1 major levels (daily SMA 50/100/200, D1 trendlines, horizontal S/R): "
            f"{float(d1_score_value):.1f}/10 ({market.get('d1_state')}) | "
            f"break pressure {market.get('d1_pressure')} | confidence {market.get('d1_confidence')} | "
            f"{d1_test_count} resolved D1 tests"
        )
    else:
        lines.append(
            "D1 major levels: building - no D1 level test has resolved yet this session."
        )
    if intraday_test_count > 0 and market.get("intraday_score") is not None:
        lines.append(
            f"Intraday M5 levels (VWAP/EMA/bands): {float(market.get('intraday_score')):.1f}/10 "
            f"({market.get('intraday_state')}); {intraday_test_count} resolved tests"
        )
    lines.extend(
        [
            f"Break pressure: {pressure} | confidence: {confidence}",
            f"Evidence: {int(market.get('test_count') or 0)} resolved tests across "
            f"{int(market.get('symbol_count') or 0)} symbols; {int(market.get('pending_count') or 0)} pending.",
            "1 means levels are breaking easily; 10 means levels are earning repeated respect.",
            "Advisory only; coverage is the symbols BounceBot has scanned, not a full-market census.",
        ]
    )
    weak = snapshot.get("weakest_industries") or []
    strong = snapshot.get("strongest_industries") or []
    if weak:
        lines.append("Weakest industries:")
        lines.extend(
            f"- {row.get('label')} {float(row.get('score') or 0.0):.1f}/10 {row.get('state')} "
            f"({row.get('pressure')}, n={row.get('test_count')})"
            for row in weak[:3]
        )
    if strong:
        lines.append("Strongest industries:")
        lines.extend(
            f"- {row.get('label')} {float(row.get('score') or 0.0):.1f}/10 {row.get('state')} "
            f"({row.get('pressure')}, n={row.get('test_count')})"
            for row in strong[:3]
        )
    if headline_pressure == "BEARISH":
        color = "#f85149"
    elif headline_pressure == "BULLISH":
        color = "#3fb950"
    elif headline_state in {"VERY WEAK", "WEAK"}:
        color = "#d29922"
    elif headline_state in {"FIRM", "STRONG"}:
        color = "#58a6ff"
    else:
        color = "#c9d1d9"
    return chip, "\n".join(lines), color
