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
import math
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from market_session import normalize_market_local_datetime
from project_paths import get_diagnostics_dir


FEATURE_VERSION = "technical_integrity_v1"
SNAPSHOT_SCHEMA = "technical_integrity_snapshot_v1"
EVENT_SCHEMA = "technical_integrity_event_v1"
STATE_SCHEMA = "technical_integrity_monitor_state_v1"
CALIBRATION_SCHEMA = "technical_integrity_calibration_v1"


@dataclass(frozen=True)
class TechnicalIntegrityConfig:
    prior_weight: float = 2.0
    prior_respect_probability: float = 0.5
    held_value: float = 1.0
    reclaimed_value: float = 0.65
    chop_value: float = 0.5
    broke_value: float = 0.0
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


def completed_m5_bars(rows: Any, *, now: datetime | None = None) -> list[dict[str, Any]]:
    """Normalize valid bars and exclude any M5 candle that can still form."""
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
    latest_date = complete[-1]["_start_local"].date()
    return [row for row in complete if row["_start_local"].date() == latest_date]


def _score_state(score: float, test_count: int) -> str:
    if test_count <= 0:
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
    symbols: set[str] = set()
    for event in events:
        value = _outcome_value(str(event.get("outcome") or ""), config)
        if value is None:
            continue
        try:
            weight = max(0.0, float(event.get("event_weight") or 1.0))
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        weighted_value += weight * value
        total_weight += weight
        resolved_weight += weight
        test_count += 1
        symbol = str(event.get("symbol") or "").strip().upper()
        if symbol:
            symbols.add(symbol)
    probability = weighted_value / total_weight if total_weight > 0 else 0.5
    return probability, resolved_weight, test_count, symbols


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
    probability, _weight, count, _symbols = _integrity_probability(side_events, config)
    return round(1.0 + 9.0 * probability, 1) if count else None


def _entity_row(
    events: list[Mapping[str, Any]],
    *,
    entity_type: str,
    entity_key: str,
    label: str,
    config: TechnicalIntegrityConfig,
) -> dict[str, Any]:
    probability, resolved_weight, test_count, symbols = _integrity_probability(events, config)
    score = round(1.0 + 9.0 * probability, 1)
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
        sub_probability, sub_weight, sub_count, sub_symbols = _integrity_probability(subset, config)
        sub_score = round(1.0 + 9.0 * sub_probability, 1) if sub_count else None
        sub_pressure, _up, _down = _pressure(subset)
        row.update(
            {
                f"{prefix}score": sub_score,
                f"{prefix}state": _score_state(sub_score if sub_score is not None else 5.5, sub_count),
                f"{prefix}pressure": sub_pressure,
                f"{prefix}confidence": _confidence(sub_weight, len(sub_symbols)),
                f"{prefix}test_count": sub_count,
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
    weakest_industries = sorted(industries, key=lambda row: (row["score"], -row["test_count"], row["label"]))[:5]
    strongest_industries = sorted(industries, key=lambda row: (-row["score"], -row["test_count"], row["label"]))[:5]
    weakest_sectors = sorted(sectors, key=lambda row: (row["score"], -row["test_count"], row["label"]))[:5]
    strongest_sectors = sorted(sectors, key=lambda row: (-row["score"], -row["test_count"], row["label"]))[:5]

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


def load_technical_integrity_snapshot(path: Path | None = None) -> dict[str, Any]:
    target = Path(path or technical_integrity_snapshot_path())
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) and payload.get("schema") == SNAPSHOT_SCHEMA else {}


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
    ) -> None:
        self.events_path = Path(events_path or technical_integrity_events_path())
        self.state_path = Path(state_path or technical_integrity_state_path())
        self.snapshot_path = Path(snapshot_path or technical_integrity_snapshot_path())
        self.config = config or TechnicalIntegrityConfig()
        self._lock = threading.RLock()
        self.session_date = ""
        self.pending: dict[str, dict[str, Any]] = {}
        self.seen_test_ids: set[str] = set()
        self.resolved_events: list[dict[str, Any]] = []
        self._load_state()

    @property
    def pending_count(self) -> int:
        return len(self.pending)

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
        self._load_resolved_events()

    def _load_resolved_events(self) -> None:
        self.resolved_events = []
        if not self.session_date or not self.events_path.exists():
            return
        started: dict[str, dict[str, Any]] = {}
        resolved: dict[str, dict[str, Any]] = {}
        try:
            with self.events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict) or str(row.get("session_date") or "") != self.session_date:
                        continue
                    event_id = str(row.get("event_id") or "")
                    if not event_id:
                        continue
                    event_type = str(row.get("event_type") or "")
                    self.seen_test_ids.add(event_id)
                    if event_type == "level_test_started":
                        started[event_id] = row
                    elif event_type == "level_resolved":
                        resolved[event_id] = row
        except OSError:
            self.resolved_events = []
            return
        self.resolved_events = sorted(
            resolved.values(),
            key=lambda row: (str(row.get("resolved_at") or ""), str(row.get("event_id") or "")),
        )
        # The append-only ledger repairs a crash between event append and the
        # atomic state write. Resolved IDs suppress stale pending state.
        recovered_pending = {
            event_id: row for event_id, row in started.items() if event_id not in resolved
        }
        recovered_pending.update(
            {
                event_id: row
                for event_id, row in self.pending.items()
                if event_id not in resolved and event_id not in recovered_pending
            }
        )
        self.pending = recovered_pending

    def _ensure_session(self, session_date: str) -> None:
        if self.session_date == session_date:
            return
        self.session_date = session_date
        self.pending = {}
        self.seen_test_ids = set()
        self._load_resolved_events()

    def _append_event(self, row: Mapping[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _save_state(self) -> None:
        _atomic_write_json(
            self.state_path,
            {
                "schema": STATE_SCHEMA,
                "feature_version": FEATURE_VERSION,
                "session_date": self.session_date,
                "pending": self.pending,
                "seen_test_ids": sorted(self.seen_test_ids),
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
            changed = False
            for event_id, pending in list(self.pending.items()):
                if str(pending.get("symbol") or "").upper() != sym:
                    continue
                resolved = _resolve_pending(pending, bars, self.config)
                if resolved is None:
                    continue
                self._append_event(resolved)
                self.resolved_events.append(resolved)
                self.pending.pop(event_id, None)
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


def load_resolved_technical_integrity_events(path: Path | None = None) -> list[dict[str, Any]]:
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
    score = float(market.get("score") or 5.5)
    pressure = str(market.get("pressure") or "BALANCED")
    confidence = str(market.get("confidence") or "LOW")
    short = {"HIGH": "HIGH", "MEDIUM": "MED", "LOW": "LOW"}
    confidence_short = short.get(confidence, confidence)
    d1_test_count = int(market.get("d1_test_count") or 0)
    intraday_test_count = int(market.get("intraday_test_count") or 0)
    # D1 major levels are the headline; intraday M5 levels stay visible but
    # secondary. Without any resolved D1 test yet, fall back to the combined
    # score so the chip is never blank early in the session.
    if d1_test_count > 0:
        d1_score = float(market.get("d1_score") or 5.5)
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
    else:
        chip = f"Technicals D1: building · M5 {state} {score:.1f}/10 | {pressure} | {confidence_short}"
        headline_pressure = pressure
        headline_state = state
    lines = [
        f"Scanned-market Technical Integrity: {score:.1f}/10 ({state})",
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
