from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


LEVEL_STORE_SCHEMA_VERSION = 1
HV_RELVOL_GREEN = 3.0
HV_RELVOL_RED = 2.0
HV_VOL_SMA = 50
LEVEL_TOL_ATR_FRACTION = 0.05
LEVEL_BREAK_ATR = 0.25
LEVEL_FORWARD_BARS = 5
LEVEL_TOUCH_WEIGHT = 0.08
LEVEL_TOUCH_CAP = 0.40
LEVEL_BUCKET_WEIGHTS = {"green": 1.0, "red": 0.35}


def _coerce_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    work = df.copy()
    work.rename(columns={column: str(column).strip().lower() for column in work.columns}, inplace=True)
    if "datetime" not in work.columns:
        for candidate in ("date", "time", "timestamp"):
            if candidate in work.columns:
                work.rename(columns={candidate: "datetime"}, inplace=True)
                break
    required = {"datetime", "open", "high", "low", "close", "volume"}
    if required - set(work.columns):
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    if work.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    return work.sort_values("datetime").reset_index(drop=True)


def _date_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        value = value.date()
    elif hasattr(value, "date") and not isinstance(value, date):
        value = value.date()
    if hasattr(value, "isoformat"):
        return value.isoformat()[:10]
    return str(value)[:10]


def _date_key(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _earnings_origin_dates(work: pd.DataFrame, earnings_dates: Iterable[str]) -> set[str]:
    earnings_keys = {_date_key(str(value)[:10]) for value in earnings_dates or [] if str(value or "").strip()}
    earnings_keys = {value for value in earnings_keys if value is not None}
    if not earnings_keys or work.empty:
        return set()

    date_texts = [_date_text(value) for value in work["datetime"]]
    date_keys = [_date_key(value) for value in date_texts]
    origin_dates: set[str] = set()
    for idx, trade_date in enumerate(date_keys):
        if trade_date not in earnings_keys:
            continue
        for neighbor_idx in (idx - 1, idx, idx + 1):
            if 0 <= neighbor_idx < len(date_texts):
                origin_dates.add(date_texts[neighbor_idx])
    return origin_dates


def _level_tolerance(atr20: float | None, price: float | None = None, *, tol_frac: float = LEVEL_TOL_ATR_FRACTION) -> float:
    atr_value = _coerce_float(atr20)
    if atr_value is not None and atr_value > 0:
        return float(atr_value) * float(tol_frac)
    price_value = abs(_coerce_float(price) or 0.0)
    return max(0.01, price_value * 0.001)


def compute_relvol(df: pd.DataFrame | None, vol_sma: int = HV_VOL_SMA) -> pd.Series:
    work = _normalize_frame(df)
    if work.empty:
        return pd.Series(dtype="float64")
    lookback = max(1, int(vol_sma or HV_VOL_SMA))
    volume = pd.to_numeric(work["volume"], errors="coerce")
    volume_sma = volume.rolling(lookback, min_periods=lookback).mean()
    return volume / volume_sma


def extract_hv_levels(
    df: pd.DataFrame | None,
    atr20: float | None,
    *,
    green: float = HV_RELVOL_GREEN,
    red: float = HV_RELVOL_RED,
    vol_sma: int = HV_VOL_SMA,
    earnings_dates: Iterable[str] = (),
) -> list[dict]:
    work = _normalize_frame(df)
    if work.empty:
        return []
    relvol = compute_relvol(work, vol_sma=vol_sma)
    earnings_set = _earnings_origin_dates(work, earnings_dates)
    candidates: list[dict] = []
    for idx, row in work.iterrows():
        rv = _coerce_float(relvol.iloc[idx] if idx < len(relvol) else None)
        if rv is None or rv < float(red):
            continue
        bucket = "green" if rv >= float(green) else "red"
        trade_date = _date_text(row.get("datetime"))
        earnings_origin = trade_date in earnings_set
        for origin_side, price_key in (("high", "high"), ("low", "low")):
            price = _coerce_float(row.get(price_key))
            if price is None:
                continue
            candidates.append(
                {
                    "kind": "hv_horizontal",
                    "price": round(float(price), 4),
                    "origin_side": origin_side,
                    "bucket": bucket,
                    "relvol": round(float(rv), 4),
                    "first_seen": trade_date,
                    "last_seen": trade_date,
                    "earnings_origin": bool(earnings_origin),
                    "non_earnings_anchor_candidate": bool(bucket == "green" and not earnings_origin),
                    "atr20_at_origin": _coerce_float(atr20),
                    "source_bar_index": int(idx),
                }
            )
    return candidates


def _level_strength(level: dict) -> float:
    bucket = str(level.get("bucket") or "red").lower()
    touch_count = int(level.get("touch_count", 0) or 0)
    return round(
        float(LEVEL_BUCKET_WEIGHTS.get(bucket, LEVEL_BUCKET_WEIGHTS["red"]))
        + min(LEVEL_TOUCH_CAP, touch_count * LEVEL_TOUCH_WEIGHT),
        4,
    )


def _cluster_from_members(members: list[dict], atr20: float | None) -> dict:
    weights = [max(_coerce_float(member.get("relvol")) or 0.0, 0.01) for member in members]
    prices = [float(member["price"]) for member in members]
    total_weight = sum(weights) or 1.0
    price = sum(price * weight for price, weight in zip(prices, weights)) / total_weight
    bucket = "green" if any(str(member.get("bucket")) == "green" for member in members) else "red"
    first_seen_values = [str(member.get("first_seen") or "") for member in members if member.get("first_seen")]
    last_seen_values = [
        str(member.get("last_seen") or member.get("first_seen") or "")
        for member in members
        if member.get("last_seen") or member.get("first_seen")
    ]
    level = {
        "kind": "hv_horizontal",
        "price": round(float(price), 4),
        "band": [round(min(prices), 4), round(max(prices), 4)],
        "origin_sides": sorted({str(member.get("origin_side") or "") for member in members if member.get("origin_side")}),
        "bucket": bucket,
        "relvol": round(max(_coerce_float(member.get("relvol")) or 0.0 for member in members), 4),
        "first_seen": min(first_seen_values) if first_seen_values else "",
        "last_seen": max(last_seen_values) if last_seen_values else "",
        "earnings_origin": any(bool(member.get("earnings_origin")) for member in members),
        "non_earnings_anchor_candidate": any(bool(member.get("non_earnings_anchor_candidate")) for member in members),
        "member_count": len(members),
        "tol_atr_fraction": LEVEL_TOL_ATR_FRACTION,
        "atr20_at_update": _coerce_float(atr20),
        "touch_count": 0,
        "respect_count": 0,
        "break_count": 0,
    }
    level["strength"] = _level_strength(level)
    return level


def cluster_levels(
    candidates: list[dict] | None,
    atr20: float | None,
    *,
    tol_frac: float = LEVEL_TOL_ATR_FRACTION,
) -> list[dict]:
    valid = [
        dict(candidate)
        for candidate in (candidates or [])
        if _coerce_float(candidate.get("price")) is not None
    ]
    if not valid:
        return []
    valid.sort(key=lambda item: float(item["price"]))
    clusters: list[list[dict]] = []
    current: list[dict] = []
    current_max = None
    for candidate in valid:
        price = float(candidate["price"])
        tolerance = _level_tolerance(atr20, price, tol_frac=tol_frac)
        if current and current_max is not None and price > float(current_max) + tolerance:
            clusters.append(current)
            current = []
        current.append(candidate)
        current_max = max(float(current_max) if current_max is not None else price, price)
    if current:
        clusters.append(current)
    return [_cluster_from_members(members, atr20) for members in clusters]


def recompute_touch_stats(
    levels: list[dict] | None,
    df: pd.DataFrame | None,
    atr20: float | None,
    *,
    tol_frac: float = LEVEL_TOL_ATR_FRACTION,
    break_atr: float = LEVEL_BREAK_ATR,
    forward_bars: int = LEVEL_FORWARD_BARS,
) -> list[dict]:
    work = _normalize_frame(df)
    if not levels:
        return []
    if work.empty:
        return [dict(level) for level in levels]
    break_tolerance = _level_tolerance(atr20, tol_frac=float(break_atr))
    output = []
    for raw_level in levels:
        level = dict(raw_level)
        price = _coerce_float(level.get("price"))
        if price is None:
            continue
        first_seen = _date_key(level.get("first_seen"))
        tolerance = _level_tolerance(atr20, price, tol_frac=tol_frac)
        touch_count = 0
        respect_count = 0
        break_count = 0
        post_break_returns = []
        last_touch = ""
        last_break = ""
        for idx, row in work.iterrows():
            trade_date_text = _date_text(row.get("datetime"))
            trade_date = _date_key(trade_date_text)
            if first_seen is not None and trade_date is not None and trade_date <= first_seen:
                continue
            high_value = _coerce_float(row.get("high"))
            low_value = _coerce_float(row.get("low"))
            close_value = _coerce_float(row.get("close"))
            if high_value is None or low_value is None or close_value is None:
                continue
            touched = float(low_value) <= float(price) + tolerance and float(high_value) >= float(price) - tolerance
            broke_up = float(close_value) > float(price) + break_tolerance
            broke_down = float(close_value) < float(price) - break_tolerance
            if not touched:
                continue
            touch_count += 1
            last_touch = trade_date_text
            if broke_up or broke_down:
                break_count += 1
                last_break = trade_date_text
                future_idx = min(len(work) - 1, idx + max(1, int(forward_bars or LEVEL_FORWARD_BARS)))
                future_close = _coerce_float(work.iloc[future_idx].get("close"))
                if future_close is not None and close_value:
                    post_break_returns.append(round(((future_close - close_value) / close_value) * 100.0, 4))
            else:
                respect_count += 1
        level.update(
            {
                "touch_count": int(touch_count),
                "respect_count": int(respect_count),
                "break_count": int(break_count),
                "last_touch": last_touch,
                "last_break": last_break,
                "avg_post_break_return_pct": (
                    round(sum(post_break_returns) / len(post_break_returns), 4)
                    if post_break_returns
                    else None
                ),
                "strength": _level_strength({"bucket": level.get("bucket"), "touch_count": touch_count}),
                "atr20_at_update": _coerce_float(atr20),
            }
        )
        output.append(level)
    output.sort(key=lambda item: (float(item.get("price") or 0.0), str(item.get("first_seen") or "")))
    return output


def default_level_store(symbol: str = "") -> dict:
    return {
        "schema_version": LEVEL_STORE_SCHEMA_VERSION,
        "symbol": str(symbol or "").strip().upper(),
        "updated": "",
        "atr20_at_update": None,
        "levels": [],
    }


def merge_into_store(
    store: dict | None,
    clusters: list[dict] | None,
    *,
    symbol: str = "",
    atr20: float | None = None,
    updated: str | None = None,
) -> dict:
    payload = default_level_store(symbol)
    if isinstance(store, dict):
        payload.update({key: value for key, value in store.items() if key != "levels"})
        payload["levels"] = [dict(level) for level in store.get("levels", []) if isinstance(level, dict)]
    payload["schema_version"] = LEVEL_STORE_SCHEMA_VERSION
    payload["symbol"] = str(symbol or payload.get("symbol") or "").strip().upper()
    tolerance = _level_tolerance(atr20)
    merged = [dict(level) for level in payload.get("levels", [])]
    for cluster in clusters or []:
        price = _coerce_float(cluster.get("price"))
        if price is None:
            continue
        match = None
        for existing in merged:
            if str(existing.get("kind") or "") != str(cluster.get("kind") or ""):
                continue
            existing_price = _coerce_float(existing.get("price"))
            if existing_price is not None and abs(float(existing_price) - float(price)) <= tolerance:
                match = existing
                break
        if match is None:
            merged.append(dict(cluster))
            continue
        first_seen_values = [str(match.get("first_seen") or ""), str(cluster.get("first_seen") or "")]
        first_seen_values = [value for value in first_seen_values if value]
        match.update(cluster)
        if first_seen_values:
            match["first_seen"] = min(first_seen_values)
    payload["levels"] = sorted(merged, key=lambda item: (float(item.get("price") or 0.0), str(item.get("first_seen") or "")))
    payload["updated"] = str(updated or datetime.now().date().isoformat())
    payload["atr20_at_update"] = _coerce_float(atr20)
    return payload


def levels_near(
    store: dict | None,
    price: float | None,
    atr20: float | None,
    *,
    tol_frac: float = LEVEL_TOL_ATR_FRACTION,
    min_strength: float = 0.0,
) -> list[dict]:
    entry_price = _coerce_float(price)
    if entry_price is None or not isinstance(store, dict):
        return []
    tolerance = _level_tolerance(atr20, entry_price, tol_frac=tol_frac)
    matches = []
    for level in store.get("levels", []) or []:
        level_price = _coerce_float(level.get("price"))
        if level_price is None:
            continue
        strength = _coerce_float(level.get("strength")) or 0.0
        if strength < float(min_strength or 0.0):
            continue
        distance = float(level_price) - float(entry_price)
        if abs(distance) <= tolerance:
            item = dict(level)
            item["distance"] = round(distance, 4)
            item["distance_atr"] = None if not atr20 else round(distance / float(atr20), 4)
            item["abs_distance_atr"] = None if not atr20 else round(abs(distance) / float(atr20), 4)
            item["position"] = "above" if distance > 0 else "below" if distance < 0 else "at"
            matches.append(item)
    matches.sort(key=lambda item: (abs(float(item.get("distance") or 0.0)), -float(item.get("strength") or 0.0)))
    return matches


def levels_blocking_entry(
    store: dict | None,
    side: str,
    entry_price: float | None,
    atr20: float | None,
    *,
    tol_frac: float = LEVEL_TOL_ATR_FRACTION,
    min_strength: float = LEVEL_BUCKET_WEIGHTS["green"],
) -> list[dict]:
    nearby = levels_near(store, entry_price, atr20, tol_frac=tol_frac, min_strength=min_strength)
    normalized_side = str(side or "").strip().upper()
    if normalized_side == "SHORT":
        return [level for level in nearby if float(level.get("distance") or 0.0) <= 0]
    return [level for level in nearby if float(level.get("distance") or 0.0) >= 0]


def level_store_path(levels_dir: Path, symbol: str) -> Path:
    safe_symbol = "".join(ch for ch in str(symbol or "").strip().upper() if ch.isalnum() or ch in {"_", "-"})
    return Path(levels_dir) / f"{safe_symbol}.json"


def load_level_store(path: Path, symbol: str = "") -> dict:
    target = Path(path)
    if not target.exists():
        return default_level_store(symbol)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return default_level_store(symbol)
    if not isinstance(payload, dict):
        return default_level_store(symbol)
    payload.setdefault("levels", [])
    payload.setdefault("schema_version", LEVEL_STORE_SCHEMA_VERSION)
    payload.setdefault("symbol", str(symbol or "").strip().upper())
    return payload


def save_level_store(path: Path, store: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")
