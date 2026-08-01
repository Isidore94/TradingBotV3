"""Evening mode's morning briefing: the desk is ready before the trader is.

Armed the night before a sleep-in morning, Auto EVENING keeps discovery
identical to DESK (picks stage for chart approval, nothing self-applies) and
this module turns the first hour into a briefing:

- 07:00 / 07:15 / 07:30 strength checks snapshot every staged intraday pick
  against its day extreme. A name that was pressing its HOD at 07:00 but
  slipped by 07:30 is FADED - presented, but explicitly not recommended
  (trader directive: "if it starts to dip maybe don't recommend it").
- The 07:00 early Master AVWAP run (autopilot_core early slot) supplies the
  swing rows; the best D1s per side are ranked by expected R.
- Overnight price-alert crossings (price_alerts trigger log) are folded in so
  the briefing answers "what happened while I slept" in one read.

Everything here is pure computation over plain dicts (file I/O helpers
aside) so the persistence rules are testable without Qt, IB, or yfinance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Iterable, Mapping

from project_paths import EVENING_BRIEFING_FILE, EVENING_BRIEFING_STATE_FILE

# Local-clock strength-check slots for a normal 06:30 session. The trader
# arrives 07:00-07:30; three looks are enough to tell "still pressing" from
# "faded off the open push" without burning API budget.
EVENING_STRENGTH_CHECK_SLOTS = ("07:00", "07:15", "07:30")

# A pick "holds" when it is still within this of its HOD/LOD at the latest
# check (same bar as the near-extreme adds) ...
PERSISTENCE_NEAR_EXTREME_PCT = 1.0
# ... and its price has not gone the wrong way by more than this since the
# first check. Half a percent of give keeps normal pullback noise from
# flagging a strong name as faded.
PERSISTENCE_FADE_TOLERANCE_PCT = 0.5

BRIEFING_TOP_D1_PER_SIDE = 5
BRIEFING_MAX_INTRADAY_PICKS = 8


def _slot_time(slot: str) -> dt_time:
    hours, minutes = str(slot).strip().split(":", 1)
    return dt_time(int(hours), int(minutes))


def due_strength_check(
    now: datetime,
    recorded_slots: Iterable[str],
    slots: tuple[str, ...] = EVENING_STRENGTH_CHECK_SLOTS,
) -> str | None:
    """The strength-check slot to take now, or ``None``.

    Only the LATEST due slot ever runs: starting the day at 07:20 takes the
    07:15 look immediately and the 07:30 look on time, rather than replaying
    07:00 with 07:20 data. A recorded slot also retires every earlier one.
    """
    due = [slot for slot in slots if _slot_time(slot) <= now.time()]
    if not due:
        return None
    candidate = due[-1]
    recorded = {str(slot) for slot in recorded_slots}
    if any(_slot_time(slot) >= _slot_time(candidate) for slot in recorded if slot in slots):
        return None
    return candidate


def load_evening_state(
    now: datetime,
    path: Path = EVENING_BRIEFING_STATE_FILE,
) -> dict[str, Any]:
    today = now.date().isoformat()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    if not isinstance(payload, dict) or payload.get("date") != today:
        payload = {"date": today}
    payload.setdefault("checks", {})
    payload.setdefault("announced_at", None)
    return payload


def save_evening_state(state: Mapping[str, Any], path: Path = EVENING_BRIEFING_STATE_FILE) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logging.exception("Evening briefing state save failed")


def staged_picks_from_pending(pending_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten load_auto_populate_pending_picks() output to briefing picks."""
    picks: list[dict[str, Any]] = []
    pending = pending_payload.get("pending") if isinstance(pending_payload, Mapping) else {}
    if not isinstance(pending, Mapping):
        return picks
    for side in ("long", "short"):
        bucket = pending.get(side)
        if not isinstance(bucket, Mapping):
            continue
        for symbol, meta in bucket.items():
            meta = meta if isinstance(meta, Mapping) else {}
            picks.append(
                {
                    "symbol": str(symbol).strip().upper(),
                    "side": side,
                    "score": float(meta.get("score") or 0.0),
                    "reason": str(meta.get("reason") or ""),
                }
            )
    return [pick for pick in picks if pick["symbol"]]


def record_strength_check(
    state: dict[str, Any],
    slot: str,
    staged_picks: Iterable[Mapping[str, Any]],
    snapshot: Mapping[str, Mapping[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    """Store one look at every staged pick's distance from its day extreme."""
    observed: dict[str, Any] = {}
    for pick in staged_picks:
        symbol = str(pick.get("symbol") or "").strip().upper()
        side = "short" if str(pick.get("side") or "").lower().startswith("short") else "long"
        if not symbol:
            continue
        quote = snapshot.get(symbol) or {}
        try:
            last = float(quote["last"])
            day_high = float(quote["day_high"])
            day_low = float(quote["day_low"])
        except (KeyError, TypeError, ValueError):
            continue
        if last <= 0 or day_high <= 0:
            continue
        if side == "long":
            distance_pct = (day_high - last) / day_high * 100.0
        else:
            distance_pct = (last - day_low) / last * 100.0
        observed[symbol] = {
            "side": side,
            "score": float(pick.get("score") or 0.0),
            "reason": str(pick.get("reason") or ""),
            "last": last,
            "distance_pct": distance_pct,
        }
    state.setdefault("checks", {})[str(slot)] = {
        "at": now.strftime("%H:%M:%S"),
        "symbols": observed,
    }
    return state


def assess_pick_persistence(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Held/faded verdict per symbol across the recorded strength checks.

    Held = still within PERSISTENCE_NEAR_EXTREME_PCT of the day extreme at
    the latest look AND the price has not moved against the side by more than
    PERSISTENCE_FADE_TOLERANCE_PCT since the first look.
    """
    checks = state.get("checks") or {}
    ordered_slots = sorted(checks, key=_slot_time)
    verdicts: dict[str, dict[str, Any]] = {}
    for slot in ordered_slots:
        for symbol, reading in (checks[slot].get("symbols") or {}).items():
            series = verdicts.setdefault(
                symbol,
                {"side": reading.get("side", "long"), "score": 0.0, "reason": "", "readings": []},
            )
            series["side"] = reading.get("side", series["side"])
            series["score"] = max(float(series["score"]), float(reading.get("score") or 0.0))
            series["reason"] = str(reading.get("reason") or series["reason"])
            series["readings"].append(
                {
                    "slot": slot,
                    "last": float(reading.get("last") or 0.0),
                    "distance_pct": float(reading.get("distance_pct") or 0.0),
                }
            )
    for symbol, series in verdicts.items():
        readings = series.pop("readings")
        first, latest = readings[0], readings[-1]
        side = series["side"]
        if side == "long":
            price_ok = latest["last"] >= first["last"] * (1 - PERSISTENCE_FADE_TOLERANCE_PCT / 100.0)
        else:
            price_ok = latest["last"] <= first["last"] * (1 + PERSISTENCE_FADE_TOLERANCE_PCT / 100.0)
        near_ok = latest["distance_pct"] <= PERSISTENCE_NEAR_EXTREME_PCT
        held = bool(price_ok and near_ok)
        extreme = "HOD" if side == "long" else "LOD"
        if held:
            detail = f"{latest['distance_pct']:.2f}% off {extreme} at {latest['slot']}"
        elif not near_ok:
            detail = f"faded to {latest['distance_pct']:.2f}% off {extreme} by {latest['slot']}"
        else:
            drift = (latest["last"] / first["last"] - 1) * 100.0 if first["last"] else 0.0
            detail = f"price drifted {drift:+.2f}% against the {side} since {first['slot']}"
        series["verdict"] = "held" if held else "faded"
        series["detail"] = detail + (" (single check)" if len(readings) == 1 else "")
        series["checks"] = len(readings)
        series["last"] = latest["last"]
    return verdicts


def select_best_d1_rows(swing_rows: Iterable[Any], per_side: int = BRIEFING_TOP_D1_PER_SIDE) -> dict[str, list[dict[str, Any]]]:
    """Top swing-scan rows per side by expected R (the best D1 setups)."""

    def _field(row: Any, name: str) -> Any:
        if isinstance(row, Mapping):
            return row.get(name)
        return getattr(row, name, None)

    best: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    rows = []
    for row in swing_rows or []:
        symbol = str(_field(row, "symbol") or "").strip().upper()
        side = str(_field(row, "side") or "").strip().lower()
        if not symbol or side not in ("long", "short"):
            continue
        expected = _field(row, "expected_r")
        try:
            expected = float(expected) if expected is not None else None
        except (TypeError, ValueError):
            expected = None
        rows.append(
            {
                "symbol": symbol,
                "side": side,
                "expected_r": expected,
                "bucket": str(_field(row, "bucket_label") or _field(row, "bucket") or "").strip(),
            }
        )
    for side in ("long", "short"):
        side_rows = [row for row in rows if row["side"] == side]
        side_rows.sort(key=lambda row: (row["expected_r"] is None, -(row["expected_r"] or 0.0)))
        best[side] = side_rows[:per_side]
    return best


def build_evening_briefing(
    *,
    now: datetime,
    regime: str,
    swing_rows: Iterable[Any],
    swing_data_current: bool,
    persistence: Mapping[str, Mapping[str, Any]],
    overnight_triggers: Iterable[Mapping[str, Any]],
    checks_done: Iterable[str],
) -> dict[str, Any]:
    held = [
        {"symbol": symbol, **dict(info)}
        for symbol, info in persistence.items()
        if info.get("verdict") == "held"
    ]
    faded = [
        {"symbol": symbol, **dict(info)}
        for symbol, info in persistence.items()
        if info.get("verdict") == "faded"
    ]
    held.sort(key=lambda item: -float(item.get("score") or 0.0))
    faded.sort(key=lambda item: -float(item.get("score") or 0.0))
    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "regime": str(regime or "unknown"),
        "best_d1": select_best_d1_rows(swing_rows),
        "swing_data_current": bool(swing_data_current),
        "held_picks": held[:BRIEFING_MAX_INTRADAY_PICKS],
        "faded_picks": faded[:BRIEFING_MAX_INTRADAY_PICKS],
        "overnight_alerts": [dict(trigger) for trigger in overnight_triggers],
        "checks_done": sorted(set(str(slot) for slot in checks_done), key=_slot_time),
    }


def render_evening_briefing(payload: Mapping[str, Any]) -> str:
    def _d1_line(row: Mapping[str, Any]) -> str:
        expected = row.get("expected_r")
        expected_text = f" | {float(expected):.2f}R" if expected is not None else ""
        bucket = str(row.get("bucket") or "").strip()
        bucket_text = f" | {bucket}" if bucket else ""
        return f"{row.get('symbol')}{bucket_text}{expected_text}"

    def _pick_line(item: Mapping[str, Any]) -> str:
        reason = str(item.get("reason") or "").strip()
        reason_text = f" - {reason}" if reason else ""
        return (
            f"{item.get('symbol')} ({str(item.get('side', '')).upper()}) "
            f"score {float(item.get('score') or 0.0):.2f} | {item.get('detail', '')}{reason_text}"
        )

    def _alert_line(item: Mapping[str, Any]) -> str:
        direction = "ABOVE" if str(item.get("side")) == "above" else "BELOW"
        return f"{item.get('at', '')} {item.get('symbol')} {item.get('last')} crossed {direction} {item.get('level')}"

    def _section(lines: list[str]) -> str:
        return "\n".join(lines) if lines else "(none)"

    best = payload.get("best_d1") or {}
    checks_done = payload.get("checks_done") or []
    swing_note = (
        ""
        if payload.get("swing_data_current")
        else " (from the PREVIOUS session - today's early scan has not completed)"
    )
    held = [_pick_line(item) for item in payload.get("held_picks") or []]
    faded = [_pick_line(item) for item in payload.get("faded_picks") or []]
    sections = [
        "EVENING MODE - MORNING BRIEFING",
        f"Generated: {payload.get('generated_at', '')}",
        f"Market environment: {payload.get('regime', 'unknown')}",
        f"Strength checks completed: {', '.join(checks_done) if checks_done else '(none yet)'}",
        "",
        f"== BEST D1 SWING SETUPS - LONG{swing_note} ==",
        _section([_d1_line(row) for row in best.get("long") or []]),
        "",
        f"== BEST D1 SWING SETUPS - SHORT{swing_note} ==",
        _section([_d1_line(row) for row in best.get("short") or []]),
        "",
        "== INTRADAY PICKS THAT STAYED STRONG ==",
        _section(held),
        "",
        "== FADED SINCE THE FIRST CHECK - NOT RECOMMENDED ==",
        _section(faded),
        "",
        "== OVERNIGHT PRICE ALERTS ==",
        _section([_alert_line(item) for item in payload.get("overnight_alerts") or []]),
        "",
        "Picks above are staged in the Alert Center for chart approval.",
        "Flip Auto mode off EVENING (to DESK or OFF) to resume live recommendations.",
        "",
    ]
    return "\n".join(sections)


def briefing_summary_lines(payload: Mapping[str, Any], limit: int = 12) -> list[str]:
    """Compact lines for the phone report's briefing section."""
    lines = [
        f"Environment: {payload.get('regime', 'unknown')} | checks: "
        f"{', '.join(payload.get('checks_done') or []) or 'none yet'}",
    ]
    best = payload.get("best_d1") or {}
    for side in ("long", "short"):
        rows = best.get(side) or []
        if rows:
            tickers = ", ".join(str(row.get("symbol")) for row in rows)
            lines.append(f"Best D1 {side}s: {tickers}")
    held = payload.get("held_picks") or []
    if held:
        lines.append("Held strong: " + ", ".join(str(item.get("symbol")) for item in held))
    faded = payload.get("faded_picks") or []
    if faded:
        lines.append("Faded (skip): " + ", ".join(str(item.get("symbol")) for item in faded))
    alerts = payload.get("overnight_alerts") or []
    if alerts:
        lines.append(f"Overnight price alerts: {len(alerts)} fired (see Price Alerts tab)")
    return lines[:limit]


def write_evening_briefing_file(text: str, path: Path = EVENING_BRIEFING_FILE) -> bool:
    """Drop the rendered briefing next to autopilot_today.txt on the shared
    store (one tap in the Drive app). Layer-1 writer role gates it like every
    other shared export; a refusal just means the phone copy stays stale."""
    try:
        from autopilot_core import shared_write_refusal

        refusal = shared_write_refusal(Path(path))
    except Exception:
        refusal = ""
    if refusal:
        logging.info("Evening briefing file not written: %s", refusal)
        return False
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        return True
    except OSError:
        logging.exception("Evening briefing file write failed")
        return False
