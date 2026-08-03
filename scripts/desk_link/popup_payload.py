"""Build and serialize self-contained alert popup payloads.

The main desk captures everything its own popup renders — the alert
identity plus the exact D1/M5 snapshot dicts from ``chart_snapshot`` — and
ships it as JSON. A satellite reverses the process and hands the snapshots
straight to the same chart widgets, so one rendering path serves both
machines and a TWS-less satellite never fetches anything.

Qt-free on purpose: payload building runs on the main's GUI thread today
(same cost its own popup already pays) and must stay unit-testable
headless. Bars carry ``dt`` datetimes; JSON gets ISO-8601 strings with
their timezone preserved (plan.md sec 5: timestamps carry explicit
timezones), restored to datetimes on the satellite because the chart axis
formats real datetimes.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Mapping

PAYLOAD_SCHEMA = "desk_link.alert_popup.v1"


def bars_to_wire(bars: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for bar in bars or []:
        row = dict(bar)
        moment = row.get("dt")
        if isinstance(moment, datetime):
            row["dt"] = moment.isoformat()
        wire.append(row)
    return wire


def bars_from_wire(bars: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    restored: list[dict[str, Any]] = []
    for bar in bars or []:
        row = dict(bar)
        moment = row.get("dt")
        if isinstance(moment, str) and moment:
            try:
                row["dt"] = datetime.fromisoformat(moment)
            except ValueError:
                pass  # leave as text; the chart tolerates missing datetimes
        restored.append(row)
    return restored


def _snapshot_to_wire(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    wire = dict(snapshot)
    wire["bars"] = bars_to_wire(wire.get("bars") or [])
    return wire


def _snapshot_from_wire(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    restored = dict(snapshot)
    restored["bars"] = bars_from_wire(restored.get("bars") or [])
    return restored


def build_alert_popup_payload(
    alert: Any,
    *,
    d1_snapshot: Mapping[str, Any],
    m5_snapshot: Mapping[str, Any],
    armed_kinds: list[str] | None = None,
    armed_levels: list[dict[str, Any]] | None = None,
    armed_d1_events: list[dict[str, Any]] | None = None,
    guidance_text: str = "",
) -> dict[str, Any]:
    """Assemble the wire payload from an alert + prebuilt snapshot dicts.

    ``alert`` is a BounceAlert dataclass (or any object ``asdict`` accepts,
    or already a plain dict). Snapshots come from
    ``chart_snapshot.build_d1_snapshot`` / ``build_m5_snapshot`` exactly as
    the main's own popup consumes them.
    """
    if isinstance(alert, dict):
        alert_fields = dict(alert)
    else:
        alert_fields = asdict(alert)
    return {
        "schema": PAYLOAD_SCHEMA,
        "alert": alert_fields,
        "armed": {
            "kinds": list(armed_kinds or []),
            "levels": [dict(level) for level in (armed_levels or [])],
            "d1_events": [dict(event) for event in (armed_d1_events or [])],
        },
        "guidance_text": str(guidance_text or ""),
        "d1": _snapshot_to_wire(d1_snapshot),
        "m5": _snapshot_to_wire(m5_snapshot),
    }


def capture_alert_popup(
    alert: Any,
    *,
    bot: Any = None,
    armed_kinds: list[str] | None = None,
    armed_levels: list[dict[str, Any]] | None = None,
    armed_d1_events: list[dict[str, Any]] | None = None,
    guidance_text: str = "",
) -> dict[str, Any]:
    """Build the full popup payload for an alert on the main desk.

    Same synchronous local reads as the main's own popup
    (SymbolSnapshotWidget._build_snapshots): the bot's in-memory M5 cache
    plus the mtime-cached daily store — never a fetch. The satellite has no
    store to backfill, so unlike the local popup no staleness backfill is
    triggered here; a stale tail renders as-is with its date visible.
    """
    import chart_snapshot  # local import keeps desk_link import-light for the transport tests

    symbol = str(getattr(alert, "symbol", "") or (alert.get("symbol") if isinstance(alert, dict) else "") or "")
    m5_bars: list[dict[str, Any]] = []
    if bot is not None and symbol:
        try:
            m5_bars = bot.m5_chart_bars(symbol, max_sessions=2)
        except Exception:
            m5_bars = []
    d1 = chart_snapshot.build_d1_snapshot(symbol, intraday_bars=m5_bars)
    m5 = chart_snapshot.build_m5_snapshot(symbol, m5_bars)
    return build_alert_popup_payload(
        alert,
        d1_snapshot=d1,
        m5_snapshot=m5,
        armed_kinds=armed_kinds,
        armed_levels=armed_levels,
        armed_d1_events=armed_d1_events,
        guidance_text=guidance_text,
    )


def restore_alert_popup_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild render-ready structures from a wire payload.

    Returns ``{"alert": dict, "armed": dict, "guidance_text": str,
    "d1": snapshot, "m5": snapshot}`` with bar datetimes restored. Raises
    ValueError on an unknown schema so a version skew between machines is a
    visible error, not a half-drawn chart.
    """
    schema = str(payload.get("schema") or "")
    if schema != PAYLOAD_SCHEMA:
        raise ValueError(f"unsupported alert popup payload schema: {schema!r}")
    return {
        "alert": dict(payload.get("alert") or {}),
        "armed": {
            "kinds": list((payload.get("armed") or {}).get("kinds") or []),
            "levels": [dict(level) for level in (payload.get("armed") or {}).get("levels") or []],
            "d1_events": [dict(event) for event in (payload.get("armed") or {}).get("d1_events") or []],
        },
        "guidance_text": str(payload.get("guidance_text") or ""),
        "d1": _snapshot_from_wire(payload.get("d1") or {}),
        "m5": _snapshot_from_wire(payload.get("m5") or {}),
    }
