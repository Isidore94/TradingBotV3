"""Every RVOL number the desk computes, from one frozen input set (A4b golden).

Shared by ``test_rvol_golden.py`` and the one-off scratch build of
``fixtures/rvol_golden_v1.json``: the same calls produce the expected values
and the actual values, so the golden pins today's code exactly.

Covers the three RVOL implementations and their callers' readings:
- ``rvol.py``: slot baselines, bar/session rvol, session rvol from a
  precomputed baseline (the bounce bot's live path), daily rvol and bands;
- ``intraday_rvol_service.reading_from_bars`` (chart header);
- ``movers_scan``: offset-keyed baseline, recent rvol, rvol weight.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

NY = ZoneInfo("America/New_York")
#: Bars into the 2026-09-25 session at which every reading is taken.
DEPTHS = (1, 2, 3, 6, 12, 24, 42, 60, 77, 78)


def _stamp(day: str, start: str, index: int) -> datetime:
    hour, minute = (int(part) for part in start.split(":"))
    base = datetime.fromisoformat(day).replace(hour=hour, minute=minute)
    return base + timedelta(minutes=5 * index)


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, (date, datetime)) else value


def compute_symbol(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    import movers_scan
    import rvol
    from ui.services.intraday_rvol_service import reading_from_bars

    prior_vols = [[float(v) for v in s["volumes"]] for s in sessions[:-1]]
    today_vols = [float(v) for v in sessions[-1]["volumes"]]
    today_day = date.fromisoformat(sessions[-1]["date"])

    naive_prior = [
        {"dt": _stamp(s["date"], s["start"], i), "volume": float(v)}
        for s in sessions[:-1]
        for i, v in enumerate(s["volumes"])
    ]
    naive_today = [
        {"dt": _stamp(sessions[-1]["date"], sessions[-1]["start"], i), "volume": float(v)}
        for i, v in enumerate(sessions[-1]["volumes"])
    ]
    aware = [
        {"dt": bar["dt"].replace(tzinfo=NY), "volume": bar["volume"]}
        for bar in naive_prior + naive_today
    ]
    aware_today = aware[len(naive_prior):]

    slots = rvol.slot_baselines(prior_vols)
    movers_base = movers_scan.build_rvol_baseline(aware, before=today_day, local_tz=NY)
    movers_base15 = movers_scan.build_rvol_baseline(
        aware, before=today_day, local_tz=NY, sessions=15
    )

    out: dict[str, Any] = {
        "rvol_slot_baselines": slots,
        "movers_baseline": {str(k): v for k, v in sorted(movers_base.items())},
        "movers_baseline_15": {str(k): v for k, v in sorted(movers_base15.items())},
        "depths": {},
    }
    for depth in DEPTHS:
        today = today_vols[:depth]
        reading = reading_from_bars(
            "X", naive_prior + naive_today[:depth], fetched_at=datetime(2026, 9, 25, 16, 5)
        )
        recent3 = movers_scan.recent_rvol(aware_today[:depth], movers_base, bars=3)
        out["depths"][str(depth)] = {
            "rvol_bar_rvol": rvol.bar_rvol(today, prior_vols),
            "rvol_session_rvol": rvol.session_rvol(today, prior_vols),
            "rvol_session_rvol_from_baseline": rvol.session_rvol_from_baseline(today, slots),
            "service_session_rvol": reading.session_rvol,
            "service_last_bar_rvol": reading.last_bar_rvol,
            "service_prior_sessions": reading.prior_sessions,
            "service_session_date": _iso(reading.session_date),
            "service_last_bar_at": _iso(reading.last_bar_at),
            "movers_recent_rvol_3": recent3,
            "movers_recent_rvol_6": movers_scan.recent_rvol(aware_today[:depth], movers_base, bars=6),
            "movers_recent_rvol_span": movers_scan.recent_rvol(
                aware_today[:depth], movers_base, bars=depth
            ),
            "movers_rvol_weight_3": movers_scan.rvol_weight(recent3),
        }
    return out


def compute_all(inputs: dict[str, Any]) -> dict[str, Any]:
    import rvol

    daily = rvol.daily_rvol_series(inputs["spy_daily"]["volumes"])
    return {
        "m5": {sym: compute_symbol(sessions) for sym, sessions in inputs["m5_sessions"].items()},
        "spy_daily_rvol": daily,
        "spy_daily_band": [rvol.rvol_band(value) for value in daily],
    }
