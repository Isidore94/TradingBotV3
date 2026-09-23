"""Golden replays of the two real 2026-09-23 Pullback alerts.

Real tapes (see the fixture's ``_about``): QDEL SHORT was a correct alert and
must keep firing; ARM LONG ``reclaim_then_lrsi`` fired with no pullback at
all - ARM sat ~50 points above its M30 75-SMA - and is the case the dip gate
exists for.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE = ROOT_DIR / "tests" / "fixtures" / "pullback_dip_gate_real_tapes_v1.json"


def _load():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    now = datetime.fromisoformat(payload["now"])
    symbols = {}
    for symbol, entry in payload["symbols"].items():
        parsed = dict(entry)
        for key in ("M15", "M30", "D1"):
            parsed[key] = [
                dict(bar, dt=datetime.fromisoformat(bar["dt"])) for bar in entry[key]
            ]
        parsed["armed_at"] = datetime.fromisoformat(entry["armed_at"])
        symbols[symbol] = parsed
    return now, symbols


def _evaluate_m30(symbol: str):
    from indicators.pullback_sma_reclaim import evaluate

    now, symbols = _load()
    entry = symbols[symbol]
    kwargs = {}
    try:
        import inspect

        if "daily_bars" in inspect.signature(evaluate).parameters:
            kwargs["daily_bars"] = entry["D1"]
    except (TypeError, ValueError):  # pragma: no cover
        pass
    return evaluate(
        entry["M30"],
        side=entry["side"],
        sma_length=75,
        bar_minutes=30,
        armed_at=entry["armed_at"],
        now=now,
        companion_bars=entry["M15"],
        companion_minutes=15,
        **kwargs,
    )


def test_qdel_short_m30_loss_with_lrsi_cross_still_fires_on_the_real_tape():
    result = _evaluate_m30("QDEL")
    assert result is not None
    fires = {fire.trigger: fire for fire in result.fired}
    assert set(fires) == {"sma_reclaim_lrsi"}
    fire = fires["sma_reclaim_lrsi"]
    assert fire.bar_dt == datetime(2026, 9, 23, 6, 30)
    assert fire.cross_bar_dt == datetime(2026, 9, 22, 12, 30)
    assert fire.cross_timeframe == "M30"
    assert fire.lrsi_from_below_50 is True
    assert round(fire.sma, 4) == round(10.923817354838054, 4)


def test_arm_long_reclaim_then_lrsi_fired_on_the_real_tape_before_the_gate():
    result = _evaluate_m30("ARM")
    assert result is not None
    fires = {fire.trigger: fire for fire in result.fired}
    assert set(fires) == {"reclaim_then_lrsi"}
    fire = fires["reclaim_then_lrsi"]
    assert fire.cross_timeframe == "M15"
    assert fire.cross_bar_dt == datetime(2026, 9, 23, 6, 45)
